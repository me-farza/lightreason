"""Imitation and reward-based training for the experimental memory policy."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from time import perf_counter

import torch


@dataclass
class Trajectory:
    """Per-episode outcomes and per-action scores with zero padding."""

    log_probs: torch.Tensor
    entropies: torch.Tensor
    rewards: torch.Tensor
    metrics: dict[str, tuple[int, int]] | None = None


def _batch_size(data):
    if not data:
        raise ValueError("An episode collection must contain batched tensors.")
    if any(not isinstance(value, torch.Tensor) or value.ndim == 0 for value in data.values()):
        raise ValueError("Each episode field must have a leading batch dimension.")
    sizes = {len(value) for value in data.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("Episode fields must have the same nonzero length.")
    return sizes.pop()


def _take(data, indexes):
    return {key: value.index_select(0, indexes.to(value.device)) for key, value in data.items()}


def _deadline(end):
    if end is not None and perf_counter() >= end:
        raise TimeoutError("The memory-policy training deadline elapsed.")


def _check_trajectory(trajectory, size):
    if (trajectory.log_probs.ndim != 2 or trajectory.log_probs.shape[0] != size
            or trajectory.log_probs.shape[1] == 0
            or trajectory.entropies.shape != trajectory.log_probs.shape
            or trajectory.rewards.shape != (size,)):
        raise ValueError("The memory rollout returned incompatible action or episode dimensions.")


@contextmanager
def _torch_threads(count):
    if count is None:
        yield
        return
    if type(count) is not int or count < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@torch.no_grad()
def evaluate(policy, data, rollout, *, batch_size=64, deadline=None):
    """Score greedily on complete unseen episodes, with no updates."""

    count = _batch_size(data)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    prior_mode = policy.training
    policy.eval()
    metrics = {}
    rewards = []
    started = perf_counter()
    try:
        for start in range(0, count, batch_size):
            _deadline(deadline)
            stop = min(start + batch_size, count)
            subset = {key: value[start:stop] for key, value in data.items()}
            trajectory = rollout(policy, subset, mode="greedy")
            _check_trajectory(trajectory, stop - start)
            rewards.append(trajectory.rewards.detach().float().cpu())
            for label, (correct, total) in (trajectory.metrics or {}).items():
                if not 0 <= int(correct) <= int(total):
                    raise ValueError("A rollout reported invalid correctness counts.")
                accumulated = metrics.setdefault(label, [0, 0])
                accumulated[0] += int(correct)
                accumulated[1] += int(total)
        _deadline(deadline)
    finally:
        policy.train(prior_mode)
    reward = torch.cat(rewards)
    if not bool(torch.isfinite(reward).all()):
        raise ValueError("Evaluation produced nonfinite rewards.")
    return {
        "episodes": count,
        "mean_reward": reward.mean().item(),
        "reward_std": reward.std(unbiased=False).item(),
        "metrics": {
            label: {"correct": correct, "total": total, "accuracy": correct / total if total else None}
            for label, (correct, total) in sorted(metrics.items())
        },
        "elapsed_s": perf_counter() - started,
    }


def _snapshot(policy):
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}


def _batch_baseline(rewards):
    # Compare each episode to other sampled episodes. Its own action cannot
    # change its baseline. A single episode uses zero as the baseline.
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    return (rewards.sum() - rewards) / (rewards.numel() - 1)


def train(policy, train_batch, dev_batch, rollout, *, warmstart_steps=200, rl_steps=200,
          batch_size=64, learning_rate=0.001, rl_learning_rate=0.0003, entropy_weight=0.002,
          baseline="batch", seed=42, timeout_s=600, report_every=50, progress=None,
          cpu_threads=None, matched_teacher=False, aligned_teacher=False):
    """Run teacher imitation, then categorical REINFORCE on new rollouts.

    The callback receives train-only teacher targets in teacher mode. It must
    perform actual policy-selected state transitions in sample and greedy
    modes. Reinforcement uses episode rewards with no teacher action loss.
    The optional matched control continues teacher imitation from the same
    checkpoint, with the reinforcement stage's batches, learning rate and
    number of optimizer steps.
    """

    count = _batch_size(train_batch)
    _batch_size(dev_batch)
    for name, value, minimum in (
        ("warmstart_steps", warmstart_steps, 0), ("rl_steps", rl_steps, 0),
        ("batch_size", batch_size, 1), ("report_every", report_every, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}.")
    for name, value in (
        ("learning_rate", learning_rate), ("rl_learning_rate", rl_learning_rate), ("timeout_s", timeout_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number.")
    if not math.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("entropy_weight must be finite and nonnegative.")
    if baseline not in {"batch", "greedy"}:
        raise ValueError("baseline must be batch or greedy.")
    if type(matched_teacher) is not bool:
        raise ValueError("matched_teacher must be a boolean.")
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The policy has no trainable parameters.")
    device = parameters[0].device
    indexes = torch.Generator().manual_seed(seed)
    actions = torch.Generator(device=device).manual_seed(seed + 1)
    batch_size = min(batch_size, count)
    end = perf_counter() + timeout_s
    snapshots, report = {}, {"settings": {
        "seed": seed, "batch_size": batch_size, "warmstart_steps": warmstart_steps, "rl_steps": rl_steps,
        "learning_rate": learning_rate, "rl_learning_rate": rl_learning_rate, "entropy_weight": entropy_weight,
        "baseline": baseline, "matched_teacher": matched_teacher, "device": str(device), "training_episodes": count,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }, "stages": {}}
    total_started = perf_counter()
    prior_mode = policy.training

    def run_stage(name, mode, steps, rate):
        optimizer = torch.optim.AdamW(parameters, lr=rate)
        started = perf_counter()
        rows = []
        policy.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(1, steps + 1):
            _deadline(end)
            batch = _take(train_batch, torch.randperm(count, generator=indexes)[:batch_size])
            if mode == "aligned_teacher":
                batch["write_admit"] = batch["write_admit"].masked_fill(batch["write_transient"], 0)
            trajectory = rollout(policy, batch, mode="teacher" if mode == "aligned_teacher" else mode,
                                 generator=actions)
            _check_trajectory(trajectory, batch_size)
            reward = trajectory.rewards.detach().float()
            entropy = trajectory.entropies.float().sum(dim=-1)
            took_action = trajectory.log_probs.float().sum(dim=-1)
            outperformed_baseline = None
            if mode in {"teacher", "aligned_teacher"}:
                loss = -took_action.mean()
            else:
                if baseline == "greedy":
                    with torch.no_grad():
                        greedy = rollout(policy, batch, mode="greedy")
                        _check_trajectory(greedy, batch_size)
                        reference = greedy.rewards.float()
                else:
                    reference = _batch_baseline(reward)
                advantage = reward - reference
                outperformed_baseline = (advantage > 0).float().mean().item() if (
                    step == 1 or step % report_every == 0 or step == steps
                ) else None
                loss = -(advantage * took_action + entropy_weight * entropy).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % report_every == 0 or step == steps:
                row = {
                    "step": step, "loss": loss.detach().item(), "training_reward": reward.mean().item(),
                    "entropy": entropy.detach().mean().item(), "gradient_norm": norm.item(),
                    "elapsed_s": perf_counter() - started,
                }
                if outperformed_baseline is not None:
                    row["fraction_above_baseline"] = outperformed_baseline
                rows.append(row)
                if progress is not None:
                    progress({"stage": name, **row})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _deadline(end)
        peak_cuda_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        return {
            "steps": steps, "optimization_s": perf_counter() - started, "peak_cuda_allocated_bytes": peak_cuda_bytes,
            "progress": rows,
        }

    def record_stage(name, optimization=None):
        snapshots[name] = _snapshot(policy)
        report["stages"][name] = {
            **(optimization or {}),
            "development": evaluate(policy, dev_batch, rollout, batch_size=batch_size, deadline=end),
        }
        if progress is not None:
            progress({"stage": name, "development": report["stages"][name]["development"]})

    with _torch_threads(cpu_threads):
        try:
            record_stage("untrained")
            record_stage("supervised", run_stage("supervised", "teacher", warmstart_steps, learning_rate))
            if matched_teacher or aligned_teacher:
                batch_rng = indexes.get_state()
                action_rng = actions.get_state()
                for enabled, stage, mode in (
                    (matched_teacher, "matched_teacher", "teacher"),
                    (aligned_teacher, "aligned_teacher", "aligned_teacher"),
                ):
                    if enabled:
                        record_stage(stage, run_stage(stage, mode, rl_steps, rl_learning_rate))
                        policy.load_state_dict(snapshots["supervised"])
                        indexes.set_state(batch_rng)
                        actions.set_state(action_rng)
            record_stage("rl", run_stage("rl", "sample", rl_steps, rl_learning_rate))
        finally:
            policy.train(prior_mode)
    report["elapsed_s"] = perf_counter() - total_started
    return snapshots, report
