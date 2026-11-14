"""Bounded policy coaching on private vectors and verified behavioral feedback."""

from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter

import torch

from .rl_policy import CAPACITY, Bank, _action, _selected
from .rl_train import _torch_threads

MAX_REPLAY = 24
MAX_CAPTURE_TOKENS = 128
_SAMPLES = 24


@dataclass(frozen=True)
class DreamExample:
    features: torch.Tensor
    mask: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor | None = None
    write: bool | None = None
    key_position: int | None = None
    value_position: int | None = None
    bank: Bank | None = None
    slot: int | None = None


@dataclass(frozen=True)
class DreamResult:
    accepted: bool
    candidate_state: dict[str, torch.Tensor] | None
    stats: dict


def _cpu_copy(value):
    return value.detach().cpu().clone()


def _checked_example(example):
    def tensor(value, name, dimensions, boolean=False):
        if (not isinstance(value, torch.Tensor) or value.ndim != dimensions
                or value.shape[0] != 1 or value.numel() > 128 * 4096
                or (value.dtype != torch.bool if boolean else not value.is_floating_point())
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid coaching tensor: {name}.")
        result = _cpu_copy(value)
        return result if boolean else result.float()

    features = tensor(example.features, "features", 3)
    mask = tensor(example.mask, "mask", 2, boolean=True)
    keys = tensor(example.keys, "keys", 3)
    values = tensor(example.values, "values", 3) if example.values is not None else None
    if (not 1 <= features.shape[1] <= MAX_CAPTURE_TOKENS
            or mask.shape != features.shape[:2] or keys.shape[:2] != mask.shape
            or (values is not None and values.shape[:2] != mask.shape) or not bool(mask.any())):
        raise ValueError("Incompatible coaching capture shapes.")

    bank = None
    if example.write is None:
        if example.bank is None or type(example.slot) is not int or not 0 <= example.slot <= CAPACITY:
            raise ValueError("Read feedback needs a captured bank and verified slot or abstention.")
        bank = Bank(tensor(example.bank.keys, "bank keys", 3), tensor(example.bank.values, "bank values", 3),
                    tensor(example.bank.occupied, "bank occupancy", 2, boolean=True))
        if (bank.occupied.shape != (1, CAPACITY) or bank.keys.shape != (1, CAPACITY, keys.shape[-1])
                or bank.values.shape[:2] != bank.occupied.shape
                or (example.slot < CAPACITY and not bool(bank.occupied[0, example.slot]))):
            raise ValueError("Read feedback refers to an invalid memory bank.")
        positions = (example.key_position,) if example.key_position is not None else ()
        if example.value_position is not None:
            raise ValueError("Read feedback cannot specify a write value pointer.")
    elif type(example.write) is bool:
        if example.bank is not None or example.slot is not None:
            raise ValueError("Write feedback must specify pointers rather than a read slot.")
        if example.write:
            if values is None:
                raise ValueError("Verified writes need captured value vectors.")
            positions = (example.key_position, example.value_position)
        else:
            if example.key_position is not None or example.value_position is not None:
                raise ValueError("Ignore feedback must not specify pointer targets.")
            positions = ()
    else:
        raise ValueError("Admission feedback must be true, false or absent.")
    if any(type(pos) is not int or not 0 <= pos < mask.shape[1] or not bool(mask[0, pos]) for pos in positions):
        raise ValueError("A verified pointer does not reference a captured token position.")
    return DreamExample(features, mask, keys, values, example.write, example.key_position,
                        example.value_position, bank, example.slot)


def _behavior(policy, example):
    if example.write is not None:
        admission, pointers = policy.write_scores(example.features, example.mask)
        admitted = bool(admission.argmax(-1).item())
        if not admitted:
            return ("ignored",)
        key_position, value_position = pointers[0].argmax(-1).tolist()
        if example.values is None:
            return ("wrote", key_position, value_position)
        return ("wrote", example.keys[0, key_position], example.values[0, value_position])
    pointers, abstain = policy.read_scores(example.features, example.mask)
    key = _selected(example.keys, pointers.argmax(-1))
    slot = int(policy.slot_scores(key, example.bank, abstain).argmax(-1).item())
    if slot == CAPACITY:
        return ("unknown",)
    return ("read", slot, example.bank.values[0, slot])


def _expected(example):
    if example.write is True:
        return ("wrote", example.keys[0, example.key_position], example.values[0, example.value_position])
    if example.write is False:
        return ("ignored",)
    if example.slot == CAPACITY:
        return ("unknown",)
    return ("read", example.slot, example.bank.values[0, example.slot])


def _equal_behavior(actual, expected):
    if len(actual) != len(expected) or actual[0] != expected[0]:
        return False
    return all(torch.equal(a, b) if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) else a == b
               for a, b in zip(actual[1:], expected[1:], strict=True))


def _sample_cost(logits, target, generator):
    action, log_prob, entropy = _action(logits, "sample", generator, temperature=2.0)
    return _reinforce(action.eq(target).float(), log_prob, entropy)


def _reinforce(reward, log_prob, entropy):
    # Leave each action out of its own baseline.
    advantage = (reward * _SAMPLES - reward.sum(dim=0)) / (_SAMPLES - 1)
    return -(advantage * log_prob + .002 * entropy).mean()


def _same_vectors(rows, target_position):
    return (rows == rows[:, target_position:target_position + 1]).all(-1)


def _pointer_cost(logits, verified, generator):
    action, log_prob, entropy = _action(logits, "sample", generator, temperature=2.0)
    match = verified.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    return _reinforce(match.float(), log_prob, entropy)


def _sample_loss(policy, example, generator):
    if example.write is not None:
        admission, pointers = policy.write_scores(example.features, example.mask)
        loss = _sample_cost(admission.expand(_SAMPLES, -1), int(example.write), generator)
        if example.write:
            # The three writer actions are independent given the observation.
            # Score their verified transitions separately to avoid sparse credit.
            pointer_logits = pointers.expand(_SAMPLES, -1, -1)
            pointer_targets = torch.stack((_same_vectors(example.keys, example.key_position),
                                           _same_vectors(example.values, example.value_position)), dim=1)
            loss = loss + _pointer_cost(pointer_logits, pointer_targets.expand(_SAMPLES, -1, -1), generator)
        return loss

    pointers, abstain = policy.read_scores(example.features, example.mask)
    position, pointer_prob, pointer_entropy = _action(
        pointers.expand(_SAMPLES, -1), "sample", generator, temperature=2.0,
    )
    selected_keys = example.keys[0, position]
    slot_scores = policy.slot_scores(selected_keys, example.bank, abstain.expand(_SAMPLES, -1))
    slot, slot_prob, slot_entropy = _action(slot_scores,
                                           "sample", generator, temperature=2.0)
    reward = slot.eq(example.slot).float()
    pointer_reward = reward
    if example.key_position is not None:
        pointer_reward = (reward + _same_vectors(example.keys, example.key_position)[0, position].float()) / 2
    return _reinforce(reward, slot_prob, slot_entropy) + _reinforce(pointer_reward, pointer_prob, pointer_entropy)


@torch.inference_mode(False)
@torch.enable_grad()
def coach(policy, focus, rehearsal=(), *, guard=(), max_steps=48, timeout_s=2.0, seed=41):
    """Try a new policy on CPU. Only return weights that pass greedy replay gates."""
    if type(max_steps) is not int or not 1 <= max_steps <= 128:
        raise ValueError("Coaching must use 1–128 optimization steps.")
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
        raise ValueError("Coaching must use a finite timeout of at most 30 seconds.")
    if type(seed) is not int:
        raise ValueError("Coaching seed must be an integer.")
    if len(rehearsal) + len(guard) > MAX_REPLAY:
        raise ValueError(f"Coaching accepts at most {MAX_REPLAY} previous captures.")
    started = perf_counter()
    focus = _checked_example(focus)
    rehearsal = tuple(_checked_example(row) for row in rehearsal)
    guard = tuple(_checked_example(row) for row in guard)
    candidate = deepcopy(policy).cpu().eval()
    replay = (*rehearsal, *guard)
    expected = _expected(focus)
    steps = 0
    identical_checks = 0
    regressed = 0
    guard_correct = 0
    focus_before = focus_after = False

    def result(reason, accepted=False):
        state = {key: _cpu_copy(value) for key, value in candidate.state_dict().items()} if accepted else None
        return DreamResult(accepted, state, {
            "reason": reason, "steps": steps, "elapsed_s": perf_counter() - started, "focus_before": focus_before,
            "focus_after": focus_after, "rehearsal_rows": len(rehearsal), "holdout_rows": len(guard),
            "guard_before_correct": guard_correct, "behavior_checks": identical_checks,
            "guard_regressions": regressed,
        })

    with _torch_threads(1):
        with torch.no_grad():
            focus_before = focus_after = _equal_behavior(_behavior(candidate, focus), expected)
            before = tuple(_behavior(candidate, row) for row in replay)
            guard_correct = sum(_equal_behavior(old, _expected(row)) for old, row in zip(before, replay, strict=True))
        if focus_before:
            return result("already_correct")
        if perf_counter() - started >= timeout_s:
            return result("timeout")
        generator = torch.Generator().manual_seed(seed)
        # Ignore feedback only identifies the admission decision. Keep the
        # shared encoder and span heads fixed for that update.
        update_scope = candidate.admit if focus.write is False else candidate
        parameters = [parameter for parameter in update_scope.parameters() if parameter.requires_grad]
        if not parameters:
            return result("no_trainable_parameters")
        if update_scope is not candidate:
            candidate.requires_grad_(False)
            for parameter in parameters:
                parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(parameters, lr=.003, weight_decay=0)
        while steps < max_steps and perf_counter() - started < timeout_s:
            loss = _sample_loss(candidate, focus, generator)
            if rehearsal:
                first = steps % len(rehearsal)
                batch = [rehearsal[(first + offset) % len(rehearsal)] for offset in range(min(3, len(rehearsal)))]
                loss = loss + torch.stack([_sample_loss(candidate, row, generator) for row in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            except RuntimeError:
                return result("invalid_gradient")
            optimizer.step()
            steps += 1
            if perf_counter() - started >= timeout_s:
                break
            with torch.no_grad():
                focus_after = _equal_behavior(_behavior(candidate, focus), expected)
                if not focus_after:
                    continue
                identical_checks += 1
                regressed = sum(not _equal_behavior(_behavior(candidate, row), old)
                                for old, row in zip(before, replay, strict=True))
            if not regressed and perf_counter() - started < timeout_s:
                return result("accepted", accepted=True)
    return result("timeout" if perf_counter() - started >= timeout_s else "behavior_gate_failed")
