"""Train and retain a local decision policy over frozen Qwen features."""

from hashlib import file_digest, sha256
import os
from pathlib import Path

import torch
from safetensors.torch import load
from torch.nn.functional import normalize

from lightreason_memory.latent_freetoken_hook import HIDDEN_SIZE, MAX_ROWS, embedding_digest
from lightreason_memory.neural import _atomic_write, _private_open, _require_private_directory
from lightreason_memory.rl_memory import FEATURE_WIDTH
from lightreason_memory.rl_policy import _action


_FORMAT = "vicena-skill-memory-1"
_MAX_ACTIONS = 24
_MAX_BYTES = 2 * 1024 * 1024
_SAMPLES = 24
_STEPS = 96
_MIN_CONFIDENCE = 0.55


def _hash(action):
    return sha256(action.encode()).digest()


def _features(encoder, question):
    with torch.no_grad():
        features, mask, _, _ = encoder.request(question, encode_values=False)
        pooled = features[0, mask[0]].float().mean(0).cpu()
    result = torch.cat([normalize(part, dim=0) for part in pooled.chunk(2)])
    if result.shape != (FEATURE_WIDTH,) or not torch.isfinite(result).all():
        raise ValueError("Invalid frozen Qwen features.")
    return result


def _select(state, features):
    if not state:
        return 0
    scores = (state["weights"] @ features + state["biases"]).softmax(0)
    confidence, action = scores.max(0)
    return int(action) if confidence >= _MIN_CONFIDENCE else 0


def _selection_key(state, action):
    return bytes(state["keys"][action - 1].tolist()) if action else None


def _rows(state, action):
    if not action:
        return None
    slot = action - 1
    return state["vectors"][slot, :int(state["lengths"][slot])]


def _validate_examples(labels, training, checks):
    if not 1 <= len(labels) <= _MAX_ACTIONS or any(
        not isinstance(action, str) or not action.strip() or not isinstance(lesson, str) or not lesson.strip()
        for action, lesson in labels.items()
    ):
        raise ValueError("Supply 1 to 24 named actions with approved lesson text.")
    if not training or not checks or len(training) + len(checks) > 128:
        raise ValueError("Supply training and held-out cases, up to 128 cases in total.")
    for row in (*training, *checks):
        if (not isinstance(row["id"], str) or not isinstance(row["question"], str) or not row["question"].strip()
                or (row["action"] is not None and row["action"] not in labels)):
            raise ValueError("Each case needs a question, ID, and a known action or abstention.")
    if {row["question"].strip().casefold() for row in training} & {row["question"].strip().casefold() for row in checks}:
        raise ValueError("Training and held-out questions must be distinct.")
    if len({row["id"] for row in checks}) != len(checks):
        raise ValueError("Held-out cases must have distinct IDs.")


class SkillMemory:
    """Persist policy weights and destination input vectors in an owner-only checkpoint."""

    def __init__(self, path, row_encoder):
        self.path = Path(path)
        self.row_encoder = row_encoder

    def _load(self):
        if not self.path.parent.is_dir():
            return {}
        _require_private_directory(self.path.parent)
        try:
            fd = _private_open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return {}
        with os.fdopen(fd, "rb") as stream:
            payload = stream.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            raise ValueError("Skill checkpoint is too large.")
        state = load(payload)
        if set(state) != {"keys", "weights", "biases", "vectors", "lengths"} or state["keys"].ndim != 2:
            raise ValueError("Invalid skill checkpoint layout.")
        count = len(state["keys"])
        specs = {
            "keys": ((count, 32), torch.uint8),
            "weights": ((count + 1, FEATURE_WIDTH), torch.float32),
            "biases": ((count + 1,), torch.float32),
            "vectors": ((count, MAX_ROWS, HIDDEN_SIZE), torch.bfloat16),
            "lengths": ((count,), torch.long),
        }
        if (not 1 <= count <= _MAX_ACTIONS
                or any((state[name].shape, state[name].dtype) != spec or not torch.isfinite(state[name]).all()
                       for name, spec in specs.items())
                or not state["lengths"].ge(1).all() or not state["lengths"].le(MAX_ROWS).all()
                or len(state["keys"].unique(dim=0)) != count):
            raise ValueError("Invalid skill checkpoint tensors.")
        return state

    def recall(self, encoder, question):
        state = self._load()
        if not state:
            return None
        rows = _rows(state, _select(state, _features(encoder, question)))
        return None if rows is None else rows.clone()

    def evaluate(self, encoder, checks, *, labels=None):
        state = self._load()
        names = {_hash(action): action for action in (labels or {})}
        names.update({_hash(case["action"]): case["action"] for case in checks if case["action"] is not None})
        outcomes = []
        for case in checks:
            action = _select(state, _features(encoder, case["question"])) if state else 0
            rows = _rows(state, action)
            outcomes.append({
                **self._outcome(state, action, case, names),
                "vector_rows": len(rows) if rows is not None else 0,
                "vector_sha256": embedding_digest(rows) if rows is not None else None,
            })
        return outcomes

    @staticmethod
    def _outcome(state, action, case, names):
        key = _selection_key(state, action)
        expected = case["action"]
        name = names.get(key, key.hex()[:12]) if key else None
        return {"id": case["id"], "question": case["question"], "expected": expected, "actual": name,
                "reward": 1 if name == expected else -1, "retain": bool(case.get("retain"))}

    def train(self, encoder, *, labels, training, checks, progress=None, supersedes=None):
        _validate_examples(labels, training, checks)
        previous = self._load()
        supersedes = supersedes or {}
        if any(new not in labels or old not in labels or new == old for new, old in supersedes.items()):
            raise ValueError("Corrections must name distinct known actions.")
        retired = set(supersedes.values())
        if retired.intersection(case["action"] for case in (*training, *checks)):
            raise ValueError("A retired action cannot be an active reward target.")
        actions = sorted(action for action in labels if action not in retired)
        slots = {action: index for index, action in enumerate([None, *actions])}
        names = {_hash(action): action for action in labels}
        encoded = {}
        for case in (*training, *checks):
            question = case["question"]
            if question not in encoded:
                encoded[question] = _features(encoder, question)
        if progress:
            progress(f"Evaluating {len(checks)} held-out scenarios before rehearsal.")

        prior_actions = [_select(previous, encoded[case["question"]]) for case in checks]
        before = [self._outcome(previous, choice, case, names)
                  for choice, case in zip(prior_actions, checks, strict=True)]
        expected = torch.tensor([slots[case["action"]] for case in training])
        inputs = torch.stack([encoded[case["question"]] for case in training])
        keys = torch.tensor([list(_hash(action)) for action in actions], dtype=torch.uint8)
        prior_slots = {_selection_key(previous, slot): slot for slot in range(len(previous.get("keys", ())) + 1)}
        with torch.enable_grad():
            policy = torch.nn.Linear(FEATURE_WIDTH, len(slots))
            with torch.no_grad():
                policy.weight.zero_()
                policy.bias.zero_()
                for key, slot in [(None, 0), *[(_hash(supersedes.get(name, name)), slots[name]) for name in actions]]:
                    if previous and key in prior_slots:
                        prior = prior_slots[key]
                        policy.weight[slot].copy_(previous["weights"][prior])
                        policy.bias[slot].copy_(previous["biases"][prior])
            # A correction inherits a decision boundary. Use small replay steps
            # to preserve it while the policy learns the revised action.
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.001 if supersedes else 0.05)
            generator = torch.Generator().manual_seed(42)
            for step in range(1, _STEPS + 1):
                logits = policy(inputs)
                chosen, log_prob, entropy = _action(
                    logits.unsqueeze(0).expand(_SAMPLES, -1, -1), "sample", generator, temperature=2.0,
                )
                reward = chosen.eq(expected).float()
                advantage = (_SAMPLES * reward - reward.sum(dim=0)) / (_SAMPLES - 1)
                loss = -(advantage * log_prob + 0.002 * entropy).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    certainty = policy(inputs).softmax(-1).gather(1, expected[:, None])
                    if certainty.min() > 0.985:
                        break
        candidate = {"keys": keys, "weights": policy.weight.detach(), "biases": policy.bias.detach()}
        result = []
        for case, baseline in zip(checks, before, strict=True):
            choice = _select(candidate, encoded[case["question"]])
            result.append({"before": baseline["actual"], **self._outcome(candidate, choice, case, names)})

        before_passed = sum(row["reward"] == 1 for row in before)
        after_passed = sum(row["reward"] == 1 for row in result)
        promoted = after_passed == len(checks) and after_passed > before_passed
        digest = None
        if promoted:
            rows = [self.row_encoder(labels[action]) for action in actions]
            if any(item is None for item in rows):
                raise ValueError("Approved lesson exceeds the answering model’s native vector limit.")
            vectors, lengths = zip(*rows, strict=True)
            candidate.update(vectors=torch.stack(vectors), lengths=torch.tensor(lengths, dtype=torch.long))
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _require_private_directory(self.path.parent)
            _atomic_write(candidate, self.path, {"format": _FORMAT})
            with self.path.open("rb") as stream:
                digest = file_digest(stream, "sha256").hexdigest()
        return {"promoted": promoted, "optimizer_steps": step, "checkpoint_sha256": digest,
                "before_passed": before_passed, "after_passed": after_passed, "checks": result}
