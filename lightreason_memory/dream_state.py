"""Private tensor-only session metadata and replay for online memory coaching."""

from dataclasses import dataclass, field
import os

import torch
from safetensors.torch import load

from .dream_train import DreamExample, coach
from .neural import KEY_DIM, MAX_TOKENS, MAX_VALUE_TOKENS, TARGET_WIDTH, VALUE_DIM, _private_open
from .rl_policy import Bank, CAPACITY


FEEDBACK_LIMIT = 8
GUARD_LIMIT = 4
STATE_BYTE_LIMIT = 5 * 1024 * 1024
STATUSES = ("collecting", "awaiting_feedback", "unchanged", "rejected", "promoted", "training_failed")
_COUNTERS = ("messages", "pending", "status", "generation", "attempts", "steps")


@dataclass
class AlignedValues:
    binding: torch.Tensor
    rows: torch.Tensor = field(default_factory=lambda: torch.zeros(
        (CAPACITY, MAX_VALUE_TOKENS, TARGET_WIDTH), dtype=torch.bfloat16))
    lengths: torch.Tensor = field(default_factory=lambda: torch.zeros(CAPACITY, dtype=torch.int16))

    def replace(self, slot, value):
        self.rows[slot].zero_()
        self.lengths[slot] = 0 if value is None else len(value)
        if value is not None:
            self.rows[slot, :len(value)] = value

    def tensors(self):
        return {"target_binding": self.binding, "target_rows": self.rows, "target_lengths": self.lengths}


@dataclass
class DreamState:
    messages: int = 0
    pending: int = 0
    status: str = "collecting"
    generation: int = 0
    attempts: int = 0
    steps: int = 0
    feedback: list = field(default_factory=list)
    guards: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    aligned: AlignedValues | None = None

    def report(self):
        return {name: getattr(self, name) for name in _COUNTERS}

    def add_feedback(self, example):
        self.feedback.append(example)
        self.feedback = self.feedback[-FEEDBACK_LIMIT:]
        self.pending = min(self.pending + 1, len(self.feedback))

    def add_guard(self, example):
        self.guards = [*self.guards, example][-GUARD_LIMIT:]

    def tick(self, policy, cadence, max_steps):
        self.messages += 1
        if self.messages % cadence:
            self.status = "collecting"
            return
        if not self.pending:
            self.status = "awaiting_feedback"
            return
        focus_index = len(self.feedback) - self.pending
        self.attempts += 1
        try:
            result = coach(policy, self.feedback[focus_index], self.feedback[:focus_index],
                           guard=(*self.feedback[focus_index + 1:], *self.guards),
                           max_steps=max_steps, timeout_s=3.0, seed=41 + self.attempts)
        except (ValueError, RuntimeError, TimeoutError):
            self.status = "training_failed"
            self.steps = 0
            return
        self.pending -= 1
        self.steps = result.stats["steps"]
        if result.accepted:
            self.weights = result.candidate_state
            self.generation += 1
            self.status = "promoted"
        else:
            self.status = "unchanged" if result.stats["reason"] == "already_correct" else "rejected"

    def tensors(self):
        counters = [STATUSES.index(self.status) if name == "status" else getattr(self, name) for name in _COUNTERS]
        result = {"dream": torch.tensor(counters, dtype=torch.int64)}
        for label, examples in (("feedback", self.feedback), ("guard", self.guards)):
            for index, example in enumerate(examples):
                result.update({f"{label}.{index}.{name}": value for name, value in _capture(example).items()})
        result.update({f"policy.{name}": value for name, value in self.weights.items()})
        return result


def _capture(example):
    result = {"features": example.features, "mask": example.mask, "keys": example.keys}
    if example.write is not None:
        result["target"] = torch.tensor([int(example.write),
                                         -1 if example.key_position is None else example.key_position,
                                         -1 if example.value_position is None else example.value_position])
        if example.values is not None:
            result["values"] = example.values
    else:
        result["target"] = torch.tensor([example.slot])
        result.update(bank_keys=example.bank.keys, bank_values=example.bank.values, bank_occupied=example.bank.occupied)
    return result


def _read_capture(tensors, prefix, feature_width, feedback):
    features = tensors.pop(prefix + "features")
    length = features.shape[1] if features.ndim == 3 else 0

    def require(value, shape, dtype):
        if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError("Invalid persisted coaching capture.")
        return value

    if not 1 <= length <= MAX_TOKENS:
        raise ValueError("Invalid persisted coaching length.")
    require(features, (1, length, feature_width), torch.float16)
    keys = require(tensors.pop(prefix + "keys"), (1, length, KEY_DIM), torch.float16)
    mask = require(tensors.pop(prefix + "mask"), (1, length), torch.bool)
    if not mask.any():
        raise ValueError("An empty capture cannot be replayed.")
    target = tensors.pop(prefix + "target")
    require(target, (3 if feedback else 1,), torch.int64)
    if feedback:
        admitted, key, value = target.tolist()
        if admitted == 0 and (key, value) == (-1, -1):
            return DreamExample(features, mask, keys, write=False)
        if admitted != 1 or any(not 0 <= position < length or not mask[0, position] for position in (key, value)):
            raise ValueError("Invalid persisted coaching targets.")
        values = require(tensors.pop(prefix + "values"), (1, length, VALUE_DIM), torch.float16)
        return DreamExample(features, mask, keys, values, True, key, value)
    slot = target.item()
    bank = Bank(require(tensors.pop(prefix + "bank_keys"), (1, CAPACITY, KEY_DIM), torch.float16),
                require(tensors.pop(prefix + "bank_values"), (1, CAPACITY, VALUE_DIM), torch.float16),
                require(tensors.pop(prefix + "bank_occupied"), (1, CAPACITY), torch.bool))
    if not 0 <= slot <= CAPACITY or (slot != CAPACITY and not bank.occupied[0, slot]):
        raise ValueError("Invalid persisted read guard.")
    return DreamExample(features, mask, keys, bank=bank, slot=slot)


def load_state(path, binding, device, *, feature_width, policy_template):
    bank = Bank.empty(1, KEY_DIM, VALUE_DIM, device)
    dream = DreamState()
    try:
        fd = _private_open(path, os.O_RDONLY)
    except FileNotFoundError:
        return bank, dream
    with os.fdopen(fd, "rb") as stream:
        payload = stream.read(STATE_BYTE_LIMIT + 1)
    if len(payload) > STATE_BYTE_LIMIT:
        raise ValueError("Invalid memory state size.")
    rows = load(payload)
    if not torch.equal(rows.pop("binding", torch.empty(0)), binding):
        raise ValueError("Memory state belongs to a different base policy.")
    loaded = []
    for name in ("keys", "values", "occupied"):
        tensor = rows.pop(name, torch.empty(0))
        dtype = torch.bool if name == "occupied" else torch.float16
        if tensor.dtype != dtype or tensor.shape != getattr(bank, name).shape or not torch.isfinite(tensor).all():
            raise ValueError("Invalid tensor memory state.")
        loaded.append(tensor.to(device=device, dtype=torch.bool if name == "occupied" else torch.float32))
    bank = Bank(*loaded)
    if "target_binding" in rows:
        identity = rows.pop("target_binding")
        aligned = AlignedValues(identity)
        if identity.shape != (32,) or identity.dtype != torch.uint8:
            raise ValueError("Invalid destination checkpoint binding.")
        for key, field_name in (("target_rows", "rows"), ("target_lengths", "lengths")):
            actual = rows.pop(key, torch.empty(0))
            expected = getattr(aligned, field_name)
            if (actual.shape != expected.shape or actual.dtype != expected.dtype
                    or not torch.isfinite(actual).all()):
                raise ValueError("Invalid target-aligned state.")
            setattr(aligned, field_name, actual)
        if (not 0 <= aligned.lengths.min() <= aligned.lengths.max() <= MAX_VALUE_TOKENS
                or ((aligned.lengths > 0) & ~bank.occupied[0].cpu()).any()):
            raise ValueError("Invalid target-aligned row lengths.")
        dream.aligned = aligned
    counters = rows.pop("dream", None)
    if counters is None and not rows:
        return bank, dream
    if counters is None or counters.shape != (len(_COUNTERS),) or counters.dtype != torch.int64:
        raise ValueError("Invalid dreaming counters.")
    values = counters.tolist()
    if min(values) < 0 or values[2] >= len(STATUSES):
        raise ValueError("Invalid dreaming status.")
    values[2] = STATUSES[values[2]]
    dream = DreamState(**dict(zip(_COUNTERS, values, strict=True)), aligned=dream.aligned)
    try:
        for label, limit, examples in (("feedback", FEEDBACK_LIMIT, dream.feedback), ("guard", GUARD_LIMIT, dream.guards)):
            for index in range(limit):
                prefix = f"{label}.{index}."
                if prefix + "target" not in rows:
                    break
                examples.append(_read_capture(rows, prefix, feature_width, label == "feedback"))
        for name in policy_template:
            value = rows.pop(f"policy.{name}", None)
            if value is not None:
                expected = policy_template[name]
                if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
                    raise ValueError("Invalid active policy tensor.")
                dream.weights[name] = value
    except KeyError as exc:
        raise ValueError("Incomplete persisted coaching capture.") from exc
    if rows or (dream.weights and set(dream.weights) != set(policy_template)) or dream.pending > len(dream.feedback):
        raise ValueError("Invalid or incomplete dreaming state.")
    return bank, dream
