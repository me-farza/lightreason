"""A Qwen-conditioned episodic writer and reader with tensor-only state."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .rl_train import Trajectory


CAPACITY = 8
QUERY_KINDS = ("initial", "retention", "distractor", "correction", "unknown", "transient")
MASK = -1e4


@dataclass
class Bank:
    keys: torch.Tensor
    values: torch.Tensor
    occupied: torch.Tensor

    @classmethod
    def empty(cls, batch, key_width, value_width, device):
        return cls(torch.zeros(batch, CAPACITY, key_width, device=device),
                   torch.zeros(batch, CAPACITY, value_width, device=device),
                   torch.zeros(batch, CAPACITY, device=device, dtype=torch.bool))


class MemoryPolicy(nn.Module):
    """Scores memory actions from frozen contextual token features."""

    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 96), nn.GELU(), nn.LayerNorm(96))
        self.lexical = nn.Sequential(nn.Linear(width // 2, 64), nn.LayerNorm(64))
        self.writer = nn.Linear(64, 2)
        self.admit = nn.Linear(96, 2)
        self.query = nn.Linear(64, 1)
        self.abstain = nn.Linear(96, 1)
        self.match_scale = nn.Parameter(torch.tensor(18.0))

    def write_scores(self, features, mask):
        hidden = self.encoder(features.float())
        summary = (hidden * mask.unsqueeze(-1)).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)
        spans = self.writer(self.lexical(features[..., :features.shape[-1] // 2].float()))
        spans = spans.movedim(-1, -2).masked_fill(~mask.unsqueeze(-2), MASK)
        return self.admit(summary), spans

    def read_scores(self, features, mask):
        hidden = self.encoder(features.float())
        summary = (hidden * mask.unsqueeze(-1)).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)
        pointers = self.query(self.lexical(features[..., :features.shape[-1] // 2].float()))
        return pointers.squeeze(-1).masked_fill(~mask, MASK), self.abstain(summary)

    def slot_scores(self, key, bank, abstain):
        scores = (bank.keys * key.unsqueeze(-2)).sum(-1) * self.match_scale.clamp(5, 60)
        scores = scores.masked_fill(~bank.occupied, MASK)
        return torch.cat((scores, abstain + 6.0), dim=-1)


def _action(logits, mode, generator, teacher=None, temperature=1.0):
    if mode == "sample":
        logits = logits / temperature
    log_prob = logits.log_softmax(-1)
    probability = log_prob.exp()
    if mode == "teacher":
        choice = teacher
    elif mode == "sample":
        flat = probability.detach().reshape(-1, probability.shape[-1])
        choice = torch.multinomial(flat, 1, generator=generator).reshape(probability.shape[:-1])
    else:
        choice = logits.argmax(-1)
    selected = log_prob.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_prob * probability).sum(-1)
    return choice, selected, entropy


def _selected(rows, indexes):
    return rows.gather(-2, indexes[..., None, None].expand(*indexes.shape, 1, rows.shape[-1])).squeeze(-2)


def write_slots(bank, admit, key):
    related = ((bank.keys * key.unsqueeze(-2)).sum(-1) >= .80) & bank.occupied
    matching = related.to(torch.int8).argmax(-1)
    free = (~bank.occupied).to(torch.int8).argmax(-1)
    slot = torch.where(related.any(-1), matching, free)
    # Capacity overflow abstains in this pilot. Training has fewer than eight facts.
    valid = admit.bool() & (related.any(-1) | ~bank.occupied.all(-1))
    return F.one_hot(slot, CAPACITY).bool() & valid.unsqueeze(-1)


def replace_slots(bank, selected, key, value):
    return Bank(torch.where(selected.unsqueeze(-1), key.unsqueeze(-2), bank.keys),
                torch.where(selected.unsqueeze(-1), value.unsqueeze(-2), bank.values),
                bank.occupied | selected)


def update_bank(bank, admit, key, value):
    """Apply the chosen write. Content matching replaces the addressed slot."""
    return replace_slots(bank, write_slots(bank, admit, key), key, value)


def write_action(policy, features, mask):
    admission, spans = policy.write_scores(features, mask)
    return admission.argmax(-1), spans.argmax(-1)


def write_one(policy, bank, features, mask, keys, values):
    admit, positions = write_action(policy, features, mask)
    return update_bank(bank, admit, _selected(keys, positions[..., 0]), _selected(values, positions[..., 1])), admit


def read_action(policy, bank, features, mask, keys):
    pointers, abstain = policy.read_scores(features, mask)
    key = _selected(keys, pointers.argmax(-1))
    return policy.slot_scores(key, bank, abstain).argmax(-1)


def read_one(policy, bank, features, mask, keys):
    action = read_action(policy, bank, features, mask, keys)
    return _selected(bank.values, action.clamp(max=CAPACITY - 1)), action.ne(CAPACITY)


def make_rollout(value_catalog, *, control=None):
    """Create the scorer. Reference labels are used only by reward or imitation."""
    def rollout(policy, batch, *, mode, generator=None):
        bank = Bank.empty(len(batch["writes"]), batch["write_keys"].shape[-1],
                          batch["write_values"].shape[-1], batch["writes"].device)
        scores, entropies = [], []
        snapshots = []
        write_admission, write_spans = policy.write_scores(batch["writes"], batch["write_mask"])
        teach_admit = batch["write_admit"] if mode == "teacher" else None
        teach_spans = batch["write_spans"] if mode == "teacher" else None
        admits, admission_lp, admission_h = _action(write_admission, mode, generator, teach_admit, temperature=3.0)
        spans, spans_lp, spans_h = _action(write_spans, mode, generator, teach_spans)
        scores.append(admission_lp + spans_lp.sum(-1) * admits)
        entropies.append(admission_h + spans_h.sum(-1) * admits)
        keys = _selected(batch["write_keys"], spans[..., 0])
        values = _selected(batch["write_values"], spans[..., 1])
        for step in range(keys.shape[1]):
            bank = update_bank(bank, admits[:, step], keys[:, step], values[:, step])
            snapshots.append(bank)

        rows = torch.arange(len(keys), device=keys.device)[:, None]
        time = batch["after_write"] - 1
        visible = Bank(*(torch.stack([getattr(bank, field) for bank in snapshots], dim=1)[rows, time]
                         for field in ("keys", "values", "occupied")))
        pointer, abstain = policy.read_scores(batch["queries"], batch["query_mask"])
        teacher_pointer = batch["query_spans"] if mode == "teacher" else None
        position, pointer_lp, pointer_h = _action(pointer, mode, generator, teacher_pointer)
        query_keys = _selected(batch["query_keys"], position)
        if control == "empty":
            visible = Bank(visible.keys, visible.values, torch.zeros_like(visible.occupied))
        if control == "shuffled_queries":
            query_keys = query_keys.roll(1, dims=0)
        slot_scores = policy.slot_scores(query_keys, visible, abstain)
        expected = batch["targets"]
        if mode == "teacher":
            related = (visible.keys * query_keys.unsqueeze(-2)).sum(-1)
            related = related.masked_fill(~visible.occupied, MASK)
            teacher_slot = torch.where(expected.eq(len(value_catalog)), CAPACITY, related.argmax(-1))
        else:
            teacher_slot = None
        slot, slot_lp, slot_h = _action(slot_scores, mode, generator, teacher_slot)
        scores.append(pointer_lp + slot_lp)
        entropies.append(pointer_h + slot_h)

        retrieved = _selected(visible.values, slot.clamp(max=CAPACITY - 1))
        similarity, nearest = (retrieved @ value_catalog.T).max(-1)
        answers = torch.where(slot.eq(CAPACITY), len(value_catalog),
                              torch.where(similarity > .995, nearest, -1))
        correct = answers.eq(expected)
        # Unsupported guesses and stale values cost more than abstaining.
        per_query = torch.where(correct, 1.0, torch.where(slot.eq(CAPACITY), -.35, -1.0))
        # A small cost for each write discourages gratuitous writes. Future
        # recall outcomes supply the credit for admitting useful observations.
        rewards = per_query.mean(-1) - .12 * admits.float().mean(-1)
        metrics = None
        if mode == "greedy":
            metrics = {"overall": (int(correct.sum()), correct.numel()),
                       "write_admission": (int(admits.eq(batch["write_admit"]).sum()), admits.numel()),
                       "transient_written": (int((admits.bool() & batch["write_transient"]).sum()),
                                             int(batch["write_transient"].sum())),
                       "unsupported_false_positive": (int((~correct & expected.eq(len(value_catalog))).sum()),
                                                      int(expected.eq(len(value_catalog)).sum()))}
            for index, label in enumerate(QUERY_KINDS):
                subset = batch["query_kind"].eq(index)
                metrics[label] = (int((correct & subset).sum()), int(subset.sum()))
        return Trajectory(torch.cat(scores, -1), torch.cat(entropies, -1), rewards, metrics)
    return rollout
