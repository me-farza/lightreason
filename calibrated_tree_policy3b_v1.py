"""Development-only proposal-ranking policies over the unchanged PARD lattice.

The auxiliary still emits the same top-K IDs and full-vocabulary log masses.
Retained normalization and temperature alter ONLY CPU tree priorities. They do
not calibrate true target conditional probabilities and confer no output-quality
claim. The original BF16 target still verifies every emitted token, EOS, and
persistent cache input. Existing frozen source and weights remain untouched.
"""
from __future__ import annotations

import math

from cost_aware_tree3b_v1 import CostAwareTree3BRunner

RANKING_POLICIES = ("original", "retained")
RETAINED_TEMPERATURES = (0.5, 0.75, 1.0)


def validate_ranking_policy(policy, temperature):
    if policy not in RANKING_POLICIES:
        raise ValueError("Use ranking_policy='original' or 'retained'")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("Use a finite ranking temperature")
    temperature = float(temperature)
    if policy == "original" and temperature != 1.0:
        raise ValueError("Original ranking is an exact no-op at temperature1 only")
    if policy == "retained" and temperature not in RETAINED_TEMPERATURES:
        raise ValueError("Use the predefined retained-ranking temperatures0.5,0.75,1.0")
    return policy, temperature


def calibrate_lattice_logps(rows, policy="original", temperature=1.0):
    """Return proposal-only log priorities without changing candidate IDs.

    For retained ranking, each row becomes log-softmax(original_logp / T).
    Subtracting the retained-set log mass discards unknown outside-top-K mass.
    Sharpening can favor deeper paths; only measured target acceptance can
    establish whether it improves complete-cycle throughput.
    """
    policy, temperature = validate_ranking_policy(policy, temperature)
    rows = [list(row) for row in rows]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Use a nonempty rectangular draft-depth by top-K lattice")
    checked = []
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Use numeric marginal log probabilities")
            value = float(value)
            if math.isnan(value) or value == math.inf or value > 1e-5:
                raise ValueError("Marginal log probabilities must be nonpositive; -infinity is allowed")
            values.append(value)
        if not any(math.isfinite(value) for value in values):
            raise ValueError("Every retained row needs at least one finite candidate")
        checked.append(values)
    if policy == "original":
        return checked
    transformed = []
    for row in checked:
        # Stable for tiny original retained mass, sharp temperatures, and -inf.
        scaled = [min(value, 0.0) / temperature for value in row]
        maximum = max(scaled)
        shifted = [value - maximum for value in scaled]
        normalizer = math.log(sum(math.exp(value) for value in shifted))
        transformed.append([min(0.0, value - normalizer) for value in shifted])
    return transformed


def _cpu_logps_matrix(value):
    """Read CPU inputs only; no hidden device transfer in tree construction."""
    if hasattr(value, "device"):
        if value.device.type != "cpu":
            raise ValueError("Transfer lattice log probabilities in the timed lattice_transfer stage")
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError("Use CPU log probabilities[B,K] or[1,B,K]")
        return value.tolist()
    rows = list(value)
    if len(rows) == 1 and rows[0] and isinstance(rows[0][0], (list, tuple)):
        rows = rows[0]
    return [list(row) for row in rows]


class CalibratedTreePolicy3BRunner(CostAwareTree3BRunner):
    """Actual packed-family runner with separately labeled proposal priorities."""
    def __init__(self, *args, ranking_policy="original", ranking_temperature=1.0, **kwargs):
        validate_ranking_policy(ranking_policy, ranking_temperature)
        super().__init__(*args, **kwargs)
        self.set_ranking_policy(ranking_policy, ranking_temperature)

    @classmethod
    def from_control(cls, control, *, ranking_policy="original", ranking_temperature=1.0,
                     tree_policy=None, **kwargs):
        validate_ranking_policy(ranking_policy, ranking_temperature)
        result = super().from_control(control, **kwargs)
        result.set_ranking_policy(ranking_policy, ranking_temperature, tree_policy=tree_policy)
        return result

    def set_ranking_policy(self, policy, temperature=1.0, *, tree_policy=None):
        policy, temperature = validate_ranking_policy(policy, temperature)
        chosen_tree_policy = self.tree_policy if tree_policy is None else tree_policy
        if chosen_tree_policy not in ("best-first", "trunk"):
            raise ValueError("Use best-first or the already existing mandatory-trunk policy")
        if chosen_tree_policy == "trunk":
            if self.cycle_costs_ms is not None:
                raise ValueError("Fixed-budget trunk retests do not use CAST best-first mass adaptation")
            if any(size < self.draft_tokens + 1 for size in self.prepared_packed_budgets):
                raise ValueError("Every trunk family must hold the entire supported draft plus root")
        self.ranking_policy, self.ranking_temperature = policy, temperature
        self.tree_policy = chosen_tree_policy
        self.metadata = {**self.metadata,
            "variant": "original_BF16_target_proposal_ranking_calibration_v1",
            "ranking_policy": policy,
            "ranking_temperature": temperature,
            "tree_policy": chosen_tree_policy,
            "existing_mandatory_trunk_policy_reused": chosen_tree_policy == "trunk",
            "ranking_score_label": ("original_full_vocabulary_marginal_log_probability" if policy == "original"
                                    else "temperature_scaled_retained_topK_log_probability_proposal_only"),
            "ranking_transform_scope": "CPU candidate priorities only; unchanged original auxiliary lattice",
            "ranking_original_noop": policy == "original",
            "original_auxiliary_logits_IDs_and_weights_unchanged": True,
            "original_target_logits_weight_values_BF16_and_acceptance_unchanged": True,
            "ranking_scores_are_not_target_conditional_probabilities": True,
            "retained_probability_calibrated_to_target_claimed": False,
            "candidate_ranking_acceptance_or_speed_improvement_requires_measurement": True}

    def _make_tree(self, anchor, top_ids, logps, available_rows=None):
        priorities = logps
        if self.ranking_policy != "original":
            priorities = calibrate_lattice_logps(_cpu_logps_matrix(logps),
                                                  self.ranking_policy, self.ranking_temperature)
        tree, diagnostics = super()._make_tree(anchor, top_ids, priorities, available_rows)
        tree["ranking_policy"] = self.ranking_policy
        tree["ranking_temperature"] = self.ranking_temperature
        tree["ranking_score_label"] = self.metadata["ranking_score_label"]
        return tree, diagnostics

    def run(self, ids, n, eos_ids=None, stream_callback=None):
        tokens, metrics = super().run(ids, n, eos_ids=eos_ids, stream_callback=stream_callback)
        for row in self.last_ranking_trace:
            row.update(ranking_policy=self.ranking_policy,
                       ranking_temperature=self.ranking_temperature,
                       tree_policy=self.tree_policy,
                       ranking_score_label=self.metadata["ranking_score_label"])
        metrics.update(ranking_policy=self.ranking_policy,
                       ranking_temperature=self.ranking_temperature,
                       tree_policy=self.tree_policy,
                       ranking_score_label=self.metadata["ranking_score_label"],
                       final_acceptance_by_original_target=True)
        return tokens, metrics
