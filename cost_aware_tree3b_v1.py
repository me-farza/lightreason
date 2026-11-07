"""Cost-aware, genuinely packed tree families for the unchanged BF16 Qwen target.

This is a separate experimental runner. Existing target, auxiliary, cache
commit, greedy acceptance, EOS, and frozen controls are not rewritten.
Budgets count physical rows INCLUDING the pending root (CAST uses N+1).
A measured complete-cycle cost curve selects among separately captured M-row
verifier graphs. No compute reduction is claimed for masking a larger graph.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import math
import time
from types import FunctionType

from graph_prefill_compiled_pard2_lora_tree3b_v1 import GraphPrefillCompiledPARD2LoRATree3BRunner

PACKED_BUDGETS = (16, 32, 48, 64, 96, 128)


def checked_budgets(values):
    values = tuple(values)
    if not values or any(type(v) is not int or v not in PACKED_BUDGETS for v in values):
        raise ValueError("Use actual packed row counts from 16,32,48,64,96,128")
    if len(set(values)) != len(values):
        raise ValueError("Packed row counts must be unique")
    return tuple(sorted(values))


def public_packed_families(prompt_length, output_budget, shapes, max_len, attention_buckets):
    """All legal actual M/bucket families; reserve remains the control's 64."""
    if any(type(v) is not int for v in (prompt_length, output_budget, max_len)) or prompt_length < 1 or output_budget < 1:
        raise ValueError("Use positive public prompt/output capacities")
    if prompt_length + output_budget + 64 > max_len:
        raise ValueError("Prompt/output must fit the frozen control's 64-row reserve")
    shapes = checked_budgets(shapes)
    buckets = tuple(attention_buckets)
    if not buckets or tuple(sorted(set(buckets))) != buckets or buckets[-1] != max_len:
        raise ValueError("Use sorted immutable attention buckets ending at max_len")
    families = []
    if output_budget == 1:
        return families
    for size in shapes:
        last_base = min(prompt_length + output_budget - 2, max_len - size)
        if last_base < prompt_length:
            continue
        first_end, last_end = prompt_length + size, last_base + size
        first_bucket = next(b for b in buckets if b >= first_end)
        last_bucket = next(b for b in buckets if b >= last_end)
        families.extend((size, b) for b in buckets if first_bucket <= b <= last_bucket)
    return sorted(families)


def capacity_budget(preferred, available_rows, fallback):
    if any(type(v) is not int or v < 1 for v in (preferred, available_rows, fallback)):
        raise ValueError("Use positive integer physical capacity/budgets")
    if preferred <= available_rows:
        return preferred, False
    if fallback <= available_rows:
        return fallback, True
    raise RuntimeError("No prepared physical verifier family fits remaining cache capacity")


def checkpoint_training_depth(metadata):
    """The admitted checkpoint already exposes its validated training identity."""
    try:
        depth = metadata["fold"]["loaded_adaptation"]["validated_identity"]["training_config"]["draft_tokens"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing admitted auxiliary checkpoint training draft depth") from exc
    if type(depth) is not int or not 1 <= depth <= 32:
        raise ValueError("Invalid checkpoint training draft depth")
    return depth


def select_cost_aware_budget(log_scores, cycle_costs_ms, budgets=PACKED_BUDGETS):
    """CAST plug-in expected commits / measured COMPLETE cycle time.

    log_scores include root zero. Products of position marginals rank proposals,
    not target probabilities; calibration against accepted paths is separate.
    The exact measured curve permits non-affine kernel-boundary cost jumps.
    """
    budgets = checked_budgets(budgets)
    if len(log_scores) < budgets[-1]:
        raise ValueError("Score the full configured candidate universe first")
    previous = math.inf
    scores = []
    for value in log_scores:
        value = float(value)
        if not math.isfinite(value) or value > 1e-5 or value > previous + 1e-10:
            raise ValueError("Use finite best-first nonincreasing log prefix masses")
        value = min(0.0, value)
        scores.append(value)
        previous = value
    if abs(scores[0]) > 1e-8:
        raise ValueError("Root score must be zero")
    mass, by_size = 1.0, {}
    for index in range(1, budgets[-1]):
        mass += math.exp(scores[index])
        if index + 1 in budgets:
            by_size[index + 1] = mass
    candidates = []
    for size in budgets:
        cost = cycle_costs_ms.get(size, cycle_costs_ms.get(str(size)))
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError("Every packed shape needs a positive measured complete-cycle cost")
        candidates.append({"packed_rows": size, "predicted_committed_tokens": by_size[size],
                           "complete_cycle_cost_ms": float(cost),
                           "predicted_tokens_per_ms": by_size[size] / cost})
    # Deterministic tie favors less target compute, without changing acceptance.
    best = max(candidates, key=lambda row: (row["predicted_tokens_per_ms"], -row["packed_rows"]))
    return best["packed_rows"], candidates


def private_bucket_forward(function, suffix):
    """Private code identity; avoid a process-global Dynamo limit change."""
    if not isinstance(function, FunctionType):
        raise ValueError("Use the immutable bucket's Python forward closure")
    code = function.__code__.replace(co_name=function.__code__.co_name + "_" + str(suffix))
    result = FunctionType(code, function.__globals__, code.co_name,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    return result


def ranking_calibration(traces, budgets=PACKED_BUDGETS):
    """Offline score calibration. This never changes final target acceptance.

    Projecting a large observed path onto a prefix is a ranking diagnostic,
    NOT a substitute for timing/equivalence of actual smaller BF16 graphs.
    EOS/output-budget-censored rounds are omitted from the estimates.
    """
    budgets = checked_budgets(budgets)
    boundaries = (0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.00000001)
    bins = [{"lower": lo, "upper": hi, "nodes": 0, "proposal_mass_sum": 0.0,
             "accepted_nodes": 0} for lo, hi in zip(boundaries, boundaries[1:])]
    depths, projected = {}, {m: [] for m in budgets}
    usable, censored = 0, 0
    for row in traces:
        if row.get("output_budget_or_EOS_can_censor_acceptance"):
            censored += 1
            continue
        count = row["packed_rows"]
        masses, accepted, row_depths = row["log_prefix_masses"], row["accepted_nonroot_nodes"], row["depths"]
        if not (len(masses) == len(accepted) == len(row_depths) == count):
            raise ValueError("Ranking trace arrays must cover every actual physical node")
        usable += 1
        for index in range(1, count):
            mass = math.exp(masses[index])
            if not 0 <= mass <= 1.000001 or type(accepted[index]) is not bool:
                raise ValueError("Malformed proposal mass/accepted-prefix trace")
            mass = min(mass, 1.0)
            item = next(item for item in bins if item["lower"] <= mass < item["upper"])
            item["nodes"] += 1
            item["proposal_mass_sum"] += mass
            item["accepted_nodes"] += accepted[index]
            depth = depths.setdefault(row_depths[index], {"nodes": 0, "accepted_nodes": 0})
            depth["nodes"] += 1
            depth["accepted_nodes"] += accepted[index]
        for size in budgets:
            if size <= count:
                projected[size].append(1 + sum(accepted[1:size]))
    for item in bins:
        item["mean_proposal_prefix_mass"] = item["proposal_mass_sum"] / item["nodes"] if item["nodes"] else None
        item["observed_accepted_prefix_frequency"] = item["accepted_nodes"] / item["nodes"] if item["nodes"] else None
    for value in depths.values():
        value["observed_accepted_prefix_frequency"] = value["accepted_nodes"] / value["nodes"]
    return {"uncensored_cycles": usable, "censored_cycles_excluded": censored,
            "proposal_prefix_mass_bins": bins, "acceptance_by_depth": depths,
            "projected_prefix_budget_commits": {size: {"support_cycles": len(values),
                "mean_committed_tokens": sum(values) / len(values) if values else None}
                for size, values in projected.items()},
            "projection_is_ranking_diagnostic_only": True,
            "physical_smaller_shape_timing_or_numerical_equivalence_claimed": False,
            "marginal_products_are_not_target_conditional_probabilities": True}


class CostAwareTree3BRunner(GraphPrefillCompiledPARD2LoRATree3BRunner):
    """Fixed shape first; optional complete-cost-calibrated per-round dispatch.

    from_control shares frozen trained models but owns the target cache and
    graph families. Use sequentially: the shared auxiliary's request cache is
    reset by every generation. No caller may overlap the two runners.
    """
    def __init__(self, *args, packed_budgets=None, cycle_costs_ms=None,
                 trace_ranking=False, **kwargs):
        fixed = kwargs.get("node_budget", 64)
        budgets = checked_budgets(packed_budgets if packed_budgets is not None else (fixed,))
        kwargs["node_budget"] = max(budgets)
        super().__init__(*args, **kwargs)
        self._shared_control = None
        self._isolate_bucket_code(self.verifier._owner)
        self._configure_cost_policy(budgets, cycle_costs_ms, trace_ranking,
                                    fixed if fixed in budgets else budgets[-1])

    @classmethod
    def from_control(cls, control, *, node_budget=64, packed_budgets=None,
                     cycle_costs_ms=None, trace_ranking=False):
        import torch
        from conditioned_pard2_tree3b_v1 import _VerifiedFeatureBoundProposer
        from feature4_3b_tree_verifier import Feature4BTreeVerifier
        from eager_hf_prefill_graph_v1 import TargetFeature4PrefillGraph
        from graph_prefill_compiled_pard2_lora_tree3b_v1 import _TargetPrefillProxy
        budgets = checked_budgets(packed_budgets if packed_budgets is not None else (node_budget,))
        if control.verifier.pending is not None:
            raise RuntimeError("Finish the control's pending verification first")
        result = cls.__new__(cls)
        result.model, result.tokenizer = control.model, control.tokenizer
        result.max_len, result.node_budget = control.max_len, max(budgets)
        result.draft_tokens, result.top_k, result.tree_policy = control.draft_tokens, control.top_k, control.tree_policy
        result.prefill_cache_lengths = control.prefill_cache_lengths
        original_owner = control.verifier._owner
        owner = Feature4BTreeVerifier(result.model, max_len=result.max_len,
            node_budget=result.node_budget, buckets=original_owner.attention_buckets,
            cuda_graph=original_owner.cuda_graph, compile_static=original_owner.compile_static,
            share_pool=True, warmups=original_owner.warmups,
            validate_topology=original_owner.validate_topology,
            validate_acceptance=original_owner.validate_acceptance,
            commit_engine=original_owner.commit_engine)
        result._isolate_bucket_code(owner)
        result.target_prefill_component = TargetFeature4PrefillGraph(owner, warmups=2)
        result.verifier = _TargetPrefillProxy(owner, result.target_prefill_component)
        result.proposer = _VerifiedFeatureBoundProposer(control.proposer.conditioned, owner)
        with torch.inference_mode(False), torch.no_grad():
            result.visible = torch.empty((1, result.max_len), device=result.model.device, dtype=torch.long)
        result.setup_s, result.prefill_setup_s, result.prepared = 0.0, 0.0, False
        result._prefill_lengths = OrderedDict()
        result.prefill_bank_metadata = {"maximum_retained_target_aux_length_pairs": result.prefill_cache_lengths,
            "retained_public_target_lengths": [], "captures": 0, "evicted_pairs": 0, "setup_s": 0.0,
            "requests": [], "auxiliary_bank_owned_by_frozen_control": True}
        result.metadata = {**control.metadata, "target_verifier": owner.metadata,
                           "prefill_graph_bank": result.prefill_bank_metadata}
        result._shared_control = control
        result._configure_cost_policy(budgets, cycle_costs_ms, trace_ranking,
                                     node_budget if node_budget in budgets else budgets[-1])
        return result

    @staticmethod
    def _isolate_bucket_code(owner):
        import torch
        if owner.entries or owner.graphs:
            raise RuntimeError("Private compiler closures must be installed before graph preparation")
        for bucket, eager in list(owner._eager_bucket_forwards.items()):
            private = private_bucket_forward(eager, f"packed_{id(owner)}_{bucket}")
            owner._eager_bucket_forwards[bucket] = private
            owner._bucket_forwards[bucket] = (torch.compile(private, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False}) if owner.compile_static else private)
        owner.metadata["private_per_bucket_compiler_code_identity"] = True
        owner.metadata["global_compiler_configuration_changed"] = False

    def _configure_cost_policy(self, budgets, costs, trace_ranking, fixed):
        supported = checkpoint_training_depth(self.proposer.conditioned.metadata)
        if not 1 <= self.draft_tokens <= supported:
            raise ValueError(f"Draft depth {self.draft_tokens} exceeds checkpoint-trained depth {supported}")
        if self.tree_policy != "best-first" and len(budgets) > 1 and costs is not None:
            raise ValueError("CAST marginal-mass adaptation requires best-first candidates")
        self.packed_budgets, self.fixed_packed_budget = budgets, fixed
        self.capacity_fallback_budget = min(64, max(budgets))
        self.prepared_packed_budgets = tuple(sorted(set(budgets) | {self.capacity_fallback_budget}))
        self.cycle_costs_ms = dict(costs) if costs is not None else None
        self.trace_ranking, self.stage_hook = bool(trace_ranking), None
        self.last_cycles, self.last_ranking_trace = [], []
        self.metadata = {**self.metadata,
            "variant": "original_BF16_target_cost_aware_actual_packed_tree_v1",
            "node_budget": self.node_budget,
            "packed_budgets_including_root": list(budgets),
            "prepared_packed_budgets_including_capacity_fallback": list(self.prepared_packed_budgets),
            "capacity_fallback_budget": self.capacity_fallback_budget,
            "request_output_capacity_reserve_unchanged": 64,
            "capacity_fallback_uses_public_current_cache_position_only": True,
            "checkpoint_trained_draft_depth": supported,
            "cost_curve_scope": "measured complete speculative cycle, not target forward alone",
            "fixed_packed_budget": fixed,
            "target_controller_inherited_unchanged": False,
            "original_greedy_accept_path_inherited_unchanged": True,
            "original_cache_commit_inherited_unchanged": True,
            "actual_smaller_verifier_graphs_required": True,
            "unchanged_target_weight_values_BF16": True,
            "shared_auxiliary_models_sequential_requests_only": self._shared_control is not None,
            "candidate_masses_are_uncalibrated_proposal_scores": True}

    def execution_signature(self):
        """CPU metadata only; no replay, allocation, model call, or synchronization."""
        auxiliary = self.proposer.conditioned
        def bank(mapping):
            return tuple((repr(key), id(value), id(getattr(value, "graph", value)))
                         for key, value in sorted(mapping.items(), key=lambda item: repr(item[0])))
        signature = {
            "target_owner": id(self.verifier._owner),
            "target_cache": id(self.verifier.cache),
            "target_graphs": bank(self.verifier.graphs),
            "target_entries": bank(self.verifier.entries),
            "auxiliary_owner": id(auxiliary),
            "auxiliary_graphs": bank(auxiliary.graphs),
            "target_prefill_entries": bank(self.target_prefill_component.graph.entries),
            "auxiliary_prefill_entries": bank(auxiliary.prefill_component.graph.entries),
            "packed_families": tuple(sorted(self.verifier.graphs)),
            "prepared_packed_budgets": self.prepared_packed_budgets,
        }
        if self._shared_control is not None:
            signature["shared_control_target_prefill_entries"] = bank(
                self._shared_control.target_prefill_component.graph.entries)
        return signature

    def select_fixed_budget(self, packed_rows):
        if packed_rows not in self.packed_budgets:
            raise ValueError("Prepare/configure the true packed family before selecting it")
        self.fixed_packed_budget, self.cycle_costs_ms = packed_rows, None
        self.metadata["fixed_packed_budget"] = packed_rows

    def set_cycle_costs_ms(self, costs):
        if self.tree_policy != "best-first":
            raise ValueError("Adaptive marginal-mass selection requires best-first candidates")
        # Validate the entire measured curve before changing the policy.
        select_cost_aware_budget([0.0] * max(self.packed_budgets), costs, self.packed_budgets)
        self.cycle_costs_ms = dict(costs)

    def _stage(self, name):
        return nullcontext() if self.stage_hook is None else self.stage_hook(name)

    def _prepare_public_prefills(self, length):
        if self._shared_control is None:
            return super()._prepare_public_prefills(length)
        # The auxiliary graph bank stays under ONE owner's eviction accounting.
        # Never evict an auxiliary entry behind the frozen control's bookkeeping.
        import torch
        shared = self._shared_control._prepare_public_prefills(length)
        began = time.perf_counter()
        target_new = 0
        if length in self._prefill_lengths:
            self._prefill_lengths.move_to_end(length)
        else:
            if len(self._prefill_lengths) >= self.prefill_cache_lengths:
                old, _ = self._prefill_lengths.popitem(last=False)
                self.target_prefill_component.graph.entries.pop(old, None)
                self.prefill_bank_metadata["evicted_pairs"] += 1
            info = self.target_prefill_component.prepare(length)
            target_new = int(info.get("new_graph", False))
            self._prefill_lengths[length] = None
        self.verifier.reset_for_new_request()
        self.proposer.reset()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        self.prefill_setup_s += elapsed
        self.target_prefill_component.graph.metadata["prepared_lengths"] = sorted(
            self.target_prefill_component.graph.entries)
        self.prefill_bank_metadata.update(retained_public_target_lengths=list(self._prefill_lengths),
            setup_s=self.prefill_setup_s, captures=self.prefill_bank_metadata["captures"] + target_new)
        return {"setup_s": elapsed + shared["setup_s"],
                "new_graphs": target_new + shared.get("new_graphs", 0),
                "shared_auxiliary_bank": shared, "public_target_length": length}

    def _check(self, ids, n):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Only one nonempty unpadded original-tokenized prompt is supported")
        if type(n) is not int or n < 1:
            raise ValueError("Use the unchanged positive public output budget")
        if ids.shape[1] + n + 64 > self.max_len:
            raise ValueError("Prompt/output must fit the frozen control's unchanged 64-row reserve")

    def prepare(self, ids, n):
        import torch
        self._check(ids, n)
        if self.verifier.pending is not None:
            raise RuntimeError("Finish pending verification before graph preparation")
        self.verifier.reset_for_new_request()
        self.proposer.reset()
        torch.cuda.synchronize()
        began = time.perf_counter()
        length = ids.shape[1]
        families = public_packed_families(length, n, self.prepared_packed_budgets,
                                         self.max_len, self.verifier.attention_buckets)
        target = self.verifier._prepare_families(families) if families else {"new_graphs": 0}
        draft_buckets = self._reachable(self.proposer, length + self.draft_tokens,
                                        length + n + self.draft_tokens - 2)
        draft = self.proposer.prepare(self.draft_tokens, buckets=draft_buckets) if n > 1 else {"new_graphs": 0}
        prefill = self._prepare_public_prefills(length)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        self.setup_s += elapsed
        self.prepared = True
        return {"setup_s": elapsed, "target": target, "draft": draft, "prefill_graphs": prefill,
                "new_graphs": target.get("new_graphs", 0) + draft.get("new_graphs", 0) + prefill.get("new_graphs", 0),
                "actual_packed_shapes": list(self.prepared_packed_budgets)}

    def _make_tree(self, anchor, top_ids, logps, available_rows=None):
        from pard_tree import best_first_tree
        from pard_tree_policies import trunk_best_first_tree
        adaptive = self.cycle_costs_ms is not None
        available_rows = self.max_len if available_rows is None else available_rows
        legal_budgets = tuple(size for size in self.packed_budgets if size <= available_rows)
        if adaptive and legal_budgets:
            maximum = max(legal_budgets)
            capacity_fallback = False
        else:
            maximum, capacity_fallback = capacity_budget(self.fixed_packed_budget,
                available_rows, self.capacity_fallback_budget)
            adaptive = False
        builder = best_first_tree if self.tree_policy == "best-first" else trunk_best_first_tree
        tree = builder(anchor, top_ids, logps, node_budget=maximum)
        if tree["node_count"] != maximum:
            raise RuntimeError("Finite lattice did not fill the configured physical verifier shape")
        diagnostics = None
        if adaptive:
            selected, diagnostics = select_cost_aware_budget(tree["log_scores"].tolist(),
                                                            self.cycle_costs_ms, legal_budgets)
            if selected != maximum:
                tree = {**tree, "tokens": tree["tokens"][:, :selected].contiguous(),
                    "parents": tree["parents"][:selected].contiguous(),
                    "depths": tree["depths"][:selected].contiguous(),
                    "ancestors": tree["ancestors"][:selected, :selected].contiguous(),
                    "log_scores": tree["log_scores"][:selected].contiguous(),
                    "node_count": selected, "node_budget": selected}
        tree["capacity_fallback"] = capacity_fallback
        return tree, diagnostics

    def run(self, ids, n, eos_ids=None, stream_callback=None):
        import torch
        self._check(ids, n)
        prepared = self.prepare(ids, n)
        eos = set(eos_ids or [])
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.proposer.reset()
            self.visible[:, :ids.shape[1]].copy_(ids)
            with self._stage("prefill"):
                anchor = self.verifier.prefill(ids)
                self.visible[:, ids.shape[1]:ids.shape[1] + 1].copy_(anchor)
                tokens = [int(anchor.item())]
            if stream_callback is not None:
                with self._stage("streaming"):
                    stream_callback(tokens.copy())
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            cycles, ranking = [], []
            while len(tokens) < n and (not eos or tokens[-1] not in eos):
                cycle_start = time.perf_counter()
                visible_length = ids.shape[1] + len(tokens)
                with self._stage("drafting"):
                    top_ids, logps = self.proposer.propose_lattice(self.visible[:, :visible_length], self.draft_tokens)
                with self._stage("lattice_transfer"):
                    cpu_ids, cpu_logps = top_ids.detach().cpu(), logps.detach().cpu()
                with self._stage("selection_tree"):
                    tree, cost_diagnostics = self._make_tree(tokens[-1], cpu_ids, cpu_logps,
                        self.max_len - self.verifier.current_position)
                with self._stage("verification"):
                    predicted = self.verifier.verify_tree(tree["tokens"], tree["depths"],
                                                         tree["ancestors"], self.verifier.current_position)
                with self._stage("prediction_transfer"):
                    predictions = predicted.reshape(-1).cpu().tolist()
                with self._stage("acceptance"):
                    path, emitted = self.accept_path(tree, predictions, n - len(tokens), eos)
                if not emitted or len(path) != len(emitted):
                    raise RuntimeError("Rooted original-target verification failed to advance")
                committed = len(emitted)
                with self._stage("visible_update"):
                    selected = torch.tensor(path, device=self.model.device, dtype=torch.long)
                    self.visible[:, visible_length:visible_length + committed].copy_(
                        predicted.index_select(0, selected).view(1, -1))
                    tokens.extend(emitted)
                with self._stage("cache_commit"):
                    self.verifier.commit_path(path)
                with self._stage("anchor_sync"):
                    if int(self.verifier.anchor.item()) != tokens[-1]:
                        raise RuntimeError("Compacted target cache/anchor invariant failed")
                if stream_callback is not None:
                    with self._stage("streaming"):
                        stream_callback(tokens.copy())
                cycle = {"node_count": tree["node_count"], "committed_tokens": committed,
                         "matched_draft_tokens": committed - 1, "accepted_node_indices": path,
                         "cycle_wall_s": time.perf_counter() - cycle_start,
                         "capacity_fallback": tree["capacity_fallback"]}
                if cost_diagnostics is not None:
                    cycle["cost_policy_candidates"] = cost_diagnostics
                if self.trace_ranking:
                    accepted = set(path[1:])
                    ranking.append({"cycle": len(cycles), "packed_rows": tree["node_count"],
                        "depths": tree["depths"].tolist(), "log_prefix_masses": tree["log_scores"].tolist(),
                        "accepted_nonroot_nodes": [i in accepted for i in range(tree["node_count"])],
                        "actual_committed_tokens": committed,
                        "output_budget_or_EOS_can_censor_acceptance": len(tokens) >= n or tokens[-1] in eos})
                cycles.append(cycle)
            torch.cuda.synchronize()
            end = time.perf_counter()
        self.verifier.assert_target_unchanged()
        self.proposer.assert_draft_unchanged()
        decode_s = end - prefill_end
        self.last_cycles, self.last_ranking_trace = cycles, ranking
        return tokens, {"prefill_s": prefill_end - start, "decode_s": decode_s,
            "total_s": end - start, "decode_tokens_s": (len(tokens) - 1) / decode_s if len(tokens) > 1 else 0.0,
            "generated_tokens": len(tokens), "cycles": len(cycles),
            "mean_committed_tokens_per_cycle": (len(tokens) - 1) / len(cycles) if cycles else 0.0,
            "mean_cycle_ms": 1000.0 * decode_s / len(cycles) if cycles else 0.0,
            "verified_nodes": sum(c["node_count"] for c in cycles),
            "actual_packed_rows_used": sorted(set(c["node_count"] for c in cycles)),
            "capacity_fallback_cycles": sum(c["capacity_fallback"] for c in cycles),
            "excluded_setup_s": prepared["setup_s"], "cumulative_setup_s": self.setup_s,
            "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens" if eos else "fixed_budget",
            "all_output_tokens_target_verified": True,
            "streaming_callback_enabled": stream_callback is not None,
            "decode_includes_drafting_tree_building_verification_compaction_and_cpu_transfers": True,
            "variant": self.metadata["variant"]}

    def run_stream(self, ids, n, eos_ids=None, stream_callback=None):
        if stream_callback is None:
            raise ValueError("run_stream requires an observer callback")
        return self.run(ids, n, eos_ids=eos_ids, stream_callback=stream_callback)
