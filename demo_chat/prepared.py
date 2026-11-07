"""Synthetic-only preparation for a real, original-first inference demo.

Preparation warms execution shapes, never a user's answer. Admission checks
the live graph banks, so eviction, a process restart or a new required family
cannot silently turn a presentation into a cold compilation run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

if __package__:
    from .interactive import InteractiveLiveInference
    from .inference import MAX_LEN, TREE_RESERVE, ALLOWED_BUDGETS, require, valid_budget
    from .messages import chat_messages
else:
    from interactive import InteractiveLiveInference
    from inference import MAX_LEN, TREE_RESERVE, ALLOWED_BUDGETS, require, valid_budget
    from messages import chat_messages


DEFAULT_DEMO_PROMPT = "Write a Python function that returns the first n Fibonacci numbers and explain it."
DEFAULT_DEMO_BUDGET = 256


def _source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _buckets(choices, first, last):
    choices = tuple(choices)
    lower = next((v for v in choices if v >= first), None)
    upper = next((v for v in choices if v >= last), None)
    require(lower is not None and upper is not None, "Public demo shape exceeds the prepared cache capacity")
    return tuple(v for v in choices if lower <= v <= upper)


def _identity(value):
    if value is None:
        return None
    graph = value.get("graph") if isinstance(value, dict) else getattr(value, "graph", value)
    return id(value), id(graph) if graph is not None else None


def _all_graphs(runner, mode):
    """CPU metadata only; no replay, synchronization, tensor read or allocation."""
    if runner is None:
        return ()
    if mode == "strict":
        return tuple((repr(key), _identity(value)) for key, value in sorted(runner.graphs.items()))
    aux = runner.proposer.conditioned
    banks = (runner.verifier.graphs, aux.graphs,
             runner.target_prefill_component.graph.entries,
             aux.prefill_component.graph.entries)
    return tuple(tuple((repr(key), _identity(value)) for key, value in sorted(bank.items()))
                 for bank in banks)


def _registered_graphs(runner, mode):
    if mode == "strict":
        banks = (("strict", runner.graphs),)
    else:
        aux = runner.proposer.conditioned
        banks = (("target", runner.verifier.graphs), ("draft", aux.graphs),
                 ("target_prefill", runner.target_prefill_component.graph.entries),
                 ("aux_prefill", aux.prefill_component.graph.entries))
    return {(name, key): _identity(value) for name, bank in banks for key, value in bank.items()}


class PreparedLiveInference(InteractiveLiveInference):
    def initialize(self, callback):
        metadata = super().initialize(callback)
        self._prepared_source_sha = _source_sha()
        self._demo_warm_lengths = {}
        self._demo_admitted_graphs = {}
        self._demo_preparing = False
        self.metadata.update(demo_preparation_supported=True, demo_ready=False,
                             preparation_uses_synthetic_values_only=True,
                             prepared_answers_cached=False)
        return dict(self.metadata)

    def _source_guards(self):
        super()._source_guards()
        require(_source_sha() == self._prepared_source_sha,
                "Demo preparation source changed since startup")

    def _shape(self, prompt, budget, mode):
        require(isinstance(prompt, str) and bool(prompt.strip()) and len(prompt) <= 8000,
                "Enter a nonempty prompt of at most 8,000 characters")
        require(valid_budget(budget), "Output must fit in the available model context")
        require(mode != "original" and mode in self.available_modes, "Choose an available Lightreason runner")
        tokens = self.tokenizer.apply_chat_template(
            chat_messages(prompt), add_generation_prompt=True, tokenize=True)
        length = len(tokens)
        require(length > 0 and length + budget + TREE_RESERVE <= MAX_LEN,
                f"Prompt has {length} tokens. Prompt + {budget} output tokens + 64 tree reserve must fit in {MAX_LEN}.")
        return length

    def _families_ready(self, length, budget, mode):
        admitted = getattr(self, "_demo_admitted_graphs", {}).get(mode, {})
        runner = self.strict if mode == "strict" else self.fast
        if runner is None:
            return False
        live = _registered_graphs(runner, mode)
        if mode == "strict":
            wanted = {((position // 128) + 1) * 128
                      for position in range(length, length + budget - 1)}
            required = [("strict", bucket) for bucket in wanted]
            return all(key in live and live[key] is not None and live[key] == admitted.get(key) and live[key][1] is not None
                       for key in required)
        fast = runner
        aux = fast.proposer.conditioned
        targets = _buckets(fast.verifier.attention_buckets,
                           length + fast.node_budget,
                           length + budget + fast.node_budget - 2)
        drafts = _buckets(aux.attention_buckets,
                          length + fast.draft_tokens,
                          length + budget + fast.draft_tokens - 2)
        target_prefill = fast.target_prefill_component.graph.entries.get(length)
        aux_prefill = aux.prefill_component.graph.entries.get(length + 1)
        if target_prefill is None or aux_prefill is None or getattr(target_prefill, "graph", None) is None or getattr(aux_prefill, "graph", None) is None:
            return False
        required = [("target_prefill", length), ("aux_prefill", length + 1)]
        required += [("target", (fast.node_budget, bucket)) for bucket in targets]
        required += [("draft", (fast.draft_tokens, width, bucket)) for bucket in drafts
                     for width in range(fast.draft_tokens, 2 * fast.draft_tokens + 1)]
        return all(key in live and live[key] is not None and live[key] == admitted.get(key) and live[key][1] is not None
                   for key in required)

    def is_prepared(self, prompt, budget, mode):
        """Read-only CPU admission; call from the HTTP side only while idle."""
        if not self.ready or self.tokenizer is None:
            return {"prepared": False, "demo_ready": False, "detail": "The model is still loading.",
                    "prompt_tokens": None, "budget": budget, "mode": mode,
                    "prepared_modes": [], "setup_s": None, "synthetic_only": True,
                    "answers_cached": False}
        length = self._shape(prompt, budget, mode)
        record = getattr(self, "_demo_warm_lengths", {}).get(length)
        owners_match = bool(record and record["owners"] == (id(self.model), id(self.fast), id(self.strict)))
        required = self.available_modes[1:]
        modes = []
        if owners_match and not getattr(self, "_demo_preparing", False):
            modes = [name for name in required if self._families_ready(length, budget, name)]
        prepared = tuple(modes) == required
        return {"prepared": prepared, "demo_ready": prepared,
                "detail": "Available runners are prepared for this prompt and output limit." if prepared else
                          "Prepare this demo before presenting; its GPU execution shape is not fully warm.",
                "prompt_tokens": length, "budget": budget, "mode": mode,
                "prepared_modes": modes,
                "setup_s": record["setup_s"] if owners_match else None,
                "synthetic_only": True, "answers_cached": False}

    prepared_status = is_prepared

    def _warm_original(self, synthetic):
        from run_numeric_quality import raw_hf_natural
        # No true prompt values, reference tokens, expected answer or EOS
        # distance enter warmup. This result is discarded immediately.
        raw_hf_natural(self.model, synthetic, 3, set())

    def prepare_demo(self, prompt, budget, mode, callback):
        self._thread()
        require(self.ready, "Inference is not ready")
        length = self._shape(prompt, budget, mode)
        self._request_guards()
        self._source_guards()
        import torch
        began = time.perf_counter()
        self._demo_preparing = True
        self.metadata["demo_ready"] = False
        receipts = {}
        try:
            self._status(callback, "preparing_demo", "Preparing the original runner with synthetic inputs…",
                         demo_ready=False, prompt_tokens=length, budget=budget)
            with torch.inference_mode(False), torch.no_grad():
                synthetic = torch.zeros((1, length), device=self.model.device, dtype=torch.long)
                self._warm_original(synthetic)
                torch.cuda.synchronize()
                if self.fast is not None:
                    self._status(callback, "preparing_demo", "Preparing Lightreason Turbo…",
                                 demo_ready=False, prompt_tokens=length, budget=budget)
                    receipts["optimized"] = self.fast.prepare(synthetic, budget)
                    torch.cuda.synchronize()
                self._status(callback, "preparing_demo", "Preparing the exact-output reference runner…",
                             demo_ready=False, prompt_tokens=length, budget=budget)
                self._ensure_strict(callback)
                receipts["strict"] = self.strict.prepare(synthetic, budget)
                torch.cuda.synchronize()
                del synthetic
            self._request_guards()
            self._source_guards()
            require(time.perf_counter() - began <= 600, "Demo preparation exceeded its bounded 600-second segment")
            self._demo_warm_lengths[length] = {
                "owners": (id(self.model), id(self.fast), id(self.strict)),
                "setup_s": time.perf_counter() - began,
                "prepared_budget": budget,
            }
            self._demo_admitted_graphs = {"strict": _registered_graphs(self.strict, "strict")}
            if self.fast is not None:
                self._demo_admitted_graphs["optimized"] = _registered_graphs(self.fast, "optimized")
        finally:
            self._demo_preparing = False
        status = self.is_prepared(prompt, budget, mode)
        require(status["prepared"], "Demo preparation returned without all required live graph families")
        self.metadata["demo_ready"] = True
        event = {"type": "prepared", **status, "receipts": receipts,
                 "preparation_before_presentation": True}
        callback(event)
        self._status(callback, "demo_ready", "Available runners are warm and ready for live generation.",
                     demo_ready=True, prompt_tokens=length, budget=budget)
        return event

    def compare(self, prompt, budget, mode, callback, request_index=0,
                execution="sequential", request_started_s=None, require_prepared=True):
        self._thread()
        require(type(require_prepared) is bool, "Use an explicit prepared-demo policy")
        admitted = self.is_prepared(prompt, budget, mode)
        if require_prepared:
            require(admitted["prepared"], admitted["detail"])
        prepared = admitted["prepared"]
        before = None
        if prepared:
            before = (_all_graphs(self.fast, "optimized"), _all_graphs(self.strict, "strict"))
        comparison_events = []

        def forward(event):
            if event.get("type") == "comparison":
                comparison_events.append(event)
                return
            if prepared and event.get("type") == "status" and event.get("state") == "preparing":
                event = dict(event, state="demo_ready",
                             detail="The original finished. Starting the already prepared optimized runner.",
                             demo_ready=True)
            callback(event)

        result = super().compare(prompt, budget, mode, forward, request_index=request_index,
                                 execution=execution, request_started_s=request_started_s)
        if prepared:
            after = (_all_graphs(self.fast, "optimized"), _all_graphs(self.strict, "strict"))
            require(before == after, "An unexpected GPU graph capture occurred during the prepared demo")
            require(result["comparison"]["prepared"].get("new_graphs") == 0,
                    "The prepared demo unexpectedly created GPU graphs")
        require(len(comparison_events) == 1, "Expected one complete original-first comparison")
        comparison = result["comparison"]
        comparison.update(demo_prepared=prepared,
                          preparation_before_demo_s=admitted["setup_s"],
                          new_gpu_graphs_during_comparison=0 if prepared else comparison["prepared"].get("new_graphs"),
                          answers_cached=False)
        result["integrity_gates"]["prepared_demo_created_no_new_gpu_graphs"] = bool(prepared)
        callback(comparison)
        return result
