"""Responsive real streaming: original first, then the optimized runner.

Startup retains the inherited full target/auxiliary byte and checkpoint audits.
Requests use inexpensive object/storage/version/forward guards and source
hashes, rather than rescanning model/checkpoint bytes or pausing for cooldowns.
The frozen inference implementations and their arithmetic are unchanged.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

if __package__:
    from .inference import LiveInference, TARGET_SHA, MAX_LEN, TREE_RESERVE, ALLOWED_BUDGETS, compare_outputs, require, valid_budget
    from .messages import chat_messages
else:
    from inference import LiveInference, TARGET_SHA, MAX_LEN, TREE_RESERVE, ALLOWED_BUDGETS, compare_outputs, require, valid_budget
    from messages import chat_messages


SCOPE = "Synchronized original-first interactive generation; streaming observation included; setup, queueing and rendering separate"
PROTOCOL = "interactive_original_first_no_artificial_cooldown"


def _demo_sources():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("inference.py", "interactive.py")}


def _tokenizer_controls(tokenizer):
    return (id(tokenizer), type(tokenizer), tokenizer.eos_token_id,
            tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.chat_template,
            getattr(tokenizer.apply_chat_template, "__func__", tokenizer.apply_chat_template),
            getattr(tokenizer.decode, "__func__", tokenizer.decode))


class InteractiveLiveInference(LiveInference):
    def initialize(self, callback):
        metadata = super().initialize(callback)
        from pard2_proposer3b_v1 import model_state
        self._guard_models = {"target": self.model}
        if self.fast is not None:
            conditioned = self.fast.proposer.conditioned
            self._guard_models.update(effective_auxiliary=conditioned.model,
                                      original_auxiliary_reference=conditioned.base.reference.model,
                                      projector=conditioned.adapter.projector)
        self._guard_model_states = {name: model_state(model) for name, model in self._guard_models.items()}
        self._tokenizer_control_state = _tokenizer_controls(self.tokenizer)
        self._demo_source_state = _demo_sources()
        self._root_source_state = dict(self.startup_audit["source_sha256"])
        self.metadata.update(
            timing_scope=SCOPE, comparison_protocol=PROTOCOL,
            default_execution="sequential", fixed_order=["baseline", "candidate"],
            per_request_full_byte_audit=False,
            integrity_checks="startup_full_hash_plus_request_version_storage_module_guards",
            artificial_cooldown_s=0.0)
        self._status(callback, "ready", "Ready. The original streams first, then the optimized runner.", **self.metadata)
        return dict(self.metadata)

    def _request_guards(self):
        """No model-byte copies, checkpoint reads, or vocabulary sorting."""
        from pard2_proposer3b_v1 import model_state
        self._identity()
        require(all(model_state(model) == self._guard_model_states[name]
                    for name, model in self._guard_models.items()),
                "Target/auxiliary object, storage, version, mode or forward identity changed")
        require(_tokenizer_controls(self.tokenizer) == self._tokenizer_control_state,
                "Original tokenizer identity, template, control IDs or methods changed")

    def _source_guards(self):
        from benchmark_conditioned_pard2_tree3b_v1 import source_hashes
        require(source_hashes() == self._root_source_state,
                "Original inference source changed since the full startup audit")
        require(_demo_sources() == self._demo_source_state,
                "Interactive streaming source changed since startup")

    def compare(self, prompt, budget, mode, callback, request_index=0,
                execution="sequential", request_started_s=None):
        require(execution == "sequential", "This chat runs the original first, then the optimized runner")
        self._thread()
        require(self.ready, "Inference is not ready")
        require(isinstance(prompt, str) and bool(prompt.strip()) and len(prompt) <= 8000,
                "Enter a nonempty prompt of at most 8,000 characters")
        require(valid_budget(budget), "Output must fit in the available model context")
        require(mode != "original" and mode in self.available_modes, "Choose an available Lightreason runner")
        began = time.perf_counter()
        received = began if request_started_s is None else request_started_s
        require(isinstance(received, (int, float)) and math.isfinite(received) and received <= began,
                "Invalid request arrival timestamp")
        deadline = began + 600.0
        import torch

        def check_deadline():
            require(time.perf_counter() < deadline, "Comparison exceeded its 600-second bounded session")

        self._request_guards()
        tokenized = time.perf_counter()
        prompt_ids = self.tokenizer.apply_chat_template(
            chat_messages(prompt), add_generation_prompt=True, tokenize=True)
        require(bool(prompt_ids) and len(prompt_ids) + budget + TREE_RESERVE <= MAX_LEN,
                f"Prompt has {len(prompt_ids)} tokens. Prompt + {budget} output tokens + 64 tree reserve must fit in {MAX_LEN}; shorten the prompt or lower the output limit.")
        tokenization_s = time.perf_counter() - tokenized
        ids = torch.tensor([prompt_ids], device=self.model.device, dtype=torch.long)
        torch.cuda.synchronize()
        completed = {}

        def run_one(runner_name):
            check_deadline()
            self._request_guards()
            callback({"type": "started", "runner": runner_name, "mode": mode,
                      "execution": execution})
            torch.cuda.synchronize()
            start = time.perf_counter()
            first_token_s = request_first_token_s = None

            def emit(tokens):
                nonlocal first_token_s, request_first_token_s
                now = time.perf_counter()
                elapsed = now - start
                require(elapsed <= 120.0, "Active generation exceeded its 120-second limit")
                require(bool(tokens) and len(tokens) <= budget, "Invalid live observed tokens")
                if first_token_s is None:
                    first_token_s = elapsed
                    request_first_token_s = now - received
                callback({"type": "token_ids", "runner": runner_name,
                          "token_ids": list(tokens), "tokens": len(tokens), "elapsed_s": elapsed,
                          "request_to_first_token_s": request_first_token_s})

            if runner_name == "baseline":
                tokens, internal = self._run_baseline(ids, budget, emit)
            elif mode == "strict":
                tokens, internal = self._run_strict(ids, budget, emit)
            else:
                tokens, internal = self._run_fast(ids, budget, emit)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            require(bool(tokens) and len(tokens) <= budget and seconds > 0 and first_token_s is not None,
                    "Invalid generation result or missing real streamed output")
            require(not any(token in self.eos for token in tokens[:-1]),
                    "Generation continued after native EOS")
            self._request_guards()
            finished = tokens[-1] in self.eos
            event = {"type": "completed", "runner": runner_name, "mode": mode,
                     "execution": execution, "token_ids": list(tokens),
                     "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                     "metrics": {"seconds": seconds, "tokens": len(tokens),
                                 "tokens_per_second": len(tokens) / seconds,
                                 "finished_eos": finished,
                                 "stop_reason": "eos" if finished else "max_new_tokens",
                                 "time_to_first_token_s": first_token_s,
                                 "request_to_first_token_s": request_first_token_s,
                                 "stream_callback_overhead_included": True,
                                 "timing_scope": SCOPE},
                     "internal_metrics": internal}
            completed[runner_name] = event
            callback(event)

        # The user sees the original answer before any candidate setup starts.
        self._status(callback, "generating", "Streaming the original runner first", execution=execution)
        run_one("baseline")

        self._status(callback, "preparing", "Original finished. Preparing the optimized runner; new prompt lengths may need one-time GPU setup.",
                     execution=execution, prompt_tokens=len(prompt_ids), mode=mode)
        setup_started = time.perf_counter()
        if mode == "strict":
            self._ensure_strict(callback)
        selected = self.fast if mode == "optimized" else self.strict
        # Only public shape and zeros enter the candidate's preparation route.
        with torch.inference_mode(False), torch.no_grad():
            synthetic = torch.zeros((1, len(prompt_ids)), device=self.model.device, dtype=torch.long)
            prepared = selected.prepare(synthetic, budget)
            torch.cuda.synchronize()
            del synthetic
        setup_s = time.perf_counter() - setup_started
        check_deadline()
        self._request_guards()
        self._status(callback, "generating", "Streaming the optimized runner now", execution=execution,
                     setup_s=setup_s)
        run_one("candidate")

        checked = time.perf_counter()
        self._request_guards()
        self._source_guards()
        check_deadline()
        gates = {
            "startup_full_identity_audit_passed": True,
            "original_target_parameter_object_storage_version_and_forward_guards_passed": True,
            "tokenizer_controls_and_methods_unchanged": True,
            "original_and_interactive_sources_unchanged_since_startup": True,
            "no_full_model_or_checkpoint_byte_scan_during_this_request": True,
        }
        if self.fast is not None:
            gates["auxiliary_reference_effective_projector_module_guards_passed"] = True
        comparison = compare_outputs(completed["baseline"], completed["candidate"])
        comparison.update({
            "type": "comparison", "mode": mode, "execution": execution,
            "order": ["baseline", "candidate"], "fixed_original_first_order": True,
            "isolated_benchmark": True, "isolated_run_timings": True,
            "benchmark_protocol": PROTOCOL,
            "both_streams_started_together": False,
            "setup_s": setup_s, "prepared": prepared, "tokenization_s": tokenization_s,
            "post_run_guard_s": time.perf_counter() - checked,
            "elapsed_wall_s": time.perf_counter() - received,
            "request_to_first_token_s": completed["baseline"]["metrics"]["request_to_first_token_s"],
            "max_new_tokens": budget, "prompt_tokens": len(prompt_ids),
            "cooldown_s_per_run": 0.0, "per_request_full_byte_audit": False,
            "integrity_checks": "startup_full_hash_plus_request_version_storage_module_guards",
            "target_sha256": TARGET_SHA, "integrity_passed": True,
            "timing_scope": SCOPE, "single_gpu_serial_order": True,
            "quality_note": "The original streams first, followed by fresh optimized inference. This is an interactive comparison, not the archived alternating multi-repeat benchmark. Exact matching is checked for this prompt; arbitrary-prompt quality is not certified.",
        })
        callback(comparison)
        return {"prompt": prompt, "mode": mode, "execution": execution,
                "baseline": completed["baseline"], "candidate": completed["candidate"],
                "comparison": comparison, "integrity_gates": gates,
                "startup": self.metadata}
