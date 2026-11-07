"""Single-runner live code generation using the same prepared Qwen target.

This adapter adds no code execution or answer reuse. Browser rendering belongs
to the sandboxed preview. Original HF, Lightreason Exact and Lightreason Turbo
retain their existing raw-greedy/native-EOS inference implementations.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

if __package__:
    from .prepared import PreparedLiveInference, _all_graphs
    from .inference import TARGET_SHA, MAX_LEN, TREE_RESERVE, require
    from .messages import PromptText, chat_messages, source_sha256 as messages_source_sha256
else:
    from prepared import PreparedLiveInference, _all_graphs
    from inference import TARGET_SHA, MAX_LEN, TREE_RESERVE, require
    from messages import PromptText, chat_messages, source_sha256 as messages_source_sha256


RUNNER_NAMES = {"original": "Original Qwen", "optimized": "Lightreason Turbo", "strict": "Lightreason Exact"}
SINGLE_SCOPE = "Synchronized single-runner generation; token observation included; preparation and preview rendering separate"
MAX_PROMPT_CHARACTERS = 8000


def build_prompt(instruction, previous_code=None):
    """Make the explicit update context shared by prepare/check/generate/compare.

    A first request is unchanged. A follow-up explicitly contains the page the
    user chose to retain; there is no hidden conversation state or truncation.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Enter a prompt first.")
    if previous_code is not None and not isinstance(previous_code, str):
        raise ValueError("The current page must be HTML text.")
    if previous_code:
        prompt = ("Current HTML document:\n```html\n" + previous_code +
                  "\n```\n\nRequested update:\n" + instruction +
                  "\n\nReturn the complete updated HTML document in one ```html code block. "
                  "Keep CSS and JavaScript inline and do not use external libraries.")
    else:
        prompt = instruction
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError("This page and update exceed the local prompt limit. Start a new page or use a shorter document; no context was truncated.")
    if isinstance(instruction, PromptText):
        return PromptText(prompt, instruction.system_prompt)
    return prompt


def _source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _policy_event(event, until_finished, budget):
    """Describe the real public capacity; never change tokens or EOS behavior."""
    result = dict(event)
    if isinstance(result.get("metrics"), dict):
        metrics = dict(result["metrics"])
        if until_finished and not metrics.get("finished_eos") and metrics.get("stop_reason") == "max_new_tokens":
            metrics["stop_reason"] = "context_limit"
        metrics.update(until_finished=until_finished,
                       output_limit_requested=None if until_finished else budget,
                       context_capacity_new_tokens=budget)
        result["metrics"] = metrics
    return result


class StudioLiveInference(PreparedLiveInference):
    def initialize(self, callback):
        metadata = super().initialize(callback)
        self._studio_source_sha = _source_sha()
        self._messages_source_sha = messages_source_sha256()
        self.metadata.update(single_runner_generation_supported=True,
                             runner_names={mode: RUNNER_NAMES[mode] for mode in self.available_modes},
                             generated_code_executed_on_gpu_host=False)
        return dict(self.metadata)

    def _source_guards(self):
        super()._source_guards()
        require(_source_sha() == self._studio_source_sha,
                "Studio generation source changed since startup")
        require(messages_source_sha256() == self._messages_source_sha,
                "Request message construction source changed since startup")

    def _shape(self, prompt, budget, mode):
        require(mode in self.available_modes, "Choose an available runner")
        return super()._shape(prompt, budget, self.default_mode if mode == "original" else mode)

    def is_prepared(self, prompt, budget, mode):
        require(mode in self.available_modes, "Choose an available runner")
        status = super().is_prepared(prompt, budget, mode)
        if mode == "original" and "optimized" not in self.available_modes and status["prompt_tokens"] is not None:
            status.update(prepared=True, detail="The Standard runner is ready and uses no prepared GPU graphs.",
                          prepared_modes=["original", *status["prepared_modes"]], setup_s=0.0)
        return dict(status, mode=mode, runner_name=RUNNER_NAMES[mode])

    prepared_status = is_prepared

    def prepare_demo(self, prompt, budget, mode, callback):
        if mode != "original" or "optimized" in self.available_modes:
            return super().prepare_demo(prompt, budget, mode, callback)
        self._thread()
        status = self.is_prepared(prompt, budget, mode)
        require(status["prepared"], status["detail"])
        self._request_guards()
        self._source_guards()
        event = {"type": "prepared", **status, "receipts": {}}
        callback(event)
        return event

    def compare(self, prompt, budget, mode, callback, request_index=0,
                execution="sequential", request_started_s=None, require_prepared=True,
                until_finished=False):
        require(mode != "original" and mode in self.available_modes, "Choose an available Lightreason runner")
        require(type(until_finished) is bool, "Choose a valid output policy")

        def forward(event):
            event = _policy_event(event, until_finished, budget)
            if event.get("type") == "comparison":
                event.update(until_finished=until_finished,
                             output_limit_requested=None if until_finished else budget,
                             context_capacity_new_tokens=budget,
                             runner_names={"baseline": RUNNER_NAMES["original"], "candidate": RUNNER_NAMES[mode]})
            callback(event)

        result = super().compare(prompt, budget, mode, forward, request_index=request_index,
                                 execution=execution, request_started_s=request_started_s,
                                 require_prepared=require_prepared)
        for name in ("baseline", "candidate"):
            result[name] = _policy_event(result[name], until_finished, budget)
        result["comparison"].update(until_finished=until_finished,
                                    output_limit_requested=None if until_finished else budget,
                                    context_capacity_new_tokens=budget,
                                    runner_names={"baseline": RUNNER_NAMES["original"], "candidate": RUNNER_NAMES[mode]})
        return result

    def generate(self, prompt, budget, mode, callback, request_started_s=None,
                 require_prepared=True, until_finished=False):
        require(mode in self.available_modes, "Choose an available runner")
        require(type(require_prepared) is bool and type(until_finished) is bool, "Use explicit demo and output policies")
        require(type(budget) is int and 1 <= budget <= MAX_LEN - TREE_RESERVE - 1,
                "The output budget must fit in the local model context")
        self._thread()
        require(self.ready, "Inference is not ready")
        began = time.perf_counter()
        received = began if request_started_s is None else request_started_s
        require(isinstance(received, (int, float)) and math.isfinite(received) and received <= began,
                "Invalid request arrival timestamp")
        admitted = self.is_prepared(prompt, budget, mode)
        # Even an internal caller cannot accidentally compile a Turbo/Exact
        # shape during the presentation. Original HF needs no decode graphs.
        require(admitted["prepared"] or (mode == "original" and not require_prepared), admitted["detail"])
        prepared = admitted["prepared"]
        self._request_guards()
        self._source_guards()
        before = (_all_graphs(self.fast, "optimized"), _all_graphs(self.strict, "strict")) if prepared else None
        import torch

        tokenized = time.perf_counter()
        prompt_ids = self.tokenizer.apply_chat_template(
            chat_messages(prompt), add_generation_prompt=True, tokenize=True)
        require(bool(prompt_ids) and len(prompt_ids) + budget + TREE_RESERVE <= MAX_LEN,
                f"Prompt has {len(prompt_ids)} tokens. The prompt, output capacity and {TREE_RESERVE}-token verification reserve must fit in {MAX_LEN}; start a new page or choose a smaller output limit.")
        tokenization_s = time.perf_counter() - tokenized
        ids = torch.tensor([prompt_ids], device=self.model.device, dtype=torch.long)
        torch.cuda.synchronize()
        callback({"type": "status", "state": "generating", "detail": f"{RUNNER_NAMES[mode]} is generating the page live"})
        callback({"type": "started", "runner": "single", "mode": mode, "runner_name": RUNNER_NAMES[mode]})
        start = time.perf_counter()
        first_token_s = request_first_token_s = None

        def emit(tokens):
            nonlocal first_token_s, request_first_token_s
            now = time.perf_counter()
            elapsed = now - start
            require(elapsed <= 120.0, "Active generation exceeded its 120-second limit")
            require(bool(tokens) and len(tokens) <= budget, "Invalid single-runner stream")
            if first_token_s is None:
                first_token_s, request_first_token_s = elapsed, now - received
            callback({"type": "token_ids", "runner": "single", "mode": mode,
                      "token_ids": list(tokens), "tokens": len(tokens), "elapsed_s": elapsed,
                      "request_to_first_token_s": request_first_token_s})

        if mode == "original":
            tokens, internal = self._run_baseline(ids, budget, emit)
        elif mode == "strict":
            tokens, internal = self._run_strict(ids, budget, emit)
        else:
            tokens, internal = self._run_fast(ids, budget, emit)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        require(bool(tokens) and len(tokens) <= budget and seconds > 0 and first_token_s is not None,
                "Generation did not produce a complete measured stream")
        require(all(type(token) is int and 0 <= token < 151936 for token in tokens), "Invalid target token IDs")
        require(not any(token in self.eos for token in tokens[:-1]), "Generation continued after native EOS")
        self._request_guards()
        self._source_guards()
        if prepared:
            after = (_all_graphs(self.fast, "optimized"), _all_graphs(self.strict, "strict"))
            require(before == after, "An unexpected GPU graph capture occurred during single-runner generation")
        finished_eos = tokens[-1] in self.eos
        event = _policy_event({
            "type": "completed", "runner": "single", "mode": mode,
            "runner_name": RUNNER_NAMES[mode], "token_ids": list(tokens),
            "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "metrics": {"seconds": seconds, "tokens": len(tokens),
                        "tokens_per_second": len(tokens) / seconds,
                        "time_to_first_token_s": first_token_s,
                        "request_to_first_token_s": request_first_token_s,
                        "finished_eos": finished_eos,
                        "stop_reason": "eos" if finished_eos else "max_new_tokens",
                        "stream_callback_overhead_included": True,
                        "timing_scope": SINGLE_SCOPE},
            "internal_metrics": internal,
        }, until_finished, budget)
        callback(event)
        gates = {"startup_full_identity_audit_passed": True,
                 "target_storage_version_and_forward_guards_passed": True,
                 "original_and_studio_source_guards_passed": True,
                 "no_new_gpu_graphs_during_generation": prepared}
        if self.fast is not None:
            gates["auxiliary_storage_version_and_forward_guards_passed"] = True
        generation = {**event, "type": "generation", "prompt": prompt,
                      "demo_prepared": admitted.get("demo_ready", prepared), "setup_s": 0.0,
                      "preparation_before_demo_s": admitted.get("setup_s"),
                      "new_gpu_graphs_during_generation": 0,
                      "answers_cached": False,
                      "same_original_target_weights": True, "target_sha256": TARGET_SHA,
                      "integrity_passed": all(gates.values()),
                      "elapsed_wall_s": time.perf_counter() - received,
                      "tokenization_s": tokenization_s,
                      "preview_rendering_included_in_generation_time": False}
        callback(generation)
        return {"prompt": prompt, "mode": mode, "generation": generation,
                "integrity_gates": gates, "startup": self.metadata}
