"""Real, single-GPU comparison adapter; the audited runners stay unchanged.

All methods touching CUDA must run on the same application worker thread.
Callbacks only enqueue JSON-compatible events: UTF-8 decoding/network writes
belong to the HTTP thread. The small callback cost is included in inference
wall time on BOTH sides. Graph preparation and full integrity audits are
reported separately and never represented as first-request acceleration.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import threading
import time

if __package__:
    from .messages import chat_messages
else:
    from messages import chat_messages


TARGET_SHA = "ab4dc8672b41311882ed404da089baad41db202eda849c68c14df3352c06e9a2"
CHECKPOINT_SHA = "8e5b53abec75a33825afca5c08a151d2cefd0a8a567bfff2ebc507bdd4ecc6b6"
EFFECTIVE_AUX_SHA = "e7f784c2cc8cc201cf6f36155a529ddfc7fc35c5eb557714200ef5bcb60e3f13"
MAX_LEN = 4096
TREE_RESERVE = 64
ALLOWED_BUDGETS = (64, 128, 256, 512, 1024)


def valid_budget(value):
    """Fixed UI limits or the remaining capacity for native-EOS generation."""
    return type(value) is int and 1 <= value <= MAX_LEN - TREE_RESERVE - 1


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def configured_modes(draft_path, checkpoint_path):
    if bool(draft_path) != bool(checkpoint_path):
        raise ValueError("Provide both --draft-path and --checkpoint-path to enable Lightreason Turbo.")
    return ("original", "optimized", "strict") if draft_path else ("original", "strict")


@contextmanager
def _instance_observers(runner, emit):
    """Observe outputs only; restore even inherited/proxy methods exactly.

    object.__setattr__ intentionally places the hooks ON the proxy instance.
    Its normal __setattr__ would instead modify the original verifier owner.
    Original bound methods and their outputs remain the authority.
    """
    verifier = runner.verifier
    original_prefill = verifier.prefill
    original_accept = runner.accept_path
    original_commit = verifier.commit_path
    observed = []
    pending = []
    saved = []

    def observe_prefill(*args, **kwargs):
        anchor = original_prefill(*args, **kwargs)
        observed.append(int(anchor.item()))
        emit(list(observed))
        return anchor

    def observe_accept(*args, **kwargs):
        result = original_accept(*args, **kwargs)
        pending[:] = list(result[1])
        return result

    def observe_commit(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        require(bool(pending), "Observed commit has no original target-accepted tokens")
        observed.extend(pending)
        pending.clear()
        emit(list(observed))
        return result

    try:
        for owner, name, replacement in (
            (verifier, "prefill", observe_prefill),
            (runner, "accept_path", observe_accept),
            (verifier, "commit_path", observe_commit),
        ):
            existing = vars(owner)
            saved.append((owner, name, name in existing, existing.get(name)))
            object.__setattr__(owner, name, replacement)
        yield observed
    finally:
        for owner, name, existed, original in reversed(saved):
            if existed:
                object.__setattr__(owner, name, original)
            else:
                object.__delattr__(owner, name)


def compare_outputs(baseline, candidate):
    left, right = baseline["token_ids"], candidate["token_ids"]
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    lm, rm = baseline["metrics"], candidate["metrics"]
    left_decode = lm.get("decode_tokens_per_s")
    right_decode = rm.get("decode_tokens_per_s")
    decode_speedup = None
    if left_decode and right_decode and math.isfinite(left_decode) and math.isfinite(right_decode):
        decode_speedup = right_decode / left_decode
    return {
        "speedup": lm["seconds"] / rm["seconds"],
        "token_rate_speedup": rm["tokens_per_second"] / lm["tokens_per_second"],
        "decode_rate_speedup": decode_speedup,
        "exact_match": left == right,
        "prefix_match_tokens": prefix,
        "baseline_tokens": len(left),
        "candidate_tokens": len(right),
        "same_output_latency_speedup": lm["seconds"] / rm["seconds"] if left == right else None,
        "baseline_finished_eos": lm["finished_eos"],
        "candidate_finished_eos": rm["finished_eos"],
        "both_finished_eos": lm["finished_eos"] and rm["finished_eos"],
        "quality_equivalence_certified": False,
    }


class LiveInference:
    def __init__(self, model_path="model-3b", draft_path=None, checkpoint_path=None,
                 cooldown_s=8.0):
        self.available_modes = configured_modes(draft_path, checkpoint_path)
        self.default_mode = self.available_modes[1]
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.draft_path = str(Path(draft_path).expanduser().resolve()) if draft_path else None
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None
        require(math.isfinite(cooldown_s) and 0 <= cooldown_s <= 30, "Invalid cooldown")
        self.cooldown_s = float(cooldown_s)
        self.worker_thread = None
        self.model = self.tokenizer = self.fast = self.strict = None
        self.ready = False
        self.metadata = {}

    def _thread(self):
        current = threading.get_ident()
        if self.worker_thread is None:
            self.worker_thread = current
        require(current == self.worker_thread, "All CUDA operations require the single inference worker thread")

    @staticmethod
    def _status(callback, state, detail, **extra):
        callback({"type": "status", "state": state, "detail": detail, **extra})

    def initialize(self, callback):
        self._thread()
        require(not self.ready, "Inference is already initialized")
        self._status(callback, "loading", "Loading the original Qwen 3B model on the GPU")
        # Deferred imports let the HTTP process start and expose startup status
        # without requiring CUDA in its own request-handler thread.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from run_model_3b import validate_target, eos_token_ids
        from exact10_validation import hash_model_parameters
        from pard2_proposer3b_v1 import file_sha, model_state, tokenizer_snapshot
        from benchmark_conditioned_pard2_tree3b_v1 import source_hashes

        require(torch.cuda.is_available(), "A CUDA GPU is required; no simulated inference is provided")
        require(torch.__version__.split("+")[0] == "2.9.1", "Use the pinned Torch 2.9.1 runtime")
        require(__import__("transformers").__version__ == "4.57.1", "Use pinned Transformers 4.57.1")
        require("4090" in torch.cuda.get_device_name(), "Use the measured RTX 4090 host")
        if self.checkpoint_path is not None:
            require(file_sha(self.checkpoint_path) == CHECKPOINT_SHA, "Selected auxiliary checkpoint SHA differs")
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").eval().requires_grad_(False).cuda()
        torch.cuda.synchronize()
        self.parameter_count, _ = validate_target(self.model)
        require(hash_model_parameters(self.model) == TARGET_SHA, "Original target BF16 parameter bytes differ")
        self.versions = [(p, p._version, id(p), p.data_ptr()) for p in self.model.parameters()]
        before_state = model_state(self.model)
        self.tokenizer_before = tokenizer_snapshot(self.tokenizer)
        self.eos = eos_token_ids(self.model, self.tokenizer)
        loaded_s = time.perf_counter() - started
        constructor_s = 0.0
        if self.draft_path is not None:
            from feature_target_setup3b import validate_lazy_forward_transition
            from graph_prefill_compiled_pard2_lora_tree3b_v1 import GraphPrefillCompiledPARD2LoRATree3BRunner
            self._status(callback, "loading", "Building Lightreason Turbo with the original target weights")
            constructed = time.perf_counter()
            self.fast = GraphPrefillCompiledPARD2LoRATree3BRunner(
                self.model, self.tokenizer, self.draft_path,
                adaptation_checkpoint_path=self.checkpoint_path, max_len=MAX_LEN,
                node_budget=64, draft_tokens=15, top_k=8,
                compile_static=True, compile_draft=True, cuda_graph=True)
            self.fast.set_graph_prefill_enabled(True)
            torch.cuda.synchronize()
            constructor_s = time.perf_counter() - constructed
            validate_lazy_forward_transition(before_state, model_state(self.model))
        self.initialized_model_state = model_state(self.model)
        if self.fast is None:
            self.startup_audit = {"original_target_live_sha256": TARGET_SHA, "source_sha256": source_hashes()}
        else:
            self._status(callback, "auditing", "Verifying full target and auxiliary identities")
            self.startup_audit = self._full_audit()
        self._check_full_audit(self.startup_audit, self.startup_audit)
        self._identity()
        self.metadata = {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "gpu": torch.cuda.get_device_name(),
            "parameters": self.parameter_count,
            "target_sha256": TARGET_SHA,
            "auxiliary_checkpoint_sha256": CHECKPOINT_SHA if self.fast is not None else None,
            "available_modes": list(self.available_modes),
            "default_mode": self.default_mode,
            "target_dtype": "BF16",
            "same_target_instance": True,
            "model_load_s": loaded_s,
            "fast_constructor_s": constructor_s,
            "max_context_tokens": MAX_LEN,
            "allowed_budgets": list(ALLOWED_BUDGETS),
            "native_eos": True,
            "timing_scope": "Synchronized inference wall time including queued token events; graph preparation, cooldown, auditing, tokenization, networking and UTF-8 rendering separately excluded",
            "one_gpu_sequential": True,
        }
        self.ready = True
        self._status(callback, "ready", "Ready to compare Qwen runners on the same GPU", **self.metadata)
        return dict(self.metadata)

    def _identity(self):
        require(all(p._version == version and id(p) == identity and p.data_ptr() == pointer
                    for p, version, identity, pointer in self.versions), "An original target parameter changed")
        if self.fast is not None:
            self.fast.verifier.assert_target_unchanged()
            self.fast.proposer.assert_draft_unchanged()

    def _full_audit(self):
        if self.fast is not None:
            from benchmark_folded_pard2_lora_tree3b_v1 import full_audit
            return full_audit(self.fast)
        from benchmark_conditioned_pard2_tree3b_v1 import source_hashes
        from exact10_validation import hash_model_parameters
        return {"original_target_live_sha256": hash_model_parameters(self.model), "source_sha256": source_hashes()}

    def _check_full_audit(self, before, after):
        if self.fast is None:
            require(before["original_target_live_sha256"] == after["original_target_live_sha256"] == TARGET_SHA,
                    "Original target BF16 parameter bytes differ")
            require(before["source_sha256"] == after["source_sha256"], "Original inference source changed")
            return {"original_target_full_live_bytes_unchanged": True, "whole_run_source_hashes_unchanged": True}
        from benchmark_folded_pard2_lora_tree3b_v1 import identity_gates
        gates = identity_gates(before, after, CHECKPOINT_SHA)
        require(all(gates.values()), "Full model/source/auxiliary integrity gate failed: " +
                ", ".join(key for key, passed in gates.items() if not passed))
        require(before.get("effective_adapted_auxiliary_live_sha256") ==
                after.get("effective_adapted_auxiliary_live_sha256") == EFFECTIVE_AUX_SHA,
                "Selected effective auxiliary identity differs")
        return gates

    def _ensure_strict(self, callback):
        if self.strict is not None:
            return 0.0
        from exact3b_dynamic import Bucketed3BRunner
        import torch
        self._status(callback, "preparing", "Building the preserved exact-output runner")
        started = time.perf_counter()
        self.strict = Bucketed3BRunner(
            self.model, max_len=MAX_LEN, max_output=MAX_LEN,
            norm_mode="triton", pointwise="compiled", gate_mode="compiled",
            fuse_cache=True, combine_projections=True, combine_mode="kv",
            share_pool=True, screen_head=True, verifier_lanes=16)
        torch.cuda.synchronize()
        return time.perf_counter() - started

    def _run_baseline(self, ids, budget, emit):
        """Exactly the original HF calls/argmax/EOS policy, plus observers."""
        import torch
        with torch.inference_mode():
            output = self.model(ids, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            tokens = [int(token.item())]
            emit(tokens.copy())
            while len(tokens) < budget and tokens[-1] not in self.eos:
                output = self.model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                emit(tokens.copy())
        return tokens, {}

    def _run_strict(self, ids, budget, emit):
        import torch
        with torch.inference_mode(False), torch.no_grad():
            self.strict.prefill(ids)
            tokens = [int(self.strict.token.item())]
            emit(tokens.copy())
            while len(tokens) < budget and tokens[-1] not in self.eos:
                self.strict.replay_one()
                tokens.append(int(self.strict.token.item()))
                emit(tokens.copy())
        if hasattr(self.strict, "assert_target_unchanged"):
            self.strict.assert_target_unchanged()
        return tokens, {"all_output_tokens_target_verified": True}

    def _run_fast(self, ids, budget, emit):
        with _instance_observers(self.fast, emit) as observed:
            tokens, metrics = self.fast.run(ids, budget, eos_ids=self.eos)
        require(observed == tokens, "Observed tokens differ from original runner result")
        return tokens, metrics

    def compare(self, prompt, budget, mode, callback, request_index=0):
        self._thread()
        require(self.ready, "Inference is not ready")
        require(isinstance(prompt, str) and bool(prompt.strip()) and len(prompt) <= 20000,
                "Enter a nonempty prompt of at most 20,000 characters")
        require(valid_budget(budget), "Output must fit in the available model context")
        require(mode != "original" and mode in self.available_modes, "Choose an available Lightreason runner")
        import torch
        from exact10_validation import hash_model_parameters
        from pard2_proposer3b_v1 import model_state, tokenizer_snapshot
        from run_numeric_quality import raw_hf_natural

        began = time.perf_counter()
        deadline = began + 600.0

        def check_deadline():
            require(time.perf_counter() < deadline, "Comparison exceeded its 600-second bounded session")

        self._identity()
        require(tokenizer_snapshot(self.tokenizer) == self.tokenizer_before, "Original tokenizer changed")
        tokenized = time.perf_counter()
        prompt_ids = self.tokenizer.apply_chat_template(
            chat_messages(prompt), add_generation_prompt=True, tokenize=True)
        require(bool(prompt_ids) and len(prompt_ids) + budget + TREE_RESERVE <= MAX_LEN,
                f"Prompt has {len(prompt_ids)} tokens. Prompt + {budget} output tokens + 64 tree reserve must fit in {MAX_LEN}; shorten the prompt or lower the output limit.")
        tokenization_s = time.perf_counter() - tokenized
        self._status(callback, "preparing", "Preparing GPU graphs for this prompt length; setup is shown separately",
                     prompt_tokens=len(prompt_ids), mode=mode)
        setup_started = time.perf_counter()
        if mode == "strict":
            self._ensure_strict(callback)
        selected = self.fast if mode == "optimized" else self.strict
        # Actual request values enter CUDA only after all synthetic preparation.
        with torch.inference_mode(False), torch.no_grad():
            synthetic = torch.zeros((1, len(prompt_ids)), device=self.model.device, dtype=torch.long)
            prepared = selected.prepare(synthetic, budget)
            raw_hf_natural(self.model, synthetic, 3, set())
            torch.cuda.synchronize()
            del synthetic
        setup_s = time.perf_counter() - setup_started
        check_deadline()
        self._status(callback, "auditing", "Checking model identities before generation", setup_s=setup_s)
        before_audit = self._full_audit()
        self._check_full_audit(self.startup_audit, before_audit)
        self._identity()
        ids = torch.tensor([prompt_ids], device=self.model.device, dtype=torch.long)
        torch.cuda.synchronize()
        order = ["baseline", "candidate"] if request_index % 2 == 0 else ["candidate", "baseline"]
        completed = {}
        for runner_name in order:
            check_deadline()
            if self.cooldown_s:
                self._status(callback, "cooldown", f"Cooling the shared GPU before the {runner_name} run",
                             runner=runner_name, cooldown_s=self.cooldown_s)
                time.sleep(self.cooldown_s)
            self._identity()
            callback({"type": "started", "runner": runner_name, "mode": mode})
            torch.cuda.synchronize()
            start = time.perf_counter()
            first_token_s = None

            def emit(tokens):
                nonlocal first_token_s
                elapsed = time.perf_counter() - start
                require(elapsed <= 120.0, "Active generation exceeded its 120-second limit")
                if first_token_s is None:
                    first_token_s = elapsed
                callback({"type": "token_ids", "runner": runner_name,
                          "token_ids": tokens, "tokens": len(tokens), "elapsed_s": elapsed})

            if runner_name == "baseline":
                tokens, internal = self._run_baseline(ids, budget, emit)
            elif mode == "strict":
                tokens, internal = self._run_strict(ids, budget, emit)
            else:
                tokens, internal = self._run_fast(ids, budget, emit)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            require(bool(tokens) and len(tokens) <= budget and seconds > 0, "Invalid generation result")
            require(not any(token in self.eos for token in tokens[:-1]), "Generation continued after native EOS")
            self._identity()
            finished = tokens[-1] in self.eos
            event = {"type": "completed", "runner": runner_name, "mode": mode,
                     "token_ids": tokens, "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                     "metrics": {"seconds": seconds, "tokens": len(tokens),
                                 "tokens_per_second": len(tokens) / seconds,
                                 "finished_eos": finished, "stop_reason": "eos" if finished else "max_new_tokens",
                                 "time_to_first_token_s": first_token_s,
                                 "stream_callback_overhead_included": True},
                     "internal_metrics": internal}
            completed[runner_name] = event
            callback(event)
        check_deadline()
        self._status(callback, "auditing", "Checking unchanged model weights and source after both runs")
        audit_started = time.perf_counter()
        after_audit = self._full_audit()
        gates = self._check_full_audit(before_audit, after_audit)
        require(hash_model_parameters(self.model) == TARGET_SHA, "Original target full bytes changed")
        require(model_state(self.model) == self.initialized_model_state, "Original target module/forward identity changed")
        require(tokenizer_snapshot(self.tokenizer) == self.tokenizer_before, "Original tokenizer changed")
        self._identity()
        check_deadline()
        comparison = compare_outputs(completed["baseline"], completed["candidate"])
        comparison.update({"type": "comparison", "mode": mode, "order": order,
                           "setup_s": setup_s, "prepared": prepared,
                           "tokenization_s": tokenization_s,
                           "post_run_audit_s": time.perf_counter() - audit_started,
                           "elapsed_wall_s": time.perf_counter() - began,
                           "max_new_tokens": budget, "prompt_tokens": len(prompt_ids),
                           "cooldown_s_per_run": self.cooldown_s,
                           "target_sha256": TARGET_SHA, "integrity_passed": all(gates.values()),
                           "timing_scope": self.metadata["timing_scope"],
                           "single_gpu_serial_order": True,
                           "quality_note": "Exact token matching is measured for this prompt; it does not certify arbitrary-prompt quality equivalence."})
        callback(comparison)
        return {"prompt": prompt, "mode": mode, "baseline": completed["baseline"],
                "candidate": completed["candidate"], "comparison": comparison,
                "integrity_gates": gates, "startup": self.metadata}
