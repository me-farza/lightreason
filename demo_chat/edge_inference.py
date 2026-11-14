"""Expose the Qwen3.6 BF16 text benchmark runners to the Studio worker."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import time
from types import SimpleNamespace

if __package__:
    from .inference import compare_outputs
    from .messages import chat_messages
else:
    from inference import compare_outputs
    from messages import chat_messages


MODEL_NAME = "Qwen3.6-35B-A3B"
CONTEXT_CAPACITY = 4096
RUNNER_NAMES = {"original": "Scheduled reference", "optimized": "Lightreason Hybrid"}
PACKED_RUNNER = "Lightreason Packed"
PACKED_REPLAY_RUNNER = "Lightreason Packed Replay"
EXACT_GRAPH_RUNNER = "Lightreason Exact"
EXECUTION_MODES = {"original": ("scheduled", 12), "optimized": ("hybrid", 15), "strict": ("scheduled", 15)}


def runner_names(*, packed=False, packed_replay=False, exact_graphs=False):
    names = dict(RUNNER_NAMES)
    if packed:
        names["optimized"] = PACKED_REPLAY_RUNNER if packed_replay else PACKED_RUNNER
    if exact_graphs:
        names["strict"] = EXACT_GRAPH_RUNNER
    return names


class EdgeLiveInference:
    def __init__(self, model_path, *, packed=False, packed_replay=False, exact_graphs=False):
        self.model_path = Path(model_path)
        self.packed = packed
        self.packed_replay = packed_replay
        self.runner_names = runner_names(packed=packed, packed_replay=packed_replay, exact_graphs=exact_graphs)
        self.scheduled_mixer_graphs = SimpleNamespace()

    def initialize(self, callback, *, shared_experts=None, gpu_weight_views=None):
        import torch
        from edge35b.benchmark import load_target
        from edge35b.cpu_route import _kernel
        from edge35b.download import GIB
        from edge35b.expert_stream import ExpertPool

        _kernel()
        self.model, self.tokenizer, self.eos, self.metadata = load_target(
            self.model_path, shared_experts=shared_experts, gpu_weight_views=gpu_weight_views)
        cache_gib = 14 if shared_experts is not None else 15
        if torch.cuda.mem_get_info()[0] < (cache_gib + 3) * GIB:
            raise RuntimeError(f"The {cache_gib} GiB expert cache needs 3 GiB of additional free VRAM after loading.")
        self.pool = ExpertPool(device="cuda:0", capacity_bytes=15 * GIB, non_blocking=True)
        callback(dict(type="status", state="preparing", detail="Warming the 35B runners on the RTX 4090…"))
        for mode in self.runner_names:
            self._run("Count to three.", 2, mode)

    def _input_ids(self, prompt):
        return self.tokenizer.apply_chat_template(chat_messages(prompt), add_generation_prompt=True,
                                                  tokenize=True, return_dict=True, enable_thinking=True)["input_ids"]

    def resolve_budget(self, prompt, budget):
        remaining = CONTEXT_CAPACITY - len(self._input_ids(prompt))
        if remaining < 1 or (budget is not None and budget > remaining):
            raise ValueError("The prompt and output must fit the 4,096-token demo context. Shorten the prompt or choose a smaller output limit.")
        return remaining if budget is None else budget

    def prepared_status(self, prompt, budget, mode):
        return dict(prepared=True, demo_ready=True, prepared_modes=list(self.runner_names),
                    mode=mode, setup_s=0.0, detail="The Qwen3.6 runners are warm.")

    def prepare_demo(self, prompt, budget, mode, callback):
        return self.prepared_status(prompt, budget, mode)

    def _run(self, prompt, budget, mode, *, on_token=None, on_start=None):
        import torch
        from edge35b.benchmark import generate
        from edge35b.download import GIB
        from edge35b.expert_stream import patch_experts

        if mode not in self.runner_names:
            raise ValueError("Choose an available Qwen3.6 runner")
        execution, cache_gib = EXECUTION_MODES[mode]
        packed = self.packed and mode == "optimized"
        delta = nullcontext()
        if packed:
            from edge35b.delta_decode import fused_delta

            cache_gib = 14
            delta = fused_delta(self.model)
        ids = torch.tensor([self._input_ids(prompt)], device="cuda:0", dtype=torch.long)
        self.pool._capacity_bytes = cache_gib * GIB
        self.pool.packed_hybrid = packed
        with patch_experts(self.model, mode=execution, pool=self.pool), delta, ExitStack() as stack:
            if packed and self.packed_replay:
                from edge35b.packed_native import configure, mixer_graphs

                stack.enter_context(configure(self.pool, "graph"))
                stack.enter_context(mixer_graphs(self.model, self.pool))
            elif mode == "strict":
                from edge35b.packed_native import mixer_graphs
                from edge35b.scheduled_graph import replay_experts

                stack.enter_context(mixer_graphs(self.model, self.scheduled_mixer_graphs))
                stack.enter_context(replay_experts(self.pool))
            try:
                if on_start is not None:
                    on_start()
                return generate(self.model, ids, budget, self.eos, self.pool, on_token=on_token)
            finally:
                self.pool.clear()
                torch.cuda.empty_cache()

    def _stream(self, prompt, budget, mode, callback, runner, request_started_s, until_finished):
        first_request_s = None
        label = self.runner_names[mode]

        def observe(ids, elapsed):
            nonlocal first_request_s
            if first_request_s is None:
                first_request_s = time.perf_counter() - request_started_s
            callback(dict(type="token_ids", runner=runner, mode=mode, token_ids=list(ids),
                          tokens=len(ids), elapsed_s=elapsed))

        def started():
            callback(dict(type="started", runner=runner, mode=mode, runner_name=label))

        result = self._run(prompt, budget, mode, on_token=observe, on_start=started)
        ids = result["token_ids"]
        seconds = result["time_s"]
        finished = result["stop_reason"] == "eos"
        metrics = dict(seconds=seconds, tokens=len(ids), tokens_per_second=len(ids) / seconds,
                       decode_tokens_per_s=result["decode_tokens_per_s"],
                       time_to_first_token_s=result["first_token_s"], request_to_first_token_s=first_request_s,
                       finished_eos=finished, stop_reason="eos" if finished else
                       ("context_limit" if until_finished else "max_new_tokens"),
                       until_finished=until_finished, output_limit_requested=None if until_finished else budget)
        event = dict(type="completed", runner=runner, mode=mode, runner_name=label, token_ids=ids,
                     text=self.tokenizer.decode(ids, skip_special_tokens=True), metrics=metrics)
        callback(event)
        return event

    def generate(self, prompt, budget, mode, callback, request_started_s, require_prepared=True, until_finished=False):
        complete = self._stream(prompt, budget, mode, callback, "single", request_started_s, until_finished)
        event = dict(complete, type="generation", model=MODEL_NAME)
        callback(event)
        return dict(generation=event)

    def compare(self, prompt, budget, mode, callback, request_started_s, until_finished=False, **_):
        baseline = self._stream(prompt, budget, "original", callback, "baseline", request_started_s, until_finished)
        candidate = self._stream(prompt, budget, mode, callback, "candidate", request_started_s, until_finished)
        event = dict(compare_outputs(baseline, candidate), type="comparison", model=MODEL_NAME, execution="sequential")
        callback(event)
        return dict(comparison=event, baseline=baseline, candidate=candidate)
