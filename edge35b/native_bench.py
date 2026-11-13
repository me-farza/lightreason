"""Paired Packed ablations, runtime BF16 audit and optional Nsight capture."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import ctypes
import json
from pathlib import Path
import resource
from time import perf_counter
from types import SimpleNamespace

import torch

from edge35b import packed_hybrid
from edge35b.benchmark import PROMPTS, generate, load_target, summarize, tokenize, validate_bytes, write_report
from edge35b.delta_decode import fused_delta
from edge35b.download import GIB, REVISION
from edge35b.expert_stream import ExpertPool, patch_experts


PROMPT_SET = {
    "known-math": ("What is 17 times 19?", 768),
    "known-logic": (PROMPTS["logic"], 192),
    "reasoning": ("Mira has 17 coins, all quarters and dimes. Their total value is $2.75. How many are quarters? Explain the calculation in at most three sentences.", 512),
    "coding": ("Write a small Python function to return the index of the first duplicate in an iterable, or -1 if there is none. Include the complexity. Keep your answer concise.", 768),
    "chat": ("Give me a friendly two-sentence reply to a colleague who offered to cover my shift. Accept their offer and say I can cover Friday in return.", 512),
}


@contextmanager
def annotate(model, pool):
    """Opt-in ranges. Timed performance runs leave all hooks off."""
    handles = []
    step = 0

    def before(_module, _args):
        nonlocal step
        torch.cuda.nvtx.range_push("prefill" if step == 0 else "decode-token")
        step += 1

    def after(*_):
        torch.cuda.nvtx.range_pop()

    handles.append(model.register_forward_pre_hook(before))
    handles.append(model.register_forward_hook(after, always_call=True))
    original = packed_hybrid.packed_forward
    original_kernel = packed_hybrid._kernel
    original_sync = torch.cuda.Event.synchronize
    original_copy = pool._copy

    def wrapped(fn, label):
        def call(*args, **kwargs):
            with torch.cuda.nvtx.range(label):
                return fn(*args, **kwargs)
        return call

    packed_hybrid.packed_forward = wrapped(original, "packed-layer")
    packed_hybrid._kernel = lambda: wrapped(original_kernel(), "cpu-experts")
    torch.cuda.Event.synchronize = wrapped(original_sync, "route-and-hidden-fence")
    pool._copy = wrapped(original_copy, "expert-weight-copy")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        packed_hybrid.packed_forward = original
        packed_hybrid._kernel = original_kernel
        torch.cuda.Event.synchronize = original_sync
        pool._copy = original_copy


def resident_audit(pool):
    count = elements = 0
    for key, ((gpu_gate, gpu_down), _size) in pool._rows.items():
        expert, source_gate, source_down = key[-3:]
        for live, source in ((gpu_gate, source_gate), (gpu_down, source_down)):
            if live.dtype != torch.bfloat16 or not torch.equal(live.cpu().view(torch.int16), source[expert].view(torch.int16)):
                raise RuntimeError(f"Runtime expert {expert} bytes differ from the BF16 host source")
            elements += live.numel()
        count += 1
    return {"expert_rows": count, "bf16_elements": elements}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", help="Comma-separated built-in cases. Defaults to all five without a Studio export")
    p.add_argument("--modes", default="packed")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--resident-audit", action="store_true")
    p.add_argument("--audit-logits", action="store_true", help="Capture full logit byte hashes; timings are diagnostic")
    p.add_argument("--no-observer", action="store_true", help="Do not copy committed IDs to the observation callback")
    p.add_argument("--studio-export", type=Path, help="Reproduce a Qwen3.6 Studio comparison with its system prompt")
    args = p.parse_args()
    modes = args.modes.split(",")
    names = args.cases.split(",") if args.cases else []
    if not args.cases and not args.studio_export:
        names = list(PROMPT_SET)
    if args.out.exists() or not set(names) <= PROMPT_SET.keys() or args.repeats < 1:
        p.error("Use a fresh result path, known cases, and at least one repeat")
    if args.profile and (len(modes) != 1 or len(names) + bool(args.studio_export) != 1 or args.repeats != 1):
        p.error("Capture one request and one mode at a time")
    cases = {}
    if args.studio_export:
        exported = json.loads(args.studio_export.read_text())
        request = exported["request"]
        if (exported["service"]["model_profile"] != "35b" or request.get("app_action", "new") != "new"
                or request.get("previous_code") or args.cap):
            p.error("Use a Qwen3.6 new-request export with its original output budget")
        cases = {"studio": (request["prompt"], request["max_new_tokens"], request.get("system_prompt"))}
    cases.update({name: (*PROMPT_SET[name], None) for name in names})
    model, tokenizer, eos, setup = load_target(args.model_dir)
    budgets = {mode: 15 if mode in ("scheduled-graph15", "scheduled-experts15")
               else 12 if mode.startswith("scheduled") else 14
               for mode in modes}
    if torch.cuda.mem_get_info()[0] < (max(budgets.values()) + 3) * GIB:
        raise RuntimeError("Need 3 GiB of GPU headroom beyond the expert cache after loading")
    pool = ExpertPool(capacity_bytes=14 * GIB)
    scheduled_mixer_graphs = SimpleNamespace()
    report = {"revision": REVISION, "setup": setup, "eos_ids": sorted(eos), "warmup_s": {},
              "dtype": "bfloat16", "greedy": True, "thinking": True, "batch_size": 1,
              "expert_cache_gib": 14, "cold_expert_cache_each_request": True,
              "expert_cache_gib_by_mode": budgets,
              "token_observer": not args.no_observer,
              "logits_audit": args.audit_logits,
              "torch": torch.__version__, "cuda": torch.version.cuda, "rows": [], "profiled": args.profile}

    def run(mode, ids, cap, *, capture=False):
        pool.clear()
        torch.cuda.empty_cache()
        scheduled = mode.startswith("scheduled")
        pool._capacity_bytes = budgets[mode] * GIB
        pool.packed_hybrid = not scheduled
        callback = None if args.no_observer else lambda ids, _elapsed: list(ids)
        ranges = annotate(model, pool) if capture else nullcontext()
        delta = nullcontext() if scheduled else fused_delta(model)
        route = "scheduled" if scheduled else "hybrid"
        with patch_experts(model, mode=route, pool=pool), delta, ranges, ExitStack() as stack:
            if mode in ("scheduled-graph", "scheduled-graph15", "scheduled-experts15"):
                from edge35b.packed_native import mixer_graphs

                stack.enter_context(mixer_graphs(model, scheduled_mixer_graphs))
                if mode == "scheduled-experts15":
                    from edge35b.scheduled_graph import replay_experts

                    stack.enter_context(replay_experts(pool))
            elif mode not in ("packed", "scheduled"):
                from edge35b.packed_native import configure, mixer_graphs
                mixer_mode = mode == "mixer-graph"
                stack.enter_context(configure(pool, "graph" if mixer_mode else mode))
                if mixer_mode:
                    stack.enter_context(mixer_graphs(model, pool))
            if capture:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                before = resource.getrusage(resource.RUSAGE_SELF)
                result = generate(model, ids, cap, eos, pool, audit_logits=args.audit_logits, on_token=callback)
                after = resource.getrusage(resource.RUSAGE_SELF)
                result["major_page_faults"] = after.ru_majflt - before.ru_majflt
                return result
            finally:
                if capture:
                    torch.cuda.cudart().cudaProfilerStop()

    warm = tokenize(tokenizer, "Count to three.", True)
    for mode in modes:
        start = perf_counter()
        run(mode, warm, 3)
        report["warmup_s"][mode] = perf_counter() - start
    for index, (name, (prompt, cap, system)) in enumerate(cases.items()):
        cap = args.cap or cap
        if system:
            from demo_chat.messages import PromptText, chat_messages

            ids = tokenizer.apply_chat_template(chat_messages(PromptText(prompt, system)),
                                                add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                enable_thinking=True)["input_ids"].cuda()
        else:
            ids = tokenize(tokenizer, prompt, True)
        cap = cap if cap is not None else 4096 - ids.shape[-1]
        for repeat in range(args.repeats):
            order = (index + repeat) % len(modes)
            row = {"case": name, "prompt": prompt, "prompt_ids": ids[0].tolist(), "cap": cap, "repeat": repeat, "runs": []}
            if system:
                row["system_prompt"] = system
            report["rows"].append(row)
            for mode in modes[order:] + modes[:order]:
                result = run(mode, ids, cap, capture=args.profile)
                result.update(mode=mode, text=tokenizer.decode(result["token_ids"], skip_special_tokens=True))
                row["runs"].append(result)
                report["summary"] = summarize(report["rows"], modes)
                write_report(args.out, report)
                print(name, repeat, mode, round(result["decode_tokens_per_s"], 2),
                      len(result["token_ids"]), result["stop_reason"], flush=True)
    report["captured_mixer_graphs"] = len(getattr(pool, "mixer_graphs", {}))
    report["captured_scheduled_mixer_graphs"] = len(getattr(scheduled_mixer_graphs, "mixer_graphs", {}))
    if args.resident_audit:
        report["runtime_resident_audit"] = resident_audit(pool)
    pool.clear()
    report["post_run_checkpoint_bytes_match"] = validate_bytes(model, args.model_dir) == setup["target_parameters"]
    write_report(args.out, report)


if __name__ == "__main__":
    main()
