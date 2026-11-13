"""Run an isolated local FreeToken server against Packed's exact prompt IDs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from statistics import median
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from edge35b.download import REVISION


class ModelLoading(RuntimeError):
    pass


def completion(url, prompt, prompt_tokens, cap, stream, *, on_text=None):
    payload = {"model": "qwen36-bf16", "prompt": prompt, "max_tokens": cap,
               "temperature": 0, "top_p": 1, "n": 1, "stream": stream}
    if stream:
        payload["stream_options"] = {"include_usage": True}
        if on_text is not None:
            payload["stream_options"]["continuous_usage_stats"] = True
    request = Request(url + "/v1/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    start = perf_counter()
    text = []
    usage = timings = finish = first = token_ids = None

    def consume(message):
        nonlocal usage, timings, finish, first, token_ids
        chunk_usage = message.get("usage") or {}
        usage = chunk_usage or usage
        timings = message.get("timings") or timings
        for choice in message.get("choices", []):
            finish = choice.get("finish_reason") or finish
            if "token_ids" in choice:
                token_ids = choice["token_ids"]
            piece = choice.get("text")
            if piece:
                if first is None:
                    first = perf_counter() - start
                text.append(piece)
                if on_text is not None:
                    on_text(piece, perf_counter() - start, chunk_usage.get("completion_tokens"))

    try:
        with urlopen(request, timeout=240) as response:
            if stream:
                for raw in response:
                    if not raw.startswith(b"data:"):
                        continue
                    item = raw[5:].strip()
                    if item == b"[DONE]":
                        break
                    consume(json.loads(item))
            else:
                consume(json.load(response))
    except HTTPError as e:
        message = e.read().decode(errors="replace")
        if e.code == 503:
            raise ModelLoading(message) from e
        raise RuntimeError(message) from e
    if not usage or not timings or finish is None:
        raise RuntimeError(f"Missing timing, usage or stop record: {timings}, {usage}, {finish}")
    if usage["prompt_tokens"] != prompt_tokens:
        raise RuntimeError(f"Prompt length changed: {usage['prompt_tokens']} vs {prompt_tokens}")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timings.get("cache_n"))
    if token_ids is not None and len(token_ids) != usage["completion_tokens"]:
        raise RuntimeError("Reported output IDs and completion token count differ")
    return {"text": "".join(text), "finish_reason": finish, "usage": usage,
            "timings": timings, "http_wall_s": perf_counter() - start,
            "http_first_text_s": first if stream else None,
            "cached_prompt_tokens": cached,
            "raw_output_token_ids_available": token_ids is not None,
            **({"token_ids": token_ids} if token_ids is not None else {})}


def flush_prefix_cache(url):
    request = Request(url + "/v1/cache/flush?timeout=10", method="POST")
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    if result.get("status") != "ok":
        raise RuntimeError(f"Unable to flush input state: {result}")


def studio_reference(path):
    exported = json.loads(path.read_text())
    request, service = exported["request"], exported["service"]
    cap = request["max_new_tokens"]
    if (service["model_profile"] != "35b" or not service["thinking"] or not service["same_target_weights"]
            or request.get("app_action", "new") != "new" or request.get("previous_code")
            or not isinstance(cap, int) or cap < 1):
        raise ValueError("Expected a capped, same-weight Qwen3.6 Studio export with thinking enabled")
    finished = {event["runner"]: event for event in exported["events"] if event["type"] == "completed"}
    if (finished["baseline"]["runner_name"] != "Scheduled reference"
            or finished["candidate"]["runner_name"] not in {"Lightreason Packed", "Lightreason Packed Replay"}):
        raise ValueError("Expected a comparison between Scheduled and a Packed runner")
    runs = []
    for runner, mode in (("baseline", "scheduled"), ("candidate", "packed")):
        done = finished[runner]
        runs.append({"mode": mode, "runner_name": done["runner_name"], "text": done["text"],
                     "token_ids": done["token_ids"], **done["metrics"]})
    return {"revision": REVISION, "rows": [{"case": "studio", "prompt": request["prompt"],
            "system_prompt": request.get("system_prompt"), "cap": cap, "repeat": 0, "runs": runs}]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ft", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    reference = p.add_mutually_exclusive_group(required=True)
    reference.add_argument("--packed", type=Path)
    reference.add_argument("--studio-export", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--cases", default="known-math,reasoning,coding,chat")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--decode-trials", type=Path, help="Sweep routing policies with the decode trial patch")
    p.add_argument("--width-trials", type=Path, help="Fit and benchmark real verifier widths and paired CPU experts")
    p.add_argument("--evolve", action="store_true", help="Search real verifier policies and test the winner on held-out prompts")
    p.add_argument("--evolution-trials", type=int, default=8, help="Maximum number of live candidate policies")
    p.add_argument("--evolution-development-cases", type=int, default=1, help="Use this many leading cases for search")
    p.add_argument("--evolution-proposals", type=Path, help="Optional JSON policies to try before mutations")
    p.add_argument("--width-pair-retest", action="store_true", help="Repeat CPU ablations on the fixed eight-position graph")
    p.add_argument("--cpu-pair-kernel", metavar="WIDEN,CHUNK,MAX_ROUTES",
                   help="Compare a native CPU kernel candidate with fixed width and pairing")
    p.add_argument("--cpu-pair-width", type=int, choices=(4, 8), default=8)
    p.add_argument("--pass-diagnostics", action="store_true", help="Profile expert stages and audit draft candidates")
    p.add_argument("--packed-transfer", action="store_true", help="Compare raw and packed BF16 fetches")
    p.add_argument("--packed-transfer-width", type=int, choices=(4, 8), default=4)
    p.add_argument("--packed-transfer-reference", type=Path, help="Frozen eight-position paired-1 result archive")
    p.add_argument("--hot-expert-plan", type=Path, help="Compare the frozen transport with an extra read-only BF16 GPU source")
    p.add_argument("--hot-expert-autotune", action="store_true", help="Select the enabled hot rows on development prompts")
    p.add_argument("--gather-evolution", type=int, default=0,
                   help="Number of tiled hot-source gather candidates to measure against the tuned control")
    p.add_argument("--native-packed-gather", metavar="GEOMETRY",
                   help="Test an Ada-native packed gather with the same hot expert replicas")
    p.add_argument("--cpu-read-policy", metavar="POLICY", help="Compare CPU read policies on the same native CUDA graph")
    p.add_argument("--cpu-read-profile", type=Path, help="Frozen native CUDA pass profile for ranking CPU expert banks")
    p.add_argument("--cpu-read-layers", type=int, default=20, help="Maximum layers with extra packed CPU expert banks")
    p.add_argument("--shared-cpu-gather", metavar="GEOMETRY", help="Gather directly from the packed CPU layout")
    p.add_argument("--shared-cpu-layers", type=int, help="Shared CUDA layer budget, defaulting to the CPU layer budget")
    p.add_argument("--shared-cpu-control-layers", type=int, help="Compare two budgets for the shared CUDA gather")
    p.add_argument("--cpu-read-control-layers", type=int, help="CPU packing budget in the shared CUDA control")
    p.add_argument("--expert-reuse-window", type=int, choices=range(1, 7),
                   help="Test how many verification passes to protect repeatedly used experts")
    p.add_argument("--draft-budget", action="store_true",
                   help="Fit a draft-confidence policy using real verifier widths and expert traffic")
    p.add_argument("--draft-prefix", type=Path, metavar="CALIBRATION",
                   help="Benchmark prefix selection and adaptive verification with the frozen packed runner")
    p.add_argument("--draft-prefix-teacher", action="store_true",
                   help="Also verify frozen continuation blocks as an offline acceptance diagnostic")
    p.add_argument("--cpu-three-routes", action="store_true",
                   help="Also compare two- and three-route CPU read grouping on the fixed graph")
    p.add_argument("--gpu-resident-plan", type=Path,
                   help="Compare raw and lossless packed GPU expert replicas against the fixed runner")
    p.add_argument("--gpu-resident-mib", type=int, default=896, help="Maximum additional shared GPU arena size")
    p.add_argument("--packed-primary", action="store_true",
                   help="Use the existing BF16 bank allocation for a larger packed primary decode cache")
    p.add_argument("--packed-primary-layout", choices=("staged-primary", "compact-tail", "fixed-prefix"),
                   default="staged-primary")
    p.add_argument("--packed-primary-reserve-rows", type=int, default=512,
                   help="Number of raw rows whose storage is reused for packing during decode")
    p.add_argument("--selector-calibration", action="store_true", help="Collect scores, fit a draft selector, and benchmark it")
    p.add_argument("--cache-gib", type=float, default=14)
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--mtp-steps", type=int, default=0)
    p.add_argument("--draft-dir", type=Path, help="Use a verified DFlash proposal checkpoint")
    p.add_argument("--draft-steps", type=int, default=0)
    p.add_argument("--host-embedding", action="store_true")
    p.add_argument("--expert-load", choices=("auto", "serial", "parallel"), default="auto")
    p.add_argument("--attention", default="triton")
    p.add_argument("--cache-type", choices=("naive", "radix"), default="naive")
    p.add_argument("--prefill-hit-d2d", action="store_true")
    p.add_argument("--prefix-trials", action="store_true",
                   help="Measure uncached input, a different request sharing the system prompt, and a repeat")
    p.add_argument("--full-studio-run", action="store_true",
                   help="Also stream the Studio prompt up to EOS or the 4,096-token context limit")
    args = p.parse_args()
    if args.out.exists() or args.server_log.exists():
        p.error("Use fresh output paths")
    if args.repeats < 1 or (args.repeats != 1 and args.prefix_trials):
        p.error("Use at least one repeat. Prefix trials use their own fixed order.")
    if args.decode_trials and (args.cache_type != "naive" or args.prefix_trials or args.full_studio_run
                               or args.decode_trials.exists() or args.draft_steps != 7):
        p.error("Decode trials require seven proposals, a fresh policy path, and naive input caching.")
    if args.width_trials and (args.decode_trials or args.cache_type != "naive" or args.prefix_trials
                              or args.full_studio_run or args.draft_steps != 7
                              or args.width_trials.exists() or args.width_trials.with_suffix(".passes.jsonl").exists()):
        p.error("Width trials require seven proposals, fresh policy and trace paths, and naive input caching.")
    if args.width_pair_retest and not args.width_trials:
        p.error("--width-pair-retest requires --width-trials")
    cpu_pair_kernel = None
    if args.cpu_pair_kernel is not None:
        if (not args.width_trials or args.evolve or args.width_pair_retest or args.pass_diagnostics
                or args.selector_calibration or args.packed_transfer or args.repeats < 2 or args.cache_gib != 15):
            p.error("CPU kernel comparison needs its own width trial, two or more repeats, and a 15 GiB expert cache.")
        from edge35b.width_autotune import DEFAULT_CPU_PAIR_TUNING, validate_cpu_pair_tuning

        try:
            cpu_pair_kernel = validate_cpu_pair_tuning(tuple(map(int, args.cpu_pair_kernel.split(","))))
        except ValueError as exc:
            p.error(f"Invalid CPU kernel candidate: {exc}")
        if cpu_pair_kernel == DEFAULT_CPU_PAIR_TUNING:
            p.error("Choose CPU tuning that differs from the fixed control.")
    if args.evolve and (not args.width_trials or args.width_pair_retest or args.pass_diagnostics
                        or args.packed_transfer or args.selector_calibration or args.repeats < 2
                        or not 1 <= args.evolution_trials <= 64 or args.cache_gib != 15):
        p.error("Evolution needs its own width trial, at least two repeats, 1–64 candidates, and a 15 GiB expert cache.")
    if args.evolution_proposals and not args.evolve:
        p.error("--evolution-proposals requires --evolve")
    if args.pass_diagnostics and (not args.width_trials or args.width_pair_retest
                                 or args.width_trials.with_suffix(".details.jsonl").exists()):
        p.error("Pass diagnostics require width trials and a fresh details log.")
    if args.packed_transfer and (not args.width_trials or args.width_pair_retest or args.pass_diagnostics
                                 or args.selector_calibration or args.evolve or args.repeats < 2
                                 or args.cache_gib != 15
                                 or args.width_trials.with_suffix(".details.jsonl").exists()):
        p.error("Packed transfer requires its own width trial, fresh logs, at least two repeats, and a 15 GiB cache.")
    if args.packed_transfer_reference and (not args.packed_transfer or args.packed_transfer_width != 8):
        p.error("The frozen reference is for the eight-position transfer trial.")
    if args.hot_expert_plan and not args.packed_transfer_reference:
        p.error("The hot expert trial requires the exact eight-position reference.")
    hot_experts, hot_base = (), "packed"
    if args.hot_expert_plan:
        from edge35b.tune_hot_expert_cache import read_plan

        try:
            hot_base, hot_experts = read_plan(args.hot_expert_plan)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid hot expert plan: {exc}")
    if args.hot_expert_autotune and (hot_base != "raw" or not hot_experts or len(args.cases.split(",")) < 2):
        p.error("Hot expert autotune needs a raw-source plan and separate development and held-out cases.")
    if args.gather_evolution:
        from edge35b.width_autotune import GATHER_GEOMETRIES

        if (not 1 <= args.gather_evolution <= len(GATHER_GEOMETRIES) or not args.packed_transfer
                or args.packed_transfer_width != 8 or not args.packed_transfer_reference or hot_base != "raw"
                or not hot_experts or args.hot_expert_autotune or len(args.cases.split(",")) < 2):
            p.error("Gather evolution needs the fixed eight-position hot-source runner and a held-out set.")
    if args.native_packed_gather:
        from edge35b.native_bf16_gather import select_geometry

        try:
            select_geometry(args.native_packed_gather)
        except ValueError as exc:
            p.error(str(exc))
        if (not args.packed_transfer or args.packed_transfer_width != 8 or not args.packed_transfer_reference
                or hot_base != "raw" or not hot_experts or args.hot_expert_autotune or args.gather_evolution
                or len(args.cases.split(",")) < 2):
            p.error("Native packing needs a frozen eight-position control, hot plan, and held-out prompts.")
    cpu_read_layers, cpu_read_coverage, shared_cpu_layers = (), 0.0, ()
    control_cpu_layers, control_shared_layers = (), ()
    cpu_read_profile = None
    if args.cpu_read_policy:
        from edge35b.cpu_read_packing import CPU_READ_POLICIES, packed_banks, rank_cpu_layers

        if (not args.native_packed_gather or not args.cpu_read_profile or args.cpu_read_policy == "stock"
                or args.cpu_read_policy not in CPU_READ_POLICIES or args.cases.split(",")[0] != "studio"
                or not 1 <= args.cpu_read_layers <= 40):
            p.error("CPU read trials need a nonstock policy, native CUDA, the frozen profile, and a Studio development case.")
        if packed_banks(args.cpu_read_policy):
            try:
                cpu_read_profile = json.loads(args.cpu_read_profile.read_text())
                cpu_read_layers, cpu_read_coverage = rank_cpu_layers(cpu_read_profile, args.cpu_read_layers)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                p.error(f"Invalid CPU read profile: {exc}")
    elif args.cpu_read_profile:
        p.error("--cpu-read-profile requires --cpu-read-policy.")
    if args.shared_cpu_gather:
        from edge35b.cpu_read_packing import packed_banks, rank_shared_layers
        from edge35b.native_bf16_gather import select_geometry

        if cpu_read_profile is None or packed_banks(args.cpu_read_policy) != (0, 1):
            p.error("Shared CUDA reads require a native CPU read trial with both projections packed.")
        try:
            select_geometry(args.shared_cpu_gather, shared=True)
            shared_limit = len(cpu_read_layers) if args.shared_cpu_layers is None else args.shared_cpu_layers
            shared_cpu_layers = rank_shared_layers(cpu_read_profile, cpu_read_layers, shared_limit)
        except (ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid shared CPU gather: {exc}")
    if args.shared_cpu_control_layers is not None or args.cpu_read_control_layers is not None:
        if not shared_cpu_layers or args.shared_cpu_control_layers is None or args.cpu_read_control_layers is None:
            p.error("A shared-layout control needs both its CPU and CUDA layer budgets.")
        try:
            control_cpu_layers, _ = rank_cpu_layers(cpu_read_profile, args.cpu_read_control_layers)
            control_shared_layers = rank_shared_layers(cpu_read_profile, control_cpu_layers,
                                                       args.shared_cpu_control_layers)
        except (ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid shared-layout control: {exc}")
        if (not set(control_cpu_layers) <= set(cpu_read_layers)
                or not set(control_shared_layers) <= set(shared_cpu_layers)
                or (control_cpu_layers == cpu_read_layers and control_shared_layers == shared_cpu_layers
                    and not (args.expert_reuse_window or args.draft_budget or args.gpu_resident_plan))):
            p.error("The candidate must expand the shared control's CPU or CUDA bank coverage.")
    if args.expert_reuse_window and (not control_shared_layers or control_cpu_layers != cpu_read_layers
                                     or control_shared_layers != shared_cpu_layers):
        p.error("Expert retention requires identical shared CPU and CUDA coverage in both arms.")
    if args.draft_budget and (args.expert_reuse_window or not control_shared_layers
                              or control_cpu_layers != cpu_read_layers or control_shared_layers != shared_cpu_layers
                              or len(shared_cpu_layers) != 34 or not args.packed_transfer_reference):
        p.error("The draft budget trial needs the frozen eight-position runner and identical 34-layer coverage.")
    if args.gpu_resident_plan:
        from edge35b.plan_gpu_resident_cache import read_plan

        if (args.draft_budget or args.expert_reuse_window or not control_shared_layers
                or control_cpu_layers != cpu_read_layers or control_shared_layers != shared_cpu_layers
                or len(shared_cpu_layers) != 34 or not args.packed_transfer_reference
                or not 128 <= args.gpu_resident_mib <= 3072):
            p.error("GPU residency needs identical shared 34-layer coverage, a frozen control, and 128–3072 MiB.")
        try:
            read_plan(args.gpu_resident_plan, hot_experts)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid GPU resident plan: {exc}")
    if args.cpu_three_routes and not (args.draft_budget or args.gpu_resident_plan):
        p.error("CPU grouping requires a paired shared graph.")
    if args.packed_primary and not (args.gpu_resident_plan and args.cpu_three_routes):
        p.error("Packed primary requires the current packed-resident control and three-route CPU path.")
    joint_calibration = None
    if args.draft_prefix:
        if (not args.gpu_resident_plan or not args.cpu_three_routes or not args.packed_transfer
                or args.packed_primary or args.gpu_resident_mib != 880):
            p.error("Draft prefix trials need the frozen 880 MiB packed-resident runner.")
        try:
            from edge35b.draft_budget import validate

            archived = json.loads(args.draft_prefix.read_text())
            if archived["revision"] != REVISION or archived["dtype"] != "bfloat16":
                raise ValueError("Use the pinned BF16 calibration archive")
            joint_calibration = archived["draft_budget"]["policy"]
            validate(joint_calibration)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid DFlash2 calibration: {exc}")
    elif args.draft_prefix_teacher:
        p.error("The teacher diagnostic requires --draft-prefix.")
    if args.packed_primary:
        bounds = (1024, 1536) if args.packed_primary_layout == "fixed-prefix" else (64, 512)
        if not bounds[0] <= args.packed_primary_reserve_rows <= bounds[1]:
            p.error(f"Use {bounds[0]}–{bounds[1]} reserved rows with this packed primary layout.")
    if args.selector_calibration and (not args.width_trials or args.width_pair_retest or args.pass_diagnostics
                                     or args.width_trials.with_suffix(".details.jsonl").exists()):
        p.error("Selector calibration requires its own width trial and a fresh details log.")
    if (args.mtp_steps and args.draft_dir) or (args.draft_steps and not args.draft_dir):
        p.error("Choose one draft method and provide its checkpoint when required")
    if args.prefix_trials and (args.cache_type != "radix" or args.cases != "studio"):
        p.error("Prefix trials require one Studio case and the radix cache")
    if args.full_studio_run and not args.studio_export:
        p.error("The full Studio run requires a Studio export")
    baseline = studio_reference(args.studio_export) if args.studio_export else json.loads(args.packed.read_text())
    model_manifest = json.loads((args.model / "lightreason-download.json").read_text())
    if model_manifest["revision"] != baseline["revision"]:
        p.error("FreeToken and Packed must use the same checkpoint revision")
    by_case = {row["case"]: row for row in reversed(baseline["rows"]) if row["repeat"] == 0}
    cases = args.cases.split(",")
    if not set(cases) <= by_case.keys():
        p.error("Baseline must include every requested case")
    if (args.evolve or args.packed_transfer or cpu_pair_kernel is not None) and (
            baseline["revision"] != REVISION or len(set(cases)) != len(cases)):
        p.error("Use the pinned BF16 target and distinct cases for these trials.")
    frozen_transfer = {}
    if args.packed_transfer_reference:
        from edge35b.expert_transport_eval import frozen_controls

        try:
            frozen_transfer = frozen_controls(
                json.loads(args.packed_transfer_reference.read_text()),
                {case: by_case[case]["cap"] for case in cases}, REVISION,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(f"Invalid frozen transfer reference: {exc}")
    if args.evolve:
        if not 1 <= args.evolution_development_cases < len(cases):
            p.error("Use distinct development and held-out cases.")
        from edge35b.evolve_policy import evolve, load_proposals

        try:
            proposals = load_proposals(args.evolution_proposals)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            p.error(f"Invalid evolution proposals: {exc}")
    # The pinned release accepts prompt strings. Its API schema also accepts
    # token lists, but passes them through the chat-template path. Verify the
    # raw string round trip with the same tokenizer and options as the server.
    prepare = """
import json, sys
from transformers import AutoTokenizer
from demo_chat.messages import PromptText, chat_messages
t = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
result = {}
records = json.load(sys.stdin)
for name, record in records.items():
    messages = chat_messages(PromptText(record["prompt"], record.get("system_prompt")))
    text = t.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    ids = record.get("prompt_ids")
    if ids is None:
        ids = t.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                    return_dict=True, enable_thinking=True)["input_ids"]
    if t.encode(text, add_special_tokens=True) != ids:
        raise RuntimeError(f"FreeToken's raw-string encoding differs for {name}")
    result[name] = {"text": text, "tokens": len(ids)}
if sys.argv[2] == "True":
    system = records["studio"].get("system_prompt")
    if not system:
        raise RuntimeError("Prefix trials require a nonempty shared system prompt")
    user = {"role": "user", "content": "List three colors."}
    if records["studio"]["prompt"] == user["content"]:
        raise RuntimeError("The prefix seed must use different user input")
    for name, messages in [("unrelated", [user]), ("shared", [{"role": "system", "content": system}, user])]:
        text = t.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        result[name] = {"text": text, "tokens": len(t.encode(text, add_special_tokens=True))}
print(json.dumps(result))
"""
    inputs = {name: by_case[name] for name in cases}
    calibration = {}
    if args.selector_calibration:
        from edge35b.selector_calibration import prompts as calibration_prompts

        inputs.update(calibration_prompts())
    if args.draft_budget:
        from edge35b.benchmark_draft_budget import calibration_suite

        calibration = calibration_suite()
        inputs.update(calibration)
        by_case.update(calibration)
    prompts = json.loads(subprocess.check_output([str(args.ft.with_name("python")), "-c", prepare,
                                                  str(args.model), str(args.prefix_trials)],
                        input=json.dumps(inputs), text=True))
    url = f"http://127.0.0.1:{args.port}"
    command = [str(args.ft), "serve", "--model", str(args.model), "--served-model-name", "qwen36-bf16",
               "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16", "--text-model-only",
               "--max-running-requests", "1", "--max-seq-len-override", "4096", "--cache-type", args.cache_type,
               "--moe-strategy", "hybrid", "--moe-cache-mem", f"{args.cache_gib:g}GiB", "--moe-cpu-threads", "8",
               "--cuda-graph-max-bs", "1", "--attn", args.attention, "--enable-timings-report", "--moe-collect-stats",
               "--decode-log-interval", "8192" if args.decode_trials or args.width_trials else "64"]
    if args.cache_type == "radix":
        command += ["--enable-cache-report"]
    if args.prefill_hit_d2d:
        command += ["--moe-prefill-hit-d2d"]
    if args.mtp_steps:
        command += ["--speculative-algorithm", "mtp", "--speculative-num-steps", str(args.mtp_steps)]
    if args.draft_dir:
        command += ["--speculative-algorithm", "dflash", "--speculative-draft-model-path", str(args.draft_dir)]
        if args.draft_steps:
            command += ["--speculative-num-steps", str(args.draft_steps)]
    if args.host_embedding:
        command += ["--embed-weights", "host"]
    if args.expert_load != "auto":
        command += ["--expert-load", args.expert_load]
    result = {"revision": baseline["revision"], "dtype": "bfloat16", "command": command,
              "server_log": str(args.server_log), "packed_reference": str(args.packed or args.studio_export),
              "ft_version": subprocess.check_output([str(args.ft), "--version"], text=True).strip(),
              "prompt_tokenizer_roundtrip_match": True, "prefix_trials": args.prefix_trials, "rows": []}

    def generate(name, cap, stream):
        prompt = prompts[name]
        return completion(url, prompt["text"], prompt["tokens"], cap, stream)

    def save():
        staged = args.out.with_suffix(".part")
        staged.write_text(json.dumps(result, indent=2) + "\n")
        staged.replace(args.out)

    def measure(name, stream, scenario=None, repeat=0):
        reference = by_case[name]
        if name in calibration:
            packed = scheduled = None
        else:
            packed = next(x for x in reference["runs"] if x["mode"] == "packed")
            scheduled = next((x for x in reference["runs"] if x["mode"] == "scheduled"), None)
        run = generate(name, reference["cap"], stream)
        if (args.decode_trials or args.width_trials) and ("token_ids" not in run or run["cached_prompt_tokens"] != 0):
            raise RuntimeError("Runner trials require raw output IDs and zero cached input tokens")
        run.update(case=name, stream=stream, cap=reference["cap"], scenario=scenario, repeat=repeat)
        if packed is not None:
            run.update(packed_text_match=(run["text"] == packed["text"]),
                       packed_exact_ids=(run["token_ids"] == packed["token_ids"]) if "token_ids" in run else None,
                       packed_n=len(packed["token_ids"]),
                       packed_stop_reason=packed["stop_reason"],
                       packed_decode_tokens_per_s=packed["decode_tokens_per_s"])
        if scheduled is not None:
            run["scheduled_text_match"] = run["text"] == scheduled["text"]
        result["rows"].append(run)
        save()
        print(name, scenario, stream, run["timings"]["predicted_per_second"], run["finish_reason"],
              run["cached_prompt_tokens"], run["http_wall_s"], run.get("packed_text_match"), flush=True)

    trials = {
        "control": (0, 0, 0),
        "vector": (1, 0, 0),
        "reuse-2": (1, 2, 0),
        "reuse-4": (1, 4, 0),
        "reuse-8": (1, 8, 0),
        "fetch-40": (1, 0, 40),
        "fetch-65": (1, 0, 65),
        "reuse-4-fetch-65": (1, 4, 65),
    }

    def set_trial(name):
        staged = args.decode_trials.with_suffix(".part")
        staged.write_text(" ".join(str(value) for value in trials[name]) + "\n")
        staged.replace(args.decode_trials)

    width_table = {}
    width_log = None
    detail_log = None
    if args.width_trials:
        width_log = args.width_trials.with_suffix(".passes.jsonl")
        detail_log = args.width_trials.with_suffix(".details.jsonl")

    def set_width(width, minimum, diagnostics="off", selector=None, *, transport="raw", decision_table=None,
                  cpu_pair_tuning=None, hot_expert_limit=None, cpu_read_policy="stock",
                  draft_budget=None, draft_confidence=False, three_route_reuse=False, gpu_resident_layout=None,
                  draft_prefix_selection=False, draft_teacher=None):
        staged = args.width_trials.with_suffix(".part")
        table = width_table if decision_table is None else decision_table
        settings = {"width": width, "min_pairs": minimum, "table": table,
                    "diagnostics": diagnostics, "selector": selector, "transport": transport,
                    "hot_expert_limit": hot_expert_limit}
        if cpu_pair_tuning is not None:
            settings["cpu_pair_tuning"] = cpu_pair_tuning
        if cpu_read_policy != "stock":
            settings["cpu_read_policy"] = cpu_read_policy
        if draft_budget is not None:
            settings["draft_budget"] = draft_budget
        if draft_confidence:
            settings["draft_confidence"] = True
        if three_route_reuse:
            settings["three_route_reuse"] = True
        if gpu_resident_layout is not None:
            settings["gpu_resident_layout"] = gpu_resident_layout
        if draft_prefix_selection:
            settings["draft_prefix_selection"] = True
        if draft_teacher is not None:
            settings["draft_teacher"] = draft_teacher
        staged.write_text(json.dumps(settings))
        staged.replace(args.width_trials)

    def measure_width(case, stream, trial, width, minimum, repeat=0, diagnostics="off", selector=None,
                      *, transport="raw", decision_table=None, cpu_pair_tuning=None, hot_expert_limit=None,
                      cpu_read_policy="stock", draft_budget=None, draft_confidence=False, three_route_reuse=False,
                      gpu_resident_layout=None, draft_prefix_selection=False, draft_teacher=None):
        set_width(width, minimum, diagnostics, selector, transport=transport, decision_table=decision_table,
                  cpu_pair_tuning=cpu_pair_tuning, hot_expert_limit=hot_expert_limit, cpu_read_policy=cpu_read_policy,
                  draft_budget=draft_budget, draft_confidence=draft_confidence, three_route_reuse=three_route_reuse,
                  gpu_resident_layout=gpu_resident_layout, draft_prefix_selection=draft_prefix_selection,
                  draft_teacher=draft_teacher)
        prior = len(width_log.read_text().splitlines())
        prior_details = len(detail_log.read_text().splitlines()) if detail_log.exists() else 0
        measure(case, stream, trial, repeat)
        records = width_log.read_text().splitlines()
        if len(records) != prior + 1:
            raise RuntimeError("Expected one width trace per request")
        run = result["rows"][-1]
        trace = json.loads(records[-1])
        timings = run["timings"]
        passes = trace["passes"]
        if (trace["width"] != str(width) or trace["min_pairs"] != minimum
                or (isinstance(width, int) and any(w != width for w, _, _ in passes))
                or trace.get("transport", "raw") != transport
                or trace.get("cpu_read_policy", "stock") != cpu_read_policy
                or trace.get("three_route_reuse", False) != three_route_reuse
                or trace.get("gpu_resident_layout") != gpu_resident_layout
                or trace.get("draft_prefix_selection", False) != draft_prefix_selection
                or trace.get("draft_teacher", False) != (draft_teacher is not None)
                or trace.get("hot_expert_limit") != hot_expert_limit
                or sum(w - 1 for w, _, _ in passes) != timings["draft_n"]
                or sum(n for _, n, _ in passes) != run["usage"]["completion_tokens"] - 1):
            raise RuntimeError("Width trace disagrees with the generation accounting")
        run.update(verify_width=width, pair_threshold=minimum, passes=passes, transport=transport)
        if gpu_resident_layout is not None:
            run["gpu_resident_layout"] = gpu_resident_layout
        if draft_prefix_selection:
            run["draft_prefix_selection"] = True
        if draft_teacher is not None:
            run["draft_teacher"] = True
        if args.cpu_read_policy:
            run["cpu_read_policy"] = cpu_read_policy
        if hot_expert_limit is not None:
            run["hot_expert_limit"] = hot_expert_limit
        if cpu_pair_tuning is not None:
            if tuple(trace.get("cpu_pair_tuning", ())) != cpu_pair_tuning or any(w != width for w, _, _ in passes):
                raise RuntimeError("The requested fixed width or CPU pair kernel was not recorded.")
            run["cpu_pair_tuning"] = cpu_pair_tuning
        if diagnostics != "off":
            details = detail_log.read_text().splitlines()
            if len(details) != prior_details + 1:
                raise RuntimeError("Expected one diagnostics record per request")
            detailed = json.loads(details[-1])
            if (detailed["mode"] != diagnostics
                    or [(p["width"], p["committed"]) for p in detailed["passes"]] != [tuple(p[:2]) for p in passes]
                    or sum(min(p["accepted_drafts"], p["committed"]) for p in detailed["passes"])
                    != timings["draft_n_accepted"]):
                raise RuntimeError("Diagnostics disagree with scheduler pass accounting")
            run["diagnostics"] = detailed
        save()
        return run

    def run_pass_diagnostics():
        development = cases[0]
        result["diagnostic_warmup"] = []
        warm_configs = ((4, "off"), (4, "profile"), (8, "profile"), (8, "audit"))
        for width, mode in warm_configs:
            set_width(width, 1, mode)
            run = generate(development, 4, False)
            result["diagnostic_warmup"].append(dict(width=width, mode=mode, seconds=run["http_wall_s"]))
        ordinary = [("plain-4", 4, 0), ("paired-4", 4, 1), ("paired-8", 8, 1)]
        for repeat in range(args.repeats):
            order = ordinary if repeat % 2 == 0 else reversed(ordinary)
            for trial, width, minimum in order:
                measure_width(development, True, trial, width, minimum, repeat)
        for repeat in range(args.repeats):
            measure_width(development, True, "profile-4", 4, 1, repeat, diagnostics="profile")
        measure_width(development, True, "profile-8", 8, 1, diagnostics="profile")
        for i, case in enumerate(cases):
            measure_width(case, i % 2 == 0, "audit-8", 8, 1, diagnostics="audit")

    def run_gather_evolution():
        from edge35b.gather_evolution import CONTROL, evolve
        from edge35b.width_autotune import GATHER_GEOMETRIES

        result["gather_warmup"] = {}
        for mode in (CONTROL, *GATHER_GEOMETRIES):
            set_width(8, 1, transport=mode)
            warm = generate(cases[0], 8, False)
            result["gather_warmup"][mode] = warm["http_wall_s"]

        def checkpoint(report):
            result["gather_evolution"] = report
            save()

        def measure_policy(case, stream, transport, repeat):
            return measure_width(case, stream, transport, 8, 1, repeat, transport=transport)

        evolve(measure_policy, frozen_transfer, cases[:1], cases[1:], args.gather_evolution, args.repeats,
               checkpoint=checkpoint)

    def run_packed_transfer():
        from edge35b.analyze_pass_diagnostics import profile_pass
        from edge35b.expert_transport_eval import compare_runs, summarize_transport
        from edge35b.width_autotune import (
            NATIVE_PACKED_TRANSPORT, SHARED_CPU_CONTROL_TRANSPORT, SHARED_CPU_TRANSPORT, SHARED_CPU_TRANSPORTS,
        )

        development = cases[0]
        width = args.packed_transfer_width
        modes = (hot_base, "hot") if hot_experts else ("raw", "packed")
        if args.native_packed_gather:
            modes = ("hot", NATIVE_PACKED_TRANSPORT)
        if args.cpu_read_policy:
            modes = (NATIVE_PACKED_TRANSPORT, "native-with-cpu-reads")
        if args.shared_cpu_gather:
            modes = ("native-with-cpu-reads", SHARED_CPU_TRANSPORT)
        if control_shared_layers:
            modes = (SHARED_CPU_CONTROL_TRANSPORT, SHARED_CPU_TRANSPORT)

        def options(mode):
            if mode == "native-with-cpu-reads":
                return {"transport": NATIVE_PACKED_TRANSPORT, "cpu_read_policy": args.cpu_read_policy}
            if mode in SHARED_CPU_TRANSPORTS:
                return {"transport": mode, "cpu_read_policy": args.cpu_read_policy}
            return {"transport": mode}

        canonical = dict(frozen_transfer)
        result["transport_settings"] = {
            "width": width, "cpu_pair_threshold": 1, "expert_cache_gib": args.cache_gib,
            "prompt_cache": "empty", "expert_cache": "reset before prefill",
            "frozen_reference": str(args.packed_transfer_reference) if frozen_transfer else None,
            "hot_expert_plan": str(args.hot_expert_plan) if hot_experts else None,
            "hot_expert_count": len(hot_experts),
            "native_geometry": args.native_packed_gather,
        }
        if args.shared_cpu_gather:
            result["transport_settings"].update(shared_cpu_geometry=args.shared_cpu_gather,
                                                shared_cpu_layers=shared_cpu_layers)
        if control_shared_layers:
            result["transport_settings"]["shared_cpu_control_layers"] = control_shared_layers
        if args.expert_reuse_window:
            result["transport_settings"]["expert_reuse_window_passes"] = args.expert_reuse_window
        if args.cpu_read_policy:
            result["cpu_read_settings"] = {
                "candidate_policy": args.cpu_read_policy, "layers": cpu_read_layers,
                "development_cpu_expert_coverage": cpu_read_coverage,
                "ranking_profile": str(args.cpu_read_profile),
            }
        if control_cpu_layers:
            result["cpu_read_settings"]["control_layers"] = control_cpu_layers
        result["transport_warmup"] = []
        result["transport_pairs"] = []
        for diagnostics in ("off", "profile"):
            for mode in modes:
                set_width(width, 1, diagnostics, **options(mode))
                warm = generate(development, 4, False)
                result["transport_warmup"].append({"scenario": mode, "diagnostics": diagnostics,
                                                   "seconds": warm["http_wall_s"], **options(mode)})

        def pair(case, stream, repeat, order, diagnostics="off", *, limit=None, tuning=False):
            runs = {}
            for mode in order:
                scenario = f"{mode}-tune-{limit}" if tuning else f"{mode}-{diagnostics}"
                runs[mode] = measure_width(
                    case, stream, scenario, width, 1, repeat, diagnostics=diagnostics,
                    hot_expert_limit=limit if mode == "hot" else None, **options(mode),
                )
            control, candidate = (runs[mode] for mode in modes)
            reference = canonical.setdefault(case, control)
            comparison = {"case": case, "stream": stream, "repeat": repeat,
                          "diagnostics": diagnostics, "order": list(order), "hot_expert_limit": limit,
                          **compare_runs(control, candidate, reference)}
            if diagnostics == "profile":
                comparison["same_fetch_plan"] = (
                    [p["expert_fetch_ids"] for p in control["diagnostics"]["passes"]]
                    == [p["expert_fetch_ids"] for p in candidate["diagnostics"]["passes"]]
                )
                if not args.expert_reuse_window:
                    comparison["exact"] &= comparison["same_fetch_plan"]
            for mode, run in runs.items():
                committed = sum(n for _, n, _ in run["passes"])
                decode_s = run["timings"]["predicted_ms"] / 1000
                summary = {"decode_tok_s": committed / decode_s, "decode_s": decode_s,
                           "committed_decode_tokens": committed, "verify_passes": len(run["passes"]),
                           "committed_per_pass": committed / len(run["passes"]),
                           "first_text_s": run["http_first_text_s"], "request_s": run["http_wall_s"],
                           "prefill_ms": run["timings"]["prompt_ms"],
                           "pass_ms": sum(p[2] for p in run["passes"]) / len(run["passes"])}
                if diagnostics == "profile":
                    profiles = [profile_pass(record, pass_timing[2]) for record, pass_timing in zip(
                        run["diagnostics"]["passes"], run["passes"], strict=True)]
                    for component in ("pcie_gather_ms", "cpu_wait_ms", "cpu_service_ms",
                                      "gpu_experts_ms", "fetched", "missed"):
                        summary[component] = sum(p[component] for p in profiles) / len(profiles)
                    for key, field in (("pcie_payload_bytes_per_pass", "expert_pcie_payload_bytes"),
                                       ("gpu_source_bytes_per_pass", "expert_gpu_source_bytes")):
                        summary[key] = sum(sum(p[field]) for p in run["diagnostics"]["passes"]) / len(profiles)
                comparison[mode] = summary
            result["hot_tuning_pairs" if tuning else "transport_pairs"].append(comparison)
            save()
            return comparison

        limit = None
        if args.hot_expert_autotune:
            from edge35b.tune_hot_expert_cache import BUDGETS

            result["hot_tuning_pairs"] = []
            budgets = sorted({n for n in BUDGETS if n <= len(hot_experts)} | {len(hot_experts)})
            tested = {n: [] for n in budgets}
            for rep in range(2):
                for n in budgets if rep == 0 else reversed(budgets):
                    tested[n].append(pair(development, rep == 0, rep, modes if rep == 0 else modes[::-1],
                                          limit=n, tuning=True))
            scored = {}
            for n, trials in tested.items():
                scored[n] = {
                    "exact": all(r["exact"] for r in trials),
                    "decode_speedup": median(r["hot"]["decode_tok_s"] / r["raw"]["decode_tok_s"] for r in trials),
                    "request_speedup": median(r["raw"]["request_s"] / r["hot"]["request_s"] for r in trials),
                }
            eligible = [n for n in budgets if scored[n]["exact"]]
            result["hot_tuning"] = {"development_case": development, "scores": scored}
            if not eligible:
                save()
                raise RuntimeError("No cache budget reproduced the frozen output; see the tuning pairs.")
            limit = max(eligible, key=lambda n: (scored[n]["decode_speedup"], -n))
            result["hot_tuning"]["selected"] = limit
            save()

        def measure_cases(selected):
            for repeat in range(args.repeats):
                for case in selected if repeat % 2 == 0 else reversed(selected):
                    for stream in (True, False) if repeat % 2 == 0 else (False, True):
                        order = modes if stream == (repeat % 2 == 0) else modes[::-1]
                        pair(case, stream, repeat, order, limit=limit)

        held_out = cases[1:]
        if args.expert_reuse_window:
            measure_cases(cases[:1])
            summary = summarize_transport(result["transport_pairs"], *modes)
            gain = median(row["paired_decode_speedup"] for row in summary["ordinary"])
            passed = summary["exact"] and gain > 1
            result["expert_reuse_development"] = {"decode_speedup": gain, "advanced": passed}
            save()
            print(f"Expert retention development: {gain:.4f}x decode, exact={summary['exact']}", flush=True)
            if not passed:
                held_out = []
            measure_cases(held_out)
        elif args.cpu_read_policy:
            measure_cases(cases[:1])
            if not all(row["exact"] for row in result["transport_pairs"]):
                raise RuntimeError("CPU read mutation did not match the frozen outputs. See the saved development pairs.")
            measure_cases(held_out)
        elif args.native_packed_gather:
            from edge35b.evolve_policy import MIN_DECODE_RATIO

            measure_cases(cases[:1])
            summary = summarize_transport(result["transport_pairs"], *modes)
            gain = median(row["paired_decode_speedup"] for row in summary["ordinary"])
            request = median(row["paired_request_speedup"] for row in summary["ordinary"])
            passed = summary["exact"] and gain >= MIN_DECODE_RATIO and request >= 1
            result["native_development"] = {"decode_speedup": gain, "request_speedup": request, "passed": passed}
            save()
            print(f"Native gather development: {gain:.4f}x decode, {request:.4f}x request, "
                  f"{'testing held-out prompts' if passed else 'did not advance'}", flush=True)
            if not passed:
                held_out = []
            measure_cases(held_out)
        else:
            measure_cases(cases)
        for repeat in range(args.repeats):
            pair(development, True, repeat, modes if repeat % 2 == 0 else modes[::-1],
                 diagnostics="profile", limit=limit)
        for index, case in enumerate(held_out):
            pair(case, True, 0, modes if index % 2 else modes[::-1], diagnostics="profile", limit=limit)
        result["transport_summary"] = summarize_transport(result["transport_pairs"], *modes)
        save()
        if not result["transport_summary"]["exact"]:
            raise RuntimeError("BF16 transfer failed exact token, stop, or pass comparison; see the saved report.")

    def run_selector_calibration():
        from edge35b.analyze_pass_diagnostics import summarize_recall
        from edge35b.selector_calibration import TOKEN_LIMIT, prompts as calibration_prompts

        set_width(8, 1, "train")
        records = []
        result["selector_corpus"] = []
        for case in calibration_prompts():
            run = generate(case, TOKEN_LIMIT, False)
            lines = detail_log.read_text().splitlines()
            if len(lines) != len(records) + 1:
                raise RuntimeError("Expected one score capture per development prompt")
            record = json.loads(lines[-1])
            if record["mode"] != "train":
                raise RuntimeError("Score capture used the wrong mode")
            summarize_recall({**run, "case": case, "diagnostics": record})
            records.append(record)
            result["selector_corpus"].append({"case": case, "tokens": len(run["token_ids"]),
                                              "passes": len(record["passes"]), "seconds": run["http_wall_s"]})
            save()
            print("selector capture", case, len(run["token_ids"]), flush=True)
        fit = json.loads(subprocess.check_output(
            [str(args.ft.with_name("python")), "-m", "edge35b.selector_calibration"],
            input=json.dumps(records), text=True,
        ))
        result["selector_training"] = fit
        save()
        print("selector fit", fit["selection"], fit["models"][fit["selection"]]["validation"], flush=True)
        for i, case in enumerate(cases):
            for repeat in range(args.repeats if i == 0 else 1):
                controls = [("stock-selector", None), ("trained-selector", fit["selected"])]
                if (i + repeat) % 2:
                    controls.reverse()
                for tag, parameters in controls:
                    measure_width(case, i % 2 == 0, tag, 8, 1, repeat, selector=parameters)

    def run_width_trials():
        from edge35b.width_autotune import fit_table

        development = cases[0]
        fixed = [("fixed-8", 8, 0), ("fixed-4", 4, 0), ("fixed-6", 6, 0)]
        cpu = [("paired-1", 8, 1), ("paired-3", 8, 3)]
        adaptive = [("rule", "rule", 0), ("learned", "learned", 0)]

        def sweep(specs):
            for repeat in range(args.repeats):
                ordered = specs if repeat % 2 == 0 else reversed(specs)
                for trial, width, minimum in ordered:
                    measure_width(development, True, trial, width, minimum, repeat)

        def select(specs):
            def rate(spec):
                return median(row["timings"]["predicted_per_second"] for row in result["rows"]
                              if row["case"] == development and row["scenario"] == spec[0])
            return max(specs, key=rate)

        if args.width_pair_retest:
            sweep([fixed[0], *cpu])
            paired = select(cpu)
            holdout = [fixed[0], paired]
            result["width_trial_selection"] = {"cpu": paired}
        else:
            sweep(fixed + cpu)
            width_table.update(fit_table([row["passes"] for row in result["rows"]
                                          if row["scenario"].startswith("fixed-")]))
            sweep(adaptive)
            winner = select(adaptive)
            paired = select(cpu)
            combo = ("combined", winner[1], paired[2])
            sweep([combo])
            holdout = [fixed[0], paired, winner, combo]
            result["width_trial_selection"] = {"table": width_table, "adaptive": winner,
                                               "cpu": paired, "best_fixed": select(fixed)}
        save()
        for i, case in enumerate(cases[1:]):
            for repeat in range(args.repeats if args.width_pair_retest else 1):
                forward = (i + repeat) % 2 == 0
                order = holdout if forward else reversed(holdout)
                for trial, width, minimum in order:
                    measure_width(case, forward, trial, width, minimum, repeat)

    def run_evolution():
        def measure_policy(case, stream, name, policy, repeat):
            run = measure_width(case, stream, name, policy.width, policy.min_pairs,
                                repeat, decision_table=policy.table)
            policy.check_trace(run["passes"])
            return run

        def checkpoint(history):
            result["evolution"] = {"development": history, "complete": False}
            save()

        split = args.evolution_development_cases
        report = evolve(measure_policy, cases[:split], cases[split:], args.repeats,
                        args.evolution_trials, proposals, progress=checkpoint)
        result["evolution"] = dict(report, complete=True)
        save()

    def run_cpu_pair_retest():
        from edge35b.evolve_policy import compare_trials

        control, candidate = [], []
        for index, case in enumerate(cases):
            for repeat in range(args.repeats):
                for stream in ((True, False) if repeat % 2 == 0 else (False, True)):
                    pair = [("paired-control", DEFAULT_CPU_PAIR_TUNING, control),
                            ("paired-kernel-candidate", cpu_pair_kernel, candidate)]
                    if (index + repeat + int(not stream)) % 2:
                        pair.reverse()
                    for label, tuning, rows in pair:
                        run = measure_width(case, stream, label, args.cpu_pair_width, 1, repeat, cpu_pair_tuning=tuning)
                        rows.append(run)
        result["cpu_pair_kernel_retest"] = {
            "width": args.cpu_pair_width, "control": DEFAULT_CPU_PAIR_TUNING, "candidate": cpu_pair_kernel,
            **compare_trials(control, candidate, cases, 2 * args.repeats),
        }
        save()

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    if args.decode_trials:
        set_trial("control")
        env["LIGHTREASON_DECODE_TRIAL_PATH"] = str(args.decode_trials.resolve())
        result["decode_trials"] = trials
    if args.width_trials:
        if cpu_pair_kernel is not None:
            set_width(args.cpu_pair_width, 1, cpu_pair_tuning=DEFAULT_CPU_PAIR_TUNING)
        else:
            set_width(8, int(args.pass_diagnostics or args.selector_calibration or args.packed_transfer))
        env["LIGHTREASON_WIDTH_TRIAL_PATH"] = str(args.width_trials.resolve())
    if args.pass_diagnostics or args.packed_transfer:
        env["LIGHTREASON_PASS_DIAGNOSTICS"] = "1"
    if args.packed_transfer:
        env["LIGHTREASON_PACKED_BF16_TRANSFER"] = "1"
        env["LIGHTREASON_PACKED_BF16_WIDTH"] = str(args.packed_transfer_width)
        if args.hot_expert_plan:
            env["LIGHTREASON_PACKED_BF16_HOT_PLAN"] = str(args.hot_expert_plan.resolve())
    if args.gather_evolution:
        env["LIGHTREASON_GATHER_EVOLUTION"] = "1"
    if args.native_packed_gather:
        env["LIGHTREASON_NATIVE_PACKED_GATHER"] = args.native_packed_gather
    if args.cpu_read_policy:
        env["LIGHTREASON_CPU_READ_POLICY"] = args.cpu_read_policy
        env["LIGHTREASON_CPU_READ_LAYERS"] = ",".join(map(str, cpu_read_layers))
    if args.shared_cpu_gather:
        env["LIGHTREASON_SHARED_CPU_GATHER"] = args.shared_cpu_gather
        env["LIGHTREASON_SHARED_CPU_LAYERS"] = ",".join(map(str, shared_cpu_layers))
    if control_shared_layers:
        env["LIGHTREASON_SHARED_CPU_CONTROL_LAYERS"] = ",".join(map(str, control_shared_layers))
        env["LIGHTREASON_CPU_READ_CONTROL_LAYERS"] = ",".join(map(str, control_cpu_layers))
    if args.expert_reuse_window:
        env["LIGHTREASON_EXPERT_REUSE_WINDOW"] = str(args.expert_reuse_window)
    if args.draft_budget or args.draft_prefix:
        env["LIGHTREASON_DRAFT_BUDGET"] = "1"
    if args.draft_prefix:
        env["LIGHTREASON_DRAFT_PREFIX"] = "1"
    if args.gpu_resident_plan:
        resident_manifest = args.width_trials.with_suffix(".resident.json").resolve()
        if resident_manifest.exists() or resident_manifest.with_suffix(".loads.jsonl").exists():
            p.error("Use fresh GPU resident manifest paths")
        env["LIGHTREASON_GPU_RESIDENT_PLAN"] = str(args.gpu_resident_plan.resolve())
        env["LIGHTREASON_GPU_RESIDENT_MIB"] = str(args.gpu_resident_mib)
        env["LIGHTREASON_GPU_RESIDENT_REPORT"] = str(resident_manifest)
    if args.packed_primary:
        primary_manifest = args.width_trials.with_suffix(".primary.json").resolve()
        if primary_manifest.exists():
            p.error("Use a fresh packed primary manifest path")
        env["LIGHTREASON_PACKED_PRIMARY_REPORT"] = str(primary_manifest)
        env["LIGHTREASON_PACKED_PRIMARY_LAYOUT"] = args.packed_primary_layout
        env["LIGHTREASON_PACKED_PRIMARY_RESERVE_ROWS"] = str(args.packed_primary_reserve_rows)
    if args.cpu_three_routes:
        env["LIGHTREASON_CPU_THREE_ROUTES"] = "1"
    if args.selector_calibration:
        env["LIGHTREASON_SELECTOR_LAB"] = "1"

    start = perf_counter()
    with args.server_log.open("xb") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while perf_counter() - start < 1000:
                if child.poll() is not None:
                    raise RuntimeError(f"FreeToken exited {child.returncode}; see {args.server_log}")
                try:
                    with urlopen(url + "/v1/models", timeout=1) as response:
                        if any(card["id"] == "qwen36-bf16" for card in json.load(response)["data"]):
                            break
                except (TimeoutError, URLError):
                    pass
                sleep(1)
            else:
                raise TimeoutError(f"FreeToken did not become ready; see {args.server_log}")
            warm = "unrelated" if args.prefix_trials else cases[0]
            while perf_counter() - start < 1000:
                if child.poll() is not None:
                    raise RuntimeError(f"FreeToken exited {child.returncode}; see {args.server_log}")
                try:
                    result["warmup"] = generate(warm, 3, False)
                    break
                except ModelLoading:
                    sleep(3)
            else:
                raise TimeoutError(f"FreeToken model did not become ready; see {args.server_log}")
            result["startup_s"] = perf_counter() - start - result["warmup"]["http_wall_s"]
            if args.prefix_trials:
                result["shared_prefix_seeds"] = []
                for stream in (True, False):
                    flush_prefix_cache(url)
                    measure("studio", stream, "uncached-input")
                    flush_prefix_cache(url)
                    result["shared_prefix_seeds"].append(generate("shared", 1, False))
                    measure("studio", stream, "shared-system-prompt")
                    measure("studio", stream, "repeated-input")
            elif args.decode_trials:
                for repeat in range(args.repeats):
                    order = tuple(trials) if repeat % 2 == 0 else tuple(reversed(trials))
                    for trial in order:
                        set_trial(trial)
                        measure(cases[0], True, trial, repeat)

                def rate(name):
                    return median(row["timings"]["predicted_per_second"] for row in result["rows"]
                                  if row["scenario"] == name)

                best = max(trials, key=rate)
                result["best_trial"] = best
                save()
                for case in cases[1:]:
                    for trial in dict.fromkeys(("control", best)):
                        set_trial(trial)
                        measure(case, True, trial)
            elif args.selector_calibration:
                run_selector_calibration()
            elif args.pass_diagnostics:
                run_pass_diagnostics()
            elif args.gather_evolution:
                run_gather_evolution()
            elif args.draft_budget:
                from edge35b.benchmark_draft_budget import benchmark

                benchmark(result, calibration, cases, args.repeats, frozen_transfer, shared_cpu_layers,
                          args.cpu_read_policy, set_width, measure_width, generate, save,
                          cpu_three_routes=args.cpu_three_routes)
            elif args.draft_prefix:
                from edge35b.benchmark_dflash_prefix import benchmark

                benchmark(result, cases, args.repeats, frozen_transfer, shared_cpu_layers, args.cpu_read_policy,
                          joint_calibration, resident_manifest, set_width, measure_width, generate, save,
                          teacher=args.draft_prefix_teacher)
            elif args.gpu_resident_plan:
                if args.packed_primary:
                    from edge35b.benchmark_packed_primary_cache import benchmark

                    benchmark(result, cases, args.repeats, frozen_transfer, shared_cpu_layers, args.cpu_read_policy,
                              primary_manifest, resident_manifest, set_width, measure_width, generate, save)
                else:
                    from edge35b.benchmark_gpu_resident_cache import benchmark

                    benchmark(result, cases, args.repeats, frozen_transfer, shared_cpu_layers, args.cpu_read_policy,
                              args.cpu_three_routes, resident_manifest, set_width, measure_width, generate, save)
            elif args.packed_transfer:
                run_packed_transfer()
            elif cpu_pair_kernel is not None:
                run_cpu_pair_retest()
            elif args.evolve:
                run_evolution()
            elif args.width_trials:
                run_width_trials()
            else:
                for repeat in range(args.repeats):
                    order = cases[::-1] if repeat % 2 else cases
                    for i, name in enumerate(order):
                        for stream in ((False, True) if (i + repeat) % 2 else (True, False)):
                            measure(name, stream, repeat=repeat)
            if args.full_studio_run:
                if args.cache_type == "radix":
                    flush_prefix_cache(url)
                cap = 4096 - prompts["studio"]["tokens"]
                run = generate("studio", cap, True)
                run.update(cap=cap, scenario="uncached-input-until-finished")
                result["full_studio_run"] = run
                save()
                print("full-studio-run", run["usage"]["completion_tokens"], run["finish_reason"],
                      run["http_wall_s"], flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    main()
