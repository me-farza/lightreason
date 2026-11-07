"""Chunked native-EOS collection for public numeric-quality cases; no gold input.

  python run_numeric_quality.py --engine baseline --model-path model-3b \
    --cases results3b/numeric-cases.json --out results3b/numeric-hf.json \
    --start 0 --limit 32 --original-reference results3b/hf-reference.json
  # Same file, next chunk (all prior verified cases retained):
  python run_numeric_quality.py --engine baseline --model-path model-3b \
    --cases results3b/numeric-cases.json --out results3b/numeric-hf.json \
    --start 32 --limit 32 --resume
  # Candidate options match numerical3b_runner.make_quality_runner:
  python run_numeric_quality.py --engine numerical --model-path model-3b \
    --cases results3b/numeric-cases.json --out results3b/numeric-fp.json \
    --draft-path /tmp/lightreason-models/pard-qwen0.5b --draft-engine graph

Only public prompts/budgets enter generation. No answer-key file is opened.
Optional old-HF import consumes only verified original32 generation fields,
never answer/score fields, and never passes cached outputs to a candidate.
Actual target hashes and critical source snapshots bracket EACH process chunk.
Resume requires unchanged identity/config/critical sources; interrupted chunks
without a completed integrity check are discarded and must be recollected.
All recorded latency is quality-harness diagnostics, not speed evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from exact3b_validate import (atomic_json, generation_eos, hash_model_parameters,
    model_identity, raw_hf_natural, sha256_file)


ROOT = Path(__file__).resolve().parent
SCHEMA = "lightreason-numeric-quality-generations-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_cases(path, plan):
    data = json.loads(Path(path).read_text())
    require(data.get("schema") == "lightreason-quality3b-public-cases-v1" and
            set(data) <= {"schema", "plan_sha256", "dataset_sha256", "cases"}, "Require public case file, never a gold key")
    require(data["plan_sha256"] == sha256_file(ROOT / "quality_3b_numeric-plan.json"), "Frozen quality plan differs")
    cases = data["cases"]
    require(len(cases) == len({row["id"] for row in cases}) == 160, "Require all160 unique public cases")
    for row in cases:
        require(set(row) <= {"id", "suite", "source_index", "prompt", "max_new_tokens"}, "Nonpublic field in case")
        expected = 1024 if row["suite"] == "gsm8k-test128" else 512 if row["suite"] == "original-diagnostics32" else None
        require(row["max_new_tokens"] == expected and isinstance(row["prompt"], str), "Public suite/budget changed")
    require(data["dataset_sha256"] == plan["dataset"]["sha256"], "Public dataset hash differs")
    return cases


def critical_sources(engine, draft_engine):
    names = ["run_numeric_quality.py", "exact3b_validate.py", "exact10_validation.py", "benchmark.py"]
    if engine == "numerical":
        names += ["numerical3b_runner.py", "relaxed3b_verifier.py", "pard_proposer.py"]
        if draft_engine == "graph":
            names.append("graph_pard_proposer.py")
    return {name: sha256_file(ROOT / name) for name in names}


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def valid_stop(tokens, reason, budget, eos):
    require(isinstance(tokens, list) and bool(tokens) and all(type(t) is int and t >= 0 for t in tokens)
            and len(tokens) <= budget, "Invalid generated token sequence")
    first = next((i for i, token in enumerate(tokens) if token in eos), None)
    require((reason == "eos" and first == len(tokens) - 1) or
            (reason == "max_new_tokens" and first is None and len(tokens) == budget),
            "Generation did not stop at first native EOS or full public budget")


def import_original(path, identity, weight, cases, tokenizer, eos):
    """Only baseline generation fields are extracted; no gold/score is accessed."""
    data = json.loads(Path(path).read_text())
    env = data["environment"]
    require(data.get("mode") == "baseline" and data["summary"].get("complete_reference_gate_passed") is True,
            "Original reference is not a complete verified HF baseline")
    require(data["identity"] == identity and identity["runtime"]["transformers"] == "4.57.1",
            "Old HF checkpoint/tokenizer/runtime identity differs")
    require(env.get("target_parameter_bytes_sha256_before") == env.get("target_parameter_bytes_sha256_after") == weight
            and env.get("target_parameters_unchanged") is True and env.get("project_python_sources_unchanged") is True,
            "Old HF reference weight/source provenance failed")
    old = {row["id"]: row for row in data["cases"]}
    imported, source_hash = {}, sha256_file(path)
    for case in cases:
        if case["suite"] != "original-diagnostics32":
            continue
        row = old[case["id"]]
        require(row["prompt"] == case["prompt"] and row["max_new_tokens"] == case["max_new_tokens"],
                "Cached diagnostic prompt/public budget differs")
        ids = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
            add_generation_prompt=True, tokenize=True)
        require(row["input_token_ids_sha256"] == token_hash(ids), "Cached diagnostic tokenization differs")
        valid_stop(row["token_ids"], row["stop_reason"], case["max_new_tokens"], eos)
        require(row["stop_reason"] == "eos" and row["text"] == tokenizer.decode(row["token_ids"], skip_special_tokens=True),
                "Cached diagnostic answer incomplete/text differs")
        imported[case["id"]] = {**case, "input_tokens": len(ids), "input_token_ids_sha256": token_hash(ids),
            "token_ids": row["token_ids"], "text": row["text"], "raw_text": tokenizer.decode(row["token_ids"]),
            "stop_reason": row["stop_reason"], "cached_hf_reference_sha256": source_hash,
            "diagnostic_metrics_not_speed_evidence": {"imported_verified_original_hf": True}}
    return imported


def resume_state(path, resume, scope):
    if not Path(path).exists():
        return {}, []
    require(resume, "Output exists; use --resume or choose a new file")
    data = json.loads(Path(path).read_text())
    require(data.get("schema") == SCHEMA and data["environment"]["collection_scope"] == scope,
            "Resume engine/config/public cases/critical source identity differs")
    chunks, discarded = [], set()
    for chunk in data["chunks"]:
        if chunk.get("integrity_passed") is None:
            discarded.add(chunk["index"])
        else:
            require(chunk["integrity_passed"] is True, "Cannot resume a failed weight/source integrity chunk")
            chunks.append(chunk)
    rows = {row["id"]: row for row in data["cases"] if row.get("collected_chunk") not in discarded and not row.get("error")}
    return rows, chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["baseline", "numerical"], required=True)
    parser.add_argument("--model-path", default="model-3b")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--original-reference", help="Baseline-only import of verified original32 HF generations")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--draft-path", default="/tmp/lightreason-models/pard-qwen0.5b")
    parser.add_argument("--block-width", type=int, default=8)
    parser.add_argument("--cache-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-static", action="store_true")
    parser.add_argument("--draft-engine", choices=["hf", "graph"], default="hf")
    parser.add_argument("--draft-attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--share-pool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    require(args.start >= 0 and (args.limit is None or args.limit > 0), "Use start>=0 and a positive limit")
    require(not args.original_reference or args.engine == "baseline", "Cached reference import is baseline-only")
    inputs = [args.cases, ROOT / "quality_3b_numeric-plan.json"] + ([args.original_reference] if args.original_reference else [])
    require(Path(args.out).resolve() not in {Path(path).resolve() for path in inputs}, "Output must not replace an input")
    plan = json.loads((ROOT / "quality_3b_numeric-plan.json").read_text())
    cases = public_cases(args.cases, plan)
    require(args.start < len(cases), "Chunk start is outside public cases")
    sources = critical_sources(args.engine, args.draft_engine)
    settings = {name: getattr(args, name) for name in ("draft_path", "block_width", "cache_mode", "cuda_graph",
        "compile_static", "draft_engine", "draft_attention", "share_pool", "warmups")} if args.engine == "numerical" else {}
    if settings:
        settings["draft_path"] = str(Path(settings["draft_path"]).resolve())
        settings["draft_files_sha256"] = {p.name: sha256_file(p) for p in sorted(Path(settings["draft_path"]).glob("*.safetensors"))}
        require(bool(settings["draft_files_sha256"]), "No local pretrained draft weights")
    scope = {"engine": args.engine, "max_len": args.max_len, "settings": settings,
             "public_cases_sha256": sha256_file(args.cases), "plan_sha256": sha256_file(ROOT / "quality_3b_numeric-plan.json"),
             "critical_sources_sha256": sources}
    rows, chunks = resume_state(args.out, args.resume, scope)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().requires_grad_(False).cuda()
    identity = model_identity(model, tokenizer, SimpleNamespace(model_path=args.model_path, tasks=str(ROOT / "quality_tasks.json")))
    if Path(args.out).exists() and args.resume:
        require(json.loads(Path(args.out).read_text())["identity"] == identity, "Resume model/tokenizer/runtime identity differs")
    eos = set(generation_eos(model, tokenizer))
    require(sorted(eos) == plan["target"]["eos_token_ids"], "Target EOS identity differs from frozen quality plan")
    before = hash_model_parameters(model)
    require(before == plan["target"]["live_parameter_sha256"], "Original BF16 target bytes differ")
    versions = [(p, p._version) for p in model.parameters()]
    index = max((chunk["index"] for chunk in chunks), default=-1) + 1
    chunk = {"index": index, "start": args.start, "limit": args.limit, "generated": 0, "imported": 0,
             "target_parameter_sha256_before": before, "critical_sources_sha256_before": sources, "integrity_passed": None}
    chunks.append(chunk)
    runner, metadata, fatal = None, None, None

    def payload():
        good = all(c.get("integrity_passed") is True for c in chunks)
        ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
        return {"schema": SCHEMA, "identity": identity, "environment": {
            "collection_scope": scope, "public_cases_sha256": scope["public_cases_sha256"],
            "target_parameter_bytes_sha256_before": before,
            "target_parameter_bytes_sha256_after": chunk.get("target_parameter_sha256_after"),
            "target_parameters_unchanged": good if chunk["integrity_passed"] is not None else None,
            "target_parameter_versions_unchanged": chunk.get("target_parameter_versions_unchanged"),
            "project_python_sources_unchanged": good if chunk["integrity_passed"] is not None else None,
            "source_snapshot_scope": "Critical generator/selected-engine files only; unrelated project sources excluded",
            "candidate_metadata": metadata, "quality_latency_is_speed_evidence": False,
            "gold_answer_key_loaded": False}, "chunks": chunks,
            "summary": {"collected": len(ordered), "required": len(cases),
                "complete": good and len(ordered) == len(cases) and all(not row.get("error") for row in ordered),
                "native_eos": sum(row.get("stop_reason") == "eos" for row in ordered),
                "budget_exhausted": sum(row.get("stop_reason") == "max_new_tokens" for row in ordered),
                "fatal_error": fatal}, "cases": ordered}
    try:
        if args.original_reference:
            imported = import_original(args.original_reference, identity, before, cases, tokenizer, eos)
            for name, row in imported.items():
                if name not in rows:
                    rows[name] = {**row, "collected_chunk": index}; chunk["imported"] += 1
        if args.engine == "numerical":
            from numerical3b_runner import make_quality_runner
            factory_args = {name: value for name, value in settings.items() if name != "draft_files_sha256"}
            with torch.inference_mode(False), torch.no_grad():
                adapter = make_quality_runner(model, tokenizer, SimpleNamespace(max_len=args.max_len, **factory_args))
            runner, metadata = adapter.runner, adapter.runner.metadata
        atomic_json(args.out, payload())
        stop = len(cases) if args.limit is None else min(len(cases), args.start + args.limit)
        for case in cases[args.start:stop]:
            if case["id"] in rows:
                continue
            ids = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                add_generation_prompt=True, return_tensors="pt").to(model.device)
            row = {**case, "collected_chunk": index, "input_tokens": ids.shape[1],
                   "input_token_ids_sha256": token_hash(ids[0].cpu().tolist())}
            try:
                reserve = args.block_width if args.engine == "numerical" else 0
                require(ids.shape[1] + case["max_new_tokens"] + reserve <= args.max_len, "Prompt/public budget exceeds capacity")
                torch.cuda.synchronize(); started = time.perf_counter()
                if runner is None:
                    tokens, reason, metrics = raw_hf_natural(model, ids, case["max_new_tokens"], eos)
                else:
                    tokens, metrics = runner.run(ids, case["max_new_tokens"], eos_ids=eos)
                    reason = metrics["stop_reason"]
                torch.cuda.synchronize()
                valid_stop(tokens, reason, case["max_new_tokens"], eos)
                row.update(token_ids=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True),
                    raw_text=tokenizer.decode(tokens), stop_reason=reason,
                    diagnostic_metrics_not_speed_evidence={**metrics, "quality_harness_s": time.perf_counter() - started})
            except Exception as error:
                row.update(error=f"{type(error).__name__}: {error}", token_ids=[], text="", raw_text="", stop_reason="error")
                rows[case["id"]] = row; atomic_json(args.out, payload()); raise
            rows[case["id"]] = row; chunk["generated"] += 1
            atomic_json(args.out, payload())
            print(json.dumps({"id": case["id"], "collected": len(rows), "tokens": len(tokens), "stop_reason": reason}), flush=True)
    except Exception as error:
        fatal = f"{type(error).__name__}: {error}"
    finally:
        chunk["target_parameter_sha256_after"] = hash_model_parameters(model)
        chunk["target_parameter_versions_unchanged"] = all(p._version == version for p, version in versions)
        chunk["critical_sources_sha256_after"] = critical_sources(args.engine, args.draft_engine)
        chunk["integrity_passed"] = (before == chunk["target_parameter_sha256_after"] and
            chunk["target_parameter_versions_unchanged"] and sources == chunk["critical_sources_sha256_after"])
        chunk["fatal_error"] = fatal
        result = payload(); atomic_json(args.out, result)
        print(json.dumps({"summary": result["summary"], "chunk_integrity_passed": chunk["integrity_passed"]}), flush=True)
    if fatal or not chunk["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
