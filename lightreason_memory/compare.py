"""Run the benchmark-history pre-recall and evidence-only comparison on the GPU host."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

from .backend import QwenMemoryBackend
from .demo import DEMO_QUESTION, DEMO_UNKNOWN, seed_demo
from .reasoning import reasoner_prompt
from .remote import Reasoner35B, SampledMemory, comparison_lease
from .service import MemoryService
from .store import Store


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    expected: dict[str, str]
    unsupported: dict[str, str]
    abstain: bool = False


_WEATHER_EXACT = (
    r"(?:weather|scheduled)[^.\n]{0,140}(?:\b(?:were|are|was|is) identical\b"
    r"|\bmatched (?:every|all)\b|\bpreserved complete output agreement\b)"
)
CASES = (
    Case("agreement", DEMO_QUESTION, {
        "strict 3B agreed": r"(?:38\s*/\s*38|38 out of 38|all 38|matched 38)",
        "packed 35B agreed": r"(?:40\s*/\s*40|40 out of 40|all 40|40 comparisons)",
        "weather diverged at 30-token prefix": r"(?:diverg\w*|differ\w*|only matched|only the first)[^.\n]{0,100}\b30\b|\b30\b[^.\n]{0,100}(?:diverg\w*|differ\w*)",
    }, {"weather exact match": _WEATHER_EXACT}),
    Case("weather-paraphrase", "Did the weather demo's roughly 5.7x decode advantage reproduce the Scheduled answer exactly?", {
        "weather did not match": r"\b(?:no|not|diverg\w*|differ\w*)\b",
        "shared 30-token prefix": r"\b30\b",
    }, {"weather exact match": _WEATHER_EXACT}),
    Case("cache-paraphrase", "For coding, how fast was the optional packed GPU expert cache, and which baseline was used to check agreement?", {
        "coding rate": r"\b51\.(?:08|1)\b",
        "tuned fixed-width baseline": r"\b(?:tuned|fixed[- ]eight|fixed[- ]8)\b",
    }, {"wrong baseline": r"\bagainst (?:the )?(?:original )?(?:scheduled|hugging face)\b"}),
    Case("absent-energy", DEMO_UNKNOWN, {
        "abstention": r"\b(?:unknown|no |not |unavailable|cannot|can't|insufficient|doesn't)\b",
    }, {"unsupported energy number": r"\b\d[\d,.]*\s*(?:joules?\b|J\b)"}, abstain=True),
)


def _score(case, result):
    answer = result["answer"]
    checks = {label: bool(re.search(pattern, answer, re.I)) for label, pattern in case.expected.items()}
    unsupported = [label for label, pattern in case.unsupported.items() if re.search(pattern, answer, re.I)]
    return dict(
        rubric_passed=bool(answer) and all(checks.values()) and not unsupported,
        checks=checks, flagged_unsupported=unsupported,
        abstained=case.abstain and all(checks.values()) and not unsupported,
    )


def _evidence(rows, text_key):
    return [
        {"id": row["memory_id"], "kind": row["kind"], "excerpt": row[text_key]}
        for row in rows
    ]


def _contexts(selection, recalled, records):
    return {
        "direct": {"evidence": _evidence(selection.evidence, "excerpt")},
        "memory": {"evidence": _evidence(recalled["supporting_excerpts"], "excerpt"),
                   "status": recalled["status"], "unknowns": recalled["unknowns"]},
        "full": {"evidence": _evidence(records, "content")},
    }


def _write_report(path, report):
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                            encoding="utf-8", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _status(event):
    if event["type"] == "status":
        print(event["detail"], file=sys.stderr, flush=True)


def _backend(args):
    return QwenMemoryBackend(args.model_3b, runtime_path=args.runtime_3b,
                             python_executable=args.python_3b, timeout_s=args.memory_timeout)


def _recall(service, case, args):
    began = perf_counter()
    selection = service.select(case.question, args.memory_tokens)
    recalled = service.read(selection)
    return selection, recalled, perf_counter() - began


def _answer(reasoner, case, context, mode, args):
    result = reasoner.answer(reasoner_prompt(case.question, context), args.reasoner_tokens)
    result.update(case=case.name, mode=mode, question=case.question, scoring=_score(case, result))
    return result


def _paired_phase(args, store, records, report):
    memory = _backend(args)
    service = MemoryService(store, memory)
    saved = {}
    try:
        with SampledMemory() as sample:
            for case in CASES:
                selection, recalled, elapsed = _recall(service, case, args)
                saved[case.name] = selection, recalled, elapsed
            report["component_3b_release_s"] = memory.release_gpu()
        report["component_3b_memory"] = sample.report()
    finally:
        service.close()
    report["recall_components"] = [
        {"case": case.name, "component_elapsed_s": saved[case.name][2],
         "temperature": "process-cold" if case is CASES[0] else "warm-worker-or-no-evidence",
         "recall": saved[case.name][1]}
        for case in CASES
    ]
    _write_report(args.out, report)

    reasoner = Reasoner35B(args.model_35b, args.stage_35b)
    memory = _backend(args)
    service = MemoryService(store, memory)
    first = CASES[0]
    try:
        with SampledMemory() as cold_sample:
            cold_began = perf_counter()
            selected, recalled, recall_elapsed = _recall(service, first, args)
            recall_completed = perf_counter()
            release_s = memory.release_gpu()
            service.close()
            report["cold_memory"] = {"recall": recalled, "recall_elapsed_s": recall_elapsed,
                                     "memory_worker_release_s": release_s}
            _write_report(args.out, report)
            setup_s = reasoner.start(_status)
            reasoner_ready = perf_counter()
            contexts = _contexts(selected, recalled, records)
            reasoner_request_started = perf_counter()
            cold = _answer(reasoner, first, contexts["memory"], "memory", args)
            cold["end_to_end_s"] = perf_counter() - cold_began
            first_text = cold["reasoner_first_final_text_s"]
            cold["request_to_first_final_text_s"] = (
                reasoner_request_started - cold_began + first_text if first_text is not None else None
            )
            cold.update(temperature="process-cold", setup_35b_s=setup_s,
                        memory_to_reasoner_ready_s=reasoner_ready - recall_completed,
                        memory_worker_release_s=release_s, memory_recall_elapsed_s=recall_elapsed)
            report["reasoner_runs"].append(cold)
        report["cold_pipeline_memory"] = cold_sample.report()
        _write_report(args.out, report)

        # Alternate A/B order across prompts. Use the very same selected rows for
        # both arms. Warm 3B timings were measured earlier, before loading 35B.
        with SampledMemory() as warm_sample:
            for index, case in enumerate(CASES):
                selection, recalled, elapsed = (selected, recalled, recall_elapsed) if case is first else saved[case.name]
                contexts = _contexts(selection, recalled, records)
                modes = ("direct", "memory", "full") if index % 2 == 0 else ("memory", "direct", "full")
                for mode in modes:
                    if mode == "full" and args.skip_full:
                        continue
                    started = perf_counter()
                    result = _answer(reasoner, case, contexts[mode], mode, args)
                    result.update(temperature="warm-reasoner", request_elapsed_s=perf_counter() - started,
                                  direct_evidence_ids=[row["memory_id"] for row in selection.evidence],
                                  memory_source_ids=recalled["source_ids"])
                    if mode == "memory":
                        result["separate_memory_component_s"] = elapsed
                    elif mode == "direct":
                        result["separate_retrieval_component_s"] = selection.retrieval_s
                        result["separate_selection_tokenization_s"] = selection.tokenization_s
                    report["reasoner_runs"].append(result)
                    _write_report(args.out, report)
        report["warm_reasoner_memory"] = warm_sample.report()
        _write_report(args.out, report)
    finally:
        try:
            service.close()
        finally:
            reasoner.close()


def _cold_direct_phase(args, store, report):
    memory = _backend(args)
    service = MemoryService(store, memory)
    reasoner = Reasoner35B(args.model_35b, args.stage_35b)
    try:
        with SampledMemory() as sample:
            began = perf_counter()
            case = CASES[0]
            selection = service.select(case.question, args.memory_tokens)
            service.close()
            setup_s = reasoner.start(_status)
            request_started = perf_counter()
            result = _answer(reasoner, case, {"evidence": _evidence(selection.evidence, "excerpt")}, "direct", args)
            first = result["reasoner_first_final_text_s"]
            result.update(temperature="process-cold", end_to_end_s=perf_counter() - began,
                          setup_35b_s=setup_s, separate_retrieval_component_s=selection.retrieval_s,
                          separate_selection_tokenization_s=selection.tokenization_s,
                          request_to_first_final_text_s=request_started - began + first if first is not None else None)
        report["cold_direct_memory"] = sample.report()
        report["reasoner_runs"].append(result)
        _write_report(args.out, report)
    finally:
        try:
            service.close()
        finally:
            reasoner.close()


def _stop(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-3b", type=Path, required=True)
    parser.add_argument("--runtime-3b", type=Path, required=True)
    parser.add_argument("--python-3b", type=Path, required=True)
    parser.add_argument("--model-35b", type=Path, required=True)
    parser.add_argument("--stage-35b", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory-tokens", type=int, default=256, choices=range(1, 513), metavar="1..512")
    parser.add_argument("--reasoner-tokens", type=int, default=384, choices=range(1, 2049), metavar="1..2048")
    parser.add_argument("--memory-timeout", type=float, default=180)
    parser.add_argument("--cold-direct", action="store_true", help="Add a second 35B startup for the process-cold direct arm")
    parser.add_argument("--skip-full", action="store_true", help="Omit the full-context reference arm")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output path.")
    signal.signal(signal.SIGTERM, _stop)
    report = {
        "outcome": "incomplete", "reasoner_runs": [],
        "settings": {"memory_output_budget": args.memory_tokens, "reasoner_output_budget": args.reasoner_tokens,
                     "reasoner_temperature": 0, "reasoner_thinking": False, "reasoner_expert_cache_gib": 15,
                     "cold_direct_enabled": args.cold_direct},
        "method": (
            "Fixed benchmark records and held-out paraphrases. Both paired arms start from the same selected "
            "excerpts. Full uses all three records. Process-cold B includes worker startup and observed 3B "
            "release through 35B ready. Warm-reasoner B reuses a previously measured 3B recall, so it has "
            "no measured warm pipeline latency. OS file caches were left intact. The scoring rubric checks "
            "selected facts and known failure modes. Review the saved final answers for other unsupported claims."
        ),
    }
    try:
        with comparison_lease(), TemporaryDirectory(prefix="lightreason-memory-comparison-") as temporary:
            store = Store(Path(temporary) / "demo.sqlite3", "benchmark", "lightreason-history")
            records = seed_demo(store, args.artifacts)
            report["input_sources"] = _evidence(records, "content")
            _paired_phase(args, store, records, report)
            if args.cold_direct:
                _cold_direct_phase(args, store, report)
            report["outcome"] = "complete"
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _write_report(args.out, report)
    print(args.out)


if __name__ == "__main__":
    main()
