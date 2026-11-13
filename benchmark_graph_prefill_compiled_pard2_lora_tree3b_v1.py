"""New combined prefill variant in the FROZEN paired original-target benchmark.

Original HF baseline/timers/controller and whole-run source/raw-reference/
effective/target/projector/checkpoint audits are inherited unchanged. Scoped
constructor selection restores on success, partial trial failure or audit error.
Capture setup is separate; every online prefill copy/projection remains timed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


@contextmanager
def runner_binding(module,replacement):
    original=module.FoldedPARD2LoRATree3BRunner
    module.FoldedPARD2LoRATree3BRunner=replacement
    try:yield
    finally:module.FoldedPARD2LoRATree3BRunner=original


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prefill-cache-lengths",type=int,default=4)
    parser.add_argument("--prefill-warmups",type=int,default=2)
    parser.add_argument("--out",default="results3b/graph-prefill-compiled-folded-pard2-lora-v1.json")
    args,values=parser.parse_known_args(list(sys.argv[1:]if argv is None else argv))
    from graph_prefill_compiled_pard2_lora_tree3b_v1 import GraphPrefillCompiledPARD2LoRATree3BRunner,checked_capacity
    checked_capacity(args.prefill_cache_lengths)
    if args.prefill_warmups not in(1,2,3):parser.error("Use public prefill warmups1..3")
    destination=Path(args.out).expanduser().resolve()
    if destination.exists():parser.error("Use a fresh artifact path")
    values.extend(("--out",str(destination)))
    if "--compile-draft"not in values:values.append("--compile-draft")
    import folded_pard2_lora_tree3b_v1 as runtime
    from benchmark_folded_pard2_lora_tree3b_v1 import main as audited_benchmark
    from benchmark_conditioned_pard2_tree3b_v1 import file_sha,atomic_json
    runners=[]
    class SelectedRunner(GraphPrefillCompiledPARD2LoRATree3BRunner):
        def __init__(self,*positional,**kwargs):
            super().__init__(*positional,prefill_cache_lengths=args.prefill_cache_lengths,prefill_warmups=args.prefill_warmups,**kwargs)
            runners.append(self)
            self.metadata["combined_prefill_benchmark_wrapper"]={"source":Path(__file__).name,"sha256":file_sha(__file__),
                "original_baseline_and_paired_request_timer_unchanged":True,
                "online_target_and_auxiliary_prefill_graph_replays_and_all_copies_timed":True,
                "prefill_capture_public_shapes_only_excluded_and_reported_separately":True}
    original,error_tb,completed=None,None,False
    try:
        with runner_binding(runtime,SelectedRunner):audited_benchmark(values)
        completed=True
    except BaseException as exc:original,error_tb=exc,exc.__traceback__
    gates={"frozen_audited_benchmark_returned_successfully":completed,"exactly_one_actual_combined_runner":len(runners)==1}
    if len(runners)==1:
        runner=runners[0];owner=runner.verifier._owner
        aux_counts=runner.proposer.conditioned.prefill_dispatch_counts
        gates.update(original_true_target_owner_is_still_feature_bound=runner.proposer.verifier is owner,
            graph_prefill_enabled_for_entire_benchmark=runner.metadata.get("graph_prefill_enabled")is True and
                runner.verifier._enabled is True and runner.proposer.conditioned.graph_prefill_enabled is True,
            every_candidate_target_prefill_used_prepared_graph=runner.verifier._counts["original"]==0 and runner.verifier._counts["graph"]>0,
            every_candidate_auxiliary_prefill_used_prepared_graph=aux_counts["original"]==0 and
                aux_counts["unprepared_original_fallback"]==0 and aux_counts["graph"]>0,
            prefill_bank_within_explicit_limit=len(runner._prefill_lengths)<=args.prefill_cache_lengths,
            no_source_or_model_forward_monkeypatch=runner.metadata.get("target_or_auxiliary_model_forward_replaced")is False)
    try:
        result=json.loads(destination.read_text())if destination.exists()else{"environment":{},"summary":{},"trials":[],"underlying_artifact_missing":True}
        gates["frozen_complete_benchmark_and_whole_identity_audits_passed"]=result.get("summary",{}).get("complete_benchmark_and_integrity_passed")is True
        metadata={"source":Path(__file__).name,"sha256":file_sha(__file__),"gates":gates,
            "constructor_binding_restored":runtime.FoldedPARD2LoRATree3BRunner is not SelectedRunner,
            "prefill_cache_lengths":args.prefill_cache_lengths,"prefill_warmups":args.prefill_warmups,
            "underlying_error":None if original is None else type(original).__name__+": "+str(original),
            "scope":"Integrated single-request paired wall latency; component ratios are not speedup evidence",
            "quality_assessed":False}
        if len(runners)==1:metadata.update(excluded_prefill_graph_setup_s=runners[0].prefill_setup_s,prefill_bank=runners[0].prefill_bank_metadata)
        result.setdefault("environment",{})["combined_prefill_integration"]=metadata
        summary=result.setdefault("summary",{});summary["combined_prefill_complete_benchmark_and_integrity_passed"]=all(gates.values())
        if not all(gates.values()):summary.update(complete_benchmark_and_integrity_passed=False,ten_x_timing_demonstrated=False)
        atomic_json(destination,result)
    except BaseException as exc:
        if original is None:raise
        print("Combined artifact update also failed: "+type(exc).__name__+": "+str(exc),file=sys.stderr)
    if original is not None:raise original.with_traceback(error_tb)
    if not all(gates.values()):raise RuntimeError("Combined prefill benchmark/provenance gate failed")


def self_test():
    from types import SimpleNamespace
    from graph_prefill_compiled_pard2_lora_tree3b_v1 import self_test as runtime_test
    original,new=object(),object();module=SimpleNamespace(FoldedPARD2LoRATree3BRunner=original)
    with runner_binding(module,new):assert module.FoldedPARD2LoRATree3BRunner is new
    assert module.FoldedPARD2LoRATree3BRunner is original
    failure=RuntimeError("partial trial/audit error fixture")
    try:
        with runner_binding(module,new):raise failure
    except RuntimeError as exc:assert exc is failure
    assert module.FoldedPARD2LoRATree3BRunner is original
    return {**runtime_test(),"CPU_actual_benchmark_constructor_binding_success_failure_restoration_passed":True,
        "original_failure_identity_preserved":True}


if __name__=="__main__":
    if sys.argv[1:]==["--self-test"]:print(json.dumps(self_test(),indent=2))
    else:main()
