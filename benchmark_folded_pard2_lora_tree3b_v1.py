"""Paired original-target timing with an intentionally ADAPTED auxiliary.

The pinned reference alone gets raw-checkpoint equality audits. The separate
effective folded model gets its own full hash/version/module/fold-lineage
checks. No unchanged-original-auxiliary claim is applied to adapted weights.
All audits/folding are untimed setup; every online conditioning/proposal/tree
cost remains in the frozen paired benchmark. Quality remains a separate test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from benchmark_conditioned_pard2_tree3b_v1 import file_sha,source_hashes,atomic_json,_valid_sha,EXPECTED_TARGET_SHA


def full_audit(runner):
    from audit_pard2_state_coverage_v1 import audit_loaded_checkpoint_weights
    began=time.perf_counter()
    proposer=getattr(runner.proposer,"conditioned",runner.proposer)
    effective=proposer.base;reference=effective.reference
    if effective.model is not proposer.model or proposer.adapter.core is not effective.model or runner.verifier.model is not runner.model:
        raise RuntimeError("Adapted audit does not refer to the actual online inference models")
    admission=proposer.metadata["checkpoint_admission"]
    if admission.get("kind")!="verified_LoRA_projector_training_segment"or admission.get("segment_integrity_verified")is not True:
        raise ValueError("Use a verified published LoRA/projector checkpoint")
    record=runner.audit_integrity(verify_bytes=True,verify_files=True)
    record.update(original_reference_raw_checkpoint_audit=audit_loaded_checkpoint_weights(reference),
        original_reference_files_reverified=True,
        adaptation_checkpoint_sha256=file_sha(proposer.projector_checkpoint_path),
        adaptation_checkpoint_bytes_unchanged=record["projector_checkpoint_unchanged"],
        projector_full_live_sha256=record["projector_runtime_sha256"],
        original_target_live_sha256=runner.metadata["original_target_live_sha256"],
        adaptation_admission=admission,fold_lineage=proposer.metadata["fold"],
        actual_effective_model_not_reference_is_online=True,
        effective_auxiliary_raw_equality_to_original_checkpoint_claimed=False,
        source_sha256=source_hashes())
    record["excluded_whole_run_audit_wall_s"]=time.perf_counter()-began
    return record


def identity_gates(before,after,expected):
    before,after=before or{},after or{}
    source=lambda r:r.get("original_reference_audit",{})
    raw=lambda r:r.get("original_reference_raw_checkpoint_audit",{})
    fold=lambda r:r.get("fold_lineage",{})
    return {
        "original_target_full_live_and_state_unchanged":all(r.get("original_target_live_sha256")==EXPECTED_TARGET_SHA and
            r.get("original_target_identity_checked")is True and r.get("target_live_bytes_verified")is True for r in(before,after)),
        "original_pinned_reference_full_live_bytes_unchanged":_valid_sha(before.get("original_reference_live_sha256"))and
            before.get("original_reference_live_sha256")==after.get("original_reference_live_sha256"),
        "original_pinned_reference_raw_checkpoint_equality_before_and_after":all(
            raw(r).get("all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly")is True and raw(r).get("tensor_keys_checked",0)>0
            for r in(before,after))and raw(before)==raw(after),
        "original_reference_module_version_file_tokenizer_guards":all(r.get("original_reference_files_reverified")is True and
            source(r).get("auxiliary_parameter_module_versions_unchanged")is True and
            source(r).get("original_target_tokenizer_unchanged")is True for r in(before,after)),
        "effective_adapted_auxiliary_full_bytes_and_versions_unchanged_during_run":
            _valid_sha(before.get("effective_adapted_auxiliary_live_sha256"))and
            before.get("effective_adapted_auxiliary_live_sha256")==after.get("effective_adapted_auxiliary_live_sha256")and
            all(r.get("effective_auxiliary_parameter_module_versions_unchanged")is True for r in(before,after)),
        "adaptation_checkpoint_bytes_unchanged":_valid_sha(expected)and
            before.get("adaptation_checkpoint_sha256")==after.get("adaptation_checkpoint_sha256")==expected and
            all(r.get("adaptation_checkpoint_bytes_unchanged")is True for r in(before,after)),
        "new_runtime_projector_bytes_and_versions_unchanged":_valid_sha(before.get("projector_full_live_sha256"))and
            before.get("projector_full_live_sha256")==after.get("projector_full_live_sha256")and
            all(r.get("projector_parameter_module_identity_unchanged")is True for r in(before,after)),
        "published_adaptation_segment_identity_unchanged":bool(before.get("adaptation_admission"))and
            before.get("adaptation_admission")==after.get("adaptation_admission")and all(
            r.get("adaptation_admission",{}).get("kind")=="verified_LoRA_projector_training_segment"and
            r["adaptation_admission"].get("segment_integrity_verified")is True for r in(before,after)),
        "explicit_fold_formula_nonadapted_values_noalias_and_source_checkpoint_lineage":all(
            fold(r).get("all140_folded_matrices_match_explicit_FP32_formula")is True and
            fold(r).get("all_other_state_tensors_bit_equal_original_reference")is True and
            fold(r).get("no_parameter_storage_alias_to_reference_training_base_A_B_or_projector")is True and
            fold(r).get("ordinary_HF_core_without_LoRA_wrappers")is True and fold(r).get("source_checkpoint_sha256")==expected
            for r in(before,after))and fold(before)==fold(after),
        "adapted_effective_model_is_never_mislabeled_original_checkpoint_equal":all(
            r.get("effective_auxiliary_is_intentionally_adapted")is True and
            r.get("effective_auxiliary_raw_equality_to_original_checkpoint_claimed")is False and
            r.get("actual_effective_model_not_reference_is_online")is True for r in(before,after)),
        "whole_run_source_hashes_unchanged":bool(before.get("source_sha256"))and before.get("source_sha256")==after.get("source_sha256")}


def evaluate_gates(result,before,after,*,expected,completed,prompts,repeats,tokens,audit_error=None):
    rows=result.get("trials",[]);env=result.get("environment",{});summary=result.get("summary",{})
    pairs={(r.get("prompt_index"),r.get("repeat"))for r in rows};wanted={(p,r)for p in range(prompts)for r in range(repeats)}
    gates={"underlying_benchmark_returned_successfully":bool(completed),
        "all_requested_paired_trials_complete":len(rows)==len(wanted)and pairs==wanted and summary.get("trials")==len(wanted)and
            all(len(r.get(name,[]))==tokens for r in rows for name in("baseline_tokens","candidate_tokens")),
        "original_target_timing_SHA_and_versions_passed":env.get("target_parameter_bytes_sha256_before")==
            env.get("target_parameter_bytes_sha256_after")==EXPECTED_TARGET_SHA and env.get("target_parameters_unchanged")is True,
        "original_timing_source_guard_passed":env.get("project_python_sources_unchanged")is True,
        "both_full_audits_completed_without_error":before is not None and after is not None and audit_error is None}
    gates.update(identity_gates(before,after,expected));gates["complete_benchmark_and_integrity_passed"]=all(gates.values())
    return gates


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adaptation-checkpoint-path",required=True);parser.add_argument("--adaptation-checkpoint-sha256")
    parser.add_argument("--proposal-mode");parser.add_argument("--draft-path",default="/tmp/lightreason-models/pard2-qwen3-8b")
    parser.add_argument("--max-len",type=int,default=2048);parser.add_argument("--node-budget",type=int,default=64)
    parser.add_argument("--draft-tokens",type=int,default=15);parser.add_argument("--prompts",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=2);parser.add_argument("--tokens",type=int,default=128)
    parser.add_argument("--out",default="results3b/folded-pard2-lora-tree-audited-v1.json")
    args,remainder=parser.parse_known_args(list(sys.argv[1:]if argv is None else argv))
    if args.proposal_mode not in(None,"tree"):parser.error("Select the original-target-verified tree controller")
    checkpoint=Path(args.adaptation_checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():parser.error("Provide a published immutable LoRA/projector checkpoint")
    expected=file_sha(checkpoint)
    if args.adaptation_checkpoint_sha256 is not None and args.adaptation_checkpoint_sha256!=expected:parser.error("Adaptation checkpoint SHA differs")
    destination=Path(args.out).expanduser().resolve()
    if destination.exists():parser.error("Use a fresh output path; preserve existing evidence")
    forward=[*remainder,"--proposal-mode","tree","--draft-path",args.draft_path,"--max-len",str(args.max_len),
        "--node-budget",str(args.node_budget),"--draft-tokens",str(args.draft_tokens),"--prompts",str(args.prompts),
        "--repeats",str(args.repeats),"--tokens",str(args.tokens),"--out",str(destination)]
    from folded_pard2_lora_tree3b_v1 import FoldedPARD2LoRATree3BRunner
    import benchmark_numerical3b,numerical3b_tree_runner
    runners=[];audit={"before":None,"after":None,"error":None,"expected_adaptation_checkpoint_sha256":expected}
    class AuditedRunner(FoldedPARD2LoRATree3BRunner):
        def __init__(self,*positional,**kwargs):
            super().__init__(*positional,adaptation_checkpoint_path=str(checkpoint),**kwargs)
            if runners:raise RuntimeError("Expected exactly one paired candidate")
            runners.append(self);audit["before"]=full_audit(self)
            if audit["before"]["adaptation_checkpoint_sha256"]!=expected:raise RuntimeError("Adaptation changed during setup")
            self.metadata["whole_run_separate_reference_effective_integrity_before"]=audit["before"]
            self.metadata["benchmark_wrapper"]={"source":Path(__file__).name,"sha256":file_sha(__file__),
                "reference_raw_equality_and_effective_adapted_hash_are_separate":True,
                "all_folding_and_full_audits_excluded_from_request_timers":True}
    old,oldargv=numerical3b_tree_runner.Numerical3BTreeRunner,sys.argv
    original,error_tb,completed=None,None,False
    try:
        numerical3b_tree_runner.Numerical3BTreeRunner=AuditedRunner;sys.argv=[str(Path(__file__)),*forward]
        benchmark_numerical3b.main();completed=True
    except BaseException as exc:original,error_tb=exc,exc.__traceback__
    finally:numerical3b_tree_runner.Numerical3BTreeRunner=old;sys.argv=oldargv
    audit_exc,audit_tb=None,None
    try:
        if len(runners)==1:audit["after"]=full_audit(runners[0])
        else:audit["error"]="No completed adapted candidate available"
    except BaseException as exc:audit["error"]=type(exc).__name__+": "+str(exc);audit_exc,audit_tb=exc,exc.__traceback__
    gates=None
    try:
        result=json.loads(destination.read_text())if destination.exists()else{"environment":{},"summary":{},"trials":[],"underlying_artifact_missing":True}
        gates=evaluate_gates(result,audit["before"],audit["after"],expected=expected,completed=completed,
            prompts=args.prompts,repeats=args.repeats,tokens=args.tokens,audit_error=audit["error"])
        result.setdefault("environment",{})["whole_run_reference_effective_adaptation_integrity"]={**audit,"gates":gates,
            "underlying_exception":None if original is None else type(original).__name__+": "+str(original),
            "raw_checkpoint_equality_applies_to_original_reference_only":True,
            "effective_auxiliary_is_intentionally_adapted":True,"all_full_audits_are_untimed":True}
        summary=result.setdefault("summary",{});summary["complete_benchmark_and_integrity_passed"]=gates["complete_benchmark_and_integrity_passed"]
        if not gates["complete_benchmark_and_integrity_passed"]:summary.update(ten_x_timing_demonstrated=False,timing_evidence_status="Incomplete/failed integrity; not passing speed evidence")
        else:summary["timing_evidence_status"]="Complete selected paired original-target protocol; adapted auxiliary separately identified; quality not assessed"
        atomic_json(destination,result);print(json.dumps({"reference_effective_adaptation_integrity_gates":gates}),flush=True)
    except BaseException as exc:
        if original is not None:print("Post-run artifact/audit update also failed: "+type(exc).__name__+": "+str(exc),file=sys.stderr)
        else:raise
    if original is not None:raise original.with_traceback(error_tb)
    if audit_exc is not None:raise audit_exc.with_traceback(audit_tb)
    if gates is None or not gates["complete_benchmark_and_integrity_passed"]:raise RuntimeError("Adapted auxiliary whole-run audit failed")


def self_test():
    from copy import deepcopy
    sha="a"*64
    audit={"original_reference_live_sha256":"b"*64,"effective_adapted_auxiliary_live_sha256":"c"*64,
        "original_target_live_sha256":EXPECTED_TARGET_SHA,"original_target_identity_checked":True,"target_live_bytes_verified":True,
        "original_reference_raw_checkpoint_audit":{"all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly":True,"tensor_keys_checked":3},
        "original_reference_files_reverified":True,"original_reference_audit":{"auxiliary_parameter_module_versions_unchanged":True,"original_target_tokenizer_unchanged":True},
        "effective_auxiliary_parameter_module_versions_unchanged":True,"adaptation_checkpoint_sha256":sha,"adaptation_checkpoint_bytes_unchanged":True,
        "projector_full_live_sha256":"d"*64,"projector_parameter_module_identity_unchanged":True,
        "adaptation_admission":{"kind":"verified_LoRA_projector_training_segment","segment_integrity_verified":True},
        "fold_lineage":{"all140_folded_matrices_match_explicit_FP32_formula":True,"all_other_state_tensors_bit_equal_original_reference":True,
            "no_parameter_storage_alias_to_reference_training_base_A_B_or_projector":True,"ordinary_HF_core_without_LoRA_wrappers":True,"source_checkpoint_sha256":sha},
        "effective_auxiliary_is_intentionally_adapted":True,"effective_auxiliary_raw_equality_to_original_checkpoint_claimed":False,
        "actual_effective_model_not_reference_is_online":True,"source_sha256":{"runtime.py":"f"*64}}
    result={"environment":{"target_parameter_bytes_sha256_before":EXPECTED_TARGET_SHA,"target_parameter_bytes_sha256_after":EXPECTED_TARGET_SHA,
        "target_parameters_unchanged":True,"project_python_sources_unchanged":True},"summary":{"trials":2},"trials":[
        {"prompt_index":0,"repeat":r,"baseline_tokens":[1,2],"candidate_tokens":[1,2]}for r in range(2)]}
    check=lambda data,b,a,done=True:evaluate_gates(data,b,a,expected=sha,completed=done,prompts=1,repeats=2,tokens=2)
    assert check(result,audit,audit)["complete_benchmark_and_integrity_passed"]
    # Effective may equal original when LoRABzero; never require a difference.
    zero=deepcopy(audit);zero["effective_adapted_auxiliary_live_sha256"]=zero["original_reference_live_sha256"]
    assert check(result,zero,zero)["complete_benchmark_and_integrity_passed"]
    for key,value in(("original_reference_live_sha256","e"*64),("effective_adapted_auxiliary_live_sha256","e"*64),
        ("adaptation_checkpoint_sha256","e"*64),("projector_full_live_sha256","e"*64),
        ("effective_auxiliary_raw_equality_to_original_checkpoint_claimed",True)):
        bad=deepcopy(audit);bad[key]=value;assert not check(result,audit,bad)["complete_benchmark_and_integrity_passed"]
    partial=deepcopy(result);partial["trials"].pop();assert not check(partial,audit,audit)["complete_benchmark_and_integrity_passed"]
    assert not check(result,audit,audit,False)["complete_benchmark_and_integrity_passed"]
    return {"CPU_separate_original_reference_effective_adaptation_hash_lineage_partial_and_failure_gates_passed":True,"GPU_work_performed":False}


if __name__=="__main__":
    if sys.argv[1:]==["--self-test"]:print(json.dumps(self_test(),indent=2))
    else:main()
