"""Verified TRAINED projector + frozen PARD2/core/target paired benchmark.

The original paired benchmark owns all request timers, raw HF baselines and
token records. Full target/core/projector/file/source audits bracket the WHOLE
run outside those timers. All online conditioning, genuine prefill, Feature4
extraction, projection, copies, tree/controller and verification costs remain
inside the unchanged runner timings. Diagnostics are not admitted here.

The quality factory uses an explicit expected projector SHA in factory args to
bind resume scope; a one-shot source callback audits the entire collector chunk
without adding per-request full byte scans. No frozen file/model is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from benchmark_pard2_tree3b_v2 import evaluate_gates as original_gates,EXPECTED_TARGET_SHA


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb")as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def source_hashes():return{p.name:file_sha(p)for p in sorted(Path(__file__).parent.glob("*.py"))if p.is_file()}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def _valid_sha(value):return isinstance(value,str)and len(value)==64 and all(c in"0123456789abcdef"for c in value)


def full_audit(runner):
    """Unwrap the feature delegate, retaining the actual core/projector objects."""
    started=time.perf_counter()
    from audit_pard2_state_coverage_v1 import audit_loaded_checkpoint_weights
    proposer=getattr(runner.proposer,"conditioned",runner.proposer)
    base=proposer.base
    if proposer.model is not base.model or proposer.adapter.core is not base.model or runner.verifier.model is not runner.model:
        raise RuntimeError("Audit/core/target objects differ from actual inference objects")
    admission=proposer.metadata.get("checkpoint_admission",{})
    if (proposer.metadata.get("diagnostic_test_mode")is not False or
            admission.get("kind")!="verified_training_segment"or admission.get("segment_integrity_verified")is not True):
        raise ValueError("Benchmark/quality requires a verified training segment; diagnostic checkpoints are forbidden")
    record=runner.audit_integrity(verify_bytes=True,verify_files=True)
    record.update(auxiliary_live_parameter_sha256=record["auxiliary_core_live_parameter_sha256"],
        pinned_auxiliary_files_reverified=True,
        original_auxiliary_checkpoint_weights=audit_loaded_checkpoint_weights(base),
        original_target_live_sha256=runner.metadata["original_target_live_sha256"],
        projector_checkpoint_path=str(proposer.projector_checkpoint_path),
        projector_checkpoint_sha256=file_sha(proposer.projector_checkpoint_path),
        runtime_projector_full_live_sha256=record["projector_runtime_sha256"],
        projector_admission=admission,diagnostic_test_mode=False,
        actual_feature_bound_delegate_unwrapped=True,
        source_sha256=source_hashes())
    record["excluded_whole_run_audit_wall_s"]=time.perf_counter()-started
    return record


def identity_gates(before,after,expected_projector_sha256):
    before,after=before or{},after or{}
    raw=lambda record:record.get("original_auxiliary_checkpoint_weights",{})
    return {
        "core_module_version_file_tokenizer_guards_passed":all(
            record.get("auxiliary_parameter_module_versions_unchanged")is True and
            record.get("original_target_tokenizer_unchanged")is True and
            record.get("pinned_auxiliary_files_reverified")is True for record in(before,after)),
        "original_target_full_live_audits_passed":all(record.get("original_target_live_sha256")==EXPECTED_TARGET_SHA and
            record.get("original_target_identity_checked")is True and record.get("target_live_bytes_verified")is True
            for record in(before,after)),
        "projector_checkpoint_full_bytes_unchanged":_valid_sha(expected_projector_sha256)and
            before.get("projector_checkpoint_sha256")==after.get("projector_checkpoint_sha256")==expected_projector_sha256 and
            all(record.get("projector_checkpoint_unchanged")is True for record in(before,after)),
        "projector_full_live_bytes_and_versions_unchanged":_valid_sha(before.get("runtime_projector_full_live_sha256"))and
            before.get("runtime_projector_full_live_sha256")==after.get("runtime_projector_full_live_sha256")and
            all(record.get("projector_parameter_module_identity_unchanged")is True for record in(before,after)),
        "verified_training_segment_and_no_diagnostic_mode":all(record.get("diagnostic_test_mode")is False and
            record.get("projector_admission",{}).get("kind")=="verified_training_segment"and
            record.get("projector_admission",{}).get("segment_integrity_verified")is True for record in(before,after))and
            bool(before.get("projector_admission"))and before.get("projector_admission")==after.get("projector_admission"),
        "full_raw_auxiliary_tensor_audit_before_and_after":all(
            raw(record).get("all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly")is True and
            raw(record).get("tensor_keys_checked",0)>0 for record in(before,after))and raw(before)==raw(after),
        "whole_run_all_project_python_sources_unchanged":bool(before.get("source_sha256"))and
            before.get("source_sha256")==after.get("source_sha256")}


def evaluate_gates(result,before,after,*,expected_projector_sha256,completed,prompts,repeats,tokens,audit_error=None):
    gates=original_gates(result,before,after,completed=completed,prompts=prompts,repeats=repeats,tokens=tokens,audit_error=audit_error)
    gates.pop("complete_benchmark_and_integrity_passed")
    gates.update(identity_gates(before,after,expected_projector_sha256))
    gates["complete_benchmark_and_integrity_passed"]=all(gates.values())
    return gates


def main(argv=None):
    values=list(sys.argv[1:]if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projector-checkpoint-path",required=True)
    parser.add_argument("--projector-checkpoint-sha256")
    parser.add_argument("--proposal-mode")
    parser.add_argument("--draft-path",default="/tmp/lightreason-models/pard2-qwen3-8b")
    parser.add_argument("--max-len",type=int,default=2048)
    parser.add_argument("--node-budget",type=int,default=64)
    parser.add_argument("--draft-tokens",type=int,default=15)
    parser.add_argument("--out",default="results3b/conditioned-pard2-tree-audited-v1.json")
    parser.add_argument("--prompts",type=int,default=2);parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--tokens",type=int,default=128)
    selected,remainder=parser.parse_known_args(values)
    if selected.proposal_mode not in(None,"tree"):parser.error("This explicitly selected conditioned runner uses tree mode")
    checkpoint=Path(selected.projector_checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():parser.error("Provide the immutable trained projector checkpoint")
    expected=file_sha(checkpoint)
    if selected.projector_checkpoint_sha256 is not None and selected.projector_checkpoint_sha256!=expected:
        parser.error("Projector checkpoint differs from explicitly supplied SHA")
    destination=Path(selected.out).expanduser().resolve()
    if destination.exists():parser.error("Use a fresh --out path; preserve existing evidence")
    forwarded=[*remainder,"--proposal-mode","tree","--draft-path",selected.draft_path,
        "--max-len",str(selected.max_len),"--node-budget",str(selected.node_budget),"--draft-tokens",str(selected.draft_tokens),
        "--out",str(destination),"--prompts",str(selected.prompts),"--repeats",str(selected.repeats),"--tokens",str(selected.tokens)]
    from conditioned_pard2_tree3b_v1 import ConditionedPARD2Tree3BRunner
    import benchmark_numerical3b
    import numerical3b_tree_runner
    runners=[];audit={"before":None,"after":None,"error":None,"expected_projector_checkpoint_sha256":expected}
    class AuditedRunner(ConditionedPARD2Tree3BRunner):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,projector_checkpoint_path=str(checkpoint),diagnostic_test_mode=False,**kwargs)
            if runners:raise RuntimeError("Expected exactly one conditioned paired-benchmark candidate")
            runners.append(self)
            audit["before"]=full_audit(self)
            if audit["before"]["projector_checkpoint_sha256"]!=expected:raise RuntimeError("Projector changed during constructor/setup")
            self.metadata["whole_benchmark_conditioned_integrity_before"]=audit["before"]
            self.metadata["whole_benchmark_audit_wrapper"]={"source":Path(__file__).name,"sha256":file_sha(__file__),
                "scope":"Full target/core/projector/checkpoint/raw-core/source audits bracketing whole run, outside request timers",
                "all_online_conditioning_prefill_features_projection_copies_tree_and_verification_remain_timed":True}
    original,old_argv=numerical3b_tree_runner.Numerical3BTreeRunner,sys.argv
    original_error,original_traceback,completed=None,None,False
    try:
        numerical3b_tree_runner.Numerical3BTreeRunner=AuditedRunner
        sys.argv=[str(Path(__file__)),*forwarded]
        benchmark_numerical3b.main();completed=True
    except BaseException as exc:original_error,original_traceback=exc,exc.__traceback__
    finally:
        numerical3b_tree_runner.Numerical3BTreeRunner=original;sys.argv=old_argv
    audit_exception,audit_traceback=None,None
    try:
        if len(runners)==1:audit["after"]=full_audit(runners[0])
        else:audit["error"]="No completed conditioned candidate constructor available"
    except BaseException as exc:
        audit["error"]=type(exc).__name__+": "+str(exc)
        audit_exception,audit_traceback=exc,exc.__traceback__
    # Persist a failing audit even if the underlying benchmark failed before
    # its first trial; no incomplete run can acquire a passing summary.
    gates=None
    try:
        result=json.loads(destination.read_text())if destination.exists()else{"environment":{},"trials":[],"summary":{},"underlying_artifact_missing":True}
        gates=evaluate_gates(result,audit["before"],audit["after"],expected_projector_sha256=expected,
            completed=completed,prompts=selected.prompts,repeats=selected.repeats,tokens=selected.tokens,audit_error=audit["error"])
        result.setdefault("environment",{})["whole_benchmark_conditioned_integrity"]={**audit,"gates":gates,
            "underlying_exception":None if original_error is None else type(original_error).__name__+": "+str(original_error),
            "all_full_audits_excluded_from_request_timers":True}
        summary=result.setdefault("summary",{})
        summary.update(complete_benchmark_and_integrity_passed=gates["complete_benchmark_and_integrity_passed"],
            auxiliary_full_live_bytes_unchanged=gates["auxiliary_full_live_bytes_unchanged"],
            projector_checkpoint_and_live_bytes_unchanged=gates["projector_checkpoint_full_bytes_unchanged"]and
                gates["projector_full_live_bytes_and_versions_unchanged"],
            all_requested_paired_trials_complete=gates["all_requested_paired_trials_complete"])
        if not gates["complete_benchmark_and_integrity_passed"]:
            summary.update(ten_x_timing_demonstrated=False,timing_evidence_status="Incomplete or failed integrity; not passing speed evidence")
        else:summary["timing_evidence_status"]="Complete selected paired protocol, verified original target/core/projector identities; answer quality separate"
        atomic_json(destination,result)
        print(json.dumps({"whole_benchmark_conditioned_integrity_gates":gates}),flush=True)
    except BaseException as exc:
        if original_error is not None:
            print("Conditioned post-run artifact/audit update also failed: "+type(exc).__name__+": "+str(exc),file=sys.stderr)
        else:raise
    if original_error is not None:raise original_error.with_traceback(original_traceback)
    if audit_exception is not None:raise audit_exception.with_traceback(audit_traceback)
    if gates is None or not gates["complete_benchmark_and_integrity_passed"]:raise RuntimeError("Whole-run conditioned benchmark integrity/completeness failed: "+str(audit["error"]))


def make_quality_runner(model,tokenizer,args):
    """Gold-blind factory with projector SHA bound to collector resume scope."""
    from conditioned_pard2_tree3b_v1 import ConditionedPARD2Tree3BRunner
    from numerical3b_runner import NumericalGenerationAdapter
    if getattr(args,"diagnostic_test_mode",False):raise ValueError("Quality factory does not admit diagnostic checkpoints")
    expected=getattr(args,"projector_checkpoint_sha256",None)
    checkpoint=Path(args.projector_checkpoint_path).expanduser().resolve()
    if not _valid_sha(expected)or file_sha(checkpoint)!=expected:
        raise ValueError("Supply projector_checkpoint_sha256 in factory args to bind immutable bytes across resumed chunks")
    collector=sys.modules.get("run_numeric_quality")
    if collector is None or not hasattr(collector,"critical_sources"):
        raise RuntimeError("This whole-chunk-audited factory requires the frozen run_numeric_factory_quality collector")
    names=("node_budget","draft_tokens","top_k","compile_static","compile_draft","cuda_graph","share_pool",
           "warmups","validate_topology","commit_engine","tree_policy","attention_buckets")
    runner=ConditionedPARD2Tree3BRunner(model,tokenizer,args.draft_path,projector_checkpoint_path=str(checkpoint),
        max_len=args.max_len,diagnostic_test_mode=False,**{name:getattr(args,name)for name in names if hasattr(args,name)})
    before=full_audit(runner)
    if before["projector_checkpoint_sha256"]!=expected:raise RuntimeError("Projector changed while quality candidate was loading")
    metadata={"before":before,"after":None,"gates":None,"expected_projector_checkpoint_sha256":expected,
        "scope":"Whole collector chunk, no per-request full scans; quality latency is not speed evidence"}
    runner.metadata["whole_quality_chunk_conditioned_integrity"]=metadata
    original_sources=collector.critical_sources
    def final_sources(engine,draft_engine):
        try:
            metadata["after"]=full_audit(runner)
            gates=identity_gates(before,metadata["after"],expected)
            # Reuse auxiliary bytes/modules/files/tokenizer guards too.
            gates["auxiliary_full_live_bytes_unchanged"]=(before["auxiliary_live_parameter_sha256"]==metadata["after"]["auxiliary_live_parameter_sha256"])
            metadata["gates"]=gates
            if not all(gates.values()):raise RuntimeError("Whole quality chunk core/projector/source integrity failed")
        finally:collector.critical_sources=original_sources
        return original_sources(engine,draft_engine)
    collector.critical_sources=final_sources
    configured=model.generation_config.eos_token_id
    eos=[configured]if isinstance(configured,int)else list(configured or [])
    if tokenizer.eos_token_id is not None:eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner,tokenizer,sorted(set(eos)))


def self_test():
    from copy import deepcopy
    sha="a"*64
    proof={"kind":"verified_training_segment","segment_integrity_verified":True,"updates":25}
    audit={"auxiliary_live_parameter_sha256":"b"*64,"auxiliary_parameter_module_versions_unchanged":True,
        "original_target_tokenizer_unchanged":True,"pinned_auxiliary_files_reverified":True,
        "original_target_live_sha256":EXPECTED_TARGET_SHA,"original_target_identity_checked":True,"target_live_bytes_verified":True,
        "projector_checkpoint_sha256":sha,"projector_checkpoint_unchanged":True,
        "runtime_projector_full_live_sha256":"c"*64,"projector_parameter_module_identity_unchanged":True,
        "diagnostic_test_mode":False,"projector_admission":proof,"source_sha256":{"runtime.py":"d"*64},
        "original_auxiliary_checkpoint_weights":{"all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly":True,"tensor_keys_checked":3}}
    result={"environment":{"target_parameter_bytes_sha256_before":EXPECTED_TARGET_SHA,"target_parameter_bytes_sha256_after":EXPECTED_TARGET_SHA,
        "target_parameters_unchanged":True,"project_python_sources_unchanged":True},"summary":{"trials":2},"trials":[
        {"prompt_index":0,"repeat":r,"baseline_tokens":[1,2],"candidate_tokens":[1,2]}for r in range(2)]}
    check=lambda data,b,a,done=True:evaluate_gates(data,b,a,expected_projector_sha256=sha,completed=done,prompts=1,repeats=2,tokens=2)
    assert check(result,audit,audit)["complete_benchmark_and_integrity_passed"]
    for key,value in(("auxiliary_live_parameter_sha256","e"*64),("runtime_projector_full_live_sha256","f"*64),
            ("projector_checkpoint_sha256","0"*64),("diagnostic_test_mode",True),("source_sha256",{})):
        changed=deepcopy(audit);changed[key]=value
        assert not check(result,audit,changed)["complete_benchmark_and_integrity_passed"]
    for key in("segment_integrity_verified","kind"):
        changed=deepcopy(audit);changed["projector_admission"][key]=False
        assert not check(result,audit,changed)["complete_benchmark_and_integrity_passed"]
    partial=deepcopy(result);partial["trials"].pop()
    assert not check(partial,audit,audit)["complete_benchmark_and_integrity_passed"]
    duplicate=deepcopy(result);duplicate["trials"][1]["repeat"]=0
    assert not check(duplicate,audit,audit)["complete_benchmark_and_integrity_passed"]
    assert not check(result,audit,audit,False)["complete_benchmark_and_integrity_passed"]
    return {"CPU_whole_core_projector_checkpoint_source_admission_partial_and_failure_gates_passed":True,"GPU_work_performed":False}


if __name__=="__main__":
    if sys.argv[1:]==["--self-test"]:print(json.dumps(self_test(),indent=2))
    else:main()
