"""Whole-run auxiliary integrity audit around the unchanged paired benchmark.

All request timings and target gates come from the frozen benchmark. Actual
full auxiliary live hashes and module/version/file/tokenizer audits bracket the
WHOLE run outside request timing. A failed/interrupted/partial underlying run
cannot acquire a passing summary, and its original exception is re-raised.
Use a fresh --out path. Optional --audit-state-coverage adds metadata coverage
of every stored auxiliary tensor against the already loaded state dictionary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


EXPECTED_TARGET_SHA="ab4dc8672b41311882ed404da089baad41db202eda849c68c14df3352c06e9a2"


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def evaluate_gates(result,before,after,*,completed,prompts,repeats,tokens,audit_error=None):
    """No partial/mixed trial set or unaudited bytes can become passing evidence."""
    rows=result.get("trials",[]);summary=result.get("summary",{});env=result.get("environment",{})
    pairs={(row.get("prompt_index"),row.get("repeat")) for row in rows}
    expected={(p,r) for p in range(prompts) for r in range(repeats)}
    all_lengths=all(len(row.get(name,[]))==tokens for row in rows
                    for name in ("baseline_tokens","candidate_tokens"))
    before_sha=before.get("auxiliary_live_parameter_sha256") if before else None
    after_sha=after.get("auxiliary_live_parameter_sha256") if after else None
    valid_hash=lambda value:isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)
    gates={"underlying_benchmark_returned_successfully":bool(completed),
        "all_requested_paired_trials_complete":len(rows)==len(expected) and pairs==expected and
            summary.get("trials")==len(expected) and all_lengths,
        "original_target_live_SHA_unchanged":env.get("target_parameter_bytes_sha256_before")==
            env.get("target_parameter_bytes_sha256_after")==EXPECTED_TARGET_SHA and env.get("target_parameters_unchanged") is True,
        "original_benchmark_source_guard_passed":env.get("project_python_sources_unchanged") is True,
        "auxiliary_full_live_bytes_unchanged":valid_hash(before_sha) and before_sha==after_sha,
        "all_original_auxiliary_tensor_bytes_loaded_exactly":before is not None and
            before.get("original_auxiliary_checkpoint_weights",{}).get("all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly") is True and
            before["original_auxiliary_checkpoint_weights"].get("tensor_keys_checked",0)>0,
        "auxiliary_module_version_file_tokenizer_audit_passed":audit_error is None and
            before is not None and after is not None and all(a.get("auxiliary_parameter_module_versions_unchanged") is True and
                a.get("original_target_tokenizer_unchanged") is True and a.get("pinned_auxiliary_files_reverified") is True
                for a in (before,after))}
    gates["complete_benchmark_and_integrity_passed"]=all(gates.values())
    return gates


def main(argv=None):
    values=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--proposal-mode");parser.add_argument("--out",default="results3b/pard2-tree-audited-v2.json")
    parser.add_argument("--prompts",type=int,default=2);parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--tokens",type=int,default=128)
    parser.add_argument("--audit-state-coverage",action="store_true")
    selected,remainder=parser.parse_known_args(values)
    if selected.proposal_mode not in (None,"tree"):parser.error("Select tree proposals for this original-target-verified adapter")
    destination=Path(selected.out).expanduser().resolve()
    if destination.exists():parser.error("Use a fresh --out path; preserve existing evidence")
    # Rebuild consumed protocol arguments; the frozen parser remains final
    # authority for all underlying options and ordinary CLI error behavior.
    forwarded=[*remainder,"--proposal-mode","tree","--out",str(destination),
        "--prompts",str(selected.prompts),"--repeats",str(selected.repeats),"--tokens",str(selected.tokens)]
    from pard2_proposer3b_v1 import PARD2Numerical3BTreeRunner
    import benchmark_numerical3b
    import numerical3b_tree_runner
    runners=[];audit={"before":None,"after":None,"error":None,"state_coverage_requested":selected.audit_state_coverage}
    def full_audit(base,*,check_original_weights=False):
        started=time.perf_counter();record=base.audit_integrity(verify_bytes=True,verify_files=True)
        record["pinned_auxiliary_files_reverified"]=True
        if check_original_weights:
            from audit_pard2_state_coverage_v1 import audit_loaded_checkpoint_weights
            record["original_auxiliary_checkpoint_weights"]=audit_loaded_checkpoint_weights(base)
        if selected.audit_state_coverage:
            from audit_pard2_state_coverage_v1 import audit_loaded_auxiliary
            record["full_state_metadata_coverage"]=audit_loaded_auxiliary(base)
        record["excluded_audit_wall_s"]=time.perf_counter()-started
        return record
    class AuditedRunner(PARD2Numerical3BTreeRunner):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if runners:raise RuntimeError("Expected exactly one paired-benchmark candidate instance")
            runners.append(self)
            audit["before"]=full_audit(self.proposer.base,check_original_weights=True)
            self.metadata["whole_benchmark_auxiliary_integrity_before"]=audit["before"]
            self.metadata["whole_benchmark_audit_wrapper"]={"source":Path(__file__).name,
                "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audit_scope":"One full live hash/module/version/file/tokenizer audit before and after the whole benchmark; outside request timing"}
    previous,previous_argv=numerical3b_tree_runner.Numerical3BTreeRunner,sys.argv
    original_error,original_traceback,completed=None,None,False
    try:
        numerical3b_tree_runner.Numerical3BTreeRunner=AuditedRunner
        sys.argv=[str(Path(__file__)),*forwarded]
        benchmark_numerical3b.main();completed=True
    except BaseException as error:
        original_error,original_traceback=error,error.__traceback__
    finally:
        numerical3b_tree_runner.Numerical3BTreeRunner=previous;sys.argv=previous_argv
    try:
        if len(runners)==1:
            audit["after"]=full_audit(runners[0].proposer.base)
        else:
            audit["error"]="No completed candidate constructor was available for the whole-run audit"
    except BaseException as error:
        audit["error"]=f"{type(error).__name__}: {error}"
    gates=None
    if destination.exists():
        try:
            result=json.loads(destination.read_text())
            gates=evaluate_gates(result,audit["before"],audit["after"],completed=completed,
                prompts=selected.prompts,repeats=selected.repeats,tokens=selected.tokens,audit_error=audit["error"])
            result.setdefault("environment",{})["whole_benchmark_auxiliary_integrity"]={**audit,
                "underlying_exception":None if original_error is None else f"{type(original_error).__name__}: {original_error}",
                "all_audits_excluded_from_request_timing":True,"gates":gates}
            summary=result.setdefault("summary",{})
            summary["complete_benchmark_and_integrity_passed"]=gates["complete_benchmark_and_integrity_passed"]
            summary["auxiliary_full_live_bytes_unchanged"]=gates["auxiliary_full_live_bytes_unchanged"]
            summary["all_requested_paired_trials_complete"]=gates["all_requested_paired_trials_complete"]
            if not gates["complete_benchmark_and_integrity_passed"]:
                summary["ten_x_timing_demonstrated"]=False
                summary["timing_evidence_status"]="Incomplete or failed integrity; not passing speed evidence"
            else:
                summary["timing_evidence_status"]="Complete selected paired protocol with original target and auxiliary integrity gates"
            atomic_json(destination,result)
            print(json.dumps({"whole_benchmark_auxiliary_integrity_gates":gates}),flush=True)
        except BaseException as error:
            if original_error is not None:
                print("Post-run audit/artifact update also failed: "+f"{type(error).__name__}: {error}",file=sys.stderr,flush=True)
            else:raise
    if original_error is not None:
        raise original_error.with_traceback(original_traceback)
    if gates is None or not gates["complete_benchmark_and_integrity_passed"]:
        raise RuntimeError("Whole-run auxiliary/target/source/completeness audit failed: "+str(audit["error"]))


def self_test():
    from copy import deepcopy
    sha="a"*64
    audit={"auxiliary_live_parameter_sha256":sha,"auxiliary_parameter_module_versions_unchanged":True,
           "original_target_tokenizer_unchanged":True,"pinned_auxiliary_files_reverified":True,
           "original_auxiliary_checkpoint_weights":{"all_original_auxiliary_checkpoint_tensor_bytes_loaded_exactly":True,
               "tensor_keys_checked":3}}
    result={"environment":{"target_parameter_bytes_sha256_before":EXPECTED_TARGET_SHA,
        "target_parameter_bytes_sha256_after":EXPECTED_TARGET_SHA,"target_parameters_unchanged":True,
        "project_python_sources_unchanged":True},"summary":{"trials":2},"trials":[
        {"prompt_index":0,"repeat":r,"baseline_tokens":[1,2],"candidate_tokens":[1,2]} for r in range(2)]}
    check=lambda d,b,a,done=True:evaluate_gates(d,b,a,completed=done,prompts=1,repeats=2,tokens=2)
    assert check(result,audit,audit)["complete_benchmark_and_integrity_passed"]
    changed=deepcopy(audit);changed["auxiliary_live_parameter_sha256"]="b"*64
    assert not check(result,audit,changed)["complete_benchmark_and_integrity_passed"]
    assert not check(result,audit,audit,False)["complete_benchmark_and_integrity_passed"]
    partial=deepcopy(result);partial["trials"].pop()
    assert not check(partial,audit,audit)["complete_benchmark_and_integrity_passed"]
    mixed=deepcopy(result);mixed["trials"][1]["repeat"]=0
    assert not check(mixed,audit,audit)["complete_benchmark_and_integrity_passed"]
    assert not check(result,None,audit)["complete_benchmark_and_integrity_passed"]
    return {"CPU_full_run_hash_mutation_partial_failure_duplicate_and_missing_audit_gates_passed":True,
            "GPU_work_performed":False}


if __name__=="__main__":
    if sys.argv[1:]==["--self-test"]:print(json.dumps(self_test()))
    else:main()
