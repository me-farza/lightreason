"""V2 timeout-only variant: single PUBLIC Fibonacci complete native-EOS paired timing.

Exactly three alternating pairs compare live original raw-greedy Qwen3B HF
with the original target plus graph-prefill/private-compiled folded450 AUX.
This is one public prompt, never six/30-prompt or general quality evidence.
All 1024-token public-budget graph families prepare with synthetic values
BEFORE the true prompt is admitted. Every online cost, EOS check, tree/cache
operation and host token delivery is in an external synchronized call timer.

Complete IDs, raw/decoded text, first EOS and actual latencies are retained.
Any missing EOS/failed/incomplete trial or identity guard invalidates accepted
speed evidence. Full SAME-OUTPUT speedup additionally requires all full token
IDs and UTF8 strings to match. Neither an oracle generation counter nor a
cached answer is an input to generation. No gold/training data is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

SCHEMA="lightreason-single-public-fibonacci-native-eos-pairs-v2"
PROMPT_INDEX=1
PUBLIC_PROMPT="Write a Python function that returns the first n Fibonacci numbers and explain it."
MAX_NEW_TOKENS=1024
MAX_CACHE=2048
PAIR_COUNT=3
COOLDOWN_S=8.0
CHECKPOINT450_SHA="8e5b53abec75a33825afca5c08a151d2cefd0a8a567bfff2ebc507bdd4ecc6b6"
EFFECTIVE450_SHA="e7f784c2cc8cc201cf6f36155a529ddfc7fc35c5eb557714200ef5bcb60e3f13"
TARGET_SHA="ab4dc8672b41311882ed404da089baad41db202eda849c68c14df3352c06e9a2"


def require(condition,message):
    if not condition:raise ValueError(message)


def paired_orders():
    return [("baseline","candidate")if index%2==0 else("candidate","baseline")for index in range(PAIR_COUNT)]


def token_sha(tokens):
    return hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()


def first_eos_check(tokens,reason,eos,budget=MAX_NEW_TOKENS):
    valid=isinstance(tokens,list)and bool(tokens)and len(tokens)<=budget and all(type(t)is int and 0<=t<151936 for t in tokens)
    first=next((i for i,t in enumerate(tokens)if t in eos),None)if valid else None
    return {"tokens_valid":valid,"first_EOS_index":first,
        "stopped_at_first_native_EOS":valid and reason=="eos"and first==len(tokens)-1,
        "actual_tokens":len(tokens)if isinstance(tokens,list)else None,"public_max_new_tokens":budget}


def output_comparison(left,right):
    a,b=left.get("token_ids",[]),right.get("token_ids",[])
    first=next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),None)
    if first is None and len(a)!=len(b):first=min(len(a),len(b))
    exact=a==b and bool(a)
    return {"complete_token_IDs_exact":exact,"first_different_token_index":first,
        "raw_UTF8_bytes_exact":isinstance(left.get("raw_text"),str)and isinstance(right.get("raw_text"),str)and
            left["raw_text"].encode("utf8")==right["raw_text"].encode("utf8"),
        "decoded_UTF8_bytes_exact":isinstance(left.get("text"),str)and isinstance(right.get("text"),str)and
            left["text"].encode("utf8")==right["text"].encode("utf8"),
        "baseline_tokens":len(a),"candidate_tokens":len(b),
        "different_completion_lengths_or_content_can_confound_latency_ratio":not exact}


def invoke_generation(engine,model,candidate,ids,eos,baseline_function):
    """No expected/reference IDs, scores or output lengths cross this boundary."""
    if engine=="baseline":return baseline_function(model,ids,MAX_NEW_TOKENS,eos)
    if engine=="candidate":
        tokens,metrics=candidate.run(ids,MAX_NEW_TOKENS,eos_ids=eos)
        return tokens,metrics.get("stop_reason"),metrics
    raise ValueError("Unknown benchmark engine")


def summarize(rows,active_runs,*,completed,integrity_passed):
    wanted={(PROMPT_INDEX,i)for i in range(PAIR_COUNT)}
    seen={(row.get("prompt_index"),row.get("repeat"))for row in rows}
    order_exact=len(rows)==PAIR_COUNT and all(row.get("repeat")==i and row.get("order")==list(paired_orders()[i])for i,row in enumerate(rows))
    expected_sequence=[(i,engine)for i,order in enumerate(paired_orders())for engine in order]
    actual_sequence=[(row.get("repeat"),row.get("engine"))for row in active_runs]
    pairs_complete=len(rows)==PAIR_COUNT and seen==wanted and order_exact and all(
        all(row.get(name,{}).get("completed")is True for name in("baseline","candidate"))for row in rows)
    run_keys={(row.get("repeat"),row.get("engine"))for row in active_runs}
    expected_runs={(i,e)for i in range(PAIR_COUNT)for e in("baseline","candidate")}
    six_complete=len(active_runs)==2*PAIR_COUNT and run_keys==expected_runs and actual_sequence==expected_sequence and all(r.get("completed")is True for r in active_runs)
    natural=pairs_complete and all(row[name].get("native_EOS",{}).get("stopped_at_first_native_EOS")is True
        for row in rows for name in("baseline","candidate"))
    complete=bool(completed and pairs_complete and six_complete and natural and integrity_passed)
    equal=complete and all(all(row.get("comparison",{}).get(k)is True for k in(
        "complete_token_IDs_exact","raw_UTF8_bytes_exact","decoded_UTF8_bytes_exact"))for row in rows)
    ratios=[row["baseline"]["external_active_wall_s"]/row["candidate"]["external_active_wall_s"]for row in rows
        if row.get("baseline",{}).get("completed")and row.get("candidate",{}).get("completed")and
        row["baseline"].get("external_active_wall_s",0)>0 and row["candidate"].get("external_active_wall_s",0)>0]
    result={"scope":"ONE public prompt index1 ONLY; single-request native completion, no quality/generalization claim",
        "expected_pairs":PAIR_COUNT,"completed_pairs":len(rows),"expected_active_runs":2*PAIR_COUNT,
        "all_three_requested_pairs_and_six_active_runs_complete":pairs_complete and six_complete,
        "requested_alternating_order_executed_exactly":order_exact and actual_sequence==expected_sequence,
        "all_six_generations_reached_first_native_EOS":natural,
        "complete_benchmark_and_integrity_passed":complete,
        "all_full_outputs_exact":bool(equal),"same_output_speedup_eligible":bool(equal),
        "same_output_median_speedup":statistics.median(ratios)if equal else None,
        "same_output_minimum_speedup":min(ratios)if equal else None,
        "same_output_ge10_in_ALL_three_pairs":bool(equal and min(ratios)>=10),
        "same_output_median_ge10":bool(equal and statistics.median(ratios)>=10),
        "general_single_request_ten_x_goal_achieved":False,"general_quality_preservation_established":False,
        "exact_mathematical_equivalence_certified":False,
        "paired_ratio_diagnostics_NOT_accepted_if_partial_or_different_outputs":ratios,
        "partial_or_failed_evidence_is_not_accepted_speedup":not complete}
    if complete:
        result["complete_native_completion_latency_ratio_median_REPORT_ONLY_if_outputs_differ"]=statistics.median(ratios)
        result["active_latency_s_by_engine"]={engine:[row[engine]["external_active_wall_s"]for row in rows]for engine in("baseline","candidate")}
        result["generated_tokens_per_active_second_by_engine"]={engine:[row[engine]["native_EOS"]["actual_tokens"]/row[engine]["external_active_wall_s"]for row in rows]for engine in("baseline","candidate")}
    return result


def file_manifest(directory,sha):
    base=Path(directory)
    names=("model.safetensors.index.json","config.json","generation_config.json","tokenizer.json",
        "tokenizer_config.json","vocab.json","merges.txt","special_tokens_map.json","added_tokens.json")
    files=sorted(base.glob("*.safetensors"))+[base/name for name in names if(base/name).is_file()]
    require(any(p.suffix==".safetensors"for p in files),"Require original local safetensors; no downloading")
    return {p.name:sha(p)for p in files}


def graph_signature(runner):
    def bank(mapping):
        return [[repr(key),id(value),id(value.get("graph"))if isinstance(value,dict)else id(getattr(value,"graph",None))]
            for key,value in sorted(mapping.items(),key=lambda item:repr(item[0]))]
    return {"target_decode_graphs":bank(runner.verifier.graphs),
        "auxiliary_query_graphs":bank(runner.proposer.conditioned.graphs),
        "target_prefill_graphs":bank(runner.target_prefill_component.graph.entries),
        "auxiliary_prefill_graphs":bank(runner.proposer.conditioned.prefill_component.graph.entries),
        "prefill_capture_count":getattr(runner,"prefill_bank_metadata",{}).get("captures"),
        "target_prefill_python_forward_calls":getattr(runner.target_prefill_component.graph,"python_forward_calls",None),
        "auxiliary_prefill_python_forward_calls":getattr(runner.proposer.conditioned.prefill_component.graph,"python_forward_calls",None),
        "compiled_auxiliary_prepared_family_widths":[[repr(key),sorted(value)]for key,value in sorted(
            getattr(runner.proposer.conditioned,"_full_query_widths_seen",{}).items(),key=lambda item:repr(item[0]))]}


def self_test():
    from copy import deepcopy
    from types import SimpleNamespace
    import ast
    original=ast.parse((Path(__file__).parent/"benchmark.py").read_text())
    source_prompts=next(ast.literal_eval(node.value)for node in original.body if isinstance(node,ast.Assign)and
        any(isinstance(target,ast.Name)and target.id=="PROMPTS"for target in node.targets))
    assert source_prompts[PROMPT_INDEX]==PUBLIC_PROMPT
    assert paired_orders()==[("baseline","candidate"),("candidate","baseline"),("baseline","candidate")]
    eos={151645};tokens=[1,2,151645]
    assert first_eos_check(tokens,"eos",eos)["stopped_at_first_native_EOS"]
    for value,reason in (([1,151645,2],"eos"),([1,2],"max_new_tokens"),([],"eos"),([True,151645],"eos")):
        assert not first_eos_check(value,reason,eos)["stopped_at_first_native_EOS"]
    calls=[];ids=object();model=object()
    class Candidate:
        def run(self,actual,budget,**kwargs):
            calls.append(("candidate",actual,budget,kwargs));return tokens,{"stop_reason":"eos"}
    def baseline(actual_model,actual,budget,actual_eos):
        calls.append(("baseline",actual,budget,actual_eos));assert actual_model is model
        return tokens,"eos",{}
    invoke_generation("baseline",model,Candidate(),ids,eos,baseline)
    invoke_generation("candidate",model,Candidate(),ids,eos,baseline)
    assert all(row[1]is ids and row[2]==MAX_NEW_TOKENS for row in calls)
    assert calls[-1][3]=={"eos_ids":eos}and not any("reference"in str(k)for k in calls[-1][3])
    generation={"completed":True,"token_ids":tokens,"raw_text":"complete<EOS>","text":"complete",
        "native_EOS":first_eos_check(tokens,"eos",eos),"external_active_wall_s":1.}
    rows=[];active=[]
    for i in range(PAIR_COUNT):
        left=deepcopy(generation);left["external_active_wall_s"]=11.
        right=deepcopy(generation)
        rows.append({"prompt_index":PROMPT_INDEX,"repeat":i,"order":list(paired_orders()[i]),"baseline":left,"candidate":right,
            "comparison":output_comparison(left,right)})
        active.extend({"repeat":i,"engine":engine,"completed":True}for engine in paired_orders()[i])
    good=summarize(rows,active,completed=True,integrity_passed=True)
    assert good["same_output_median_speedup"]==11. and good["same_output_ge10_in_ALL_three_pairs"]
    assert not good["general_single_request_ten_x_goal_achieved"]and not good["general_quality_preservation_established"]
    for incomplete in (rows[:-1],rows+[deepcopy(rows[0])]):
        out=summarize(incomplete,active,completed=True,integrity_passed=True)
        assert not out["complete_benchmark_and_integrity_passed"]and out["same_output_median_speedup"]is None
    wrong_order=deepcopy(rows);wrong_order[1]["order"]=["baseline","candidate"]
    assert not summarize(wrong_order,active,completed=True,integrity_passed=True)["complete_benchmark_and_integrity_passed"]
    assert not summarize(rows,list(reversed(active)),completed=True,integrity_passed=True)["complete_benchmark_and_integrity_passed"]
    bad=deepcopy(rows);bad[1]["candidate"]["native_EOS"]["stopped_at_first_native_EOS"]=False
    assert not summarize(bad,active,completed=True,integrity_passed=True)["complete_benchmark_and_integrity_passed"]
    bad=deepcopy(rows);bad[0]["candidate"]["token_ids"]=[9,151645]
    bad[0]["comparison"]=output_comparison(bad[0]["baseline"],bad[0]["candidate"])
    different=summarize(bad,active,completed=True,integrity_passed=True)
    assert different["complete_benchmark_and_integrity_passed"]and not different["same_output_speedup_eligible"]
    for kwargs in({"completed":False,"integrity_passed":True},{"completed":True,"integrity_passed":False}):
        assert summarize(rows,active,**kwargs)["same_output_median_speedup"]is None
    signature_owner=SimpleNamespace(verifier=SimpleNamespace(graphs={(64,256):{"graph":object()}}),
        proposer=SimpleNamespace(conditioned=SimpleNamespace(graphs={(15,15,256):{"graph":object()}},
            prefill_component=SimpleNamespace(graph=SimpleNamespace(entries={25:SimpleNamespace(graph=object())})))),
        target_prefill_component=SimpleNamespace(graph=SimpleNamespace(entries={24:SimpleNamespace(graph=object())})))
    a=graph_signature(signature_owner);assert a==graph_signature(signature_owner)
    signature_owner.prefill_bank_metadata={"captures":2}
    assert a!=graph_signature(signature_owner)
    b=graph_signature(signature_owner)
    signature_owner.target_prefill_component.graph.python_forward_calls=4
    assert b!=graph_signature(signature_owner)
    b=graph_signature(signature_owner)
    signature_owner.proposer.conditioned._full_query_widths_seen={(15,256):{15,16}}
    assert b!=graph_signature(signature_owner)
    b=graph_signature(signature_owner)
    signature_owner.verifier.graphs[(64,512)]={"graph":object()}
    assert b!=graph_signature(signature_owner)
    return {"CPU_three_alternating_pairs_fixed_public_budget_and_first_native_EOS_gates_passed":True,
        "CPU_no_reference_output_or_stopping_length_enters_generation_route":True,
        "CPU_partial_failed_missing_EOS_different_full_output_and_wrong_integrity_cannot_claim_same_output_speedup":True,
        "CPU_graph_family_identity_signature_detects_new_runtime_capture":True,"GPU_work_performed":False}


def main():
    if sys.argv[1:]==["--self-test"]:
        print(json.dumps(self_test(),indent=2));return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",default="model-3b")
    parser.add_argument("--draft-path",default="/tmp/lightreason-models/pard2-qwen3-8b")
    parser.add_argument("--adaptation-checkpoint-path",required=True)
    parser.add_argument("--adaptation-checkpoint-sha256",default=CHECKPOINT450_SHA)
    parser.add_argument("--effective-auxiliary-sha256",default=EFFECTIVE450_SHA)
    parser.add_argument("--draft-tokens",type=int,choices=(15,31),default=15)
    parser.add_argument("--node-budget",type=int,choices=(64,128),default=64)
    parser.add_argument("--top-k",type=int,choices=(8,),default=8)
    parser.add_argument("--max-new-tokens",type=int,choices=(MAX_NEW_TOKENS,),default=MAX_NEW_TOKENS)
    parser.add_argument("--max-len",type=int,choices=(MAX_CACHE,),default=MAX_CACHE)
    parser.add_argument("--repeats",type=int,choices=(PAIR_COUNT,),default=PAIR_COUNT)
    parser.add_argument("--cooldown-seconds",type=float,choices=(COOLDOWN_S,),default=COOLDOWN_S)
    parser.add_argument("--max-seconds",type=float,default=600.)
    parser.add_argument("--out",default="results3b/native-eos-fibonacci-graph-prefill-compiled450-v2.json")
    args=parser.parse_args()
    if not 0<args.max_seconds<=600:parser.error("Use a process segment deadline<=600s")
    if args.adaptation_checkpoint_sha256!=CHECKPOINT450_SHA or args.effective_auxiliary_sha256!=EFFECTIVE450_SHA:
        parser.error("This version pins the immutable450 checkpoint and effective auxiliary; use a NEW version for a different adaptation")
    destination=Path(args.out)
    if destination.exists():parser.error("Preserve earlier evidence: choose a fresh output path")
    if destination.resolve()==Path(args.adaptation_checkpoint_path).resolve():parser.error("Output may not replace immutable checkpoint")
    import torch
    from benchmark import PROMPTS
    from benchmark_3b import load
    from run_numeric_quality import raw_hf_natural
    from exact3b_validate import generation_eos
    from exact10_validation import hash_model_parameters,project_python_hashes
    from pard2_proposer3b_v1 import model_state,tokenizer_snapshot
    from benchmark_folded_pard2_lora_tree3b_v1 import full_audit,identity_gates
    from benchmark_conditioned_pard2_tree3b_v1 import file_sha,atomic_json
    from graph_prefill_compiled_pard2_lora_tree3b_v1 import GraphPrefillCompiledPARD2LoRATree3BRunner
    require(PROMPTS[PROMPT_INDEX]==PUBLIC_PROMPT,"Original PUBLIC Fibonacci prompt bytes changed")
    require(file_sha(args.adaptation_checkpoint_path)==CHECKPOINT450_SHA,"Immutable450 checkpoint file differs")
    began=time.perf_counter();sources_before=project_python_hashes();disk_before=file_manifest(args.model_path,file_sha)
    report={"schema":SCHEMA,"scope":"SINGLE PUBLIC PROMPT index1 ONLY; native full completion, no general speed or quality claim",
        "timeout_only_variant_of":"benchmark_native_eos_fibonacci3b_v1.py",
        "max_process_segment_seconds":args.max_seconds,"original_V1_source_preserved":True,
        "performance_protocol_and_all_validation_gates_unchanged_from_V1":True,
        "prompt_index":PROMPT_INDEX,"prompt":PUBLIC_PROMPT,"public_max_new_tokens":MAX_NEW_TOKENS,
        "public_max_cache_tokens":MAX_CACHE,"expected_pairs":PAIR_COUNT,"paired_orders":[list(p)for p in paired_orders()],
        "cooldown_between_active_requests_excluded_s":COOLDOWN_S,
        "timer_policy":"external CUDA-synchronized wall call; includes every online prefill, native EOS check, proposal/query/tree/cache/commit/host token delivery",
        "baseline_generator":"run_numeric_quality.raw_hf_natural (frozen exact3b_validate helper), independent first-native-EOS raw greedy",
        "baseline_helper_internal_quality_context_metrics_are_not_used_as_timers":True,
        "candidate_internal_timing_reported_separately_not_mixed_with_external_baseline_timer":True,
        "shared_prompt_tokenization_and_initial_H2D_excluded_for_both_as_in_frozen_Reference":True,
        "all_online_output_ID_host_delivery_and_internal_prefill_KV_transfers_included":True,
        "UTF8_decoding_and_posthoc_output_comparison_excluded_from_both_active_timers":True,
        "all_public_shape_setup_before_actual_input_admission":True,"real_output_warmup_or_answer_cache_used":False,
        "internal_prepare_check_wall_may_be_positive_without_new_capture_and_is_included_in_external_call_timer":True,
        "gold_answer_or_training_corpus_opened":False,"reference_tokens_or_length_passed_to_candidate":False,
        "immutable_adaptation_checkpoint_sha256":CHECKPOINT450_SHA,"immutable_effective_auxiliary_sha256":EFFECTIVE450_SHA,
        "target_compile_static":True,"target_forward_or_weights_replaced":False,
        "source_before":sources_before,"original_target_files_before":disk_before,
        "active_runs":[],"trials":[],"untimed_cooldowns":[],"completed_all_requested_pairs":False,
        "complete_benchmark_and_integrity_passed":False}
    runner=model=tokenizer=None;target_before=target_state_before=target_versions=None
    initialized_state=None;before=after=None;failure=None;loop_completed=False
    def deadline():require(time.perf_counter()-began<args.max_seconds,"Bounded native-EOS benchmark segment exceeded")
    def checkpoint():atomic_json(destination,report)
    try:
        model,tokenizer,environment=load(args);report["environment"]=environment
        require("4090"in environment["gpu"]and environment["torch"].split("+")[0]=="2.9.1"and environment["transformers"]=="4.57.1",
            "Require same4090 and pinned Torch2.9.1/Transformers4.57.1")
        target_before=hash_model_parameters(model);require(target_before==TARGET_SHA,"Loaded authoritative original BF16 target differs")
        target_versions=[(p,p._version,id(p),p.data_ptr())for p in model.parameters()]
        target_state_before=model_state(model);tokenizer_before=tokenizer_snapshot(tokenizer)
        eos=generation_eos(model,tokenizer);report["native_target_EOS_IDs"]=eos
        # Only public tokenization and length inform synthetic preparation.
        token_ids=tokenizer.apply_chat_template([{"role":"user","content":PUBLIC_PROMPT}],add_generation_prompt=True,tokenize=True)
        require(isinstance(token_ids,list)and token_ids and len(token_ids)+MAX_NEW_TOKENS+max(args.node_budget,args.draft_tokens)<=MAX_CACHE,
            "Public prompt/full budget/tree reserve exceed fixed cache")
        report["input_token_ids"]=token_ids;report["input_token_ids_sha256"]=token_sha(token_ids)
        report["tokenizer_snapshot_before"]=tokenizer_before
        deadline();constructed=time.perf_counter()
        runner=GraphPrefillCompiledPARD2LoRATree3BRunner(model,tokenizer,args.draft_path,
            adaptation_checkpoint_path=args.adaptation_checkpoint_path,max_len=MAX_CACHE,node_budget=args.node_budget,
            draft_tokens=args.draft_tokens,top_k=args.top_k,compile_static=True,compile_draft=True,cuda_graph=True)
        runner.set_graph_prefill_enabled(True)
        report["constructor_setup_excluded_s"]=time.perf_counter()-constructed
        initialized_state=model_state(model)
        from feature_target_setup3b import validate_lazy_forward_transition
        report["constructor_original_HF_lazy_forward_bookkeeping_transition"]=validate_lazy_forward_transition(target_state_before,initialized_state)
        report["candidate"]=runner.metadata
        require(runner.metadata.get("compile_static")is True,"Candidate target private compilation must match existing4.52x configuration")
        synthetic=torch.zeros((1,len(token_ids)),device=model.device,dtype=torch.long)
        prepared=time.perf_counter();public_setup=runner.prepare(synthetic,MAX_NEW_TOKENS);torch.cuda.synchronize()
        report["public_synthetic_prepare"]=public_setup;report["public_prepare_wall_excluded_s"]=time.perf_counter()-prepared
        report["prepared_graph_signature"]=graph_signature(runner);prepared_signature=graph_signature(runner)
        deadline();warmed=time.perf_counter()
        raw_hf_natural(model,synthetic,3,set());torch.cuda.synchronize()
        report["synthetic_original_HF_three_step_warmup_excluded_s"]=time.perf_counter()-warmed
        del synthetic
        before=full_audit(runner);report["integrity_before"]=before
        require(all(identity_gates(before,before,CHECKPOINT450_SHA).values()),"Initial full target/reference/effective/fold/source identity failed")
        require(before["effective_adapted_auxiliary_live_sha256"]==EFFECTIVE450_SHA and before["adaptation_admission"].get("updates")==450,
            "Effective auxiliary/update450 admission differs")
        require(initialized_state==model_state(model),"Synthetic setup changed initialized original target state")
        prefill_counts_before={"target":dict(runner.verifier._counts),"auxiliary":dict(runner.proposer.conditioned.prefill_dispatch_counts)}
        ids=torch.tensor([token_ids],device=model.device,dtype=torch.long)
        report["actual_prompt_GPU_admission_after_synthetic_setup_and_before_timer"]=True
        checkpoint()
        for repeat,order in enumerate(paired_orders()):
            row={"prompt_index":PROMPT_INDEX,"repeat":repeat,"order":list(order)};report["trials"].append(row)
            for engine in order:
                deadline();cool_start=time.perf_counter();time.sleep(COOLDOWN_S)
                report["untimed_cooldowns"].append({"repeat":repeat,"before_engine":engine,"actual_s":time.perf_counter()-cool_start})
                deadline();event={"repeat":repeat,"engine":engine,"completed":False}
                report["active_runs"].append(event);row[engine]=event
                torch.cuda.synchronize();started=time.perf_counter()
                try:
                    tokens,reason,metrics=invoke_generation(engine,model,runner,ids,eos,raw_hf_natural)
                    torch.cuda.synchronize();event["external_active_wall_s"]=time.perf_counter()-started
                except Exception as exc:
                    torch.cuda.synchronize();event["external_active_wall_s"]=time.perf_counter()-started
                    event["error"]=type(exc).__name__+": "+str(exc);raise
                event.update(completed=True,token_ids=list(tokens),token_ids_sha256=token_sha(tokens),
                    stop_reason=reason,native_EOS=first_eos_check(tokens,reason,eos),
                    raw_text=tokenizer.decode(tokens,skip_special_tokens=False),text=tokenizer.decode(tokens,skip_special_tokens=True),
                    internal_metrics=metrics)
                require(event["external_active_wall_s"]>0 and math.isfinite(event["external_active_wall_s"]),"Invalid external request latency")
                if engine=="candidate":
                    require(isinstance(metrics.get("excluded_setup_s"),(int,float))and math.isfinite(metrics["excluded_setup_s"])and
                        metrics["excluded_setup_s"]>=0. and metrics.get("all_output_tokens_target_verified")is True,
                        "Invalid candidate internal prepare-check time or missing final target verification")
                    require(graph_signature(runner)==prepared_signature,
                        "Online generation created/replaced a graph/capture, called Python prefill, or changed a compiled public family")
                require(token_sha(ids.reshape(-1).cpu().tolist())==report["input_token_ids_sha256"],"Prompt IDs changed during generation")
                checkpoint()
            row["comparison"]=output_comparison(row["baseline"],row["candidate"])
            row["external_complete_latency_ratio_baseline_over_candidate"]=row["baseline"]["external_active_wall_s"]/row["candidate"]["external_active_wall_s"]
            checkpoint()
        loop_completed=True;report["completed_all_requested_pairs"]=True
        report["prefill_dispatch_counts_before"]=prefill_counts_before
        report["prefill_dispatch_counts_after"]={"target":dict(runner.verifier._counts),"auxiliary":dict(runner.proposer.conditioned.prefill_dispatch_counts)}
        deltas={scope:{k:report["prefill_dispatch_counts_after"][scope].get(k,0)-prefill_counts_before[scope].get(k,0)
            for k in report["prefill_dispatch_counts_after"][scope]}for scope in("target","auxiliary")}
        report["timed_prefill_dispatch_count_deltas"]=deltas
        require(deltas["target"].get("graph")==PAIR_COUNT and deltas["target"].get("original")==0 and
            deltas["auxiliary"].get("graph",0)>=PAIR_COUNT and deltas["auxiliary"].get("original")==0 and
            deltas["auxiliary"].get("unprepared_original_fallback")==0,
            "Measured calls did not exclusively use prepared target/auxiliary graph prefills")
    except Exception as exc:
        failure=type(exc).__name__+": "+str(exc);report["error"]=failure
    finally:
        errors=[]
        if runner is not None:
            try:after=full_audit(runner);report["integrity_after"]=after
            except Exception as exc:errors.append("final full audit: "+type(exc).__name__+": "+str(exc))
        gates=identity_gates(before,after,CHECKPOINT450_SHA)
        gates["pinned_effective450_SHA_before_and_after"]=bool(before and after and before.get("effective_adapted_auxiliary_live_sha256")==after.get("effective_adapted_auxiliary_live_sha256")==EFFECTIVE450_SHA)
        if model is not None and target_before is not None:
            try:
                after_hash=hash_model_parameters(model);report["target_full_parameter_sha256_before"]=target_before;report["target_full_parameter_sha256_after"]=after_hash
                gates["original_target_full_live_SHA_objects_storage_versions_preserved"]=(after_hash==target_before==TARGET_SHA and all(
                    p._version==v and id(p)==identity and p.data_ptr()==pointer for p,v,identity,pointer in target_versions))
                gates["initialized_original_target_module_forwards_modes_and_parameters_preserved"]=initialized_state is not None and model_state(model)==initialized_state
                report["target_feature_lazy_bookkeeping_only_setup_transition"]=runner.metadata.get("original_HF_feature_setup")if runner is not None else None
                report["tokenizer_snapshot_after"]=tokenizer_snapshot(tokenizer)
                gates["original_target_tokenizer_unchanged"]=report["tokenizer_snapshot_after"]==report["tokenizer_snapshot_before"]
            except Exception as exc:errors.append("target/tokenizer final guard: "+type(exc).__name__+": "+str(exc))
        else:gates["original_target_full_live_SHA_objects_storage_versions_preserved"]=False
        try:
            report["source_after"]=project_python_hashes();report["original_target_files_after"]=file_manifest(args.model_path,file_sha)
            report["adaptation_checkpoint_sha256_after"]=file_sha(args.adaptation_checkpoint_path)
            gates["all_project_source_files_unchanged"]=report["source_after"]==sources_before
            gates["all_original_target_weight_config_tokenizer_files_unchanged"]=report["original_target_files_after"]==disk_before
            gates["immutable450_checkpoint_file_unchanged"]=report["adaptation_checkpoint_sha256_after"]==CHECKPOINT450_SHA
        except Exception as exc:errors.append("final source/files guard: "+type(exc).__name__+": "+str(exc))
        gates["no_runtime_or_final_audit_error"]=failure is None and not errors
        report["identity_and_integrity_gates"]=gates;report["integrity_errors"]=errors
        report["summary"]=summarize(report["trials"],report["active_runs"],completed=loop_completed,integrity_passed=all(gates.values()))
        report["complete_benchmark_and_integrity_passed"]=report["summary"]["complete_benchmark_and_integrity_passed"]
        report["whole_process_elapsed_s"]=time.perf_counter()-began;checkpoint()
    print(json.dumps({"out":str(destination),"complete_benchmark_and_integrity_passed":report["complete_benchmark_and_integrity_passed"],
        "same_output_speedup_eligible":report["summary"]["same_output_speedup_eligible"],
        "same_output_median_speedup":report["summary"]["same_output_median_speedup"],"error":failure}))
    if not report["complete_benchmark_and_integrity_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
