"""Bounded prefill graphs + frozen compiled auxiliary + ORIGINAL target tree.

Only fresh-request target/auxiliary prefill dispatch changes. The original
Feature4 owner, full-vocabulary verification, acceptance/EOS/controller and
compiled auxiliary query graphs remain unchanged. Captures use public lengths
and synthetic private values before request timers. No model forward is patched.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!="torch":raise
    torch=None

if torch is not None:
    from compiled_folded_pard2_lora_proposer3b_v1 import CompiledFoldedPARD2LoRALatticeProposer
    from compiled_folded_pard2_lora_tree3b_v1 import CompiledFoldedPARD2LoRATree3BRunner
else:
    CompiledFoldedPARD2LoRALatticeProposer=CompiledFoldedPARD2LoRATree3BRunner=object


def checked_capacity(value):
    if type(value)is not int or not 1<=value<=16:raise ValueError("Use a bounded public prefill-length bank1..16")
    return value


class _TargetPrefillProxy:
    """ALL state/methods belong to the original owner; only prefill dispatches.

    The verified-feature-bound proposer retains the TRUE owner, never this
    proxy. Every state update made by either prefill mode is visible there.
    """
    def __init__(self,owner,component):
        object.__setattr__(self,"_owner",owner);object.__setattr__(self,"_component",component)
        object.__setattr__(self,"_enabled",True)
        object.__setattr__(self,"_counts",{"original":0,"graph":0})
    def __getattr__(self,name):return getattr(self._owner,name)
    def __setattr__(self,name,value):
        if name in("_owner","_component","_enabled","_counts"):object.__setattr__(self,name,value)
        else:setattr(self._owner,name,value)
    def set_graph_prefill_enabled(self,enabled):
        if type(enabled)is not bool:raise ValueError("Use an explicit boolean prefill diagnostic toggle")
        if self._owner.pending is not None:raise RuntimeError("Finish pending verification before switching prefill mode")
        self._enabled=enabled
    def prefill(self,ids):
        if self._enabled:
            if ids.shape[1]not in self._component.graph.entries:raise RuntimeError("Prepare this PUBLIC target length before inference; no online capture")
            self._counts["graph"]+=1;return self._component.prefill(ids)
        self._counts["original"]+=1;return self._owner.prefill(ids)


class GraphPrefillCompiledPARD2LoRALatticeProposer(CompiledFoldedPARD2LoRALatticeProposer):
    def __init__(self,*args,prefill_warmups=2,**kwargs):
        super().__init__(*args,**kwargs)
        from eager_hf_prefill_graph_v1 import ConditionedAuxPrefillGraph
        self.prefill_component=ConditionedAuxPrefillGraph(self,warmups=prefill_warmups)
        self.graph_prefill_enabled=True
        self.prefill_dispatch_counts={"original":0,"graph":0,"unprepared_original_fallback":0}
        self.metadata={**self.metadata,"fresh_genuine_prefill_variant":"public_length_original_eager_HF_graph_v1",
            "prefill_graphs_capture_original_effective_HF_not_compiled_private_query":True,
            "unprepared_unexpected_full_prefill_uses_original_eager_never_online_capture":True,
            "prefill_dispatch_counts":self.prefill_dispatch_counts}

    def _prefill_conditioned_history(self,ids,features,p):
        if self.graph_prefill_enabled and ids.shape[1]in self.prefill_component.graph.entries:
            self.prefill_dispatch_counts["graph"]+=1
            return self.prefill_component.prefill(ids,features,p)
        if self.graph_prefill_enabled:self.prefill_dispatch_counts["unprepared_original_fallback"]+=1
        self.prefill_dispatch_counts["original"]+=1
        return super()._prefill_conditioned_history(ids,features,p)


class GraphPrefillCompiledPARD2LoRATree3BRunner(CompiledFoldedPARD2LoRATree3BRunner):
    """Same frozen run/controller; a SAME-OBJECT toggle isolates prefill effects."""
    def __init__(self,model,tokenizer,draft_path,*,adaptation_checkpoint_path,max_len=2048,
                 node_budget=64,draft_tokens=15,top_k=8,compile_static=False,compile_draft=True,
                 cuda_graph=True,share_pool=True,warmups=3,validate_topology=True,
                 commit_engine="triton",tree_policy="best-first",attention_buckets=(256,512,1024,2048),
                 prefill_cache_lengths=4,prefill_warmups=2):
        if torch is None:raise RuntimeError("Use the pinned CUDA runtime")
        from conditioned_pard2_tree3b_v1 import _VerifiedFeatureBoundProposer
        from feature4_3b_tree_verifier import Feature4BTreeVerifier
        from feature_target_setup3b import initialize_feature_output_bookkeeping
        from pard2_proposer3b_v1 import PARD2Proposer,file_sha,require
        from folded_pard2_lora_proposer3b_v1 import _assert_no_alias
        from exact10_validation import hash_model_parameters
        from eager_hf_prefill_graph_v1 import TargetFeature4PrefillGraph
        self.prefill_cache_lengths=checked_capacity(prefill_cache_lengths)
        require(type(prefill_warmups)is int and 1<=prefill_warmups<=3,"Use public prefill warmups1..3")
        require(compile_draft is True and cuda_graph is True and type(compile_static)is bool,
                "Use explicit target compiler policy, compiled auxiliary and CUDA graphs")
        require(type(node_budget)is int and 2<=node_budget<=128 and type(draft_tokens)is int and
            1<=draft_tokens<=32 and type(top_k)is int and 1<=top_k<=64 and tree_policy in("best-first","trunk"),"Use public tree settings")
        possible=1;level=1
        for _ in range(draft_tokens):
            level*=top_k;possible+=level
            if possible>=node_budget:break
        require(possible>=node_budget,"Public B/K cannot fill the fixed target tree")
        setup=initialize_feature_output_bookkeeping(model);target_sha=hash_model_parameters(model)
        self.model,self.tokenizer,self.max_len=model,tokenizer,max_len
        self.node_budget,self.draft_tokens,self.top_k,self.tree_policy=node_budget,draft_tokens,top_k,tree_policy
        buckets=tuple(sorted({int(v)for v in attention_buckets if 1<=int(v)<=max_len}|{max_len}))
        owner=Feature4BTreeVerifier(model,max_len=max_len,node_budget=node_budget,buckets=buckets,
            cuda_graph=True,compile_static=compile_static,share_pool=share_pool,warmups=warmups,
            validate_topology=validate_topology,commit_engine=commit_engine)
        reference=PARD2Proposer(draft_path,target_tokenizer=tokenizer,device=model.device,
            cache_mode="incremental",max_cache_len=max_len,attn_implementation="sdpa")
        _assert_no_alias(reference.model,model.parameters())
        auxiliary=GraphPrefillCompiledPARD2LoRALatticeProposer(reference,adaptation_checkpoint_path,
            expected_target_sha256=target_sha,max_len=max_len,attention_buckets=buckets,top_k=top_k,
            compile_model=True,share_pool=share_pool,warmups=warmups,prefill_warmups=prefill_warmups)
        _assert_no_alias(auxiliary.model,model.parameters())
        self.target_prefill_component=TargetFeature4PrefillGraph(owner,warmups=prefill_warmups)
        self.verifier=_TargetPrefillProxy(owner,self.target_prefill_component)
        self.proposer=_VerifiedFeatureBoundProposer(auxiliary,owner)
        with torch.inference_mode(False),torch.no_grad():self.visible=torch.empty((1,max_len),device=model.device,dtype=torch.long)
        self.setup_s,self.prepared=0.,False
        self._prefill_lengths=OrderedDict();self.prefill_setup_s=0.
        self.prefill_bank_metadata={"maximum_retained_target_aux_length_pairs":self.prefill_cache_lengths,
            "default_four_pairs_limits_long_context_VRAM":True,"retained_public_target_lengths":[],
            "captures":0,"evicted_pairs":0,"setup_s":0.,"requests":[]}
        self.metadata={"variant":"original_target_compiled_folded_LoRA_PARD2_with_eager_HF_prefill_graph_v1",
            "original_target_live_sha256":target_sha,"target_weights_and_equations_unchanged":True,
            "target_verifier":owner.metadata,"proposer":auxiliary.metadata,
            "adaptation_checkpoint_path":str(Path(adaptation_checkpoint_path).resolve()),
            "effective_auxiliary_is_intentionally_adapted":True,"original_pinned_reference_kept_unchanged":True,
            "effective_auxiliary_equal_original_checkpoint_claimed":False,"node_budget":node_budget,
            "draft_tokens":draft_tokens,"top_k":top_k,"tree_policy":tree_policy,
            "compile_static":compile_static,"compile_draft":True,"max_len":max_len,"attention_buckets":list(buckets),
            "original_HF_feature_setup":setup,"target_controller_inherited_unchanged":True,
            "verified_feature_proposer_bound_to_true_owner":self.proposer.verifier is owner,
            "all_online_feature_prefill_projection_cache_tree_and_verification_costs_timed":True,
            "prefill_capture_setup_uses_public_lengths_and_synthetic_private_values_only":True,
            "prefill_capture_on_actual_request_values_or_inside_request_timer":False,
            "graph_prefill_enabled":True,
            "target_or_auxiliary_model_forward_replaced":False,"prefill_graph_bank":self.prefill_bank_metadata,
            "strict_token_matching_required":False,"exact_token_agreement_must_be_reported":True,
            "quality_and_end_to_end_speed_require_separate_measurement":True,"runner_source_sha256":file_sha(__file__)}

    def set_graph_prefill_enabled(self,enabled):
        self.verifier.set_graph_prefill_enabled(enabled)
        self.proposer.conditioned.graph_prefill_enabled=enabled
        self.metadata["graph_prefill_enabled"]=enabled

    def _prepare_public_prefills(self,length):
        """Receives LENGTH ONLY; never tokens, features, answers or EOS distance."""
        from eager_hf_prefill_graph_v1 import valid_length
        valid_length(length,self.max_len-1)
        target=self.target_prefill_component;aux=self.proposer.conditioned.prefill_component
        if self.verifier.pending is not None:raise RuntimeError("No capture/eviction during a pending verification")
        if length in self._prefill_lengths:
            self._prefill_lengths.move_to_end(length)
            self.prefill_bank_metadata["retained_public_target_lengths"]=list(self._prefill_lengths)
            return {"setup_s":0.,"new_graphs":0,"public_target_length":length,"bank_hit":True}
        torch.cuda.synchronize();began=time.perf_counter();evicted=None
        if len(self._prefill_lengths)>=self.prefill_cache_lengths:
            evicted,_=self._prefill_lengths.popitem(last=False)
            target.graph.entries.pop(evicted);aux.graph.entries.pop(evicted+1)
            for graph in(target.graph,aux.graph):graph.metadata["prepared_lengths"]=sorted(graph.entries)
            self.prefill_bank_metadata["evicted_pairs"]+=1
        target_info=aux_info=None
        try:
            target_info=target.prepare(length);aux_info=aux.prepare(length+1)
            self._prefill_lengths[length]=None
        except BaseException:
            target.graph.entries.pop(length,None);aux.graph.entries.pop(length+1,None)
            for graph in(target.graph,aux.graph):graph.metadata["prepared_lengths"]=sorted(graph.entries)
            raise
        finally:
            # Captures wrote SYNTHETIC physical rows. Every next request starts
            # from genuine prefill; no synthetic prefix is declared committed.
            self.verifier.reset_for_new_request();self.proposer.reset()
            self.prefill_bank_metadata["retained_public_target_lengths"]=list(self._prefill_lengths)
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        self.prefill_setup_s+=elapsed;self.setup_s+=elapsed
        self.prefill_bank_metadata.update(retained_public_target_lengths=list(self._prefill_lengths),
            setup_s=self.prefill_setup_s,captures=self.prefill_bank_metadata["captures"]+2)
        result={"setup_s":elapsed,"new_graphs":2,"public_target_length":length,"bank_hit":False,
            "evicted_public_target_length":evicted,"target":target_info,"auxiliary":aux_info}
        self.prefill_bank_metadata["requests"].append(result)
        return result

    def prepare(self,ids,n):
        # Parent preparation selects ONLY public decode families and resets
        # logical state before synthetic captures. Its controller is unchanged.
        prepared=super().prepare(ids,n)
        prefill=self._prepare_public_prefills(ids.shape[1])
        return {**prepared,"setup_s":prepared["setup_s"]+prefill["setup_s"],
            "new_graphs":prepared.get("new_graphs",0)+prefill["new_graphs"],"prefill_graphs":prefill}


def self_test():
    from types import SimpleNamespace
    for value in(1,4,16):assert checked_capacity(value)==value
    for value in(0,17,True):
        try:checked_capacity(value)
        except ValueError:pass
        else:raise AssertionError("Unbounded/invalid graph-bank capacity accepted")
    calls=[];owner=SimpleNamespace(pending=None,model=object(),current_position=0)
    owner.prefill=lambda ids:calls.append(("original",ids))or"original-anchor"
    component=SimpleNamespace(graph=SimpleNamespace(entries={3:object()}),prefill=lambda ids:calls.append(("graph",ids))or"graph-anchor")
    proxy=_TargetPrefillProxy(owner,component);ids=SimpleNamespace(shape=(1,3))
    assert proxy.prefill(ids)=="graph-anchor"and proxy.model is owner.model
    proxy.current_position=5;assert owner.current_position==5
    proxy.set_graph_prefill_enabled(False);assert proxy.prefill(ids)=="original-anchor"
    proxy.set_graph_prefill_enabled(True)
    before=list(calls)
    try:proxy.prefill(SimpleNamespace(shape=(1,4)))
    except RuntimeError:pass
    else:raise AssertionError("Missing public length admitted online capture")
    assert calls==before
    owner.pending={}
    try:proxy.set_graph_prefill_enabled(False)
    except RuntimeError:pass
    else:raise AssertionError("Prefill changed mode during pending target verification")
    assert proxy._enabled is True
    # Exercise the ACTUAL length-only LRU preparation method without Torch.
    # No fake model/GPU execution is called or represented as correctness.
    global torch
    original_torch=torch;events=[]
    class Component:
        fail_length=None
        def __init__(self,label):self.label=label;self.graph=SimpleNamespace(entries={},metadata={"prepared_lengths":[]})
        def prepare(self,length):
            events.append((self.label,"capture_public_length",length))
            if length==self.fail_length:raise ValueError("expected auxiliary capture failure")
            self.graph.entries[length]=object();self.graph.metadata["prepared_lengths"]=sorted(self.graph.entries)
            return {"setup_s":.001,"new_graph":True,"length":length}
    try:
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append("synchronize")))
        instance=object.__new__(GraphPrefillCompiledPARD2LoRATree3BRunner)
        target,aux=Component("target"),Component("aux")
        instance.max_len=64;instance.prefill_cache_lengths=2;instance._prefill_lengths=OrderedDict()
        instance.target_prefill_component=target
        instance.verifier=SimpleNamespace(pending=None,reset_for_new_request=lambda:events.append("target_reset"))
        instance.proposer=SimpleNamespace(conditioned=SimpleNamespace(prefill_component=aux),reset=lambda:events.append("aux_reset"))
        instance.prefill_setup_s=instance.setup_s=0.
        instance.prefill_bank_metadata={"evicted_pairs":0,"captures":0,"requests":[]}
        for length in(3,4):assert instance._prepare_public_prefills(length)["new_graphs"]==2
        assert set(target.graph.entries)=={3,4}and set(aux.graph.entries)=={4,5}
        count=len(events);assert instance._prepare_public_prefills(3)["bank_hit"]and len(events)==count
        assert list(instance._prefill_lengths)==[4,3]
        result=instance._prepare_public_prefills(5)
        assert result["evicted_public_target_length"]==4 and set(target.graph.entries)=={3,5}and set(aux.graph.entries)=={4,6}
        assert events[-2:]==["aux_reset","synchronize"]and instance.prefill_bank_metadata["evicted_pairs"]==1
        aux.fail_length=8
        try:instance._prepare_public_prefills(7)
        except ValueError as exc:assert str(exc)=="expected auxiliary capture failure"
        else:raise AssertionError("Failed partial capture was accepted")
        assert 7 not in target.graph.entries and 8 not in aux.graph.entries and 7 not in instance._prefill_lengths
        assert events[-2:]==["target_reset","aux_reset"]
        before=list(events);instance.verifier.pending={}
        try:instance._prepare_public_prefills(9)
        except RuntimeError:pass
        else:raise AssertionError("Pending verification admitted capture/eviction")
        assert events==before
    finally:torch=original_torch
    # Execute the new auxiliary dispatch SOURCE against a stub parent. This
    # tests actual override branches without importing any CUDA dependency.
    import ast,inspect,textwrap
    auxiliary_class=ast.parse(inspect.getsource(GraphPrefillCompiledPARD2LoRALatticeProposer)).body[0]
    auxiliary_class.bases=[ast.Name(id="Parent",ctx=ast.Load())]
    parent_calls=[]
    class Parent:
        def _prefill_conditioned_history(self,ids,features,p):parent_calls.append((ids,features,p));return"original"
    namespace={"Parent":Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[auxiliary_class],type_ignores=[])),__file__,"exec"),namespace)
    Auxiliary=namespace["GraphPrefillCompiledPARD2LoRALatticeProposer"]
    candidate=object.__new__(Auxiliary);candidate.graph_prefill_enabled=True
    dispatch=[];candidate.prefill_component=SimpleNamespace(graph=SimpleNamespace(entries={4:object()}),prefill=lambda *args:dispatch.append(args)or"graph")
    candidate.prefill_dispatch_counts={"original":0,"graph":0,"unprepared_original_fallback":0}
    data,features=SimpleNamespace(shape=(1,4)),object()
    assert candidate._prefill_conditioned_history(data,features,3)=="graph"
    candidate.graph_prefill_enabled=False;assert candidate._prefill_conditioned_history(data,features,3)=="original"
    candidate.graph_prefill_enabled=True;missing=SimpleNamespace(shape=(1,5))
    assert candidate._prefill_conditioned_history(missing,features,4)=="original"
    assert dispatch==[(data,features,3)]and parent_calls==[(data,features,3),(missing,features,4)]
    assert candidate.prefill_dispatch_counts=={"original":2,"graph":1,"unprepared_original_fallback":1}
    assert set(candidate.prefill_component.graph.entries)=={4}
    owner_calls=[node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(GraphPrefillCompiledPARD2LoRATree3BRunner.__init__))))
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=="Feature4BTreeVerifier"]
    assert len(owner_calls)==1 and any(kw.arg=="compile_static"and isinstance(kw.value,ast.Name)and kw.value.id=="compile_static"for kw in owner_calls[0].keywords)
    return {"CPU_target_proxy_original_owner_state_and_same_object_toggle_passed":True,
        "missing_public_shape_rejected_without_online_capture":True,
        "bounded_graph_bank_capacity_contract_passed":True,
        "actual_length_only_LRU_hit_eviction_partial_failure_cleanup_and_logical_reset_passed":True,
        "actual_auxiliary_override_original_graph_and_unprepared_fallback_routes_passed":True,
        "existing_target_compile_static_policy_forwarded_unchanged":True,
        "GPU_work_performed":False}


if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");args=parser.parse_args()
    if not args.self_test:parser.error("Runtime API; standalone mode is --self-test")
    print(json.dumps(self_test(),indent=2))
