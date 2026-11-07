"""Separate full-query compilation of an AUXILIARY conditioned PARD2 model.

The frozen original implementation compiles only the private core forward.
This variant compiles positions/mask, embedding+projector, the SAME core HF
forward, argmax/top-K/logsumexp and persistent output copies together. Each
(B, immutable attention bucket) has its own callable; the inherited prepare
warms/captures EVERY input width B..2B before any real prefix is admitted.

No target model, global compiler setting, core/projector forward function or
weight is replaced. Original conditioned history checks, row alignment,
conditioning prefill, overwritten mask KV and final-target authority remain
inherited. Compiler FP-order differences affect unverified proposals only.
Actual graph correctness, acceptance, quality and speed are unestablished.
"""
from __future__ import annotations

from pathlib import Path
from types import FunctionType

from conditioned_pard2_proposer3b_v1 import (ConditionedPARD2LatticeProposer,
    FEATURE_WIDTH,conditioning_rows,cpu_contract_tests as inherited_cpu_tests)
from pard2_proposer3b_v1 import file_sha,require

try:
    import torch
except ModuleNotFoundError as error:
    if error.name!="torch":raise
    torch=None


def full_query_key(block,bucket,attention_buckets):
    require(type(block)is int and 1<=block<=32 and type(bucket)is int and
        bucket in attention_buckets and bucket>=2*block,"Use a public B/bucket family supporting all B..2B widths")
    return block,bucket


def private_code_copy(function,label):
    """Independent code cache for this NEW private closure, never core.forward.

    Torch2.9 documents compile caches/recompile limits per code object:
    https://docs.pytorch.org/docs/2.9/generated/torch.compile.html
    Separate private code objects avoid exhausting that shared cache across
    projector instances/buckets without changing any global compiler limit.
    Only name metadata changes; bytecode, constants and closure cells match.
    """
    code=function.__code__.replace(co_name="full_query_"+label,co_qualname="full_query_"+label)
    copied=FunctionType(code,function.__globals__,code.co_name,function.__defaults__,function.__closure__)
    copied.__kwdefaults__=function.__kwdefaults__
    return copied


class CompiledConditionedPARD2LatticeProposer(ConditionedPARD2LatticeProposer):
    def __init__(self,*args,compile_model=True,**kwargs):
        require(torch is not None,"Use pinned CUDA PyTorch for the compiled auxiliary")
        require(compile_model is True,"Select this separately labeled variant with compile_model=True")
        self._eager_full_queries,self._compiled_full_queries={},{}
        self._full_query_widths_seen={}
        # Disable the old core-only wrapper: ONE private full-query compiler
        # owns this path, with no nested CUDA graphs or core.forward mutation.
        super().__init__(*args,compile_model=False,**kwargs)
        self.compile_model=True
        self.metadata={**self.metadata,
            "auxiliary_implementation":"PARD2_conditioned_projector_full_query_compile_v1",
            "compile_model":True,"core_only_compile_model":False,
            "compiled_auxiliary_scope":"positions+mask+embedding+projector+original_core_HF+argmax+topk+logsumexp+output_copies",
            "compiled_callable_key":"(B, immutable attention bucket); dynamic query width, fixed outer graph per B..2B entry",
            "compile_dynamic_query_width":True,"private_per_family_code_cache":True,
            "global_compiler_configuration_changed":False,"core_or_target_forward_replaced":False,
            "inherited_conditioning_cache_controller_unchanged":True,
            "all_public_shapes_compiled_before_real_prefix":True,
            "compile_variant_fp_order_may_change_unverified_proposals":True,
            "full_query_compile_source_sha256":file_sha(__file__),
            "new_performance_acceptance_or_quality_established":False}

    def _make_full_query(self,block,bucket):
        # Capture only immutable object/tensor references, never logical CPU
        # prefix length, actual features, future target rows or mutable shapes.
        core,adapter=self.model,self.adapter
        base=self.gpu_base;keys=self.key_positions[:bucket]
        view=self.cache_views[bucket];K=self.top_k
        output_ids=self.outputs[block]
        lattice_ids,lattice_logps=self.lattice_ids[block],self.lattice_logps[block]
        def query(input_ids,feature_inputs,offsets):
            positions=base+offsets
            allowed=keys.view(1,1,1,-1)<=positions.view(1,1,-1,1)
            embeddings=adapter.conditioned_embeddings(input_ids,feature_inputs)
            output=core(input_ids=None,inputs_embeds=embeddings,past_key_values=view,
                cache_position=positions,position_ids=positions.view(1,-1),attention_mask=allowed,
                use_cache=True,logits_to_keep=block,output_attentions=False,
                output_hidden_states=False,return_dict=True)
            raw=output.logits[:,-block:]
            output_ids.copy_(raw.argmax(-1))
            logits=raw.float()
            values,ids=torch.topk(logits,K,dim=-1,largest=True,sorted=True)
            lattice_ids.copy_(ids)
            lattice_logps.copy_(values-torch.logsumexp(logits,dim=-1,keepdim=True))
            # Raw output is exposed for own-callable parity and eager drift
            # diagnostics, never used as a target acceptance certificate.
            return raw,output_ids,lattice_ids,lattice_logps
        return query

    def _query_for_preparation(self,block,bucket):
        key=full_query_key(block,bucket,self.attention_buckets)
        if key not in self._compiled_full_queries:
            require(not self.cached_visible_ids,"New auxiliary compiled families require an empty logical request")
            eager=private_code_copy(self._make_full_query(*key),f"instance{id(self):x}_B{block}_K{bucket}")
            compiled=torch.compile(eager,fullgraph=True,dynamic=True,
                options={"triton.cudagraphs":False})
            self._eager_full_queries[key],self._compiled_full_queries[key]=eager,compiled
            self._full_query_widths_seen[key]=set()
        return self._compiled_full_queries[key]

    def _call(self,entry,block):
        if "feature_inputs"not in entry:
            require(not self.cached_visible_ids,"Allocate auxiliary feature buffers only in public preparation")
            entry["feature_inputs"]=torch.zeros((1,entry["input_ids"].shape[1],FEATURE_WIDTH),
                device=self.device,dtype=torch.bfloat16)
        bucket=entry["attention_bucket"]
        query=self._query_for_preparation(block,bucket)
        query(entry["input_ids"],entry["feature_inputs"],entry["offsets"])
        self._full_query_widths_seen[(block,bucket)].add(entry["input_ids"].shape[1])

    def full_query_for(self,block,bucket,*,compiled=True):
        """Exact graph callable, or SAME source eager diagnostic; both write KV.

        The callable takes (input_ids, aligned_verified_features, offsets) and
        returns (raw_logits, greedy_ID_bank, lattice_ID_bank, logprob_bank).
        Returned banks are reused; diagnostics must clone before another call.
        This accessor never creates/compiles a missing public family.
        """
        key=full_query_key(block,bucket,self.attention_buckets)
        functions=self._compiled_full_queries if compiled else self._eager_full_queries
        require(key in functions,"Prepare the requested full-query B/bucket family before use")
        return functions[key]

    def prepare(self,num_draft_tokens,buckets=None):
        result=super().prepare(num_draft_tokens,buckets=buckets)
        requested=self.attention_buckets if buckets is None else tuple(dict.fromkeys(buckets))
        required=set(range(num_draft_tokens,2*num_draft_tokens+1))
        require(all(self._full_query_widths_seen.get((num_draft_tokens,bucket),set())==required
            for bucket in requested),"Public preparation did not compile/warm every B..2B input width")
        return {**result,"full_query_compile":True,"all_requested_static_widths_seen":True,
            "compiled_family_keys":[list(key)for key in sorted(self._compiled_full_queries)]}


def cpu_contract_tests():
    inherited=inherited_cpu_tests()
    for block in (7,15,31):
        for bucket in (256,512,1024,2048):
            assert full_query_key(block,bucket,(256,512,1024,2048))==(block,bucket)
            for true_rows in range(1,block+2):
                width=true_rows+block-1
                assert block<=width<=2*block
                p=240;common=p+1-true_rows
                plan=conditioning_rows(common,p+1,block,p)
                assert len(plan["feature_rows"])==width and max(i for i in plan["feature_rows"] if i is not None)<p
    for block,bucket in ((True,256),(0,256),(33,256),(31,32),(15,300)):
        try:full_query_key(block,bucket,(256,512,1024,2048))
        except ValueError:pass
        else:raise AssertionError("Invalid full-query public family accepted")
    original_code=full_query_key.__code__
    copy=private_code_copy(full_query_key,"CPU_test_only")
    assert copy.__code__ is not original_code and full_query_key.__code__ is original_code
    assert copy.__code__.co_code==original_code.co_code and copy.__code__.co_consts==original_code.co_consts
    assert copy(15,256,(256,512))==full_query_key(15,256,(256,512))
    # Actual accessor/preparation routing with CPU sentinels, not a mirrored
    # GPU controller: unprepared families reject before compiler invocation.
    obj=object.__new__(CompiledConditionedPARD2LatticeProposer)
    obj.attention_buckets=(256,512,1024,2048);obj.cached_visible_ids=(1,2,3)
    ready=object();obj._compiled_full_queries={(15,256):ready};obj._eager_full_queries={(15,256):ready}
    assert obj.full_query_for(15,256) is ready and obj._query_for_preparation(15,256) is ready
    for call in (lambda:obj.full_query_for(31,256),lambda:obj._query_for_preparation(31,256),
        lambda:obj._call({},31)):
        try:call()
        except ValueError:pass
        else:raise AssertionError("An active request admitted an unprepared compiler/feature-allocation path")
    assert set(obj._compiled_full_queries)=={(15,256)} and obj.cached_visible_ids==(1,2,3)
    return {**inherited,"CPU_full_query_public_family_dispatch_and_no_future_feature_tests_passed":True,
        "CPU_actual_unprepared_accessor_compiler_and_buffer_allocation_prewrite_rejections_passed":True,
        "GPU_compile_graph_projection_or_cache_validation_performed":False}


if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if not args.self_test:p.error("Use --self-test or the separate root-run probe/benchmark")
    print(json.dumps(cpu_contract_tests(),indent=2))
