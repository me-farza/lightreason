"""Full PRIVATE-query compilation of the intentionally ADAPTED auxiliary.

The original pinned reference retains its raw-checkpoint byte-equality audit.
The effective folded BF16 auxiliary has its OWN hash, versions and explicit
original+BA fold lineage. The frozen original3B target is never compiled or
modified here. No auxiliary/source identity is relabeled as the original.

Only the new private call compiles: aligned features -> projector/embedding ->
effective ordinary HF core -> full-vocab logits -> argmax/topK/logsumexp/copies.
All conditioning/cache visibility and effective/reference integrity contracts
are inherited from frozen folded_pard2_lora_proposer3b_v1.py. Public families
compile/capture B..2B before real prefixes. No global compiler change or core
forward reassignment occurs. Correctness/acceptance/speed still need GPU proof.
"""
from __future__ import annotations

from folded_pard2_lora_proposer3b_v1 import (FoldedPARD2LoRALatticeProposer,
    cpu_contract_tests as folded_cpu_tests)
from compiled_conditioned_pard2_proposer3b_v1 import (CompiledConditionedPARD2LatticeProposer,
    cpu_contract_tests as compiled_cpu_tests)
from pard2_proposer3b_v1 import file_sha,require


class CompiledFoldedPARD2LoRALatticeProposer(FoldedPARD2LoRALatticeProposer):
    # These frozen helper methods contain no super()/admission/loading logic.
    # They capture THIS instance's effective model, adapter, view and banks.
    _make_full_query=CompiledConditionedPARD2LatticeProposer._make_full_query
    _query_for_preparation=CompiledConditionedPARD2LatticeProposer._query_for_preparation
    _call=CompiledConditionedPARD2LatticeProposer._call
    full_query_for=CompiledConditionedPARD2LatticeProposer.full_query_for

    def __init__(self,*args,compile_model=True,**kwargs):
        require(compile_model is True,"This separately labeled folded variant requires full private-query compilation")
        self._eager_full_queries,self._compiled_full_queries={},{}
        self._full_query_widths_seen={}
        # Preserve the entire audited fold/admission/identity constructor; only
        # disable its old partial core-only wrapper to avoid nested compilation.
        super().__init__(*args,compile_model=False,**kwargs)
        self.compile_model=True
        self.metadata={**self.metadata,
            "variant":"intentionally_adapted_folded_LoRA_PARD2_full_private_query_compile_v1",
            "auxiliary_implementation":"ordinary_BF16_folded_LoRA_effective_model_full_query_compile",
            "compile_model":True,"core_only_compile_model":False,
            "compiled_scope":"positions+mask+aligned_features_projector+embedding+EFFECTIVE_core_HF+full_vocab_argmax_topk_logsumexp+output_copies",
            "compile_dynamic_query_width":True,"private_per_B_immutable_bucket_code_cache":True,
            "original_reference_RAW_checkpoint_equality_scope":"untouched original reference ONLY",
            "effective_auxiliary_is_intentionally_adapted":True,
            "effective_auxiliary_raw_equality_to_original_checkpoint_claimed":False,
            "full_query_model_is_effective_adapted_model":self.adapter.core is self.model and self.base.model is self.model,
            "source_adaptation_checkpoint_reference_SHA_never_rewritten":True,
            "all_public_input_widths_B_through_2B_prepared_before_real_prefix":True,
            "core_or_target_forward_replaced":False,"global_compiler_configuration_changed":False,
            "inherited_fold_conditioning_cache_and_reference_effective_audits_unchanged":True,
            "compile_FP_order_can_change_unverified_proposals":True,
            "compiled_helper_source_sha256":file_sha(CompiledConditionedPARD2LatticeProposer._call.__code__.co_filename),
            "compiled_folded_runtime_source_sha256":file_sha(__file__),
            "performance_acceptance_or_quality_established":False}
        require(self.metadata["full_query_model_is_effective_adapted_model"],"Compiled auxiliary captures wrong inference core")

    def prepare(self,num_draft_tokens,buckets=None):
        result=super().prepare(num_draft_tokens,buckets=buckets)
        requested=self.attention_buckets if buckets is None else tuple(dict.fromkeys(buckets))
        required=set(range(num_draft_tokens,2*num_draft_tokens+1))
        require(all(self._full_query_widths_seen.get((num_draft_tokens,bucket),set())==required for bucket in requested),
            "Public folded preparation did not compile/warm every required B..2B input width")
        return {**result,"full_effective_auxiliary_query_compile":True,
            "all_requested_static_widths_seen":True,
            "compiled_effective_family_keys":[list(key)for key in sorted(self._compiled_full_queries)]}


def cpu_contract_tests():
    result={**folded_cpu_tests(),**compiled_cpu_tests()}
    # Actual NEW class routes through the SAME helper functions, while fold
    # construction and audit remain the original separately scoped functions.
    assert CompiledFoldedPARD2LoRALatticeProposer._call is CompiledConditionedPARD2LatticeProposer._call
    assert CompiledFoldedPARD2LoRALatticeProposer.audit_integrity is FoldedPARD2LoRALatticeProposer.audit_integrity
    obj=object.__new__(CompiledFoldedPARD2LoRALatticeProposer)
    obj.attention_buckets=(256,512,1024,2048);obj.cached_visible_ids=(1,2)
    marker=object();obj._compiled_full_queries={(15,256):marker};obj._eager_full_queries={(15,256):marker}
    assert obj.full_query_for(15,256) is marker and obj._query_for_preparation(15,256) is marker
    for action in (lambda:obj.full_query_for(31,256),lambda:obj._query_for_preparation(31,256),lambda:obj._call({},31)):
        try:action()
        except ValueError:pass
        else:raise AssertionError("Active folded prefix admitted missing compiler/buffer family")
    assert obj.cached_visible_ids==(1,2)and set(obj._compiled_full_queries)=={(15,256)}
    return {**result,"CPU_compiled_folded_actual_dispatch_original_audit_and_separate_identity_inheritance_passed":True,
        "GPU_work_performed":False}


if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if not args.self_test:p.error("Use --self-test or the separate root-run GPU proof/benchmark")
    print(json.dumps(cpu_contract_tests(),indent=2))
