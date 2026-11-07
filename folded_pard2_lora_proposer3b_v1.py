"""Independent folded LoRA AUXILIARY; original PARD2/reference/target intact.

Raw checkpoint equality is asserted ONLY for the pinned original reference.
The effective auxiliary intentionally has adapted BF16 weights, its own SHA,
and versions. Its source checkpoint keeps the ORIGINAL reference SHA; no fake
projector-only checkpoint or rewritten source identity is manufactured.
"""
from __future__ import annotations

from pathlib import Path

try:
    import torch
    from pard2_proposer3b_v1 import PARD2Proposer,BucketedPARD2LatticeProposer,model_state,tensor_sha,file_sha,require
    from conditioned_pard2_proposer3b_v1 import ConditionedPARD2LatticeProposer
except ModuleNotFoundError as exc:
    if exc.name not in("torch","transformers"):raise
    torch=None;PARD2Proposer=BucketedPARD2LatticeProposer=ConditionedPARD2LatticeProposer=object


def canonical_modules(layers=28):
    return [f"model.layers.{i}.{suffix}"for i in range(layers)for suffix in
        ("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]


def validate_published_adaptation(saved):
    """Additional segment proof; the frozen helper separately checks all state."""
    if (saved.get("diagnostic_only")is True or saved.get("segment_integrity_verified")is not True or
            saved.get("auxiliary_effective_model_adaptation")is not True or
            type(saved.get("updates"))is not int or saved["updates"]<0):
        raise ValueError("Use a published verified LoRA/projector training segment, not a diagnostic")
    proof=saved.get("segment_integrity",{})
    if not isinstance(proof,dict)or not proof:
        raise ValueError("Adaptation checkpoint must include its complete training segment integrity proof")
    # The exact field contract is intentionally explicit; do not infer safety
    # from generic truthy values or the adaptation flag alone.
    required=("original_target_parameter_versions_and_bytes_unchanged",
        "original_auxiliary_reference_unchanged",
        "all_cloned_auxiliary_base_tensor_bytes_equal_original","source_and_all_feature_files_unchanged")
    optimizer=proof.get("optimizer_adaptation_only",{})
    if not(all(proof.get(key)is True for key in required)and
            proof.get("target_parameters_trained")is False and
            proof.get("frozen_reference_or_base_parameters_trained")is False and
            proof.get("auxiliary_effective_model_adaptation")is True and
            proof.get("new_LoRA_and_projector_parameters_trained")is True and isinstance(optimizer,dict)and
            optimizer.get("only_new_LoRA_and_projector_trainable")is True and
            optimizer.get("original_target_reference_and_cloned_base_excluded")is True and
            optimizer.get("new_trainable_parameters")==14811136):
        raise ValueError("LoRA training proof must preserve original target/reference/cloned base and train only new A/B/P")
    return {"kind":"verified_LoRA_projector_training_segment","segment_integrity_verified":True,
        "updates":saved["updates"],"segment_integrity":proof,"diagnostic_checkpoint":False}


def _assert_no_alias(effective,forbidden):
    storage={(str(p.device),p.untyped_storage().data_ptr())for p in forbidden}
    require(all((str(p.device),p.untyped_storage().data_ptr())not in storage for p in effective.parameters()),
            "Effective adapted auxiliary aliases a protected reference/training parameter")


def verify_fold_formula(adapter,folded):
    """Independent value check of140 folds and every unchanged state tensor."""
    reference=adapter._reference_core
    source=reference.state_dict(keep_vars=True);actual=folded.state_dict(keep_vars=True)
    require(set(source)==set(actual),"Effective ordinary HF state coverage differs from original reference")
    names=canonical_modules()
    require(list(adapter._lora_layers)==names and adapter.rank==16 and adapter.alpha==16.,
            "Use the declared140-module rank16/alpha16 auxiliary adaptation")
    expected_keys={name+".weight":name for name in names}
    checked=0;changed=0
    with torch.no_grad(),torch.autocast(device_type=next(folded.parameters()).device.type,enabled=False):
        for key,value in actual.items():
            original=source[key]
            require(tuple(value.shape)==tuple(original.shape)and value.dtype==original.dtype==torch.bfloat16,
                    "Fold changed an auxiliary state tensor shape/dtype: "+key)
            if key in expected_keys:
                layer=adapter._lora_layers[expected_keys[key]]
                if bool((layer.lora_B==0).all()):expected=original.detach()
                else:expected=(original.float()+(layer.alpha/layer.rank)*(layer.lora_B.float()@layer.lora_A.float())).to(original.dtype)
                checked+=1
                changed+=not bool(torch.equal(value.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8)))
            else:expected=original
            require(bool(torch.equal(value.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))),
                    "Effective folded auxiliary differs from declared original+BA formula: "+key)
    from pard2_lora_feature_adapter3b import AuxiliaryLoRALinear
    require(checked==140,"Fold lineage did not check every declared adapted weight")
    require(not any(isinstance(module,AuxiliaryLoRALinear)for module in folded.modules()),
            "Online effective HF model still contains LoRA wrappers")
    _assert_no_alias(folded,list(reference.parameters())+list(adapter.parameters()))
    return {"all140_folded_matrices_match_explicit_FP32_formula":checked==140,
        "adapted_matrices_checked":checked,"adapted_matrices_differing_from_reference":changed,
        "all_other_state_tensors_bit_equal_original_reference":True,
        "unchanged_state_keys_checked":len(actual)-checked,"ordinary_HF_core_without_LoRA_wrappers":True,
        "no_parameter_storage_alias_to_reference_training_base_A_B_or_projector":True,
        "B_zero_preserves_original_BF16_bytes_including_signed_zero":True,
        "nonzero_auxiliary_unfolded_vs_folded_FP_order_equivalence_not_claimed":True}


class EffectiveAdaptedPARD2Base(PARD2Proposer):
    """Graph-compatible base with DISTINCT original/effective integrity scopes."""
    def __init__(self,reference,model,*,reference_sha256,fold_metadata):
        from exact10_validation import hash_model_parameters
        self.reference=reference;self.model=model
        self.device=reference.device;self.mask_id=reference.mask_id;self.target_vocab_size=reference.target_vocab_size
        self.cache_mode=reference.cache_mode;self.max_cache_len=reference.max_cache_len
        self.reference_sha256=reference_sha256;self.effective_sha256=hash_model_parameters(model)
        self.parameter_versions=[(p,p._version)for p in model.parameters()]
        self._effective_state=model_state(model)
        self.metadata={"role":"Intentionally adapted auxiliary ONLY; original target freshly verifies every final token",
            "variant":"ordinary_BF16_PARD2_copy_with_folded_LoRA",
            "effective_auxiliary_is_intentionally_adapted":True,
            "effective_auxiliary_raw_equality_to_original_checkpoint_claimed":False,
            "original_reference_repo":reference.metadata["repo_id"],"original_reference_revision":reference.metadata["revision"],
            "original_reference_live_sha256":reference_sha256,"effective_auxiliary_live_sha256":self.effective_sha256,
            "effective_hash_equals_original_reference":self.effective_sha256==reference_sha256,
            "fold":fold_metadata,"dtype":"torch.bfloat16","mask_id":self.mask_id,
            "draft_attention":"sdpa","cache_mode":self.cache_mode,"max_cache_len":self.max_cache_len}
        self.reset()

    def assert_draft_unchanged(self):
        self.reference.assert_draft_unchanged()
        require(all(p._version==version for p,version in self.parameter_versions),
                "Effective adapted auxiliary parameter version changed")

    def audit_integrity(self,*,verify_bytes=True,verify_files=True):
        self.assert_draft_unchanged()
        reference_audit=self.reference.audit_integrity(verify_bytes=verify_bytes,verify_files=verify_files)
        require(model_state(self.model)==self._effective_state,"Effective adapted auxiliary objects/functions/modes changed")
        if verify_bytes:
            from exact10_validation import hash_model_parameters
            reference_sha=reference_audit["auxiliary_live_parameter_sha256"]
            effective_sha=hash_model_parameters(self.model)
            require(reference_sha==self.reference_sha256,"Original pinned auxiliary reference live bytes changed")
            require(effective_sha==self.effective_sha256,"Effective adapted auxiliary live bytes changed after setup")
        else:reference_sha=self.reference_sha256;effective_sha=self.effective_sha256
        return {"original_reference_audit":reference_audit,"original_reference_live_sha256":reference_sha,
            "effective_adapted_auxiliary_live_sha256":effective_sha,
            "effective_auxiliary_parameter_module_versions_unchanged":True,
            "original_target_tokenizer_unchanged":reference_audit["original_target_tokenizer_unchanged"],
            "effective_auxiliary_is_intentionally_adapted":True,
            "effective_auxiliary_raw_equality_to_original_checkpoint_claimed":False,
            "live_bytes_verified":bool(verify_bytes)}


class FoldedPARD2LoRALatticeProposer(ConditionedPARD2LatticeProposer):
    """Same frozen conditioning/cache methods, independently adapted core."""
    def __init__(self,reference,adaptation_checkpoint_path,*,expected_target_sha256,max_len=2048,
                 attention_buckets=(256,512,1024,2048),top_k=8,compile_model=False,share_pool=True,warmups=3,copy_output=True):
        from exact10_validation import hash_model_parameters
        from audit_pard2_state_coverage_v1 import audit_loaded_checkpoint_weights
        from pard2_lora_feature_adapter3b import PARD2LoRAFeatureAdapter3B,load_adaptation_checkpoint
        from pard2_feature_adapter3b import PARD2FeatureAdapter3B
        require(isinstance(reference,PARD2Proposer)and not isinstance(reference,EffectiveAdaptedPARD2Base),
                "Provide the untouched pinned original PARD2 reference")
        path=Path(adaptation_checkpoint_path).expanduser().resolve();checkpoint_sha=file_sha(path)
        saved=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        admission=validate_published_adaptation(saved);del saved
        raw_reference=audit_loaded_checkpoint_weights(reference)
        source_hash=hash_model_parameters(reference.model);reference_state=model_state(reference.model)
        train_adapter=PARD2LoRAFeatureAdapter3B(reference.model,rank=16,alpha=16)
        loaded=load_adaptation_checkpoint(train_adapter,path,expected_core_live_sha256=source_hash,
                                          expected_target_sha256=expected_target_sha256)
        cloned_base_audit=train_adapter.audit_frozen_base_bytes(raw_reference)
        require(not torch.backends.cuda.matmul.allow_tf32,"Disable TF32 for the explicit FP32 auxiliary fold")
        with torch.no_grad(),torch.autocast(device_type=reference.device.type,enabled=False):
            folded,fold_metadata=train_adapter.fold_auxiliary_copy()
            fold_formula=verify_fold_formula(train_adapter,folded)
        train_adapter.audit_frozen_base_bytes(raw_reference)
        adapter=PARD2FeatureAdapter3B(folded)
        with torch.no_grad():adapter.projector.load_state_dict(train_adapter.projector.state_dict(),strict=True)
        adapter.projector.to(device=reference.device,dtype=torch.bfloat16).eval().requires_grad_(False)
        require(model_state(reference.model)==reference_state and hash_model_parameters(reference.model)==source_hash,
                "Preparing adapted auxiliary changed the original pinned reference")
        require(file_sha(path)==checkpoint_sha,"Immutable LoRA/projector source checkpoint changed during fold")
        fold_metadata={**fold_metadata,**fold_formula,"source_checkpoint_sha256":checkpoint_sha,
            "original_reference_live_sha256":source_hash,"original_reference_raw_tensor_audit":raw_reference,
            "cloned_frozen_base_audit":cloned_base_audit,"loaded_adaptation":loaded,
            "fold_arithmetic":"FP32(originalW)+alpha/r*(FP32B@FP32A), castBF16; B0exactcopy",
            "TF32_disabled":True,"autocast_disabled_during_fold":True,
            "proposal_agreement_acceptance_and_quality_not_inferred_from_fold":True}
        effective=EffectiveAdaptedPARD2Base(reference,folded,reference_sha256=source_hash,fold_metadata=fold_metadata)
        # Bypass ONLY the old projector-only constructor/admission. All online
        # conditioning/cache methods are inherited byte-for-byte unchanged.
        BucketedPARD2LatticeProposer.__init__(self,effective,max_len=max_len,attention_buckets=attention_buckets,
            top_k=top_k,compile_model=compile_model,share_pool=share_pool,warmups=warmups,copy_output=copy_output)
        self.adapter=adapter;self.reference=reference
        self._core_initial_sha256=effective.effective_sha256
        self._adapter_config=dict(adapter.adapter_config())
        self._adapter_methods=tuple((getattr(getattr(adapter,name),"__func__",getattr(adapter,name)),
            adapter.__dict__.get(name))for name in("project_features","conditioned_embeddings"))
        self._projector_state=model_state(adapter.projector);self._projector_bytes_sha256=tensor_sha(adapter.projector.weight)
        self.projector_checkpoint_path=path;self._projector_checkpoint_sha256=checkpoint_sha
        with torch.inference_mode(False),torch.no_grad():
            self._verified_feature_snapshot=torch.empty((1,self.capacity,8192),device=self.device,dtype=torch.bfloat16)
        self._verified_feature_count=0
        self.metadata={**self.metadata,"variant":"intentionally_adapted_folded_LoRA_PARD2_Feature4_auxiliary_v1",
            "auxiliary_implementation":"ordinary_HF_BF16_folded_LoRA_PARD2_copy_with_Feature4_projector",
            "target_features_for_auxiliary_conditioning_only":True,
            "effective_auxiliary_is_intentionally_adapted":True,
            "original_reference_unchanged":True,"original_target_unchanged":True,
            "source_checkpoint_kind":"LoRA+projector adaptation; not old projector-only schema",
            "adaptation_checkpoint_path":str(path),"adaptation_checkpoint_sha256":checkpoint_sha,
            "checkpoint_admission":admission,"diagnostic_test_mode":False,
            "projector_runtime_sha256":self._projector_bytes_sha256,"projector_runtime_dtype":"torch.bfloat16",
            "original_source_reference_SHA_retained_in_checkpoint_identity":True,
            "effective_auxiliary_live_sha256":effective.effective_sha256,
            "fold":fold_metadata,"feature_taps_hf_tuple":[2,18,33,36],"feature_width":8192,
            "all_online_conditioning_cache_math_inherited_unchanged":True,
            "unfolded_vs_folded_auxiliary_proposal_identity_not_assumed":True,
            "performance_acceptance_target_output_agreement_and_quality_unestablished":True,
            "folded_runtime_source_sha256":file_sha(__file__)}
        self.assert_draft_unchanged()

    def audit_integrity(self,*,verify_bytes=True,verify_files=True):
        result=super().audit_integrity(verify_bytes=verify_bytes,verify_files=verify_files)
        # Do not expose the old ambiguous 'core unchanged from checkpoint'
        # interpretation: this hash is the intentionally adapted effective core.
        effective=result.pop("auxiliary_core_live_parameter_sha256",self._core_initial_sha256)
        return {**result,"effective_adapted_auxiliary_live_sha256":effective,
            "adaptation_checkpoint_sha256":file_sha(self.projector_checkpoint_path),
            "effective_auxiliary_raw_equality_to_original_checkpoint_claimed":False,
            "fold_lineage_verified_at_setup":self.metadata["fold"]["all140_folded_matrices_match_explicit_FP32_formula"]}


def cpu_contract_tests():
    names=canonical_modules();assert len(names)==140 and len(set(names))==140
    assert names[0]=="model.layers.0.self_attn.q_proj"and names[-1]=="model.layers.27.mlp.down_proj"
    proof={key:True for key in("original_target_parameter_versions_and_bytes_unchanged",
        "original_auxiliary_reference_unchanged",
        "all_cloned_auxiliary_base_tensor_bytes_equal_original","source_and_all_feature_files_unchanged")}
    proof.update(target_parameters_trained=False,frozen_reference_or_base_parameters_trained=False,
        auxiliary_effective_model_adaptation=True,new_LoRA_and_projector_parameters_trained=True,
        optimizer_adaptation_only={"only_new_LoRA_and_projector_trainable":True,"original_target_reference_and_cloned_base_excluded":True,
            "new_trainable_parameters":14811136})
    saved={"updates":25,"auxiliary_effective_model_adaptation":True,"segment_integrity_verified":True,"segment_integrity":proof}
    assert validate_published_adaptation(saved)["kind"]=="verified_LoRA_projector_training_segment"
    for bad in({**saved,"diagnostic_only":True},{**saved,"segment_integrity":{}},
        {**saved,"segment_integrity":{**proof,"frozen_reference_or_base_parameters_trained":True}}):
        try:validate_published_adaptation(bad)
        except ValueError:pass
        else:raise AssertionError("Unverified or original-base-trained checkpoint admitted")
    return {"CPU140canonical_modules_and_separate_adaptation_admission_passed":True,"GPU_work_performed":False}


if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");args=parser.parse_args()
    if not args.self_test:parser.error("Runtime API; only --self-test is standalone")
    print(json.dumps(cpu_contract_tests(),indent=2))
