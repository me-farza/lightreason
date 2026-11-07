"""Frozen PARD2 plus a separate learned projector; auxiliary proposals ONLY.

The target history excludes its unprocessed anchor. Genuine auxiliary input at
absolute positioni receives target featurei-1 (position0 gets zeros). Every
mask receives the last ALREADY VERIFIED featurep-1 when visible length=p+1.
New genuine rows overwrite former mask KV using fresh verified conditioning.
No target/cache/model forward is monkeypatched and no target token is emitted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import torch
    from pard2_proposer3b_v1 import (PARD2Proposer,BucketedPARD2LatticeProposer,
        model_state,tensor_sha,file_sha,require)
except ModuleNotFoundError as exc:
    if exc.name not in ("torch","transformers"):raise
    torch=None;PARD2Proposer=BucketedPARD2LatticeProposer=object


FEATURE_TAPS=(2,18,33,36)
FEATURE_WIDTH=8192


def _no_grad(function):
    return torch.no_grad()(function)if torch is not None else function


def conditioning_rows(common,visible_length,draft_tokens,verified_prefix_length):
    """Public row-index oracle. None means the exact zero feature vector."""
    if (any(type(v)is not int for v in (common,visible_length,draft_tokens,verified_prefix_length))or
            not 0<=common<visible_length or visible_length!=verified_prefix_length+1 or
            verified_prefix_length<1 or not 1<=draft_tokens<=32):
        raise ValueError("Use visible length=p+1, p>=1, common within history and B1..32")
    true=[None if position==0 else position-1 for position in range(common,visible_length)]
    masks=[verified_prefix_length-1]*(draft_tokens-1)
    result=true+masks
    if any(index is not None and not 0<=index<verified_prefix_length for index in result):
        raise AssertionError("Conditioning accessed an unverified/future feature")
    return {"feature_rows":result,"true_rows":len(true),"mask_rows":len(masks),
        "absolute_input_positions":list(range(common,visible_length+draft_tokens-1)),
        "last_verified_feature_row":verified_prefix_length-1}


def checkpoint_admission(saved,*,diagnostic_test_mode=False):
    """Fail closed on unverified training segments; diagnostic mode is explicit."""
    if saved.get("diagnostic_only")is True:
        if not(diagnostic_test_mode is True and type(saved.get("updates"))is int and
                saved["updates"]==0 and saved.get("training_performed")is False and
                saved.get("diagnostic_kind")in("zero","random_small")):
            raise ValueError("Diagnostic projector requires explicit test mode, diagnostic flags and update0")
        return {"kind":"diagnostic_only","diagnostic_kind":saved["diagnostic_kind"],
            "updates":0,"training_performed":False,"trained_or_quality_claim_permitted":False}
    proof=saved.get("segment_integrity",{})
    optimizer=proof.get("optimizer_projector_only",{})if isinstance(proof,dict)else{}
    required=("original_target_parameter_versions_and_bytes_unchanged","pretrained_auxiliary_core_bytes_unchanged",
              "all_original_auxiliary_tensors_preserved","source_and_all_feature_files_unchanged")
    optimizer_required=("original_target_parameters_and_storage_excluded",
                        "pretrained_auxiliary_core_and_original_target_excluded","only_projector_weight_trainable")
    if not(saved.get("segment_integrity_verified")is True and type(saved.get("updates"))is int and
            saved["updates"]>=0 and isinstance(proof,dict)and
            all(proof.get(key)is True for key in required)and
            proof.get("target_parameters_trained")is False and proof.get("auxiliary_core_parameters_trained")is False and
            isinstance(optimizer,dict)and all(optimizer.get(key)is True for key in optimizer_required)):
        raise ValueError("Inference requires a published verified projector-only training segment")
    return {"kind":"verified_training_segment","segment_integrity_verified":True,
        "segment_integrity":proof,"updates":saved.get("updates"),
        "trained_or_quality_equivalence_established_by_admission":False}


class ConditionedPARD2LatticeProposer(BucketedPARD2LatticeProposer):
    def __init__(self,checked_base,projector_checkpoint_path,*,expected_target_sha256,
                 max_len=2048,attention_buckets=(256,512,1024,2048),top_k=8,
                 compile_model=False,share_pool=True,warmups=3,copy_output=True,diagnostic_test_mode=False):
        from exact10_validation import hash_model_parameters
        from pard2_feature_adapter3b import PARD2FeatureAdapter3B,load_projector_checkpoint
        require(isinstance(checked_base,PARD2Proposer),"Use the pinned audited original PARD2 loader")
        require(isinstance(expected_target_sha256,str)and len(expected_target_sha256)==64,
                "Bind the projector to the original target live SHA")
        self._core_initial_sha256=hash_model_parameters(checked_base.model)
        core_state=model_state(checked_base.model)
        self.projector_checkpoint_path=Path(projector_checkpoint_path).resolve()
        checkpoint_before=file_sha(self.projector_checkpoint_path)
        saved=torch.load(self.projector_checkpoint_path,map_location="cpu",weights_only=True,mmap=True)
        admission=checkpoint_admission(saved,diagnostic_test_mode=diagnostic_test_mode)
        del saved
        adapter=PARD2FeatureAdapter3B(checked_base.model)
        require(adapter.core is checked_base.model,"Feature helper replaced the original auxiliary core")
        loaded=load_projector_checkpoint(adapter,self.projector_checkpoint_path,
            expected_core_live_sha256=self._core_initial_sha256,expected_target_sha256=expected_target_sha256)
        require(isinstance(loaded,dict),"Projector loader must return JSON-only validated metadata")
        # Cast/freeze ONLY the independent projector. The original core's
        # storage, dtype, modules and forward functions are never reassigned.
        adapter.projector.to(device=checked_base.device,dtype=torch.bfloat16).eval().requires_grad_(False)
        require(tuple(adapter.projector.weight.shape)==(1024,FEATURE_WIDTH)and adapter.projector.bias is None,
                "Use the declared bias-free8192->1024 auxiliary projector")
        require(model_state(checked_base.model)==core_state and
                hash_model_parameters(checked_base.model)==self._core_initial_sha256,
                "Projector construction/loading changed the original auxiliary core")
        require(file_sha(self.projector_checkpoint_path)==checkpoint_before,"Projector checkpoint changed while loading")
        super().__init__(checked_base,max_len=max_len,attention_buckets=attention_buckets,
            top_k=top_k,compile_model=compile_model,share_pool=share_pool,warmups=warmups,copy_output=copy_output)
        self.adapter=adapter
        self._adapter_config=dict(adapter.adapter_config())
        self._adapter_methods=tuple((getattr(getattr(adapter,name),"__func__",getattr(adapter,name)),
            adapter.__dict__.get(name))for name in("project_features","conditioned_embeddings"))
        self._projector_state=model_state(adapter.projector)
        self._projector_bytes_sha256=tensor_sha(adapter.projector.weight)
        self._projector_checkpoint_sha256=checkpoint_before
        with torch.inference_mode(False),torch.no_grad():
            self._verified_feature_snapshot=torch.empty((1,self.capacity,FEATURE_WIDTH),
                device=self.device,dtype=torch.bfloat16)
        self._verified_feature_count=0
        self.metadata={**self.metadata,
            "auxiliary_implementation":"PARD2_four_target_features_separate_projector_v1",
            "feature_taps_hf_tuple":list(FEATURE_TAPS),"feature_width":FEATURE_WIDTH,
            "conditioning":"Eaux(ids)+0.02*P(aligned_verified_features)",
            "genuine_position_i_feature":"i-1; position0 zero",
            "all_mask_feature_rows":"last verified featurep-1; never anchor/future features",
            "projector_checkpoint":str(self.projector_checkpoint_path),
            "projector_checkpoint_sha256":checkpoint_before,"projector_loader_metadata":loaded,
            "checkpoint_admission":admission,"diagnostic_test_mode":bool(diagnostic_test_mode),
            "projector_runtime_dtype":"torch.bfloat16","projector_runtime_sha256":self._projector_bytes_sha256,
            "projector_runtime_parameters_frozen":True,"auxiliary_core_live_sha256":self._core_initial_sha256,
            "original_target_live_sha256":expected_target_sha256,
            "shared_genuine_feature_prefix_validated":True,
            "full_auxiliary_prefill_uses_same_conditioned_embeddings":True,
            "all_old_mask_KV_overwritten_when_genuine":True,
            "model_forward_functions_assigned":False,"auxiliary_is_final_authority":False,
            "performance_acceptance_output_agreement_and_quality_unestablished":True,
            "conditioned_runtime_sha256":file_sha(__file__)}
        self.assert_draft_unchanged()

    def reset(self):
        super().reset()
        self._verified_feature_count=0

    def assert_draft_unchanged(self):
        super().assert_draft_unchanged()
        if hasattr(self,"_projector_state"):
            require(self.adapter.core is self.model and self.adapter.adapter_config()==self._adapter_config and
                tuple((getattr(getattr(self.adapter,name),"__func__",getattr(self.adapter,name)),
                    self.adapter.__dict__.get(name))for name in("project_features","conditioned_embeddings"))==self._adapter_methods,
                "Auxiliary conditioning core/config/functions changed")
            require(model_state(self.adapter.projector)==self._projector_state,
                    "Frozen auxiliary projector identity/versions/modules changed")

    def audit_integrity(self,*,verify_bytes=True,verify_files=True):
        self.assert_draft_unchanged()
        result=self.base.audit_integrity(verify_bytes=False,verify_files=verify_files)
        require(file_sha(self.projector_checkpoint_path)==self._projector_checkpoint_sha256,
                "Immutable auxiliary projector checkpoint bytes changed")
        if verify_bytes:
            from exact10_validation import hash_model_parameters
            core=hash_model_parameters(self.model)
            projector=tensor_sha(self.adapter.projector.weight)
            require(core==self._core_initial_sha256,"Original auxiliary core live bytes changed")
            require(projector==self._projector_bytes_sha256,"Frozen runtime projector live bytes changed")
            result.update(auxiliary_core_live_parameter_sha256=core,projector_runtime_sha256=projector)
        return {**result,"projector_checkpoint_unchanged":True,
                "projector_parameter_module_identity_unchanged":True,"live_bytes_verified":bool(verify_bytes)}

    def _call(self,entry,block):
        if "feature_inputs"not in entry:
            # First synthetic warmup only, before graph capture; no real request.
            entry["feature_inputs"]=torch.zeros((1,entry["input_ids"].shape[1],FEATURE_WIDTH),
                device=self.device,dtype=torch.bfloat16)
        bucket=entry["attention_bucket"]
        positions=self.gpu_base+entry["offsets"]
        allowed=self.key_positions[:bucket].view(1,1,1,-1)<=positions.view(1,1,-1,1)
        embeddings=self.adapter.conditioned_embeddings(entry["input_ids"],entry["feature_inputs"])
        output=self.forward(input_ids=None,inputs_embeds=embeddings,past_key_values=self.cache_views[bucket],
            cache_position=positions,position_ids=positions.view(1,-1),attention_mask=allowed,use_cache=True,
            logits_to_keep=block,output_attentions=False,output_hidden_states=False,return_dict=True)
        raw=output.logits[:,-block:]
        self.outputs[block].copy_(raw.argmax(-1))
        logits=raw.float();values,ids=torch.topk(logits,self.top_k,dim=-1,largest=True,sorted=True)
        self.lattice_ids[block].copy_(ids)
        self.lattice_logps[block].copy_(values-torch.logsumexp(logits,dim=-1,keepdim=True))

    def _prefill_conditioned_history(self,ids,features,p):
        aligned=torch.zeros((1,p+1,FEATURE_WIDTH),device=self.device,dtype=torch.bfloat16)
        aligned[:,1:].copy_(features[:,:p])
        embeddings=self.adapter.conditioned_embeddings(ids,aligned)
        output=self.model(input_ids=None,inputs_embeds=embeddings,attention_mask=None,use_cache=True,
            logits_to_keep=1,return_dict=True,output_attentions=False,output_hidden_states=False)
        dynamic=output.past_key_values
        require(len(dynamic.layers)==len(self.cache.layers),"Conditioned genuine prefill layer count differs")
        for destination,source in zip(self.cache.layers,dynamic.layers):
            require(destination.keys is not None,"Prepare public graph families before conditioned prefill")
            destination.keys[:,:,:p+1].copy_(source.keys)
            destination.values[:,:,:p+1].copy_(source.values)
        self.prefill_count+=1

    @_no_grad
    def propose(self,visible_ids,committed_features,committed_prefix_length,num_draft_tokens):
        p,B=committed_prefix_length,num_draft_tokens
        require(isinstance(visible_ids,torch.Tensor)and visible_ids.dtype==torch.long and
                visible_ids.ndim==2 and visible_ids.shape[0]==1,"Use original visible IDs[1,p+1]")
        require(type(p)is int and p>=1 and visible_ids.shape[1]==p+1 and type(B)is int and 1<=B<=32,
                "Visible IDs must end in the unprocessed anchor; use public B1..32")
        require(isinstance(committed_features,torch.Tensor)and committed_features.ndim==3 and
            committed_features.shape[0]==1 and committed_features.shape[1]>=p and
            committed_features.shape[2]==FEATURE_WIDTH and committed_features.dtype==torch.bfloat16 and
            committed_features.device==self.device,"Pass original verified Feature4 history excluding the anchor")
        self.assert_draft_unchanged()
        length=p+1;bucket=self.select_bucket(length+B-1)
        require(all((B,width,bucket)in self.graphs for width in range(B,2*B+1)),
                "Prepare the public conditioned graph family before inference")
        visible=tuple(visible_ids.detach().cpu().reshape(-1).tolist())
        require(min(visible)>=0 and max(visible)<self.target_vocab_size,"Original visible IDs outside target vocabulary")
        ids=visible_ids.to(self.device).contiguous()
        common=0
        for left,right in zip(self.cached_visible_ids,visible[:-1]):
            if left!=right:break
            common+=1
        shared_features=max(0,common-1)
        require(shared_features<=self._verified_feature_count,"Cached feature-prefix accounting differs")
        require(torch.equal(committed_features[:,:shared_features],self._verified_feature_snapshot[:,:shared_features]),
                "Verified features conditioning reused genuine KV changed; reset request")
        used_prefill=False
        try:
            if not self.cached_visible_ids or common==0 or length-common>B+1:
                self._prefill_conditioned_history(ids,committed_features,p)
                common=length-1;used_prefill=True
            count=length-common
            entry=self.graphs[(B,count+B-1,bucket)]
            entry["input_ids"][:,:count].copy_(ids[:,common:])
            aligned=entry["feature_inputs"]
            if common==0:
                aligned[:,:1].zero_()
                if count>1:aligned[:,1:count].copy_(committed_features[:,:p])
            else:aligned[:,:count].copy_(committed_features[:,common-1:p])
            if B>1:aligned[:,count:].copy_(committed_features[:,p-1:p].expand(1,B-1,FEATURE_WIDTH))
            self.gpu_base.fill_(common);entry["graph"].replay()
            output=self.outputs[B].clone()if self.copy_output else self.outputs[B]
            self.cached_visible_ids=visible
            self._verified_feature_snapshot[:,:p].copy_(committed_features[:,:p])
            self._verified_feature_count=p
            self.assert_draft_unchanged()
        except Exception:
            self.reset()
            raise
        self.last_report={"visible_tokens":length,"verified_feature_prefix":p,"num_draft_tokens":B,
            "reused_true_prefix_tokens":common,"processed_true_tokens":count,"mask_tokens":B-1,
            "last_verified_mask_feature_row":p-1,"used_conditioned_full_prefill":used_prefill,
            "graph_key":[B,count+B-1,bucket],"attention_bucket":bucket,
            "genuine_and_mask_feature_rows":conditioning_rows(common,length,B,p)["feature_rows"],
            "all_conditioning_features_already_verified":True,"auxiliary_never_final_authority":True}
        return output

    @_no_grad
    def propose_lattice(self,visible_ids,committed_features,committed_prefix_length,num_draft_tokens):
        self.propose(visible_ids,committed_features,committed_prefix_length,num_draft_tokens)
        ids,logps=self.lattice_ids[num_draft_tokens],self.lattice_logps[num_draft_tokens]
        return(ids.clone(),logps.clone())if self.copy_output else(ids,logps)


def cpu_contract_tests():
    from pard2_feature_adapter3b import window_spec,FEATURES_HF_TAPS
    assert tuple(FEATURES_HF_TAPS)==FEATURE_TAPS
    for p in (1,2,24,255,256,511):
        for B in (1,7,15,31):
            ids=list(range(p+B+3));teacher=ids[1:]+[-100]
            training=window_spec(ids,[True]*len(ids),teacher,p,B,set())
            assert conditioning_rows(0,p+1,B,p)["feature_rows"]==training["feature_rows"]
            for common in (0,max(0,p-3),p):
                plan=conditioning_rows(common,p+1,B,p)
                assert plan["feature_rows"][:plan["true_rows"]]==[
                    None if i==0 else i-1 for i in range(common,p+1)]
                assert plan["feature_rows"][plan["true_rows"]:]==[p-1]*(B-1)
                assert all(i is None or i<p for i in plan["feature_rows"])
    # A row that was a mask now becomes genuine: old f3 must be replaced by f4.
    before=conditioning_rows(4,5,7,4)
    after=conditioning_rows(5,7,7,6)
    assert before["feature_rows"][1]==3 and after["feature_rows"][0]==4
    # Pure zero-projector algebra; BF16 byte/argmax equivalence requires GPU proof.
    embedding=[-2.,0.,3.5];features=[7.,-9.,2.,11.]
    zero=[[0.]*len(features)for _ in embedding]
    assert[e+.02*sum(w*f for w,f in zip(row,features))for e,row in zip(embedding,zero)]==embedding
    for bad in ((0,4,7,4),(0,1,7,0),(5,5,7,4),(0,5,33,4)):
        try:conditioning_rows(*bad)
        except ValueError:pass
        else:raise AssertionError("Invalid future/length feature contract accepted")
    proof={key:True for key in("original_target_parameter_versions_and_bytes_unchanged","pretrained_auxiliary_core_bytes_unchanged",
        "all_original_auxiliary_tensors_preserved","source_and_all_feature_files_unchanged")}
    proof.update(target_parameters_trained=False,auxiliary_core_parameters_trained=False,
        optimizer_projector_only={key:True for key in("original_target_parameters_and_storage_excluded",
            "pretrained_auxiliary_core_and_original_target_excluded","only_projector_weight_trainable")})
    trained={"segment_integrity_verified":True,"segment_integrity":proof,"updates":25}
    assert checkpoint_admission(trained)["kind"]=="verified_training_segment"
    for bad in ({**trained,"segment_integrity_verified":False},{**trained,"segment_integrity":{}},
            {**trained,"segment_integrity":{**proof,"target_parameters_trained":True}}):
        try:checkpoint_admission(bad)
        except ValueError:pass
        else:raise AssertionError("Unverified or target-trained segment admitted")
    diagnostic={"diagnostic_only":True,"updates":0,"training_performed":False,"diagnostic_kind":"random_small"}
    assert checkpoint_admission(diagnostic,diagnostic_test_mode=True)["trained_or_quality_claim_permitted"]is False
    for bad,mode in ((diagnostic,False),({**diagnostic,"updates":1},True),({**diagnostic,"training_performed":True},True)):
        try:checkpoint_admission(bad,diagnostic_test_mode=mode)
        except ValueError:pass
        else:raise AssertionError("Diagnostic checkpoint escaped explicit test-only admission")
    return {"CPU_true_shift_mask_repeat_cache_overwrite_and_zero_projector_algebra_passed":True,
            "CPU_training_segment_and_explicit_diagnostic_admission_tests_passed":True,
            "GPU_zero_projection_bits_or_model_equivalence_tested":False}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if not args.self_test:parser.error("Runtime API; use --self-test for CPU alignment checks")
    print(json.dumps(cpu_contract_tests(),indent=2))
