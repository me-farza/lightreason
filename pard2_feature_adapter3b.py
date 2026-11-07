"""New projector-only auxiliary ablation; original target/PARD2 core frozen.

Feature contract is explicitly HF[2,18,33,36], concatenating existing three
residual taps with the final normalized original3B state. It is NOT the8B
PARD2 feature warp. At true token positioni use target featurei-1 (zero at0).
Every future mask uses the same last already-verified featurecut-1; future
teacher features/IDs are loss targets only and never proposal inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    if error.name!="torch":raise
    torch=nn=F=None
_Module=object if nn is None else nn.Module

FEATURE_SCHEMA="lightreason_pard2_projector_HF2_18_33_36_shifted_inputs_v1"
CHECKPOINT_SCHEMA="lightreason_pard2_projector_checkpoint_v1"
FEATURES_HF_TAPS=(2,18,33,36)
GAIN=0.02
CORE_REPO="amd/PARD2-Qwen3-8B"
CORE_REVISION="67a1516c8f6fc145cda99916799a0cbb3a4af135"
MASK_ID=151670


def require(condition,message):
    if not condition:raise ValueError(message)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def window_spec(ids,loss_mask,teacher_ids,cut,B,eos_ids,label_source="teacher"):
    """Pure causal alignment oracle; no future ID/feature enters input rows."""
    require(len(ids)==len(loss_mask)==len(teacher_ids) and type(cut)is int and
        1<=cut<len(ids)-1 and bool(loss_mask[cut]) and type(B)is int and 1<=B<=32,
        "Choose a valid assistant predictor cutpoint and B1..32")
    require(label_source in ("teacher","actual"),"Choose explicit original teacher-top1 or actual on-policy labels")
    inputs=list(ids[:cut+1])+[MASK_ID]*(B-1)
    feature_rows=[None]+list(range(cut))+[cut-1]*(B-1)
    labels,actual,valid=[],[],[];stopped=False
    for j in range(B):
        q=cut+j
        okay=(not stopped and q+1<len(ids) and bool(loss_mask[q]) and teacher_ids[q]>=0)
        labels.append(int(teacher_ids[q]) if okay else -100)
        actual.append(int(ids[q+1]) if okay else -100)
        valid.append(okay)
        if okay and (ids[q+1] in eos_ids or (label_source=="teacher" and teacher_ids[q] in eos_ids)):stopped=True
    require(any(valid) and all(row is None or row<cut for row in feature_rows),
        "Window lacks valid labels or exposes future target features")
    return {"input_ids":inputs,"feature_rows":feature_rows,"teacher_labels":labels,
        "actual_next_ids":actual,"valid":valid,"cutpoint":cut,"B":B,
        "last_available_target_feature":cut-1,"label_source":label_source}


class PARD2FeatureAdapter3B(_Module):
    def __init__(self,core,*,feature_width=8192,gain=GAIN):
        if torch is None:raise RuntimeError("Use a PyTorch environment for the projector helper")
        super().__init__()
        require(not core.training and all(not p.requires_grad for p in core.parameters()),
            "Provide the already audited frozen/eval auxiliary core")
        require(type(feature_width)is int and feature_width>0 and gain==GAIN,"Use declared feature width and fixed0.02gain")
        self.core=core;self.feature_width=feature_width;self.gain=GAIN
        embedding=core.get_input_embeddings().weight
        self.projector=nn.Linear(feature_width,embedding.shape[1],bias=False,
            device=embedding.device,dtype=torch.float32)
        nn.init.zeros_(self.projector.weight)

    def train(self,mode=True):
        super().train(mode);self.core.eval();return self

    def adapter_config(self):
        return {"feature_hf_taps":list(FEATURES_HF_TAPS),"feature_width":self.feature_width,
            "auxiliary_hidden_size":self.projector.out_features,"gain":self.gain,"bias":False,
            "feature_shift":"True tokeni uses featurei-1; row0zero; masks repeat last verified feature"}

    def project_features(self,features):
        require(features.shape[-1]==self.feature_width,"Explicit HF2/18/33/36 feature width differs")
        # Training uses FP32 masters inside autocast. Runtime may cast ONLY the
        # projector toBF16. The frozen core never receives an in-place cast.
        projected=self.projector(features.to(dtype=self.projector.weight.dtype))*self.gain
        return projected.to(dtype=self.core.get_input_embeddings().weight.dtype)

    def conditioned_embeddings(self,ids,features):
        require(tuple(ids.shape)==tuple(features.shape[:-1]),"Token/feature row geometry differs")
        return self.core.get_input_embeddings()(ids)+self.project_features(features)

    def make_window(self,record,cut,B,eos_ids,label_source="teacher"):
        spec=window_spec(record["input_ids"].tolist(),record["loss_mask"].tolist(),
            record["teacher_next_ids"].tolist(),cut,B,eos_ids,label_source)
        device=self.projector.weight.device
        true=torch.cat((record["aux_hidden_state"],record["hidden_state"]),dim=-1)
        require(true.shape[1]==self.feature_width,"Cached residual/finalnorm feature dimensions differ")
        # Gather ONLY rows<cut; no future reference feature is copied to CUDA.
        known=true[:cut].to(device=device,dtype=self.core.get_input_embeddings().weight.dtype)
        features=torch.cat((torch.zeros_like(known[:1]),known,known[-1:].expand(B-1,-1)),dim=0)[None]
        ids=torch.tensor([spec["input_ids"]],device=device,dtype=torch.long)
        labels=torch.tensor([spec["teacher_labels"] if label_source=="teacher" else spec["actual_next_ids"]],device=device,dtype=torch.long)
        return ids,features,labels,spec

    def window_logits(self,record,cut,B,eos_ids,label_source="teacher"):
        ids,features,labels,spec=self.make_window(record,cut,B,eos_ids,label_source)
        embeddings=self.conditioned_embeddings(ids,features)
        positions=torch.arange(ids.shape[1],device=ids.device,dtype=torch.long)[None]
        output=self.core(input_ids=None,inputs_embeds=embeddings,position_ids=positions,
            attention_mask=None,use_cache=False,logits_to_keep=B,return_dict=True,
            output_hidden_states=False,output_attentions=False)
        return output.logits[:,-B:],labels,spec

    def window_loss(self,record,cut,B,eos_ids,label_source="teacher"):
        logits,labels,spec=self.window_logits(record,cut,B,eos_ids,label_source)
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
        predicted=logits.argmax(-1)[0].detach().cpu().tolist()
        valid=sum(spec["valid"])
        matched=0
        for got,wanted,okay in zip(predicted,spec["actual_next_ids"],spec["valid"]):
            if not okay or got!=wanted:break
            matched+=1
            if wanted in eos_ids:break
        return loss,{"valid_teacher_labels":valid,
            "teacher_top1_correct":sum(okay and got==wanted for got,wanted,okay in
                zip(predicted,spec["teacher_labels"],spec["valid"])),
            "actual_trace_matching_draft_prefix":matched,"valid_actual_trace_horizon":valid,
            "proposal_IDs":predicted,"cutpoint":cut,
            "label_source":label_source,
            "diagnostic_scope":"Free parallel proposals on a fixed genuine prefix; offline trace comparison, not online acceptance/speed/quality"}

    @torch.no_grad() if torch is not None else lambda fn:fn
    def zero_baseline_check(self,record,cut,B,eos_ids):
        require(bool((self.projector.weight==0).all()),"Zero baseline check requires the original zero projector")
        ids,_,_,_=self.make_window(record,cut,B,eos_ids)
        positions=torch.arange(ids.shape[1],device=ids.device,dtype=torch.long)[None]
        reference=self.core(input_ids=ids,position_ids=positions,use_cache=False,logits_to_keep=B,
            output_hidden_states=False,output_attentions=False,return_dict=True).logits[:,-B:]
        actual,_,_=self.window_logits(record,cut,B,eos_ids)
        equal=reference.shape==actual.shape and reference.dtype==actual.dtype and bool(torch.equal(
            reference.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)))
        require(equal,"Zero projector did not reproduce original token-only auxiliary logits bit-for-bit")
        return {"zero_projector_full_logits_bits_equal_token_only_core":True,"draft_tokens":B,
            "target_quality_or_speed_claim":False}


def assert_projector_only_optimizer(adapter,optimizer,original_target_parameters=()):
    from feature_draft3b import assert_optimizer_excludes_target
    forbidden=list(original_target_parameters)+list(adapter.core.parameters())
    result=assert_optimizer_excludes_target(adapter,optimizer,forbidden)
    require(len([p for group in optimizer.param_groups for p in group["params"]])==1 and
        optimizer.param_groups[0]["params"][0] is adapter.projector.weight and
        adapter.projector.weight.dtype==torch.float32,"Only the independent FP32 projector may be optimized")
    return {**result,"pretrained_auxiliary_core_and_original_target_excluded":True,"only_projector_weight_trainable":True}


def load_projector_checkpoint(adapter,path,*,expected_core_live_sha256,expected_target_sha256):
    """Validate/load only the new projector; return JSON-only provenance."""
    path=Path(path).resolve();before=file_sha(path)
    saved=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    identity=saved.get("identity",{})
    require(saved.get("schema")==CHECKPOINT_SCHEMA and saved.get("feature_schema")==FEATURE_SCHEMA and
        saved.get("auxiliary_only") is True and saved.get("authoritative") is False and
        saved.get("adapter_config")==adapter.adapter_config() and
        identity.get("auxiliary_core_repo")==CORE_REPO and identity.get("auxiliary_core_revision")==CORE_REVISION and
        identity.get("auxiliary_core_live_parameter_sha256")==expected_core_live_sha256 and
        identity.get("original_target_live_parameter_sha256")==expected_target_sha256 and
        identity.get("source_sha256",{}).get(Path(__file__).name)==file_sha(__file__),
        "Projector checkpoint feature/core/target/source contract differs")
    state=saved.get("projector_state",{})
    require(set(state)=={"weight"} and tuple(state["weight"].shape)==(1024,8192) and
        state["weight"].dtype==torch.float32 and bool(torch.isfinite(state["weight"]).all()),
        "Projector checkpoint must contain only finite original FP32[1024,8192] weight")
    require(adapter.projector.weight.dtype==torch.float32,"Load FP32 masters before explicitly casting ONLY the runtime projector")
    adapter.projector.load_state_dict(state,strict=True)
    require(file_sha(path)==before,"Immutable projector checkpoint changed during loading")
    result={"projector_checkpoint_path":str(path),"projector_checkpoint_sha256":before,
        "feature_schema":FEATURE_SCHEMA,"adapter_config":saved["adapter_config"],
        "validated_identity":identity,"updates":saved.get("updates"),
        "auxiliary_only":True,"authoritative":False,"optimizer_state_loaded":False,
        "core_weights_loaded_or_changed_by_this_function":False}
    json.dumps(result)  # Artifacts receive JSON-only provenance, never optimizer/tensor state.
    return result


def cpu_contract_tests():
    ids=list(range(10,20));mask=[False]*3+[True]*6+[False];teacher=list(range(11,20))+[-100]
    spec=window_spec(ids,mask,teacher,4,4,{18})
    assert spec["input_ids"]==ids[:5]+[MASK_ID]*3
    assert spec["feature_rows"]==[None,0,1,2,3,3,3,3]
    assert spec["teacher_labels"]==[15,16,17,18]
    last=window_spec(ids,mask,teacher,7,4,{18})
    assert last["teacher_labels"]==[18,-100,-100,-100] and last["actual_next_ids"]==[18,-100,-100,-100]
    changed=list(teacher);changed[4]=18
    assert window_spec(ids,mask,changed,4,4,{18})["valid"]==[True,False,False,False]
    altered=list(ids);altered[5:]=[100]*5
    altered_teacher=teacher[:];altered_teacher[5:]=[100]*5
    other=window_spec(altered,mask,altered_teacher,4,4,{18})
    assert other["input_ids"]==spec["input_ids"] and other["feature_rows"]==spec["feature_rows"]
    return {"CPU_causal_input_feature_shift_no_future_leak_and_EOS_mask_tests_passed":True,"GPU_work_performed":False}


def cpu_tensor_tests():
    """Actual tiny CPU gradient/zero-baseline tests; no original model loaded."""
    if torch is None:raise RuntimeError("PyTorch is required for --cpu-torch-test")
    from types import SimpleNamespace
    class TinyCore(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(MASK_ID+1,4);self.head=nn.Linear(4,32,bias=False)
        def get_input_embeddings(self):return self.embedding
        def forward(self,input_ids=None,inputs_embeds=None,logits_to_keep=1,**kwargs):
            x=self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
            causal=x.cumsum(1)/torch.arange(1,x.shape[1]+1,dtype=x.dtype)[None,:,None]
            return SimpleNamespace(logits=self.head(causal)[:,-logits_to_keep:])
    torch.manual_seed(7)
    core=TinyCore().eval().requires_grad_(False)
    adapter=PARD2FeatureAdapter3B(core,feature_width=8)
    record={"input_ids":torch.tensor([1,2,3,4,5,6,7,8]),"loss_mask":torch.tensor([False,False,True,True,True,True,True,False]),
        "teacher_next_ids":torch.tensor([2,3,4,5,6,7,8,-100]),
        "aux_hidden_state":torch.randn(8,6,dtype=torch.bfloat16),"hidden_state":torch.randn(8,2,dtype=torch.bfloat16)}
    adapter.train();assert not core.training and all(not p.requires_grad for p in core.parameters())
    adapter.zero_baseline_check(record,3,3,{8})
    a,_,_=adapter.window_logits(record,3,3,{8})
    changed={k:v.clone() for k,v in record.items()};changed["aux_hidden_state"][3:]=500;changed["hidden_state"][3:]=-500
    changed["input_ids"][4:]=9;changed["teacher_next_ids"][4:]=10
    b,_,_=adapter.window_logits(changed,3,3,{8})
    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),"Future teacher data affected proposal inputs"
    original=nn.Parameter(torch.ones(1),requires_grad=False)
    optimizer=torch.optim.AdamW(adapter.projector.parameters(),lr=1e-3)
    assert_projector_only_optimizer(adapter,optimizer,[original])
    frozen={name:p.detach().clone() for name,p in core.named_parameters()}
    loss,_=adapter.window_loss(record,3,3,{8},label_source="actual")
    loss.backward()
    assert adapter.projector.weight.grad is not None and bool(torch.isfinite(adapter.projector.weight.grad).all())
    assert bool((adapter.projector.weight.grad!=0).any())
    assert all(p.grad is None for p in core.parameters()) and original.grad is None
    optimizer.step();assert bool((adapter.projector.weight!=0).any())
    assert all(torch.equal(p.view(torch.uint8),frozen[name].view(torch.uint8)) for name,p in core.named_parameters())
    bad=torch.optim.AdamW([adapter.projector.weight,next(core.parameters())],lr=1e-3)
    try:assert_projector_only_optimizer(adapter,bad,[original])
    except RuntimeError:pass
    else:raise AssertionError("Original frozen auxiliary core entered the optimizer")
    return {"actual_tiny_CPU_zero_baseline_causality_gradient_frozen_core_and_optimizer_tests_passed":True,
        "GPU_work_performed":False}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true");parser.add_argument("--cpu-torch-test",action="store_true")
    args=parser.parse_args()
    if args.cpu_torch_test:print(json.dumps(cpu_tensor_tests()))
    elif args.self_test:print(json.dumps(cpu_contract_tests()))
    else:parser.error("Use --self-test or --cpu-torch-test; this helper never emits authoritative outputs")
