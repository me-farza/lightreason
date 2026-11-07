"""Separate AUXILIARY LoRA+feature adaptation; original target never changes.

The pinned PARD2 reference is kept intact. A distinct frozen-weight core copy
gets zero-output LoRA branches in q/k/v/o and MLP down projections. Only new
FP32 A/B matrices and the independent8192->1024 feature projector train.
Feature alignment and loss/window functions reuse the frozen projector helper.
Fast export folds LoRA into yet another auxiliary copy, with explicitly
reported BF16 rounding differences; no original reference tensor is written.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

from pard2_feature_adapter3b import (PARD2FeatureAdapter3B,FEATURE_SCHEMA,CORE_REPO,CORE_REVISION,
    require,file_sha,torch,nn,F)

_Module=object if nn is None else nn.Module
CHECKPOINT_SCHEMA="lightreason_pard2_lora_feature_checkpoint_v1"
SUFFIXES=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.down_proj")


def expected_parameters(rank=16,layers=28,feature_width=8192,hidden=1024):
    # Actual smallQwen3 q/o dimensions are2048 despite hidden1024.
    lora=layers*rank*((1024+2048)+(1024+1024)+(1024+1024)+(2048+1024)+(3072+1024))
    projector=feature_width*hidden
    return {"LoRA":lora,"projector":projector,"total":lora+projector,
        "FP32_masters_gradients_Adam_moments_bytes":(lora+projector)*16}


class AuxiliaryLoRALinear(_Module):
    def __init__(self,base,rank,alpha):
        super().__init__()
        require(isinstance(base,nn.Linear) and base.bias is None and not base.weight.requires_grad,
            "LoRA wraps only the separate frozen bias-free auxiliary Linear copy")
        require(type(rank)is int and 1<=rank<=min(base.in_features,base.out_features) and alpha>0,"Invalid LoRA geometry")
        self.base=base;self.rank=rank;self.alpha=float(alpha);self.scale=self.alpha/rank
        self.in_features,self.out_features=base.in_features,base.out_features
        self.lora_A=nn.Parameter(torch.empty((rank,base.in_features),device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros((base.out_features,rank),device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=5**.5)

    @property
    def weight(self):return self.base.weight

    @property
    def bias(self):return self.base.bias

    def forward(self,x):
        original=self.base(x)
        delta=F.linear(F.linear(x.to(self.lora_A.dtype),self.lora_A),self.lora_B)*self.scale
        return original+delta.to(original.dtype)


class PARD2LoRAFeatureAdapter3B(PARD2FeatureAdapter3B):
    def __init__(self,reference_core,*,rank=16,alpha=16,feature_width=8192):
        from pard2_proposer3b_v1 import model_state
        require(not reference_core.training and all(not p.requires_grad for p in reference_core.parameters()),
            "Provide the immutable audited original auxiliary reference")
        copied=deepcopy(reference_core).eval().requires_grad_(False)
        original_storage={(str(p.device),p.untyped_storage().data_ptr()) for p in reference_core.parameters()}
        require(all((str(p.device),p.untyped_storage().data_ptr()) not in original_storage for p in copied.parameters()),
            "Auxiliary adaptation copy aliases an original reference weight")
        super().__init__(copied,feature_width=feature_width)
        object.__setattr__(self,"_reference_core",reference_core)
        self._reference_signature=model_state(reference_core)
        self._frozen_base_parameters=dict(copied.named_parameters(remove_duplicate=False))
        self._frozen_versions={name:(id(p),p.data_ptr(),p._version,tuple(p.shape),str(p.dtype))
            for name,p in self._frozen_base_parameters.items()}
        self.rank,self.alpha=rank,float(alpha)
        layers={}
        for index in range(len(copied.model.layers)):
            for suffix in SUFFIXES:
                name=f"model.layers.{index}.{suffix}"
                parent_path,leaf=name.rsplit(".",1);parent=copied.get_submodule(parent_path)
                wrapped=AuxiliaryLoRALinear(getattr(parent,leaf),rank,alpha)
                setattr(parent,leaf,wrapped);layers[name]=wrapped
        object.__setattr__(self,"_lora_layers",layers)  # Already registered through core; no duplicate module paths.
        self.train()

    def adapter_config(self):
        return {**super().adapter_config(),"auxiliary_lora":{"rank":self.rank,"alpha":self.alpha,
            "scale":self.alpha/self.rank,"target_modules":list(self._lora_layers),
            "base_copy_is_independent":True,"original_reference_and_target_are_immutable":True}}

    def lora_parameters(self):
        return [p for layer in self._lora_layers.values() for p in (layer.lora_A,layer.lora_B)]

    def trainable_parameters(self):return list(self.projector.parameters())+self.lora_parameters()

    def assert_frozen_bases_unchanged(self):
        from pard2_proposer3b_v1 import model_state
        require(model_state(self._reference_core)==self._reference_signature,"Original auxiliary reference changed")
        require(not self.core.training and all(not p.requires_grad and p.grad is None and
            (id(p),p.data_ptr(),p._version,tuple(p.shape),str(p.dtype))==self._frozen_versions[name]
            for name,p in self._frozen_base_parameters.items()),"A cloned original base parameter changed or received gradients")

    def audit_frozen_base_bytes(self,original_raw_audit):
        from pard2_proposer3b_v1 import tensor_sha
        self.assert_frozen_bases_unchanged()
        wanted=original_raw_audit["tensor_sha256"]
        require(set(wanted)==set(self._frozen_base_parameters),"Copied auxiliary base keys differ from original raw tensors")
        hashes={name:tensor_sha(p) for name,p in self._frozen_base_parameters.items()}
        require(all(hashes[name]==wanted[name]["stored_sha256"] for name in hashes),
            "A cloned frozen base tensor differs from the original auxiliary checkpoint")
        return {"all_cloned_auxiliary_base_tensor_bytes_equal_original":True,"keys_checked":len(hashes),
            "frozen_base_sha256_by_original_name":hashes,
            "effective_auxiliary_is_adapted":True,"LoRA_and_projector_values_intentionally_excluded":True}

    def trainable_state_dict(self):
        return {"projector":{name:p.detach().cpu().clone() for name,p in self.projector.state_dict().items()},
            "lora":{name:{"A":layer.lora_A.detach().cpu().clone(),"B":layer.lora_B.detach().cpu().clone()}
                for name,layer in self._lora_layers.items()}}

    def load_trainable_state_dict(self,state):
        require(set(state)=={"projector","lora"} and set(state["projector"])=={"weight"} and
            set(state["lora"])==set(self._lora_layers),"Adapted auxiliary trainable state keys differ")
        pairs=[(self.projector.weight,state["projector"]["weight"])]
        for name,layer in self._lora_layers.items():
            require(set(state["lora"][name])=={"A","B"},"LoRA matrix names differ")
            pairs.extend(((layer.lora_A,state["lora"][name]["A"]),(layer.lora_B,state["lora"][name]["B"])))
        require(all(tuple(p.shape)==tuple(v.shape) and p.dtype==v.dtype==torch.float32 and
            bool(torch.isfinite(v).all()) for p,v in pairs),"LoRA/projector master geometry/dtype/values differ")
        with torch.no_grad():
            for p,value in pairs:p.copy_(value)

    @torch.no_grad() if torch is not None else lambda fn:fn
    def zero_baseline_check(self,record,cut,B,eos_ids):
        require(all(bool((layer.lora_B==0).all()) for layer in self._lora_layers.values()),"Zero baseline needs every LoRABzero")
        result=super().zero_baseline_check(record,cut,B,eos_ids)
        ids,_,_,_=self.make_window(record,cut,B,eos_ids)
        positions=torch.arange(ids.shape[1],device=ids.device,dtype=torch.long)[None]
        original=self._reference_core(input_ids=ids,position_ids=positions,use_cache=False,logits_to_keep=B,
            return_dict=True,output_hidden_states=False,output_attentions=False).logits[:,-B:]
        actual,_,_=self.window_logits(record,cut,B,eos_ids)
        require(torch.equal(original.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)),
            "Zero LoRA/projector did not preserve original auxiliary reference logits bit-for-bit")
        return {**result,"zero_LoRA_and_projector_logits_bits_equal_original_auxiliary_reference":True}

    @torch.no_grad() if torch is not None else lambda fn:fn
    def fold_auxiliary_copy(self):
        """Return a NEW ordinary HF auxiliary; never mutate reference/base copy."""
        self.assert_frozen_bases_unchanged()
        require(not torch.backends.cuda.matmul.allow_tf32,"Disable TF32 for the explicit FP32 LoRA fold")
        folded=deepcopy(self._reference_core).eval().requires_grad_(False)
        for name,layer in self._lora_layers.items():
            destination=folded.get_submodule(name)
            if bool((layer.lora_B==0).all()):weight=layer.base.weight.detach().clone()
            else:weight=(layer.base.weight.float()+layer.scale*(layer.lora_B.float()@layer.lora_A.float())).to(layer.base.weight.dtype)
            destination.weight.copy_(weight)
        forbidden={(str(p.device),p.untyped_storage().data_ptr()) for p in
            list(self._reference_core.parameters())+list(self.core.parameters())}
        require(all((str(p.device),p.untyped_storage().data_ptr()) not in forbidden for p in folded.parameters()),
            "Folded auxiliary aliases a reference or training parameter")
        self.assert_frozen_bases_unchanged()
        return folded,{"role":"Separately adapted AUXILIARY only; original3B must verify every output",
            "original_target_or_pinned_auxiliary_reference_changed":False,"new_independent_auxiliary_copy":True,
            "LoRA_folded_into_BF16_auxiliary_weights":True,
            "nonzero_fold_has_different_auxiliary_floating_point_evaluation_order":True,
            "projector_still_separate":True,"acceptance_quality_and_speed_require_measurement":True}


def assert_adaptation_only_optimizer(adapter,optimizer,original_target_parameters=()):
    from feature_draft3b import assert_optimizer_excludes_target
    forbidden=list(original_target_parameters)+list(adapter._reference_core.parameters())+list(adapter._frozen_base_parameters.values())
    result=assert_optimizer_excludes_target(adapter,optimizer,forbidden)
    wanted=adapter.trainable_parameters();actual=[p for group in optimizer.param_groups for p in group["params"]]
    require(len(wanted)==len(actual) and {id(p) for p in wanted}=={id(p) for p in actual} and
        all(p.dtype==torch.float32 for p in actual),"Optimizer must contain every new LoRA/projector master exactly once")
    return {**result,"only_new_LoRA_and_projector_trainable":True,"original_target_reference_and_cloned_base_excluded":True,
        "new_trainable_parameters":sum(p.numel() for p in wanted)}


def load_adaptation_checkpoint(adapter,path,*,expected_core_live_sha256,expected_target_sha256):
    path=Path(path).resolve();before=file_sha(path)
    saved=torch.load(path,map_location="cpu",weights_only=True,mmap=True);identity=saved.get("identity",{})
    require(saved.get("schema")==CHECKPOINT_SCHEMA and saved.get("feature_schema")==FEATURE_SCHEMA and
        saved.get("auxiliary_only") is True and saved.get("authoritative") is False and
        saved.get("auxiliary_effective_model_adaptation") is True and saved.get("segment_integrity_verified") is True and
        saved.get("adapter_config")==adapter.adapter_config() and identity.get("auxiliary_core_repo")==CORE_REPO and
        identity.get("auxiliary_core_revision")==CORE_REVISION and
        identity.get("auxiliary_core_live_parameter_sha256")==expected_core_live_sha256 and
        identity.get("original_target_live_parameter_sha256")==expected_target_sha256 and
        identity.get("source_sha256",{}).get(Path(__file__).name)==file_sha(__file__) and
        identity.get("source_sha256",{}).get("pard2_feature_adapter3b.py")==file_sha(Path(__file__).with_name("pard2_feature_adapter3b.py")),
        "Adapted auxiliary checkpoint contract differs")
    adapter.load_trainable_state_dict(saved["trainable_state"])
    require(file_sha(path)==before,"Immutable adaptation checkpoint changed during loading")
    result={"checkpoint_path":str(path),"checkpoint_sha256":before,"feature_schema":FEATURE_SCHEMA,
        "adapter_config":saved["adapter_config"],"validated_identity":identity,"updates":saved.get("updates"),
        "auxiliary_only":True,"authoritative":False,"original_target_unchanged":True,
        "auxiliary_effective_model_adapted":True,"optimizer_state_loaded":False}
    json.dumps(result);return result


def cpu_contract_tests():
    from pard2_feature_adapter3b import cpu_contract_tests as causal_tests
    result=causal_tests();counts=expected_parameters()
    assert counts=={"LoRA":6422528,"projector":8388608,"total":14811136,
        "FP32_masters_gradients_Adam_moments_bytes":236978176}
    result["CPU_LoRA_geometry_and_trainable_memory_counts_passed"]=True
    return result


def cpu_tensor_tests():
    if torch is None:raise RuntimeError("PyTorch is required for actual tiny CPU tests")
    from types import SimpleNamespace
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.self_attn=nn.Module()
            for name,inside,outside in (("q_proj",4,8),("k_proj",4,4),("v_proj",4,4),("o_proj",8,4)):
                setattr(self.self_attn,name,nn.Linear(inside,outside,bias=False))
            self.mlp=nn.Module();self.mlp.up_proj=nn.Linear(4,12,bias=False);self.mlp.down_proj=nn.Linear(12,4,bias=False)
        def forward(self,x):
            a=self.self_attn;x=x+a.o_proj(a.q_proj(x))+a.k_proj(x)+a.v_proj(x)
            return x+self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))
    class Core(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(151671,4);self.head=nn.Linear(4,32,bias=False)
            self.model=nn.Module();self.model.layers=nn.ModuleList([Layer(),Layer()])
        def get_input_embeddings(self):return self.embedding
        def forward(self,input_ids=None,inputs_embeds=None,logits_to_keep=1,**kwargs):
            x=self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
            for layer in self.model.layers:x=layer(x)
            return SimpleNamespace(logits=self.head(x)[:,-logits_to_keep:])
    torch.manual_seed(7);reference=Core().eval().requires_grad_(False)
    before={name:p.detach().clone() for name,p in reference.named_parameters()}
    adapter=PARD2LoRAFeatureAdapter3B(reference,rank=2,alpha=2,feature_width=8)
    record={"input_ids":torch.tensor([1,2,3,4,5,6,7,8]),"loss_mask":torch.tensor([False,False,True,True,True,True,True,False]),
        "teacher_next_ids":torch.tensor([2,3,4,5,6,7,8,-100]),"aux_hidden_state":torch.randn(8,6,dtype=torch.bfloat16),
        "hidden_state":torch.randn(8,2,dtype=torch.bfloat16)}
    adapter.zero_baseline_check(record,3,3,{8})
    target=nn.Parameter(torch.ones(1),requires_grad=False)
    optimizer=torch.optim.AdamW([{"params":adapter.projector.parameters(),"lr":1e-6},
        {"params":adapter.lora_parameters(),"lr":1e-5}],betas=(.9,.95),weight_decay=0.)
    assert_adaptation_only_optimizer(adapter,optimizer,[target]);loss,_=adapter.window_loss(record,3,3,{8},"actual")
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.trainable_parameters())
    assert any(bool((layer.lora_B.grad!=0).any()) for layer in adapter._lora_layers.values())
    assert all(p.grad is None for p in adapter._frozen_base_parameters.values())
    optimizer.step();adapter.assert_frozen_bases_unchanged()
    assert all(torch.equal(p.view(torch.uint8),before[name].view(torch.uint8)) for name,p in reference.named_parameters())
    folded,meta=adapter.fold_auxiliary_copy();assert meta["new_independent_auxiliary_copy"]
    assert not any(isinstance(m,AuxiliaryLoRALinear) for m in folded.modules())
    saved=adapter.trainable_state_dict();fresh=PARD2LoRAFeatureAdapter3B(reference,rank=2,alpha=2,feature_width=8)
    fresh.load_trainable_state_dict(saved)
    for a,b in zip(adapter.trainable_parameters(),fresh.trainable_parameters()):assert torch.equal(a,b)
    bad=torch.optim.AdamW(adapter.trainable_parameters()+[next(reference.parameters())],lr=1e-5)
    try:assert_adaptation_only_optimizer(adapter,bad,[target])
    except RuntimeError:pass
    else:raise AssertionError("Original auxiliary reference entered optimizer")
    return {"CPU_zero_baseline_new_LoRA_gradients_original_and_cloned_base_immutability_fold_copy_and_state_roundtrip_passed":True,
        "GPU_work_performed":False}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--cpu-torch-test",action="store_true");args=parser.parse_args()
    print(json.dumps(cpu_tensor_tests() if args.cpu_torch_test else cpu_contract_tests()))
