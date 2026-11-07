"""Fixed-public-shape CUDA graphs of ORIGINAL eager HF fresh-cache prefill.

No compile, forward replacement, padded attention, cache prefix reuse, changed
weights or answer cache. Each warmup/capture constructs a NEW DynamicCache
inside the original HF call. Replay runs only its captured GPU work; that
cache's Python state is never used as an incremental cache. Inputs are fully
overwritten private buffers, and original source kernels/math are retained.
Backend graph workspace choices may still change FP order: strict identity
requires measured bit agreement, otherwise this is a numerical candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!="torch":raise
    torch=None


def valid_length(length,max_len):
    if type(length)is not int or not 1<=length<max_len:
        raise ValueError("Use one public nonempty prefix length below fixed capacity")
    return length


@dataclass
class PrefillOutputs:
    logits: object
    token_ids: object
    key_values: tuple
    hidden_states: object
    features: object
    raw_output: object


@dataclass
class _Entry:
    length: int
    input_ids: object
    aligned_features: object=None
    graph: object=None
    output: object=None
    setup_s: float=0.


class EagerHFPrefillGraph:
    """prepare(L) before real prefix; replay(ids,verified_features=None).

    conditioned_embeddings is an unchanged auxiliary helper method. Its input
    features are shifted here: input0 gets zero; inputi gets verified featurei-1.
    consumer optionally captures exact copies into an EXISTING static owner.
    Returned outputs alias graph memory and must be cloned before another replay.
    """
    def __init__(self,model,*,max_len=2048,feature_indices=(),conditioned_embeddings=None,
                 feature_width=8192,consumer=None,warmups=2,share_pool=False):
        if torch is None:raise RuntimeError("Use the pinned GPU PyTorch runtime")
        import transformers
        from pard2_proposer3b_v1 import model_state
        if torch.__version__.split("+")[0]!="2.9.1"or transformers.__version__!="4.57.1":
            raise ValueError("Use existing Torch2.9.1/Transformers4.57.1")
        if model.training or any(p.requires_grad or p.dtype!=torch.bfloat16 for p in model.parameters()):
            raise ValueError("Original/effective source model must be frozen/eval allBF16")
        self.model=model;self.device=model.get_input_embeddings().weight.device
        if self.device.type!="cuda"or type(warmups)is not int or warmups<1:
            raise ValueError("Use CUDA and positive public warmup count")
        self.max_len,self.feature_indices=max_len,tuple(feature_indices)
        self.conditioned_embeddings,self.feature_width=conditioned_embeddings,feature_width
        self.consumer,self.warmups=consumer,warmups
        self._state=model_state(model);self.entries={};self.python_forward_calls=0
        self._pool=torch.cuda.graph_pool_handle()if share_pool else None
        self.metadata={"implementation":"original_eager_HF_fresh_DynamicCache_prefill_graph_v1",
            "compiled":False,"model_forward_replaced":False,"padded_attention":False,
            "fresh_DynamicCache_created_during_every_warmup_and_capture":True,
            "captured_cache_Python_state_never_incrementally_reused":True,
            "private_inputs_fully_overwritten_each_replay":True,"prepared_lengths":[],
            "conditioned_embeddings":conditioned_embeddings is not None,"feature_indices":list(self.feature_indices),
            "existing_static_storage_copy_captured":consumer is not None,
            "strict_output_identity_requires_measured_bit_gate":True,"share_pool":bool(share_pool),"setup_s":0.}

    def assert_model_unchanged(self):
        from pard2_proposer3b_v1 import model_state
        if model_state(self.model)!=self._state:
            raise RuntimeError("Source model objects/forwards/modes/parameters changed; initialize HF bookkeeping before capture")

    def _forward(self,entry):
        from transformers.cache_utils import DynamicCache
        self.python_forward_calls+=1
        # FRESH Python cache object INSIDE this call/capture. NEVER replay a
        # Python forward using the cache object retained from a prior call.
        cache=DynamicCache(config=self.model.config)
        inputs={"input_ids":entry.input_ids}
        if self.conditioned_embeddings is not None:
            inputs={"input_ids":None,"inputs_embeds":self.conditioned_embeddings(entry.input_ids,entry.aligned_features)}
        raw=self.model(**inputs,attention_mask=None,past_key_values=cache,use_cache=True,logits_to_keep=1,
            return_dict=True,output_attentions=False,output_hidden_states=bool(self.feature_indices))
        features=(torch.cat(tuple(raw.hidden_states[index]for index in self.feature_indices),dim=-1)
                  if self.feature_indices else None)
        output=PrefillOutputs(raw.logits,raw.logits[:,-1].argmax(-1,keepdim=True),
            tuple((layer.keys,layer.values)for layer in raw.past_key_values.layers),raw.hidden_states,features,raw)
        if self.consumer is not None:self.consumer(output,entry.length)
        return output

    def prepare(self,length):
        valid_length(length,self.max_len)
        if length in self.entries:return {"setup_s":0.,"new_graph":False,"length":length}
        self.assert_model_unchanged();began=time.perf_counter()
        with torch.inference_mode(False),torch.no_grad():
            entry=_Entry(length,torch.zeros((1,length),device=self.device,dtype=torch.long))
            if self.conditioned_embeddings is not None:
                entry.aligned_features=torch.zeros((1,length,self.feature_width),device=self.device,dtype=torch.bfloat16)
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream),torch.inference_mode(False),torch.no_grad():
                for _ in range(self.warmups):temporary=self._forward(entry)
                del temporary
            torch.cuda.current_stream(self.device).wait_stream(stream);torch.cuda.synchronize(self.device)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self._pool,stream=stream),torch.inference_mode(False),torch.no_grad():
                entry.output=self._forward(entry)
            torch.cuda.synchronize(self.device);entry.graph=graph
        self.assert_model_unchanged()
        if len(entry.output.key_values)!=self.model.config.num_hidden_layers:
            raise RuntimeError("Fresh HF prefill cache layer coverage differs")
        if any(k.shape[-2]!=length or v.shape[-2]!=length for k,v in entry.output.key_values):
            raise RuntimeError("Captured prefill retained anything beyond the fresh public prefix")
        entry.setup_s=time.perf_counter()-began;self.entries[length]=entry
        self.metadata["prepared_lengths"]=sorted(self.entries);self.metadata["setup_s"]+=entry.setup_s
        return {"setup_s":entry.setup_s,"new_graph":True,"length":length,"cache_layers":len(entry.output.key_values)}

    def replay(self,ids,verified_features=None):
        if not isinstance(ids,torch.Tensor)or ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1:
            raise ValueError("Use original CUDA/CPU LongTensor[1,L] IDs")
        length=valid_length(ids.shape[1],self.max_len)
        if length not in self.entries:raise RuntimeError("Prepare the fixed public prefill length before real data")
        entry=self.entries[length]
        if self.conditioned_embeddings is not None:
            if (not isinstance(verified_features,torch.Tensor)or verified_features.shape!=(1,length-1,self.feature_width)or
                    verified_features.dtype!=torch.bfloat16 or verified_features.device!=self.device):
                raise ValueError("Auxiliary prefill needs ONLY verified features[1,L-1,8192], excluding anchor")
        elif verified_features is not None:raise ValueError("Original target prefill does not accept auxiliary features")
        with torch.inference_mode(False),torch.no_grad():
            entry.input_ids.copy_(ids)
            if entry.aligned_features is not None:
                entry.aligned_features[:,:1].zero_()
                entry.aligned_features[:,1:].copy_(verified_features)
            entry.graph.replay()
        return entry.output


class TargetFeature4PrefillGraph:
    """Component adapter over an EXISTING verifier; no installation/patching."""
    def __init__(self,owner,*,warmups=2):
        self.owner=owner
        owner.assert_storage_consistent()
        if len(owner.cache.layers)!=owner.model.config.num_hidden_layers or any(
                layer.keys is None or layer.values is None for layer in owner.cache.layers):
            raise ValueError("Target component requires the EXISTING initialized static cache")
        def consume(output,length):
            if len(output.key_values)!=len(owner.cache.layers):raise RuntimeError("Target prefill copy layer coverage differs")
            owner.cache.reset()
            for layer,(key,value)in zip(owner.cache.layers,output.key_values):
                layer.keys[:,:,:length].copy_(key);layer.values[:,:,:length].copy_(value)
            owner.committed_feature_history[:,:length].copy_(output.features)
            owner._anchor_buffer.copy_(output.token_ids)
        self.graph=EagerHFPrefillGraph(owner.model,max_len=owner.max_len,feature_indices=(2,18,33,36),
                                     consumer=consume,warmups=warmups)

    def prepare(self,length):return self.graph.prepare(length)

    def prefill(self,ids):
        owner=self.owner;owner._check_ids(ids);owner.assert_target_unchanged();owner.assert_storage_consistent()
        torch.cuda.synchronize(owner.device);began=time.perf_counter()
        output=self.graph.replay(ids);length=ids.shape[1]
        owner.last_logits=output.logits[:,-1,:]
        owner.anchor=owner._anchor_buffer;owner.last_next_ids=owner._anchor_buffer.reshape(-1)
        owner.current_position,owner.pending=length,None
        owner._pending_entry=owner._pending_topology=None
        owner.verify_calls=owner.committed_input_tokens=owner.commit_calls=owner.compacted_input_tokens=0
        owner.last_depths=owner.last_ancestors=owner.last_input_ids=None
        owner.last_bucket=owner.last_entry_key=owner.last_node_features=None
        owner.last_report={"operation":"original_eager_HF_fresh_cache_prefill_graph_component","prompt_tokens":length,
            "cache_excludes_unprocessed_anchor":True,"feature_indices":[2,18,33,36]}
        torch.cuda.synchronize(owner.device);owner.prefill_s=time.perf_counter()-began
        owner.assert_storage_consistent()
        return owner.anchor


class ConditionedAuxPrefillGraph:
    """Component adapter over an EXISTING proposer; no private method patch."""
    def __init__(self,owner,*,warmups=2):
        self.owner=owner
        if len(owner.cache.layers)!=owner.model.config.num_hidden_layers or any(
                layer.keys is None or layer.values is None or layer.keys.shape[-2]!=owner.capacity or
                layer.values.shape[-2]!=owner.capacity for layer in owner.cache.layers):
            raise ValueError("Auxiliary component requires the EXISTING full-capacity initialized static cache")
        for view in owner.cache_views.values():view.assert_storage_consistent(require_initialized=True)
        def consume(output,length):
            if len(output.key_values)!=len(owner.cache.layers):raise RuntimeError("Auxiliary prefill copy layer coverage differs")
            for layer,(key,value)in zip(owner.cache.layers,output.key_values):
                layer.keys[:,:,:length].copy_(key);layer.values[:,:,:length].copy_(value)
        self.graph=EagerHFPrefillGraph(owner.model,max_len=owner.capacity,conditioned_embeddings=owner.adapter.conditioned_embeddings,
                                     consumer=consume,warmups=warmups)

    def prepare(self,length):return self.graph.prepare(length)

    def prefill(self,ids,features,p):
        if type(p)is not int or ids.shape[1]!=p+1:raise ValueError("Auxiliary prefix must end at its unprocessed target anchor")
        self.graph.replay(ids,features[:,:p]);self.owner.prefill_count+=1


def cpu_contract_tests():
    for length in(1,2,255,256,257):assert valid_length(length,2048)==length
    for length in(0,-1,2048,True):
        try:valid_length(length,2048)
        except ValueError:pass
        else:raise AssertionError("Invalid/dynamic public shape accepted")
    # Exact static temporal alignment used by replay:0zero; inputi<-featurei-1.
    for length in(1,2,17):
        source=list(range(length-1));aligned=[None]+source
        assert len(aligned)==length and all(index is None or index<length-1 for index in aligned)
    # Exercise the ACTUAL replay method with tensor/context stubs: complete
    # private overwrite and temporal shift must precede one GPU-only replay.
    from contextlib import nullcontext
    from copy import deepcopy
    from types import SimpleNamespace
    global torch
    original_torch=torch;events=[]
    class Tensor:
        def __init__(self,data,shape,dtype="bf16"):
            self.data,self.shape,self.dtype=deepcopy(data),shape,dtype
            self.ndim,self.device=len(shape),"cuda:0"
        def copy_(self,other):self.data=deepcopy(other.data);events.append("ids");return self
        def __getitem__(self,index):
            parent=self;rows=list(range(self.shape[1]))[index[1]]
            class Region:
                def zero_(self):
                    for row in rows:parent.data[0][row]=[0]*parent.shape[2]
                    events.append("zero");return self
                def copy_(self,other):
                    for row,source in zip(rows,other.data[0]):parent.data[0][row]=list(source)
                    events.append("features");return self
            return Region()
    try:
        torch=SimpleNamespace(Tensor=Tensor,long="long",bfloat16="bf16",inference_mode=lambda *_:nullcontext(),no_grad=nullcontext)
        ids=Tensor([[10,11,12,13]],(1,4),"long")
        features=Tensor([[[1,2,3],[4,5,6],[7,8,9]]],(1,3,3))
        entry=_Entry(4,Tensor([[99]*4],(1,4),"long"),Tensor([[[99]*3]*4],(1,4,3)))
        expected=[[[0,0,0],[1,2,3],[4,5,6],[7,8,9]]]
        def replay():
            assert entry.input_ids.data==ids.data and entry.aligned_features.data==expected
            events.append("graph")
        entry.graph=SimpleNamespace(replay=replay);entry.output=object()
        instance=EagerHFPrefillGraph.__new__(EagerHFPrefillGraph)
        instance.max_len,instance.device,instance.feature_width=16,"cuda:0",3
        instance.conditioned_embeddings=True;instance.entries={4:entry}
        assert instance.replay(ids,features)is entry.output and events==["ids","zero","features","graph"]
        for invalid in(None,Tensor([[[1,2,3]]],(1,1,3)),Tensor(features.data,(1,3,3),"float32")):
            before=list(events)
            try:instance.replay(ids,invalid)
            except ValueError:pass
            else:raise AssertionError("Invalid/future/missing feature contract accepted")
            assert events==before,"Invalid feature shape performed writes before rejection"
        instance.conditioned_embeddings=None;before=list(events)
        try:instance.replay(ids,features)
        except ValueError:pass
        else:raise AssertionError("Auxiliary features entered original-target component")
        assert events==before
    finally:torch=original_torch
    return {"CPU_public_shape_and_verified_feature_shift_contracts_passed":True,
        "actual_replay_private_overwrite_shift_GPU_dispatch_order_passed":True,
        "invalid_conditioning_rejected_before_any_input_write":True,
        "target_rejects_auxiliary_feature_inputs":True,"GPU_capture_or_output_parity_tested":False}


if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");args=parser.parse_args()
    if not args.self_test:parser.error("Component API; standalone mode is --self-test")
    print(json.dumps(cpu_contract_tests(),indent=2))
