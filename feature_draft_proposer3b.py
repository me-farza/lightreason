"""Auxiliary feature-draft runtime, never a final target or token authority.

Stable auxiliary physical rowj contains the decoder KV for semantic tokenj+1,
conditioned on VERIFIED target features atj. At target committed prefixp, the
visible IDs containp+1 tokens: the last token is the unprocessed target anchor.
Auxiliary teacher refill processes that anchor and predicts its first draft.
Every later refill overwrites new stable rows using verified target features,
including rows that had speculative KV with different predicted features.

prepare() captures all public refill/recurrent families BEFORE real prefix data.
Initial full-prefix auxiliary prefill runs during propose() and is timed. Eager
control uses the same static-cache/mask functions as graph replay. This module
does not commit target KV, emit final tokens, interpret auxiliary EOS, train a
model, load evaluation answers or assert any speed/acceptance/quality result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = F = None

from feature_draft3b import FeatureDraft3B, FEATURE_SCHEMA, TARGET_REVISION, TARGET_TAPS


DEFAULT_BUCKETS = (256, 512, 1024, 2048)


def bucket_for_end(end, buckets):
    if type(end) is not int or end < 1:
        raise ValueError("Use a positive exclusive physical end")
    for bucket in buckets:
        if end <= bucket:
            return bucket
    raise ValueError("No attention bucket covers the required physical end")


def refill_alignment(old_prefix, new_prefix):
    """(physical KV row, verified feature row, semantic/input token position)."""
    if type(old_prefix) is not int or type(new_prefix) is not int or not 0 <= old_prefix <= new_prefix:
        raise ValueError("Committed prefixes must advance monotonically")
    return [(row, row, row+1) for row in range(old_prefix, new_prefix)]


def public_families(prompt_length, max_new_tokens, draft_tokens, buckets):
    """Exact families reachable without inspecting EOS, predictions or answers."""
    if (type(prompt_length) is not int or prompt_length < 1 or
            type(max_new_tokens) is not int or max_new_tokens < 1 or
            type(draft_tokens) is not int or not 1 <= draft_tokens <= 64):
        raise ValueError("Use positive public lengths and1..64 draft tokens")
    buckets = tuple(buckets)
    if not buckets or any(type(b) is not int or b < 1 for b in buckets) or tuple(sorted(set(buckets))) != buckets:
        raise ValueError("Buckets must be unique, positive and increasing")
    if max_new_tokens == 1:
        return {"prefill": (), "refill": (), "recurrent": ()}
    final_prefix = prompt_length+max_new_tokens-2
    bucket_for_end(final_prefix+draft_tokens-1, buckets)
    prefill = ((prompt_length, bucket_for_end(prompt_length, buckets)),)
    recurrent = {(draft_tokens, bucket_for_end(p+draft_tokens-1, buckets))
                 for p in range(prompt_length, final_prefix+1)}
    refill = set()
    for count in range(1, draft_tokens+2):
        # The prior proposal had stable prefix>=prompt_length. The next call
        # must leave at least one output token after the current anchor.
        for new_prefix in range(prompt_length+count, final_prefix+1):
            refill.add((count, bucket_for_end(new_prefix, buckets)))
    return {"prefill": prefill, "refill": tuple(sorted(refill)),
            "recurrent": tuple(sorted(recurrent))}


def route_proposal(old_prefix, prefix, draft_tokens, plan, buckets, prepared):
    """Host-only prewrite routing/shape/budget oracle, also used at runtime."""
    length, budget, width = plan
    if draft_tokens != width or budget < 2 or not length <= prefix <= length+budget-2:
        raise ValueError("Proposal lies outside the prepared public request")
    if old_prefix is None:
        if prefix != length:
            raise ValueError("Fresh request must start at its prepared prompt length")
        kind = "prefill"; count = prefix
    else:
        if prefix < old_prefix:
            raise ValueError("Committed prefix moved backwards; reset before a new request")
        count = prefix-old_prefix
        if count > width+1:
            raise ValueError("Master committed more inputs than this draft family permits")
        kind = "reuse" if count == 0 else "refill"
    fill_key = None if kind == "reuse" else (count, bucket_for_end(prefix, buckets))
    recur_key = (draft_tokens, bucket_for_end(prefix+draft_tokens-1, buckets))
    if kind != "reuse" and fill_key not in prepared[kind]:
        raise ValueError("Unprepared auxiliary "+kind+" family: "+str(fill_key))
    if recur_key not in prepared["recurrent"]:
        raise ValueError("Unprepared auxiliary recurrent family: "+str(recur_key))
    return {"kind":kind, "count":count, "fill_key":fill_key, "recur_key":recur_key,
            "old_prefix":0 if old_prefix is None else old_prefix,
            "prefix":prefix, "physical_recurrent_end":prefix+draft_tokens-1}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_state(model):
    parameters = {name:(id(p), p.data_ptr(), p._version, tuple(p.shape), tuple(p.stride()),
                        str(p.dtype), str(p.device), p.requires_grad)
                  for name,p in model.named_parameters(remove_duplicate=False)}
    modules = {}
    for name,module in model.named_modules(remove_duplicate=False):
        forward = module.forward
        modules[name] = (id(module), type(module), getattr(forward,"__func__",forward),
                         module.__dict__.get("forward"), module.training)
    return parameters,modules


@dataclass
class _FillEntry:
    rows: int
    bucket: int
    features: object
    tokens: object
    base: object
    offsets: object
    key_positions: object
    graph: object = None


@dataclass
class _RecurrentEntry:
    width: int
    bucket: int
    output: object
    key_positions: object
    graph: object = None


class FeatureDraftProposer3B:
    """One immutable auxiliary model/cache, with explicit request preparation."""

    def __init__(self, target, checkpoint_path, *, max_len=2048, cuda_graph=True,
                 share_pool=True, buckets=None, verify_prefix=True):
        if torch is None:
            raise RuntimeError("PyTorch is required for the auxiliary runtime")
        from transformers import __version__ as transformers_version
        if torch.__version__.split("+")[0] != "2.9.1" or transformers_version != "4.57.1":
            raise ValueError("Use the existing pinned Torch2.9.1 / Transformers4.57.1 runtime")
        if type(max_len) is not int or max_len < 2:
            raise ValueError("Use a full auxiliary cache capacity>=2")
        selected = tuple(b for b in DEFAULT_BUCKETS if b <= max_len) if buckets is None else tuple(buckets)
        if (not selected or selected != tuple(sorted(set(selected))) or
                any(type(b) is not int or not 1 <= b <= max_len for b in selected)):
            raise ValueError("Use fixed increasing attention buckets within full capacity")
        started = time.perf_counter()
        from exact10_validation import hash_model_parameters
        self.target = target
        self._target_snapshot = _target_state(target)
        self._source_hash = hash_model_parameters(target)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self._checkpoint_sha256 = sha256_file(self.checkpoint_path)
        saved = torch.load(self.checkpoint_path,map_location="cpu",weights_only=True,mmap=True)
        if (saved.get("schema") != "lightreason_aux_feature_checkpoint_v1" or
                saved.get("feature_schema") != FEATURE_SCHEMA or saved.get("auxiliary_only") is not True or
                saved.get("authoritative") is not False or
                saved.get("identity",{}).get("source_live_parameter_sha256") != self._source_hash):
            raise ValueError("Checkpoint is not an auxiliary trained against this unchanged target")
        model_source = Path(__file__).with_name("feature_draft3b.py")
        if saved["identity"].get("source_sha256",{}).get(model_source.name) != sha256_file(model_source):
            raise ValueError("Auxiliary model implementation differs from checkpoint training")
        mapping = saved["aux_to_target"]
        auxiliary = FeatureDraft3B.from_target(target,mapping)
        if asdict(auxiliary.config) != saved.get("model_config"):
            raise ValueError("Checkpoint target geometry/feature contract differs")
        auxiliary.load_trainable_state_dict(saved["trainable_state"])
        self.device = target.get_input_embeddings().weight.device
        if self.device.type != "cuda":
            raise ValueError("Load the authoritative target on its final CUDA device first")
        # Only this independent auxiliary receives device/dtype/mode changes.
        self.model = auxiliary.eval().requires_grad_(False).to(device=self.device,dtype=torch.bfloat16)
        self._auxiliary_snapshot = _target_state(self.model)
        self.max_len, self.buckets = max_len, selected
        self.cuda_graph, self.share_pool, self.verify_prefix = bool(cuda_graph),bool(share_pool),bool(verify_prefix)
        self._pool = torch.cuda.graph_pool_handle() if cuda_graph and share_pool else None
        self.prefill_entries, self.refill_entries, self.recurrent_entries = {},{},{}
        self.graphs = {}
        h,kv,d = self.model.config.hidden_size,self.model.config.num_key_value_heads,self.model.config.head_dim
        with torch.inference_mode(False),torch.no_grad():
            self.key_cache = torch.zeros((1,kv,max_len,d),device=self.device,dtype=torch.bfloat16)
            self.value_cache = torch.zeros_like(self.key_cache)
            self.stable_state = torch.empty((1,1,h),device=self.device,dtype=torch.bfloat16)
            self.stable_next_token = torch.empty((1,1),device=self.device,dtype=torch.long)
            self.state = torch.empty_like(self.stable_state)
            self.token = torch.empty_like(self.stable_next_token)
            self.position = torch.zeros((1,),device=self.device,dtype=torch.long)
            self.verified_visible_ids = torch.empty((1,max_len+1),device=self.device,dtype=torch.long)
        self._cache_identity = self._cache_state()
        self.stable_prefix_length, self.request_plan = None,None
        self.last_route = None
        self.last_output = None
        self.prefill_calls = self.refill_calls = self.reused_prefix_calls = 0
        self.metadata = {"role":"auxiliary_feature_proposer_never_authoritative",
            "target_model_replaced_or_changed":False,"checkpoint_sha256":self._checkpoint_sha256,
            "checkpoint_updates":saved.get("updates"),"checkpoint_training_identity":saved.get("identity"),
            "target_revision":TARGET_REVISION,"target_live_parameter_sha256":self._source_hash,
            "target_taps":list(TARGET_TAPS),"model":self.model.metadata(),
            "runtime":{"torch":torch.__version__,"transformers":transformers_version},
            "auxiliary_dtype":"torch.bfloat16","source_dtype":"torch.bfloat16",
            "auxiliary_parameters_frozen":True,
            "direct_aux_to_target_mapping":True,"full_static_cache_capacity":max_len,
            "physical_rowj_semantic_token_position":"j+1",
            "verified_feature_refill_replaces_speculative_KV":True,
            "initial_auxiliary_full_prefix_prefill_is_timed":True,
            "CUDA_graphs":self.cuda_graph,"shared_graph_pool":self.share_pool,
            "prefix_ID_validation_in_propose":self.verify_prefix,
            "auxiliary_EOS_authority":False,"final_token_commit_authority":False,
            "setup_s":0.0,"graph_setup_s":0.0,"graph_count":0,
            "quality_acceptance_or_speed_verified":False,
            "implementation_sha256":{p.name:sha256_file(p) for p in
                (Path(__file__),Path(__file__).with_name("feature_draft3b.py"))}}
        self.assert_target_unchanged(verify_bytes=True,verify_checkpoint=True)
        self.metadata["setup_s"] = time.perf_counter()-started

    def _cache_state(self):
        return tuple((id(value),value.data_ptr(),tuple(value.shape),tuple(value.stride()),
                      str(value.dtype),str(value.device)) for value in (self.key_cache,self.value_cache))

    def assert_storage_consistent(self):
        if self._cache_state() != self._cache_identity:
            raise RuntimeError("Auxiliary full backing KV storage/pointers changed")
        return {"auxiliary_full_KV_storage_identity_unchanged":True}

    def assert_target_unchanged(self, *, verify_bytes=False, verify_checkpoint=False):
        """Untimed source/checkpoint audit; never inside capture or latency work."""
        if _target_state(self.target) != self._target_snapshot:
            raise RuntimeError("Authoritative target modules/forwards/parameter state changed")
        if _target_state(self.model) != self._auxiliary_snapshot:
            raise RuntimeError("Frozen auxiliary module/parameter/version identity changed")
        if verify_bytes:
            from exact10_validation import hash_model_parameters
            if hash_model_parameters(self.target) != self._source_hash:
                raise RuntimeError("Authoritative target live BF16 bytes changed")
        if verify_checkpoint and sha256_file(self.checkpoint_path) != self._checkpoint_sha256:
            raise RuntimeError("Auxiliary checkpoint bytes changed")
        return {"target_module_parameter_and_version_identity_unchanged":True,
                "target_live_bytes_verified":bool(verify_bytes),
                "checkpoint_bytes_verified":bool(verify_checkpoint)}

    def reset_for_new_request(self):
        """Logical reset only; prepared graphs and physical cache remain allocated."""
        self.stable_prefix_length, self.request_plan = None,None
        self.last_route = self.last_output = None
        return {"logical_auxiliary_prefix_reset":True,"KV_bytes_cleared":False}

    def _allocate_fill(self, rows, bucket):
        with torch.inference_mode(False),torch.no_grad():
            return _FillEntry(rows,bucket,
                torch.zeros((1,rows,6144),device=self.device,dtype=torch.bfloat16),
                torch.zeros((1,rows),device=self.device,dtype=torch.long),
                torch.tensor([bucket-rows],device=self.device,dtype=torch.long),
                torch.arange(rows,device=self.device,dtype=torch.long),
                torch.arange(bucket,device=self.device,dtype=torch.long))

    def _static_attention(self, q, k, v, physical_positions, key_positions):
        self.key_cache.index_copy_(2,physical_positions,k)
        self.value_cache.index_copy_(2,physical_positions,v)
        bucket = key_positions.shape[0]
        mask = (key_positions[None,:] <= physical_positions[:,None])[None,None]
        groups = self.model.config.num_attention_heads // self.model.config.num_key_value_heads
        keys = self.key_cache[:,:,:bucket].repeat_interleave(groups,dim=1)
        values = self.value_cache[:,:,:bucket].repeat_interleave(groups,dim=1)
        return F.scaled_dot_product_attention(q,keys,values,attn_mask=mask,
                                             dropout_p=0.0,is_causal=False)

    def _run_fill(self, entry):
        physical = entry.base+entry.offsets
        semantic = (physical+1).view(1,-1)
        state = self.model.project_target_features(entry.features)
        q,k,v = self.model._qkv(state,entry.tokens,semantic)
        attended = self._static_attention(q,k,v,physical,entry.key_positions)
        state = self.model._finish(state,attended)
        last = state[:,-1:]
        logits = self.model.compute_logits(last)
        self.stable_state.copy_(last)
        self.stable_next_token.copy_(self.model.aux_to_target[logits.argmax(-1)])

    def _run_recurrent(self, entry):
        self.state.copy_(self.stable_state)
        self.token.copy_(self.stable_next_token)
        entry.output[:,:1].copy_(self.token)
        for index in range(entry.width-1):
            semantic = (self.position+1).view(1,1)
            q,k,v = self.model._qkv(self.state,self.token,semantic)
            attended = self._static_attention(q,k,v,self.position,entry.key_positions)
            state = self.model._finish(self.state,attended)
            logits = self.model.compute_logits(state)
            self.state.copy_(state)
            self.token.copy_(self.model.aux_to_target[logits.argmax(-1)])
            entry.output[:,index+1:index+2].copy_(self.token)
            self.position.add_(1)
        return entry.output

    def _capture(self, fn, reset=None):
        if reset is not None:
            reset()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream),torch.no_grad():
            fn()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        if reset is not None:
            reset()
            torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=self._pool),torch.no_grad():
            fn()
        return graph

    def prepare(self, prompt_length, max_new_tokens, B):
        if self.stable_prefix_length is not None:
            raise RuntimeError("Reset the auxiliary logical request before preparation")
        families = public_families(prompt_length,max_new_tokens,B,self.buckets)
        if prompt_length > self.max_len:
            raise ValueError("Prompt exceeds full auxiliary backing storage")
        began = time.perf_counter()
        with torch.inference_mode(False),torch.no_grad():
            for key in families["prefill"]:
                if key not in self.prefill_entries:
                    # Buffers may be prepared, but the real prefix forward still
                    # executes during the first timed propose() call.
                    self.prefill_entries[key] = self._allocate_fill(*key)
            for key in families["refill"]:
                if key not in self.refill_entries:
                    entry = self._allocate_fill(*key)
                    if self.cuda_graph:
                        entry.graph = self._capture(lambda entry=entry:self._run_fill(entry))
                        self.graphs[("refill",)+key] = entry.graph
                    self.refill_entries[key] = entry
            for key in families["recurrent"]:
                if key not in self.recurrent_entries:
                    width,bucket = key
                    entry = _RecurrentEntry(width,bucket,
                        torch.empty((1,width),device=self.device,dtype=torch.long),
                        torch.arange(bucket,device=self.device,dtype=torch.long))
                    if self.cuda_graph:
                        # Capture executes the loop once after its warmup; reset
                        # GPU position before each pass so no tail exceedsbucket.
                        representative = bucket-width+1
                        self.stable_state.zero_(); self.stable_next_token.zero_()
                        entry.graph = self._capture(
                            lambda entry=entry:self._run_recurrent(entry),
                            reset=lambda representative=representative:self.position.fill_(representative))
                        self.graphs[("recurrent",)+key] = entry.graph
                    self.recurrent_entries[key] = entry
        self.request_plan = (prompt_length,max_new_tokens,B)
        self.metadata["graph_setup_s"] += time.perf_counter()-began
        self.metadata["graph_count"] = len(self.graphs)
        self.metadata["prepared_refill_families"] = [list(key) for key in sorted(self.refill_entries)]
        self.metadata["prepared_recurrent_families"] = [list(key) for key in sorted(self.recurrent_entries)]
        self.assert_storage_consistent()
        return {"families":families,"setup_s":time.perf_counter()-began,
                "initial_prefix_forward_executed":False}

    def propose(self, visible_ids, committed_target_features, committed_prefix_length,
                num_draft_tokens, *, use_graphs=None):
        """Return reused GPU original-ID buffer[1,B]; never authorize/commit it.

        Clone the returned buffer if retaining it beyond the next same-family
        proposal. use_graphs=False is the same-operation eager control path.
        """
        if self.request_plan is None:
            raise RuntimeError("Call prepare() for the public request before propose()")
        p, B = committed_prefix_length,num_draft_tokens
        use_graphs = self.cuda_graph if use_graphs is None else bool(use_graphs)
        if type(p) is not int or type(B) is not int or not 1 <= p <= self.max_len:
            raise ValueError("Use positive CPU prefix/draft metadata within full capacity")
        if (visible_ids.shape != (1,p+1) or visible_ids.dtype != torch.long or visible_ids.device != self.device or
                committed_target_features.ndim != 3 or committed_target_features.shape[0] != 1 or
                committed_target_features.shape[1] < p or committed_target_features.shape[2] != 6144 or
                committed_target_features.dtype != torch.bfloat16 or committed_target_features.device != self.device):
            raise ValueError("Visible IDs/verified features violate the device/shape/position contract")
        prepared = {"prefill":self.prefill_entries,"refill":self.refill_entries,"recurrent":self.recurrent_entries}
        route = route_proposal(self.stable_prefix_length,p,B,self.request_plan,self.buckets,prepared)
        old = route["old_prefix"]
        if self.stable_prefix_length is not None and self.verify_prefix:
            if not torch.equal(visible_ids[:,:old+1],self.verified_visible_ids[:,:old+1]):
                raise ValueError("Committed token IDs changed; reset before rebuilding auxiliary history")
        with torch.no_grad():
            if route["kind"] != "reuse":
                entry = prepared[route["kind"]][route["fill_key"]]
                entry.features.copy_(committed_target_features[:,old:p])
                entry.tokens.copy_(visible_ids[:,old+1:p+1])
                entry.base.fill_(old)
                if entry.graph is None or not use_graphs:
                    self._run_fill(entry)
                else:
                    entry.graph.replay()
                self.verified_visible_ids[:,:p+1].copy_(visible_ids)
                if route["kind"] == "prefill": self.prefill_calls += 1
                else: self.refill_calls += 1
            else:
                self.reused_prefix_calls += 1
            self.stable_prefix_length = p
            recurrent = self.recurrent_entries[route["recur_key"]]
            self.position.fill_(p)
            if recurrent.graph is None or not use_graphs:
                output = self._run_recurrent(recurrent)
            else:
                recurrent.graph.replay(); output = recurrent.output
        self.last_route,self.last_output = route,output
        return output


def cpu_contract_tests():
    assert refill_alignment(0,4) == [(r,r,r+1) for r in range(4)]
    assert refill_alignment(4,7) == [(r,r,r+1) for r in range(4,7)]
    # Exhaustive public route/family correspondence across bucket boundaries.
    for length in (1,31,63,64,65,127,128,129):
        for budget in (1,2,3,8,19):
            for width in (1,3,7):
                buckets=(64,128,256)
                families=public_families(length,budget,width,buckets)
                if budget==1:
                    assert not any(families.values()); continue
                plan=(length,budget,width)
                for p in range(length,length+budget-1):
                    route=route_proposal(None if p==length else p-1,p,width,plan,buckets,families)
                    assert route["physical_recurrent_end"]==p+width-1
                    for count in range(1,min(width+1,p-length)+1):
                        route=route_proposal(p-count,p,width,plan,buckets,families)
                        assert route["fill_key"]==(count,bucket_for_end(p,buckets))
    # Simulate stable rows and poisoned hypotheses. Accepted IDs alone never
    # permit retaining KV based on predicted instead of verified features.
    tokens=list(range(30)); cache=[None]*32
    def true_refill(old,new):
        for physical,feature,semantic in refill_alignment(old,new):
            cache[physical]=("verified_feature",feature,tokens[semantic],semantic)
    true_refill(0,4); original=list(cache[:4])
    cache[4]=("hypothesis",999,tokens[5],5); cache[5]=("hypothesis",998,tokens[6],6)
    true_refill(4,6)
    assert cache[:4]==original
    assert cache[4]==("verified_feature",4,tokens[5],5)
    assert cache[5]==("verified_feature",5,tokens[6],6)
    true_refill(6,10)  # AllB proposals accepted plus target bonus: c=B+1.
    assert cache[9]==("verified_feature",9,tokens[10],10)
    prepared=public_families(4,12,3,(16,32))
    for invalid in ((7,6,3),(4,9,3),(None,5,3),(None,4,2)):
        try: route_proposal(*invalid,(4,12,3),(16,32),prepared)
        except ValueError: pass
        else: raise AssertionError("Invalid proposal route was accepted")
    missing={**prepared,"recurrent":()}
    try: route_proposal(None,4,3,(4,12,3),(16,32),missing)
    except ValueError: pass
    else: raise AssertionError("Unprepared recurrent family was accepted")
    missing={**prepared,"refill":()}
    try: route_proposal(4,5,3,(4,12,3),(16,32),missing)
    except ValueError: pass
    else: raise AssertionError("Unprepared refill family was accepted")
    fake=FeatureDraftProposer3B.__new__(FeatureDraftProposer3B)
    fake.stable_prefix_length=7; fake.request_plan=(4,12,3)
    fake.last_route={"old_prefix":4}; fake.last_output=[1,2,3]
    fake.key_cache=list(cache); fake.value_cache=list(cache)
    fake.refill_entries={"prepared":"unchanged"}; saved_keys=list(fake.key_cache)
    fake.reset_for_new_request()
    assert fake.stable_prefix_length is None and fake.request_plan is None and fake.last_route is None
    assert fake.key_cache==saved_keys and fake.refill_entries=={"prepared":"unchanged"}
    return {"CPU_public_bucket_routing_and_accepted_feature_refill_oracles_passed":True,
            "actual_torch_CUDA_or_graph_execution":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if not args.self_test:
        parser.error("This module is a runtime API; only --self-test is a standalone command")
    print(json.dumps(cpu_contract_tests(),indent=2))
