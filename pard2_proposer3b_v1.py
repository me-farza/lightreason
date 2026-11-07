"""Pinned token-only PARD-2 auxiliary; the original3B target stays authoritative.

This checkpoint's name describes its training teacher. Its token-only model is
Qwen3-0.6B geometry. No Qwen3 target, feature warp, tokenizer chat template or
upstream inference CLI is used. Every proposed token requires fresh original
Qwen2.5-3B full-vocabulary verification by the separately selected tree runner.

Pinned semantics: AMD-AGI/PARD6f279bf3f1680e0b5d71c562ca5b91bdeef4c038,
pard/pard2_infer.py109-112,218-230: non-None p2 disables feature conditioning;
ordinary causal base-model logits over genuine history plus B-1 mask IDs are
proposals. Auxiliary FP-order differences do not authorize final outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from pard_proposer import PARDProposer
    from bucketed_pard_proposer import BucketedPARDLatticeProposer
    from bucketed3b_tree_runner import BucketedNumerical3BTreeRunner
except ModuleNotFoundError as exc:
    if exc.name not in ("torch", "transformers"):
        raise
    torch = AutoConfig = AutoModelForCausalLM = None
    PARDProposer = BucketedPARDLatticeProposer = BucketedNumerical3BTreeRunner = object


REPO = "amd/PARD2-Qwen3-8B"
REVISION = "67a1516c8f6fc145cda99916799a0cbb3a4af135"
UPSTREAM_COMMIT = "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
VOCAB_SIZE, MASK_ID = 151936, 151670
FILES = {
    "config.json": (975, "ad6a223718a3c5df8a65fa51a24b66ec07f2ed63bd7ceb51c326ac8f42994259"),
    "model.safetensors": (1503300456, "bf1a6c0e69db76420f1f58381b03263711e8f1ae5938b955a0ce49b7aa60502e"),
    "aux-tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
}
ALLOWED_EXTRAS = {"<tool_response>":151665, "</tool_response>":151666,
                  "<think>":151667, "</think>":151668}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_files(directory):
    directory = Path(directory)
    result = {}
    require({p.name for p in directory.glob("*.safetensors")} == {"model.safetensors"},
            "Require only the pinned ordinary auxiliary safetensors file")
    for name, (size, expected) in FILES.items():
        path = directory/name
        require(path.is_file() and path.stat().st_size == size, "Pinned auxiliary file missing or wrong size: "+name)
        observed = file_sha(path)
        require(observed == expected, "Pinned auxiliary file SHA mismatch: "+name)
        result[name] = {"bytes":size, "sha256":observed}
    return result


def tokenizer_snapshot(tokenizer):
    values = {"vocab": sorted(tokenizer.get_vocab().items()),
        "special": {name:getattr(tokenizer,name,None) for name in
                    ("bos_token_id", "eos_token_id", "pad_token_id", "chat_template")}}
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_token_ids(target_tokenizer, auxiliary_json):
    """Direct ID compatibility, including all target controls; no retokenizing."""
    ordinary = auxiliary_json["model"]["vocab"]
    require(len(ordinary) == 151643 and len(set(ordinary.values())) == 151643 and
            set(ordinary.values()) == set(range(151643)), "Pinned auxiliary ordinary vocabulary malformed")
    auxiliary = dict(ordinary)
    for row in auxiliary_json["added_tokens"]:
        token, value = row["content"], row["id"]
        require(token not in auxiliary or auxiliary[token] == value, "Conflicting auxiliary token IDs")
        auxiliary[token] = value
    target = target_tokenizer.get_vocab()
    require(len(target) == 151665 and all(type(value) is int and 0 <= value < VOCAB_SIZE for value in target.values()),
            "Use the original target tokenizer with151643 ordinary and22 added tokens")
    require(all(auxiliary.get(token) == value for token,value in target.items()),
            "An original target token differs from the pinned auxiliary ID map")
    require({token:value for token,value in auxiliary.items() if token not in target} == ALLOWED_EXTRAS,
            "Uninspected auxiliary-only token IDs")
    require(target.get("<|endoftext|>") == 151643 and target.get("<|im_start|>") == 151644 and
            target.get("<|im_end|>") == 151645 and target_tokenizer.eos_token_id == 151645 and
            target_tokenizer.pad_token_id == 151643 and target_tokenizer.bos_token_id is None,
            "Original target chat/EOS/padding IDs differ")
    return {"all_target_token_IDs_identical":True, "target_mapping_entries":len(target),
        "all_151643_ordinary_IDs_identical":True, "all_22_original_added_IDs_identical":True,
        "auxiliary_only_registered_tokens":ALLOWED_EXTRAS, "mask_id":MASK_ID,
        "mask_registered_in_target":MASK_ID in target.values(),
        "target_tokenizer_modified":False, "auxiliary_tokenizer_used_for_encoding":False,
        "original_target_chat_template_used":True}


def matrix_inventory(path):
    """Small local safetensors-header reader; never deserializes executable code."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Missing safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        require(0 < size <= 2*1024*1024, "Implausible auxiliary header")
        raw = stream.read(size)
        require(len(raw) == size, "Truncated auxiliary header")
        header = json.loads(raw)
    result = {}
    for name, row in header.items():
        if name == "__metadata__":continue
        begin,end = row["data_offsets"]
        require(0 <= begin <= end and 8+size+end <= path.stat().st_size, "Auxiliary tensor offsets invalid")
        result[name] = {"path":path, "offset":8+size+begin, "bytes":end-begin,
                        "dtype":row["dtype"], "shape":row["shape"]}
    return result


def compare_stored_embedding_head(inventory):
    embed, head = inventory.get("model.embed_tokens.weight"), inventory.get("lm_head.weight")
    require(embed is not None and head is not None, "Pinned auxiliary stores both embedding and head matrices")
    require(all(embed[key] == head[key] for key in ("dtype","shape","bytes")) and
            embed["dtype"] == "BF16", "Incompatible original auxiliary embedding/head geometry")
    hashes = [hashlib.sha256(),hashlib.sha256()]; equal = True
    with embed["path"].open("rb") as left, head["path"].open("rb") as right:
        left.seek(embed["offset"]); right.seek(head["offset"]); remaining = embed["bytes"]
        while remaining:
            size = min(4*1024*1024,remaining); a,b = left.read(size),right.read(size)
            require(len(a) == len(b) == size, "Truncated auxiliary matrix bytes")
            hashes[0].update(a);hashes[1].update(b);equal = equal and a == b;remaining -= size
    return {"config_ties_embeddings":True, "both_original_matrices_stored":True,
        "embedding_head_storage_bits_equal":equal, "compared_bytes_per_matrix":embed["bytes"],
        "embedding_sha256":hashes[0].hexdigest(),"head_sha256":hashes[1].hexdigest(),
        "effective_tie_word_embeddings":equal,
        "distinct_matrices_preserved_by_auxiliary_tieFalse":not equal}


def model_state(model):
    parameters = {name:(id(p),p.data_ptr(),p._version,tuple(p.shape),tuple(p.stride()),str(p.dtype),str(p.device),p.requires_grad)
                  for name,p in model.named_parameters(remove_duplicate=False)}
    modules = {name:(id(m),type(m),getattr(m.forward,"__func__",m.forward),m.__dict__.get("forward"),m.training)
               for name,m in model.named_modules(remove_duplicate=False)}
    return parameters,modules


def tensor_sha(value):
    """Untimed original stored-value audit, with bounded GPU-to-CPU chunks."""
    digest=hashlib.sha256();flat=value.detach().reshape(-1)
    for first in range(0,flat.numel(),1024*1024):
        digest.update(flat[first:first+1024*1024].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class PARD2Proposer(PARDProposer):
    """New loader, frozen original causal proposal/cache methods inherited."""
    def __init__(self, model_path, *, target_tokenizer, device="cuda", target_vocab_size=VOCAB_SIZE,
                 cache_mode="incremental", max_cache_len=None, attn_implementation="sdpa"):
        if torch is None:raise RuntimeError("Use the pinned remote PyTorch runtime")
        import transformers
        require(torch.__version__.split("+")[0] == "2.9.1" and transformers.__version__ == "4.57.1",
                "Use pinned Torch2.9.1/Transformers4.57.1")
        require(cache_mode in ("incremental","stateless") and attn_implementation in ("sdpa","eager"),
                "Choose supported auxiliary cache/attention settings")
        require(target_vocab_size == VOCAB_SIZE, "Original target full vocabulary must remain151936")
        directory = Path(model_path).expanduser().resolve()
        files = verify_local_files(directory)
        target_before = tokenizer_snapshot(target_tokenizer)
        compatibility = validate_token_ids(target_tokenizer,json.loads((directory/"aux-tokenizer.json").read_text()))
        config = AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        require((config.model_type,config.hidden_size,config.num_hidden_layers,config.num_attention_heads,
                 config.num_key_value_heads,config.head_dim,config.intermediate_size,config.vocab_size) ==
                ("qwen3",1024,28,16,8,128,3072,VOCAB_SIZE) and getattr(config,"pard2",False) is True and
                config.spd_type == "pard2" and config.pard_token == MASK_ID and config.rope_theta == 1e6 and
                config.tie_word_embeddings is True and not getattr(config,"use_sliding_window",False),
                "Use the exact pinned small PARD2 Qwen3 auxiliary config")
        inventory = matrix_inventory(directory/"model.safetensors")
        require(all(row["dtype"] == "BF16" for row in inventory.values()), "Pinned auxiliary matrices must all be BF16")
        tying = compare_stored_embedding_head(inventory)
        if not tying["embedding_head_storage_bits_equal"]:
            config.tie_word_embeddings = False  # AUXILIARY ONLY: retain both actual stored matrices.
        with torch.inference_mode(False),torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(directory,config=config,local_files_only=True,
                trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation=attn_implementation).eval().requires_grad_(False).to(device)
        embedding,head = self.model.get_input_embeddings().weight,self.model.lm_head.weight
        require(tuple(embedding.shape) == tuple(head.shape) == (VOCAB_SIZE,1024) and
                all(p.dtype == torch.bfloat16 and not p.requires_grad for p in self.model.parameters()),
                "Auxiliary loaded parameter geometry/dtype differs")
        require((embedding is head) == tying["effective_tie_word_embeddings"], "Auxiliary loader tying disagrees with raw matrices")
        loaded_embedding_sha,loaded_head_sha=tensor_sha(embedding),tensor_sha(head)
        require(loaded_embedding_sha == tying["embedding_sha256"] and loaded_head_sha == tying["head_sha256"],
                "HF auxiliary loader did not preserve both original stored matrices")
        tying.update(loaded_original_embedding_bytes_exact=True,loaded_original_head_bytes_exact=True)
        require(tokenizer_snapshot(target_tokenizer) == target_before, "Original target tokenizer changed during auxiliary loading")
        self.mask_id,self.target_vocab_size = MASK_ID,VOCAB_SIZE
        self.device,self.cache_mode = embedding.device,cache_mode
        limit = int(config.max_position_embeddings)
        self.max_cache_len = limit if max_cache_len is None else min(int(max_cache_len),limit)
        require(self.max_cache_len >= 2, "Auxiliary cache too small")
        self.parameter_versions = [(p,p._version) for p in self.model.parameters()]
        self._model_snapshot = model_state(self.model)
        self._target_tokenizer,self._target_tokenizer_snapshot = target_tokenizer,target_before
        self._directory,self._verified_files = directory,files
        self.metadata = {"role":"Auxiliary token proposals only; original target verifies all final tokens",
            "variant":"pinned_PARD2_Qwen3_06B_token_only_for_original_Qwen25_3B",
            "repo_id":REPO,"revision":REVISION,"upstream_source_commit":UPSTREAM_COMMIT,
            "model_path":str(directory),"mask_id":MASK_ID,"draft_attention":attn_implementation,
            "dtype":"torch.bfloat16","cache_mode":cache_mode,"max_cache_len":self.max_cache_len,
            "parameter_count":sum(p.numel() for p in self.model.parameters()),
            "checkpoint_stored_elements":sum(__import__('math').prod(r["shape"]) for r in inventory.values()),
            "checkpoint_files":files,"checkpoint_tying":tying,"tokenizer_compatibility":compatibility,
            "draft_architecture":{"model_type":"qwen3","hidden_size":1024,"layers":28,
                "attention_heads":16,"kv_heads":8,"head_dim":128,"q_and_k_norm":True},
            "target_feature_conditioning":False,"warp_model_loaded":False,
            "target_weights_or_template_changed":False,"generation_config_bypassed":True,
            "raw_draft_argmax":True,"raw_draft_logits_full_vocabulary":VOCAB_SIZE,
            "config_sha256":files["config.json"]["sha256"],
            "implementation_sha256":file_sha(__file__),"performance_and_quality_unestablished":True}
        self.last_report,self.last_confidence = None,None
        self.reset()

    def audit_integrity(self, *, verify_bytes=True, verify_files=True):
        """Untimed full audit; lightweight inherited version guards run online."""
        self.assert_draft_unchanged()
        require(model_state(self.model) == self._model_snapshot, "Auxiliary parameters/modules/functions/modes changed")
        require(tokenizer_snapshot(self._target_tokenizer) == self._target_tokenizer_snapshot,
                "Original target tokenizer changed")
        if verify_files:require(verify_local_files(self._directory) == self._verified_files, "Pinned auxiliary files changed")
        result = {"auxiliary_parameter_module_versions_unchanged":True,"original_target_tokenizer_unchanged":True}
        if verify_bytes:
            from exact10_validation import hash_model_parameters
            result["auxiliary_live_parameter_sha256"] = hash_model_parameters(self.model)
        return result


class BucketedPARD2LatticeProposer(BucketedPARDLatticeProposer):
    def __init__(self, checked_base, **kwargs):
        require(isinstance(checked_base,PARD2Proposer), "Pass the explicitly checked pinned PARD2 auxiliary loader")
        super().__init__(checked_base,**kwargs)
        self.metadata = {**self.metadata,"auxiliary_implementation":"PARD2_token_only_full_storage_bucketed_lattice",
            "Qwen3_head_dim_and_KV_geometry_from_original_auxiliary_config":True,
            "inherited_genuine_prefix_mask_visibility_controller_unchanged":True,
            "adapter_sha256":file_sha(__file__)}


class PARD2Numerical3BTreeRunner(BucketedNumerical3BTreeRunner):
    """Reuse the frozen tree run/accept/commit and bucket prepare controllers."""
    def __init__(self,model,tokenizer,draft_path,*,max_len=2048,node_budget=64,draft_tokens=15,top_k=8,
                 compile_static=False,compile_draft=False,cuda_graph=True,share_pool=True,warmups=3,
                 validate_topology=True,commit_engine="triton",tree_policy="best-first",
                 attention_buckets=(256,512,1024,2048)):
        from bucketed3b_tree_verifier import BucketedNumericalTreeVerifier
        require(type(node_budget) is int and 2 <= node_budget <= 128 and type(draft_tokens) is int and
                1 <= draft_tokens <= 32 and tree_policy in ("best-first","trunk"), "Use valid public tree sizes/policy")
        self.model,self.tokenizer,self.max_len = model,tokenizer,max_len
        self.node_budget,self.draft_tokens,self.top_k = node_budget,draft_tokens,top_k
        self.tree_policy = tree_policy
        buckets = tuple(sorted({int(v) for v in attention_buckets if 1 <= int(v) <= max_len} | {max_len}))
        self.verifier = BucketedNumericalTreeVerifier(model,max_len=max_len,node_budget=node_budget,buckets=buckets,
            cuda_graph=cuda_graph,compile_static=compile_static,share_pool=share_pool,warmups=warmups,
            validate_topology=validate_topology,commit_engine=commit_engine)
        base = PARD2Proposer(draft_path,target_tokenizer=tokenizer,device=model.device,
            cache_mode="incremental",max_cache_len=max_len,attn_implementation="sdpa")
        self.proposer = BucketedPARD2LatticeProposer(base,max_len=max_len,attention_buckets=buckets,
            top_k=top_k,compile_model=compile_draft,share_pool=share_pool,warmups=warmups)
        with torch.inference_mode(False),torch.no_grad():
            self.visible = torch.empty((1,max_len),device=model.device,dtype=torch.long)
        self.setup_s,self.prepared = 0.0,False
        self.metadata = {"variant":"original_BF16_target_bucketed_tree_PARD2_token_only_auxiliary",
            "strict_token_matching_required":False,"exact_token_agreement_must_be_reported":True,
            "target_weights_and_equations_unchanged":True,"target_verifier":self.verifier.metadata,
            "proposer":self.proposer.metadata,"node_budget":node_budget,"draft_tokens":draft_tokens,
            "top_k":top_k,"tree_policy":tree_policy,"compile_static":compile_static,"compile_draft":compile_draft,
            "max_len":max_len,"attention_buckets":list(buckets),"target_controller_inherited_unchanged":True,
            "all_online_prefill_proposal_cache_tree_verification_and_acceptance_costs_timed":True,
            "generation":"Original target raw greedy/full vocabulary/template/nativeEOS; auxiliary never final authority",
            "preparation_uses_only_public_lengths":True,"runtime_graph_capture_permitted":False,
            "runner_source_sha256":file_sha(__file__)}


def make_quality_runner(model,tokenizer,args):
    from numerical3b_runner import NumericalGenerationAdapter
    names=("node_budget","draft_tokens","top_k","compile_static","compile_draft","cuda_graph","share_pool",
           "warmups","validate_topology","commit_engine","tree_policy","attention_buckets")
    runner=PARD2Numerical3BTreeRunner(model,tokenizer,args.draft_path,max_len=args.max_len,
        **{name:getattr(args,name) for name in names if hasattr(args,name)})
    configured=model.generation_config.eos_token_id
    eos=[configured] if isinstance(configured,int) else list(configured or [])
    if tokenizer.eos_token_id is not None:eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner,tokenizer,sorted(set(eos)))


def self_test():
    # Pure metadata/mapping tests; no target, auxiliary weights or GPU needed.
    vocab={"t"+str(i):i for i in range(151643)}
    controls={"<|endoftext|>":151643,"<|im_start|>":151644,"<|im_end|>":151645}
    controls.update({"control"+str(i):i for i in range(151646,151665)})
    target_vocab=vocab|controls
    class Tokenizer:
        bos_token_id=None;eos_token_id=151645;pad_token_id=151643;chat_template="original"
        def get_vocab(self):return dict(target_vocab)
    aux={"model":{"vocab":vocab},"added_tokens":[{"content":k,"id":v} for k,v in (controls|ALLOWED_EXTRAS).items()]}
    assert validate_token_ids(Tokenizer(),aux)["all_target_token_IDs_identical"]
    changed=json.loads(json.dumps(aux));changed["added_tokens"][-1]["id"]=151669
    try:validate_token_ids(Tokenizer(),changed)
    except ValueError:pass
    else:raise AssertionError("Uninspected extra control accepted")
    changed=json.loads(json.dumps(aux));changed["model"]["vocab"]["t0"]=1
    try:validate_token_ids(Tokenizer(),changed)
    except ValueError:pass
    else:raise AssertionError("Changed ordinary mapping accepted")
    return {"CPU_exact_ID_mapping_and_uninspected_extra_mutation_tests_passed":True,
            "GPU_work_performed":False}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if not args.self_test:parser.error("Use the explicit probe or benchmark adapter; this loader never generates final answers")
    print(json.dumps(self_test()))
