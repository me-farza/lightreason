"""Pretrained AMD PARD proposals; the unchanged target must verify every token.

API:
    draft = PARDProposer(local_path, target_tokenizer=target_tokenizer)
    proposed_ids = draft.propose(visible_ids, num_draft_tokens=7)  # [1,7]

visible_ids contains only the public prompt and already committed target IDs.
No target model, reference answer, future token, or EOS distance is accepted.
This module never commits output. Draft EOS/mask predictions are proposals too;
the authoritative target controller must apply its own native EOS rule.

AMD's PARD-1 uses the ordinary causal Qwen2 forward. Given visible history H,
append B-1 integer config.pard_token placeholders and take the last B logits.
The last true-history position predicts the first draft token; the following
mask positions predict the rest in parallel. No learned head is added here.
For this checkpoint pard_token=151665 is an embedding-row ID, not a registered
tokenizer token. Do not add tokens or resize either model's embeddings.

Incremental mode reuses only genuine visible-prefix K/V. Every mask K/V row
is cropped immediately after proposing, and prefix rollback crops to the last
common visible prefix. Stateless mode is available for acceptance probes.
Proposal quality and latency on the actual GPU remain empirical questions.

Semantic reference, inspect independently before changing this adapter:
https://github.com/AMD-AGI/PARD/blob/master/pard/pard_infer.py
https://huggingface.co/amd/PARD-Qwen2.5-0.5B
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _tensor_inventory(directory):
    """Read local safetensors headers only; never deserialize executable code."""
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Invalid safetensors header: {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= 128 * 1024 ** 2:
                raise ValueError(f"Implausible safetensors header size: {path.name}")
            header = json.loads(handle.read(header_size))
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Duplicate checkpoint tensor key: {key}")
            begin, end = entry["data_offsets"]
            if not 0 <= begin <= end or 8 + header_size + end > path.stat().st_size:
                raise ValueError(f"Invalid tensor offsets: {key}")
            tensors[key] = {"path": path, "offset": 8 + header_size + begin,
                            "bytes": end - begin, "dtype": entry["dtype"], "shape": entry["shape"]}
    if not tensors:
        raise ValueError("Require the locally downloaded safetensors checkpoint")
    return tensors


def _check_tied_checkpoint(directory, config):
    """Prevent HF tying from silently discarding distinct stored draft weights."""
    if not getattr(config, "tie_word_embeddings", False):
        return {"config_ties_embeddings": False, "raw_duplicate_check_needed": False}
    inventory = _tensor_inventory(directory)
    embed = inventory.get("model.embed_tokens.weight")
    head = inventory.get("lm_head.weight")
    if embed is None:
        raise ValueError("The draft checkpoint has no original token-embedding tensor")
    if head is None:
        return {"config_ties_embeddings": True, "raw_duplicate_check_needed": False,
                "stored_head_is_deduplicated": True}
    if any(embed[name] != head[name] for name in ("bytes", "dtype", "shape")):
        raise ValueError("The checkpoint declares tied embeddings but stores incompatible embedding/head tensors")
    hashes = (hashlib.sha256(), hashlib.sha256())
    equal = True
    with embed["path"].open("rb") as left, head["path"].open("rb") as right:
        left.seek(embed["offset"])
        right.seek(head["offset"])
        remaining = embed["bytes"]
        while remaining:
            size = min(4 * 1024 ** 2, remaining)
            a, b = left.read(size), right.read(size)
            if len(a) != size or len(b) != size:
                raise ValueError("Truncated draft embedding/head checkpoint data")
            equal = equal and a == b
            hashes[0].update(a)
            hashes[1].update(b)
            remaining -= size
    record = {"config_ties_embeddings": True, "raw_duplicate_check_needed": True,
              "embedding_head_storage_bits_equal": equal, "compared_bytes_per_matrix": embed["bytes"],
              "embedding_sha256": hashes[0].hexdigest(), "head_sha256": hashes[1].hexdigest()}
    if not equal:
        raise ValueError("Draft checkpoint stores distinct embedding/head values despite tie_word_embeddings=True; "
                         "refusing to let HF silently discard either pretrained matrix. " + json.dumps(record))
    return record


def check_tokenizer_ids(target_tokenizer, draft_tokenizer, target_vocab_size, draft_vocab_size, mask_id):
    """Require exact target token->ID correspondence; never retokenize inputs."""
    if target_vocab_size != draft_vocab_size:
        raise ValueError(f"Target/draft classifier widths differ: {target_vocab_size}/{draft_vocab_size}")
    target = target_tokenizer.get_vocab()
    draft = draft_tokenizer.get_vocab()
    mismatches = [(token, value, draft.get(token)) for token, value in target.items()
                  if draft.get(token) != value]
    if mismatches:
        raise ValueError(f"Target/draft token-ID mappings differ ({len(mismatches)} entries); first: {mismatches[:8]}")
    if any(type(value) is not int or not 0 <= value < target_vocab_size for value in target.values()):
        raise ValueError("Target tokenizer emits IDs outside the original target classifier")
    extras = {token: value for token, value in draft.items() if token not in target}
    if any(value != mask_id for value in extras.values()):
        raise ValueError("Unexpected draft-only token IDs: " + str(list(extras.items())[:8]))
    # Padding metadata may differ while all vocabulary IDs match. This adapter
    # accepts one unpadded target-tokenized sequence and never inserts a pad.
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(target_tokenizer, name, None) != getattr(draft_tokenizer, name, None):
            raise ValueError(f"Target/draft tokenizer special ID differs: {name}")
    canonical = json.dumps(sorted(target.items()), ensure_ascii=False, separators=(",", ":")).encode()
    return {"target_token_mapping_entries": len(target), "target_mapping_sha256": hashlib.sha256(canonical).hexdigest(),
            "all_target_token_ids_identical": True, "classifier_vocab_size": target_vocab_size,
            "draft_extra_tokens": extras, "mask_id": mask_id,
            "mask_registered_by_tokenizer": mask_id in set(draft.values()),
            "target_pad_token_id": target_tokenizer.pad_token_id,
            "draft_pad_token_id": draft_tokenizer.pad_token_id,
            "input_padding_used": False,
            "draft_tokenizer_used_to_encode_inputs": False,
            "target_tokenizer_modified": False}


class PARDProposer:
    """Local frozen PARD-1 drafter. Returns proposals, never verified output."""
    def __init__(self, model_path, *, target_tokenizer, device="cuda",
                 target_vocab_size=151936, cache_mode="incremental", max_cache_len=None,
                 attn_implementation="eager"):
        if cache_mode not in ("incremental", "stateless"):
            raise ValueError("Choose cache_mode='incremental' or 'stateless'")
        if attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Choose auxiliary attn_implementation='eager' or 'sdpa'")
        directory = Path(model_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Local PARD checkpoint directory does not exist: {directory}")
        config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        if config.model_type != "qwen2" or getattr(config, "spd_type", None) != "pard":
            raise ValueError("Use the pretrained Qwen2 PARD-1 checkpoint, not an ordinary AR model")
        mask_id = getattr(config, "pard_token", None)
        if type(mask_id) is not int or not 0 <= mask_id < config.vocab_size:
            raise ValueError("PARD config must provide a valid integer pard_token embedding ID")
        checkpoint_check = _check_tied_checkpoint(directory, config)
        draft_tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        compatibility = check_tokenizer_ids(target_tokenizer, draft_tokenizer, target_vocab_size,
                                             config.vocab_size, mask_id)
        with torch.inference_mode(False), torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(directory, config=config, local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation).eval().requires_grad_(False).to(device)
        if self.model.get_input_embeddings().weight.shape[0] != config.vocab_size or self.model.lm_head.weight.shape[0] != config.vocab_size:
            raise ValueError("Draft embedding/output dimensions differ from checkpoint vocabulary")
        if any(parameter.dtype != torch.bfloat16 or parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Draft parameters must remain frozen BF16")
        self.mask_id, self.target_vocab_size = mask_id, target_vocab_size
        self.device = self.model.get_input_embeddings().weight.device
        self.cache_mode = cache_mode
        limit = int(config.max_position_embeddings)
        self.max_cache_len = limit if max_cache_len is None else min(int(max_cache_len), limit)
        if self.max_cache_len < 2:
            raise ValueError("PARD cache capacity must be at least two positions")
        self.metadata = {"role": "auxiliary proposal source only; original target must verify every committed token",
            "model_path": str(directory), "mask_id": mask_id, "draft_attention": attn_implementation,
            "dtype": "torch.bfloat16", "cache_mode": cache_mode, "max_cache_len": self.max_cache_len,
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "draft_architecture": {"model_type": config.model_type, "hidden_size": config.hidden_size,
                "layers": config.num_hidden_layers, "attention_heads": config.num_attention_heads,
                "kv_heads": config.num_key_value_heads},
            "generation_config_bypassed": True, "raw_draft_argmax": True,
            "tokenizer_compatibility": compatibility, "checkpoint_tying": checkpoint_check,
            "config_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()}
        self.parameter_versions = [(parameter, parameter._version) for parameter in self.model.parameters()]
        self.last_report = None
        self.last_confidence = None
        self.reset()

    def reset(self):
        """Begin a new request, without importing any preceding request output."""
        self.cache = DynamicCache(config=self.model.config) if self.cache_mode == "incremental" else None
        self.cached_visible_ids = ()
        self.last_confidence = None
        self.last_report = None

    def new_session(self, *, cache_mode=None):
        """Fresh independent cache; shares the same immutable pretrained weights."""
        mode = self.cache_mode if cache_mode is None else cache_mode
        if mode not in ("incremental", "stateless"):
            raise ValueError("Choose cache_mode='incremental' or 'stateless'")
        session = object.__new__(type(self))
        session.__dict__.update(self.__dict__)
        session.cache_mode = mode
        session.metadata = {**self.metadata, "cache_mode": mode}
        session.reset()
        return session

    def assert_draft_unchanged(self):
        if any(parameter._version != version for parameter, version in self.parameter_versions):
            raise RuntimeError("A pretrained auxiliary draft parameter changed in place")

    @torch.no_grad()
    def propose(self, visible_ids, num_draft_tokens):
        """Return LongTensor[1,B] using only these visible, authoritative IDs."""
        return self._propose(visible_ids, num_draft_tokens, confidence=False)

    @torch.no_grad()
    def propose_with_confidence(self, visible_ids, num_draft_tokens):
        """Return (LongTensor[1,B], FP32 Tensor[1,B]); scores are drafter-only."""
        tokens = self._propose(visible_ids, num_draft_tokens, confidence=True)
        return tokens, self.last_confidence

    def _propose(self, visible_ids, num_draft_tokens, confidence):
        if not isinstance(visible_ids, torch.Tensor) or visible_ids.ndim != 2 or visible_ids.shape[0] != 1 or visible_ids.shape[1] < 1:
            raise ValueError("visible_ids must be a nonempty target-token tensor[1,L]")
        if visible_ids.dtype != torch.long:
            raise ValueError("Provide the original integer target IDs as torch.long")
        if type(num_draft_tokens) is not int or not 1 <= num_draft_tokens <= 32:
            raise ValueError("Use a public proposal count between1 and32; common probes are3/7/15")
        length = visible_ids.shape[1]
        if length + num_draft_tokens - 1 > self.max_cache_len:
            raise ValueError("Visible history plus draft masks exceeds auxiliary capacity; caller must choose a public feasible budget")
        # This prefix comparison synchronizes GPU inputs to the host. It is a
        # correctness-first adapter; include this overhead in any online timing.
        visible = tuple(visible_ids.detach().cpu().reshape(-1).tolist())
        if min(visible) < 0 or max(visible) >= self.target_vocab_size:
            raise ValueError("Visible IDs are outside the original target vocabulary")
        ids = visible_ids.to(device=self.device).contiguous()
        common = 0
        old_length = len(self.cached_visible_ids)
        if self.cache_mode == "incremental":
            # Reprocess at least the final true token so the last B logits
            # always start at a true-history position, including repeat calls.
            for left, right in zip(self.cached_visible_ids, visible[:-1]):
                if left != right:
                    break
                common += 1
            self.cache.crop(common)
            if int(self.cache.get_seq_length()) != common:
                self.reset()
                raise RuntimeError("Auxiliary cache length did not match the visible common prefix")
        true_input = ids[:, common:]
        masks = torch.full((1, num_draft_tokens - 1), self.mask_id, dtype=torch.long, device=self.device)
        model_input = torch.cat((true_input, masks), dim=-1)
        positions = torch.arange(common, length + num_draft_tokens - 1, device=self.device, dtype=torch.long)
        try:
            # These masks are inputs to the auxiliary model only. Ordinary
            # causal attention prevents later masks influencing true prefixes.
            with torch.inference_mode(False), torch.no_grad():
                output = self.model(input_ids=model_input, attention_mask=None,
                    position_ids=positions.view(1, -1), cache_position=positions,
                    past_key_values=self.cache, use_cache=self.cache_mode == "incremental",
                    logits_to_keep=num_draft_tokens, return_dict=True,
                    output_attentions=False, output_hidden_states=False)
                logits = output.logits[:, -num_draft_tokens:, :]
                if logits.shape != (1, num_draft_tokens, self.target_vocab_size):
                    raise RuntimeError("Unexpected parallel PARD output shape")
                tokens = logits.argmax(-1)
                self.last_confidence = None
                if confidence:
                    logits32 = logits.float()
                    selected = logits32.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                    self.last_confidence = torch.exp(selected - torch.logsumexp(logits32, dim=-1))
                if self.cache_mode == "incremental":
                    self.cache = output.past_key_values
                    self.cache.crop(length)  # discard EVERY speculative mask K/V
                    if int(self.cache.get_seq_length()) != length:
                        raise RuntimeError("Auxiliary mask cache rows survived cropping")
                    self.cached_visible_ids = visible
            self.assert_draft_unchanged()
        except Exception:
            # A partially completed forward may have updated only some layers.
            # No uncertain K/V state is retained for a later proposal.
            self.reset()
            raise
        self.last_report = {"visible_tokens": length, "num_draft_tokens": num_draft_tokens,
            "mask_tokens": num_draft_tokens - 1, "mask_id": self.mask_id,
            "reused_true_prefix_tokens": common, "processed_true_tokens": length - common,
            "discarded_prior_cache_tokens": old_length - common if self.cache_mode == "incremental" else 0,
            "retained_cache_tokens": length if self.cache_mode == "incremental" else 0,
            "cache_mode": self.cache_mode, "all_mask_kv_discarded": True,
            "authoritative_target_verification_still_required": True}
        return tokens
