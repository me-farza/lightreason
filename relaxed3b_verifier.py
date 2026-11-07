"""Original-weight HF batched verification with explicitly relaxed FP order.

This is a SEPARATE numerical variant. It uses the original frozen BF16 target
modules, logits and equations, but an M-token GEMM/attention evaluation can
round differently from M serial GEMVs. Target verification therefore does not
establish exact agreement with the strict serial reference. Report agreement
and quality separately; preserve the strict runner and its measurements.

The caller supplies [anchor, draft1, ...] and owns proposal construction/EOS.
The cache excludes the unprocessed anchor. verify() tentatively writes all M
input rows and returns M target argmax IDs. commit(c) retains c processed input
rows (normally c=matched_drafts+1), and makes prediction[c-1] the next anchor.
Rejected target K/V is physically cropped in DynamicCache mode. Static mode
retains stale storage but excludes it with an absolute-position causal mask;
later blocks overwrite each newly visible row before attention can read it.

Default DynamicCache mode is the correctness-first functional implementation.
Optional HF StaticCache mode uses the same original model forward with a 4D
prepared mask. Optional CUDA graphs capture one call per block size, with GPU
positions/masks. Prefill is always original HF DynamicCache, bit-copied into
static storage. Setup/capture time is recorded separately from online replay.

Pinned source contracts:
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/masking_utils.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/sdpa_attention.py
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from transformers.cache_utils import DynamicCache, StaticCache


@dataclass
class _StaticEntry:
    input_ids: torch.Tensor
    base: torch.Tensor
    offsets: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    graph: object = None
    logits: torch.Tensor | None = None
    next_ids: torch.Tensor | None = None


class Relaxed3BVerifier:
    """Frozen original-target block verifier; no proposer or acceptance oracle.

    Public lifecycle:
        anchor = verifier.prefill(prompt_ids)              # [1,1]
        verifier.prepare((1,4,8,16))                      # optional, untimed
        predicted = verifier.verify(input_block, basep)   # [M]
        verifier.commit(matched_drafts + 1)               # new cache length

    `current_position` is authoritative. Never infer static-cache length by
    scanning nonzero K/V: rejected rows deliberately remain allocated.
    `last_logits` is [M,V], and may be overwritten by the next graph replay.
    """
    def __init__(self, model, max_len=1024, *, cache_mode="dynamic",
                 cuda_graph=False, compile_static=False, warmups=3,
                 share_pool=False, block_sizes=(1, 2, 4, 8, 16, 32)):
        started = time.perf_counter()
        if cache_mode not in ("dynamic", "static"):
            raise ValueError("cache_mode must be 'dynamic' or 'static'")
        if cuda_graph and cache_mode != "static":
            raise ValueError("CUDA graphs require persistent HF StaticCache storage")
        if compile_static and cache_mode != "static":
            raise ValueError("compile_static applies only to the static forward")
        if type(max_len) is not int or max_len < 2 or type(warmups) is not int or warmups < 1:
            raise ValueError("Use positive cache capacity>=2 and at least one warmup")
        sizes = tuple(dict.fromkeys(int(size) for size in block_sizes))
        if not sizes or any(not 1 <= size <= 32 for size in sizes):
            raise ValueError("Allowed block sizes must be integers from1 through32")
        config = model.config
        geometry = (config.model_type, config.hidden_size, config.num_attention_heads,
                    config.num_key_value_heads, config.num_hidden_layers)
        if geometry != ("qwen2", 2048, 16, 2, 36):
            raise ValueError("This variant is scoped to the original Qwen2.5-3B target")
        if getattr(model.model, "has_sliding_layers", False):
            raise ValueError("This absolute causal-mask variant does not support sliding attention")
        if getattr(config, "_attn_implementation", None) != "sdpa":
            raise ValueError("Use the original HF target loaded with attn_implementation='sdpa'")
        if model.training or any(parameter.requires_grad or parameter.dtype != torch.bfloat16
                                 for parameter in model.parameters()):
            raise ValueError("Target must already be eval(), frozen and original BF16")
        self.model, self.max_len = model, max_len
        self.device = model.get_input_embeddings().weight.device
        if self.device.type != "cuda" and (cuda_graph or compile_static):
            raise ValueError("This graph/compiled experiment requires CUDA")
        self.cache_mode, self.cuda_graph = cache_mode, bool(cuda_graph)
        self.compile_static, self.warmups, self.share_pool = bool(compile_static), warmups, bool(share_pool)
        self.block_sizes = sizes
        self.parameter_versions = [(parameter, parameter._version) for parameter in model.parameters()]
        self.entries, self.graphs, self.capture_records = {}, {}, []
        self.pool = torch.cuda.graph_pool_handle() if cuda_graph and share_pool else None
        self.setup_s, self.prefill_s = 0.0, 0.0
        self.verify_calls, self.committed_input_tokens = 0, 0
        self.key_positions = torch.arange(max_len, device=self.device, dtype=torch.long).view(1, 1, 1, -1)
        self.cache = None
        if cache_mode == "static":
            self.cache = StaticCache(config=config, max_cache_len=max_len)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            with torch.inference_mode(False), torch.no_grad():
                self.cache.early_initialization(batch_size=1, num_heads=config.num_key_value_heads,
                    head_dim=head_dim, dtype=torch.bfloat16, device=self.device)
        self._static_forward_fn = self._forward_static
        if compile_static:
            # The wrapper belongs to THIS verifier. Source modules, parameters,
            # configuration and strict/reference callables are not replaced.
            self._static_forward_fn = torch.compile(self._forward_static, fullgraph=True, mode="default")
        self.current_position = 0
        self.pending = None
        # Do not alias a CUDA graph output: another replay, even one later
        # discarded, must never overwrite the prior committed anchor.
        self._anchor_buffer = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self.anchor, self.last_logits, self.last_next_ids, self.last_report = None, None, None, None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.constructor_s = time.perf_counter() - started
        self.metadata = {
            "variant": "relaxed_fp_order_original_weight_batched_hf",
            "strict_output_identity_claimed": False,
            "target_model_weights_changed": False,
            "target_mathematical_equations_changed": False,
            "cache_mode": cache_mode, "cuda_graph": self.cuda_graph,
            "compile_static": self.compile_static, "max_len": max_len,
            "block_sizes": list(sizes), "dtype": "torch.bfloat16",
            "causal_semantics": "key_absolute_position <= query_absolute_position",
            "prefill": "original HF DynamicCache, original target full head",
            "sampling": "raw target argmax only; caller handles EOS and acceptance",
            "quality_and_exact_agreement_must_be_measured": True,
        }

    @property
    def position(self):
        return self.current_position

    def assert_target_unchanged(self):
        if any(parameter._version != version for parameter, version in self.parameter_versions):
            raise RuntimeError("An original target parameter changed in place")

    def _check_ids(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Provide original target IDs as LongTensor[1,L]")
        if ids.device != self.device or ids.shape[1] < 1:
            raise ValueError("Input IDs must be nonempty and already on the target device")

    def _dynamic_length(self):
        lengths = [int(layer.get_seq_length()) for layer in self.cache.layers]
        if not lengths or len(set(lengths)) != 1:
            raise RuntimeError("Target DynamicCache layers have inconsistent lengths")
        return lengths[0]

    @torch.no_grad()
    def prefill(self, ids):
        """Start/reset one unpadded request; return its unprocessed target anchor."""
        self._check_ids(ids)
        length = ids.shape[1]
        if length >= self.max_len:
            raise ValueError("Prompt must leave room for at least one target input")
        self.assert_target_unchanged()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode(False), torch.no_grad():
            # A fresh, independent original HF prefill prevents rejected rows
            # or graph warmups from leaking across requests.
            original = self.model(input_ids=ids, attention_mask=None,
                past_key_values=DynamicCache(config=self.model.config),
                use_cache=True, logits_to_keep=1, return_dict=True,
                output_attentions=False, output_hidden_states=False)
            source_cache = original.past_key_values
            if self.cache_mode == "dynamic":
                self.cache = source_cache
            else:
                self.cache.reset()
                if len(source_cache.layers) != len(self.cache.layers):
                    raise RuntimeError("Original prefill/cache layer counts differ")
                for destination, source in zip(self.cache.layers, source_cache.layers):
                    destination.keys[:, :, :length, :].copy_(source.keys)
                    destination.values[:, :, :length, :].copy_(source.values)
            self.last_logits = original.logits[:, -1, :]
            self._anchor_buffer.copy_(self.last_logits.argmax(-1, keepdim=True))
            self.anchor = self._anchor_buffer
            self.last_next_ids = self.anchor.reshape(-1)
        self.current_position, self.pending = length, None
        self.verify_calls, self.committed_input_tokens = 0, 0
        self.last_report = {"operation": "prefill", "prompt_tokens": length,
            "cache_excludes_unprocessed_anchor": True, "original_hf_prefill": True}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.prefill_s = time.perf_counter() - started
        return self.anchor

    def _new_entry(self, size):
        entry = _StaticEntry(
            input_ids=torch.zeros((1, size), device=self.device, dtype=torch.long),
            base=torch.tensor([self.current_position], device=self.device, dtype=torch.long),
            offsets=torch.arange(size, device=self.device, dtype=torch.long),
            positions=torch.empty(size, device=self.device, dtype=torch.long),
            mask=torch.empty((1, 1, size, self.max_len), device=self.device, dtype=torch.bool))
        self.entries[size] = entry
        return entry

    def _forward_static(self, input_ids, positions, mask):
        # A prepared4D mask bypasses HF mask-size/cache-length inference.
        # Bare is_causal=True on rectangular Q/K would have the wrong offset.
        output = self.model(input_ids=input_ids, attention_mask=mask,
            position_ids=positions.view(1, -1), cache_position=positions,
            past_key_values=self.cache, use_cache=True, logits_to_keep=input_ids.shape[1],
            return_dict=True, output_attentions=False, output_hidden_states=False)
        logits = output.logits.squeeze(0)
        return logits, logits.argmax(-1)

    def _static_step(self, entry):
        torch.add(entry.base, entry.offsets, out=entry.positions)
        torch.le(self.key_positions, entry.positions.view(1, 1, -1, 1), out=entry.mask)
        entry.logits, entry.next_ids = self._static_forward_fn(entry.input_ids, entry.positions, entry.mask)
        return entry.next_ids

    @torch.no_grad()
    def prepare(self, block_sizes=None):
        """Optional untimed setup; never commits warmup/capture target rows."""
        sizes = (self.block_sizes if block_sizes is None else (block_sizes,) if type(block_sizes) is int
                 else tuple(dict.fromkeys(block_sizes)))
        if any(type(size) is not int or size not in self.block_sizes for size in sizes):
            raise ValueError("Prepare only constructor-authorized block sizes")
        if self.cache_mode != "static":
            return {"new_graphs": 0, "setup_s": 0.0, "cache_mode": self.cache_mode}
        if self.pending is not None:
            raise RuntimeError("Commit or discard the preceding block before graph preparation")
        # Capture may precede the first request. Zero prefix/cache values are
        # disposable warmup data; the next original-HF prefill resets storage.
        representative_base = self.current_position
        missing = [size for size in sizes if size not in self.entries or
                   (self.cuda_graph and size not in self.graphs)]
        if any(representative_base + size > self.max_len for size in missing):
            raise ValueError("Prepared block would exceed target cache capacity")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for size in missing:
            entry = self.entries.get(size) or self._new_entry(size)
            entry.base.fill_(representative_base)
            entry.input_ids.fill_(0)
            if not self.cuda_graph:
                if self.compile_static:
                    warm_started = time.perf_counter()
                    with torch.inference_mode(False), torch.no_grad():
                        for _ in range(self.warmups):
                            self._static_step(entry)
                    torch.cuda.synchronize(self.device)
                    self.capture_records.append({"block_size": size,
                        "warmup_s": time.perf_counter() - warm_started, "capture_s": 0.0,
                        "representative_base": representative_base,
                        "captured_before_first_prefill": self.current_position == 0,
                        "compiled_without_cuda_graph": True})
                continue
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            warm_started = time.perf_counter()
            with torch.cuda.stream(stream), torch.inference_mode(False), torch.no_grad():
                for _ in range(self.warmups):
                    self._static_step(entry)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            warmup_s = time.perf_counter() - warm_started
            graph = torch.cuda.CUDAGraph()
            capture_started = time.perf_counter()
            with torch.cuda.graph(graph, pool=self.pool, stream=stream), torch.inference_mode(False), torch.no_grad():
                self._static_step(entry)
            torch.cuda.synchronize(self.device)
            entry.graph = graph
            self.graphs[size] = graph
            self.capture_records.append({"block_size": size, "warmup_s": warmup_s,
                "capture_s": time.perf_counter() - capture_started,
                "representative_base": representative_base,
                "captured_before_first_prefill": self.current_position == 0})
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.setup_s += elapsed
        return {"new_graphs": len(missing) if self.cuda_graph else 0,
                "new_entries": len(missing), "setup_s": elapsed, "cache_mode": self.cache_mode}

    @torch.no_grad()
    def verify(self, inputs, basep=None):
        """Tentatively evaluate M target inputs, returning raw argmax IDs[M]."""
        self._check_ids(inputs)
        if self.cache is None or self.current_position < 1:
            raise RuntimeError("Call prefill() before verification")
        if self.pending is not None:
            raise RuntimeError("Commit or discard the preceding block before verifying another")
        basep = self.current_position if basep is None else basep
        if type(basep) is not int or basep != self.current_position:
            raise ValueError("basep must equal the current committed target cache length")
        size = inputs.shape[1]
        if size not in self.block_sizes or basep + size > self.max_len:
            raise ValueError("Block size is unauthorized or exceeds target cache capacity")
        try:
            with torch.inference_mode(False), torch.no_grad():
                if self.cache_mode == "dynamic":
                    if self._dynamic_length() != basep:
                        raise RuntimeError("Dynamic target cache length differs from committed position")
                    positions = torch.arange(basep, basep + size, device=self.device, dtype=torch.long)
                    output = self.model(input_ids=inputs, attention_mask=None,
                        position_ids=positions.view(1, -1), cache_position=positions,
                        past_key_values=self.cache, use_cache=True, logits_to_keep=size,
                        return_dict=True, output_attentions=False, output_hidden_states=False)
                    self.cache = output.past_key_values
                    self.last_logits = output.logits.squeeze(0)
                    self.last_next_ids = self.last_logits.argmax(-1)
                    if self._dynamic_length() != basep + size:
                        raise RuntimeError("Dynamic target did not append the whole verification block")
                else:
                    if size not in self.entries or (self.cuda_graph and size not in self.graphs):
                        self.prepare((size,))
                    entry = self.entries[size]
                    entry.input_ids.copy_(inputs)
                    entry.base.fill_(basep)
                    if self.cuda_graph:
                        entry.graph.replay()
                    else:
                        self._static_step(entry)
                    self.last_logits, self.last_next_ids = entry.logits, entry.next_ids
                if self.last_logits.shape != (size, self.model.config.vocab_size):
                    raise RuntimeError("Target returned an unexpected classifier block shape")
        except Exception:
            # Writes are confined to the speculative tail. Dynamic forwards
            # may update only a subset of layers before failing, so crop ALL.
            if self.cache_mode == "dynamic" and self.cache is not None:
                self.cache.crop(basep)
            self.pending = None
            raise
        self.pending = {"basep": basep, "input_count": size}
        self.verify_calls += 1
        self.last_report = {"operation": "verify", "basep": basep, "input_count": size,
            "tentative_end": basep + size, "cache_mode": self.cache_mode,
            "cuda_graph_replay": self.cuda_graph,
            "strict_serial_token_agreement_not_assumed": True}
        return self.last_next_ids

    @torch.no_grad()
    def commit(self, acceptedinputcount):
        """Keep this many input K/V rows; zero discards the entire proposal."""
        if self.pending is None:
            raise RuntimeError("No pending verification block to commit")
        count = acceptedinputcount
        if type(count) is not int or not 0 <= count <= self.pending["input_count"]:
            raise ValueError("Commit count must be an integer from0 through block size")
        new_position = self.pending["basep"] + count
        if self.cache_mode == "dynamic":
            self.cache.crop(new_position)
            if self._dynamic_length() != new_position:
                raise RuntimeError("Rejected target cache rows survived DynamicCache.crop")
        if count:
            self._anchor_buffer.copy_(self.last_next_ids[count - 1:count].view(1, 1))
            self.anchor = self._anchor_buffer
        self.committed_input_tokens += count
        self.current_position, self.pending = new_position, None
        self.last_report = {"operation": "commit", "accepted_input_count": count,
            "new_position": new_position, "physical_crop": self.cache_mode == "dynamic",
            "rejected_static_tail_excluded_by_absolute_mask": self.cache_mode == "static"}
        return new_position

    def discard(self):
        return self.commit(0)

    def memory_report(self):
        values = [value for layer in ([] if self.cache is None else self.cache.layers)
                  for value in (layer.keys, layer.values) if value is not None]
        report = {"cache_bytes": sum(value.numel() * value.element_size() for value in values),
            "graph_count": len(self.graphs), "graph_block_sizes": sorted(self.graphs),
            "constructor_s": self.constructor_s, "setup_s": self.setup_s,
            "last_prefill_s": self.prefill_s, "capture_records": self.capture_records,
            "current_position": self.current_position, "pending": self.pending,
            "strict_output_identity_claimed": False}
        if self.device.type == "cuda":
            report.update({"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                           "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)})
        return report


# Short import alias for controller code; both names denote the same variant.
HFBlockVerifier = Relaxed3BVerifier
