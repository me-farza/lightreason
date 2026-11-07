"""Persistent-cache CUDA-graph PARD proposals; target verification is mandatory.

GraphPARDProposer(path, target_tokenizer=tok, attn_implementation='sdpa')
    .prepare(B) captures B+1 graphs for input lengths B..2B, outside inference.
    .propose(visible_ids[1,L], B) returns unverified auxiliary IDs[1,B].

A graph consumes a genuine visible extension followed by B-1 PARD masks.
Explicit GPU positions and a4D boolean causal mask allow only K<=absolute Q.
Old mask K/V may remain physically allocated, but never become visible unless
overwritten by genuine input or this call's new masks. True-prefix metadata is
the authority; StaticCache.get_seq_length is not used to authorize mask rows.

First use, large growth, or prefix replacement uses the original draft HF
DynamicCache prefill and copies its genuine K/V into the persistent cache.
This affects only the auxiliary drafter. Its floating-point output can differ
from stateless PARD. No target model, expected output or future EOS is accepted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch
from transformers import StaticCache

from pard_proposer import PARDProposer


class GraphPARDProposer:
    def __init__(self, model_path_or_proposer, *, target_tokenizer=None, device="cuda",
                 target_vocab_size=151936, max_cache_len=1024, attn_implementation="sdpa",
                 compile_model=False, warmups=3, copy_output=True,
                 max_len=None, share_pool=True):
        if max_len is not None:
            max_cache_len = max_len
        if isinstance(model_path_or_proposer, PARDProposer):
            self.base = model_path_or_proposer
        else:
            if target_tokenizer is None:
                raise ValueError("A target tokenizer is required when loading a local PARD checkpoint")
            self.base = PARDProposer(model_path_or_proposer, target_tokenizer=target_tokenizer,
                device=device, target_vocab_size=target_vocab_size,
                cache_mode="incremental", max_cache_len=max_cache_len,
                attn_implementation=attn_implementation)
        self.model = self.base.model
        self.device = self.base.device
        if getattr(self.model.config, "_attn_implementation", None) != "sdpa":
            raise ValueError("Construct the graph auxiliary drafter with attn_implementation='sdpa'; its4D boolean mask uses SDPA semantics")
        if self.device.type != "cuda":
            raise ValueError("CUDA-graph PARD requires a CUDA auxiliary model")
        if type(max_cache_len) is not int or max_cache_len < 64 or warmups < 1:
            raise ValueError("Use max_cache_len>=64 and positive warmups")
        if getattr(self.model.config, "use_sliding_window", False):
            raise ValueError("This adapter requires the ordinary full-attention PARD checkpoint")
        self.capacity = min(max_cache_len, int(self.model.config.max_position_embeddings))
        self.mask_id = self.base.mask_id
        self.target_vocab_size = self.base.target_vocab_size
        self.warmups, self.copy_output = warmups, copy_output
        self.compile_model = compile_model
        self.cached_visible_ids = ()
        self.graphs, self.outputs = {}, {}
        self.last_report = None
        self.setup_s, self.prefill_count = 0.0, 0
        self.capture_records = []
        with torch.inference_mode(False), torch.no_grad():
            self.cache = StaticCache(config=self.model.config, max_cache_len=self.capacity)
            self.gpu_base = torch.zeros((1,), device=self.device, dtype=torch.long)
            self.key_positions = torch.arange(self.capacity, device=self.device, dtype=torch.long)
            self.pool = torch.cuda.graph_pool_handle() if share_pool else None
            self.capture_stream = torch.cuda.Stream(device=self.device)
        self.forward = self.model.forward
        if compile_model:
            # The outer graphs own capture/replay; avoid nested Inductor graphs.
            self.forward = torch.compile(self.model.forward, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False})
        self.metadata = {**self.base.metadata, "cache_mode": "persistent_static_cuda_graph",
            "draft_attention": getattr(self.model.config, "_attn_implementation", "unknown"),
            "capacity": self.capacity, "compile_model": compile_model,
            "shared_graph_pool": share_pool, "copy_output": copy_output,
            "target_verification_mandatory": True, "mask_kv_physically_zeroed_after_proposal": False,
            "visibility_rule": "explicit4D boolean K<=absolute_query_position; tracked true-prefix length",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def reset(self):
        """Invalidate the logical true prefix; captured storage remains stable."""
        self.cached_visible_ids = ()
        self.last_report = None

    def assert_draft_unchanged(self):
        self.base.assert_draft_unchanged()

    def _call(self, entry, block):
        positions = self.gpu_base + entry["offsets"]
        allowed = self.key_positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
        output = self.forward(input_ids=entry["input_ids"], past_key_values=self.cache,
            cache_position=positions, position_ids=positions.view(1, -1),
            attention_mask=allowed, use_cache=True, logits_to_keep=block,
            output_attentions=False, output_hidden_states=False, return_dict=True)
        selected = output.logits[:, -block:].argmax(-1)
        self.outputs[block].copy_(selected)

    @torch.no_grad()
    def prepare(self, num_draft_tokens):
        block = num_draft_tokens
        if type(block) is not int or not 1 <= block <= 32:
            raise ValueError("Choose a public PARD proposal count between1 and32")
        if 2 * block > self.capacity:
            raise ValueError("Persistent auxiliary cache is too small for this graph family")
        missing = [length for length in range(block, 2 * block + 1) if (block, length) not in self.graphs]
        if not missing:
            return {"setup_s": 0.0, "new_graphs": 0, "block": block,
                    "total_graphs": len(self.graphs), "logical_prefix_invalidated": False}
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Invalidate before the first synthetic write, including failure paths.
        self.reset()
        with torch.inference_mode(False), torch.no_grad():
            if block not in self.outputs:
                self.outputs[block] = torch.empty((1, block), device=self.device, dtype=torch.long)
            for length in missing:
                true_count = length - (block - 1)
                entry = {"input_ids": torch.full((1, length), self.mask_id, device=self.device, dtype=torch.long),
                         "offsets": torch.arange(length, device=self.device, dtype=torch.long),
                         "true_count": true_count}
                entry["input_ids"][:, :true_count].zero_()
                self.gpu_base.zero_()
                self.capture_stream.wait_stream(torch.cuda.current_stream())
                warm_started = time.perf_counter()
                with torch.cuda.stream(self.capture_stream):
                    for _ in range(self.warmups):
                        self._call(entry, block)
                torch.cuda.current_stream().wait_stream(self.capture_stream)
                torch.cuda.synchronize()
                warm_ended = time.perf_counter()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.capture_stream, pool=self.pool):
                    self._call(entry, block)
                torch.cuda.synchronize()
                self.graphs[(block, length)] = {**entry, "graph": graph}
                self.capture_records.append({"block": block, "input_length": length,
                    "true_count": true_count, "warmup_compile_s": warm_ended - warm_started,
                    "capture_s": time.perf_counter() - warm_ended})
            # Warmups wrote synthetic tokens and mask K/V. Never reuse them as
            # visible context; the next call must restore a genuine HF prefix.
            self.reset()
        elapsed = time.perf_counter() - started
        self.setup_s += elapsed
        self.assert_draft_unchanged()
        return {"setup_s": elapsed, "new_graphs": len(missing), "block": block,
                "total_graphs": len(self.graphs), "logical_prefix_invalidated": True,
                "cumulative_setup_s": self.setup_s}

    @torch.no_grad()
    def _prefill_true_history(self, ids):
        with torch.inference_mode(False), torch.no_grad():
            # Original auxiliary HF forward, ordinary DynamicCache, no PARD
            # masks. Prefill does not alter the authoritative target's cache.
            output = self.model(input_ids=ids, attention_mask=None, use_cache=True,
                logits_to_keep=1, return_dict=True, output_attentions=False,
                output_hidden_states=False)
            dynamic = output.past_key_values
            length = ids.shape[1]
            if len(dynamic.layers) != len(self.cache.layers):
                raise RuntimeError("Auxiliary dynamic/static prefill layer counts differ")
            for destination, source in zip(self.cache.layers, dynamic.layers):
                if getattr(destination, "keys", None) is None:
                    raise RuntimeError("Call prepare(B) before copying into persistent draft storage")
                destination.keys[:, :, :length].copy_(source.keys)
                destination.values[:, :, :length].copy_(source.values)
            self.prefill_count += 1

    @torch.no_grad()
    def propose(self, visible_ids, num_draft_tokens):
        if not isinstance(visible_ids, torch.Tensor) or visible_ids.dtype != torch.long or visible_ids.ndim != 2 or visible_ids.shape[0] != 1 or visible_ids.shape[1] < 1:
            raise ValueError("Provide only visible original-target LongTensor IDs[1,L]")
        block = num_draft_tokens
        if type(block) is not int or not 1 <= block <= 32:
            raise ValueError("Choose a public PARD proposal count between1 and32")
        length = visible_ids.shape[1]
        if length + block - 1 > self.capacity:
            raise ValueError("Visible prefix plus PARD masks exceeds auxiliary cache capacity")
        visible = tuple(visible_ids.detach().cpu().reshape(-1).tolist())
        if min(visible) < 0 or max(visible) >= self.target_vocab_size:
            raise ValueError("Visible history contains an ID outside the target classifier")
        prepared = self.prepare(block)
        ids = visible_ids.to(device=self.device).contiguous()
        common = 0
        for left, right in zip(self.cached_visible_ids, visible[:-1]):
            if left != right:
                break
            common += 1
        used_prefill = False
        # Keep the graph family bounded to1..B+1 true inputs. Larger growth or
        # a replacement prefix gets a complete genuine-history prefill first.
        if not self.cached_visible_ids or common == 0 or length - common > block + 1:
            try:
                self._prefill_true_history(ids)
            except Exception:
                self.reset()
                raise
            common = length - 1
            used_prefill = True
        true_count = length - common
        total_input = true_count + block - 1
        if (block, total_input) not in self.graphs:
            self.reset()
            raise RuntimeError("Public true-extension length is outside the prepared graph family")
        entry = self.graphs[(block, total_input)]
        try:
            with torch.inference_mode(False), torch.no_grad():
                # Placeholder slots were filled once during prepare and are
                # never written by a runtime call for this fixed input length.
                entry["input_ids"][:, :true_count].copy_(ids[:, common:])
                self.gpu_base.fill_(common)
                entry["graph"].replay()
                tokens = self.outputs[block].clone() if self.copy_output else self.outputs[block]
            self.cached_visible_ids = visible
            self.assert_draft_unchanged()
        except Exception:
            self.reset()
            raise
        self.last_report = {"visible_tokens": length, "num_draft_tokens": block,
            "reused_true_prefix_tokens": common, "processed_true_tokens": true_count,
            "mask_tokens": block - 1, "input_length": total_input,
            "retained_logical_visible_length": length, "used_original_hf_prefill": used_prefill,
            "prefill_count": self.prefill_count, "graph_prepare": prepared,
            "graph_key": [block, total_input], "physical_cache_capacity": self.capacity,
            "stale_mask_rows_hidden_by_causal_mask": True,
            "authoritative_target_verification_still_required": True,
            "returned_output_is_a_copy": self.copy_output}
        return tokens
