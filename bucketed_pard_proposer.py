"""Separate PARD graph family that attends only to a public K/V prefix bucket.

The full static cache and all pretrained weights remain intact. Immutable
prefix views change only the attention workload. Graph keys include the
bucket; one logical genuine-prefix history is shared across those graphs.
Preparation is explicit and must precede an active request.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch

from bucketed_cache_view import StaticCachePrefixView
from pard_tree import GraphPARDLatticeProposer


class BucketedPARDLatticeProposer(GraphPARDLatticeProposer):
    def __init__(self, *args, attention_buckets=(256, 512, 1024, 2048),
                 compile_model=False, **kwargs):
        super().__init__(*args, compile_model=False, **kwargs)
        values = sorted(set(int(v) for v in attention_buckets if int(v) <= self.capacity))
        if self.capacity not in values:
            values.append(self.capacity)
        if not values or min(values) < 1:
            raise ValueError("Use positive attention buckets within full cache capacity")
        self.attention_buckets = tuple(values)
        config = self.model.config
        head = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        with torch.inference_mode(False), torch.no_grad():
            self.cache.early_initialization(batch_size=1, num_heads=config.num_key_value_heads,
                head_dim=head, dtype=torch.bfloat16, device=self.device)
        self.cache_views = {v: StaticCachePrefixView(self.cache, v) for v in self.attention_buckets}
        self.compile_model = bool(compile_model)
        if compile_model:
            self.forward = torch.compile(self.model.forward, fullgraph=True, dynamic=True,
                options={"triton.cudagraphs": False})
        self.metadata = {**self.metadata,
            "auxiliary_implementation": "full_storage_bucketed_attention_pard",
            "attention_buckets": list(self.attention_buckets),
            "full_storage_capacity": self.capacity,
            "compile_model": self.compile_model, "compile_dynamic": self.compile_model,
            "graph_key": "draft_depth,input_length,attention_bucket",
            "cache_storage_replaced": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def select_bucket(self, required_end):
        if type(required_end) is not int or not 1 <= required_end <= self.capacity:
            raise ValueError("Required physical attention end is outside full storage")
        return next(v for v in self.attention_buckets if v >= required_end)

    def _call(self, entry, block):
        bucket = entry["attention_bucket"]
        positions = self.gpu_base + entry["offsets"]
        allowed = self.key_positions[:bucket].view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
        output = self.forward(input_ids=entry["input_ids"], past_key_values=self.cache_views[bucket],
            cache_position=positions, position_ids=positions.view(1, -1),
            attention_mask=allowed, use_cache=True, logits_to_keep=block,
            output_attentions=False, output_hidden_states=False, return_dict=True)
        raw = output.logits[:, -block:]
        self.outputs[block].copy_(raw.argmax(-1))
        logits = raw.float()
        values, ids = torch.topk(logits, self.top_k, dim=-1, largest=True, sorted=True)
        self.lattice_ids[block].copy_(ids)
        self.lattice_logps[block].copy_(values - torch.logsumexp(logits, dim=-1, keepdim=True))

    @torch.no_grad()
    def prepare(self, num_draft_tokens, buckets=None):
        block = num_draft_tokens
        if type(block) is not int or not 1 <= block <= 32:
            raise ValueError("Choose a public PARD proposal count1..32")
        requested = self.attention_buckets if buckets is None else tuple(dict.fromkeys(buckets))
        if not requested or any(v not in self.cache_views or v < 2 * block for v in requested):
            raise ValueError("Requested buckets must support the complete prepared input family")
        missing = [(block, length, bucket) for bucket in requested
                   for length in range(block, 2 * block + 1)
                   if (block, length, bucket) not in self.graphs]
        if not missing:
            return {"setup_s": 0.0, "new_graphs": 0, "buckets": list(requested)}
        if self.cached_visible_ids:
            raise RuntimeError("Reset genuine-prefix metadata before preparing new bucket graphs")
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.inference_mode(False), torch.no_grad():
            if block not in self.outputs:
                self.outputs[block] = torch.empty((1, block), device=self.device, dtype=torch.long)
                self.lattice_ids[block] = torch.empty((1, block, self.top_k), device=self.device, dtype=torch.long)
                self.lattice_logps[block] = torch.empty((1, block, self.top_k), device=self.device, dtype=torch.float32)
            for key in missing:
                _, length, bucket = key
                count = length - block + 1
                entry = {"input_ids": torch.full((1, length), self.mask_id, device=self.device, dtype=torch.long),
                    "offsets": torch.arange(length, device=self.device, dtype=torch.long),
                    "true_count": count, "attention_bucket": bucket}
                entry["input_ids"][:, :count].zero_()
                self.gpu_base.zero_()
                self.capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.capture_stream):
                    for _ in range(self.warmups):
                        self._call(entry, block)
                torch.cuda.current_stream().wait_stream(self.capture_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.capture_stream, pool=self.pool):
                    self._call(entry, block)
                torch.cuda.synchronize()
                self.graphs[key] = {**entry, "graph": graph}
            self.reset()
        for view in self.cache_views.values():
            view.assert_storage_consistent(require_initialized=True)
        elapsed = time.perf_counter() - began
        self.setup_s += elapsed
        self.assert_draft_unchanged()
        return {"setup_s": elapsed, "new_graphs": len(missing), "buckets": list(requested),
                "total_graphs": len(self.graphs), "logical_prefix_invalidated": True}

    @torch.no_grad()
    def propose(self, visible_ids, num_draft_tokens):
        if not isinstance(visible_ids, torch.Tensor) or visible_ids.dtype != torch.long or visible_ids.ndim != 2 or visible_ids.shape[0] != 1 or visible_ids.shape[1] < 1:
            raise ValueError("Use only visible genuine LongTensor IDs[1,L]")
        block, length = num_draft_tokens, visible_ids.shape[1]
        if type(block) is not int or not 1 <= block <= 32:
            raise ValueError("Choose a public PARD proposal count1..32")
        bucket = self.select_bucket(length + block - 1)
        if any((block, width, bucket) not in self.graphs for width in range(block, 2 * block + 1)):
            raise RuntimeError("Prepare the public bucket graph family before inference")
        visible = tuple(visible_ids.detach().cpu().reshape(-1).tolist())
        if min(visible) < 0 or max(visible) >= self.target_vocab_size:
            raise ValueError("Visible IDs are outside the original target vocabulary")
        ids = visible_ids.to(self.device).contiguous()
        common = 0
        for left, right in zip(self.cached_visible_ids, visible[:-1]):
            if left != right:
                break
            common += 1
        used_prefill = False
        if not self.cached_visible_ids or common == 0 or length - common > block + 1:
            try:
                self._prefill_true_history(ids)
            except Exception:
                self.reset()
                raise
            common, used_prefill = length - 1, True
        count = length - common
        key = (block, count + block - 1, bucket)
        entry = self.graphs[key]
        try:
            with torch.inference_mode(False), torch.no_grad():
                entry["input_ids"][:, :count].copy_(ids[:, common:])
                self.gpu_base.fill_(common)
                entry["graph"].replay()
                output = self.outputs[block].clone() if self.copy_output else self.outputs[block]
            self.cached_visible_ids = visible
            self.assert_draft_unchanged()
        except Exception:
            self.reset()
            raise
        self.last_report = {"visible_tokens": length, "num_draft_tokens": block,
            "reused_true_prefix_tokens": common, "processed_true_tokens": count,
            "mask_tokens": block - 1, "input_length": count + block - 1,
            "retained_logical_visible_length": length, "used_original_hf_prefill": used_prefill,
            "graph_key": list(key), "physical_cache_capacity": self.capacity,
            "attention_bucket": bucket, "full_storage_unchanged": True}
        return output

    @torch.no_grad()
    def propose_lattice(self, visible_ids, num_draft_tokens):
        self.propose(visible_ids, num_draft_tokens)
        ids, logps = self.lattice_ids[num_draft_tokens], self.lattice_logps[num_draft_tokens]
        return (ids.clone(), logps.clone()) if self.copy_output else (ids, logps)
