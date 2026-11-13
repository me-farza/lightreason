"""Reuse the BF16 GPU bank allocation as a larger lossless decode cache."""

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
import triton
import triton.language as tl

from edge35b.cpu_read_packing import TILE_VALUES


ROUTES = 64
PACKED_TILE_BYTES = TILE_VALUES * 3 // 2


@triton.jit(do_not_specialize=["layer", "slot_base"])
def _stage_plan(slots, cpu_ids, slot_to_id, evicted, fetched, read_plan, count, gemm_slots, layer, slot_base,
                PACKED: tl.constexpr, HOT_RAW: tl.constexpr, ROUTES: tl.constexpr, EXPERTS: tl.constexpr):
    i = tl.arange(0, ROUTES)
    slot = tl.load(slots + i)
    on_gpu = tl.load(cpu_ids + i) == -1
    identity = tl.load(slot_to_id + slot, mask=on_gpu, other=0)
    expert = identity - layer * EXPERTS
    active = on_gpu
    # Decode each unique weight once. The first route owns the staged row.
    same = (expert[:, None] == expert[None, :]) & active[None, :]
    leader = tl.min(tl.where(same, i[None, :], ROUTES), axis=1)
    first = (leader == i) & active
    index = tl.cumsum(first.to(tl.int32)) - 1
    row = tl.sum(tl.where(leader[:, None] == i[None, :], index[None, :], 0), axis=1)
    tl.store(read_plan + index, expert, mask=first)
    physical = slot + slot_base
    tl.store(read_plan + ROUTES + index, physical, mask=first)
    inserted = tl.load(evicted + i, i < tl.load(fetched), other=-3)
    miss = tl.sum((slot[:, None] == inserted[None, :]).to(tl.int32), axis=1) > 0
    tl.store(read_plan + 2 * ROUTES + index, miss, mask=first)
    tl.store(count, tl.sum(first.to(tl.int32)))
    tl.store(gemm_slots + i, tl.where(active, tl.where(physical >= PACKED,
                                                      physical - PACKED, HOT_RAW + row + 1), HOT_RAW))


class PackedPrimaryCache:
    """Cache slots use compact payloads for packable layers and raw BF16 otherwise."""

    def __init__(self, cache, transport, read_banks, layers, report_path, *, layout="staged-primary",
                 reserve_rows=512, replica_plan=None):
        from freetoken.kernel.pinned import device_ptr

        start = perf_counter()
        self.cache, self.transport = cache, transport
        self.layers = frozenset(layers) - {0}
        self.device = cache.device
        self.report_path = Path(report_path)
        self.slots = cache.cache_size
        if layout not in {"staged-primary", "compact-tail", "fixed-prefix"}:
            raise ValueError("Invalid packed primary layout")
        self.tiered = layout == "compact-tail"
        self.fixed = layout == "fixed-prefix"
        if self.tiered and not 64 <= reserve_rows <= 512:
            raise ValueError("The packed tier must fit in the prefill buffer region")
        self.reserve_rows = reserve_rows
        self.raw_experts = cache.num_experts
        self.scratch = ROUTES + 1  # row zero is the original GPU weight-zero route
        self.native = transport.native
        self.baseline_bytes = sum(bank.nbytes for _, bank in cache.banks)
        if (self.slots != 2560 or cache.pool_sizes != [self.slots] or self.raw_experts != 256
                or not self.layers or self.native is None or transport.gpu_resident is None):
            raise ValueError("Packed primary trials require the frozen 34-layer hybrid Qwen BF16 runner")
        strides = {}
        row_sizes = {layer: [] for layer in self.layers}
        self.prefix = {layer: [] for layer in self.layers}
        for bank in range(len(cache.banks)):
            for layer in self.layers:
                prepared = read_banks.shared.get((layer, bank))
                if prepared is None:
                    raise ValueError(f"Missing packed source for layer {layer}, bank {bank}")
                _, bases, _ = prepared
                fallback = (bases.cpu().numpy().reshape(self.raw_experts, -1, 32) == 255)
                flags = np.packbits(fallback, axis=-1, bitorder="little").view(np.uint32).squeeze(-1)
                counts = fallback.sum(axis=-1, dtype=np.uint64)
                offsets = counts.cumsum(axis=-1) - counts
                header = (offsets << 32) | flags
                self.prefix[layer].append(torch.from_numpy(header.view(np.int64)).to(self.device))
                tiles = fallback.shape[1] * 32
                sizes = tiles * PACKED_TILE_BYTES + counts.sum(axis=-1) * 128
                row_sizes[layer].append(sizes)
                stride = int(sizes.max())
                strides.setdefault(layer, []).append(stride)
        # Layers with unusually wide exponent ranges use raw slots. Every
        # compact row in the remaining layers fits its fixed-stride slot.
        max_packed_share = 0.875 if self.fixed else 0.82
        if self.tiered or self.fixed:
            # Only the admitted experts need to fit the tail stride. Outliers
            # still use the original copy path and never claim a tail slot.
            self.max_stride = [int(max_packed_share * arena[0].nbytes) // 128 * 128 for _, arena in cache.banks]
            cacheable = {layer: np.logical_and.reduce([sizes <= limit for sizes, limit in
                                                       zip(rows, self.max_stride, strict=True)])
                         for layer, rows in row_sizes.items()}
            self.layers = frozenset(layer for layer, keep in cacheable.items() if keep.any())
            self.cacheable = {layer: torch.from_numpy(keep).to(self.device) for layer, keep in cacheable.items()}
        else:
            self.layers = frozenset(layer for layer, stride in strides.items()
                                    if all(s <= max_packed_share * arena[0].nbytes
                                           for s, (_, arena) in zip(stride, cache.banks, strict=True)))
            self.max_stride = [max(strides[layer][bank] for layer in self.layers) for bank in range(len(cache.banks))]
        if not self.layers:
            raise ValueError("No layers fit the packed primary slot budget")
        self.prefix = {layer: self.prefix[layer] for layer in self.layers}
        self.prefix_ptrs = {layer: self.pointer_tensor(ptrs) for layer, ptrs in self.prefix.items()}

        # The same underlying allocation holds compact rows, the raw tier and
        # BF16 scratch. No full raw shadow cache exists during decode.
        capacities = []
        for (_, arena), stride in zip(cache.banks, self.max_stride, strict=True):
            row_bytes = arena[0].nbytes
            if self.fixed:
                storage = (reserve_rows - self.scratch) * row_bytes
            elif self.tiered:
                storage = reserve_rows * row_bytes
            else:
                storage = arena.nbytes - (self.raw_experts + self.scratch) * row_bytes
            capacities.append(storage // stride)
        if self.fixed and not 512 <= reserve_rows <= min(min(capacities), self.slots - 1):
            raise ValueError("The fixed packed prefix and BF16 scratch must fit before the raw slots")
        self.packed_experts = reserve_rows if self.fixed else min(capacities)
        raw_slots = self.slots - reserve_rows if self.tiered or self.fixed else self.raw_experts
        self.capacity = self.packed_experts + raw_slots

        self.views, descriptors = [], []
        self.tail_storage = []
        self.replica_space = []
        for (_, arena), stride in zip(cache.banks, self.max_stride, strict=True):
            row_bytes = arena[0].nbytes
            base = arena.data_ptr()
            if self.fixed:
                scratch_start = reserve_rows - self.scratch
                self.views.append(arena[scratch_start:])
                descriptors.append((base, base + reserve_rows * row_bytes, base + scratch_start * row_bytes,
                                    stride, row_bytes))
                self.replica_space.append(arena.view(torch.uint8).view(-1)[reserve_rows * stride:scratch_start * row_bytes])
            else:
                # The raw tier and scratch are one contiguous BF16 GEMM view.
                raw = arena[:self.raw_experts + self.scratch]
                self.views.append(raw)
                descriptors.append((base + raw.nbytes, base, base + self.raw_experts * row_bytes, stride, row_bytes))
            if self.tiered:
                self.tail_storage.append(arena[:reserve_rows])
        self.layouts = torch.tensor(descriptors, device=self.device, dtype=torch.int64)
        self.slot_to_id = torch.full((self.capacity,), -1, device=self.device, dtype=torch.int32)
        self.usage = torch.zeros(self.capacity, device=self.device, dtype=torch.int64)
        self.step = torch.zeros((), device=self.device, dtype=torch.int64)
        self.slot_for_id = torch.full_like(cache.slot_for_id, -1)
        self.recency = torch.full_like(cache.expert_recency, -1)
        self.plan = torch.empty((3, ROUTES), device=self.device, dtype=torch.int32)
        self.count = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.gemm_slots = torch.empty((8, 8), device=self.device, dtype=torch.int32)
        self.sources = {}
        for layer in range(cache.num_layers):
            table = transport.shared_descriptors.get(layer)
            if table is None:
                pointers = [device_ptr(sources[layer]) for sources, _ in cache.banks]
                hot = {e: i for i, (l, e) in enumerate(transport.hot_experts) if self.fixed and l == layer}
                table = torch.tensor([[(0, 0, transport.hot_storage[b][hot[e]].data_ptr()
                                         if e in hot else base + e * bank[0].nbytes) for e in range(self.raw_experts)]
                                      for b, (base, (_, bank)) in enumerate(zip(pointers, cache.banks, strict=True))],
                                     device=self.device, dtype=torch.int64)
            elif layer in self.layers:
                table = table.clone()
                # The compact row's fallbacks and offsets describe the original
                # host tile layout even when a GPU replica supplies its bytes.
                for bank, (sources, _) in enumerate(cache.banks):
                    source = sources[layer]
                    owner, bases, _ = read_banks.shared[layer, bank]
                    expert = torch.arange(self.raw_experts, device=self.device, dtype=torch.int64)
                    tiles = source[0].numel() // TILE_VALUES
                    table[bank, :, 0] = device_ptr(owner.tensor) + expert * (tiles * PACKED_TILE_BYTES)
                    table[bank, :, 1] = bases.data_ptr() + expert * tiles
                    table[bank, :, 2] = device_ptr(source) + expert * source[0].nbytes
            self.sources[layer] = table

        # Raw/compact choices are captured into the two prefill-independent
        # graphs. GPU replicas remain a separate, read-only miss source.
        self.miss_sources = {
            layer: transport.gpu_resident.tables.get(layer, transport.shared_descriptors.get(layer, table))
            for layer, table in self.sources.items()
        }

        self.report = {"raw_control_slots": self.slots, "primary_slots": self.capacity,
                       "packed_slots": self.packed_experts, "raw_slots": raw_slots,
                       "packed_layers": sorted(self.layers), "scratch_rows": 0 if self.tiered else self.scratch,
                       "bank_bytes": self.baseline_bytes, "layout": layout,
                       "packed_stride_bytes": self.max_stride,
                       "setup_s": perf_counter() - start}
        if self.tiered:
            self.tail = CompactTail(self)
            self.report["prefill_rows_reused"] = self.reserve_rows
        elif self.fixed:
            from edge35b.fixed_packed_primary import FixedPackedPrefix

            self.fixed_prefix = FixedPackedPrefix(self, replica_plan)
        self.report["setup_s"] = perf_counter() - start
        self.report_path.write_text(json.dumps(self.report, indent=2) + "\n")

    def pointer_tensor(self, tensors):
        return torch.tensor([t.data_ptr() for t in tensors], device=self.device, dtype=torch.int64)

    def reset(self):
        if self.fixed:
            self.fixed_prefix.reset()
            return
        if self.tiered:
            self.tail.reset()
            return
        self.slot_to_id.fill_(-1)
        self.slot_for_id.fill_(-1)
        self.usage.zero_()
        self.step.zero_()
        self.recency.fill_(-1)
        for bank in self.views:
            bank[self.raw_experts].zero_()

    def bounds(self, layer):
        return (0, self.packed_experts) if layer in self.layers else (self.packed_experts, self.capacity)

    def ensure(self, layer, experts, weights, clamp):
        if self.fixed:
            return self.cache.ensure_experts_hybrid(layer, experts, weights, clamp=clamp)
        if self.tiered:
            return self.tail.ensure(layer, experts, weights, clamp)
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        cache = self.cache
        fields = ("id_of_slot", "_pool_id_of_slot", "_pool_usage", "slot_for_id", "step", "expert_recency")
        original = [getattr(cache, field) for field in fields]
        lo, hi = self.bounds(layer)
        # The existing Triton planner uses tier-local slot indices. The
        # reverse map is sliced by tier, so a victim cannot cross formats.
        replacement = (self.slot_to_id, [self.slot_to_id[lo:hi]], [self.usage[lo:hi]],
                       self.slot_for_id, self.step, self.recency)
        try:
            for field, value in zip(fields, replacement, strict=True):
                setattr(cache, field, value)
            fraction = cache.hybrid_fetch_fraction[layer] if cache.hybrid_fetch_fraction is not None else 0.0
            routes = ensure_experts_hybrid(cache, layer, experts, weights, cache.hybrid_max_fetch,
                                          fraction, clamp=clamp)
        finally:
            for field, value in zip(fields, original, strict=True):
                setattr(cache, field, value)
        return routes

    def stage(self, layer, physical_slots, cpu_ids, *, profile=False):
        if self.fixed:
            return self.fixed_prefix.stage(layer, physical_slots, cpu_ids, profile)
        if self.tiered:
            self.tail.fill(layer)
            return self.tail.raw_views, physical_slots
        lo, _ = self.bounds(layer)
        _stage_plan[(1,)](physical_slots, cpu_ids, self.slot_to_id[lo:], self.cache.evict_slots, self.cache.num_indices,
                         self.plan, self.count,
                         self.gemm_slots, layer, lo, self.packed_experts, self.raw_experts, ROUTES, self.raw_experts,
                         num_warps=4)
        prefix = self.prefix_ptrs.get(layer, next(iter(self.prefix_ptrs.values())))
        self.native.stage_primary(self.sources[layer], self.miss_sources[layer], self.layouts, prefix, self.plan, self.count,
                                  experts=self.raw_experts, packed_slots=self.packed_experts)
        return tuple(self.views), self.gemm_slots

    def audit(self):
        """Check sampled real rows through fill, packed hit, raw hit and refill."""
        if self.fixed:
            return self.fixed_prefix.audit()
        if self.tiered:
            return self.tail.audit()
        checked = 0
        for layer in sorted({0, min(self.layers), max(self.layers), self.cache.num_layers - 1}):
            lo, _ = self.bounds(layer)
            candidates = [e for l, e in self.transport.gpu_resident.keys["packed"] if l == layer]
            chosen = list(dict.fromkeys([*candidates[:1], 0, 127, 255]))
            for expert in chosen:
                self.slot_to_id[lo] = layer * self.raw_experts + expert
                slots = torch.zeros((8, 8), device=self.device, dtype=torch.int32)
                cpu_ids = torch.full_like(slots, -1)
                self.cache.evict_slots[0] = 0
                for fill in (1, 0):
                    self.cache.num_indices.fill_(fill)
                    views, _ = self.stage(layer, slots, cpu_ids)
                    out = self.raw_experts + 1 if layer in self.layers else 0
                    for bank, (source, _) in zip(views, self.cache.banks, strict=True):
                        if not torch.equal(bank[out].view(torch.int16).cpu(), source[layer][expert].view(torch.int16)):
                            raise RuntimeError(f"Primary slot audit failed at layer {layer}, expert {expert}, fill={fill}")
                        checked += bank[out].numel()
        self.reset()
        self.report["sampled_bf16_values_audited"] = checked
        self.report_path.write_text(json.dumps(self.report, indent=2) + "\n")

    def copy_profile(self):
        if self.fixed:
            self.fixed_prefix.copy_profile()
        elif self.tiered:
            self.tail.host_hits.copy_(self.tail.hit_trace, non_blocking=True)

    def adjust_profile(self, info, raw_expert_bytes):
        if self.fixed:
            self.fixed_prefix.adjust_profile(info, raw_expert_bytes)
            return
        if not self.tiered:
            return
        hits = self.tail.host_hits.tolist()
        counts = [len(ids) for ids in info["expert_fetch_ids"]]
        info["packed_tail_hits"] = [sum(row[:n]) for row, n in zip(hits, counts, strict=True)]
        payload = self.transport.gpu_resident.payload["packed"]
        for layer, (ids, row) in enumerate(zip(info["expert_fetch_ids"], hits, strict=True)):
            avoided = sum(int(payload[layer, expert]) for expert, hit in zip(ids, row[:len(ids)], strict=True) if hit)
            info["expert_pcie_payload_bytes"][layer] -= avoided
            info["expert_gpu_source_bytes"][layer] += sum(row[:len(ids)]) * raw_expert_bytes

    def begin_decode(self):
        if self.fixed:
            self.fixed_prefix.begin_decode()


@triton.jit(do_not_specialize=["layer"])
def _tail_plan(keys, usage, step_ptr, expert_to_entry, fetched, fetch_count, skip, plan, hit_trace,
               layer, EXPERTS: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr,
               WIDTH: tl.constexpr, SAVE: tl.constexpr):
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    entries = tl.arange(0, BLOCK)
    planned = tl.arange(0, WIDTH)
    n = tl.load(fetch_count)
    expert = tl.load(fetched + planned, planned < n, other=0)
    cached = tl.load(expert_to_entry + layer * EXPERTS + expert, planned < n, other=-1)
    for row in tl.range(n):
        entry = tl.sum(tl.where(planned == row, cached, 0), axis=0)
        if entry >= 0:
            tl.store(usage + entry, step)
    tl.debug_barrier()
    score = tl.load(usage + entries, mask=entries < SLOTS, other=9223372036854775807).to(tl.int64)
    for row in tl.range(n):
        e = tl.load(fetched + row)
        key = layer * EXPERTS + e
        entry = tl.load(expert_to_entry + key)
        hit = entry >= 0
        if SAVE and not hit and tl.load(skip + e) == 0:
            victim = tl.argmin(score, axis=0).to(tl.int32)
            score = tl.where(entries == victim, step, score)
            old = tl.load(keys + victim)
            if old >= 0:
                tl.store(expert_to_entry + old, -1)
            tl.store(keys + victim, key)
            tl.store(usage + victim, step)
            tl.store(expert_to_entry + key, victim)
            entry = victim
        tl.store(plan + row, entry)
        tl.store(plan + WIDTH + row, hit)
        tl.store(hit_trace + layer * WIDTH + row, hit)


class CompactTail:
    """Reuse the prefill buffers as a packed secondary tier during decode."""

    def __init__(self, primary):
        self.primary = primary
        self.cache = primary.cache
        self.device = primary.device
        self.size = primary.packed_experts
        self.keys = torch.full((self.size,), -1, device=self.device, dtype=torch.int32)
        self.usage = torch.zeros(self.size, device=self.device, dtype=torch.int64)
        self.step = torch.zeros((), device=self.device, dtype=torch.int64)
        self.mapping = torch.full_like(self.cache.slot_for_id, -1)
        self.plan = torch.empty((2, ROUTES), device=self.device, dtype=torch.int32)
        self.hit_trace = torch.empty((self.cache.num_layers, ROUTES), device=self.device, dtype=torch.int32)
        self.host_hits = torch.empty_like(self.hit_trace, device="cpu", pin_memory=True)
        self.slot_base = primary.reserve_rows
        self.id_of_slot = self.cache.id_of_slot[self.slot_base:]
        self.raw_usage = self.cache.usage[self.slot_base:]
        self.raw_slots = torch.full_like(self.cache.slot_for_id, -1)
        self.raw_step = torch.zeros_like(self.cache.step)
        self.raw_recency = torch.full_like(self.cache.expert_recency, -1)
        self.raw_views = tuple(bank[self.slot_base:] for _, bank in self.cache.banks)
        self.destinations = primary.pointer_tensor(self.raw_views)
        self.storage = torch.tensor([(buf.data_ptr(), stride) for buf, stride in
                                     zip(primary.tail_storage, primary.max_stride, strict=True)],
                                    device=self.device, dtype=torch.int64)
        self.skip = {}
        static = set(primary.transport.gpu_resident.keys["packed"]) | set(primary.transport.hot_experts)
        for layer in range(self.cache.num_layers):
            skip = torch.tensor([(layer, e) in static for e in range(self.cache.num_experts)], device=self.device)
            if layer in primary.layers:
                skip |= ~primary.cacheable[layer]
            self.skip[layer] = skip.to(torch.int32)

    def reset(self):
        self.keys.fill_(-1)
        self.usage.zero_()
        self.step.zero_()
        self.mapping.fill_(-1)
        self.id_of_slot.fill_(-1)
        self.raw_usage.zero_()
        self.raw_slots.fill_(-1)
        self.raw_step.zero_()
        self.raw_recency.fill_(-1)

    def ensure(self, layer, experts, weights, clamp):
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        cache = self.cache
        fields = ("_pool_id_of_slot", "_pool_usage", "slot_for_id", "step", "expert_recency")
        original = [getattr(cache, field) for field in fields]
        replacement = ([self.id_of_slot], [self.raw_usage], self.raw_slots, self.raw_step, self.raw_recency)
        try:
            for field, value in zip(fields, replacement, strict=True):
                setattr(cache, field, value)
            fraction = cache.hybrid_fetch_fraction[layer] if cache.hybrid_fetch_fraction is not None else 0.0
            return ensure_experts_hybrid(cache, layer, experts, weights, cache.hybrid_max_fetch, fraction, clamp=clamp)
        finally:
            for field, value in zip(fields, original, strict=True):
                setattr(cache, field, value)

    def fill(self, layer):
        cache, primary = self.cache, self.primary
        _tail_plan[(1,)](self.keys, self.usage, self.step, self.mapping, cache.src_indices, cache.num_indices,
                         self.skip[layer], self.plan, self.hit_trace, layer,
                         cache.num_experts, self.size, triton.next_power_of_2(self.size), ROUTES,
                         layer in primary.layers, num_warps=4)
        prefix = primary.prefix_ptrs.get(layer, next(iter(primary.prefix_ptrs.values())))
        primary.native.copy_compact_tail(primary.miss_sources[layer], self.destinations,
                                         cache._copy_feat_bytes[cache.layer_layout[layer]],
                                         prefix, self.storage, cache.src_indices, cache.evict_slots, cache.num_indices,
                                         self.plan, primary.sources[layer], cache.num_experts)

    def audit(self):
        checked = 0
        cache, primary = self.cache, self.primary
        for layer in sorted(primary.layers)[:2]:
            eligible = [e for e, skip in enumerate(self.skip[layer].cpu().tolist()) if not skip]
            for expert in (eligible[0], eligible[-1]):
                cache.src_indices[0] = expert
                cache.evict_slots[0] = 0
                cache.num_indices.fill_(1)
                for _ in range(2):
                    self.fill(layer)
                    for (sources, _), output in zip(cache.banks, self.raw_views, strict=True):
                        if not torch.equal(output[0].view(torch.int16).cpu(), sources[layer][expert].view(torch.int16)):
                            raise RuntimeError(f"Compact tier BF16 audit failed on layer {layer}, expert {expert}")
                        checked += output[0].numel()
                if not bool(self.hit_trace[layer, 0]):
                    raise RuntimeError("The audit did not use the packed tier on the final fetch")
        self.reset()
        primary.report["sampled_bf16_values_audited"] = checked
        primary.report_path.write_text(json.dumps(primary.report, indent=2) + "\n")
