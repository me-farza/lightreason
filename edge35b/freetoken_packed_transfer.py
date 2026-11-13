"""CUDA-graph capture of lossless PCIe fetches into FreeToken's BF16 slots."""

from math import prod
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from edge35b.bf16_expert_codec import LAYERS, pack_verified_row, plan_bank, row_bytes
from edge35b.width_autotune import (
    NATIVE_PACKED_TRANSPORT, PACKED_PRIMARY_TRANSPORT, SHARED_CPU_CONTROL_TRANSPORT, SHARED_CPU_TRANSPORT,
)


@triton.jit
def _planned_row(descriptors, destinations, src_indices, dst_indices, bank, ordinal, row_size,
                 EXPERTS: tl.constexpr):
    expert = tl.load(src_indices + ordinal)
    slot = tl.load(dst_indices + ordinal).to(tl.int64)
    row = descriptors + (bank * EXPERTS + expert) * 3
    src = tl.load(row)
    base = tl.load(row + 1).to(tl.int32)
    escapes = tl.load(row + 2).to(tl.int32)
    dst = tl.load(destinations + bank) + slot * row_size
    return src, dst, base, escapes


@triton.jit
def _copy_rows(descriptors, destinations, sizes, src_indices, dst_indices, count,
               EXPERTS: tl.constexpr, PAIRS: tl.constexpr, BLOCKS: tl.constexpr):
    bank = tl.program_id(1)
    size = tl.multiple_of(tl.load(sizes + bank).to(tl.int32), 4 * PAIRS)
    tiles = size // (4 * PAIRS)
    copies = tl.load(count).to(tl.int32)
    for tile in range(tl.program_id(0), tiles * copies, BLOCKS):
        src, dst, base, _ = _planned_row(descriptors, destinations, src_indices, dst_indices,
                                         bank, tile // tiles, size, EXPERTS)
        src = tl.multiple_of(src, 16)
        dst = tl.multiple_of(dst, 16)
        i = (tile % tiles) * PAIRS + tl.arange(0, PAIRS)
        if base < 0:
            bits = tl.load(tl.cast(src, tl.pointer_type(tl.uint32)) + i)
        else:
            # Each exponent byte encodes two BF16 values, with their sign and mantissa intact.
            packed = tl.cast(src, tl.pointer_type(tl.uint8))
            exp = tl.load(packed + size // 2 + i).to(tl.uint32)
            sm = tl.load(tl.cast(src, tl.pointer_type(tl.uint16)) + i).to(tl.uint32)
            lo = ((sm & 128) << 8) | (((exp & 15) + base) << 7) | (sm & 127)
            hi = (sm & 32768) | (((exp >> 4) + base) << 7) | ((sm >> 8) & 127)
            bits = lo | (hi << 16)
        tl.store(tl.cast(dst, tl.pointer_type(tl.uint32)) + i, bits)


@triton.jit
def _repair_rows(descriptors, destinations, sizes, src_indices, dst_indices, count,
                 EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    ordinal, bank = tl.program_id(0), tl.program_id(1)
    if ordinal < tl.load(count):
        size = tl.load(sizes + bank).to(tl.int32)
        src, dst, _, escapes = _planned_row(descriptors, destinations, src_indices, dst_indices,
                                             bank, ordinal, size, EXPERTS)
        tail = src + 3 * size // 4
        indices = tl.cast(tail, tl.pointer_type(tl.int32))
        values = tl.cast(tail + 4 * escapes, tl.pointer_type(tl.uint16))
        out = tl.cast(dst, tl.pointer_type(tl.uint16))
        for start in range(0, escapes, BLOCK):
            i = start + tl.arange(0, BLOCK)
            where = tl.load(indices + i, i < escapes, 0)
            value = tl.load(values + i, i < escapes, 0)
            tl.store(out + where, value, i < escapes)


@triton.jit(do_not_specialize=["layer"])
def _snapshot_fetched(fetched, count, src, layer, MAX_COPIES: tl.constexpr):
    index = tl.arange(0, MAX_COPIES)
    copies = tl.load(count)
    ids = tl.load(src + index, index < copies, other=-1)
    tl.store(fetched + layer * MAX_COPIES + index, ids)


@triton.jit
def _split_hot_misses(src, dst, count, hot_by_expert, limit,
                      host_src, host_dst, host_count, gpu_src, gpu_dst, gpu_count,
                      MAX_COPIES: tl.constexpr):
    ordinal = tl.arange(0, MAX_COPIES)
    valid = ordinal < tl.load(count)
    expert = tl.load(src + ordinal, valid, 0)
    slot = tl.load(dst + ordinal, valid, -1)
    resident = tl.load(hot_by_expert + expert, valid, -1)
    is_resident = (resident >= 0) & (resident < tl.load(limit))
    cold = valid & ~is_resident
    hot = valid & is_resident
    i = tl.cumsum(cold.to(tl.int32)) - 1
    j = tl.cumsum(hot.to(tl.int32)) - 1
    tl.store(host_src + i, expert, cold)
    tl.store(host_dst + i, slot, cold)
    tl.store(gpu_src + j, resident, hot)
    tl.store(gpu_dst + j, slot, hot)
    tl.store(host_count, tl.sum(cold.to(tl.int64)))
    tl.store(gpu_count, tl.sum(hot.to(tl.int64)))


@triton.jit
def _copy_raw_tiles(destinations, host_sources, gpu_sources, sizes, src_indices, dst_indices, count,
                    hot_by_expert, limit, STRIPES: tl.constexpr, TILE: tl.constexpr, HAS_HOT: tl.constexpr):
    stripe, bank, ordinal = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if ordinal < tl.load(count):
        expert = tl.load(src_indices + ordinal).to(tl.int64)
        slot = tl.load(dst_indices + ordinal).to(tl.int64)
        address = tl.load(host_sources + bank)
        if HAS_HOT:
            resident = tl.load(hot_by_expert + expert)
            if (resident >= 0) & (resident < tl.load(limit)):
                address = tl.load(gpu_sources + bank)
                expert = resident.to(tl.int64)
        # Both Qwen BF16 banks have a whole number of tiles per row.
        words = tl.multiple_of(tl.load(sizes + bank).to(tl.int32) // 4, TILE)
        source = tl.cast(tl.multiple_of(address, 16), tl.pointer_type(tl.uint32)) + expert * words
        dest = tl.cast(tl.multiple_of(tl.load(destinations + bank), 16), tl.pointer_type(tl.uint32)) + slot * words
        columns = tl.arange(0, TILE)
        for offset in range(stripe * TILE, words, STRIPES * TILE):
            tl.store(dest + offset + columns, tl.load(source + offset + columns))


class ExpertFetchTrace:
    """Read back the existing miss plan in the separate instrumented graphs."""

    def __init__(self, layers, width, device):
        self.max_copies = width * 8
        self.device_ids = torch.empty((layers, self.max_copies), device=device, dtype=torch.int32)
        self.host_ids = torch.empty((layers, self.max_copies), pin_memory=True, dtype=torch.int32)

    def capture(self, layer, cache):
        _snapshot_fetched[(1,)](self.device_ids, cache.num_indices, cache.src_indices, layer, self.max_copies,
                                num_warps=1)

    def copy_to_host(self):
        self.host_ids.copy_(self.device_ids, non_blocking=True)

    def report(self, stamps, payload_by_expert, raw_expert_bytes):
        ids_by_layer = []
        bytes_by_layer = []
        for layer, (row, stamp) in enumerate(zip(self.host_ids.tolist(), stamps, strict=True)):
            count = stamp[7]
            ids = row[:count]
            if not (0 <= count <= self.max_copies and len(set(ids)) == count
                    and all(0 <= e < 256 for e in ids) and all(e == -1 for e in row[count:])):
                raise ValueError(f"Invalid fetched expert IDs for layer {layer}: {row}, count {count}")
            ids_by_layer.append(ids)
            bytes_by_layer.append(count * raw_expert_bytes if payload_by_expert is None
                                  else sum(int(payload_by_expert[layer, e]) for e in ids))
        return {
            "expert_fetch_ids": ids_by_layer,
            "expert_pcie_payload_bytes": bytes_by_layer,
            "expert_gpu_source_bytes": [
                sum(1 for e in ids if payload_by_expert is not None and payload_by_expert[layer, e] == 0)
                * raw_expert_bytes for layer, ids in enumerate(ids_by_layer)
            ],
        }


class PackedBf16Transport:
    """Keep raw banks for the CPU, with a compact second source for selected GPU copies."""

    def __init__(self, cache, hot_plan=None, *, native_geometry=None, prepare_native_banks=True):
        from freetoken.kernel.pinned import device_ptr
        from freetoken.moe.host_banks import HostBank, HostResidency

        shapes = ((256, 1024, 2048), (256, 2048, 512))
        if (cache.decode_target != "hybrid" or not cache._copy_fused_ok or cache._unpinned_layers
                or cache.num_layers != 40 or cache.num_experts != 256
                or tuple(cache.bank_schema) != ("gate_up", "down")):
            raise ValueError("Packed BF16 transport requires the pinned Qwen3.6 hybrid expert cache")
        for layer in range(cache.num_layers):
            for (banks, _), shape in zip(cache.banks, shapes, strict=True):
                source = banks[layer]
                if source.shape != shape or source.dtype != torch.bfloat16 or not source.is_contiguous():
                    raise ValueError(f"Unsupported BF16 expert bank for layer {layer}")
        # One-percent escapes cap the packed payload at 78% of raw. Allow room
        # for alignment, one row's scratch, and the rest of the process.
        from edge35b.tune_hot_expert_cache import read_plan

        self.hot_base, hot_experts = read_plan(hot_plan) if hot_plan else (None, ())
        self.native = None
        self.native_geometry = native_geometry
        self.capture_native = False
        self.capture_shared = False
        self.capture_shared_control = False
        self.shared_geometry = None
        self.shared_descriptors = {}
        self.shared_payload_by_expert = None
        self.shared_control_layers = frozenset()
        self.shared_control_payload_by_expert = None
        self.gpu_resident = None
        self.packed_primary = None
        self.capture_packed_primary = False
        if native_geometry is not None:
            if self.hot_base != "raw":
                raise ValueError("Native packed verification compares against the raw hot-source plan")
            from edge35b.native_bf16_gather import NativeBf16Gather

            self.native = NativeBf16Gather()
        layers = LAYERS if self.hot_base != "raw" or (self.native is not None and prepare_native_banks) else ()
        bound = sum(prod(shape) for shape in shapes) * 2 * len(layers) * 79 // 100
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        if available < bound + 8 * 2**30:
            raise RuntimeError("Insufficient host RAM for packed expert banks plus an 8 GiB reserve")

        self.raw_expert_bytes = sum(2 * prod(shape[1:]) for shape in shapes)
        self.payload_by_expert = np.full((cache.num_layers, cache.num_experts), self.raw_expert_bytes, dtype=np.int64)
        self.native_payload_by_expert = self.payload_by_expert.copy() if self.native is not None else None
        packed_payload = self.payload_by_expert if self.native_payload_by_expert is None else self.native_payload_by_expert
        self.storage = []
        self.descriptors = {}
        self.repair_layers = set()
        self.hot_descriptors = {}
        self.hot_by_expert = {}
        self.hot_storage = []
        self.hot_payload_by_expert = None
        self.hot_experts = hot_experts
        self.hot_limit = None
        self.gather_geometry = None
        for layer in layers:
            packed_payload[layer].fill(0)
            descriptors = []
            for sources, _ in cache.banks:
                source = sources[layer]
                rows = source.view(torch.int16).numpy().view(np.uint16).reshape(cache.num_experts, -1)
                plans, size = plan_bank(rows)
                packed_address = 0
                if size:
                    bank = HostBank((size,), torch.uint8, backing="mmap")
                    payload = bank.tensor.numpy()
                    for row, plan in zip(rows, plans, strict=True):
                        if plan is not None:
                            offset, base, escapes = plan
                            pack_verified_row(row, payload[offset:offset + row_bytes(row.size, escapes)], base, escapes)
                    bank.pin()
                    if bank.residency != HostResidency.PINNED:
                        raise RuntimeError("Packed expert source needs CUDA-mapped host memory")
                    packed_address = device_ptr(bank.tensor)
                    self.storage.append(bank)
                raw_address = device_ptr(source)
                table = []
                for expert, plan in enumerate(plans):
                    if plan is None:
                        packed_payload[layer, expert] += rows[expert].nbytes
                        table.append((raw_address + expert * rows[expert].nbytes, -1, 0))
                    else:
                        offset, base, escapes = plan
                        packed_payload[layer, expert] += row_bytes(rows[expert].size, escapes)
                        table.append((packed_address + offset, base, escapes))
                        if escapes:
                            self.repair_layers.add(layer)
                descriptors.append(table)
            self.descriptors[layer] = torch.tensor(descriptors, device=cache.device, dtype=torch.int64)

        if hot_experts:
            self._prepare_hot_sources(cache, hot_experts)

    def _prepare_hot_sources(self, cache, experts):
        from freetoken.kernel.pinned import device_ptr
        free, _ = torch.cuda.mem_get_info(cache.device)
        required = len(experts) * self.raw_expert_bytes
        if free < required + 2**30:
            raise RuntimeError("Hot expert source needs its BF16 budget plus 1 GiB of free VRAM")
        self.hot_payload_by_expert = self.payload_by_expert.copy()
        by_layer = {}
        for slot, (layer, expert) in enumerate(experts):
            self.hot_payload_by_expert[layer, expert] = 0
            if self.native_payload_by_expert is not None:
                self.native_payload_by_expert[layer, expert] = 0
            by_layer.setdefault(layer, []).append((expert, slot))
        for sources, _ in cache.banks:
            shape = sources[0].shape[1:]
            bank = torch.empty((len(experts), *shape), dtype=torch.bfloat16, device=cache.device)
            for slot, (layer, expert) in enumerate(experts):
                bank[slot].copy_(sources[layer][expert], non_blocking=True)
            for slot, (layer, expert) in enumerate(experts):
                if not torch.equal(bank[slot].view(torch.int16).cpu(), sources[layer][expert].view(torch.int16)):
                    raise RuntimeError(f"Hot BF16 row failed its bit audit: layer {layer}, expert {expert}")
            self.hot_storage.append(bank)
        if self.hot_base == "raw":
            self.hot_limit = torch.full((1,), len(experts), device=cache.device, dtype=torch.int32)
            self.hot_source_ptrs = torch.tensor(
                [bank.data_ptr() for bank in self.hot_storage], device=cache.device, dtype=torch.int64,
            )
            self.hot_split = tuple(
                (torch.empty(64, device=cache.device, dtype=torch.int32),
                 torch.empty(64, device=cache.device, dtype=torch.int32),
                 torch.empty(1, device=cache.device, dtype=torch.int64))
                for _ in range(2)
            )
            for layer, rows in by_layer.items():
                lookup = [-1] * cache.num_experts
                for expert, slot in rows:
                    lookup[expert] = slot
                self.hot_by_expert[layer] = torch.tensor(lookup, device=cache.device, dtype=torch.int32)
                if self.native is not None and layer in self.descriptors:
                    # Native graphs use the same full set of audited hot replicas.
                    indices = [expert for expert, _ in rows]
                    for bank_index, bank in enumerate(self.hot_storage):
                        gpu_rows = [(bank[slot].data_ptr(), -1, 0) for _, slot in rows]
                        self.descriptors[layer][bank_index, indices] = torch.tensor(
                            gpu_rows, device=cache.device, dtype=torch.int64,
                        )
            return
        for layer, rows in by_layer.items():
            if layer in self.descriptors:
                descriptors = self.descriptors[layer].clone()
            else:
                raw = []
                for sources, _ in cache.banks:
                    bank = sources[layer]
                    addr = device_ptr(bank)
                    stride_bytes = bank[0].numel() * bank.element_size()
                    raw.append([(addr + expert * stride_bytes, -1, 0) for expert in range(cache.num_experts)])
                descriptors = torch.tensor(raw, device=cache.device, dtype=torch.int64)
            indices = [expert for expert, _ in rows]
            for bank_index, bank in enumerate(self.hot_storage):
                # A negative base means a raw BF16 row regardless of the source address.
                table = [(bank[slot].data_ptr(), -1, 0) for _, slot in rows]
                descriptors[bank_index, indices] = torch.tensor(table, device=cache.device, dtype=torch.int64)
            self.hot_descriptors[layer] = descriptors

    def set_hot_limit(self, limit):
        if self.hot_base != "raw":
            if limit is not None:
                raise ValueError("A request-level hot expert limit requires raw base transport")
            return
        limit = len(self.hot_experts) if limit is None else limit
        if not 1 <= limit <= len(self.hot_experts):
            raise ValueError("Hot expert limit exceeds the prepared source")
        self.hot_limit.fill_(limit)

    def install_shared_sources(self, cache, read_banks, layers, geometry, *, control_layers=()):
        from edge35b.cpu_read_packing import TILE_VALUES
        from freetoken.kernel.pinned import device_ptr

        if self.native is None or not layers or self.hot_base != "raw":
            raise ValueError("Shared CPU and CUDA reads need the native CUDA transport and hot replicas.")
        if control_layers and not set(LAYERS) <= set(control_layers) <= set(layers):
            raise ValueError("The shared control must cover the original native CUDA layers.")
        self.shared_geometry = geometry
        self.shared_payload_by_expert = self.native_payload_by_expert.copy()
        hot = {expert_key: slot for slot, expert_key in enumerate(self.hot_experts)}
        for layer in layers:
            descriptors = []
            self.shared_payload_by_expert[layer].fill(0)
            for bank, (sources, _) in enumerate(cache.banks):
                raw = sources[layer]
                row_values = raw[0].numel()
                row_bytes = row_values * 2
                row_tiles = row_values // TILE_VALUES
                packed_row_bytes = row_values * 3 // 2
                raw_base = device_ptr(raw)
                prepared = read_banks.shared.get((layer, bank))
                packed_base, bases_addr, encoded = 0, 0, [0] * cache.num_experts
                if prepared is not None:
                    owner, bases, encoded = prepared
                    packed_base = device_ptr(owner.tensor)
                    bases_addr = bases.data_ptr()
                table = []
                for expert, count in enumerate(encoded):
                    raw_address = raw_base + expert * row_bytes
                    slot = hot.get((layer, expert))
                    if slot is not None:
                        table.append((0, 0, self.hot_storage[bank][slot].data_ptr()))
                    elif count:
                        table.append((packed_base + expert * packed_row_bytes, bases_addr + expert * row_tiles,
                                      raw_address))
                        self.shared_payload_by_expert[layer, expert] += row_bytes - count * (TILE_VALUES // 2)
                    else:
                        table.append((0, 0, raw_address))
                        self.shared_payload_by_expert[layer, expert] += row_bytes
                descriptors.append(table)
            self.shared_descriptors[layer] = torch.tensor(descriptors, device=cache.device, dtype=torch.int64)
        if control_layers:
            self.shared_control_layers = frozenset(control_layers)
            self.shared_control_payload_by_expert = self.native_payload_by_expert.copy()
            indices = list(control_layers)
            self.shared_control_payload_by_expert[indices] = self.shared_payload_by_expert[indices]

    def hot_payload(self, limit):
        if limit is None or limit == len(self.hot_experts):
            return self.hot_payload_by_expert
        payload = self.payload_by_expert.copy()
        for layer, expert in self.hot_experts[:limit]:
            payload[layer, expert] = 0
        return payload

    def profile_payload(self, transport, hot_limit):
        if transport == SHARED_CPU_CONTROL_TRANSPORT:
            return self.shared_control_payload_by_expert
        if transport in {SHARED_CPU_TRANSPORT, PACKED_PRIMARY_TRANSPORT}:
            if self.gpu_resident is not None:
                return self.gpu_resident.payload[self.gpu_resident.mode]
            return self.shared_payload_by_expert
        if transport == NATIVE_PACKED_TRANSPORT:
            return self.native_payload_by_expert
        if transport == "hot":
            return self.hot_payload(hot_limit)
        return self.payload_by_expert if transport == "packed" else None

    def _copy_raw_with_hot(self, cache, layer, max_copies, lookup):
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        (h_src, h_dst, h_count), (g_src, g_dst, g_count) = self.hot_split
        _split_hot_misses[(1,)](cache.src_indices, cache.evict_slots, cache.num_indices, lookup, self.hot_limit,
                               h_src, h_dst, h_count, g_src, g_dst, g_count,
                               triton.next_power_of_2(max_copies), num_warps=1)
        dst = cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer]]
        feature_bytes = cache._copy_feat_bytes[cache.layer_layout[layer]]
        blocks = cache._copy_blocks_per_bank
        fast_index_copy_multi_jit(dst, cache._copy_src_ptrs[layer], feature_bytes, h_dst, h_src, h_count,
                                  blocks_per_bank=blocks)
        fast_index_copy_multi_jit(dst, self.hot_source_ptrs, feature_bytes, g_dst, g_src, g_count,
                                  blocks_per_bank=blocks)

    def copy_missing(self, cache, layer, max_copies, *, hot=False):
        lookup = self.hot_by_expert.get(layer) if hot else None
        descriptors = self.hot_descriptors.get(layer) if hot else None
        native_geometry = self.native_geometry
        shared_layer = self.capture_shared and (not self.capture_shared_control or layer in self.shared_control_layers)
        resident = self.gpu_resident if shared_layer and not self.capture_shared_control else None
        if resident is not None and layer in resident.tables:
            descriptors = resident.tables[layer]
            native_geometry = (*self.shared_geometry[:2], "resident")
        elif shared_layer and layer in self.shared_descriptors:
            descriptors = self.shared_descriptors[layer]
            native_geometry = self.shared_geometry
        elif descriptors is None and (self.hot_base != "raw" or self.capture_native):
            descriptors = self.descriptors.get(layer)
        if self.gather_geometry is None and descriptors is None and lookup is None:
            return False
        supported_counts = (32, 48, 64) if self.capture_shared else (32, 64)
        if cache._pending_src_layer != layer or cache._pending_whole_layer or max_copies not in supported_counts:
            raise RuntimeError("Packed transport received an unsupported hybrid copy plan")
        if self.capture_native and descriptors is not None:
            self.native.copy(descriptors, cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer]],
                             cache._copy_feat_bytes[cache.layer_layout[layer]], cache.src_indices,
                             cache.evict_slots, cache.num_indices, max_copies=max_copies,
                             experts=cache.num_experts, geometry=native_geometry)
            return True
        if self.gather_geometry is not None:
            stripes, tile = self.gather_geometry
            _copy_raw_tiles[(stripes, 2, max_copies)](
                cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer]], cache._copy_src_ptrs[layer],
                self.hot_source_ptrs, cache._copy_feat_bytes[cache.layer_layout[layer]], cache.src_indices,
                cache.evict_slots, cache.num_indices, lookup, self.hot_limit, stripes, tile, lookup is not None,
                num_warps=4,
            )
            return True
        if lookup is not None:
            self._copy_raw_with_hot(cache, layer, max_copies, lookup)
            return True
        args = (descriptors, cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer]],
                cache._copy_feat_bytes[cache.layer_layout[layer]], cache.src_indices, cache.evict_slots, cache.num_indices)
        blocks = 128
        _copy_rows[(blocks, 2)](*args, cache.num_experts, 4096, blocks, num_warps=8)
        if layer in self.repair_layers:
            _repair_rows[(max_copies, 2)](*args, cache.num_experts, 256, num_warps=4)
        return True
