"""Exact GPU replicas for the existing BF16 expert fetch plan."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from edge35b.cpu_read_packing import TILE_VALUES
from edge35b.download import available_memory
from edge35b.plan_gpu_resident_cache import read_plan


MIB = 2**20
GIB = 2**30
RAW_TILE_BYTES = 2 * TILE_VALUES
PACKED_TILE_BYTES = 3 * TILE_VALUES // 2
GEOMETRY = (4, 256, "resident")


def encode(source):
    """Pack consecutive tiles with inline raw fallbacks and prefix offsets."""
    from freetoken.kernel import _cpu_moe

    raw = source.view(torch.int16).numpy().view(np.uint8).reshape(-1, RAW_TILE_BYTES)
    packed = np.empty((len(raw), PACKED_TILE_BYTES), dtype=np.uint8)
    bases = np.empty(len(raw), dtype=np.uint8)
    encoded = _cpu_moe.pack_cpu_read_tiles(source.data_ptr(), packed.ctypes.data, bases.ctypes.data, len(raw))
    compressed = bases != 255
    raw_before = np.cumsum(~compressed, dtype=np.uint32) - (~compressed)
    headers = (raw_before << 8) | bases
    lengths = np.where(compressed, PACKED_TILE_BYTES, RAW_TILE_BYTES)
    size = int(lengths.sum())
    if size + headers.nbytes >= raw.nbytes:
        return raw.reshape(-1), None, 0, len(raw)
    if encoded != int(compressed.sum()):
        raise RuntimeError("The BF16 codec's base and tile counts disagree")
    tiles = np.empty_like(raw)
    tiles[compressed, :PACKED_TILE_BYTES] = packed[compressed]
    tiles[~compressed] = raw[~compressed]
    return tiles[np.arange(RAW_TILE_BYTES) < lengths[:, None]], headers.view(np.uint8), encoded, len(raw) - encoded


def device_rows(weights, raw):
    """Build descriptors for independently owned compact device rows."""
    rows, owners = [], []
    total = 0
    for i, source in enumerate(weights):
        data, meta, _, _ = encode(source)
        if meta is None:
            rows.append((0, 0, raw[i].data_ptr()))
            total += source.nbytes
        else:
            d, h = (torch.from_numpy(x).to(raw.device) for x in (data, meta))
            owners.extend((d, h))
            rows.append((d.data_ptr(), h.data_ptr(), -1))
            total += data.size + meta.size
    return torch.tensor(rows, dtype=torch.int64, device=raw.device), owners, total


class GpuResidentExperts:
    """One physical GPU arena with two audited, interchangeable layouts."""

    def __init__(self, cache, transport, plan, requested_mib, report_path):
        from freetoken.kernel.pinned import device_ptr

        start = perf_counter()
        self.cache, self.native = cache, transport.native
        self.device = cache.device
        self.manifest_path = Path(report_path)
        self.load_log = self.manifest_path.with_suffix(".loads.jsonl")
        self.mode = None
        keys = read_plan(plan, transport.hot_experts)
        self.raw_bytes = transport.raw_expert_bytes
        self.free_before = torch.cuda.mem_get_info(self.device)[0]
        # Reserve space for the four capture pools and the ordinary request
        # workspaces. The pool is sized once, independently of the prompt.
        self.capacity = min(requested_mib * MIB, self.free_before - 512 * MIB) // (16 * MIB) * (16 * MIB)
        if self.capacity < 128 * MIB or available_memory() < self.capacity + 8 * GIB:
            raise MemoryError("Insufficient spare GPU or pinned host memory for the resident trial")
        self.buffer = torch.empty(self.capacity, dtype=torch.uint8, device=self.device)
        self.packed_host = torch.empty(self.capacity, dtype=torch.uint8, pin_memory=True)
        self.base = self.buffer.data_ptr()
        self.keys = {"raw": keys[:self.capacity // self.raw_bytes], "packed": []}
        self.layouts = {"raw": {}, "packed": {}}
        self.payload = {mode: transport.shared_payload_by_expert.copy() for mode in self.keys}

        used, encoded, fallback = 0, 0, 0
        for key in keys:
            projections = [encode(self.source(key, bank)) for bank in range(len(cache.banks))]
            layout = []
            end = used
            for data, headers, _, _ in projections:
                data_offset = end
                end += data.size
                meta_offset = end if headers is not None else None
                if headers is not None:
                    end += headers.size
                end = (end + 127) // 128 * 128
                layout.append((data_offset, meta_offset))
            if end > self.capacity:
                break
            for (offset, meta), (data, headers, n_packed, n_raw) in zip(layout, projections, strict=True):
                self.packed_host[offset:offset + data.size].copy_(torch.from_numpy(data))
                if headers is not None:
                    self.packed_host[meta:meta + headers.size].copy_(torch.from_numpy(headers))
                encoded += n_packed
                fallback += n_raw
            used = end
            self.layouts["packed"][key] = layout
            self.keys["packed"].append(key)
        self.used = {"raw": len(self.keys["raw"]) * self.raw_bytes, "packed": used}
        if not set(self.keys["raw"]) < set(self.keys["packed"]):
            raise RuntimeError("Packed cache did not fit more rows than its equal-budget raw control")

        offset = 0
        for key in self.keys["raw"]:
            self.layouts["raw"][key] = []
            for bank in range(len(cache.banks)):
                self.layouts["raw"][key].append((offset, None))
                offset += self.source(key, bank).numel() * 2

        layers = sorted({layer for layer, _ in self.keys["packed"]})
        hot = {key: slot for slot, key in enumerate(transport.hot_experts)}
        base_tables = {}
        for layer in layers:
            table = transport.shared_descriptors.get(layer)
            if table is not None:
                base_tables[layer] = table.cpu()
                continue
            rows = []
            for bank, (sources, _) in enumerate(cache.banks):
                source = sources[layer]
                base = device_ptr(source)
                stride = source[0].numel() * source.element_size()
                rows.append([(0, 0, transport.hot_storage[bank][hot[layer, e]].data_ptr()
                              if (layer, e) in hot else base + e * stride) for e in range(cache.num_experts)])
            base_tables[layer] = torch.tensor(rows, dtype=torch.int64)
        self.tables = {layer: table.to(self.device) for layer, table in base_tables.items()}
        self.host_tables = {}
        for mode, layout in self.layouts.items():
            tables = {layer: table.clone() for layer, table in base_tables.items()}
            for (layer, expert), placements in layout.items():
                for bank, (offset, meta) in enumerate(placements):
                    pointers = ((0, 0, self.base + offset) if meta is None
                                else (self.base + offset, self.base + meta, -1))
                    tables[layer][bank, expert] = torch.tensor(pointers, dtype=torch.int64)
                self.payload[mode][layer, expert] = 0
            self.host_tables[mode] = tables

        audit = {}
        for mode in ("raw", "packed"):
            self.select(mode)
            audit[mode] = self.audit()
        self.report = {"requested_mib": requested_mib, "arena_bytes": self.capacity,
                       "free_gpu_bytes_before_arena": self.free_before, "used_bytes": self.used,
                       "experts": self.keys, "packed_tiles": encoded, "raw_tiles": fallback,
                       "bf16_values_audited": audit, "setup_s": perf_counter() - start}
        self.save_report()

    def source(self, key, bank):
        layer, expert = key
        return self.cache.banks[bank][0][layer][expert]

    def select(self, mode):
        if mode == self.mode:
            return
        if self.packed_host is None:
            raise RuntimeError("The fixed-prefix benchmark keeps the replica layout packed")
        start = perf_counter()
        if mode == "packed":
            size = self.used[mode]
            self.buffer[:size].copy_(self.packed_host[:size], non_blocking=True)
        else:
            for key, placements in self.layouts[mode].items():
                for bank, (offset, _) in enumerate(placements):
                    raw = self.source(key, bank).view(torch.uint8).reshape(-1)
                    self.buffer[offset:offset + raw.numel()].copy_(raw, non_blocking=True)
        for layer, table in self.tables.items():
            table.copy_(self.host_tables[mode][layer], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        self.mode = mode
        with self.load_log.open("a") as stream:
            stream.write(json.dumps({"mode": mode, "bytes": self.used[mode], "seconds": perf_counter() - start}) + "\n")

    def audit(self):
        """Reconstruct every loaded projection with the production Ada kernel."""
        mode = self.mode
        key = self.keys[mode][0]
        output = [torch.empty_like(self.source(key, bank), device=self.device) for bank in range(len(self.cache.banks))]
        destinations = torch.tensor([t.data_ptr() for t in output], device=self.device, dtype=torch.int64)
        sizes = torch.tensor([t.numel() * t.element_size() for t in output], device=self.device, dtype=torch.int64)
        ids = torch.zeros(1, device=self.device, dtype=torch.int32)
        count = torch.ones(1, device=self.device, dtype=torch.int64)
        descriptors = torch.empty((len(output), 1, 3), device=self.device, dtype=torch.int64)
        values = 0
        for layer, expert in self.keys[mode]:
            descriptors.copy_(self.host_tables[mode][layer][:, expert:expert + 1])
            self.native.copy(descriptors, destinations, sizes, ids, ids, count,
                             max_copies=1, experts=1, geometry=GEOMETRY)
            for bank, reconstructed in enumerate(output):
                if not torch.equal(reconstructed.view(torch.int16).cpu(), self.source((layer, expert), bank).view(torch.int16)):
                    raise RuntimeError(f"GPU {mode} weight audit failed at layer {layer}, expert {expert}, bank {bank}")
                values += reconstructed.numel()
        return values

    def save_report(self):
        self.manifest_path.write_text(json.dumps(self.report, indent=2) + "\n")
