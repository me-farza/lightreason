"""Prepare and audit the optional lossless AVX2 expert representation."""

from collections import Counter
import os
from weakref import finalize

from edge35b.download import GIB, REVISION, available_memory


TILE_VALUES = 256
CPU_READ_POLICIES = {
    "stock": 0,
    "raw-no-prefetch": 1,
    "raw-prefetch-256": 2,
    "raw-prefetch-1024": 3,
    "packed-gate": 4,
    "packed-down": 5,
    "packed-both": 6,
    "packed-both-16": 7,
    "packed-both-prefetch": 8,
}


def packed_banks(policy):
    mode = CPU_READ_POLICIES[policy]
    if mode < 4:
        return ()
    return (0,) if mode == 4 else (1,) if mode == 5 else (0, 1)


def development_passes(report):
    if (report["revision"] != REVISION or report["dtype"] != "bfloat16"
            or not report["transport_summary"]["exact"]):
        raise ValueError("Use the exact native CUDA profile.")
    return [step for row in report["rows"]
            if row["case"] == "studio" and row["scenario"] == "native-packed-hot-profile"
            for step in row["diagnostics"]["passes"]]


def rank_cpu_layers(report, count):
    if not 1 <= count <= 40:
        raise ValueError("Use a valid CPU layer budget.")
    reads = Counter()
    for step in development_passes(report):
        for layer, stamp in enumerate(step["gpu_layer_stamps"]):
            reads[layer] += stamp[8] - stamp[7]
    if len(reads) != 40 or reads.total() <= 0:
        raise ValueError("Expected the 40-layer development trace with CPU expert work.")
    layers = sorted(reads, key=lambda layer: (-reads[layer], layer))[:count]
    return layers, sum(reads[layer] for layer in layers) / reads.total()


def rank_shared_layers(report, cpu_layers, count):
    if not len(cpu_layers) <= count <= 40:
        raise ValueError("The shared plan must cover the CPU plan and fit the 40 layers.")
    traffic = Counter()
    for step in development_passes(report):
        for layer, size in enumerate(step["expert_pcie_payload_bytes"]):
            traffic[layer] += size
    if len(traffic) != 40 or traffic.total() <= 0:
        raise ValueError("Expected a development profile with PCIe weight traffic.")
    extra = [layer for layer, _ in traffic.most_common() if layer not in cpu_layers]
    return [*cpu_layers, *extra[:count - len(cpu_layers)]]


def release_studio_pins(registrations, owners):
    from freetoken.kernel.pinned import host_unregister
    from freetoken.moe.host_banks import refund_pin

    for address, size in reversed(registrations):
        host_unregister(address)
        refund_pin(size)
    for owner in owners:
        owner.release()


class PackedReadBanks:
    def __init__(self, layers):
        import numpy as np

        self.tables = np.zeros((4, layers), dtype=np.uint64)
        self.storage = []
        self.audit = []
        self.shared = {}
        self.layer_tables = {}
        self.studio_pins = None

    def pin_studio_bank(self, owner, row_bytes, chunk_rows=16):
        from freetoken.kernel.pinned import device_ptr, host_register
        from freetoken.moe.host_banks import charge_pin, refund_pin

        if row_bytes % 4096:
            raise ValueError("Packed expert rows must occupy whole host pages.")
        if self.studio_pins is None:
            self.studio_pins = ([], [])
            self.release_pins = finalize(self, release_studio_pins, *self.studio_pins)
            self.release_pins.atexit = False
        registrations, owners = self.studio_pins
        owners.append(owner)
        chunk = chunk_rows * row_bytes
        gpu_base = None
        for offset in range(0, owner.nbytes, chunk):
            size = min(chunk, owner.nbytes - offset)
            if available_memory() < 8 * GIB:
                raise MemoryError("Insufficient RAM to map the packed experts with an 8 GiB reserve.")
            address = owner.addr + offset
            charge_pin(size)
            try:
                host_register(address, size)
            except RuntimeError:
                refund_pin(size)
                raise
            registrations.append((address, size))
            mapped = device_ptr(owner.tensor[offset:offset + size])
            if gpu_base is None:
                gpu_base = mapped
            elif mapped != gpu_base + offset:
                raise RuntimeError("Packed expert mappings must be contiguous on the GPU.")

    def add(self, bank, layer, weights, *, use_cpu=True, gpu_device=None):
        import torch
        from freetoken.kernel import _cpu_moe

        if (weights.device.type != "cpu" or weights.dtype != torch.bfloat16
                or weights.shape[-1] not in (512, 2048) or not weights.is_contiguous()):
            raise ValueError("Expected contiguous Qwen CPU BF16 expert rows.")
        tiles = weights.numel() // TILE_VALUES
        size = weights.numel() * 3 // 2
        owner = None
        if gpu_device is None:
            packed = torch.empty(size, dtype=torch.uint8, device="cpu")
        else:
            from freetoken.moe.host_banks import HostBank

            owner = HostBank((size,), torch.uint8, backing="mmap")
            packed = owner.tensor
        bases = torch.empty(tiles, dtype=torch.uint8, device="cpu")
        encoded = _cpu_moe.pack_cpu_read_tiles(weights.data_ptr(), packed.data_ptr(), bases.data_ptr(), tiles)
        if encoded and use_cpu:
            self.storage.extend((packed, bases))
            self.tables[2 * bank, layer] = packed.data_ptr()
            self.tables[2 * bank + 1, layer] = bases.data_ptr()
        if encoded and owner is not None:
            from freetoken.moe.host_banks import HostResidency

            if os.getenv("LIGHTREASON_DRAFT_PREFIX") == "1":
                self.pin_studio_bank(owner, size // weights.shape[0], chunk_rows=4)
            elif os.getenv("LIGHTREASON_STUDIO_CACHE_PARK") == "1":
                self.pin_studio_bank(owner, size // weights.shape[0])
            else:
                owner.pin()
                if owner.residency != HostResidency.PINNED:
                    raise RuntimeError("The shared packed CPU bank needs CUDA-mapped host memory.")
            expert_tiles = (bases.view(weights.shape[0], -1) != 255).sum(dim=1).tolist()
            self.shared[layer, bank] = (owner, bases.to(gpu_device), expert_tiles)
        self.audit.append({"layer": layer, "bank": bank, "tiles": tiles, "encoded": encoded,
                           "allocation_bytes": packed.numel() + bases.numel() if encoded else 0})

    def configure(self, executor, layers=None):
        tables = self.tables
        if layers is not None:
            import numpy as np

            key = tuple(layers)
            if key not in self.layer_tables:
                subset = np.zeros_like(tables)
                subset[:, key] = tables[:, key]
                self.layer_tables[key] = subset
            tables = self.layer_tables[key]
        executor.configure_cpu_read_tables(*(table.ctypes.data for table in tables))


def prepare_cpu_reads(cache, executor, policy, layers, *, shared_layers=()):
    banks = packed_banks(policy)
    if not banks:
        if shared_layers:
            raise ValueError("Shared CPU and CUDA reads require both packed expert projections.")
        return None
    if (not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < cache.num_layers for layer in layers)
            or tuple(cache.bank_schema) != ("gate_up", "down") or cache.num_experts != 256):
        raise ValueError("Packed CPU reads require the pinned Qwen expert banks and a distinct layer plan.")
    if shared_layers and (banks != (0, 1) or len(set(shared_layers)) != len(shared_layers)
                          or not set(layers) <= set(shared_layers)
                          or any(not 0 <= layer < cache.num_layers for layer in shared_layers)):
        raise ValueError("Use a shared layer plan that covers both CPU projections.")
    if shared_layers:
        import torch

        gpu_bytes = sum(cache.banks[bank][0][layer].numel() // TILE_VALUES
                        for layer in shared_layers for bank in banks)
        if torch.cuda.mem_get_info(cache.device)[0] < gpu_bytes + GIB:
            raise MemoryError("Shared CPU tile metadata needs its budget plus 1 GiB of free VRAM.")
    packed = PackedReadBanks(cache.num_layers)
    for layer in shared_layers or layers:
        for bank in banks:
            source = cache.banks[bank][0][layer]
            size = source.numel() * 3 // 2 + source.numel() // TILE_VALUES
            if available_memory() < size + 8 * GIB:
                raise MemoryError("Insufficient host memory for the next CPU packed bank plus the 8 GiB reserve.")
            packed.add(bank, layer, source, use_cpu=layer in layers,
                       gpu_device=cache.device if shared_layers else None)
    packed.configure(executor)
    return packed
