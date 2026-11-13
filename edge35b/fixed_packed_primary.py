"""Pack physical BF16 rows while retaining the frozen logical cache planner."""

import json

import torch
import triton
import triton.language as tl

from edge35b.gpu_resident_experts import GEOMETRY, encode
from edge35b.packed_primary_cache import ROUTES
from edge35b.plan_gpu_resident_cache import read_plan


@triton.jit(do_not_specialize=["layer"])
def _plan_fixed_prefix(slots, cpu_ids, slot_to_id, evicted, fetched, eligible, packed_ids,
                       plan, count, gemm_slots, refill_trace, layer,
                       PACKED: tl.constexpr, SCRATCH: tl.constexpr, EXPERTS: tl.constexpr,
                       WIDTH: tl.constexpr, PROFILE: tl.constexpr):
    i = tl.arange(0, WIDTH)
    logical_slot = tl.load(slots + i)
    on_gpu = tl.load(cpu_ids + i) == -1
    identity = tl.load(slot_to_id + logical_slot, mask=on_gpu, other=layer * EXPERTS)
    expert = identity - layer * EXPERTS
    first_route = tl.min(tl.where((logical_slot[:, None] == logical_slot[None, :]) & on_gpu[None, :],
                                 i[None, :], WIDTH), axis=1)
    first = on_gpu & (first_route == i)
    index = tl.cumsum(first.to(tl.int32)) - 1
    row = tl.sum(tl.where(first_route[:, None] == i[None, :], index[None, :], 0), axis=1)

    prefix = on_gpu & (logical_slot < PACKED)
    pack = prefix & tl.load(eligible + expert, mask=prefix, other=False)
    packed_id = tl.load(packed_ids + logical_slot, mask=pack & first, other=-1)
    inserted = tl.load(evicted + i, i < tl.load(fetched), other=-3)
    miss = tl.sum((logical_slot[:, None] == inserted[None, :]).to(tl.int32), axis=1) > 0
    refill = miss | (prefix & (packed_id != identity))
    tl.store(packed_ids + logical_slot, identity, mask=pack & first)
    tl.store(plan + index, expert, mask=first)
    tl.store(plan + WIDTH + index, tl.where(prefix & ~pack, -1, logical_slot), mask=first)
    tl.store(plan + 2 * WIDTH + index, refill, mask=first)
    tl.store(count, tl.sum(first.to(tl.int32)))
    tl.store(gemm_slots + i, tl.where(on_gpu, tl.where(prefix, row + 1, SCRATCH + logical_slot - PACKED), 0))
    if PROFILE:
        tl.store(refill_trace + layer * WIDTH + i, tl.where(first & refill, expert, -1))


class FixedPackedPrefix:
    def __init__(self, primary, replica_plan):
        self.primary = primary
        self.cache = primary.cache
        self.device = primary.device
        self.packed_ids = torch.full((primary.reserve_rows,), -1, device=self.device, dtype=torch.int32)
        self.ineligible = torch.zeros(primary.raw_experts, device=self.device, dtype=torch.bool)
        self.trace = torch.empty((self.cache.num_layers, ROUTES), device=self.device, dtype=torch.int32)
        self.host_trace = torch.empty_like(self.trace, device="cpu", pin_memory=True)
        self._prepare_replicas(replica_plan)

    def _prepare_replicas(self, replica_plan):
        primary = self.primary
        transport = primary.transport
        resident = transport.gpu_resident
        used = [0] * len(primary.replica_space)
        space = [arena.nbytes for arena in primary.replica_space]
        if sum(space) > resident.packed_host.nbytes:
            raise ValueError("The fixed prefix exceeds the existing replica upload buffer")
        # After the ordinary replicas are installed, their GPU arena remains
        # packed throughout this A/B. Reuse the established pinned buffer.
        self.upload_buffer = resident.packed_host
        host_replicas = list(self.upload_buffer[:sum(space)].split(space))
        self.payload = resident.payload["packed"].copy()
        existing = set(transport.hot_experts) | set(resident.keys["packed"])
        self.replica_keys = []
        self.refill_tables = {layer: table.clone() for layer, table in primary.miss_sources.items()}
        print(f"Lightreason fixed primary replica budget: {sum(space) / 2**20:.0f} MiB", flush=True)
        for key in read_plan(replica_plan, transport.hot_experts):
            if key in existing:
                continue
            layer, expert = key
            rows = [encode(sources[layer][expert]) for sources, _ in self.cache.banks]
            end = [(offset + data.size + (0 if meta is None else meta.size) + 127) // 128 * 128
                   for offset, (data, meta, _, _) in zip(used, rows, strict=True)]
            if any(last > size for last, size in zip(end, space, strict=True)):
                break
            for bank, ((data, meta, _, _), offset) in enumerate(zip(rows, used, strict=True)):
                host = host_replicas[bank]
                base = primary.replica_space[bank].data_ptr()
                host[offset:offset + data.size].copy_(torch.from_numpy(data))
                if meta is None:
                    desc = (0, 0, base + offset)
                else:
                    start = offset + data.size
                    host[start:start + meta.size].copy_(torch.from_numpy(meta))
                    desc = (base + offset, base + start, -1)
                self.refill_tables[layer][bank, expert] = torch.tensor(desc, device=self.device, dtype=torch.int64)
            used = end
            self.replica_keys.append(key)
            self.payload[layer, expert] = 0
        self.replica_keys = frozenset(self.replica_keys)
        print(f"Lightreason fixed primary encoded {len(self.replica_keys)} extra replicas", flush=True)
        self.uploads = [(arena[:n], host[:n])
                        for arena, host, n in zip(primary.replica_space, host_replicas, used, strict=True)]
        if resident.mode != "packed":
            raise RuntimeError("The fixed prefix requires the packed replicas to be loaded before capture")
        resident.packed_host = None
        primary.report.update(replica_space_bytes=sum(space), replica_upload_bytes=sum(used),
                              replica_upload_buffer_reused=True,
                              extra_replicas=sorted(self.replica_keys),
                              prefill_upload_each_request=True)

    def reset(self):
        self.packed_ids.fill_(-1)

    def begin_decode(self):
        # Routed prefill can write past the staging prefix. Load these rows
        # after prefill and before the first decode graph for this request.
        for device, host in self.uploads:
            device.copy_(host, non_blocking=True)
        for view in self.primary.views:
            view[0].zero_()

    def stage(self, layer, slots, cpu_ids, profile):
        primary, cache = self.primary, self.cache
        _plan_fixed_prefix[(1,)](slots, cpu_ids, cache.id_of_slot, cache.evict_slots, cache.num_indices,
                                 primary.cacheable.get(layer, self.ineligible), self.packed_ids,
                                 primary.plan, primary.count, primary.gemm_slots, self.trace, layer,
                                 primary.reserve_rows, primary.scratch, cache.num_experts, ROUTES, profile, num_warps=4)
        prefix = primary.prefix_ptrs.get(layer, next(iter(primary.prefix_ptrs.values())))
        primary.native.stage_primary(primary.sources[layer], self.refill_tables[layer], primary.layouts,
                                     prefix, primary.plan, primary.count, experts=cache.num_experts,
                                     packed_slots=primary.reserve_rows)
        return tuple(primary.views), primary.gemm_slots

    def copy_profile(self):
        self.host_trace.copy_(self.trace, non_blocking=True)

    def adjust_profile(self, info, raw_expert_bytes):
        refills = [[e for e in row if e >= 0] for row in self.host_trace.tolist()]
        info["primary_refill_ids"] = refills
        info["primary_additional_refills"] = [len(set(actual) - set(logical)) for actual, logical in
                                              zip(refills, info["expert_fetch_ids"], strict=True)]
        info["extra_replica_hits"] = [sum((layer, e) in self.replica_keys for e in ids)
                                     for layer, ids in enumerate(refills)]
        info["expert_pcie_payload_bytes"] = [sum(int(self.payload[layer, e]) for e in ids)
                                             for layer, ids in enumerate(refills)]
        info["expert_gpu_source_bytes"] = [sum(1 for e in ids if self.payload[layer, e] == 0) * raw_expert_bytes
                                           for layer, ids in enumerate(refills)]

    def audit(self):
        primary, cache = self.primary, self.cache
        self.begin_decode()
        layers = sorted(primary.layers)
        checked = 0
        for layer in sorted({0, layers[0], layers[len(layers) // 2], layers[-1]}):
            eligible = primary.cacheable.get(layer, self.ineligible).tolist()
            chosen = list(dict.fromkeys([0, 127, 255, *[e for e, fits in enumerate(eligible) if fits][:1]]))
            for expert in chosen:
                for slot in (0, primary.reserve_rows, primary.slots - 1):
                    cache.id_of_slot[slot] = layer * cache.num_experts + expert
                    slots = torch.full((8, 8), slot, device=self.device, dtype=torch.int32)
                    cpu_ids = torch.full_like(slots, -1)
                    cpu_ids[0, 0] = expert
                    cache.evict_slots[0] = slot
                    self.packed_ids.fill_(-1)
                    for fill in (1, 0):
                        cache.num_indices.fill_(fill)
                        views, gemm_slots = self.stage(layer, slots, cpu_ids, False)
                        out = 1 if slot < primary.reserve_rows else primary.scratch + slot - primary.reserve_rows
                        if gemm_slots[0, 0].item() != 0 or not bool((gemm_slots.reshape(-1)[1:] == out).all()):
                            raise RuntimeError("Fixed primary slot remapping failed")
                        for view, (sources, _) in zip(views, cache.banks, strict=True):
                            if not torch.equal(view[out].view(torch.int16).cpu(), sources[layer][expert].view(torch.int16)):
                                raise RuntimeError(f"Fixed primary BF16 audit failed for {(layer, expert, slot, fill)}")
                            checked += view[out].numel()
        # Exercise every extra replica using the same descriptors as decode.
        outputs = [view[1] for view in primary.views]
        dest = primary.pointer_tensor(outputs)
        sizes = torch.tensor([out.nbytes for out in outputs], device=self.device, dtype=torch.int64)
        ids = torch.zeros(1, device=self.device, dtype=torch.int32)
        count = torch.ones(1, device=self.device, dtype=torch.int64)
        for layer, expert in sorted(self.replica_keys):
            descriptor = self.refill_tables[layer][:, expert:expert + 1].contiguous()
            primary.native.copy(descriptor, dest, sizes, ids, ids, count, max_copies=1, experts=1, geometry=GEOMETRY)
            for out, (sources, _) in zip(outputs, cache.banks, strict=True):
                if not torch.equal(out.view(torch.int16).cpu(), sources[layer][expert].view(torch.int16)):
                    raise RuntimeError(f"Extra GPU replica BF16 audit failed for {(layer, expert)}")
                checked += out.numel()
        self.reset()
        cache.reset()
        primary.report["sampled_bf16_values_audited"] = checked
        primary.report_path.write_text(json.dumps(primary.report, indent=2) + "\n")
