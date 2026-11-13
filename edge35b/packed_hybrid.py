"""Run cached CUDA experts in two grouped launches alongside the cold CPU experts."""

import ctypes

import torch
import triton
import triton.language as tl

from edge35b.cpu_route import _kernel
from edge35b.fused_experts import _matvec


@triton.jit
def _combine(Table, Hot, Cold, Router, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0.0, tl.float32)
    for expert in tl.static_range(8):
        slot = tl.load(Table + 4 * expert + 2)
        cold_row = tl.load(Table + 4 * expert + 3)
        scale = tl.load(Router + slot).to(tl.float32)
        if cold_row >= 0:
            value = tl.load(Cold + cold_row * WIDTH + offsets).to(tl.float32)
        else:
            value = tl.load(Hot + expert * WIDTH + offsets).to(tl.float32)
        product = (value * scale).to(tl.bfloat16).to(tl.float32)
        result = (result + product).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + offsets, result)


class Workspace:
    def __init__(self, device):
        self.ids = torch.empty(8, dtype=torch.long, pin_memory=True)
        self.hidden = torch.empty((1, 2048), dtype=torch.bfloat16, pin_memory=True)
        self.host_table = torch.empty((8, 4), dtype=torch.long, pin_memory=True)
        self.table_view = (ctypes.c_int64 * 32).from_address(self.host_table.data_ptr())
        self.device_table = torch.empty_like(self.host_table, device=device)
        self.host_results = torch.empty((8, 2048), dtype=torch.bfloat16, pin_memory=True)
        self.cold_results = torch.empty_like(self.host_results, device=device)
        self.hot_results = torch.empty_like(self.host_results, device=device)
        self.gate_up = torch.empty((8, 1024), device=device, dtype=torch.bfloat16)
        self.inbound = torch.cuda.Event()


def packed_forward(layer, hidden, indices, weights):
    pool = layer.pool
    if pool.packed_workspace is None:
        pool.packed_workspace = Workspace(hidden.device)
    work = pool.packed_workspace
    work.ids.copy_(indices[0], non_blocking=True)
    work.hidden.copy_(hidden, non_blocking=True)
    work.inbound.record()
    work.inbound.synchronize()
    schedule = sorted(enumerate(work.ids.tolist()), key=lambda route: route[1])
    layer.decode_step += 1
    hot, gate_ptrs, down_ptrs = [], [], []
    for index, (slot, expert) in enumerate(schedule):
        key = pool._row_key(layer, expert)
        recent = layer.decode_step - layer.last_route.get(expert, -4) <= 4
        layer.last_route[expert] = layer.decode_step
        if key in pool._rows or recent:
            gate, down = pool._expert(layer, expert, cached=True)
            hot.extend((gate, down))
            row = (gate.data_ptr(), down.data_ptr(), slot, -1)
        else:
            row = (0, 0, slot, len(gate_ptrs))
            gate_ptrs.append(layer.gate_up[expert].data_ptr())
            down_ptrs.append(layer.down[expert].data_ptr())
        work.table_view[4 * index:4 * index + 4] = row
    work.device_table.copy_(work.host_table, non_blocking=True)
    if hot:
        _matvec[(128, 8)](work.device_table, hidden, work.gate_up, 2048, 1024, 0, 8, True, 4, num_warps=8)
        gate, up = work.gate_up.chunk(2, dim=-1)
        active = layer.act_fn(gate) * up
        _matvec[(256, 8)](work.device_table, active, work.hot_results, 512, 2048, 1, 8, False, 4, num_warps=4)
    cold = len(gate_ptrs)
    if cold:
        pointer_array = ctypes.c_void_p * cold
        _kernel()(pointer_array(*gate_ptrs), pointer_array(*down_ptrs), work.hidden.data_ptr(),
                  work.host_results.data_ptr(), cold, 8)
        work.cold_results[:cold].copy_(work.host_results[:cold], non_blocking=True)
        pool.stats.cpu_expert_rows += cold
    result = torch.empty_like(hidden)
    _combine[(8,)](work.device_table, work.hot_results, work.cold_results, weights, result,
                   2048, 256, enable_fp_fusion=False)
    return result
