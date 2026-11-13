"""Grouped CUDA projection of the original BF16 experts for short target blocks."""

import ctypes
from functools import lru_cache
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _project(Pointers, Routes, X, Y, IN: tl.constexpr, OUT: tl.constexpr,
             DOWN: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr, REDUCE: tl.constexpr):
    expert = tl.program_id(0)
    row = tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    offsets = tl.arange(0, REDUCE)
    token = tl.load(Routes + expert * ROWS + row)
    weight = tl.load(Pointers + 2 * expert + int(DOWN)).to(tl.pointer_type(tl.bfloat16))
    input_row = expert * ROWS + row if DOWN else tl.maximum(token, 0)
    acc = tl.full((ROWS, COLS), 0.0, tl.float32)
    for start in range(0, IN, REDUCE):
        k = start + offsets
        lhs = tl.load(X + input_row[:, None] * IN + k[None, :], token[:, None] >= 0, 0)
        rhs = tl.load(weight + col[None, :] * IN + k[:, None])
        acc = tl.dot(lhs, rhs, acc)
    tl.store(Y + (expert * ROWS + row[:, None]) * OUT + col[None, :], acc)


@triton.jit
def _combine(Inverse, X, Cold, Router, Out, WIDTH: tl.constexpr, ROWS: tl.constexpr,
             TILE: tl.constexpr, HYBRID: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    result = tl.full((TILE,), 0.0, tl.float32)
    for route in tl.static_range(8):
        descriptor = Inverse + (token * 8 + route) * 3
        expert = tl.load(descriptor)
        row = tl.load(descriptor + 1)
        router_slot = tl.load(descriptor + 2)
        if HYBRID and expert < 0:
            value = tl.load(Cold + row * WIDTH + col).to(tl.float32)
        else:
            value = tl.load(X + (expert * ROWS + row) * WIDTH + col).to(tl.float32)
        scale = tl.load(Router + token * 8 + router_slot).to(tl.float32)
        weighted = (value * scale).to(tl.bfloat16).to(tl.float32)
        result = (result + weighted).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + token * WIDTH + col, result)


@lru_cache(maxsize=1)
def _cpu_block_kernel():
    fn = ctypes.CDLL(os.environ["EDGE35B_CPU_EXPERT_LIBRARY"]).expert_block_rows
    pointer = ctypes.c_void_p
    fn.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(pointer), pointer,
                   ctypes.POINTER(ctypes.c_int32), pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = None
    return fn


class CpuBlock:
    def __init__(self, device):
        self.hidden = torch.empty((16, 2048), dtype=torch.bfloat16, pin_memory=True)
        self.indices = torch.empty((16, 8), dtype=torch.long, pin_memory=True)
        self.output = torch.empty((128, 2048), dtype=torch.bfloat16, pin_memory=True)
        self.device_output = torch.empty_like(self.output, device=device)
        self.inbound = torch.cuda.Event()

    def read(self, hidden, indices):
        width = hidden.shape[0]
        self.hidden[:width].copy_(hidden, non_blocking=True)
        self.indices[:width].copy_(indices, non_blocking=True)
        self.inbound.record()
        self.inbound.synchronize()
        return self.indices[:width].tolist()

    def compute(self, layer, width, experts):
        size = len(experts)
        pointers = ctypes.c_void_p * size
        gate = pointers(*(layer.gate_up[expert].data_ptr() for expert, _ in experts))
        down = pointers(*(layer.down[expert].data_ptr() for expert, _ in experts))
        inputs = (ctypes.c_int32 * size)(*(token for _, token in experts))
        _cpu_block_kernel()(gate, down, self.hidden.data_ptr(), inputs, self.output.data_ptr(), width, size, 8)
        self.device_output[:size].copy_(self.output[:size], non_blocking=True)
        layer.pool.stats.cpu_expert_rows += size


def grouped_block(layer, hidden, indices, weights):
    count, width = hidden.shape
    if not 2 <= count <= 16 or width != 2048 or indices.shape != (count, 8):
        raise ValueError("Grouped expert verification needs 2–16 Qwen3.6 target positions")
    if not hidden.is_contiguous() or not weights.is_contiguous():
        raise ValueError("Grouped expert verification requires contiguous target activations")
    pool = layer.pool
    host = None
    if pool.hybrid_verifier:
        if pool.cpu_block is None:
            pool.cpu_block = CpuBlock(hidden.device)
        host = pool.cpu_block
        layer.decode_step += 1
    routes = {}
    for token, top_k in enumerate(host.read(hidden, indices) if host else indices.tolist()):
        for slot, expert in enumerate(top_k):
            routes.setdefault(expert, []).append((token, slot))
    pairs, inputs, cold, inverse = [], [], [], [[] for _ in range(count)]
    for expert, matches in sorted(routes.items()):
        if host:
            recent = layer.decode_step - layer.last_route.get(expert, -4) <= 4
            layer.last_route[expert] = layer.decode_step
            if len(matches) == 1 and not recent and pool._row_key(layer, expert) not in pool._rows:
                token, slot = matches[0]
                inverse[token].append((-1, len(cold), slot))
                cold.append((expert, token))
                continue
        index = len(inputs)
        gate, down = pool._expert(layer, expert, cached=True)
        pairs.extend((gate, down))
        inputs.append([token for token, _ in matches] + [-1] * (16 - len(matches)))
        for row, (token, slot) in enumerate(matches):
            inverse[token].append((index, row, slot))
    device = hidden.device
    inverse = torch.tensor(inverse, device=device, dtype=torch.int32)
    gpu_experts = len(inputs)
    projected = torch.empty((max(gpu_experts, 1), 16, 2048), device=device, dtype=torch.bfloat16)
    if gpu_experts:
        pointers = torch.tensor([weight.data_ptr() for weight in pairs], device=device, dtype=torch.int64)
        inputs = torch.tensor(inputs, device=device, dtype=torch.int32)
        gates = torch.empty((gpu_experts, 16, 1024), device=device, dtype=torch.bfloat16)
        _project[(gpu_experts, 32)](pointers, inputs, hidden, gates, 2048, 1024, False, 16, 32, 64)
        gate, up = gates.chunk(2, dim=-1)
        active = layer.act_fn(gate) * up
        _project[(gpu_experts, 64)](pointers, inputs, active, projected, 512, 2048, True, 16, 32, 64)
    if cold:
        host.compute(layer, count, cold)
    result = torch.empty_like(hidden)
    _combine[(count, 8)](inverse, projected, host.device_output if host else projected, weights, result,
                          width, 16, 256, bool(host), enable_fp_fusion=False)
    return result
