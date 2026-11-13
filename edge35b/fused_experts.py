"""Evaluate all eight single-token experts with batched BF16 matvecs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _matvec(table, x, result, K: tl.constexpr, N: tl.constexpr,
            WEIGHT_COLUMN: tl.constexpr, BLOCK_ROWS: tl.constexpr, SHARED_X: tl.constexpr,
            STRIDE: tl.constexpr = 3):
    expert = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, K)
    address = tl.load(table + STRIDE * expert + WEIGHT_COLUMN)
    if STRIDE == 3 or address != 0:
        weights = address.to(tl.pointer_type(tl.bfloat16))
        matrix = tl.load(weights + rows[:, None] * K + cols[None, :], rows[:, None] < N, 0).to(tl.float32)
        if SHARED_X:
            vector = tl.load(x + cols).to(tl.float32)
        else:
            vector = tl.load(x + expert * K + cols).to(tl.float32)
        values = tl.sum(matrix * vector[None, :], axis=1)
        tl.store(result + expert * N + rows, values, rows < N)


@triton.jit
def _ordered_sum(table, values, weights, out, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    features = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0.0, tl.float32)
    for expert in tl.static_range(8):
        slot = tl.load(table + 3 * expert + 2)
        router = tl.load(weights + slot).to(tl.float32)
        value = tl.load(values + expert * HIDDEN + features, features < HIDDEN, 0).to(tl.float32)
        weighted = (value * router).to(tl.bfloat16).to(tl.float32)
        result = (result + weighted).to(tl.bfloat16).to(tl.float32)
    tl.store(out + features, result, features < HIDDEN)


def fused_forward(layer, hidden, indices, weights, *, torch_projections=False):
    if (hidden.shape != (1, 2048) or indices.shape != (1, 8) or weights.shape != (1, 8)
            or hidden.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16
            or not hidden.is_contiguous() or not weights.is_contiguous()
            or layer.gate_up.shape[1:] != (1024, 2048) or layer.down.shape[1:] != (2048, 512)):
        raise ValueError("Fused expert decode requires the single-token 35B route shape")
    pool = layer.pool
    schedule = sorted(enumerate(indices[0].tolist()), key=lambda pair: pair[1])
    if len({expert for _, expert in schedule}) != 8 or not all(
            0 <= expert < layer.module.num_experts for _, expert in schedule):
        raise ValueError("Fused expert decode requires eight distinct valid routes")
    experts = [pool._expert(layer, expert, cached=True) for _, expert in schedule]
    pointers = torch.tensor([(gate.data_ptr(), down.data_ptr(), slot)
                             for (slot, _), (gate, down) in zip(schedule, experts)],
                            device=hidden.device, dtype=torch.int64)
    if torch_projections:
        linear = torch.nn.functional.linear
        gate_up = torch.cat([linear(hidden, gate) for gate, _ in experts], dim=0)
    else:
        gate_up = torch.empty((8, 1024), device=hidden.device, dtype=torch.bfloat16)
        _matvec[(128, 8)](pointers, hidden, gate_up, 2048, 1024, 0, 8, True, num_warps=8)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = layer.act_fn(gate) * up
    if torch_projections:
        projected = torch.cat([linear(activated[i:i + 1], down) for i, (_, down) in enumerate(experts)], dim=0)
    else:
        projected = torch.empty((8, 2048), device=hidden.device, dtype=torch.bfloat16)
        _matvec[(256, 8)](pointers, activated, projected, 512, 2048, 1, 8, False, num_warps=4)
    result = torch.empty_like(hidden)
    _ordered_sum[(8,)](pointers, projected, weights, result, 2048, 256, enable_fp_fusion=False)
    return result
