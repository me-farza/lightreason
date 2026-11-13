"""Experimental AVX2 expert projection using original mapped BF16 rows."""

import ctypes
from functools import lru_cache
import os

import torch


@lru_cache(maxsize=1)
def _kernel():
    path = os.environ.get("EDGE35B_CPU_EXPERT_LIBRARY")
    if not path:
        raise RuntimeError("Set EDGE35B_CPU_EXPERT_LIBRARY to the compiled AVX2 library")
    fn = ctypes.CDLL(path).expert_rows
    pointer = ctypes.c_void_p
    fn.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(pointer), pointer, pointer,
                   ctypes.c_int, ctypes.c_int]
    fn.restype = None
    return fn


def cpu_forward(layer, hidden, indices, weights, *, hybrid=False):
    if (hidden.shape != (1, 2048) or indices.shape != (1, 8) or weights.shape != (1, 8)
            or hidden.dtype != torch.bfloat16 or layer.gate_up.shape[1:] != (1024, 2048)
            or layer.down.shape[1:] != (2048, 512)):
        raise ValueError("The AVX2 kernel requires the original 35B single-token expert layout")
    order = sorted(enumerate(indices[0].tolist()), key=lambda pair: pair[1])
    pool = layer.pool
    reuse_window = pool.hybrid_reuse_window if hybrid else None
    if reuse_window is not None:
        layer.decode_step += 1
    cpu_rows = []
    gpu_rows = []
    for position, (_, expert) in enumerate(order):
        key = pool._row_key(layer, expert) if hybrid else None
        if reuse_window is None:
            seen = key in pool._cpu_seen
        else:
            last = layer.last_route.get(expert, -reuse_window)
            seen = layer.decode_step - last <= reuse_window
            layer.last_route[expert] = layer.decode_step
        if hybrid and (key in pool._rows or seen):
            gpu_rows.append((position, expert))
        else:
            cpu_rows.append((position, expert))
            if hybrid and reuse_window is None:
                pool._cpu_seen.add(key)
    input_cpu = hidden.cpu().contiguous() if cpu_rows else None
    projected = [None] * len(order)
    linear = torch.nn.functional.linear
    for position, expert in gpu_rows:
        gate_up, down = pool._expert(layer, expert, cached=True)
        gate, up = linear(hidden, gate_up).chunk(2, dim=-1)
        projected[position] = linear(layer.act_fn(gate) * up, down)
    if cpu_rows:
        count = len(cpu_rows)
        output_cpu = torch.empty((count, 2048), device="cpu", dtype=torch.bfloat16)
        gate_ptr = (ctypes.c_void_p * count)(*(layer.gate_up[e].data_ptr() for _, e in cpu_rows))
        down_ptr = (ctypes.c_void_p * count)(*(layer.down[e].data_ptr() for _, e in cpu_rows))
        _kernel()(gate_ptr, down_ptr, input_cpu.data_ptr(), output_cpu.data_ptr(), count, 8)
        values = output_cpu.to(hidden.device)
        pool.stats.cpu_expert_rows += count
        for i, (position, _) in enumerate(cpu_rows):
            projected[position] = values[i:i + 1]
    result = torch.zeros_like(hidden)
    for i, (slot, _) in enumerate(order):
        result.add_(projected[i] * weights[:, slot:slot + 1])
    return result
