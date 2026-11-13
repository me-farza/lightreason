"""Replay the Scheduled expert operations with byte-identical staging."""

from contextlib import contextmanager
from operator import itemgetter
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from edge35b.expert_stream import SCHEDULED_TOP_K, TransferStats, _Layer


@triton.jit
def _stage(table, destination, router, staged_router, ELEMENTS: tl.constexpr,
           COLUMN: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(1)
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(table + 3 * expert + COLUMN).to(tl.pointer_type(tl.bfloat16))
    weights = tl.load(source + offsets)
    tl.store(destination + expert * ELEMENTS + offsets, weights)
    if COLUMN == 0 and block == 0:
        router_slot = tl.load(table + 3 * expert + 2)
        tl.store(staged_router + expert, tl.load(router + router_slot))


class Workspace:
    def __init__(self, layer, hidden, router, scheduled):
        self.hidden = torch.zeros_like(hidden)
        self.router = torch.zeros_like(router)
        self.gate = torch.zeros((8, 1024, 2048), dtype=hidden.dtype, device=hidden.device)
        self.down = torch.zeros((8, 2048, 512), dtype=hidden.dtype, device=hidden.device)
        self.host_table = torch.empty((8, 3), dtype=torch.int64, pin_memory=True)
        self.table_view = self.host_table.view(-1).numpy()
        self.device_table = torch.empty_like(self.host_table, device=hidden.device)

        # Feed sorted local expert IDs into the original Scheduled routine.
        # Static weight rows and router weights are populated in that order.
        def row(_layer, index, *, cached):
            return self.gate[index], self.down[index]

        pool = SimpleNamespace(_torch=torch, _expert=row, _schedule_slots=layer.pool._schedule_slots,
                               stats=TransferStats())
        view = SimpleNamespace(pool=pool, module=layer.module, act_fn=layer.act_fn)
        ids = torch.arange(SCHEDULED_TOP_K, dtype=torch.long)[None]
        scheduled(view, self.hidden, ids, self.router)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = scheduled(view, self.hidden, ids, self.router)

    def replay(self, hidden, weights):
        self.device_table.copy_(self.host_table, non_blocking=True)
        self.hidden.copy_(hidden)
        block = 8192
        for column, matrix in enumerate((self.gate, self.down)):
            elements = matrix.numel() // SCHEDULED_TOP_K
            _stage[(elements // block, SCHEDULED_TOP_K)](
                self.device_table, matrix, weights, self.router, elements, column, block,
                num_warps=8,
            )
        self.graph.replay()
        return self.output


@contextmanager
def replay_experts(pool):
    original = _Layer._scheduled_forward

    def forward(layer, hidden, indices, weights):
        if (layer.pool is not pool or hidden.shape != (1, 2048) or indices.shape != (1, 8)
                or weights.shape != (1, 8) or hidden.dtype != torch.bfloat16
                or indices.dtype != torch.long or weights.dtype != torch.bfloat16
                or not weights.is_contiguous() or layer.gate_up.shape[1:] != (1024, 2048)
                or layer.down.shape[1:] != (2048, 512)):
            return original(layer, hidden, indices, weights)

        # Reading these IDs synchronizes the preceding layer on this stream.
        # Its graph and pointer-table DMA are done before we reuse the buffers.
        ids = indices[0].tolist()
        if len(set(ids)) != SCHEDULED_TOP_K or min(ids) < 0 or max(ids) >= layer.module.num_experts:
            raise ValueError("Scheduled decode requires eight distinct in-range expert IDs")
        pool.stats.scheduled_router_d2h_bytes += indices.numel() * indices.element_size()
        pool.stats.scheduled_layer_calls += 1
        work = getattr(pool, "scheduled_workspace", None)
        if work is None:
            work = pool.scheduled_workspace = Workspace(layer, hidden, weights, original)

        rows = []
        for i, (slot, expert) in enumerate(sorted(enumerate(ids), key=itemgetter(1))):
            gate, down = pool._expert(layer, expert, cached=True)
            rows.extend((gate, down))
            work.table_view[3 * i:3 * i + 3] = (gate.data_ptr(), down.data_ptr(), slot)
        output = work.replay(hidden, weights)
        # Keep transient or evicted source tensors alive until the stage launch.
        # Their allocator stream is the same stream that reads them.
        return output

    _Layer._scheduled_forward = forward
    try:
        yield
    finally:
        _Layer._scheduled_forward = original
