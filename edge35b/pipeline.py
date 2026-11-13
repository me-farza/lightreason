"""Batch the router read and overlap exact expert copies with CUDA work."""

from ctypes import memmove

import torch
from torch.nn.functional import linear


class TransferPipeline:
    def __init__(self, pool, gate_shape, down_shape):
        self.pool = pool
        self.stream = torch.cuda.Stream(device=pool.device)
        self.layout = (gate_shape, down_shape)
        gate_count = gate_shape[0] * gate_shape[1]
        down_count = down_shape[0] * down_shape[1]
        self.sizes = (gate_count, down_count)
        with torch.inference_mode(False):
            self.host = torch.empty((8, gate_count + down_count), dtype=torch.bfloat16, pin_memory=True)
        self.ready = [torch.cuda.Event() for _ in range(self.host.shape[0])]
        self.used = 0
        self.events = {}

    def start(self):
        if self.used:
            self.ready[self.used - 1].synchronize()
        self.used = 0
        self.events.clear()
        self.stream.wait_stream(torch.cuda.current_stream(self.pool.device))

    def copy(self, gate_up, down, *, rows):
        if rows != 1 or self.used >= len(self.ready):
            raise ValueError("The transfer pipeline expects at most eight individual expert rows")
        slot = self.used
        host = self.host[slot]
        host_gate, host_down = host.split(self.sizes)
        for target, source in ((host_gate, gate_up), (host_down, down)):
            memmove(target.data_ptr(), source.data_ptr(), source.numel() * source.element_size())
        device_row = torch.empty_like(host, device=self.pool.device)
        device_row.record_stream(self.stream)
        event = self.ready[slot]
        with torch.cuda.stream(self.stream):
            try:
                device_row.copy_(host, non_blocking=True)
            finally:
                event.record(self.stream)
                self.used += 1
        gate, down = device_row.split(self.sizes)
        self.events[device_row.data_ptr()] = event
        self.pool.stats.transferred_expert_rows += 1
        self.pool.stats.transferred_bytes += host.numel() * host.element_size()
        return gate.view(self.layout[0]), down.view(self.layout[1])

    def wait(self, weights):
        event = self.events.get(weights[0].data_ptr())
        if event is not None:
            torch.cuda.current_stream(self.pool.device).wait_event(event)


def _routes(indices, schedule_slots):
    if indices.shape[0] == 1:
        return [(expert, schedule_slots[0], schedule_slots[slot])
                for slot, expert in sorted(enumerate(indices[0].tolist()), key=lambda pair: pair[1])]
    by_expert = {}
    for slot, token_routes in enumerate(indices.t().tolist()):
        for token, expert in enumerate(token_routes):
            by_expert.setdefault(expert, []).append((token, slot))
    ordered = sorted(by_expert.items())
    flat = [pair for _, pairs in ordered for pair in pairs]
    device_indices = torch.tensor(flat, dtype=torch.long, device=indices.device).t()
    result = []
    start = 0
    for expert, pairs in ordered:
        end = start + len(pairs)
        result.append((expert, device_indices[0, start:end], device_indices[1, start:end]))
        start = end
    return result


def routed_forward(layer, hidden_states, top_k_index, top_k_weights, *, pipeline=None):
    """Preserve eager expert and per-expert token order, including BF16 adds."""
    pool = layer.pool
    cached = hidden_states.shape[0] == 1 or pool.cache_multi_token
    routes = _routes(top_k_index, pool._schedule_slots)
    keep = set()
    if not cached and layer.mode in ("seeded", "hybrid"):
        by_frequency = sorted(routes, key=lambda route: (-route[1].numel(), route[0]))
        keep = {expert for expert, _, _ in by_frequency[:pool._seed_rows]}
    result = torch.zeros_like(hidden_states)
    for start in range(0, len(routes), 8):
        group = routes[start:start + 8]
        if pipeline is not None:
            pipeline.start()
        copy = pipeline.copy if pipeline is not None else None
        weights = [pool._expert(layer, expert, cached=cached or expert in keep, copy=copy)
                   for expert, _, _ in group]
        contributions = [None] * len(group)
        order = range(len(group))
        if pipeline is not None:
            order = sorted(order, key=lambda i: weights[i][0].data_ptr() in pipeline.events)
        for i in order:
            _, token_idx, slot_idx = group[i]
            if pipeline is not None:
                pipeline.wait(weights[i])
            gate_up, down = weights[i]
            gate, up = linear(hidden_states[token_idx], gate_up).chunk(2, dim=-1)
            current = linear(layer.act_fn(gate) * up, down)
            contributions[i] = current * top_k_weights[token_idx, slot_idx, None]
        for (_, token_idx, _), contribution in zip(group, contributions):
            result.index_add_(0, token_idx, contribution.to(result.dtype))
    return result
