"""Optional replay and native worker ablations for the frozen Packed path."""

from contextlib import contextmanager
import ctypes
import os
from types import SimpleNamespace

import torch

from edge35b import packed_hybrid
from edge35b.fused_experts import _matvec


class Workspace(packed_hybrid.Workspace):
    def __init__(self, device, library):
        super().__init__(device)
        self.hidden_gpu = torch.empty_like(self.hidden, device=device)
        self.router = torch.empty((1, 8), device=device, dtype=torch.bfloat16)
        self.result = torch.empty_like(self.hidden_gpu)
        self.hot_graph = self.combine_graph = None
        self.library = library
        self.experts = (ctypes.c_int32 * 8)()
        self.recent = (ctypes.c_uint8 * 8)()
        self.gate = (ctypes.c_void_p * 8)()
        self.down = (ctypes.c_void_p * 8)()
        self.staging = None
        self.copy_slot = 0
        self.ada_gate = None

    def pinned_copy(self, gate, down, *, rows):
        if self.staging is None:
            elements = gate.numel() + down.numel()
            self.staging = torch.empty((8, elements), dtype=torch.bfloat16, pin_memory=True)
        host = self.staging[self.copy_slot]
        self.copy_slot += 1
        gate_bytes = gate.numel() * gate.element_size()
        down_bytes = down.numel() * down.element_size()
        ctypes.memmove(host.data_ptr(), gate.data_ptr(), gate_bytes)
        ctypes.memmove(host.data_ptr() + gate_bytes, down.data_ptr(), down_bytes)
        device = host.to(self.hidden_gpu.device, non_blocking=True)
        first, second = device.split((gate.numel(), down.numel()))
        return first.view(gate.shape), second.view(down.shape)

    def hot(self, layer):
        if self.ada_gate is not None:
            self.ada_gate(self.device_table, self.hidden_gpu, self.active)
            active = self.active
        else:
            _matvec[(128, 8)](self.device_table, self.hidden_gpu, self.gate_up, 2048, 1024, 0, 8, True, 4, num_warps=8)
            gate, up = self.gate_up.chunk(2, dim=-1)
            active = layer.act_fn(gate) * up
        _matvec[(256, 8)](self.device_table, active, self.hot_results, 512, 2048, 1, 8, False, 4, num_warps=4)

    def combine(self):
        packed_hybrid._combine[(8,)](self.device_table, self.hot_results, self.cold_results, self.router,
                                     self.result, 2048, 256, enable_fp_fusion=False)

    def capture(self, operation):
        operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        return graph


def native_library():
    library = ctypes.CDLL(os.environ["EDGE35B_PACKED_WORKERS"])
    pointer = ctypes.c_void_p
    library.packed_plan.argtypes = [pointer, pointer, pointer, ctypes.c_int, pointer, pointer]
    library.packed_plan.restype = None
    library.packed_workers_create.argtypes = [ctypes.c_int]
    library.packed_workers_create.restype = pointer
    library.packed_workers_destroy.argtypes = [pointer]
    library.packed_workers_destroy.restype = None
    library.packed_workers_submit.argtypes = [pointer, pointer, pointer, pointer, pointer, ctypes.c_int]
    library.packed_workers_submit.restype = None
    library.packed_workers_wait.argtypes = [pointer]
    library.packed_workers_wait.restype = None
    return library


class StaticMixerCache:
    def __init__(self, layer_index, source):
        self.layers = {layer_index: SimpleNamespace(conv_states=torch.empty_like(source.conv_states),
                                                    recurrent_states=torch.empty_like(source.recurrent_states))}

    def has_previous_state(self, _index):
        return True

    def update_recurrent_state(self, state, index):
        target = self.layers[index].recurrent_states
        target.copy_(state)
        return target


@contextmanager
def mixer_graphs(model, pool):
    """Replay single-token Gated DeltaNet in layer order on one request stream."""
    graphs = getattr(pool, "mixer_graphs", None)
    if graphs is None:
        graphs = pool.mixer_graphs = {}
        pool.mixer_graph_memory = torch.cuda.graph_pool_handle()
    originals = []

    def bind(original, index):
        def forward(hidden_states, cache_params=None, attention_mask=None):
            if (hidden_states.shape[:2] != (1, 1) or cache_params is None
                    or not cache_params.has_previous_state(index) or attention_mask is not None):
                return original(hidden_states, cache_params=cache_params, attention_mask=attention_mask)
            current = cache_params.layers[index]
            if index not in graphs:
                static_cache = StaticMixerCache(index, current)
                static_input = torch.empty_like(hidden_states)
                buffers = static_cache.layers[index]
                buffers.conv_states.copy_(current.conv_states)
                buffers.recurrent_states.copy_(current.recurrent_states)
                static_input.copy_(hidden_states)
                original(static_input, cache_params=static_cache, attention_mask=None)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool.mixer_graph_memory):
                    output = original(static_input, cache_params=static_cache, attention_mask=None)
                graphs[index] = (graph, static_input, static_cache, output)
            graph, static_input, static_cache, output = graphs[index]
            buffers = static_cache.layers[index]
            # Captured addresses persist across requests. Only the active
            # request's cache supplies state and receives the next state.
            static_input.copy_(hidden_states)
            buffers.conv_states.copy_(current.conv_states)
            buffers.recurrent_states.copy_(current.recurrent_states)
            graph.replay()
            current.conv_states.copy_(buffers.conv_states)
            current.recurrent_states.copy_(buffers.recurrent_states)
            return output
        return forward

    try:
        for index, layer in enumerate(model.model.layers):
            if hasattr(layer, "linear_attn"):
                mixer = layer.linear_attn
                original = mixer.forward
                originals.append((mixer, mixer.__dict__.get("forward")))
                mixer.forward = bind(original, index)
        yield
    finally:
        for mixer, previous in originals:
            if previous is None:
                del mixer.forward
            else:
                mixer.forward = previous


@contextmanager
def configure(pool, mode):
    if mode not in ("buffers", "graph", "native", "overlap", "pinned", "ada"):
        raise ValueError(f"Unknown Packed ablation: {mode}")
    replay = mode != "buffers"
    library = native_library() if mode in ("native", "overlap", "pinned", "ada") else None
    worker = library.packed_workers_create(7) if library else None
    original = packed_hybrid.packed_forward
    # The same graph and buffers can service every layer and request on this pool.
    workspaces = getattr(pool, "native_workspaces", None)
    if workspaces is None:
        workspaces = pool.native_workspaces = {}

    def forward(layer, hidden, indices, weights):
        work = workspaces.get(mode)
        if work is None:
            work = workspaces[mode] = Workspace(hidden.device, library)
            if mode == "ada":
                from edge35b.packed_ada import load_gate
                work.ada_gate = load_gate()
                work.active = torch.empty((8, 512), dtype=torch.bfloat16, device=hidden.device)
        work.ids.copy_(indices[0], non_blocking=True)
        work.hidden.copy_(hidden, non_blocking=True)
        work.hidden_gpu.copy_(hidden)
        work.router.copy_(weights)
        work.inbound.record()
        work.inbound.synchronize()
        work.copy_slot = 0
        layer.decode_step += 1
        if library:
            if not hasattr(layer, "native_last_route"):
                layer.native_last_route = (ctypes.c_int32 * 256)(*([-4] * 256))
            library.packed_plan(work.ids.data_ptr(), work.host_table.data_ptr(), layer.native_last_route,
                                layer.decode_step, work.experts, work.recent)
            routes = ((work.table_view[4 * i + 2], work.experts[i], work.recent[i]) for i in range(8))
        else:
            routes = []
            for slot, expert in sorted(enumerate(work.ids.tolist()), key=lambda route: route[1]):
                recent = layer.decode_step - layer.last_route.get(expert, -4) <= 4
                layer.last_route[expert] = layer.decode_step
                routes.append((slot, expert, recent))

        hot = []
        cold = 0
        for i, (slot, expert, recent) in enumerate(routes):
            key = pool._row_key(layer, expert)
            if key in pool._rows or recent:
                copy = work.pinned_copy if mode in ("pinned", "ada") else None
                gate, down = pool._expert(layer, expert, cached=True, copy=copy)
                hot.extend((gate, down))
                row = (gate.data_ptr(), down.data_ptr(), slot, -1)
            else:
                row = (0, 0, slot, cold)
                work.gate[cold] = layer.gate_up[expert].data_ptr()
                work.down[cold] = layer.down[expert].data_ptr()
                cold += 1
            work.table_view[4 * i:4 * i + 4] = row
        work.device_table.copy_(work.host_table, non_blocking=True)
        if work.copy_slot:
            pool.stats.transferred_expert_rows += work.copy_slot
            pool.stats.transferred_bytes += work.copy_slot * layer.row_bytes
            pool.stats.packed_h2d_copies += work.copy_slot
        submitted = bool(library and cold and mode in ("overlap", "pinned", "ada"))
        if submitted:
            library.packed_workers_submit(worker, work.gate, work.down, work.hidden.data_ptr(),
                                          work.host_results.data_ptr(), cold)
        try:
            if hot:
                if replay:
                    if work.hot_graph is None:
                        work.hot_graph = work.capture(lambda: work.hot(layer))
                    work.hot_graph.replay()
                else:
                    work.hot(layer)
            if cold:
                if library:
                    if not submitted:
                        library.packed_workers_submit(worker, work.gate, work.down, work.hidden.data_ptr(),
                                                      work.host_results.data_ptr(), cold)
                    library.packed_workers_wait(worker)
                    submitted = False
                else:
                    packed_hybrid._kernel()(work.gate, work.down, work.hidden.data_ptr(),
                                           work.host_results.data_ptr(), cold, 8)
                work.cold_results[:cold].copy_(work.host_results[:cold], non_blocking=True)
                pool.stats.cpu_expert_rows += cold
            if replay:
                if work.combine_graph is None:
                    work.combine_graph = work.capture(work.combine)
                work.combine_graph.replay()
            else:
                work.combine()
            return work.result
        finally:
            if submitted:
                library.packed_workers_wait(worker)

    packed_hybrid.packed_forward = forward
    try:
        yield
    finally:
        packed_hybrid.packed_forward = original
        if worker:
            library.packed_workers_destroy(worker)
