"""Stream frozen BF16 Qwen experts without changing eager expert arithmetic.

The transfer controls call the Transformers 5.7.0 Qwen3_5MoeExperts eager
body. Dense transfers all rows, selected transfers rows on demand, cached
reuses selected rows, and packed groups single-token transfers in pinned host
buffers. Scheduled derives the single-token routing schedule on the host while
preserving the eager per-expert arithmetic and accumulation order.

Use one pool on one CUDA device, thread, and stream. The caller times complete
synchronized requests and clears the pool between cold or competing runs.
The cache budget counts retained weight bytes; activations, transient copies,
and the CUDA allocator's reserved memory are separate.
"""

import ast
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from inspect import getsource, signature
from operator import itemgetter
from textwrap import dedent
import threading
from time import perf_counter
from types import SimpleNamespace


TRANSFORMERS_VERSION = "5.7.0"
MODES = ("dense", "selected", "cached", "packed", "scheduled", "grouped", "pipelined",
         "fused", "batched", "cpu", "seeded", "hybrid")
PACKED_ROWS = 8
SCHEDULED_TOP_K = 8
SCHEDULED_EAGER_AST_SHA256 = "2c120b7959a23b4ecb91119942942e5574ab710cfe3fa7d4ced71105dc450211"


@dataclass
class TransferStats:
    layer_calls: int = 0
    dense_layer_calls: int = 0
    selected_layer_calls: int = 0
    cached_layer_calls: int = 0
    packed_layer_calls: int = 0
    scheduled_layer_calls: int = 0
    prefill_cache_bypasses: int = 0
    prefill_packed_bypasses: int = 0
    prefill_scheduled_bypasses: int = 0
    scheduled_router_d2h_bytes: int = 0
    packed_router_d2h_bytes: int = 0
    packed_reuse_fences: int = 0
    packed_h2d_copies: int = 0
    routing_scalar_reads: int = 0
    transferred_bytes: int = 0
    transferred_expert_rows: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    peak_cached_bytes: int = 0
    cpu_expert_rows: int = 0


class ExpertPool:
    """A byte-bounded LRU shared by all patched expert layers.

    Hits and misses count lookups in cached and scheduled modes. Dense and selected-only
    transfers have independent row/byte counts and do not consult the LRU.
    """

    def __init__(self, device="cuda:0", *, capacity_bytes=0, non_blocking=True):
        import torch

        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("capacity_bytes must be a nonnegative integer")
        if type(non_blocking) is not bool:
            raise ValueError("non_blocking must be a boolean")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Expert streaming requires a CUDA destination")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.device = device
        self._torch = torch
        self._capacity_bytes = capacity_bytes
        self._non_blocking = non_blocking
        self._rows = OrderedDict()
        self._cached_bytes = 0
        self._packed = None
        self._pipeline = None
        self._schedule_slots = None
        self._seed_rows = 0
        self._cpu_seen = set()
        self.cache_multi_token = False
        self.grouped_verifier = False
        self.hybrid_verifier = False
        self.cpu_block = None
        self.hybrid_reuse_window = None
        self.packed_hybrid = False
        self.packed_workspace = None
        self._owner = None
        self._attached = False
        self.stats = TransferStats()

    def snapshot(self):
        return {
            **asdict(self.stats),
            "capacity_bytes": self._capacity_bytes,
            "cached_bytes": self._cached_bytes,
            "cached_rows": len(self._rows),
            "non_blocking": self._non_blocking,
            "packed_pinned_bytes": self._packed.bytes if self._packed is not None else 0,
            "packed_setup_s": self._packed.setup_s if self._packed is not None else 0.0,
        }

    def reset_stats(self):
        """Reset counters while keeping any already warmed rows."""
        self.stats = TransferStats(peak_cached_bytes=self._cached_bytes)

    def clear(self):
        """Release cached rows while retaining the packed host staging area."""
        self._rows.clear()
        self._cpu_seen.clear()
        self._cached_bytes = 0

    def _check_context(self):
        current_stream = self._torch.cuda.current_stream(self.device).cuda_stream
        owner = (threading.get_ident(), current_stream)
        if self._owner is None:
            self._owner = owner
        elif owner != self._owner:
            raise RuntimeError("Use the pool on its original CUDA thread and stream")

    def _copy(self, gate_up, down, *, rows):
        # CPU backing tensors outlive the queued copies. Preserve their dtype
        # and strides; both packed parameter tensors are contiguous.
        pair = (
            gate_up.detach().to(self.device, non_blocking=self._non_blocking),
            down.detach().to(self.device, non_blocking=self._non_blocking),
        )
        self.stats.transferred_bytes += _bytes(gate_up) + _bytes(down)
        self.stats.transferred_expert_rows += rows
        return pair

    @staticmethod
    def _row_key(layer, index):
        # Keep source objects alive alongside their stamps so a newly loaded
        # parameter cannot inherit a cached row by reusing an object address.
        return (layer.module, *layer.source_stamps, index, layer.gate_up, layer.down)

    def _expert(self, layer, index, *, cached, copy=None):
        copy = self._copy if copy is None else copy
        if not cached:
            return copy(layer.gate_up[index], layer.down[index], rows=1)

        key = self._row_key(layer, index)
        entry = self._rows.get(key)
        if entry is not None:
            self.stats.cache_hits += 1
            self._rows.move_to_end(key)
            return entry[0]

        self.stats.cache_misses += 1
        row_bytes = layer.row_bytes
        if row_bytes > self._capacity_bytes:
            return copy(layer.gate_up[index], layer.down[index], rows=1)

        while self._cached_bytes + row_bytes > self._capacity_bytes:
            _, (_, evicted_bytes) = self._rows.popitem(last=False)
            self._cached_bytes -= evicted_bytes
            self.stats.cache_evictions += 1
        pair = copy(layer.gate_up[index], layer.down[index], rows=1)
        self._rows[key] = (pair, row_bytes)
        self._cached_bytes += row_bytes
        self.stats.peak_cached_bytes = max(self.stats.peak_cached_bytes, self._cached_bytes)
        return pair


class _PackedBuffers:
    """One shared pinned gate/down pair for the eight active decode experts."""

    def __init__(self, pool, layout):
        torch = pool._torch
        started = perf_counter()
        self.layout = layout
        with torch.inference_mode(False):
            self.gate_up, self.down = (
                torch.empty((PACKED_ROWS, *shape), dtype=torch.bfloat16,
                            device="cpu", pin_memory=True)
                for shape in layout
            )
        self.bytes = _bytes(self.gate_up) + _bytes(self.down)
        self._fence = torch.cuda.Event() if pool._non_blocking else None
        self._pending = False
        self._recorded = False
        if self._fence is not None:
            # Create the CUDA event during setup; first record is lazy.
            self._fence.record(torch.cuda.current_stream(pool.device))
            self._fence.synchronize()
        self.setup_s = perf_counter() - started

    def gather(self, layer, top_k_index):
        pool = layer.pool
        torch = pool._torch
        if (top_k_index.ndim != 2 or top_k_index.shape[0] != 1
                or not 1 <= top_k_index.shape[1] <= PACKED_ROWS
                or top_k_index.dtype != torch.long):
            raise ValueError("Packed decode requires one token and at most eight routed expert IDs")
        ids = sorted(top_k_index[0].tolist())
        pool.stats.packed_router_d2h_bytes += _bytes(top_k_index)
        if len(set(ids)) != len(ids) or ids[0] < 0 or ids[-1] >= layer.module.num_experts:
            raise ValueError("Packed decode requires distinct in-range expert IDs")

        if self._pending:
            if not self._recorded:
                raise RuntimeError("The packed host buffers lost their CUDA reuse fence")
            self._fence.synchronize()
            pool.stats.packed_reuse_fences += 1
            self._pending = False

        rows = len(ids)
        index = torch.tensor(ids, dtype=torch.long, device="cpu")
        gate_up, down = self.gate_up[:rows], self.down[:rows]
        torch.index_select(layer.gate_up, 0, index, out=gate_up)
        torch.index_select(layer.down, 0, index, out=down)

        # Record after the two transfers and before eager compute. Even if a
        # later transfer fails, the host must not overwrite a queued source.
        self._pending = self._fence is not None
        self._recorded = False
        try:
            device_gate, device_down = pool._copy(gate_up, down, rows=rows)
            pool.stats.packed_h2d_copies += 2
        finally:
            if self._pending:
                self._fence.record(torch.cuda.current_stream(pool.device))
                self._recorded = True
        return ids, device_gate, device_down


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _weight_stamp(weight):
    try:
        version = weight._version
    except RuntimeError as exc:
        raise ValueError("Load expert parameters outside torch.inference_mode()") from exc
    return (id(weight), weight.data_ptr(), version, weight.device, weight.dtype,
            tuple(weight.shape), weight.stride())


class _ProjectionRows:
    """The two tensors' indexing interface used by the pinned eager body."""

    def __init__(self, selection, *, is_gate):
        self._selection = selection
        self._is_gate = is_gate

    def __getitem__(self, expert_index):
        if self._is_gate:
            return self._selection.gate(expert_index)
        return self._selection.down(expert_index)


class _Selection:
    def __init__(self, layer, pool, *, cached, packed=None):
        self.layer = layer
        self.pool = pool
        self.cached = cached
        self.packed = packed
        self.position = 0
        self.pending = None

    def gate(self, index):
        if self.pending is not None:
            raise RuntimeError("Unexpected eager expert gate/down access order")
        # Use the actual upstream index. The down lookup reuses the same row
        # and needs no second device-to-host scalar read.
        expert = int(index.item())
        self.pool.stats.routing_scalar_reads += 1
        if not 0 <= expert < self.layer.module.num_experts:
            raise RuntimeError("Unexpected expert index")
        if self.packed is None:
            gate_up, down = self.pool._expert(self.layer, expert, cached=self.cached)
        else:
            ids, gates, downs = self.packed
            position = self.position
            if position >= len(ids) or expert != ids[position]:
                raise RuntimeError("Packed row order disagrees with the eager expert indices")
            gate_up, down = gates[position], downs[position]
            self.position += 1
        self.pending = (index, down)
        return gate_up

    def down(self, index):
        if self.pending is None or index is not self.pending[0]:
            raise RuntimeError("Unexpected eager expert gate/down access order")
        _, down = self.pending
        self.pending = None
        return down

    def check_complete(self):
        if self.pending is not None:
            raise RuntimeError("Eager expert did not consume its down projection")
        if self.packed is not None and self.position != len(self.packed[0]):
            raise RuntimeError("Eager expert did not consume every packed row")


class _Layer:
    def __init__(self, name, module, eager_forward, pool, mode):
        self.name = name
        self.module = module
        self.eager_forward = eager_forward
        self.pool = pool
        self.mode = mode
        self.gate_up = module.gate_up_proj
        self.down = module.down_proj
        self.act_fn = module.act_fn
        self.decode_step = 0
        self.last_route = {}

        if "forward" in vars(module) or getattr(module, "_hf_hook", None) is not None:
            raise ValueError(f"{name}: expert forwards must have no existing offload hooks")
        if module.num_experts <= 0 or module.hidden_dim <= 0 or module.intermediate_dim <= 0:
            raise ValueError(f"{name}: invalid expert dimensions")
        expected = (
            (module.num_experts, 2 * module.intermediate_dim, module.hidden_dim),
            (module.num_experts, module.hidden_dim, module.intermediate_dim),
        )
        for source, shape in zip((self.gate_up, self.down), expected):
            if (source.device.type != "cpu" or source.dtype != pool._torch.bfloat16
                    or source.requires_grad or not source.is_contiguous()
                    or tuple(source.shape) != shape):
                raise ValueError(f"{name}: expected frozen contiguous CPU BF16 expert parameters")
        self.source_stamps = (_weight_stamp(self.gate_up), _weight_stamp(self.down))
        self.row_bytes = (_bytes(self.gate_up) + _bytes(self.down)) // module.num_experts
        self._check_sources()

    def _check_sources(self):
        module = self.module
        if module.training or module.config._experts_implementation != "eager":
            raise RuntimeError(f"{self.name}: use eval mode and experts_implementation='eager'")
        if (module.gate_up_proj is not self.gate_up or module.down_proj is not self.down
                or module.act_fn is not self.act_fn
                or self.gate_up.requires_grad or self.down.requires_grad
                or (_weight_stamp(self.gate_up), _weight_stamp(self.down)) != self.source_stamps):
            raise RuntimeError(f"{self.name}: original expert weights or activation changed")

    def _scheduled_forward(self, hidden_states, top_k_index, top_k_weights):
        pool = self.pool
        torch = pool._torch
        route_shape = (1, SCHEDULED_TOP_K)
        if (tuple(hidden_states.shape) != (1, self.module.hidden_dim)
                or hidden_states.dtype != torch.bfloat16
                or tuple(top_k_index.shape) != route_shape or top_k_index.dtype != torch.long
                or tuple(top_k_weights.shape) != route_shape or top_k_weights.dtype != torch.bfloat16):
            raise ValueError("Scheduled decode requires one BF16 token and eight BF16/long router slots")

        ids = top_k_index[0].tolist()
        pool.stats.scheduled_router_d2h_bytes += _bytes(top_k_index)
        if len(set(ids)) != SCHEDULED_TOP_K or min(ids) < 0 or max(ids) >= self.module.num_experts:
            raise ValueError("Scheduled decode requires eight distinct in-range expert IDs")
        pool.stats.scheduled_layer_calls += 1

        # For one token with distinct routes, the upstream expert_mask has
        # one hit per selected expert. nonzero visits those experts ascending;
        # where yields token zero and the expert's original top-k slot.
        # Keep its tensor index shapes and per-expert operations intact.
        # Derived from the pinned Qwen3_5MoeExperts eager forward, Apache-2.0.
        slots = pool._schedule_slots
        token_idx = slots[0]
        linear = torch.nn.functional.linear
        result = torch.zeros_like(hidden_states)
        for slot, expert in sorted(enumerate(ids), key=itemgetter(1)):
            gate_up, down = pool._expert(self, expert, cached=True)
            current = hidden_states[token_idx]
            gate, up = linear(current, gate_up).chunk(2, dim=-1)
            current = linear(self.act_fn(gate) * up, down)
            current = current * top_k_weights[token_idx, slots[slot], None]
            result.index_add_(0, token_idx, current.to(result.dtype))
        return result

    def forward(self, hidden_states, top_k_index, top_k_weights):
        pool = self.pool
        pool._check_context()
        self._check_sources()
        if pool._torch.is_grad_enabled():
            raise RuntimeError("Use expert streaming under torch.no_grad() or torch.inference_mode()")
        if (hidden_states.ndim != 2 or hidden_states.device != pool.device
                or top_k_index.device != pool.device or top_k_weights.device != pool.device):
            raise ValueError(f"{self.name}: routed states and indices must be on the pool CUDA device")

        pool.stats.layer_calls += 1
        single_token = hidden_states.shape[0] == 1
        if pool.grouped_verifier and pool.cache_multi_token and not single_token:
            from edge35b.block_experts import grouped_block

            return grouped_block(self, hidden_states, top_k_index, top_k_weights)
        if self.mode in ("cpu", "hybrid") and single_token:
            if self.mode == "hybrid" and pool.packed_hybrid:
                from edge35b.packed_hybrid import packed_forward

                return packed_forward(self, hidden_states, top_k_index, top_k_weights)
            from edge35b.cpu_route import cpu_forward

            return cpu_forward(self, hidden_states, top_k_index, top_k_weights, hybrid=self.mode == "hybrid")
        if self.mode in ("fused", "batched") and single_token:
            from edge35b.fused_experts import fused_forward

            return fused_forward(self, hidden_states, top_k_index, top_k_weights,
                                 torch_projections=self.mode == "batched")
        if self.mode == "pipelined" or (
                self.mode in ("grouped", "fused", "batched", "cpu", "seeded", "hybrid") and not single_token):
            from edge35b.pipeline import routed_forward

            return routed_forward(self, hidden_states, top_k_index, top_k_weights,
                                  pipeline=pool._pipeline if self.mode == "pipelined" else None)
        if self.mode in ("scheduled", "grouped", "seeded"):
            if single_token:
                return self._scheduled_forward(hidden_states, top_k_index, top_k_weights)
            if hidden_states.shape[0] == 0:
                raise ValueError("Scheduled mode requires at least one token")
            pool.stats.prefill_scheduled_bypasses += 1
        selection = None
        if self.mode == "dense":
            pool.stats.dense_layer_calls += 1
            gate_up, down = pool._copy(self.gate_up, self.down, rows=self.module.num_experts)
        else:
            # Multi-token prefill can touch many experts in each layer. A
            # streaming prefill uses the same selected-only path without
            # evicting the single-token decode working set.
            cached = self.mode == "cached" and single_token
            packed = None
            if self.mode == "packed" and single_token:
                pool.stats.packed_layer_calls += 1
                packed = pool._packed.gather(self, top_k_index)
            elif cached:
                pool.stats.cached_layer_calls += 1
            else:
                pool.stats.selected_layer_calls += 1
                if self.mode == "cached":
                    pool.stats.prefill_cache_bypasses += 1
                if self.mode == "packed":
                    pool.stats.prefill_packed_bypasses += 1
            selection = _Selection(self, pool, cached=cached, packed=packed)
            gate_up = _ProjectionRows(selection, is_gate=True)
            down = _ProjectionRows(selection, is_gate=False)

        view = SimpleNamespace(num_experts=self.module.num_experts, act_fn=self.act_fn,
                               gate_up_proj=gate_up, down_proj=down)
        result = self.eager_forward(view, hidden_states, top_k_index, top_k_weights)
        if selection is not None:
            selection.check_complete()
        return result


@contextmanager
def patch_experts(model, *, mode, pool):
    """Install one transfer mode, then restore the model's expert forwards.

    The target should already be in eval mode with frozen ordinary CPU expert
    parameters, CUDA non-expert modules, and explicit eager expert dispatch.
    """
    if mode not in MODES:
        raise ValueError(f"Choose an expert transfer mode from {MODES}")
    if pool._attached:
        raise RuntimeError("The pool already has active expert forwards")

    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Use Transformers {TRANSFORMERS_VERSION} for the audited eager body")
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    # use_experts_implementation installs a @wraps(original_forward) dispatch
    # function. Use its original body for the transfer controls and prefill,
    # so framework dispatch cannot silently choose grouped GEMMs.
    eager_forward = getattr(Qwen3_5MoeExperts.forward, "__wrapped__", None)
    if not callable(eager_forward):
        raise RuntimeError("The pinned Qwen eager expert forward is unavailable")
    layers = [_Layer(name, module, eager_forward, pool, mode)
              for name, module in model.named_modules() if type(module) is Qwen3_5MoeExperts]
    if not layers:
        raise ValueError("No Transformers Qwen3_5MoeExperts modules found")

    pool._check_context()
    if mode in ("scheduled", "grouped", "pipelined", "fused", "batched", "cpu", "seeded", "hybrid"):
        tree = ast.parse(dedent(getsource(eager_forward)))
        # Newer Python omits empty fields by default. Keep the 3.12 AST dump
        # representation used to audit the installed Transformers wheel.
        options = {"show_empty": True} if "show_empty" in signature(ast.dump).parameters else {}
        fingerprint = sha256(ast.dump(tree, include_attributes=False, **options).encode()).hexdigest()
        if fingerprint != SCHEDULED_EAGER_AST_SHA256:
            raise RuntimeError("Scheduled decode requires the audited eager expert source")
        if pool._schedule_slots is None:
            pool._schedule_slots = pool._torch.arange(
                SCHEDULED_TOP_K, dtype=pool._torch.long, device=pool.device).split(1)
    if mode in ("seeded", "hybrid"):
        pool._seed_rows = pool._capacity_bytes // sum(layer.row_bytes for layer in layers)
    if mode in ("packed", "pipelined"):
        layouts = {(tuple(layer.gate_up.shape[1:]), tuple(layer.down.shape[1:])) for layer in layers}
        if len(layouts) != 1:
            raise ValueError("Packed expert layers must share one gate/down layout")
        layout = layouts.pop()
        if mode == "pipelined":
            from edge35b.pipeline import TransferPipeline

            if pool._pipeline is None:
                pool._pipeline = TransferPipeline(pool, *layout)
            elif pool._pipeline.layout != layout:
                raise ValueError("Use a new pool for a different pipelined expert layout")
        elif pool._packed is None:
            pool._packed = _PackedBuffers(pool, layout)
        elif pool._packed.layout != layout:
            raise ValueError("Use a new pool for a different packed expert layout")
    pool._attached = True
    installed = []
    try:
        for layer in layers:
            layer.module.forward = layer.forward
            installed.append(layer)
        yield pool
    finally:
        for layer in reversed(installed):
            del layer.module.forward
        pool._attached = False
