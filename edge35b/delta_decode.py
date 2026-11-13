"""Short BF16 target Gated DeltaNet updates with selectable prefix states."""

from contextlib import contextmanager

import torch
import triton
import triton.language as tl
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm


@triton.jit(do_not_specialize=["T"])
def _delta(Q, K, V, G, Beta, Initial, Result, States, T,
           H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
           TILE: tl.constexpr, RECORD: tl.constexpr):
    head = tl.program_id(0)
    row = tl.arange(0, D)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    index = (head * D + row[:, None]) * E + col[None, :]
    state = tl.load(Initial + index).to(tl.float32)
    for token in range(T):
        query = tl.load(Q + (token * H + head) * D + row).to(tl.float32) * (D ** -0.5)
        key = tl.load(K + (token * H + head) * D + row).to(tl.float32)
        value = tl.load(V + (token * H + head) * E + col).to(tl.float32)
        state = state * tl.load(G + token * H + head)
        memory = tl.sum(state * key[:, None], axis=0)
        delta = (value - memory) * tl.load(Beta + token * H + head).to(tl.float32)
        state = state + key[:, None] * delta[None, :]
        out = tl.sum(state * query[:, None], axis=0)
        tl.store(Result + (token * H + head) * E + col, out)
        # Serial target decode rounds every committed recurrent state to the cache dtype.
        state = state.to(Initial.dtype.element_ty).to(tl.float32)
        if RECORD:
            tl.store(States + token * H * D * E + index, state)
    if not RECORD:
        tl.store(States + index, state)


def _run(query, key, value, g, beta, initial, *, record, normalize):
    batch, steps, heads, key_width = query.shape
    value_width = value.shape[-1]
    if (batch != 1 or key_width not in (64, 128, 256) or value_width % 16
            or initial.shape != (batch, heads, key_width, value_width)
            or initial.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError(f"Unsupported recurrent layout {query.shape}, {value.shape}, {initial.shape}, {initial.dtype}")
    if normalize:
        query, key = l2norm(query), l2norm(key)
    out = torch.empty_like(value, memory_format=torch.contiguous_format)
    states = initial.new_empty((steps if record else 1, heads, key_width, value_width))
    _delta[(heads, value_width // 16)](query.contiguous(), key.contiguous(), value.contiguous(),
                                      g.exp().contiguous(), beta.contiguous(), initial.contiguous(), out, states,
                                      steps, heads, key_width, value_width, 16, record,
                                      num_warps=4, enable_fp_fusion=False)
    return out, states


class PrefixStates:
    def __init__(self):
        self.conv = {}
        self.recurrent = {}

    @contextmanager
    def record(self, cache):
        self.conv.clear()
        self.recurrent.clear()
        self.cache = cache
        try:
            yield
        finally:
            self.cache = None

    def commit(self, cache, prefix, accepted, width):
        if accepted == width:
            return
        if not 1 <= accepted < width or self.conv.keys() != self.recurrent.keys() or not self.conv:
            raise RuntimeError("Missing target state for the accepted speculative prefix")
        cache.crop(prefix + accepted)
        for index, (old, projected) in self.conv.items():
            layer = cache.layers[index]
            length = old.shape[-1]
            layer.conv_states.copy_(torch.cat((old, projected[..., :accepted]), dim=-1)[..., -length:])
            layer.recurrent_states.copy_(self.recurrent[index][accepted - 1:accepted])


@contextmanager
def fused_delta(model):
    """Keep long prefill stock and record short blocks for causal rollback."""
    patches = []
    histories = PrefixStates()
    histories.cache = None
    try:
        for index, layer in enumerate(model.model.layers):
            if not hasattr(layer, "linear_attn"):
                continue
            mixer = layer.linear_attn

            def projection(_module, _args, output, index=index):
                if histories.cache is not None and output.shape[1] > 1:
                    state = histories.cache.layers[index].conv_states
                    histories.conv[index] = (state.clone(), output.transpose(1, 2))

            def bind(original, index):
                def decode(query, key, value, g, beta, initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False):
                    if initial_state is None or query.shape[1] > 16:
                        return original(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                        output_final_state=output_final_state,
                                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                    record = histories.cache is not None and query.shape[1] > 1
                    output, states = _run(query, key, value, g, beta, initial_state,
                                          record=record, normalize=use_qk_l2norm_in_kernel)
                    if record:
                        histories.recurrent[index] = states
                    return output, states[-1:] if output_final_state else None
                return decode

            chunk, step = mixer.chunk_gated_delta_rule, mixer.recurrent_gated_delta_rule
            hook = mixer.in_proj_qkv.register_forward_hook(projection)
            patches.append((mixer, chunk, step, hook))
            mixer.chunk_gated_delta_rule = bind(chunk, index)
            mixer.recurrent_gated_delta_rule = bind(step, index)
        yield histories
    finally:
        for mixer, chunk, step, hook in patches:
            mixer.chunk_gated_delta_rule = chunk
            mixer.recurrent_gated_delta_rule = step
            hook.remove()
