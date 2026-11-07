"""Original BF16 Qwen2.5-3B with exactness-gated CUDA graph optimizations.

The baseline always uses the original HF model. This runner has a separate
decode core; prefill and original target weights are never replaced.
"""
import time
import torch

from exact3b_graphs import LinearExactGraphRunner
from exact3b_core import Exact3BDecoder
from exact_graph import ExactNaturalEOS


class Exact3BRunner(LinearExactGraphRunner):
    def __init__(self, model, max_len=1024, max_output=512, warmups=3,
                 norm_mode="split", pointwise="compiled", gate_mode="compiled",
                 fuse_cache=False, mean_threads=None, norm_warps=4,
                 share_pool=False, track_logits=False, screen_head=False,
                 verifier_lanes=16, verify_head_steps=False,
                 combine_projections=False, combine_mode="kv"):
        super().__init__(model, max_len, max_output, warmups, share_pool,
                         track_logits, precompute_rope=False)
        self.decoder = Exact3BDecoder(model, self.cache, norm_mode=norm_mode,
            pointwise=pointwise, gate_mode=gate_mode, fuse_cache=fuse_cache,
            mean_threads=mean_threads, norm_warps=norm_warps,
            combine_projections=combine_projections, combine_mode=combine_mode)
        self.screen_head = screen_head
        self.verifier_lanes = verifier_lanes
        self.verify_head_steps = verify_head_steps
        self.head_verified_steps = 0
        self.head_index_setup_s = 0.0
        if verify_head_steps and not screen_head:
            raise ValueError("Head audits require a screened head")
        if screen_head:
            from exact3b_head import BoundedHead3B
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.decoder.head_override = BoundedHead3B(model.lm_head,
                verifier_lanes=verifier_lanes, compact=True)
            torch.cuda.synchronize()
            self.head_index_setup_s = time.perf_counter() - start
            if verify_head_steps:
                self.last_head_query = torch.empty((1, 1, model.config.hidden_size),
                    device=self.token.device, dtype=model.lm_head.weight.dtype)
                def save_query(module, arguments):
                    self.last_head_query.copy_(arguments[0])
                self.head_query_hook = self.decoder.head_override.register_forward_pre_hook(save_query)

    def _step(self):
        position = self.capture_position
        logits = self.decoder.forward_one(self.token, self.fixed_positions[position], position)
        selected = logits.argmax(-1)
        if self.track_logits:
            self.last_logits.copy_(logits)
        self.absolute_output[position + 1:position + 2].copy_(selected)
        self.token.copy_(selected.view(1, 1))

    def replay_one(self):
        output = super().replay_one()
        if self.verify_head_steps:
            record = self.decoder.head_override.audit_buffers(self.last_head_query)
            required = ("selected_bits_exact", "all_finite_reference_logits_enclosed",
                        "reference_winner_selected", "argmax_exact")
            if not all(record[key] for key in required):
                raise RuntimeError("Original full-head audit failed: " + str(record))
            self.head_verified_steps += 1
        return output


def make_quality_runner(model, tokenizer, args):
    settings = {name: getattr(args, name) for name in
                ["norm_mode", "pointwise", "gate_mode", "fuse_cache", "mean_threads",
                 "norm_warps", "share_pool", "track_logits", "screen_head",
                 "verifier_lanes", "verify_head_steps", "combine_projections",
                 "combine_mode"] if hasattr(args, name)}
    runner = Exact3BRunner(model, args.max_len, args.max_len, **settings)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return ExactNaturalEOS(runner, tokenizer, eos)
