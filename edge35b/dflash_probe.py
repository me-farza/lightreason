"""Benchmark DFlash proposals verified by the unmodified BF16 target weights."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from time import perf_counter

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half

from edge35b.benchmark import PROMPTS, generate, load_target, summarize, tokenize, validate_bytes, write_report
from edge35b.delta_decode import fused_delta
from edge35b.download import GIB, REVISION
from edge35b.expert_stream import ExpertPool, patch_experts


class Draft(nn.Module):
    """DFlash draft with a cache containing only committed target features."""

    def __init__(self, path):
        super().__init__()
        raw = json.loads((path / "config.json").read_text())
        config = Qwen3Config.from_pretrained(path, local_files_only=True)
        self.layer_ids = raw["dflash_config"]["target_layer_ids"]
        self.mask_id = raw["dflash_config"]["mask_token_id"]
        self.block_size = raw["dflash_config"]["block_size"]
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.types = config.layer_types
        self.rotary = Qwen3RotaryEmbedding(config)
        with torch.device("meta"):
            self.layers = nn.ModuleList(Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers))
            self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.fc = nn.Linear(len(self.layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.load_state_dict(load_file(path / "model.safetensors"), assign=True)
        self.eval().requires_grad_(False).cuda()

    def forward(self, features, hidden, prefix, past):
        width = hidden.shape[1]
        context = self.hidden_norm(self.fc(features))
        positions = torch.arange(prefix - context.shape[1], prefix + width, device=hidden.device)[None, :]
        cos, sin = (tensor.unsqueeze(1) for tensor in self.rotary(hidden, positions))
        query_pos = positions[0, -width:, None]
        offset = torch.arange(prefix + width, device=hidden.device)[None, :]
        sliding_mask = (offset <= query_pos) & (query_pos - offset < self.window)
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            normal = layer.input_layernorm(hidden)
            context_and_noise = torch.cat((context, normal), dim=1)
            query = attn.q_norm(attn.q_proj(normal).unflatten(-1, (-1, self.head_dim))).transpose(1, 2)
            keys = attn.k_norm(attn.k_proj(context_and_noise).unflatten(-1, (-1, self.head_dim))).transpose(1, 2)
            values = attn.v_proj(context_and_noise).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
            query = query * cos[..., -width:, :] + rotate_half(query) * sin[..., -width:, :]
            keys = keys * cos + rotate_half(keys) * sin
            if past[index] is not None:
                old_keys, old_values = past[index]
                keys = torch.cat((old_keys, keys), dim=-2)
                values = torch.cat((old_values, values), dim=-2)
            if keys.shape[-2] != prefix + width:
                raise RuntimeError("DFlash context does not align with the committed target prefix")
            past[index] = (keys[..., :prefix, :], values[..., :prefix, :])
            mask = sliding_mask if self.types[index] == "sliding_attention" else None
            result = scaled_dot_product_attention(query, keys, values, attn_mask=mask, enable_gqa=True)
            result = result.transpose(1, 2).reshape(1, width, -1)
            hidden = hidden + attn.o_proj(result)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.norm(hidden)


def features_from(output, draft, length):
    return torch.cat([output.hidden_states[i + 1][:, :length] for i in draft.layer_ids], dim=-1)


@torch.inference_mode()
def decode(model, draft, ids, cap, eos, pool, width, histories, on_token=None, profile_cycles=False,
           adaptive=False):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pool.reset_stats()
    started = perf_counter()
    output = model(input_ids=ids, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    cache = output.past_key_values
    token_ids = [int(output.logits[0, 0].argmax().item())]
    first = perf_counter()
    prefill_experts = pool.snapshot()
    if on_token is not None:
        on_token(token_ids, first - started)
    prefix = ids.shape[1]
    context = features_from(output, draft, prefix)
    del output
    draft_cache = [None] * len(draft.layers)
    accepted_per_block = []
    committed_per_block = []
    positions_per_block = []
    proposed = verified = 0
    cycle_samples = []
    pool.cache_multi_token = True
    try:
        while len(token_ids) < cap and token_ids[-1] not in eos:
            if profile_cycles:
                cycle_start = perf_counter()
                start_event = torch.cuda.Event(enable_timing=True)
                draft_end = torch.cuda.Event(enable_timing=True)
                target_end = torch.cuda.Event(enable_timing=True)
                commit_end = torch.cuda.Event(enable_timing=True)
                start_event.record()
            count = min(width, cap - len(token_ids))
            block = torch.full((1, count), draft.mask_id, device=ids.device, dtype=torch.long)
            block[0, 0] = token_ids[-1]
            if count > 1:
                hidden = draft(context, model.get_input_embeddings()(block), prefix, draft_cache)
                block[:, 1:] = model.lm_head(hidden[:, 1:]).argmax(dim=-1)
            if profile_cycles:
                draft_end.record()
            with histories.record(cache):
                output = model(input_ids=block, past_key_values=cache, use_cache=True, output_hidden_states=True)
            # Pull all decisions in one synchronization. The bonus is always the target argmax.
            decisions = torch.cat((output.logits.argmax(dim=-1), block[:, 1:]), dim=-1)[0].tolist()
            if profile_cycles:
                target_end.record()
            prediction, guesses = decisions[:count], decisions[count:]
            matches = 0
            for guess, chosen in zip(guesses, prediction):
                if guess != chosen:
                    break
                matches += 1
            proposed += len(guesses)
            verified += count
            accepted_per_block.append(matches)
            added = 0
            for chosen in prediction[:matches + 1]:
                token_ids.append(chosen)
                added += 1
                if chosen in eos or len(token_ids) == cap:
                    break
            committed_per_block.append(added)
            positions_per_block.append(count)
            if adaptive and count == width:
                if added >= width - 1:
                    width = min(width + 4, draft.block_size)
                elif added <= width // 2:
                    width = max(4, width - 4)
            if on_token is not None:
                on_token(token_ids, perf_counter() - started)
            stopped = len(token_ids) == cap or token_ids[-1] in eos
            if not stopped:
                histories.commit(cache, prefix, added, count)
                context = features_from(output, draft, added)
                prefix += added
            if profile_cycles:
                commit_end.record()
                commit_end.synchronize()
                cycle_samples.append({"committed": added, "verified_positions": count,
                                      "wall_ms_with_instrumentation": (perf_counter() - cycle_start) * 1000,
                                      "draft_stream_span_ms": start_event.elapsed_time(draft_end),
                                      "verify_and_decide_stream_span_ms": draft_end.elapsed_time(target_end),
                                      "commit_and_observe_stream_span_ms": target_end.elapsed_time(commit_end)})
            if stopped:
                break
    finally:
        pool.cache_multi_token = False
    torch.cuda.synchronize()
    end = perf_counter()
    return dict(token_ids=token_ids, stop_reason="eos" if token_ids[-1] in eos else "limit",
                time_s=end - started, first_token_s=first - started, decode_s=end - first,
                decode_tokens=len(token_ids) - 1, decode_tokens_per_s=(len(token_ids) - 1) / (end - first),
                proposed=proposed, accepted=sum(accepted_per_block), verified=verified,
                accepted_per_block=accepted_per_block, committed_per_block=committed_per_block,
                positions_per_block=positions_per_block, peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                prefill_experts=prefill_experts, total_experts=pool.snapshot(), logits_sha256=None,
                **({"cycle_samples": cycle_samples} if profile_cycles else {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--modes", default="hybrid,delta,flash4,flash8")
    parser.add_argument("--cases", default="math,code,logic")
    parser.add_argument("--prompt", help="Run one explicit prompt instead of the built-in cases")
    parser.add_argument("--also-full-answer", action="store_true")
    parser.add_argument("--cache-gib", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--grouped-verifier", action="store_true")
    parser.add_argument("--hybrid-verifier", action="store_true", help="Run cold singleton verify experts on the CPU")
    parser.add_argument("--stream", action="store_true", help="Include live token-observation copies in timing")
    parser.add_argument("--profile-cycles", action="store_true", help="Record synchronized phase spans for diagnosis")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if args.out.exists() or not 2 <= args.tokens <= 1024 or args.repeats < 1:
        parser.error("Use a new result file, 2–1024 tokens and at least one repeat")
    cases = {"custom": args.prompt} if args.prompt else {name: PROMPTS[name] for name in args.cases.split(",")}
    widths = {mode: int(mode[5:]) for mode in modes if mode.startswith("flash") and mode[5:].isdigit()}
    if "flash-adaptive" in modes:
        widths["flash-adaptive"] = 8
    if set(modes) - {"scheduled", "hybrid", "delta", "delta-reuse", "delta-packed", *widths} or any(not 2 <= w <= 16 for w in widths.values()):
        parser.error("Choose scheduled, hybrid, delta, delta-reuse, delta-packed, flash-adaptive or flashN (2–16)")
    if widths and args.draft_dir is None:
        parser.error("Provide the verified DFlash draft weights")
    cache_gib = args.cache_gib if args.cache_gib is not None else (14 if widths else 15)
    if not 12 <= cache_gib <= 15:
        parser.error("Choose a cache between 12 and 15 GiB")
    model, tokenizer, eos, setup = load_target(args.model_dir)
    draft = Draft(args.draft_dir) if widths else None
    if widths and max(widths.values()) > draft.block_size:
        parser.error(f"The draft checkpoint supports at most {draft.block_size} positions")
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < (cache_gib + 3) * GIB:
        raise RuntimeError(f"The probe needs {cache_gib} GiB of expert cache and 3 GiB of remaining VRAM")
    pool = ExpertPool(capacity_bytes=cache_gib * GIB)
    pool.grouped_verifier = args.grouped_verifier or args.hybrid_verifier
    pool.hybrid_verifier = args.hybrid_verifier
    report = dict(setup=setup, target_revision=REVISION, cap=args.tokens, modes=modes,
                  expert_cache_gib=cache_gib, streaming=args.stream, profiled_cycles=args.profile_cycles,
                  grouped_verifier=pool.grouped_verifier, hybrid_verifier=pool.hybrid_verifier, rows=[])

    def run(mode, inputs, cap):
        pool.clear()
        torch.cuda.empty_cache()
        pool._capacity_bytes = (12 if mode == "scheduled" else cache_gib) * GIB
        pool.hybrid_reuse_window = 4 if mode == "delta-reuse" else None
        pool.packed_hybrid = mode == "delta-packed"
        route = "scheduled" if mode == "scheduled" else "hybrid"
        context = nullcontext() if mode in ("scheduled", "hybrid") else fused_delta(model)
        updates = []

        def observe(tokens, elapsed):
            updates.append((list(tokens), elapsed))

        callback = observe if args.stream else None
        with patch_experts(model, mode=route, pool=pool), context as histories:
            if mode in widths:
                result = decode(model, draft, inputs, cap, eos, pool, widths[mode], histories,
                                on_token=callback, profile_cycles=args.profile_cycles,
                                adaptive=mode == "flash-adaptive")
            else:
                result = generate(model, inputs, cap, eos, pool, on_token=callback)
        if callback is not None:
            if updates[-1][0] != result["token_ids"]:
                raise RuntimeError("The streamed and returned token IDs diverged")
            result["stream_updates"] = len(updates)
        return result

    warm_ids = tokenize(tokenizer, "Count to three.", True)
    report["warmup_s"] = {}
    for mode in modes:
        warm = run(mode, warm_ids, widths.get(mode, 1) + 1)
        report["warmup_s"][mode] = warm["time_s"]
        print(f"Warmed {mode} in {warm['time_s']:.2f}s", flush=True)
    if args.also_full_answer:
        cases["full-answer"] = "What is 17 times 19?"
    for index, (name, prompt) in enumerate(cases.items()):
        ids = tokenize(tokenizer, prompt, True)
        case_cap = 768 if name == "full-answer" else args.tokens
        for repeat in range(args.repeats):
            rotation = (index + repeat) % len(modes)
            row = dict(case=name, prompt=prompt, cap=case_cap, repeat=repeat, runs=[])
            report["rows"].append(row)
            for mode in modes[rotation:] + modes[:rotation]:
                result = run(mode, ids, case_cap)
                result.update(mode=mode, text=tokenizer.decode(result["token_ids"], skip_special_tokens=True))
                row["runs"].append(result)
                report["summary"] = summarize(report["rows"], modes)
                write_report(args.out, report)
                print(name, repeat, mode, round(result["decode_tokens_per_s"], 2),
                      result.get("accepted"), result.get("proposed"), flush=True)
    pool.clear()
    report["post_run_checkpoint_bytes_match"] = validate_bytes(model, args.model_dir) == setup["target_parameters"]
    write_report(args.out, report)


if __name__ == "__main__":
    main()
