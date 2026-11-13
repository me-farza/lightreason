"""Test the native Qwen MTP head with BF16 target verification and rollback."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

from safetensors import safe_open
import torch
from torch import nn
from transformers.cache_utils import DynamicCache, LinearAttentionLayer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    torch_recurrent_gated_delta_rule,
)

from edge35b.benchmark import PROMPTS, generate, load_target, summarize, tokenize, validate_bytes, write_report
from edge35b.download import GIB
from edge35b.expert_stream import ExpertPool, patch_experts


class NativeMTP(nn.Module):
    def __init__(self, target, checkpoint):
        super().__init__()
        self.config = deepcopy(target.config)
        self.config.num_hidden_layers = 1
        self.config.layer_types = ["full_attention"]
        dim, eps = self.config.hidden_size, self.config.rms_norm_eps
        with torch.device("meta"):
            self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(dim, eps)
            self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(dim, eps)
            self.fc = nn.Linear(2 * dim, dim, bias=False)
            self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(self.config, 0)])
            self.norm = Qwen3_5MoeRMSNorm(dim, eps)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        files = defaultdict(list)
        for key, file in index.items():
            if key.startswith("mtp."):
                files[file].append(key)
        state = {}
        for file, keys in files.items():
            with safe_open(checkpoint / file, framework="pt", device="cpu") as source:
                for key in keys:
                    state[key.removeprefix("mtp.")] = source.get_tensor(key)
        if any(value.dtype != torch.bfloat16 for value in state.values()):
            raise ValueError("Expected original BF16 weights for the native MTP head")
        self.load_state_dict(state, assign=True, strict=True)
        self.eval().requires_grad_(False).cuda()

    def forward(self, target, hidden, ids, cache):
        embedded = self.pre_fc_norm_embedding(target.model.embed_tokens(ids))
        hidden = self.fc(torch.cat((embedded, self.pre_fc_norm_hidden(hidden)), dim=-1))
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :] + cache.get_seq_length()
        mask = create_causal_mask(config=self.config, inputs_embeds=hidden, attention_mask=None,
                                  past_key_values=cache, position_ids=positions)
        rotary = target.model.rotary_emb(hidden, positions[None, ...].expand(3, -1, -1))
        hidden = self.layers[0](hidden, position_embeddings=rotary, attention_mask=mask,
                                position_ids=positions, past_key_values=cache, use_cache=True)
        return self.norm(hidden)


def snapshot(cache):
    return [(layer, layer.conv_states.clone(), layer.recurrent_states.clone())
            for layer in cache.layers if isinstance(layer, LinearAttentionLayer)]


def restore(cache, length, states):
    cache.crop(length)
    for layer, conv, recurrent in states:
        layer.conv_states.copy_(conv)
        layer.recurrent_states.copy_(recurrent)


def argmax(model, hidden):
    return model.lm_head(hidden).argmax(dim=-1)


@contextmanager
def recurrent_verify(model, enabled):
    """Use the stock recurrent operation for short cached verification blocks."""
    original = []
    if enabled:
        for layer in model.model.layers:
            if not hasattr(layer, "linear_attn"):
                continue
            mixer = layer.linear_attn
            chunk = mixer.chunk_gated_delta_rule

            def run(query, key, value, *, initial_state, chunk=chunk, **kwargs):
                if initial_state is not None and query.shape[1] <= 5:
                    return torch_recurrent_gated_delta_rule(query, key, value, initial_state=initial_state, **kwargs)
                return chunk(query, key, value, initial_state=initial_state, **kwargs)

            original.append((mixer, chunk))
            mixer.chunk_gated_delta_rule = run
    try:
        yield
    finally:
        for mixer, chunk in original:
            mixer.chunk_gated_delta_rule = chunk


@torch.inference_mode()
def speculative_generate(model, draft, ids, cap, eos, pool, draft_tokens):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pool.reset_stats()
    started = perf_counter()
    initial = model.model(input_ids=ids, use_cache=True)
    target_cache = initial.past_key_values
    hidden = initial.last_hidden_state
    current = argmax(model, hidden[:, -1:])
    tokens = [int(current.item())]
    first = perf_counter()
    prefill_transfers = pool.snapshot()
    draft_cache = DynamicCache(config=draft.config)
    if ids.shape[1] > 1 and tokens[-1] not in eos:
        draft(model, hidden[:, :-1], ids[:, 1:], draft_cache)
    last_hidden = hidden[:, -1:]
    del hidden, initial
    rounds = proposed = accepted = replays = 0
    pool.cache_multi_token = True
    try:
        while len(tokens) < cap and tokens[-1] not in eos:
            prefix = target_cache.get_seq_length()
            draft_prefix = draft_cache.get_seq_length()
            if draft_prefix != prefix - 1:
                raise RuntimeError("The native draft and target cache positions diverged")
            pieces = [current]
            draft_hidden = last_hidden
            for _ in range(min(draft_tokens, cap - len(tokens) - 1)):
                draft_hidden = draft(model, draft_hidden, pieces[-1], draft_cache)
                pieces.append(argmax(model, draft_hidden))
            block = torch.cat(pieces, dim=1)
            states = snapshot(target_cache) if block.shape[1] > 1 else None
            verified = model.model(input_ids=block, past_key_values=target_cache, use_cache=True).last_hidden_state
            prediction, guesses = argmax(model, verified)[0].tolist(), block[0, 1:].tolist()
            matches = 0
            for guess, truth in zip(guesses, prediction):
                if guess != truth:
                    break
                matches += 1
            rounds += 1
            proposed += len(guesses)
            if matches < len(guesses):
                restore(target_cache, prefix, states)
                verified = model.model(input_ids=block[:, :matches + 1], past_key_values=target_cache,
                                       use_cache=True).last_hidden_state
                prediction = argmax(model, verified)[0].tolist()
                replays += 1
                if prediction[:matches] != guesses[:matches]:
                    restore(target_cache, prefix, states)
                    verified = model.model(input_ids=current, past_key_values=target_cache,
                                           use_cache=True).last_hidden_state
                    prediction = argmax(model, verified)[0].tolist()
                    matches = 0
                    replays += 1
            accepted += matches
            added = guesses[:matches] + [prediction[matches]]
            for token in added:
                tokens.append(token)
                if token in eos or len(tokens) == cap:
                    break
            if tokens[-1] in eos or len(tokens) == cap:
                break
            last_hidden = verified[:, matches:matches + 1]
            current = torch.tensor([[tokens[-1]]], dtype=torch.long, device=ids.device)
            draft_cache.crop(draft_prefix + 1)
            if matches:
                draft(model, verified[:, :matches], block[:, 1:matches + 1], draft_cache)
    finally:
        pool.cache_multi_token = False
    torch.cuda.synchronize()
    end = perf_counter()
    return {"token_ids": tokens, "stop_reason": "eos" if tokens[-1] in eos else "limit",
            "time_s": end - started, "first_token_s": first - started,
            "decode_s": end - first, "decode_tokens": len(tokens) - 1,
            "decode_tokens_per_s": (len(tokens) - 1) / (end - first),
            "rounds": rounds, "draft_proposed": proposed, "draft_accepted": accepted, "replays": replays,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "prefill_experts": prefill_transfers, "total_experts": pool.snapshot(), "logits_sha256": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--draft-tokens", type=int, default=1)
    parser.add_argument("--recurrent-verify", action="store_true")
    parser.add_argument("--cases", choices=(*PROMPTS, "all"), default="all")
    args = parser.parse_args()
    if args.out.exists() or not 2 <= args.tokens <= 512 or not 1 <= args.draft_tokens <= 4:
        parser.error("Use a fresh output, 2–512 output tokens and 1–4 draft tokens")
    model, tokenizer, eos, setup = load_target(args.model_dir)
    draft = NativeMTP(model, args.model_dir)
    if torch.cuda.mem_get_info()[0] < 15 * GIB:
        raise MemoryError("Need 12 GiB for experts plus at least 3 GiB of CUDA headroom")
    pool = ExpertPool(capacity_bytes=12 * GIB)
    modes = ("grouped", "mtp")
    report = {"setup": setup, "cap": args.tokens, "draft_tokens": args.draft_tokens,
              "recurrent_verify": args.recurrent_verify, "rows": []}
    prompts = PROMPTS if args.cases == "all" else {args.cases: PROMPTS[args.cases]}
    with patch_experts(model, mode="grouped", pool=pool):
        generate(model, tokenize(tokenizer, "Count to three.", True), 2, eos, pool)
        for i, (case, prompt) in enumerate(prompts.items()):
            row = {"case": case, "prompt": prompt, "runs": []}
            report["rows"].append(row)
            ids = tokenize(tokenizer, prompt, True)
            for mode in (modes if i % 2 == 0 else reversed(modes)):
                pool.clear()
                torch.cuda.empty_cache()
                if mode == "mtp":
                    with recurrent_verify(model, args.recurrent_verify):
                        run = speculative_generate(model, draft, ids, args.tokens, eos, pool, args.draft_tokens)
                else:
                    run = generate(model, ids, args.tokens, eos, pool)
                run.update(mode=mode, text=tokenizer.decode(run["token_ids"], skip_special_tokens=True))
                row["runs"].append(run)
                print(case, mode, round(run["decode_tokens_per_s"], 3),
                      run.get("draft_accepted"), run.get("draft_proposed"), flush=True)
                report["summary"] = summarize(report["rows"], modes)
                write_report(args.out, report)
    pool.clear()
    report["post_run_checkpoint_bytes_match"] = validate_bytes(model, args.model_dir) == setup["target_parameters"]
    write_report(args.out, report)


if __name__ == "__main__":
    main()
