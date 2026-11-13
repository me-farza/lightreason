"""Localize a speculative next-token disagreement and test prefix rollback."""

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from edge35b.benchmark import PROMPTS, load_target, tokenize, validate_bytes, write_report
from edge35b.delta_decode import fused_delta
from edge35b.dflash_probe import Draft, decode
from edge35b.download import GIB
from edge35b.expert_stream import ExpertPool, patch_experts


def distance(a, b):
    if a.shape != b.shape:
        return {"shapes": [list(a.shape), list(b.shape)]}
    af, bf = a.float(), b.float()
    return {"changed": int((a != b).sum()), "count": a.numel(),
            "max_abs": float((af - bf).abs().max()), "rms": float((af - bf).square().mean().sqrt())}


def state_diff(a, b):
    rows = []
    for layer, (first, second) in enumerate(zip(a.layers, b.layers, strict=True)):
        row = {"layer": layer}
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            left = getattr(first, name, None)
            right = getattr(second, name, None)
            if left is not None and right is not None:
                row[name] = distance(left, right)
        rows.append(row)
    return {"lengths": [a.get_seq_length(), b.get_seq_length()], "layers": rows}


def scores(a, b):
    candidates = sorted(set(a.topk(6).indices.tolist() + b.topk(6).indices.tolist()))
    return {"argmax": [int(a.argmax()), int(b.argmax())],
            "margins": [float(x.float().topk(2).values.diff().abs()[0]) for x in (a, b)],
            "leading_logits": {int(i): [float(a[i]), float(b[i])] for i in candidates},
            "distance": distance(a, b)}


@contextmanager
def record_routes(model, enabled=lambda: True):
    routes = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def capture(_module, args, index=index):
            if enabled():
                routes[index] = (args[1].detach().clone(), args[2].detach().clone())
        handles.append(layer.mlp.experts.register_forward_pre_hook(capture))
    try:
        yield routes
    finally:
        for handle in handles:
            handle.remove()


def compare_routes(a, b, column):
    return [{"layer": i, "serial": a[i][0][-1].tolist(), "block": b[i][0][column].tolist(),
             "weight_difference": distance(a[i][1][-1], b[i][1][column])}
            for i in a if i in b]


@contextmanager
def record_first_layer(model, enabled=lambda: True):
    outputs = {}
    handles = []
    layer = model.model.layers[0]
    for name, module in layer.named_modules():
        if name in ("input_layernorm", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                    "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj",
                    "linear_attn", "post_attention_layernorm", "mlp.experts", "mlp.shared_expert", "mlp"):
            def capture(_module, _args, output, name=name):
                if enabled() and isinstance(output, torch.Tensor):
                    outputs[name] = output.detach().reshape(-1, output.shape[-1]).clone()
            handles.append(module.register_forward_hook(capture))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def compare_first_layer(reference, candidate, column):
    return {key: distance(tensor[-1], candidate[key][column])
            for key, tensor in reference.items() if key in candidate}


@torch.inference_mode()
def investigate(model, tokenizer, eos, draft, pool, index):
    prompt = tokenize(tokenizer, PROMPTS["logic"], True)
    prompt_len = prompt.shape[-1]
    reference = []
    replay_states = {}
    recent_prefixes = {}
    pool.packed_hybrid = False
    with patch_experts(model, mode="hybrid", pool=pool), record_routes(model) as routes, record_first_layer(model) as internals:
        output = model(input_ids=prompt, use_cache=True, output_hidden_states=True, logits_to_keep=1)
        cache = output.past_key_values
        for step in range(index + 5):
            token = int(output.logits[0, -1].argmax())
            reference.append(token)
            if step == index - 1:
                clean_prefix = deepcopy(cache)
            if step == index:
                reference_output = output
                reference_routes = dict(routes)
                reference_internals = dict(internals)
            if index <= step <= index + 3:
                replay_states[step - index + 1] = deepcopy(cache)
            if index - 5 <= step <= index - 1:
                recent_prefixes[cache.get_seq_length()] = deepcopy(cache)
            output = model(input_ids=prompt.new_tensor([[token]]), past_key_values=cache, use_cache=True,
                           output_hidden_states=True, logits_to_keep=1)
    pool.clear()
    torch.cuda.empty_cache()

    live = {}
    decision_input = prompt_len + index - 1

    def before(_module, _args, kwargs):
        ids, cache = kwargs["input_ids"], kwargs.get("past_key_values")
        if cache is not None:
            start = cache.get_seq_length()
            if start <= decision_input < start + ids.shape[-1]:
                live.update(start=start, block=ids[0].tolist(), capture=True, start_cache=deepcopy(cache))

    def after(_module, _args, _kwargs, output):
        if live.get("capture"):
            column = decision_input - live["start"]
            live["output"] = output
            live["column"] = column
            live["capture"] = False

    pool.grouped_verifier = pool.hybrid_verifier = True
    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
               model.register_forward_hook(after, with_kwargs=True)]
    try:
        with patch_experts(model, mode="hybrid", pool=pool), fused_delta(model) as history:
            enabled = lambda: live.get("capture", False)
            with record_routes(model, enabled) as routes, record_first_layer(model, enabled) as internals:
                actual = decode(model, draft, prompt, index + 9, eos, pool, 4, history)
                actual_routes = dict(routes)
                actual_internals = dict(internals)
    finally:
        for handle in handles:
            handle.remove()
    live_output = live["output"]
    column = live["column"]
    serial_logits = reference_output.logits[0, -1]
    live_logits = live_output.logits[0, column]
    result = {"prompt_ids": prompt[0].tolist(), "reference_ids": reference,
              "actual_ids": actual["token_ids"], "suspect_index_zero_based": index,
              "first_difference_zero_based": next((i for i, (a, b) in enumerate(
                  zip(reference, actual["token_ids"])) if a != b), None),
              "same_generated_prefix": reference[:index] == actual["token_ids"][:index],
              "actual_block": {"input_ids": live["block"], "absolute_start": live["start"], "column": column,
                               "reference_input_prefix": reference[live["start"] - prompt_len:index],
                               "next_token": scores(serial_logits, live_logits),
                               "starting_state_vs_serial": state_diff(recent_prefixes[live["start"]], live["start_cache"]),
                               "first_layer": compare_first_layer(reference_internals, actual_internals, column),
                               "routes": compare_routes(reference_routes, actual_routes, column),
                               "hidden_states": [distance(a[0, -1], b[0, column]) for a, b in
                                                 zip(reference_output.hidden_states, live_output.hidden_states)]},
              "isolations": []}
    # Rebuild the exact same cache and residency history for each arm.
    # A clean sequential prefix ends just before the token predicting the suspect.
    def prepared():
        output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        for token in reference[:index - 1]:
            model(input_ids=prompt.new_tensor([[token]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        return cache

    variants = ("serial-hybrid", "serial-packed", "one-token", "oracle-gpu", "oracle-hybrid",
                "reject-first", "reject-middle", "reject-first-gpu", "reject-middle-gpu")
    oracle_states = {}
    for variant in variants:
        pool.clear()
        torch.cuda.empty_cache()
        pool.packed_hybrid = False
        pool.cache_multi_token = False
        pool.hybrid_verifier = not variant.endswith("-gpu")
        with patch_experts(model, mode="hybrid", pool=pool):
            cache = prepared()
            assert cache.get_seq_length() == clean_prefix.get_seq_length()
            starting_state = state_diff(clean_prefix, cache)
            prefix = cache.get_seq_length()
            pool.packed_hybrid = variant == "serial-packed"
            pool.cache_multi_token = True
            if variant in ("serial-hybrid", "serial-packed", "one-token"):
                ids = reference[index - 1:index]
            else:
                ids = reference[index - 1:index + 3]
            accepted = len(ids)
            if variant.startswith("reject"):
                accepted = 1 if variant.startswith("reject-first") else 2
                ids[accepted] = (ids[accepted] + 17) % model.config.vocab_size

            serial = variant == "serial-hybrid"
            manager = nullcontext() if serial else fused_delta(model)
            with manager as history, record_routes(model) as routes, record_first_layer(model) as internals:
                with history.record(cache) if history else nullcontext():
                    candidate = model(input_ids=prompt.new_tensor([ids]), past_key_values=cache,
                                      use_cache=True, output_hidden_states=True)
                candidate_routes = dict(routes)
                candidate_internals = dict(internals)
                if variant == "oracle-gpu":
                    for accepted_length in (1, 2):
                        clone = deepcopy(cache)
                        history.commit(clone, prefix, accepted_length, len(ids))
                        oracle_states[accepted_length] = clone
                if history:
                    history.commit(cache, prefix, accepted, len(ids))
            predictions = candidate.logits[0].argmax(dim=-1).tolist()
            accepted_by_argmax = 1
            for proposed, prediction in zip(ids[1:], predictions):
                if proposed != prediction:
                    break
                accepted_by_argmax += 1
            result["isolations"].append({
                "variant": variant, "block": ids, "committed_inputs": accepted,
                "accepted_by_argmax": accepted_by_argmax, "predictions": predictions,
                "starting_state": starting_state,
                "next_token": scores(serial_logits, candidate.logits[0, 0]),
                "routes": compare_routes(reference_routes, candidate_routes, 0),
                "first_layer": compare_first_layer(reference_internals, candidate_internals, 0),
                "hidden_states": [distance(a[0, -1], b[0, 0]) for a, b in
                                  zip(reference_output.hidden_states, candidate.hidden_states)],
                "committed_vs_serial": state_diff(replay_states[accepted], cache),
                "committed_vs_same_width_oracle": state_diff(oracle_states[accepted], cache)
                if variant.startswith("reject") and variant.endswith("-gpu") else None,
            })
        print(variant, result["isolations"][-1]["next_token"]["argmax"], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--draft-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model, tokenizer, eos, setup = load_target(args.model_dir)
    draft = Draft(args.draft_dir)
    pool = ExpertPool(capacity_bytes=14 * GIB)
    result = investigate(model, tokenizer, eos, draft, pool, 33)
    pool.clear()
    result["checkpoint_parameters_after"] = validate_bytes(model, args.model_dir)
    result["setup"] = setup
    write_report(args.out, result)


if __name__ == "__main__":
    main()
