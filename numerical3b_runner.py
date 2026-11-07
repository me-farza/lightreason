"""Separately labeled original-weight speculative runner, relaxed FP order.

The original BF16 target verifies every committed token. Batched GEMM and
attention may round differently from the strict serial HF implementation;
exact token agreement and answer quality must therefore be reported separately.
The pretrained PARD model only proposes. It never authorizes an output or EOS.
"""
from __future__ import annotations

import time
import torch

from pard_proposer import PARDProposer
from relaxed3b_verifier import Relaxed3BVerifier


class Numerical3BRunner:
    def __init__(self, model, tokenizer, draft_path, *, max_len=1024,
                 block_width=8, cache_mode="static", cuda_graph=True,
                 compile_static=False, draft_engine="hf", draft_attention="sdpa",
                 share_pool=True, warmups=3):
        if type(block_width) is not int or not 2 <= block_width <= 32:
            raise ValueError("Use target block widths2..32")
        self.model, self.tokenizer = model, tokenizer
        self.max_len, self.block_width = max_len, block_width
        self.draft_tokens = block_width - 1
        self.verifier = Relaxed3BVerifier(model, max_len=max_len,
            cache_mode=cache_mode, cuda_graph=cuda_graph,
            compile_static=compile_static, share_pool=share_pool,
            warmups=warmups, block_sizes=(block_width,))
        base = PARDProposer(draft_path, target_tokenizer=tokenizer,
            device=model.device, cache_mode="incremental",
            attn_implementation=draft_attention, max_cache_len=max_len)
        if draft_engine == "graph":
            from graph_pard_proposer import GraphPARDProposer
            self.proposer = GraphPARDProposer(base, max_len=max_len,
                share_pool=share_pool, warmups=warmups)
        elif draft_engine == "hf":
            self.proposer = base
        else:
            raise ValueError("Use draft_engine hf or graph")
        self.visible = torch.empty((1, max_len), device=model.device, dtype=torch.long)
        self.setup_s, self.prepared = 0.0, False
        self.metadata = {"variant": "same_target_weights_numerical_order_speculation",
            "strict_token_matching_required": False,
            "exact_token_agreement_must_be_reported": True,
            "target_weights_and_equations_unchanged": True,
            "target_verifier": self.verifier.metadata,
            "proposer": base.metadata, "draft_engine": draft_engine,
            "block_width": block_width, "max_len": max_len,
            "generation": "raw target greedy argmax, original chat template; target EOS only"}

    def prepare(self, ids, n):
        self._check(ids, n)
        if self.prepared:
            return {"setup_s": 0.0, "new_graphs": 0}
        torch.cuda.synchronize()
        start = time.perf_counter()
        prepared = self.verifier.prepare((self.block_width,))
        draft_prepared = None
        if hasattr(self.proposer, "prepare"):
            draft_prepared = self.proposer.prepare(self.draft_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.setup_s += elapsed
        self.prepared = True
        return {"setup_s": elapsed, "target": prepared,
                "draft": draft_prepared, "new_graphs": len(self.verifier.graphs)}

    def _check(self, ids, n):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Only one nonempty unpadded target-tokenized prompt is supported")
        if type(n) is not int or n < 1:
            raise ValueError("Use a positive public output budget")
        # Reserve one full public verification block, even on the final round.
        # Neither draft length nor proposal construction learns an EOS distance.
        if ids.shape[1] + n + self.block_width > self.max_len:
            raise ValueError("Prompt, full output budget and speculative reserve exceed max_len")

    @torch.no_grad()
    def run(self, ids, n, eos_ids=None):
        self._check(ids, n)
        prepared = self.prepare(ids, n)
        eos = set(eos_ids or [])
        with torch.inference_mode(False), torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.proposer.reset()
            self.visible[:, :ids.shape[1]].copy_(ids)
            anchor = self.verifier.prefill(ids)
            self.visible[:, ids.shape[1]:ids.shape[1] + 1].copy_(anchor)
            tokens = [int(anchor.item())]
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            cycles = []
            while len(tokens) < n and (not eos or tokens[-1] not in eos):
                visible_length = ids.shape[1] + len(tokens)
                proposal = self.proposer.propose(self.visible[:, :visible_length],
                                                 num_draft_tokens=self.draft_tokens)
                if proposal.shape != (1, self.draft_tokens) or proposal.dtype != torch.long:
                    raise RuntimeError("Auxiliary proposer returned an invalid public block")
                block = torch.cat((anchor, proposal), dim=1)
                position = self.verifier.current_position
                predicted = self.verifier.verify(block, position)
                # This sync and acceptance cost are part of online latency.
                predicted_ids = predicted.reshape(-1).cpu().tolist()
                draft_ids = proposal.reshape(-1).cpu().tolist()
                count = 1
                while count < self.block_width and draft_ids[count - 1] == predicted_ids[count - 1]:
                    count += 1
                proposed_commit = count
                emitted = predicted_ids[:min(count, n - len(tokens))]
                if eos:
                    terminal = next((i for i, token in enumerate(emitted) if token in eos), None)
                    if terminal is not None:
                        emitted = emitted[:terminal + 1]
                count = len(emitted)
                if count < 1:
                    raise RuntimeError("A nonterminal speculative cycle failed to advance")
                self.visible[:, visible_length:visible_length + count].copy_(predicted[:count].view(1, -1))
                tokens.extend(emitted)
                self.verifier.commit(count)
                anchor = self.verifier.anchor
                if anchor is None or int(anchor.item()) != tokens[-1]:
                    raise RuntimeError("Target cache/anchor acceptance invariant failed")
                cycles.append({"base_position": position, "matching_drafts": proposed_commit - 1,
                               "potential_commits": proposed_commit, "committed_tokens": count})
            torch.cuda.synchronize()
            end = time.perf_counter()
        self.verifier.assert_target_unchanged()
        self.proposer.assert_draft_unchanged()
        decode_s = end - prefill_end
        metrics = {"prefill_s": prefill_end - start, "decode_s": decode_s,
            "total_s": end - start, "decode_tokens_s": (len(tokens) - 1) / decode_s if len(tokens) > 1 else 0.0,
            "generated_tokens": len(tokens), "cycles": len(cycles),
            "mean_committed_tokens_per_cycle": (len(tokens) - 1) / len(cycles) if cycles else 0.0,
            "mean_potential_commits_per_cycle": sum(c["potential_commits"] for c in cycles) / len(cycles) if cycles else 0.0,
            "excluded_setup_s": prepared["setup_s"], "cumulative_setup_s": self.setup_s,
            "stop_reason": "eos" if tokens[-1] in eos else "max_new_tokens" if eos else "fixed_budget",
            "all_output_tokens_target_verified": True,
            "decode_includes_auxiliary_prefill_drafting_verification_acceptance_and_cpu_transfers": True,
            "variant": self.metadata["variant"]}
        self.last_cycles = cycles
        return tokens, metrics


class NumericalGenerationAdapter:
    def __init__(self, runner, tokenizer, eos_ids):
        self.runner, self.tokenizer, self.eos_ids = runner, tokenizer, eos_ids

    def __call__(self, prompt, max_new_tokens):
        ids = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(self.runner.model.device)
        tokens, metrics = self.runner.run(ids, max_new_tokens, eos_ids=self.eos_ids)
        return {"token_ids": tokens, "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "raw_text": self.tokenizer.decode(tokens, skip_special_tokens=False),
            "stop_reason": metrics["stop_reason"], "metrics": metrics}


def make_quality_runner(model, tokenizer, args):
    settings = {key: getattr(args, key) for key in ("block_width", "cache_mode", "cuda_graph",
        "compile_static", "draft_engine", "draft_attention", "share_pool", "warmups") if hasattr(args, key)}
    runner = Numerical3BRunner(model, tokenizer, args.draft_path, max_len=args.max_len, **settings)
    values = model.generation_config.eos_token_id
    eos = [values] if isinstance(values, int) else list(values or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner, tokenizer, sorted(set(eos)))
