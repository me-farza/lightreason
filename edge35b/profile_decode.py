"""Measure the scheduled runner and collect exact expert access traces."""

import argparse
from pathlib import Path
from time import perf_counter

from edge35b.benchmark import PROMPTS, generate, load_target, tokenize, write_report
from edge35b.download import GIB
from edge35b.expert_stream import ExpertPool, patch_experts


class TracePool(ExpertPool):
    def reset_stats(self):
        super().reset_stats()
        self.routes = []
        self.copy_host_s = 0.0

    def _copy(self, gate_up, down, *, rows):
        started = perf_counter()
        result = super()._copy(gate_up, down, rows=rows)
        if self.routes:
            self.copy_host_s += perf_counter() - started
        return result

    def _expert(self, layer, index, *, cached):
        if cached:
            self.routes.append((layer.name, index))
        return super()._expert(layer, index, cached=cached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    if args.out.exists() or not 2 <= args.tokens <= 128:
        parser.error("Use a fresh output and a token cap from 2 to 128")
    model, tokenizer, eos, setup = load_target(args.model_dir)
    pool = TracePool(capacity_bytes=12 * GIB)
    report = {"setup": setup, "tokens": args.tokens, "cases": {}}
    with patch_experts(model, mode="scheduled", pool=pool):
        generate(model, tokenize(tokenizer, "Count to three.", True), 2, eos, pool)
        for case, prompt in PROMPTS.items():
            pool.clear()
            run = generate(model, tokenize(tokenizer, prompt, True), args.tokens, eos, pool)
            run["routes"] = pool.routes
            run["copy_host_s"] = pool.copy_host_s
            report["cases"][case] = run
            write_report(args.out, report)
            print(case, run["decode_tokens_per_s"], pool.copy_host_s, flush=True)


if __name__ == "__main__":
    main()
