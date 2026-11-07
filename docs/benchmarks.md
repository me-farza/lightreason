# Qwen2.5-3B benchmark notes

## Target-only Studio runner

**Standard** runs the original Hugging Face BF16 forward one token at a time, takes the greedy argmax, and stops at the model's first native EOS. **Lightreason Exact** uses the same Qwen weights and greedy stop rules. It replays CUDA graphs to reduce launch overhead, keeps a preallocated compact KV cache, and packs projections while preserving the original BF16 values. The output head uses a conservative bound to screen vocabulary rows, then evaluates surviving rows with the original BF16 weights. The runner checks token matching against Standard in Studio. [Graph and cache code](../exact3b_graphs.py) · [Target-only runner](../exact3b_runner.py)

Use the [Studio quickstart](../README.md#run-on-a-gpu-machine) to reproduce an interactive comparison with just the public Qwen checkpoint. Run the same input once to prepare the required graphs, then repeat it with the same output limit. Record generation time, first-token latency, token IDs and stop reason separately from graph setup and browser rendering.

## GPU coordination experiment

The lab test used the same pinned BF16 target, an RTX 4090 at stock settings, native EOS, and greedy token selection. We froze the original Hugging Face runner and the prior optimized runner before profiling.

1. The prior optimized runner used one PARD2 draft pass with the lab's trained auxiliary LoRA/projector. The adapter was trained for the draft and folded into a separate auxiliary copy. The draft proposed up to 15 positions per path, with a top-8 candidate list at each depth. The original Qwen target verified a packed 64-position tree with ancestor-only attention and correct path positions. The target chose which tokens to commit. The packed target forward could produce small floating-point differences from sequential Hugging Face forwards. [Frozen optimized runner](../graph_prefill_compiled_pard2_lora_tree3b_v1.py)
2. We profiled the full cycle and measured drafting, tree construction, verification, acceptance, KV commitment, host waits, and streaming. The retained change moved the same ranking and tie-breaking, acceptance, and KV compaction onto the GPU. It uses preallocated buffers, appends the verified suffix, and returns one small pinned-memory receipt per cycle. [GPU coordination implementation](../gpu_tree_control3b_v1.py)
3. We reran six separate held-out prompts in alternating order with three repeats in both streaming modes. Preparation was recorded separately. We checked token IDs, stop reasons and answer quality, and ran a larger 160-case output regression against the prior optimized runner.

| Paired comparison | Non-streaming | Streaming |
| --- | ---: | ---: |
| GPU coordination vs. frozen optimized runner | 1.062× | 1.062× |
| GPU coordination vs. original Hugging Face | 9.33× | 9.39× |

The candidate matched the frozen optimized runner on 36/36 held-out comparisons and 160/160 regression cases. It matched original Hugging Face token-for-token on 12/36 held-out comparisons. Two prompts had exact Hugging Face output and exceeded 10× in every repeat. The suite medians versus Hugging Face include differences in output. These figures apply to the optional GPU coordination runner.

On the separately archived weather prompt, the retained runner matched all 492 original output IDs and measured 8.78× non-streaming and 8.88× streaming versus Hugging Face. The kernel times below come from its diagnostic trace. Warm speed ratios come from separate runs without the profiler.

| GPU component | ms per cycle |
| --- | ---: |
| Draft | 3.468 |
| Tree construction | 0.096 |
| Target verification | 9.965 |
| Acceptance | 0.006 |
| KV commitment | 0.004 |

The host tree interval fell from 0.498 ms to 0.132 ms. The host wait for the receipt averaged 11.546 ms and overlapped queued GPU work. The streaming callback averaged 0.034 ms. Held-out non-streaming cycle time fell from 13.927 ms to 13.070 ms at the same average 6.339 committed tokens per cycle. Target verification remained the largest measured GPU cost.

### What else we tested

We used CAST as a reference for trading off acceptance against verification cost. We timed separately captured verifier shapes of 16, 32, 48, 64, 96 and 128 rows on the development set. In the experimental floating-point variant, 16 rows averaged 5.77 committed tokens in 11.68 ms, 64 rows averaged 6.93 in 12.03 ms, and 128 rows averaged 7.18 in 13.60 ms. The 64-row shape led in committed tokens per second. The attention changes tested in that variant failed the larger quality check. We kept the existing 64-row shape and its verification arithmetic for the retained implementation. [CAST paper](https://arxiv.org/abs/2610.00321)

The measured lossless target-weight packing prototype ran slower than the existing GEMM. Auxiliary INT8 head savings were small and a second draft query cost more than its additional accepted tokens saved. We kept GPU coordination because it improved the held-out result and preserved the optimized control's output.

### Reproduce with the matching adapter

Obtain the optional draft and authorized adapter below. The input-only [suite](../optimization_20261106/suite.json) fixes six development and six held-out prompts and their output budgets. This public command reruns the retained configuration with three peers. The lab's four-peer experiment, regression outputs, and original traces are separate artifacts outside this release.

```bash
python benchmark_cost_aware_tree3b_v1.py \
  --controls-manifest optimization_20261106/controls/runtime-manifest.json \
  --model-path model-3b --draft-path models/pard2 \
  --adaptation-checkpoint-path models/adapter.pt \
  --candidate-factory gpu_tree_control3b_v1:GPUTreeControl3BRunner \
  --split heldout --node-budget 64 --repeats 3 --stream-modes both \
  --out /tmp/lightreason-heldout.json
```

The report records preparation, external wall times, token IDs, stop reasons, and pairwise output comparisons. To score the fixed cases for answer correctness and format after timing, extract the response records with `jq` and run the fixture grader.

```bash
jq '[.trials[].runs[]]' /tmp/lightreason-heldout.json > /tmp/lightreason-heldout-responses.json
python quality_cost_aware_tree3b_v1.py --results /tmp/lightreason-heldout-responses.json \
  --out /tmp/lightreason-heldout-quality.json
```

Profile the frozen optimized control on one of the published development cases with the same assets. Use a fresh output path for every invocation. The profiler also writes a Chrome trace beside the report.

```bash
python profile_cost_aware_tree3b_v1.py \
  --controls-manifest optimization_20261106/controls/runtime-manifest.json \
  --model-path model-3b --draft-path models/pard2 \
  --adaptation-checkpoint-path models/adapter.pt \
  --suite optimization_20261106/suite.json --case-id dev-code-prefix \
  --streaming --out /tmp/lightreason-cycle-profile.json
```

For a development budget sweep, use the benchmark command with `--split development --sweep --packed-budgets 16,32,48,64,96,128`, omit `--candidate-factory` to select the cost-aware implementation, and give it a fresh `--out`. Record the actual graph shapes and compare committed tokens per complete cycle time before evaluating a selected setting on held-out inputs. The sweep uses the public cost-aware implementation. The experimental attention composition used for the numerical results above is outside the release.

## Patch-edit experiment

Two alternating repeats per case used one generated-page fixture on the lab's Turbo runner. The CSS and counter changes passed both full-page and JSON patch checks. The JSON heading patch was ambiguous and rejected. The full-page heading edit changed unrelated behavior. The marker format passed heading and CSS but failed the counter edit. A different JavaScript edit also showed that a unique textual match alone cannot establish behavioral correctness.

| Passing case | Full page, median HTTP completion | JSON patch, median HTTP completion | Generated tokens |
| --- | ---: | ---: | ---: |
| Button background | 1.416 s | 0.513 s | 749 → 27 |
| Counter increment | 1.366 s | 0.549 s | 748 → 29 |

The four jointly passing JSON pairs had median paired ratios of 7.96× for model generation and 2.76× through local HTTP completion. Browser paint and preparation were measured separately from generation. These results compare different edit response formats on one runner.

With Studio running locally, benchmark the same edit formats using the stock target. Select `original` to identify the Standard runner in the report.

```bash
python -m demo_chat.benchmark_patch_edit --mode original --patch-format json \
  --repeats 2 --out /tmp/lightreason-patch-edit.json
```

## Optional draft and adapter setup

The draft model and auxiliary tokenizer are public upstream. Fetch the files the runtime uses with these commands.

```bash
hf download amd/PARD2-Qwen3-8B config.json model.safetensors \
  --revision 67a1516c8f6fc145cda99916799a0cbb3a4af135 --local-dir models/pard2
hf download Qwen/Qwen3-0.6B tokenizer.json \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir models/aux-tokenizer
cp models/aux-tokenizer/tokenizer.json models/pard2/aux-tokenizer.json
```

Turbo and the coordination benchmark require the lab's trained adapter with SHA-256 `8e5b53abec75a33825afca5c08a151d2cefd0a8a567bfff2ebc507bdd4ecc6b6`. Provide it as `models/adapter.pt` if you have an authorized copy. The runner checks the adapter, target, and auxiliary identities. The public draft by itself is insufficient for this configuration.

```bash
python -m demo_chat.server --model-path model-3b \
  --draft-path models/pard2 --checkpoint-path models/adapter.pt
```
