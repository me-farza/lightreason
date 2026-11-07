# Qwen2.5-3B benchmark notes

## GPU coordination experiment

The lab test used the same pinned BF16 target, an RTX 4090 at stock settings, native EOS, and the same greedy token selection. The GPU candidate moved tree construction, acceptance, and cache commitment onto the GPU. The target chose accepted tokens. Preparation was outside the repeated warm generation timings.

| Paired comparison | Non-streaming | Streaming |
| --- | ---: | ---: |
| GPU coordination vs. frozen optimized runner | 1.062× | 1.062× |
| GPU coordination vs. original Hugging Face | 9.33× | 9.39× |

The candidate matched the frozen optimized runner on 36/36 held-out comparisons and 160/160 regression cases. It matched original Hugging Face token-for-token on 12/36 held-out comparisons. Two prompts had exact Hugging Face output and exceeded 10× in every repeat. The suite medians versus Hugging Face include differences in output; these figures do not describe the default Studio Exact runner. The separate experimental attention variant showed two new answer regressions and is outside this release.

To reproduce the retained runner comparison, obtain the optional draft and adapter below. The input-only [suite](../optimization_20261106/suite.json) fixes six development and six held-out prompts and their output budgets.

```bash
python benchmark_cost_aware_tree3b_v1.py \
  --controls-manifest optimization_20261106/controls/runtime-manifest.json \
  --model-path model-3b --draft-path models/pard2 \
  --adaptation-checkpoint-path models/adapter.pt \
  --candidate-factory gpu_tree_control3b_v1:GPUTreeControl3BRunner \
  --split heldout --node-budget 64 --repeats 3 --stream-modes both \
  --out /tmp/lightreason-heldout.json
```

## Patch-edit experiment

Two alternating repeats per case used one generated-page fixture on the lab's Turbo runner. The CSS and counter changes passed both full-page and JSON patch checks. The JSON heading patch was ambiguous and rejected. The full-page heading edit changed unrelated behavior. The marker format passed heading and CSS but failed the counter edit. A different JavaScript edit also showed that a unique textual match alone cannot establish behavioral correctness.

| Passing case | Full page, median HTTP completion | JSON patch, median HTTP completion | Generated tokens |
| --- | ---: | ---: | ---: |
| Button background | 1.416 s | 0.513 s | 749 → 27 |
| Counter increment | 1.366 s | 0.549 s | 748 → 29 |

The four jointly passing JSON pairs had median paired ratios of 7.96× for model generation and 2.76× through local HTTP completion. Browser paint and preparation were measured separately from generation. These results compare different edit response formats on one runner.

With Studio running locally, this also works on the stock target without the optional draft. Choose a runner explicitly to identify the configuration in the output:

```bash
python -m demo_chat.benchmark_patch_edit --mode original --patch-format json \
  --repeats 2 --out /tmp/lightreason-patch-edit.json
```

## Optional draft and adapter setup

The draft model and auxiliary tokenizer are public upstream. These commands fetch only the files the runtime uses:

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
