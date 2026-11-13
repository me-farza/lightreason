# How Lightreason got faster

These runs use one stock RTX 4090 and one request at a time. Each experiment keeps its original Qwen target weights in BF16. Both chart panels show **decode tokens per second after the first token**, excluding model load, graph preparation and prompt processing. Each plotted multiple divides its decode rate by the initial runner’s rate on the same workload and output budget. The axes use different ranges. A filled point matches the initial runner on the whole timed sequence and stop condition. The tables report paired request latency separately. [Upstream credits and experiment history](experiments.md) · [Per-task coding results](coding-benchmarks.md)

## Qwen2.5-3B

The 3.09B dense model fits on the card. The starting Hugging Face runner sends a new series of GPU operations for every output token. Lightreason Exact captures and replays the recurring operations, stores the attention cache in preallocated capacity buckets, combines compatible BF16 projections, and skips vocabulary rows that a conservative bound can rule out. Surviving vocabulary candidates still use the original BF16 output weights. On the fixed suite, decode rose from about **63 to 140 tokens/s**. The median of paired prompt-plus-generation speedups was **2.19×**. [Target-only timings and validation](data/qwen25-3b/target-only.json) · [Target-only implementation](../exact3b_runner.py)

The later runner adds a smaller draft model. A compiled PARD2 proposer offers branches, and Qwen verifies a packed tree in one forward pass with ancestor-only attention and the corresponding positions. The accepted prefix supplies both the output and the persistent attention-cache update. Public-length CUDA graph preparation happens before the timed calls. The later GPU-coordinated version moves ranking, acceptance, and cache compaction into preallocated device buffers, with a small host receipt each cycle. This saves dispatch and synchronization without changing the preceding tree verifier's output on the held-out regression. [Technical walkthrough and adapter requirements](benchmarks.md) · [GPU coordination implementation](../gpu_tree_control3b_v1.py)

| Milestone | Decode tok/s vs. Hugging Face | Decode gain | Median paired warmed request speedup | Output evidence |
| --- | ---: | ---: | ---: | --- |
| Original Hugging Face, six prompts | 63.0 | 1.00× | 1.00× | Reference |
| LR · CUDA graphs + compact KV, six prompts | 63.0 → 139.6 | 2.22× | 2.19× | 30/30 timing pairs and 38/38 separate validation cases matched |
| LR · Same target-only runner, three coding tasks | 59.4 → 138.6 | 2.33× | 2.32× | 9/9 matched through native EOS |
| LR · Compiled PARD2 draft, six prompts | 63.3 → 308.0 | 4.87× | 4.76× | 15/30 capped pairs matched |
| LR · PARD2 tree verifier, weather dashboard | 58.3 → 495.1 | 8.49× | 8.38× streamed | 3/3 matched all 492 IDs and native EOS |
| LR · GPU tree coordination, same weather dashboard | 58.3 → 525.2 | 9.00× | 8.88× streamed | 3/3 matched all 492 IDs and native EOS |
| LR · PARD2 on Fibonacci, best exact code slice | 63.6 → 724.4 | 11.40× | 10.79× | 5/5 matched through the 128-token cap |

The capped Fibonacci slice comes from the earlier compiled-draft suite. A separate new benchmark uses only the public target and measures three complete, matching Python, JavaScript, and SQL responses at **2.32–2.34×** decode speed. [Coding benchmarks and reproduction](coding-benchmarks.md)

The weather follow-up moved median warmed generation time from **8.435 s** in Hugging Face to **1.006 s** with the prior tree verifier and **0.950 s** with GPU coordination. The same trials measured **58.3 → 495.1 → 525.2 decode tok/s** using the time from first driver emission to the end of the generation call, including the stream consumer drain. That is one complete workload, repeated three times per streaming mode. The broader six-prompt native-EOS follow-up showed **9.39×** median streamed latency versus Hugging Face with **12/36** exact matches across both modes. Against the prior tree runner, GPU coordination gave **1.062×** and **36/36** exact pairs. [Compiled draft measurements](data/qwen25-3b/compiled-draft.json) · [Weather and held-out paired timings](data/qwen25-3b/gpu-coordination.json)

The earlier compiled-draft variant scored 100/128 on GSM8K with both runners, with one answer improving and one regressing. Strict diagnostic format fell from 26/32 to 25/32. On a separate complete Fibonacci answer the generated code passed the same 132 input checks, but the responses differed in text and length. The **10.79× exact result** in the chart refers to the capped coding slice above. [Quality and complete-answer evidence](data/qwen25-3b/compiled-draft.json)

## Qwen3.6-35B

Qwen3.6-35B-A3B is a mixture-of-experts model. The text checkpoint contains **34.66B BF16 parameters (64.56 GiB)**, including **60 GiB** of routed expert weights. Each token selects eight of 256 experts per layer. The card has 24 GiB of VRAM. The original Lightreason implementation established the eager baseline by moving each full layer's expert bank from host memory to the card. Selecting just the routed experts and keeping recent rows on the card was the first large improvement. [Initial implementation and weight audit](../edge35b/README.md)

The earliest whole-layer offload test took **40.04 s** for its six-token workload. Routed caching took **4.60 s**, with **4/4** exact paired outputs and **8.71×** median paired request speedup. This early reference did far more transfer work per token. The more useful long-form reference is **Scheduled**, which fetches only the selected misses and reads routing IDs together. In a separate complete 341-ID arithmetic answer, Scheduled measured **7.52 tok/s** and **47.29 s**. Packed CPU/GPU execution reached **12.85 tok/s** and **28.39 s** with the same answer. A four-position DFlash experiment reached **18.29 tok/s** and **20.54 s** on that answer, with a mismatch on a separate logic prompt. [Whole-layer comparison](../edge35b/results/baseline.json) · [Scheduled answer](../edge35b/results/short-answer.json) · [Packed and early DFlash](../edge35b/results/target50/hybrid-verify.json)

| Milestone | Decode tok/s | Gain vs. Scheduled | What changed |
| --- | ---: | ---: | --- |
| LR · Original Scheduled, weather | 7.79 | 1.00× | Fetch selected rows and reuse the GPU cache |
| LR · Packed Replay, weather | 17.18 | 2.21× | Split expert work across CPU and GPU, replay hot CUDA work and recurrent mixers |
| Stock FreeToken on Python | 32.88 | 4.15× | One upstream BF16 run on the 768-ID Python fixture |
| LR · Ada gather + DFlash2, weather | 39.22 | 5.04× | Check seven guesses per pass with FreeToken, move and unpack fewer expert bytes |
| LR · Shared host packing, weather | 45.80 | 5.88× | Feed the GPU gather and AVX2 workers from the same lossless host representation on 34 layers |
| LR · Packed GPU replicas, weather | 46.80 | 6.01× | Keep 187 additional packed experts in 880 MiB and reconstruct directly into BF16 cache slots |
| LR · Packed GPU replicas on Python | **51.08** | **6.44×** | Same runtime on the 768-ID first-duplicate prompt, whose Scheduled reference is 7.93 tok/s |

The weather rows use the same dashboard input and 256-ID cap across archived sessions. The earliest and latest weather outputs first differ at token **31**. The new Python Scheduled reference matches the archived packed run’s 44 input IDs and 768-token cap. Those outputs first differ at token **116**. The same-input multiples include changes to the execution stack and apply to decode rates. [Archived prompts and weather reference](../edge35b/benchmark-suite.json) · [New Scheduled Python records](data/qwen35b/scheduled-coding.json) · [First-token and request timings](coding-benchmarks.md#35b-html-and-python)

Before the Ada packing work, stock FreeToken reached a median **31.57 tok/s** across the four-prompt BF16 suite and both HTTP modes. Its streaming and non-streaming arithmetic outputs differed in length from the earlier Packed reference. Seven-proposal DFlash2 reached **41.41 tok/s** on an earlier streamed weather run. Those measurements helped identify the CPU/GPU overlap and speculative verifier as starting points. [Stock FreeToken comparison](../edge35b/results/native50/freetoken-bf16.json) · [Early seven-proposal DFlash2 runs](../edge35b/results/native50/dflash2-232049.json)

The strongest controlled comparison for the last change is **50.39 → 51.08 tok/s on coding**, with **20/20** identical outputs, stop reasons, and commit sequences across the four-prompt packed-replica trial. The two ordinary coding runs took a median **16.05 s** through the cap and **1.02 s** to first streamed text. The equal-capacity arena held 146 raw experts or 187 packed experts. The production gather decoded and checked **588,251,136 packed BF16 weight values** during setup. Model, bank and graph preparation took **283.86 s**, recorded separately. [Packed-replica evidence](../edge35b/results/native50/gpu-resident-qwen.json.gz) · [Packed-replica implementation and profiles](../edge35b/THROUGHPUT100.md#lossless-gpu-resident-expert-weights)

### Why the retained runner stops around 50 tok/s

FreeToken supplies the graph-compatible CPU/GPU pipeline and DFlash2 integration. It keeps hot experts on the GPU and sends overflow to CPU workers, overlapping computation with PCIe reads where possible. DFlash2 proposes seven tokens, then Qwen verifies the pending token plus those guesses. Qwen commits only the verified prefix and its own next-token decision, and updates attention, recurrent, and convolution state for that prefix. Lightreason adds selected-row prefill, CPU weight reuse across routes, lossless host packing, and a packed GPU replica tier. Lossless packing stores compact exponent codes and falls back to raw tiles where needed. The original BF16 bits are restored for the same expert operations. [FreeToken](https://github.com/FlashML-org/FreeToken) · [DFlash2](https://inco.ai/blog/dflash2/) · [Ada gather](../edge35b/bf16_gather_ada.cu)

The latest DFlash2 experiment jointly tuned prefix selection and real four-, six-, and eight-position verifier graphs. Adaptive width reached **54.23 tok/s** streamed on coding, and every candidate changed output on its 20 comparisons with the frozen packed runner. With known continuation tokens, the target committed eight matching IDs per pass for 30 arithmetic passes at **67.34 tok/s**. That diagnostic used saved proposals and stopped being exact after output ID 246. Transfers alone took **86 ms** per pass, already above the **80 ms** total budget for 100 tok/s at eight commits. This makes host weight movement a priority for the next optimization. [Latest ablations and reproduction](../edge35b/THROUGHPUT100.md#joint-dflash2-and-verifier-tuning) · [Latest raw IDs and timings](../edge35b/results/native50/dflash-prefix-qwen.json.gz)

Unchanged weights and target verification establish which model made the decision. Exact output claims require comparing the actual IDs and stop condition against a named reference. Different matrix shapes or CPU/GPU expert assignments can change floating-point accumulation and shift the target's top token. [Identical-prefix diagnosis](../edge35b/NATIVE50.md#token-34-and-speculation-timing)

## Reproduce or port it

- Start with the public [3B Studio quickstart](../README.md#run-on-a-gpu-machine) or the [35B benchmark setup](qwen35b.md). Use the [experiment history](experiments.md) to see which candidate to compare with which reference.
- Follow the [hardware porting guide](porting.md) to rebuild the baseline and profile on your device. The [agent skill](../.agents/skills/lightreason-hardware/SKILL.md) uses the same protocol.
- Regenerate the README image with Matplotlib and CairoSVG, which renders the repository logo. The [plotted data](data/speedup-history.json) identifies each benchmark source and its output check.

```sh
python -m pip install matplotlib cairosvg
python tools/render_speedup.py
```

Large measurement files are gzip-compressed. Use `gzip -dc` to pass an uncompressed copy to an older analysis command that expects plain JSON. Archived commands retain the flags used in the lab, with workstation paths and the device identifier replaced by placeholders. The [35B guide](qwen35b.md) gives the portable checkout, build, and launch sequence.
