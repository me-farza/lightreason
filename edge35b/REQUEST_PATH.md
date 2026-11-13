# From prompt to streamed tokens on the RTX 4090

The comparison uses the same 151 input IDs, the Qwen3.6-35B-A3B BF16 checkpoint, greedy decoding, and a 256-ID output cap. The two Studio exports share the same user and system prompts. Scheduled and Packed Replay ran sequentially in one export. FreeToken ran live in the later export, alongside a saved Packed Replay result. Decode and first-token clocks are runner clocks. The FreeToken runner clock is its local HTTP client and includes its first-text handling. Browser time and comparison queueing use separate clocks.

| Runner | First token or text | Decode after first token | Complete runner request | Decode rate |
| --- | ---: | ---: | ---: | ---: |
| Scheduled reference | 4.73 s | 32.75 s | 37.49 s | 7.79 tok/s |
| Lightreason Packed Replay | 4.38 s | 14.85 s | 19.23 s | 17.18 tok/s |
| FreeToken, seven proposals | 1.13 s | 6.71 s | 7.85 s | 38.01 tok/s |

The live FreeToken browser saw first text at 1.69 s and completion at 8.28 s. Packed Replay's first browser text in the original comparison was at 42.49 s because Scheduled ran first. Its own runner needed 4.38 s from the start of prefill. All three reached the cap. FreeToken first differed from both saved sequences at ID 31. [Comparison timing boundaries](NATIVE50.md#the-256-token-studio-comparison-and-decode-cache-policy)

| Stage | Scheduled reference | Packed Replay | FreeToken |
| --- | --- | --- | --- |
| Prompt enters | Studio validates the request, applies Qwen's chat template, tokenizes and queues it. | Same Studio entry and tokenizer. | Studio checks a raw prompt against Qwen's tokenized template and calls FreeToken's local API. FreeToken tokenizes and queues the same IDs. |
| Prefill and first token | Runs all prompt positions through the target, stages the selected expert weights per layer, and takes the maximum logit at the final position. | Uses the routed prefill path and seeds the GPU expert cache while building the initial model state. | Stages the union of routed expert rows for this 151-ID prompt and uses resident GPU copies when available. Keeps intermediate target features for its draft model. |
| Read the existing sequence state | Reads full-attention KV pages and Gated DeltaNet's recurrent state and convolution history. | Replays the single-token Gated DeltaNet path through CUDA graphs with fixed buffers. | Uses temporary slots for speculative positions and replays a wider target graph. |
| Route and compute experts | Reads eight route IDs per layer on the host and runs those experts in the original PyTorch order, fetching weights on a cache miss. | Copies route IDs and activations into pinned host buffers and waits. Runs grouped hot experts on the GPU and cold experts on CPU workers, with repeat-based admission to its GPU cache. Replays the expert and combine operations. | Resolves the cache split on the GPU across eight verifier positions. Executes resident and newly fetched experts on the GPU while native workers compute the CPU assignments. |
| Decide and commit | Takes the target argmax and advances one position. | Same one-position loop, with different expert arithmetic from Scheduled. | DFlash2 proposes seven IDs in parallel. The target scores the pending position and seven draft positions. The scheduler keeps the matching prefix and the target's next token, releases rejected slots, and commits the accepted recurrent and convolution state. |
| Deliver text | Reports output IDs and streams updates to Studio. | Same output path. | Copies the accepted block's IDs to the host, decodes it to text and passes incremental events through the local HTTP stream and Studio to the browser. |

The target has 40 routed layers. The 30 Gated DeltaNet layers hold recurrent and convolution state, while the 10 full-attention layers keep KV state. The model also applies shared experts, residuals, and normalization before the final projection to vocabulary logits. An expert uses three matrices, so an expert's weights here occupy 6 MiB. CUDA graph replay reduces repeated launch and allocation work. The original model's mathematical operations still run. [Scheduled loop](benchmark.py) · [Packed expert dispatch](packed_hybrid.py) · [Packed graph replay](packed_native.py) · [DFlash paper](https://arxiv.org/abs/2602.06036)

## New trace of the FreeToken verifier

The isolated, instrumented run used the same seven-proposal runtime and greedy BF16 target as Studio. The streamed weather request committed 255 output IDs across 53 passes after the first token. CUDA events bracketed the engine's stream, including idle gaps caused by host dependencies. The target span includes its projection to vocabulary logits, expert copies, CPU expert work and synchronization. The publish span includes work through the scheduler handing off the reply. Browser rendering and network transit are outside these spans.

| Span | Streamed weather mean per pass | Streamed arithmetic mean per pass |
| --- | ---: | ---: |
| Scheduler before draft | 0.13 ms | 0.13 ms |
| Draft and stage proposals | 2.48 ms | 2.49 ms |
| Exposed metadata after draft | 0.02 ms | 0.02 ms |
| Original BF16 target verifier | 120.23 ms | 138.74 ms |
| Greedy selection, prefix comparison and ID readback | 0.03 ms | 0.03 ms |
| Post-verify context, state commit and scheduler publication | 0.98 ms | 0.98 ms |
| Sum of traced spans | **123.87 ms** | **142.38 ms** |
| Decode wall time divided by passes | 123.91 ms | 142.43 ms |

Target verification occupied about 97% of the measured pass. With weather acceptance fixed, removing every measured span outside the verifier would raise the calculated rate from 38.8 to about 40.0 tok/s. Keeping the surrounding work at its observed cost leaves roughly 92.6 ms for target verification at 50 tok/s. It would need to lose 27.6 ms per pass. The first five weather target spans consumed 1.01 s combined. The first was 307.6 ms and the later 48 averaged 111.8 ms. The eight-position graph runs the same shape in every pass, so extra work on a cold start and changes in routed experts deserve separate investigation.

In that streamed weather request, all 2,120 layer calls were counted before the periodic stats reset. Each layer call averaged 64 route assignments, 35.89 distinct active experts, 5.89 GPU fetches, 5.20 distinct CPU experts and **6.63 CPU route assignments**. Grouping shared weights would remove **1.43 redundant logical weight reads per layer**, about 22% of CPU assignments and 0.36 GB of logical reads per pass. Some repeated reads may already hit a CPU cache. A separate memory bandwidth trace is needed to count physical DRAM traffic. The periodic stats reset occurred during the other measured requests, so their expert counters cover shorter windows and their timing rows still cover every pass.

| Request | Mode | Output | Mean committed IDs per pass | Decode tok/s | HTTP request |
| --- | --- | --- | ---: | ---: | ---: |
| Weather | Streaming | 256, cap | 4.81 | 38.83 | 7.93 s |
| Weather | Non-streaming | 256, cap | 5.00 | 40.36 | 7.44 s |
| Arithmetic | Streaming | 341, EOS | 5.40 | 37.89 | 9.39 s |
| Arithmetic | Non-streaming | 375, EOS | 6.03 | 41.57 | 9.48 s |

These were sequential diagnostic runs. The modes had different preceding requests, expert-cache histories and output IDs. Those differences confound an estimate of streaming overhead. Both arithmetic answers gave 323. The streamed arithmetic case matched the saved Packed IDs through EOS. Both weather cases reached the cap and differed from the saved reference. Setup took about 158 s outside the request clocks. Peak temperatures were 46 °C GPU and 63.25 °C CPU. [Stage events and watchdog](results/native50/path-trace-stages.json) · [HTTP timings, output IDs and exact-match fields](results/native50/path-trace-http.json)

## Which work can go away

**Prefill.** The earlier selected-row ablation reduced the staging list from 10,240 to 5,577 rows and moved first streamed text from 2.38 to 1.24 s with matching outputs in all 16 control pairs. If an incoming request shares a verified token prefix, restoring its KV, recurrent, convolution, and draft context can avoid recomputing those positions. A prior radix experiment reused 64 IDs of a shared system prefix and reduced first text from 2.30 to 1.09 s with that experiment's routed-prefill setup. Those two gains overlap. Current Studio uses the naive input cache. [Selected prefill](NATIVE50.md#uncached-prefill-and-cold-experts) · [Radix experiment](results/native50/train-routed-prefill.json)

**Target decoding.** FreeToken amortizes a target pass across several output tokens. It committed 255 IDs in 53 passes after the initial token, an average of 4.81 IDs per pass. At 126.60 ms per pass, 50 tok/s would require 6.33 IDs per pass. At its observed acceptance, 50 tok/s requires 96.23 ms per pass. The original token-by-token execution has a true dependency on the preceding token, so speculation uses guesses and verifies them against Qwen. Greedy verification equals a sequential walk when each candidate prefix gives the same target argmax. The earlier identical-prefix investigation measured BF16 projection differences between batched and serial shapes, so the strict token and state check remains separate. [Identical-prefix audit](NATIVE50.md#token-34-and-speculation-timing)

The new weather trace had nine passes that committed only one ID and fourteen that committed all eight. Clipping those recorded decisions to a six-position verifier would retain an average 4.17 IDs per old pass. At that rate the six-position cycle needs 83.4 ms to reach 50 tok/s. This replay calculation holds old pass boundaries fixed. A real shorter verifier changes later prefixes, arithmetic and expert demand. An adaptive width could keep long passes where the draft looks reliable and use smaller captured graphs where it does not. The earlier four- and fifteen-proposal measurements illustrate why width should be scored by committed IDs per millisecond.

**Cold expert weights.** The fastest path can group verifier positions by expert. For one expert, the inputs for several positions can use the same loaded gate, up and down weights. Each position still needs its own dot products, SiLU, and router-weighted result. FreeToken already deduplicates GPU fetches, but its CPU BF16 executor schedules separate dots per route. An AVX2 microkernel can load and widen the same BF16 vector once for two routed positions and use independent accumulators with the original reduction order. Whether it saves DRAM reads depends on how often CPU routes share experts and how much is already retained by the CPU caches.

**Bytes per expert.** The transport probe reconstructed 100,663,296 original BF16 values bit for bit. On three sampled layers it used roughly 25% fewer bytes and took 1.51 ms to gather and reconstruct eight experts, versus 2.31 ms for raw mapped-host reads. The fourth layer needs a raw fallback with that codec. The earlier weather cache counters showed about 2.77 GB of cold expert weights per pass under unique-expert accounting. Packing only the GPU-bound portion would remove about 0.37 GB of traffic on that configuration, roughly 12 ms at the measured shared bandwidth before reconstruction and scheduling effects. A compact host bank with CPU tile unpack and GPU reconstruction would attack both sides. [Transport probe](NATIVE50.md#the-256-token-studio-comparison-and-decode-cache-policy) · [ZipServ](https://arxiv.org/abs/2603.17435)

**Scheduling and publication.** A specialized single-request path can generate position and page metadata in persistent GPU buffers, combine greedy selection and prefix matching, and copy one compact result to the host. FreeToken already uses CUDA graph replay, pinned buffers, asynchronous token readback, and a direct greedy argmax. The earlier vector-only routing experiment matched every weather ID and saved 0.12 ms per pass on the median. Its result puts a scale on that specific optimization. [Routing ablation](NATIVE50.md#the-256-token-studio-comparison-and-decode-cache-policy)

## Reproduce the trace

Apply [the diagnostic patch](freetoken_path_trace.patch) from the `python` directory of a fresh copy of the selected-prefill and token-audit FreeToken checkout. Use `FT_SOURCE` for that directory and the checkpoint, proposer, Python environment and 256-ID suite from the [decode policy reproduction](NATIVE50.md#the-256-token-studio-comparison-and-decode-cache-policy). Run on an idle GPU with the existing watchdog.

```sh
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_PATH_TRACE=1 LIGHTREASON_BENCH_TOKEN_IDS=1
export LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80 LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
taskset -c 2-3 "$PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 900 --out /tmp/path-trace-watch.json \
  --monitor-command taskset -c 8-15 python3 -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math --repeats 1 \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
  --server-log /tmp/path-trace-server.log --out /tmp/path-trace-http.json
```

The server log writes one `LIGHTREASON_PATH_TRACE` record per request, including the warmup. Markers enqueue without a new per-pass host wait. A final event wait and counter read occur after reply publication. The trace adds CUDA events and one CPU route-count kernel per routed layer, so compare its stage proportions with normal request timings.
