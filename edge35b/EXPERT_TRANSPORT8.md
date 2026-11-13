# Eight-position BF16 transport and expert-cache trials

The control is the fixed eight-position verifier from [the width retest](WIDTH_AUTOTUNE.md), with one-pair CPU fusion and seven DFlash2 proposals. Both arms run the original Qwen3.6-35B-A3B BF16 target (revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`) and DFlash2 (revision `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`). FreeToken 0.2.1, greedy decoding, the thinking template, prompt IDs, caps, stop settings, and the 15 GiB primary GPU expert cache stay fixed. Each request starts with uncached input and a reset primary expert cache. Only the target commits output tokens.

The RTX 4090 was restored with a one-time boot into `7.0.0-31-generic` and the installed NVIDIA 580.173.02 module. These measurements use vicena-0's Ryzen 9 5950X, 121 GiB usable host RAM, and PCIe Gen4 ×16 under load. The watchdog enforces the existing 70 °C GPU and 75 °C CPU stops at the stock 450 W power limit. The transfer run peaked at 40 °C GPU and 72.75 °C CPU. [Hardware and input manifest](results/native50/transport8-hardware.txt) · [Watchdog summary](results/native50/transport8-watch-summary.json)

## Raw versus packed

The packed arm reconstructs original BF16 bits directly into the same GPU cache slots. The stock planner, CPU overflow, GPU expert math, state commitment and proposer remain the same. Twelve layers use the packed source or per-row raw fallback. The CPU row audit covers their actual checkpoint bytes. A graph replay fixture also checks the CUDA writes at production row sizes, including mixed raw/packed rows and exception repair. [Codec audit](PASS_DIAGNOSTICS.md#lossless-expert-fetch-trial) · [CUDA fixture watchdog](results/native50/transport8-cuda-check.json)

All **21 pairs** matched the frozen eight-position token IDs, stop reasons and commits per pass: 16 ordinary pairs and 5 profiled pairs. The profiled runs also matched the ordered expert fetch IDs for every layer and pass. This gate refers to the frozen eight-position runner. It makes no additional parity claim against the separately archived Packed Replay or Scheduled runners.

| Prompt | Mode | Raw → packed decode tok/s | Raw → packed request | Raw → packed first text |
| --- | --- | ---: | ---: | ---: |
| Weather | Stream | 38.08 → 37.06 | 8.13 → 8.31 s | 1.428 → 1.429 s |
| Weather | Non-stream | 38.04 → 34.55 | 8.13 → 8.84 s | — |
| Arithmetic | Stream | 41.56 → 40.32 | 8.89 → 9.14 s | 0.637 → 0.634 s |
| Arithmetic | Non-stream | 38.98 → 40.11 | 9.48 → 9.19 s | — |
| Coding | Stream | 42.10 → 40.97 | 19.24 → 19.74 s | 1.020 → 1.011 s |
| Coding | Non-stream | 42.21 → 41.02 | 19.18 → 19.71 s | — |
| Chat | Stream | 36.83 → 35.80 | 23.34 → 23.99 s | 0.916 → 0.916 s |
| Chat | Non-stream | 35.96 → 35.73 | 23.90 → 24.03 s | — |

Cells are medians of two alternating ordinary pairs per prompt and mode. All repeats, including slow samples, are retained. Weather (256 IDs) and coding (768 IDs) reached their original caps. Arithmetic (344 IDs) and chat (827 IDs) stopped naturally. Decode rates exclude the initial prefill token and use the server's decode interval, which includes drafting, verification and state updates. Request time includes prefill, decode and delivery. Preparation was 196.6 s before the separate short graph warmups and includes approximately 38 s building and auditing the pinned 13.04 GiB packed banks. [Raw paired measurements and exact IDs](results/native50/transport8.json.gz)

| Profiled weather component | Raw | Packed |
| --- | ---: | ---: |
| Requested PCIe expert payload per pass | 1.470 GB | 1.345 GB |
| Gather, including unpack and repair for packed | 90.81 ms | 95.92 ms |
| Exposed CPU wait | 11.31 ms | 9.04 ms |
| CPU expert service, mostly concurrent | 102.37 ms | 102.14 ms |
| GPU expert math | 9.54 ms | 9.56 ms |
| Committed IDs per pass | 4.636 | 4.636 |

Packing lowered payload by **8.47%**, but the gather stage took **5.11 ms longer**. Effective weight payload per gather second fell from 16.18 to 14.02 GB/s. Gather time in the 12 changed layers rose from 32.40 to 37.47 ms, while the other layers stayed at 58.4 ms. The data separates the slowdown from routing and target arithmetic; it does not isolate the GPU unpack instructions from the memory access pattern and repair kernel. The reduction in exposed CPU wait reflects extra overlap with the longer gather. Summing CPU service and gather would double count overlapping work. Raw remains the control for the cache trial.

At 4.636 committed IDs per pass, 50 tok/s needs **92.73 ms per pass**, about 28.7 ms below the current ordinary streamed control. The measured raw profile had 233.6 GPU fetches and 206.4 unique CPU overflow experts per pass, about 2.77 GB of combined original expert payload under unique-row accounting. Faster GPU copies alone would expose more CPU time unless that service or contention also falls. This motivates changes to both host data paths, or more committed target tokens for the same amount of work.

## Cache tuning method

The [cache planner](tune_hot_expert_cache.py) ranks layer/expert pairs by refetches in the development weather trace. It projects 1.26%, 2.51%, and 4.56% fewer PCIe bytes for 32, 64, and 128 rows respectively. The frozen 128-row plan's replay coverage on the withheld traces is 1.79% for arithmetic, 1.98% for coding, and 1.90% for chat. These are payload projections, not timing results. [Frozen plan and budgets](results/native50/hot8-plan.json)

The runtime leaves the primary cache and GPU/CPU assignment untouched. A read-only GPU copy of up to 128 experts costs 768 MiB, with a 1 GiB free-memory guard. Every prepared replica is compared with the original BF16 bits. A GPU kernel splits the stock miss list into rows read from pinned host memory and rows copied within the GPU. Both copies use the original vectorized gather and write the original destination slots. The three-budget sweep enables prefixes of the same prepared plan, so it can alternate with the raw runner without reloading the checkpoint. Only the development weather case chooses the enabled budget; the selected configuration is repeated on the separate math, code and chat cases.

A separate [trace replay](results/native50/hot8-replay-bounds.json) shows that a 128-row LRU fed only by stock misses has zero hits on these four traces. A future-aware replacement oracle that can retain or bypass each fetched row tops out at 6.62–8.26% avoided fetches under the same restriction, even before copy costs. This is not a bound on a cache that can also retain rows during prefill, but it rules out simple LRU behind the existing primary cache as the way to 50.

## Cache measurements

The development weather sweep selected **128 rows**. All six tuning pairs matched the frozen token IDs, stop reasons and commits per pass. The median paired decode gains were 0.88% with 32 enabled rows, 1.27% with 64, and 1.70% with 128.

| Prompt | Mode | Raw → hot-source decode tok/s | Raw → hot-source request | Raw → hot-source first text |
| --- | --- | ---: | ---: | ---: |
| Weather | Stream | 38.05 → 38.72 | 8.13 → 8.02 s | 1.429 → 1.430 s |
| Weather | Non-stream | 37.85 → 38.54 | 8.17 → 8.05 s | — |
| Arithmetic | Stream | 41.23 → 41.53 | 8.96 → 8.90 s | 0.636 → 0.634 s |
| Arithmetic | Non-stream | 41.31 → 41.54 | 8.94 → 8.89 s | — |
| Coding | Stream | 42.09 → 42.42 | 19.24 → 19.09 s | 1.017 → 1.013 s |
| Coding | Non-stream | 42.14 → 42.51 | 19.22 → 19.06 s | — |
| Chat | Stream | 36.80 → 36.92 | 23.36 → 23.29 s | 0.916 → 0.916 s |
| Chat | Non-stream | 36.82 → 36.91 | 23.35 → 23.30 s | — |

These are two alternating unprofiled repeats per mode. All 27 cache pairs passed the exact-output and pass-count gate: six tuning, 16 ordinary, and five profiled. The profiled fetch plans matched the raw control. The cache run peaked at 42 °C GPU and 71.75 °C CPU. [Paired timings and tokens](results/native50/hot8.json.gz) · [Watchdog summary](results/native50/hot8-watch-summary.json) The highest repeated median was 42.51 tok/s. Preparation took 157.6 s, separate from inference and graph warmups. Gains on held-out streamed prompts were 0.33–0.79% by median paired ratio; these short repeats cannot establish tighter uncertainty bounds.

In the weather profiles the cache moved 1.403 GB per pass over PCIe, down from 1.470 GB. Gather time fell from 91.19 to 87.96 ms, and exposed CPU wait rose from 11.24 to 11.68 ms. The GPU sources served 10.65 of 233.6 planned fetches per pass on average. The fixed eight-position path now needs about 27 ms more per pass removed on weather to reach 50 tok/s at unchanged acceptance. The cache variant is reproducible and opt-in; it is not the Studio default.

## Gather kernel tuning

The [gather benchmark](benchmark_packed_gather8.py) replays 12 actual fetch lists, with 5–15 experts per layer, using pinned checkpoint rows and exact BF16 cache destinations. It ran 16 alternating repeats per list for the original raw gather and 13 packed launch geometries. Every variant reconstructed all tested weight bits. Layers 2 and 1 chose the geometry; layers 3 and 4 were withheld.

The winner used 4,096 BF16 pairs per tile, 128 blocks per bank and four warps per block. On the withheld shapes it gave **1.12×** the throughput of the original packed geometry but only **0.86×** the throughput of stock raw. This is a copy-only diagnostic with no concurrent CPU experts. It does not justify another full-model test of the same packed format. [Gather timings](results/native50/packed-gather8.json) · [Watchdog summary](results/native50/packed-gather8-watch-summary.json)

## Evolution of BF16 copy scheduling

The tuned 128-row source is the timing control. Its verifier launches a split of the existing fetch list, followed by separate host-to-device and device-to-device copies. The tiled variant assigns a group of CUDA blocks to each requested row and selects the host or GPU source inside one copy kernel. Both BF16 banks are copied as raw 32-bit words into the original cache slots. The verifier width, CPU expert math, cache policy, proposer, and target commitment stay fixed. CUDA graph tests check the full Qwen row sizes, including empty plans, mixed sources, and slots outside the plan.

The [search](gather_evolution.py) starts with two copy geometries and changes one launch dimension of the fastest exact candidate. It scores complete weather requests, including the real proposer and target verification. Each trial alternates the arm order across streaming and non-streaming requests. A faster candidate must reproduce the frozen eight-position control's token IDs, stop reason, and commits per pass. One development winner can advance to repeated alternating measurements on arithmetic, coding, and chat.

On the 4090 the four candidates matched **8/8** pairs. The 256-token weather prompt was the development case. One streamed and one non-streamed pair tested each candidate.

| Copy geometry | Tuned control decode | Candidate decode | Control → candidate request | Paired decode change |
| --- | ---: | ---: | ---: | ---: |
| 4 stripes, 4,096 words | 39.06 tok/s | 38.59 tok/s | 7.953 → 8.033 s | −1.21% |
| 8 stripes, 2,048 words | 38.89 tok/s | 38.62 tok/s | 7.982 → 8.032 s | −0.70% |
| 2 stripes, 2,048 words | 39.02 tok/s | 38.76 tok/s | 7.961 → 8.005 s | −0.67% |
| 2 stripes, 4,096 words | 38.98 tok/s | 38.82 tok/s | 7.969 → 7.995 s | −0.39% |

The best candidate averaged 119.04 ms per verification pass versus 118.57 ms for its control. It committed the same 4.636 tokens per pass. Both streamed runs took about 1.425 s to first text. The search rejected all candidates on development performance and kept the math, coding, and chat set untouched. Setup took 157.58 s before separate short warmup requests. At the measured acceptance, 50 tok/s requires 92.73 ms per pass, about 25.84 ms below this control. The run used the matching NVIDIA 580.173.02 libraries and PCIe Gen4 ×16. Peak sampled temperatures were 44 °C GPU and 69 °C CPU. [Full timings and token IDs](results/native50/gather-evolution.json)

FreeToken's `measure_overlap_bw` probe then measured BF16 CPU experts and PCIe gathers concurrently for one second at Qwen's 2,048 × 512 shape. Two pairs per layout used the current eight workers on cores 8–15 as control. The slower of the two byte transfer times, using 1.299 GB of CPU expert reads and 1.403 GB of PCIe reads from the weather trace, served as the screening score. Eight, ten, and twelve workers spread across both CPU chiplets improved CPU throughput at the cost of PCIe throughput. Their projected overlap rates were 4.4%, 3.1%, and 6.7% lower than control. Six workers on cores 8–13 were 3.2% faster in one pair and 0.1% slower in the other. This synthetic probe did not measure generation and did not advance a layout to a model trial. [Worker layout data](results/native50/gather-worker-layout.json)

Apply [the gather graph patch](freetoken_gather_evolution.patch) with `patch -p1 -d "$FT_SOURCE"` in an isolated copy of the tuned FreeToken source described below. Use the same `FT`, `MODEL`, `DRAFT`, `SUITE`, `CONTROL`, and the frozen 128-row `HOT_PLAN`. Run the CUDA check before serving the full model. Use fresh paths under `RUN`.

```sh
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
FT_PYTHON="${FT%/*}/python"
taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 150 --out "$RUN/copy-watch.json" \
  --monitor-command taskset -c 8-15 env LIGHTREASON_TEST_PACKED_CUDA=1 "$FT_PYTHON" -m unittest \
  edge35b.test_freetoken_packed_transfer.PackedTransportTests.test_raw_tiles_cuda_graph
taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1650 --out "$RUN/watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials "$RUN/width.cfg" --packed-transfer --packed-transfer-width 8 \
  --packed-transfer-reference "$CONTROL" --hot-expert-plan "$HOT_PLAN" --gather-evolution 4 \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1924 \
  --server-log "$RUN/server.log" --out "$RUN/gather.json"
```

## Native Ada CUDA unpack and copy

The [native kernel](bf16_gather_ada.cu) loads eight sign/fraction bytes and four exponent bytes for every 16 bytes of BF16 output. Each CUDA block owns a contiguous portion of the row. It reconstructs the bits with integer instructions, waits for that portion's writes, and repairs its sorted exponent exceptions in the same launch. A descriptor with a negative exponent base reads original bytes, including BF16 rows already resident on the GPU. NVRTC compiled both load policies for `sm_89`, and the driver verified binary architecture 89.

The [copy diagnostic](results/native50/native-cuda-gather.json) used actual checkpoint rows and 15 fetch lists from the fixed-width trace, with 20 alternating graph replays per configuration. All six native configurations matched the original bytes, including exception rows, the predominantly raw layer 0, untouched slots, and empty fetch plans. Layers 2 and 1 selected 8 stripes of 128 threads with the default load policy. Over the six held-out lists from layers 3 and 4, the sum of per-list medians was **12.254 ms** for stock raw, **14.066 ms** for the earlier tuned Triton unpack, and **11.254 ms** for native CUDA. This gives **1.089×** stock raw and **1.250×** tuned Triton throughput at identical outputs. Layer 0 stayed out of the full-model experiment because its raw fallback was slightly slower than stock raw. The copy test has no concurrent CPU expert work.

For the full Qwen comparison, apply [the graph patch](freetoken_native_packed_gather.patch) after the tiled gather patch. The two arms share the same pinned CPU banks and audited 128-row GPU source. Only the candidate reads the existing lossless packed layout for the 12 high-traffic layers. Warmup and packing are outside the request timings. The verifier runs its original greedy target computation and the same proposer. A development gain and exact output gate select whether to run the held-out cases. Profiled pairs verify the original fetch IDs and expose CPU wait and gather time separately from ordinary request timings.

| Prompt | Tuned → native decode, streaming | Tuned → native request, streaming | Median paired decode gain across both HTTP modes |
| --- | ---: | ---: | ---: |
| Weather | 38.84 → 39.22 tok/s | 7.992 → 7.928 s | +1.10% |
| Arithmetic | 41.81 → 41.98 tok/s | 8.842 → 8.810 s | +0.66% |
| Coding | 42.38 → 42.67 tok/s | 19.124 → 18.987 s | +0.57% |
| Chat | 37.03 → 37.38 tok/s | 23.224 → 23.018 s | +0.92% |

Both arms matched the frozen eight-position IDs, commits per pass and stop reasons on all **16 ordinary pairs** and **five profiled pairs**. The profiles also matched the ordered fetch plans. Weather and coding used the existing 256- and 768-token caps. Arithmetic and chat stopped naturally. All four prompts improved in paired median complete request time. The best repeated median was **42.73 tok/s** on non-streaming coding. [Ordinary and profiled results](results/native50/native-cuda-qwen.json)

The two weather profiles moved **1.403 → 1.285 GB** over PCIe per pass, with gather time **87.68 → 86.80 ms**, exposed CPU wait **11.70 → 11.52 ms**, and CPU expert service **100.75 → 99.63 ms**. CPU service overlaps the gather. GPU expert math remained about 9.56 ms, and both arms committed 4.636 tokens per pass. The candidate retains 13.04 GiB of extra pinned host memory, and full-model setup took 199.65 s before the short graph warmups. Peak sampled temperatures were 51 °C GPU and 71.75 °C CPU under the existing limits. The implementation remains an opt-in benchmark.

Use the pinned runtime and watchdog commands above. For the isolated CUDA test, run the following under the watchdog with the same CUDA library environment:

```sh
"$FT_PYTHON" -m edge35b.benchmark_packed_gather8 --model "$MODEL" \
  --trace "$TRANSFER_RUN/transport.json" --out "$RUN/native-gather.json" \
  --native-cuda --repeats 20
```

For full Qwen, replace `--gather-evolution 4` in the preceding full-model command with `--native-packed-gather cuda-8-128-default` and use fresh output, trace, and log paths.

## CPU expert reads with the native CUDA gather

The paired CPU kernel already shares each weight load across two routes to the same expert. The CPU read search changes the bytes read for those dots and for single routes. It starts with prefetch distance and projection choices, then mutates the fastest exact variants by changing the covered projections, AVX2 unpack width, or packed prefetch distance. Four weather-derived fixtures select a policy. Three fixtures from other layers and prompts check that policy. Each fixture uses 40 real experts, eight pinned workers, and 20 timed pairs with alternating order. The synthetic inputs reproduce observed route and expert counts. CPU outputs are compared as BF16 bit patterns on every pair. [CPU read search](benchmark_cpu_reads.py)

The codec stores the original low byte and a four-bit index for the high byte within each 256-value tile. The index stores the sign and an offset in the tile's eight-value high-byte window. The preparation audit reconstructs each encoded weight bit for bit. Tiles outside that range read from the original banks. AVX2 restores the BF16 bits before widening and preserves the existing FP32 FMA and reduction order. The setup ranks layers using unique CPU expert demand on the weather profile and prepares both projections for the top 20 layers. This covers 56.5% of distinct CPU expert reads in the development profile before raw fallbacks. [CPU codec](cpu_bf16_reads.h)

The search tested eight policies. Packing both projections with 32-value AVX2 unpacking and prefetching was **1.160×** on development and **1.167×** on held-out fixtures. All tested CPU outputs matched bit for bit. Preparing the actual runtime audited **59,544,012 of 62,914,560** tiles against the original BF16 banks. The other tiles use raw reads. Packing took 36.49 s and allocated 22.18 GiB of additional host RAM. The entire service took 235.65 s to prepare before the separate graph warmups. [CPU fixture timings](results/native50/cpu-read-evolution.json)

The ordinary pairs below used the same native CUDA gather, 128 hot replicas, eight verifier positions, seven DFlash2 proposals, and original BF16 target in both arms. Each cell is a median of two alternating pairs. Decode excludes the initial prefill token and includes drafting, verification, transfers and state commitment. Request time includes prefill and delivery.

| Prompt | Mode | Native CUDA → combined decode tok/s | Native CUDA → combined request | Native CUDA → combined first text |
| --- | --- | ---: | ---: | ---: |
| Weather | Stream | 39.34 → 40.73 | 7.907 → 7.688 s | 1.424 → 1.425 s |
| Weather | Non-stream | 39.17 → 41.02 | 7.937 → 7.643 s | — |
| Arithmetic | Stream | 42.19 → 43.71 | 8.768 → 8.485 s | 0.636 → 0.634 s |
| Arithmetic | Non-stream | 42.04 → 44.09 | 8.797 → 8.416 s | — |
| Coding | Stream | 42.71 → 44.79 | 18.981 → 18.140 s | 1.018 → 1.013 s |
| Coding | Non-stream | 42.75 → 44.71 | 18.957 → 18.169 s | — |
| Chat | Stream | 37.23 → 39.05 | 23.102 → 22.072 s | 0.916 → 0.916 s |
| Chat | Non-stream | 37.40 → 39.11 | 23.008 → 22.036 s | — |

The median paired decode gain was **4.67%** across the 16 ordinary comparisons. All **21 pairs** matched the frozen eight-position token IDs, stop reasons, and commits per pass. The five profiled pairs also matched the ordered fetch IDs and payloads. Arithmetic (344 output IDs) and chat (827) stopped naturally. Weather (256) and coding (768) reached their existing caps. The highest repeated median was 44.79 tok/s and the fastest individual run was 45.11 tok/s. The GPU and CPU peaked at 51 °C and 71.75 °C under the existing 70 °C and 75 °C stops. The PCIe link ran at Gen4 ×16 under load. [Paired requests and profiles](results/native50/cpu-native-qwen.json.gz)

| Profiled prompt | CPU expert service | Exposed CPU wait | PCIe gather | Profiled pass |
| --- | ---: | ---: | ---: | ---: |
| Weather | 99.85 → 91.36 ms | 11.97 → 9.26 ms | 86.73 → 84.20 ms | 119.81 → 114.58 ms |
| Arithmetic | 124.96 → 113.23 ms | 15.15 → 10.95 ms | 109.00 → 105.68 ms | 145.04 → 137.53 ms |
| Coding | 102.49 → 93.93 ms | 11.31 → 8.88 ms | 90.79 → 88.41 ms | 123.47 → 118.67 ms |
| Chat | 105.67 → 96.79 ms | 11.89 → 9.47 ms | 93.39 → 90.79 ms | 126.74 → 121.73 ms |

CPU service overlaps the PCIe gather. Weather moved 1.285 GB per pass in both arms. Faster gather with the same PCIe payload is consistent with less contention for host memory. GPU expert math stayed at 9.62 ms. At the measured weather acceptance of 4.636 committed tokens per pass, 50 tok/s requires 92.73 ms per pass. Gather and GPU expert math alone took 93.81 ms in the combined profiles. With acceptance fixed, weather needs 21.11 ms per ordinary streaming pass removed, coding needs 12.22 ms, and chat needs 26.33 ms. Eliminating just the exposed CPU wait at the measured transfer rate would leave a gap on each of those prompts.

Start with the isolated native CUDA source used above and its original vendor `_cpu_moe` shared library. The C++ patch applies after `freetoken_bf16_pairs.patch`. If this staging directory already has the pair extension rebuilt, restore the original vendor library from `liblightreason_cpu_moe_base.so` in the isolated directory before rebuilding. The runtime patch applies after the native gather graph patch. `NATIVE_PROFILE` is the completed `native-cuda-qwen.json` report. Use the same driver libraries and Triton cache as the native CUDA run and a fresh `RUN` directory.

```sh
install -m 0644 edge35b/cpu_bf16_reads.h \
  "$FT_SOURCE/freetoken/kernel/csrc/cpu_moe/lightreason_cpu_bf16_reads.h"
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_cpu_reads.patch
bash edge35b/build_freetoken_bf16_pairs.sh "$FT_PYTHON" "$FT_SOURCE"
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_cpu_reads_runtime.patch

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 300 --out "$RUN/cpu-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -m edge35b.benchmark_cpu_reads \
  --model "$MODEL" --profile "$NATIVE_PROFILE" --repeats 20 --trials 8 --out "$RUN/cpu-reads.json"

CPU_READ_POLICY=$("$FT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' \
  "$RUN/cpu-reads.json")
taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1800 --out "$RUN/qwen-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials "$RUN/width.cfg" --packed-transfer --packed-transfer-width 8 \
  --packed-transfer-reference "$CONTROL" --hot-expert-plan "$HOT_PLAN" \
  --native-packed-gather cuda-8-128-default --cpu-read-policy "$CPU_READ_POLICY" \
  --cpu-read-profile "$NATIVE_PROFILE" --cpu-read-layers 20 \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1924 \
  --server-log "$RUN/server.log" --out "$RUN/cpu-native-qwen.json"
```

### CUDA reads of the same packed host banks

The shared kernel gives one warp each 256-value tile. GPU-resident metadata specifies the exponent window or a raw fallback. The warp reconstructs BF16 and writes directly to the original cache slot. GPU hot replicas still supply original BF16 rows. The same packed host storage feeds AVX2 and the CUDA gather on overlapping layers. [Ada kernel](bf16_gather_ada.cu)

Three shared-kernel geometries copied 18 traced fetch lists on the RTX 4090, with 20 alternating replays per list. The checks covered original BF16 bytes, empty fetch plans, untouched slots, and the raw fallback on layer 0. Two development layers selected four stripes of 256 threads. On nine held-out lists from layers 3, 4, and 17, the sum of medians was **12.41 ms**, down from **16.21 ms** with the current gather for those layers. The copy-only rate was **1.306×** with no concurrent CPU experts. [Exact copy probe](results/native50/shared-cpu-cuda-gather.json)

The full trial keeps CPU packing enabled on the same 20 layers in both arms. Both arms share the same host allocations and the same 128 GPU replicas. The control's 12-layer packed source is also prepared so the two graphs can be measured back to back. The watchdog stopped the first, 28-layer setup when temperature readings went stale during memory preparation. The sampled temperatures reached 46 °C GPU and 62.75 °C CPU. The retry uses the same 20-layer plan for shared CUDA reads and the existing watchdog settings.

| Prompt | Stream decode tok/s | Non-stream decode tok/s | Stream request | Stream first text |
| --- | ---: | ---: | ---: | ---: |
| Weather | 41.10 → 43.26 | 41.21 → 43.29 | 7.631 → 7.321 s | 1.425 → 1.424 s |
| Arithmetic | 44.07 → 46.04 | 44.16 → 46.34 | 8.422 → 8.087 s | 0.637 → 0.635 s |
| Coding | 45.01 → 47.05 | 44.93 → 47.10 | 18.059 → 17.316 s | 1.018 → 1.011 s |
| Chat | 39.29 → 41.20 | 39.32 → 41.21 | 21.942 → 20.967 s | 0.915 → 0.915 s |

The 20-layer shared gather improved median paired decode throughput by **4.92%** over the CPU-packed control with the earlier CUDA gather. All 21 pairs matched the frozen output IDs, stop reasons, and commits. The profiles also matched fetch IDs. Weather's requested PCIe payload dropped from 1.285 to 1.219 GB per pass, and gather time fell from 84.08 to 77.37 ms. Exposed CPU wait was 9.45 → 9.54 ms, and CPU expert service was 91.52 → 88.44 ms. Peak temperatures were 53 °C GPU and 72 °C CPU. Auditing, packing, and pinning the 22.18 GiB shared banks took 54.19 s of the 253.76 s setup. [Exact full-model pairs and timings](results/native50/cpu-shared-qwen.json.gz)

Apply the shared graph patch after the CPU read runtime patch. Run the copy probe against the original raw versus packed transport archive. To benchmark complete requests, repeat the full CPU read command with a fresh `RUN` directory and the two shared-layout options below. [Shared graph patch](freetoken_shared_cpu_gather.patch)

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_shared_cpu_gather.patch
taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 300 --out "$RUN/shared-copy-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -m edge35b.benchmark_packed_gather8 \
  --model "$MODEL" --trace "$TRANSFER_RUN/transport.json" --native-cuda --shared-cpu-layout \
  --repeats 20 --out "$RUN/shared-cuda-gather.json"
# Add to the full CPU read benchmark command.
# --shared-cpu-gather cuda-4-256-cpu-tiles --shared-cpu-layers 20
```

### Expanding the shared layer budget

The next comparison prepares one shared bank for both arms and captures separate graphs for their layer budgets. The control enables the audited 20-layer plan for AVX2 and CUDA. The candidate enables 34 layers, which cover 88.1% of distinct CPU expert reads and 87.0% of PCIe payload on the original weather trace. Both figures are development coverage before raw fallback. The 20-layer plan already covers the earlier CUDA gather's 12 packed layers, so this comparison can omit the separate 13.04 GiB CUDA-only bank. Both arms use the same host allocations, expert routing policy, and GPU replicas.

| Prompt | Stream decode tok/s | Non-stream decode tok/s | Stream request | Stream first text |
| --- | ---: | ---: | ---: | ---: |
| Weather | 42.84 → 45.80 | 43.17 → 45.81 | 7.384 → 7.002 s | 1.435 → 1.424 s |
| Arithmetic | 45.97 → 49.22 | 46.13 → 49.20 | 8.110 → 7.614 s | 0.646 → 0.639 s |
| Coding | 47.03 → 50.07 | 47.12 → 50.04 | 17.341 → 16.333 s | 1.027 → 1.013 s |
| Chat | 41.20 → 43.70 | 41.14 → 43.81 | 20.983 → 19.823 s | 0.932 → 0.920 s |

The median paired decode gain was **6.28%** and all 21 pairs matched the frozen IDs, stops, and commits. Two of four ordinary coding runs exceeded 50 tok/s, with a range of 49.76–50.38. In the weather profiles, the requested PCIe payload dropped from 1.219 to 1.109 GB and gather time fell from 77.29 to 71.37 ms. CPU expert service fell from 89.45 to 82.62 ms, and exposed wait fell from 10.49 to 9.32 ms. Profiled pass time moved from 108.57 to 101.54 ms at 4.636 committed tokens per pass. The preparation audited 102,404,549 packed BF16 tiles and allocated 37.97 GiB of additional host RAM. The packing and pinning took 85.94 s of the 244.67 s setup. Peak sampled temperatures were 47 °C GPU and 71.25 °C CPU. [34-layer exact pairs and timings](results/native50/cpu-shared-coverage34.json.gz)

For this comparison, add the following arguments to the full CPU read benchmark and use fresh output paths. [Shared graph patch](freetoken_shared_cpu_gather.patch)

```sh
--cpu-read-layers 34 --shared-cpu-gather cuda-4-256-cpu-tiles --shared-cpu-layers 34 \
--cpu-read-control-layers 20 --shared-cpu-control-layers 20
```

Both 37-layer attempts stopped during host bank preparation when temperature telemetry went stale for more than 750 ms. The second attempt registered 32 contiguous expert rows at a time. That variant reproduced the original BF16 bytes in a CUDA copy across registration boundaries. Its full run stopped before timed requests, with sampled temperatures of at most 47 °C GPU and 62.5 °C CPU. The 34-layer runtime is the retained result. All three completed full-model comparisons passed 63 exact pairs combined. [Hardware watchdog summaries](results/native50/cpu-read-watch-summary.json)

At unchanged acceptance, the streamed 34-layer weather run needs another 8.50 ms per pass for 50 tok/s, arithmetic needs 1.91 ms, and chat needs 13.53 ms. Chat still spends 77.35 ms per pass on the PCIe gather and 9.84 ms in exposed CPU wait. Even eliminating that exposed wait at the measured transfer rate leaves a gap. [34-layer profiles](results/native50/cpu-shared-coverage34.json.gz)

## Reproduce

Apply [the transfer patch](freetoken_packed_bf16_transfer.patch) to an isolated copy of the fused, instrumented FreeToken source from [the pass audit](PASS_DIAGNOSTICS.md#reproduce). Do not modify the frozen source. Use the same pinned `FT`, `FT_SOURCE`, `MODEL`, `DRAFT`, and 256-token Studio `SUITE` from that audit. Preserve its FreeToken JIT and Triton cache settings.

```sh
export CONTROL="$PWD/edge35b/results/native50/width-pair-retest.json"
export OUT_ROOT=/tmp/lightreason-expert-transport8
bash edge35b/run_expert_transport8.sh

# After the raw-versus-packed trial completes, use its reported output directory.
python3 -m edge35b.tune_hot_expert_cache "$TRANSFER_RUN/transport.json" \
  --base raw --development studio --budget 128 --out /tmp/hot-experts-128.json
HOT_PLAN=/tmp/hot-experts-128.json HOT_TUNE=1 PORT=1922 bash edge35b/run_expert_transport8.sh

python3 -m edge35b.replay_hot_expert_cache "$TRANSFER_RUN/transport.json" \
  /tmp/hot-experts-128.json --out /tmp/cache-replay.json
```

For the isolated gather sweep, run the following from the same isolated FreeToken environment on an idle 4090:

```sh
taskset -c 2-3 "${FT%/*}/python" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 300 --out /tmp/gather8-watch.json \
  --monitor-command taskset -c 8-15 "${FT%/*}/python" -m edge35b.benchmark_packed_gather8 \
  --model "$MODEL" --trace "$TRANSFER_RUN/transport.json" --repeats 16 --out /tmp/gather8.json
```

The launcher checks for an idle RTX 4090 and runs the CUDA graph correctness fixtures before timing. Ordinary pairs have no per-layer profile markers. Instrumented pairs separately record actual fetch IDs, calculated payload, GPU gather and expert math, CPU service and exposed CPU wait. Full traces include all timings and token comparisons.
