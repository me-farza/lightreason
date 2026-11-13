# Single-request BF16 decode on the RTX 4090

The control uses the pinned Qwen3.6-35B-A3B BF16 target, the seven-proposal
DFlash2 draft, a 15 GiB primary GPU expert cache, 128 GPU expert replicas, and
34 layers with shared lossless CPU and CUDA weight packing. Each request uses
an uncached prompt and starts with an empty primary expert cache. The control and its four streamed
benchmark medians are in the [34-layer report](EXPERT_TRANSPORT8.md#expanding-the-shared-layer-budget).

The following numbers use the first instrumented 34-layer run for each prompt.
The decode interval excludes prefill. Milliseconds per pass divide the server
decode interval by the actual number of verification passes.

| Prompt | Committed IDs/pass | Decode ms/pass | Gather ms/pass | Budget at 100 tok/s |
| --- | ---: | ---: | ---: | ---: |
| Weather | 4.636 | 102.13 | 71.35 | 46.36 ms |
| Arithmetic | 6.018 | 123.06 | 90.32 | 60.18 ms |
| Coding | 5.253 | 106.13 | 75.21 | 52.53 ms |
| Chat | 4.693 | 108.99 | 77.35 | 46.93 ms |

The eight-position graph can commit up to eight IDs per pass. At the measured
weather pass time, eight commits per pass would give 78.3 tok/s. At the
measured acceptance, 100 tok/s would require a 55% reduction in pass time.
The weather passes request 1.109 GB over PCIe and at least 0.974 GB of CPU expert
weights under the current tile format. The CPU estimate counts each distinct
overflow expert once and assigns its smallest 4.5 MiB packed size. Servicing
those requests within 46.36 ms would require at least 44.9 GB/s of combined
weight reads. The comparable demand on the other prompts is 41.3–47.4 GB/s.
These figures describe the measured routing, cache behavior, and acceptance.
Changing them can change the demand. [Per-pass measurements](results/native50/cpu-shared-coverage34.json.gz)

## Asynchronous CUDA tile reads

An Ada CUDA experiment pipelined reads from mapped host memory with two or
four stages per warp. It used the same packed BF16 data, raw fallback tiles,
fetch IDs, and destinations as the control. All 18 traced fetch lists passed
bitwise checks, including empty plans and untouched slots. Each test ran 24
graph replays in alternating order.

The current kernel took 12.400 ms across the held-out medians. The fastest
asynchronous geometry took 12.403 ms. The timings cover standalone copy and
unpack. Studio retains the current kernel. [Measurements](results/native50/ada-async-gather.json)

Apply [the experiment patch](ada_async_gather_experiment.patch) in a separate
copy of the benchmark sources and run the shared CPU layout copy probe in the
[reproduction instructions](EXPERT_TRANSPORT8.md#cuda-reads-of-the-same-packed-host-banks)
with `--repeats 24`.

## Repeated-expert retention

The [retention patch](freetoken_expert_reuse.patch) gives recently reused
experts up to three passes of extra eviction priority. Both graphs use the
same cache capacity, model, draft, and 34 packed layers. The candidate can
change which expert routes use the CPU and which use the GPU. A first launch
was stopped before timing to match the frozen prefill settings and enable
raw token ID capture. The completed run used those settings in both arms.

| Weather, 256 output IDs | Frozen policy | Retention |
| --- | ---: | ---: |
| Stream decode | 45.89 tok/s | 43.48 tok/s |
| Non-stream decode | 46.09 tok/s | 43.67 tok/s |
| Stream first text | 1.434 s | 1.428 s |
| Stream complete request | 6.994 s | 7.294 s |
| Committed IDs per verification pass | 4.636 | 4.397 |
| Profiled PCIe payload per pass | 1.109 GB | 1.112 GB |
| Profiled gather per pass | 70.35 ms | 70.39 ms |

Each ordinary rate is the median of two alternating pairs. Two more
alternating streamed pairs collected the per-pass profile. All six controls
matched the archived IDs, stop reason, and commits per pass. All six
candidates differed first at output token 251. Both stopped at the original
cap. The candidate required 58 passes versus 55 for the control. The cause
of the output change remains open. Preparation took
253.64 s before the graph warmups. Peak sampled temperatures were 54 °C GPU
and 69.5 °C CPU at the existing limits. The benchmark rejected the candidate
on both exactness and speed, so the held-out requests were skipped. Studio
uses the frozen policy. [Paired timings, profiles, and raw IDs](results/native50/expert-reuse-100.json.gz)

To reproduce, apply the retention patch to an isolated copy of the complete
34-layer FreeToken build from the [34-layer instructions](EXPERT_TRANSPORT8.md#expanding-the-shared-layer-budget).
Use the same model, cache calibration, pinned draft, temperature watchdog,
and benchmark suite. Replace the control budgets with 34 and add the
retention flag to the full-model command. The expected rejection writes the
report before exiting with an error.

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_expert_reuse.patch
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
# Use these arguments with the 34-layer command and fresh output paths.
# --cpu-read-control-layers 34 --shared-cpu-control-layers 34 --expert-reuse-window 3
```

## Headroom in the existing fetch plan

The [offline replacement replay](replay_hot_expert_cache.py) processes the
first instrumented 34-layer run for each prompt. It keeps the original
primary-cache fetch plan and excludes reads served by the existing
[128-row GPU plan](results/native50/hot8-plan.json). An additional cache can
retain or bypass each requested PCIe row, starts empty at decode, and knows
all future fetches. The table gives the maximum fraction of recorded
fetches avoided under those assumptions. It ignores metadata, GPU copy
costs, and any cache state carried over from prefill.

| Prompt | 128 extra rows (0.75 GiB) | 256 extra rows (1.5 GiB) | 512 extra rows (3 GiB) |
| --- | ---: | ---: | ---: |
| Weather | 6.74% | 11.25% | 19.60% |
| Arithmetic | 6.99% | 11.82% | 20.95% |
| Coding | 7.97% | 13.86% | 23.36% |
| Chat | 8.41% | 14.90% | 25.13% |

An ordinary LRU overlay hit zero times on these traces at all three
capacities. Even the future-aware 128-row overlay could avoid at most 8.53%
of weather PCIe bytes, charging each avoided row the maximum 6 MiB. The
additional capacity is hypothetical, and the replay holds primary-cache
decisions fixed.

The validated 34-layer runner still measures 43.70–50.07 tok/s across the
four streamed medians. On weather, 100 tok/s needs 4.64 commits every 46 ms
or eight every 80 ms. The best next candidate needs to amortize host reads
across more committed tokens. The earlier
[16-position test](NATIVE50.md#wider-verification-on-the-rtx-4090) increased
acceptance but increased expert traffic enough to reduce decode throughput.

## Choosing verification width from the draft and expert reads

The [draft patch](freetoken_draft_budget.patch) exposes the top candidate's
shortlist probability at each of the seven DFlash2 positions. The policy
fits those scores to observed conditional acceptance and estimates the cost
of each real verifier width from its measured expert payload and pass times.
The fit uses eight separate calibration prompts and distinguishes the first
24 decoded tokens from later passes. This follows the observation that draft
positions can increase the union of expert weights read per pass.
[EcoSpec](https://arxiv.org/abs/2607.12696)

The candidate drafts with the same checkpoint and chooses a four-, six-, or
eight-position target graph before it allocates verification rows. Each graph
has the matching attention, recurrent, and convolution scratch. The complete
cycle includes the full DFlash2 draft, the seven-score readback, the decision,
target verification, the accepted-prefix commit, and streaming when enabled.
The fixed control uses the original draft graph and eight target positions.
The only outputs counted are IDs committed by Qwen. The output gate compares
all IDs and the stop reason. The pass schedule is recorded separately.

| Prompt | Stream decode, fixed → adaptive | Non-stream decode, fixed → adaptive | Stream request, fixed → adaptive | First differing output ID |
| --- | ---: | ---: | ---: | ---: |
| Weather | 45.92 → 48.99 tok/s | 46.03 → 49.00 tok/s | 6.979 → 6.636 s | 31 |
| Arithmetic | 49.38 → 49.17 tok/s | 49.39 → 49.18 tok/s | 7.590 → 7.560 s | 247 |
| Coding | 50.33 → 53.36 tok/s | 50.25 → 53.70 tok/s | 16.251 → 15.394 s | 14 |
| Chat | 44.01 → 49.32 tok/s | 44.00 → 49.46 tok/s | 19.694 → 14.615 s | 307 |

The ordinary timings use two alternating pairs per prompt and mode. One
additional streamed pair per prompt collected the profiles. All 20 fixed
runs matched the frozen IDs, stops, and commits. Every adaptive run differed
from its corresponding fixed run. The first mismatch reproduced across
repeats, HTTP modes, and profiling. Both arithmetic answers and both chat
answers stopped naturally. Their output counts were 344 → 341 and 827 → 676,
respectively. The shorter chat answer contributes to its lower request time.
Weather and coding reached their original caps. The adaptive candidate
failed the exact-output gate. The cause of the first differing decisions
has yet to be isolated on identical prefixes.

| Profiled prompt | PCIe MB/committed ID | Estimated CPU weight MB/committed ID | Mean pass time | Committed IDs/pass |
| --- | ---: | ---: | ---: | ---: |
| Weather | 239.19 → 221.97 | 233.28 → 210.23 | 100.10 → 92.16 ms | 4.636 → 4.554 |
| Arithmetic | 232.13 → 231.30 | 220.80 → 216.07 | 122.20 → 108.01 ms | 6.018 → 5.312 |
| Coding | 222.62 → 206.43 | 210.88 → 193.23 | 104.23 → 94.61 ms | 5.253 → 5.079 |
| Chat | 255.64 → 229.27 | 240.14 → 216.17 | 106.52 → 107.29 ms | 4.693 → 5.315 |

Weather used widths 4, 6, and 8 on 12, 26, and 18 passes. It saved 7.2%
of PCIe payload per committed token, and the gather fell from 70.48 to
63.81 ms per pass. The candidate's first streamed text took 1.429 s on
weather, compared with 1.424 s for the fixed control. These profiles use the
candidate's actual outputs. They establish the traffic cost of those runs.
Ordinary throughput comes from the uninstrumented pairs.

The fitted steady-state cycle costs were 59.22, 74.86, and 95.18 ms for the
four-, six-, and eight-position graphs. Perfect acceptance at those costs
would yield 67.55, 80.15, and 84.05 tok/s. This diagnostic holds cycle cost
fixed. Different routing, wider blocks, or fewer weight reads can change it.
[Live pairs, confidence calibration, and per-pass profiles](results/native50/draft-budget-qwen.json)

## Reusing CPU weight loads across three routes

The [native CPU patch](freetoken_cpu_three_routes.patch) groups three routes
to the same expert when doing so saves a weight read versus the original
pair grouping. Each route retains its FP32 accumulation lanes, BF16 activation
rounding, and original router-order reduction. The [isolated trial](results/native50/cpu-three-routes.json)
used actual BF16 Qwen rows and synthetic route placements at shapes observed
in the existing profiles. All 576 alternating pairs matched bit for bit.
With packed reads in the two clustered cases where a read was saved, coding
went from eight to seven groups and 0.486 to 0.460 ms. Chat went from ten to
nine groups and 0.659 to 0.632 ms. The balanced cases did not exercise the
three-route path. This is CPU kernel timing without concurrent PCIe reads.

The full-model ablation keeps the target at eight positions. Both arms use
the rebuilt extension, with the original pairing path selected in the
control. It compares token IDs, stops, pass commits, and profiled PCIe
fetch lists. The profile counts scheduled CPU weight-read groups. A read
group accounts for one full expert's worth of weight references across the
workers. Physical DDR traffic can differ because of CPU cache hits and
raw-tile fallback. The profile reports logical CPU payload separately from
the measured PCIe copy plan.

| Prompt | Original → grouped decode | Streamed request | CPU read groups/pass | PCIe MB/committed ID |
| --- | ---: | ---: | ---: | ---: |
| Weather | 45.97 → 46.15 tok/s | 6.976 → 6.959 s | 220.47 → 211.05 | 239.19 → 239.19 |
| Coding | 50.02 → 50.32 tok/s | 16.344 → 16.271 s | 225.35 → 215.59 | 222.62 → 222.62 |

Ordinary results are medians of two alternating streamed pairs per prompt.
One additional profiled pair per prompt measured the read counts. All six
pairs matched the original IDs, stops, and per-pass commits through the
existing caps. Both profiled pairs had identical ordered PCIe fetch lists.
Logical CPU weight-read groups fell by 4.27% on weather and 4.33% on coding.
Weather's exposed CPU wait fell from 9.58 to 8.89 ms per pass. Its gather
took 70.65 and 70.47 ms per pass. The measured benefit from sharing CPU
reads is small with that gather time still on the critical path.

Weather had 206.42 distinct CPU experts per pass, leaving at most 4.64
additional CPU weight-read groups per pass to remove by sharing across more
routes on that trace. The profiled exact candidate still takes 100.16 ms per
pass. At 4.636 commits per pass, 100 tok/s leaves 46.36 ms for the entire
pass, 24.11 ms less than its gather span alone. The CPU grouping patch is
available as an opt-in experiment. [Exact pairs and profiles](results/native50/draft-budget-qwen.json)

Preparation for the successful paired run took 259.69 s before separate
graph warmups and confidence calibration. It audited 102,404,549 encoded
BF16 tiles. The launcher initially lacked a calibration module, and two
subsequent startups exposed four- and eight-position assumptions in the
six-position copy and profiling paths. The successful run checked all three
route counts before starting the server. Peak sampled temperatures were
53 °C GPU and 72.25 °C CPU under the existing stock limits.

### Reproduction

Start from an isolated copy of the 34-layer build and use the pinned inputs,
CPU affinity, driver libraries, Triton cache, and watchdog from the
[34-layer experiment](EXPERT_TRANSPORT8.md#expanding-the-shared-layer-budget).
Restore the original vendor extension from `liblightreason_cpu_moe_base.so`
in the isolated copy before rebuilding, as in the CPU read experiment.

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_draft_budget.patch
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_cpu_three_routes.patch
bash edge35b/build_freetoken_bf16_pairs.sh "$FT_PYTHON" "$FT_SOURCE"
export PYTHONPATH="$PWD:$FT_SOURCE"
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 450 --out "$RUN/cpu-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -m edge35b.benchmark_cpu_three_routes \
  --model "$MODEL" --profile "$NATIVE_PROFILE" --repeats 24 --out "$RUN/cpu-three-routes.json"

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1800 --out "$RUN/width-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials "$RUN/width.cfg" --packed-transfer --packed-transfer-width 8 \
  --packed-transfer-reference "$CONTROL" --hot-expert-plan "$HOT_PLAN" \
  --native-packed-gather cuda-8-128-default --cpu-read-policy packed-both-prefetch \
  --cpu-read-profile "$NATIVE_PROFILE" --cpu-read-layers 34 \
  --shared-cpu-gather cuda-4-256-cpu-tiles --shared-cpu-layers 34 \
  --cpu-read-control-layers 34 --shared-cpu-control-layers 34 \
  --draft-budget --cpu-three-routes --draft-dir "$DRAFT" --draft-steps 7 \
  --expert-load parallel --cache-gib 15 --host-embedding --cache-type naive --prefill-hit-d2d --port 1924 \
  --server-log "$RUN/width-server.log" --out "$RUN/draft-budget.json"
```

## Lossless GPU-resident expert weights

The [resident cache](gpu_resident_experts.py) adds a preloaded source for
expert weights behind the existing 2,560-slot primary cache. The router,
CPU/GPU assignment, destination slots, BF16 expert computation, and target
commit stay on the frozen eight-position path. All three arms use the
34 shared host-packed layers, the original 128 raw GPU replicas, and the
exact three-route CPU kernel. Requests use uncached input and start with
an empty primary expert cache.

The extra arena has two layouts. The raw layout uses a prefix of the ranked
expert list. The packed layout fits a longer prefix in the same arena.
Each compressible tile stores the low BF16 bytes and four-bit high-byte
codes. A per-tile header identifies the exponent base and the offset from
preceding inline raw fallback tiles. The Ada gather reconstructs the
original BF16 bytes directly into the planned primary slot. The
[ranking script](plan_gpu_resident_cache.py) uses eight independent
calibration prompts and weights their fetches by estimated PCIe payload
per committed ID. The [ranked list](results/native50/gpu-resident-plan.json)
excludes the original 128 replicas. No generation state is stored in the
weight arena.

The guarded RTX 4090 run fit **146 raw or 187 packed additional experts**
in **880 MiB**, a 28.08% increase in rows. The requested cap was 896 MiB.
The allocator retained 512 MiB for graph capture and runtime workspace.
The raw layout used 876.00 MiB.
The packed layout used 878.66 MiB including headers and raw fallbacks.
The production Ada gather reconstructed and compared all **459,276,288
raw** and **588,251,136 packed** BF16 values before timing. The [CUDA graph
fixture](test_gpu_resident_experts.py) exercised mixed packed and raw tiles,
descriptor changes between replays, empty fetches, and untouched slots.
The arena setup and audits took 3.55 s, and 348.75 MiB of VRAM was free after
graph capture. Both layouts reuse the same GPU allocation and descriptor
addresses. Layout loads and short warm requests are logged separately from
the timed requests.

| Prompt | Fixed decode | Raw resident decode | Packed resident decode | Raw → packed PCIe payload saved vs fixed |
| --- | ---: | ---: | ---: | ---: |
| Weather | 43.93 tok/s | 46.74 tok/s | 46.80 tok/s | 2.57% → 3.36% |
| Arithmetic | 49.60 tok/s | 49.86 tok/s | 50.33 tok/s | 2.87% → 3.65% |
| Coding | 50.39 tok/s | 51.06 tok/s | 51.08 tok/s | 3.25% → 4.24% |
| Chat | 44.00 tok/s | 44.59 tok/s | 44.86 tok/s | 3.36% → 4.37% |

The decode columns are medians of two ordinary streamed runs per arm and
prompt. The first streamed weather control ran at 41.79 tok/s and the
reverse-order control ran at 46.07. All samples are retained. The second
weather pair measured 46.07 → 46.80 tok/s with packing. The cause of the
slow first control was not isolated. The highest non-streaming median was
51.33 tok/s on coding, with a paired control of 50.33 and raw arena of
51.01 tok/s. Packed coding took 16.048 s for the streamed request and
1.024 s to first text, compared with 16.236 s and 1.014 s for the control.

Across all ordinary streamed and non-streamed pairs, raw residency gained
1.44% and packed residency gained 1.76% by median paired decode ratio.
Packed gained 0.42% against equal-budget raw. Each arm matched the original
IDs, stop, and commits on all 20 pairs, including four separately profiled
pairs. Every profiled comparison matched the original ordered fetch plan.
Arithmetic and chat stopped naturally. Weather and coding reached their
original caps. Profiled payload is derived from executed fetch IDs and the
bytes required by their source format. The packed cache served 390, 545,
1,362, and 1,738 fetches on the four requests, respectively.

On weather, PCIe payload fell from 239.19 to 231.15 MB per committed ID.
The expert gather fell from 70.43 to 68.40 ms per pass. Exposed CPU wait
rose from 8.66 to 10.28 ms, and the same 211.05 CPU read groups were issued
per pass. The profiled cycle still took 99.47 ms with 4.636 IDs committed.
The packed chat profile ran at 41.54 tok/s, compared with its ordinary
44.86 median. All speed comparisons use the ordinary pairs.

Model load, bank preparation, and graph setup took 283.86 s. Median arena
fill time was 39.1 ms raw and 39.3 ms packed when switching between test
arms. A steady layout stays resident between requests. Peak sampled
temperatures were 53 °C GPU and 71.5 °C CPU at the existing limits.
[Paired results and exact output IDs](results/native50/gpu-resident-qwen.json.gz)

At weather's current acceptance, 100 tok/s allows 46.36 ms per pass. If
the fixed control's other work holds at 29.45 ms, the gather has 16.91 ms.
With gather time proportional to payload at its measured rate, that
corresponds to about 76% fewer PCIe bytes. A hindsight ranking of the
weather trace needs roughly 3,800 extra resident rows to cover that share
of payload, or about 17 GiB at the measured packed row size. This estimate
assigns each fetch its layer's average payload and keeps the primary fetch
plan fixed. The real cache would need to reduce CPU wait as well.

### Resident cache reproduction

Use an isolated copy of the preceding 34-layer runtime with the draft-budget
patch and rebuilt three-route CPU extension. Use the same model and draft
revisions, FreeToken environment, bandwidth calibration cache, driver
libraries, CPU affinity, and watchdog. The new runtime patch enables the
resident experiment when the benchmark supplies a plan.

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_gpu_resident_cache.patch
export PYTHONPATH="$PWD:$FT_SOURCE"
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
python3 -m edge35b.plan_gpu_resident_cache edge35b/results/native50/draft-budget-qwen.json \
  "$HOT_PLAN" --out "$RUN/gpu-resident-plan.json"

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 120 --out "$RUN/resident-fixture-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -m unittest edge35b.test_gpu_resident_experts

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1800 --out "$RUN/resident-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials "$RUN/resident.cfg" --packed-transfer --packed-transfer-width 8 \
  --packed-transfer-reference "$CONTROL" --hot-expert-plan "$HOT_PLAN" \
  --native-packed-gather cuda-8-128-default --cpu-read-policy packed-both-prefetch \
  --cpu-read-profile "$NATIVE_PROFILE" --cpu-read-layers 34 \
  --shared-cpu-gather cuda-4-256-cpu-tiles --shared-cpu-layers 34 \
  --cpu-read-control-layers 34 --shared-cpu-control-layers 34 --cpu-three-routes \
  --gpu-resident-plan "$RUN/gpu-resident-plan.json" --gpu-resident-mib 896 \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1924 \
  --server-log "$RUN/resident-server.log" --out "$RUN/gpu-resident.json"
```

### Reading the packed weights inside expert computation

The [GEMM probe](benchmark_resident_gemm.py) used actual Qwen gate/up and
down weights from four layers, with eight target positions and synthetic
routes shaped by the profiled CPU assignments. Each method retained the
original Triton dot-product tiles and BF16 outputs. It tested unpacking in
the load path, sharing each metadata header across its input block, and
staging each active packed expert in a temporary BF16 buffer before the
ordinary GEMM. Fusing decompression with the compute path is also the
approach evaluated in [ZipServ](https://arxiv.org/abs/2603.17435).

| Method | Sum of eight projection medians | BF16 output matches |
| --- | ---: | ---: |
| Raw GPU weights | 0.488 ms | Reference |
| Direct packed reads | 1.662 ms | 8/8 |
| Direct packed reads with shared headers | 1.369 ms | 8/8 |
| CUDA unpack to staging followed by raw GEMM | 1.621 ms | 8/8 |

Each projection ran 24 alternating samples with eight graph replays per
sample. These are isolated GPU projection timings. All three packed
compute methods were slower at the measured shapes. The full generation
experiment uses packed replicas only to replace selected host fetches.
[First direct probe](results/native50/resident-gemm-naive.json) · [Four-way probe](results/native50/resident-gemm-methods.json)

Apply the [GEMM patch](freetoken_resident_gemm_probe.patch) to a separate
copy of the runtime and run on an idle 4090 under the same watchdog.

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_resident_gemm_probe.patch
taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 240 --out "$RUN/resident-gemm-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.benchmark_resident_gemm \
  --model "$MODEL" --trace edge35b/results/native50/draft-budget-qwen.json \
  --repeats 24 --out "$RUN/resident-gemm-methods.json"
```

## Packing the primary GPU expert cache

The control is the 880 MiB packed-replica runner above, with its 15 GiB
raw primary cache, 34 shared host-packed layers, and three-route CPU
grouping. The [primary-cache patch](freetoken_packed_primary_cache.patch)
reuses the same 15 GiB allocation after prefill. The target remains the
pinned BF16 checkpoint. The proposer, eight verification positions,
sampling, prompt suite, and output caps are unchanged. The storage and
scheduling tradeoff also appears in [ZipMoE](https://arxiv.org/abs/2601.21198).

The target revision is `345d467b117d1fd5eda9b879a5988de9ae5f47a7` and
the DFlash2 revision is `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`.
Each request starts with an uncached input and an empty logical expert
cache. The weather and coding runs cap generation at 256 and 768 IDs.
Arithmetic and chat run through their natural stops in the frozen control.
Decode counts only committed IDs after the first token. Setup, first text,
and total request time are stored separately. [Workstation and reference](NATIVE50.md)

### Compact slots with BF16 staging

The first [implementation](packed_primary_cache.py) stores eligible
experts in compact slots with inline raw fallback tiles. The device-side
refill writes the compact cache and reconstructs active rows for the
original BF16 expert GEMM. It uses a raw tier for layers whose worst-case
row exceeds the packed stride. The compact bank holds 2,947 rows. Another
256 rows remain raw, and 65 rows provide BF16 staging. The total is
3,203 cache slots versus 2,560 in the control. Only five of 40 layers
qualified for the compact tier under the chosen fixed stride. The audit
matched 88,080,384 sampled BF16 values through fills, hits and refills.

| Prompt | Control stream decode | Staged primary stream decode | PCIe MB per committed ID | First different ID |
| --- | ---: | ---: | ---: | ---: |
| Weather | 44.34 tok/s | 16.22 tok/s | 231.15 → 740.26 | 31 |
| Arithmetic | 50.00 tok/s | 21.78 tok/s | 223.66 → 563.62 | 247 |
| Coding | 51.13 tok/s | 19.55 tok/s | 213.18 → 623.64 | 116 |
| Chat | 44.78 tok/s | 17.34 tok/s | 244.47 → 699.08 | 306 |

Decode rates are two ordinary streamed runs per prompt and arm, with
alternating order. PCIe payload comes from the separate profiled pairs.
All 20 controls matched the frozen token IDs, stops and pass commits.
The staged path changed the output IDs in all 20 pairs. It changed
cache residency and CPU/GPU assignment even though its sampled weights
matched. The smaller raw tier drove more misses. On weather, gather time
rose from 68.73 to 199.21 ms per pass, and exposed CPU wait rose from 9.37
to 38.83 ms. The cause of each token mismatch was not isolated on identical
prefixes. Preparation took 286.72 s before the per-graph warmups. Peak
sampled temperatures were 53 °C GPU and 72.5 °C CPU at the existing limits.
[Full timings, profiles and IDs](results/native50/primary-cache-staged.json.gz)

### Reusing the prefill buffers as a compact tier

The second design keeps the usual BF16 prefill buffers. During decoding,
it uses their 512-row address range as a packed cache and the remaining
2,048 raw slots for the unchanged BF16 expert GEMM. The packed tier
only serves the raw planner's misses. Its copy kernel reconstructs the
original BF16 slot and writes a cacheable miss in compact form without
a second host read. It uses per-expert size checks so a large outlier
cannot overflow a packed slot. This changes the raw planner's capacity,
so exact output is still a measured gate.

The [offline replay](replay_primary_cache.py) on the existing runner's
fetch IDs removed the GPU replicas and gave a 624-row tier no prefill
seed. Plain LRU yielded zero hits on each trace. The future-aware bound
was 23–30% of the remaining fetches. Requiring a recent second fetch
retained some of the reuse, especially on the longer coding and chat
traces. This replay holds the raw planner's fetch sequence fixed. The
live split has a smaller raw tier and can see a different sequence.
[Replay counts](results/native50/primary-cache-replay.json)

The live compact tier fit 624 packed rows for 2,672 physical slots total.
The sampled fill and hit audit matched 12,582,912 BF16 values. The run
ended after nine ordinary pairs. No profiled pairs were recorded. The
watchdog recorded an external-command failure, with no temperature stop.

| Prompt | Control stream decode | Compact tail stream decode | First different ID |
| --- | ---: | ---: | ---: |
| Weather | 41.74 tok/s | 39.57 tok/s | Same output |
| Arithmetic | 50.50 tok/s | 36.77 tok/s | 247 |
| Coding | 50.83 tok/s | 36.58 tok/s | 116 |
| Chat | 44.80 tok/s | 37.41 tok/s | 306 |

Each row is one streamed pair. The non-streamed weather run also matched
all output IDs and the stop. Both weather runs took 56 passes with the
compact tier and 55 with the control. Chat first diverged at token 307
and token 187 in its two non-streamed runs. All nine candidates failed
the full exact-output and pass-commit gate. The control matched the
frozen records in all nine comparisons.
[Partial timings and output IDs](results/native50/primary-cache-compact-tail-partial.json)

### Packing physical rows with the frozen planner

The [fixed-prefix path](fixed_packed_primary.py) keeps all 2,560 logical
slots and uses the original CPU/GPU planner. A configurable prefix uses
compact physical storage for eligible rows during decode.
The GPU reconstructs the active packed rows into 65 BF16 scratch rows
before the existing expert GEMM. A packed logical hit whose row exceeds
the stride must be refetched into scratch. The live profile accounts for
those refills separately from the planner's fetch list.

A 2,048-slot prefix frees 1,146 MiB for replicas in the same 15 GiB
arena. The first attempt at that size and a 1,024-slot retry with chunked
host pinning both stopped during setup when sensor readings went stale.
Neither recorded timed pairs. The latter encoded 73 extra replicas within
378 MiB before the stop. The implementation now reuses the established
pinned upload buffer from the ordinary replica arena, which stays packed
throughout the comparison. Two retries of this version also stopped when
temperature readings went stale during shared host-bank setup, including
one using the established chunked registration path. The fixed-prefix
full-model comparison remains unverified. [Watchdog outcomes](results/native50/primary-cache-watch-summary.json)

The extra replicas continue the calibration ranking after the already
resident rows. Routed prefill can write across the arena, so this version
reloads the extra replicas before its first decode pass. The request and
decode clocks include that copy. The profile reports its bytes separately
from the per-pass expert payload.

On the frozen fetch trace, the next 240 ranked rows cover approximately
4.35% of weather's PCIe payload and 4.58–5.44% on the other prompts.
Selecting the rows with hindsight covers 8.08% on weather. Those are
replay estimates at unchanged acceptance, based on the average payload
of each layer's fetch list. They exclude extra refills and staging costs.

### Isolated Ada expert computation

The [layer benchmark](benchmark_fixed_primary_layer.py) loaded original
Qwen BF16 experts from four layers. Each fixture used 24 distinct experts,
eight positions and synthetic routes with CPU-assignment counts drawn
from the frozen profiles. The control uses FreeToken's actual decode GEMV
and reduction. The staged path uses the same arithmetic after reconstructing
the active rows. The direct path reads the lossless encoding inside GEMV.
The layers were timed independently over 24 alternating samples of eight
graph replays each.

| Layer | Raw BF16 | Staged prefix | Unpack and remap alone | Direct packed |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.181 ms | 0.363 ms | 0.185 ms | 0.908 ms |
| 17 | 0.181 ms | 0.361 ms | 0.185 ms | 1.056 ms |
| 37 | 0.181 ms | 0.360 ms | 0.184 ms | 1.109 ms |
| 0 | 0.181 ms | 0.361 ms | 0.188 ms | 0.762 ms |

Staging matched the original routed BF16 weights and expert outputs on
all four fixtures through fill and reuse. The direct kernel changed
13,782–13,977 of 16,384 BF16 output values per fixture. The maximum absolute
error ranged from 0.000244 to 0.000610. Its numerical cause was not isolated.
All sources in this probe were on the GPU. Its timing covers the expert
layers and reconstruction, outside the full verifier and without host
PCIe traffic or CPU overflow computation. [Layer timings and checks](results/native50/primary-cache-layer.json)

### Remaining decode budget

The exact packed-replica control's weather profile committed 4.636 IDs in
99.47 ms per pass. At that acceptance, 100 tok/s allows 46.36 ms per pass.
The gather alone took 68.40 ms. Holding the other stages fixed leaves
15.29 ms for the gather, about 78% below the measured time. The coding
control had 5.253 commits in 102.25 ms per pass, leaving 52.53 ms at the
target rate. It would need about 70% less gather time if its other stages
hold. CPU work overlaps the gather and can become exposed as copies speed
up. These are timing budgets at the measured acceptance and routing.

At the current weather pass time, committing all eight verifier positions
would yield about 80 tok/s. Further work needs to reduce weight traffic and
verification cost together or test a different proposer with enough
acceptance to amortize that traffic. The measured improvement from a small
resident tier and the offline replacement bounds constrain what another
cache-only change can deliver.

### Reproduction

Use the [resident experiment command](#resident-cache-reproduction) with the same
pinned model, draft, CPU affinity, power and temperature limits, and
benchmark suite. Use `--gpu-resident-mib 880`, set fresh output paths for
each run, and apply the primary patch after the resident patch.
The chunked host-bank preparation is shared by both arms when enabled.

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_packed_primary_cache.patch
# Add one of these sets of arguments to the resident experiment command.
# --packed-primary --packed-primary-layout staged-primary
# --packed-primary --packed-primary-layout compact-tail --packed-primary-reserve-rows 512
# --packed-primary --packed-primary-layout fixed-prefix --packed-primary-reserve-rows 1024
# Use the existing chunked host registration for fixed-prefix preparation.
# export LIGHTREASON_STUDIO_CACHE_PARK=1

python3 -m edge35b.replay_primary_cache edge35b/results/native50/gpu-resident-qwen.json \
  "$HOT_PLAN" --resident-plan "$RUN/gpu-resident-plan.json" --out "$RUN/primary-cache-replay.json"

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --final-report-only --safety-sampling \
  --command-timeout-s 180 --out "$RUN/primary-cache-layer-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -m edge35b.benchmark_fixed_primary_layer \
  --model "$MODEL" --trace edge35b/results/native50/gpu-resident-qwen.json \
  --repeats 24 --out "$RUN/primary-cache-layer.json"
```

## Joint DFlash2 and verifier tuning

The paired experiment uses the existing 880 MiB packed-replica runner as
the control. The target, checkpoint, prompt IDs, greedy settings, 2,560
primary slots, 34 host-packed layers, and three-route CPU kernels stay
fixed. Each request starts with an empty expert LRU and an uncached prompt.
The control uses the original seven DFlash2 proposals and eight target
positions. The three candidates isolate prefix selection at width eight,
adaptive width with the stock selector, and both changes together.

The [selector](dflash_prefix_selection.py) evaluates the existing DFlash2
transition scores for each adjacent pair of shortlisted tokens. For
conditional probability `p` and expected subsequent acceptance `v`, it
scores choosing a token as `p * (1 + v)` and works backward across the
seven positions. Only Qwen's verifier accepts output or commits target
state. The adaptive arm uses real four-, six-, and eight-position graphs.
It uses the earlier development-prompt calibration for conditional
acceptance and verification costs. Applying the same calibration to the
new selector is an assumption under test. The conditional scoring and
cost-aware width selection draw on [DCOCO](https://arxiv.org/abs/2610.12570)
and [HybridSpec](https://arxiv.org/abs/2611.04055).

The [shortlist replay](dflash_joint_analysis.py) checks that the archived
control and candidate audit used identical token IDs, stops and pass
boundaries. It then gives the selector the actual subsequent tokens
wherever they appear in the recorded shortlist. The table holds each
control pass cost fixed. A different live proposal changes pass boundaries
and can change expert traffic and floating-point evaluation.

| Prompt | Actual committed IDs/pass | Shortlist oracle IDs/pass | Oracle tok/s at current cost | All eight accepted at current cost |
| --- | ---: | ---: | ---: | ---: |
| Weather | 4.64 | 6.82 | 68.55 | 80.43 |
| Arithmetic | 6.02 | 7.58 | 63.20 | 66.72 |
| Coding | 5.25 | 7.29 | 71.28 | 78.24 |
| Chat | 4.69 | 6.45 | 61.74 | 76.59 |

Even the shortlist oracle would need weather verification to finish in
68.18 ms per pass for 100 tok/s, down from 99.47 ms. On coding the
corresponding budget is 72.88 ms, down from 102.25 ms.
[Replay inputs and estimates](results/native50/dflash-prefix-headroom.json)

### Full generation on the RTX 4090

Each prompt ran twice with streaming and twice without it, alternating
execution order. A separate streamed run profiled each arm. All 20 control
requests reproduced the archived packed runner's token IDs, stop reasons,
and commits per pass. The table uses the medians of the ordinary runs.
Decode rates exclude prefill and count IDs emitted after the first token.

| Prompt | Packed control | Prefix at width 8 | Stock selector, adaptive width | Prefix and adaptive width |
| --- | ---: | ---: | ---: | ---: |
| Weather | 42.19 / 46.42 | 42.51 / 43.74 | 46.77 / 49.68 | 46.62 / 49.31 |
| Arithmetic | 50.39 / 50.42 | 48.34 / 48.72 | 49.66 / 49.83 | 50.31 / 50.42 |
| Coding | 51.08 / 51.23 | 46.43 / 46.43 | 54.23 / 54.40 | 51.43 / 51.64 |
| Chat | 44.74 / 44.58 | 46.33 / 46.50 | 50.32 / 50.22 | 50.62 / 50.76 |

Each cell gives streamed / non-streamed tok/s. The first streamed weather
control ran at 37.93 tok/s and its reverse-order repeat reached 46.44.
The cause of the slow first sample was unresolved. All samples contribute
to the medians. The median paired decode change across ordinary runs was
−3.15% for prefix at width eight, +6.54% for stock adaptive, and +3.35% for
the joint arm.

| Prompt | Packed first text / complete request | Joint first text / complete request | First differing ID, prefix / stock adaptive / joint |
| --- | ---: | ---: | ---: |
| Weather | 1.438 / 7.542 s | 1.434 / 6.921 s | 113 / 31 / 31 |
| Arithmetic | 0.641 / 7.449 s | 0.641 / 7.401 s | 211 / 247 / 247 |
| Coding | 1.013 / 16.031 s | 1.012 / 15.931 s | 200 / 14 / 14 |
| Chat | 0.913 / 19.378 s | 0.921 / 17.136 s | 187 / 307 / 429 |

All 60 candidate comparisons failed the exact-output gate, with the same
first mismatch across repeats, HTTP modes, and profiling. Each arm
stopped naturally on arithmetic and chat, and reached the original caps
on weather and coding. Chat ended at 827 IDs for Packed, 914 for fixed
prefix, 676 for stock adaptive, and 822 for the joint arm. Changed output
length contributes to complete-request time. The cause of each first
differing logit has yet to be isolated. For example, the fixed-width
prefix selector's ordered expert fetches first differed on chat after
181 output IDs, while both arms still agreed on width and commits. The
first changed output was ID 187. A previous identical-prefix
diagnostic established that shape-dependent arithmetic can change the
argmax and that the hybrid planner's CPU/GPU assignments can depend on
the proposed suffix. The exact-output runner remains the deployment
choice. [Prior diagnosis](NATIVE50.md#token-34-and-speculation-timing)

| Profiled prompt | Committed IDs/pass, Packed → joint | Cycle time | Gather time | PCIe MB/ID | Estimated CPU weight MB/ID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weather | 4.64 → 4.64 | 99.89 → 94.06 ms | 68.52 → 64.55 ms | 231.15 → 219.16 | 223.35 → 206.16 |
| Arithmetic | 6.02 → 5.23 | 119.24 → 104.16 ms | 86.55 → 73.01 ms | 223.66 → 219.23 | 211.64 → 205.25 |
| Coding | 5.25 → 5.08 | 102.14 → 98.13 ms | 71.55 → 67.95 ms | 213.18 → 210.10 | 201.78 → 197.57 |
| Chat | 4.69 → 4.61 | 104.41 → 90.78 ms | 73.32 → 61.89 ms | 244.47 → 211.11 | 230.98 → 197.08 |

The CPU payload estimate counts weight read groups, with the minimum tile
size for packed layers. It includes no adjustment for cache hits or raw
fallback tiles. Gather occupied 68–73% of the control's pass time and
overlapped CPU expert computation. Exposed CPU wait took 9.50–12.05 ms
per pass and GPU expert math took 9.41–9.68 ms. The fixed-prefix arm's
coding profile reduced average gather time by 1.21 ms but committed 4.73
IDs per pass versus 5.25, raising the PCIe payload per committed ID by 9.3%.
The reused calibration forecast 5.14 commits per joint pass versus 4.87
observed on 445 profiled passes away from stop and length boundaries. With
the stock selector, it forecast 5.31 versus 5.14 observed on 394 passes.
The four evaluation prompts were outside the calibration set.
In the archived control traces, grouping every CPU route of each layer's
distinct expert into one read could remove only 1.6–3.3% of the already
grouped CPU reads. That bound holds for the measured placements and does
not include reuse across verification passes.

The model, banks, and graphs took 298.78 s to prepare, including 27.52 s
for replica setup and audits. The production gather checked 588,251,136
packed BF16 weight values, and setup verified 102,404,549 shared host
tiles. The first launch stopped during preparation when thermal readings
went stale. The completed retry used the existing chunked host registration
path for shared banks and peaked at 53 °C GPU, 72.25 °C CPU, and 158.34 W.
The watchdog limits were 70 °C GPU, 75 °C CPU, and stock 450 W.
[Live pairs, IDs, and per-pass profiles](results/native50/dflash-prefix-qwen.json.gz) · [Watchdog outcomes](results/native50/dflash-prefix-watch-summary.json)

### Verifying known continuations

The live diagnostic ran the full DFlash2 draft and replaced its proposed
IDs with the archived continuation only while the generated prefix
matched. Qwen still verified and committed every emitted ID. This measures
how better proposals affect acceptance and the cost of each pass. The
table covers complete passes before the first mismatch and includes
the scheduling, real drafting, transfers, target verification, and commit
inside those passes. Startup passes are included.

| Prompt | Complete matching passes | IDs/pass | Cycle | Decode | First differing ID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weather | 3 | 8.00 | 219.09 ms | 36.52 tok/s | 31 |
| Arithmetic | 30 | 8.00 | 118.79 ms | 67.34 tok/s | 247 |
| Coding | 10 | 8.00 | 126.14 ms | 63.42 tok/s | 86 |
| Chat | 38 | 8.00 | 120.27 ms | 66.52 tok/s | 306 |

On arithmetic, these passes lowered PCIe payload from 214.90 to 167.16 MB
per committed ID versus control passes ending before the same mismatch.
They committed eight IDs while gather rose from 80.36 to 85.98 ms per
pass. At eight commits, 100 tok/s needs an 80 ms cycle. The measured
gather alone already exceeded that budget. Once the prefix diverged, the
test fell back to DFlash2 for the rest of the request. These rates describe
partial outputs with foreknowledge and cannot be used as live runner rates.
[Matching-prefix measurements](results/native50/dflash-prefix-headroom.json)

### Reusing GPU expert weights across verifier rows

The [GPU probe](benchmark_verifier_expert_reuse.py) reuses each weight
tile across two routes to the same expert, preserving each route's BF16
arithmetic and the final reduction. It also tests skipping zero-weight
routes alone. The final CUDA graph replays matched the stock expert output
bit for bit across all four single-layer fixtures and eight rotating
fixtures. The rotating run
used 1,482 MiB of original Qwen weights across eight layers, with synthetic
routes sized by the measured GPU and CPU assignment counts. Pairing cut
logical weight-read groups from 416 to 280. The cost rose from 0.2178 to
0.2209 ms per layer. Skipping zero-weight routes alone took 0.2199 ms.
These replays measure GPU expert math without PCIe gathers, CPU overflow,
or full generation. The profiler puts the entire GPU expert stage at
roughly 10 ms per pass, so this stage has limited headroom for 100 tok/s.
The probe did not advance to a full-model trial.
[Single-layer fixtures](results/native50/verifier-expert-reuse-layer.json) · [Rotating fixtures](results/native50/verifier-expert-reuse-rotating.json)

### Joint benchmark reproduction

Start from the isolated [resident cache runtime](#resident-cache-reproduction).
Use the same model, draft, prompt suite, environment, and watchdog. Use
fresh output paths under `RUN`. The optional
`--draft-prefix-teacher` diagnostic verifies saved continuation blocks
after the ordinary pairs and their profiles. It includes a real draft
forward and the normal Qwen verification and commitment, while supplying
known draft IDs for prefixes that still match the frozen continuation.

```sh
patch --batch --fuzz=0 -p1 -d "$FT_SOURCE" < edge35b/freetoken_dflash_prefix.patch
"$FT_PYTHON" -m unittest edge35b.test_dflash_prefix_selection
taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --final-report-only --safety-sampling \
  --command-timeout-s 1800 --out "$RUN/dflash-prefix-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials "$RUN/dflash-prefix.cfg" --packed-transfer --packed-transfer-width 8 \
  --packed-transfer-reference "$CONTROL" --hot-expert-plan "$HOT_PLAN" \
  --native-packed-gather cuda-8-128-default --cpu-read-policy packed-both-prefetch \
  --cpu-read-profile "$NATIVE_PROFILE" --cpu-read-layers 34 \
  --shared-cpu-gather cuda-4-256-cpu-tiles --shared-cpu-layers 34 \
  --cpu-read-control-layers 34 --shared-cpu-control-layers 34 --cpu-three-routes \
  --gpu-resident-plan "$RUN/gpu-resident-plan.json" --gpu-resident-mib 880 \
  --draft-prefix edge35b/results/native50/draft-budget-qwen.json --draft-prefix-teacher \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1924 \
  --server-log "$RUN/dflash-prefix-server.log" --out "$RUN/dflash-prefix.json"

python3 -m edge35b.dflash_joint_analysis edge35b/results/native50/pass-audit.json \
  edge35b/results/native50/gpu-resident-qwen.json --live "$RUN/dflash-prefix.json" \
  --out "$RUN/dflash-prefix-headroom.json"

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --final-report-only --safety-sampling \
  --command-timeout-s 180 --out "$RUN/verifier-reuse-watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -m edge35b.benchmark_verifier_expert_reuse \
  --model "$MODEL" --trace edge35b/results/native50/gpu-resident-qwen.json \
  --repeats 24 --rotating --out "$RUN/verifier-reuse.json"
```
