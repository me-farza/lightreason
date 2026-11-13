# Slow verifier passes and DFlash2 candidate recall

The experiment uses the same Qwen3.6-35B-A3B BF16 checkpoint, DFlash2 checkpoint, greedy settings, 15 GiB GPU expert cache, and prompts as the [width and CPU pairing work](WIDTH_AUTOTUNE.md). The eight-position verifier with the fused CPU reduction is the exact-output reference for this follow-up. Each request starts with an empty expert cache and zero cached input tokens.

The ordinary weather comparison warms the four-position graph, then alternates the plain four-position graph, the same graph with the fused CPU kernel, and the eight-position graph with the fused CPU kernel. Timed and candidate-audit graphs warm separately. Ordinary decode timings have neither device timing markers nor candidate copies.

## What the pass trace measures

The [runtime patch](freetoken_pass_diagnostics.patch) captures separate instrumented four- and eight-position verifier graphs. Device markers bracket route planning, CPU dispatch, PCIe gather, GPU expert math, the wait on CPU completion, and the final combine. The trace records the number of BF16 experts gathered and the number of CPU assignments per layer. Each gathered expert accounts for 6 MiB of weight data. The [native timing patch](freetoken_expert_timing_native.patch) measures CPU service from the coordinator reading the task through worker completion.

The device wait measures how much CPU work was still outstanding after the GPU finished its own expert work. CPU service overlaps PCIe gather and GPU expert math. The graph total therefore uses the measured device wait and does not add CPU service. The remaining target time contains the other model kernels and gaps between marked expert blocks. Timing kernels and copies add diagnostic overhead, so the ordinary measurements are the speed comparison.

## Measured bottleneck

| Ordinary weather runner | Decode tok/s | Mean pass time | Complete HTTP request |
| --- | ---: | ---: | ---: |
| Warm fixed four | 42.51 | 81.06 ms | 7.42 s |
| Warm fixed four with fused CPU pairing | 42.60 | 80.90 ms | 7.41 s |
| Warm fixed eight with fused CPU pairing | 38.28 | 121.13 ms | 8.09 s |

Rates and times are the medians of two streamed runs in reverse order. Both four-position pairs matched all output IDs and the stop reason. First text was 1.42 s in all three configurations. The profiled four-position runs measured 42.35 and 42.41 tok/s, and the profiled eight-position run measured 38.24 tok/s. Every profiled run matched its corresponding ordinary verifier.

| Mean per four-position pass | Full run | Fastest 32 passes in sequence | Outside the fast stretch | Outside, excluding first five |
| --- | ---: | ---: | ---: | ---: |
| Scheduler elapsed | 80.91 ms | 62.60 ms | 94.86 ms | 85.88 ms |
| PCIe expert gather | 57.27 ms | 40.97 ms | 69.68 ms | 61.76 ms |
| BF16 weight bytes gathered | 0.954 GB | 0.700 GB | 1.147 GB | 1.021 GB |
| GPU expert math | 6.01 ms | 6.12 ms | 5.92 ms | 6.19 ms |
| Exposed wait for CPU | 6.72 ms | 4.81 ms | 8.18 ms | 6.90 ms |
| Other target work outside expert markers | 6.76 ms | 6.63 ms | 6.86 ms | 6.82 ms |
| CPU service, mostly concurrent | 63.07 ms | 45.07 ms | 76.77 ms | 67.75 ms |

The table averages two profiled runs. PCIe gather accounts for **28.71 of the 32.26 ms** difference between the fast stretch and the remaining passes. It accounts for **20.79 of 23.28 ms** even after removing the first five passes. GPU expert math was nearly constant. The first five passes gathered about 2.08 GB per pass, so startup and routing changes within the answer both matter. Across the complete eight-position trace, the GPU gathered 1.47 GB per pass and spent 89.96 ms on that stage, with 10.17 ms of exposed CPU wait.

The full four-position run needs about 12 ms less per pass at its measured acceptance to reach 50 tok/s. An ideal 25% reduction of its 57.27 ms gather span would save about 14.3 ms if bandwidth and overlap held. CPU service is already comparable in length to the GPU branch, so CPU wait can grow after a faster gather. The measured priorities are fewer cold fetches and less host weight traffic across both branches. These results motivate a paired lossless transport test with the CPU branch still active. [Raw timings and tokens](results/native50/pass-audit.json) · [Timing summary](results/native50/pass-audit-summary.json) · [Watchdog summary](results/native50/pass-audit-watch-summary.json)

In both four-position traces, the 12 highest-traffic layers other than layer 0 carried 35.74% of the fetched weight bytes and about 36.3% of gather time. A 25% reduction in those layers' gather time corresponds to only 5.2 ms per pass before considering CPU overlap. A partial rollout can validate the codec and its exact output but has limited headroom. The recorded layer IDs, ordered by bytes, were 2, 1, 3, 4, 11, 5, 10, 39, 6, 23, 22, and 34.

## What the candidate audit measures

For every eight-position pass, the audit copies the draft's seven ranked candidate lists, selected draft IDs, target decisions, and the accepted length. First-rejection recall counts whether Qwen's correction was present in the list at the first rejected position. Accepted prefixes and corrections are checked against the response's raw committed IDs.

For later positions, the analysis takes the actual future continuation from the completed request. Target logits for rows after a wrong draft would correspond to the wrong prefix. The fixed-pool oracle follows the actual continuation as long as every next token appears in the saved candidate list. It adds the target's next token, subject to the actual request limit. This is a ceiling on the observed candidate pools and pass boundaries. Choosing different candidates can change boundaries, target arithmetic and expert traffic, so the ceiling is not a speed measurement or a guarantee about a newly trained selector.

| Prompt | Qwen correction in the top 16 at first rejection | Live IDs/pass | Fixed-pool oracle IDs/pass | Oracle at ordinary pass time |
| --- | ---: | ---: | ---: | ---: |
| Weather | 38/41 | 4.64 | 6.82 | 56.29 tok/s |
| Arithmetic | 21/24 | 6.02 | 7.58 | 52.37 tok/s |
| Coding | 89/95 | 5.25 | 7.29 | 58.62 tok/s |
| Shift cover | 101/120 | 4.69 | 6.45 | 50.59 tok/s |

The correction was present in **249 of 280** observed first rejections. All four audit outputs matched their ordinary eight-position references. Weather and coding used the same caps as the previous experiments. Arithmetic and chat ended naturally. The calculation uses the ordinary benchmark's per-pass time and assumes the captured candidate pools and target continuation stay fixed. The weather threshold is 6.06 committed IDs per pass at the 121.13 ms ordinary time. The selector would need to recover about 65% of the observed oracle gap by itself. The chat case has less slack. [Candidate lists, verified decisions and pass counts](results/native50/pass-audit.json)

## Selector training pilot

The [selector patch](freetoken_selector_calibration.patch) exposes the existing DFlash2 unary scores and low-rank transition scores under the verified predecessor. Training examples stop at the first rejected proposal. Rows whose target token falls outside the shortlist carry no selector label. The candidate pool, draft backbone and Qwen remain fixed.

The [CPU trainer](selector_calibration.py) fits one unary-score scale per draft position, with or without a bias for each unary rank. There are at most 119 learned scalars. The pilot uses 12 synthetic training prompts and four separate calibration validation prompts. The four original benchmark prompts stay outside both sets. The live comparison uses the same eight-position graph and fused CPU kernel on each side. Both sides include the score adjustment operation. The identity settings use scale 1 and bias 0.

The development capture produced 1,536 output IDs in 48 seconds of requests. It yielded 1,043 valid training rows and 346 validation rows with the target in the candidate pool. The unadjusted score reconstruction reproduced every native choice on those rows. The scale-only fit won validation accuracy. It fixed two choices and preserved every previously correct choice in the 346-row validation split. The larger scale-and-rank fit lost two net correct choices there. These are scores at prefixes visited by the original selector. A different choice in a live rollout changes the next proposal context and can change expert traffic, so the complete paired run is the decision criterion.

| Benchmark prompt | Stock → fitted decode | Stock → fitted IDs/pass | Stock → fitted request | First different ID |
| --- | ---: | ---: | ---: | ---: |
| Weather | 37.94 → 36.82 tok/s | 4.64 → 4.55 | 8.15 → 8.35 s | 201 |
| Arithmetic | 41.12 → 39.12 tok/s | 6.02 → 5.41 | 8.98 → 9.76 s | 247 |
| Coding | 42.21 → 41.00 tok/s | 5.25 → 5.08 | 19.19 → 19.72 s | 116 |
| Shift cover | 36.80 → 36.58 tok/s | 4.69 → 5.00 | 23.36 → 28.12 s | 500 |

Weather used two alternating streamed pairs. Coding used one streamed pair, and arithmetic and shift cover used one non-streamed pair each. Every identity-control response matched the earlier eight-position fused CPU runner through its cap or stop. The fitted response differed in all five pairs. Both arithmetic responses gave 323, and both shift-cover responses accepted the offer and offered Friday in two sentences. The code hit the cap during thinking, which leaves its final quality unevaluated. The shift-cover result also shows why the benchmark targets tokens per second. Higher acceptance came with slower verification passes and a longer answer.

The fitted selector is kept as a reproducible experiment. All four measured decode rates fell, so Studio uses the stock selector. The audit's headroom still motivates a contextual selector trained against accepted prefix length on diverse prompts. The weight-traffic measurements give a more direct runtime optimization to implement first. [Live results and fitted parameters](results/native50/selector-calibration.json) · [Watchdog summary](results/native50/selector-calibration-watch-summary.json)

## Reproduce

Start from the isolated fused reduction source described in [the previous reproduction](WIDTH_AUTOTUNE.md#reproduce). Restore its original compiled library in the new copy before rebuilding, since the build script links that library as the frozen base. Use the same pinned Python and model paths.

```sh
cd "$FT_SOURCE"
patch -p1 < "$REPO/edge35b/freetoken_expert_timing_native.patch"
patch -p1 < "$REPO/edge35b/freetoken_pass_diagnostics.patch"
bash "$REPO/edge35b/build_freetoken_bf16_pairs.sh" "${FT%/*}/python" "$FT_SOURCE"
cd "$REPO"
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
taskset -c 2-3 "$PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out /tmp/pass-audit-watch.json \
  --monitor-command taskset -c 8-15 python3 -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials /tmp/pass-audit.cfg --pass-diagnostics \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
  --server-log /tmp/pass-audit-server.log --out /tmp/pass-audit.json

python3 -m edge35b.analyze_pass_diagnostics /tmp/pass-audit.json \
  --frozen edge35b/results/native50/width-pair-retest.json --out /tmp/pass-audit-summary.json
```

For the selector pilot, apply `freetoken_selector_calibration.patch` on top of the runtime patch in a separate source copy. Run the same benchmark command with `--selector-calibration` in place of `--pass-diagnostics` and use fresh policy, result, and server log paths. The benchmark first captures the development scores, fits the calibration on CPU, and evaluates the selected parameters with live target verification. The output JSON records the fitted parameters and offline scores. The policy's sibling `.details.jsonl` file contains the development score captures.

## Lossless expert fetch trial

The [transfer kernel](freetoken_packed_transfer.py) consumes the existing GPU miss plan and restores the original BF16 bits into the planned cache slots. CPU overflow still reads the original banks and the target uses the same expert GEMMs. Only the selected 12 layers use the extra pinned source. Each row has a 16-value exponent window and stores original values for exceptions. Rows with more than 1% exceptions use their raw BF16 source. The [runtime patch](freetoken_packed_bf16_transfer.patch) captures separate raw and packed four-position graphs, each with a separate timed variant.

The CPU audit checked all 6,144 projection rows in the selected layers. **5,930** were packed and reconstructed bit for bit, and **214** used the raw fallback. Their combined payload is **24.05% smaller** than the raw payload, including the fallback rows. The compact banks require **13.04 GiB** in addition to the original host banks. Layer 0 had 508 fallback rows out of 512 and remains raw in the trial. The audit took 70.7 seconds on one CPU core and reached 58.25 °C. [Checkpoint row audit](results/native50/bf16-expert-row-audit.json)

The mixed-row copy and repair fixture passed in the Triton CPU interpreter. Both kernels compiled for `sm_89`. The initial live preflight stopped on vicena-0's `7.0.0-42-generic` kernel because the NVIDIA module was unavailable. That four-position trial was not timed. The workstation was subsequently restored and the [eight-position transfer and cache experiments](EXPERT_TRANSPORT8.md) ran the CUDA fixture and compared actual generation against the frozen eight-position control.

Apply the transfer patch in a fresh copy of the pass-audit FreeToken source, with its fused CPU reduction and timing patches already present. Use the same pinned Qwen and DFlash2 paths and the [256-token Studio suite](NATIVE50.md#the-256-token-studio-comparison-and-decode-cache-policy). Set `FT_SOURCE` to the patched Python source directory and start from the Lightreason checkout. The commands assume an idle GPU with a working driver.

```sh
patch -p1 -d "$FT_SOURCE" < edge35b/freetoken_packed_bf16_transfer.patch
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
FT_PYTHON="${FT%/*}/python"
trial=$(mktemp -d /tmp/lightreason-packed-bf16.XXXXXX)

taskset -c 8 "$FT_PYTHON" -m edge35b.audit_bf16_expert_transport \
  --model "$MODEL" --out "$trial/codec-audit.json"
TRITON_INTERPRET=1 "$FT_PYTHON" -m unittest -v edge35b.test_freetoken_packed_transfer

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 120 --out "$trial/cuda-check.json" \
  --monitor-command taskset -c 8-15 env LIGHTREASON_TEST_PACKED_CUDA=1 \
  "$FT_PYTHON" -m unittest -v edge35b.test_freetoken_packed_transfer

taskset -c 2-3 "$FT_PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out "$trial/watch.json" \
  --monitor-command taskset -c 8-15 "$FT_PYTHON" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials "$trial/width.cfg" --packed-transfer \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
  --server-log "$trial/server.log" --out "$trial/transport.json"
```

The harness warms all four graphs before alternating raw and packed requests in streaming and non-streaming modes. It records output IDs, stop reasons, commits per pass, first streamed text, request time and decode rate. Profiled weather pairs measure GPU gather and the exposed CPU wait separately from ordinary speed runs.
