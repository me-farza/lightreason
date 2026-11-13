# Real verifier widths and paired BF16 CPU experts

This experiment keeps the pinned Qwen3.6-35B-A3B BF16 target, the seven-proposal DFlash2 checkpoint, the 15 GiB GPU expert cache, and the same greedy prompts and stop settings. Every accepted token comes from the target. The fixed eight-position verifier with CPU pairing disabled is the direct control. The frozen Scheduled and Packed Replay results are separate comparisons.

The [verifier patch](freetoken_width_autotune.patch) captures actual 4-, 6-, and 8-position target graphs. The additional graphs have separate attention capture buffers, recurrent and convolution scratch, and graph memory pools. They share the same weights and persistent per-request model state. The scheduler allocates and releases KV slots for the chosen width and commits only the matching prefix. The proposer still runs its complete eight-position block, including the pending token, and the target receives the selected prefix of that block. This isolates changes to target verification. Both the unused proposals and the controller overhead stay inside the timed request.

The [width controller](width_autotune.py) supports fixed widths, a threshold rule, and a table fitted to development runs. The table picks the width with the best observed ratio of committed IDs to time for four states of the preceding pass. It uses prior acceptance available on the host, so it adds no GPU readback. Fixed-width development passes provide the training data. The benchmark keeps the resulting table fixed for the held-out prompts.

The [CPU patch](freetoken_bf16_pairs.patch) groups two positions routed to the same expert. The paired AVX2 dot loads and widens each BF16 weight vector once. It keeps separate FP32 accumulators for the two positions, with the same lane and reduction order as FreeToken. It applies SiLU and BF16 intermediate rounding per route and reduces each token's output in its original router order. A per-request threshold chooses the minimum number of pairs before using the paired path. The implementation is scoped to pinned BF16 Qwen experts with hidden size 2,048, inner size 512, and eight routes per token. The native extension links the unmodified GPU and alternate-format symbols from the pinned library. Its isolated Qwen build omits the optional disk-locked fetch binding.

## Isolated CPU measurements

Eight CPU workers ran 30 alternating trials per fixture using real expert weights from layer 13. Every output hash matched the frozen stock extension, including the multiple-route fixture. The median comparisons below use the same rebuilt extension with pairing disabled or enabled at one pair, so the build and worker implementation are controlled.

| Route fixture | Pairing off | Pairing enabled |
| --- | ---: | ---: |
| Eight distinct expert assignments | 0.797 ms | 0.795 ms |
| Two pairs | 0.372 ms | 0.363 ms |
| Four pairs | 0.220 ms | 0.194 ms |
| Eight pairs | 1.595 ms | 0.842 ms |

These timings isolate the CPU and intentionally vary repetition. In complete inference, GPU transfers run concurrently with the CPU and the layer waits for both. A large gain on the eight-pair fixture gives limited information about ordinary layers with fewer repeated assignments. [Frozen CPU fixtures](results/native50/cpu-pairs-stock.json) · [Paired CPU fixtures](results/native50/cpu-pairs-grouped.json)

The [second CPU patch](freetoken_bf16_pairs_reduce.patch) assigns each down-projection output stripe to one worker for all tokens. The worker immediately reduces its stripe in original router order. This removes the extra shared scratch round trip and the barrier between down projection and reduction. Its hashes also matched the frozen native module on all four fixtures. With pairing on, the two-, four-, and eight-pair fixtures improved to 0.331, 0.173, and 0.812 ms. [Fused reduction CPU fixtures](results/native50/cpu-pairs-reduce.json)

## Full-model width and pairing sweep

| Weather configuration | Decode tok/s | IDs/pass | ms/pass | Complete HTTP request | First different ID vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed eight positions | 37.80 | 4.64 | 122.67 | 8.17 s | None |
| Fixed four positions | 40.81 | 3.45 | 84.57 | 7.68 s | 31 |
| Fixed six positions | 38.72 | 4.05 | 104.55 | 8.01 s | 201 |
| Eight positions with pairing at one pair | 37.36 | 4.64 | 124.10 | 8.25 s | None |
| Eight positions with pairing at three pairs | 37.84 | 4.64 | 122.51 | 8.17 s | None |
| Width rule | 39.72 | 4.25 | 107.00 | 7.85 s | 102 |
| Fitted width table | 42.20 | 3.45 | 81.67 | 7.47 s | 31 |
| Fitted width table and pairing at three pairs | 42.41 | 3.45 | 81.25 | 7.44 s | 31 |

Weather rows are medians of two streamed runs with reversed order within each sweep. All produced 256 IDs. First streamed text was about 1.42 s. Model load and graph setup took 161 s separately. After the extra four- and six-position graphs captured, 2.19 GiB of GPU memory was free versus 2.27 GiB after capturing eight positions alone. The first fixed-four run spent about 665 ms in its initial pass, compared with 224 ms in the repeated fixed-four run. The repeated fixed-four request ran at **42.37 tok/s** with the same IDs as its first run. This limits what the apparent median advantage of the learned policy establishes.

The fitted table chose six positions at the start and four thereafter on each evaluated request. The controller's “middle” state cannot occur immediately after a four-position pass, so its trained choice of eight positions for that state did not get exercised. The stronger result is the measured reduction from about 123 to 81 ms per pass with narrower verification. This table does not establish a benefit for switching during a request over a warmed fixed-four graph.

| Held-out prompt | Fixed eight decode | CPU pairing decode | Fitted width decode | Combined decode | Combined first different ID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic, streaming | 40.92 | 40.99 | 42.06 | 42.34 | 34 |
| First duplicate, non-streaming | 41.54 | 41.67 | 41.04 | 41.04 | 133 |
| Shift cover, streaming | 35.93 | 36.22 | 40.77 | 40.83 | 187 |

Each held-out row is one run per configuration. Pairing alone matched every output ID and stop reason across the weather repeats and all three held-out prompts. The combined run matched the fitted-width sequence on every prompt. Both arithmetic runs answered 323, and both chat variants accepted the offer in two sentences and offered to cover Friday. The coding runs reached the cap inside the thinking section, which leaves the final answer unevaluated. The combined arithmetic request took 9.40 s versus 9.02 s with the control because it used 372 versus 344 IDs. Chat went from 23.91 to 14.36 s as output shortened from 827 to 550 IDs. Strict matching and those task checks measure different properties.

At the combined weather acceptance of 3.45 IDs per pass, 50 tok/s needs about **69 ms/pass**, roughly **12 ms** below the observed time. The exact eight-position control needs about 93 ms/pass at 4.64 IDs per pass. The previous token-34 diagnosis showed that different batched arithmetic can move a close greedy decision. The new narrow runs report the mismatches and do not establish their individual causes or general quality equivalence. [Raw timing, passes and IDs](results/native50/width-trials.json) · [Watchdog summary](results/native50/width-trials-watch-summary.json)

In the first combined weather run, the fastest contiguous 32-pass window committed 105 IDs at 52.69 tok/s by scheduler pass timestamps. It averaged 62.3 ms and 3.28 IDs per pass. The remaining passes averaged 95.0 ms and 3.57 IDs. This selected window shows that pass cost varied more than acceptance on this request. The complete request remains the benchmark. A breakdown of weight movement, CPU wait and GPU work on the slow passes would identify which cost to reduce.

## Full-model retest of the fused CPU reduction

| Prompt | Fixed eight decode | Fused CPU decode | Fixed eight request | Fused CPU request | Exact pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weather | 37.77 tok/s | 38.37 tok/s | 8.18 s | 8.07 s | 2/2 |
| Arithmetic | 40.85 tok/s | 41.58 tok/s | 9.03 s | 8.88 s | 2/2 |
| First duplicate | 41.67 tok/s | 42.26 tok/s | 19.42 s | 19.17 s | 2/2 |
| Shift cover | 36.23 tok/s | 36.82 tok/s | 23.72 s | 23.35 s | 2/2 |

The median paired rate gain was 1.62%. Weather used two streamed runs per arm. The other prompts used both streaming modes and reversed arm order. The one-pair threshold won on the development prompt and was kept fixed for the held-out runs. Every ID and stop reason matched the fixed eight-position control in the same trial. The resulting sequences also matched the fixed eight-position runs from the first sweep. First-token scheduler times were effectively unchanged. Setup took 158 s outside the timing window. The first and second full sweeps peaked at 50 °C and 43 °C on the GPU and 71.25 °C and 70.25 °C on the CPU. [Repeated exact comparisons](results/native50/width-pair-retest.json) · [Watchdog summary](results/native50/width-pair-retest-watch-summary.json)

The fused CPU reduction is the validated exact improvement in these experiments. The width and CPU combination used the first CPU implementation, and the fused CPU retest held width at eight. The best median decode rate of either complete weather configuration was 42.41 tok/s. At the fused CPU weather rate the verifier averaged 120.8 ms per pass, about 28 ms above the 50 tok/s budget at its observed acceptance. The adaptive configuration reduced that gap to about 12 ms and produced different output IDs.

The [follow-up pass diagnostics](PASS_DIAGNOSTICS.md) profile the slow passes, benchmark the fused reduction at four positions, and measure candidate recall and selector training.

## Reproduce

Use the selected-prefill, token-audit source from the [previous BF16 experiments](NATIVE50.md#the-256-token-studio-comparison-and-decode-cache-policy). Build in a fresh copy with the pinned FreeToken Python environment. The CPU extension build uses the FreeToken wheel's existing GPU kernels and the installed CUDA runtime headers.

```sh
cd "$FT_SOURCE"
patch -p1 < "$REPO/edge35b/freetoken_bf16_pairs.patch"
patch -p1 < "$REPO/edge35b/freetoken_width_autotune.patch"
bash "$REPO/edge35b/build_freetoken_bf16_pairs.sh" "${FT%/*}/python" "$FT_SOURCE"
cd "$REPO"
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
taskset -c 8-15 "${FT%/*}/python" -m edge35b.benchmark_cpu_bf16_pairs \
  --model "$MODEL" --out /tmp/cpu-pairs-grouped.json
taskset -c 2-3 "$PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1350 --out /tmp/width-trials-watch.json \
  --monitor-command taskset -c 8-15 python3 -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --width-trials /tmp/width-trials.cfg \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
  --server-log /tmp/width-trials-server.log --out /tmp/width-trials.json
```

Run the CPU benchmark with the frozen selected-prefill source on `PYTHONPATH` to obtain the stock output hashes. The GPU harness checks that each request starts with zero cached input IDs and that the real graph widths and committed output agree with the server counters. The cache metadata resets between requests so each arm starts with the same empty expert cache. No prompt or answer is precomputed.

Apply `freetoken_bf16_pairs_reduce.patch` on top of the paired CPU patch before building to reproduce the fused reduction version. Add `--width-pair-retest` to the GPU command to keep verification fixed at eight positions and repeat the CPU comparisons in alternating streaming modes. Use fresh output, policy, and log paths.

## Search for a policy with measured feedback

`freetoken_bench --evolve` keeps the fixed eight-position, pairing-off runner as its control. It measures changes to the verifier width for each prior-pass state and the minimum CPU pair count. It tries fixed widths and pairing first, then mutates the best eligible policies. Each trial pairs the candidate with fresh control requests and reverses runner order across repeats in both HTTP modes. The configured expert cache and original BF16 target stay fixed. The benchmark verifies the actual verifier width on every pass.

A candidate must reproduce every control token ID and stop reason, improve median paired decode rate by at least 0.5%, and have a median HTTP completion time no slower than the control on each prompt. A repeated control with differing outputs fails this gate. Search sees only the development prompts. One winner goes to the held-out prompts, where it must pass the same checks. The raw results also retain the comparison with Packed Replay.

Use the fused CPU reduction build from above. Add `--evolve --evolution-trials 8` to the width trial command and give the run fresh output paths. Allow a longer watchdog command timeout for the extra paired requests. With the four cases above, Studio weather is development and arithmetic, code, and chat are held out. To use more development prompts, put them first in `--cases` and set `--evolution-development-cases`.

An external proposer can supply bounded configurations with `--evolution-proposals`. The file contains a JSON array of objects with `width_by_state` (keys `first`, `short`, `middle`, `long` and values 4, 6, or 8) and `min_pairs` (0, 1, or 3). Proposed settings go through the same measurements as mutations.

Apply the evaluator to existing trials on a CPU:

```sh
python3 -m edge35b.evolve_policy edge35b/results/native50/width-pair-retest.json
python3 -m edge35b.evolve_policy edge35b/results/native50/width-trials.json
```

The fused reduction archive selects one-pair CPU work at width eight. It matches all six held-out pairs and improves median paired decode rate by 1.62% there. In the earlier width sweep, the combined policy improved the development decode rate by 12.21% and differed at token 31. It fails the exact-output gate. A fresh CPU test on checkpoint weights also reproduced identical BF16 result hashes across four fixtures. The eight-pair fixture fell from 1.58 to 0.81 ms. [Recorded request evaluations and fresh CPU timings](results/native50/evolution-bootstrap.json)

## Reuse FP32 conversions in paired expert work

The CPU candidate expands each paired input to FP32 once per token. It expands the intermediate once per route, after the existing BF16 rounding. The dot products use the same accumulation and reduction order. A second change lets each worker claim two or four output blocks at once.

The CPU search timed real Qwen checkpoint weights on the Ryzen 9 5950X with eight workers. It used synthetic activations and deterministic route layouts with counts taken from the four- and eight-position GPU traces. Forty paired repeats alternated arm order. The initial selection used earlier passes and layers 0–19. Later passes and layers 20–39 supplied the first held-out set. The cutoff retest used the previously measured fixtures for tuning and reserved different combinations of passes, layers, and route shapes for validation.

| CPU candidate | Held-out four-position median | Held-out eight-position median | Overall held-out median |
| --- | ---: | ---: | ---: |
| Reuse both conversions and claim two blocks | 1.180× | 0.988× | 1.055× |
| Enable that kernel at five CPU routes or fewer | 0.999× | 1.007× | 1.000× |

Every measured output hash matched the frozen fused CPU build. The cutoff's repeated eight-position fixture gained 1.377×, but the cutoff failed the selection gate. The eight-position live runner keeps its existing settings. Sampled CPU temperatures peaked at 51.5 °C with a 75 °C cutoff. These are isolated CPU timings. [Paired timings, shapes, output hashes, and telemetry](results/native50/cpu-pair-tuning-v2.json)

To reproduce the first search, apply `freetoken_bf16_pair_tuning.patch` on a copy of the fused reduction source and rebuild it as described above. Point `FT_SOURCE` at that copy, then run:

```sh
PYTHONPATH="$PWD:$FT_SOURCE" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  taskset -c 8-15 "${FT%/*}/python" -m edge35b.benchmark_cpu_pair_tuning \
  --model "$MODEL" --profile edge35b/results/native50/pass-audit.json \
  --repeats 40 --out /tmp/paired-tuning.json
```

For the cutoff run, apply `freetoken_bf16_pair_route_gate.patch` before rebuilding and add `--route-gate-from /tmp/paired-tuning.json` with a fresh output path. Run either benchmark with `--frozen-control` on the original fused build to verify the native output hashes.

For the full model, apply `freetoken_cpu_pair_tuning_retest.patch` to the isolated candidate source. Use the supervised width benchmark command above with `--cpu-pair-kernel 3,2,64 --cpu-pair-width 4` for the ungated candidate, or `--cpu-pair-kernel 3,2,5 --cpu-pair-width 8` for the cutoff candidate. Keep a fresh width configuration and output path for each test. Each arm uses the same physical verifier width and one-pair threshold. The report compares full requests and checks every token ID and stop reason against that width's fused CPU control. Live testing awaits a working NVIDIA driver.
