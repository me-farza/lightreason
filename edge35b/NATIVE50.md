# Qwen3.6 BF16 native runtime experiments

The target is 50 committed output tokens/s for one request on the same RTX 4090, a budget of 20 ms per committed token. The frozen Studio result for Lightreason Packed is 12.95 tokens/s on the interactive run and 12.85 tokens/s on the 341-token arithmetic answer. The arithmetic run took 28.39 seconds from prefill through EOS. The Scheduled runner remains the original control. The new runners are separate, opt-in implementations.

## Test setup and weight audit

- Target: [Qwen/Qwen3.6-35B-A3B at `345d467b117d1fd5eda9b879a5988de9ae5f47a7`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/345d467b117d1fd5eda9b879a5988de9ae5f47a7), text only. 34,660,610,688 BF16 parameters; 4.56 GiB of shared, attention, routing, embedding and output weights on CUDA; 60 GiB of routed experts backed by host memory. The target is unquantized.
- Workstation: AMD Ryzen 9 5950X, 121.5 GiB OS-visible RAM, RTX 4090 with 24 GiB VRAM, PCIe 4.0 ×16 under load, driver 580.173.02. Linux ran the benchmark on eight physical CPU cores (`taskset -c 8-15`). DIMM population and memory clock were not readable from the unprivileged account. CUDA 12.8, PyTorch 2.9.1, Transformers 5.7.0 for Packed. The stock-limit watchdog stops at 70 °C GPU or 75 °C CPU. The native ablations peaked at 53 °C GPU and 73.75 °C CPU; no hardware settings changed. [Hardware inventory](results/native50/hardware.txt) · [Watchdog summary](results/native50/thermal-summary.json)
- Greedy, batch size one, thinking enabled, original chat template and EOS IDs 248044 and 248046. Each run begins with empty model and expert caches and warmed code unless stated otherwise. The timing runner clears the previous request's expert LRU. The kernel warmup, checkpoint load, and audits are outside generation time. Generation wall time begins at prefill and ends after the last token and GPU synchronization. Decode rate is `(committed IDs − 1) / (last − first)` and includes EOS. Studio records request-to-first-token separately; a callback that observes token IDs is lighter than HTTP streaming.
- Studio Packed uses a 14 GiB GPU expert LRU. It seeds the cache at prefill, runs cold experts through AVX2 from the mapped BF16 tensors, and admits repeated experts within a four-decode-step window. It groups the GPU experts into Triton launches and uses a fused Gated DeltaNet update. Packed changes floating-point evaluation order compared with Scheduled. The checkpoint audit compares live text model parameters with the official file bytes before and after inference. The additional resident audit compares every expert row still in the CUDA LRU with its BF16 host source. It checked 2,389 rows (7,515,144,192 BF16 elements) in the first native ablation. Neither audit certifies the bytes of every transient activation or evicted row.

## Where time goes

Nsight Systems capture · [Extracted timing](results/native50/nsys-summary.json). The instrumented math run took 97.45 ms/token over 79 decode steps; ordinary Packed runs are around 77–82 ms/token. The trace is useful for locating work. The instrumentation changes the elapsed time.

| Decode work | Instrumented ms/token | Interpretation |
| --- | ---: | --- |
| GPU kernels | 11.80 | Union on the GPU timeline; expert matvec kernels sum to 1.92 ms. |
| GPU copies | 9.57 | About 110 MiB H2D, 30 MiB D2D and 0.16 MB D2H per token. |
| CPU experts | 16.94 | Just 0.23 ms overlapped GPU kernels or copies. |
| Explicit route/hidden fences | 0.20 | CPU scope; the required work can also wait elsewhere. |
| Host launch API | 13.85 | Can overlap GPU work. |
| Host weight copy calls | 11.84 | Inclusive; do not add to device copy time. |
| No GPU work or instrumented CPU expert calls | 59.36 | Includes host routing and scheduling, driver activity, and profiler overhead. It is not a measure of removable time. |

Prefill for this prompt took 2.16 seconds on the trace, including 1.22 seconds of GPU copies and 97.6 ms of GPU kernels. It transferred about 15.02 GB H2D. The largest individual decode kernel groups were cuBLAS BF16 GEMV variants and the Triton expert matvec; much of the rest consisted of short elementwise and reduction kernels. [Kernel breakdown](results/native50/kernel-breakdown.txt) · [Nsight Compute log](results/native50/ncu-200701.txt). Nsight Compute could not read hardware counters under the current driver permissions. The custom kernel was evaluated in ordinary full-generation runs. See [NVIDIA's explanation of the counter permission](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Paired native ablations

Two fixed development prompts, two repeats each, 64-token cap. Generation ran sequentially with a token observer and no profiler. See [the full timing and ID records](results/native50/ablations-200701.json).

| Runner | Median decode tok/s | Median first token | Median request | Exact IDs and stop vs Packed |
| --- | ---: | ---: | ---: | ---: |
| Packed control | 12.76 | 2.62 s | 7.56 s | Reference |
| Persistent buffers | 12.96 | 2.57 s | 7.43 s | 4/4 |
| Buffers + hot expert and combine graph replay | 13.39 | 2.56 s | 7.27 s | 4/4 |
| Add compiled route planning and a C++ worker pool | 13.18 | 2.67 s | 7.47 s | 4/4 |
| Start native CPU work before hot CUDA work | 13.21 | 2.63 s | 7.40 s | 4/4 |
| Coalesce each missed expert's two H2D weight copies | 13.37 | 2.62 s | 7.34 s | 4/4 |
| Add Ada CUDA gate/up projection and SiLU fusion | 13.41 | 2.63 s | 7.33 s | 4/4 |

The CPU workers use the same AVX2 arithmetic and preserve the LRU policy. The `sm_89` CUDA experiment keeps BF16 weights and the same accumulation order across experts; it did not show a meaningful gain over graph replay by itself. Replaying all 30 single-token Gated DeltaNet mixers as well as the expert operations raised the initial 48-token math trial from 12.52 to 16.53 tok/s with identical output. It preserves the original attention path and copies the convolution and recurrent cache into and out of each captured mixer. Python still schedules CPU work between graph launches. [Mixer trial](results/native50/mixer-201800.json)

Across the repeated long-prefix and complete-answer suite, **Packed Replay reached 16.66 tok/s versus 12.94 tok/s for Packed**, and all eight paired runs matched every output ID and stop condition. Median paired request time improved by **1.26×**. The arithmetic prompt stopped naturally in both repeats; the other three reached their initial caps. [Repeated runs and runtime audit](results/native50/heldout-201800.json)

| Prompt | Output and stop | Decode tok/s, Packed → Replay | First token, Packed → Replay | Request, Packed → Replay |
| --- | --- | ---: | ---: | ---: |
| Arithmetic | 341, EOS | 12.71 → 16.32 | 1.98 s → 2.00 s | 28.74 s → 22.83 s |
| Held-out reasoning | 512, cap | 12.56 → 16.02 | 3.33 s → 3.27 s | 44.03 s → 35.16 s |
| Held-out coding | 768, cap | 13.24 → 17.18 | 3.10 s → 3.14 s | 61.03 s → 47.78 s |
| Held-out chat | 512, cap | 13.16 → 17.02 | 2.84 s → 2.91 s | 41.66 s → 32.94 s |

The replay runner takes about 60 ms per committed token on the median run. The 20 ms budget needs about another 40 ms removed from each decode step, or enough verified speculation to amortize the model and expert work. Prefill did not speed up because the new mixer graphs apply only to single-token decode.

The non-streaming follow-up used a 1,536-token cap and no token-observation callback. It matched all three held-out pairs. Reasoning reached **1,200 IDs and EOS** (102.19 → 81.20 seconds; 12.14 → 15.39 tok/s). Chat reached **576 IDs and EOS** (46.18 → 36.38 seconds). Coding matched through the full **1,536-token cap** (120.05 → 93.69 seconds; 13.13 → 16.95 tok/s). Median paired request speedup was 1.27×. These runs used the same checkpoint bytes backed by disk, after the startup audit. The first reasoning control incurred 32 major page faults; the other timed requests had none. [Non-streaming results](results/native50/eos-205420.json)

## Token 34 and speculation timing

[Identical-prefix isolation and state comparisons](results/native50/diag-v2-201800.json). The real DFlash run and the serial reference agree on the first 33 output IDs. Both consume the same prefix for the disputed decision. Serial Hybrid chose ID 30790 at logit 24.75 with a 0.25 lead. The four-position DFlash verifier put IDs 30790 and 8618 at 24.625 and selected 8618. The largest absolute logit difference was 1.09375. These are measured differences, including a change of greedy token.

One-position verification on clean serial state kept 30790. A four-position GPU verifier fed the reference continuation also produced the tie from clean serial state, without any previous speculative rollback. The first layer's input normalization matched bitwise; the very next batched QKV projection changed seven BF16 elements by up to 0.00390625, and the gate projection changed two. That precedes recurrent updates and expert routing. Downstream routing sets then diverged. The four-position hybrid verifier on that same clean cache kept 30790, showing that placement and the block's starting state matter. The real DFlash block entered with cache values that already differed from serial replay, despite the same tokens and cache length. This isolates a shape-dependent arithmetic path and its propagation; it does not establish serial numerical equivalence for the hybrid verifier.

Forced first and middle suffix rejections tested attention keys and values, Gated DeltaNet recurrent state, and convolution history. With GPU expert placement held fixed, trimming to the selected prefix matched the corresponding same-width reference block bitwise in all four cache families. The cropped state was still different from a clean sequential replay. The hybrid planner can change CPU/GPU assignment based on all positions in the candidate block, so a different suffix can also change the arithmetic used for earlier positions. The existing DFlash verifier remains a separate experiment and is not enabled in Studio.

The earlier **18.50 tok/s** coding result used the real [DFlash proposer](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/tree/f181eece646affea2c38b2765f1aaa01a9734ccd). In 2.541 seconds of decoding it committed 47 new IDs after the first token, accepted 32 of 43 draft IDs, and evaluated 58 target positions in 15 cycles. The time includes the proposer, rejected work, expert CPU/GPU transfers, target verification, cache commitment, and token observation. It excludes model loading, warmup, prefill (3.100 s), and HTTP streaming. The complete request took 5.641 s. This was not the oracle test. On the complete arithmetic answer DFlash reached 18.29 tok/s, matching 341 IDs and native EOS on that case. It committed 3.505 tokens per cycle at 191.7 ms/cycle. At the same acceptance, 50 tok/s would require about 70.1 ms/cycle. [Raw earlier timings](results/target50/hybrid-verify.json)

That DFlash arithmetic run transferred 77.0 GB of expert weights after prefill and evaluated 19,345 cold expert rows on the CPU. At the same traffic and acceptance, reaching 50 tok/s requires roughly 11.3 GB/s of expert H2D transfers and 17.9 GB/s of CPU expert weight reads concurrently, before activation work, target computation, or scheduling. The cycle needs less traffic, more overlap, or more accepted tokens. Removing the profiled 1.92 ms of Packed hot expert matvec alone cannot close the gap.

The follow-up changed the actual target input width for the real proposer and measured the complete decode cycle. Each grouped expert kernel still pads its internal row tile to 16 positions; target sequence width, route count, cache traffic and committed tokens vary. All runs used a 48-token cap. [Width sweep and output IDs](results/native50/verify-width-205420.json)

| Positions | Logic: tok/s; tokens/cycle; ms/cycle | Coding: tok/s; tokens/cycle; ms/cycle | Exact pairs vs Packed |
| ---: | --- | --- | ---: |
| 2 | 15.02; 1.81; 120.32 | 16.06; 1.88; 117.07 | 2/2 |
| 4 | 16.10; 3.13; 194.56 | 18.08; 3.13; 173.27 | 1/2 |
| 6 | 17.34; 4.27; 246.39 | 16.79; 3.62; 215.31 | 2/2 |
| 8 | 15.01; 4.27; 284.58 | 17.20; 4.27; 248.47 | 1/2 |

Logic first differed at token 34 for widths 4 and 8. All widths matched the capped coding output. On logic, widening from six to eight did not increase committed tokens per cycle; H2D weight traffic rose from 1.18 to 1.50 GB per cycle and CPU expert work from 254 to 273 rows per cycle. A short exact pair does not qualify a speculative width for Studio.

For the four-position coding run, a separate instrumented profile measured **172.81 ms/cycle** across 15 cycles: **6.26 ms** for the draft stream span, **166.21 ms** for target verification and reading its decisions, and **0.31 ms** for commitment and observation. These spans include host-side gaps between GPU events; they are not pure GPU compute times. The target span includes expert dispatch and transfers, target arithmetic, and host synchronization to read decisions. The proposer is a small share of this critical path. [Cycle profile](results/native50/verify-profile-205420.json)

## FreeToken BF16 comparison

[FreeToken nightly](https://github.com/FlashML-org/FreeToken/releases/tag/nightly) `0.2.1+g14119b8cb` ran the same checkpoint in an isolated environment with PyTorch 2.14.1+cu130 and Triton 3.8.0. The source checkpoint and the persistent copy matched all 26 pinned SHA-256 digests. The server reported `dtype=torch.bfloat16` and its AVX2 worker reported `fmt=bf16` for all 40 MoE layers. This confirms the chosen precision and loader configuration; the FreeToken runtime's entire resident representation was not byte-audited. Expert bank creation required RAM beyond Packed's mapped checkpoint, consistent with [FreeToken's RAM requirements](https://github.com/FlashML-org/FreeToken/issues/84). After moving the staging paths to the verified disk copy, about 92 GiB was available to the loader.

The harness round-tripped the formatted input through FreeToken's tokenizer and checked all resulting prompt IDs against Packed. Both engines used greedy sampling, thinking enabled, one request, and the same caps. FreeToken used a 14 GiB unified LRU, eight CPU threads, a batch-size-one CUDA graph, pinned host expert banks, and the runtime's GPU/CPU handshake kernel. The log showed a fixed fetch cap of one miss per layer per step, with other misses on the CPU, and overlap during prefill. A representative chat decode interval saw roughly 64–71% expert cache hits. FreeToken keeps its expert cache across requests; its naive KV cache did not reuse prompt tokens in these tests. Packed clears its expert cache each request. This compares these working configurations; it does not isolate the gain from each difference.

| Prompt | Streaming decode | Non-streaming decode | Output comparison |
| --- | ---: | ---: | --- |
| Arithmetic | 29.10 tok/s | 31.86 tok/s | Both answered 323 at natural stop; reasoning differed and output lengths were 299 and 345 vs Packed's 341. |
| Reasoning | 28.86 tok/s | 29.50 tok/s | Text matched Packed through the 512-token cap in both runs. |
| Coding | 32.88 tok/s | 32.74 tok/s | Both reached the 768-token cap with different text. |
| Chat | 31.28 tok/s | 32.65 tok/s | Both reached the 512-token cap with different text. |

Median decode was **31.57 tok/s**. FreeToken's first-token scheduler times were 2.77–2.82 seconds. Complete arithmetic HTTP requests took 13.02 and 13.64 seconds, with different output lengths. Setup took 109.56 seconds before the first warmup request on the successful run. The unmodified completion API did not expose raw output token IDs, so matching text is not a strict ID match claim. Different arithmetic output lengths establish non-parity for that case; the two API measurements also had different expert cache histories. The faster external runtime stays separate from Studio's exact-output comparison. [Timings and responses](results/native50/freetoken-bf16.json) · [Runtime log](results/native50/ft-204432-server.txt)

The isolated FreeToken bandwidth probe used the actual BF16 expert shape (hidden 2,048, inner 512, top 8, eight CPU threads). CPU expert work measured **27.2 GB/s** alone and PCIe gather **26.3 GB/s** alone. Together they achieved **14.1 and 15.7 GB/s** respectively. At the DFlash arithmetic run's observed traffic of about 1.25 GB of CPU expert reads and 0.79 GB of H2D weights per cycle, the CPU portion alone would need about 89 ms at that concurrent rate. Even a perfectly balanced assignment at their 29.8 GB/s aggregate rate would consume about 69 ms per cycle before non-expert work. That is almost the entire 70.1 ms budget at the observed acceptance. These are diagnostic bounds for the measured traffic and rates; changing the amount of data moved or the acceptance can change them. [Bandwidth measurement](results/native50/ft-bw-205420.json)

A separate three-draft FreeToken MTP launch was stopped before generation when the hardware watchdog stopped receiving fresh temperature samples. The safety thresholds were unchanged, so there is no FreeToken MTP speed result. [Watchdog record](results/native50/ft-mtp-205042.txt) The earlier [native Lightreason MTP experiment](results/speed2/v4-recurrent-mtp.json) improved its then-current control from 7.38 to 8.26 tok/s on three short cases and matched all three token sequences. That older result does not characterize the current Packed runner.

## Reproduce

Run from the workspace on the GPU machine. The model and draft paths must point at the pinned checkpoints above. Use fresh paths for result files. The C++ workers and Ada replacement are only required for their corresponding ablations.

```bash
g++ -O3 -std=c++20 -mavx2 -mfma -fopenmp -pthread -shared -fPIC \
  edge35b/packed_workers.cpp -o /tmp/lightreason-packed-workers.so
gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC edge35b/cpu_experts.c -lm \
  -o /tmp/lightreason-packed-experts.so
export EDGE35B_PACKED_WORKERS=/tmp/lightreason-packed-workers.so
export EDGE35B_CPU_EXPERT_LIBRARY=/tmp/lightreason-packed-experts.so
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export MODEL=/dev/shm/lightreason-edge35b/model
export DRAFT=/tmp/lightreason-edge35b/dflash-f181eece

python hardware_exact10.py --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1000 \
  --out /tmp/packed-native-watch.json --monitor-command taskset -c 8-15 \
  python -u -m edge35b.native_bench --model-dir "$MODEL" --modes packed,buffers,graph,native,overlap,pinned,ada \
  --cases known-math,known-logic --cap 64 --repeats 2 --resident-audit --out /tmp/packed-native.json

python -m edge35b.native_bench --model-dir "$MODEL" --modes packed,graph,mixer-graph \
  --cases reasoning,coding,chat,known-math --repeats 2 --resident-audit --out /tmp/packed-replay.json
python -m edge35b.native_bench --model-dir "$MODEL" --modes packed,mixer-graph \
  --cases reasoning,coding,chat --cap 1536 --repeats 1 --no-observer --resident-audit --out /tmp/replay-eos.json
python -m edge35b.verify_diagnose --model-dir "$MODEL" --draft-dir "$DRAFT" --out /tmp/token-34.json
python -m edge35b.dflash_probe --model-dir "$MODEL" --draft-dir "$DRAFT" \
  --modes delta-packed,flash2,flash4,flash6,flash8 --hybrid-verifier --cases logic,code \
  --tokens 48 --stream --out /tmp/dflash-width.json
python -m edge35b.dflash_probe --model-dir "$MODEL" --draft-dir "$DRAFT" \
  --modes flash4 --hybrid-verifier --cases code --tokens 48 --profile-cycles --stream \
  --out /tmp/dflash-cycle-profile.json

nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --stats=false -o /tmp/packed \
  python -u -m edge35b.native_bench --model-dir "$MODEL" --modes packed \
  --cases known-math --repeats 1 --cap 80 --profile --out /tmp/packed-profile.json
nsys export --type sqlite -o /tmp/packed.sqlite /tmp/packed.nsys-rep
python -m edge35b.nsys_summary /tmp/packed.sqlite
```

To compare FreeToken with the same test requests, install the pinned wheels above in a separate Python 3.12 environment. Use the audited disk-backed checkpoint and allow enough free RAM for FreeToken's expert banks. The harness runs one local server, sends requests sequentially, and records scheduler timings and HTTP timings separately:

```bash
python -m edge35b.freetoken_bench --ft /path/to/freetoken/bin/ft \
  --model /path/to/verified/model --packed /tmp/packed-replay.json \
  --out /tmp/freetoken-bf16.json --server-log /tmp/freetoken-bf16.log
```

On the workstation with the Tailscale route to vicena-0, start Studio with the Replay runner. The Scheduled runner remains the original comparison. Launch with `--edge35b-packed` alone to choose the frozen Packed runner:

```bash
python3 demo_chat/start.py --model 35b --edge35b-packed --edge35b-packed-replay \
  --edge35b-model-path /path/to/lightreason-benchmark/model
```

The live HTTP smoke test streamed 64 incremental updates from each runner, with the Scheduled stream completing before Replay started. All 64 final IDs matched. Scheduled ran at 7.01 decode tok/s and took 11.08 seconds; Replay ran at 16.86 decode tok/s and took 5.69 seconds. This is a separate capped comparison against Scheduled, not the held-out Packed ablation. [Studio stream record](results/native50/studio-replay.json)

## Studio greeting follow-up

The directly launched 35B server imported the prompt class under two module names. The request and export preserved the design system prompt, but the 35B tokenizer dropped it because it saw a different class. The launcher and runner now share the same import path. The effective input in the older greeting run was 12 prompt IDs. The corrected request uses 114 prompt IDs and starts generating a page.

The legacy effective input reproduced all 243 Scheduled IDs and all 246 Replay IDs from the export. Frozen Packed reproduced the Replay sequence exactly, including the divergence from Scheduled at output token 116. With the design prompt applied, Packed and Replay also matched each other, while both first differed from Scheduled at token 47. [Legacy greeting and complete arithmetic records](results/native50/studio-effective-222204.json) · [Corrected request and ablation](results/native50/studio-repro-221244.json)

Lightreason Exact stages BF16 expert bytes into static CUDA buffers and captures Scheduled's existing PyTorch operations in their original order. It also replays the original single-token Gated DeltaNet operations and raises the candidate's GPU expert cache to 15 GiB. The 12 GiB Scheduled reference remains the timing control. The target checkpoint byte audit passed.

| Input | Output | Scheduled → Exact decode tok/s | Scheduled → Exact request |
| --- | --- | ---: | ---: |
| Legacy greeting | 243 identical IDs through EOS | 7.75 → 13.28 | 32.82 s → 19.83 s |
| Greeting with design prompt | 256 identical IDs through the cap | 7.69 → 13.05 | 37.53 s → 23.91 s |
| Arithmetic | 341 identical IDs through EOS | 7.43 → 12.24 | 47.88 s → 29.88 s |

The Exact arithmetic run moved 501 MB of expert weights per decode token. At the measured 23.15 GB/s linear H2D rate, that volume takes about 21.6 ms per token before computation. The 50 tok/s target allows 20 ms per token. This bound applies to the measured cache and transfer policy. [Bandwidth measurement](results/native50/ft-bw-205420.json)

Enable the Exact choice by adding `--edge35b-exact-graphs` to the Studio launch command above. To reproduce the corrected request and the graph ablations, run `python -m edge35b.native_bench --model-dir "$MODEL" --studio-export /path/to/comparison.json --modes scheduled,packed,mixer-graph,scheduled-graph,scheduled-graph15,scheduled-experts15 --repeats 1 --resident-audit --out /tmp/studio-exact.json`. For an export from the old directly launched server, set `request.system_prompt` to an empty string in a copy to reproduce the effective legacy input.

The separate DFlash2 experiment used the same BF16 target and a checksummed public proposer. The newer FreeToken source used the installed native modules after comparing their complete source tree and build script. The first startup hit the watchdog's stale-telemetry cutoff. With the parallel loader and a checked host RAM reserve, the retry finished. The 14 GiB expert cache used the measured bandwidth calibration to assign about 52.7% of cold expert misses to GPU fetch and the rest to CPU compute. [DFlash2 proposer](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) · [FreeToken source](https://github.com/FlashML-org/FreeToken/tree/8215a94f92449c1aff07627d95ef73377363315d)

With up to five proposals per cycle, the corrected Studio prompt measured **40.15 tok/s streaming and 39.90 tok/s non-streaming**, at the original 256-token cap. Arithmetic measured **38.31 and 37.46 tok/s**, respectively, through EOS. The two API modes agreed in text for each prompt. Both arithmetic runs answered 323 and matched the earlier non-streaming FreeToken arithmetic text. Both prompts differed from Packed's text, and the API did not expose raw output IDs. At 4.72 committed tokens per cycle, the fastest prompt averaged 117.60 ms per cycle and would need 94.44 ms per cycle to reach 50 tok/s at that acceptance. [DFlash2 request measurements](results/native50/dflash2-222802.json) · [Watchdog result](results/native50/dflash2-222802.txt)

Seven proposals, a 15 GiB GPU expert cache, and a host-resident input embedding table together raised the four-run median from **39.10 to 41.51 tok/s**. The corrected Studio prompt measured 41.41 tok/s streaming and 41.61 non-streaming. Arithmetic measured 42.42 tok/s streaming and 37.74 non-streaming. Both answered 323 through EOS. The 341-token non-streaming text matched Packed and the 330-token streaming text differed. The fastest run committed about 6.21 tokens per 146.33 ms cycle. At that acceptance, 50 tok/s requires about 124.15 ms per cycle. The experiment changed three settings together, so it does not isolate their gains. [Wider DFlash2 measurements](results/native50/dflash2-232049.json) · [Watchdog result](results/native50/dflash2-232049.txt)

After restarting Studio, the live comparison sent the original design prompt and used a 64-token limit for the smoke test. Both streams produced 64 updates in order and matched the saved Scheduled prefix exactly. Scheduled ran at 8.22 tok/s and took 12.03 seconds. Lightreason Exact ran at 14.64 tok/s and took 8.65 seconds. [Live Exact comparison](results/native50/studio-exact-live.json)

## The 10× Studio request

The November 7 train comparison used Qwen3.6-35B-A3B BF16 with thinking enabled, a 137-token prompt including the Studio design instructions, greedy decoding, and a 128-token cap. Scheduled took **22.77 s** with a **4.92 s** first token and **7.12 decode tok/s**. Packed Replay took **12.81 s** with a **4.53 s** first token and **15.34 decode tok/s**. The speedup was **1.78×** for the request and **2.16×** for decode. Both reached the cap and produced partial HTML. The first 60 output IDs matched.

For the same request, **10× requires 2.277 s** from input to the last token. The 127 decode tokens require at least 55.8 tok/s even with zero time to first token. A 0.5 s first token raises that requirement to 71.5 tok/s.

The FreeToken harness formats the export with Studio's prompt formatter and verifies the raw-completion tokenizer round trip. It sends single requests sequentially, records HTTP wall time and server prefill/decode time, and reports text parity separately from raw output IDs. Both the four- and seven-proposal tests used the same pinned BF16 target, the DFlash2 proposer, a 15 GiB expert cache, device-side copies of expert cache hits during prefill, and the radix cache for input state. The unrelated warm-up and a different user prompt that primes the common design prefix run separately from the measured request. Cache flush clears input state between scenarios and leaves the expert LRU warm from preceding work.

| Proposals per cycle | Input state | Cached input tokens | First streamed text | Streamed request | Non-streamed request |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 | Uncached | 0 | 3.15 s | 7.76 s | 6.46 s |
| 7 | Shared design prompt | 64 | 2.21 s | 6.75 s | 6.64 s |
| 7 | Repeated input | 128 | 2.56 s | 6.87 s | 6.50 s |
| 4 | Uncached | 0 | 2.71 s | 6.99 s | 6.16 s |
| 4 | Shared design prompt | 64 | 2.16 s | 6.22 s | 6.15 s |
| 4 | Repeated input | 128 | 2.17 s | 6.22 s | 6.14 s |

Four proposals improved all six measurements in this sweep. On the streamed shared-prefix case, it reduced the average cycle from about 138 to 96 ms while committed tokens per cycle fell from 3.85 to 3.02. Decode rose from 27.99 to 31.34 tok/s. Input reuse removed attention and recurrent-state work for the cached positions, but the backend still copied complete layers of expert weights, using the GPU cache for resident rows. Priming the design prefix took about 2.2 s and server setup took 155–164 s. The tests do not isolate expert LRU state or guarantee the same gain on other prompts. All twelve completions showed the correct 72 km/h calculation, hit the same output cap, and differed from Scheduled's text. One of the seven-proposal outputs matched Packed Replay's text. The API did not expose raw IDs, and the unfinished app cannot serve as a quality test. [Seven proposals](results/native50/train-radix.json) · [Four proposals](results/native50/train-width4.json)

The routed-prefill prototype uses the existing loop-based cache lookup to move only the experts selected by up to 80 new positions. It admits all selected routes to the GPU cache and uses the original BF16 prefill GEMM. Runtime logs confirmed the path ran for the 73-position shared suffix and 9-position repeated suffix. The 137-position fresh request still takes the full-layer path. An initial version called an unrolled lookup and generated 4.5 MB of PTX during warmup. The watchdog halted that attempt for stale telemetry. The corrected run finished at peak temperatures of 48 °C GPU and 63 °C CPU.

| Input state | Cached input tokens | First streamed text | Streamed request | Non-streamed request |
| --- | ---: | ---: | ---: | ---: |
| Uncached | 0 | 2.30 s | 6.34 s | 6.21 s |
| Shared design prompt | 64 | 1.09 s | 5.04 s | 5.12 s |
| Repeated input | 128 | 1.11 s | 5.11 s | 4.04 s |

The shared-prefix run improved from 6.22 to **5.04 s** over the four-proposal control, with prefill falling from 2.16 to 1.09 s. It measured **4.52×** against the saved Scheduled request time and **2.54×** against Packed Replay. The warm repeated non-streaming run reached **4.04 s** and 33.26 decode tok/s with 0.21 s of server prefill. The first streamed repeat was slower, so 0.21 s is a warm observation. Preparation took 161 s and seeding the common prefix with a different user prompt took about 2.2 s. All six runs produced 128 tokens and contained the correct 72 km/h calculation. All six texts differed from the corresponding unpatched four-proposal run and from Scheduled. The shared-prefix stream matched Packed Replay's text. The altered cache occupancy can also change CPU/GPU placement during decode. This sweep does not isolate every numerical difference or certify output quality. [Routed-prefill measurements](results/native50/train-routed-prefill.json)

At the shared-prefix run's 1.09 s to first text, the remaining 10× budget would require about 107 committed tok/s. Even at 0.21 s to first token, the target requires about 61.5 tok/s. The shared-prefix trial committed 3.18 tokens per 98.6 ms cycle. Achieving 10× with a 0.5 s first token at that acceptance requires about 44 ms per cycle. The remaining gap requires reducing verifier cost, improving acceptance, or both.

The restored Studio streamed the same train prompt through Scheduled and Lightreason Exact. Both emitted 128 updates sequentially and matched all 128 IDs from the original Scheduled export. The new Scheduled run took 22.81 s at 7.09 decode tok/s. Exact took 16.14 s at 11.29 decode tok/s, a **1.41×** request speedup. Time to first token was 4.91 and 4.89 s, respectively. The FreeToken experiment remains separate from the Studio runners. [Live exact stream record](results/native50/train-studio-exact.json)

Run the same test from the workspace on the GPU machine with the pinned FreeToken source, audited target checkpoint, DFlash2 checkpoint and bandwidth calibration described above. `FT_SOURCE` points at the checkout's `python` directory. For the routed variant, apply `edge35b/freetoken_routed_prefill.patch` with `patch -p1` in a separate copy of that directory, point `FT_SOURCE` at the copy, and export `LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80`. This activates the routed variant only for short extensions using the pinned BF16 expert banks. The unpatched control uses `--draft-steps 4` or `7`.

```bash
export PYTHONPATH="$PWD:$FT_SOURCE"
export TRITON_CACHE_DIR=/path/to/lightreason-benchmark/freetoken-triton
export XDG_CACHE_HOME=/path/to/lightreason-benchmark/freetoken-dflash2-cache
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
taskset -c 2-3 python hardware_exact10.py --gpu-stop-c 70 --cpu-stop-c 75 \
  --command-timeout-s 900 --out /tmp/train-watch.json --monitor-command \
  taskset -c 8-15 python -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --studio-export "$EXPORT" --cases studio \
  --draft-dir "$DRAFT" --draft-steps 4 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type radix --prefill-hit-d2d --prefix-trials \
  --server-log /tmp/train-server.log --out /tmp/train.json
```

## Weather dashboard export

The November 7 Studio export compared Scheduled and Packed Replay on the same 151-token input with the design prompt, thinking enabled, and a 128-token output cap. Scheduled took 20.14 s with 4.73 s to first token. Packed Replay took 10.77 s with 4.34 s to first token. Decode rose from 8.24 to 19.75 tok/s, or **2.40×**. The **1.87×** request speedup includes prefill. The first 100 output IDs matched and both streams reached the cap with an incomplete response.

The separate routed-prefill FreeToken run used seven DFlash2 proposals and the same BF16 checkpoint. It decoded at 42.44 tok/s on uncached input and 43.52 tok/s on shared design input. The 128-token streaming measurements were:

| Input state | Cached input tokens | First streamed text | Request | Speedup vs. exported Scheduled |
| --- | ---: | ---: | ---: | ---: |
| Uncached | 0 | 2.31 s | 5.30 s | 3.80× |
| Shared design prompt | 64 | 2.23 s | 5.15 s | 3.91× |
| Repeated input | 128 | 0.69 s | 3.87 s | 5.20× |

Server setup took 152 s and is excluded from request time. The expert LRU remains warm across tests. The fresh and shared design requests had 151 and 87 new input positions, respectively, beyond the routed-only path's 80-position threshold. The six experimental capped outputs all differed from the exported outputs in text. The separate full-page trial produced a closed HTML document and its inline JavaScript parsed successfully. It took 95.39 s for 3,938 output tokens, at the limit reported by the speculative server. There is no full-page Scheduled control in the export. [Raw weather timings and outputs](results/native50/weather-width7.json)

A paired Packed Replay trial enabled the existing `TransferPipeline` only for its multi-token hybrid forward. It preserved all 128 IDs in both alternating pairs and moved the same 35.04 GB of BF16 weights during prefill. The pinned staging and copy stream raised first-token time from 4.32–4.44 s to 5.32–5.33 s. Request time rose from 10.70–10.84 s to 11.70–11.72 s. The post-run checkpoint byte audit passed. The live runner uses its original transfer path. [Paired Packed Replay measurements](results/native50/weather-pinned-prefill.json)

LaneZero's public homepage plays back a fixed program. On November 7 it displayed 1,364 tok/s on GLM-5.3-Flash against a 145 tok/s OpenRouter provider median, a 9.4× ratio. The displayed 1.10 s is 1,500 output tokens divided by that rate. LaneZero's separate playground labels its rate as first to last token. The site and public API docs do not disclose the serving hardware, weight precision, or a reproducible benchmark for that claim. [LaneZero homepage](https://www.lanezero.ai/) · [LaneZero playground](https://try.lanezero.ai/) · [LaneZero API docs](https://www.lanezero.ai/docs)

A fresh anonymous timer request in LaneZero's playground reported 1,292 tok/s and 2,182 output tokens across eight model calls. Its public frontend divides the total completion tokens by the sum of each call's first-to-last arrival interval, about 1.69 s in this trial. The full turn took 58 s with file edits and several preview checks that timed out in this browser. This measures the service from the browser. GPU timing and server-side buffering remain unobserved. OpenRouter's published throughput definition includes time to first token, so timing boundaries should be checked before using the homepage ratio as a controlled comparison. Live trial screenshot · [OpenRouter metric definitions](https://openrouter.ai/blog/insights/evaluate-llm-provider-performance/)

On November 8, two follow-ups asked for 140 lines of five-word sentences, first repeating a fixed sentence, then varying each sentence. Both used the same model, low reasoning effort and streaming settings, and each made one model call with zero tool calls. They ran in sequence in the same conversation, so the second included the first response in its context. The API counted 2,611 and 3,568 input tokens, respectively. Neither output had exactly 140 lines. The samples measure behavior on those inputs and do not control for all workload differences.

| Output | Completion tokens | First response bytes | Last positive payload | Browser decode rate | Text chunks in the SSE response |
| --- | ---: | ---: | ---: | ---: | --- |
| Repeated sentence | 886 | 1.571 s | 2.282 s | 1,248 tok/s | 106 of 108 contained exactly eight tokens |
| Varied sentences | 953 | 1.468 s | 4.433 s | 322 tok/s | 407 of 521 contained one or two tokens |

The Chrome DevTools trace independently reproduces the rate to within rounding, using output counts divided by the first-to-last positive payload span. The size pattern is in the server-sent JSON records. The browser received those records in four and 193 network payload events, respectively. The chunk counts come from tokenizing the concatenated visible response with Z.ai's official tokenizer and mapping character offsets back to the records. Both traces had zero chunk boundaries that split a token. Visible tokens plus the reported reasoning counts and one remaining token account for all reported completion tokens in both runs. Sanitized wire trace · Pinned tokenizer and chunk analysis

The repeated eight-token chunks and much smaller chunks for varied text fit speculative decoding with a seven-token draft plus one token from the verifier. The public vLLM recipe for the same GLM model uses that exact width for DFlash2. This is a candidate explanation for LaneZero's behavior. Streaming buffers can also change emission patterns, and the endpoint exposed no acceptance counts, server timing fields, or system fingerprint. We did not identify their drafter, hardware or precision. [vLLM's GLM-5.3-Flash recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/zai-org/GLM-5.3-Flash.yaml) · Varied run screenshot

## Wider verification on the RTX 4090

The native sweep used the same [z-lab DFlash checkpoint](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/tree/f181eece646affea2c38b2765f1aaa01a9734ccd) for all speculative runs. This checkpoint was trained for 16 positions, including the pending target token. The target evaluated real blocks of 8, 12, or 16 positions, with a shorter block at the output cap. The adaptive mode started at 8 and moved in four-position increments between 4 and 16. It widened after committing at least width minus one, narrowed after committing at most half, and held the width otherwise. Each mode started with an empty expert LRU. The timed callback copies emitted IDs to the host. It omits HTTP and rendering. Mode order rotated across prompts. The three 96-token runs covered train reasoning, interval-merge coding, and arrival-order logic. Arithmetic was allowed to end naturally under the existing 768-token cap. [Timings, exact IDs and per-cycle counts](results/native50/lanezero-wide-v1.json)

| Runner | Median decode tok/s, three capped prompts | Arithmetic decode tok/s | Arithmetic first token | Arithmetic request | Exact vs. Scheduled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scheduled | 7.42 | 7.38 | 2.10 s | 48.17 s | 4/4 |
| Native Packed control | 12.80 | 12.67 | 1.97 s | 28.80 s | 3/4 |
| 8 positions | 16.98 | 17.01 | 1.97 s | 21.96 s | 2/4 |
| 12 positions | 15.31 | 14.65 | 1.97 s | 25.18 s | 4/4 |
| 16 positions | 12.31 | 13.33 | 1.98 s | 27.48 s | 2/4 |
| Adaptive 4–16 positions | 15.46 | 16.11 | 1.98 s | 23.70 s | 1/4 |

On the train prompt, increasing the width from 8 to 16 raised committed output from 5.28 to 6.79 per cycle. Cycle time rose from 302 to 512 ms, cache misses from 4,514 to 7,617, and weight transfers from 28.40 to 47.92 GB. On the full arithmetic answer, 8 and 16 positions committed 5.48 and 7.08 IDs per cycle. Eight positions completed the request 2.19× as fast as Scheduled and 1.31× as fast as native Packed. A 10× result on the same arithmetic answer would require 4.82 s total, or about 120 decode tok/s with the measured 1.97 s first-token latency.

Every runner stopped at EOS and answered 323 on arithmetic. The adaptive run diverged from Scheduled at token 93 and took 351 IDs versus 341. The other four candidates matched Scheduled through EOS. The 8-position and 16-position runs reproduced the known token-34 logic mismatch. The 12-position run matched the 96-token prefixes on this sample. The capped prompts stop in reasoning, so they establish prefix agreement where observed and leave final-answer quality untested. All 34,660,610,688 live BF16 target parameters matched the checkpoint bytes before and after the sweep.

Run the native sweep from the workspace on the GPU machine with the installed benchmark environment. `MODEL` and `DRAFT16` point to the pinned local checkpoints and `PYTHON` is the audited Transformers 5.7.0 environment.

```sh
gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC edge35b/cpu_experts.c -lm -o /tmp/wide-experts.so
export EDGE35B_CPU_EXPERT_LIBRARY=/tmp/wide-experts.so
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
taskset -c 2-3 "$PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out /tmp/wide-watch.json \
  --monitor-command taskset -c 8-15 "$PYTHON" -u -m edge35b.dflash_probe \
  --model-dir "$MODEL" --draft-dir "$DRAFT16" \
  --modes scheduled,delta-packed,flash8,flash12,flash16,flash-adaptive \
  --cases math,code,logic --tokens 96 --stream --hybrid-verifier --also-full-answer \
  --out /tmp/lanezero-wide-v1.json
```

The separate [FreeToken DFlash 1 patch](freetoken_dflash1.patch) runs the 16-position draft on FreeToken's CUDA-graph verifier. It reuses the existing target feature capture and draft KV ring, uses a causal window for the first five draft layers, and gives the sixth full attention over the 4,096-token context. It emits greedy proposals from the target head. The DFlash 2 control uses its own eight-position checkpoint and path selector. The native sweep isolates width with one drafter. The FreeToken comparison measures the combined difference in width and proposer.

Both FreeToken runs use the pinned BF16 target, a 15 GiB expert cache, one request, greedy decoding, and the same prompt, cap, and template for each paired case. Input prefix caching is disabled, and the expert LRU can retain weights between requests. The harness alternates the order of streaming and non-streaming by case. Runtime logs confirm BF16 expert execution. The native sweep audits live parameter bytes. FreeToken is using the audited checkpoint copy, but this HTTP experiment does not independently audit every live runtime tensor. Its API returns output text and counts rather than raw token IDs.

| Prompt | HTTP mode | 7 proposals decode tok/s | 15 proposals decode tok/s | 7 proposals request | 15 proposals request |
| --- | --- | ---: | ---: | ---: | ---: |
| Weather dashboard | Streaming | 39.38 | 28.10 | 5.90 s | 7.19 s |
| Weather dashboard | Non-streaming | 46.72 | 28.25 | 4.93 s | 6.70 s |
| Arithmetic | Streaming | 40.84 | 33.28 | 7.73 s | 10.62 s |
| Arithmetic | Non-streaming | 38.14 | 35.07 | 9.40 s | 10.23 s |
| First-duplicate function | Streaming | 41.54 | 29.69 | 19.38 s | 26.91 s |
| First-duplicate function | Non-streaming | 41.58 | 26.49 | 19.28 s | 29.84 s |
| Shift-cover reply | Streaming | 35.89 | 27.55 | 31.81 s | 37.06 s |
| Shift-cover reply | Non-streaming | 36.33 | 27.08 | 25.11 s | 20.78 s |

On streamed arithmetic, 15 proposals increased output per cycle from 5.98 to 7.12 while raising full cycle time from 146 to 213 ms. The wider variant reduced decode rate in all eight pairs. It finished the non-streaming chat sooner because it generated 542 IDs versus 884. Startup took 161 seconds for the control and 155 seconds for the wider variant, outside the request timings. [Seven-proposal HTTP results](results/native50/lanezero-ft-v2.json) · [Fifteen-proposal HTTP results](results/native50/lanezero-ft-v1.json)

All eight complete arithmetic and chat responses met the task checks. Arithmetic gave 323, and each chat reply used two sentences, accepted the shift cover, and offered Friday in return. Weather and coding reached the caps. The streamed weather prefix was identical in text across the two FreeToken variants, and the control's non-streamed arithmetic text matched Packed. The other FreeToken comparisons against Packed differed in text. These task checks leave broader quality and strict token parity unverified for FreeToken.

On the weather export's fixed 128-token comparison, the seven-proposal streamed result measured 3.41× against the saved 20.14-second Scheduled request and 1.83× against the 10.77-second Packed Replay request. Time to first streamed text was 2.68 seconds, above the entire 2.01-second budget for 10×. The paired tests show that increasing width alone does not reach that target. The observed bottlenecks are the added expert work within a verification pass and uncached prefill on the longer input. The width experiments are opt-in.

The three watchdog runs finished successfully. The maximum sampled temperatures across all experiments were 52 °C GPU and 71 °C CPU, below the configured 70 °C and 75 °C stops.

For reproduction, use the routed-prefill checkout above for the 7-proposal control. For 15 proposals, apply `freetoken_dflash1.patch` with `patch -p1` from the `python` directory of a separate copy. Set `FT_SOURCE` to that copy and select the corresponding `DRAFT` and `STEPS`. Build the input suite from the weather-dashboard export and saved native controls.

```sh
python3 - "$EXPORT" /tmp/wide-suite.json <<'PY'
import json
from pathlib import Path
import sys
from edge35b.freetoken_bench import studio_reference

suite = studio_reference(Path(sys.argv[1]))
reports = Path("edge35b/results/native50")
for file, cases in (
    ("heldout-201800.json", {"known-math", "coding"}),
    ("eos-205420.json", {"chat"}),
):
    report = json.loads((reports / file).read_text())
    assert report["revision"] == suite["revision"]
    suite["rows"].extend(row for row in report["rows"] if row["repeat"] == 0 and row["case"] in cases)
Path(sys.argv[2]).write_text(json.dumps(suite))
PY

export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
taskset -c 2-3 "$PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out "/tmp/wide-ft-$STEPS-watch.json" \
  --monitor-command taskset -c 8-15 python3 -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed /tmp/wide-suite.json --cases studio,known-math,coding,chat \
  --draft-dir "$DRAFT" --draft-steps "$STEPS" --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
  --server-log "/tmp/wide-ft-$STEPS.log" --out "/tmp/wide-ft-$STEPS.json"
```

## Uncached prefill and cold experts

The comparison keeps the same seven-proposal FreeToken runner, audited Qwen3.6-35B-A3B BF16 checkpoint copy, greedy settings, and four prompts from the wider-verification test. All measured requests have zero input-prefix hits. Expert weights can stay resident between requests. The control and the selected-row trial use two repeats per streaming mode and reverse prompt order for the second repeat. They run in separate processes with exclusive GPU access and identical warmup requests.

The [selected prefill patch](freetoken_selected_prefill.patch) changes the path for 81–192 input positions. It forms the union of a layer's actual top-k expert routes on the GPU. The existing copy kernel stages cold rows from pinned host memory and resident rows from GPU memory. The staging buffers still place each row at its original expert ID, so the same BF16 prefill GEMM and routing weights are used. The short-prefill path for 80 positions and below keeps its existing cutoff. The three short prompts therefore act as checks on whether the weather request leaves the same state for subsequent requests.

The [token audit patch](freetoken_benchmark_token_ids.patch) forwards the IDs directly from FreeToken's detokenizer to the final HTTP completion record. The harness validates their count against reported usage. Both arms include the same instrumentation. The output comparisons use these raw IDs and the stop reason.

| Prompt | HTTP mode | Control request | Selected-row request | Control decode tok/s | Selected-row decode tok/s | Exact pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Weather dashboard | Streaming | 5.53 s | 4.40 s | 40.40 | 40.25 | 2/2 |
| Weather dashboard | Non-streaming | 5.20 s | 4.08 s | 42.29 | 42.07 | 2/2 |
| Arithmetic | Streaming | 9.19 s | 9.25 s | 40.15 | 39.84 | 2/2 |
| Arithmetic | Non-streaming | 9.27 s | 9.27 s | 39.08 | 39.08 | 2/2 |
| First-duplicate function | Streaming | 19.56 s | 19.59 s | 41.14 | 41.06 | 2/2 |
| First-duplicate function | Non-streaming | 19.63 s | 19.58 s | 40.84 | 40.95 | 2/2 |
| Shift-cover reply | Streaming | 23.70 s | 23.63 s | 36.36 | 36.48 | 2/2 |
| Shift-cover reply | Non-streaming | 27.95 s | 27.86 s | 36.23 | 36.33 | 2/2 |

Medians are across two runs of each prompt and HTTP mode. All 16 comparisons preserved the control's output IDs and stop reason. Arithmetic and chat stopped naturally and passed the same answer checks. Weather and coding reached their fixed caps. The seven-proposal control matched Packed's exact IDs in one of the 16 records. Both streamed weather responses first differed from the saved Scheduled response at ID 31. These results establish parity with FreeToken's control on this suite. Setup took 161 s for the control and 158 s for the selected-row runner. [Raw control results](results/native50/prefill-cold-control.json) · [Raw selected-row results](results/native50/prefill-cold-selected.json)

Weather first streamed text fell from 2.38 to 1.24 s. The 151-position input used 5,577 expert rows across 40 layers, down from 10,240 staged rows with the whole-layer path. The four timed requests fetched 3,788–5,326 of those rows from pinned host memory, equivalent to 23.83–33.51 GB of BF16 payload. The remaining rows came from GPU memory. These timings include one readback of the new counters per selected prefill. The row reduction explains the prefill gain. Decode acceptance and cycle times were essentially unchanged.

The streamed weather request was 1.26× faster than the seven-proposal control and 4.58× faster than the saved Scheduled run. A 10× result against Scheduled allows 2.01 seconds total. At the measured 1.24 seconds to first text, the other 127 tokens would need about 165 tok/s. At the measured 40.25 tok/s, decoding alone takes about 3.16 seconds.

The isolated copy sweep verified the destination bytes after transferring the two BF16 weight banks, totaling 6 MiB per expert. At 13 experts, the default eight blocks per bank sustained 25.65–26.14 GB/s, versus 25.10 GB/s with 32 blocks and 24.77 GB/s with 64. At 150 experts, the default reached 26.28 GB/s and both larger grids stayed around 25.83 GB/s. Six-expert copies varied across repeats. The default remains the faster measured configuration for the larger copies. [Copy measurements](results/native50/prefill-cold-copy.json)

The decode-capacity ablation kept the whole-layer prefill path. The 16 GiB process could not allocate the KV cache under the existing memory budget and produced no results. The watchdog also recorded stale temperature telemetry during that attempt. The 15.5 GiB trial completed with the same requested context length and ran one repeat per HTTP mode. All eight pairs diverged from the corresponding control IDs. Arithmetic and chat still passed the task checks. On the first streamed weather pair, the reported verifier cycle fell from 107.4 to 89.6 ms, while committed IDs per cycle fell from 4.27 to 3.37. Decode dropped from 39.41 to 37.29 tok/s and the first different ID was at token 73. The faster individual cycles did not transfer to a faster weather request. Capacity also changes which experts run on CPU and GPU, so this ablation does not isolate transfer latency or prove numerical equivalence. The selected-row result uses the original 15 GiB capacity. [Capacity timings and raw IDs](results/native50/prefill-cold-cache155.json)

The recorded peaks were 50 °C GPU and 71.25 °C CPU under the existing 70 °C and 75 °C stops. [Watchdog outcomes and expert counts](results/native50/prefill-cold-watch-summary.json)

Run the new comparison with the same `MODEL`, `FT`, `DRAFT`, and `PYTHON` as the seven-proposal control and the `/tmp/wide-suite.json` constructed above. In an isolated copy of the routed-prefill FreeToken checkout, apply the token audit and selected prefill patches from the checkout's `python` directory. Point `FT_SOURCE` at that directory. Use a fresh server process for each arm.

```sh
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_PREFILL_COUNTERS=1
for selected_limit in 0 192; do
  export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS="$selected_limit"
  taskset -c 2-3 "$PYTHON" hardware_exact10.py \
    --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out "/tmp/prefill-${selected_limit}-watch.json" \
    --monitor-command taskset -c 8-15 python3 -u -m edge35b.freetoken_bench \
    --ft "$FT" --model "$MODEL" --packed /tmp/wide-suite.json --cases studio,known-math,coding,chat \
    --repeats 2 --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
    --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
    --server-log "/tmp/prefill-${selected_limit}.log" --out "/tmp/prefill-${selected_limit}.json"
done
```

For the 15.5 GiB ablation, use `LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=0`, `--repeats 1`, and `--cache-gib 15.5` with fresh output paths.

The FreeToken Studio session streamed the same weather prompt and design instructions as the November 8 256-token export. Both requests used 151 input tokens with zero prefix-cache hits. Backend decode measured 39.84 and 40.44 tok/s. Studio's backend request took 7.51 and 7.42 s, with about 1.11 s to first text. The client received the first text in about 1.53 s and the terminal event in 8.10 and 7.84 s. The saved Packed Replay request took 19.23 s at 17.18 decode tok/s. The two FreeToken runs matched the first 30 IDs of the saved Scheduled and Packed Replay outputs and differed from each other. All runs reached the 256-token cap. [Studio streaming results](results/native50/studio-freetoken-256.json)

## The 256-token Studio comparison and decode cache policy

In the later Studio export (`lightreason-compare-2026-11-08T19-44-49-297Z.json`), the saved Packed Replay request took 19.230 s, with 4.385 s to first token and 17.18 tok/s for decoding. The live seven-proposal FreeToken request took 7.846 s, with 1.135 s to first text and 38.01 tok/s for decoding. This is 2.45× in backend request time and 2.21× in decode rate. The client observed first text at 1.687 s and the terminal event at 8.278 s. Both arms stopped at 256 IDs; the common prefix was 30 IDs. The export compares a saved result with a live one. It does not show strict parity or both engines competing for the same GPU.

Packed Replay replays parts of the original single-token execution. FreeToken also uses the DFlash2 proposer and verifies several positions with the BF16 target per pass. Of 371 proposed IDs, 202 were accepted; 53 verifier passes committed the 255 IDs after the initial prefill token. This is 4.81 committed IDs per pass and 126.60 ms per pass. At that acceptance, 50 tok/s requires 96.23 ms per pass. At that cycle time it would require 6.33 committed IDs per pass. Selected-row prefill reduces a separate cost and does not account for the decode-rate gain.

The [decode experiment](freetoken_decode_trial.patch) adds a vectorized routing metadata kernel, a repeat-based admission policy adapted from Packed, and an adjustable ratio for fetching cold experts to the GPU versus computing on the CPU. It runs the same seven-proposal graph, 15 GiB cache, target checkpoint and greedy settings throughout. Input state and expert-cache metadata start cold for each request. A device configuration tensor selects the policy between requests. No policy change alters the BF16 weights or lets the proposer commit a token. Changing CPU/GPU placement can change floating-point evaluation and therefore target tokens. The control in this experiment includes the opt-in policy instrumentation. Compare its measurements against other rows in this experiment; Studio allows expert weights to stay resident between requests.

| Weather policy | Decode tok/s | HTTP first text | HTTP request | IDs/pass | ms/pass | First different ID vs control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 37.75 | 1.423 s | 8.18 s | 4.64 | 122.81 | None |
| Vectorized routing | 37.79 | 1.423 s | 8.17 s | 4.64 | 122.69 | None |
| Repeat window 2 | 37.01 | 1.423 s | 8.31 s | 4.47 | 120.87 | 102 |
| Repeat window 4 | 37.47 | 1.423 s | 8.23 s | 4.64 | 123.74 | None |
| Repeat window 8 | 34.98 | 1.424 s | 8.72 s | 4.40 | 125.69 | 248 |
| Fetch 40% | 34.05 | 1.424 s | 8.91 s | 4.25 | 124.80 | 31 |
| Fetch 65% | 38.07 | 1.423 s | 8.12 s | 4.64 | 121.78 | 31 |
| Repeat window 4, fetch 65% | 39.79 | 1.422 s | 7.83 s | 5.00 | 125.65 | 31 |

Rates and times are medians of two streamed runs. Every arm after the control includes vectorized routing. Both exact arms matched every ID and stop condition in both repeats. The weather winner gained 5.4% in decode rate by changing accepted output and committing more IDs per pass. Its pass took longer and the cache recorded more misses (11.76 vs. 11.00 per layer). It did not reduce the decode cycle cost. Setup took 158 s and is separate from all table timings.

| Held-out prompt | Decode tok/s, control → trial | HTTP request, control → trial | First different ID | Stop |
| --- | ---: | ---: | ---: | --- |
| Arithmetic | 40.81 → 41.98 | 9.04 → 8.30 s | 34 | Natural |
| First-duplicate function | 41.23 → 40.08 | 19.62 → 20.15 s | 116 | 768-ID cap |
| Shift-cover reply | 36.19 → 37.53 | 23.74 → 17.20 s | 306 | Natural |

The held-out set uses one streamed pair per prompt. Both arithmetic runs answered 323. Both chat runs used two sentences, accepted the shift cover, and offered Friday in return. The shorter chat answer took 612 IDs vs. 827, so most of the request-time gain was due to length. Both coding runs reached the cap in the thinking section. These checks do not establish quality equivalence. The comparison did not promote a policy to Studio. [Raw timings and IDs](results/native50/decode-policy.json) · [Expert counts and watchdog summary](results/native50/decode-policy-diagnostics.json). Peak measured temperatures were 47 °C GPU and 68.5 °C CPU.

To reproduce, use the 256-token Packed Replay export (`lightreason-compare-2026-11-08T19-18-11-827Z.json`) as the `studio_reference` in the suite-construction command above. Keep `known-math`, `coding` and `chat` from the same reports, and set `SUITE` to the path of the resulting suite. In a fresh copy of the selected-prefill and token-audit source tree, apply `freetoken_decode_trial.patch` with `patch -p1` from the `python` directory. Set `FT_SOURCE` to that directory. The weather policies alternate forward and reverse; the winning development policy is checked on the three other prompts.

```sh
export PYTHONPATH="$PWD:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192
taskset -c 2-3 "$PYTHON" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out /tmp/decode-policy-watch.json \
  --monitor-command taskset -c 8-15 python3 -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed "$SUITE" --cases studio,known-math,coding,chat \
  --repeats 2 --decode-trials /tmp/decode-policy.cfg \
  --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port 1921 \
  --server-log /tmp/decode-policy-server.log --out /tmp/decode-policy.json
```

The [BF16 transport probe](benchmark_bf16_transport.py) stores sign and fraction as one byte and uses a four-bit exponent offset with indexed full-bit escapes. It audits the reconstructed tensors bitwise. This follows the exponent-compression direction explored by [ZipServ](https://arxiv.org/abs/2603.17435) and [SplitZip](https://arxiv.org/abs/2605.01708). It times the PCIe copy and reconstruction using real expert rows, separately from inference. To reproduce on the idle GPU, use the FreeToken Python environment and run `python -m edge35b.benchmark_bf16_transport --model "$MODEL" --experts 8 --repeats 12 --gather --out /tmp/bf16-transport.json` inside the same thermal watchdog. This probe neither modifies the checkpoint nor installs the codec in Studio.

The probe matched all 100,663,296 BF16 values bitwise in each transfer mode. In layers 13, 26 and 39, packed data used 75.02–75.05% of the original bytes. Median mapped-host gather time across those three layers went from 2.31 to 1.51 ms per eight experts, including reconstruction. Layer 0 was an exception: the codec used 208% of the original bytes and gather time increased from 2.26 to 4.31 ms. An integrated implementation needs a raw fallback or a different exponent codebook. These timings exclude CPU expert work and use contiguous samples, so they cannot be substituted for generation timings. [Transport measurements](results/native50/bf16-transport.json)

At the cold weather control's 4.64 IDs per pass, 50 tok/s requires 92.7 ms per pass, a 30.1 ms reduction. With roughly 11 cold experts per layer across 40 layers, the request processed about 2.77 GB of cold expert weights per pass. About 1.47 GB were fetched to the GPU. Applying the three-layer codec ratio to that fetch volume would save about 0.37 GB per pass before accounting for fallbacks and concurrent CPU work. At the previously measured 29.75 GB/s aggregate CPU plus PCIe bandwidth, that's about 12 ms of bandwidth time. This is a traffic estimate, not measured critical-path time. Packing the PCIe side alone leaves a gap; packing CPU reads as well, retaining more BF16 expert rows, or increasing acceptance without extra verifier work are the next tests with enough theoretical headroom to matter. A packed transport path can preserve routing and every BF16 weight value, which allows a direct token and state comparison before any claim of speed or quality in Studio.

The loaded runner had about 28 GiB of available host RAM. A second full 75%-size copy of the roughly 62.5 GiB routed expert bank would require about 47 GiB. A full experiment therefore needs to build the compact representation in place of the raw host bank, with exact tile unpacking for CPU experts and exact unpacking into the existing GPU slots. It should keep the existing placement and accumulation policy fixed for the first paired run so token differences can be attributed independently of a cache or proposer change.

The [verifier width and CPU expert pairing report](WIDTH_AUTOTUNE.md) records the subsequent physical-width sweep, the learned policy, and the repeated exact CPU reduction measurements.
