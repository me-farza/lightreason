# Qwen3.6-35B-A3B on one RTX 4090

The pinned public checkpoint contains 66.97 GiB of BF16 tensor data. The RTX 4090 on the test host has 24 GiB of VRAM and the host has 121 GiB of RAM. This experiment targets text generation with the original Qwen weights and thinking template. The visual encoder and native multi-token predictor are outside this text-only run. [Qwen model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [Pinned checkpoint](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/345d467b117d1fd5eda9b879a5988de9ae5f47a7)

The runner keeps attention, the router, shared experts, embeddings, and output head on CUDA. Routed expert weights remain in CPU-mapped safetensors. The transfer modes feed the original BF16 values to the Transformers eager CUDA expert forward. The scheduled variant changes how it reads router IDs during single-token decode and preserves the same projection operations, routing weights, and accumulation order. Every mode shares the greedy policy and EOS rules. The code audits all text parameter bytes against the official files before and after a complete benchmark. [Upstream eager implementation](https://github.com/huggingface/transformers/blob/v5.7.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L656-L689)

| Mode | Transfer policy |
| --- | --- |
| Dense | Move the full expert bank for the current layer. The 40-layer model transfers 60 GiB of expert weights per decode step in this control. |
| Selected | Move only experts picked by the original router. Eight of 256 routed experts per layer require 1.875 GiB per decode step before any reuse. |
| Cached | Store selected expert rows in a bounded GPU LRU and fetch only misses. Multi-token prefill uses the selected path. |
| Packed | Gather selected rows in pinned host staging and make two bulk transfers per decode layer. Multi-token prefill uses the selected path. |
| Scheduled | Reuse the GPU LRU and read all eight router IDs at once. Visit them in the original ascending order, using the original router slots. Multi-token prefill uses the selected path. |

## Original-weight layer result

Layer 4 used its original checkpoint tensors, verified by the official shard SHA-256. The reference kept the whole layer resident on CUDA and called the original eager forward. Each mode ran two alternating 16-step trials with identical synthetic hidden states and routing. Half the routes repeated each step. The stock watchdog sampled a maximum of 47 °C GPU and 54 °C CPU during the test. [Raw result](original-layer.json)

| Mode | Median ms per step | Mean MiB transferred per step | Bitwise match to resident CUDA |
| --- | ---: | ---: | --- |
| Dense | 164.33 | 1,536 | All steps |
| Selected | 6.34 | 48 | All steps |
| Cached | 4.23 | 25.5 | All steps |
| Packed | 9.11 | 48 | All steps |

The cache was 1.50× faster than selected-only transfer on these routes. Packed transfer was slower, even with its 25 ms staging setup outside timing. The dense comparison measures the cost of whole-layer transfers. These results describe an isolated layer. Full text generation requires the complete checkpoint and a separate benchmark.

A separate alternating run of the same original-weight layer reduced the cached path from **4.31 ms to 3.58 ms per step** with scheduled routing, a 1.20× gain. All outputs matched the fully resident CUDA reference at the bit level. Both modes transferred the same number of BF16 weight bytes. [Scheduled comparison](scheduled-layer.json)

## End-to-end generation

The full text model loaded **34,660,610,688 BF16 parameters**, or **64.56 GiB** of text weights. Routed experts accounted for 60 GiB of the text weights. The 4.56 GiB of non-routed weights stayed on CUDA. The loader compared every live text tensor against the pinned checkpoint before and after inference. On the four-token math audit, full logits and generated IDs matched across the dense, selected, and cached modes. [Audit](results/audit.json)

Two prompts, two alternating repeats per prompt, and a six-token cap gave four paired headless comparisons with thinking enabled. All four pairs matched the dense reference in token IDs and stop reason. The timing scope excludes loading, audit, and warmup. [Repeated timings](results/baseline.json)

| Mode | Median request time | Median first token | Median decode rate |
| --- | ---: | ---: | ---: |
| Dense | 40.04 s | 7.33 s | 0.15 tok/s |
| Selected | 4.98 s | 3.45 s | 3.27 tok/s |
| Cached | 4.60 s | 3.44 s | 4.33 tok/s |

Caching measured **8.71×** median paired request speedup over whole-layer transfer and **1.08×** over selected-only transfer on these short requests. The whole-layer offload control transfers a large volume of inactive expert weights. This comparison measures the gain from changing what crosses PCIe in this setup.

On three fixed prompts with two alternating repeats and a 48-token cap, the cache reduced median request time from **17.50 s to 11.10 s** and raised median decode throughput from **3.32 to 6.09 tok/s**. The paired request speedup was **1.58×** over selected-only transfer. All six pairs matched exactly. The median decode cache hit rate was 69.5%, cutting transferred weight bytes from 1.875 to 0.572 GiB per token. Peak CUDA tensor allocation was 16.61 GiB. [Sustained comparison](results/sustained.json)

In a separate six-pair comparison on the same inputs, scheduled routing reduced the median request from **11.07 s to 9.70 s**, a **1.14×** paired gain over the cache with eager routing. Median decode reached **7.38 tok/s** and first token took **3.32 s**. All six pairs matched exactly. A separate eight-token audit matched every logit byte, and the post-run checkpoint audit passed. [Scheduled timings](results/scheduled-full.json) · [Logit audit](results/scheduled-audit.json)

On a single 256-token reasoning prefix, the cache reduced the request from 80.84 s to 44.45 s with the same 256 output IDs. Both modes reached the output cap before finishing the answer. [256-token comparison](results/reasoning.json)

On the algebra prompt with a 1,024-token cap, cached and scheduled generation produced the same 1,024 IDs and both stopped at the cap. Scheduled routing took **148.35 s** versus **177.63 s** with eager routing, a **1.20×** request gain. Both used 16.66 GiB peak CUDA tensor allocation. [Algebra prefix](results/algebra-prefix.json)

On the direct question `What is 17 times 19?`, both modes generated the same **341 tokens through native EOS** with thinking enabled and answered **323**. Scheduled routing took **47.29 s** versus **57.39 s** with eager routing. [Completed reasoning answer](results/short-answer.json) · [Hardware watchdog summary](results/hardware-summary.json)

## Further speed experiments

The reference for this pass is the scheduled runner with a 12 GiB expert cache. Each timing comparison uses identical greedy inputs and caps on the RTX 4090. The candidate called `batched` reads the prefill routing table once per layer, retains the original per-expert PyTorch projections, and combines the pointwise operations across the eight active experts. Its 15 GiB cache and batching together reduced median first-token latency from **3.36 s to 3.02 s**. All six 48-token pairs matched token-for-token. The separate eight-token audit matched every logit byte. [Timing](results/speed2/v6-batched-timing.json) · [Logit audit](results/speed2/v6-batched-audit.json)

| Comparison | Median request | Median decode rate | Exact token pairs |
| --- | ---: | ---: | ---: |
| Scheduled → grouped prefill and 15 GiB cache | 9.65 s → 9.14 s | 7.46 → 7.74 tok/s | 6/6 |
| Grouped → overlapped pinned transfers, both at 15 GiB | 9.14 s → 9.60 s | 7.74 → 7.89 tok/s | 6/6 |
| Scheduled → batched operations and 15 GiB cache | 9.77 s → 8.39 s | 7.36 → 8.89 tok/s | 6/6 |
| Grouped → new CUDA matrix-vector kernel, both at 12 GiB | 9.39 s → 8.40 s | 7.40 → 8.95 tok/s | 3/3 |
| Grouped → direct AVX2 expert evaluation | 9.31 s → 8.82 s | 7.44 → 8.05 tok/s | 3/3 |
| Grouped → native drafting with short recurrent verification | 9.43 s → 8.70 s | 7.38 → 8.26 tok/s | 3/3 |
| Batched → host/GPU expert split, both at 15 GiB | 8.31 s → 7.38 s | 8.96 → 10.90 tok/s | 3/3 |

The new CUDA matrix-vector kernel changed logit bytes on its audit despite matching the generated IDs. The AVX2 path uses a different floating-point reduction order. These paths are experimental. Native drafting uses the original checkpoint's MTP head, verifies proposals with the BF16 target, and restores the convolution and recurrent state before replaying a rejected prefix. One proposal with the stock chunk verifier accepted 68 of 71 drafts and slowed generation. The three-proposal recurrent experiment accepted 99 of 122 drafts and measured a 1.08× median paired request gain. [Transfer timing](results/speed2/v2-pipeline-timing.json) · [CUDA logit comparison](results/speed2/v4-fused-audit.json) · [CUDA timing](results/speed2/v4-fused-timing.json) · [AVX2 comparison](results/speed2/v5-cpu.json) · [Native draft results](results/speed2/v4-recurrent-mtp.json)

On the six-token workload, `seeded` retained the most-used prompt experts in each layer during prefill. It used the same BF16 decode operations and reduced the median request from **4.62 s to 3.82 s** against the original cache runner, a **1.21×** paired gain. All four pairs matched. Median first-token time was **3.22 s**. [Six-token rerun](results/speed2/v7-six-token.json)

On the same complete arithmetic reasoning prompt, `batched` reduced time from **47.41 s to 37.54 s** and increased decode throughput from **7.50 to 9.53 tok/s**. Both modes generated the same 341 token IDs through native EOS, gave the answer 323, and passed the full checkpoint audit. [Complete reasoning rerun](results/speed2/v7-full-answer.json)

The host/GPU split seeds the CUDA cache from the prompt. It evaluates newly seen cold experts with the AVX2 kernel from their original mapped BF16 weights and promotes repeated experts to CUDA. The median PCIe weight volume fell from **0.522 to 0.161 GiB per decoded token**, with about 52 of 320 routed expert evaluations per token handled by the CPU. The CPU reduction order differs from the CUDA control. A separate complete reasoning comparison matched all 341 tokens and native EOS, taking **37.72 s** with the split and **46.97 s** with the scheduled control. [Host/GPU timing](results/speed2/v8-hybrid.json) · [Complete answer](results/speed2/v8-hybrid-answer.json) · [Watchdog summary](results/speed2/hardware.json)

The live profile saw 54–60 ms of host-side time inside expert copy calls per decoded token. The scheduled reference transferred about 0.572 GiB per decoded token. At the 10× target of roughly 74 tok/s, that is about **45 GB/s**. Even the 15 GiB GPU-only cache path transferred about **0.522 GiB per token**, which would require about **41 GB/s**. PCIe 4.0 ×16 provides a theoretical **31.5 GB/s** per direction. The host/GPU split reduces that traffic, and its 10.90 tok/s still leaves a gap to the 74 tok/s target. Time to first token remains a separate limit for six-token requests. [Route trace](results/speed2/profile.json.gz) · [PCI-SIG link performance](https://pcisig.com/sites/default/files/files/PCI-SIG-Seamless_Transition_to_PCIe_5.0_in_System_Implementations_FINAL.pdf)

## 50 tokens per second experiments

See [the native runtime report](NATIVE50.md) for the current Packed baseline,
CUDA graph and kernel ablations, and the token-34 verifier diagnosis.

The probe uses the same audited BF16 target. `delta-packed` groups the resident
expert projections on CUDA, uses pinned buffers for the cold experts, and
retains CUDA rows when routes recur within four decode steps. `flash4` and
`flash8` use the public DFlash draft. The BF16 target verifies each block and
commits its matching prefix, using the target's recurrent and convolution
state for that prefix. The target selects the first token and any correction.
[DFlash draft model](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/tree/f181eece646affea2c38b2765f1aaa01a9734ccd)

The first trial used a 14 GiB cache for each runner and three 48-token outputs.
Exact matches compare complete token IDs and the stop reason for these capped
runs. The fused reductions use a different order of floating-point operations.
[Raw first trial](results/target50/first-trial.json) · [Watchdog](results/target50/first-trial-watch.json)

| Runner | Median decode tok/s | Exact matches vs Hybrid |
| --- | ---: | ---: |
| Hybrid | 10.55 | Reference |
| Fused recurrent update | 10.91 | 2/3 |
| Fused recurrent update with recent expert admission | 11.15 | 2/3 |
| Packed CPU/GPU experts with fused recurrent update | 13.26 | 3/3 |
| DFlash with a four-position GPU verifier | 13.26 | 3/3 |
| DFlash with an eight-position GPU verifier | 11.59 | 3/3 |

The four-position GPU verifier copied a median 0.505 GiB of expert weights per
committed decode token. At 50 tok/s it would need roughly 27 GB/s just for those
weights. The packed CPU/GPU runner copied 0.114 GiB per token and read roughly
0.399 GiB per token in its CPU expert projections. These are weight volumes,
excluding activation and state traffic. The target remains 50 tok/s on a single
stream with reasoning enabled.

The second trial records a token stream for each run. The four-position
verifier sent cold singleton experts to the CPU while CUDA ran the cached and
repeated experts. It rounds the recurrent state at each position to match the
serial BF16 cache policy. The three short runs used the same 48-token cap.
[Raw streaming trial](results/target50/hybrid-verify.json) · [Watchdog](results/target50/hybrid-verify-watch.json)

| Runner | Median short decode tok/s | Exact short matches vs Hybrid | Full-answer decode tok/s | Full-answer total |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 10.70 | Reference | 9.19 | 38.93 s |
| Packed CPU/GPU experts with fused recurrent update | 13.25 | 3/3 | 12.85 | 28.39 s |
| DFlash with a four-position CPU/GPU verifier | 17.97 | 2/3 | 18.29 | 20.54 s |

All three runners produced the same 341 tokens and native EOS on `What is 17
times 19?`. That sequence also matches the archived scheduled reference. The
DFlash run accepted 245 of 291 proposals on that answer. Its short coding run
reached 18.50 tok/s. Its short logic run first differed at token 34, so Studio
exposes the packed runner in this profile. The checkpoint byte audit passed
after generation and the watchdog measured peaks of 48 °C GPU and 61.5 °C CPU.

To reproduce on vicena-0, obtain `config.json` and `model.safetensors` from the
pinned DFlash revision above. The SHA-256 of `model.safetensors` is
`1fb90ef50a32bfb8dd2abfe601dd3608d6d5b59dc342820a98830f76f8cd72b7`.
Use the same Python and CUDA environment as the earlier experiments. Pick fresh
report paths and a writable Triton cache directory.

```bash
gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC edge35b/cpu_experts.c -lm \
  -o /tmp/lightreason-verify-experts.so
export EDGE35B_CPU_EXPERT_LIBRARY=/tmp/lightreason-verify-experts.so
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python hardware_exact10.py --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 900 \
  --out /tmp/lightreason-verify-watch.json --monitor-command taskset -c 8-15 \
  python -u -m edge35b.dflash_probe --model-dir /dev/shm/lightreason-edge35b/model \
  --draft-dir /tmp/lightreason-edge35b/dflash-f181eece --modes hybrid,delta-packed,flash4 \
  --hybrid-verifier --cases math,code,logic --tokens 48 --also-full-answer --stream \
  --out /tmp/lightreason-verify.json
```

## Reproduce

Use a separate Python 3.12 environment with PyTorch `2.9.1+cu128`, Transformers `5.7.0`, and safetensors `0.8.0`. On the test host the downloader stages the model across tmpfs mounts with a host memory reserve. Two shards use a third volume to stay within the per-user quotas. Choose writable scratch paths with enough room for the selected shard split. `--mirror` uses an official Qwen mirror for the large shards after checking its size and SHA against the pinned Hub metadata. Each completed file is verified before it enters the model directory.

```bash
python -m edge35b.download \
  --model-dir /dev/shm/lightreason-edge35b/model \
  --shard-dir /tmp/lightreason-edge35b/shards \
  --overflow-dir "$XDG_RUNTIME_DIR/lightreason-edge35b" \
  --overflow-shard model-00001-of-00026.safetensors \
  --overflow-shard model-00006-of-00026.safetensors --workers 10 --mirror
```

The initial end-to-end comparison is greedy with reasoning enabled, native EOS, and a small cap for the whole-layer transfer control. The logits audit runs separately from timing. Reports include checkpoint checks, full token IDs, stop reasons, first-token and decode time, CUDA memory, and expert transfer bytes and cache hits.

```bash
python hardware_exact10.py --out /tmp/edge35b-audit-watch.json --command-timeout-s 900 \
  --monitor-command python -m edge35b.benchmark \
  --model-dir /dev/shm/lightreason-edge35b/model --out /tmp/edge35b-audit.json \
  --cases math --tokens 4 --repeats 1 --audit-logits

python hardware_exact10.py --out /tmp/edge35b-timing-watch.json --command-timeout-s 900 \
  --monitor-command python -m edge35b.benchmark \
  --model-dir /dev/shm/lightreason-edge35b/model --out /tmp/edge35b-timing.json \
  --cases math,logic --tokens 6 --repeats 2

python hardware_exact10.py --out /tmp/edge35b-scheduled-watch.json --command-timeout-s 900 \
  --monitor-command python -m edge35b.benchmark \
  --model-dir /dev/shm/lightreason-edge35b/model --out /tmp/edge35b-scheduled.json \
  --modes cached,scheduled --cases math,logic,code --tokens 48 --repeats 2
```

Use `--modes selected,cached` and fresh output paths to reproduce the sustained selected-only versus cached comparison.

For the batched comparison, use the 48-token command with `--modes scheduled,batched --cache-gib 15 --reference-cache-gib 12`. Use fresh output paths and add `--audit-logits --cases math --tokens 8 --repeats 1` for the separate logit check. Set `TRITON_CACHE_DIR` to a writable directory with free space for the kernel cache.

For the six-token rerun, use `--modes cached,grouped,seeded,batched --cache-gib 15 --reference-cache-gib 12 --cases math,logic --tokens 6 --repeats 2`.

To reproduce the host/GPU experiment on the tested AVX2/FMA Linux host, build the expert kernel and use `--modes batched,hybrid --cache-gib 15 --tokens 48 --repeats 1` with the same watchdog command and fresh output paths.

```bash
gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC edge35b/cpu_experts.c -lm -o /tmp/edge35b-experts.so
export EDGE35B_CPU_EXPERT_LIBRARY=/tmp/edge35b-experts.so OMP_WAIT_POLICY=PASSIVE
```

Run one prompt with the scheduled cache runner and print the generated text. The staged files live on tmpfs and are cleared by a reboot.

```bash
python hardware_exact10.py --out /tmp/edge35b-prompt-watch.json --command-timeout-s 900 \
  --monitor-command python -m edge35b.benchmark \
  --model-dir /dev/shm/lightreason-edge35b/model --out /tmp/edge35b-prompt.json \
  --modes scheduled --tokens 1024 --repeats 1 --prompt 'What is 17 times 19?'
```

For the original-weight layer test, supply the pinned Hugging Face files metadata JSON used for the shard hash check. For a full staged checkpoint the index resolves every shard in the model directory.

```bash
curl -fsSL 'https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B/revision/345d467b117d1fd5eda9b879a5988de9ae5f47a7?blobs=true' \
  -o /tmp/edge35b-files.json
python -m edge35b.probe_experts --model-dir /dev/shm/lightreason-edge35b/model \
  --checkpoint-files /tmp/edge35b-files.json --layer 4 --out /tmp/edge35b-layer.json
```
