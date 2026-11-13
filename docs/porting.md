# Port Lightreason to another GPU

The published Studio build is validated on Linux with an RTX 4090. Start a 3B port with the target-only runner. Start a 35B port with Scheduled and a stock FreeToken BF16 run if supported, then measure which part of the hybrid path costs the most. The 3B draft-assisted path adds an auxiliary and a matching adapter. The 35B path uses the public DFlash2 draft. [3B Studio quickstart](../README.md#run-on-a-gpu-machine) · [35B reproduction](qwen35b.md) · [Measured experiments and rejected changes](experiments.md)

| Hardware | First milestone |
| --- | --- |
| RTX 4090 on Linux | Reproduce 3B Standard vs. Exact in Studio. For 35B, check available system RAM and reproduce Scheduled before testing packed replicas. |
| Other BF16-capable NVIDIA CUDA GPU | Run the original CLI with a compatible PyTorch wheel. Then validate graph replay and kernels on that architecture. The 3B Studio has a 4090 guard, and the 35B Ada gather explicitly targets SM 8.9. |
| AMD GPU | Establish a BF16 PyTorch ROCm reference on a supported GPU. Use ROCm kernels and graph APIs where the CUDA-specific paths require replacement. [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) |
| Apple Silicon or CPU | Establish a BF16 reference with the chosen backend first. Check BF16 operator support and memory before porting the runner. The released GPU graph and timing paths are CUDA-specific. [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) |

## 1. Freeze the reference

Record GPU model, memory, architecture, driver, OS, PyTorch, Transformers, attention backend, and project commit. Start at stock clocks and power limits. Pin the target and tokenizer revisions and verify the target parameter hash and BF16 dtype. Fix the prompt and system prompt, rendered input IDs, greedy policy, native EOS set, context capacity, and maximum output tokens. Keep a clean baseline and preserve any already-optimized comparison as a second control.

On a compatible CUDA GPU, start with the public target checkpoint and the following smoke test. On other backends, make an equivalent original Hugging Face greedy runner using that backend's synchronization and BF16 support.

```bash
python download_3b.py --destination model-3b
python run_model_3b.py --model-path model-3b --engine original \
  --prompt 'Explain why the sky is blue in three sentences.' --tokens 128 --json
```

The downloader is backend-independent and verifies the official files. To check the live parameter bytes before and after a candidate run, use `exact10_validation.hash_model_parameters(model)` outside the timed region. The expected BF16 hash is in the [runtime manifest](../optimization_20261106/controls/runtime-manifest.json). Confirm the model is on the intended device and save full IDs including EOS and the stop reason. Compare across devices separately, since the reference itself may differ across numerical backends.

For ROCm, create a separate environment with a PyTorch ROCm build that supports your GPU and OS. Check that `torch.version.hip` is set and `torch.cuda.is_available()` is true, then record `torch.cuda.get_device_properties(0)`. Run a BF16 operation and verify the loaded model's dtype, device, and generation before profiling. The cached greedy loop in [`run_model_3b.py`](../run_model_3b.py) is the starting reference. Use the corresponding device and synchronization methods on MPS or CPU. A load failure, dtype fallback, or unsupported operator blocks the same-precision comparison until resolved.

## 2. Profile before changing kernels

Measure model loading and graph preparation separately. On warm runs collect prefill, first token, decode, and total latency. For a speculative runner also measure drafting, ranking and tree construction, verification, acceptance, committed KV updates, transfers and host waits, and streaming. Distinguish GPU time from CPU wall intervals that overlap it. Run final trials with profiling disabled.

For target-only work, compare graph replay, cache layout, projection packing and output-head evaluation one change at a time. For draft-assisted work, tune the actual packed verifier row count and time the whole cycle, including any extra draft passes or checks. Use committed tokens divided by complete cycle time. Check whether draft rank and score predict actual acceptance. Keep checkpoint-supported draft depth, ancestor attention, path positions, and EOS behavior intact. A mask on the original graph leaves its physical verifier shape unchanged.

If the trace shows coordination cost, try device-side bookkeeping and preallocated suffix writes. If drafting dominates, any draft-only precision change needs a fresh acceptance and quality measurement. If target weight reads dominate, test the local GEMM shape and memory traffic in isolation. A lossless repack must recover every original BF16 value. A full request benchmark still decides whether to keep the change.

### Translate the 35B gains to a different machine

1. **Decide where the weights can live.** The full 35B text target needs 64.56 GiB of BF16 storage before runtime workspace, attention cache, or draft weights. When the target fits in device memory, benchmark a fully resident BF16 baseline first. When it spills, record host RAM, NUMA layout, link generation and width under load, and free VRAM after loading. Keep workspace and the target's state within budget before increasing the expert cache.
2. **Measure CPU and link bandwidth together.** Run `ft bench bw --model "$MODEL" --formats bf16 --cpu-threads 8` with the intended CPU affinity. Record standalone and overlapped results. Choose the CPU/GPU expert split on development prompts. Both streams can compete for the same DRAM bandwidth.
3. **Retune the actual workload.** Measure verified tokens per pass, expert misses, host bytes per committed token, and complete cycle time. Try the resident expert budget, CPU reuse and verifier widths independently. The supplied 34-layer and replica plans are calibrated on the 4090 setup. To generate a plan for a different workload, collect fresh fetch profiles and run [the replica ranking script](../edge35b/plan_gpu_resident_cache.py).
4. **Keep the reference arithmetic and generation settings fixed where you claim exact output.** Re-run the same candidate inputs and verify which tokens were committed, the EOS or cap, and the accepted state after rejection. A different CPU/GPU split or kernel reduction can change the greedy decision while retaining every BF16 weight bit. Check the old archive and the new device's live control as separate comparisons. The [35B reproduction](qwen35b.md) describes how to bring up the measured configuration.

## 3. Test on held-out inputs and report the result

Tune on development inputs and reserve a separate held-out set. Repeat warm baseline, prior optimized control if available, and candidate runs in alternating order. Run both streaming and non-streaming. Record latency, token count, tokens per second, first-token time, end-to-end time, and preparation. Keep stream transport and browser rendering distinct from model timings. Report per-case measurements and paired summaries.

Report full token and EOS matching against each control. For a strict change, require exact token and EOS equality and check committed state where the implementation promises it. For a change in floating-point evaluation order, identify the changed operation, keep original BF16 weight values and target verification, and label the variant separately. Evaluate task correctness and format, inspect divergent cases, and report paired gains and regressions on held-out inputs. A speculative proposal can only commit tokens or persistent target KV through the original target.

Document the commands, input fixtures, environment, preparation cost, and measured hardware limits. Keep a candidate only when it meets the chosen correctness contract and improves the held-out metric. The 4090 measurements are specific to those workloads and configurations.

## Give this workflow to an agent

From the repository on `main`, ask Codex to use `$lightreason-hardware`. For other coding agents, supply the [skill file](../.agents/skills/lightreason-hardware/SKILL.md). Codex discovers project skills in `.agents/skills`. [Codex skills documentation](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)

```text
Use the Lightreason hardware skill to establish a BF16 baseline for Qwen2.5-3B
or Qwen3.6-35B on my machine. Profile the complete request, change one measured
bottleneck, and compare repeated held-out token IDs, stops, and timings.
Give me the commands and the hardware limits you actually measured.
```
