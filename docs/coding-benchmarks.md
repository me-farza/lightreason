# Coding benchmarks

All runs used the original BF16 Qwen target, greedy decoding and batch size one on one stock RTX 4090. The decode ratios divide the median tokens per second by the matching reference median. The request columns include prefill and generation. Graph preparation and model load are recorded separately. The outputs, prompts, and checks are available alongside the timings. [Measurement definitions](performance.md)

## 3B with only the public Qwen checkpoint

These are new runs of the same Lightreason Exact implementation used in Studio. Each case had one untimed warmup per runner, then three alternating pairs. Both runners stopped at native EOS in every trial under the same 768-token cap. The first case prepared seven graphs in 4.38 s after runner construction. All later cases reused those graphs. Timings are for the non-streaming generation call, including per-token host EOS checks.

| Task | IDs to EOS | Hugging Face → LR decode tok/s | Decode gain | Hugging Face → LR generation | Exact pairs | Answer check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Python · stable deduplication | 249 | 59.66 → 139.15 | **2.33×** | 4.176 → 1.803 s | 3/3 | Passed seven output and input-mutation fixtures |
| JavaScript · trailing-edge debounce | 493 | 59.36 → 137.63 | **2.32×** | 8.307 → 3.596 s | 3/3 | Both return a wrapper without the required `cancel()` |
| SQLite · top customers with date and count filters | 646 | 59.21 → 138.55 | **2.34×** | 10.913 → 4.676 s | 3/3 | Passed the ranking, tie, minimum-order, and year-boundary fixture |

The target’s live parameter hash matched the archived BF16 hash before and after these tests. The answer checks ran offline on the shared output for each task. [Full prompts, output IDs, and timings](data/qwen25-3b/coding-target-only.json) · [Answer checks and SQL fixture](data/qwen25-3b/coding-target-only-checks.json)

From the repository root, install the pinned dependencies and download the 3B checkpoint using the Studio quickstart. This command uses the public checkpoint alone. [3B setup](../README.md#run-on-a-gpu-machine)

```bash
python -m tools.benchmark_coding_3b --model-path model-3b --repeats 3 \
  --out /tmp/lightreason-3b-coding.json
```

## 3B with a PARD2 draft

These archived research runs used the lab’s matching auxiliary adapter. The first row is the fastest exact coding slice. The next two are held-out Python tasks with complete responses. They show how the result varies with the prompt and how far exactness extends.

| Task and LR runner | Hugging Face → LR decode tok/s | Decode gain | Hugging Face → LR generation | Output and task checks |
| --- | ---: | ---: | ---: | --- |
| Fibonacci · compiled PARD2 · 128-token cap | 63.56 → 724.36 | **11.40×** | 2.017 → 0.187 s | 5/5 exact through the cap |
| Stable deduplication · GPU tree coordination · EOS | 58.31 → 555.27 | **9.52×** | 4.273 → 0.458 s | 0/6 exact vs. Hugging Face, 6/6 vs. prior tree. Both pass all seven behavior and mutation checks |
| Merge intervals · GPU tree coordination · EOS | 57.72 → 429.82 | **7.45×** | 7.367 → 1.027 s | 0/6 exact vs. Hugging Face, 6/6 vs. prior tree. Both return the right intervals but mutate input pairs on three of eight fixtures |

The Fibonacci cap trial used five non-streamed repeats. Its **10.79×** figure is the median paired warm request speedup. The six-prompt suite recorded **63.44 s** of graph setup outside the timing. The held-out rate and time columns use three streamed repeats per task, and their exact counts include another three non-streamed pairs. Streamed timings cover driver delivery and consumer drain. [Capped Fibonacci inputs and output IDs](data/qwen25-3b/compiled-draft.json) · [Held-out timing pairs](data/qwen25-3b/gpu-coordination.json) · [Responses and offline grades](data/qwen25-3b/coding-draft-checks.json)

The three separate Fibonacci pairs run to native EOS. They generated **604 → 567 IDs** in **10.166 → 0.910 s** at the medians. Complete-call rates were **59.41 → 623.28 tok/s** including prefill, with a **10.49×** median paired rate ratio. Full responses differed. All six functions had the same parsed Python program and passed 132 input checks apiece. [Full Fibonacci outputs and checks](data/qwen25-3b/fibonacci-native-eos.json)

With the pinned public PARD2 draft and the separately supplied adapter, run the held-out suite from the benchmark notes. The first command below reproduces the five-pair Fibonacci slice with the other five fixed prompts. The second runs the complete Fibonacci answer. [Adapter and held-out commands](benchmarks.md#reproduce-with-the-matching-adapter)

```bash
python benchmark_graph_prefill_compiled_pard2_lora_tree3b_v1.py \
  --model-path model-3b --draft-path models/pard2 --adaptation-checkpoint-path models/adapter.pt \
  --adaptation-checkpoint-sha256 8e5b53abec75a33825afca5c08a151d2cefd0a8a567bfff2ebc507bdd4ecc6b6 \
  --compile-static --max-len 2048 --draft-tokens 15 --node-budget 64 \
  --tokens 128 --prompts 6 --repeats 5 --out /tmp/lightreason-3b-compiled-capped.json

python benchmark_native_eos_fibonacci3b_v2.py --model-path model-3b \
  --draft-path models/pard2 --adaptation-checkpoint-path models/adapter.pt \
  --out /tmp/lightreason-fibonacci-native.json
```

## 35B HTML and Python

The 35B tests use the thinking template. The weather task asks for a single-file HTML/CSS/JavaScript dashboard. The Python task asks for the index of the first duplicate in an iterable. The limits are part of the original fixtures. Both comparisons stopped at their respective caps. The dashboard ended in partial HTML and the fastest Python runs ended in the thinking section. Treat these as decode measurements.

| Task | Output cap | LR Original Scheduled → LR Packed GPU replicas | Decode gain | First token/text | Request or runner time | Exact against Scheduled |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Weather dashboard | 256 | 7.79 → 46.80 tok/s | **6.01×** | 4.74 → 1.43 s | 37.49 → 6.88 s | First differs at token 31 |
| First duplicate in Python | 768 | 7.93 → 51.08 tok/s | **6.44×** | 3.31 → 1.03 s | 100.07 → 16.05 s | First differs at token 116 |

The Python Scheduled number is the median of two new runs with the same 44 rendered input IDs and 12 GiB expert cache as the original runner uses. Both Scheduled runs match each other. The packed figures are the medians of two streamed runs per prompt from the archived 880 MiB replica trial. Scheduled reports in-process latency and Packed reports local HTTP latency. The decode rates are calculated inside the engines, and the rows compare separate sessions. [Prompt suite and weather baseline](../edge35b/benchmark-suite.json) · [New Scheduled Python runs](data/qwen35b/scheduled-coding.json) · [Packed trials and output IDs](../edge35b/results/native50/gpu-resident-qwen.json.gz)

The fresh packed retries stopped during preparation when temperature readings went stale. Whole-bank, 16-row, and four-row host registration attempts yielded zero timed samples. The watchdog’s 70 °C GPU and 75 °C CPU limits remained in force. The table uses the previously completed packed runs.

On that same Python prompt, stock FreeToken BF16 ran at **32.88 tok/s** in one archived streaming run. The closer paired reference, LR fixed eight with no replica tier, ran at **50.39 tok/s** streaming and **50.33 tok/s** non-streaming. LR Packed GPU replicas ran at **51.08** and **51.33 tok/s** respectively, with matching 768 IDs, stops, and per-pass commits in all four unprofiled Python pairs. This isolates the final replica change. The larger gains over Scheduled include the change of execution stack and speculation. [Stock FreeToken trace](../edge35b/results/native50/freetoken-bf16.json) · [Paired packed comparison](../edge35b/THROUGHPUT100.md#lossless-gpu-resident-expert-weights)

Use the pinned Qwen download and separate Scheduled environment from the 35B guide, then run the original Python case twice. To reproduce the packed figures and their exact checks, use the full fixed-eight and replica benchmark from the same guide. [35B download and build](qwen35b.md)

```bash
"$BASE_PYTHON" -m edge35b.native_bench --model-dir "$MODEL" \
  --modes scheduled --cases coding --repeats 2 --out "$OUT_ROOT/scheduled-coding.json"
bash edge35b/run_packed_replicas.sh
```
