# Reproduce the 35B BF16 benchmarks

The text-only Qwen3.6-35B-A3B target has 64.56 GiB of BF16 weights. The full download, including the checkpoint's other tensors, is about 67 GiB. Our Linux x86-64 workstation had an RTX 4090 with 24 GiB of VRAM, a Ryzen 9 5950X, 121.5 GiB of OS-visible RAM, and PCIe 4.0 ×16. The packed runner uses a 15 GiB GPU expert cache, 34 extra packed host banks, and an 880 MiB GPU replica arena. Store the checkpoint on disk so the expert banks have room in system RAM. Benchmark one request at a time on an idle GPU at stock settings. [Results and timing definitions](performance.md#qwen36-35b)

## Download and verify the models

From the repository root, choose a writable **disk-backed** directory with room for the checkpoint, build tools, and logs. The example uses `bench-35b/`, which Git ignores. The downloader verifies every file against the pinned Hub revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7` and writes the manifest used by the harness. [Qwen target](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

```bash
export WORK="$PWD/bench-35b"
export MODEL="$WORK/qwen35b" DRAFT="$WORK/dflash2" OUT_ROOT="$WORK/results"
mkdir -p "$WORK"
python3 -m edge35b.download --model-dir "$MODEL" --shard-dir "$MODEL" --workers 6
```

Scheduled runs with this checkpoint alone. For the packed runner, install the Hugging Face Hub CLI in your tools environment and download the pinned DFlash2 draft too. [Inco AI draft card](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) · [Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

```bash
hf download incoai/Qwen3.6-35B-A3B-DFlash2 config.json model.safetensors \
  --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 --local-dir "$DRAFT"
(
  cd "$DRAFT"
  sha256sum -c <<'SHA256'
86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59  config.json
d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d  model.safetensors
SHA256
)
```

## Run the original Scheduled baseline

Use a separate Python 3.12 environment for the original Transformers baseline. It runs the original eager BF16 expert operations with scheduled transfers and verifies the live text parameters against the checkpoint. The commands below exercise the same 341-ID arithmetic answer recorded in the history. [Original baseline and other workloads](../edge35b/README.md#reproduce)

```bash
python3.12 -m venv "$WORK/venv-scheduled"
BASE_PYTHON="$WORK/venv-scheduled/bin/python"
"$BASE_PYTHON" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
"$BASE_PYTHON" -m pip install 'transformers==5.7.0' 'safetensors==0.8.0'
mkdir -p "$OUT_ROOT"
"$BASE_PYTHON" -m edge35b.benchmark --model-dir "$MODEL" --modes cached,scheduled \
  --prompt 'What is 17 times 19?' --tokens 1024 --repeats 2 --out "$OUT_ROOT/scheduled.json"
```

The separate Python coding comparison uses the original 768-token cap and the same rendered input as the packed runner. [Coding prompts and reproduction](coding-benchmarks.md#35b-html-and-python)

## Prepare the pinned FreeToken runtime

The measured Python source revision is `8215a94f92449c1aff07627d95ef73377363315d`. The lab reused native extensions from the earlier `0.2.1+g14119b8cb` CPython 3.12 wheel after comparing its native source against the newer checkout. FreeToken's nightly wheel URL is rolling. The setup below builds those upstream extensions locally from the pinned source. The full patch stack applied cleanly to the public source archive. The patched C++ build and CUDA replay checks passed on the 4090 using the lab's native extensions. Validate a fresh source build before comparing its timings to the archive. [FreeToken installation and build notes](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)

Use Python 3.12, the CUDA 13.0 development toolkit (including `nvcc`), a matching NVIDIA driver, a C++20 compiler, Ninja, `patch`, and `uv`. The lab used PyTorch 2.14.1, Triton 3.8.0, Transformers 5.18.0 and `cuda-bindings==13.5.0` in this environment. The native build uses the Torch wheel's CUDA libraries and headers for the patched CPU extension. Keep this environment separate from 3B Studio.

```bash
git clone https://github.com/FlashML-org/FreeToken.git "$WORK/FreeToken"
git -C "$WORK/FreeToken" checkout --detach 8215a94f92449c1aff07627d95ef73377363315d
uv venv --python 3.12 "$WORK/venv-ft"
export FT="$WORK/venv-ft/bin/ft"
FT_PYTHON="$WORK/venv-ft/bin/python"
uv pip install --python "$FT_PYTHON" 'torch==2.14.1' 'setuptools>=77' wheel ninja
uv pip install --python "$FT_PYTHON" --no-build-isolation \
  "$WORK/FreeToken[accel]" 'transformers==5.18.0' 'triton==3.8.0' 'cuda-bindings==13.5.0'

export FT_SOURCE="$WORK/patched-python"
bash edge35b/prepare_freetoken.sh "$WORK/FreeToken" "$FT_SOURCE"
"$FT_PYTHON" - "$FT_SOURCE" <<'PY'
from pathlib import Path
import shutil
import sys
import sysconfig
import freetoken

source = Path(freetoken.__file__).resolve().parent / "kernel"
target = Path(sys.argv[1]) / "freetoken" / "kernel"
suffix = sysconfig.get_config_var("EXT_SUFFIX")
for name in ("_cpu_moe", "_pinned_tensor", "_row_store"):
    shutil.copy2(source / (name + suffix), target / (name + suffix))
PY
bash edge35b/build_freetoken_bf16_pairs.sh "$FT_PYTHON" "$FT_SOURCE"
PYTHONPATH="$PWD:$FT_SOURCE" "$FT_PYTHON" -m unittest \
  edge35b.test_gpu_resident_experts edge35b.test_dflash_prefix_selection
```

Run FreeToken's bandwidth calibration on the target machine. It compares expert compute and PCIe gathers while both use the same system memory. Match the benchmark's eight CPU workers and use the same cache directory during calibration and inference. Set `BENCH_CORES` and `MONITOR_CORES` to suitable disjoint CPU sets if your core numbering differs.

```bash
export XDG_CACHE_HOME="$WORK/cache"
export TRITON_CACHE_DIR="$WORK/triton-cache"
export BENCH_CORES=8-15 MONITOR_CORES=2-3
taskset -c "$BENCH_CORES" "$FT" bench bw --model "$MODEL" --formats bf16 --cpu-threads 8
bash edge35b/run_packed_replicas.sh
```

The benchmark warms the graphs, resets the expert cache before each request and alternates the fixed eight-position control, raw replicas and packed replicas. The four fixtures fix the system prompts and caps. It records all generated IDs, stop reasons, fetch plans, preparation, time to first streamed text and complete request time. It verifies the archived control and exits unsuccessfully if the packed candidate fails the exactness check. A different machine or library build may produce different BF16 decisions. For a port, use a fresh reference on that device and report differences from the archive separately. [Prompt suite](../edge35b/benchmark-suite.json) · [Frozen output archive](../edge35b/results/native50/width-pair-retest.json) · [Packed-replica raw measurements](../edge35b/results/native50/gpu-resident-qwen.json.gz)

To collect a fresh fixed-width reference for that port, use the built-in `--width-pair-retest` experiment with the new device's verified FreeToken tree, checkpoint, draft, and prompt suite. Use the resulting `paired-1` rows as the archive passed to `--packed-transfer-reference` for the next transport test. Keep the new device's CPU/GPU split, cache budget, and verifier shape fixed across that pair. [Width retest commands](../edge35b/WIDTH_AUTOTUNE.md#reproduce) · [Hardware porting](porting.md)

Run the selector and adaptive-width experiment with the same build and environment.

```bash
DRAFT_PREFIX_TRIAL=1 bash edge35b/run_packed_replicas.sh
```

That experiment includes saved-continuation diagnostics after all ordinary and profiled pairs. The saved IDs are used only by the diagnostic, and Qwen still verifies every token. The ordinary experiments never receive reference outputs as proposals. [Adaptive-width results](../edge35b/THROUGHPUT100.md#joint-dflash2-and-verifier-tuning) · [Raw comparisons](../edge35b/results/native50/dflash-prefix-qwen.json.gz)
