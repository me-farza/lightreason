#!/usr/bin/env bash
# Run with the isolated FreeToken source and pinned artifacts from PASS_DIAGNOSTICS.md.
set -euo pipefail
: "${FT:?FreeToken executable required}" "${FT_SOURCE:?Patched FreeToken Python source required}"
: "${MODEL:?Pinned Qwen checkpoint required}" "${DRAFT:?Pinned DFlash2 checkpoint required}"
: "${SUITE:?Pinned prompt suite required}" "${CONTROL:?Frozen eight-position output archive required}"
: "${OUT_ROOT:?Output directory required}"

repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
python="${FT%/*}/python"
mkdir -p -- "$OUT_ROOT"
exec 9>"$OUT_ROOT/.benchmark.lock"
flock -n 9
nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fx 'NVIDIA GeForce RTX 4090'
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active" ]]; then
    printf 'An existing CUDA workload is using the GPU.\n' >&2
    exit 1
fi
export PYTHONPATH="$repo:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192

run=$(mktemp -d "$OUT_ROOT/transport8.XXXXXX")
hot_args=()
if [[ -n "${HOT_PLAN:-}" ]]; then
    hot_args=(--hot-expert-plan "$HOT_PLAN")
    if [[ "${HOT_TUNE:-0}" == 1 ]]; then hot_args+=(--hot-expert-autotune); fi
fi
printf 'Benchmark output: %s\n' "$run"
{
    uname -r
    nvidia-smi --query-gpu=name,driver_version,pstate,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,memory.total,power.limit --format=csv
    lscpu
    free -h
    sha256sum "$MODEL/lightreason-download.json" "$DRAFT/config.json" "$SUITE" "$CONTROL"
    if [[ -n "${HOT_PLAN:-}" ]]; then sha256sum "$HOT_PLAN"; fi
} > "$run/hardware-and-inputs.txt"

taskset -c 2-3 "$python" hardware_exact10.py \
    --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 120 \
    --out "$run/cuda-check.json" \
    --monitor-command taskset -c 8-15 env LIGHTREASON_TEST_PACKED_CUDA=1 \
    "$python" -m unittest -v edge35b.test_freetoken_packed_transfer

exec taskset -c 2-3 "$python" hardware_exact10.py \
    --gpu-stop-c 70 --cpu-stop-c 75 --command-timeout-s 1200 --out "$run/watch.json" \
    --monitor-command taskset -c 8-15 "$python" -u -m edge35b.freetoken_bench \
    --ft "$FT" --model "$MODEL" --packed "$SUITE" \
    --cases studio,known-math,coding,chat --repeats 2 --width-trials "$run/width.cfg" \
    --packed-transfer --packed-transfer-width 8 --packed-transfer-reference "$CONTROL" "${hot_args[@]}" \
    --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
    --host-embedding --cache-type naive --prefill-hit-d2d --port "${PORT:-1921}" \
    --server-log "$run/server.log" --out "$run/transport.json"
