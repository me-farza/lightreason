#!/usr/bin/env bash
# Run the published, fixed-width comparison and require exact IDs and stops.
set -euo pipefail
: "${FT:?Path to the pinned FreeToken executable required}"
: "${FT_SOURCE:?Path to the patched FreeToken Python source required}"
: "${MODEL:?Path to the verified Qwen checkpoint required}"
: "${DRAFT:?Path to the pinned DFlash2 draft required}"
: "${OUT_ROOT:?Writable benchmark output directory required}"

repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
python="${FT%/*}/python"
data=edge35b/results/native50
mkdir -p -- "$OUT_ROOT"
exec 9>"$OUT_ROOT/.benchmark.lock"
flock -n 9
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "The benchmark needs an idle GPU." >&2
  exit 1
fi
run=$(mktemp -d "$OUT_ROOT/packed-replicas.XXXXXX")
printf 'Benchmark output: %s\n' "$run"
export PYTHONPATH="$repo:$FT_SOURCE"
export OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export LIGHTREASON_BENCH_TOKEN_IDS=1 LIGHTREASON_ROUTED_PREFILL_MAX_TOKENS=80
export LIGHTREASON_SELECTED_PREFILL_MAX_TOKENS=192

joint=()
if [[ "${DRAFT_PREFIX_TRIAL:-0}" == 1 ]]; then
  joint=(--draft-prefix "$data/draft-budget-qwen.json" --draft-prefix-teacher)
fi

taskset -c "${MONITOR_CORES:-2-3}" "$python" hardware_exact10.py \
  --gpu-stop-c 70 --cpu-stop-c 75 --final-report-only --safety-sampling \
  --command-timeout-s 1800 --out "$run/watch.json" \
  --monitor-command taskset -c "${BENCH_CORES:-8-15}" "$python" -u -m edge35b.freetoken_bench \
  --ft "$FT" --model "$MODEL" --packed edge35b/benchmark-suite.json \
  --cases studio,known-math,coding,chat --repeats 2 --width-trials "$run/width.cfg" \
  --packed-transfer --packed-transfer-width 8 --packed-transfer-reference "$data/width-pair-retest.json" \
  --hot-expert-plan "$data/hot8-plan.json" --native-packed-gather cuda-8-128-default \
  --cpu-read-policy packed-both-prefetch --cpu-read-profile "$data/native-cuda-qwen.json" \
  --cpu-read-layers 34 --shared-cpu-gather cuda-4-256-cpu-tiles --shared-cpu-layers 34 \
  --cpu-read-control-layers 34 --shared-cpu-control-layers 34 --cpu-three-routes \
  --gpu-resident-plan "$data/gpu-resident-plan.json" --gpu-resident-mib 880 \
  "${joint[@]}" --draft-dir "$DRAFT" --draft-steps 7 --expert-load parallel --cache-gib 15 \
  --host-embedding --cache-type naive --prefill-hit-d2d --port "${PORT:-1924}" \
  --server-log "$run/server.log" --out "$run/benchmark.json"

if [[ "${DRAFT_PREFIX_TRIAL:-0}" != 1 ]]; then
  "$python" - "$run/benchmark.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    comparison = json.load(stream)["gpu_resident_cache"]["summary"]["gpu-packed"]
print(f"{comparison['exact_pairs']}/{comparison['pairs']} exact packed-replica pairs")
if not comparison["exact"]:
    raise SystemExit("The packed configuration diverged from the frozen control.")
PY
fi
