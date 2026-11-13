#!/usr/bin/env bash
# Apply the measured patch stack to a fresh copy of the pinned FreeToken Python tree.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PINNED_FREETOKEN_CHECKOUT NEW_RUNTIME_DIRECTORY" >&2
  exit 2
fi

source_dir="$1/python/freetoken"
destination=$2
here=$(cd -- "$(dirname -- "$0")" && pwd)
test -f "$source_dir/engine/engine.py"
mkdir -- "$destination"
cp -R -- "$source_dir" "$destination/freetoken"
# An existing checkout can contain editable-build symlinks or in-place extensions.
# The benchmark copies native modules from the selected Python environment.
find "$destination/freetoken" \( -name '*.so' -o -name '*.so.*' -o -name '*.pyd' \) -delete

patches=(
  routed_prefill selected_prefill benchmark_token_ids
  bf16_pairs width_autotune bf16_pairs_reduce expert_timing_native pass_diagnostics
  packed_bf16_transfer gather_evolution native_packed_gather cpu_reads cpu_reads_runtime
  shared_cpu_gather draft_budget cpu_three_routes gpu_resident_cache dflash_prefix
)
for name in "${patches[@]}"; do
  patch --batch --fuzz=0 -p1 -d "$destination" < "$here/freetoken_$name.patch"
done
install -m 0644 "$here/cpu_bf16_reads.h" \
  "$destination/freetoken/kernel/csrc/cpu_moe/lightreason_cpu_bf16_reads.h"
