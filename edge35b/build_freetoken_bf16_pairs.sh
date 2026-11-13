#!/usr/bin/env bash
# Rebuild the two changed CPU translation units and retain the pinned GPU code.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 FREETOKEN_PYTHON ISOLATED_FREETOKEN_PYTHON_SOURCE" >&2
  exit 2
fi
ft_python=$1
source_dir=$(realpath "$2")
kernel="$source_dir/freetoken/kernel"
cpu="$kernel/csrc/cpu_moe"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mapfile -t build < <("$ft_python" - <<'PY'
from pathlib import Path
import sysconfig
import torch

torch_root = Path(torch.__file__).resolve().parent
cuda = torch_root.parent / "nvidia" / "cu13"
print(sysconfig.get_paths()["include"])
print(sysconfig.get_config_var("EXT_SUFFIX"))
print(torch_root)
print(cuda)
print(int(torch.compiled_with_cxx11_abi()))
PY
)
python_include=${build[0]}
extension="$kernel/_cpu_moe${build[1]}"
torch_root=${build[2]}
cuda_root=${build[3]}
abi=${build[4]}
base=liblightreason_cpu_moe_base.so
test ! -e "$kernel/$base"
test -f "$extension"
test -f "$cuda_root/include/cuda_runtime.h"
test -f "$cuda_root/lib/libcudart.so.13"
cp "$extension" "$scratch/$base"

${CXX:-g++} -std=c++20 -O3 -fPIC -pthread -shared \
  -D_GLIBCXX_USE_CXX11_ABI="$abi" -DTORCH_EXTENSION_NAME=_cpu_moe \
  -I"$python_include" -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
  -I"$kernel/csrc/include" -I"$cuda_root/include" \
  -I"$torch_root/../triton/backends/nvidia/include" \
  "$cpu/cpu_moe_ext.cpp" "$cpu/kernels/bf16.cpp" \
  -L"$torch_root/lib" -L"$cuda_root/lib" -L"$scratch" \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$torch_root/lib" -Wl,-rpath,"$cuda_root/lib" \
  -l:"$base" -l:libcudart.so.13 -lc10 -ltorch -ltorch_cpu -ltorch_python \
  -o "$scratch/_cpu_moe${build[1]}"
cp "$extension" "$kernel/$base"
install -m 0755 "$scratch/_cpu_moe${build[1]}" "$extension"
