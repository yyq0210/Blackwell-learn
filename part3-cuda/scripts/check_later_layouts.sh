#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd ../.. && pwd)
CUDA="$ROOT/.toolchains/cuda-13.0"
mkdir -p build results
"$CUDA/bin/nvcc" -std=c++17 -O2 -arch=sm_103a --expt-relaxed-constexpr \
  -ccbin /opt/rh/devtoolset-8/root/usr/bin/g++ \
  -I"$ROOT/cutlass/include" -Iinclude --cudart shared \
  -L"$CUDA/lib" -Xlinker -rpath -Xlinker "$CUDA/lib" \
  scripts/inspect_later_layouts.cu -o build/inspect_later_layouts
./build/inspect_later_layouts > results/later-layout-inspection.txt
cat results/later-layout-inspection.txt
