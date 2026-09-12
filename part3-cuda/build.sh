#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
CUDA="$ROOT/.toolchains/cuda-13.0"
mkdir -p build results
"$CUDA/bin/nvcc" --default-stream per-thread -std=c++17 -O3 -arch=sm_103a --expt-relaxed-constexpr \
 -ccbin /opt/rh/devtoolset-8/root/usr/bin/g++ \
 -I"$ROOT/cutlass/include" -I"$ROOT/cutlass/tools/util/include" -Iinclude \
 --cudart shared -L"$CUDA/lib" -Xlinker -rpath -Xlinker "$CUDA/lib" \
 -lcublas -lcublasLt -Xptxas=-v "kernels/$1.cu" -o "build/$1" 2>"results/$1.compile.log"
