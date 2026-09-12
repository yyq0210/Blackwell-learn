#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
using H = cutlass::half_t;
#define CUDA_OK(x)                                                             \
  do {                                                                         \
    auto cuda_check_status = (x);                                              \
    if (cuda_check_status != cudaSuccess) {                                    \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                \
              cudaGetErrorString(cuda_check_status));                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
struct Problem {
  int m, n, k, sms;
  H *a, *b, *d;
};
// 原书的 grouped-M rasterization：M方向8个tile一组，支持最后不足8行的一组。
__device__ inline void grouped_tile(int ti, int mt, int nt, int &bm, int &bn) {
  int group = ti / (8 * nt), base = group * 8, rows = min(8, mt - base),
      local = ti % (8 * nt);
  bm = base + local % rows;
  bn = local / rows;
}
