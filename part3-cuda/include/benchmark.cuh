#pragma once
#include <algorithm>
#include <cmath>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <functional>
#include <vector>
#define BLAS_OK(x)                                                             \
  do {                                                                         \
    auto bs = (x);                                                             \
    if (bs != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(bs), __FILE__,         \
              __LINE__);                                                       \
      exit(5);                                                                 \
    }                                                                          \
  } while (0)
struct Timing {
  float median, p10, p90;
};
inline Timing graph_time(std::function<void()> fn, int nodes = 20) {
  for (int i = 0; i < 5; ++i)
    fn();
  CUDA_OK(cudaDeviceSynchronize());
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreate(&stream));
  // All launch functions use stream 0. Capture the legacy stream is prohibited;
  // capture stream is selected through per-thread default stream compilation.
  CUDA_OK(cudaStreamDestroy(stream));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_OK(
      cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < nodes; ++i)
    fn();
  CUDA_OK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
  CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < 3; ++i)
    CUDA_OK(cudaGraphLaunch(exec, cudaStreamPerThread));
  CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
  cudaEvent_t s, e;
  CUDA_OK(cudaEventCreate(&s));
  CUDA_OK(cudaEventCreate(&e));
  std::vector<float> times;
  for (int j = 0; j < 15; ++j) {
    CUDA_OK(cudaEventRecord(s, cudaStreamPerThread));
    CUDA_OK(cudaGraphLaunch(exec, cudaStreamPerThread));
    CUDA_OK(cudaEventRecord(e, cudaStreamPerThread));
    CUDA_OK(cudaEventSynchronize(e));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, s, e));
    times.push_back(ms / nodes);
  }
  std::sort(times.begin(), times.end());
  Timing r{times[7], times[1], times[13]};
  CUDA_OK(cudaEventDestroy(s));
  CUDA_OK(cudaEventDestroy(e));
  CUDA_OK(cudaGraphExecDestroy(exec));
  CUDA_OK(cudaGraphDestroy(graph));
  return r;
}
struct BlasReference {
  cublasHandle_t h;
  float *ref;
  H *out;
  Problem p;
  BlasReference(Problem problem) : p(problem) {
    BLAS_OK(cublasCreate(&h));
    BLAS_OK(cublasSetStream(h, cudaStreamPerThread));
    BLAS_OK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
    CUDA_OK(cudaMalloc(&ref, size_t(p.m) * p.n * 4));
    CUDA_OK(cudaMalloc(&out, size_t(p.m) * p.n * 2));
  }
  void run_ref() {
    float a = 1, b = 0;
    BLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k, &a, p.b,
                         CUDA_R_16F, p.k, p.a, CUDA_R_16F, p.k, &b, ref,
                         CUDA_R_32F, p.n, CUBLAS_COMPUTE_32F_PEDANTIC,
                         CUBLAS_GEMM_DEFAULT));
  }
  void run() {
    float a = 1, b = 0;
    BLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k, &a, p.b,
                         CUDA_R_16F, p.k, p.a, CUDA_R_16F, p.k, &b, out,
                         CUDA_R_16F, p.n, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT));
  }
  ~BlasReference() {
    cudaFree(ref);
    cudaFree(out);
    cublasDestroy(h);
  }
};
struct LtReference {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulAlgo_t algo;
  void *workspace;
  H *out;
  Problem p;
  size_t workspace_bytes = 256ULL << 20;
  int candidates = 0, best_index = -1;
  LtReference(Problem problem) : p(problem) {
    BLAS_OK(cublasLtCreate(&h));
    BLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    BLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                           &trans, sizeof(trans)));
    BLAS_OK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, p.k, p.n, p.k));
    BLAS_OK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, p.k, p.m, p.k));
    BLAS_OK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, p.n, p.m, p.n));
    BLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    BLAS_OK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes)));
    CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
    CUDA_OK(cudaMalloc(&out, size_t(p.m) * p.n * 2));
    cublasLtMatmulHeuristicResult_t choices[64];
    BLAS_OK(cublasLtMatmulAlgoGetHeuristic(h, op, a, b, c, c, pref, 64, choices,
                                           &candidates));
    float fastest = 1e30f;
    for (int i = 0; i < candidates; ++i) {
      if (choices[i].state != CUBLAS_STATUS_SUCCESS)
        continue;
      algo = choices[i].algo;
      auto t = graph_time([&] { run(); }, 10);
      if (t.median < fastest) {
        fastest = t.median;
        best_index = i;
      }
    }
    if (best_index < 0) {
      fprintf(stderr, "No cuBLASLt candidate\n");
      exit(6);
    }
    algo = choices[best_index].algo;
  }
  void run() {
    float alpha = 1, beta = 0;
    BLAS_OK(cublasLtMatmul(h, op, &alpha, p.b, a, p.a, b, &beta, out, c, out, c,
                           &algo, workspace, workspace_bytes,
                           cudaStreamPerThread));
  }
  ~LtReference() {
    cudaFree(workspace);
    cudaFree(out);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(h);
  }
};
