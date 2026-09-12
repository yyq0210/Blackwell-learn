#pragma once
#include "benchmark.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
// Deterministic CPU double references for every output on small cases;
// stratified plus random checks on large cases. Full cuBLAS checking is added
// in benchmark.cpp.
int main(int argc, char **argv) {
  int m = argc > 1 ? atoi(argv[1]) : 128, n = argc > 2 ? atoi(argv[2]) : 128,
      k = argc > 3 ? atoi(argv[3]) : 64;
  if (m % 128 || n % 128 || k % 64 || m < 128 || n < 128 || k < 64 ||
      (VERSION <= 2 && (m != 128 || n != 128)) || (VERSION == 1 && k != 64) ||
      (VERSION == 8 && (m % 256 || n % 256)) ||
      (VERSION == 9 && (m % 512 || n % 256))) {
    std::cerr << "Unsupported shape for version " << VERSION << "\n";
    return 2;
  }
  CUDA_OK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  std::vector<H> a(size_t(m) * k), b(size_t(n) * k), d(size_t(m) * n);
  std::mt19937 gen(argc > 4 ? atoi(argv[4]) : 20260912);
  std::uniform_real_distribution<float> dist(-1, 1);
  for (auto &v : a)
    v = H(dist(gen));
  for (auto &v : b)
    v = H(dist(gen));
  Problem p{m, n, k, prop.multiProcessorCount, nullptr, nullptr, nullptr};
  CUDA_OK(cudaMalloc(&p.a, a.size() * 2));
  CUDA_OK(cudaMalloc(&p.b, b.size() * 2));
  CUDA_OK(cudaMalloc(&p.d, d.size() * 2));
  CUDA_OK(cudaMemcpy(p.a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(p.b, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(p.d, 0xff, d.size() * 2));
  launch(p);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(d.data(), p.d, d.size() * 2, cudaMemcpyDeviceToHost));
  double maxerr = 0;
  size_t checks = d.size() <= 65536 ? d.size() : 4096;
  for (auto v : d)
    if (!std::isfinite(float(v))) {
      std::cerr << "nonfinite output\n";
      return 3;
    }
  for (size_t i = 0; i < checks; ++i) {
    size_t idx = checks == d.size() ? i
                                    : (i < 1024 ? i * (d.size() - 1) / 1023
                                                : gen() % d.size());
    int r = idx / n, c = idx % n;
    double ref = 0;
    for (int z = 0; z < k; ++z)
      ref += double(float(a[size_t(r) * k + z])) * float(b[size_t(c) * k + z]);
    double err = std::abs(float(d[idx]) - ref);
    maxerr = std::max(maxerr, err);
    if (err > 0.005 + 0.002 * std::abs(ref)) {
      std::cerr << "FAIL at " << r << "," << c << " got " << float(d[idx])
                << " ref " << ref << "\n";
      return 4;
    }
  }
  // Isolate one custom launch when using CUDA sanitizers; skip vendor kernels and Graphs.
  if (std::getenv("STUDY_SANITIZER_ONLY")) {
    CUDA_OK(cudaFree(p.a)); CUDA_OK(cudaFree(p.b)); CUDA_OK(cudaFree(p.d));
    std::cout << "{\"sanitizer_only\":true,\"cpu_checked\":" << checks << "}\n";
    return 0;
  }
  BlasReference blas(p);
  blas.run_ref();
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> reference(d.size());
  CUDA_OK(cudaMemcpy(reference.data(), blas.ref, reference.size() * 4,
                     cudaMemcpyDeviceToHost));
  auto check = [&](H *ptr) {
    CUDA_OK(cudaMemcpy(d.data(), ptr, d.size() * 2, cudaMemcpyDeviceToHost));
    double mx = 0;
    for (size_t i = 0; i < d.size(); ++i) {
      double e = std::abs(float(d[i]) - reference[i]);
      mx = std::max(mx, e);
      if (!std::isfinite(float(d[i])) ||
          e > 0.005 + 0.002 * std::abs(reference[i])) {
        std::cerr << "Full check FAILED " << i << " got " << float(d[i])
                  << " ref " << reference[i] << "\n";
        exit(7);
      }
    }
    return mx;
  };
  maxerr = check(p.d);
  blas.run();
  CUDA_OK(cudaDeviceSynchronize());
  check(blas.out);
  LtReference lt(p);
  lt.run();
  CUDA_OK(cudaDeviceSynchronize());
  check(lt.out);
  // Warm input/output replay policy, same graph size and sample count for all
  // methods.
  auto custom = graph_time([&] { launch(p); });
  auto bt = graph_time([&] { blas.run(); });
  auto lt_time = graph_time([&] { lt.run(); });
  int bv = 0;
  BLAS_OK(cublasGetVersion(blas.h, &bv));
  std::cout << std::setprecision(9) << "{\"version\":" << VERSION
            << ",\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
            << ",\"correct\":true,\"checked\":" << d.size()
            << ",\"max_abs\":" << maxerr << ",\"median_ms\":" << custom.median
            << ",\"p10_ms\":" << custom.p10 << ",\"p90_ms\":" << custom.p90
            << ",\"tflops\":" << 2.0 * m * n * k / (custom.median * 1e9)
            << ",\"cublas_ms\":" << bt.median
            << ",\"cublasLt_ms\":" << lt_time.median
            << ",\"speedup_vs_lt\":" << lt_time.median / custom.median
            << ",\"lt_candidates\":" << lt.candidates
            << ",\"lt_choice\":" << lt.best_index
            << ",\"cublas_version\":" << bv
            << ",\"cache_policy\":\"warm_replay\",\"gpu\":\"" << prop.name
            << "\",\"sms\":" << prop.multiProcessorCount << "}\n";
  CUDA_OK(cudaFree(p.a));
  CUDA_OK(cudaFree(p.b));
  CUDA_OK(cudaFree(p.d));
}
