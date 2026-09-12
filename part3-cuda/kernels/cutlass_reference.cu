// 独立诊断基线：CUTLASS 已优化的 collective builder；不是第9步的手写移植。
#include "common.cuh"
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
using RefTile = Shape<_256, _256, _64>;
using Cluster = Shape<_2, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, RefTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, H,
    cutlass::layout::RowMajor, 8, H, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, H,
    cutlass::layout::RowMajor, 8, H, cutlass::layout::ColumnMajor, 8, float,
    RefTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
        typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                    Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
void launch(Problem p) {
  static Gemm gemm;
  static bool ready = false;
  static void *workspace = nullptr;
  if (!ready) {
    auto sa = cutlass::make_cute_packed_stride(Kernel::StrideA{},
                                               make_shape(p.m, p.k, 1));
    auto sb = cutlass::make_cute_packed_stride(Kernel::StrideB{},
                                               make_shape(p.n, p.k, 1));
    auto sd = cutlass::make_cute_packed_stride(Kernel::StrideD{},
                                               make_shape(p.m, p.n, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = p.sms;
    Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                         {p.m, p.n, p.k, 1},
                         {p.a, sa, p.b, sb},
                         {{}, p.d, sd, p.d, sd},
                         hw};
    args.epilogue.thread.alpha = 1;
    args.epilogue.thread.beta = 0;
    args.scheduler.max_swizzle_size = 8;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "can_implement %d\n", int(status));
      exit(8);
    }
    CUDA_OK(cudaMalloc(&workspace,
                       std::max(size_t(1), Gemm::get_workspace_size(args))));
    status = gemm.initialize(args, workspace, cudaStreamPerThread);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "initialize %d\n", int(status));
      exit(8);
    }
    ready = true;
  }
  auto status = gemm.run(cudaStreamPerThread);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "run %d\n", int(status));
    exit(8);
  }
}
constexpr int VERSION = 10;
#include "runner_graph.cuh"
