# 第三部分：Blackwell GEMM 的 CUDA C++ / CuTe 重写

**阅读入口：[本地交互课程](http://localhost:8000/study/part3-cuda/docs/index.html)**。也可直接打开 [docs/index.html](docs/index.html)。

这里对应原书第三部分的GEMM三章。包含9个独立CUDA C++ kernel、1个第9版调优变体，以及每版的逐行中文讲解和交互可视化。所有kernel已在本机B300上编译运行；没有Python DSL/PyTorch依赖。

v02–v09及调优版已各补充完整计算流程图与机制放大图，共18张SVG。每版从上一版的变化出发，跟踪具体元素与layout视图，并说明线程分工、缓冲复用、barrier及复习答案。交互页可修改nk、kt、任务轮数、consumer、peer和输出ei，观察坐标与phase如何联动。

- [CuTe C++前置知识](docs/00-CuTe前置知识.md)
- [实测结果与“TIR为何能持平cuBLAS”](docs/performance.md)
- [共享C++测试代码逐行讲解](docs/testing.md)
- [额外CUTLASS collective诊断基线](docs/cutlass_reference.md)

## 计算契约

`D[m,n] = sum_k A[m,k] * B[n,k]`。A[M,K]、B[N,K]、D[M,N]均为row-major FP16，Tensor Core做FP32累加，写回转换FP16。数学运算是`A × Bᵀ`，alpha1、beta0。

只支持当前tile整除的尺寸，不实现任意尾部、任意stride或融合运算。一个命令行进程对应一个shape和一组固定指针。

|版本|独立源码|逐行讲义|核心变化|
|---|---|---|---|
|1|[v01_single_tile.cu](kernels/v01_single_tile.cu)|[讲义](docs/v01_single_tile.md)|单CTA、单K tile，普通copy|
|2|[v02_k_loop.cu](kernels/v02_k_loop.cu)|[讲义](docs/v02_k_loop.md)|K-loop累加|
|3|[v03_multi_cta.cu](kernels/v03_multi_cta.cu)|[讲义](docs/v03_multi_cta.md)|输出空间多CTA分块|
|4|[v04_tma.cu](kernels/v04_tma.cu)|[讲义](docs/v04_tma.md)|TMA load/store，立即等待|
|5|[v05_double_buffer.cu](kernels/v05_double_buffer.cu)|[讲义](docs/v05_double_buffer.md)|双缓冲预取|
|6|[v06_persistent.cu](kernels/v06_persistent.cu)|[讲义](docs/v06_persistent.md)|持久化、8行M分组调度|
|7|[v07_warp_specialized.cu](kernels/v07_warp_specialized.cu)|[讲义](docs/v07_warp_specialized.md)|producer/MMA/epilogue分工|
|8|[v08_two_cta.cu](kernels/v08_two_cta.cu)|[讲义](docs/v08_two_cta.md)|2-CTA MMA，256×256输出|
|9|[v09_multi_consumer.cu](kernels/v09_multi_consumer.cu)|[讲义](docs/v09_multi_consumer.md)|两个独立MMA issue warp共享B，512×256输出|
|9T|[v09_tuned.cu](kernels/v09_tuned.cu)|[讲义](docs/v09_tuned.md)|宽TMEM load，减少epilogue指令|

## 结果

4096³的调优版约104μs，同轮cuBLAS约93μs，**未beat cuBLAS**。8192³重复3轮比普通cuBLAS略快约3%–4%，但与cuBLASLt基本持平、有胜有负，不能宣称稳定超过最强库基线。完整数字、波动和条件见[实测报告](docs/performance.md)。

九版与调优版均做完整FP32参考检查；另有11组不同seed/shape/短K验证。最终kernel通过指定case的memcheck、synccheck和隔离racecheck。工具日志中的插桩计时不用于性能报告。

## 环境与编译

- GPU0：NVIDIA B300 SXM6 AC，148 SM，目标`sm_103a`；驱动580.105.08。
- CUTLASS v4.6.0，commit `e6233cbac5d7c7a865c19c91cd684ceece19513c`，位于工作区`cutlass/`。
- 独立CUDA13.0.2组件，位于工作区`.toolchains/cuda-13.0`，不替换系统CUDA12.2。
- GCC8.3.1：`/opt/rh/devtoolset-8/root/usr/bin/g++`，C++17。
- cuBLAS API版本130100。NVIDIA重分发包的URL与SHA256已锁定在`results/cuda-13.0.2-components.json`。

```bash
cd /home/hadoop-scale-llm/yyq/blackwell-study/part3-cuda
./build.sh v01_single_tile
./build/v01_single_tile 128 128 64
./build.sh v09_tuned
./build/v09_tuned 4096 4096 4096
# 第4个位置参数可指定不同随机种子
./build/v09_tuned 8192 4096 192 12345
```

重建独立工具链：`python3 fetch_toolchain.py`（只写工作区；校验已下载包）。

## 验证与产物

```bash
./scripts/validate_all.sh  # 顺序编译，每版运行完整参考与Graph基准
./scripts/edge_cases.sh   # 短K、非方形、跨tile stage/phase复用
STUDY_SANITIZER_ONLY=1 ../../.toolchains/cuda-13.0/compute-sanitizer/compute-sanitizer \
  --tool racecheck --error-exitcode 99 build/v09_tuned 1024 512 192
```

- `results/final/`：统一计时口径的主结果；`results/edge/`：附加形状；`results/repeats/`：8192³重复实验。
- `results/*.compile.log`：ptxas寄存器、spill与编译诊断；`results/*.log`：检查日志。
- `results/v09_tuned.sass`：实际机器码，包含2-CTA TMA/MMA和LDTM.x32。
- `docs/`：逐行Markdown、独立SVG与无需前端依赖的交互HTML。

每个`.cu`包含完整算法；公共参数和测试入口在`include/common.cuh`、`include/runner_graph.cuh`与`include/benchmark.cuh`。开发时的共享模板保留在`include/basic.cuh`、`include/tma.cuh`、`include/warp_specialized.cuh`，**不是阅读独立源码的必经路径**。`scripts/materialize.py`会从模板重新生成所有版本，覆盖手动修改的对应`.cu`；学习时直接修改并编译某一份`.cu`即可。

后续版本的布局证据见 [坐标诊断结果](results/later-layout-inspection.txt)：`./scripts/check_later_layouts.sh` 在B300上用单个GPU线程检查单/双CTA分片及窄/宽load的坐标映射，不执行MMA/TMEM数据指令。讲义中的性能数据沿用原实测，本轮文档更新没有重新计时。

生成讲义使用现有docs312环境，运行 `./build_docs.sh` 即可重建所有页面与校验链接。C++编译不依赖这些Python文档工具。

API参考：[NVIDIA CuTe Blackwell C++ tutorials](https://github.com/NVIDIA/cutlass/tree/v4.6.0/examples/cute/tutorial/blackwell)。源码版本及许可见`THIRD_PARTY_NOTICES.md`。
