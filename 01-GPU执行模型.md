# GPU 执行模型：第一轮问答

记录日期：2026-09-11

教材：[本地章节](http://localhost:8000/zh/chapter_background/index.html) · [章节源码](../modern-gpu-programming-for-mlsys/zh/chapter_background/index.md)

## 1. 多个 CTA 驻留同一个 SM，SMEM、寄存器、Tensor Core 是共用的吗？

**物理资源来自同一个 SM，但“容量共用”“数据可互访”“计算单元共用”是不同的事。**

| 资源 | 多个驻留 CTA 如何使用 | 是否可直接读写另一个 CTA 的数据 |
| --- | --- | --- |
| SMEM | SM 的容量按 CTA 分配，每个 CTA 有自己的区域 | 普通 shared 访问不可以；同 cluster 可通过 DSMEM 机制访问 |
| 寄存器文件 | 从 SM 的资源中为线程/warp 分配寄存器，寄存器值属于各线程 | 不可以；warp shuffle 等是显式通信指令，也不意味着任意访问他人寄存器 |
| Tensor Core | 驻留 CTA 提交的矩阵运算共用计算吞吐量 | 它是计算单元，不是程序可直接寻址的数据空间 |
| CUDA Core 等执行单元 | 执行来自驻留 warp 的指令，共用执行吞吐量 | 同上 |

所以 CTA A 阻塞时，CTA B 的就绪 warp 有机会继续执行，以隐藏延迟。但增加 CTA 数量不会复制 Tensor Core，也不保证性能提升：可能增加带宽竞争，或迫使每个 CTA 使用更小的 tile。

教学例子：假设 SM 可用于分配的 SMEM 预算是 192 KiB，每个 CTA 占 64 KiB，**只看 SMEM** 可驻留 3 个 CTA。实际还受寄存器、线程/warp 数、CTA 槽位、分配粒度及其他资源约束，可能只能驻留 1～2 个。这是假设数值，不是本机资源测量结果。

来源：[CUDA Hardware Multithreading](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#hardware-multithreading)。

## 2. 为什么书里说 SMEM 是“每个 CTA（一个 SM 内）”？

**“一个 SM 内”描述物理位置，“每个 CTA”描述分配和访问作用域，两者不矛盾。**

```text
SM 0
├─ SMEM 物理容量
│  ├─ 分给 CTA A 的区域：A 内所有线程可协作访问
│  ├─ 分给 CTA B 的区域：B 内所有线程可协作访问
│  └─ 尚未分配的区域
├─ 寄存器文件：分别分配给驻留线程/warp
└─ 计算流水线：执行来自这些 CTA 的工作
```

同一个 kernel 中写 `__shared__ float tile[1024];`，每个 CTA 都会得到自己的一份 `tile`。两个 CTA 的 `tile[0]` 不是同一个存储位置，即使它们在同一个 SM 上。

书中表格可以更准确地理解成：**SMEM 物理位于 SM，按 CTA 分配；默认由该 CTA 的线程共享，cluster 内可通过 DSMEM 扩展访问范围。**

## 3. CTA 运行在哪个 SM，是如何调度的？

分成两个层次理解：

1. **CTA 放置/驻留**：启动 kernel 后，GPU 的硬件工作分发机制把待执行 CTA 放到资源足够的 SM 上。判断条件包括 kernel 的 SMEM、寄存器、线程数需求和硬件驻留上限。已有 CTA 完成并释放资源后，可接纳后续 CTA。
2. **SM 内指令发射**：CTA 的线程分成 warp，SM 内的 warp scheduler 从就绪 warp 中选择工作，向适当的执行流水线发射指令。数据依赖、barrier 和流水线资源等会影响就绪状态。

一般 CUDA kernel 没有“把 block 7 固定到 SM 3”的常规启动接口。`blockIdx` 是逻辑索引，不是 SM 编号；CTA 的执行顺序也没有保证。不要依赖 `blockIdx % SM数量` 推断实际位置。

你主要通过 block 大小、资源用量、grid 大小以及 cluster 配置影响调度条件。硬件如何精确选择目标 SM、如何打破平局，不是 CUDA 程序应依赖的契约。

这里“CTA 在单个 SM 执行”说明其线程的执行和资源归属；不要把它扩展成对抢占、上下文切换等所有底层行为的保证。

Persistent kernel 通常是让已驻留 CTA 连续领取多个逻辑 tile，从而在软件层面安排 tile 到 CTA 的分配；它仍不等于应用可以任意指定物理 SM。

来源：[CUDA Programming Model](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#programming-model)、[Hardware Multithreading](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#hardware-multithreading)。

## 4. Cluster 的 CTA 个数有限制吗？同 SM 的 CTA 就是 cluster 吗？

**有数量限制。同一个 SM 上驻留的 CTA 不会自动组成 cluster。**

Cluster 是启动时明确指定的 CTA 协作分组。CUDA 保证一个 cluster 的 CTA 在同一个 **GPC（GPU Processing Cluster，包含多个 SM 的硬件组织）** 内协同调度，支持 cluster 级同步和 DSMEM 访问。GPC 是硬件组织，thread block cluster 是程序的协作分组，它们不是同一个概念。

| 问题 | 答案 |
| --- | --- |
| 多个 CTA 在同一个 SM 上，是否自动属于一个 cluster？ | 否，物理共驻留不决定 cluster 成员关系 |
| 一个 cluster 是否必须只在一个 SM 上？ | 否，它的 CTA 可以跨 SM；保证的范围是同一个 GPC |
| Cluster 是否就是“当前恰好一起运行的 CTA”？ | 否，成员关系由启动配置确定 |
| `cta_group::2` 是否意味着所有 cluster 都只能有 2 个 CTA？ | 否，它是 tcgen05 的 CTA pair 协作模式，不是通用 cluster 大小上限 |

CUDA 可移植的 cluster 大小上限为 **8 个 CTA**。较小设备或 MIG 配置可能低于 8；某些 GPU 支持显式启用更大大小。例如官方 Blackwell 调优指南明确列出 **B200 支持非可移植的 16-CTA cluster**，需设置 `cudaFuncAttributeNonPortableClusterSizeAllowed`。

**不能据此直接宣称本机 B300 对任何 kernel 都支持 16。** 应对实际设备和 kernel 启动配置使用 `cudaOccupancyMaxPotentialClusterSize` 查询可用上限，并使用 `cudaOccupancyMaxActiveClusters` 估算活跃 cluster 数。资源用量也会影响可用并发度。

来源：[CUDA Thread Block Clusters](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#thread-block-clusters)、[Blackwell Thread Block Clusters](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#thread-block-clusters)。

## 5. 如何创建、操纵或调度 cluster？

### 指定分组大小

编译期可以用 `__cluster_dims__(2, 1, 1)` 指定每个 cluster 有 2 个 CTA；启动期也可以用 `cudaLaunchKernelEx` 和 `cudaLaunchAttributeClusterDimension` 配置。

以下是说明分组语义的最小 CUDA 示例，**尚未编译运行**：

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) example(int* out) {
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) {
        out[blockIdx.x] = static_cast<int>(cluster.block_rank());
    }
    cluster.sync();  // cluster 内所有线程必须按要求共同参与
}

// Host 端示意：out 已分配，至少容纳 8 个 int。
// example<<<8, 128>>>(out);
// grid 有 8 个 CTA，每个 cluster 有 2 个 CTA，共 4 个 cluster。
```

`gridDim` 仍然以 CTA 为单位，不是 cluster 个数；grid 的各维度必须能被对应 cluster 维度整除。若 kernel 已在编译期固定 cluster 大小，启动时不能随意改成另一种大小。

### 在 cluster 内协作

- `cg::this_cluster()`：取得当前 cluster 的协作组。
- `cluster.block_rank()`：当前 CTA 在 cluster 内的逻辑编号。
- `cluster.sync()`：cluster 级同步。
- `cluster.map_shared_rank(local_ptr, peer_rank)`：把本 CTA 的 shared 地址映射到目标 CTA 对应的 shared 存储位置，用于 DSMEM 访问。

这些 API 控制分组和协作，不指定物理 SM ID。后续 CLC、persistent scheduling 等章节讨论更高级的工作分配，仍需区分“哪个 CTA 处理哪个 tile”和“硬件把 CTA 放在哪个 SM”。

来源：[CUDA Thread Block Clusters](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#thread-block-clusters)、[Distributed Shared Memory](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#distributed-shared-memory)。

## 6. Epilogue 是什么？

Epilogue 是 kernel 的**结果收尾处理阶段**，并不是一种硬件单元。GEMM 的主循环沿 K 维不断执行矩阵乘累加；得到 accumulator 后，通常还要转换成最终输出。

例如：

```text
主循环：acc = Σ_k A[:, k] × B[k, :]
Epilogue：D = cast_fp16(ReLU(α × acc + β × C + bias))
```

Epilogue 可以包括缩放、加 bias、残差融合、激活、量化、dtype 转换和输出布局转换，具体取决于算子。

在本书的 Blackwell GEMM 中，常见路径为：

```text
TMEM accumulator → 寄存器 → 输出处理 → GMEM
                                  或 → SMEM staging → TMA store → GMEM
```

将后处理融合在 GEMM epilogue 中，可以减少中间结果写回、重读和额外 kernel launch。它也会消耗执行资源，不能仅凭“融合了”就认定一定更快。

“收尾”是相对某个输出 tile 的数据依赖关系而言；采用流水线后，一个 tile 的 epilogue 可以与其他 tile 的工作重叠，并不要求整个 grid 的所有 MMA 都结束。

来源：[教材 GPU 执行模型：GEMM 数据流水线](../modern-gpu-programming-for-mlsys/zh/chapter_background/index.md)。

## 7. DSMEM 是引擎，还是共享内存空间？

**DSMEM 是分布式共享内存及其访问地址空间，不是 TMA 那样的数据搬运引擎。**

它让 cluster 内的 CTA 可以对彼此的 SMEM 做读、写和原子操作。物理存储仍然是各 CTA 原有的 shared memory；并没有额外生成一块同样容量的新存储。

假设一个 cluster 有 4 个 CTA，每个 CTA 分配 32 KiB SMEM，则 cluster 的分布式 shared 存储总量为 128 KiB，分散在 4 个 CTA 的区域中。它不会自动变成本 CTA 的一个可以随意线性越界索引的 128 KiB 数组，跨 CTA 访问需要正确的地址映射。

```text
CTA 0 的 SMEM ←─ cluster 内远程访问能力 ─→ CTA 1 的 SMEM
```

DSMEM 描述“能访问哪些存储”；TMA 描述“谁执行特定异步数据搬运”。普通 DSMEM load/store 并不要求每次都先发起 TMA。

访问远端 shared 前必须保证目标 CTA 已存在、数据已经准备好；退出前必须保证其他 CTA 已完成相关远程访问。教材和 CUDA 示例使用 `cluster.sync()` 建立必要的协作和生命周期保证。异步路径还需遵循相应完成协议。

来源：[CUDA Distributed Shared Memory](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#distributed-shared-memory)。

## 8. TMEM 只能由 Tensor Core 写入、只能输出到寄存器吗？

**不是。前一轮介绍的路径是常见 GEMM accumulator 路径，不能理解成完整的硬件访问限制。**

| 数据路径 | 指令或用途 |
| --- | --- |
| Tensor Core → TMEM | `tcgen05.mma` 将计算结果/累加结果写入 TMEM |
| 寄存器 → TMEM | `tcgen05.st`，warp 协同执行的异步写入 |
| SMEM → TMEM | `tcgen05.cp`，支持指定 shape/layout 的异步复制 |
| TMEM → 寄存器 | `tcgen05.ld`，warp 协同执行的异步读取 |
| TMEM → Tensor Core | MMA 读取旧 accumulator；支持的形式还能读取 operand A、scale factors 等 |

本书的常规 MMA 路径中，A 可以来自 SMEM 或受支持的 TMEM 输入形式，B 通常来自 SMEM。不能据此认为任意 operand、dtype、shape 都能自由选择存储空间，具体以目标架构和 PTX 形式为准。

```text
SMEM ──tcgen05.cp──> TMEM <──tcgen05.st── 寄存器
                     │  │
                     │  └──tcgen05.ld──> 寄存器
                     │
                     ↕ 支持的输入、旧累加值与计算结果
                  Tensor Core
```

TMEM 也不是普通 `float*` 指向的可通用 load/store 空间。书中 tcgen05 路径没有通用的“直接 TMEM → GMEM store”；写回通常先读到寄存器，再直接或经 SMEM 写到 GMEM。

同步要区分：MMA 完成通知、TMEM load 完成、以及线程间可见性/顺序。`tcgen05.ld` 发出后，使用目标寄存器前要执行 `tcgen05.wait::ld`；不能以“发出了读取指令”代替“读取已经完成”。

来源：[PTX Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory)、[tcgen05.cp](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-cp)、[tcgen05.st](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-st)、[tcgen05 MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma)。

## 9. 同一个 SM 上，可以有多个 CTA 的 warp 同时执行吗？

原问题：SM 内的 warp scheduler 从就绪 warp 中选择工作，是否意味着同一个 SM 上多个 CTA 的 warp 可以同时执行？

**可以。SM 不需要先把 CTA A 整体执行完，再执行 CTA B。来自多个驻留 CTA 的 warp 可以并发推进，指令在硬件资源允许时也可以并行执行。**

要区分三个概念：

| 概念 | 含义 |
| --- | --- |
| 驻留（resident） | warp 的执行上下文和所需资源已经在 SM 上，不代表此刻正在执行指令 |
| 就绪（ready/eligible） | 下一条指令的依赖已满足，并符合当前发射条件，有机会被调度器选择 |
| 发射与执行（issue/execute） | 指令被发射后进入对应流水线；它可能跨多个周期执行，也可能启动更长的异步操作 |

现代 SM 有多个 warp scheduler。每个 scheduler 在它所负责的 warp 集合中选择就绪 warp，而不是所有 scheduler 都任意挑选整个 SM 的全部 warp。来自不同 CTA 的 warp 可以分布在这些集合中。

因此，并发包含两种情况：

1. **交错发射**：同一个 scheduler 在不同发射时刻选择不同 CTA 的 warp。A 等待访存时，B 可能得到执行机会。
2. **并行或重叠执行**：多个 scheduler 在硬件允许时发射指令；不同 warp 的指令也可同时处于不同流水线阶段。A 的已发起矩阵计算尚未完成时，B 的其他工作可能继续进行。

以下只是可能的调度示意，不是周期精确的硬件执行表：

```text
SM 0 驻留：CTA A 的 A0、A1……；CTA B 的 B0、B1……

发射时刻     scheduler S0       scheduler S1
   t0        发射 A0 的指令      发射 B0 的指令
   t1        A0 等待，选择 B1    发射 A1 的指令
```

示意中假定 A0/B1 属于 S0 的 warp 集合，B0/A1 属于 S1 的集合，且资源满足发射条件。实际分配方式、发射吞吐量和执行延迟取决于架构及指令。

**多个 CTA 驻留并不保证所有 warp 同时执行，也不保证每个周期都能发射多个 CTA 的指令。** 依赖、执行单元吞吐量、访存和 barrier 都会限制进展。CTA A 内的普通 `__syncthreads()` 也不要求不相关的 CTA B 一起等待。

来源：[CUDA Hardware Multithreading](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#hardware-multithreading)、[CUDA Compute Capabilities 中各架构的 SM 调度说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#compute-capabilities)。

## 10. GPC 是硬件单元吗？与 SM 是什么关系？

原问题：GPC 是硬件单元吗，和 SM 是什么关系？

**是。GPC（GPU Processing Cluster）是 GPU 内的硬件组织，包含多个 SM 以及相关的组织和互连资源；SM 是实际承载 CTA 执行的计算单元。**

下面是省略中间层和其他硬件模块的包含关系，SM 数量仅用于示意：

```text
GPU
├─ GPC 0
│  ├─ SM 0：驻留 CTA，包含 warp schedulers、执行单元和片上存储
│  ├─ SM 1
│  └─ …
├─ GPC 1
│  ├─ SM …
│  └─ …
└─ …
```

不能把图中层次理解为软件一一映射：

- **GPC、SM 是硬件组织。** 它们的数量和具体组成取决于 GPU 型号。
- **Thread block cluster、CTA 是软件执行/协作组织。** 其大小由 kernel 和启动配置决定。
- CTA 的线程在一个 SM 内执行；同一个 thread block cluster 的 CTA 被保证在同一个 GPC 内协同调度。
- 一个 thread block cluster 不等于整个 GPC，也不代表独占该 GPC。来自其他 cluster 的 CTA 能否共驻留取决于资源和调度限制。
- 同一个 GPC 内的任意两个 CTA 也不能因此直接通过 DSMEM 互访；它们还必须属于同一个 thread block cluster，并满足访问及同步规则。

可以记住两个映射约束：**CTA → 一个 SM；thread block cluster → 一个 GPC 内。** 它们都不是“独占”或“一一对应”的保证。

来源：[CUDA Thread Block Clusters](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#thread-block-clusters)。

## 复习题与参考答案

### 1. 两个 CTA 同驻留一个 SM，为什么不能直接用自己的 `tile[0]` 访问对方的 `tile[0]`？

**因为 shared memory 按 CTA 分配，同名数组在各 CTA 中对应独立的存储实例。** 物理上来自同一个 SM 的容量池，并不会使普通 shared 地址变成跨 CTA 的访问地址。

若两者属于同一 cluster，可通过 DSMEM 地址映射访问对方对应的位置，并配合正确同步；否则需要使用 GMEM 等合法通信途径。同 SM 驻留本身不授予跨 CTA 的 shared 访问权。

### 2. 一个 4-CTA cluster，每个 CTA 分配 32 KiB SMEM，为什么本 CTA 不能把本地 shared 指针直接当成 128 KiB 数组使用？

**128 KiB 是四份分散存储的总容量，不是本地 shared 数组的分配长度。** 本 CTA 只有自己分配的 32 KiB；直接越过这个范围不会自动切换到其他 CTA，属于非法越界访问。

跨 CTA 访问需要指定目标 CTA，例如使用 `cluster.map_shared_rank(local_ptr, peer_rank)` 映射对应地址，再在目标分配范围内访问。还必须确保远端 CTA 和数据已准备好，并在其退出前完成访问。

### 3. 改成异步 TMA/MMA 后，如果每次发出操作都立即等待，是否就已经实现搬运与计算重叠？

**没有。异步指令提供重叠的可能，实际依赖和指令安排才决定是否发生重叠。**

若每一轮都是下面的顺序，该数据流仍然串行：

```text
load(k) → 等 load(k) → MMA(k) → 等 MMA(k)
        → load(k+1) → 等 load(k+1) → MMA(k+1) → …
```

要重叠，需要在 MMA(k) 执行期间允许 load(k+1) 推进，例如采用不同 SMEM stages 的双缓冲/多缓冲，并提前发起后续搬运：

```text
阶段              预热           稳态 1          稳态 2
TMA              load(0)        load(1)         load(2)
Tensor Core       等数据         MMA(0)          MMA(1)
```

该图是依赖示意，不假设 load 与 MMA 耗时相同。实现时要保证：

- MMA 消费一个 stage 前，该 stage 的 TMA load 已完成。
- 覆盖或复用一个 stage 前，之前的 MMA 已不再读取它。
- 生产者/消费者之间正确维护 barrier、phase 和必要的内存顺序。

其他 CTA 或 warp 可能仍在执行工作，因此“本数据流没有 overlap”不等于“整个 SM 完全空闲”。另外，立刻等待也不抹去 TMA 减少线程搬运指令等其他潜在收益。

## 后续实操待验证

- 本机 B300 上，选定 kernel 和启动配置的最大 cluster 大小、活跃 cluster 数。
- CUDA/TIRx 工具链对本机编译目标的适配。
- 本笔记的 cluster 示例尚未编译运行；本轮仅完成文档核对和问答整理。
