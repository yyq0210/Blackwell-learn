# Kernel 性能从何而来：第一轮问答

记录日期：2026-09-11

教材：[本地章节](http://localhost:8000/zh/chapter_performance/index.html) · [章节源码](../modern-gpu-programming-for-mlsys/zh/chapter_performance/index.md)

## 1. Roofline 的“高于拐点”，是指已经在峰值吞吐水平线上吗？

原问题：roofline 模型高于拐点，意思是水平线也就是峰值吞吐上？

**不是。这里比较的是横轴上的算术强度，更准确的说法是“位于拐点右侧”。右侧的理论上限是水平线，但实测点不一定在这条线上。**

定义：

- 横轴 I：算术强度，单位 FLOP/byte。
- 纵轴 P：性能，单位 FLOP/s。
- BW：对应内存层级的带宽，本章默认 HBM。
- P_peak：当前数据类型和计算路径对应的峰值计算吞吐。

$$
P_{roof}(I)=\min(P_{peak},BW\times I)
$$

$$
I_{ridge}=\frac{P_{peak}}{BW}
$$

```text
性能 P
  ↑
  │             拐点────────────────── 计算上限 P_peak
  │             /           ● 实测点可以低于上限
  │            /
  │           / 带宽上限 BW × I
  └──────────┼────────────────────────→ 算术强度 I
           I_ridge
```

图是定性示意。右侧仍需取两个上限中较小的计算上限，不能沿斜线继续向上推。

采用教材近似值 P_peak = 2000 TFLOP/s、BW = 8 TB/s，拐点为 250 FLOP/byte：

| 算术强度 I | BW × I | Roofline 上限 | 解释 |
| --- | --- | --- | --- |
| 100 FLOP/byte | 800 TFLOP/s | 800 TFLOP/s | 拐点左侧，带宽上限更低 |
| 500 FLOP/byte | 4000 TFLOP/s | 2000 TFLOP/s | 拐点右侧，计算上限更低 |

如果第二个 kernel 实测只有 600 TFLOP/s，它依然在拐点右侧，但只达到示例计算上限的 30%。这个 600 是教学假设，并非本机测量。

**右侧只意味着此模型给出的上限由计算吞吐决定，不证明实际 kernel 已经被 Tensor Core 峰值吞吐限制。** 同步等待、指令发射、SMEM/L2 带宽、数据布局、并行度等仍可能使它远低于上限。

还要保持口径一致：若横坐标使用理想流量计算，真实实现存在额外 HBM 流量，实际算术强度就会更低。不要把理想 AI 对应的 roofline 分类直接当成实际瓶颈诊断。

来源：[教材 Roofline 模型与优化阶梯](../modern-gpu-programming-for-mlsys/zh/chapter_performance/index.md)。本节沿用教材近似值，不是 B300 实测或精确规格。

## 2. GEMM 的算术强度公式是怎样推导的？

原问题：GEMM 的算子强度的 2N*N / 3 * 2*N*N 是咋算出来的？

这里通常称为“算术强度”（Arithmetic Intensity，AI）。教材原式分子是 **2N³**，完整括号形式为：

```text
AI = (2 * N * N * N) / (3 * 2 * N * N) = N / 3
```

分子的计算量单位是 FLOP，分母的数据量单位是 byte。

### 分子：为什么是 2N³？

对于 N×N 方阵乘法 C = A @ B：

$$
C_{ij}=\sum_{k=0}^{N-1} A_{ik}B_{kj}
$$

输出有 N² 个元素；每个输出需要沿 K=N 进行 N 次乘加。按性能分析惯例，一次 FMA 计为 2 FLOP，因此：

$$
F=\underbrace{N^2}_{输出元素数}\times
\underbrace{N}_{每个输出的乘加次数}\times
\underbrace{2}_{每次乘加的FLOP数}=2N^3
$$

精确地按点积的 N 次乘法和 N−1 次加法数，是 N²(2N−1)；GEMM 性能报告通常使用 2N³。Tensor Core 用少量指令完成很多 FLOP，因此 FLOP 数不等于指令条数。

### 分母：为什么是 3 × 2N²？

假设 A、B、输出 C 都以 FP16 存储，每个元素 2 bytes；A/B 各从 HBM 读一次，C 向 HBM 写一次：

| HBM 流量 | 元素数 | 每元素大小 | 字节数 |
| --- | --- | --- | --- |
| 读取 A | N² | 2 bytes | 2N² |
| 读取 B | N² | 2 bytes | 2N² |
| 写出 C | N² | 2 bytes | 2N² |
| 总计 | | | 6N² = 3 × 2N² |

所以：

$$
AI=\frac{2N^3\;\mathrm{FLOP}}{6N^2\;\mathrm{byte}}
=\frac{N}{3}\;\mathrm{FLOP/byte}
$$

两个“2”来源不同：分子的 2 表示乘法加加法的 FLOP 计数；分母的 2 表示 FP16 的字节数。分母的 3 表示两次整矩阵读取加一次整矩阵写出。

### 这个结果有哪些前提？

- 计算 C=A@B，或等价 beta=0 的 GEMM，不需要读取旧 C。
- A/B/C 的 HBM 存储类型都是 FP16；FP32 accumulator 若仅保存在片上，不增加这里的 HBM 字节数。
- 全局复用理想，A/B 各读取一次，没有额外 metadata、padding、临时张量或冗余 HBM 访问。
- 若缓存预热使输入来自 L2，实际 HBM 流量也可能与这个冷数据假设不同，必须明确测量层级。

一般形状和不同存储 dtype 下，理想 beta=0 模型为：

$$
AI\approx\frac{2MNK}{s_A MK+s_B KN+s_C MN}
$$

其中 s_A、s_B、s_C 是每个元素在 HBM 中的字节数。

对于方阵，若输入 FP16、输出 FP32，总流量变成 2N²+2N²+4N²=8N²，AI 约为 N/4。若输出仍为 FP16，但 beta≠0 需要再读取一次旧 C，总流量也约为 8N²，忽略低阶 epilogue 运算时 AI 同样约为 N/4。

### 不要和后面的 tile 模型混淆

N/3 是整个方阵 GEMM 在理想 HBM 流量下的估计。章节后面的 B/s 只统计一个 K-stage 的 A/B tile 读取，并忽略输出写回；B 在那里是 tile 边长，不是整个矩阵的维度 N。

两者统计范围不同，不能直接认为矛盾。Global load 也可能命中 L2，不能把所有 CTA 的 global load 字节简单等同于 HBM 实际字节。

来源：[教材常见算子的算术强度：GEMM](../modern-gpu-programming-for-mlsys/zh/chapter_performance/index.md)。

## 3. SM occupancy 说的是驻留 CTA 还是 warp？

原问题：SM 占用率描述一个 SM 上能够同时驻留多少工作，这句话是驻留 CTA 还是 warp？

**CUDA occupancy 的标准计量单位是 warp，通常表示为比例。**

$$
Occupancy=\frac{SM上驻留的active\ warps数}{该SM支持的最大active\ warps数}
$$

这里 active 表示尚未完成、驻留中的 warp；不要求它此刻就绪或正在发射指令。等待访存/barrier 的驻留 warp 也计入 occupancy。

教材“能够同时驻留多少工作”是概括性说法，更接近**理论 occupancy**：给定 kernel 资源用量和 block 大小，资源允许的最大驻留 warp 比例。

### CTA 数如何影响 occupancy？

对固定 block 大小，计算最大驻留 CTA 数后，可以换算理论 occupancy：

$$
W_{CTA}=\lceil threadsPerCTA/32\rceil
$$

$$
Occupancy_{theoretical}=
\frac{B_{resident,max}\times W_{CTA}}{W_{SM,max}}
$$

B_resident,max 要综合寄存器、SMEM、warp slots、CTA slots 等限制；不能只看硬件最大 CTA 个数。

示例：假设一个 SM 最多支持 64 个驻留 warp；每个 CTA 有 256 个线程，即 8 个 warp；资源允许每 SM 驻留 4 个 CTA，则理论 occupancy 为：

$$
\frac{4\times8}{64}=50\%
$$

若同样是 4 个 CTA，但每个 CTA 只有 128 个线程，则为 4×4/64=25%。这些是说明单位的假设配置，不是本机 kernel 测量。

因此，CTA 数是决定驻留资源和换算 warp 数的中间量；单独报告“4 个 CTA”无法确定 occupancy。

### Occupancy 不是什么？

- 不等于正在发射指令的 warp 比例。
- 不等于 Tensor Core 利用率，也不等于 achieved FLOP/s 除以峰值 FLOP/s。
- 不等于 GPU 有多少个 SM 被使用。
- 不保证越高越快：更多驻留 warp 有助于隐藏延迟，但可能以更小 tile、更少 pipeline stages 为代价。

理论 occupancy 是资源推算的上限。实际 achieved occupancy 通常是执行期间 active warp 数的时间/SM 平均比例；grid 太小、启动和收尾、负载不均等会让它低于理论值。实际指标以 profiling 工具的定义和采样范围为准。

低 occupancy 的 Tensor Core kernel 仍可能通过异步流水线让计算单元持续繁忙；高 occupancy 的 kernel 也可能有大量 warp 同时等待。

来源：[CUDA Best Practices Guide：Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)、[Calculating Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy)。

## 本轮结论

- 拐点比较的是横轴 AI；右侧的 roof 是计算上限，实测点不必达到 roof。
- GEMM 理想 AI = 计算量 / 对应层级的数据字节数，FP16 方阵特定假设下为 N/3。
- Occupancy 按驻留 warp 比例定义；CTA 数影响其上限，但不是百分比定义本身。
