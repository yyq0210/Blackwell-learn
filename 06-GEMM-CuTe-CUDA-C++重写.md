# GEMM：CUDA C++ / CuTe重写与学习入口

本轮要求：按原书第三部分的顺序写CUDA C++（不是Python DSL），每版在本机编译运行，每份kernel配逐行中文文档和可视化，最终与cuBLAS比较。

已经整理为[独立课程文件夹](part3-cuda/README.md)。

- [本地交互入口](http://localhost:8000/study/part3-cuda/docs/index.html)
- [CuTe前置知识](part3-cuda/docs/00-CuTe前置知识.md)
- [第1版：单CTA、单K tile](part3-cuda/docs/v01_single_tile.md)
- [第9版：两个独立MMA消费者](part3-cuda/docs/v09_multi_consumer.md)
- [最终实测报告](part3-cuda/docs/performance.md)

## 问：TIR版本为什么能和cuBLAS持平？

因为作者把TMA、流水线、持久化调度、warp specialization、双CTA和多消费者复用组合成了完整kernel。TIR表达这些机制后仍生成GPU指令；CUDA C++ / CuTe也能生成相同类别的指令，语言名本身不能解释性能差距。

书中的持平是B200、4096³、锁定时钟、1000次计时下双方94μs。我们测的是B300、未锁时钟、不同cuBLAS版本，不能直接把书中的94μs当作本机cuBLAS的时间。

移植时最初把两个accumulator交给一个MMA issue warp，削弱了作者的独立消费者机制。已修正成两个issue warp，回收B的barrier等待2次MMA完成通知；另补齐8行M分组调度，并用宽TMEM load调优。单独修正issue warp没有明显提速，因此也不能把全部差距归因于这一处。

最终4096³调优版约104μs，同轮cuBLAS约93μs，尚未beat。8192³三轮对普通cuBLAS有3%–4%的小幅优势，但对cuBLASLt基本持平、有胜有负。当前证据不支持“稳定超过最强库基线”，也不支持“TIR语言天然更快”。具体源码、全部输出检查、重复数据和工具日志都在课程目录中。


## 问：第 1 版代码看不懂，能否从计算过程开始详细讲？

新增 [第一个 kernel：从一行点积读懂 CuTe 和 Blackwell](part3-cuda/docs/01-single-tile-walkthrough.md)，对应 [完整交互图文页](part3-cuda/docs/01-single-tile-walkthrough.html)。讲解包含小矩阵手算、四次 K16 累加、变量与实际存储的区别、shape/stride、SMEM swizzle 地址实例、descriptor、barrier 和每个线程的 TMEM 写回分工。原第 1 版逐行页也已链接该讲义，并修正混入后续版本概念的注释。


## 问：画图解释 CuTe API 以及各种线程编排

新增 [CuTe API 图册](part3-cuda/docs/01-cute-api-atlas.md) 和 [交互页面](part3-cuda/docs/01-cute-api-atlas.html)。五张 SVG 串起坐标选块、MMA 分层坐标、descriptor fragment、输入与输出的两种线程映射，以及 TMEM 协作源窗口到各 lane 的寄存器结果。

本版实际输入搬运者 `t=64*(row%2)+k`，输出写回者 `t=m`；`cooperative_copy<128>` 默认向量宽度为 16 bits。`mma.get_slice(0)` 在本原语中取 CTA 份额，`cp.get_slice(t)` 取线程份额。TMEM `partition_S` 表示协作源窗口，不能直接把它的元素数当成该线程接收的寄存器数量。上述结论均已结合当前 CUTLASS 进行布局/普通复制诊断，并补回原讲义。


## 问：把 v01 全过程串起来，画清 bm/bn/nk、layout，以及 A[3,18] 的 swizzle 地址

已补进[现有 v01 讲义的第 0 节](part3-cuda/docs/01-single-tile-walkthrough.md)，配[完整流程交互](part3-cuda/docs/01-single-tile-walkthrough.html#full-trace)。用同一个 A[3,18] 与 D[3,5] 贯穿 ga/pa/sa/ra/pd/acc/src/dst/rf/rh；明确 bm=bn=0、nk=1、kt=0、kb=0…3。第二张新增图逐格画出第 3 行的 sector 重排，以及 thread82 将 GMEM 元素210搬至 SMEM 元素202；线程逻辑所有权、物理槽位、元素/字节偏移分别解释。
