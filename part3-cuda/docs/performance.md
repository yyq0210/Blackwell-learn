# 本机实测结果与“TIR为什么能持平cuBLAS”

作者最终版能持平，靠的是完整的硬件数据路径、计算/搬运重叠和输入复用；TIR是表达这些机制的语言，不是一个比CUDA C++更快的执行引擎。CUDA C++ / CuTe也能生成tcgen05、TMA和TMEM指令，关键是最终指令、布局、流水线和测试条件是否对应。

## 原书的结论适用什么条件

原书记录的是NVIDIA B200、4096³、FP16、锁定时钟、每版计时1000次；第9版与cuBLAS同为94μs。94μs对应约1.46 PFLOP/s，不能直接当作所有Blackwell卡、所有cuBLAS版本的统一标准。原书表中第1步70ms来自将串行思路扩展到完整矩阵的baseline，不是运行128×128×64的教学kernel。

当前测试是B300、驱动580.105.08、CUDA13.0.2组件、cuBLAS13.1.0（API版本130100），没有锁时钟。使用FP16 A/B/D、FP32累加，warm replay和CUDA Graph。尚未在同一机器上重新运行作者的TIRx二进制，因此没有足够证据把剩余差距归因于语言或编译器。

## 本机4096³结果

v1/v2只计算一个输出tile，单列形状，不参与完整4096³演进比较。所有数值均通过完整FP32参考检查。每行cuBLAS/cuBLASLt来自同次进程；不同进程之间的少量波动不应当作优化效果。

|版本|M×N×K|手写 μs|TFLOP/s|cuBLAS μs|cuBLASLt μs|相对当轮更快库基线|
|---|---|---:|---:|---:|---:|---:|
|[v01_single_tile](v01_single_tile.html)|128×128×64|26.44|0.1|1.54|1.56|0.058×|
|[v02_k_loop](v02_k_loop.html)|128×128×4096|1289.44|0.1|5.58|4.38|0.003×|
|[v03_multi_cta](v03_multi_cta.html)|4096×4096×4096|15276.99|9.0|83.02|100.83|0.005×|
|[v04_tma](v04_tma.html)|4096×4096×4096|493.04|278.8|82.51|90.37|0.167×|
|[v05_double_buffer](v05_double_buffer.html)|4096×4096×4096|392.92|349.8|82.47|107.26|0.210×|
|[v06_persistent](v06_persistent.html)|4096×4096×4096|330.46|415.9|82.58|102.94|0.250×|
|[v07_warp_specialized](v07_warp_specialized.html)|4096×4096×4096|149.35|920.3|96.32|96.53|0.645×|
|[v08_two_cta](v08_two_cta.html)|4096×4096×4096|141.77|969.5|86.21|110.64|0.608×|
|[v09_multi_consumer](v09_multi_consumer.html)|4096×4096×4096|121.66|1129.7|86.70|91.28|0.713×|
|[v09_tuned](v09_tuned.html)|4096×4096×4096|103.98|1321.8|92.96|100.39|0.894×|

**结论：4096³尚未beat cuBLAS。** 宽TMEM load改善了最后一版，但不能用略快于某轮cuBLASLt的结果，忽略更快的普通cuBLAS。

## 8192³重复实验

|轮次|手写 μs|cuBLAS μs|cuBLASLt μs|相对更快库基线|
|---|---:|---:|---:|---:|
|1|835.31|865.65|825.89|0.989×|
|2|833.40|863.80|859.42|1.031×|
|3|833.99|870.10|828.35|0.993×|

三轮都比普通cuBLAS略快，约3%–4%；相对cuBLASLt则接近1×且有胜有负。加上未锁时钟、单轮分位波动，不宣称稳定超过最强库基线。8192输入工作集很大，“warm replay”表示相同地址重复执行，不表示所有数据都在L2。

## 移植过程中发现了哪些真实差异

- 最初把两个accumulator交给一个MMA issue warp轮流处理，没有保留作者的两个独立consumer。已修正为两个独立issue warp，并将输入回收barrier计数改成2。这项修正保证机制对应，但单独修改没有带来显著提速，不能说它解释了全部差距。
- 已补齐作者M方向8行分组的持久化任务编号；consumer和producer都使用同一映射。原先单纯M-major编号在更大网格上改变了cache复用距离。
- TMEM窄load和宽load是不同机器指令。最终调优版使用x32，减少epilogue指令数。反汇编可看到`LDTM.x32`、`UTCHMMA.2CTA`、`UTMALDG.2D.2CTA`，说明实际执行的确是Blackwell数据路径。
- 各版本的stage数量与epilogue子块大小明确写在源码和文档中；本项目v8选6个输入stages和64列写回块，与书中示例参数不完全相同。因此版本演进用于教学，不应把某次性能变化视为严格“只改一个变量”的消融实验。
- 单独加入CUTLASS collective builder诊断基线，也在约100μs量级，未自动超过当轮cuBLAS。换成更高层C++接口不等于免费获得最优shape配置。

因此可以说：书展示了这组优化足以在其B200测试条件下达到cuBLAS；本项目已经跑通同样的核心机制，但本机4096³实测尚未达到更快的cuBLAS配置。进一步解释差距，需要同机同条件的TIRx生成代码和指令/性能计数器对照，不能凭两张不同环境的时间表下结论。

## 正确性与检查覆盖

- 九个阶段和第9版调优版均已编译运行；前三版没有提前使用TMA，WS/双CTA/多消费者按顺序引入。
- 主测试逐元素对照cuBLAS FP32参考。4096²输出检查16,777,216个元素；8192²检查67,108,864个元素。容差与CPU参考方法见[共享测试讲解](testing.html)。
- 11组附加case覆盖不同seed、K=64/192、非方形、持久化跨tile、stage回绕和最小双CTA输出。包括K/64=3的奇数情形，避免phase只在偶数循环上成立。
- 最终调优kernel的memcheck和synccheck报告0 errors。初次混合库/Graph的racecheck触发工具内部异常；使用`STUDY_SANITIZER_ONLY=1`隔离手写kernel后，racecheck报告0 hazards。保留两份日志，不把工具异常当作一次成功检查。
- Sanitizer是对指定case的检查，不是所有shape、所有数据的形式化证明。Sanitizer日志中的计时受插桩影响，不能进入性能比较表。
- 交互页脚本通过V8加轻量DOM模型检查：10页的初始化、源码行点击、搜索、K推进、consumer切换及tile边界映射。静态SVG也作为独立文件提供；该检查不等于真实浏览器截图对比。

## 复现

```bash
cd /home/hadoop-scale-llm/yyq/blackwell-study/part3-cuda
./scripts/validate_all.sh
./scripts/edge_cases.sh
STUDY_SANITIZER_ONLY=1 ../../.toolchains/cuda-13.0/compute-sanitizer/compute-sanitizer \
  --tool racecheck --error-exitcode 99 ./build/v09_tuned 1024 512 192
```

每个命令行进程只处理一个shape/一组指针。当前实现不支持任意尾部尺寸、任意stride或不同的融合epilogue，也不包含端到端数据分配/搬运耗时。

原书：[最终优化结果](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/ebccca2e5675966f68fb3d4880d4448194bd638d/zh/chapter_gemm_advanced/index.md#完整优化结果)。CuTe C++ API参考：[NVIDIA官方Blackwell tutorials](https://github.com/NVIDIA/cutlass/tree/v4.6.0/examples/cute/tutorial/blackwell)。
