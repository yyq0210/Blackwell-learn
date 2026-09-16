# 测试与基准：共享 C++ 代码逐行讲解

所有版本用同一测试程序。不是用Python/DSL驱动GPU：输入分配、数值检查、cuBLAS调用和CUDA Graph计时均在C++中执行。

1. 固定seed的[-1,1]浮点输入，量化FP16；输出先填NaN。
2. 小问题逐元素CPU double参考，大问题额外分层/随机抽查4096位置。
3. cuBLAS以FP32累加和FP32输出生成完整参考；手写kernel、cuBLAS FP16基线和最终cuBLASLt基线的全部输出都对照它检查。
4. 容差 `abs(actual-reference) <= 0.005 + 0.002*abs(reference)`。这是误差阈值，不是逐bit一致。
5. 每种方法5次warmup；捕获20个连续调用的CUDA Graph；3次图预热；15次event采样，每次图耗时除20，取中位数。p10/p90用第2/14个排序样本近似表示波动。
6. 输入/输出地址在图重放中保持固定，属于warm replay；8k输入工作集超过缓存时也不等于“全部L2命中”。计时不含分配、初始化、descriptor构造、候选搜索与图实例化。
7. cuBLASLt请求最多64候选，实际返回数量写入JSON。对有效候选计时选优，workspace限制256MiB。普通cuBLAS与cuBLASLt都报告，判断是否beat应对照两者中更快的一条。
8. 本机未锁GPU时钟，因此记录分位和重复实验。书中的B200锁时钟94μs不可直接作为当前B300的目标基准。

## 重要约束

每个可执行文件处理一个固定shape和一组固定指针。host的静态配置缓存只适用于这个命令行测试程序，不能原样当作接受任意多次不同参数的生产API。当前教学kernel只支持分块整除形状，不实现通用尾部predication。

下面解释完整C++共享代码；这些文件与每版`.cu`共同构成可执行程序。

## include/common.cuh

|行|代码|解释|
|---|---|---|
|1|`#pragma once`|让这个共享头文件在一个编译单元中只展开一次，避免重复声明。|
|2|`#include <cstdio>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|3|`#include <cstdlib>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|4|`#include <cuda_runtime.h>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|5|`#include <cute/tensor.hpp>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|6|`#include <cute/algorithm/cooperative_copy.hpp>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|7|`#include <cute/arch/tmem_allocator_sm100.hpp>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|8|`#include <cutlass/arch/barrier.h>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|9|`#include <cutlass/cluster_launch.hpp>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|10|`#include <cutlass/half.h>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|11|`using H = cutlass::half_t;`|把CUTLASS的half_t命名为H，保证输入输出都是FP16。|
|12|`#define CUDA_OK(x)                                                             \`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|13|`do {                                                                         \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|14|`auto cuda_check_status = (x);                                              \`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|15|`if (cuda_check_status != cudaSuccess) {                                    \`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|16|`fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|17|`cudaGetErrorString(cuda_check_status));                          \`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|18|`exit(1);                                                                 \`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|19|`}                                                                          \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|20|`} while (0)`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|21|`struct Problem {`|定义共享的参数/计时结果结构，不执行GPU运算。|
|22|`int m, n, k, sms;`|声明结构中的字段：尺寸/SM数、设备指针，或中位数与分位计时。|
|23|`H *a, *b, *d;`|声明结构中的字段：尺寸/SM数、设备指针，或中位数与分位计时。|
|24|`};`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|25|`// 原书的 grouped-M rasterization：M方向8个tile一组，支持最后不足8行的一组。`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|26|`__device__ inline void grouped_tile(int ti, int mt, int nt, int &bm, int &bn) {`|设备侧tile编号映射帮助函数，无内存访问；所有WS角色都用同一个映射。|
|27|`int group = ti / (8 * nt), base = group * 8, rows = min(8, mt - base),`|grouped-M调度的整数计算：每8行M tile为一组，组内先M后N；最后不足8行时使用实际行数。|
|28|`local = ti % (8 * nt);`|grouped-M调度的整数计算：每8行M tile为一组，组内先M后N；最后不足8行时使用实际行数。|
|29|`bm = base + local % rows;`|grouped-M调度的整数计算：每8行M tile为一组，组内先M后N；最后不足8行时使用实际行数。|
|30|`bn = local / rows;`|grouped-M调度的整数计算：每8行M tile为一组，组内先M后N；最后不足8行时使用实际行数。|
|31|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|

## include/runner_graph.cuh

|行|代码|解释|
|---|---|---|
|1|`#pragma once`|让这个共享头文件在一个编译单元中只展开一次，避免重复声明。|
|2|`#include "benchmark.cuh"`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|3|`#include <algorithm>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|4|`#include <chrono>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|5|`#include <cmath>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|6|`#include <fstream>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|7|`#include <iomanip>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|8|`#include <iostream>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|9|`#include <random>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|10|`#include <vector>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|11|`// Deterministic CPU double references for every output on small cases;`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|12|`// stratified plus random checks on large cases. Full cuBLAS checking is added`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|13|`// in benchmark.cpp.`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|14|`int main(int argc, char **argv) {`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|15|`int m = argc > 1 ? atoi(argv[1]) : 128, n = argc > 2 ? atoi(argv[2]) : 128,`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|16|`k = argc > 3 ? atoi(argv[3]) : 64;`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|17|`if (m % 128 \|\| n % 128 \|\| k % 64 \|\| m < 128 \|\| n < 128 \|\| k < 64 \|\|`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|18|`(VERSION <= 2 && (m != 128 \|\| n != 128)) \|\| (VERSION == 1 && k != 64) \|\|`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|19|`(VERSION == 8 && (m % 256 \|\| n % 256)) \|\|`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|20|`(VERSION == 9 && (m % 512 \|\| n % 256))) {`|解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。|
|21|`std::cerr << "Unsupported shape for version " << VERSION << "\n";`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|22|`return 2;`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|23|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|24|`CUDA_OK(cudaSetDevice(0));`|选择本机GPU0并读取GPU名称、SM数等信息。持久化grid使用实际SM数量。|
|25|`cudaDeviceProp prop;`|保存CUDA查询到的GPU属性，如设备名与SM数量。|
|26|`CUDA_OK(cudaGetDeviceProperties(&prop, 0));`|选择本机GPU0并读取GPU名称、SM数等信息。持久化grid使用实际SM数量。|
|27|`std::vector<H> a(size_t(m) * k), b(size_t(n) * k), d(size_t(m) * n);`|分配CPU侧输入、输出、参考或计时样本。它不是CUDA device内存。|
|28|`std::mt19937 gen(argc > 4 ? atoi(argv[4]) : 20260912);`|用固定或CLI指定的随机种子生成[-1,1]浮点输入，再转换FP16；不是只测试容易恰好精确的整数。|
|29|`std::uniform_real_distribution<float> dist(-1, 1);`|用固定或CLI指定的随机种子生成[-1,1]浮点输入，再转换FP16；不是只测试容易恰好精确的整数。|
|30|`for (auto &v : a)`|遍历CPU输入或输出，生成随机数据或检查所有元素的有限性。|
|31|`v = H(dist(gen));`|生成一个随机浮点数并转换FP16，作为实际传给GPU的输入值。|
|32|`for (auto &v : b)`|遍历CPU输入或输出，生成随机数据或检查所有元素的有限性。|
|33|`v = H(dist(gen));`|生成一个随机浮点数并转换FP16，作为实际传给GPU的输入值。|
|34|`Problem p{m, n, k, prop.multiProcessorCount, nullptr, nullptr, nullptr};`|封装矩阵形状、实际SM数和GPU指针；这是轻量host参数对象，不拥有矩阵内存。|
|35|`CUDA_OK(cudaMalloc(&p.a, a.size() * 2));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|36|`CUDA_OK(cudaMalloc(&p.b, b.size() * 2));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|37|`CUDA_OK(cudaMalloc(&p.d, d.size() * 2));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|38|`CUDA_OK(cudaMemcpy(p.a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));`|在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。|
|39|`CUDA_OK(cudaMemcpy(p.b, b.data(), b.size() * 2, cudaMemcpyHostToDevice));`|在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。|
|40|`CUDA_OK(cudaMemset(p.d, 0xff, d.size() * 2));`|将输出填成NaN位模式，便于发现没有写满输出的kernel。只检查少量随机位置无法替代这一步后的全矩阵有限性检查。|
|41|`launch(p);`|运行本版本的CUDA kernel。首次用于检查，后续通过graph_time测量。|
|42|`CUDA_OK(cudaGetLastError());`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|43|`CUDA_OK(cudaDeviceSynchronize());`|等待当前GPU工作完成，以确保检查或捕获开始前没有未完成操作；同步开销不作为kernel计时。|
|44|`CUDA_OK(cudaMemcpy(d.data(), p.d, d.size() * 2, cudaMemcpyDeviceToHost));`|在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。|
|45|`double maxerr = 0;`|记录最大绝对误差；完整结果也写入JSON，便于比较FP32累加与FP16舍入。|
|46|`size_t checks = d.size() <= 65536 ? d.size() : 4096;`|小问题对全部位置做CPU double参考，大问题选择分层与随机位置；完整验证另由FP32矩阵参考负责。|
|47|`for (auto v : d)`|遍历CPU输入或输出，生成随机数据或检查所有元素的有限性。|
|48|`if (!std::isfinite(float(v))) {`|检查全部输出是否有限，防止遗漏写入的NaN、越界结果或溢出被抽样漏过。|
|49|`std::cerr << "nonfinite output\n";`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|50|`return 3;`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|51|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|52|`for (size_t i = 0; i < checks; ++i) {`|遍历检查位置或完整参考矩阵；正确性检查发生在计时之外。|
|53|`size_t idx = checks == d.size() ? i`|小问题对全部位置做CPU double参考，大问题选择分层与随机位置；完整验证另由FP32矩阵参考负责。|
|54|`: (i < 1024 ? i * (d.size() - 1) / 1023`|三元表达式的后半部分：前1024个检查位置分层覆盖整个矩阵，其余使用随机位置。|
|55|`: gen() % d.size());`|小问题对全部位置做CPU double参考，大问题选择分层与随机位置；完整验证另由FP32矩阵参考负责。|
|56|`int r = idx / n, c = idx % n;`|小问题对全部位置做CPU double参考，大问题选择分层与随机位置；完整验证另由FP32矩阵参考负责。|
|57|`double ref = 0;`|把最大误差统计或CPU参考累加器初始化为0。|
|58|`for (int z = 0; z < k; ++z)`|CPU参考逐个K元素做double累加，与GPU分块实现相互独立。|
|59|`ref += double(float(a[size_t(r) * k + z])) * float(b[size_t(c) * k + z]);`|CPU用double逐项累计独立参考，小矩阵检查所有位置，大矩阵额外抽查4096个位置。后面还有完整cuBLAS FP32参考。|
|60|`double err = std::abs(float(d[idx]) - ref);`|计算当前输出元素与参考的绝对误差；后面同时检查绝对和相对误差阈值。|
|61|`maxerr = std::max(maxerr, err);`|记录最大绝对误差；完整结果也写入JSON，便于比较FP32累加与FP16舍入。|
|62|`if (err > 0.005 + 0.002 * std::abs(ref)) {`|逐元素错误阈值是绝对误差0.005加参考值绝对值的0.002倍。这是数值近似检查，不是bitwise相等承诺。|
|63|`std::cerr << "FAIL at " << r << "," << c << " got " << float(d[idx])`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|64|`<< " ref " << ref << "\n";`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|65|`return 4;`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|66|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|67|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|68|`// Isolate one custom launch when using CUDA sanitizers; skip vendor kernels and Graphs.`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|69|`if (std::getenv("STUDY_SANITIZER_ONLY")) {`|检查是否启用仅一次手写kernel的sanitizer模式；该模式跳过cuBLAS/Graph，避免工具混合路径异常，不用其时间作性能比较。|
|70|`CUDA_OK(cudaFree(p.a)); CUDA_OK(cudaFree(p.b)); CUDA_OK(cudaFree(p.d));`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|71|`std::cout << "{\"sanitizer_only\":true,\"cpu_checked\":" << checks << "}\n";`|将版本、shape、正确性、完整元素数、延迟、吞吐和库版本输出为可机器读取的JSON。|
|72|`return 0;`|测试成功返回操作系统；sanitizer模式在此结束，标准模式则继续或在main末尾隐式成功返回。|
|73|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|74|`BlasReference blas(p);`|RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。|
|75|`blas.run_ref();`|执行该性能基线一次，随后等待并完整验证；正式计时另由graph_time执行。|
|76|`CUDA_OK(cudaDeviceSynchronize());`|等待当前GPU工作完成，以确保检查或捕获开始前没有未完成操作；同步开销不作为kernel计时。|
|77|`std::vector<float> reference(d.size());`|分配CPU侧输入、输出、参考或计时样本。它不是CUDA device内存。|
|78|`CUDA_OK(cudaMemcpy(reference.data(), blas.ref, reference.size() * 4,`|在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。|
|79|`cudaMemcpyDeviceToHost));`|在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。|
|80|`auto check = [&](H *ptr) {`|定义完整矩阵核对过程；后续同时用于手写结果和两条库基线。|
|81|`CUDA_OK(cudaMemcpy(d.data(), ptr, d.size() * 2, cudaMemcpyDeviceToHost));`|在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。|
|82|`double mx = 0;`|把最大误差统计或CPU参考累加器初始化为0。|
|83|`for (size_t i = 0; i < d.size(); ++i) {`|遍历检查位置或完整参考矩阵；正确性检查发生在计时之外。|
|84|`double e = std::abs(float(d[i]) - reference[i]);`|计算当前输出元素与参考的绝对误差；后面同时检查绝对和相对误差阈值。|
|85|`mx = std::max(mx, e);`|记录最大绝对误差；完整结果也写入JSON，便于比较FP32累加与FP16舍入。|
|86|`if (!std::isfinite(float(d[i])) \|\|`|检查全部输出是否有限，防止遗漏写入的NaN、越界结果或溢出被抽样漏过。|
|87|`e > 0.005 + 0.002 * std::abs(reference[i])) {`|逐元素错误阈值是绝对误差0.005加参考值绝对值的0.002倍。这是数值近似检查，不是bitwise相等承诺。|
|88|`std::cerr << "Full check FAILED " << i << " got " << float(d[i])`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|89|`<< " ref " << reference[i] << "\n";`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|90|`exit(7);`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|91|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|92|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|93|`return mx;`|返回最大误差或计时统计对象，供调用方记录结果。|
|94|`};`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|95|`maxerr = check(p.d);`|记录最大绝对误差；完整结果也写入JSON，便于比较FP32累加与FP16舍入。|
|96|`blas.run();`|执行该性能基线一次，随后等待并完整验证；正式计时另由graph_time执行。|
|97|`CUDA_OK(cudaDeviceSynchronize());`|等待当前GPU工作完成，以确保检查或捕获开始前没有未完成操作；同步开销不作为kernel计时。|
|98|`check(blas.out);`|对用于性能比较的cuBLAS/cuBLASLt输出也使用同一参考和阈值检查。|
|99|`LtReference lt(p);`|RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。|
|100|`lt.run();`|执行该性能基线一次，随后等待并完整验证；正式计时另由graph_time执行。|
|101|`CUDA_OK(cudaDeviceSynchronize());`|等待当前GPU工作完成，以确保检查或捕获开始前没有未完成操作；同步开销不作为kernel计时。|
|102|`check(lt.out);`|对用于性能比较的cuBLAS/cuBLASLt输出也使用同一参考和阈值检查。|
|103|`// Warm input/output replay policy, same graph size and sample count for all`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|104|`// methods.`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|105|`auto custom = graph_time([&] { launch(p); });`|通过相同的warmup、Graph节点数、事件计时和采样流程测量给定调用。候选搜索与最终报告分别计时。|
|106|`auto bt = graph_time([&] { blas.run(); });`|执行该性能基线一次，随后等待并完整验证；正式计时另由graph_time执行。|
|107|`auto lt_time = graph_time([&] { lt.run(); });`|执行该性能基线一次，随后等待并完整验证；正式计时另由graph_time执行。|
|108|`int bv = 0;`|准备接收cuBLAS版本号，写入结果JSON。|
|109|`BLAS_OK(cublasGetVersion(blas.h, &bv));`|管理cuBLAS handle或记录库版本；便于复现并与书中使用的版本区分。|
|110|`std::cout << std::setprecision(9) << "{\"version\":" << VERSION`|将版本、shape、正确性、完整元素数、延迟、吞吐和库版本输出为可机器读取的JSON。|
|111|`<< ",\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|112|`<< ",\"correct\":true,\"checked\":" << d.size()`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|113|`<< ",\"max_abs\":" << maxerr << ",\"median_ms\":" << custom.median`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|114|`<< ",\"p10_ms\":" << custom.p10 << ",\"p90_ms\":" << custom.p90`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|115|`<< ",\"tflops\":" << 2.0 * m * n * k / (custom.median * 1e9)`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|116|`<< ",\"cublas_ms\":" << bt.median`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|117|`<< ",\"cublasLt_ms\":" << lt_time.median`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|118|`<< ",\"speedup_vs_lt\":" << lt_time.median / custom.median`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|119|`<< ",\"lt_candidates\":" << lt.candidates`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|120|`<< ",\"lt_choice\":" << lt.best_index`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|121|`<< ",\"cublas_version\":" << bv`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|122|`<< ",\"cache_policy\":\"warm_replay\",\"gpu\":\"" << prop.name`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|123|`<< "\",\"sms\":" << prop.multiProcessorCount << "}\n";`|继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。|
|124|`CUDA_OK(cudaFree(p.a));`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|125|`CUDA_OK(cudaFree(p.b));`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|126|`CUDA_OK(cudaFree(p.d));`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|127|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|

## include/benchmark.cuh

|行|代码|解释|
|---|---|---|
|1|`#pragma once`|让这个共享头文件在一个编译单元中只展开一次，避免重复声明。|
|2|`#include <algorithm>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|3|`#include <cmath>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|4|`#include <cublasLt.h>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|5|`#include <cublas_v2.h>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|6|`#include <functional>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|7|`#include <vector>`|引入CUDA/CuTe或项目公共声明；这里只包含头文件，没有执行数据搬运或计算。|
|8|`#define BLAS_OK(x)                                                             \`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|9|`do {                                                                         \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|10|`auto bs = (x);                                                             \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|11|`if (bs != CUBLAS_STATUS_SUCCESS) {                                         \`|根据参数、API状态或数值条件选择处理路径；失败路径必须终止当前测试。|
|12|`fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(bs), __FILE__,         \`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|13|`__LINE__);                                                       \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|14|`exit(5);                                                                 \`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|15|`}                                                                          \`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|16|`} while (0)`|错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。|
|17|`struct Timing {`|定义共享的参数/计时结果结构，不执行GPU运算。|
|18|`float median, p10, p90;`|声明结构中的字段：尺寸/SM数、设备指针，或中位数与分位计时。|
|19|`};`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|20|`inline Timing graph_time(std::function<void()> fn, int nodes = 20) {`|通过相同的warmup、Graph节点数、事件计时和采样流程测量给定调用。候选搜索与最终报告分别计时。|
|21|`for (int i = 0; i < 5; ++i)`|执行warmup、Graph节点录制、重复采样或候选遍历；计时范围由event边界明确限定。|
|22|`fn();`|调用传入graph_time的GEMM函数，用于warmup或捕获连续图节点。|
|23|`CUDA_OK(cudaDeviceSynchronize());`|等待当前GPU工作完成，以确保检查或捕获开始前没有未完成操作；同步开销不作为kernel计时。|
|24|`cudaStream_t stream;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|25|`CUDA_OK(cudaStreamCreate(&stream));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|26|`// All launch functions use stream 0. Capture the legacy stream is prohibited;`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|27|`// capture stream is selected through per-thread default stream compilation.`|说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。|
|28|`CUDA_OK(cudaStreamDestroy(stream));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|29|`cudaGraph_t graph;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|30|`cudaGraphExec_t exec;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|31|`CUDA_OK(`|检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。|
|32|`cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));`|在per-thread stream开始CUDA Graph捕获。后面的20次launch被记录成GPU执行图，计时不包含这次捕获。|
|33|`for (int i = 0; i < nodes; ++i)`|执行warmup、Graph节点录制、重复采样或候选遍历；计时范围由event边界明确限定。|
|34|`fn();`|调用传入graph_time的GEMM函数，用于warmup或捕获连续图节点。|
|35|`CUDA_OK(cudaStreamEndCapture(cudaStreamPerThread, &graph));`|结束捕获并取出包含连续kernel调用的CUDA Graph。|
|36|`CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));`|实例化CUDA Graph，完成执行准备。实例化时间不计入GEMM延迟。|
|37|`for (int i = 0; i < 3; ++i)`|执行warmup、Graph节点录制、重复采样或候选遍历；计时范围由event边界明确限定。|
|38|`CUDA_OK(cudaGraphLaunch(exec, cudaStreamPerThread));`|重放GPU工作图。用它减少每个小kernel之间的Python/C++ host发射间隙；所有对照方法使用同样流程。|
|39|`CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));`|等待结束事件或stream完成，确保时间戳和输出已经可读。|
|40|`cudaEvent_t s, e;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|41|`CUDA_OK(cudaEventCreate(&s));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|42|`CUDA_OK(cudaEventCreate(&e));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|43|`std::vector<float> times;`|分配CPU侧输入、输出、参考或计时样本。它不是CUDA device内存。|
|44|`for (int j = 0; j < 15; ++j) {`|执行warmup、Graph节点录制、重复采样或候选遍历；计时范围由event边界明确限定。|
|45|`CUDA_OK(cudaEventRecord(s, cudaStreamPerThread));`|在同一stream记录GPU时间戳；前后事件之间的图执行时间除以节点数，得到每次kernel的平均时间。|
|46|`CUDA_OK(cudaGraphLaunch(exec, cudaStreamPerThread));`|重放GPU工作图。用它减少每个小kernel之间的Python/C++ host发射间隙；所有对照方法使用同样流程。|
|47|`CUDA_OK(cudaEventRecord(e, cudaStreamPerThread));`|在同一stream记录GPU时间戳；前后事件之间的图执行时间除以节点数，得到每次kernel的平均时间。|
|48|`CUDA_OK(cudaEventSynchronize(e));`|等待结束事件或stream完成，确保时间戳和输出已经可读。|
|49|`float ms;`|保存CUDA event测得的毫秒数。|
|50|`CUDA_OK(cudaEventElapsedTime(&ms, s, e));`|读取两个CUDA event之间的毫秒数。返回单位是ms；TFLOPS换算中必须使用相应的1e9因子。|
|51|`times.push_back(ms / nodes);`|把整张Graph的毫秒数除以节点数，记录每次GEMM的平均GPU耗时。|
|52|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|53|`std::sort(times.begin(), times.end());`|排序15个图重放平均值，报告中位数及近似10/90分位；不会把最快一次单独拿来宣称性能。|
|54|`Timing r{times[7], times[1], times[13]};`|第8个有序样本是15次采样的中位数；第2/14个用作近似低/高分位观察波动。|
|55|`CUDA_OK(cudaEventDestroy(s));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|56|`CUDA_OK(cudaEventDestroy(e));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|57|`CUDA_OK(cudaGraphExecDestroy(exec));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|58|`CUDA_OK(cudaGraphDestroy(graph));`|创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。|
|59|`return r;`|返回最大误差或计时统计对象，供调用方记录结果。|
|60|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|61|`struct BlasReference {`|RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。|
|62|`cublasHandle_t h;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|63|`float *ref;`|保存GPU分配的输出/参考/workspace地址；指针声明本身不分配内存。|
|64|`H *out;`|保存GPU分配的输出/参考/workspace地址；指针声明本身不分配内存。|
|65|`Problem p;`|封装矩阵形状、实际SM数和GPU指针；这是轻量host参数对象，不拥有矩阵内存。|
|66|`BlasReference(Problem problem) : p(problem) {`|封装矩阵形状、实际SM数和GPU指针；这是轻量host参数对象，不拥有矩阵内存。|
|67|`BLAS_OK(cublasCreate(&h));`|管理cuBLAS handle或记录库版本；便于复现并与书中使用的版本区分。|
|68|`BLAS_OK(cublasSetStream(h, cudaStreamPerThread));`|让cuBLAS在同一per-thread stream上执行，保证Graph能捕获并使用相同计时边界。|
|69|`BLAS_OK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));`|设置cuBLAS数学模式。本项目输入是FP16，参考另用COMPUTE_32F_PEDANTIC；不用TF32近似FP32参考。|
|70|`CUDA_OK(cudaMalloc(&ref, size_t(p.m) * p.n * 4));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|71|`CUDA_OK(cudaMalloc(&out, size_t(p.m) * p.n * 2));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|72|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|73|`void run_ref() {`|封装一次库GEMM调用，分别用于数值参考或性能对照。|
|74|`float a = 1, b = 0;`|设置alpha=1、beta=0，所以数学运算只有A×Bᵀ，不读取或累加旧D。|
|75|`BLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k, &a, p.b,`|调用cuBLAS GEMM。row-major D被解释为column-major Dᵀ，所以参数顺序是B、A，操作为T、N，问题维度是N、M、K。|
|76|`CUDA_R_16F, p.k, p.a, CUDA_R_16F, p.k, &b, ref,`|本行继续上方表达式的参数/类型：调用cuBLAS GEMM。row-major D被解释为column-major Dᵀ，所以参数顺序是B、A，操作为T、N，问题维度是N、M、K。|
|77|`CUDA_R_32F, p.n, CUBLAS_COMPUTE_32F_PEDANTIC,`|FP32参考采用严格FP32累加，避免低精度参考掩盖错误；输出参考矩阵也是FP32。|
|78|`CUBLAS_GEMM_DEFAULT));`|本行继续上方表达式的参数/类型：FP32参考采用严格FP32累加，避免低精度参考掩盖错误；输出参考矩阵也是FP32。|
|79|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|80|`void run() {`|封装一次库GEMM调用，分别用于数值参考或性能对照。|
|81|`float a = 1, b = 0;`|设置alpha=1、beta=0，所以数学运算只有A×Bᵀ，不读取或累加旧D。|
|82|`BLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k, &a, p.b,`|调用cuBLAS GEMM。row-major D被解释为column-major Dᵀ，所以参数顺序是B、A，操作为T、N，问题维度是N、M、K。|
|83|`CUDA_R_16F, p.k, p.a, CUDA_R_16F, p.k, &b, out,`|本行继续上方表达式的参数/类型：调用cuBLAS GEMM。row-major D被解释为column-major Dᵀ，所以参数顺序是B、A，操作为T、N，问题维度是N、M、K。|
|84|`CUDA_R_16F, p.n, CUBLAS_COMPUTE_32F,`|性能基线保持FP32累加，输入输出仍为FP16，与手写kernel相同。|
|85|`CUBLAS_GEMM_DEFAULT));`|本行继续上方表达式的参数/类型：性能基线保持FP32累加，输入输出仍为FP16，与手写kernel相同。|
|86|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|87|`~BlasReference() {`|RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。|
|88|`cudaFree(ref);`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|89|`cudaFree(out);`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|90|`cublasDestroy(h);`|管理cuBLAS handle或记录库版本；便于复现并与书中使用的版本区分。|
|91|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|92|`};`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|93|`struct LtReference {`|RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。|
|94|`cublasLtHandle_t h;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|95|`cublasLtMatmulDesc_t op;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|96|`cublasLtMatrixLayout_t a, b, c;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|97|`cublasLtMatmulPreference_t pref;`|配置或释放算法选择偏好，其中workspace上限固定为256MiB。|
|98|`cublasLtMatmulAlgo_t algo;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|99|`void *workspace;`|保存GPU分配的输出/参考/workspace地址；指针声明本身不分配内存。|
|100|`H *out;`|保存GPU分配的输出/参考/workspace地址；指针声明本身不分配内存。|
|101|`Problem p;`|封装矩阵形状、实际SM数和GPU指针；这是轻量host参数对象，不拥有矩阵内存。|
|102|`size_t workspace_bytes = 256ULL << 20;`|将cuBLASLt算法workspace上限固定为256MiB，记录为对照条件。|
|103|`int candidates = 0, best_index = -1;`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|104|`LtReference(Problem problem) : p(problem) {`|封装矩阵形状、实际SM数和GPU指针；这是轻量host参数对象，不拥有矩阵内存。|
|105|`BLAS_OK(cublasLtCreate(&h));`|管理cuBLASLt资源，位于kernel计时之外。|
|106|`BLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));`|性能基线保持FP32累加，输入输出仍为FP16，与手写kernel相同。|
|107|`cublasOperation_t trans = CUBLAS_OP_T;`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|108|`BLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,`|把第一操作数设为转置：存储的B为[K,N]列主序视图，转置后得到[N,K]，计算Dᵀ。|
|109|`&trans, sizeof(trans)));`|本行继续上方表达式的参数/类型：把第一操作数设为转置：存储的B为[K,N]列主序视图，转置后得到[N,K]，计算Dᵀ。|
|110|`BLAS_OK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, p.k, p.n, p.k));`|为cuBLASLt描述矩阵类型、行列数和leading dimension。本项目交换A/B来适配row-major输出。|
|111|`BLAS_OK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, p.k, p.m, p.k));`|为cuBLASLt描述矩阵类型、行列数和leading dimension。本项目交换A/B来适配row-major输出。|
|112|`BLAS_OK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, p.n, p.m, p.n));`|为cuBLASLt描述矩阵类型、行列数和leading dimension。本项目交换A/B来适配row-major输出。|
|113|`BLAS_OK(cublasLtMatmulPreferenceCreate(&pref));`|配置或释放算法选择偏好，其中workspace上限固定为256MiB。|
|114|`BLAS_OK(cublasLtMatmulPreferenceSetAttribute(`|配置或释放算法选择偏好，其中workspace上限固定为256MiB。|
|115|`pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,`|将cuBLASLt算法workspace上限固定为256MiB，记录为对照条件。|
|116|`sizeof(workspace_bytes)));`|将cuBLASLt算法workspace上限固定为256MiB，记录为对照条件。|
|117|`CUDA_OK(cudaMalloc(&workspace, workspace_bytes));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|118|`CUDA_OK(cudaMalloc(&out, size_t(p.m) * p.n * 2));`|分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。|
|119|`cublasLtMatmulHeuristicResult_t choices[64];`|声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。|
|120|`BLAS_OK(cublasLtMatmulAlgoGetHeuristic(h, op, a, b, c, c, pref, 64, choices,`|请求最多64个候选，实际返回数量由cuBLASLt决定；本机该形状通常返回8个，并不是64个全部有效。|
|121|`&candidates));`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|122|`float fastest = 1e30f;`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|123|`for (int i = 0; i < candidates; ++i) {`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|124|`if (choices[i].state != CUBLAS_STATUS_SUCCESS)`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|125|`continue;`|跳过不可用的cuBLASLt候选，不把失败算法列入比较。|
|126|`algo = choices[i].algo;`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|127|`auto t = graph_time([&] { run(); }, 10);`|通过相同的warmup、Graph节点数、事件计时和采样流程测量给定调用。候选搜索与最终报告分别计时。|
|128|`if (t.median < fastest) {`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|129|`fastest = t.median;`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|130|`best_index = i;`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|131|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|132|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|133|`if (best_index < 0) {`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|134|`fprintf(stderr, "No cuBLASLt candidate\n");`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|135|`exit(6);`|在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。|
|136|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|137|`algo = choices[best_index].algo;`|遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。|
|138|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|139|`void run() {`|封装一次库GEMM调用，分别用于数值参考或性能对照。|
|140|`float alpha = 1, beta = 0;`|设置alpha=1、beta=0，所以数学运算只有A×Bᵀ，不读取或累加旧D。|
|141|`BLAS_OK(cublasLtMatmul(h, op, &alpha, p.b, a, p.a, b, &beta, out, c, out, c,`|按选中的算法与workspace执行FP16 I/O、FP32累加GEMM；alpha1/beta0，不融合其他计算。|
|142|`&algo, workspace, workspace_bytes,`|将cuBLASLt算法workspace上限固定为256MiB，记录为对照条件。|
|143|`cudaStreamPerThread));`|本行继续上方表达式的参数/类型：将cuBLASLt算法workspace上限固定为256MiB，记录为对照条件。|
|144|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|145|`~LtReference() {`|RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。|
|146|`cudaFree(workspace);`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|147|`cudaFree(out);`|释放此前分配的GPU内存；不属于GEMM计算时间。|
|148|`cublasLtMatmulPreferenceDestroy(pref);`|配置或释放算法选择偏好，其中workspace上限固定为256MiB。|
|149|`cublasLtMatrixLayoutDestroy(a);`|管理cuBLASLt资源，位于kernel计时之外。|
|150|`cublasLtMatrixLayoutDestroy(b);`|管理cuBLASLt资源，位于kernel计时之外。|
|151|`cublasLtMatrixLayoutDestroy(c);`|管理cuBLASLt资源，位于kernel计时之外。|
|152|`cublasLtMatmulDescDestroy(op);`|管理cuBLASLt资源，位于kernel计时之外。|
|153|`cublasLtDestroy(h);`|管理cuBLASLt资源，位于kernel计时之外。|
|154|`}`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
|155|`};`|结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。|
