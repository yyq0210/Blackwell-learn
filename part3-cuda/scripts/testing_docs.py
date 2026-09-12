from pathlib import Path
import html,re,json
from docs import notes,R
EXTRA=[
(r'std::getenv','检查是否启用仅一次手写kernel的sanitizer模式；该模式跳过cuBLAS/Graph，避免工具混合路径异常，不用其时间作性能比较。','host'),
(r'return 0','测试成功返回操作系统；sanitizer模式在此结束，标准模式则继续或在main末尾隐式成功返回。','host'),

(r'#pragma once','让这个共享头文件在一个编译单元中只展开一次，避免重复声明。','host'),
(r'^<<','继续输出JSON字段或错误诊断，将相邻变量写到同一输出记录中。','host'),
(r'^: \(i <','三元表达式的后半部分：前1024个检查位置分层覆盖整个矩阵，其余使用随机位置。','host'),
(r'H \*out|float \*ref|void \*workspace','保存GPU分配的输出/参考/workspace地址；指针声明本身不分配内存。','host'),
(r'__LINE__|auto bs|^do \{|^\}.*\\|^\} while','错误检查宏的组成部分：求值一次，失败时报文件行号，并用do-while(0)保持语句语义。','host'),
(r'blas.run|lt.run','执行该性能基线一次，随后等待并完整验证；正式计时另由graph_time执行。','host'),
(r'^continue','跳过不可用的cuBLASLt候选，不把失败算法列入比较。','host'),
(r'cudaDeviceProp','保存CUDA查询到的GPU属性，如设备名与SM数量。','host'),
(r'double e =|double err =','计算当前输出元素与参考的绝对误差；后面同时检查绝对和相对误差阈值。','host'),
(r'double mx =|double ref =','把最大误差统计或CPU参考累加器初始化为0。','host'),
(r'float ms','保存CUDA event测得的毫秒数。','host'),
(r'^fn\(','调用传入graph_time的GEMM函数，用于warmup或捕获连续图节点。','host'),
(r'for \(int z =','CPU参考逐个K元素做double累加，与GPU分块实现相互独立。','host'),
(r'int bv =','准备接收cuBLAS版本号，写入结果JSON。','host'),
(r'return mx|return r','返回最大误差或计时统计对象，供调用方记录结果。','host'),
(r'times.push_back','把整张Graph的毫秒数除以节点数，记录每次GEMM的平均GPU耗时。','host'),
(r'v = H\(dist','生成一个随机浮点数并转换FP16，作为实际传给GPU的输入值。','host'),

(r'cudaMalloc','分配GPU内存；字节数按FP16=2B、FP32=4B计算。分配在计时与Graph捕获之外。','host'),
(r'cudaMemcpy','在CPU与GPU之间复制输入、输出或参考结果；方向参数决定传输方向。这些复制不计入kernel时间。','host'),
(r'cudaMemset','将输出填成NaN位模式，便于发现没有写满输出的kernel。只检查少量随机位置无法替代这一步后的全矩阵有限性检查。','host'),
(r'cudaSetDevice|cudaGetDeviceProperties','选择本机GPU0并读取GPU名称、SM数等信息。持久化grid使用实际SM数量。','host'),
(r'cudaDeviceSynchronize','等待当前GPU工作完成，以确保检查或捕获开始前没有未完成操作；同步开销不作为kernel计时。','sync'),
(r'cudaStreamBeginCapture','在per-thread stream开始CUDA Graph捕获。后面的20次launch被记录成GPU执行图，计时不包含这次捕获。','host'),
(r'cudaStreamEndCapture','结束捕获并取出包含连续kernel调用的CUDA Graph。','host'),
(r'cudaGraphInstantiate','实例化CUDA Graph，完成执行准备。实例化时间不计入GEMM延迟。','host'),
(r'cudaGraphLaunch','重放GPU工作图。用它减少每个小kernel之间的Python/C++ host发射间隙；所有对照方法使用同样流程。','host'),
(r'cudaEventRecord','在同一stream记录GPU时间戳；前后事件之间的图执行时间除以节点数，得到每次kernel的平均时间。','host'),
(r'cudaEventElapsedTime','读取两个CUDA event之间的毫秒数。返回单位是ms；TFLOPS换算中必须使用相应的1e9因子。','host'),
(r'cudaEventSynchronize|cudaStreamSynchronize','等待结束事件或stream完成，确保时间戳和输出已经可读。','sync'),
(r'cudaEventCreate|cudaEventDestroy|cudaGraph.*Destroy|cudaStream(Create|Destroy)','创建或释放计时/graph/stream资源；这些管理操作在计时区间之外。','host'),
(r'cudaFree','释放此前分配的GPU内存；不属于GEMM计算时间。','host'),
(r'cublasGemmEx','调用cuBLAS GEMM。row-major D被解释为column-major Dᵀ，所以参数顺序是B、A，操作为T、N，问题维度是N、M、K。','mma'),
(r'CUBLAS_COMPUTE_32F_PEDANTIC','FP32参考采用严格FP32累加，避免低精度参考掩盖错误；输出参考矩阵也是FP32。','mma'),
(r'CUBLAS_COMPUTE_32F','性能基线保持FP32累加，输入输出仍为FP16，与手写kernel相同。','mma'),
(r'cublasSetStream','让cuBLAS在同一per-thread stream上执行，保证Graph能捕获并使用相同计时边界。','host'),
(r'cublasSetMathMode','设置cuBLAS数学模式。本项目输入是FP16，参考另用COMPUTE_32F_PEDANTIC；不用TF32近似FP32参考。','host'),
(r'cublas(Create|Destroy|GetVersion)','管理cuBLAS handle或记录库版本；便于复现并与书中使用的版本区分。','host'),
(r'cublasLtMatmulDescCreate','创建cuBLASLt运算描述：FP32累加，alpha/beta是FP32。矩阵数据另由layout和指针提供。','host'),
(r'cublasLtMatmulDescSetAttribute','把第一操作数设为转置：存储的B为[K,N]列主序视图，转置后得到[N,K]，计算Dᵀ。','host'),
(r'cublasLtMatrixLayoutCreate','为cuBLASLt描述矩阵类型、行列数和leading dimension。本项目交换A/B来适配row-major输出。','host'),
(r'cublasLtMatmulPreference','配置或释放算法选择偏好，其中workspace上限固定为256MiB。','host'),
(r'cublasLtMatmulAlgoGetHeuristic','请求最多64个候选，实际返回数量由cuBLASLt决定；本机该形状通常返回8个，并不是64个全部有效。','host'),
(r'cublasLtMatmul\(','按选中的算法与workspace执行FP16 I/O、FP32累加GEMM；alpha1/beta0，不融合其他计算。','mma'),
(r'cublasLt(Create|Destroy)|cublasLt.*Destroy','管理cuBLASLt资源，位于kernel计时之外。','host'),
(r'graph_time','通过相同的warmup、Graph节点数、事件计时和采样流程测量给定调用。候选搜索与最终报告分别计时。','host'),
(r'std::sort','排序15个图重放平均值，报告中位数及近似10/90分位；不会把最快一次单独拿来宣称性能。','host'),
(r'Timing r|times\[7\]','第8个有序样本是15次采样的中位数；第2/14个用作近似低/高分位观察波动。','host'),
(r'std::mt19937|uniform_real_distribution','用固定或CLI指定的随机种子生成[-1,1]浮点输入，再转换FP16；不是只测试容易恰好精确的整数。','host'),
(r'isfinite','检查全部输出是否有限，防止遗漏写入的NaN、越界结果或溢出被抽样漏过。','host'),
(r'err >|e >','逐元素错误阈值是绝对误差0.005加参考值绝对值的0.002倍。这是数值近似检查，不是bitwise相等承诺。','host'),
(r'maxerr|std::max\(mx','记录最大绝对误差；完整结果也写入JSON，便于比较FP32累加与FP16舍入。','host'),
(r'ref \+=','CPU用double逐项累计独立参考，小矩阵检查所有位置，大矩阵额外抽查4096个位置。后面还有完整cuBLAS FP32参考。','host'),
(r'blas.run_ref','运行FP32输出的cuBLAS参考，随后把完整矩阵取回CPU逐元素核对。','host'),
(r'check\(p.d\)','对手写kernel的全部输出做FP32参考检查；不是只检查抽样点。','host'),
(r'check\((blas|lt).out\)','对用于性能比较的cuBLAS/cuBLASLt输出也使用同一参考和阈值检查。','host'),
(r'launch\(p\)','运行本版本的CUDA kernel。首次用于检查，后续通过graph_time测量。','host'),
(r'choices\[i\]|candidates|best_index|fastest','遍历有效cuBLASLt候选，实际计时后选择更快者；最终测量使用选定算法。候选搜索不计入GEMM时间。','host'),
(r'std::cout|setprecision','将版本、shape、正确性、完整元素数、延迟、吞吐和库版本输出为可机器读取的JSON。','host'),
(r'std::cerr|fprintf|exit\(|return [2345678]','在CUDA错误、非法shape或数值不一致时输出诊断并以非零状态退出，脚本立即停止。','host'),
(r'Problem p','封装矩阵形状、实际SM数和GPU指针；这是轻量host参数对象，不拥有矩阵内存。','host'),
(r'argc|atoi|Unsupported|m %|n %|k %|VERSION ==','解析CLI的M,N,K和可选随机种子，拒绝本版本不能整除分块的形状。','host'),
(r'std::vector','分配CPU侧输入、输出、参考或计时样本。它不是CUDA device内存。','host'),
(r'size_t checks|size_t idx|gen\(\)|r = idx','小问题对全部位置做CPU double参考，大问题选择分层与随机位置；完整验证另由FP32矩阵参考负责。','host'),
(r'auto check =','定义完整矩阵核对过程；后续同时用于手写结果和两条库基线。','host'),
(r'float alpha|float a =|beta =|float a=','设置alpha=1、beta=0，所以数学运算只有A×Bᵀ，不读取或累加旧D。','host'),
(r'for \(int sample|for \(int j =|for \(int i =','执行warmup、Graph节点录制、重复采样或候选遍历；计时范围由event边界明确限定。','host'),
(r'for \(auto','遍历CPU输入或输出，生成随机数据或检查所有元素的有限性。','host'),
(r'for \(size_t','遍历检查位置或完整参考矩阵；正确性检查发生在计时之外。','host'),
(r'cudaGraph_t|cudaGraphExec_t|cudaEvent_t|cudaStream_t|cublas.*_t','声明CUDA/BLAS资源句柄，真正创建和释放由相应API负责。','host'),
(r'BlasReference|LtReference','RAII基线对象保存handle、输出和workspace，构造/析构在计时区域之外。','host'),
(r'workspace_bytes','将cuBLASLt算法workspace上限固定为256MiB，记录为对照条件。','host'),
(r'CUDA_OK|BLAS_OK|cuda_check_status|cudaGetErrorString|#define','检查每个CUDA/BLAS API的返回值并报告失败位置，避免把错误或没有运行的kernel当作通过。','host'),
(r'group =|bm =|rows =|local =|base =|bn =','grouped-M调度的整数计算：每8行M tile为一组，组内先M后N；最后不足8行时使用实际行数。','schedule'),
(r'int group|__device__ inline void grouped_tile','设备侧tile编号映射帮助函数，无内存访问；所有WS角色都用同一个映射。','schedule'),
(r'using H','把CUTLASS的half_t命名为H，保证输入输出都是FP16。','host'),
(r'struct Problem|struct Timing','定义共享的参数/计时结果结构，不执行GPU运算。','host'),
(r'float median|int m, n|H \*a','声明结构中的字段：尺寸/SM数、设备指针，或中位数与分位计时。','host'),
(r'void run_ref|void run\(','封装一次库GEMM调用，分别用于数值参考或性能对照。','host'),
(r'if \(|if\(','根据参数、API状态或数值条件选择处理路径；失败路径必须终止当前测试。','host'),
]
import docs
docs.RULES=EXTRA+docs.RULES
INTRO='''# 测试与基准：共享 C++ 代码逐行讲解

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
'''
files=['include/common.cuh','include/runner_graph.cuh','include/benchmark.cuh']
md=INTRO;blocks=[]
for rel in files:
 ls=notes((R/rel).read_text().splitlines());md+='\n## '+rel+'\n\n|行|代码|解释|\n|---|---|---|\n';rows=[]
 for x in ls:
  md+='|'+str(x['line'])+'|`'+x['code'].strip().replace('|','\\|')+'`|'+x['note'].replace('|','\\|')+'|\n'
  rows.append('<tr><td>'+str(x['line'])+'</td><td><code>'+html.escape(x['code'])+'</code></td><td>'+html.escape(x['note'])+'</td></tr>')
 blocks.append('<h2>'+rel+'</h2><table>'+''.join(rows)+'</table>')
(R/'docs/testing.md').write_text(md)
from markdown_it import MarkdownIt
body=MarkdownIt().render(INTRO)
(R/'docs/testing.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>C++测试代码逐行讲解</title><style>body{max-width:1400px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#0f172a}table{border-collapse:collapse;width:100%;font-size:13px}td{vertical-align:top;border-bottom:1px solid #ddd;padding:8px}td:nth-child(2){white-space:pre-wrap;width:48%}a{color:#0369a1}</style><a href="index.html">← 版本目录</a>'+body+''.join(blocks)+'</html>')
print('shared docs generated; unmatched lines',len(docs.UNKNOWN))
(R/'results/testing-unclassified-lines.txt').write_text('\n'.join(sorted(docs.UNKNOWN)))
