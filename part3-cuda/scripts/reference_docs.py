from pathlib import Path
import html,json
import docs
from testing_docs import EXTRA
from markdown_it import MarkdownIt
R=docs.R
rules=[
(r'CollectiveBuilder|CollectiveOp','使用CUTLASS collective builder按以下模板参数构造MMA主循环或epilogue。内部已有专用流水线与同步实现，不等于逐行复刻原书。','host'),
(r'using RefTile|using Cluster','定义256×256×64 MMA tile及2×1×1 CTA cluster。','host'),
(r'cutlass::arch::Sm100|OpClassTensorOp','选择Blackwell Tensor Core架构类。nvcc仍以sm_103a编译，支持本机B300。','host'),
(r'Layout|layout::','描述FP16输入的row-major A、column-major B数学视图，以及row-major输出；alignment8表示16-byte对齐。','host'),
(r'EpilogueTileAuto|EpilogueScheduleAuto|KernelScheduleAuto','让CUTLASS选择合法的epilogue tile或流水线策略，不能保证自动得到当前shape的全局最优性能。','host'),
(r'StageCountAutoCarveout','根据epilogue占用扣除SMEM后，自动计算可用的输入pipeline stage数。','smem'),
(r'GemmUniversal|GemmUniversalAdapter|using Kernel|using Gemm','组装GPU kernel，并用device adapter提供参数初始化、workspace查询和launch入口。','host'),
(r'make_cute_packed_stride','从矩阵形状计算CUTLASS所需的紧凑stride。B的张量逻辑次序按[N,K,L]描述，数学矩阵仍是[K,N]列主序。','host'),
(r'KernelHardwareInfo|hw\.','给CUTLASS调度器提供设备编号和真实SM数。','schedule'),
(r'Gemm::Arguments|GemmUniversalMode','组装GEMM参数：problem shape、A/B地址与stride、epilogue输入输出、硬件信息。','host'),
(r'args.epilogue','alpha1/beta0，保持D=A×Bᵀ，不读取旧C作为有效输入。','host'),
(r'args.scheduler','设置调度swizzle上限8，以改善L2任务局部性。','schedule'),
(r'can_implement','检查参数是否满足kernel限制；失败时停止，不把没有执行当作性能结果。','host'),
(r'gemm.initialize','建立kernel参数与workspace，仅在首次调用做。当前命令行程序每进程只处理一组固定指针/shape。','host'),
(r'gemm.run','在相同per-thread stream上启动CUTLASS kernel，便于用同一Graph基准比较。','host'),
(r'get_workspace_size','查询CUTLASS需要的workspace并分配，位于计时区域之外。','host'),
(r'static Gemm|static bool|static void|ready =','复用adapter初始化结果，不在每次被计时的kernel调用中重建资源。','host'),
]
docs.RULES=rules+EXTRA+docs.RULES
src=R/'kernels/cutlass_reference.cu';ls=docs.notes(src.read_text().splitlines())
intro='''# 独立CUTLASS C++诊断基线

这份代码调用CUTLASS成熟的collective builder，用来检查“换成库内已优化流水线后是什么水平”。它明确作为额外基线，不冒充原书第9版的手写实现。它与九个版本使用相同输入和测试程序。

固定配置为FP16输入/输出、FP32累加、256×256×64 tile、2×1×1 cluster，主循环和epilogue schedule由CUTLASS选择。当前测得约100μs，仍未超过当轮约91μs的cuBLAS；这说明仅改用高层builder不能自动解决剩余差距。

```bash
./build.sh cutlass_reference
./build/cutlass_reference 4096 4096 4096
```

数据路径：GMEM → TMA pipeline → SMEM → tcgen05 → TMEM → CUTLASS epilogue → D。内部阶段和warp调度由所选collective提供；详细实现可沿builder生成类型进入NVIDIA头文件，而不是把全部库内部实现伪装成几十行手写kernel。
'''
md=intro+'\n|行|代码|解释|\n|---|---|---|\n';trs=[]
for x in ls:
 md+='|'+str(x['line'])+'|`'+x['code'].strip().replace('|','\\|')+'`|'+x['note']+'|\n';trs.append('<tr><td>'+str(x['line'])+'</td><td><code>'+html.escape(x['code'])+'</code></td><td>'+html.escape(x['note'])+'</td></tr>')
(R/'docs/cutlass_reference.md').write_text(md)
(R/'docs/cutlass_reference.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CUTLASS诊断基线</title><style>body{max-width:1300px;margin:30px auto;padding:20px;font:16px/1.8 system-ui}table{border-collapse:collapse;font-size:13px}td{padding:8px;border-bottom:1px solid #ddd;vertical-align:top}td:nth-child(2){white-space:pre-wrap;width:48%}a{color:#0369a1}</style><a href="index.html">← 目录</a>'+MarkdownIt().render(intro)+'<img width="100%" src="v08_two_cta.svg" alt="双CTA数据路径示意；具体内部调度由CUTLASS决定"><table>'+''.join(trs)+'</table></html>')
(R/'docs/00-CuTe前置知识.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CuTe前置知识</title><style>body{max-width:1050px;margin:30px auto;padding:20px;font:16px/1.8 system-ui;color:#0f172a}pre{background:#f1f5f9;padding:18px;overflow:auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}a{color:#0369a1}</style><a href="index.html">← 目录</a>'+MarkdownIt().enable('table').render((R/'docs/00-CuTe前置知识.md').read_text())+'</html>')
print('reference and prerequisites docs ready')
