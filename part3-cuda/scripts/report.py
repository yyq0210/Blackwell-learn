from pathlib import Path
import json,html,statistics,hashlib,subprocess
from doc_content import TITLES
from docs import NAMES,R
from markdown_it import MarkdownIt
rows=[]
for name in NAMES:
 p=R/'results/final'/f'{name}.json'
 if p.exists():rows.append((name,json.loads(p.read_text())))
rep=[json.loads(p.read_text()) for p in sorted((R/'results/repeats').glob('v09_tuned_8192_r*.json'))]
body='''# 本机实测结果与“TIR为什么能持平cuBLAS”

作者最终版能持平，靠的是完整的硬件数据路径、计算/搬运重叠和输入复用；TIR是表达这些机制的语言，不是一个比CUDA C++更快的执行引擎。CUDA C++ / CuTe也能生成tcgen05、TMA和TMEM指令，关键是最终指令、布局、流水线和测试条件是否对应。

## 原书的结论适用什么条件

原书记录的是NVIDIA B200、4096³、FP16、锁定时钟、每版计时1000次；第9版与cuBLAS同为94μs。94μs对应约1.46 PFLOP/s，不能直接当作所有Blackwell卡、所有cuBLAS版本的统一标准。原书表中第1步70ms来自将串行思路扩展到完整矩阵的baseline，不是运行128×128×64的教学kernel。

当前测试是B300、驱动580.105.08、CUDA13.0.2组件、cuBLAS13.1.0（API版本130100），没有锁时钟。使用FP16 A/B/D、FP32累加，warm replay和CUDA Graph。尚未在同一机器上重新运行作者的TIRx二进制，因此没有足够证据把剩余差距归因于语言或编译器。

## 本机4096³结果

v1/v2只计算一个输出tile，单列形状，不参与完整4096³演进比较。所有数值均通过完整FP32参考检查。每行cuBLAS/cuBLASLt来自同次进程；不同进程之间的少量波动不应当作优化效果。

|版本|M×N×K|手写 μs|TFLOP/s|cuBLAS μs|cuBLASLt μs|相对当轮更快库基线|
|---|---|---:|---:|---:|---:|---:|
'''
for name,r in rows:
 ratio=min(r['cublas_ms'],r['cublasLt_ms'])/r['median_ms']
 body+=f"|[{name}]({name}.html)|{r['m']}×{r['n']}×{r['k']}|{r['median_ms']*1000:.2f}|{r['tflops']:.1f}|{r['cublas_ms']*1000:.2f}|{r['cublasLt_ms']*1000:.2f}|{ratio:.3f}×|\n"
body+='''
**结论：4096³尚未beat cuBLAS。** 宽TMEM load改善了最后一版，但不能用略快于某轮cuBLASLt的结果，忽略更快的普通cuBLAS。

## 8192³重复实验

|轮次|手写 μs|cuBLAS μs|cuBLASLt μs|相对更快库基线|
|---|---:|---:|---:|---:|
'''
for i,r in enumerate(rep,1):body+=f"|{i}|{r['median_ms']*1000:.2f}|{r['cublas_ms']*1000:.2f}|{r['cublasLt_ms']*1000:.2f}|{min(r['cublas_ms'],r['cublasLt_ms'])/r['median_ms']:.3f}×|\n"
body+='''
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
STUDY_SANITIZER_ONLY=1 ../../.toolchains/cuda-13.0/compute-sanitizer/compute-sanitizer \\
  --tool racecheck --error-exitcode 99 ./build/v09_tuned 1024 512 192
```

每个命令行进程只处理一个shape/一组指针。当前实现不支持任意尾部尺寸、任意stride或不同的融合epilogue，也不包含端到端数据分配/搬运耗时。

原书：[最终优化结果](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/ebccca2e5675966f68fb3d4880d4448194bd638d/zh/chapter_gemm_advanced/index.md#完整优化结果)。CuTe C++ API参考：[NVIDIA官方Blackwell tutorials](https://github.com/NVIDIA/cutlass/tree/v4.6.0/examples/cute/tutorial/blackwell)。
'''
(R/'docs/performance.md').write_text(body)
style='<style>body{max-width:1250px;margin:30px auto;padding:20px;font:16px/1.8 system-ui;color:#0f172a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #cbd5e1;text-align:left}pre{padding:18px;background:#f1f5f9;overflow:auto}a{color:#0369a1}</style>'
(R/'docs/performance.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>性能实测与TIR对照</title>'+style+'<a href="index.html">← 版本目录</a>'+MarkdownIt().enable('table').render(body)+'</html>')
# Comparison chart uses only full-size rows; the cuBLAS reference is from the tuned version's run.
chart=[]
for name,r in rows:
 if r['m']==4096:chart.append((name,r['tflops']))
if rows:
 last=rows[-1][1];chart.append(('cuBLAS（调优版同轮）',2*4096**3/(last['cublas_ms']*1e9)));chart.append(('cuBLASLt（调优版同轮）',2*4096**3/(last['cublasLt_ms']*1e9)))
maxv=max([t for _,t in chart]+[1]);bars=[]
for i,(n,t) in enumerate(chart):
 y=55+i*34;w=650*t/maxv;color='#64748b' if n.startswith('cuBLAS') else '#0d9488'
 bars.append(f'<text x="12" y="{y+15}" font-size="13">{html.escape(n)}</text><rect x="225" y="{y}" width="{w:.1f}" height="22" fill="{color}" rx="3"/><text x="{235+w:.1f}" y="{y+16}" font-size="12">{t:.0f}</text>')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 {100+len(chart)*34}"><rect width="980" height="100%" fill="#f8fafc"/><text x="12" y="25" font-size="18">B300 · 4096³ · TFLOP/s（越高越快）</text>'+''.join(bars)+'</svg>'
(R/'docs/performance.svg').write_text(svg)
cards=''
for i,n in enumerate(NAMES):
 if i in (0,3,6,9):cards+='<h2 style="grid-column:1/-1;margin-bottom:0">'+{0:'第一章：构建 Tiled GEMM',3:'第二章：异步与持久化',6:'第三章：Warp specialization 与 cluster',9:'调优与验证'}[i]+'</h2>'
 cards+=f'<a class="card" href="{n}.html"><b>{"09T" if i==9 else f"{i+1:02d}"} · {TITLES[i]}</b><span>独立 .cu · 逐行中文解释 · 可操作图示</span></a>'
index='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell GEMM · CUDA C++ / CuTe</title><style>body{background:#f1f5f9;color:#0f172a;font:16px/1.8 system-ui;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:32px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}.card{display:block;background:white;border:1px solid #cbd5e1;border-radius:10px;padding:20px;text-decoration:none;color:#0f172a}.card:hover{border-color:#0d9488}.card span{display:block;color:#64748b;font-size:14px}.panel{background:white;padding:24px;border-radius:12px;margin:24px 0}a{color:#0369a1}svg{width:100%}code{background:#e2e8f0;padding:3px}li{margin:7px 0}</style><main><a href="/zh/">← 返回原书中文网页</a><h1>Blackwell GEMM：CUDA C++ / CuTe 重写</h1><p>按原书第三部分的9步顺序学习。源码为真正的CUDA C++，不依赖Python DSL。B300上逐版编译、核对输出并计时。</p><div class="panel"><b>结论先看</b><p>4096³最终调优版仍慢于cuBLAS。8192³重复实验比普通cuBLAS略快，但与cuBLASLt基本持平，不能宣称稳定超过最强库基线。</p><a href="performance.html">完整结果、测试口径，以及“TIR为何能持平” →</a></div><div class="grid">'''+cards+'''</div><div class="panel"><h2>如何阅读</h2><p><a href="01-single-tile-walkthrough.html">第一个 kernel 看不懂？从小矩阵、四次 MMA 和实际地址图开始 →</a></p><p><a href="00-CuTe前置知识.html">先读：从CUDA/TIRx到CuTe C++的符号与layout前置知识</a></p><ol><li>先读本版开头的目的、数据流和barrier协议。</li><li>移动K tile滑块，观察stage/phase；移动输出tile滑块，观察A/B复用。</li><li>点击源码行，把CuTe表达式对应到存储、计算或同步环节。</li><li>自己编译运行，再对照逐行解释；按顺序进入下一版。</li></ol><p><a href="testing.html">共享C++测试程序逐行讲解</a> · <a href="cutlass_reference.html">独立CUTLASS collective诊断基线</a> · <a href="../README.md">编译与目录说明</a></p></div><div class="panel"><h2>同尺寸的性能演进</h2>'''+svg+'''<p>每种库基线取最后调优版同轮测量；每版自己的完整对照值见实测报告。v1/v2是单输出tile，不混入此图。</p></div></main></html>'''
(R/'docs/index.html').write_text(index)
print('report/index generated',len(rows),'versions',len(rep),'repeats')
