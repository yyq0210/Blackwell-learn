"""Render the hand-written v01 guide and its teaching SVGs/interactive page."""
from pathlib import Path
from html import escape
from markdown_it import MarkdownIt
R = Path(__file__).resolve().parents[1]
D = R / 'docs'
COLORS = ['#2563eb', '#0d9488', '#d97706', '#9333ea']
def text(x,y,s,size=17,color='#172554'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(s)}</text>'
def rect(x,y,w,h,color,stroke='#cbd5e1'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{color}" stroke="{stroke}"/>'
def svg(body,title,height):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 {height}" role="img"><title>{escape(title)}</title><rect width="1100" height="100%" fill="#f8fafc"/><g font-family="system-ui, sans-serif">{body}</g></svg>'
s=text(30,38,'四段 K，更新同一块 D',27)
s+=text(30,68,'A[m,k] × B[n,k]：A 的第 m 行与 B 的第 n 行做点积。B 在内存中仍按 (N,K) 存储。',17)
for x,label,row_label in [(35,'A：128 × 64','m'),(390,'B：128 × 64','n')]:
    s+=text(x,105,label,20)
    for i,c in enumerate(COLORS):
        s+=rect(x+i*70,125,68,192,c)+text(x+i*70+8,153,f'K {i*16}–',14,'white')+text(x+i*70+8,176,str(i*16+15),14,'white')
    s+=f'<rect x="{x}" y="212" width="278" height="5" fill="#fef08a"/>'
    s+=text(x,344,f'黄色高亮：第 {row_label} 行的 64 个输入')
s+=text(745,105,'acc / D：128 × 128',20)+rect(745,125,290,192,'#e2e8f0')
s+=rect(854,211,12,12,'#e11d48')+text(777,276,'红点：一个 D[m,n]',18)
s+=text(745,344,'四次都更新整块 128 × 128')
s+=text(35,393,'放大一个结果：设 A[m,k]=1，B[n,k]=k+1',21)
for i,(part,total) in enumerate([(136,136),(392,528),(648,1176),(904,2080)]):
    x=35+i*265
    s+=rect(x,415,245,115,'white',COLORS[i])+text(x+15,445,f'MMA {i} · K=[{16*i},{16*i+16})',18,COLORS[i])
    s+=text(x+15,478,('覆盖：' if i==0 else '累加：')+str(part),19)+text(x+15,511,f'acc[m,n] = {total}',20)
s+=text(35,567,'128×128×64 是工作块；128×128×16 是一条指令。颜色表示 K 区间，不表示四个 warp。',17)
(D/'v01-matmul-steps.svg').write_text(svg(s,'四次 MMA 的数学分块与累加',595))
s=text(30,38,'先准备数据，再提交计算，等待后写回',27)
boxes=[(30,'GMEM','A、B 各 16 KiB'),(245,'SMEM','A、B 各 16 KiB'),(460,'Tensor Core','4 × MMA K16'),(675,'TMEM','128×128 FP32'),(890,'REG → GMEM','FP32 → FP16')]
for x,a,b in boxes:
    s+=rect(x,80,180,110,'white')+text(x+12,112,a,21)+text(x+12,147,b,15)
    if x<890:s+=text(x+186,140,'→',25)
s+=text(30,223,'普通 load/store；按 LA 写入 SMEM       descriptor 引导读取       结果累加       tcgen05.ld 后再写 D',16)
labels=['申请 / 初始化','搬 A、B','可见性 + 集合','提交 4 次 MMA','等待计算完成','读取 / 转换 / 写回','集合 / 释放']
for i,label in enumerate(labels):s+=text(172+i*127,290,label,14)
for w in range(4):
    y=312+w*68;s+=text(28,y+31,f'warp {w}',20)
    vals=[('分配 + 初始化' if w==0 else '等初始化'), '合作搬运','fence + sync',('issue' if w==0 else '等待'),'wait mma','TMEM → D',('sync + free' if w==0 else 'sync')]
    for i,v in enumerate(vals):
        color='#dbeafe' if i in (1,5) else '#ede9fe' if i==3 and w==0 else '#e2e8f0'
        s+=rect(165+i*127,y,120,48,color)+text(172+i*127,y+30,v,13)
s+=text(30,614,'每个 warp 有 32 个线程。时间框只表达依赖顺序，宽度不是测量时间；发射指令不等于计算立即完成。',16)
(D/'v01-memory-timeline.svg').write_text(svg(s,'四个 warp 的职责和同步顺序',640))
md=(D/'01-single-tile-walkthrough.md').read_text()
body=MarkdownIt().enable('table').render(md)
lab='''<section id="lab"><h2>交互计算图：只看一个输出如何累加</h2>
<p>示例输入固定为 A[m,k]=1+(m mod 4)，B[n,k]=(1+(n mod 4))×(k+1)。默认 m=n=0，四次结果正好是 136、528、1176、2080。图画出真实维度，放大点积用数学值展示；不模拟 GPU 时间或 FP16 舍入。</p>
<div class="controls"><label>输出行 m <input id="m" type="number" min="0" max="127" value="0"></label><label>输出列 n <input id="n" type="number" min="0" max="127" value="0"></label><label>已完成 MMA <input id="step" type="range" min="0" max="4" value="0"></label><button id="next">下一步</button><button id="reset">重置</button></div>
<p id="stage" aria-live="polite"></p><canvas id="matrices" width="1050" height="365" aria-label="A、B 的 K16 区间和被更新的输出点">请启用 JavaScript 查看交互图，上方 SVG 可直接阅读。</canvas>
<div id="sums"></div><p id="owner"></p>
<h3>同一个 A 元素：GMEM 地址如何变成 SMEM 地址？</h3>
<p>使用上面的 m 作为 A 行号，选择 k。下图的格子是 16-byte sector，每格含 8 个 FP16。上排按逻辑 K 次序排列，下排按 SMEM 物理位置排列；示意固定使用 1024-byte 对齐的数组基地址。</p>
<label>输入列 k <input id="k" type="range" min="0" max="63" value="18"></label><p id="address" aria-live="polite"></p>
<canvas id="sectors" width="1050" height="165" aria-label="八个 sector 的逻辑次序与 swizzle 后物理次序"></canvas>
<p>例如把 m 改成 3、k 改成 18：逻辑 sector 2 移到物理 sector 1；GMEM 元素 210 被复制到 SMEM 元素 202。</p></section>'''
js=r'''
const $=id=>document.getElementById(id);
const colors=['#2563eb','#0d9488','#d97706','#9333ea'];
function readInt(id,max){const v=Number($(id).value);const n=Math.max(0,Math.min(max,Number.isFinite(v)?Math.floor(v):0));$(id).value=n;return n;}
function model(m,n,step,k){
 const factor=(1+m%4)*(1+n%4), partial=[136,392,648,904].map(v=>v*factor);
 const sector=Math.floor(k/8), physical=sector^(m%8);
 return {factor,partial,total:partial.slice(0,step).reduce((a,b)=>a+b,0),sector,physical,gm:m*64+k,sm:m*64+physical*8+k%8};
}
function drawMatrix(ctx,x,title,cols,rows,row,col,step,isD){
 const w=285,h=240;ctx.fillStyle='#172554';ctx.font='17px system-ui';ctx.fillText(title,x,27);
 ctx.fillStyle='#e2e8f0';ctx.fillRect(x,50,w,h);
 if(!isD){for(let b=0;b<4;b++){ctx.globalAlpha=b<step?0.75:0.12;ctx.fillStyle=colors[b];ctx.fillRect(x+b*w/4,50,w/4,h);}ctx.globalAlpha=1;
  ctx.fillStyle='#e11d48';ctx.fillRect(x,50+row*h/rows,w,Math.max(2,h/rows));
  ctx.font='14px system-ui';ctx.fillStyle='#334155';for(let b=0;b<4;b++)ctx.fillText(`${b*16}–${b*16+15}`,x+b*w/4+6,315);
 }else{ctx.fillStyle='#e0e7ff';ctx.fillRect(x,50,w,h);ctx.fillStyle='#e11d48';ctx.fillRect(x+col*w/cols-2,50+row*h/rows-2,6,6);}
 ctx.strokeStyle='#94a3b8';ctx.strokeRect(x,50,w,h);ctx.fillStyle='#334155';ctx.font='14px system-ui';ctx.fillText(isD?'每条 MMA 都更新整个矩阵':`红线：第 ${row} 行；横轴是 K`,x,345);
}
function render(){
 const m=readInt('m',127),n=readInt('n',127),step=readInt('step',4),k=readInt('k',63),v=model(m,n,step,k);
 $('stage').textContent=step===0?'准备阶段：A/B 已放进 SMEM，但尚未发出第一条 MMA；TMEM 的旧值未初始化，不能当成 0 读取。':`已完成 ${step} 条 MMA，覆盖 K=[0,${step*16})。D[${m},${n}] 对应的 FP32 累加值 = ${v.total}。`;
 const ctx=$('matrices').getContext('2d');ctx.clearRect(0,0,1050,365);
 drawMatrix(ctx,15,'A：128×64',64,128,m,0,step,false);drawMatrix(ctx,365,'B：128×64，仍按 (N,K) 显示',64,128,n,0,step,false);drawMatrix(ctx,715,'acc：128×128',128,128,m,n,step,true);
 $('sums').innerHTML='<table><thead><tr><th>MMA</th><th>K 区间</th><th>该点的本段贡献</th><th>完成后的累计值</th></tr></thead><tbody>'+v.partial.map((p,i)=>`<tr style="opacity:${i<step?1:0.45}"><td>${i} · ${i===0?'Zero':'One'}</td><td>[${i*16},${i*16+16})</td><td>${p}</td><td>${v.partial.slice(0,i+1).reduce((a,b)=>a+b,0)}</td></tr>`).join('')+'</tbody></table>';
 $('owner').textContent=`写回分工：D[${m},${n}] 由 thread ${m}（warp ${Math.floor(m/32)}，lane ${m%32}）读入寄存器后写回；该线程还负责同一行的另外 127 个结果。此规律只描述本版输出 copy，不描述 A/B 的 cooperative_copy。`;
 $('address').textContent=`A[${m},${k}]：GMEM 元素 ${v.gm}（字节 ${v.gm*2}） → SMEM 元素 ${v.sm}（字节 ${v.sm*2}）。sector ${v.sector} XOR 行低三位 ${m%8} = 物理 sector ${v.physical}；sector 内偏移 ${k%8} 个 FP16 不变。`;
 const c=$('sectors').getContext('2d');c.clearRect(0,0,1050,165);c.font='16px system-ui';
 for(let q=0;q<8;q++){const phys=q^(m%8);for(const [x,y,chosen] of [[145+q*108,20,q===v.sector],[145+phys*108,95,q===v.sector]]){c.fillStyle=chosen?'#fde68a':'#dbeafe';c.fillRect(x,y,100,45);c.strokeStyle=chosen?'#b45309':'#94a3b8';c.strokeRect(x,y,100,45);c.fillStyle='#172554';c.fillText(`逻辑 ${q}`,x+18,y+28);}}
 c.fillStyle='#172554';c.fillText('逻辑 K 次序',10,48);c.fillText('物理 SMEM',10,123);
}
for(const id of ['m','n','step','k'])$(id).addEventListener('input',render);
$('next').addEventListener('click',()=>{$('step').value=Math.min(4,readInt('step',4)+1);render();});
$('reset').addEventListener('click',()=>{$('step').value=0;render();});
render();
'''
css='''body{max-width:1120px;margin:30px auto;padding:0 22px 70px;color:#172554;font:17px/1.85 system-ui,sans-serif}h1{font-size:32px}h2{margin-top:2.4em;border-bottom:2px solid #dbeafe;padding-bottom:8px}h3{margin-top:1.8em}a{color:#0369a1}pre{padding:20px;background:#f1f5f9;border-radius:9px;overflow:auto;font-size:14px;line-height:1.7}code{font-family:ui-monospace,monospace;font-size:.9em}table{border-collapse:collapse;display:block;overflow:auto;margin:18px 0}th,td{padding:10px 14px;border:1px solid #cbd5e1;vertical-align:top}th{background:#eff6ff}img,canvas{width:100%;height:auto}nav,.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center}input[type=number]{width:65px;padding:6px}button{padding:8px 16px;cursor:pointer;border:1px solid #93c5fd;background:#eff6ff;border-radius:6px}.controls{padding:20px;background:#f1f5f9}#lab{scroll-margin-top:20px}'''
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>从一行点积读懂第一个 CuTe kernel</title><style>'+css+'</style></head><body><nav><a href="index.html">课程目录</a><a href="v01_single_tile.html">原逐行代码页</a><a href="01-single-tile-walkthrough.md">Markdown 讲义</a><a href="#lab">直接试交互图</a></nav>'+body+lab+'<script>'+js+'</script></body></html>'
(D/'01-single-tile-walkthrough.html').write_text(page)
print('v01 walkthrough: Markdown + HTML + 2 SVGs')
