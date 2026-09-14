"""Build the v01 API atlas from its Markdown and measured copy ownership."""
from pathlib import Path
from html import escape
import json
from markdown_it import MarkdownIt
R=Path(__file__).resolve().parents[1]; D=R/'docs'
COL=['#2563eb','#0d9488','#d97706','#9333ea']
def t(x,y,s,size=17,color='#172554'):
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(s))}</text>'
def box(x,y,w,h,lines,color='#eff6ff'):
 s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" stroke="#94a3b8"/>'
 for i,l in enumerate(lines):
  units=sum(1 if ord(c)>255 else .58 for c in l)
  fs=min(17 if i==0 else 14,(w-24)/max(1,units))
  s+=t(x+12,y+28+i*25,l,round(fs,1))
 return s
def arr(x1,y1,x2,y2,label='',color='#64748b'):
 return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'+ (t((x1+x2)/2+8,(y1+y2)/2-8,label,13,color) if label else '')
def save(name,s,h,title):
 (D/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 {h}" role="img"><title>{escape(title)}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#64748b"/></marker></defs><rect width="1120" height="100%" fill="#f8fafc"/><g font-family="system-ui,sans-serif">{s}</g></svg>')
s=t(25,38,'API 接力：先描述访问方式，再执行搬运与计算',26)
s+=t(25,73,'蓝框：视图 / 配置；橙框：实际数据动作。连线主要表示依赖，不表示每一步都复制矩阵。',16)
xs=[25,395,765]
heads=['A/B 输入路线','MMA 操作数路线','结果读取路线']
for x,h in zip(xs,heads):s+=t(x,113,h,21)
left=[['make_coord(bm,bn,_)','组装数学块坐标'],['local_tile(a/b,Tile,coord,Step)','完整 GMEM → 子块视图 ga/gb'],['cta.partition_A/B(ga/gb)','ga/gb → MMA 层次视图 pa/pb'],['cooperative_copy<128>(t,pa,sa)','128 线程按各自份额搬 GMEM → SMEM']]
mid=[['mma.get_slice(0)','本版唯一 CTA 的 MMA 接口'],['make_tensor(smem_ptr,LA)','sa/sb：地址 + swizzled layout'],['cta.make_fragment_A/B(sa/sb)','ra/rb：描述完整输入片的 descriptor'],['gemm(mma,ra,rb,acc)','warp0 发射；计算单元读取 SMEM']]
right=[['cta.make_fragment_C(pd)','acc：TMEM 视图；还需申请/绑定'],['make_tmem_copy(load_atom,acc)','cp：推导 copy 的线程/数值映射'],['cp.get_slice(t)','th：本 CUDA 线程的 copy 接口'],['th.partition_S(acc) / _D(pd)','源协作窗口 / 本线程目标位置']]
for x,items in zip(xs,[left,mid,right]):
 for i,lines in enumerate(items):
  y=135+i*132;s+=box(x,y,330,96,lines,'#ffedd5' if lines[0].startswith(('cooperative_copy','gemm')) else '#eff6ff')
  if i<3:s+=arr(x+165,y+97,x+165,y+131)
s+=box(25,690,1065,98,['真正的数据路径：GMEM → SMEM → Tensor Core → TMEM → RF → RH → GMEM D','cooperative_copy 搬入；gemm 计算；copy(cp,src,rf) 读出；FP32→FP16 转换；copy(rh,dst) 写回。'],'#fff7ed')
s+=t(25,646,'D 路线：local_tile(d) → gd → cta.partition_C(gd) → pd；pd 同时提供 acc 的逻辑形状和写回目标。',16)
s+=t(25,822,'make_fragment_A/B 不把 FP16 矩阵搬进寄存器；make_fragment_C 不把 D 的旧值加载进 TMEM。',16)
save('v01-api-chain.svg',s,850,'CuTe API 输入、计算与结果路线')
s=t(25,38,'local_tile：coord=(1,2,_)，同一数学块在 A/B/D 中的窗口',25)
s+=t(25,73,'演示尺寸 M=384、N=512、K=128；每格表示一个 128×64 输入块或 128×128 输出块。',16)
for x,label,rows,cols,sel in [(35,'A：M × K',3,2,1),(360,'B：N × K',4,2,2),(685,'D：M × N',3,4,1)]:
 s+=t(x,120,label,21)
 cw=125 if cols==2 else 90;ch=70
 for row in range(rows):
  for col in range(cols):
   chosen=(row==sel and (cols==2 or col==2)); color='#fde68a' if chosen else '#e2e8f0'
   s+=box(x+col*cw,155+row*ch,cw-5,ch-5,[f'({row},{col})'],color)
 s+=t(x,155+rows*ch+30,('保留两个 K64 块' if cols==2 else '仅选择一个输出 tile'),16)
s+=t(35,500,'A 忽略 bn：保留 (M,K)；B 忽略 bm：保留 (N,K)；D 忽略 K：保留 (M,N)。',18)
s+=box(35,530,1020,118,['ga(3,18,1) = A[1×128+3, 1×64+18] = A[131,82]','gb(5,18,1) = B[2×128+5, 1×64+18] = B[261,82]','gd(3,5) = D[1×128+3, 2×128+5] = D[131,261]'])
s+=t(35,690,'图中的 (row,col) 是块编号。_ 使 ga/gb 保留 K 块轴；它不会自动启动两个 K 任务。',16)
save('v01-local-tile.svg',s,725,'make_coord 和 local_tile 的投影与窗口')
s=t(25,38,'partition 改坐标层次；fragment 改操作数表示',26)
s+=box(30,90,310,110,['ga(m,k,kt)','子块内：m=3，k=18','较大演示中 kt=1'])
s+=arr(350,145,485,145,'k=16×kb+ki')
s+=box(495,90,590,110,['pa((m,ki),0,kb,kt)','pa((3,2),0,1,1) 与 ga(3,18,1) 指向同一元素','只有访问坐标变了；没有物理搬运'])
s+=t(30,250,'一个 K64 大块拆成四条 K16 指令：',20)
for q,c in enumerate(COL):
 x=30+q*268;s+=box(x,275,250,110,[f'kb={q}，K=[{q*16},{q*16+16})',f'ki=0..15，k={q*16}+ki', '包含 k=18 → ki=2' if q==1 else '同样覆盖全部 128 行'], '#fef3c7' if q==1 else '#eff6ff')
s+=box(30,440,490,113,['sa：((128,16),1,4)','每个 K16 切片描述 128×16 个 FP16 数值','cooperative_copy 后，数值实际存入 SMEM'])
s+=arr(530,494,605,494)
s+=box(615,440,470,113,['ra：(1,1,4)','每个 K16 切片由一个 descriptor 位置描述','ra(_,_,kb) 供一条 MMA 使用'])
s+=t(30,610,'同一数学元素可以有不同坐标表达；整个输入片可以用一个描述符指定，但输入数值没有消失。',17)
s+=t(30,645,'partition_C：gd(m,n) → pd((m,n),0,0)；make_fragment_C(pd) 再建立对应的 TMEM 视图。',17)
save('v01-partition-fragment.svg',s,680,'partition 的嵌套坐标与 fragment 的描述符表示')
s=t(25,38,'同一批 128 个线程，两套不同的数据归属',26)
s+=t(25,78,'颜色表示 warp：',17)
for w,c in enumerate(COL):s+=box(205+w*210,53,185,42,[f'warp{w} · t={32*w}–{32*w+31}'],c.replace('#','#'))
# White labels in color legend.
s=s.replace('fill="#172554">warp','fill="white">warp')
s+=t(35,137,'输入 A/B：偶数行 / 奇数行 × K 的前半 / 后半',20)
for row in range(8):
 s+=t(35,183+row*37,f'行 {row}',15)
 for half in range(2):
  w=2*(row%2)+half;x=110+half*212;y=157+row*37
  s+=f'<rect x="{x}" y="{y}" width="205" height="33" fill="{COL[w]}"/>'+t(x+12,y+23,f'warp{w}：K={half*32}–{half*32+31}',15,'white')
s+=t(110,487,'… 按奇偶行重复到第 127 行',16)
s+=t(600,137,'输出 D：每个 warp 负责连续 32 行',20)
for w,c in enumerate(COL):s+=box(600,157+w*78,475,70,[f'warp{w}：输出行 {w*32}–{w*32+31}','每个线程固定一行，覆盖全部 128 列'], '#dbeafe' if w==0 else '#ccfbf1' if w==1 else '#fef3c7' if w==2 else '#ede9fe')
s+=box(35,535,1040,110,['thread82 = warp2 / lane18','输入：A/B 的所有奇数行、K=18，共各 64 个 FP16','输出：D[82,0:128]，共 128 个结果'],'#fff7ed')
s+=t(35,689,'输入 owner = 64×(row mod 2)+k；输出 owner = m。此关系已用当前 CUTLASS 实际执行/布局检查。',16)
save('v01-thread-map.svg',s,725,'输入合作搬运与输出写回的线程归属对比')
s=t(25,38,'cp.get_slice(37)：warp1 的协作窗口，lane5 的结果',26)
s+=t(25,76,'以固定输出列 n 为例：一次 32dp32b1x 操作，由 warp1 的 32 个线程协作执行。',17)
rows=[32,33,34,35,36,37,38,63]
for i,row in enumerate(rows):
 y=120+i*53;hi=row==37;shade='#fde68a' if hi else '#eff6ff'
 s+=box(35,y,290,42,[f'TMEM[{row}, n]'],shade)
 s+=arr(340,y+21,495,y+21)
 s+=box(510,y,210,42,[f'lane {row-32}'],shade)
 s+=arr(735,y+21,865,y+21)
 s+=box(875,y,210,42,[f'thread{row}：rf[n]'],shade)
s+=t(45,570,'图中省略行 39–62。warp1 的源窗口是行 32–63；每个线程最终只接收自己的一行。',16)
s+=box(35,605,1050,110,['partition_S：按协作指令表示源窗口，warp 内不同线程的源视图可以重合','partition_D：thread37 的目标是 D[37,0:128]，不与其他线程重合','重复覆盖 n=0..127，再 FP32→FP16，最后由各线程写回 GMEM'])
s+=t(35,755,'不能把 partition_S 的逻辑元素数直接当成该线程接收的寄存器数；须结合 copy atom 的 Src/Dst 映射。',16)
save('v01-tmem-slices.svg',s,790,'warp 协作的 TMEM source 与各 lane 的寄存器 destination')
md=MarkdownIt().enable('table').render((D/'01-cute-api-atlas.md').read_text())
data=json.loads((R/'results/v01-api-mapping.json').read_text())
assert data['failures']==0 and len(data['input_owner'])==8192
lab='''<section id="lab"><h2>交互：选块与选线程是两件事</h2><h3>① local_tile：改变的是数学窗口</h3><p>布局演示：M=384、N=512、K=128，tile=(128,128,64)。每格是一块；高亮 A/B 的当前 K64 块和对应 D tile。_ 保留全部 K 块，kt 滑块展示后续再固定哪一个。</p><div class="controls"><label>bm <input id="bm" type="range" min="0" max="2" value="1"></label><label>bn <input id="bn" type="range" min="0" max="3" value="2"></label><label>kt <input id="kt" type="range" min="0" max="1" value="1"></label></div><canvas id="tiles" width="1080" height="340"></canvas><pre id="coords"></pre><h3>② 真实 v01：同一个线程的输入 / 输出份额</h3><p>下图使用从 CuTe 实际执行得到的输入 owner 表。背景颜色是 warp 编号；红色是当前线程负责的逻辑元素。输入每线程 64 个 FP16，输出每线程 128 个结果。没有显示 GPU 的执行时钟。</p><div class="controls"><label>threadIdx.x <input id="tid" type="number" min="0" max="127" value="82"></label><button id="prev">上一个线程</button><button id="next">下一个线程</button><label>输出列 n <input id="ncol" type="range" min="0" max="127" value="5"></label></div><p id="role" aria-live="polite"></p><canvas id="owners" width="1080" height="625"></canvas><pre id="threadcoords"></pre><h3>③ 放大一次 TMEM load：32 个 lane 各接收一个结果</h3><canvas id="lanes" width="1080" height="205"></canvas><p id="source" aria-live="polite"></p><p>此处 get_slice 只选择既有操作映射的参与者，不创建 CTA/warp，也不控制 SM 的实际硬件调度。</p></section>'''
js=r'''
const $=id=>document.getElementById(id), colors=['#2563eb','#0d9488','#d97706','#9333ea'];
const measured=JSON.parse($('ownership').textContent).input_owner;
function value(id,max){let v=Number($(id).value);v=Math.max(0,Math.min(max,Number.isFinite(v)?Math.floor(v):0));$(id).value=v;return v;}
function inputCells(t){let cells=[];for(let i=0;i<measured.length;i++)if(measured[i]===t)cells.push([Math.floor(i/64),i%64]);return cells;}
function tileCoords(bm,bn,kt){return {a:[128*bm+3,64*kt+18],b:[128*bn+5,64*kt+18],d:[128*bm+3,128*bn+5]};}
function render(){
 const bm=value('bm',2),bn=value('bn',3),kt=value('kt',1),tid=value('tid',127),n=value('ncol',127),warp=Math.floor(tid/32),lane=tid%32;
 const tc=$('tiles').getContext('2d');tc.clearRect(0,0,1080,340);tc.font='17px system-ui';
 for(const [x,title,rs,cs,rsel,csel] of [[15,'A：M×K',3,2,bm,kt],[365,'B：N×K',4,2,bn,kt],[715,'D：M×N',3,4,bm,bn]]){
  tc.fillStyle='#172554';tc.fillText(title,x,30);let w=310/cs,h=58;
  for(let r=0;r<rs;r++)for(let c=0;c<cs;c++){tc.fillStyle=r===rsel&&c===csel?'#fbbf24':r===rsel&&cs===2?'#fef3c7':'#e2e8f0';tc.fillRect(x+c*w,55+r*h,w-5,h-5);tc.fillStyle='#172554';tc.fillText(`(${r},${c})`,x+c*w+10,88+r*h);}
 }
 const co=tileCoords(bm,bn,kt);$('coords').textContent=`make_coord(${bm}, ${bn}, _) → ga.shape=(128,64,2)\n固定 kt=${kt}：ga(3,18,${kt}) = A[${co.a}]\npartition_A：pa((3,2),0,1,${kt}) = 同一 A[${co.a}]\ngb(5,18,${kt}) = B[${co.b}]；gd(3,5) = D[${co.d}]`;
 const c=$('owners').getContext('2d');c.clearRect(0,0,1080,625);c.font='17px system-ui';c.fillStyle='#172554';c.fillText('输入 A/B 的逻辑坐标：128 行 × 64 K',15,30);c.fillText('输出 D 的逻辑坐标：128 行 × 128 列',570,30);
 for(let m=0;m<128;m++){
  for(let k=0;k<64;k++){let t=measured[m*64+k];c.globalAlpha=t===tid?1:.33;c.fillStyle=t===tid?'#e11d48':colors[Math.floor(t/32)];c.fillRect(15+k*7.8,55+m*4,7.8,4);}
  c.globalAlpha=m===tid?1:.33;c.fillStyle=m===tid?'#e11d48':colors[Math.floor(m/32)];c.fillRect(570,55+m*4,499,4);
 }
 c.globalAlpha=1;c.strokeStyle='#172554';c.lineWidth=2;c.strokeRect(570+n*(499/128),55+tid*4,499/128,4);c.fillStyle='#172554';c.font='15px system-ui';c.fillText('红色 64 个元素：每隔一行，在固定 K 列',15,597);c.fillText('红色 128 个元素：同一行，覆盖全部 N 列',570,597);
 const cells=inputCells(tid);$('role').textContent=`thread ${tid} = warp ${warp} / lane ${lane}。输入搬运与输出写回对应两套不同的坐标。`;
 $('threadcoords').textContent=`输入 A/B：${cells.slice(0,6).map(v=>`[${v}]`).join('、')} … [${cells[cells.length-1]}]（共 ${cells.length} 个）\n输出 D：D[${tid},0:128]（128 个）\n计算阶段：${warp===0?'此线程进入 warp0 的 MMA 封装；内部选一个线程实际发射':'此 warp 不发射 MMA，随后等待 MMA 完成'}。`;
 const l=$('lanes').getContext('2d');l.clearRect(0,0,1080,205);l.font='15px system-ui';
 for(let i=0;i<32;i++){let x=15+(i%16)*66,y=10+Math.floor(i/16)*93;let row=warp*32+i;l.fillStyle=i===lane?'#fde68a':'#dbeafe';l.fillRect(x,y,61,83);l.fillStyle='#172554';l.fillText(`lane${i}`,x+3,y+21);l.fillText(`行${row}`,x+3,y+47);l.fillText(`t${row}`,x+3,y+71);}
 $('source').textContent=`固定列 n=${n}：warp${warp} 协作读取 TMEM 行 ${warp*32}–${warp*32+31}。lane${lane} 接收 acc[${tid},${n}] 到自己的 rf[${n}]，最终写 D[${tid},${n}]。warp 内 source 视图描述同一协作窗口；destination 的接收结果按 lane 区分。`;
}
for(const id of ['bm','bn','kt','tid','ncol'])$(id).addEventListener('input',render);
$('prev').addEventListener('click',()=>{$('tid').value=Math.max(0,value('tid',127)-1);render();});
$('next').addEventListener('click',()=>{$('tid').value=Math.min(127,value('tid',127)+1);render();});
render();
'''
css='''body{max-width:1140px;margin:30px auto;padding:0 22px 70px;color:#172554;font:17px/1.85 system-ui,sans-serif}h1{font-size:31px}h2{margin-top:2.3em;padding-bottom:7px;border-bottom:2px solid #dbeafe}h3{margin-top:1.7em}a{color:#0369a1}pre{padding:18px;background:#f1f5f9;overflow:auto;border-radius:9px;font-size:14px;line-height:1.8}code{font-family:ui-monospace,monospace;font-size:.9em}table{border-collapse:collapse;display:block;overflow:auto;margin:18px 0}td,th{padding:10px 13px;border:1px solid #cbd5e1;vertical-align:top}th{background:#eff6ff}img,canvas{width:100%;height:auto}nav,.controls{display:flex;flex-wrap:wrap;align-items:center;gap:18px}.controls{padding:18px;background:#f1f5f9}input[type=number]{width:65px;padding:6px}button{padding:8px 13px;border:1px solid #93c5fd;background:#eff6ff;border-radius:6px;cursor:pointer}#lab{scroll-margin-top:20px}'''
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CuTe API 与线程编排图册</title><style>'+css+'</style></head><body><nav><a href="index.html">课程目录</a><a href="01-single-tile-walkthrough.html">计算过程讲义</a><a href="v01_single_tile.html">逐行代码</a><a href="#lab">交互图</a><a href="01-cute-api-atlas.md">Markdown</a></nav>'+md+lab+'<script id="ownership" type="application/json">'+json.dumps(data,separators=(',',':'))+'</script><script>'+js+'</script></body></html>'
(D/'01-cute-api-atlas.html').write_text(page)
print('API atlas: 5 SVGs and interactive HTML generated')
