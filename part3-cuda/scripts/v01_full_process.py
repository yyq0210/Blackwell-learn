"""Supplement the existing v01 walkthrough with one continuous trace and address zoom."""
from pathlib import Path
from html import escape
R=Path(__file__).resolve().parents[1]

def text(x,y,value,size=17,color='#172554'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(value))}</text>'
def box(x,y,w,h,fill='#eff6ff',stroke='#94a3b8'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}"/>'
def line(x1,y1,x2,y2,color='#64748b',width=2):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#tip)"/>'
def wrap_svg(body,h,title):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" role="img"><title>{escape(title)}</title><defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#64748b"/></marker></defs><rect width="1200" height="100%" fill="#f8fafc"/><g font-family="system-ui,sans-serif">{body}</g></svg>'

def build_figures():
    s=text(30,40,'v01 全过程：始终追踪 A[3,18] 与 D[3,5]',29)
    s+=text(30,78,'固定 M=N=128、K=64；1 个 CTA、128 个线程。示例 A[3,k]=4，B[5,k]=2(k+1)。',18)
    s+=box(30,104,1140,106,'#fef3c7')
    s+=text(48,135,'bm=0、bn=0：唯一输出块；nk=1：只有一个 K64 大块；kt=0；kb=0,1,2,3。',20)
    s+=text(48,167,'块内输入 k = 16×kb + ki：18 = 16×1 + 2。nk 是块的数量，kt/kb 才是循环编号。',18)
    s+=text(48,196,'蓝色框建立视图 / 地址；橙色框搬运数据或计算。acc 的中间数值是数学解释，须等完成后才能读取。',16)
    panels=[
      ('① CPU 构造完整矩阵视图', ['a/b：(128,64):(64,1)', 'd：(128,128):(128,1)'], ['a(3,18) → GMEM p.a[210]，数值 4', 'b(5,18) → GMEM p.b[338]，数值 38', 'd(3,5) → GMEM p.d[389]，尚未写结果'], False),
      ('② 选输出块，保留 K64 块轴', ['coord = make_coord(0,0,_)', 'local_tile → ga / gb / gd'], ['ga(3,18,0) 仍是 A[3,18]', 'ga/gb shape=(128,64,1)；gd=(128,128)', 'bm/bn 选位置；最后的 1 是 K64 块数量'], False),
      ('③ MMA 分层坐标 + 操作数视图', ['cta = mma.get_slice(0)', 'partition_* + make_fragment_*'], ['pa((3,2),0,1,0) 仍指向 GMEM A[3,18]', 'sa((3,2),0,1) 指定其目标 SMEM 位置', 'ra(0,0,1) 描述第二个 K16 输入片；acc 是 TMEM 视图'], False),
      ('④ 申请 TMEM，绑定结果地址', ['warp0：allocate(128,&s.tmem)', 'thread0 初始化 barrier；CTA 同步'], ['申请的是 128 列 TMEM，不是 128 个线程', 'acc.data()=s.tmem；此时 acc 数值仍未初始化', 'ScaleOut::Zero 让第一条 MMA 忽略旧累加值'], False),
      ('⑤ 128 线程装入整个 A/B tile', ['cooperative_copy<128>(t,pa,sa)', 'shared proxy fence + __syncthreads'], ['thread82 搬 A[3,18]：p.a[210] → s.a[202]', '同一线程也搬 B[5,18]：p.b[338] → s.b[378]', '此处 SMEM 地址示例采用 1024-byte 对齐起点'], True),
      ('⑥ kt=0 内，warp0 提交 4 次 MMA', ['kb=0 → 1 → 2 → 3', '每条都覆盖同一块 128×128 acc'], ['K[0,16)：1088；再加 3136 → 4224', '再加 K[32,48) 的 5184 → 9408', '再加 K[48,64) 的 7232 → 16640'], True),
      ('⑦ 完成等待，再把 TMEM 读到 REG', ['umma_arrive + wait_barrier(phase=0)', 'make_tmem_copy → get_slice(t)'], ['cp.get_slice(3)：warp0/lane3 的 copy 份额', 'src：warp0 协作窗口；dst：D 第 3 行', 'copy + wait::ld 后，thread3 的 rf[5]=16640'], True),
      ('⑧ 转换、写回并释放', ['rh[i]=H(rf[i])；copy(rh,dst)', '全 CTA 同步；warp0 释放 TMEM'], ['thread3：rh[5] → GMEM D[3,5] = p.d[389]', '全部 128 线程合作写完 16384 个 FP16 结果', 'A[3,18] 只是 64 项点积中的一个输入，并非完整结果'], True)
    ]
    for i,(title,left,right,action) in enumerate(panels):
        y=242+i*160
        s+=box(30,y,1140,136,'#fff7ed' if action else '#eff6ff')
        s+=text(46,y+29,title,21)
        for j,v in enumerate(left):s+=text(46,y+66+j*29,v,17)
        s+=f'<path d="M492,{y+42} L492,{y+122}" stroke="#cbd5e1"/>'
        for j,v in enumerate(right):s+=text(515,y+59+j*29,v,16)
        if i<7:s+=line(600,y+138,600,y+157)
    s+=text(30,1562,'关键：ga → pa 只改变看待数据的坐标方式；pa → sa 才搬数据；ra 描述输入，acc 存累计结果。',17)
    (R/'docs/v01-full-process.svg').write_text(wrap_svg(s,1595,'v01 的 bm/bn/nk、布局视图与完整数据计算过程'))
    s=text(30,40,'A[3,18]：谁搬它，与放到哪里，是两个问题',29)
    s+=text(30,78,'以下假设 A 的 SMEM 数组起点为 1024-byte 对齐。所有元素索引均相对各自数组起点。',17)
    s+=box(385,105,430,75,'#fef3c7')+text(405,135,'逻辑身份：A[3,18]，本示例数值 4',20)+text(405,164,'m=3，k=18；无论存到哪里都不改变',17)
    s+=line(435,182,270,232)+line(765,182,930,232)
    s+=box(30,240,535,120,'#ede9fe')+text(48,272,'问题 1：哪个线程搬？——输入 copy 映射',20)
    s+=text(48,308,'t = 64×(m mod 2) + k = 64×1 + 18 = 82',19)+text(48,340,'thread82 = warp2 / lane18；不代入物理列号。',17)
    s+=box(600,240,570,120,'#ccfbf1')+text(618,272,'问题 2：放到哪里？——目标 swizzle 布局',20)
    s+=text(618,308,'q=18/8=2，u=18%8=2；p=q XOR 3=1',18)+text(618,340,'每个 sector 是 16 bytes，容纳 8 个 FP16。',17)
    s+=text(30,405,'第 3 行：行首元素偏移 3×64=192，行首字节偏移 384。',20)
    s+=text(30,444,'GMEM 地址顺序',17)
    s+=text(30,477,'行主序：槽位 = q',15)
    for q in range(8):
        x=220+q*116;fill='#fde68a' if q==2 else '#e2e8f0'
        s+=box(x,460,108,93,fill)+text(x+8,486,f'槽位 {q}',18)+text(x+8,514,f'k {q*8}–{q*8+7}',15)+text(x+8,540,f'元素 {192+q*8}–{199+q*8}',12)
    s+=line(220+2*116+54,558,220+1*116+54,628,'#dc2626',4)
    s+=text(565,588,'选中的逻辑 sector 2 → 物理 sector 1',18,'#b91c1c')
    s+=text(565,617,'sector 内 u=2 保持不变',17,'#b91c1c')
    s+=text(30,669,'SMEM 地址顺序',17)
    s+=text(30,700,'槽位 = q XOR 3',15)
    for p in range(8):
        q=p^3;x=220+p*116;fill='#fde68a' if q==2 else '#e2e8f0'
        s+=box(x,640,108,95,fill)+text(x+8,665,f'物理 p={p}',16)+text(x+8,694,f'装逻辑 q={q}',14)+text(x+8,721,f'元素 {192+p*8}–{199+p*8}',12)
    s+=text(30,780,'放大被搬运的 sector：逻辑 k=16…23，整体从源 sector 2 放入目标 sector 1。',19)
    for u in range(8):
        x=220+u*116;fill='#fbbf24' if u==2 else '#f1f5f9'
        s+=box(x,808,108,107,fill)+text(x+9,836,f'逻辑 k={16+u}',15)+text(x+9,867,f'GMEM {208+u}',14)+text(x+9,899,f'SMEM {200+u}',14)
    s+=box(30,950,1140,154,'#fff7ed')
    s+=text(48,983,'thread82：读取 p.a[210] 的 4 → 临时寄存器 → 写入 s.a[202]，数值仍为 4。',21)
    s+=text(48,1017,'GMEM 元素：192+2×8+2=210；SMEM 元素：192+1×8+2=202。',19)
    s+=text(48,1052,'字节偏移：420 XOR 48 = 404；404 / 2 = 202。不能对元素偏移直接套字节 swizzle。',17)
    s+=text(48,1084,'物理行内位置 202−192=10，不是新的逻辑 k；不能据此把搬运线程改算成 64+10=74。',17)
    s+=text(30,1150,'两排都按各自内存地址递增绘制。GMEM 的行主序恰好与逻辑 K 次序一致；SMEM 的 swizzle 不一致。',17)
    s+=text(30,1180,'A[3,18] 是两处数据共同的逻辑身份；210 与 202 是它在两个不同缓冲区中的元素偏移。',17)
    (R/'docs/v01-swizzle-ownership.svg').write_text(wrap_svg(s,1220,'thread82 搬运 A[3,18] 到 SMEM 元素202的地址分解'))

TRACE_HTML='''<section id="full-trace"><h2>交互：把 v01 从启动到写回完整走一遍</h2><p>固定观察 A[3,18] 与 D[3,5]：A[3,k]=4、B[5,k]=2(k+1)，最终 D[3,5]=16640；与下方交互矩阵采用相同输入规则。bm=bn=0、nk=1、kt=0 固定不变。</p><div class="controls"><button id="trace-prev">上一步</button><label>执行阶段 <input id="trace-step" type="range" min="0" max="11" value="0"></label><button id="trace-next">下一步</button><button id="trace-reset">回到启动</button></div><p id="trace-title" aria-live="polite"></p><p id="trace-action"></p><pre id="trace-alias"></pre><div id="trace-state"></div><p>中途显示的累计值只解释数学贡献：提交异步 MMA 后不能立即读取 acc，必须等完成 barrier。图不模拟 GPU 时钟或各 warp 的精确交错。</p></section>'''
TRACE_JS=r'''
function traceModel(stage){
 const totals=[1088,4224,9408,16640];
 const rows=[
 ['0 · CPU 构造视图并启动','1 CTA × 128 线程。bm=bn=0；nk=1 表示一个 K64 大块。','a(3,18) → p.a[210]=4\nb(5,18) → p.b[338]=38\nd(3,5) → p.d[389]，尚未产生本次结果'],
 ['1 · make_coord + local_tile','coord=(0,0,_)，只选唯一输出块，并保留大小为 1 的 K64 块轴。','ga(3,18,0) = a(3,18)\ngb(5,18,0) = b(5,18)\nga/gb shape=(128,64,1)，gd=(128,128)'],
 ['2 · partition + fragment','取得 CTA0 的 MMA 份额，建立 GMEM / SMEM / descriptor / TMEM 视图；尚未搬入矩阵。','pa((3,2),0,1,0) 仍指向 GMEM A[3,18]\nsa((3,2),0,1) 指向 SMEM 目标；ra(0,0,1) 描述第二个 K16 输入片\npd((3,5),0,0) 与 acc((3,5),0,0) 逻辑输出相同，存储空间不同'],
 ['3 · 分配 TMEM 并初始化同步','warp0 分配 128 列；thread0 初始化 barrier；全 CTA 同步后绑定 acc 地址。','acc.data() = s.tmem\nacc 尚未初始化；ScaleOut::Zero 只让第一条 MMA 忽略旧值，并未在此执行清零'],
 ['4 · kt=0：合作装入 A/B','全部 128 线程合作。thread82 搬入 A[3,18] 和 B[5,18]；随后 fence + CTA 同步。','pa((3,2),0,1,0) → sa((3,2),0,1)\np.a[210]=4 → s.a[202]=4\np.b[338]=38 → s.b[378]=38（SMEM 起点 1024-byte 对齐的示意）'],
 ['5 · 提交 kb=0，K=[0,16)','warp0 提交第一条 K16 MMA，ScaleOut::Zero；此后模式变成 One。','ra(0,0,0) / rb(0,0,0)\n对 D[3,5] 的数学贡献：8×(1+…+16)=1088\n整块 128×128 acc 都在本条操作的覆盖范围内'],
 ['6 · 提交 kb=1，K=[16,32)','第二条 MMA 使用包含 A[3,18] 的输入片，继续累加。','ra(0,0,1) / rb(0,0,1)\nA[3,18]×B[5,18]=4×38=152，只是本段 16 项中的一项\n本段贡献 3136；累计数学目标 4224'],
 ['7 · 提交 kb=2，K=[32,48)','第三条 MMA 累加到同一个 acc；kt 仍然是 0。','ra(0,0,2) / rb(0,0,2)\n本段贡献 5184；累计数学目标 9408'],
 ['8 · 提交 kb=3，K=[48,64)','第四条 MMA 仍更新同一块 acc；不是另算第四块输出。','ra(0,0,3) / rb(0,0,3)\n本段贡献 7232；累计数学目标 16640'],
 ['9 · 等待全部 MMA 完成','umma_arrive 关联此前 MMA，所有线程 wait_barrier(phase=0)。此时结果才可读取。','acc((3,5),0,0) = 16640\nTMEM 相对坐标：TLane=3，TCol=5；不是普通线性内存 stride'],
 ['10 · TMEM → REG → FP16 → GMEM','cp.get_slice(3) 取 thread3 的 copy 份额。协作读取并 wait::ld 后转换、写回。','src：warp0 的 TMEM 行 0–31 协作窗口\nthread3 的 rf[5]=16640 → rh[5] → dst[5]\ndst[5] 对应 pd((3,5),0,0) → p.d[389]'],
 ['11 · CTA 同步并回收 TMEM','所有线程结束读取/写回流程，warp0 释放 TMEM。kernel 完成。','D[3,5] = FP16(16640)\n其余 16383 个输出也已由对应线程写回；本次 TMEM 存储已释放']
 ];
 return {row:rows[stage],smem:stage>=4?'A: s.a[202]=4；B: s.b[378]=38':'输入尚未装入',acc:stage<3?'未申请':stage<5?'已绑定，但内容未初始化':stage<9?'MMA 已提交；数学目标 '+totals[stage-5]+'，尚不可直接读取':stage<11?'计算完成：acc[3,5]=16640':'已释放',output:stage<10?'D 尚未写入本次结果':'D[3,5]=16640'};
}
function traceRender(){
 const stage=readInt('trace-step',11),v=traceModel(stage);
 $('trace-title').textContent=v.row[0];$('trace-action').textContent=v.row[1];$('trace-alias').textContent=v.row[2];
 $('trace-state').innerHTML='<table><tr><th>输入 SMEM</th><td>'+v.smem+'</td></tr><tr><th>结果 TMEM</th><td>'+v.acc+'</td></tr><tr><th>输出 GMEM</th><td>'+v.output+'</td></tr></table>';
}
$('trace-step').addEventListener('input',traceRender);
$('trace-prev').addEventListener('click',()=>{$('trace-step').value=Math.max(0,readInt('trace-step',11)-1);traceRender();});
$('trace-next').addEventListener('click',()=>{$('trace-step').value=Math.min(11,readInt('trace-step',11)+1);traceRender();});
$('trace-reset').addEventListener('click',()=>{$('trace-step').value=0;traceRender();});
traceRender();
'''
