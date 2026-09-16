"""Version-specific diagrams and lesson assembly. Does not modify kernels."""
from pathlib import Path
from html import escape
from later_lesson_text import DETAILS
R=Path(__file__).resolve().parents[1]
NAMES=list(DETAILS)
SPEC={
 'v02_k_loop':(2,1,1,1,128,128,'128×128×192','K64 大块 × 3，acc 保留'),
 'v03_multi_cta':(3,1,1,1,128,128,'256×384×192','输出tile网格 2×3'),
 'v04_tma':(4,1,1,1,128,128,'256×384×192','TMA输入 + 分片TMA输出'),
 'v05_double_buffer':(5,1,1,2,128,128,'256×384×192','下一K块预取到另一stage'),
 'v06_persistent':(6,1,1,2,128,128,'2048×2048×192','CTA0：ti0 → ti148（148SM例子）'),
 'v07_warp_specialized':(7,1,1,4,256,128,'2048×2048×192','producer / MMA / epilogue 三角色'),
 'v08_two_cta':(8,2,1,6,256,256,'512×512×448','两CTA输入，共同256×256结果'),
 'v09_multi_consumer':(9,2,2,4,384,512,'1024×512×192','两A共用一B，两个独立MMA warp'),
 'v09_tuned':(9,2,2,4,384,512,'1024×512×192','相同计算，TMEM load 1x → 32x'),
}
FLOW={
'v02_k_loop':[
 ('准备','bm=bn=0；nk=3；128线程','ga=(128,64,3)，pa=((128,16),1,4,3)'),
 ('kt0','GMEM K[0,64) → 唯一SMEM','fence+sync → 4次MMA → wait phase0'),
 ('kt1','同一SMEM改装 K[64,128)','acc不清零；继续4次MMA → wait phase1'),
 ('kt2','改装 K[128,192)','最后4次MMA → wait phase0；累计完整192项'),
 ('写回','TMEM → 128线程的RF → FP16 → GMEM','每线程负责一行；最后同步、释放TMEM')],
'v03_multi_cta':[
 ('启动','grid=(2,3)，每CTA128线程','bm=blockIdx.x，bn=blockIdx.y'),
 ('选块','CTA(1,2)选D[128:256,256:384]','A行128:256；B行256:384；K=192'),
 ('建视图','D[131,261] → 块内(3,5)','A[131,82] → pa((3,2),0,1,1)'),
 ('独立归约','每CTA独立做kt0/1/2，各4次MMA','SMEM、TMEM、barrier均属于各自CTA'),
 ('独立写回','各CTA只写自己的一格输出','没有跨CTA的K归约，也不需要atomic')],
'v04_tma':[
 ('CPU','创建A/B load descriptor与D store descriptor','get_tma_tensor生成TMA坐标视图'),
 ('输入','warp0选一线程，TMA搬A+B共32768bytes','pa/pb + sa/sb → tma_partition → ag/as、bg/bs'),
 ('等到达','wait full[0]，然后MMA读SMEM','输入到齐与输入用完是两个不同事件'),
 ('等用完','4次MMA后commit到done并等待','只有此时才能下一轮覆盖唯一输入stage'),
 ('输出ei0/1','TMEM→RF→RH→sd，再TMA→GMEM','每次128×64；fence/sync/commit/wait后复用sd')],
'v05_double_buffer':[
 ('启动预取','先TMA装kt0到stage0','acc唯一；A/B分别有2套stage'),
 ('kt0','等stage0，预取kt1→stage1，然后算kt0','TMA(kt1)与MMA(kt0)有重叠机会'),
 ('kt1','先前done已保护stage0；预取kt2→stage0','等stage1 full，再读stage1计算'),
 ('kt2','等stage0 full phase1；不再预取','full按stage各自计次，done按全部MMA批次计次'),
 ('写回','完成整段K归约后再处理ei0/1','输入双缓冲不等于输出也有两个并行结果')],
'v06_persistent':[
 ('一次初始化','启动min(tile数量,SM数量)个CTA','同一CTA循环复用SMEM/TMEM/barrier'),
 ('任务ti0','grouped_tile→(bm0,bn0)','Zero→完整K循环→ei写回；保留barrier历史'),
 ('任务ti148','148SM示例：CTA0下一任务','16×16网格中映射为(bm12,bn2)'),
 ('重用','换ga/gb/gd；同一物理输入与结果缓冲','acc重新Zero；full phase不能简单全部重置0'),
 ('退出','所有分配给此CTA的任务写完后释放','逻辑顺序用于局部性，不固定实际SM调度')],
'v07_warp_specialized':[
 ('256线程','WG0写回；warp4算；warp7装','4个输入stage；独立角色有独立循环计数'),
 ('producer','等empty或首次直接用，TMA→stage','本stage到齐后full释放MMA等待'),
 ('MMA','等full→4次MMA→通知empty','全K完成后再通知acc_full'),
 ('epilogue','WG0等acc_full，分两个ei写回','TMEM读完/写回流程结束后128线程通知acc_empty'),
 ('下一任务','MMA等自己的acc_empty再覆盖TMEM','输入环可提前推进；SMEM与TMEM回收分开')],
'v08_two_cta':[
 ('显式cluster','2CTA，每CTA256线程','peer=blockIdx.x%2；输出合作块256×256'),
 ('分输入','每侧A128×64、B128×64','两侧producer；leader登记完整65536bytes'),
 ('共同计算','只有leader warp4发2CTA MMA','A每侧128行；两侧B一起覆盖全部256列'),
 ('广播完成','empty/acc_full通知mask=3两侧','本地TMEM各128×256；输入环S=6'),
 ('分侧写回','每侧4个ei；两侧256线程通知acc_empty','退出cluster_sync后两侧协作释放TMEM')],
'v09_multi_consumer':[
 ('512×256任务','bm → 两个consumer的256行tile','c选A/acc；peer再选每侧128行'),
 ('输入共享','每stage每侧2A+1B=48KiB','两侧producer合计98304bytes，S=4'),
 ('独立MMA','leader warp8算c0，warp9算c1','每个consumer独立推进同一任务的K循环'),
 ('安全复用B','每consumer用完后各通知empty一次','count=2：少一位都不能覆盖共享B'),
 ('独立写回','WG0写c0，WG1写c1，各有sd/acc_full','各acc_empty分别收集两侧256线程')],
'v09_tuned':[
 ('前半段不变','仍2CTA、2consumer、4stage','MMA结果与TMEM的两个256列slot保持一致'),
 ('选ei','每侧结果256列，仍拆成4个64列片','观察c1/peer1/D[387,133]：ei2，片内列5'),
 ('make_tmem_copy','改用32dp32b32x构造cp','相应src/dst/rf形状由CuTe重新推导'),
 ('宽读出','64列：按atom覆盖为2次32列操作','每线程总64个FP32，并非只需32个寄存器值'),
 ('后半段不变','wait::ld→转换→sd→TMA store→回收','实际有LDTM.x32；是否更快仍看完整实测')]
}
def text(x,y,s,sz=18,color='#172554'):
 units=sum(1 if ord(c)>255 else .57 for c in s);sz=min(sz,(1170-x)/max(units,1))
 return f'<text x="{x}" y="{y}" font-size="{sz:.1f}" fill="{color}">{escape(s)}</text>'
def rect(x,y,w,h,color='#eff6ff'):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" stroke="#94a3b8" rx="7"/>'
def card(x,y,w,title,lines,color='#eff6ff'):
 out=rect(x,y,w,48+len(lines)*29,color)+text(x+13,y+29,title,min(21,(w-26)/max(1,sum(1 if ord(c)>255 else .57 for c in title))))
 for j,s in enumerate(lines):
  fs=min(17,(w-26)/max(1,sum(1 if ord(c)>255 else .57 for c in s)))
  out+=text(x+13,y+59+j*29,s,fs)
 return out
def arrow(x1,y1,x2,y2):return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#64748b" fill="none" stroke-width="2" marker-end="url(#a)"/>'
def save(stem,body,height,title):
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" role="img"><title>{escape(title)}</title><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10Z" fill="#64748b"/></marker></defs><rect width="1200" height="100%" fill="#f8fafc"/><g font-family="system-ui,sans-serif">{body}</g></svg>'
 (R/'docs'/f'{stem}.svg').write_text(svg)
def flow(name):
 spec=SPEC[name];s=text(30,40,name+'：从输入到输出的完整过程',28)+text(30,79,'例子 M×N×K='+spec[6]+'；'+spec[7],18)
 for i,(title,a,b) in enumerate(FLOW[name]):
  y=115+i*150;s+=card(35,y,1130,f'{i+1}. '+title,[a,b],'#eff6ff' if i<2 else '#fff7ed')
  if i<4:s+=arrow(600,y+109,600,y+146)
 s+=text(30,906,'箭头表示依赖和代码逻辑，不是GPU时钟比例；异步提交后必须等待相应完成事件。',17)
 save(name+'-process',s,940,name+'完整计算流程')
def mechanism(name):
 v,G,C,S,T,cols,_,_=SPEC[name]
 s=text(30,40,name+'：本版最重要的空间或同步关系',27)
 if v==2:
  s+=text(30,85,'三个K64块写入同一SMEM，并累计到同一TMEM；示例 A=1，B[k]=k+1。',19)
  for i,(p,total) in enumerate([(2080,2080),(6176,8256),(10272,18528)]):
   x=35+i*385;s+=card(x,130,360,f'kt={i}，K=[{64*i},{64*i+64})',[f'4条K16 MMA；贡献 {p}',f'完成后累计 {total}',f'barrier等待 phase {i%2}'])
  s+=card(35,340,1130,'唯一的输入缓冲与唯一的结果累加器',['SMEM：kt0输入 → 等读完 → kt1输入 → 等读完 → kt2输入','TMEM：第一次Zero，之后11条MMA都是One；到最后才写回'])
 elif v==3:
  s+=text(30,85,'M=256,N=384：每格128×128，六CTA写不同输出；同列共用B、同行共用A的数学范围。',18)
  for bm in range(2):
   for bn in range(3):s+=card(40+bn*380,130+bm*150,345,f'CTA({bm},{bn})',[f'D行[{128*bm},{128*(bm+1)})',f'D列[{128*bn},{128*(bn+1)})'],'#fde68a' if (bm,bn)==(1,2) else '#eff6ff')
  s+=text(35,510,'每个CTA内独立完成全部K，没有跨CTA归约；硬件可以按不同顺序运行这些CTA。',19)
 elif v==4:
  for y,title,lines in [(125,'full：输入到达',['登记 A16KiB + B16KiB = 32768bytes','TMA真正搬完 → full就绪 → MMA才可读SMEM']), (290,'done：输入已用完',['4条MMA提交后仍可能异步执行','MMA真正完成 → done就绪 → 才可覆盖输入']), (455,'输出sd：也是一份会被复用的缓冲',['ei0填sd → fence/sync → TMA store commit/wait','全CTA同步后才覆盖sd，处理ei1'])]:s+=card(35,y,1130,title,lines)
 elif v==5:
  rows=[('启动','load kt0 → S0','等待输入'),('kt0','load kt1 → S1','MMA kt0读S0 → 等done'),('kt1','load kt2 → S0','MMA kt1读S1 → 等done'),('kt2','没有下一块','MMA kt2读S0 → 等done')]
  for i,(r,a,b) in enumerate(rows):s+=card(35,110+i*125,1130,r,[a+'；'+b])
  s+=text(35,658,'只有两个不同缓冲能支持这种重叠；上一轮done保护当前被预取覆盖的目标。',18)
 elif v==6:
  s+=text(30,86,'grouped-M示例 mt=10、nt=4；格内是逻辑ti，最后一组只有2行。',18)
  for m in range(10):
   for n in range(4):
    ti=n*8+m if m<8 else 32+n*2+(m-8)
    x=45+n*170;y=125+m*46;s+=rect(x,y,160,40,'#dbeafe' if m<8 else '#fde68a')+text(x+10,y+27,f'({m},{n}) : {ti}',16)
  s+=card(760,125,400,'持久化 != ti顺序执行',['实际CTA0：ti0 → ti148','148SM、16×16网格示例','ti148 → bm12,bn2'])
  s+=card(760,320,400,'两种轮数',['ti：矩阵中的任务编号','tile_round：本CTA第几次做任务','phase使用后者的使用历史'])
  s+=text(35,640,'nk=3时每输出使用S0两次、S1一次：第二输出full等待序列为0、1、1。',18)
 elif v==7:
  s+=card(35,110,1130,'线程角色',['WG0/thread0–127：写回；warp4/thread128–159：MMA；warp7/thread224–255：TMA'])
  for x,title,lines in [(35,'输入SMEM环',['TMA → full[st] → MMA','MMA完成 → empty[st] → TMA','4槽；按连续iteration选stage']), (630,'输出TMEM槽',['MMA整段K完成 → acc_full','WG0读出/写回 → acc_empty','128个写回线程归还这个槽'])]:s+=card(x,265,535,title,lines)
  s+=card(35,480,1130,'两种回收允许不同角色重叠',['输入stage空闲：producer可预取下个输出','结果TMEM未空闲：MMA仍必须等acc_empty；不能把empty当作acc_empty'])
 elif v==8:
  for peer in range(2):
   x=35+peer*590;s+=card(x,110,550,f'peer{peer} / 本侧SM',['A行 '+str(peer*128)+'…'+str(peer*128+127)+'，B行同范围','本地输入32KiB/stage × 6','输出：本侧128行 × 全部256列'])
  s+=card(35,335,1130,'跨侧乘积例子 D[3,133]',['A[3,k]来自peer0，B[133,k]来自peer1；结果属于peer0的TMEM','leader warp4发一次合作MMA；两侧WG0各自写回本地结果'])
  s+=card(35,510,1130,'通知路径',['full期望65536bytes；empty/acc_full完成通知multicast到两侧','各侧empty count=1；acc_empty由两侧128+128线程在leader归还'])
 elif name=='v09_multi_consumer':
  for c in range(2):
   for peer in range(2):
    x=35+c*590;y=110+peer*145;base=c*256+peer*128
    s+=card(x,y,550,f'consumer{c} / peer{peer}',[f'输出行[{base},{base+128}) × 列[0,256)',f'本地TMEM：TLane0…127 / TCol[{c*256},{(c+1)*256})'],'#dbeafe' if c==0 else '#ede9fe')
  s+=card(35,440,1130,'共享B的安全条件',['每stage：A0 + A1 + B；每位consumer用完后各通知empty一次','empty count=2才允许重写B；两份acc_full/acc_empty则分别推进'])
  s+=text(35,615,'D[387,133]：c1、peer1、local_m3 → thread131（copy tid3），TMEM列389，ei2。',18)
 else:
  s+=text(30,85,'同一warp的32行 × 64列片；每lane最终都接收64个FP32。',19)
  for q in range(64):s+=rect(35+q*17.5,145,16,55,'#dbeafe')
  s+=text(35,129,'1x：每个基础操作取1列，覆盖64列需要64份atom工作',18)
  for q in range(2):s+=card(35+q*570,270,555,f'32x 操作 {q}',[f'片内列 [{q*32},{(q+1)*32})','每lane接收32个FP32'],'#ede9fe')
  s+=text(35,243,'32x：每个基础操作取32列，同样64列用两份atom工作覆盖',18)
  s+=card(35,470,1130,'形状改变，总数据量不变',['窄fragment：每次1值 × 64重复；宽fragment：每次32值 × 2重复','仍需要wait::ld、转换、输出SMEM、TMA store与回收；不是整个kernel加速32倍'])
 s+=text(30,735,'图为当前源码的教学映射；颜色/方框不代表固定硬件执行时钟或实测耗时。',17)
 save(name+'-mechanism',s,775,name+'关键机制')
def build_lesson_figures():
 for name in NAMES:flow(name);mechanism(name)
def lesson_markdown(name):
 if name not in SPEC:return ''
 v,G,C,S,T,cols,example,focus=SPEC[name];idx=NAMES.index(name)
 prev='v01_single_tile' if idx==0 else NAMES[idx-1]
 next_link=' · [下一版]('+NAMES[idx+1]+'.html)' if idx+1<len(NAMES) else ' · [回到版本目录](index.html)'
 data_kib=(C+1)*16*S if v>=7 else 32*S
 out_kib=0 if v<=3 else 16*C
 return f'''## 从上一版走到本版：完整图解

先读：[上一版]({prev}.html){next_link}。本版重点：**{focus}**。下方保留完整逐行代码，先用图和具体例子建立整体过程，再对照行号。

|量|本版的具体含义|
|---|---|
|合作输出任务|{128*G*C}×{128*G}；每consumer的MMA输出{128*G}×{128*G}|
|CTA组大小 / 线程|{G} CTA；每CTA {T}线程|
|输入stage / consumer|{S} stage；{C}个MMA consumer（v07起为独立MMA角色）|
|每CTA输入/输出SMEM数据|输入{data_kib}KiB；输出缓冲{out_kib}KiB；barrier、字段及对齐另计|
|每CTA TMEM|{cols}列，即128×{cols}个32-bit单元|
|教学例子的M×N×K|{example}；不改变下方已有实测尺寸与数据|

![本版完整过程]({name}-process.svg)

{DETAILS[name]}

### 本版机制放大图

![本版关键机制]({name}-mechanism.svg)

### 如何核对这份讲解

对应本页链接的 `.cu` 源码；CuTe 单/双CTA分片和宽/窄load的坐标核对见 [layout诊断](../results/later-layout-inspection.txt)，可运行 `./scripts/check_later_layouts.sh` 复现。它在B300上用单个GPU线程检查CuTe identity tensor的坐标与分片形状，不执行MMA或TMEM数据读写。时间图只表达依赖；本轮完善文档不修改kernel，不把新增布局检查当作新的性能测量。已有正确性与性能数据仍见页面后半部分。

'''
