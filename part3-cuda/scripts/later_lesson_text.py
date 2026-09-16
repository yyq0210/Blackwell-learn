"""Hand-written, source-specific lessons inserted into the existing v02–v09T pages."""
DETAILS={
'v02_k_loop':r'''
### 1. 从 v01 只改一个问题：K 超过 64 时怎么办？

取合法小例子 M=N=128、K=192。只有一个 CTA，bm=bn=0；但 nk=K/64=3，kt=0、1、2。每个 kt 内 kb=0…3，所以共发 **12 条 K16 MMA**，始终更新同一块 128×128 的 acc。

```text
全局 K = 64×kt + 16×kb + ki
kt=0：K[0,64)    → 四条 MMA
kt=1：K[64,128)  → 四条 MMA，接着累加
kt=2：K[128,192) → 四条 MMA，最后才写 D
```

若 A[3,k]=1、B[5,k]=k+1，D[3,5] 三轮的大块贡献为 2080、6176、10272，累计为 2080、8256、18528。这里只为解释数学选定示例数据；基准测试使用随机输入。

### 2. 追踪 A[3,82]，与 v01 的 A[3,18] 比较

完整矩阵的行 stride 变成 192，**不要继续用 GMEM 行 stride=64**。

|视图|实际 shape|观察点|
|---|---|---|
|a|`(128,192)`|a(3,82)，GMEM 元素偏移 3×192+82=658|
|ga|`(128,64,3)`|ga(3,18,1)|
|pa|`((128,16),1,4,3)`|pa((3,2),0,1,1)|
|sa|`((128,16),1,4)`|sa((3,2),0,1)，本轮装入 K[64,128)|
|ra|`(1,1,4)`|ra(0,0,1)，当前缓冲中的第二片 K16|
|acc / pd|`((128,128),1,1)`|acc/pd((3,5),0,0)，逻辑输出不变|

ga/pa 是 GMEM 视图；sa 是固定大小的 SMEM 缓冲。下一轮改变了源的 kt，但目标仍是同一份 sa。ra 描述 SMEM 中当前这批数据，不需要拥有 nk 个 descriptor 槽。示意 SMEM 基地址为 1024-byte 对齐时，局部 (3,18) 仍放在目标元素202；源地址已变为658。

### 3. 沿真实执行顺序看一次循环

```text
初始化 TMEM、barrier；accumulate=Zero（只在整个 K 循环前设置）
kt=0：128线程装 A/B → fence+CTA同步 → 4次MMA → 等 phase0
kt=1：同一块SMEM装下一批 → 同步 → 4次MMA → 等 phase1
kt=2：再次覆盖SMEM → 同步 → 4次MMA → 等 phase0
TMEM→REG → 等load完成 → 转FP16 → REG→GMEM → 同步并释放
```

phase 是同一 barrier 完成轮次的奇偶位，不是另一块内存，也不是“还要等几个线程”。第三次等待 phase0 是等待新的一轮，不是重新读取第一轮的状态。

SMEM 能被覆盖的依据是上一轮 MMA 已完成；TMEM 则一直保留同一输出的累计值。它们的复用时刻不同。

### 4. 线程和 layout：哪些继承，哪些变化？

仍是 128 线程、4 个 warp；普通 cooperative_copy 装输入，warp0 提交 MMA，所有线程等待，再合作写回。没有额外的 producer warp，也没有输入 stage 环。

当前 `cooperative_copy<128>` 默认 MaxVecBits=16，不能解释为128-bit向量。对本版这种布局，输入分工仍按局部 K64 窗口的坐标组织；源全局 k 还需加64×kt。输出仍每线程负责一个逻辑行。

真正改变的是源 GMEM 的完整 row stride、K 大块轴的长度和循环次数，SMEM/tmem tile 的形状没有随完整 K 扩大。

### 5. 容易出错的改法及复习答案

- 每个 kt 重新设 Zero：只剩最后一轮贡献，本例会得到10272而非18528。
- 不等 MMA 就重写 sa：Tensor Core 可能还在读取上一轮输入。
- 每轮写回并重新加载 D：本版不这样做，累加留在 TMEM 到最后。
- **为什么本版仍慢？** 单 CTA 无法让整张 GPU 充分并行，而且每轮都装完、算完、等完再进入下一轮。改变 K-loop 并未引入重叠。

阅读源码时把 `2>=3` 化为 false，`2==1` 化为 false。它们是生成器的常量条件，不是 GPU 上每次随机选择的分支。
''',
'v03_multi_cta':r'''
### 1. 从“只有一个输出块”变成“很多独立输出块”

取 M=256、N=384、K=192。grid=(2,3)，共6个CTA；每个CTA128线程。`bm=blockIdx.x`，`bn=blockIdx.y`，而 kt/kb 仍像 v02 一样在每个 CTA 内循环。

```text
D 的 tile 网格（每格128×128）：
          bn0       bn1       bn2
bm0     CTA(0,0)  CTA(0,1)  CTA(0,2)
bm1     CTA(1,0)  CTA(1,1)  CTA(1,2)
```

**并行的是输出空间 M/N，归约 K 没被分给不同 CTA。** 每个CTA自己算完所属输出的全部192项，所以无需CTA之间求和，也无需atomic。

### 2. 固定 CTA(1,2)，追踪同一项乘积

它写 D[128:256,256:384]，需要 A[128:256,:] 和 B[256:384,:]。注意 B 存为(N,K)，对应输出的列范围。

```text
观察输出 D[131,261]：块内 (m,n)=(3,5)
kt=1,kb=1,ki=2 → global k=82

A[131,82] ↔ ga(3,18,1) ↔ pa((3,2),0,1,1)
B[261,82] ↔ gb(5,18,1) ↔ pb((5,2),0,1,1)
D[131,261] ↔ gd(3,5) ↔ pd((3,5),0,0)
```

A 的元素偏移是131×192+82=25234；D 的元素偏移是131×384+261=50565。GMEM 地址要用完整矩阵 stride；SMEM 输入使用局部 tile 的 layout，不把全局行号131直接代入局部行号3的位置。

### 3. 同一份代码，在六个 CTA 中各自执行

每个CTA独立申请TMEM、初始化自己的SMEM/barrier，沿K跑3轮、每轮4条MMA，最后只写自己那一格。ga/gb shape 为(128,64,3)，pa/pb为((128,16),1,4,3)，与v02的小块视图结构相同。

`mma.get_slice(0)` 仍是本次单CTA MMA的唯一参与者0，**不应改成 get_slice(blockIdx.x)**。bm选择矩阵中的窗口；slice选择该条MMA的参与者，这两个编号不在同一层。

threadIdx.x=3 在整个 grid 中并不唯一：每个 CTA 都有自己的 thread3。这里 CTA(1,2) 的 thread3 写 D 第131行的相应128列；CTA(0,2) 的 thread3 写 D 第3行的相应128列。

### 4. 复用的是哪份输入？

- 同一 bm、不同 bn：需要相同的 A 行范围，但配不同 B，因此产生同一输出行带的不同列块。
- 同一 bn、不同 bm：需要相同的 B 行范围，但配不同 A。
- 每个 CTA 的 SMEM 独立；代码没有跨CTA传送 A/B，也没有显式 cluster。重复 GMEM 读取可能命中缓存，不能说“这些CTA共用同一SMEM”。

硬件决定这些 CTA 何时运行、落到哪些 SM。grid=(2,3) 是逻辑工作网格，不是要求 GPU 的 SM 排成2×3。

### 5. 从这里开始怎样公平比较性能？

v02只算一个128×128输出，v03能覆盖完整4096×4096输出。不能直接拿两者耗时相除当作纯优化倍数。应先固定问题尺寸；后续v03→v09的4096³实测才更适合比较。

**复习：为什么 bm 相同的两个 CTA 不能写同一个 acc？** 它们输出的 bn 不同，且TMEM/SMEM各自独立；本版的每个CTA拥有一整块输出，没有部分结果合并协议。
''',
'v04_tma':r'''
### 1. 本版替换了两端的数据搬运

v03输入由128线程逐份load/store，结果直接REG→GMEM。v04改为：

```text
输入：一个发射线程提交 TMA，GMEM → SMEM
计算：warp0 提交 MMA，SMEM → Tensor Core → TMEM
输出：128线程 TMEM → REG → FP16 → 输出SMEM，再由一个线程提交 TMA → GMEM
```

异步引擎并不意味着自动重叠。本版只有一份输入buffer，实际仍是load完成→MMA完成→下一次load。

### 2. CPU 上的 descriptor 与 GPU 上的坐标视图

`make_tma_atom(SM90_TMA_LOAD{},a,LA{},Shape<128,64>{})` 编码TMA需要的全局矩阵信息和目标排列。名字SM90表示这类TMA接口的来源，并不意味着此Blackwell kernel运行在Hopper上。

`get_tma_tensor(shape(a))` 返回供TMA使用的**坐标型Tensor**。它不是一个能随便用普通 `a(m,k)` 解引用获取FP16数值的GMEM指针Tensor。数学坐标仍然一样，实际地址转换由descriptor与坐标共同完成。

`CUTE_GRID_CONSTANT` 让descriptor作为只读grid参数传入；本kernel对它的地址有要求，因此不能随意去掉这个限定。

### 3. 跟一次输入：M=256,N=384,K=192，ti=5

本版用一维grid：tiles_m=2，tiles_n=3，`bm=ti%2=1`，`bn=ti/2=2`。这与v03的CTA(1,2)选同一输出窗口，但 blockIdx 的组织变了。

A[131,82] 仍对应ga(3,18,1)、pa((3,2),0,1,1)。现在没有“thread82亲自搬这个输入元素”的分工；warp0中选出的一个线程提交TMA请求，引擎搬整片。

```cpp
auto [ag, as] = tma_partition(tma_a, ..., group_modes<0,3>(sa),
                                                   group_modes<0,3>(pa));
copy(tma_a.with(s.full[0]), ag(_,kt), as);
```

`group_modes<0,3>` 将输入片的前三个顶层mode作为一个整体，保留后面的K大块轴。`ag(_,kt)` 选择全局源tile；`as` 指定本次shared目标。`tma_partition` 只建立这两个对应视图；`copy` 才提交传输。

### 4. 两个 barrier 分别等什么？

|对象|本版保护的事件|什么时候可以做下一步？|
|---|---|---|
|full[0]|A和B的TMA输入到齐|可以让MMA读取输入SMEM|
|done|这一K64块的4条MMA完成|可以覆盖这份输入SMEM|

每片A是128×64×2=16384 bytes，B相同，合计32768 bytes。`set_barrier_transaction_bytes` 登记字节数并执行相应arrival，不是搬运指令。full的count=1不表示只传一份矩阵，也不等于有32768个线程。

```text
先提交 kt0 的TMA
kt0：等full phase0 → 4次MMA → 等done phase0 → 提交kt1
kt1：等full phase1 → 4次MMA → 等done phase1 → 提交kt2
kt2：等full phase0 → 4次MMA → 等done phase0
```

full完成不能替代done：输入已经到达，不代表MMA已经用完输入。

### 5. 输出为什么还多了一块 SMEM？

输出acc为128×128，本版TMA store每次使用128×64。`zipped_divide` 将输出拆成“子块内部坐标 + 子块编号ei”，ei=0写列0…63，ei=1写列64…127。

```text
acc/pd → acc_epi/gd_epi → 当前ei
TMEM load → rf（每线程64个FP32） → rh（64个FP16）
→ sd（128×64输出SMEM） → TMA store → 对应GMEM列块
```

`LD` 是输出SMEM布局，和输入 `LA` 用途不同。sd是实际输出缓冲的视图；`tma_partition(tma_d,sd,gd_epi)` 返回dg（GMEM坐标）与ds（SMEM源）。不要因为名字dst出现在寄存器copy中，就以为它已指向最终GMEM。

先所有写回线程执行store fence，再CTA同步；发射线程发TMA store并commit/wait；再CTA同步才让全部线程覆盖sd，处理下一个ei。只让一个发射线程等完、其他线程立刻覆盖sd，会破坏输出传输。

**复习：D[131,261]由谁写？** 本CTA的thread3负责生成该行的寄存器结果并写到sd；最终GMEM传输由TMA完成，不能再说thread3直接执行该输出元素的GMEM store。
''',
}
DETAILS.update({
'v05_double_buffer':r'''
### 1. 两份缓冲，不是两块同时计算的输出

仍用M=256,N=384,K=192，观察CTA对应的输出D[128:256,256:384]。nk=3、kt=0…2，acc仍只有一块128×128。新增的是 `s.a[2]` 与 `s.b[2]`：每个stage放一片A和一片B。

|资源|stage0|stage1|
|---|---|---|
|A输入|s.a[0]，16KiB|s.a[1]，16KiB|
|B输入|s.b[0]，16KiB|s.b[1]，16KiB|
|到达通知|full[0]|full[1]|

64KiB只是两套输入；输出SMEM另有16KiB，barrier/字段和对齐另计。stage并不是线程号，也不是TMEM列编号。

### 2. 真实代码的预取顺序

```text
启动：TMA(kt0 → stage0)

kt0：等stage0到齐
     提交 TMA(kt1 → stage1)
     提交 MMA(kt0，从stage0读)，等done

kt1：等stage1到齐
     提交 TMA(kt2 → stage0)  ← kt0的MMA已完成，可以覆盖
     提交 MMA(kt1，从stage1读)，等done

kt2：等stage0到齐
     没有下一块，不再预取
     提交 MMA(kt2)，等done，然后写回
```

第1次TMA预取kt1可能与kt0计算重叠；但每轮仍立即等done，同一warp组织加载与MMA。它还不是v07那种独立producer/consumer指令流。

### 3. 把 kt、stage 和 phase 分开

本版每CTA只处理一个输出tile，tile_round=0。`stage=kt%2`，等待full的phase为`(kt/2)&1`。

|kt|全局K范围|当前stage|full等待phase|done等待phase|本轮预取目标|
|---|---|---|---|---|---|
|0|[0,64)|0|0|0|kt1→stage1|
|1|[64,128)|1|0|1|kt2→stage0|
|2|[128,192)|0|1|0|无|
|3（若K更大）|[192,256)|1|1|1|下一块→stage0|

full[0]、full[1]各有自己的使用次数；done只有一个，所有MMA批次轮流用。**所以full phase不能简单用kt&1。** kt1第一次使用full[1]，应等phase0。

### 4. layout 没有神奇地增加 stage 轴

```cpp
auto sa = make_tensor(make_smem_ptr(s.a[stage].begin()), LA{});
```

当前源pa含kt轴，当前目标sa的shape仍为((128,16),1,4)。stage通过 **选择不同数组的基地址** 表达，不是给LA新增一个维度。`na/nb` 是预取下一stage时创建的另一组视图。

对A[131,82]：kt=1、局部k=18，选择stage1；ga/pa指全局源，sa指s.a[1]中的对应逻辑位置。下次stage1复用时数值会变，layout的解释规则不变。ra/rb由当前sa/sb构造，必须与实际读的stage一致。

### 5. 为什么这里没有 empty[stage]？

因为控制流本身保证回收：进入kt1预取kt2之前，代码已等kt0的done，stage0已安全；进入kt2前又等kt1的done。没有独立producer能越过这个等待继续前进。

若把预取移到其他独立warp，不能仅照搬这些数组而去掉等待；必须增加明确的“读完才能覆盖”协议，这正是后续empty barrier的用途。

### 6. 写回与性能边界

输入双缓冲没有改变写回：每个输出依然经两个128×64的ei，TMEM→REG→sd→TMA store。输入stage0/1与输出sd不是同一批数组，名字中的“buffer”不要混用。

**复习：nk=1会明显受益吗？** 没有下一K块可预取，重叠机会很少，却仍有两份输入存储成本。因此本版并不保证任意尺寸都比v04快。
''',
'v06_persistent':r'''
### 1. CTA 不再等于一个输出 tile

v05有多少输出tile就启动多少CTA。v06启动 `grid=min(tile数量,SM数量)` 个CTA；每个CTA循环领取固定序列：

```text
CTA blockIdx.x=b：ti=b, b+gridDim.x, b+2×gridDim.x, ...
```

这叫持久化工作循环。`ti`是逻辑任务编号，`tile_round`是**当前CTA已经处理过几个输出任务**；二者不相同。

本机148SM、M=N=2048时，mt=nt=16，总256个输出tile，启动148个CTA。CTA0处理ti=0和148；CTA107处理107和255；CTA108以后只处理一个任务。这是本机配置下的具体例子，不保证CUDA把CTA编号直接绑定到同编号SM。

### 2. grouped_tile 怎样把 ti 换成 bm/bn？

按8行M tile分组，先在组内沿M走，再换N列。用小网格mt=10、nt=4看编号顺序：

```text
ti 0..7   → (bm=0..7,bn=0)
ti 8..15  → (bm=0..7,bn=1)
...
ti 24..31 → (bm=0..7,bn=3)
ti 32,33  → (8,0),(9,0)   ← 最后一组只有2行
ti 34,35  → (8,1),(9,1)
```

源码中的 `rows=min(8,mt-base)` 必须按最后一组剩余行数计算；不能所有组都写死8。

回到上面的16×16网格：ti=148时，group=1、base=8、local=20，得到bm=12、bn=2。CTA0的第二项输出是D[1536:1664,256:384]，不是简单的“第148行”。

### 3. 一个 CTA 的资源怎样跨任务留下来？

```text
申请TMEM/初始化barrier（只一次）
任务ti0：重新建立ga/gb/gd → acc模式Zero → 完整K循环 → 全部ei写回
任务ti148：换窗口但复用原SMEM/TMEM → Zero → 完整K循环 → 写回
退出工作循环后再释放TMEM
```

sa/sb所在的两套物理数组不变；ga/gb/gd的矩阵起始坐标改变；acc绑定同一TMEM空间但代表新的输出。每个新输出必须重新设Zero，不能把上一输出的结果继续累加进来。

本版任务之间仍按代码顺序完成写回，没有独立epilogue与下个任务MMA并行；下一任务使用TMEM之前，上一个输出已经处理完。

### 4. 奇数 nk 为什么特别容易把 phase 算错？

取K=192，nk=3。每个输出从stage0开始：stage序列0、1、0。所以每个输出使用stage0两次、stage1一次。

等待full的源码公式：

```text
uses_per_output(stage) = floor((nk+S-1-stage)/S)，S=2
phase = (tile_round×uses_per_output(stage) + floor(kt/S)) & 1
```

|tile_round|kt|stage|该stage此前使用次数|full phase|done phase|
|---|---|---|---|---|---|
|0|0|0|0|0|0|
|0|1|1|0|0|1|
|0|2|0|1|1|0|
|1|0|0|2|0|1|
|1|1|1|1|1|0|
|1|2|0|3|1|1|

`done_phase`跟随每个K64 MMA批次连续翻转；full按各自stage的使用历史翻转。第二个输出第一次用stage1时等待phase1，不能重新从0开始。

### 5. stage 与布局不会自动跟 ti 绑定

本版每个新输出的kt重回0，stage也重回0；但barrier的历史并未清空。和v07的差别是：v07使用跨输出连续的iteration来选stage，未必每个输出从stage0开始。

对CTA0第二个任务中A[1539,82]：bm=12、kt=1，ga(3,18,1)与pa((3,2),0,1,1)指向该源，目标为stage1；tile_round=1使其full等待phase1。

### 6. 复用、局部性和调度保证

grouped-M让相邻逻辑任务更近地访问相同B，并限制A复用距离，期望提高缓存复用。**它没有创建跨CTA共享SMEM，也不保证硬件严格按ti顺序执行。** 持久化减少重复的CTA级资源准备，但是否提速仍取决于串行等待、负载均衡和问题规模。

**复习：tile_round能直接用ti吗？** 不能。CTA0第二次做ti148时round是1；phase依赖这份barrier实际经历了多少轮，而不是矩阵中的任务编号有多大。
''',
'v07_warp_specialized':r'''
### 1. 把三种工作交给三条可以独立前进的指令流

一个CTA现在256线程，分成两个128线程warpgroup。输出仍128×128，输入环为4个stage。

|CTA内线程|warp编号|主循环职责|
|---|---|---|
|0–127|0–3，WG0|等结果、读TMEM、转FP16、填输出SMEM并TMA写回|
|128–159|4，WG1的warp0|MMA consumer0，沿K消费输入|
|160–223|5–6|不承担主循环加载/计算/写回工作，仍参加规定的初始化/退出同步|
|224–255|7，WG1的warp3|TMA producer；warp中选一个线程提交请求|

warp编号从0开始。“WG1 warp0”指CTA的warp4，不是又一个CTA的warp0。初始化TMEM仍由CTA的warp0协作完成，与主循环计算角色不同。

### 2. 一份输入 stage 的完整生命周期

```text
producer：等 empty[st]（首次使用不用等）→ 提交 TMA
TMA完成：full[st] 就绪
MMA warp：等 full[st] → 提交4次MMA → commit到 empty[st]
MMA真正用完输入：empty[st] 就绪 → producer可复用
```

没有每轮全CTA `__syncthreads()` 来强制所有角色一起跑；producer、MMA、epilogue各自循环，通过barrier连接。异步完成通知保护真实的数据依赖。

这不表示任意角色能无限领先：输入环只有4槽，producer最终会被empty挡住；TMEM只有一份输出slot，MMA进入下一输出前可能被acc_empty挡住。

### 3. 输入和输出是两套独立的交接协议

|barrier|索引|“可以做什么”|谁等它？|
|---|---|---|---|
|full[st]|输入stage|本轮A/B可读|MMA warp|
|empty[st]|输入stage|上轮A/B已用完，可覆盖|producer warp|
|acc_full[0]|输出累加器slot|整个K归约已完成，可读TMEM|写回WG0|
|acc_empty[0]|输出累加器slot|上个结果已搬出，可重用TMEM|MMA warp|

输入SMEM用完不代表最终输出已经写回。最后一个输入stage释放后，epilogue仍可能在读TMEM；下一输出MMA不能因此直接覆盖acc。

acc_empty初始化计数为128，因为本版收集WG0的128个写回线程到达。full、empty、acc_full的count均为1，但代表的事件不同。

### 4. 这次 stage 真正跨输出连续转动

设K=192、nk=3，令r为同一CTA处理输出的轮数：

```text
i = r×nk + kt      // 各角色自己的iteration，进度可能不同
st = i % 4
MMA等 full phase = floor(i/4)%2
producer若 i>=4，等 empty phase = (floor(i/4)-1)%2
```

|输出轮r|kt|i|stage|full phase|写入前是否等旧empty|
|---|---|---|---|---|---|
|0|0|0|0|0|首次使用，不等|
|0|1|1|1|0|不等|
|0|2|2|2|0|不等|
|1|0|3|3|0|不等；与v06重新用stage0不同|
|1|1|4|0|1|等旧phase0|
|1|2|5|1|1|等旧phase0|

各角色的iteration是线程私有计数，不是所有warp共同递增的共享整数。它们按相同任务序列解释计数，通过barrier允许一方领先。

### 5. layout 视图和寄存器如何衔接？

每个stage仍是一份A128×64与B128×64：32KiB×4=128KiB；输出sd另16KiB，共144KiB有效数据加元数据。producer构造TMA坐标源pa/pb与对应sa/sb；MMA warp用同stage的SMEM descriptor ra/rb；epilogue只看完成后的acc。

写回有两个ei=0/1，每次128×64。其copy线程号是`tid=threadIdx.x%128`。本版写回只用thread0…127，因此tid数值恰好相同；不要把管理WG的thread128…255也传进去执行同一写回。

`NamedBarrier::sync(128,c)`只集合该consumer的写回WG，不能换成全CTA同步：其他warp可能正在等写回通知，形成互相等待。

### 6. 一次输出的完整故事和复习

producer填stage，MMA拿到输入后做K归约，输入可以逐stage释放；最后提交acc_full。WG0等待它，分两个ei把TMEM写到GMEM，再由128线程通知acc_empty。MMA处理下一个输出前等自己的acc_empty，而producer可在空闲stage内继续预取。

**复习：producer可以在上一输出还没写完时装下一输出吗？** 可以，只要目标输入stage的empty允许。**下一输出能直接覆盖TMEM吗？** 不能，还要等acc_empty。两份资源的生命周期不同，正是三角色能重叠又不踩数据的关键。
''',
})
DETAILS.update({
'v08_two_cta':r'''
### 1. 先区分三种尺寸：cluster、单CTA、单条MMA

本版显式启动cluster=(2,1,1)。每个cluster有两个CTA，每CTA256线程，合计512线程。一次合作输出256×256；每CTA拥有本地128×256的TMEM结果。指令是256×256×16的2-CTA MMA。

`gridDim.x=clusters*2`，`peer=blockIdx.x%2`，`cluster=blockIdx.x/2`。`mma.get_slice(peer)` 才表示这次MMA的两侧份额；它不是在选择矩阵中第几个输出tile。

每个cluster仍按 `ti=cluster, cluster+cluster_count,...` 做持久化任务。bm/bn是256×256输出块网格的坐标；它们与peer不同。

### 2. 数据如何分在两个CTA上？

观察bm=bn=0、任意kt：

|CTA|本地输入A|本地输入B|本地输出TMEM|
|---|---|---|---|
|peer0|A行0–127|B行0–127|D行0–127，列0–255|
|peer1|A行128–255|B行128–255|D行128–255，列0–255|

A/B各侧都是128×64。**结果不是CTA0只算左上角、CTA1只算右下角。** 两侧B合起来覆盖全部256个输出列，每侧A要与两侧B共同参与计算。

具体看D[3,133]的k=82项：A[3,82]由CTA0输入片提供，B[133,82]由CTA1输入片提供；结果放CTA0的TLane3/TCol133。CTA0写回thread3在ei=2处理列128…191时接收这一结果，片内列为5。

### 3. 视图链中 peer 在哪里生效？

```text
local_tile：先选完整256×64的A/B窗口、256×256输出窗口
mma.get_slice(peer).partition_A/B：各侧选128行输入
partition_C：各侧选128行结果，但保留全部256列
make_fragment_C：每侧的128×256本地TMEM视图
```

K=192时，pa/pb的本地shape为((128,16),1,4,3)，pd为((128,256),1,1)。shape看起来依旧有128，不代表整个MMA仍是128×128，而是已经选过peer份额。

TMA descriptor的创建也同时使用MMA与cluster layout。`SM100_TMA_2SM_LOAD` 不是让CTA0的普通指针随意访问另一CTA数组；它有指定的2SM操作语义。不要把单CTA的descriptor构造与barrier通知原样替换进来。

### 4. 每侧生产、leader计算、两侧写回

两侧warp7都提交自己那份输入的TMA。只有leader=peer0的warp4提交2CTA MMA；peer1的warp4不再重复发同一计算。两侧WG0都写自己的128行结果。

一次stage完整输入为2×(A16KiB+B16KiB)=65536 bytes。leader登记full的期望字节数，输入就绪交接采用2SM TMA协议；MMA读取两侧输入并把结果分布在两侧TMEM。

`umma_arrive_multicast_2x1SM(...,3)` 中mask3=二进制11，完成通知送到两侧。**它是通知目标掩码，不是三个CTA，也不是arrival计数。**

|通知|count|为什么？|
|---|---|---|
|每侧empty[stage]|1|只有一个MMA消费者完成后通知两侧，不是每CTA各有一个独立消费者|
|每侧acc_full[0]|1|这个消费者的最终MMA完成事件|
|leader的acc_empty[0]|256|两侧各128个写回线程都不再使用结果|

### 5. 六stage环与两个CTA的生存期

本实现S=6，与v07的S=4也不同；不要把实测变化全归因于“CTA数翻倍”。每CTA输入32KiB×6=192KiB，输出缓冲16KiB，合208KiB有效数据；TMEM每侧256列=128KiB。

例如nk=7：i0…5首次使用六个stage；i6复用stage0，producer先等旧empty phase0，MMA随后等full phase1。stage公式仍是连续iteration，不因输出切换自动清零。

`cluster_sync()`使两侧初始化完成后再协作；退出也要同步，避免另一侧仍有远端访问时释放。`__syncthreads()`只能同步本CTA，不能替代它。

### 6. 输出仍分小块，双CTA不等于一个巨大store

每侧128×256结果分成4个128×64的ei，每侧WG0自己准备sd并发TMA store。分工按本地tid=0…127，最终全局行要加输出tile起点和peer×128。

**复习：CTA0完成写回，能马上覆盖TMEM开始下个输出吗？** leader必须收齐两侧256次acc_empty到达；否则CTA1可能还在读它那侧的结果。两个CTA的输入/结果属于同一次协作计算。
''',
'v09_multi_consumer':r'''
### 1. 不是一个warp维护两份acc，而是两条独立MMA循环

cluster仍含2CTA，单次MMA仍输出256×256；新增consumer c=0、1，共享同一份B，各用不同A。因此一个逻辑任务覆盖512×256输出。

```text
consumer0：A的前256行 × 同一B → D前256行
consumer1：A的后256行 × 同一B → D后256行
```

这里两个消费者是 **leader CTA 中两个不同的 MMA issue warp**，各有独立iteration/round、等待和K-loop。若改成一个warp内 `for(c=0;c<2;c++)`，数值可能仍正确，但不再是本版的独立消费机制。

### 2. bm、c、peer、tid 四层编号串起来

本版CF::BM=256，BN=256，C=2。bm按512行一个逻辑输出任务计数；`make_coord(bm*2+c,bn,_)` 再选择该consumer的256行MMA tile；最后peer选择其中128行。

```text
全局输出行 = (bm*2+c)*256 + peer*128 + local_m
全局输出列 = bn*256 + local_n
本地TMEM列 = c*256 + local_n
```

以bm=bn=0、c=1、peer=1、local_m=3、local_n=133为例：全局D[387,133]，该CTA的TMEM位置TLane3/TCol389。

其输入A[387,82]对应consumer1第二CTA的本地A[3,18]；B[133,82]来自B的peer1份额。K坐标仍拆为kt=1、kb=1、ki=2。

### 3. 每CTA384线程怎样分工？

|线程|CTA内warp|职责|
|---|---|---|
|0–127|0–3，WG0|consumer0的写回，两侧都执行|
|128–255|4–7，WG1|consumer1的写回，两侧都执行|
|256–287|8，WG2 warp0|consumer0的MMA，仅leader执行|
|288–319|9，WG2 warp1|consumer1的MMA，仅leader执行|
|320–351|10|不承担主循环工作|
|352–383|11，WG2 warp3|producer，两侧各自提交TMA|

peer1的warp8/9不发MMA。初始化/释放TMEM的warp0仍是每CTA的第一个warp，不是管理WG中的“MMA warp0”。

观察D[387,133]的写回：peer1、c=1、局部tid=3，因此 **CTA内threadIdx.x=131**。调用 `cp.get_slice(tid)` 用3，不能直接传131。`ei=2`对应输出列128…191，rf中该列的片内位置为5。

### 4. 两份A、一份B，两段TMEM，两块输出buffer

```text
每CTA输入stage st：
    a[st][0]：consumer0的128×64 A
    a[st][1]：consumer1的128×64 A
    b[st]   ：共同的128×64 B

每CTA TMEM：
    consumer0：列[0,256)
    consumer1：列[256,512)

每CTA输出：d[0]、d[1]各一块128×64缓冲
```

stage有4个，共(2×16+16)KiB×4=192KiB；输出另32KiB，总224KiB有效数据加元数据。每CTA申请512列TMEM=256KiB。这里的TMEM列512是分配总量，单个consumer依然是256列。

所有输入视图只覆盖“当前stage、当前consumer或共享B”。`pa`不会因为全任务有512行就让每侧A变成512×64；consumer和peer两层选择已经在窗口与slice中发生。

### 5. B为什么不会被先完成的consumer提前覆盖？

每个consumer在本stage的MMA后各提交一次empty完成通知。`empty[st]` 的arrival count=2，所以只有两人都真正用完，producer才能重写该stage的A/B。

```text
consumer0用完B → empty收到1次：仍不能覆盖
consumer1用完B → empty凑齐2次：现在才能覆盖
```

两侧CTA都收到各consumer的multicast完成通知；每侧count是2，不是“2CTA×2consumer=4”。每次stage全输入期望字节为2CTA×(2A+B)×16384=98304 bytes。

### 6. 输出的两个生命周期不绑在一起

`acc_full[c]` 与 `acc_empty[c]` 每位consumer一对；WG0只等c0的acc_full，WG1只等c1的acc_full。c0已经完成便可先写回，无需等c1全部K循环结束。

每个acc_empty[c]仍需要两侧各128线程、合计256次到达；两个barrier分别计数，不能做成一个512次共同barrier。MMA消费者进入下个输出前只等自己的上一slot释放。

写回的 `NamedBarrier::sync(128,c)` 也按consumer区分，避免WG0与WG1误同步。不同角色可有先后，图中的两条MMA线不表示硬件保证它们同时发射或固定相差一拍。

### 7. 复习：共享B究竟省了什么？

同一stage只为两个consumer装一份B，每CTA输入从“独立两份A+B”的64KiB降为48KiB。A仍各一份，算力需求也仍是两块输出；代价是更大TMEM、两个consumer的回收协议和更多线程。能否提速依赖实际资源与调度，不是消费者数翻倍就吞吐翻倍。
''',
'v09_tuned':r'''
### 1. 把本次改动限定到一条操作原语

本版与v09相比，源码唯一算法差异是：

```cpp
// 原版
auto cp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, ae(_, _0{}));
// 调优版
auto cp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, ae(_, _0{}));
```

cluster、两个独立MMA消费者、四stage输入环、TMA、barrier与输出ei循环都保留。先读v09对它们的解释；本节只把“如何把已算好的TMEM结果搬进寄存器”放大。

### 2. 64列输出片怎样用两种atom覆盖？

每个ei处理每CTA的128×64 FP32结果，由128个写回线程分成4个warp。对一个warp，它处理32条TMEM datapath、64列：

|atom|每个基础操作覆盖|按atom粒度覆盖64列|每线程该ei总接收数|
|---|---|---|---|
|32dp32b1x|32行×1列，每lane1个FP32|64个基础窄操作|64个FP32|
|32dp32b32x|32行×32列，每lane32个FP32|2个基础宽操作|64个FP32|

这是按copy atom形状数的操作粒度，最终指令编码与调度要看编译结果，不是声称整个kernel指令总数变成1/32。两种方案搬出的结果总量相同，TMEM累计数值也相同。

### 3. fragment 的 shape 为什么会变？

make_tmem_copy把“每条操作取多少值”纳入线程/数值映射。当前每线程一个64列输出片中，窄版的目标分片顶层为`((1,1),(64))`，宽版为`((32,1),(2))`；两者元素数都是64。

因此不能只改底层PTX的名字、仍假定旧寄存器参数数量和分片格式。这里重新通过cp构造th/src/dst，再用 `shape(dst)` 创建rf/rh，让CuTe保持它们一致。

源视图仍描述warp协作窗口，不能把其逻辑元素数当成每线程寄存器数。实际输出坐标映射已用CuTe identity tensor核对；宽窄版同一线程依然负责同一个逻辑输出行。

### 4. 跟D[387,133]跨过宽load

沿用v09：peer1、consumer1的thread131，局部copy tid=3，ei=2。该输出片覆盖列128…191，D[387,133]是片内第5列。

```text
acc 的全输出局部列133，TMEM存储列389（含consumer偏移256）
→ ei2 的片内列5
→ 第一个32列宽load（片内列0…31）
→ 本线程rf中对应元素 → rh → d[consumer1]中的对应位置
→ TMA store写到D[387,133]
```

第二个宽load取片内列32…63。写回仍要经过 `tcgen05.wait::ld`，数值转换、输出SMEM fence、写回WG同步、TMA store commit/wait和acc_empty通知；宽load不取消任何生命周期约束。

### 5. 少发指令，为什么不必然超过cuBLAS？

宽load减少基础读取操作数量，但可能改变寄存器活跃范围、编译器调度和发射压力。其他瓶颈——TMA、MMA、输出store、同步、SM占用率——不会因为这一行自动消失。

本仓库保留了调优版SASS中的`LDTM.x32`，说明确实使用了宽读取；也保留完整数值检查和性能结果。4096³仍未超过同轮普通cuBLAS；8192³对普通cuBLAS略快、与cuBLASLt基本持平。它是有实测依据的局部调优，不能作为所有尺寸的胜出保证。

### 6. 自己调优时的复习清单

1. **总元素数变了吗？** 没有，每线程每ei仍64个FP32。
2. **consumer0/1的TMEM范围变了吗？** 没有，仍为[0,256)、[256,512)。
3. **可以省掉wait::ld吗？** 不能，宽度改变不等于load变成同步完成。
4. **为什么用shape(dst)分配rf而不手写数组32？** 一条atom的32个值不等于整个64列片的每线程总输出；让copy映射决定fragment结构更稳妥。
''',
})
