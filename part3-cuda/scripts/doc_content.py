"""中文逐版讲义的人工编写内容；浏览器图示由 docs.py 渲染。"""
TITLES=['单 CTA、单 K tile','沿 K 累加','多 CTA 空间分块','TMA 搬运与写回','双缓冲','持久化 tile 调度','Warp specialization','双 CTA 合作','两个独立 MMA 消费者','第9版的宽 TMEM load 调优']
INTROS=[
'''先只算 D 的一个 128×128 方块，输入 A、B 都是 128×64。每个输出元素做64次乘加。整个 kernel 只有一个 CTA、128个线程。

把数据想成一批待加工原料：global memory 是远处仓库，SMEM 是 SM 内的工作台，Tensor Core 是加工单元，TMEM 是它存放累加结果的位置，寄存器则是每个线程手里的小篮子。CuTe 的 Tensor/layout 描述“东西在哪里”，copy 才真的搬东西。

一条 MMA 指令覆盖128×128×16，所以 K=64 要发4条指令。第一次设置 ScaleOut::Zero，不读取原来未初始化的 TMEM；后3次设置 One，保留并累加已有结果。`tmem.allocate(128)` 中的128是硬件列数；每列128个32-bit lane，容量128×128×4=64 KiB。

执行顺序：128个线程合作加载 A/B → shared-memory proxy fence 和 CTA 同步 → warp0 发4条 MMA → 所有线程等待完成 barrier → 从 TMEM 取结果、转换FP16、写回。只有后一步依赖前一步结果时，才需要对应的完成等待。''',
'''输出仍只有一个128×128 tile，唯一核心扩展是允许 K>64。若 K=4096，外层 kt 跑64轮；每轮再发4条 K16 MMA，总共256条 MMA。

注意清零发生在整个 K-loop 之前，而不是每个 kt 的开头。若每轮都清零，得到的只有最后64个 K 元素的贡献。MMA 完成 barrier 会循环使用；第 kt 轮等待的 phase 是 kt&1，即0、1、0、1……。这不是“第几块共享内存”，而是同一 barrier 第几轮完成的奇偶标记。

每轮都先加载，再算，再等，资源不能重叠。这个版本用来建立正确的累加和回收关系；一个 CTA 无法充分利用整张 GPU，不能把此版本的吞吐与完整4096³ cuBLAS比较。''',
'''现在 M、N 都能包含多个128×128输出 tile。grid=(M/128,N/128)。例如 M=N=4096，grid为32×32，共1024个CTA。

CTA(bm,bn)读取 A[bm*128:(bm+1)*128, :] 和 B[bn*128:(bn+1)*128, :]，写 D 对应方块。K-loop仍和第2版一致，每个CTA独立完成其输出的全部K累加，所以无需atomic，也无需CTA之间求和。

相同bm、不同bn的CTAs会读取相同A；相同bn、不同bm的CTAs会读取相同B。目前没有显式跨CTA共享这些输入，可能由硬件cache命中提供帮助。不要把软件tiling理解成自动在CTA之间共享SMEM。

这个版本第一次适合在4096³上与后续版本比较。相较第2版的单tile，加速同时含“问题规模变大”和“并行度增加”，因此文档不计算第2→3版的简单累计加速倍数。''',
'''普通 global load→线程寄存器→shared store 被 TMA load 替换。CPU先编码TMA descriptor，GPU选出的一个线程发起搬运。descriptor描述源张量形状、stride、目标swizzle与tile形状，并不携带矩阵数据。

`CUTE_GRID_CONSTANT` 很关键：descriptor作为整个grid共享的只读kernel参数保留。否则编译器可能把取地址的参数复制到线程local memory，TMA使用该地址会失败。首次开发确实暴露过这个问题，当前源码已修正。

每次 A/B 预期字节数=128*64*2*2=32768。full barrier既等线程arrival，也等这些TMA字节完成。full就绪后才读SMEM；MMA的done完成后才复用SMEM。两条barrier保护不同事件，不能互换。

写回按128×64切两轮：TMEM→FP32寄存器→FP16寄存器→swizzled SMEM→TMA store。`tma_store_fence`使普通shared store对TMA可见，CTA同步等待所有线程完成写入；发起线程commit并wait，最后再次CTA同步才能覆盖这块输出SMEM。此版每个阶段仍立即等待，异步指令尚未充分重叠。''',
'''准备两份A/B共享内存，stage0与stage1轮换。kt=0开始前先装第0块；第0块到达后，将第1块搬进另一份buffer，再发第0块MMA。这样第1块的搬运可以和第0块计算重叠。

这里的stage是数据缓冲区，不是warp或线程。kt%2选择buffer，phase标识该buffer第几次被写满。图中的时间是教学示意，不是测得的纳秒值。

第0块MMA完成后stage0可以复用，于是在计算第1块前向stage0预取第2块。这份实现仍让同一个warp组织load和MMA；下一章的warp specialization才让两个角色拥有独立指令流。

输入SMEM从32KiB增至64KiB，输出分块buffer另占16KiB。双缓冲既有重叠收益，也有容量成本。没有足够K tiles时，预取机会很少。''',
'''启动数量限制在SM数以内，不再为所有输出tile各启动一个新CTA。CTA i处理逻辑tile i、i+gridDim.x、i+2*gridDim.x……，在一个kernel里复用TMEM分配和共享资源。

一个逻辑tile编号还必须映射到二维(bm,bn)。最终代码按原书使用M方向8行一组的顺序：先在一组M tiles内部递增，再换N column；处理完该组所有N columns后再进入下一组M。这样更接近地访问共享B的任务，也限制A的复用距离。这只改变任务编号，并不保证硬件严格按编号执行。

持久化跨越多个输出tile后，barrier phase不能每个tile都重新设0。尤其K/64是奇数时，两份buffer在一个tile内使用次数可能不同。代码为每个stage计算历史使用轮数，避免等待旧一轮的完成状态。

持久化不等于自动提速。如果串行等待仍占主导，或者一次发出全部CTAs已足够调度，单独这一步可能持平或略慢。测试记录保留实际结果，不假设版本号越大越快。''',
'''一个CTA使用两个warpgroup，共256线程。WG0的128个线程做epilogue；WG1的warp0发MMA、warp3发TMA，剩余warps不承担主循环工作。这与作者的角色划分一致。

producer可以提前填4个stage，MMA warp消费已经准备好的stage，epilogue等待当前输出完成。三条指令流能独立推进：MMA等待时producer仍可能工作；输出写回时producer还能预取下一输出tile的输入。

四类barrier形成协议：full[stage]表示输入可读；empty[stage]表示MMA已用完输入；acc_full[consumer]表示TMEM结果完成；acc_empty[consumer]表示写回线程已经用完TMEM。前两者按输入stage索引，后两者按输出消费者索引。

producer第i次使用stage=i%S。i<S时buffer初始为空，不等empty；之后等empty的((i/S)-1)&1。MMA等full的(i/S)&1。两者phase相差一轮，是因为一个等上次回收，一个等本次填充。''',
'''一个cluster包含两个CTA，分别位于两个SM，使用cta_group=2的MMA。一次合作输出256×256；每个CTA拥有128×256的本地TMEM结果。

每侧保存A的128×64和B的128×64，合起来覆盖A的256行、B的256行。同一次MMA使用两侧输入；B的两个128-row slices对应输出的两组128 columns，不是每个CTA独立做128×128。

TMA由两侧CTA各自的producer发起，完成事务集中计入leader CTA0的full barrier。预期字节数=2*(16384+16384)=65536。只有leader发MMA；MMA完成后用mask=3向两侧通知。mask=3的二进制为11，两个bit分别选择CTA0和CTA1。

输入回收必须通知两侧producer。输出TMEM回收必须收集两侧各128个writeback线程，共256次arrival。退出前cluster_sync确保远端SMEM/barrier访问已完成，再由两侧协作释放TMEM。''',
'''这里的“两个消费者”明确指两个独立MMA issue warp，不是一个warp轮流做两次MMA。WG2的warp0/1各推进自己的K-loop；warp3负责TMA。WG0/WG1分别写回consumer0/1。每CTA共384线程。

consumer0计算cluster tile的前256行，consumer1计算后256行；两者都计算相同256列，所以共享同一份B。一个cluster输出512×256。每个CTA每stage保存2份A（各128×64）和1份B（128×64），共48KiB；4stages共192KiB，两份128×64写回buffer再占32KiB。

两位消费者分别使用TMEM列[0,256)和[256,512)。消费者c在leader独立等待acc_empty[c]，消费full[stage]，发出MMA，然后通知empty[stage]。empty的初始化arrival count必须为2：只有两个消费者都确认读完，producer才能覆盖共享的B。

每个consumer最后发acc_full[c]，对应writeback WG只等待自己的slot。所以consumer0的写回无需等待consumer1的整个tile完成。任何将它们合并成一个issue warp或一个共同输出barrier的做法，都会削弱这种独立性。

每个writeback线程每轮拿64个FP32结果，转换并写回FP16；256 columns需要4轮。这是epilogue tile大小，与输入BK=64恰好同数值但含义不同。''',
'''保持第9版的两个MMA issue warp、B共享和barrier协议，只把TMEM copy atom从32dp32b1x改为32dp32b32x。

32dp表示32条TMEM datapath，32b表示每次元素宽度；32x使一条指令覆盖更多列。原来依赖大量窄load完成的64列读取，现在用更少的宽load完成。实际线程与寄存器分片由make_tmem_copy推导，不能只改指令名字却沿用不兼容的手算分片。

这属于指令粒度调优，不改变矩阵数学结果、输入复用或消费者数量。需要重新检查所有输出、寄存器数量、spill和计时；“更宽”本身不保证更快。

结果只支持所测shape与环境。若慢于cuBLAS就记录慢；只比cuBLASLt某候选快，不能宣布超过cuBLAS，因为普通cuBLAS可能选择更快的内核。'''
]


# Specific explanations for v01: avoid introducing later-version machinery.
V01_NOTES = {
 17: '定义本 CTA 的 shared-memory 存储：只有 A、B 输入数组、MMA barrier 和 TMEM 基地址字段。本版没有输出 SMEM buffer。',
 27: '这是生成器留下的常量表达式：1>=3 为假，因此 bm=bn=0；本版只处理左上角唯一的输出 tile。',
 29: '从 GMEM A 选择 M=128、K=64 的窗口；Step 保留 M/K，忽略 N。ga 的 shape 为 (128,64,1)，最后一维是 K64 大块数。',
 30: '从 GMEM B 选择 N=128、K=64 的窗口；Step 保留 N/K，忽略 M。B 按 (N,K) 存储，因此数学输出是 A×Bᵀ。',
 31: '从 GMEM D 选择 (128,128) 输出窗口；Step 保留 M/N，忽略 K。本版只有一个 CTA 拥有整块输出。',
 33: '仅重组 GMEM 访问视图，pa/pb 为 ((128,16),1,4,1)：指令内元素、M/N重复、K16块数、K64块数；不搬数据。',
 34: '将 D 组织为 ((128,128),1,1) 的 GMEM 视图，供 TMEM 布局推导和最终写回使用；此时未给 CUDA 线程分输出。',
 40: 'threadIdx.x 为 0..31 的线程属于 warp0。整个 warp 参与 TMEM 分配与 MMA 封装调用，MMA 内部再选一个线程实际发射。',
 46: '将申请到的 TMEM 基地址绑定给 acc；这是地址绑定，不是把 128×128 个累加数值赋为 s.tmem。',
 49: '本版 nk=1，所以外层 kt 只有 0；内部 kb 才遍历四个 K16 切片。',
 57: 'ra 的第 2 个顶层维度为 4；kb=0..3 依次覆盖 K=[0,16)、[16,32)、[32,48)、[48,64)。每次都更新整块 128×128 的 acc。',
 63: '本版 kt=0，所有 128 个线程等待 MMA barrier 的 phase 0 完成；之后才能从 TMEM 读取累计结果。',
 65: '构造 TMEM load 配置。基础操作是每 warp 从 32 条 datapath 各取一个 FP32；CuTe 重复该操作覆盖整个 acc。',
 68: '得到本线程的 GMEM 输出份额。本版真实映射是 thread t 写 D[t,0:128]；这项行归属只适用于当前输出 copy。',
 74: '当前线程有 128 个输出元素，因此 i=0..127，逐个执行 FP32→FP16 转换。',
 76: '将该线程的 128 个 FP16 输出写到 GMEM 的 dst；本版不经过输出 shared-memory buffer。',
 77: '等所有线程完成结果读取和写回流程，warp0 才能释放 TMEM，避免其他 warp 仍在读取时提前回收。',
 97: '常量条件均为假，因此启动 grid=(1,1)、128个线程，并为整个 CTA 分配 sizeof(BasicStorage) 字节动态 SMEM。'
}
