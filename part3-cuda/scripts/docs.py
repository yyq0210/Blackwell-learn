from pathlib import Path
import json,re,html,sys
from doc_content import TITLES,INTROS
R=Path(__file__).resolve().parents[1]
NAMES=['v01_single_tile','v02_k_loop','v03_multi_cta','v04_tma','v05_double_buffer','v06_persistent','v07_warp_specialized','v08_two_cta','v09_multi_consumer','v09_tuned']
# Each rule describes the concrete operation, not a translation of the variable name alone.
RULES=[
(r'auto \[ag, as\] =','C++结构化绑定接收TMA分片结果：ag是源global坐标视图，as是目标shared-memory视图。下一行调用tma_partition。','host'),
(r'auto \[bg, bs\] =','C++结构化绑定接收B的TMA分片：bg是源global坐标，bs是对应shared目标。','host'),
(r'int tile_round =','初始化已处理的输出tile轮数。第6版跨tile复用barrier时，用它计算每个stage应等待的phase。','sync'),

(r'CUTE_GRID_CONSTANT','TMA descriptor是grid级只读参数。防止取地址时被复制到线程local memory；TMA必须拿到可访问的descriptor地址。','host'),
(r'copy\(cp','真正执行TMEM→寄存器的load；cp规定本线程读取哪些lane/column，rf接收FP32结果。','tmem'),
(r'fence_view_async_tmem_load','发出tcgen05.wait::ld，等待本线程已发出的TMEM读取完成，再使用寄存器并允许回收TMEM。','sync'),
(r'fence_view_async_shared','使普通shared stores对异步MMA/TMA代理可见；它不替代线程之间的执行同步。','sync'),
(r'NamedBarrier::sync','只同步当前consumer的128个writeback线程。barrier ID按consumer区分，不能在这里使用要求整个CTA到齐的__syncthreads。','sync'),
(r'ClusterBarrier::arrive','写回线程报告TMEM已用完；双CTA时送到CTA0的acc_empty[c]。每个线程到达一次，合计128×CTA组大小次。','sync'),
(r'initialize_barrier\(s\.empty','初始化输入buffer回收barrier。期望到达数等于MMA消费者数；只有所有消费者读完B，producer才可覆盖该stage。','sync'),
(r'initialize_barrier\(s\.full','初始化输入就绪barrier。一个producer arrival配合预期TMA字节数，确定本轮输入是否全部到达。','sync'),
(r'initialize_barrier\(s\.acc_empty','初始化TMEM回收barrier，收集所有对应writeback线程的通知。双CTA下期望256次arrival。','sync'),
(r'initialize_barrier\(s\.acc_full','初始化结果就绪barrier，每个consumer的最后一批MMA完成后通知一次。','sync'),
(r'initialize_barrier','初始化MMA完成barrier；计数1指一个完成通知，不表示只有一个线程可以等待。','sync'),
(r'set_barrier_transaction_bytes','向full barrier登记本轮TMA应完成的字节数并arrival。它只登记期望，不会自己搬数据。','sync'),
(r'wait_barrier\(s\.acc_empty','当前MMA消费者等待上一输出tile的TMEM被用完。只等待自己的slot，不等待另一位消费者的整个输出。','sync'),
(r'wait_barrier\(s\.acc_full','writeback WG等待对应consumer的MMA完成，随后才允许读取这一段TMEM。','sync'),
(r'wait_barrier\(s\.empty','producer等待这个stage上一轮的MMA消费全部结束，之后才能覆盖它。','sync'),
(r'wait_barrier\(s\.full','MMA侧等待当前stage本轮TMA字节全部到达；phase用于区分反复使用同一barrier的不同轮次。','sync'),
(r'wait_barrier\(s\.(done|mma_bar)','等本轮异步MMA完成。没有这一步就重写输入SMEM，会在Tensor Core仍读取时覆盖数据。','sync'),
(r'umma_arrive_multicast','发出tcgen05 commit，待本warp此前MMA完成后通知mask选中的CTAs；mask3=二进制11，即两侧都通知。','sync'),
(r'umma_arrive\(&s\.mma_bar','将本warp此前MMA的完成通知绑定到mma_bar，供整个CTA等待。调用内部选出一个发射线程。','sync'),
(r'mma_arrive\(&s\.empty','当前MMA consumer提交输入消费完成通知。共享B的stage需要所有consumer的完成通知才能回收。','sync'),
(r'mma_arrive\(&s\.acc_full','本消费者整个K-loop结束，提交自己那段TMEM的结果就绪通知。','sync'),
(r'umma_arrive\(&s\.done','为这轮K tile的MMA提交完成通知，下一次复用输入SMEM前等待它。','sync'),
(r'umma_arrive\(bar','单CTA的MMA完成通知辅助函数；和双CTA分支提供相同的上层含义。','sync'),
(r'cooperative_copy','128个线程合作执行普通global load与shared store；CuTe按布局推导分配和可用的向量宽度。这不是TMA。','gmem'),
(r'copy\((ca|cb|tma_a|tma_b)\.with','发起TMA异步加载：从global坐标对应tile搬到当前stage的SMEM，并将完成字节记到with指定的barrier。','gmem'),
(r'copy\((cd|tma_d),','发起SMEM→GMEM的TMA store。这个调用返回不代表数据已经写完，后面还要commit/wait。','gmem'),
(r'tma_store_fence','把writeback线程的shared store公布给TMA异步代理；所有写入线程都要执行，然后做WG/CTA同步。','sync'),
(r'tma_store_arrive','当前发射线程提交此前TMA store组成的bulk async group。','sync'),
(r'tma_store_wait','当前发射线程等待已提交的TMA store组完成，保证输出SMEM可以安全复用。','sync'),
(r'copy\(rh, dst\)','真正执行FP16寄存器→目标存储。前三版dst是GMEM分片，TMA版本dst是swizzled输出SMEM分片。','rmem'),
(r'\b[rfh]+\(\w+\) = H\(','逐元素从FP32转换为FP16；这一步发生最终舍入，Tensor Core的内部累加精度仍是FP32。','rmem'),
(r'gemm\(mma','发一条K16的tcgen05 MMA，A/B参数是SMEM descriptor，结果累加到TMEM。调用异步；CuTe内部处理单线程发射。','mma'),
(r'ScaleOut::Zero','让后续第一条MMA忽略原TMEM值，相当于以0开始本输出tile的累加；无需另发清零kernel。','mma'),
(r'ScaleOut::One','从下一条MMA开始保留已有TMEM结果并继续累加，不能在每个K tile开头重新清零。','mma'),
(r'alloc\.allocate','由一个完整warp协作申请TMEM列数，并把基地址写到SMEM里的tmem字段；全CTA/cluster同步后其他线程才能使用。','tmem'),
(r'alloc\.release_allocation_lock','归还TMEM分配许可，让后续工作能够申请；这不等于已经释放当前TMEM内容。','tmem'),
(r'alloc\.free','释放先前申请的TMEM列。调用前必须确保所有相关线程和远端CTA已经不再访问它。','tmem'),
(r'acc\.data\(\)','为静态TMEM layout绑定真正的硬件基地址。第9版再加consumer×256列，使两个累加器互不覆盖。','tmem'),
(r'make_fragment_C','创建TMEM accumulator的布局视图；此时还没有申请硬件TMEM，或者要在下一行绑定已申请地址。','tmem'),
(r'make_fragment_[AB]','把SMEM tensor转换成MMA操作数descriptor fragment；这里不执行SMEM→寄存器的数据复制。','mma'),
(r'make_tmem_copy','用选定tcgen05 load atom和accumulator布局构造线程级复制方案；随后get_slice才取出当前线程的份额。','tmem'),
(r'SM100_TMEM_LOAD','指定TMEM load指令形状。32dp32b指32条datapath上的32-bit数据；1x或32x决定单条指令覆盖多少列。','tmem'),
(r'\.partition_S','按copy的线程映射取源分片；TMEM copy中它指当前线程要读的TMEM位置，不是另分配一块TMEM。','tmem'),
(r'\.partition_D','按同一个copy映射取目标分片，使每个源结果落到对应输出坐标；目标可能是SMEM或GMEM。','rmem'),
(r'tma_partition','把MMA分片/epilogue分片转成TMA的源、目标视图。ag/bg/dg是global坐标，as/bs/ds是shared目标或源；尚未搬数据。','host'),
(r'zipped_divide','把完整输出拆成epilogue子tile和子tile编号。128×64只限制每轮处理量，不改变完整输出的形状。','tmem'),
(r'make_tensor<float>','按当前线程目标分片的形状创建FP32寄存器fragment，暂存从TMEM读出的结果。','rmem'),
(r'make_fragment_like\(dst\)|make_tensor<H>','创建对应形状的FP16寄存器fragment，承接转换后的结果。寄存器是线程私有的。','rmem'),
(r'make_smem_ptr','把shared buffer地址和swizzled layout组合成CuTe Tensor视图；布局决定逻辑坐标如何落到物理SMEM。','smem'),
(r'make_gmem_ptr','把cudaMalloc得到的全局地址和形状/stride组合成CuTe Tensor；这个host侧对象描述内存，不复制矩阵。','host'),
(r'make_stride','描述row-major布局：外层行移动K或N个元素，内层连续维度stride=1。stride单位是元素，不是字节。','host'),
(r'make_tma_atom_A_sm100','在CPU上创建A的TMA descriptor，结合MMA和cluster layout进行CTA级分配；双CTA下每侧只搬自己那份。','host'),
(r'make_tma_atom_B_sm100','在CPU上创建B的TMA descriptor；双CTA下两侧各持一半N方向输入，两者一起服务完整MMA输出。','host'),
(r'make_tma_atom\(SM90_TMA_STORE','在CPU上为输出的128×64子tile编码TMA store descriptor，绑定D的形状和输出SMEM swizzle。','host'),
(r'make_tma_atom\(SM90_TMA_LOAD','在CPU上为输入tile编码TMA load descriptor，描述global形状/stride与目标shared layout。','host'),
(r'get_tma_tensor','从descriptor取得坐标型Tensor；里面是多轴global坐标表达式，不是普通可直接解引用的GPU指针。','host'),
(r'cta\.partition_[AB]','按MMA的CTA分片规则重排输入tile的逻辑坐标。1-CTA时全部属于本CTA；2-CTA时只取本侧输入。','mma'),
(r'cta\.partition_C','按MMA映射取得本CTA输出的逻辑布局。双CTA的每侧只拥有128行，但都覆盖MMA的全部N列。','mma'),
(r'local_tile\(a','选取当前输出任务的A窗口：M方向由bm和consumer决定，保留K tile轴用于循环。','gmem'),
(r'local_tile\(b','选取当前输出任务的B窗口：N方向由bn决定，保留K tile轴。多个consumer共用bn，所以可以复用B。','gmem'),
(r'local_tile\(d','选取本任务的输出窗口；本CTA或本consumer只写它拥有的M/N坐标，避免输出竞争。','gmem'),
(r'mma\.get_slice','取得当前CTA在MMA中的份额。Blackwell的这个slice参数是peer CTA编号，不是CUDA threadIdx。','mma'),
(r'cp\.get_slice|th = cp','取得当前CUDA线程在TMEM copy中的份额。这里的slice参数才是线程号。','rmem'),
(r'grouped_tile','把持久化逻辑tile编号映射为8行M一组的二维坐标，以限制A/B在L2中的复用距离。','schedule'),
(r'UMMA::tile_to_mma_shape','以128-byte swizzle atom为基础，铺成MMA要求的shared-memory布局；使用partition_shape得到的形状而非随意手写descriptor。','smem'),
(r'tile_to_shape','将小的swizzle atom铺成128×64输出SMEM tile；group<0,2>将两个逻辑轴打包，方便epilogue/TMA分片。','smem'),
(r'partition_shape_[AB]','在编译期推导MMA分片后的操作数shape。例如单CTA A的128×64变为((128,16),1,4)。','mma'),
(r'SM100_MMA_F16BF16_2x1SM_SS','选择双CTA的FP16/BF16 SMEM×SMEM MMA原语，FP32累加，M/N=256/256；K-major表示K连续。','mma'),
(r'SM100_MMA_F16BF16_SS','选择单CTA的FP16/BF16 SMEM×SMEM MMA原语，FP32累加；后续M/N和Major参数给出指令形状。','mma'),
(r'make_tiled_mma','把底层MMA atom包装为CuTe TiledMMA，让后续分片、descriptor和gemm调用使用一致的映射。','mma'),
(r'ArrayEngine','预留真正的shared-memory数组。cosize_v按layout的地址范围计算容量，alignas保证TMA/descriptor要求的对齐。','smem'),
(r'uint64_t.*full|uint64_t.*mma_bar','预留64-bit mbarrier对象；数组下标表示stage或consumer，不是GPU地址轴。','sync'),
(r'uint32_t tmem','在SMEM中保存TMEM基地址。它只是一个32-bit地址值，不是整个TMEM存储数组。','tmem'),
(r'extern __shared__','声明动态shared memory，由host launch第三个参数决定字节数。这里不是每个线程各分配一份。','smem'),
(r'reinterpret_cast','把动态shared字节区解释为已定义Storage结构；各字段的偏移和对齐由C++编译器确定。','smem'),
(r'__syncthreads','整个CTA的执行同步。所有需要参与的线程都必须到达；它本身不会等待尚未提交到barrier的异步MMA/TMA。','sync'),
(r'cluster_sync','两个CTA都到达cluster级同步点，保证初始化可见或退出前远端访问完成。不要把CTA内同步当作cluster同步。','sync'),
(r'cudaLaunchKernelEx','按显式cluster维度启动两个CTA合作的kernel，并放到统一的per-thread stream中以便Graph捕获。','host'),
(r'cudaLaunchAttributeClusterDimension','为launch设置cluster维度；只有这样两个CTA才获得同cluster的协作与调度保证。','host'),
(r'attr\.val\.clusterDim','cluster形状设为(2,1,1)；这是CTA的组织方式，不是CUDA block里线程的形状。','host'),
(r'config\.','填写cluster launch参数：grid/block维度、动态SMEM字节数、stream和属性数组。','host'),
(r'cudaLaunchConfig_t|cudaLaunchAttribute attr','初始化CUDA扩展启动配置结构，后续填写cluster属性。','host'),
(r'cudaFuncSetAttribute|cudaFuncAttributeMaxDynamicSharedMemorySize','允许该kernel使用指定大小的动态SMEM；配置发生在host侧且只做一次，避免污染每次发射的host成本。','host'),
(r'fn<<<','真正发射CUDA kernel：依次指定grid、每CTA线程数、动态SMEM大小。矩阵分片的Tensor与descriptor作为参数传入。','host'),
(r'auto fn =|auto fn=','选择当前Tensor/descriptor类型对应的模板实例，取得真正的kernel入口地址。','host'),
(r'static bool configured','host端记住kernel属性是否已设置；不会在每个Graph节点里反复配置。','host'),
(r'configured = true','标记属性已配置，后续launch复用。','host'),
(r'if \(!configured','仅首次host调用进入配置分支；此处不是GPU线程分支。','host'),
(r'cluster_layout =|tiled_divide\(make_layout','构造(V,M,N,K)形式的cluster布局，V表示一次MMA的peer CTA轴。双CTA时V大小为2。','schedule'),
(r'cluster_count =|cluster_count=','物理grid的CTA数量除以CTA组大小，得到本次并行驻留的逻辑cluster数。','schedule'),
(r'int clusters =|int grid =','限制持久化并行任务数不超过可用SM/cluster数，也不超过总tile数。更多工作由同一个CTA/cluster在循环里继续处理。','schedule'),
(r'int mt =','计算逻辑cluster tile网格的M/N大小；M要同时除以单consumer的BM和consumer数量。','schedule'),
(r'int tiles =','总输出tile数=(M/128)*(N/128)，用于决定普通grid或持久化grid的大小。','schedule'),
(r'int first =','初始任务编号取blockIdx.x，每个CTA从不同起点开始。','schedule'),
(r'int step =','持久化模式中每轮任务号加gridDim.x；普通多CTA模式只有一个任务。','schedule'),
(r'int warp =','计算warp号、peer CTA号和cluster号；这是线程职责与跨SM合作的基础坐标。','schedule'),
(r'bool leader','peer==0识别leader；elect_one_sync从当前warp选出一个线程，用于只能单线程发射的TMA操作。','schedule'),
(r'bool warp0','识别warp0。分配TMEM要求完整warp参与；TMA则还要进一步选一个线程。','schedule'),
(r'int c = threadIdx','根据writeback线程所在warpgroup选择consumer，并用局部tid=threadIdx%128套用TMEM copy映射。','schedule'),
(r'int c = warp','MMA issue warp号决定consumer编号，各warp有独立的K-loop和accumulator范围。','schedule'),
(r'if \(warp == 4','进入TMA producer warp。这里的常量表达式由本版consumer数量决定，和讲义中的WG角色表对应。','schedule'),
(r'if \(warp >=','只让leader CTA内的指定MMA issue warp进入；第9版两条warp指令流各推进自己的consumer。','schedule'),
(r'if \(threadIdx.x <','只有writeback warpgroup进入epilogue；producer/MMA warps不参与其128线程named barrier。','schedule'),
(r'if \(threadIdx.x == 0','仅一个线程初始化shared barrier，避免重复初始化破坏其状态。','sync'),
(r'if \(warp0 && elected\)|if \(elected\)','将TMA操作限制到一个线程；整个warp重复发射会重复搬运并破坏预期事务计数。','schedule'),
(r'if \(leader\)','仅leader登记双CTA共同的TMA完成字节或推进MMA，避免两侧重复执行同一合作计算。','schedule'),
(r'if \(tid == 0','仅本writeback WG的线程0发起并等待TMA store，其他线程通过named barrier与它会合。','schedule'),
(r'if \(warp == 0\)|if \(warp0\)','本warp承担TMEM分配/释放或串行版本的MMA发射；其他warps继续等待或负责结果搬运。','schedule'),
(r'if \(iteration >=','环形buffer第一次使用天然为空；从第二圈开始，必须等上一圈消费完成。','sync'),
(r'if \(round\)','第一次输出无需等待旧TMEM；从第二个输出tile开始等待上一轮writeback完成。','sync'),
(r'if \(kt \+ 1 <','只有确实存在下一K tile时才预取，避免末尾读取越界；还结合warp/elected限制发射者。','gmem'),
(r'for \(int ti =','持久化任务循环：当前CTA/cluster处理编号间隔为grid/cluster_count的一系列输出tiles。各角色必须使用相同编号顺序。','schedule'),
(r'for \(int kt =','沿K以64为单位分块。每一轮负责一个完整A/B输入stage，stage可跨输出tile循环复用。','mma'),
(r'for \(int kb =','一个BK64包含4个K16 MMA微块；此循环按descriptor的K偏移发出4条MMA。','mma'),
(r'for \(int ei =','遍历128×64的epilogue子块。BN128需2轮，BN256需4轮；同一输出SMEM buffer反复使用。','rmem'),
(r'for \(int c =','逐个consumer加载不同A或初始化独立barrier。最终版的MMA主循环本身由独立warp执行，不在一个warp里轮流计算。','gmem'),
(r'for \(int [ij] =','遍历固定数量的barrier或当前线程的寄存器元素；具体容量由本版stage数/fragment shape决定。','rmem'),
(r'CUTE_UNROLL','要求编译器展开静态短循环，减少循环控制；不表示这些MMA或load变成同时发射。','host'),
(r'int st =|int stage =','用绝对或当前K迭代号对stage数取模，选中要读写的那份shared buffer。','smem'),
(r'int nk =','计算K tile轮数：第1版固定1次，其余版本K/64次。K必须能被64整除。','mma'),
(r'int iteration =|int done_phase =','初始化跨迭代状态。iteration选择stage/phase；done_phase或round追踪反复使用的完成barrier。','sync'),
(r'int bm =|int bm, bn','准备输出tile的二维索引；持久化版本随后用grouped_tile按8行M分组计算。','schedule'),
(r'make_coord\(bm','把输出tile坐标和保留的K轴组合起来；下划线表示这一轴暂不固定。','schedule'),
(r'done_phase \^=','同一完成barrier每完成一轮翻转phase，避免把旧完成事件认作新事件。','sync'),
(r'\+\+tile_round','进入下一个输出tile，更新各stage的历史使用次数，以便计算正确的full phase。','sync'),
(r'static constexpr int BM','本版编译期配置：BM/BN是每个consumer合作MMA覆盖的形状，Threads含writeback和管理WG，Cols是TMEM列总数。','host'),
(r'Cols =','TMEM列数=每个consumer的BN×consumer数量；第9版256×2=512列。','tmem'),
(r'std::conditional_t|using Op =','在编译期选择单CTA或双CTA的类型。只会为本版选中的类型生成执行路径。','host'),
(r'using CopyOp','单CTA用普通TMA load，双CTA用2SM TMA load；它们的完成事务和CTA分片规则不同。','host'),
(r'using Alloc','选择单CTA或双CTA协作TMEM分配器，必须和MMA的CTA group一致。','tmem'),
(r'using Tile =|using T =','定义编译期MMA tile形状(M,N,K)，此处K=64由4条K16 MMA组成。','host'),
(r'using (CF|MM|AL|BL) =','为本版已经确定的配置/类型建立局部别名，缩短后续CuTe表达式；不会发出GPU指令。','host'),
(r'using Mma =|using MM =','取得TiledMMA类型，后续默认构造mma对象即可使用相同指令形状与布局。','host'),
(r'using (LA|LD|AShape|BShape|ShapeA)','定义静态layout或shape类型：输入layout服务MMA，LD服务128×64的输出TMA分块。','host'),
(r'\b(Mma|MM) mma;','构造本线程的轻量MMA描述对象，包含累加模式等状态；不是在每个线程里分配一个Tensor Core。','mma'),
(r'(TMEM::Allocator1Sm|Alloc) alloc;','创建TMEM分配器接口对象；真正申请发生在allocate调用。','tmem'),
(r'struct (BasicStorage|TmaStorage|Storage)','定义每CTA的shared-memory存储结构；内含A/B、输出buffer和barrier元数据。','smem'),
(r'struct WSConfig','集中定义本版的静态MMA形状、线程数、TMEM容量与SMEM结构。','host'),
(r'template <','声明C++模板参数；CuTe的Tensor/layout类型在编译时决定，运行时不做Python解释。','host'),
(r'__global__ void','定义真正运行在GPU上的CUDA kernel。每个CTA获得独立SMEM，各线程按照后面的warp条件分工。','host'),
(r'CUTE_DEVICE void mma_arrive','设备侧的MMA完成通知包装，屏蔽单/双CTA commit形式差异。','sync'),
(r'void launch','host侧入口，准备描述对象、配置kernel并发射；GPU运算在__global__函数内。','host'),
(r'constexpr int VERSION','测试程序用这个编译期版本号检查允许的输入形状，并标记输出JSON。','host'),
(r'#include "runner_graph','引入共享C++测试入口：随机输入、CPU/完整cuBLAS参考、cuBLASLt选择与Graph计时。对应独立测试文档有逐行说明。','host'),
(r'#include','引入CUDA/CuTe、错误检查或C++标准库声明；这不是运行时加载Python包。','host'),
(r'namespace study|using namespace cute','建立本项目命名空间，或简写CuTe名字；不改变数据布局或GPU调度。','host'),
]
UNKNOWN=set()
def notes(lines):
 out=[];last=('','','host')
 for i,l in enumerate(lines,1):
  z=l.strip()
  if not z:continue
  if z.startswith('//'):
   n='说明性注释，不生成机器指令。对应的中文机制解释见本节开头和下面的实际语句。';tag='host'
  elif re.fullmatch(r'[{};]+(?:\s*//.*)?',z) or z.startswith('} //'):
   n='结束当前代码块或类型声明；作用域对应上方最近的函数、循环或条件分支。';tag=last[2]
  elif z.startswith('if constexpr'):
   n='编译期分支：常量条件决定保留哪条路径，未选中的单/双CTA或单/双buffer路径不会在运行时执行。';tag='host'
  elif z.startswith('else') or z.startswith('} else'):
   n='进入与上方条件互补的分支；编译期常量条件会让其中一条路径直接消失。';tag=last[2]
  else:
   found=next(((n,t) for p,n,t in RULES if re.search(p,z)),None)
   if found:n,tag=found
   elif z.startswith(('UMMA::','decltype(','Shape<','Int<','get<','make_layout(','make_tile(','SM90_TMA_LOAD','SM100_TMA_2SM_LOAD','SM100_MMA_','TMEM::Allocator','sizeof(','cf::','CF::','cutlass::','group_modes<','set_barrier','cudaFunc','alignas(')) or last[0].rstrip().endswith((',', '(', '=', '<', '&&')) or z.startswith(('make_stride','stage]', '32768','make_coord', 'copy(', 'fn,')):
    n='本行继续上方表达式的参数/类型：'+last[1];tag=last[2]
   else:
    UNKNOWN.add(z);n='本行是相邻C++语句的组成部分；结合上一行的表达式理解：'+last[1];tag=last[2]
  out.append(dict(line=i,code=l,note=n,tag=tag));last=(z,n,tag)
 return out

def diagram(v,title):
 colors={'gmem':'#2563eb','smem':'#0e9488','mma':'#d97706','tmem':'#7c3aed','rmem':'#db2777'}
 labels=[('gmem','GMEM: A / B'),('smem','SMEM: swizzled A / B'),('mma','tcgen05 MMA'),('tmem','TMEM: FP32 accumulator'),('rmem','REG: FP32 → FP16'),('gmem','GMEM: D')]
 boxes=''
 for j,(tag,label) in enumerate(labels):
  x=24+j*155;y=80
  boxes+=f'<g id="node-{j}" data-tag="{tag}"><rect x="{x}" y="{y}" width="140" height="76" rx="10" fill="{colors[tag]}"/><text x="{x+70}" y="{y+30}" fill="white" text-anchor="middle" font-size="12">{html.escape(label.split(":")[0])}</text><text x="{x+70}" y="{y+52}" fill="white" text-anchor="middle" font-size="10">{html.escape(label.split(":")[-1])}</text></g>'
  if j<5:boxes+=f'<path d="M{x+142} 118h12" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>'
 loads='128 threads: ordinary load/store' if v<=3 else 'one TMA issue thread per CTA'
 detail='1 CTA / 1 SM' if v<=7 else 'CTA0 / SM0 + CTA1 / SM1: cooperative MMA'
 if v==9:detail+='; two independent MMA issue warps'
 return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 230" role="img" aria-label="{html.escape(title)} 数据路径"><defs><marker id="arr" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0L5 2.5L0 5" fill="none" stroke="#64748b"/></marker></defs><rect width="960" height="230" fill="#f8fafc" rx="12"/><text x="24" y="30" font-size="18" font-family="sans-serif">{html.escape(title)}</text><text x="24" y="55" font-size="13">{loads}</text>{boxes}<text x="24" y="191" font-size="13">{detail}</text><text x="24" y="214" font-size="12">Async instruction issue ≠ completion. Follow the barrier / wait before reuse.</text></svg>'''

PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
body{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.7 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:28px}a{color:#0369a1}section,.intro{background:white;border-radius:12px;padding:22px;margin:20px 0;border:1px solid #dbe3eb}svg{width:100%;max-height:260px}label{display:inline-block;margin:8px 24px 8px 0}input{vertical-align:middle}button,select{padding:7px 14px;border:1px solid #94a3b8;border-radius:5px;background:white;color:#0f172a}#buffers,#roles{display:flex;gap:12px;flex-wrap:wrap}.buffer,.role{padding:12px 18px;border:2px solid #cbd5e1;border-radius:8px}.current{border-color:#7c3aed;background:#f5f3ff}#detail{padding:14px;background:#e0f2fe;border-radius:8px;min-height:48px}.source{max-height:680px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}td{padding:5px 8px;vertical-align:top;border-bottom:1px solid #e2e8f0}td:nth-child(1){width:45px;color:#64748b}td:nth-child(2){width:55%;white-space:pre-wrap;font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}tr{cursor:pointer}tr:hover,tr.selected{background:#e0f2fe}#timeline{display:grid;grid-template-columns:100px repeat(18,1fr);gap:4px;font-size:12px}.task{min-width:28px;border-radius:4px;background:#dbeafe;padding:8px;text-align:center}.muted{color:#64748b}.code-tools{position:sticky;top:0;background:white;padding:8px}input[type=search]{width:280px;padding:8px;border:1px solid #cbd5e1}pre{white-space:pre-wrap}g.active rect{stroke:#facc15;stroke-width:5}#tilemap{width:100%;max-width:640px}#tmemmap{display:flex;gap:12px;margin:12px 0;flex-wrap:wrap}.sm{padding:12px;border:1px solid #94a3b8;border-radius:8px}.range{display:inline-block;padding:8px;background:#ede9fe;margin:3px}.chosen{background:#fef08a}#timeline{overflow:auto}#results{font-variant-numeric:tabular-nums}.var{font-family:monospace}@media(max-width:800px){main{padding:10px}td:nth-child(2){width:50%}#timeline{font-size:9px}section{padding:12px}}
</style><main><a href="index.html">← 版本目录</a><h1>__TITLE__</h1><p><a href="__SOURCE__">完整 CUDA C++ 源码</a> · <a href="__MD__">Markdown 讲义</a> · <a href="testing.html">测试代码逐行讲解</a></p><div class="intro">__INTRO__</div><section><h2>数据路径与角色</h2>__SVG__<div id="roles"></div><p class="muted">点下面任意源码行，图中高亮它操作的存储/计算环节。同步行请同时观察解释中的依赖关系。</p></section><section><h2>切换任务，观察 stage / phase / consumer</h2><label>K tile <input id="kt" type="range" min="0" max="7" value="0"><b id="ktLabel"></b></label><button id="next">推进一轮</button><label>消费者 <select id="consumer"><option value="0">0</option><option value="1">1</option></select></label><label>逻辑 tile <input id="tile" type="range" min="0" max="63" value="0"></label><div id="buffers"></div><pre id="mapping"></pre><canvas id="tilemap" width="640" height="260" aria-label="输出tile网格，蓝色共用B，绿色共用A，黄色为当前tile"></canvas><div id="tmemmap"></div><div id="timeline"></div><p class="muted">两个滑块分别演示输入环和输出任务映射；K图展示一轮输出内的前8个K tiles（第1版仅1个）。这是依赖关系示意，不代表实测时间比例。第5版开始才有输入双缓冲；第7版开始不同角色有独立warp指令流；第9版才有两个独立MMA issue warp。</p></section><section><h2>本版本实测</h2><div id="results"></div></section><section><h2>逐行讲解</h2><div class="code-tools"><input id="query" type="search" placeholder="搜索源码或解释，如 barrier / stage"><span id="count"></span><div id="detail">点击源码行查看解释并定位图示。</div></div><div class="source"><table><tbody id="code"></tbody></table></div></section><section><h2>自查</h2><p>1. 本版谁发起加载？谁等待数据可读？谁通知输入可覆盖？</p><p>2. 写回线程等待的是full[stage]还是acc_full[consumer]？为什么？</p><p>3. 下一轮能否覆盖SMEM和TMEM？分别由哪个完成事件保证？</p><p>答案：前三版加载由128线程完成，MMA完成barrier保护SMEM；TMA版本的full跟踪输入到达，MMA完成/empty跟踪输入消费。WS版writeback等待acc_full，因为它读的是最终TMEM结果；acc_empty保护下一输出的TMEM复用。第9版empty必须收到两位MMA消费者的完成通知。</p></section></main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),v=data.version,S=v<=4?1:v<=6?2:v===8?6:4,G=v>=8?2:1,C=v===9?2:1;
const tags={gmem:'GMEM',smem:'SMEM',mma:'MMA',tmem:'TMEM',rmem:'寄存器',sync:'同步',host:'host / 编译期',schedule:'任务调度'};
function renderCode(){let q=$('query').value.toLowerCase();$('code').replaceChildren();let count=0;for(const x of data.lines){if(!(x.code+' '+x.note).toLowerCase().includes(q))continue;count++;let tr=document.createElement('tr');tr.id='L'+x.line;for(const val of [x.line,x.code,x.note]){let td=document.createElement('td');td.textContent=val;tr.append(td)}tr.onclick=()=>{document.querySelectorAll('tr.selected').forEach(e=>e.classList.remove('selected'));tr.classList.add('selected');$('detail').textContent='L'+x.line+' · '+tags[x.tag]+'：'+x.note;document.querySelectorAll('g[data-tag]').forEach(g=>g.classList.toggle('active',g.dataset.tag===x.tag));history.replaceState(null,'','#L'+x.line)};$('code').append(tr)}$('count').textContent=' '+count+'行'}
function draw(){const k=+$('kt').value,c=+$('consumer').value,ti=+$('tile').value;$('ktLabel').textContent=k;$('buffers').replaceChildren();for(let i=0;i<S;i++){let b=document.createElement('div');b.className='buffer'+(i===k%S?' current':'');b.textContent='stage '+i+(i===k%S?' ← 本轮':'');$('buffers').append(b)}let mt=v>=8?4096/(128*G*C):32,nt=v>=8?16:32;let group=Math.floor(ti/(8*nt)),rows=Math.min(8,mt-group*8),local=ti%(8*nt);let bm=v>=6?group*8+local%rows:ti%mt,bn=v>=6?Math.floor(local/rows):Math.floor(ti/mt);if(v<=2){bm=0;bn=0}let row0=(bm*C+c)*128*G;
$('mapping').textContent='输入 K 区间 ['+(k*64)+', '+((k+1)*64)+')\n'+(v>=5?'stage = '+k+' % '+S+' = '+k%S+'；full phase = floor(k/'+S+') % 2 = '+Math.floor(k/S)%2+'\n':'本版只有一份输入buffer，每次必须等读完再覆盖。\n')+(v>=7?(k<S?'首次使用stage，无需等旧empty。':'复用前等empty phase = '+(Math.floor(k/S)-1)%2)+'\n':'')+'逻辑 tile ('+bm+', '+bn+')，consumer '+c+'\n输出 rows ['+row0+', '+(row0+128*G)+')，columns ['+(bn*128*G)+', '+((bn+1)*128*G)+')'+(v>=8?'\nCTA0取前128行，CTA1取后128行。':'')+'\n本CTA TMEM columns ['+(c*128*G)+', '+((c+1)*128*G)+')';
$('roles').replaceChildren();let roles=v<7?['128 threads: 协作加载/写回','warp0: MMA']:['WG0: consumer0 写回',...(C===2?['WG1: consumer1 写回']:[]),'WG'+C+' warp3: TMA',...Array.from({length:C},(_,i)=>'WG'+C+' warp'+i+': MMA consumer'+i)];roles.forEach(t=>{let e=document.createElement('div');e.className='role';e.textContent=t;$('roles').append(e)});
const cv=$('tilemap'),ctx=cv.getContext('2d');ctx.clearRect(0,0,640,260);let nr=v<=2?1:mt,nc=v<=2?1:nt,w=570/nc,h=210/nr;ctx.fillStyle='#334155';ctx.font='13px sans-serif';ctx.fillText('输出tile网格：蓝色共用B，绿色共用A，黄色为当前tile；M向下，N向右',10,18);for(let r=0;r<nr;r++)for(let col=0;col<nc;col++){ctx.fillStyle=r===bm&&col===bn?'#facc15':col===bn?'#93c5fd':r===bm?'#5eead4':'#e2e8f0';ctx.fillRect(35+col*w,35+r*h,w-2,h-2)}
$('tmemmap').replaceChildren();for(let peer=0;peer<G;peer++){let box=document.createElement('div');box.className='sm';box.textContent='CTA'+peer+' / SM'+peer+' · TMEM 128 lanes';for(let cc=0;cc<C;cc++){let rg=document.createElement('span');rg.className='range'+(cc===c?' chosen':'');rg.textContent='consumer'+cc+' cols ['+cc*128*G+', '+(cc+1)*128*G+')';box.append(rg)}$('tmemmap').append(box)}
$('timeline').replaceChildren();for(const role of ['逻辑时序','加载 / full','MMA0 / empty',...(v===9?['MMA1 / empty']:[])]){let lab=document.createElement('div');lab.textContent=role;$('timeline').append(lab);for(let t=0;t<18;t++){let e=document.createElement('div');e.className='task';let label=role==='逻辑时序'?String(t):'';for(let i=0;i<(v===1?1:8);i++){let load=v<=4?2*i:i,compute=v<=4?2*i+1:i+1;if(role==='加载 / full'&&t===load)label='K'+i+'→S'+i%S;if(role==='MMA0 / empty'&&t===compute)label='K'+i;if(role==='MMA1 / empty'&&t===compute+1)label='K'+i}if(label.includes('K'+k))e.style.background='#ddd6fe';e.textContent=label;$('timeline').append(e)}}}
$('consumer').disabled=C===1;$('kt').max=v===1?0:7;$('tile').max=v<=2?0:(4096/(128*G*C))*(4096/(128*G))-1;$('kt').oninput=draw;$('tile').oninput=draw;$('consumer').onchange=draw;$('next').onclick=()=>{$('kt').value=(+$('kt').value+1)%(+$('kt').max+1);draw()};$('query').oninput=renderCode;
const r=data.result;$('results').textContent=r?'shape '+[r.m,r.n,r.k].join('×')+'；完整检查 '+r.checked+' 个元素；kernel '+(r.median_ms*1000).toFixed(2)+' μs；cuBLAS '+(r.cublas_ms*1000).toFixed(2)+' μs；cuBLASLt '+(r.cublasLt_ms*1000).toFixed(2)+' μs。条件：同轮 warm replay、CUDA Graph、FP16 I/O / FP32 accumulation。':'最终统一复测正在生成；不要把早期不同计时条件的数值混在一起。';renderCode();draw();
</script></html>'''

def main():
 (R/'docs').mkdir(exist_ok=True)
 for idx,name in enumerate(NAMES):
  v=min(idx+1,9);title=f'{idx+1:02d} · {TITLES[idx]}' if idx<9 else '09T · '+TITLES[idx]
  src=R/'kernels'/f'{name}.cu';ls=notes(src.read_text().splitlines())
  if idx == 0:
   from doc_content import V01_NOTES
   for x in ls:
    if x['line'] in V01_NOTES: x['note'] = V01_NOTES[x['line']]
  rp=R/'results/final'/f'{name}.json';result=json.loads(rp.read_text()) if rp.exists() and rp.stat().st_size else None
  svg=diagram(v,title);(R/'docs'/f'{name}.svg').write_text(svg)
  deep_md = '先读：[从一行点积读懂本版：详细图解](01-single-tile-walkthrough.md) · [交互计算图](01-single-tile-walkthrough.html#lab)\n\n' if idx == 0 else ''
  md=f'# {title}\n\n源码：[{name}.cu](../kernels/{name}.cu) · [交互逐行讲解]({name}.html)\n\n'+deep_md+INTROS[idx]+'\n\n## 数据路径\n\n!['+title+']('+name+'.svg)\n\n## 编译运行\n\n```bash\n./build.sh '+name+'\n./build/'+name+(' 128 128 64' if v==1 else ' 128 128 4096' if v==2 else ' 4096 4096 4096')+'\n```\n\n## 逐行讲解\n\n每个有效源码行都有对应解释；纯空行不编号说明。类型/参数换行继续属于同一个CuTe表达式。\n\n|行|原始代码|解释|\n|---|---|---|\n'
  for x in ls:md+='|'+str(x['line'])+'|`'+x['code'].strip().replace('|','\\|')+'`|'+x['note'].replace('|','\\|')+'|\n'
  md+='\n## 实测\n\n'+('```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```' if result else '见 `results/final/`；最终复测结果生成后重建此页。')+'\n\n公共测试程序详见 [测试与基准](testing.md)。\n'
  (R/'docs'/f'{name}.md').write_text(md)
  intro=''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in INTROS[idx].split('\n\n'))
  if idx == 0: intro='<p><a href="01-single-tile-walkthrough.html"><b>第一次读不懂？先看从一行点积开始的完整图解 →</b></a></p>'+intro
  data=json.dumps(dict(version=v,lines=ls,result=result),ensure_ascii=False).replace('</','<\\/')
  page=PAGE.replace('__TITLE__',html.escape(title)).replace('__SOURCE__','../kernels/'+name+'.cu').replace('__MD__',name+'.md').replace('__INTRO__',intro).replace('__SVG__',svg).replace('__DATA__',data)
  (R/'docs'/f'{name}.html').write_text(page)
 (R/'results/doc-unclassified-lines.txt').write_text('\n'.join(sorted(UNKNOWN)))
 print('pages',len(NAMES),'unclassified',len(UNKNOWN))
if __name__=='__main__':main()
