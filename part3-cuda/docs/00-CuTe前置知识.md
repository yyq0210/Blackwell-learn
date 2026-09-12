# 从 CUDA / TIRx 到 CuTe C++：先把这些符号认清

CuTe C++不要求换一套GPU执行模型。thread、warp、CTA、cluster、SMEM、TMEM还是原来的硬件概念。新的是：用C++模板和layout代数，描述哪个CTA/线程操作哪些数据，并选择相应硬件指令。

## 1. Tensor首先是视图

```cpp
auto layout = make_layout(make_shape(8,16), make_stride(16,_1{}));
auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
```

`t(r,c)`的元素偏移为`16*r+c`。这两行没有分配8×16个数，也没有搬运数据；ptr必须已指向有效存储。shape告诉你逻辑范围，stride告诉你每个坐标增1时地址怎么走。

`_1{}`是编译期常量1；`Int<128>{}`是编译期常量128。普通`int m`是运行时值。静态layout让编译器提前计算分片与指令组合，运行时仍可以传入完整矩阵的M、N、K。

`decltype(expression)`只取得表达式的C++类型。`using LA = decltype(...)`是在命名一个布局类型，不是分配shared memory。

## 2. local_tile：从大矩阵选窗口

```cpp
auto coord = make_coord(bm,bn,_);
auto ga = local_tile(a,Shape<_128,_128,_64>{},coord,Step<_1,X,_1>{});
```

完整GEMM的三个轴是M、N、K，A只需要M/K，所以`Step<_1,X,_1>`保留M/K、忽略N。`_`表示K tile轴暂不固定。

例如bm=2时，`ga`覆盖A的第256–383行，形状为`(128,64,K/64)`。`ga(m,k,kt)`对应`A[256+m,64*kt+k]`。B使用`Step<X,_1,_1>`，D使用`Step<_1,_1,X>`。

## 3. partition：换成指令需要的坐标

单CTA的128×128×16 MMA要求A的核心形状是128×16。BK64包含4个K16，因此：

```text
A tile形状：       (128,64)
MMA分片形状：      ((128,16),1,4)
逻辑坐标对应：      ((m,k%16),0,k/16)
```

中间的1表示M方向只需一次该MMA形状，最后的4表示沿K发4次。这个重排不改变元素个数，也不复制数据。

`mma.get_slice(peer)`在Blackwell里先选择CTA份额。它和`copy.get_slice(threadIdx.x)`不是同一层：后者才把复制工作分到线程。

## 4. 两种fragment不要混淆

`make_fragment_A/B(smem_tensor)`在本项目的SMEM输入路径上得到descriptor fragment。它为Tensor Core描述SMEM地址、leading/stride offset和swizzle，不把矩阵搬进线程寄存器，也不需要Ampere风格的ldmatrix。

`make_tensor<float>(shape(dst))`则创建当前线程的FP32寄存器fragment。静态shape足够小的时候编译器通常放寄存器；过大可能spill，不能仅凭变量名保证全部在register。编译日志中的spill计数就是为此保留的。

## 5. SMEM layout与TMEM layout是不同地址空间

`UMMA::Layout_K_SW128_Atom<H>`是K连续、128-byte swizzle的基本布局构件。`tile_to_mma_shape`把它铺成MMA分片需要的输入布局；`tile_to_shape`把它铺成写回子块。

TMEM的lane/column是独立硬件坐标。CuTe可能把lane编码成高位stride（常见65536），column编码为低位。这不是说相邻矩阵行在普通内存里相隔65536个float！它是TMEM地址编码。可对照之前学习的`@TLane`、`@TCol`理解。

128条lane、256列FP32结果占`128*256*4=128 KiB`。`tmem.allocate(256)`参数单位是TMEM columns。第9版两个消费者合计512列，每个CTA拥有自己的两段TMEM，不是让两个SM共享一块普通线性数组。

## 6. 描述、发射与完成：分成三件事

|代码|含义|是否已完成数据搬运/计算|
|---|---|---|
|`make_tma_atom`|CPU编码descriptor|没有发射|
|`tma_partition`|建立源/目标坐标视图|没有发射|
|`copy(tma.with(bar),...)`|发起异步搬运|未必完成|
|`wait_barrier(full,phase)`|确认输入到达|对应本轮输入已完成|
|`gemm(mma,...)`|发起异步MMA|未必完成|
|`umma_arrive(bar)` + wait|提交并等待MMA完成通知|对应之前提交的MMA完成|
|`copy(tmem_copy,...)` + `wait::ld`|读TMEM并等待|可以使用寄存器结果|

`.with(bar)`只是把完成通知目标附加到copy atom上，实际指令由copy调用发出。

## 7. barrier的slot、phase和计数

以4stage、2consumer为例：

|输入迭代i|stage=i%4|MMA等full phase|producer复用前等empty phase|
|---:|---:|---:|---:|
|0|0|0|首次使用，不等|
|1|1|0|首次使用，不等|
|3|3|0|首次使用，不等|
|4|0|1|0|
|8|0|0|1|

stage选择物理buffer；phase区分同一barrier的不同轮。empty的arrival count=2表示两位MMA消费者都读完；不是phase=2，也不是有2份B。

每位消费者还有自己的`acc_full[c]`与`acc_empty[c]`。输入buffer与TMEM输出buffer是两个不同生命周期，必须分开保护。

## 8. 如何看生成的独立源码

每个`.cu`已经展开本版算法，不需要跟着三层项目头文件跳转。为了与开发模板保持对应，保留少量`if constexpr(2==1)`和`std::conditional_t<...>`。这些是编译期选择，未选分支不会变成运行时工作；不是每次kernel执行时再决定用单CTA还是双CTA。

每版仍包含两个共享头文件：`common.cuh`提供参数/错误检查/分组调度，`runner_graph.cuh`提供测试入口。它们以及基准helper都有[独立逐行讲解](testing.html)。

现在可以进入[第1版](v01_single_tile.html)，先走通一次真实数据路径。
