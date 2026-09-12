# Blackwell 学习笔记

教材：[Modern GPU Programming for MLSys](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)

本地中文版：http://localhost:8000/zh/ （远程访问需转发 8000 端口）

## 学习背景与记录约定

- 已有 CUDA thread/block、矩阵乘法基础，了解 Ampere/Hopper，有算子开发实习经历。
- 按章节记录原问题、解释、示例、易混淆点和官方来源；同章追问继续补充到同一文件。
- 区分 CUDA 编程模型的保证、硬件实现细节，以及特定 GPU 的限制。
- 书中主要目标为 sm_100a；本机检测到 B300（计算能力 10.3），运行时须另行验证编译目标和工具链。
- 示例若未编译运行，会明确标记；阅读进度不等于已经验证掌握。

## 章节索引

| 章节 | 阅读进度 | 问答笔记 |
| --- | --- | --- |
| GPU 执行模型 | 已读，两轮问答及复习题答案已整理 | [01-GPU执行模型.md](01-GPU执行模型.md) |
| Kernel 性能从何而来 | 阅读中，首轮问答已整理 | [02-Kernel性能从何而来.md](02-Kernel性能从何而来.md) |
| 数据布局及其记号 | 阅读中，已补充前置知识、通俗图解和交互实验室 | [03-数据布局及其记号.md](03-数据布局及其记号.md) |
| Tensor Core 数据布局的演进 | 阅读中，ldmatrix三图及本轮问答已整理 | [04-TensorCore数据布局演进.md](04-TensorCore数据布局演进.md) |
| 异步数据搬运：TMA | 阅读中，128B swizzle与row layout问答已整理 | [05-异步数据搬运-TMA.md](05-异步数据搬运-TMA.md) |
| 第三部分 GEMM（CUDA C++ / CuTe重写） | 9版及调优版已编译运行，配套逐行讲解和交互图 | [06-GEMM-CuTe-CUDA-C++重写.md](06-GEMM-CuTe-CUDA-C++重写.md) |

后续按教材阅读进度新增章节文件并更新本索引。

配套交互：[布局坐标实验室](http://localhost:8000/study/layout-lab.html) · [独立HTML](layout-lab.html)。直接打开 HTML 也可使用，无需额外依赖。

Tensor Core 配图：[ldmatrix x1/x2/x4](http://localhost:8000/study/ldmatrix-guide.html)。

## 克隆后阅读

```bash
git clone https://github.com/yyq0210/Blackwell-learn.git
cd Blackwell-learn
python3 -m http.server 8000 --bind 127.0.0.1
```

- Markdown 笔记可直接在 GitHub 阅读。
- 交互课程入口：http://localhost:8000/part3-cuda/docs/index.html 。
- 布局实验室：http://localhost:8000/layout-lab.html 。
- 文中旧的 `/study/` 和 `/zh/` 地址对应原教材的本地阅读服务器；本仓库独立启动时，请使用上面的入口。教材源码和工具链不包含在本仓库中。
- CUDA 代码构建及实测条件见 [part3-cuda/README.md](part3-cuda/README.md)。编译产物和 Python 缓存不纳入版本控制，HTML、SVG 以及实测记录保留。
