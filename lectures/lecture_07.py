# 中文自学导读 · Spring 2026 第 7 讲：跨 GPU 并行
# 学习目标：能逐 rank 写出 collective 的输入/输出形状，按实际损失定义判断梯度
# 应求和还是平均，并沿代码说明数据、张量、流水线并行分别在何时搬运什么。
# 先修：矩阵乘法的分块、链式法则、PyTorch 的 grad 累积、GPU 异步执行；
# 记号：P 为 world_size，B 为全局 batch，d 为隐藏宽度，L 为层数，t 为墙钟耗时。
# 阅读主线：通信原语 → 硬件路径 → API 与计时 → DP/TP/PP 的张量和梯度。
# 英文、代码及图像均保留；下文“编者补充”是推导，“原稿疑点”说明实现边界。
# 来源索引（通信/带宽/RDMA/硬件官方资料已访问；字幕用于主题及课堂补充核对）：
# [课件] https://github.com/stanford-cs336/lectures ，以本地 Spring 2026 原稿为准。
# [通信] https://docs.pytorch.org/docs/2.14/distributed.html
# [带宽] https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
# [RDMA] https://docs.nvidia.com/cuda/gpudirect-rdma/
# [硬件] https://www.nvidia.com/en-us/data-center/gb200-nvl72/
# [字幕] https://jingzechen.github.io/assets/courses/stanford-cs336-s26/lectures/007/captions.source.srt
# 字幕来自第三方保存的公开录像，非教师审校逐字稿；时间段随相邻主题标注。
# 本文件的中文 # 注释供源码阅读；没有验证 edtrace 网页是否展示这些注释。
import torch
import time
import math
import sys
import os
from inspect import isfunction
from typing import Callable
from torch import nn, tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from edtrace import text, image, link
from gpu_util import cuda_if_available
from lecture_util import article_link

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda: None  # No-op if CUDA is not available

# 整讲地图：DP 切 batch，TP 切层内矩阵，PP 切层序列。
# 三者都能增加总计算资源，但只有后两者直接把模型参数拆开；普通 DP 每卡仍需
# 保存完整模型及优化器状态。能运行、能放下、能加速是三个需要分别判断的问题。
def main():
    text("# Lecture 7: parallelism")
    text("Last week: parallelism within a single GPU")
    text("This week: parallelism across multiple GPUs")
    image("images/gpu-node-overview.png", width=700)

    text("In both cases, **compute** (arithmetic logic units) is far from inputs/outputs (**data**).")
    text("Unifying theme: orchestrate computation to avoid data transfer bottlenecks")

    # 编者补充：时间可粗写成 T_compute + T_comm，通信约为 α + S/β；α 是启动延迟，
    # S 是字节量，β 是有效带宽。实际有重叠时不能直接相加，但这个模型能解释为何
    # 小消息受延迟支配、大消息受带宽支配，以及为何 TP 更适合放在节点内高速互联。
    # HBM 是单卡内存访问，NVLink/网络是设备间传输；二者不可用同一峰值替代。
    text("Generalized hierarchy:")
    text("- Single node, single GPU: L1 cache / shared memory (fastest)")
    text("- Single node, single GPU: HBM")
    text("- Single node, multi-GPU: NVLink/NVSwitch")
    text("- Multi-node, multi-GPU: Infiniband/Ethernet (slowest)")

    text("Last week: reduce memory accesses via fusion/tiling")
    text("This week: reduce communication across GPUs/nodes via replication/sharding")

    text("Why do multi-GPU?")
    text("1. Your parameters (optimizer state + gradients + activations) don't fit on a single GPU.")
    text("2. You want to use more GPUs (more FLOPs) to train faster.")

    # When you execute this lecture directly (python lecture_07.py), it uses multiprocessing, which produces output from each process (below).
    # However, when you trace this lecture (python -m edtrace.execute -m lecture_07), we turn off multiprocessing.
    link(title="stdout for this lecture", url="var/traces/lecture_07_stdout.txt")

    text("### Part 1: building blocks of distributed communication/computation")
    collective_operations()    # Programming model
    hardware()                 # Hardware: how GPUs are connected
    torch_distributed()        # How this is implemented in NCCL/PyTorch
    benchmarking()             # Measure actual NCCL bandwidth

    text("### Part 2: distributed training")
    text("Walk through bare-bones implementations of each strategy on deep MLPs.")
    text("Recall that MLPs are the compute bottleneck in Transformers, so this is representative.")

    data_parallelism()         # Cut up along the batch dimension
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

    # 阅读边界：本讲 DP 包含一次训练更新，TP、PP 只实现前向。后文给出的反向公式
    # 是编者补充，不意味着下面的普通 dist 调用已建立跨进程 autograd 图。
    # 实际 Transformer 的注意力、序列长度和专家路由会改变通信布局；MLP 示例用于
    # 隔离最基本依赖，不能据此断言所有模型的瓶颈都是同一个矩阵乘法。
    text("What's missing?")
    text("- Communication/computation overlap")
    text("- More general models (with attention, etc.)")
    text("- Other forms of parallelism (e.g., sequence parallelism, expert parallelism, combinations)")
    text("- Jax/TPUs: just define the model, the sharding strategy, and the Jax compiler handles the rest "), link(title="levanter", url="https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html")
    text("- But we're doing PyTorch so you can see how one builds up from the primitives")

    text("### Summary")
    text("- Many ways to parallelize: data (batch), tensor/expert (width), pipeline (depth), sequence (length)")
    text("- Data parallelism: DDP (all-reduce), FSDP/ZeRO (all-gather + reduce-scatter)")
    text("- Tensor parallelism: requires very fast interconnects (e.g., NVLink)")
    text("- Pipeline parallelism: can work with slow interconnects, but need to work to reduce pipeline bubbles")
    text("- Can **re-compute** or store in **memory** or store in another GPUs memory and **communicate**")
    text("- Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure")


# 通信原语阅读法：先写“每个 rank 原来拥有什么”，再写“结束后需要什么”。
# 以下 rank0 等变量只是串行画输入/输出示意，没有真的启动四块 GPU。
# 真正执行在 collective_operations_main 中；collective 通常要求所有参与 rank
# 以相同顺序调用兼容操作，否则可能等待不返回。rank 是进程组内编号，常用一进程
# 对应一 GPU，但它本身并不等同于物理设备编号。
def collective_operations():
    text("**Collective operations** are the conceptual primitives used for distributed programming "), article_link("https://en.wikipedia.org/wiki/Collective_operation")
    text("- These are classic in the parallel programming literature from the 1980s.")
    text("- *Collective* means that you specify a general communication pattern across many devices.")
    text("- This can be better/faster than managing point-to-point communication yourself.")

    text("**Setup**:")
    image("images/ranks.png", width=500)
    text("- **Rank**: a particular device/GPU (e.g., 0, 1, 2, 3)")
    text("- **World size**: total number of devices (e.g., 4)")

    text("Operations:")
    text("- Broadcast, scatter, gather, reduce (foundations)")
    text("- All-gather, reduce-scatter, all-reduce (workhorse)")
    text("- All-to-all (for MoEs)")

    # Broadcast：root 输入 [4]，每个 rank 输出 [4]，元素不做算术运算。
    # 真实 API 中非 root 也要预先准备接收缓冲区；“没有有效输入”不等于无需分配内存。
    text("**Broadcast**: copy from rank 0 to all ranks")
    # Input
    rank0 = tensor([0., 1, 2, 3])

    # Output
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    text("Minor use case: rank 0 loads initial checkpoint and broadcasts to all ranks")

    # Scatter：root 的 [4] 被切为 P=4 个 [1]，rank r 得到第 r 块。
    # 它分配数据所有权，不复制完整输入，也不把不同 rank 的值相加。
    text("**Scatter** tensor on rank 0 to all ranks")
    # Input
    rank0 = tensor([0., 1, 2, 3])

    # Output
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    text("Note: stepping stone to understanding reduce-scatter")

    # Gather：每卡 [1] → 仅 root [4]，按 rank 顺序拼接。其他 rank 不因此拥有完整结果。
    # “相反”描述布局变化；若此前已经丢掉信息，gather 本身不会恢复被丢掉的内容。
    text("**Gather** pieces from all ranks to rank 0 (opposite of scatter)")
    # Input
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # Output
    rank0 = tensor([0., 1, 2, 3])

    text("Note: stepping stone to understanding all-gather")

    # Reduce：每卡同形状 [1] → root [1]，这里 SUM 得到 0+1+2+3=6。
    # 与 gather 的差别在于逐元素合并，输出长度不随 P 增长；SUM 和 AVG 也须明确区分。
    # 浮点加法并非严格结合，换归约顺序可能带来舍入差异，而非数学目标改变。
    text("**Reduce** pieces from all ranks to rank 0, applying some operation (e.g., sum, min, max)")
    # Input
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # Output
    rank0 = tensor([6.])  # Sum of all ranks (0 + 1 + 2 + 3)

    text("Note: stepping stone to understanding all-reduce")

    # All-gather：每卡输入 [1]，每卡输出 [4]；一般每卡 n 个元素会收集成 Pn 个。
    # 例如参数总量 N，每卡持 N/P 个，收集后每卡暂时拥有 N 个；分片可减少常驻存储，
    # 但算某层时的完整参数及通信缓冲仍可能造成峰值显存。拼接没有求和。
    text("**All-gather**: perform gather to all ranks, not just rank 0")
    # Input
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # Output
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    text("Use case: each rank holds parameter shard, gather to get full parameters for forward pass")

    # Reduce-scatter：每卡输入 [4]，每卡输出 [1]。把四行输入叠成 [P,4] 后，
    # 先沿 rank 轴求和得 [6,10,14,18]，再把第 r 块交给 rank r。
    # 这里英文 dim 0/1/2/3 是向量坐标；不是对每个一维张量调用四种不同维度归约。
    # 梯度应用：每卡都对各参数贡献局部梯度，SUM 后只保存自己负责的参数块。
    # 是否还需除 P，取决于局部损失定义；这个 SUM 原语不会自动做数据并行平均。
    text("**Reduce-scatter**: perform reduce on each dimension, scatter results")
    # Input
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # Output
    rank0 = tensor([6.])  # Sum along dim 0 (0 + 1 + 2 + 3)
    rank1 = tensor([10.]) # Sum along dim 1 (1 + 2 + 3 + 4)
    rank2 = tensor([14.]) # Sum along dim 2 (2 + 3 + 4 + 5)
    rank3 = tensor([18.]) # Sum along dim 3 (3 + 4 + 5 + 6)

    text("Use case: after backward pass, sum gradients from different data shards, but distribute storage")

    # All-reduce：每卡 [4] → 每卡 [4]，所有输出都是逐坐标归约后的完整向量。
    # 等式表达相同数学结果，不保证 NCCL 实际一定调用两个独立 API；它会选择算法。
    # FSDP/ZeRO 可以在 reduce-scatter 后就保留梯度分片、局部更新，再按需收集参数，
    # 从而避免始终复制全部训练状态。各 ZeRO 阶段分片对象不同，不能一概称全分片。
    text("**All-reduce** = reduce-scatter + all-gather")
    # Input
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # Output
    rank0 = tensor([6., 10, 14, 18])
    rank1 = tensor([6., 10, 14, 18])
    rank2 = tensor([6., 10, 14, 18])
    rank3 = tensor([6., 10, 14, 18])

    text("Use case: after backward pass, sum gradients from different data shards, but replicate full parameters")
    text("Breaking all-reduce into reduce-scatter + all-gather allows for flexibility (e.g., ZeRO/FSDP)")

    # All-to-all：这里每卡输入/输出都是 [4]；输入位置 j 表示发给目的 rank j。
    # 输出位置 i 来自源 rank i：例如 rank2 收到 [2,6,10,14]。把全网数据写成
    # “源 rank × 目的 rank”矩阵，路由等价于转置；每卡并未收到全部 16 个元素。
    # MoE 会先按目标专家打包 token，再交换、执行专家、逆向路由。各专家 token 数
    # 不均时可用不同 split size，但慢专家及最拥挤链路会决定整体等待时间。
    text("**All-to-all**: each rank sends each other rank some tensor (most general)")
    # Input
    rank0 = tensor([0., 1, 2, 3])      # send  0 to rank 0,  1 to rank 1,  2 to rank 2,  3 to rank 3
    rank1 = tensor([4., 5, 6, 7])      # send  4 to rank 0,  5 to rank 1,  6 to rank 2,  7 to rank 3
    rank2 = tensor([8., 9, 10, 11])    # send  8 to rank 0,  9 to rank 1, 10 to rank 2, 11 to rank 3
    rank3 = tensor([12., 13, 14, 15])  # send 12 to rank 0, 13 to rank 1, 14 to rank 2, 15 to rank 3

    # Output
    rank0 = tensor([0, 4, 8, 12])
    rank1 = tensor([1, 5, 9, 13])
    rank2 = tensor([2, 6, 10, 14])
    rank3 = tensor([3, 7, 11, 15])

    text("Notes:")
    text("- Useful for MoEs: each rank has split of data and subset of experts; need to route data to experts")
    text("- For balanced splits, all-to-all looks like transpose")
    text("- Also handles unbalanced splits (but want splits to be as balanced as possible)")

    text("Way to remember the terminology:")
    text("- Reduce: performs some associative/commutative operation (sum, min, max)")
    text("- Scatter is inverse of gather")
    text("- All: means destination is all devices")


# 硬件读图：GPU 节点图中的高速域适合频繁交换激活；跨节点通常通过网卡连接。
# 带宽数字须区分每卡/全机、单向/双向、十进制 GB/s 与二进制 GiB/s，以及峰值/
# 实测。下面是讲义中的示例配置，不是所有机器的保证；拓扑拥塞也会降低带宽。
def hardware():
    text("Classic (in the home):")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=500)
    text("- GPUs on same node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/s) "), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    # 原稿范围提示：家用 Ethernet 的 ~200 MB/s 是示例，不是 Ethernet 技术上限。
    # PCIe 代际、通道数和双向合计口径也要核对；不能把标称总吞吐当单 GPU 单向实测。
    text("- GPUs on different nodes communicate via Ethernet (~200 MB/s)")
    
    text("Modern (in the data center):")
    image("images/gpu-node-overview.png", width=700)

    text("Typical setup:")
    text("- 8 GPUs per node, connected by NVLink to an NVSwitch (B200s' NVLink 5.0 gets 1.8 TB/s; HBM was 8 TB/s)")
    text("- 256 nodes per pod, connected by Infiniband (via PCIe -> HCA / Infiniband NIC -> Infiniband cable) (~0.05 TB/s)")
    text("- N pods per cluster / datacenter, connected by Ethernet (via PCIe -> CPU)")

    # 编者补充：RDMA 减少数据传输中的 CPU 拷贝/协议处理；CPU 仍参与连接建立、
    # 内存注册等控制工作。GPUDirect RDMA 还要求 GPU、网卡、驱动及拓扑支持，才能
    # 让网卡直接访问 GPU 内存；普通主机内存 RDMA 不自动等于 GPU 到 GPU 零拷贝。
    # 下面“标准 Ethernet 不支持”指普通 socket 路径；随后 RoCE 就是在以太网上实现
    # RDMA 的方案，因此不能理解为任何以太网络都必须经 CPU 搬运全部有效载荷。
    text("Bypassing the CPU:")
    text("- Ethernet requires passing through the CPU (copying data to kernel socket buffer, build TCP packets, copy to NIC ring buffer)")
    text("- Remote Direct Memory Access (RDMA): allows one GPU to directly read/write another GPU's memory without involving the CPU")
    text("- Infiniband supports RDMA, but standard Ethernet does not")

    text("Advancements:")
    # 原稿疑点：官方 GB200 页面确认单机架 72 GPU、36 Grace CPU 的 NVLink 域，
    # 但“8 GPU/tray × 9 tray”不宜当作 NVL72 实体托盘布局的可靠规格；节点与机架
    # 层级不可混淆。这里保留英文，学习重点是高速互联域扩大到 72 GPU。
    text("- GB200/GB300 NVL72: 8 GPUs per tray, 9 trays per rack -> 72 GPUs in one NVLink domain")
    text("- RDMA over Converged Ethernet (RoCE): Ethernet bypasses CPU, similar but cheaper/weaker than Infiniband, used by Meta")

    # NCCL 层处理“如何传”：根据拓扑选择 ring/tree 等路径、分块和 GPU kernel。
    # 上层仍须正确选择“传什么”和“何时传”；库不会自动修正重复计算的损失或错误
    # 归一化。共享网络、PCIe 路径及 GPU 通信 kernel 也可能与训练竞争资源。
    text("### NVIDIA Collective Communication Library (NCCL)")
    text("NCCL translates collective operations into low-level packets that are sent between GPUs. "), link(title="talk", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    text("- Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)")
    text("- Optimizes the path between GPUs")
    text("- Launches GPU kernels to send/receive data")


# API 层：torch.distributed 指定进程组、源/目的、归约运算和缓冲区；后端实现传输。
# Gloo 与 NCCL 的操作支持并非完全相同，CPU fallback 不保证整讲每个操作都可运行。
# 尤其本讲 DP 使用 ReduceOp.AVG，应按所用后端/版本核对，不能仅凭有 CPU 就承诺
# 完整分布式训练通过。这里不运行昂贵 benchmark，也不把静态验证说成多卡实测。
def torch_distributed():
    text("PyTorch distributed library (`torch.distributed`) "), link(title="documentation", url="https://pytorch.org/docs/stable/distributed.html")
    text("- Provides clean interface for collective operations (e.g., `all_gather_into_tensor`)")
    text("- Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)")
    text("- Also supports higher-level algorithms (e.g., `FullyShardedDataParallel`) [not used in this course]")

    text("Let's walk through some examples.")
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):  # @inspect rank world_size
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    # 真实四进程示例：rank r 初始向量为 [r,r+1,r+2,r+3]，形状 [4]。
    # SUM 后每卡都是 [6,10,14,18]；all_reduce 原地修改 data，没有额外返回结果张量。
    # barrier 用于协调到达时刻，不保证多进程 print 绝对按 rank 排序。
    # async_op=False 避免返回 Work 句柄，但 CUDA 的主机返回与所有 GPU 工作完成仍需
    # 区分；下方 benchmark 用 synchronize 显式等待设备执行。
    ### All-reduce (dist = torch.distributed)
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    # 真实 reduce-scatter：input [P]、output [1]；P=4 时 rank r 输出 [6+4r]。
    # torch.empty 的旧内容没有意义；必须等操作写入之后才能解释输出值。
    ### Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # Input
    output = torch.empty(1, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # 接着将每卡 [1] 收集到 [P]，得到与前面 SUM all-reduce 相同的完整结果。
    # 这段代码展示的是归约结果的复制；如果只 all-gather 原始局部输入，不会自动求和。
    ### All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    text("Indeed, all-reduce = reduce-scatter + all-gather!")

    cleanup()


# 计时主线：先建立进程组和分配张量 → warmup → 同步 → 计时通信 → 再同步。
# 只计一次的大张量实验用于教学，不足以给硬件下结论；噪声、barrier 和启动开销
# 都可能进入结果。小消息应看延迟，大消息才更接近带宽受限区间。
# num_elements 是元素数，默认 float32 每个 4 字节，100*1024**2 元素为 400 MiB。
def benchmarking():
    text("How fast does communication happen?")

    # All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)

    text("References:")
    link(title="How to reason about collective operations", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")
    link(title="Sample benchmarking code", url="https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # Create tensor
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # Warmup
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # 编者推导与原稿口径核对（字幕 00:48:30–00:51:31；定义以 NCCL 官方资料为准）：
    # 设 S=size_bytes 为每卡完整输入字节数，P=world_size，t=duration。
    # ring 把 S 分成 P 块；reduce-scatter 与 all-gather 各有 P−1 步，每步每卡发送 S/P。
    # 因此每卡发送量=2(P−1)S/P，每卡接收量相同；每卡收发合计=4(P−1)S/P。
    # 全网把每次传输只在发送端计一次，总发送量=2(P−1)S；这正是下面 sent_bytes。
    # 若同时统计全网发送与接收，会将同一次传输计两遍，得到 4(P−1)S。
    # 原英文“2x because send + receive”与此口径不符：这里 2 来自归约分片、收集两阶段。
    # total_duration=P*t 是用于每卡归一化的 rank-time，不是操作实际花了 P 倍墙钟时间。
    # 最后 bandwidth=2(P−1)S/(P*t)，等于 NCCL busbw；algbw 则为 S/t。
    # P=4、S=400 MiB 时：全网发送 2400 MiB，每卡发送/接收各 600 MiB，
    # 每卡收发合计 1200 MiB。若 t=10 ms，algbw=39.0625 GiB/s，busbw=58.59375 GiB/s。
    # 这是标准通信量模型的等效指标，不是抓包得到的全部协议字节或全网总吞吐。
    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # 原稿疑点：下方约等式应在 P 很大时读成 bandwidth≈2*S/t；若仍用变量
    #  total_duration=P*t，就应写 2*S*(P−1)/total_duration，不能丢掉 P−1。
    # “与 world_size/topology 无关”指标准归一化口径；有限 P 的因子仍为 2(P−1)/P，
    # 实测 t 更会随拓扑、拥塞和算法变化，绝不保证增加 GPU 后带宽不变。
    # 打印除以 1024**3，实际单位为 GiB/s，虽原字符串写 GB/s。对照硬件十进制
    # GB/s 时应换算；双向峰值也不能直接与这里按单次发送计数的指标比较。
    # Notes:
    # - Effective bandwidth ~ 2 * size_bytes / total_duration
    # - Independent of world_size
    # - Independent of topology (ring or tree)

    cleanup()  # @stepover


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # Reduce-scatter 的同一口径（字幕 00:51:40–00:53:11）：令 S=data_bytes，
    # 它是每卡完整 input [P,num_elements] 的字节数，不是 output 的字节数。
    # 每卡 output 大小为 S/P；ring 每卡发送/接收各 (P−1)S/P，全网发送 (P−1)S。
    # 于是下面 sent_bytes 是全网发送量，bandwidth=(P−1)S/(P*t)，即 NCCL busbw；
    # algbw=S/t。每卡收发合计仍应额外乘 2，不能因无 all-gather 阶段就说没有接收。
    # 本次调用 P=4 时 S=1600 MiB、output=400 MiB，每卡发送 1200 MiB，
    # 全网发送 4800 MiB。它的完整输入比上一个 all-reduce 大 4 倍；因此下方“两倍
    # 数据、两倍时间”只适用于比较同一个 S、同一瓶颈下的理想通信模型，不适用于
    # 直接比较这两次调用的元素参数。实际 t 还含延迟和同步开销。
    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # Notes:
    # - all-reduce = reduce-scatter + all-gather
    # - all-reduce moves 2x the data in 2x the time compared to reduce-scatter, so similar bandwidth

    cleanup()  # @stepover


# DP 读图：橙线切的是数据 batch；每个 rank 的层 0…3 都完整存在。
# 前向和反向在本卡独立处理局部样本，反向结束后同步参数梯度，再用相同优化器
# 规则更新。普通 DP 减少每卡样本计算及激活量，但参数、梯度、Adam 状态仍全复制。
def data_parallelism():
    image("images/data-parallelism.png", width=300)
    text("Sharding strategy: each rank gets a slice of the data")

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    text("Notes:")
    text("- Losses are different across ranks (computed on local data)")
    text("- Gradients are all-reduced to be the same across ranks")
    text("- Therefore, parameters remain the same across ranks")

    text("Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory")


# 数据是随机 [B,d]=[128,1024]，没有标签；损失是最终激活平方的均值。
# 这用于展示通信及更新，并不代表语言模型交叉熵任务。父进程生成一个 data 后传给
# 各进程，DP 才在子进程切片；这不是生产环境按 rank 分别流式加载数据的实现。
def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)  # @stepover

    # 形状追踪：P=4，local_batch_size=32；rank2 取样本 [64:96]，data [32,1024]。
    # 四层参数各 [1024,1024]，每次 x@param 仍是 [32,1024]，GELU 逐元素作用。
    # int_divide 强制 B 能被 P 整除，避免本示例把不同局部样本数误当等权。
    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    # --- B0 ---
    # --- B1 ---
    # --- B2 ---
    # --- B3 ---
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size @stepover
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        # 损失/梯度逐步推导（编者补充；课堂平均梯度见字幕 00:58:23–01:00:43）：
        # 令 z_r 是本卡最终输出，b=B/P，局部损失 ℓ_r=Σᵢⱼ z_r[i,j]^2/(b*d)。
        # 因此 dℓ_r/dz_r=2*z_r/(b*d)，loss.backward 得到 g_r=∇θℓ_r。
        # 全局目标 ℒ=Σ全部输出平方/(B*d)=(1/P)Σ_r ℓ_r，所以 ∇θℒ=(1/P)Σ_r g_r。
        # 下面 AVG 恰好得到这个梯度；若改用 SUM，必须再除 P 才保持相同目标尺度。
        # 手算：两卡的局部平均梯度为 2、6，则全局平均为 4，SUM 为 8；若 AVG 后再除一次 P
        # 会错误得到 2。不能只看“all-reduce”名称就决定除数。
        # 若局部样本数 b_r 不等，应按 b_r/B 加权局部均值；若 loss 本来是局部总和，
        # 要得到全局均值就应 SUM 后除 B（以及这里的 d），不能机械平均局部总和。
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # 此处同步的是 param.grad，不是 param 本身；前向激活不会跨 DP rank 交换。
        # 相同初值 + 相同平均梯度 + 相同 AdamW 状态/超参数，才保证各卡更新后参数一致。
        # AdamW 有动量、二阶矩、epsilon 和解耦权重衰减，不能把梯度缩放随意说成与
        # 学习率缩放严格等价。实际 DDP 还会做梯度桶、重叠和初始同步等工程工作。
        # Sync gradients across workers (ONLY difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 原稿边界：这里 num_steps=1，没有 optimizer.zero_grad() 尚不影响首次更新。
        # 若直接增加步数，PyTorch 会把新梯度累积到旧 grad；那就不再是每步独立的 SGD/
        # AdamW 梯度。本讲保留原代码，扩展训练循环时必须明确清零或有意梯度累积的约定。
        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)  # @stepover

    cleanup()  # @stepover


# TP 读图：橙线切过每一层的宽度；同一 batch 由多个 rank 合作完成一层。
# 这里采用每层列分片，且每层都收集完整激活，便于看懂依赖，但通信频繁。
# 与 DP 对比：DP 同步参数梯度，TP 的前向中间激活本身就要跨卡交换。
def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    text("Sharding strategy: each rank gets part of each layer, transfer all data/activations")

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)  # @stepover

    data = data.to(cuda_if_available(rank))  # All ranks get the data (batch_size x num_dim)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim @stepover

    # 矩阵分块：W=[W₀|…|Wₚ₋₁]，每个 W_r [d,d/P]=[1024,256]。
    # 输入 X [128,1024] 在各卡复制，本卡计算 H_r=GELU(XW_r) [128,256]。
    # GELU 逐元素，允许先对列块非线性再拼接：GELU([A|B])=[GELU(A)|GELU(B)]。
    # 若运算跨列归一化（例如完整隐藏维 LayerNorm），就不能直接用这个分解替代。
    # Create model (each rank gets 1/world_size of the parameters)
    #  |  |  |  |
    # W0 W1 W2 W3
    #  |  |  |  |
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # Forward pass
    x = data
    for layer in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[layer]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # all_gather 的列表有 P 个 [128,256]，分别接收各源 rank 的列块；不是沿 batch 切。
        # cat(dim=1) 后 X 为 [128,1024]，所有 rank 得到相同布局，供下一层本地矩阵乘。
        # 每层每卡完整激活约 B*d 个元素，列表和拼接输出还带来额外峰值内存；因此
        # “参数变为 1/P”不意味着所有显存占用也变为 1/P。
        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank)) for _ in range(world_size)]

        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)  # @stepover

    # 反向补充（本代码未实现；字幕 01:07:21–01:08:54 也明确普通 backward 不会补齐通信）：
    # 设 A_r=XW_r，H_r=GELU(A_r)，上游局部梯度 G_r=∂ℒ/∂H_r，则
    # D_r=G_r⊙GELU′(A_r)，dW_r=XᵀD_r，dX_r=D_r W_rᵀ。
    # 各 dX_r 对应不同输出列对同一 X 的贡献，因此 dX=Σ_r dX_r：这里应 SUM，
    # 不应像 DP 那样 AVG；除 P 会缩小链式法则要求的梯度。
    # 跨 rank all-gather 的伴随是 reduce-scatter(SUM)：前向把同一块复制给各下游，
    # 反向要把所有消费者对该块的贡献相加，再交还该块拥有者，输出 [B,d/P]。
    # 输入若逻辑上复制在各卡，反传到上一层时还须正确处理这些部分贡献与复制布局；
    # 通信的选择取决于梯度所处布局，不能简单给所有张量各做一次 AVG。
    # 若最终各 rank 对同一完整输出都计算相同 loss，须约定只由一个 rank 提供损失
    # 梯度，或把每卡损失设为 ℒ/P 后合并，避免把同一目标重复算 P 次。反之最终仅
    # 按本地列计算平方和时，全局均值分母是 B*d，而非 B*(d/P)。
    # 例：X=[1,2]，W₀=[1,0]ᵀ、W₁=[0,1]ᵀ，先忽略 GELU，局部上游梯度为
    # 3、5，则 dX₀=[3,0]、dX₁=[0,5]，正确 dX=[3,5]；平均会错误变成 [1.5,2.5]。
    # 这里 dist.all_gather 使用预分配缓冲区，没有实现上述可微通信，因此不能把
    # 这段前向直接补上 loss.backward() 就当作完整 TP 训练。
    # Backward pass: homework exercise

    cleanup()  # @stepover


# PP 读图：橙线位于 layer1 与 layer2 之间，P=2，每卡持完整宽度的两层。
# 前向只需相邻阶段传递边界激活，通常比 TP 每层全组通信更容易跨较慢链路部署；
# 但阶段负载不均、通信延迟及流水线空泡会限制加速。
def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    text("Sharding strategy: each rank gets subset of layers, transfer all data/activations")

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)  # @stepover

    # Use all the data
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # 层分配：L=4、P=2，rank0 负责逻辑层0/1，rank1 负责逻辑层2/3。
    # 每卡参数量约 L*d²/P；这里各 rank 仍收到全部输入 data，但只有 rank0 使用它
    # 作为真正的模型输入，其余 rank 接收前一阶段激活。生产实现可避免这份冗余输入。
    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers @stepover

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]  # @stepover

    # Forward pass

    # 微批：B=128、M=num_micro_batches=4，每个 [32,1024]，同一微批依次经过所有阶段。
    # 它与 DP 不同：微批没有在一张卡上独立完成整网，必须按阶段继续前传。
    # 编者理想调度：假设每阶段每微批耗时 τ、通信可忽略，P 阶段处理 M 微批需
    # (M+P−1)τ，总有效工作 PMτ，利用率 M/(M+P−1)。P=2、M=4 得 4/5=80%；
    # 不分微批时为 1/2。此为前向平衡流水线模型，并非下面阻塞实现的实测利用率。
    # 微批过小会让矩阵乘效率下降、消息数增加；增大 M 并非无代价。
    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size @stepover
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank)) for _ in range(num_micro_batches)]

    # 逐次追踪：rank0 算微批0后 send，rank1 recv 后算后两层；rank0 可继续算微批1。
    # recv 前缓冲区 torch.empty 无有效激活；send/recv 的形状、dtype、对端和调用顺序
    # 必须匹配。这里用阻塞点对点接口，没有显式实现通信与本阶段计算重叠。
    # 虽然可形成不同阶段间的计算流水，填充、排空和通信等待仍然存在。
    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)  # @stepover
            dist.send(tensor=x, dst=rank + 1)

    text("Not handled: overlapping communication/computation to eliminate pipeline bubbles")

    # 反向补充（本代码未实现）：最后阶段算损失并反传，对输入边界激活求梯度
    # [32,1024]，向前一阶段发送；前一阶段收到此梯度后继续链式法则，方向与前向相反。
    # 每阶段只更新自己的层参数；不同阶段的权重含义不同，不能把它们 all-reduce 平均。
    # send/recv 只搬张量数据，未自动连接两进程计算图；需显式保留边界激活、设置梯度
    # 接口并安排反向通信。本示例既无 loss，也无 backward、optimizer 或完整训练调度。
    # 微批梯度缩放：若每个微批损失都是自身元素均值、且微批大小相同，整批均值
    # 梯度=(1/M)Σ_m ∇ℓ_m。逐微批累积 backward 时应每次对 ℓ_m/M 反传，
    # 或累积后统一除 M，最后才进行一次 optimizer.step；不能每微批更新后仍声称
    # 等价于原来一个全局 batch。大小不同则用微批元素数/总元素数加权。
    # 例如四个微批的平均梯度为 1、3、5、7，目标整批梯度为 4，直接相加得到 16。
    # 保存所有微批前向激活会增加显存；重计算用额外 FLOPs 换显存，1F1B 等调度可
    # 调整激活存活期。这里不给未实现调度的实际吞吐承诺。
    # Backward pass: homework exercise

    cleanup()  # @stepover

############################################################

# 执行环境：MASTER_ADDR/PORT 是 rendezvous 协调入口，不是全部 GPU 数据经
# rank0 转发的意思。localhost 加固定端口适用于本机教学，跨机器需可达地址，
# 并发运行还要避免端口冲突；GPU 数量也须满足 rank→device 的映射。
# 使用 NCCL 时通常应显式建立各进程正确的当前 CUDA 设备，尤其 barrier 等隐式
# 设备选择可能受影响；这里保留原实现，不把静态可编译视作多 GPU 配置已验收。
def setup(rank: int, world_size: int):
    """Initializes the distributed environment (called at start of process)."""
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Cleans up the distributed environment (called at end of process)."""
    torch.distributed.destroy_process_group()


# trace 与执行必须分开理解：此上下文把 dist 中 Python 函数替换为返回 None 的
# 空操作，用于让讲义追踪器走过控制流。它不模拟 all-reduce 的数学结果，也不会
# 把 all_gather/recv 的 torch.empty 缓冲区填好；显示的值可能未初始化或没有意义。
# 因此 trace 路径中的通信耗时、参数一致性和最终激活都不能当成多进程正确性证据。
# 退出上下文恢复函数；这是一种教学追踪技巧，不是分布式训练的测试替身。
class DisableDistributed:
    """
    Context manager that temporarily disables distributed functions (replaces with no-ops).
    This is for when we're tracing the lecture, since we can't trace through
    multiprocessing, so we just want to run the function directly without
    distributed communication.
    """
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


# 正常路径 mp.spawn 启动 P 个进程，各自以 rank=0…P−1 执行同一函数；这是
# SPMD：程序相同、rank 决定数据和通信角色。sys.gettrace() 为真时只执行 rank0，
# 但 world_size 仍传 P，配合上面的空操作仅保留代码路径，不能解释为真的四卡运行。
# 调试器也可能使 gettrace 非空，使用时应注意会进入这条特殊分支。
# kwargs.values() 依赖调用时参数顺序，适合这份固定示例，不是通用安全启动器。
def spawn(func: Callable, world_size: int, *args, **kwargs):
    """
    Launches `world_size` processes that each calls `func` on world_size, args, kwargs.
    Note: if we are being traced (inside edtrace), we just run the function directly without multiprocessing and disable distributed functions.
    """
    # Note: assume kwargs are in the same order as what main needs
    if not sys.gettrace():
        # This is the normal code path for multiprocessing
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # If we're being traced (inside edtrace), just run the function directly.
        with DisableDistributed():  # @stepover
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)


# 原稿初始化边界：每次调用都 manual_seed(0)，同形状参数会重复初始化；DP 的
# 副本因此相同，但同一模型各层也重复，TP 同形状的不同列块也可能重复，不能
# 把这个演示当作从同一个独立随机完整模型精确切片所得的 TP 基准。
# 缩放按 sqrt(num_outputs)：独立零均值单位方差输入下，线性输出方差约为
# num_inputs/num_outputs。DP 方阵该比值为1，TP [d,d/P] 则为P；所以 TP 的
# 输出尺度不能直接与 DP 输出作数值等价比较。初始化选择需结合完整逻辑矩阵。
def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    """Create parameters and put them on the `rank`-th GPU."""
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=cuda_if_available(rank)) / math.sqrt(num_outputs))


# 整除检查对应布局前提：DP 的 B/P、TP 的 d/P、PP 的 L/P 与 B/M 都要为整数。
# 更一般的非均匀分片需要显式记录大小、padding 或不同 split，不能静默取整。
def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b


# 摘要只打印形状和首元素，不能证明全张量相同、梯度正确或训练收敛。
# .item() 读取 CUDA 标量可能引入同步，因此它放在演示输出中，不属于纯通信内核。
def summarize_tensor(tensor: tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"


# 带解答自测（编者补充）：
# 1. 四卡每卡输入4个元素，all-gather 与 SUM all-reduce 输出形状分别是什么？
# 答：前者每卡16个元素，后者每卡4个；前者拼接来源，后者按位置求和。
# 2. 等大小局部 batch 的 mean loss 经 AVG 后，还应再除 P 吗？
# 答：不应；AVG 已使梯度等于全局 mean 的梯度，再除会缩小 P 倍。
# 3. TP 的 dX 部分贡献为什么不能平均？答：它们来自不同输出列，是链式法则的
# 不同加项；需要 SUM。复制相同 loss 的额外计数应在目标定义处单独处理。
# 4. P=4 时 all-reduce busbw/algbw 为多少？答：2(P−1)/P=1.5；这不代表
# 网络凭空增速，而是同一时间除以不同标准化字节数。收发合计还会多计一倍。
# 5. PP 四个等大微批分别计算 mean，累积梯度后怎样得到整批 mean？
# 答：四个梯度相加后除4；阶段间传激活梯度，各阶段参数无需互相平均。
# 6. trace 中 all-gather 后得到某个数能证明通信正确吗？答：不能，通信被禁用，
# 接收缓冲区未被真实写入；须与正常多进程路径严格区分。
if __name__ == "__main__":
    main()
