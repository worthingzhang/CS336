# 第 6 讲中文自学增强：从性能模型走到可解释的 Triton kernel。
# 学习目标：能读懂 program 的工作分配、指针与 stride、mask、归约和分块矩阵乘；
# 能区分设备执行时间、端到端延迟与首次编译成本，并解释融合为何减少访存。
# 先修：第 5 讲的算术强度 I=FLOPs/搬运字节、roofline、tiling 和在线 softmax。
# 阅读主线：硬件资源约束 → benchmark/profile → GeLU 融合 → 行归约 → K 维分块。
# 第 5 讲回答“为什么要少搬数据”，本讲逐行回答“哪些地址被读写、谁累加什么”。
# 保留原英文与代码；中文中的手算和推导为编者补充，原文疑点在相邻处注明。
# 来源索引（2026 课程；官方文档为 API/实现边界依据）：
# [S1] https://cs336.stanford.edu/ （Spring 2026 课程与本地原稿）
# [S2] https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
# [S3] https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
# [S4] https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# [S5] https://docs.pytorch.org/docs/stable/notes/cuda.html
# [S6] https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
# [S7] https://triton-lang.org/main/python-api/generated/triton.language.load.html
# [S8] https://triton-lang.org/main/python-api/generated/triton.language.dot.html
# 课堂交叉核对：控制目录 sources/lecture_06.srt，第三方公开视频字幕，非教师审校稿。
# 已核对 00:23:53–00:25:35（计时）、00:47:43–00:50:30（load/编译抽象）、
# 01:01:28–01:04:25（softmax）、01:08:42–01:10:59（分块归约）。
# 文档 main/stable 会随版本变化；具体性能及编译产物由安装版本、GPU与输入决定。
# 这里只增加注释，不把静态检查当作 GPU 正确性、性能或 edtrace 页面渲染验证。
import os
import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import triton
import triton.language as tl
from edtrace import text, link, image
from lecture_util import get_local_url
from gpu_util import cuda_if_available


# 整讲地图：每个例子增加一种协作需求。
# GeLU：每个元素独立；softmax：同行元素需共享最大值与分母；
# row sum：行太长时顺序遍历多个 tile；matmul：同时重用两个输入 tile。
# 读 kernel 时固定问四件事：program 负责什么输出？地址是什么？
# 临时张量形状是什么？边界位置用什么中性元填充？
def main():
    text("Last lecture: high-level overview of GPUs and performance")
    text("This lecture: benchmarking/profiling + writing kernels")

    review_of_gpus()
    benchmarking_and_profiling()           # Where are the bottlenecks?
    naive_vs_builtin_vs_compiled_gelu()    # Apply it to the GeLU example

    # Write Triton kernels
    triton_introduction()
    triton_gelu_example()      # Elementwise operation
    triton_softmax_example()   # Reduction (row fits in a block)
    triton_row_sum_example()   # Reduction (row doesn't fit in block)
    triton_matmul_relu_example()    # Tiling: use shared memory

    text("Summary:")
    text("- Know the programming model (PyTorch, Triton, PTX) to give you correctness")
    text("- Understand the hardware (SMs, warps, occupancy, bank conflicts, etc.) to optimize performance")
    text("- Benchmark to understand scaling")
    text("- Profile to see what's being executed for how long")
    text("- Triton: think in terms of thread blocks (read to shared memory, do stuff (fusion), write back HBM)")
    text("- Examples: GeLU (elementwise), softmax (row-wise), row sum (baby tiling), matmul (tiling)")

    text("Next time: more than one GPU!")


# 硬件复习：SM 是执行资源集合，HBM 是全卡大容量内存；
# 寄存器/shared memory 的预算按 SM 核算，L2/HBM 则由多个 SM 共享。
# 表中容量与带宽是具体型号/近似口径，不能拿不同层的峰值直接相加。
# 第 5 讲 roofline 的时间下界约为 max(F/P, Q/B)，F 是运算量、Q 是字节数；
# 降低 Q 只有在访存是主要瓶颈时才会按比例降低运行时间。
def review_of_gpus():
    text("## Hardware")
    image("images/gpu-hardware.png", width=800)
    text("| Accelerator                        | A100      | H100      | B200      |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| # SMs                              |       108 |       132 |       148 |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| Register size (per SM)             |    256 KB |    256 KB |    256 KB |", verbatim=True)
    text("| L1 cache + shared memory (per SM)  |    192 KB |    256 KB |    256 KB |", verbatim=True)
    text("| L2 cache size                      |     40 MB |     50 MB | 96-126 MB |", verbatim=True)
    text("| HBM size                           |     80 GB |     80 GB |    192 GB |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| Register bandwidth                 | ~116 TB/s | ~401 TB/s | ~447 TB/s |", verbatim=True)
    text("| L1 cache + shared memory bandwidth |  ~19 TB/s |  ~33 TB/s |  ~40 TB/s |", verbatim=True)
    text("| L2 cache bandwidth                 | ~5-8 TB/s |  ~12 TB/s |  ~17 TB/s |", verbatim=True)
    text("| HBM bandwidth                      |    2 TB/s | 3.35 TB/s |    8 TB/s |", verbatim=True)

    text("Note: the L1/L2 bandwidth numbers are rough estimates (online sources differ).")
    text("(B200s also have tensor memory (TMEM) for tensor cores (between registers and shared memory) that are invisible to programmer.)")

    # 编程层级与硬件层级不能一一等同：grid 包含多个 CTA/block；
    # 一个普通 CTA 的线程驻留在同一 SM 上，一个 SM 可同时容纳多个 CTA。
    # warp 是 NVIDIA 的 32 线程执行组，block 可有多个 warp。
    # 本讲 Triton program 通常映射一个 CTA；program 内 tl.arange 的长度
    # 表示逻辑元素数，不是物理线程数；一个线程可处理多个逻辑元素。
    text("## Programming model")
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)
    text("- *Thread*: executes code on a small part of the data")
    text("- *Thread block* or concurrent thread array (CTA): a group of threads")
    text("- *Grid*: collection of thread blocks")

    text("(H100s and B200s also have thread block clusters that enable distributed shared memory.)")

    text("Why thread blocks?")
    text("For elementwise operations (e.g., GeLU), threads are most natural: each thread processes one element.")
    text("- f(i) for i = 0, ..., N-1")
    text("However, for non-elementwise operations like softmax or matrix multiplication, threads need to communicate.")
    text("Reading/writing from HBM is slow, so use shared memory (local to SM).")
    text("Thread block: a collection of threads that access the same shared memory.")
    text("Consequently, a thread block is scheduled on one SM.")
    text("In Triton, think natively in terms of thread blocks (later).")

    text("## Interaction between programming model and hardware")
    text("Programming model provides an abstraction of the hardware.")
    text("In principle, don't need to think about anything else (for correctness).")
    text("In practice, performance is very sensitive to the hardware, so need to understand it to obtain high performance.")

    text("Let's go over some considerations.")

    # warp 的 lockstep 是入门简化：现代 NVIDIA 支持独立线程调度，
    # 但分支发散仍可能使不同路径在不同活跃掩码下执行，降低有效吞吐。
    # 不能据“zero cost”推断任意依赖和等待都免费；切换就绪 warp 能隐藏延迟，
    # 前提是有足够的就绪工作与资源。mask 也不意味着被屏蔽的槽位没有计算成本。
    text("**Warps**:")
    text("- Within a thread block, threads are grouped into warps (32 threads per warp).")
    text("- Example: thread block has 64 threads => it has 2 warps.")
    text("| TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT | TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT |", verbatim=True)
    text("- All threads within a warp must execute same instructions in lockstep on an SM.")
    text("- Control divergence: if different threads in a warp need to execute different instructions (if A, else B), must be done sequentially (bad)")
    text("| AAAAAAAAA....................... |", verbatim=True)
    text("| .........BBBBBBBBBBBBBBBBBBBBBBB |", verbatim=True)
    text("- SM runs multiple warps and switches between them (e.g., when one warp is blocked on HBM reads/writes) with zero cost.")

    text("**(Warp) occupancy**:")
    text("- Each thread can use between 0 and 255 registers.")
    text("- The more registers threads use, the fewer threads can be scheduled on an SM (low occupancy).")
    text("- Low occupancy isn't necessarily bad if each thread is doing more work.")
    text("- Example: thread coarsening (each thread processes multiple elements).")
    text("- Example: thread block has 64 threads, each using 160 registers, SM has 65536 registers")
    
    # 原英文举例写 64 个线程，下面实际计算用 128；以下以可执行代码为准。
    # 每 block 占 128×160=20480 个 32-bit 寄存器，65536//20480=3 个 block；
    # 共 3×128/32=12 个 warp，按最大 64 warp 得 occupancy=12/64=18.75%。
    # 这只是寄存器单约束估算，尚未计入分配粒度、shared memory、线程/block上限。
    # 更大 tile 可提高重用，却会增大临时值与寄存器需求，减少驻留 warp，甚至 spill。
    # 因此 occupancy 是诊断量，最高 occupancy 不等于最快；还要看数据重用和指令并行。
    # What we want to run
    num_threads_per_block = 128
    num_registers_per_thread = 160

    # What hardware offers
    max_registers = 65536  # Registers allowed per SM
    max_warps = 64         # Concurrent warps allowed per SM

    # What we can run at once
    assert num_registers_per_thread <= 255
    num_registers_per_block = num_threads_per_block * num_registers_per_thread  # @inspect num_registers_per_block
    num_blocks = max_registers // num_registers_per_block  # Limited by registers @inspect num_blocks
    num_warps = num_blocks * num_threads_per_block / 32  # @inspect num_warps
    occupancy = num_warps / max_warps  # @inspect occupancy

    # 用 32 个 bank、每个 4 字节的简化模型理解：float32 元素线性编号 i
    # 对应 bank=i mod 32。行宽32时，同一列的地址差32，落在同一 bank；
    # 行宽加到33时，同一列的 bank 随行号改变，能消除这个特定冲突。
    # 这是编者例子，不是通用最优布局；广播同一地址与访问同bank不同地址也不同。
    # matmul 本身并非必然有 bank conflict，布局、转置、swizzle 可改变访问方式。
    text("**Bank conflicts** (shared memory):")
    text("- Shared memory is divided into 32 banks, each 4 bytes wide.")
    text("B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("- Each cycle, each bank can only be accessed by one thread (if not the same exact location).")
    text("- If multiple threads access the same bank, accesses serialized (bank conflict).")
    text("- Worst case example: matrix where each row spans all banks; 32 threads accessing first column results in 32-way bank conflict!")
    text("- Unavoidable: when doing matmul A @ B, access rows of A and columns of B")
    text("- Solution: swizzling rearranges shared memory (e.g., row xor col) to avoid bank conflicts")

    # coalescing 关心一个 warp 的地址是否紧邻且对齐；bank conflict 关心共享内存 bank。
    # 原文128字节cache line是教学图示，实际事务还涉及32字节sector、对齐和缓存策略。
    # 32线程读连续float32共128字节，是好模式；每线程跨整行读一列可能浪费多个事务。
    # 这也说明相同 FLOPs 和逻辑读量，实际搬运字节及时间仍可能不同。
    text("**Memory coalescing** (HBM):")
    text("- When the 32 threads in a warp access HBM, memory accesses combined into transactions of 128 bytes (cache lines).")
    text("M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 M27 M28 M29 M30 M31", verbatim=True)
    text("M32 M33 M34 M35 M36 M37 M38 M39 M40 M41 M42 M43 M44 M45 M46 M47 M48 M49 M50 M51 M52 M53 M54 M55 M56 M57 M58 M59 M60 M61 M62 M63", verbatim=True)
    text("- Best case: full coalescing, all threads access the same cache line (32 threads x 4 bytes = 128 bytes).")

    # 此处 wave 图是假定每 SM 同时放一个 block 的简化模型。
    # 160/148 向上取整为2波，第二波只有12个block，形成尾部空闲。
    # 原文“make number of thread blocks divide # SMs”不宜按字面理解为充分条件：
    # 通常想让工作量接近有效并发容量的整数倍，并保有足够多 block；
    # 真实容量依赖每 SM 可驻留多少 block，且不同 block 的耗时也未必相同。
    text("**Block occupancy**:")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=400)
    text("- Thread blocks scheduled onto SMs in waves.")
    text("- B200 has 148 SMs, if we launch 160 thread blocks, first wave has 148 blocks, second wave has 12 blocks.")
    text("- Wave quantization problem: last wave has fewer thread blocks, leaving some SMs idle (low block occupancy).")
    text("- Solution: make number of thread blocks divide # SMs.")

    text("Summary:")
    text("- Programming model: grid (HBM) -> thread block (shared memory) -> thread (registers)")
    text("- Details of hardware (warps, bank conflicts, memory coalescing, occupancy) determine performance")


# 优化闭环先保正确，再测稳定基线，再定位瓶颈，然后只改明确的原因。
# benchmark 回答时间，profile 回答哪些 kernel 与调用消耗时间；
# profile 自身有开销，表中的时间不能不加区分地替代正常运行的基准时间。
def benchmarking_and_profiling():
    text("Recipe for success:")
    text("1. Benchmark and profile your code")
    text("2. Make changes")
    text("3. Benchmark and profile your code again")

    benchmarking()   # How long does it take?
    profiling()      # Where time is being spent?

    text("Benchmark and profile your code!")


# 本节时间趋势是特定硬件与输入的观察，不是所有尺寸的定理。
# 方阵乘 FLOPs 约2d³；小矩阵常被launch/调度主导，大矩阵才可能接近计算受限。
# run_operation2 先创建输入，再返回闭包，因此随机数生成不在 benchmark 的 run 内。
# 不要执行整讲来验证注释：这里含8192方阵和多轮GPU运行，成本很高。
def benchmarking():
    text("Benchmarking measures the wall-clock time of performing some operation.")
    text("It only gives you end-to-end time, not where time is spent (profiling).")

    text("It is still useful for:")
    text("- comparing different implementations (which is faster?), and")
    text("- understanding how performance scales (e.g., with dimension).")

    text("You can use [`torch.utils.benchmark`](https://pytorch.org/tutorials/recipes/recipes/benchmark.html).")
    text("We will roll our own to make benchmarking more transparent.")

    # Benchmark matrix multiplication
    matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
    result = benchmark(matmul)  # @inspect result

    # See how timing scales with dimension
    results = {}
    for dim in [256, 512, 1024, 2048, 4096, 8192]:
         results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))  # @inspect results @stepover

    text("Note: time is roughly constant when dimension is small, then cubic scaling.")


# 异步时间线：CPU发出launch后可以立即返回，GPU稍后才执行完。
# 只用CPU时钟包住run而不等待，往往测到提交时间，不是完成时间。
# 本函数在warmup后同步，排除之前排队工作；在同一当前stream依次记录
# start、run、end，等GPU完成后读取事件间隔，elapsed_time单位为毫秒。
# 这里虽然上文称wall-clock/end-to-end，实际返回的是CUDA事件区间时间，
# 不是包含输入准备、CPU逻辑、同步返回等环节的完整请求延迟。[S5][S6]
# 事件不会把所有CPU开销作为独立区间计入，但主机提交太慢导致GPU空闲时，
# 该空隙也可能落入事件区间；多stream工作需另建依赖，不能自动完整覆盖。
# 课堂00:23:53–00:25:35强调warmup、重复和同步；本实现仅1次warmup/3次试验，
# 适合展示机制，不足以证明稳定性能或可靠尾延迟。
# 编译成本口径：torch.compile/Triton常在首次执行时编译；warmup试图将其排除。
# 输入shape、dtype或specialization变化可能再编译，一次warmup并非永久保证。
# 若关心首次请求，需另报冷启动端到端时间；稳态加速不能隐去真实的一次性成本。
def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `func` by running it `num_trials`.  Return the average time."""
    # Warmup: first times might be slower due to compilation, etc.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times
    for trial in range(num_trials):  # Do it multiple times to capture variance
        # Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        run()  # Actually perform computation
        end_event.record()  # End timing

        torch.cuda.synchronize()  # Wait for CUDA threads to finish

        times.append((start_event.elapsed_time(end_event)))  # @inspect times

    mean_time = mean(times)   # @inspect mean_time @stepover
    return mean_time


# 看表时先确认输入规模、dtype、设备，再对应kernel名字与耗时。
# CUDA活动表主要展示设备侧活动，CPU编译/调度瓶颈需额外采集CPU活动。
# 不同尺寸可能选择不同实现，不能仅从一个长名字推断所有输入均使用同一tile。
# 若存在嵌套事件/重叠执行，直接相加时间也未必等于完整墙钟时间。
def profiling():
    text("While benchmarking looks at end-to-end time, profiling looks at where time is spent.")
    text("Independent of time, profiling also helps you understand what's going under the hood.")

    text("PyTorch has a built-in [profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).")
    text("In your assignment, you will use nsight to get more details.")

    text("## add(dim=2048)")
    add_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a + b))
    text(add_profile, verbatim=True)

    text("## matmul(dim=2048)")
    matmul_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    text("## matmul(dim=128)")
    matmul_profile = profile(run_operation2(dim=128, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    text("Observations:")
    text("- You can see which CUDA kernels are actually being called (the long names).")
    text("- Different CUDA kernels are invoked depending on the tensor dimensions.")

    text("Name of CUDA kernel tells us something about the implementation.")
    text("Example: cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bi...")
    text("- cutlass: NVIDIA's CUDA library for linear algebra")
    text("- sm100: corresponds to the NVIDIA Blackwell architecture (B200)")
    text("- f32: float32")
    text("- 64x64x16: tile shape (more on this later)")


# 这里同样先warmup并同步；profiler内同步确保退出采集前工作已完成。
# 该函数还写var/profiles.txt，运行讲义不仅会计算，也会产生文件。
# 本次只静态阅读这些路径，不用实际执行它来“检查注释”。
def profile(run: Callable, num_warmups: int = 1):
    # Warmup
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # Run the code with the profiler
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        torch.cuda.synchronize()

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=100,
                                      row_limit=10)

    # Append to profiles.txt
    with open("var/profiles.txt", "a") as f:
        f.write(f"Profile at {time.ctime()}:\n")
        f.write(table)
        f.write("\n\n")
    return table


# GeLU近似是逐元素公式，本来不要求不同元素通信。
# PyTorch eager 的每个算子却可能各自launch、读写全尺寸中间张量；
# 融合把中间值留在片上，减少launch与HBM流量，衔接第5讲算术强度。
# 内置和编译结果在本例可融合，不保证任意Python函数都能合成一个kernel。
# check_equal使用容差，说明比较数值近似，不是逐bit一致；编译也有浮点舍入边界。
def naive_vs_builtin_vs_compiled_gelu():
    text("Let's benchmark and profile the [GeLU activation function](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).")

    x = torch.tensor([1.])  # @inspect x

    # 1. Implementation naively from scratch in PyTorch (non-fused)
    y1 = naive_gelu(x)  # @inspect y1

    # 2. Built-in PyTorch implementation (fused)
    y2 = builtin_gelu(x)  # @inspect y2
    check_equal_1d(naive_gelu, builtin_gelu)  # Check it works

    # 3. Use PyTorch compiler on the naive implementation
    compiled_gelu = torch.compile(naive_gelu)  # @stepover
    y3 = compiled_gelu(x)  # @inspect y3 @stepover
    check_equal_1d(naive_gelu, compiled_gelu)  # Check it works (compilation shouldn't change semantics) @stepover

    # Benchmarking
    naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu)) # @inspect naive_time @stepover
    builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu)) # @inspect builtin_time @stepover
    compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time @stepover
    text("The builtin and compiled versions are significantly faster!")

    text("To understand why, let's look at the profiler to see where time is being spent.")

    text("## naive_gelu")
    naive_gelu_profile = profile(run_operation1(dim=16384, operation=naive_gelu))  # @stepover
    text(naive_gelu_profile, verbatim=True)

    text("## builtin_gelu")
    builtin_gelu_profile = profile(run_operation1(dim=16384, operation=builtin_gelu))  # @stepover
    text(builtin_gelu_profile, verbatim=True)

    text("## compiled_gelu")
    compiled_gelu_profile = profile(run_operation1(dim=16384, operation=compiled_gelu))  # @stepover
    text(compiled_gelu_profile, verbatim=True)

    text("Notes:")
    text("- Naive implementation: multiple kernels, requires many reads/writes from/to HBM (**no fusion**).")
    text("- Builtin and compiled versions: one kernel (**kernel fusion**), one read from HBM, one write to HBM.")
    text("- The compiled kernel is a Triton kernel.")


# Triton采用块级张量编程：Python包装器在CPU侧分配/launch，
# @triton.jit函数描述设备计算，并非逐语句在GPU解释执行普通Python。
# “load到共享内存”是直观模型，不是tl.load的API保证：load返回逻辑张量，
# 具体落在寄存器、如何跨warp归约及何时用shared memory由编译器布局决定。
# 课堂00:47:43–00:50:30也区分了源级表达与实际编译后的执行。
# 编译器负责映射不意味着资源无限；tile过大仍可能寄存器压力高或无法编译。
def triton_introduction():
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)

    text("In CUDA (developed by NVIDIA), specify what each thread does.")
    text("- Pros: fine-grained control")
    text("- Cons: need to manage more things (e.g., shared memory)")

    text("In Triton (developed by OpenAI), specify what each thread block does.")
    text("- Generally powerful enough (especially when getting started)")
    text("- Conceptual framework: load data into shared memory, operate on it, write back to global memory")


# 先读包装器怎样划分输出，再读kernel如何根据pid计算地址，最后看PTX。
# PTX是虚拟ISA，还需后端生成设备机器代码；PTX寄存器名也不是物理寄存器数。
# “一线程8元素”是所展示编译产物的观察，不是Triton对所有GPU/版本的约定。
def triton_gelu_example():
    text("Let's write the Triton kernel for GeLU.")

    x = torch.randn(8192, device=cuda_if_available())
    y = triton_gelu(x)

    check_equal_1d(triton_gelu, naive_gelu)  # Check for correctness @stepover

    text("Triton compiles down to PTX (parallel thread execution), an assembly language for GPUs.")

    text("We can see the PTX code generated by Triton.")
    link(get_local_url("var/triton_gelu-ptx.txt"))

    text("Observations:")
    text("- ld.global.* and st.global.* reads and writes from global memory")
    text("- %ctaid.x is block index, %tid.x is thread index")
    text("- %f* are floating point registers, %r* are integer registers")
    text("- One thread processes 8 elements at the same time (thread coarsening)")


# 连续性断言保证展平后offset线性访问成立；numel计的是所有维度元素。
# BLOCK_SIZE=1024是每program处理1024个逻辑元素，原注释Number of threads不准确。
# 物理线程数量与num_warps/编译布局相关；逻辑元素可由较少线程分担。
# 例如N=2500，ceil(2500/1024)=3个program；最后一个负责2048..3071，
# 只有2048..2499共452个元素有效。mask既保护读，也保护写。[S2]
def triton_gelu(x: torch.Tensor):
    # Check input
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    Block 0      |    Block 1      |     Block 2      |    Block 3     |
    num_elements = x.numel()  # @inspect num_elements
    BLOCK_SIZE = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # @inspect num_blocks

    # Launch the kernel
    kernel = triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Write out PTX (look at this later)
    output_ptx("triton_gelu", kernel)  # @stepover

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input starts at `x_ptr`
    # Output starts at `y_ptr`

    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    Block 0      |    Block 1      |     Block 2      |    Block 3     |

    # 形状追踪：pid/start是标量，offsets与mask都是[BLOCK_SIZE]。
    # x_ptr是输入起点；x_ptr+offsets是一组元素地址，不是先读取数据再相加。
    # Triton指针加1按一个所指元素前进，不按1字节前进；float32才对应4字节。
    # constexpr让BLOCK_SIZE在编译时已知，用于构造静态形状；变更它可产生新编译版本。
    pid = tl.program_id(axis=0)      # Identifies the block
    start = pid * BLOCK_SIZE         # Starting index of this block

    # Indices where this thread block should operate
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Don't read/write past the end of the tensor
    mask = offsets < num_elements

    # Read
    # masked load不会读取越界地址；这里没指定other，屏蔽槽位的值未定义。
    # 此例逐元素计算且store用同一mask，屏蔽槽位不影响有效输出；
    # 若随后要跨槽位归约，就必须指定合适other，不能照抄这个省略写法。[S7]
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    # 0.79788456近似sqrt(2/pi)，这与后面的builtin approximate="tanh"对齐。
    # 用指数实现tanh仅在数学上等价：非常大的正a可能使exp(2a)溢出，
    # 进而出现inf/inf=NaN；随机输入的allclose不足以覆盖该边界。
    # 保留原实现以便学习；不能由此推断它是适合所有幅值的生产GeLU。
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


# 从独立元素到行协作：softmax_j=exp(x_j)/Σ_k exp(x_k)，
# 每个输出都依赖整行的归一化分母；一行一个program可在片上完成归约。
# 这里不是第5讲的分块在线softmax：整行逻辑张量放进一个program，
# 不保存跨tile的running max/normalizer，适用范围受行宽与片上资源约束。
def triton_softmax_example():
    text("So far, we've looked at elementwise operations in Triton (e.g., GeLU).")
    text("Now let us look at operations that aggregate over multiple values.")

    text("We will roughly follow the Triton fused softmax tutorial: "), link("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    text("Recall the softmax operation is used in attention and generating probabilities.")
    text("Exponentiate and normalize each row of a matrix:")
    text("[0 0 0]      =>   [1/3 1/3 1/3]", verbatim=True)
    text("[1 1 -inf]        [1/2 1/2 0  ]", verbatim=True)

    text("Let's first start with the naive implementation and keep track of reads/writes.")
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=cuda_if_available())
    y1 = naive_softmax(x) # @inspect y1

    text("Now let us write the Triton kernel.")
    image("images/triton-softmax.png", width=600)
    y2 = triton_softmax(x)  # @inspect y2

    # Check our implementations are correct
    check_equal_2d(pytorch_softmax, naive_softmax) # @stepover
    check_equal_2d(pytorch_softmax, triton_softmax) # @stepover


# 数值稳定推导：令m=max_j x_j，分子分母同乘exp(-m)，概率不变。
# 此时x_j-m<=0，指数至多1；至少一个有限最大值对应exp(0)=1。
# 例如[1000,1000]直接exp会溢出，先减max得[0,0]，结果[1/2,1/2]。
# 成立条件是行内最大值有限且无NaN；整行-inf会出现-inf-(-inf)=NaN。
# 原例[0,0,100]减max后为[-100,-100,0]，前两项很小但数学上不是严格0。
def naive_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # 原读量总计漏了normalize读取denominator的M：按相同逻辑访问口径，
    # 5步读量依次MN、MN+M、MN、MN、MN+M，总计5MN+2M；
    # 写量为M、MN、MN、M、MN，即3MN+2M，与官方教程[S3]一致。
    # 这些是理想化逐算子元素访问计数，不等于缓存命中后的实测HBM事务。
    # 融合只读MN、写MN，N足够大时M项可忽略，总流量约从8MN降到2MN，故理想约4倍；
    # 指数运算、归约、launch与占用率仍在，不能把4倍当实测速度保证。
    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y


# M是行数，N是列数，grid=(M,)；N=3时BLOCK_SIZE=4，多一槽用mask补齐。
# 传入row_stride可允许行与行之间有间隔，但kernel的列地址直接+col_offsets，
# 隐含x.stride(1)=y.stride(1)=1；它不是支持任意转置/切片的通用softmax。
# 例如连续[2,3]的行stride=3；按行隔一行取样后行stride可变而列stride仍为1。
# 若每隔一列取样使列stride=2，此实现读错地址，不能只传row_stride就解决。
# 包装器未显式检查这些前提；阅读示例时要自行识别接口约束。
def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    # 地址例：一行3列、row_idx=1、row_stride=3，行首偏移3，
    # 4个逻辑地址偏移为[3,4,5,6]，mask=[真,真,真,假]。
    # 最后一项填-inf是max与exp-sum这条计算链的正确选择：
    # 它不提高最大值，减有限max后仍为-inf，exp(-inf)=0，不增加分母。
    # 不能填0：若真实行是[-2,-2,-2]，补0会引入一个虚假的指数项。
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    # 此处x_row形状是[BLOCK_SIZE]，axis=0对该逻辑向量归约成标量，
    # 不是沿原二维矩阵的“第0维”跨行归约；row_idx早已选定一行。
    # numerator仍是向量，denominator是标量，除法广播回所有槽位。
    # 手算[1,1,-inf]补到4槽：减max得[0,0,-inf,-inf]，
    # 指数[1,1,0,0]，分母2，输出有效三槽[1/2,1/2,0]。
    # store的mask最终丢弃补齐槽；它不能补救此前归约中用了错误填充值。
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


# “放不进block”应理解为一次处理整行的资源代价太大，
# 不是说逻辑元素数一旦大于物理线程数就不能处理。
# 一个program可循环读多个tile，保留固定长度acc，避免持有整行临时值。
# 但一行仍只有一个program；行数很少、行极长时可能并行度不足。
def triton_row_sum_example():
    text("In the softmax example, an entire row fits in a block, so the reduction happens within a block (handled by Triton).")
    text("What if the row doesn't fit in a block?")
    text("Example: 4096 columns, but block size is 1024...")

    text("Strategy:")
    text("- Break up row into tiles (4 in the example above)")
    text("- Each thread iterates over tiles and accumulates a sum")
    text("- Do final reduction (sum) over accumulators of each thread (shared memory or warp shuffles)")

    text("Consider the simpler example (row sum instead of softmax):")
    x = torch.tensor([[1., 2, 3, 4], [5, 6, 7, 8]], device=cuda_if_available())  # @inspect x
    y1 = builtin_row_sum(x)  # @inspect y1

    image("images/triton-row-sum.png", width=600)

    y2 = triton_row_sum(x)  # @inspect y2


# 参考语义是每行一个标量，输入[M,N]输出[M]。
# 原讲的演示只计算两个结果，未在此处assert比较；不要把演示等同完整测试。
def builtin_row_sum(x: torch.Tensor):
    return x.sum(dim=1)


# 该kernel没有传stride，而用row*N，隐含输入是连续的行主序二维矩阵。
# 输出dtype等于输入，但内部acc为float32，最后store可能转回低精度。
# 这提高了部分累加的精度，却不保证任意长行、任意输入与torch逐bit一致。
def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Which row are we processing?

    # Accumulator for each thread
    # One row: T1 T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4 (N = 12, BLOCK_SIZE = 4)
    # acc是[BLOCK_SIZE]逻辑向量，而非“每线程恰好一个标量”的保证。
    # 编者手算：N=10、BLOCK_SIZE=4、输入为1..10。
    # 三轮读[1,2,3,4]、[5,6,7,8]、[9,10,0,0]；
    # acc依次变为[1,2,3,4]、[6,8,10,12]、[15,18,10,12]，
    # 末尾sum得55。other=0是加法中性元，补齐不改变答案。
    # 循环把每隔BLOCK_SIZE的位置先局部相加，再对槽位求和；
    # 浮点加法不满足结合律，因此重排顺序可造成小的数值差异。
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over tiles
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # Final reduction from BLOCK_SIZE (all threads) to a scalar
    # 最终归约从逻辑向量得到标量；编译器可使用warp shuffle或shared memory。
    # 不能由Python变量acc断言其必定在共享内存，寄存器不够也可能spill到local memory。
    # 和第5讲在线softmax的联系：sum只需保存一个加法状态；
    # softmax若跨tile，需保存最大值m与l=Σexp(x-m)，不能直接求各tile概率再相加。
    # 新tile最大值m_t、归一化和l_t到来时，m'=max(m,m_t)，
    # l'=exp(m-m')l+exp(m_t-m')l_t；若要逐元素概率还需输出策略/再读输入。
    # 因此本row sum只是分块归约入门，没有实现完整在线softmax。
    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)


# 矩阵乘定义：A为[M,K]，B为[K,N]，C[m,n]=Σ_k A[m,k]B[k,n]。
# 同一A元素可被多个输出列重用，同一B元素可被多个输出行重用。
# 原文朴素“MKN reads”是数量级；逐输出直接读通常是2MKN个输入元素。
# 若每元素s字节，F约2MKN，输入字节约2sMKN，强度约1/s FLOP/byte。
# 方阵理想只读两输入并写一输出时，强度约2d³/(3sd²)=2d/(3s)，随d增长。
# 理想全矩阵常放不进片上内存，tiling用有限资源实现局部重用。
def triton_matmul_relu_example():
    text("Matrix multiplication is the bread and butter of deep learning.")
    a = torch.randn(1024, 1024, device=cuda_if_available())
    b = torch.randn(1024, 1024, device=cuda_if_available())
    c = naive_matmul_relu(a, b)

    text("How should we build a matmul kernel?")

    text("|        k                  n                     ", verbatim=True)
    text("|   [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", verbatim=True)
    text("| m [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", verbatim=True)
    text("|   [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", verbatim=True)

    text("**Naive approach:**")
    text("Fix any (m, n).")
    text("For each k:")
    text("- Read A[m, k] and B[k, n] from HBM.")
    text("- Multiply and accumulate.")
    text("Write result to C[m, n] in HBM.")

    text("Bottleneck: M K N reads, M N writes")
    text("Arithmetic intensity: O(1)")

    text("Computing C4 and C5 both need A4, A5, A6.")
    text("Can we read A4, A5, A6 from HBM once to compute both?")
    text("Answer: yes, using shared memory!")

    text("**Idealized approach:**")
    text("- Load all of A and B into shared memory, then compute C.")
    text("- Now we get M K + K N reads and M N writes.")
    text("- This yields the idealized O(N) arithmetic intensity from before.")
    text("- However, A and B are usually too large to fit in shared memory.")

    text("**Tiling:**")

    image("images/gemm_tiled.png", width=600)
    text("Key idea: divide the matrix C into output tiles (thread blocks).")
    text("Fix an output tile in C.")
    text("For each pair of (row tile of A, column tile of B):")
    text("- Load the corresponding A tile and B tile from HBM into shared memory.")
    text("- Perform matrix multiplication on the tiles.")
    text("- Accumulate into the partial sum (in shared memory).")
    text("Write output tile to HBM.")

    # 对一个BM×BN输出tile，每次K步读BM×BK和BK×BN个元素，
    # 做约2BM×BN×BK FLOPs；忽略输出写回时强度为
    # 2BM×BN/[s(BM+BN)]。BM=BN=T时为T/s，即O(T)。
    # BK增大会改变流水和临时资源，却在这项简化比值中约掉；不能说BK翻倍必加速。
    # 本例64×64×32、float32：读(2048+2048)×4=16384字节，
    # 做262144 FLOPs，单步强度16 FLOP/byte（未计最终写回/缓存等）。
    # 输出acc常分布在寄存器中，原文“partial sum in shared memory”为概念简化。
    # 更大BM/BN增加重用也扩大acc，须与寄存器压力、occupancy和并行tile数权衡。
    text("Arithmetic intensity: O(tile_size).")

    text("Bonus:")
    text("- Often, you want to apply an elementwise activation function.")
    text("- Example: GeLU(A @ B)")
    text("- Solution: kernel fusion!")

    text("**Implementation.**")

    text("Review: each matrix is linearized in memory")
    x = torch.tensor([[0., 1, 2, 3], [4, 5, 6, 7]])  # @inspect x
    # stride以元素为单位，表示对应维索引+1后地址前进多少元素。
    # 连续[2,4]矩阵stride=(4,1)，x[1,2]偏移1×4+2×1=6，值为6。
    # 转置后的视图stride=(1,4)，地址公式仍成立，但逻辑行列含义已改变。
    # 不要把shape当stride，也不要在指针表达式再额外乘dtype字节数。
    stride_row, stride_col = x.stride()  # @inspect stride_row stride_col
    row = 1
    col = 2
    index = row * stride_row + col * stride_col  # @inspect index

    # Compute c = a @ b
    c = triton_matmul_relu(a, b)


# 参考计算先完成所有K项求和，再ReLU=max(结果,0)。
# 不能每个K-tile后都ReLU：部分和-3和后续+2应得ReLU(-1)=0，
# 若先截断-3为0则最后变2，改变了数学结果。
def naive_matmul_relu(x: torch.Tensor, y: torch.Tensor):
    # Matmul followed by ReLU
    return torch.nn.functional.relu(x @ y)


# grid是二维输出tile网格，不是元素网格：ceil(M/64)×ceil(N/64)。
# 每个program写独立64×64输出区域，沿K循环，不同program无需原子累加。
# 原示例输入为默认float32；c的分配未显式指定dtype，使用PyTorch默认dtype。
# 因此这段包装器并非承诺任意dtype/device组合，不能照搬为通用matmul接口。
# 还要考虑tl.dot精度策略：输入float32不自动意味着全部乘法均按严格IEEE FP32。
# 具体TF32等行为依GPU、Triton版本及input_precision设置；需要匹配误差口径。[S8]
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    # A is M x K, B is K x N
    M, K = a.shape
    K, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device)

    # Determine grid
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,    # Compute c = a @ b
    M, N, K,                # a is M x K, b is K x N, c is M x N
    stride_am, stride_ak,   # How to navigate a
    stride_bk, stride_bn,   # How to navigate b
    stride_cm, stride_cn,   # How to navigate c
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 形状总表：indices_m=[BM]，indices_n=[BN]，indices_k=[BK]；
    # a_ptrs/a=[BM,BK]，b_ptrs/b=[BK,BN]，acc/c_ptrs=[BM,BN]。
    # pid_m,pid_n选择输出tile，不选择K-tile；K-tile由同一program内循环遍历。
    # 例如M=70,N=65,K=35，grid=(2,2)，共4个program，每个走2轮K。
    # 右下program起点(m,n)=(64,64)，仅6×1输出有效，其余被mask丢弃。
    # We are working on the (m, n)-th tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices of a [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Column indices of b [BLOCK_N]
    # 原英文行列说明颠倒：K索引实际对应A的列、B的行。
    # indices_k始终为0..BK-1，是tile内局部索引；循环变量k是全局K起点。
    indices_k = tl.arange(0, BLOCK_K)                    # Row indices of a = column indices of b [BLOCK_K]

    # Initial matrix of pointers of a and b
    # [:,None]把[BM]变[BM,1]，[None,:]把[BK]变[1,BK]，
    # 广播相加形成[BM,BK]地址表；不是把输入值做外积。
    # A地址为base+m*stride_am+k*stride_ak；B为base+k*stride_bk+n*stride_bn。
    # 连续A的stride为(K,1)，连续B为(N,1)。两者沿K前进的地址增量不同。
    # 小例A为2×3、B为3×2：A(1,2)偏移5，B(2,1)偏移5；
    # 虽这两个偏移碰巧相同，下一K项的增量分别是1与2。
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Move along row tiles of a, column tiles of b
    # 每轮a、b形状固定，dot把[BM,BK]@[BK,BN]变[BM,BN]，
    # 只归约本轮BK项；acc跨循环累加，最终才覆盖完整K。
    # 例如K=35,BK=32：第一轮处理0..31，第二轮32..34有效，其他补0。
    # mask分别保护M/K与K/N边界；零填充使额外乘积为0，不改变有效C。
    # 注意地址已由上一轮a_ptrs/b_ptrs递增移动，mask中的indices_k+k
    # 仅检查全局范围，不要再把k加进指针，否则会重复移动。
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak  # Advance to the next row tile of a
        b_ptrs += BLOCK_K * stride_bk  # Advance to the next column tile of b

    # Apply activation function (e.g., ReLU)
    # 全部K累加完才融合ReLU；无需把未激活C先写HBM再读回来。
    # 编者手算：A=[[1,2,3],[4,5,6]]，B=[[1,-1],[0,1],[2,-2]]；
    # AB=[[7,-5],[16,-11]]，ReLU后[[7,0],[16,0]]。
    # 若K按2+1分块，部分和分别[[1,1],[4,1]]与[[6,-6],[12,-12]]，
    # 相加恢复AB；这是“沿K累加、输出形状始终不变”的具体例子。
    acc = tl.maximum(acc, 0.0)

    # Write output tile
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))


############################################################

# 辅助闭包把输入准备移出被测操作：dim表示方阵边长，不是一维元素总数。
# 例如dim=16384有268435456个元素，单个float32张量约1GiB；
# 所以GeLU基准虽是逐元素操作，也可能有很大显存与运行成本。
def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    # Return a function to perform the operation
    return lambda : operation(x)


# 两个输入提前创建，闭包每次用相同输入再执行operation；
# 测到的缓存状态/分配路径可能与真实应用不同，报告时应注明测量范围。
def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    # Return a function to perform the operation
    return lambda : operation(x, y)


# 此处用torch.tanh，与手写Triton指数比值在极端输入的数值行为不同。
# “公式等价”与“浮点实现对所有值等价”是两回事。
def naive_gelu(x: torch.Tensor):
    # tanh approximation to the gelu activation function
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


# approximate="tanh"是重要语义设置；默认精确形式涉及高斯CDF，
# 比较实现时先对齐近似方式，再讨论速度和容差。
def builtin_gelu(x: torch.Tensor):
    # PyTorch's built-in GeLU with the tanh approximation
    return torch.nn.functional.gelu(x, approximate="tanh")


# dim=-1表示最后一维；本讲二维输入中即逐行归一化。
# softmax输出是概率而非log概率，不能直接当log_softmax使用。
def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


# 随机allclose属于有限样本数值检查，默认rtol也参与判定，
# 并不证明任意输入正确。边界尺寸、非连续stride、极端值应分别审视。
# 本次不运行这些GPU检查；新增手算只用于核对注释中的数学与形状。
def check_equal_1d(f1, f2):
    x = torch.randn(2048, device=cuda_if_available())
    y1 = f1(x)  # @stepover
    y2 = f2(x)  # @stepover
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d(f1, f2):
    x = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


# 该比较辅助函数虽已定义，但matmul演示没有调用它。
# 不能因文件里存在check_equal便声称自写matmul已经通过数值验收。
def check_equal_2d_2d(f1, f2):
    x1 = torch.randn(2048, 2048, device=cuda_if_available())
    x2 = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x1, x2)
    y2 = f2(x1, x2)
    assert torch.allclose(y1, y2, atol=1e-6)


# mean保留输入单位，因此benchmark返回仍为毫秒；换成秒要除以1000。
# 自测1：N=2050、BLOCK_SIZE=1024需要几个program？答3个，末个仅2槽有效。
# 自测2：softmax补齐槽为何不用0？答它会参与exp求和，-inf才贡献0。
# 自测3：row sum循环能直接改成对每块softmax求和吗？答不能，分母/最大值需统一。
# 自测4：matmul增大tile为何可能变慢？答acc及临时值增大可能压低并发或spill。
# 自测5：本benchmark可报告首次编译耗时吗？答不能，它旨在warmup后的事件计时；
# 首次编译与完整请求延迟需另定测量范围，不能混用。
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

    
# PTX适合核对global load/store、线程索引与指令结构；
# 真实性能仍需设备测量，资源数/机器指令可用编译信息和专门工具进一步检查。
# 本次交付只做注释静态一致性与CPU手算，不生成PTX或运行GPU性能实验。
def output_ptx(name: str, kernel):
    """Print out the PTX code generated by Triton for the given `kernel`."""
    ptx_path = f"var/{name}-ptx.txt"
    with open(ptx_path, "w") as f:
        ptx = kernel.asm["ptx"]
        f.write(ptx)


if __name__ == "__main__":
    main()
