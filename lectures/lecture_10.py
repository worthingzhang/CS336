# 第 10 讲中文自学导读：推理的资源账本、近似加速与精确采样（Spring 2026）
# 学习目标：能分别估算 prefill/decode 的 FLOPs、HBM 流量与缓存容量；能从
# batch、上下文长度、KV 头数推导时延/吞吐变化；能证明推测采样保持目标分布，
# 并解释 continuous batching 与 PagedAttention 各自解决的服务问题。
# 先修：矩阵乘法与张量形状、因果注意力、softmax/条件概率、GPU 算力与带宽。
# 阅读主线：先看 landscape 的指标，再亲算两类算子的资源账，随后理解 GQA 等
# 如何改变账本；最后区分“改变模型”的压缩与“保持目标分布”的推测采样。
# 本文件是可执行讲义：text/image/link 构造原课程内容，@inspect 展示中间量。
# 新增内容均为相邻 Python 注释；不运行整讲也可沿函数顺序学习、手算自测。
# 以下无“课堂补充”标记的推导、数值例子与纠错说明均为编者补充；英文原稿保留。
# 来源索引（2026-10-15 核验；论文实验不是所有模型/硬件上的保证）：
# [课] https://cs336.stanford.edu/ 与 https://github.com/stanford-cs336/lectures
# [字幕] https://jingzechen.github.io/assets/courses/stanford-cs336-s26/lectures/010/captions.source.srt
#   本地缓存 lecture_10.srt；第三方保存字幕，可能误识别术语，不作为教师审校逐字稿。
# [账本] https://jax-ml.github.io/scaling-book/inference/ （原稿引用的技术说明）
# [GQA] https://arxiv.org/abs/2305.13245
# [MLA] https://arxiv.org/abs/2405.04434
# [CLA] https://arxiv.org/abs/2405.12981
# [AWQ] https://arxiv.org/abs/2306.00978
# [剪枝] https://arxiv.org/abs/2407.14679
# [推测] https://arxiv.org/abs/2211.17192 与 https://arxiv.org/abs/2302.01318
# [Orca] https://www.usenix.org/conference/osdi22/presentation/yu
# [分页] https://arxiv.org/abs/2309.06180
# 来源的具体适用范围、原稿疑点及课堂时间段见对应段落；不将年份不同的讲次混用。
from dataclasses import dataclass

from sympy import symbols, oo
from edtrace import text, link, image
from lecture_util import article_link
from references import Reference, gqa_2023, mla_2024, longformer_2020, sparse_transformer_2019, mistral_7b_2023, deepseek_v4_2026

# Define symbols corresponding to the shape of the Transformer model
# 符号表：B=同时处理的请求数，S=被查询的历史/键值长度，T=本次处理的 query
# 位置数，D=隐藏宽度，F=MLP 中间宽度，N=Q 头数，K=KV 头数，H=单头宽度，
# L=层数，V=词表大小。G=N/K 是每个 KV 头服务的 Q 头数，后文以文字出现。
# 注意 K 在算法图片里还被用作“草稿长度”；中文采样说明改记 γ，以免混淆。
# sympy 的正数符号用于代数推导，非真实张量；subs/limit 都是 CPU 符号计算。
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # Just a constant that helps with taking limits
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="Scaling book chapter on Transformers", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="Scaling book chapter on inference", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    # 贯穿全讲的分类：GQA/量化/剪枝改变模型表示，必须重新评估质量；精确推测
    # 采样保留给定目标模型的分布；批处理/分页改变调度与存储布局。
    # 减少计算量、减少要搬的数据、减少空闲/碎片是不同途径，不能只比较 FLOPs。
    text("## Lecture 10: inference")
    image("images/inference-schema.png", width=600)

    text("### Understanding the inference workload")
    landscape()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    text("### Taking shortcuts (lossy)")
    reduce_kv_cache_size()
    quantization()
    model_pruning()

    text("Summary: reduce inference complexity without hurting accuracy")

    text("From scratch recipe:")
    text("1. Define faster model architecture")
    text("2. Train faster model")

    text("Distillation recipe:")
    text("1. Define faster model architecture")
    text("2. Initialize weights using original model (which has a different architecture)")
    text("3. Repair faster model (distillation)")

    text("### Use shortcuts but double check (lossless)")
    speculative_sampling()

    text("### Handling dynamic workloads")
    text("Batching over sequences in live traffic is tricky because:")
    text("1. Requests arrive at different times (waiting for batch is bad for early requests)")
    text("2. Sequences have shared prefixes (e.g., system prompts, generating multiple samples)")
    text("3. Sequences have different lengths (padding is inefficient)")

    continuous_batching()
    paged_attention()

    text("### Summary")
    text("- Inference is important (actual use, evaluation, reinforcement learning)")
    text("- Different characteristics compared to training (memory-bound, dynamic)")
    text("- Techniques: new architectures, quantization, pruning/distillation, speculative sampling")
    text("- Ideas from systems (speculative execution, paging)")
    text("- New architectures have huge potential for improvement")


def landscape():
    # 推理既用于产品，也用于评测和 RL 的 rollout；训练后反复调用，使每次调用
    # 节约的成本累积起来。下面行业 token 数是讲授时引用的数据，非实时统计；
    # 训练 token 与推理 token 的成本也不能直接相等，后者取决于上下文与批大小。
    # 软件列表帮助定位系统层：模型本身定义概率，服务引擎负责缓存、调度和内核。
    text("Inference shows up in many places:")
    text("- Actual use (chatbots, code completion, agents, batch data processing)")
    text("- Model evaluation (e.g., on instruction following)")
    text("- Reinforcement learning (sample many generations, then apply score)")

    text("Why **efficiency** matters: training is one-time cost, inference is repeated many times")
    text("- OpenAI processes ~8.6T tokens per day "), article_link("https://www.pymnts.com/artificial-intelligence-2/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/")
    text("- For reference, DeepSeek v4 was trained on 32T tokens "), link(deepseek_v4_2026)
    
    text("Moreover:")
    text("- Chatbots: most tokens are meant for human consumption (humans are bottleneck)")
    text("- Agents: query → internal trace → output for human (number of tokens generated can grow unbounded)")
    text("- Tokens generated = compute spent")

    text("Companies doing inference (a big deal for anyone who has a product or platform):")
    text("- Providers serving closed models (OpenAI, Anthropic, Google, etc.)")
    text("- Providers serving open-weight models (Together, Fireworks, Baseten, DeepInfra, Groq, Cerebras, etc.)")

    text("Open-source packages:")
    text("- vLLM: from Berkeley, pioneered PagedAttention, popular and good default "), link(title="GitHub", url="https://github.com/vllm-project/vllm")
    text("- SGLang: from Berkeley, pioneered RadixAttention, good for agentic workloads "), link(title="project", url="https://sgl-project.github.io/")
    text("- TensorRT-LLM: from NVIDIA, highly optimized for GPUs "), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    text("- llama.cpp: C++ only, supports CPU inference, runs locally "), link(title="GitHub", url="https://github.com/ggml-org/llama.cpp")

    text("Inference is huge. Important to make it fast.")

    text("What does \"fast\" mean (metrics)?")
    # TTFT 从请求到达到第一个输出 token；线上还包括排队、分词、网络与调度。
    # TPOT（time per output token）通常是首 token 后相邻输出的平均间隔：
    # 若输出 n>1 个 token，则 TPOT≈(最后 token 时间-首 token 时间)/(n-1)。
    # 总完成时间≈TTFT+(n-1)*TPOT；吞吐量是服务端单位时间所有请求的输出总数。
    # 手算：TTFT=.20s、100 个输出、TPOT=.02s，总用时≈2.18s；若稳定批 B=16
    # 每 .02s 各产 1 token，吞吐是 800 token/s，而单请求仍是 50 token/s。
    # 线上长尾要另看 p95/p99，均值会掩盖等待；输入吞吐、输出吞吐不要混计。
    text("- Time-to-first-token (TTFT): how long user waits before any generation happens (for interactive applications)")
    text("- Latency (seconds/token): how fast tokens appear for *one* query (for interactive applications)")
    text("- Throughput (tokens/second): how fast tokens appear for *many* queries (for batch processing)")

    text("What governs efficiency?")
    # Teacher forcing 时已知整段真实输入，因果 mask 让所有位置同时预测下一词；
    # 生成时第 t+1 个输入依赖第 t 个随机样本，通常不能预先把未来输入交给大模型。
    # 这是“位置之间的数据依赖”，并不禁止同一步内的头/矩阵/请求并行。
    text("- Training (supervised): you see all tokens, can parallelize over sequence (matmul in Transformer)")
    text("- Inference: you have to generate sequentially, can't parallelize over generation, so harder to fully utilize compute")


def review_transformer():
    # 读形状的诀窍是写求和指标：Y[b,t,h]=Σ_d X[b,t,d]W[d,h]。
    # 因而 BT D × D H → BT H；两个 B 不收缩表示不同请求各算各的。
    # 例如 X:[2,3,4]、W:[4,5] 得 Y:[2,3,5]，共 2*3*4*5 次乘加。
    # N 个注意力头在同一 token 上观察不同投影；GQA 仅共享 K/V，Q 仍有 N 个。
    link(scaling_book_transformers)
    text("Notation (similar to einops):")
    text("- Symbols denote dimensions (and their length): B (batch), T (sequence), D (model dim), H (head dim)")
    text("- Example: BT<font color=\"red\">D</font> x <font color=\"red\">D</font>H → BTH")
    text("- <font color=\"red\">Contracting (red)</font> dimensions appear in both operands and disappear from result")
    text("- Regular (black) dimensions appear in one operand and stay in result")
    text("- Example: <font color=\"blue\">B</font><font color=\"red\">D</font> x <font color=\"blue\">B</font><font color=\"red\">D</font> → B")
    text("- <font color=\"blue\">Batching (blue)</font> dimensions appear in both operands and stay in result")

    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=800)
    text("Conventions:")
    # F=4D 只是这一复习段的常见简化。后面的 gated MLP 有 up/gate/down 三个
    # 矩阵，Llama 配置实际 F=13824、D=5120，不能再代入 F=4D。
    # S=T 指同一段自注意力的 query/key 位置数相等；标签相对输入错一位，
    # 不是说训练能够看到未来 token。decode 中 T=1，S 随已经处理的前缀增长。
    text("- F = 4 D (MLP up-projects into 4x the model dimension)")
    text("- D = N H (model dimension split across N heads)")
    text("- N = K G (for GQA, number of heads split across K groups)")
    text("- S = T (during training, condition on S input tokens to predict T output tokens)")


def review_of_arithmetic_intensity():
    # 资源账采用一次乘加=2 FLOPs、bf16 每元素=2 bytes；HBM 是 GPU 大容量显存，
    # 算术单元需搬入数据才能运算。这里假设矩阵每元素理想地只搬一次，不算
    # tile 重读、缓存命中、内核启动、通信与非矩阵算子，得到用于判断瓶颈的模型。
    text("Setup: multiply X <font color=\"gray\">(B x D)</font> and W <font color=\"gray\">(D x F)</font> matrix")
    text("Intuition: B is batch size, D is hidden dimension, F is up-projection dimension in MLP")

    text("Let's do FLOPs and memory read/write accounting for the matrix multiplication (X * W).")
    flops = 0
    bytes_transferred = 0

    # Perform the operation
    text("1. Read X <font color=\"gray\">(B x D)</font> from HBM")
    bytes_transferred += 2*B*D   # 2 bytes for bf16
    text("2. Read W <font color=\"gray\">(D x F)</font> from HBM")
    bytes_transferred += 2*D*F
    text("3. Compute Y = X <font color=\"gray\">(B x D)</font> @ W <font color=\"gray\">(D x F)</font>")
    flops += 2*B*D*F
    text("4. Write Y <font color=\"gray\">(B x F)</font> to HBM")
    bytes_transferred += 2*B*F

    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F

    text("Recall that **arithmetic intensity** is how much compute we do per byte transferred (want to be high).")
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    # I=BDF/(BD+DF+BF)，单位 FLOP/byte。B≪D,F 时分母被权重 DF 主导，
    # 每读一个权重被 B 个输入重用，因此 I≈B，而非“batch 变大就少算 FLOPs”。
    # 编者手算：B=64,D=5120,F=13824，I≈62.922 FLOP/byte，略小于近似值 64。

    text("Assuming B is much less than D and F, then we can simplify:")
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    text("Accelerator intensity of H100:")
    flops_per_second = 989e12
    memory_bandwidth = 3.35e12
    accelerator_intensity = flops_per_second / memory_bandwidth  # @inspect accelerator_intensity
    # Roofline：t_compute=FLOPs/峰值算力，t_IO=bytes/峰值带宽；理想重叠下
    # t≥max(t_compute,t_IO)。两者相等时 I=989e12/3.35e12≈295.224。
    # 下文 iff B>295 只在 I≈B 和给定 H100 精度/峰值假设下成立；实际效率、
    # 输入形状和融合会改变交点。“compute-bound 好”指算力较易充分利用，
    # 不代表时延必小：一个巨大的矩阵乘法可以很慢但仍受算力限制。
    assert round(accelerator_intensity) == 295

    text("If computation intensity > accelerator intensity, **compute-bound** (good)")
    text("If computation intensity < accelerator intensity, **memory-bound** (bad)")
    text("Conclusion: compute-bound iff B > 295")

    text("Extreme case (B = 1, corresponding to matrix-vector product):")
    text("- Arithmetic intensity: 1")
    text("- Memory-bound (read D x F matrix, perform only 2 D F FLOPs)")
    text("- This is basically what happens with inference...")


def arithmetic_intensity_of_inference():
    # 先建立时间轴：prefill 处理完整 prompt、写缓存并产首 token 的分布；
    # 随后的 decode 将已采样 token 送入，追加该 token 的 K/V，产下一个分布。
    # 因果 Transformer 中过去位置看不到新 token，所以各层过去的 K/V 不会变；
    # Q 只服务当前查询，不需保留整个历史的 Q。这也是缓存能精确复用的原因。
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=800)
    text("Naive inference: to generate each token, feed history into Transformer")
    text("Complexity: generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2))")
    # 此处 T 暂表示“累计生成长度”，不是下方单次调用的 query 数。忽略固定 prompt，
    # 全量重算的注意力成本 Σ_{t=1}^T O(t²D)=O(T³D)，MLP 为 Σ O(tDF)=O(T²DF)。
    # 缓存后注意力为 Σ O(tD)=O(T²D)，MLP 为 O(TDF)；减少的是历史重复计算，
    # 并未把精确全局注意力变为总 O(T) 算法。有长 prompt S0 时还含 O(S0*T*D)。

    text("Observation: a lot of the work can be shared across prefixes")
    text("Solution: store **KV cache** in HBM")
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=800)
    text("KV cache: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector")
    # 必须同时存 key 与 value 两个向量：bf16 缓存= B*S*L*K*H*2*2 bytes。
    # 这是每请求独有的激活状态，不是全请求共享的模型参数，也不是训练反向的
    # 激活检查点。一般请求长度各异时，以 Σ_b S_b 替代 B*S。

    text("Two stages of inference:")
    text("1. **Prefill**: given a prompt, encode into vectors (parallelizable like in training)")
    text("2. **Generation**: generate new response tokens (sequential)")

    text("Let's compute the FLOPs and memory IO for both the MLP and attention layers.")
    text("S is the number of tokens we're conditioning on, T is the number of tokens we're generating.")
    text("Later, we'll specialize to prefill (T = S) and generation (T = 1).")

    text("### MLP layers (only looking at the matrix multiplications)")
    # 一个位置的 gated MLP：U=XWup、G=XWgate，输出=(GeLU(G)⊙U)Wdown。
    # X:[B,T,D] 可展平成 [BT,D]；前两次各得 [BT,F]，第三次回到 [BT,D]。
    # 三个矩阵均被所有 BT 个位置复用，故矩阵 FLOPs 共 3*(2BTDF)=6BTDF。
    # 原稿 IO 账是示意约定：读 X 一次、读权重一次、写 U/G/Y，不完整列出
    # 非融合实现重读 U/G/激活结果的流量；真正流量依赖融合与片上缓存。
    # 因此下式应当用来理解权重主导的极限，不能当成所有内核的实测字节数。
    flops = 0
    bytes_transferred = 0
    
    # Perform the operation
    text("1. Read X <font color=\"gray\">(B x T x D)</font> from HBM")
    bytes_transferred += 2*B*T*D
    text("2. Read Wup <font color=\"gray\">(D x F)</font>, Wgate <font color=\"gray\">(D x F)</font>, Wdown <font color=\"gray\">(F x D)</font> from HBM")
    bytes_transferred += 3 * 2*D*F
    text("3. Compute U = X <font color=\"gray\">(B x T x D)</font> @ Wup <font color=\"gray\">(D x F)</font>")
    flops += 2*B*T*D*F
    text("4. Write U <font color=\"gray\">(B x T x F)</font> to HBM")
    bytes_transferred += 2*B*T*F
    text("5. Compute G = X <font color=\"gray\">(B x T x D)</font> @ Wgate <font color=\"gray\">(D x F)</font>")
    flops += 2*B*T*D*F
    text("6. Write G <font color=\"gray\">(B x T x F)</font> to HBM")
    bytes_transferred += 2*B*T*F
    text("7. Compute Y = GeLU(G)*U <font color=\"gray\">(B x T x F)</font> @ Wdown <font color=\"gray\">(F x D)</font>")
    flops += 2*B*T*D*F
    text("8. Write Y <font color=\"gray\">(B x T x D)</font> to HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F

    # Compute the arithmetic intensity
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    text("Assume that B*T is much smaller than D and F.")
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    text("For the two stages:")
    # 编者阶段对照（本 MLP）：prefill 用 T=S，有 6BSDF FLOPs；decode 用 T=1，
    # 有 6BDF FLOPs；两者都读同样的 6DF 权重字节，前者以更多位置摊薄读取。
    # 例如 B=1,S=1024,D=5120,F=13824，代完整 IO 式，prefill I≈865.804，
    # decode I≈.999822 FLOP/byte。长 prefill 的 I≈BS 已有明显近似误差，
    # 但仍比 295 高；“prefill 受算力限制”是长输入常见情况而非无条件定理。
    text("1. Prefill: easy to make compute-bound (good) by making `B*T` large enough (large batches, long sequences)")
    text("2. Generation: two problems")
    text("- Generating one token at a time (T = 1)")
    text("- B is number of concurrent requests, unpredictable for interactive applications")

    text("### Attention layers (focusing on the matrix multiplications with FlashAttention)")
    # 这一段先按 MHA（K=N、D=NH）核算，不含 Q/K/V/O 的线性投影；投影属于
    # 前面那类共享权重矩阵乘法。实际逐头为 Q:[B,N,T,H]、K/V:[B,N,S,H]，
    # A=QK^T:[B,N,T,S]，softmax 沿 S 轴归一化，AV:[B,N,T,H] 再合并为 D。
    # 下方压平的 Q @ K 写法省略了 K 的转置和头轴，并非两个原形状直接相乘。
    # 两次乘法各 2BNTSH FLOPs；FlashAttention 避免把完整 A 写回 HBM，
    # 不消除 QK^T 的计算。理想 IO=读 Q/Y 写回 4BTD + 读 K/V 4BSD。
    # 这是理想复用模型，实际 tile 可能多次读取 K/V；因果三角 mask 还能改变
    # prefill 的 FLOPs 常数，下文公式按完整 S×T 矩形估算，忽略 softmax 等。
    text("- S is number of previous tokens (already generated)")
    text("- T is number of next tokens (to generate logits for)")
    flops = 0
    bytes_transferred = 0
    
    # Perform the operation
    text("1. Read Q <font color=\"gray\">(B x T x D)</font>, K <font color=\"gray\">(B x S x D)</font>, V <font color=\"gray\">(B x S x D)</font> from HBM")
    bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
    text("2. Compute A = Q <font color=\"gray\">(B x T x D)</font> @ K <font color=\"gray\">(B x S x D)</font>")
    flops += 2*B*S*T*D
    text("3. Compute Y = softmax(A) <font color=\"gray\">(B x S x T x K x G)</font> @ V <font color=\"gray\">(B x S x K x H)</font>")
    flops += 2*B*S*T*D
    text("4. Write Y <font color=\"gray\">(B x T x D)</font> to HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D

    # Compute the arithmetic intensity
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)
    # 分子 4BSTD 除以分母 4BD(S+T)，B 与 D 同时消去。prefill S=T 时
    # I=S/2；decode T=1 时 I=S/(S+1)，S=1024 时仅 .999024 FLOP/byte。
    # 以 B=1,S=1024,D=5120 为例：prefill 注意力约 21.475 GFLOPs/41.943 MB；
    # decode 仅 20.972 MFLOPs/20.992 MB。每层都是同一 KV 历史，算的 query
    # 数却从 1024 变为 1，所以少算很多但仍要搬历史。这与上面的 MLP 是两笔账。

    text("For the two stages:")
    text("1. Prefill: T = S")
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # Good!
    text("2. Generation: T = 1")
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # Bad!

    text("Unlike MLPs, no dependence on B, so batching doesn't help!")
    text("Why?")
    text("- In MLP layers, every sequence hits the same MLP weights (Wup, Wgate, Wdown don't depend on B)")
    text("- In attention layers, every sequence has its own KV cache vectors (Q, K, V all depend on B)")

    text("Summary:")
    text("- Prefill is compute-bound, generation is memory-bound")
    text("- Prefill MLP intensity: `B*S`")
    text("- Prefill attention intensity: `S/2`")
    text("- Generation MLP intensity: `B` (requires concurrent requests)")
    text("- Generation attention intensity: `<1` (impossible to improve)")
    # “impossible”限于当前 MHA、无跨请求缓存复用、bf16、单 query 的模型。
    # GQA 可让 G=N/K 个 Q 头复用一个 KV 头。理想共享时注意力
    # I=4BSTNH/(4BSKH+4BTNH)=STN/(SK+TN)，长历史 decode 约为 N/K。
    # B 仍会约掉；它不提高此项的理想强度，但可能改善真实内核占用率。
    # 自测：B 加倍会让 attention 的历史字节翻倍吗？答：独立请求会，FLOPs 也
    # 翻倍，故强度不变；MLP 权重不翻倍，故小 B 时批处理能提高 MLP 的强度。


@dataclass(frozen=True)
class TransformerPerformanceStats:
    # 阅读提醒：下方原 docstring 把 num_params 写成 in bytes，但函数实际返回
    # 参数“个数”；memory 才是 bytes。保留原文，以实际公式和这里的单位为准。
    # 虽类型标注 int/float，替换完全部符号前仍是 SymPy 表达式；frozen 表示
    # substitute 返回新对象，不原地改变旧配置，便于公平比较不同 B/K。
    """
    Performance stats of a Transformer:
    - num_params: number of parameters (in bytes)
    - memory: total memory usage (parameters + KV cache) in bytes
    - latency: time to generate one token (seconds/token)
    - throughput: tokens generated per second
    """
    num_params: int
    memory: int
    latency: float
    throughput: float

    def substitute(self, key, value):
        """Substitute `key` with `value` in all stats."""
        return TransformerPerformanceStats(
            self.num_params.subs(key, value).simplify(),
            self.memory.subs(key, value).simplify(),
            self.latency.subs(key, value).simplify(),
            self.throughput.subs(key, value).simplify(),
        )


def compute_transformer_performance_stats(config) -> TransformerPerformanceStats:  # @inspect config
    """Compute various performance stats for the Transformer given `config`."""

    # Number of parameters in the Transformer
    # 逐项拆开：2VD 是不共享的输入 embedding/输出词表矩阵；3DFL 是 gated MLP；
    # 每层 Q/O 各 DNH、K/V 各 DKH，总共 (2DNH+2DKH)L。若绑定 embedding，
    # 第一项应变成 VD；这里忽略 norm/bias 等小项，不能把它当精确 checkpoint 大小。
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L

    # How much memory the parameters take
    parameter_size = 2*num_params  # 2 for bf16 (training requires a larger multiple)
    
    # How much the KV cache takes per sequence (S tokens, K heads, H head dim, L layers)
    # KV_cache/seq = S*K*H*L*(key+value)*bytes/element。
    # 在默认 K=40、H=128、L=40 下，每新增 1 token 占 819200 bytes；S=1024
    # 时每请求 .8388608 GB（.78125 GiB）。GB=10^9 bytes，GiB=2^30 bytes。
    # 缓存随 B、S 线性增长；权重容量与 B、S 无关。不要把两者同除以 GQA 分组数。
    kv_cache_size_per_seq = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

    # Total memory usage
    memory = B * kv_cache_size_per_seq + parameter_size

    # *Latency* is determined by memory IO (read all parameters and KV cache for each step)
    # 这里把常驻的“权重+KV 容量”当成每步必须读取的字节量，是带宽受限 decode
    # 的简化估算；不是一般意义下内存占用=流量。未计新增 KV 写回、临时缓冲、
    # logits、内核启动、张量并行通信、排队，也假设读全部参数（embedding 等
    # 的实际访问更细）。所以此值是该模型的理想时延估算，不是测量值或严格全机界。
    # 若算力成本不再可忽略，可按层/算子求 max(FLOPs/算力,bytes/带宽) 后累加，
    # 不能把计算慢的 prefill 也直接代入这条 decode 公式。
    latency = memory / memory_bandwidth

    # *Throughput* is the inverse of latency, but we're generating B tokens in parallel
    throughput = B / latency

    # Substitute config
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return TransformerPerformanceStats(num_params, memory, latency, throughput)


def llama2_13b_config(args={}):
    return {
        S: 1024,   # Sequence length
        D: 5120,   # Model dim
        F: 13824,  # Feed-forward dim
        N: 40,     # Number of query heads
        K: 40,     # Number of key/value heads
        H: 128,    # Head dimension
        L: 40,     # Number of layers
        V: 32000,  # Vocabulary size
        memory_bandwidth: 3.35e12,  # Memory bandwidth (bytes/second)
        **args
    }


def throughput_and_latency():
    # 令 W=权重字节，C=单请求 KV 字节，β=带宽，则一步 t(B)=(W+BC)/β，
    # 单请求每步得一个 token，批吞吐 R(B)=Bβ/(W+BC)。故 dt/dB=C/β>0，
    # dR/dB=βW/(W+BC)²>0，且 B→∞ 时 R→β/C：这解释收益递减。
    # 该极限只属于忽略计算和容量约束的模型；真实服务通常先触及显存或算力上限。
    text("So we have shown that inference is memory-bound.")
    text("Let us now compute the theoretical maximum latency and throughput of a single request.")
    # 原文 maximum latency 易误读：满带宽、无开销的除法得到的是乐观的低时延
    # 估算和高吞吐估算，并不是“最大时延”。交互请求也可能因排队远远更慢。
    text("Assumption: can overlap compute and communication perfectly and ignore overhead.")

    text("Instantiate latency and throughput for Llama 2 13B on an H100:")
    config = llama2_13b_config()
    stats = compute_transformer_performance_stats(config)

    # Batch size 1
    b1 = stats.substitute(B, 1)  # @inspect b1 @stepover

    # Batch size 64
    b64 = stats.substitute(B, 64)  # @inspect b64 @stepover
    text("Result: worse latency, better throughput")

    # Batch size 256
    b256 = stats.substitute(B, 256)  # @inspect b256 @stepover
    # 编者复算（本文件 S=1024、K=40、bf16、β=3.35e12；全为十进制 GB）：
    # B=1：总容量26.869760 GB，t=8.020824 ms，R=124.675 token/s。
    # B=64：总容量79.7179904 GB，t=23.796415 ms，R=2689.481 token/s。
    # B=256：总容量240.779264 GB，t=71.874407 ms，R=3561.769 token/s。
    # 第三行超出下方 80e9 上限，因此“吞吐”只是公式外推，单卡实际无法运行。
    # 第二行也只余 .2820096 GB，忽略运行时开销的“能装下”不构成部署可行证明。
    text("Result: even worse latency, even better throughput")
    h100_memory = 80e9  # H100 memory in bytes
    assert b256.memory > h100_memory  # Doesn't fit in memory!
    text("Result: doesn't fit into memory and throughput gains are diminishing too...")

    text("What increasing batch size does:")
    text("- Worsens latency because larger KV cache (O(B) size) to read/write")
    text("- Improves throughput because amortizes the cost of reading parameters")

    text("**Tradeoff** between latency and throughput:")
    text("1. Smaller batch sizes yield better latency but worse throughput")
    text("2. Larger batch sizes yield better throughput but worse latency")

    text("Easy parallelism: if you launch M copies of the model, latency is the same, throughput increases by M!")
    # M 份副本需要 M 份硬件/权重容量并有充足流量，理想均匀分流才近似 M 倍。
    # 将同一模型切到多卡会减少每卡存储，却引入 collective 通信；此时瓶颈账本
    # 还要加入互联带宽和同步时间，不能把单卡带宽简单乘卡数就断言实测速率。
    text("Harder parallelism: shard the model and the KV cache "), link(scaling_book_inference)

    text("Note: time-to-first-token (TTFT) is essentially a function of prefill time")
    text("Use smaller batch sizes during prefill for faster TTFT")
    text("Use larger batch sizes during generation to improve throughput")
    # 自测：要减少 TTFT 是否总应扩大 B？答：否。长 prompt 的 prefill 已有
    # 大量 token 并行，再等凑批会增加队列和服务时间；decode 扩批则可摊权重。
    # 混合服务可将长 prefill 分块，避免它一次占满设备使已有请求的 TPOT 抖动；
    # 这是调度取舍，实际需按时延目标和流量测试，不由单个峰值吞吐数字决定。


def reduce_kv_cache_size():
    # 观察 B*S*L*K*H*4 这个乘积：GQA 减 K，MLA 压缩每位置表示，CLA 共享层，
    # 滑窗限制 S，量化减少每数的字节。都要分别检查质量与实际内核能否利用节省。
    # 这些属于架构/表示变化，不能把已训练 MHA 的 config 改小便当作同一个模型。
    text("Recall that memory is the bottleneck for inference.")
    text("So let's try to reduce the size of the KV cache")
    text("...but make sure we don't lose too much accuracy.")

    text("### Grouped-query attention (GQA) "), link(gqa_2023)
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=800)
    text("Idea: N query heads, but only K key and value heads, each interacting with N/K query heads")
    text("Multi-headed attention (MHA): K=N")
    text("Multi-query attention (MQA): K=1")
    text("Group-query attention (GQA): K is somewhere in between")
    # 具体形状：N=40,K=8 时 G=5，Q reshape 为 [B,T,8,5,128]，K/V 为
    # [B,S,8,128]；每组 5 个不同 Q 各有不同注意力分数，但共享同一个 K/V。
    # 不需要把共享 K/V 物理复制 5 份，否则缓存节省被抵消。Q/O 投影仍保留。
    # 图中 GQA groups 表示 KV 组数，横轴向右意味着更接近 MHA、时延更高；
    # 组数少的点接近 MQA。具体时间来自该论文实验，不能移植为 H100 预测。

    text("Latency/throughput improves: "), link(gqa_2023)
    image("images/gqa-speed.png", width=500)

    text("Why does GQA improve latency and throughput?")
    text("GQA reduces the KV cache by a factor of N/K.")
    text("Reminder: reducing memory usage leads to speedup (since we're memory-bound).")

    # Original Llama 2 13B (MHA)
    config = llama2_13b_config({K: 40, B: 64})  # @stepover
    k40_b64 = compute_transformer_performance_stats(config)  # @inspect k40_b64 @stepover

    # GQA with 1:5 ratio (K:N)
    config = llama2_13b_config({K: 8, B: 64})  # Use GQA with 1:5 ratio @stepover
    k8_b64 = compute_transformer_performance_stats(config)  # @inspect k8_b64 @stepover
    text("Result: Worse latency, but better throughput (and it fits in memory now!)")
    # 【原稿疑点，保留英文】固定 B=64，代本函数实际公式得到下表：
    # K  参数个数          权重 GB       批 KV GB       总 GB         ms/step    token/s
    # 40 13,015,449,600    26.0308992    53.6870912     79.7179904    23.796415  2689.481
    #  8 11,337,728,000    22.6754560    10.73741824    33.41287424    9.973992  6416.688
    # 复算路径：K=8 的单请求 KV=1024*8*128*40*4=167772160 bytes；
    # 总量=22675456000+64*167772160=33412874240；除 3.35e12 得秒，64/t 得吞吐。
    # 所以这里应该读成“时延改善、吞吐改善”，而非 Worse latency。
    # 只有 KV 缓存变成 1/5；K/V 投影参数也减少，Q/O、MLP、embedding 没有变，
    # 总容量并非缩到 1/5。权重差为 2 bytes*2D*(40-8)*H*L=3.3554432 GB。
    # “now fits”也需限定：在原稿 80e9 纯容量上限下，K=40,B=64 已勉强装下；
    # 此处 GQA 提供了大幅余量，不能据此断言实测前者可运行或曾经 OOM。
    # 课堂补充 · 00:49:27–00:49:59：字幕明确说固定 batch 缩小 KV 后两者都改善；
    # 00:50:15–00:50:40 才是增大 batch 后时延变差且能够装入显存的比较。

    # Now we can increase the batch size
    config = llama2_13b_config({K: 8, B: 256})  # Increase batch size @stepover
    k8_b256 = compute_transformer_performance_stats(config)  # @inspect k8_b256 @stepover
    text("Result: Worse latency, but better throughput (and still fits in memory!)")
    # 这一次固定 K=8、B 从 64→256，原句方向与公式一致：批 KV=42.94967296 GB，
    # 总容量65.62512896 GB，t=19.589591 ms，R=13068.165 token/s。
    # 与 K=40,B=256 的240.779264 GB相比，容量模型从超限变为80 GB以内；
    # 与 K=8,B=64相比，吞吐只增约2.04倍而非4倍，因为额外缓存也需要搬运。

    text("Check that accuracy doesn't drop: "), link(gqa_2023)
    image("images/gqa-accuracy.png", width=800)
    # 读图同时比较时间列与任务分数：表中 MHA-XXL 平均47.2、1.51s，GQA-8-XXL
    # 平均47.1、.28s；这是接近质量下的经验加速，不能推出每个任务严格无损。
    # [GQA] 可从 MHA 权重聚合各组 K/V 后继续训练（uptraining）；转换/训练成本
    # 及任务分布必须计入评估。这里的 Llama 改 K 只是资源模型，不是实际转换实验。

    text("### Multi-head latent attention (MLA) "), link(mla_2024)
    image("images/mla-schema.png", width=800)
    text("Normal attention: KV cache consists of K = W_K h, V = W_V h (N*H dimensions)")
    text("MLA: store compressed vector c = W_c h (C dimensions), project up to K = W_K c, V = W_V c when needed")
    text("DeepSeek v2: reduce N*H = 16384 to C = 512")
    text("Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions")
    # 读 MLA 图要看斜线“cached”：MHA/GQA 存显式 K 和 V，MLA 只存共享潜向量
    # c 以及解耦位置编码的 key。c 同时服务 K/V，512+64=576 个元素/层/token，
    # bf16 是1152 bytes，不能再乘 key/value 的2。普通两侧共2*16384=32768
    # 个元素，按这一对照缓存约压缩32768/576≈56.89倍（不同架构不能仅比 NH/C）。
    # 为什么不必每次重建全部历史？若 k_s=W_K c_s，则 q_t^T k_s=
    # (W_K^T q_t)^T c_s，可把上投影吸收到查询侧；加权值同理 Σ a_s W_V c_s
    # =W_V Σ a_s c_s。潜向量仍需训练学得，不能任意低秩截断而保证质量。
    # 原文“不兼容 RoPE”应限于直接吸收这种推导：R_t/R_s 是随位置变化的旋转，
    # q_t^T R_t^T R_s W_K c_s 不能用一个固定矩阵吸收所有 s。MLA 使用单独的
    # RoPE 分支解决，故不是 MLA 完全不能用位置编码。参见 [MLA] §2.1.2–2.1.3。
    text("Latency/throughput improvements follow similarly from the KV cache reduction as argued earlier")

    text("Let's now check the accuracy.")
    text("First, MHA is better than GQA (though more expensive) [Table 8] "), link(mla_2024)
    image("images/mla-accuracy.png", width=800)
    text("Second, MLA is even a bit better than MHA (and much cheaper) [Table 9] "), link(mla_2024)
    image("images/mla-accuracy2.png", width=800)
    # 表8/9是在论文各自配置下比较模型质量；“MLA 更好”是该消融的经验结论。
    # 应同时看参数预算、训练数据/步数与任务，不能由缓存维数推出准确率排序。

    text("### Cross-layer attention (CLA) "), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    text("Idea: share KVs across **layers** (just as GQA shares KVs across heads)")
    text("Empirically improves the pareto frontier of accuracy and KV cache size (latency and throughput)")
    # CLA 在相邻层之间复用 K/V；Q、残差和其他层运算仍各自执行。编者简例：
    # 40层按相邻两层共享一份 KV，其余条件相同时缓存层数因子从40变20，容量
    # 约减半，不表示总参数/FLOPs也减半。质量取决于这种共享如何训练。
    # Pareto 前沿指不存在另一个点同时“缓存更小且质量不差”，不是某个分数最高。
    image("images/cla-results.png", width=700)

    text("### Local (sliding window) attention "), link(longformer_2020), link(sparse_transformer_2019), link(mistral_7b_2023)
    image("images/longformer-attention.png", width=800)
    text("Idea: just look at the local context, which is most relevant for modeling")
    text("Effective context scales linearly with the number of layers")
    text("KV cache is independent of sequence length!")
    # 滑窗仅保留最近 w 个位置时，单层 KV 为4BKHL*min(S,w)的相应层份额；
    # 对全部 L 层均为相同滑窗才可直接乘 L。例：S=8192,w=1024，纯局部缓存
    # 是全局的1/8。包含自身的窗口每层最多向前传播 w-1 个位置，L层最远可达
    # 1+L(w-1) 个位置的感受野；这不是对所有远处 token 的一次精确全局查询。
    # 老信息经隐藏状态间接传递，细节可能丢失。若混合全局层，全局层缓存仍随 S
    # 增长；不能把下文 hybrid layers 也称作整个模型缓存与序列长度无关。
    text("Problem: this can still hurt accuracy")
    text("Solution: interleave local attention with global attention (hybrid layers)")

    text("### DeepSeek v4 attention")
    text("- Supports 1M context length "),  link(deepseek_v4_2026)
    image("images/deepseek-v4-attention.png", width=800)
    text("- Compressed Sparse Attention (CSA): compresses every m tokens into 1")
    text("- DeepSeek Sparse Attention (DSA): selects the top k")
    text("- Heavily Compressed Attention (HCA): compresses even more")
    # 图示阅读：左支保留滑窗细节，中支把多个 token 合并成压缩条目；indexer
    # 给压缩条目打分，top-k 再选择一部分；最后把局部与选中的条目拼接用于注意力。
    # “压缩”改变条目表示，“选择”改变参与本次查询的条目集合，二者不是同一步。
    # 编者简例：8192 个历史位置每8个合为1个，得1024个压缩条目；再取top-64
    # 只读其中64个做主注意力，但索引器和压缩状态仍有成本。这仅是示意数字，
    # 不是 DeepSeek-V4 的配置。1M 是原报告支持的上下文规格，不保证每个位置
    # 的信息都可无损找回，也不表示主注意力读到的条目数始终为1M。

    text("Summary:")
    text("- Goal: reduce the KV cache size (since inference is memory-bound) without hurting accuracy")
    text("- Lower-dimensional KV cache (GQA, MLA, CLA)")
    text("- Local attention (truncates the KV cache) on some of the layers")
    text("- Other ideas: linear attention / state-space-models (Mamba 2, GatedDeltaNet), diffusion models")
    # 线性注意力/状态空间模型把历史累积成状态，改变了显式检索历史的方式；
    # diffusion 可能改变生成顺序和迭代过程。这些是另一类模型，不是把当前
    # causal softmax Transformer 换个缓存内核就等价实现，质量/成本需另行评估。


def quantization():
    # 量化以较少位数近似原数，减小权重/缓存流量；速度收益还需匹配的内核，
    # 并扣除反量化、scale 读取等开销。原文 higher latency 应读为更低时延，
    # 更高吞吐；若变成计算受限，继续省字节也不保证同比加速。
    text("Key idea: reduce the precision of numbers")
    text("Less memory means higher latency/throughput (since inference is memory-bound).")
    text("Of course we have to worry about accuracy...")

    # Mechanics
    x = 5.2342  # @inspect x
    scale = 0.1
    zero_point = 4
    x_quant = round(x / scale) + zero_point  # Quantize  @inspect x_quant
    x_approx = (x_quant - zero_point) * scale  # Dequantize @inspect x_approx
    # 本例 round(52.342)+4=56，反量化为5.2，绝对误差.0342。
    # scale 控制网格间距，zero_point 让实数0有对应整数；真实量化还要裁剪到
    # 整数范围。无裁剪时最近舍入误差≤scale/2，超范围饱和时此界不再适用。

    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400), article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    text("- fp32 (4 bytes): needed for parameters and optimizer states during training")
    text("- bf16 (2 bytes): default for inference")
    text("- fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare "), link("https://arxiv.org/pdf/2310.18313")
    text("- int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but for inference only "), link("https://arxiv.org/pdf/2303.17951")
    text("- int4 (0.5 bytes) [-8, 7]: cheaper, even less accurate "), link("https://arxiv.org/pdf/2303.17951")
    # 原稿数值限定：NVIDIA H100 的 E4M3 最大有限值为±448，原±240不适用于
    # 该格式；参见 NVIDIA Transformer Engine 官方 FP8 说明：
    # https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html
    # 浮点还有指数/尾数，整数依靠scale映射，二者不能只按数值范围判断精度。
    # int8 并非普遍“仅能推理”，fp32 参数也不是所有训练方案都必需；这里是
    # 常见工程取舍。权重量化不自动量化 KV，必须分别说明 W/A/KV 的位宽。

    link(title="Overview of approaches", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    text("Quantization-aware training (QAT)")
    text("- During training, quantize-and-dequantize during the forward pass to simulate quantization errors")
    text("- Pro: weights are trained to work with quantization")
    text("- Con: requires expensive large-scale training")

    text("Post-training quantization (PTQ):")
    text("- Done after training, so much cheaper")
    text("- Run on sample data to determine scale and zero point for each layer or tensor")
    text("- GPTQ: use Hessian information to update non-quantized weights to account for quantization error "), link("https://arxiv.org/abs/2210.17323")
    # QAT 在前向模拟舍入/裁剪并继续训练，反向常用近似梯度；PTQ 用校准数据
    # 估计范围或重建误差，通常不做完整预训练。GPTQ 的直觉是量化一部分权重后，
    # 调整剩余权重以补偿输出误差；校准集若不代表部署输入，质量可能下降。

    text("### Activation-aware quantization (AWQ)")
    link("https://arxiv.org/abs/2306.00978")
    text("- Observation: some activation channels are large")
    text("- Weights that hit those matter more")
    text("- Allocate more precision to those weights")
    text("- Idea: select which weights (0.1-1%) to keep in high precision based on activations")
    text("- fp16 → int3 produces 4x lower memory, 3.2x speedup")
    image("images/awq-schema.png", width=800)
    # [AWQ] 图从左到右：直接舍入造成较大困惑度；保留少量显著权重为FP16
    # 可恢复质量，但混精度不利硬件；最终方法在量化前缩放通道，保护重要权重，
    # 使权重仍统一低位存储。故原文“保留0.1–1%高精度”是动机实验，不是最终算法。
    # 若输出 y=xW，令对角缩放矩阵 A，则 xW=(xA^-1)(AW)；先放大重要权重
    # 通道再量化、相应缩小激活，可减少这些通道的输出误差，但须同时考虑量化范围。
    # 原稿 int3、4x内存、3.2x速度应分开理解：纯位宽16/3≈5.33，实际还含元数据；
    # 论文 TinyChat 约4x内存/3.2–3.3x速度来自4-bit系统实验，依赖硬件与基线。


def model_pruning():
    # 结构化剪枝删除整层、头或通道，直接缩小矩阵形状；仅把零散权重置0，
    # 若仍调用密集矩阵乘法，通常不会相应变快。校准集用于判断哪些部分可删。
    # 删除后表示发生变化，蒸馏用教师输出指导学生恢复能力，例如最小化
    # Σ_x q_teacher(x|前缀) log(q_teacher(x|前缀)/p_student(x|前缀))。
    # 教师分布提供多个候选词的相对信息；学生容量较小，无法保证完全恢复质量。
    # 图的“剪枝→蒸馏”循环体现先降成本再修复；应比较同等部署预算下的任务分数，
    # 并计入校准和恢复训练成本。[剪枝] 的1024样本是实验设定，不是通用充分条件。
    text("Key idea: just rip out parts of an expensive model to make it cheaper")
    text("...and then fix it up.")

    text("Paper from NVIDIA "), link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    text("Algorithm:")
    text("1. Identify important {layer, head, hidden dimension} on a small calibration dataset (1024 samples)")
    text("2. Remove unimportant layers to get a smaller model")
    text("3. Distill the original model into pruned model")

    text("Results:")
    image("images/pruning-kd.png", width=500)

    # TODO


def speculative_sampling():
    # 以下沿用原稿 p=草稿、q=目标；γ=草稿长度（避免与 KV 头数K混淆）。
    # 草稿先顺序生成γ个候选，目标用因果mask一次计算这些已知候选前缀的分布，
    # 以更多位置重用一次权重读取。检查更快是工作负载上的机会，非所有批量下保证。
    text("Recall the two stages of inference:")
    text("- Prefill: given a sequence, encode tokens in parallel (compute-bound) [note: also gives you probabilities]")
    text("- Generation: generate one token at a time (memory-bound)")
    text("In other words, checking is faster than generation.")

    text("Speculative sampling "), link("https://arxiv.org/abs/2211.17192"), link("https://arxiv.org/abs/2302.01318")
    text("- Use a cheaper **draft model** p to guess a few tokens (e.g., 4)")
    text("- Evaluate with target model q (process tokens in parallel), and accept if it looks good")
    link(title="Speculative sampling video", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    text("This is modified rejection sampling with proposal p and target q")
    text("Modification: always generate at least one candidate (rejection sampling will keep looping)")
    text("Key property: guaranteed to be an **exact sample** from the target model!")
    # 已核验 [推测] Algorithm 2；课堂定位01:14:04–01:15:20。固定同一前缀，
    # 候选 x~p，以 a(x)=min(1,q(x)/p(x)) 接受。若拒绝，不能直接再从q采样，
    # 而应从 r(x)=(q(x)-p(x))_+/R 采样，R=Σ_x(q(x)-p(x))_+。
    # 接受路径已有 min(p,q) 的概率质量，残差只补目标尚缺的部分；R恰为总拒绝率。
    # 编者复用核对例：p=(.6,.3,.1)，q=(.2,.4,.4)，a=(1/3,1,1)。
    # 接受质量=(.2,.3,.1)，总接受率=.6；拒绝率=.4，r=(0,.25,.75)。
    # 拒绝路径补回 .4*r=(0,.1,.3)，最终输出=(.2,.4,.4)，正好是q。
    # p(x)=0的词不会成为候选，但q(x)>0时可由残差补入；R=0时不发生拒绝，
    # 无需计算除以0的残差。实际应对采样温度/top-k等处理后的分布使用同一规则。
    # 多token流程：按顺序检查，在首次拒绝位置补抽一个词并丢弃其后草稿，
    # 因为那些候选依赖已失效的前缀；回退对应KV缓存。若全部接受，可从目标
    # 最后一位置的分布额外抽一个词。因而每轮至少推进1个，最多γ+1个token。
    # “精确”指目标分布相同，不保证与逐词生成在同一随机种子下得到同一串文本。

    text("Proof by example: assume two vocabulary elements {A, B}")
    text("- Target model probabilities: [q(A), q(B)]")
    text("- Draft model probabilities: [p(A), p(B)]")
    text("- Assume p(A) > q(A) [draft model oversamples A].")
    text("- Therefore p(B) < q(B) [draft model undersamples B].")
    text("- Residual probabilities max(q-p, 0): [0, 1]")
    # 原稿这里省略了归一化：二词且p(A)>q(A)时，未归一化残差是
    # [0,q(B)-p(B)]，归一化后才是[0,1]。下方两式是在相同前缀下合并路径。
    text("Compute the probabilities of speculatively sampling a token:")
    text("- P[sampling A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)")
    text("- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)")

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)
    # 读速度/草稿长度图时留意：γ太小，摊薄目标权重读取不足；γ太大，草稿生成
    # 和被拒后浪费的检查工作增多。接受率高、草稿便宜才有利，峰值加速不是常数。
    # Medusa/EAGLE 改善候选生成方式；扩展是否保持精确分布，还取决于所用验证
    # 规则，不能仅凭“使用了目标模型检查”就断言无损。

    text("In practice:")
    text("- Target model has 70B parameters, draft model has 8B parameters")
    text("- Target model has 8B parameters, draft model has 1B parameters")
    text("- Try to make draft model as close to target (distillation)")

    text("Extensions to improve the draft model:")
    text("- Medusa: draft model generates multiple tokens in parallel "), link("https://arxiv.org/abs/2401.10774")
    text("- EAGLE: draft model takes high-level features from target model "), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=600)

    text("Summary:")
    text("- Exact sampling from target model (thanks to math)!")
    text("- Exploits asymmetry between checking and generation")
    text("- Lots of room for innovation on the draft model (involves training)")


def continuous_batching():
    # 课堂定位01:17:39–01:19:25，[Orca] 的关键是每轮重新安排活动请求。
    # 静态批等最长请求结束才换批，短请求结束后的槽位闲置；连续批可在下一轮
    # 移除结束请求、补入新请求。例：A还需1步、B还需3步，第1步A结束后，第2步
    # 即可让新C与B一起生成，而不必等B完成。新请求仍需先完成自己的prefill。
    link(title="Orca: A Distributed Serving System for Transformer-Based Generative Models", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="talk", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    text("Problem:")
    text("- Training: get a dense block of tokens (batch size x sequence length)")
    text("- Inference: requests arrive and finish at different times, so you have a ragged array")
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=600)

    text("Solution: iteration-level scheduling")
    text("- Decode step by step")
    text("- Add new requests to the batch as they arrive (so don't have to wait until generation completes)")

    text("Problem:")
    text("- Batching only works when all sequences have the same dimensionality (right?)")
    text("- But each request might have a different length")

    text("Solution: selective batching")
    text("- Training: when all sequences of the same length, operate on a B x S x H tensor")
    text("- But we might have different lengths: [3, H], [9, H], [5, H], etc.")
    text("- Attention computation: process each sequence separately")
    text("- Non-attention computation: concatenate all the sequences together to [3 + 9 + 5, H]")
    # MLP逐位置作用，可将[3,H]、[9,H]、[5,H]拼为[17,H]做共享权重乘法再拆开。
    # 注意力必须保留序列边界，不能让A的token看到B的历史；可用独立计算或
    # 支持变长序列的kernel，不要求每条请求单独启动一个kernel。decode每请求
    # 只含1个新query，上述3/9/5更适合解释变长输入，历史KV长度仍各不相同。
    # 扩大活动批可提高吞吐，却可能拉长TPOT；调度还需管理prefill、显存和排队。


def paged_attention():
    # [分页] 与连续批互补：前者管理KV存放位置，后者决定本轮处理哪些请求。
    # 课堂定位01:19:35–01:23:30。预留最大长度造成内部浪费，连续大块分配之间
    # 的小空洞形成外部碎片；分页让逻辑连续的token映射到任意空闲物理块。
    text("Paper that introduced vLLM in addition to PagedAttention "), link("https://arxiv.org/pdf/2309.06180.pdf")

    text("Previous status quo:")
    text("- Request comes in")
    text("- Allocate section of KV cache for prompt and response (up to a max length)")
    image("images/paged-attention-fragmentation.png", width=800)
    text("Problem: fragmentation (what happens to your hard drive)")
    text("- But this is wasteful since we might generate much fewer tokens (internal fragmentation)!")
    text("- Might be extra unused space between sections (external fragmentation)!")

    text("Solution: PagedAttention (remember operating systems)")
    text("- Divide the KV cache of a sequence into non-contiguous **blocks**")
    image("images/paged-attention-blocks.png", width=400)
    # 编者简例：块容量4个token，长10的序列需3块，最后空2格；块表可为[7,2,9]，
    # 第t个token定位到 block_table[t//4] 的 t%4 槽位。新增第11个token仍用块9，
    # 不需提前预留完整最大长度。仍有尾块浪费和块表开销，块越大并非总越好。

    text("Two requests share the KV caches:")
    image("images/paged-attention-logical.png", width=800)

    text("In general, multiple types of sharing KV caches across sequences:")
    image("images/paged-attention-sharing.png", width=600)
    text("- Sharing the system prompt")
    text("- Sampling multiple responses per prompt (e.g., for program synthesis)")

    text("Solution: share prefixes, copy-on-write at the block level")
    image("images/paged-attention-parallel.png", width=600)
    # 共享图中多个逻辑块可指向同一物理块；共享的是同一模型、同一前缀及位置
    # 所算出的KV，不是任意相同词串（前文不同会改变隐藏状态）。维护引用计数，
    # 分叉生成若要写共享的尾块，先复制再写，避免改变另一请求的缓存。
    # 已满且不再修改的前缀块可继续共享；后缀各自分配。这节省容量并复用prefill，
    # 但每个请求仍有自己的Q，不能据共享缓存断言attention计算全部合并。

    text("Other vLLM optimizations:")
    text("- Kernel to fuse block read and attention (reduce kernel launch overhead)")
    text("- Use latest kernels (FlashAttention, FlashDecoding)")
    text("- Use CUDA graphs to avoid kernel launch overhead")
    # 分页attention内核按块表读取并累积softmax结果；FlashAttention优化片上
    # 数据复用，CUDA graphs减少CPU重复发射开销，三者解决不同层面的成本。
    # 自测与答案：①仅分页是否让每token的KV字节变少？不变，主要减少预留浪费
    # 与增加共享。②连续批是否需要请求同时开始？不需要，在迭代边界加入。
    # ③草稿拒绝后从q直接重采是否精确？一般不精确，因为接受路径已偏向部分词，
    # 必须用归一化正残差补足。④量化后权重减半是否总加速2倍？否，KV、计算和
    # 反量化仍有成本。回到资源账本逐项判断，才知道下一步该优化哪里。

    text("Summary: use ideas from operating systems (paging) to make use of memory for dynamic workloads")


if __name__ == "__main__":
    main()
