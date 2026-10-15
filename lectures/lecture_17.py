# 中文自学导读｜第17讲：多模态模型（Spring 2026 原稿保留，新增内容均为编者注释）
# 学习目标：能写出 CLIP/SigLIP 的目标，追踪视觉特征到语言模型的张量接口，
# 解释高分辨率与视频的 token 预算，并区分“理解图像”和“生成图像”的训练要求。
# 先修：向量内积与 L2 范数、softmax/交叉熵、二分类 sigmoid、Transformer/QKV、
# 自回归语言模型的 teacher forcing。遇到不熟悉的公式，可先读下文二样本手算再返回。
# 主线：图文对齐学视觉语义 → 投影接入 LM → 分辨率/时空位置/数据配比 → 离散图像生成。
# 这里 token 指序列中的一个表示单元，既可能是实数向量，也可能是有限词表里的整数 ID。
# 来源索引：本文件原英文和图为讲义；缓存 lecture_17.srt 为第三方公开录像字幕，非教师审校稿。
# 字幕只作带时间戳的课堂转述；公式、形状例子及自测标为编者补充，不冒充课堂原话。
# 主要文献（沿用原稿链接）：CLIP https://arxiv.org/abs/2103.00020 ；
# SigLIP https://arxiv.org/abs/2303.15343 ；LLaVA https://arxiv.org/abs/2304.08485 ；
# OneVision https://arxiv.org/abs/2408.03326 ；Qwen-VL https://arxiv.org/abs/2308.12966 ；
# Qwen2-VL https://arxiv.org/abs/2409.12191 ；Qwen3-VL https://arxiv.org/abs/2511.21631 ；
# Chameleon https://arxiv.org/abs/2405.09818 ；VQ-VAE https://arxiv.org/abs/1711.00937 。
# 阅读边界：下文“最新/SOTA”等是原讲义当时的描述，不是持续更新的模型排名。
# 这些注释供阅读 Python 源码使用；不改变 edtrace 输出，未验证网页渲染。
from edtrace import text, image, link
from lecture_util import article_link, post_link

def main():
    text("## Lecture 17: multimodal models")
    text("So far: language models")
    text("> text ⇒ text")
    text("The world is multimodal:")
    image("images/multimodality.png", width=600)

    # 动机：文本可以描述“红色杯子”，却未必保留杯口形状、像素坐标或说话人的音色。
    # 把图像压成一句 caption 会丢掉后续任务需要的细节，因此让模型直接接收视觉表示。
    # “任意输入/输出”是目标，不是后面每个 VLM 都具备的能力：图像输入、文本输出
    # 只覆盖理解；能回答图片问题，并不意味着能采样生成图片。
    text("Ultimate goal: **omni model**")
    text("- Input any combination of modalities (understanding)")
    text("- Output any combination of modalities (generation)")

    text("Where we are today:")
    text("- Transformers work really well. So we gotta use them.")
    text("- Transformers speak tokens (discrete or continuous), where a token represents some ~semantic unit of information.")
    text("- Therefore, we must convert everything into tokens.")
    text("- Note: we had to do this with text (recall the tokenization lecture).")
    text("- For non-text modalities, this is more challenging...")

    text("Questions:")
    text("1. How do we input non-text data (e.g., understand images)?")
    text("2. How do we output non-text data (e.g., generate audio)?")

    # 阅读路线：先问视觉编码器学到了什么，再问 LM 如何读取它。
    # 对齐的含义也有两层：CLIP 是图文表征相似度对齐；LLaVA 第一阶段则是接口对齐。
    # 二者目标与冻结参数不同，不能把所有名为 alignment 的训练视为同一过程。
    # Encoding images
    clip()
    siglip()

    # Injecting image encodings into LLMs
    llava()
    llava_onevision()
    qwen_vl()
    qwen2_vl()
    qwen3_vl()

    # Towards Omni models
    chameleon()

    text("Summary:")
    text("- Frontier models are expected to be multimodal (natively multimodal, omni)")
    text("- Fundamental challenge: how to encode non-text modalities?")
    text("- Comprehension and generation might demand different things (semantics versus finer-grained details)")
    text("- Balance images + video (lower information density) and text for training stability")
    text("- Continuous encoders + Transformer + diffusion models for generation")


def clip():
    text("CLIP (Contrastive Language-Image Pretraining) "), link("https://arxiv.org/abs/2103.00020")

    text("Context:")
    text("- Computer vision models were trained on annotated images.")
    text("- Question: is it possible to leverage the much larger amount of (image, caption) pairs?")
    image("images/clip.png", width=800)

    # 编者推导｜设一批有 B 个匹配对 (I_i,T_i)，图像/文本编码后分别投影到共同的 d 维。
    # 令 u_i=f(I_i)/||f(I_i)||₂，v_j=g(T_j)/||g(T_j)||₂；S_ij=u_i·v_j/τ。
    # 矩阵 S 为 B×B，对角线是标注正例，其他位置作为批内负例；两边输入不是逐元素相乘。
    # 图到文：L_i2t=-(1/B)Σ_i log[exp(S_ii)/Σ_j exp(S_ij)]；
    # 文到图：L_t2i=-(1/B)Σ_i log[exp(S_ii)/Σ_j exp(S_ji)]；L=(L_i2t+L_t2i)/2。
    # 第二个分母沿列归一化，因此不是把同一行损失重复两遍；正例索引均为 i。
    # L2 归一化去掉向量长度对分数的影响；正温度 τ 控制 softmax 的尖锐程度。
    # 实现常学习 log(1/τ)，取 exp 后乘余弦相似度；温度小会放大分数差，也可能放大噪声梯度。
    # 手算：B=2，S=[[2,0],[0,2]]，每行/列正例概率 e²/(e²+1)≈0.881，
    # 两方向平均损失均约 0.127；若所有分数相同，则损失为 log 2≈0.693。
    # 这不是图像类别分类：类别是本批的文本位置，下一批候选集合会变化。
    # 字幕 00:07:48 附近讨论温度；上述公式与手算是编者补充。
    text("Method:")
    text("- Get a batch of (image, text) examples (e.g., 32768)")
    text("- Encode each image and each text")
    text("- For each image, prefer its aligned text over other texts")
    text("- For each text, prefer its aligned image over other images")
    image("images/clip-code.png", width=400)

    text("Data:")
    text("- Searched for 500K queries, get ~20K (image, text) pairs per query")
    text("- Trained on 400M image-text pairs")
    text("- Didn't release the dataset")
    text("- Reproduced in OpenCLIP (using LAION-5B dataset, which used CLIP for filtering) "), link("https://arxiv.org/abs/2212.07143")

    # 数据边界：网页文字提供弱监督，可能只提图片的一部分，甚至配错；批内非配对项
    # 也可能描述相同物体，形成假负例。因此更大批量增加候选与计算，也可能增加语义冲突。
    # 零样本分类：把各类别写成提示词，用文本编码器得到类别向量，再与图像向量比相似度。
    # “零样本”指不在目标分类数据集上训练分类头，不代表训练语料绝无相关图片或概念。
    # 下文 336 是特定模型配置，不是所有 CLIP 都固定为 336；resize/crop 会损失边缘与细字。
    text("Data processing "), link(title="code", url="https://github.com/openai/CLIP/blob/main/clip/clip.py#L79")
    text("- Images come in all resolutions (arbitrary W x H)")
    text("- Resize using bicubic interpolation so shorter side is 336 pixels")
    text("- Center crop (cuts off borders to get 336 x 336)")
    
    text("Vision encoder:")
    text("- Experimented with ResNet-50 and Vision Transformers "), link("https://arxiv.org/pdf/2010.11929")
    image("images/vit.png", width=600)
    text("- Attention pooling: do QKV with query = global average of activations")
    text("- Best model: ViT-L/14@336px (L = large, 14x14 patches, 3 channels, trained on 336x336 resolution images)")

    # 形状例子：336/14=24，一幅图有 24×24=576 个 patch，每块原像素为 14×14×3。
    # ViT 将每块展平并线性映射到隐维，加空间位置后做注意力；若含 CLS，还需计入额外位置。
    # 注意区分全局检索向量和 patch 网格：CLIP 对比目标用全局表示，VLM 常取网格特征，
    # 否则只传一个全局向量容易丢掉“左上角写了什么”的局部信息。
    # 原稿的 attention pooling 主要对应修改后的 ResNet 分支，不能据此认为所有 ViT 都同样池化。
    # 文本编码器架构沿用 GPT 风格，不等于直接使用完整 GPT-2 预训练权重；EOS 汇集序列信息。
    text("Text encoder:")
    text("- GPT-2 Transformer (63M parameters, 12 layers)")
    text("- Encode [BOS] ... [EOS], return [EOS] activation at highest layer")

    text("Headline result:")
    text("- On ImageNet, zero-shot CLIP outperformed ResNet-50 trained on 1.2M ImageNet images")

    # 读效率图：对比学习直接训练“哪幅图与哪段文字对应”，不必逐 token 还原整段文字。
    # 优势是对语义检索有效；代价是 caption 未区分的纹理、精确文字可能被压掉。
    # 分类指标上的高效不能推出 OCR、空间定位或图像重建也同样最优。
    text("Ablation:")
    text("- Alternative: predict text from images directly")
    text("- Much less compute efficient compared to CLIP-style ranking")
    image("images/clip-efficiency.png", width=400)

    text("Summary:")
    text("- Encoding of images captures semantics given by (noisy) text")
    text("- Design decisions chosen based on image classification (not very fine-grained)")
    text("- Technical: requires large batch sizes, softmax operation over full batch")


def siglip():
    text("SigLIP (Sigmoid Loss for Language Image Pre-Training) "), link("https://arxiv.org/abs/2303.15343")

    # 编者推导｜SigLIP 仍使用归一化向量，但令 z_ij=t(u_i·v_j)+b，t>0 是尺度，b 是偏置。
    # 标签 y_ij=+1（匹配）或 -1（不匹配），逐对损失为 -log σ(y_ij z_ij)。
    # 常用全批形式 L=-(1/B)Σ_iΣ_j log σ(y_ij z_ij)，等价于二元交叉熵的稳定实现。
    # 注意归一化约定：除 B 不是除 B²；改成所有 pair 的平均会改变梯度尺度，需相应考虑学习率。
    # CLIP 的每行/列概率和为 1；SigLIP 每对独立打分，不要求不同文本的匹配概率加起来为 1。
    # 例：正对 z=2 的损失 -logσ(2)≈0.127；负对 z=2 则 -logσ(-2)≈2.127。
    # 完整 B×B 中正对 B 个、负对 B(B-1) 个，负正比例 B-1，不是天然平衡的二分类。
    # 若暂不区分任何 pair，最优常数概率为 1/B，对应 b=log[(1/B)/(1-1/B)]=-log(B-1)。
    # 这说明负初始偏置的动机；实际负例采样、重加权、批量变化会影响合适的偏置与尺度。
    # σ(z) 是训练配对分布上的分数，不自动成为开放世界里“语义真匹配”的校准概率。
    # 字幕 00:23:13–00:24:00 对应逐对二分类和 log_sigmoid；公式为编者补充。
    text("Objective:")
    text("- CLIP: multiclass classification for (text, image) versus (text, image') for all image'")
    text("- SigLIP: binary classification for (text, image) - aligned or not?")
    image("images/siglip-code.png", width=500)

    text("Data:")
    text("- WebLI dataset: O(billion) (image, text) pairs "), link("https://arxiv.org/pdf/2209.06794")
    text("- Scraped from the Internet")
    text("- Used automatic OCR to extract text from images")
    text("- Keep 10% highest quality")
    text("- Supports 100 languages")

    # 为何有利并行：每对只需自己的分数/标签，不依赖跨整批 softmax 分母，可分块计算损失。
    # 并非“不需负例/不需通信”：计算远端图文 pair 仍可能交换特征，只是分块与通信组织更灵活。
    # 全量 pair 仍是 B² 个；降低峰值内存不等于自动降低全部算术量。
    # 下列运行时间同时受硬件、数据、模型与训练设置影响，不是只替换一个 loss 的严格等预算测速。
    # TPUv3/v4 的相对吞吐也取决于设备计数和工作负载，不能把括号当通用硬件性能结论。
    text("Efficiency:")
    text("- CLIP: 10 days on 256 TPUv3")
    text("- SigLIP: 5 days on 32 TPUv4 (lower FLOP/s than TPUv3) - much faster!")
    image("images/siglip-parallelism.png", width=800)

    # “解耦”指逐对目标不必用全批归一化，不意味着批量完全不影响统计或效果。
    # 批量改变负正比例、覆盖的难负例和梯度方差；32K 足够是文中实验经验，不是普遍阈值。
    text("Batch size:")
    text("- Decouple batch size from loss")
    text("- Better than CLIP for <16K batch sizes")
    text("- Go up to 1M batch size, but 32K is enough")


def llava():
    text("LLaVA (Large Language and Vision Assistant) "), link("https://arxiv.org/abs/2304.08485")

    # 接口全图：像素 → 视觉编码器 E → N 个 d_v 维特征 → 投影器 P → N 个 d_lm 维向量，
    # 再与文字 embedding 拼成 LM 输入；LM 最终用词表输出头预测回答文字。
    # 形状补充：E(I)∈R^(N×d_v)，线性 P 可用 W∈R^(d_v×d_lm)，于是 E(I)W∈R^(N×d_lm)。
    # W 只改通道维，不把 N 自动压缩成一个 token；MLP 也不自动减少序列长度。
    # 输入视觉向量绕过文字 embedding 查表，仍可参与相同的自注意力；它无需先被舍入成词 ID。
    # 从 CLIP 取出的特征也不一定已是 LM 的语言空间，维数相同更不代表语义已对齐。
    text("Vision encoder: CLIP")
    text("Text decoder: Vicuna (LLaMA fine-tuned on ShareGPT conversations) "), post_link("https://www.lmsys.org/blog/2023-03-30-vicuna/")

    text("Data:")
    text("- MS COCO has images annotated with bounding boxes and Mechanical Turk captions")
    text("- Prompt GPT-4 with captions or detected objects and generate questions or conversations")
    text("- Pair generations with original images")
    text("- 158K examples")
    image("images/llava-gen.png", width=600)

    # 合成数据链条：人工 caption/检测框 → 文本模型生成问答 → 重新配回图片。
    # 生成者若未直接看图，其答案依赖 caption 是否充分；编造细节会成为学生模型的监督噪声。
    # 因此数据数量不是唯一指标，要检查图像、问题、答案之间是否一致。
    text("Model:")
    text("- Encode images with CLIP (ViT-L/14)")
    text("- Linear projection (W) into embedding space (Flamingo and Q-former are more complex)")
    image("images/llava-architecture.png", width=600)

    text("Training:")
    # 训练解释：损失仍是回答 token 的自回归交叉熵，视觉向量作为条件，不是要 LM 复原像素。
    # teacher forcing 把正确历史回答喂回去，预测下一个回答 token；通常屏蔽图像/提示位置的监督。
    # 阶段1仅更新 W，让已有视觉/语言系统找到接口；阶段2更新 W+LM，让模型学会按问题用视觉。
    # 冻结表示“不更新参数”，不等于梯度无法穿过冻结的 LM 回到 W；切断计算图会让 W 学不到。
    # 推理则只有图片、指令与已生成 token，不再有未来正确答案。字幕 00:34:09 附近对应第二阶段。
    text("- Stage 1 (alignment): freeze vision encoder and language model, only train W")
    text("- Stage 2 (fine-tuning): freeze vision encoder and train W and language model")
    image("images/llava-example.png", width=600)


def llava_onevision():
    text("LLaVA OneVision "), link("https://arxiv.org/pdf/2408.03326")
    # OneVision 的问题从“能看一张图”转为“共享模型处理单图、多图、视频”。
    # 视觉编码器提供语义特征，两层 MLP 提供非线性接口；选哪一层特征是实现超参数，
    # 原稿对最后层前后特征的描述不应推成“所有 SigLIP 接口都必须拼这两层”。
    text("- Latest version in the LLaVA series (after LLaVA 1.5, LLaVA-Next)")
    text("- Handle multiple images, video")

    image("images/llava-onevision.png", width=600)
    text("- Vision encoder: SigLIP (use grid features before and after last Transformer layer)")
    text("- Text decoder: Qwen-2 72B")
    text("- Projector: 2-layer MLP")

    text("Data processing:")
    text("- Preserving high resolution is important (e.g., for OCR)")
    text("- CLIP resizes and crops to 336x336, which loses information")
    # 高分辨率动机：整图缩小可能把一个小字压到不到一个像素；之后的 LM 无法恢复已丢信息。
    # AnyRes 用局部裁块保留细节，常再加全局缩略图帮助理解布局；需保持裁块顺序/空间信息。
    # 来源提示：原稿把 AnyRes 归于 LLaVA 1.5，但其高分辨率任意网格方案通常与 LLaVA-NeXT
    # 系列联系；所链 1.5 文献不应单独作为该历史归属的证明。保留原稿，重点理解此处机制。
    # 编者预算例：若每裁块 576 个特征，6 块即 3456 个，另加全局图则 4032 个（未计边界符）。
    # 拼接不只增加显存：普通全注意力中视觉序列变为 k 倍，纯视觉注意力项可约增为 k² 倍。
    # 插值/池化可以降 token 数，却会抹平局部细节，这是准确率与成本之间的折中。
    text("- Solution: AnyRes, introduced in LLaVA 1.5 "), link(title="paper", url="https://static.hliu.cc/files/llava/improved_llava.pdf")
    text("- Break up image into a x b pieces (matching resolution of vision encoder), encode, concatenate")
    text("- If too many tokens (original image is too high resolution), then use bilinear interpolation")
    image("images/llava-onevision-anyres.png", width=600)
    text("Handle 3 types of input (single image, multiple images, video):")
    text("- Goal: make all of the modalities produce roughly the same length")
    image("images/llava-onevision-modalities.png", width=600)
    text("- Single image: use higher resolution")
    text("- Multiple images: use base resolution for each image")
    text("- Video: use lower resolution for each frame")

    # 字幕 00:39:52–00:40:23：强调避免重复视频帧主导数据，单图使用全图和多个裁块。
    # 视频 token 数近似 帧数×每帧 token 数；预算固定时，要在时间采样密度与单帧清晰度间分配。
    # 例如 8 帧×64=512 个视觉 token，与一帧 512 个的预算相近，但保留的信息类型完全不同。
    # 时间上太稀会漏动作，空间上太粗会看不清文字；“统一长度”不是所有任务的最优策略。
    text("Data:")
    text("- Philosophy: quality over quantity")
    image("images/llava-onevision-data-1.png", width=700)
    image("images/llava-onevision-data-2.png", width=700)

    text("Training:")
    text("- Philosophy: easier to harder")
    image("images/llava-onevision-training.png", width=700)

    # 读数据/训练图的方法：分别追踪每阶段的数据来源、图像数/帧数、序列长度和可训练模块。
    # 从单图到多图再到视频的课程安排降低难度跳跃；高质量合成问答仍需防止错误答案固化。
    # 以下迁移图给的是特定任务实例/实验现象，不足以证明所有关系推理都能零成本跨模态泛化。
    # 看图时可先问：训练中出现的是哪种输入，测试增加了哪种结构，能力是否来自共享语言推理？
    text("Transfer between modalities:")
    text("- Single image data for diagrams and charts, but generalize to multi-image")
    image("images/llava-onevision-transfer-s1.png", width=600)
    text("- OCR on single image data, relational reasoning from multi-image data, generalize to GUI-based agents")
    image("images/llava-onevision-transfer-s2.png", width=600)
    text("- Visual prompting (circle) in single images, generalize to videos")
    image("images/llava-onevision-transfer-s8.png", width=600)

    text("Summary:")
    text("- Standard VLM template: vision encoder + projector + LM")
    text("- Most work goes into data curation (heavy on synthesized, task-specific data)")
    text("- Open-source (released model weights and data)")


def qwen_vl():
    text("Qwen-VL "), link("https://arxiv.org/abs/2308.12966")

    # Qwen-VL 改用查询式适配器：令 256 个查询读取任意 N 个视觉特征，
    # Q 为 256×d，K/V 来自 N 个视觉位置，softmax(QKᵀ/√d)V 输出 256×d。
    # 这同时改长度与通道接口；与只做逐 token 线性投影不同。固定长度控制 LM 成本，也构成信息瓶颈。
    # 2D 位置使注意力区分左/右、上/下；<box>/<ref> 等特殊 token 提供结构语法，
    # 但模型仍须通过训练学会坐标与所指物体的对应，不是加几个标记就自然理解定位。
    # 比较训练表时要按版本读：本段第一阶段训练视觉端，不能套用 LLaVA 冻结视觉端的规则。
    text("Architecture:")
    text("- Vision encoder: OpenCLIP's ViT-bigC (14x14 patches) "), link("https://arxiv.org/abs/2212.07143")
    text("- Adaptor: one layer cross-attention, incorporate 2D positional encodings, maps to fixed length of 256")
    text("- Special tokens: <img>, <box>, <ref>")

    text("Training:")
    image("images/qwen-vl-stages.png", width=700)
    text("- Stage 1: large-scale low quality data; freeze LM, train vision encoder + adaptor")
    image("images/qwen-vl-stage1.png", width=400)
    text("- Stage 2: higher quality task-specific data, increase resolution; train all parameters")
    image("images/qwen-vl-stage2.png", width=400)
    text("- Stage 3: instruction tuning data; freeze visual encoder, train adaptor + LM")

    image("images/qwen-vl-examples.png", width=600)


def qwen2_vl():
    text("Qwen2-VL "), link("https://arxiv.org/abs/2409.12191")

    text("Visual encoder: larger ViT (675M)")
    image("images/qwen2-vl-architecture.png", width=700)
    text("- Key: dynamic resolution to handle varying resolutions")
    # 编者核算与原稿疑点：224/14=16，16×16=256 个 patch，2×2 合并后应为 8×8=64。
    # 因此原文“66 tokens”不能由给定尺寸推出；若存在额外特殊 token 需另行说明，不应臆造两个。
    # 动态分辨率的核心是按输入 H、W 得到可变网格，而非把所有图都强制变成一个 224 方块。
    # 对可整除尺寸，2×2 合并后的空间数量约 (H/28)(W/28)，边界符/取整需另计。
    # 真实处理通常先调整到兼容 patch/merge 的尺寸，并受最大像素/token 预算约束。
    text("- Each 224 x 224 patch encoded with ViT/14, compress every 2x2 => 66 tokens")
    text("- Video: sample 2 frames/sec, max 16384 tokens")

    text("Multimodal Rotary Position Embedding (MRoPE):")
    # MRoPE 是位置机制，不是视觉内容编码器：把 RoPE 的通道对分配给时间 t、高度 h、宽度 w。
    # 图像可共享时间坐标而 h/w 变化；视频再让 t 随帧变化；文字按一维序列位置处理。
    # 将二维图展平后，相邻序列位置未必在空间相邻，显式轴信息可避免把换行当普通水平位移。
    # 它仍需视觉特征提供“是什么”，位置旋转主要告诉注意力“在哪里/何时”，不保证几何推理正确。
    # 字幕约 00:50:43 的 66 说法与原稿一致，但不消除上面的算术问题。
    image("images/qwen2-vl-mrope.png", width=600)
    
    text("Initialize LM with Qwen2 and vision encoder from DFN "), link("https://arxiv.org/abs/2309.17425")
    text("Training (similar to Qwen-VL):")
    text("- Stage 1: train only visual encoder")
    text("- Stage 2: train all parameters")
    text("- Stage 3: train language model on instruction following datasets")

    text("Many capabilities:")
    image("images/qwen2-vl-capabilities.png", width=700)


def qwen3_vl():
    text("Qwen3-VL "), link("https://arxiv.org/abs/2511.21631")
    image("images/qwen3-vl.png", width=700)

    # Qwen3-VL 继续沿用“视觉端+接口+LM”，变化分布在长上下文、位置通道和深层融合。
    # 235B-A22B 中前者是总参数、后者是每 token 激活参数的量级，不能当作同一种成本数字。
    # 256K 是上下文容量，不保证每个位置、每种任务都能同等可靠地利用这些 token。
    text("Language model:")
    text("- Qwen-3 models (dense and MoE models up to 235B-A22B)")
    text("- Long context understanding (256K)")

    text("Vision encoder:")
    text("- SigLIP-2 (same architecture as SigLIP) "), link("https://arxiv.org/pdf/2502.14786")
    # 原稿把本节位置机制列在视觉编码器下面；概念上应与 SigLIP-2 的视觉表征分开理解：
    # MRoPE 负责多模态序列在 LM 中的时空位置，SigLIP-2 负责从像素提取特征。
    # 交错分配 t/w/h 通道，使不同轴都覆盖多种旋转频率；不是把输入帧/像素顺序交错打乱。
    # 显式时间戳提供实际秒数，可区分不同采样率的视频；帧序号本身不足以表达真实时间间隔。
    text("- Interleaved MRoPE: distribute all axes (temporal, width, height) to low- and high-frequency bands")
    text("... [t w h t w h t w h t w h] rather than [t t t t w w w w h h h h]")
    text("- Add explicit video timestamps (as separate tokens rather in positional embeddings)")
    # 损失配比补充：先区分回答监督 token 数与输入视觉 token 数，不能无条件对全部输入算语言损失。
    # 长度归一化介于按 token 总和（长样本占重）与按样本均值（每样本等重）之间。
    # 若示意权重为样本损失和/√n，则同平均单 token 损失下权重随 √n 增长，而不是随 n 增长；
    # 这是帮助理解折中的编者例子，具体 n 的定义与批内归一化须按论文实现，不能仅由标题确定。
    # 字幕 00:56:05–00:56:41 讨论长视频与长度权重，不能把它读成视觉输入位置都具有预测标签。
    text("- Square-root-normalized per-token loss: balance text and multimodal data (video examples are long, don't want to dominate)")

    text("Adapter:")
    # DeepStack 跨层注入让不同深度的视觉特征影响多个 LM 层，避免所有信息都挤在输入一次注入。
    # 这与“增加图片数量”不同，也不表示视觉编码器与 LM 各层必然一一对应；需指定映射与融合位置。
    # 预训练先适配接口，再放开更多参数和上下文；后训练 SFT/蒸馏/RL 分别提供示范、教师信号、奖励。
    # 读结果图时区分模型规模、评测任务及训练数据差异，不将多项同时改变的结果归因于单个模块。
    text("- DeepStack: cross-layer fusion to inject visual information into multiple layers "), link("https://arxiv.org/abs/2406.04334")

    text("Training:")
    text("- Pre-training has 4 stages (train adapter, train all parameters on 8K, 32K, 256K lengths)")
    image("images/qwen3-vl-pretraining.png", width=600)
    text("- Post-training: SFT on long CoT data, knowledge distillation, RL")

    image("images/qwen3-vl-results.png", width=600)

    text("Summary:")
    text("- SOTA performance")
    text("- Lots of data work, but not many details")
    text("- Minor but potentially important architectural improvements")
    text("- Scale up")


def chameleon():
    text("Chameleon "), link("https://arxiv.org/pdf/2405.09818")

    # 从理解转向生成：连续视觉特征能直接作为条件输入，但标准词表 softmax 不会直接输出任意实数图像。
    # 若另接扩散/连续生成头可以生成图像；因此原稿“need diffusion”是这条常见路线的概括，
    # 不是证明扩散是唯一办法。Chameleon 示范另一条路线：先把图像变成有限词表序列再自回归生成。
    # 字幕 01:00:07–01:00:11 明确此前例子主要在输入端多模态、输出端生成文字。
    text("So far: VLMs encode images (via CLIP or SigLIP), inject into LM")
    text("Disadvantage: can't generate images (need diffusion)")

    text("Chameleon: map everything into discrete tokens")
    text("Advantage: can analyze and generate images in a uniform way")
    image("images/chameleon.png", width=600)
    image("images/chameleon-example.png", width=600)

    text("Vision encoder "), link("https://arxiv.org/pdf/2203.13131")
    text("- Key difference: encoder needs to map to discrete tokens (so we can generate them)")
    text("- VQ-VAE (Vector Quantized Variational Autoencoder) "), link("https://arxiv.org/pdf/1711.00937")
    # VQ 手算与机制（编者补充）：编码器输出 z_e(x) 的空间网格，每格在 K 个码本向量 e_k 中
    # 选最近项 k*=argmin_k ||z_e-e_k||²，把整数 k* 作为 token；解码器从对应向量重建像素。
    # 例如码本向量 {(0,0),(1,0),(0,1)}，输入特征 (0.9,0.1) 距第二项最近，输出 ID 1（从0计）。
    # 最近邻离散选择不可普通微分，训练常用 straight-through 估计，加码本/commitment 项，
    # 不能只写重建损失便认为量化器端到端梯度自然可传；具体视觉 tokenizer 还可用感知/对抗损失。
    # 原始 VQ-VAE 常写 L=重建误差+||sg[z_e]-e||²+β||z_e-sg[e]||²，sg 表示停止梯度。
    # 第二项移动码本靠近编码器，第三项让编码器承诺使用码本；这是先修解释，不是该模型全部配方。
    # 512×512→1024=32×32 个位置意味着每边降采样16倍；K=8192 则每位置有8192种选择。
    # “1024 个 token”是序列长度，“8192”是视觉词表大小，不能互换；每 ID 最少固定编码需13 bit。
    # 量化后再查码本得到连续向量供 Transformer 使用：离散的是选择空间，不是内部计算全变整数。
    # 文本 BPE 与图像码本是不同 tokenizer；联合词表需要区分文字/图像 ID 和模态边界。
    text("- Idea: map image to a discrete codebook, decode back to image and minimize reconstruction loss")
    image("images/vq-vae.png", width=600)
    text("- Encodes 512 x 512 image into 1024 tokens (codebook of size 8192)")
    text("- Train a new BPE tokenizer")

    text("Training:")
    text("- Stage 1 (80%): large-scale, unsupervised (2.9T text tokens, 1.5T text/image tokens, 400B text/image interleaved tokens)")
    text("- Stage 2 (20%): 50% of stage 1 data, 50% of high quality data")
    
    # 两阶段别混淆：先训练能重建图像的视觉 tokenizer，再用其离散 ID 与文字训练自回归 Transformer。
    # 生成时先采样图像 ID 序列，再由视觉解码器还原像素；并非让语言模型逐个预测 RGB 字节。
    # 理解目标偏语义不变性，重建目标需要保留细节，因而 CLIP 编码器不能直接当作可逆图像压缩器。
    # 离散化会损失信息，但最终 OCR/生成效果还取决于码率、训练数据与模型，不是离散必然差的定理。
    # 稳定性解释：不同模态的序列长度、预测熵和 logit 尺度不同，混合后可能造成梯度/范数失衡。
    # QK norm 控制注意力点积尺度；z-loss 常以 (logΣ exp(logit))² 正则约束输出 logit 漂移。
    # 二者作用位置不同，也不是消除所有训练不稳定的充分条件；要与采样比例和损失权重共同考虑。
    text("Training stability")
    text("- Text tokens have low entropy, image tokens have high entropy, leads to norm growth, logit drift problem")
    text("- Fixes: QK norm, z-loss regularization")

    text("Summary:")
    text("- Elegant (just autoregressive modeling of discrete tokens)")
    text("- Not as performant (discretization loses information - think OCR)")
    text("- Training with multiple modalities is tricky")
    

# 自测与解答（编者补充）
# 1. 两个正对相似度都高，CLIP 的两个方向为何还可能损失不同？
# 答：非对角项使行、列分母不同；对称矩阵能保证两个方向相同，一般情况下不能保证。
# 2. B=4 的全量 SigLIP 有多少负对，初始常数匹配概率多少？
# 答：12 个负对、4 个正对；概率1/4，常数偏置 log(1/3)≈-1.099，不是0。
# 3. 把 576×1024 特征乘 1024×4096 的 W，能得到几个 LM 输入位置？
# 答：576 个，每个4096维；若想压成256位置，需要池化/查询适配器等额外机制。
# 4. 冻结 LM 时能否训练投影器？答：能，只要保留从 LM 损失到输入的梯度路径；冻结只禁止参数更新。
# 5. 448×224 输入、14像素 patch、2×2 merge，有几个空间 token？
# 答：(448/28)×(224/28)=16×8=128，特殊边界位置另计；该算法不能推出66。
# 6. 图像码本有8192项，一幅图1024位置，是否图像只有1024种？
# 答：不是，理论 ID 序列空间有8192^1024种；有效图像分布远小于该组合空间。
# 7. 为什么提升分辨率不能无限改善？答：受原图信息、token预算、数据与注意力成本约束；
# 插值不能创造未采集细节，更多位置也可能挤占文字/视频上下文。
if __name__ == "__main__":
    main()
