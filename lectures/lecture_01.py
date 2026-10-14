# ============================================================================
# 中文自学导读（对应 CS336 Spring 2026 第一讲，补充日期：2026-10-15）
# ============================================================================
# 本文件有两条主线：① 给整门课建立地图；② 学会文本如何变成模型处理的 token。
# 阅读目标：能够解释“数据—表示—模型—训练—系统—评估”如何相互影响，
# 并能手算一次 BPE 训练、说明它如何编码未见文本，而不只是认得术语。
#
# 【阅读路线】Python 文件的定义顺序不等于讲课顺序。以 main() 为入口：
#   welcome → why_this_course_exists → current_lm_landscape → 讲义机制/课程安排
#   → course_syllabus（依次调用 basics / systems / scaling_laws / data / alignment）
#   → tokenization（依次比较 character / byte / word / BPE）。
# 阅读分词演示时，遇到 Tokenizer、merge、train_bpe 等名字，再跳到相应定义。
# 首遍先理解每节“解决什么问题”；第二遍手算 BPE；第三遍用自测检查理解。
# 架构与系统部分是后续课程的地图，不要求在第一讲掌握每个变体的实现。
#
# 【先修知识】默认能读 Python 的循环、字典、类和函数，知道矩阵乘法、导数
# 与概率的基本含义。下面会就地补充条件概率、损失、梯度、编码等必需背景。
# 若矩阵/求导本身还陌生，可按需查教材；不必为了读第一讲先补完另一门课程。
#
# 【标记约定】“背景/推导/例子/易错点/自测”是自学补充，不冒充教师原话。
# “课堂补充 + 时间”是依据英文字幕整理的中文概括，时间对应下面的 2026 视频。
# 原英文、代码、链接和 @inspect 等标记保留；中文均为 # 注释，直接阅读源码
# 即可。注释不新增 text() 展示节点，已有网页 trace 也不会因此自动更新。
#
# 【资料与版本】以本地原讲稿为骨架；课程中的“今天/今年/下周”属于授课时点，
# 模型名单、硬件指标、作业安排属于当时背景，不当作持续更新的产品/课程说明。
# 官方课程：https://cs336.stanford.edu/
# 官方讲稿：https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py
# 官方视频：https://www.youtube.com/watch?v=JuoVZkPBiKk
# 字幕镜像：https://jingzechen.github.io/assets/courses/stanford-cs336-s26/lectures/001/captions.source.srt
# 字幕由第三方保存，未确认经过教师审校；术语和代码结果以原始资料交叉核对。
# 延伸教学：Karpathy 的 tokenizer 视频（见 tokenization()）及 minbpe 项目。
# 调查过的学习路线：https://github.com/lvy010/AI-wiki/tree/main/cs336-lab
# 该页面当时主要是先修建议与简短作业经验，未将其当作逐讲内容的依据。
# 各技术主题的原始论文已由英文 link(...) 引用；新增关键来源在对应注释中。
# ============================================================================

import os

import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
from edtrace import link, text, image
from lecture_util import article_link, post_link, video_link, get_local_url
from references import shannon_1950, lstm_1997, brants_2007, bengio_2003, glorot_2010, seq2seq_2014
from references import bahdanau_2015_attention, transformer_2017, gpt2_2019, t5_2019, kaplan_scaling_laws_2020, mup_2022
from references import dpo_2023, adamw_2017, adam_2014, grpo, ppo_2017, muon_2024
from references import large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, moe_2017, switch_transformers_2021, auxfree_2024, mtp_2024
from references import megatron_lm_2019, shazeer_2020, elmo_2018, bert_2018
from references import rms_norm_2019, layernorm_2016, pre_post_norm_2020, qk_norm_2023
from references import rope_2021, soap_2024, sparse_transformer_2019, gqa_2023, mla_2024
from references import linear_attention_2020, mamba_2_2024, gdn_2024, mamba_3_2026
from references import megabyte_2023, byt5_2021, blt_2024, tfree_2024, hnet_2025, sennrich_2016, zero_2019, gpipe_2018
from references import regmix_2025, olmix_2026, wrap_2024

from references import gpt_3_2020, gpt_4_2023, instruct_gpt_2022
from references import the_pile_2020, gpt_j_2021, opt_175b_2022, bloom_2022, palm_2022, chinchilla_2022
from references import llama_2023, llama_2_2023, llama_3_2024
from references import mistral_7b_2023, mixtral_2024
from references import deepseek_67b_2024, deepseek_v2_2024, deepseek_v3_2024, deepseek_r1_2025, deepseek_v3_2_2025
from references import qwen_2_5_2024, qwen_3_2025
from references import kimi_1_5_2025, kimi_k2_5_2026
from references import glm_4_5_2025, glm_5_2026
from references import minimax_m2_5_2026
from references import xiaomi_mimo_v2_2026

from references import marin_8b_2025, marin_32b_2025
from references import olmo_7b_2024, olmo_2_2025, olmo_3_2025
from references import nemotron_15b_2024, nemotron_3_2025

import tiktoken

def main():
    # 【总问题】在给定数据、时间、显存和算力预算下，怎样得到更好的语言模型？
    # “更好”需要先选评估目标；“更大”只是可能的手段，并非目标本身。
    welcome()
    why_this_course_exists()
    current_lm_landscape()

    what_is_this_program()

    course_logistics()
    course_syllabus()

    tokenization()  # First unit

    text("Next time: resource accounting")


def welcome():
    # 【课堂补充 00:02:33–00:02:58】from scratch 也有边界：课程挑选亲手实现
    # 收益较高的部分，不要求重新实现操作系统、GPU 驱动或整个自动微分框架。
    # 自学时可用“我能否解释接口、数据流和代价”判断掌握程度，而非只看代码量。
    text("## CS336: Language Models From Scratch (Spring 2026)"),

    image("images/course-staff.png", width=600)
    text("...bringing you the 3rd offering of CS336.")

    text("Lectures from 2nd offering (Spring 2025) are on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_).")
    text("What's new?")
    text("- Same 'from scratch' philosophy")
    text("- Prioritize high value-per-time concepts, don't lose the forest for the trees")
    text("- More coverage of modern LM ingredients (mixture of experts, long-context, agents)")


def why_this_course_exists():
    # 【背景】预训练：用大规模通用数据学习参数；微调：在已有参数上继续训练；
    # prompting：把任务放进输入，在通常的推理过程中不改变模型参数。
    # 抽象层越高，使用者要处理的细节越少，也越难控制底层训练数据、结构与算子。
    # “抽象泄漏”的例子：想让模型按字符做操作，但实际输入被切成了多字符 token；
    # 只改变提示词不能任意改变这套底层表示。理解底层才能提出更多研究方案。
    text("## Why did we make this course?")

    text("Problem: researchers are becoming **disconnected** from the underlying technology.")
    text("- 2016: researchers implemented and trained their own models.")
    text("- 2018: researchers downloaded models (e.g., BERT) and fine-tuned them.")
    text("- Today: researchers prompt API models (e.g., GPT/Claude/Gemini).")

    text("Moving up levels of abstraction boosts productivity, but")
    text("- These abstractions are leaky (in contrast to programming languages or operating systems).")
    text("- There is still fundamental research to be done that requires tearing up the stack.")

    text("**Full understanding** of this technology is necessary for **fundamental research**.")

    text("Philosophy of this course: **understanding via building**.")
    text("But there's one small problem...")

    text("## The industrialization of language models")
    image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Industrialisation.jpg", width=400)

    text("Frontier models are really expensive:")
    text("- 2023: GPT-4 supposedly cost $100M to train. "), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    text("- 2025: xAI builds cluster with 230K GPUs for training Grok. "), article_link("https://x.com/elonmusk/status/1947701807389515912")

    text("There are no public details on how frontier models are built.")
    text("From the GPT-4 technical report "), link(gpt_4_2023), text(":")
    image("images/gpt4-no-details.png", width=600)

    text("Frontier models are out of reach for us.")
    text("We could build small language models (<1B parameters), but this might not be representative of large language models.")

    text("Example 1: fraction of FLOPs spent in attention versus MLP changes with scale. "), post_link("https://x.com/stephenroller/status/1579993017234382849")
    # 【推导】设序列长度为 T、隐藏维度为 d。标准 dense Transformer 的线性
    # 投影/MLP 工作量大致随 T*d² 增长，attention 的两次序列交互矩阵乘法
    # 大致随 T²*d 增长。二者比例受 T/d 影响；放大宽度与拉长上下文不是同一件事。
    # 因而图中的某个 FLOPs 百分比不能套在所有模型上。MLP 是逐位置的前馈网络；
    # attention 则让不同位置交换信息，后面 basics() 会把二者放回整体结构。
    # 若瓶颈从 attention 变成 MLP，只优化 attention 的总收益就会减小。
    image("images/roller-flops.png", width=400)
    text("Example 2: emergence of behavior with scale "), link("https://arxiv.org/pdf/2206.07682")
    image("images/wei-emergence-plot.png", width=600)
    # 【读图提示】“涌现”在这里指某些任务分数到较大规模才明显上升的观察。
    # 图表呈现还受任务和计分方式影响，不能据此推出“所有能力都有固定突变阈值”。
    # 本讲要保留的结论是：小规模没观察到某现象，不足以排除它在大规模出现。

    text("## What can we learn in this class that transfers to frontier models?")
    text("There are three types of knowledge:")
    text("- **Mechanics**: how things work (what a Transformer is, how model parallelism works)")
    text("- **Mindset**: squeezing the most out of the hardware, taking scaling seriously")
    text("- **Intuitions**: which data and modeling decisions yield good accuracy")

    text("We can teach mechanics and mindset (these do transfer).")
    text("We can only partially teach intuitions (do not necessarily transfer across scales).")

    text("## Intuitions? 🤷")
    text("Some design decisions are simply not (yet) justifiable and just come from experimentation.")
    text("Example: Noam Shazeer paper that introduced SwiGLU "), link(shazeer_2020)
    image("images/divine-benevolence.png", width=600)
    # 【图片释义】SwiGLU 论文结尾用幽默方式承认，实验效果好并不等于已有完整
    # 理论解释。机制解释、经验结果、跨规模的猜测，要作为三类不同的知识保存。

    text("## The bitter lesson")
    text("Wrong interpretation: scale is all that matters, algorithms don't matter.")
    text("Right interpretation: algorithms that scale are what matter.")
    text("### accuracy = efficiency x resources")
    # 【易错点】这是组织思路的口号，不是可直接代入数值的精确线性定律。
    # 准确率不会随着 GPU 数量无限线性增长；真实收益还受数据质量、优化和任务限制。
    # 【背景】The Bitter Lesson 是 Sutton 对 AI 研究历史的总结：能够利用不断
    # 增长的计算资源的通用方法往往取得长期优势。这里强调“可扩展的算法”仍很重要。
    # 原文：http://www.incompleteideas.net/IncIdeas/BitterLesson.html
    text("In fact, efficiency is way more important at larger scales (can't afford to be wasteful).")
    link("https://arxiv.org/abs/2005.04305"), text(" showed 44x algorithmic efficiency on ImageNet between 2012 and 2019.")

    text("Framing: what is the best model one can build given a certain compute and data budget?")
    text("In other words, **maximize efficiency**!")
    # 【自测】为什么一个更慢但训练充分的模型可能输给一个较小的模型？
    # 解析：若训练时间固定，前者处理的数据可能更少；比较必须同时说明预算与指标。
    # 接下来的历史、系统和分词，都围绕这个“带资源约束的设计问题”展开。


def current_lm_landscape():
    # 【背景】语言模型给文本序列分配概率，而不仅是输出一个回答。
    # 对 token 序列 x_1,...,x_T，链式法则给出：
    #   P(x_1,...,x_T) = ∏_{t=1}^T P(x_t | x_1,...,x_{t-1})。
    # “预测下一个 token”就是逐步建模这些条件概率；需要完整可变长序列概率时，
    # 还要约定序列开始、结束等边界。basics() 会把它连接到训练目标。
    # 这里是教学时间线，不是严格的发明优先权列表；例如 LSTM 论文早于 2010 年。
    text("## Pre-neural (before 2010s)")
    text("- Language model to measure the entropy of English "), link(shannon_1950)
    text("- N-gram language models (used in machine translation and speech recognition systems) "), link(brants_2007)
    # 【联系】n-gram 用最近 n-1 个 token 近似全部历史，再通过计数估计概率。
    # 例如 trigram 用 P(今天 | 我, 觉得)；有限窗口忽略更远的信息，罕见组合也
    # 缺乏足够计数，需要平滑等处理。神经模型通过共享参数缓解组合稀疏问题。
    # 熵刻画平均不确定性；语言模型越能利用上下文，通常越能压低对真实文本的
    # 平均负对数概率。这是“预测”与“编码长度/压缩”之间的信息论联系。

    text("## Neural ingredients (2010s)")
    text("- Long-Short Term Memory (LSTM) "), link(lstm_1997)
    text("- First neural language model "), link(bengio_2003)
    text("- Sequence-to-sequence modeling (for machine translation) "), link(seq2seq_2014)
    text("- Adam optimizer "), link(adam_2014)
    text("- Attention mechanism (for machine translation) "), link(bahdanau_2015_attention)
    text("- Transformer architecture (for machine translation) "), link(transformer_2017)
    text("- Mixture of experts "), link(moe_2017)
    text("- Model parallelism "), link(gpipe_2018), link(zero_2019), link(megatron_lm_2019)
    # 【背景】这组名字分别补上了建模、优化与系统的不同环节：
    # - 神经语言模型用向量表示和共享参数预测词；Bengio 2003 是前馈网络，非 LSTM。
    # - RNN 递归更新状态 h_t=f(h_{t-1},x_t)；LSTM 用门控改善长期信息与梯度的传递。
    # - seq2seq 把输入序列映射到输出序列。早期单向量压缩是信息瓶颈；attention
    #   允许输出的不同位置按需读取不同输入位置，Transformer 进一步以 attention
    #   为核心组织网络。它不意味着训练时可看到未来答案，因果模型仍需遮罩。
    # - Adam 解决怎样更新参数；模型并行解决单张设备难以容纳或计算整个模型的问题。
    # 这些进展相互配合；只扩大网络，而不处理训练稳定性和硬件限制，并不足够。

    text("## Early foundation models (late 2010s)")
    text("- ELMo: pretraining with LSTMs, fine-tuning improves downstream tasks "), link(elmo_2018)
    text("- BERT: pretraining with Transformer, fine-tuning improves downstream tasks "), link(bert_2018)
    text("- Google's T5 (11B): cast everything as text-to-text "), link(t5_2019)
    # 【背景】foundation model 指在广泛数据上训练、可适配多种任务的基础模型。
    # ELMo 用双向语言模型产生上下文相关表示；BERT 常用双向 encoder 做掩码预测，
    # 不是直接按本课 decoder-only 的方式从左到右生成。T5 则把任务统一为文本输入
    # 与文本输出。这里的联系是“共享预训练所得能力”，不代表模型结构与损失相同。

    text("## Embracing scaling")
    text("- OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot "), link(gpt2_2019)
    text("- Scaling laws: provide hope / predictability for scaling "), link(kaplan_scaling_laws_2020)
    text("- OpenAI's GPT-3 (175B): in-context learning "), link(gpt_3_2020)
    text("- Google's PaLM (540B): massive scale, undertrained "), link(palm_2022)
    text("- DeepMind's Chinchilla (70B): compute-optimal scaling laws "), link(chinchilla_2022)
    # 【背景】B=10^9，T（数量后缀）=10^12；参数量是可学习数值的个数，不是文件字节数。
    # zero-shot：提示中没有任务示范；few-shot：提示中放少量示范。
    # in-context learning：通过上下文示范影响预测，通常不做梯度更新；与微调区分。
    # “undertrained”在这里是相对于预算分配而言：可能把预算过多用于扩大参数量，
    # 导致训练 token 不足。并非说该模型完全没有学到知识；详见 scaling_laws()。

    text("## Open models")
    text("Early attempts (attempts to replicate GPT-3):")
    text("- EleutherAI's open datasets (The Pile) and models (GPT-J) "), link(the_pile_2020), link(gpt_j_2021)
    text("- Meta's OPT (175B): GPT-3 replication, lots of hardware issues "), link(opt_175b_2022)
    text("- Hugging Face / BigScience's BLOOM (176B): focused on data sourcing "), link(bloom_2022)

    text("Credible open-weight models (weights + paper):")
    text("- Meta's Llama models "), link(llama_2023), link(llama_2_2023), link(llama_3_2024)
    text('- Mistral\'s models '), link(mistral_7b_2023), link(mixtral_2024)
    text("- DeepSeek\'s models "), link(deepseek_67b_2024), link(deepseek_v2_2024), link(deepseek_v3_2024), link(deepseek_r1_2025), link(deepseek_v3_2_2025)
    text("- Alibaba\'s Qwen models "), link(qwen_2_5_2024), link(qwen_3_2025)
    text("- Moonshot's Kimi models "), link(kimi_1_5_2025), link(kimi_k2_5_2026)
    text("- Z.ai's GLM models "), link(glm_4_5_2025), link(glm_5_2026)
    text("- Minimax\'s models "), link(minimax_m2_5_2026)
    text("- Xiaomi's MIMO models "), link(xiaomi_mimo_v2_2026)
    text("These models are approaching closed models (GPT, Claude, Gemini, etc.).")

    text("Open-source models (weights + paper + code + data):")
    text("- AI2's Olmo models "), link(olmo_7b_2024), link(olmo_2_2025), link(olmo_3_2025)
    text("- NVIDIA's Nemotron models "), link(nemotron_15b_2024), link(nemotron_3_2025)
    text("- Marin's models (open development) "), link(marin_8b_2025), link(marin_32b_2025)

    text("Openness is important for trust and innovation "), link("https://arxiv.org/abs/2403.07918")
    text("Ideas from open models enable us to teach CS336.")
    # 【联系】拿到权重能运行/继续训练，拿到训练代码能检查算法，拿到数据及配比才
    # 更接近解释模型从何学来。完整复现还依赖版本、超参数、随机性和硬件等细节。
    # 此处 openness 是公开程度的教学分类；具体使用许可要看各项目自身条款。

    text("What is a language model?")
    text("- 2018 (BERT): something you fine-tune")
    text("- 2020 (GPT-3): something you prompt")
    text("- 2022 (ChatGPT): something you talk to "), link(title="example conversation", url="https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/all/train?row=72&conversation-viewer=72")
    text("- 2026 (agents): something that acts autonomously "), link(title="example trace", url="https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/viewer/default/train?conversation-viewer=1")

    text("The fundamentals are the same (attention, kernels, optimization).")
    text("The specs are different (longer context, inference efficiency matters even more).")
    # 【背景】agent 通常在“模型生成动作/工具调用 → 环境执行 → 返回观察”循环中工作。
    # 模型依然生成 token，但历史里增加了工具结果和多轮行动，推理次数与上下文变长。
    # 所以应用形式的变化会反过来要求更高效的长上下文处理与推理系统。
    # 【自测】在提示里加三个例子，等于训练三次吗？解析：不等于。前者改变输入，
    # 后者通常指用损失和梯度更新权重；“学会任务”的表象不能代替机制上的区分。


def what_is_this_program():
    # 【讲义机制】text()/image()/link() 给 edtrace 提供正文、图片和链接；它们不属于
    # 语言模型算法。函数调用形成讲义层级，@inspect 标记希望展示的中间变量，
    # @stepover 表示演示时跳过内部细节。普通 # 注释由 Python 忽略。
    # 下面求和只是展示追踪器如何让读者看到中间状态，和训练神经网络无关。
    text("This is an *executable lecture*, a program whose execution delivers the content of a lecture.")
    text("Executable lectures make it possible to:")
    text("- view and run code (since everything is code!),")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    text("- see the hierarchical structure of the lecture")


def course_logistics():
    # 【自学建议】这一节是课程组织信息，可快速读过。核心方法是先在小输入上验证
    # 正确性，再研究速度与规模；“跑通”与“学得好”与“跑得快”是三个不同的目标。
    # 【课堂补充 00:23:18–00:24:19】教师强调跟读之外要动手，先在本地实现和检查，
    # 再用集群做训练/性能实验。本讲分词小例子本身不需要 GPU。
    # 原文中的作业/资源说明保留授课时状态，实际完成作业时以对应版本的要求为准。
    text("All information online: "), link(title="course website", url="https://stanford-cs336.github.io/spring2026/")

    text("This is a 5-unit class.")
    text("Comment from Spring 2024 course evaluation:")
    text("> *The entire assignment was approximately the same amount of work as all 5 assignments from CS 224n plus the final project. And that's just the first homework assignment.*")

    text("## Why you should take this course")
    text("- You have an obsessive need to understand how things work.")
    text("- You want to build up your research engineering muscles.")

    text("## Why you should not take this course")
    text("- You actually want to get research done this quarter. (Talk to your advisor.)")
    text("- You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG, etc.). (You should take a seminar class for that.)")
    text("- You want to get good results on your own application domain. (You should just prompt or fine-tune an existing model.)")

    text("## How you can follow along at home")
    text("- All lecture materials and assignments will be posted online, so feel free to follow on your own.")
    text("- Lectures are recorded via [CGOE](https://cgoe.stanford.edu/).")

    text("## Assignments")
    text("- 5 assignments (basics, systems, scaling laws, data, alignment).")
    text("- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness.")
    text("- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed).")
    text("- Leaderboard for some assignments (minimize perplexity given training budget).")

    text("## AI policy")
    text("- Coding agents can solve all the assignments, but you won't learn anything.")
    text("- AI can be tremendously useful for answering questions and tutoring.")
    text("- You must use our provided AGENTS.md file, which asks the AI to be pedagogically-minded.")
    text("- Please read our [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0).")

    text("## Compute")
    text("- Thanks to [Modal](https://modal.com/) for providing compute. 🙏")
    text("- Please read the [guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef) on how to access and use the compute.")


def course_syllabus():
    # 【整课地图】五部分相互依赖，但分开学习能控制难度：
    # basics 给出一个可训练模型；systems 让同一预算做更多有效计算；scaling laws
    # 帮助决定规模和超参数；data 决定模型看见什么；alignment 调整模型怎样使用能力。
    # 接下来出现的陌生名词先抓“作用—代价—与相邻模块的关系”，不用逐个追论文。
    basics()         # Assignment 1: tokenization, model architecture, training
    systems()        # Assignment 2: kernels, parallelism, inference
    scaling_laws()   # Assignment 3: scaling laws
    data()           # Assignment 4: evaluation, curation, transformation, filtering, deduplication, mixing
    alignment()      # Assignment 5: RLHF, RL algorithms, RL systems

    text("Remember it's all about **efficiency**:")
    text("- Resources: data + hardware (compute, memory, communication bandwidth)")
    text("- How do you train the best model given a fixed set of resources?")

    text("Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.")
    text("- Systems: clearly about efficiency")
    text("- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures")
    text("- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)")
    text("- Data filtering: avoid wasting precious compute updating on bad / irrelevant data")
    text("- Scaling laws: use less compute on smaller models to do hyperparameter tuning")

    text("Tomorrow, we will become data-constrained...")


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    # 【接口】encode: str → list[int]；decode: list[int] → str。
    # int 是词表编号（token ID）；编号大小通常不代表词义、频率或向量之间的距离。
    # 模型通过 embedding 表把编号映射到向量，而非直接把这些整数当连续数值输入。
    # Python 说明：ABC 表达基类意图，但这里未用 @abstractmethod；未实现的方法
    # 被调用时才通过 NotImplementedError 报错，不会在创建对象时强制阻止实例化。
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def basics():
    # 【背景：先补齐模型从输入到学习的全过程】
    # 1. tokenizer 把文本分成 token，再转成整数序列，例如 [12, 37, 8]。
    # 2. embedding 矩阵 E 的形状是 [V, d]：V 是词表大小，d 是隐藏维度。
    #    每个编号查出一行向量，长度 T 的输入由此变成 [T, d] 的矩阵。
    # 3. 多层网络按上下文更新这些向量，得到每个位置的表示 h_t。
    # 4. 输出投影把 h_t 变成 V 个 logits（未归一化分数）；softmax 将它们变成
    #    总和为 1 的概率。logits_j 大，不等于概率值本身大于 1。
    # 5. 用位置 t 的输出预测 x_{t+1}，根据正确答案的负对数概率计算 loss。
    # 6. 反向传播用链式法则求参数梯度；优化器据此更新参数，再处理下一批数据。
    # 推理时复用 1–4，再选取/采样下一个 token 并接到输入后；通常不计算参数梯度。
    # 这解释了为什么 tokenizer、architecture、training 必须共同构成一个闭环。
    text("Goal: be able to train a basic language model")
    text("Components: tokenization, model architecture, training")

    text("## Tokenization")
    text("What are the atoms that the model operates on?")
    text("Formally: a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens)")
    image("images/tokenized-example.png", width=600) 
    text("Popular tokenizer: **Byte-Pair Encoding** (BPE) "), link(sennrich_2016)
    text("Intuition: break input into frequently-occuring chunks")
    text("Efficiency lens")
    text("- Reduce context length (1000 bytes → ~250 tokens)")
    text("- Adaptive computation (more modeling capacity on interesting parts of input)")
    # 【课堂补充 00:28:41–00:29:15】常见字节块合成一个 token，罕见块留下更多
    # token，于是相同字节长度的不同文本占用不同数量的模型计算位置。
    # 【易错点】这是间接的计算分配，不是 tokenizer 已理解了“哪些内容更有意义”。
    # “1000 bytes → 约 250 tokens”只是例子；比例随语料、语言与词表变化。

    text("The dream: tokenizer-free model architectures, which operate directly on bytes "), link(byt5_2021), link(megabyte_2023), link(blt_2024), link(tfree_2024), link(hnet_2025)
    text("These are promising, but have not yet been scaled up to the frontier.")
    
    text("## Model architecture")
    text("Starting point: original Transformer "), link(transformer_2017)
    image("images/transformer-architecture.png", width=500)
    # 【图片释义】原始 Transformer 是翻译用的 encoder-decoder：encoder 处理源句，
    # decoder 一边看已生成的目标词，一边通过 cross-attention 读取源句表示。
    # 本课常见的自回归语言模型主要用 decoder-only：输入历史，预测后续 token。
    # 不要直接把原图的两整套堆栈照搬成所有 GPT 类模型的结构。
    #
    # 【最小 attention 背景】从输入表示 H 线性投影得到 Q、K、V（查询、键、值）。
    # attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V。
    # 行 i 的 Q 与各位置的 K 比较，权重归一化后混合那些位置的 V。
    # 因果 mask 使位置 i 不能使用未来位置；否则训练时会直接偷看预测目标。
    # 这里 V 表示 value 矩阵，和上文用于“词表大小”的 V 是不同语境中的记号。
    # 多头 attention 让多个投影空间分别提取关系，再合并；MLP 在每个位置上
    # 做非线性变换。残差连接把输入加回子层输出，给信息和梯度提供更直接的通路。
    # 原论文（架构图和公式）：https://arxiv.org/abs/1706.03762

    text("Refinements:")
    text("- Activation functions: ReLU, SwiGLU "), link(shazeer_2020)
    text("- Positional encodings: sinusoidal, RoPE "), link(rope_2021)
    text("- Normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm "), link(layernorm_2016), link(rms_norm_2019), link(qk_norm_2023), link(pre_post_norm_2020)
    text("- Attention: full, sparse/local attention, group-query attention (GQA), multi-head latent attention (MLA) "), link(sparse_transformer_2019), link(gqa_2023), link(mla_2024)
    text("- Recurrence/state-space models/linear attention: Mamba, Gated DeltaNet "), link(linear_attention_2020), link(mamba_2_2024), link(gdn_2024), link(mamba_3_2026)
    text("- MLP: dense, mixture of experts "), link(moe_2017), link(switch_transformers_2021)
    text("- Shape (hidden dimension, depth, number of heads, number of experts)")
    # 【按功能理解这些 refinement】
    # - 表达能力：激活函数让多层线性变换不至于合并为一次线性变换；ReLU 截断
    #   负值，SwiGLU 用一条带激活的支路门控另一条支路。
    # - 位置信息：attention 本身不自动知道绝对位置；sinusoidal/RoPE 等让位置
    #   进入计算。RoPE 在 Q/K 上施加与位置有关的旋转，影响相对位置的注意力分数。
    # - 数值尺度：LayerNorm/RMSNorm 控制表示的尺度；QK norm 作用于 Q/K。
    #   pre/post-norm 指归一化在子层之前/之后，影响梯度传播和训练稳定性。
    # - 上下文成本：local attention 限制可见位置；GQA 让若干 Q 头共享 K/V 头；
    #   MLA 用低维潜表示压缩 K/V 相关状态。减少 KV 存储与减少 T² 交互并非一回事。
    # - 状态压缩：递归、状态空间模型、线性 attention 尝试用不同结构汇总历史，
    #   避免完整 attention 的某些成本；它们是相关方向，不是完全等价的算法。
    # - 参数与算力：dense MLP 对每个 token 使用同一套参数；MoE 的 router 为每个
    #   token 选部分专家。总参数量可以很大，但单 token 只激活一部分，另有路由/
    #   通信成本。专家不是天然分配好“数学专家/代码专家”的人工角色。
    # 首遍能说明每类改动在改善什么即可，具体公式与实现会在架构课展开。

    text("## Training")
    text("How do you set the parameters of the model?")
    # 【推导：为什么是负对数概率】最大化训练文本的概率，相当于最大化各位置
    # 条件概率的乘积；取 log 把乘积变成求和，再加负号便于写成最小化问题：
    #   L = -(1/T) * Σ_t log p_theta(x_t | x_<t)。
    # 对 one-hot 目标，这就是平均交叉熵，也等于平均负对数似然（NLL）。
    # 若正确 token 概率从 0.1 升至 0.5，该位置 loss 从约 2.303 降至 0.693。
    # 反向传播计算 ∂L/∂theta；最简单的 SGD 更新是 theta ← theta - lr * 梯度。
    # lr 是学习率。参数由训练更新，学习率、层数等则是设计/调节的超参数。
    #
    # 【训练与生成的联系】teacher forcing 使用真实前文。例如输入 [A,B,C]，
    # 目标 [B,C,D]，一次前向计算各位置的预测；因果 mask 仍阻止使用后面的输入。
    # 生成时 B、C、D 尚未知，普通自回归采样要依次产生它们。因此“训练能并行”
    # 与“生成有顺序依赖”并不矛盾；prefill/decode 的系统差异由此而来。
    text("- Loss function (e.g., multi-token prediction) "), link(mtp_2024), link(deepseek_v3_2024)
    text("- Optimizer (e.g., AdamW, SOAP, Muon) "), link(adam_2014), link(adamw_2017), link(soap_2024), link(muon_2024)
    text("- Initialization scale (e.g., Xavier init, muP) "), link(glorot_2010), link(mup_2022)
    text("- Learning rate schedule (e.g., cosine, WSD) "), link(cosine_learning_rate_2017), link(wsd_2024)
    text("- Regularization (e.g., dropout, weight decay)")
    text("- Batch size (e.g., critical batch size) "), link(large_batch_training_2018)
    text("- MoE specific: load balancing (e.g., aux-free) "), link(auxfree_2024), link(deepseek_v3_2024)
    # 【训练术语速查】每项都在改变优化过程，而不只是配置文件里的一个名字。
    # - MTP（multi-token prediction）增加预测更远 token 的训练信号；不是把未来
    #   正确答案作为当前预测的输入，也不自动消除生成时的顺序依赖。
    # - Adam 维护梯度的一阶、二阶矩估计来调整更新；AdamW 将权重衰减与这种
    #   自适应更新解耦。SOAP/Muon 是其他更新规则，本讲先理解“优化器负责怎么走”。
    # - 初始化设定起点和各层尺度；Xavier 类方法考虑输入/输出宽度，避免信号
    #   随层数快速放大或衰减。muP 同时考虑宽度变化时的参数化与学习率迁移。
    # - 学习率 schedule 规定步长随训练如何变化；warmup 逐渐增大，cosine 余弦
    #   衰减，WSD 划分 warmup/stable/decay 阶段。
    # - dropout 在训练中随机屏蔽部分激活；weight decay 将权重向零收缩。
    #   正则化是控制过拟合等问题的工具，并非所有大模型都使用相同设置。
    # - batch 是一次梯度估计所用的数据；语言模型中常按 token 数计量。
    #   增大 batch 可提高并行度，但达到一定规模后，优化收益不会等比例增加。
    # - MoE 负载均衡避免所有 token 涌向少数专家，造成其他专家缺少训练或设备闲置。

    text("## Assignment 1 (basics)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="PDF", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2026_assignment1_basics.pdf")
    text("- Implement BPE tokenizer")
    text("- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop")
    text("- Do resource accounting")
    text("- Train on TinyStories and OpenWebText")
    text("- Leaderboard: minimize OpenWebText perplexity given 45 minutes on a B200 "), link(title="last year's leaderboard", url="https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard")
    # 【推导：perplexity（困惑度，PPL）】若 loss 使用自然对数，PPL=exp(L)。
    # 它也等于正确 token 概率倒数的几何平均。若每一步都在 k 个可能项上均匀
    # 分配概率，则 L=log(k)，PPL=k；可用“有效不确定性”作直觉，但不是字面候选数。
    # 同一评估文本、tokenizer 和协议下，PPL 越低表示模型越能预测真实文本。
    # 不同 tokenizer 改变了预测单位与分母 T，per-token PPL 不能直接跨词表比较。
    # 需要跨 tokenizer 时，可考虑在相同原始文本上按字节归一化负对数似然，
    # 同时对齐上下文和边界约定；即便如此，这也不是所有实际任务能力的总分。

    text("High-level principle: everything is about balancing the following:")
    text("- Expressivity (can represent complex dependencies in the data)")
    text("- Stability (keep parameter and gradient norms in goldilocks zone)")
    text("- Efficiency (runs fast on hardware, both training and inference)")
    # 【自测】训练 loss 一直下降，就能断言聊天效果变好吗？
    # 解析：不能。还要看独立验证数据上的泛化、实际任务评估，以及输入/输出格式。
    # 【过渡】现在有了模型和学习规则，下一问是：硬件能多快执行这些计算？


def systems():
    # 【背景：为什么模型代码正确后还有一整门系统问题】数学上相同的计算，可以
    # 用不同的数据布局、执行顺序和设备分工完成，耗时却可能差很多。
    # 参数是持久权重，激活是本次前向产生的中间值，梯度是 loss 对参数的导数，
    # 优化器状态则如 Adam 的两组矩估计；训练显存必须考虑这几类内容。
    text("Goal: squeeze the most out of the hardware (GPU or TPU)")
    text("Components: kernels, parallelism, inference")

    text("## Basics")
    text("- Resource accounting: memory and compute characteristics of a model")
    # 【推导】下面的 N=70e9 是参数数目，D=1e12 是训练中处理的 token 总数。
    # 对一个稠密矩阵 W，前向 y=xW 的每个权重参与一次乘法和加法，约 2 FLOPs；
    # 反向还要算输入梯度和权重梯度，各有类似矩阵乘法，所以训练粗估约 6ND。
    # 常见约定把一次乘加算作 2 FLOPs；这是近似，忽略/简化了 attention 的
    # 序列交互、embedding、优化器等成本。长上下文、MoE、激活重计算需要另算。
    # D 统计处理次数：同一批语料重复训练会再次计入，不代表新增了同样多的信息。
    total_flops = 6 * 70e9 * 1e12  # Training 70B parameters on 1T tokens = 4.2e23 FLOPs @inspect total_flops 
    image("images/compute-memory.png", width=300)
    text("- Model parameters must be moved from memory (HBM) to the compute (SMs)")
    text("- Example: B200 can perform 2.25 PFLOP/sec (bf16) with 8TB/sec memory bandwidth")
    text("- Roofline analysis: understand whether we're compute-bound or memory-bound")
    text("- Benchmarking and profiling (nsight): see what happens in practice")
    # 【单位】FLOPs 是总运算量；FLOP/s 是运算速率；GB/TB 是容量；GB/s/TB/s 是带宽。
    # bf16 是每个数占 16 bit（2 bytes）的浮点格式；仅 N 个 bf16 权重就需 2N bytes，
    # 70B 对应约 140 GB（十进制），训练的梯度/状态/激活还没计入。
    # HBM 是容量较大的显存；SM 是 GPU 上执行线程的计算单元，内部还有更小、更快
    # 的存储。图的要点是：数值通常必须在存储层级间搬运才能参加计算。
    #
    # 【推导：Roofline】设某计算需 F 次浮点运算、搬运 M bytes，峰值算力 P、带宽 B：
    #   理想耗时下界约 max(F/P, M/B)；算术强度 I=F/M，单位 FLOPs/byte。
    #   运算速率上界约 min(P, B*I)。I < P/B 时更可能被带宽限制。
    # 例如读取一次权重后供多个样本复用，F 增加而权重读流量未同比增加，I 就提高。
    # 实际还会受启动开销、访存模式、通信、同步等限制，故需要实测。
    # 英文硬件数字只作讲课示意；峰值还依精度/稀疏性等口径变化，不能当持续吞吐量。
    # benchmarking 测整体时间/吞吐；profiling 进一步分解时间花在哪些操作上。
    # 延伸（原文推荐书）：https://jax-ml.github.io/scaling-book/roofline/

    text("[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html):")
    image("https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png", width=500)

    text("## Kernels")
    text("- Kernel is a function that runs on GPU")
    text("- When using PyTorch, each primitive operation launches a standard kernel")
    text("- Can write custom kernels to make GPUs go brrr")
    text("- Principle: organize computation to minimize data movement")
    text("- Naive: read HBM; compute A; write HBM; read HBM; compute B; write HBM")
    text("- Fused: read HBM; compute A and B; write HBM")
    text("- Strategies: operator fusion (matmul + activation), tiling (FlashAttention)")
    # 【例子】计算 y=relu(xW+b)，若每个阶段把中间结果写回 HBM，下个阶段还得
    # 再读一次。fusion 尽量在同一 kernel 中完成相邻阶段，减少中间读写与启动。
    # tiling 把矩阵分块，在较快的片上存储中复用数据；FlashAttention 将这种思想
    # 用于 attention，避免将完整 T×T 注意力矩阵写到 HBM，仍计算精确 attention
    # （浮点舍入可能不同），并非通过删除注意力连接变成稀疏近似。
    # 原论文：https://arxiv.org/abs/2205.14135
    # “每个 PyTorch 原语启动一个 kernel”是概念简化；view 等操作可能不计算，
    # 复杂操作可能对应多个 kernel，编译器又可能融合多个操作。
    text("- Warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfers")
    text("- Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens")
    # 【术语定位】CUDA 是 GPU 编程平台；Triton 支持以较高层形式编写 GPU kernel；
    # CUTLASS/ThunderKittens 提供组织高性能计算的工具。这里的 kernel 不是操作系统内核。
    # warp divergence：同组线程走不同分支，影响并行效率；memory coalescing：
    # 合并邻近线程的访存请求；bank conflict：片上存储访问冲突；occupancy：
    # 活跃 warp 相对硬件可支持数量的占用程度；异步搬运尝试重叠访存和计算。
    # 首遍知道这些都在影响“硬件能否持续有活干”即可，具体规则留到 GPU 课程。

    text("## Parallelism")
    text("- What if we have 1024 GPUs?")
    text("- Data movement between GPUs is even slower, but same 'minimize data movement' principle holds")
    text("- Use classic collective operations (e.g., gather, reduce, all-reduce)")
    text("- Shard memory (parameters, activations, gradients, optimizer states) across GPUs")
    text("- How to split computation: {data,tensor,pipeline,sequence,expert} parallelism")
    # 【背景】数据并行：不同设备看不同样本，再汇总梯度；张量并行：将某层矩阵
    # 运算切给多个设备；流水线并行：不同设备负责不同层；序列并行：沿序列等
    # 相关维度划分工作；专家并行：将 MoE 专家分散部署并路由 token。
    # gather 收集数据，reduce 做求和等汇总，all-reduce 让各参与设备都拿到汇总结果。
    # 分片可减少单卡存储，但可能新增通信；所以 1024 张卡不会自动快 1024 倍。
    
    text("## Inference")
    text("Goal: generate tokens given a prompt (needed to actually use models!)")
    text("Inference is also needed for reinforcement learning, test-time compute, evaluation")
    text("Two phases: prefill and decode")
    image("images/prefill-decode.png", width=500)
    text("- Prefill (similar to training): tokens are given, can process all at once (compute-bound)")
    text("- Decode: need to generate one token at a time (memory-bound)")
    # 【联系】prefill 的 prompt 已全部给定，可一次处理多个位置，同时缓存各层的 K/V。
    # decode 时当前层的新 Q 与缓存 K/V 交互，避免每步重算所有历史位置的 K/V。
    # 这是 KV cache：以显存换重复计算，其大小通常随 batch、上下文长度和层数增长。
    # 小 batch 的 decode 每次使用大量权重却只算少量新 token，权重复用不足，
    # 因而常受带宽限制；大 batch、很长上下文等条件会改变瓶颈。
    # 英文 compute-bound/memory-bound 描述典型情形，不是两阶段的无条件定义。
    # 模型推理中的 decode 是“生成 token”；Tokenizer.decode 则是“ID 还原文本”，
    # 同名但不同操作。tokenization() 会使用后一个含义。
    text("Methods to speed up decoding:")
    text("- Use cheaper model (via model pruning, quantization, distillation)")
    text("- Speculative decoding: use a cheaper \"draft\" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)")
    text("- Systems optimizations: fused kernels, continuous batching")
    # 【背景】pruning 去掉部分结构/参数；quantization 使用较低精度存储/计算；
    # distillation 让较小模型学习较大模型的行为。这些通常需要重新验证质量。
    # speculative decoding 先用便宜的 draft model 提议一段，再由目标模型验证；
    # “exact”依赖正确的接受/拒绝与采样校正规则，不能把高分草稿全部直接接受。
    # 分布一致也不意味着在同一随机种子下逐字结果必然相同。
    # continuous batching 在请求完成时移出、让新请求加入，减少等待与空闲。
    # 【自测】运算次数减少一半，耗时必然减半吗？解析：不一定；若仍需搬运相同
    # 数量的权重且受带宽限制，算力减少可能几乎不改变耗时。

    text("## Assignment 2 (systems)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment2-systems"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf")
    text("- Implement a fused RMSNorm kernel in Triton")
    text("- Implement distributed data parallel training")
    text("- Implement optimizer state sharding")
    text("- Benchmark and profile the implementations")

    text("Recommended book: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)")
    text("- Nicely lays out how to approach systems for LLMs conceptually")
    text("- From Google, so it foregrounds TPUs, but high-level concepts are similar")


def scaling_laws():
    # 【过渡】有了模型，也知道怎么高效执行，但大训练只能负担少数几次。
    # 因此要把“挑一个大模型配置”转成“在小预算下研究一套随规模变化的配置规则”。
    # scaling law 是经验规律：在某套数据/架构/训练协议下，loss 随资源如何变化。
    # 它不是保证任何模型放大都成功的自然定律，也不是对所有下游能力的直接预测。
    text("Setting: if you had 1e25 FLOPs of compute, what hyperparameters would you use to train a good model?")
    text("Too expensive to do hyperparameter tuning at full scale!")

    text("Key conceptual shift: instead of a single scale, think of a **scaling recipe** (FLOPs → hyperparameters)")
    text("For a scaling recipe:")
    text("- Run experiments to compute the loss at various smaller scales (e.g., up to 1e24 FLOPs)")
    text("- Fit a scaling law to predict the loss of the scaling recipe at the target scale (e.g., 1e25 FLOPs)")

    text("Now you can:")
    text("1. Optimize the scaling recipe targeting a larger scale using smaller scale experiments")
    text("2. Predict the loss at the target scale before actually running the experiment!")
    text("Scaling laws don't happen automatically, they require careful construction of a scaling recipe.")
    text("Parameterize the model in a way to get **hyperparameter transfer** "), link(mup_2022)
    text("Predictability is at least as important as optimality!")
    # 【例子】固定一个学习率在小模型上效果最好，并不能保证把宽度放大十倍后仍稳定。
    # scaling recipe 要给出“预算 → 模型宽度/深度、训练 token、batch、学习率等”的
    # 映射。muP 研究如何通过参数化与尺度规则改善宽度变化时的超参数迁移；
    # 它不是承诺任意架构、深度、数据变化都可直接沿用小模型设置。
    # 【课堂补充 00:47:35–00:49:11】教师强调要设计可外推的训练方案，避免大规模
    # 时学习率等设置突然变得不可预测；可靠预测与局部最优同样值得重视。

    text("Question: given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)?")
    text("Classic compute-optimal scaling laws: "), link(kaplan_scaling_laws_2020), link(chinchilla_2022)
    text("- ISOFLOP curves: for multiple small FLOPs budgets, find optimal N")
    text("- Then fit a scaling law to extrapolate to large FLOPs budgets")
    image("images/chinchilla-isoflop.png", width=800)
    # 【读图与推导】固定 C≈6ND 后，D≈C/(6N)，所以不能同时无限增加 N 和 D。
    # 一条等算力曲线：横轴改变 N，纵轴记录在对应 D 下得到的验证 loss。
    # 左侧模型太小可能表达不足；右侧模型太大则训练数据/更新不足，中间可能有最优点。
    # 对多个预算重复找最优点，拟合 N_opt(C)、D_opt(C)，再预测更大预算的配置。
    #
    # 【可选推导】一种常用拟合形式是 L(N,D)=E+A/N^alpha+B/D^beta。
    # E 是拟合中的渐近项；后两项分别描述有限参数量和有限训练数据带来的差距。
    # A、B、alpha、beta 从实验拟合，不是普适常数。代入 D=C/(6N) 后：
    #   L(N)=E+A*N^(-alpha)+B*(6N/C)^beta。
    # 对 N 求导并令其为零，可得 N_opt ∝ C^[beta/(alpha+beta)]；
    # 相应 D_opt ∝ C^[alpha/(alpha+beta)]。两指数接近时，二者约随 sqrt(C) 增长。
    # 这解释了“模型与数据一起扩大”的含义；具体常数比 20 仍来自特定实验。
    # 原论文：https://arxiv.org/abs/2203.15556
    text("TL;DR: D = 20 N is roughly optimal (e.g., 70B parameter model should be trained on ~1.4T tokens)")
    text("Caveat: this doesn't take into account inference costs (want a smaller model)")
    # 【适用边界】D≈20N 是训练算力最优的粗略经验，不是数据量上限或停止训练标准。
    # 例如 N=70×10^9、D=20N=1.4×10^12，对应 C≈5.88×10^23 FLOPs。
    # 若模型上线后要服务很多请求，总成本还包括大量推理；可能宁愿多花训练成本，
    # 把小模型训练得更充分，以降低后续每次请求的成本。
    # 同样，数据质量、重复程度、tokenizer、架构和目标评估改变后，比例也可能变化。
    # 【自测】同样 C 下令 N 翻倍，按此近似 D 变成多少？解析：D 减半。
    # 但 loss 是否更好，要看原来的预算分配位置，不能仅凭参数更多下结论。

    text("Live example from Marin "), post_link("https://x.com/percyliang/status/2034367256277533100")
    image("https://pbs.twimg.com/media/HDuErvvbsAAQ5Yt?format=jpg&name=4096x4096", width=600)
    text("Should be done training this week, should see how well we match the preregistered loss!")

    text("## Assignment 3 (scaling laws)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment3-scaling"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment3-scaling/blob/master/cs336_spring2025_assignment3_scaling.pdf")
    text("- We define a training API (hyperparameters → loss) based on previous runs")
    text("- Submit \"training jobs\" (under a FLOPs budget) and gather data points")
    text("- Fit scaling laws to the data points")
    text("- Submit extrapolated hyperparameters and loss predictions")
    # 【实验方法】拟合优度只说明观察区间内的表现；外推应保留更大规模的验证点，
    # 并报告残差、预测误差与假设。否则漂亮的曲线可能只是拟合了小范围噪声。
    # 下面的作业 API 用已有实验结果模拟昂贵训练；它返回一个数值并不意味着
    # 调用者真的在那一刻启动了相应规模的训练。
    text("- Leaderboard: minimize loss given FLOPs budget")


def data():
    # 【过渡】前面似乎把数据当作可直接取用的 D 个 token；现实中还要决定数据
    # 来自哪里、保留什么、怎样评估。架构决定能表达什么，数据决定练习了什么。
    # 同样的计算预算花在重复导航栏与高质量正文上，得到的学习信号并不一样。
    text("Question: What capabilities do we want the model to have?")
    text("Multilingual?  Good at conversation?  Agentic coding capabilities?")

    text("## Evaluation")
    text("What is the purpose of evaluation?")
    text("1. Internal: guide model development (smoothness across scales, relative performance matters)")
    text("2. External: measure absolute quality of a real use case (ecological validity matters)")
    text("Examples of evaluations:")
    text("1. Perplexity: ideally run on private documents not on Internet (avoid contamination)")
    text("2. Advanced use cases: GPQA, HLE, SWE-Bench, Terminal-Bench")
    text("LMs are general purpose, require a diverse set of evaluations!")
    # 【背景】内部评估常用于比较改动、选择 checkpoint，偏好低噪声且跨规模平滑
    # 的信号；外部评估关注真实使用是否成功。ecological validity 指评估条件与
    # 实际使用情境是否相符，例如会做选择题未必等于能修复真实仓库里的 bug。
    # GPQA/HLE 是高难知识/推理类基准，SWE-Bench/Terminal-Bench 更接近代码与
    # 工具操作任务。本讲只需知道不同评估覆盖的能力不同，不必记具体榜单成绩。
    #
    # 【易错点】contamination 指训练数据含测试题、答案或高度相似内容，可能使
    # “记住了”被误当成“泛化了”。测试集未公开可降低部分风险，但还要检查数据
    # 流向；不断依据测试成绩调参也会使测试集逐渐参与开发。
    # 常见划分：训练集用于更新参数，验证集用于选择设置，测试集用于最后评估。
    # 评估损失的推导见 basics()；不同 tokenizer 的 per-token PPL 不能直接比较。

    text("## Data curation")
    text("- Data does not just fall from the sky.")
    text("- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.")
    image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)
    text("- Appeal to fair use to train on copyright data? "), link("https://arxiv.org/pdf/2303.15715.pdf")
    text("- Might have to license data (e.g., Google with Reddit data) "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- Raw data is HTML, PDF, directories (not text), requires processing")
    # 【背景】“来源”与“内容类型”都重要。网页含正文、广告、导航和脚本；PDF 可能
    # 有分栏、页眉与公式；代码仓库包含多个文件及依赖关系，简单拼接会丢失结构。
    # The Pile 图说明语料可来自多种来源，目的在于理解多样性，不是要求照搬其配比。
    # 上面的版权句子是在提出数据来源问题，不是在给出普遍适用的授权结论。

    text("## Data processing")
    text("- Transformation: convert HTML/PDF to text (extract main content)")
    text("- Filtering: keep high quality data, remove harmful content (via classifiers)")
    text("- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash")
    text("- Data mixing: how much to upweight/downweight each source? "), link(regmix_2025), link(olmix_2026)
    text("- Rewriting / synthetic data: use LM to augment real data, more similar to downstream tasks "), link(wrap_2024)
    # 【流水线例子】抓取网页 → 抽取正文 → 质量/语言筛选 → 去重 → 按来源采样
    # → tokenizer 编码 → 训练样本。实际顺序可能调整，各步都影响后面的数据分布。
    # 过滤器只是某种“质量标准”的近似，过滤过严也可能丢掉小众语言或重要知识。
    # Bloom filter 紧凑记录“可能出现过”的项，有假阳性，适合辅助精确重复检测；
    # MinHash 估计集合的 Jaccard 相似度，常将文档表示成词/字符片段集合来找近重复。
    # 它们不是可以无条件互换的两个去重算法，近重复判定还依赖阈值与分片方法。
    #
    # 【数据配比】假设按 token 采样时给代码 30%、普通文本 70%，模型受到的学习
    # 信号就偏向这种混合。提高某来源权重不是创造新内容，可能增加重复和过拟合。
    # 合成数据还需检查正确性、多样性和目标匹配程度；语言通顺不等于事实正确。

    text("Types of data:")
    text("- Pretraining data: large and diverse")
    text("- Mid-training data: high quality, including long-context")
    text("- Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)")
    # 【阶段关系】预训练通常用大量文本做下一 token 预测；mid-training 常指
    # 转向高质量、领域或长上下文数据的一段继续训练，命名并无完全统一边界。
    # SFT（监督微调）使用示范回答/行动轨迹，损失常只计算需要模型输出的位置；
    # post-training 还可包含后面介绍的偏好学习和强化学习，不只 SFT 一项。
    # 【自测】删掉重复文档就是永远不再重复训练任何 token 吗？解析：不是。
    # 去重控制语料冗余；后续训练可以多轮采样，但有效新信息和处理 token 数要区分。

    text("## Assignment 4 (data)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment4-data"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf")
    text("- Convert Common Crawl HTML to text")
    text("- Train classifiers to filter for quality and harmful content")
    text("- Deduplication using MinHash")
    text("- Leaderboard: minimize perplexity given token budget")


def alignment():
    # 【背景】“能接续互联网文本”与“能按用户意图回答”是不同目标。SFT 提供
    # 示范，而偏好/奖励提供“哪种回答更好”的信号，用来进一步塑造行为。
    # 这里的 full supervision 是说每个训练位置都有文本中的下一 token 作标签；
    # 标签通常由文本自身构造，因此预训练也称 self-supervised（自监督）。
    # weak supervision 在本节泛指较间接的评价信号，不代表评分一定不可靠：
    # 程序测试等 verifier 可以对特定目标给出很明确的结果。
    text("So far, we have trained a model on full supervision (predict the next token).")
    text("Now that the model should be reasonable, we can improve it further from **weak supervision**.")
    text("Why weak supervision?  When it is easier to critique than to generate.")

    text("Basic template:")
    text("1. Generate responses from the model.")
    text("2. Score responses with a {human, verifier, LM judge}.")
    text("3. Update the model to prefer better responses.")
    # 【例子】对于同一道题，直接写出标准解答可能很难，但检查最终答案或运行
    # 程序测试较容易。模型产生候选，评分器提供奖励，训练提高较好候选的概率。
    # 在 RL 语言中，模型是 policy（策略），前文/环境观察是状态信息，生成 token
    # 或工具调用是动作，整段生成/交互是 rollout（采样轨迹）。
    # 评分只是目标的代理：通过测试不保证所有场景正确，迎合评委也不等于实际有用。

    text("Algorithms:")
    text("- Proximal Policy Optimization (PPO) from reinforcement learning "), link(ppo_2017), link(instruct_gpt_2022)
    text("- Direct Policy Optimization (DPO): for preference data, simpler "), link(dpo_2023)
    text("- Group Relative Preference Optimization (GRPO): remove value function "), link(grpo)
    # 【勘误】保留英文原文供对照，但正式展开应为：
    # DPO = Direct Preference Optimization（不是上面的 Direct Policy）；
    # GRPO = Group Relative Policy Optimization（不是上面及作业列表中的 Preference）。
    # 原始论文：https://arxiv.org/abs/2305.18290 （DPO）
    #           https://arxiv.org/abs/2402.03300 （DeepSeekMath / GRPO）
    #
    # 【三种方法先理解到这里】PPO 用采样轨迹与奖励更新策略，通常配合 value model
    # 估计基线，并限制单次策略变化。DPO 可直接使用“同一问题下较好/较差回答”的
    # 偏好对与参考策略构造损失，常见离线形式不要求训练时在线生成或单独训练奖励模型。
    # GRPO 对同一问题采样一组回答，用组内奖励构造相对优势信号，省去单独的
    # learned value model；“省去 value model”不等于不需要奖励或参考策略约束。
    # 这也说明上面的“生成—评分—更新”是总体思路，不是每种算法每步都在线生成。

    text("Challenges:")
    text("- RL algorithms are unstable and hard to tune")
    text("- At scale, this requires a lot of new infrastructure (inference with async rollouts)")
    text("- Constantly trading off systems efficiency and on-policyness")
    # 【联系】on-policy 希望训练样本来自当前（或足够接近的）策略。异步系统中，
    # 生成服务可能还用旧权重，而训练端已更新数次；模型不匹配会影响更新的性质。
    # 等所有生成任务完成再更新可减轻滞后，却可能让设备空闲，所以这里同时有
    # 算法与系统权衡。前面 inference 的速度优化因此也会影响 RL 训练的成本。
    # 【自测】一个模型只做了下一 token 预训练，为什么不一定按问答格式工作？
    # 解析：它学习的是训练文本分布；任务格式与行为偏好需要合适的数据/目标来塑造。

    text("## Assignment 5 (alignment)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment5-alignment"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf")
    text("- Implement Direct Preference Optimization (DPO)")
    text("- Implement Group Relative Preference Optimization (GRPO)")


############################################################
# Tokenization

def tokenization():
    # 【过渡：把全课的资源问题落到第一个具体设计上】模型按 token 分配计算位置，
    # 所以切分规则既影响学习单位，也影响上下文能装多少内容和计算量。
    # 本节围绕三个问题：能否无损还原？词表多大？同一文本要多少 token？
    # “无损”是表示性质，不是说语言模型一定能理解或正确生成所有被编码的文本。
    text("This unit was inspired by Andrej Karpathy's video on tokenization; check it out! "), video_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    text("Summary:")
    text("- Tokenizer: strings ↔ tokens (indices)")
    text("- Character-based, byte-based, word-based tokenization are highly suboptimal")
    text("- BPE is an effective heuristic that is data-driven")
    text("- Tokenization is a separate step, maybe one day do it end-to-end from bytes...")

    text("But whatever solution needs to satisfy:")
    text("1. Model (e.g., Transformer) should operate on chunks (abstractions) of the sequence (text, video, DNA, etc.)")
    text("2. Chunks should be variable (allocate more model capacity to interesting chunks)")
    # 【比较总结】下表中的“可逆”指本文件的示例和有效输入，仍受具体实现边界限制。
    # 方案        基本单位           主要优势                 主要代价
    # character   Unicode 码点       容易解释和还原           码点表大、罕见项多
    # byte        UTF-8 字节         256 个基本值覆盖字节     文本序列较长
    # word        单词/文本块        常见词一个单位           未知词、长尾词表、多语言边界
    # byte BPE    学得的字节片段     词表与长度可折中         依赖语料/规则，不保证语义边界
    # 这里的“suboptimal”是在当前常见模型与资源约束下的判断，不是所有任务的定理。
    # byte-level BPE 的“训练”是离散计数与合并；常规 LM 训练时词表已经固定，
    # 它与更新神经网络参数是不同过程。改变 tokenizer 通常也会牵动模型的输入/输出表。
    #
    # 【本讲自测：先回答再看解析】
    # 1. 未见过的中文字符串能用 byte BPE 编码，是否意味着模型懂中文？
    #    解析：不意味着。字节覆盖保证可表示，能力取决于训练数据与模型。
    # 2. 词表越大是否一定更好？解析：可能缩短序列，但 embedding/输出投影变大，
    #    罕见 token 的学习信号也更少；最终要联合比较质量、显存和吞吐。
    # 3. 为什么下一讲讲资源核算？解析：本讲提出的“序列短更省、词表大更贵”需要
    #    用实际张量形状、FLOPs 和内存流量量化，才能指导工程选择。


class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    # 【代码导读】map(ord, string) 对字符串逐码点应用 ord；list 将迭代结果收集起来。
    # 本教学版本直接拿 Unicode 码点作 ID，没有另建连续、紧凑的词表映射。
    # 注意是“码点”，不一定是屏幕上一个可见字符，例子见 character_tokenizer()。
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    # 【两层编码】str.encode('utf-8') 是字符编码：str → bytes；本类的 encode
    # 是分词接口：str → list[int]。迭代 bytes 时已得到 0..255 的整数，下面的
    # map(int, ...) 不改变其数值。以 byte 值直接作 ID 时，词表天然有 256 项。
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        # 【边界】bytes(indices) 要求每项在 0..255；随后 UTF-8 解码要求整串字节
        # 格式合法。encode 的有效文本输出满足这点，但任意整数列表未必满足。
        # 例如 [240,159,140,141] 可还原 🌍，只有 [240] 则是截断的 UTF-8 序列。
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    # 【原理】这是一次针对指定 pair 的从左到右、非重叠替换；不是重新挑选 pair，
    # 也不是在同一轮递归合并新产生的 token。输入列表不变，返回新列表。
    # 例：merge([97,97,97], (97,97), 256) → [256,97]。
    # 相邻计数能看见两个 (97,97)，但两个匹配共享中间位置，不能同时替换。
    # 【循环不变量】indices[:i] 已被完整消费，其表示的字节恰等于 new_indices
    # 对应的字节；若新 token 的 vocab 内容是两项的拼接，合并就不会丢失字节。
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            # i+1 的检查先防止越界；匹配成功消费两个旧位置，故 i 前进 2。
            new_indices.append(new_index)
            i += 2
        else:
            # 不匹配就原样保留当前位置；落单的最后一个 token 也走这个分支。
            new_indices.append(indices[i])
            i += 1
    return new_indices


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    # 【两张表】vocab 回答“ID 表示哪些字节”，merges 回答“哪两个 ID 合并为新 ID”。
    # 仅有 vocab，虽然可以 decode，却不足以确定 encode 采用哪一种合法切分。
    # merges 的插入次序保存训练得到的优先级，不能把它随意排序或当无序集合。
    # dataclass 自动生成构造等方法；frozen=True 禁止重新赋值字段，但不会深度冻结
    # 字段里的字典。这里按约定将训练完成的参数当作只读，而非靠 frozen 保证一切。
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index



class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    # 【阅读路线】先看 train_bpe() 产生了什么，再看本类如何复用这些结果。
    # 训练一次、编码多次：编码新文本时不更新词表，也不重新统计语料频率。
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        # 初始序列由 bytes 构成，每个字节都有兜底 ID，未见词不会导致无处编码。
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        # 【为什么顺序重要】按 dict 插入顺序应用训练得到的规则；本例先 t+h→th，
        # 再 th+e→the，最后 the+空格→the空格。先尝试后两条时，它们的输入还不存在。
        # 不能把这理解成“在新文本里找当前最常见 pair”或“总选最长词表字符串”。
        # 这是原讲稿的简化顺序扫描版本；优化/替换实现时要明确合并优先级、
        # 非重叠规则和预分词边界，不能只比较最后的 decode 是否相同。
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)  # @stepover
        return indices

    def decode(self, indices: list[int]) -> str:
        # 【原理】vocab 已保存合并后的完整字节串，因此不需要逆序“拆开”合并树。
        # 先拼接 bytes，再统一 UTF-8 解码。单个 token 可能只有字符的一部分字节，
        # 逐 token 解码再拼字符串会失败，或在宽松解码下产生替代字符而丢失信息。
        # 若传入不在词表里的 ID，dict.get 返回 None，join 会报错；即便所有 ID
        # 都在词表里，拼接结果仍可能是非法 UTF-8。此教学版本未增加输入校验。
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:  # @inspect string indices
    """Given `string` that has been tokenized into `indices`, return the number of UTF-8 bytes per token.."""
    # 【度量】R=原始 UTF-8 字节数/token 数。对于同一份非空文本，R 越大，序列越短。
    # 分母只用 len，所以 word_tokenizer() 传入字符串块列表也能演示；类型注解
    # list[int] 并不会在运行时强制检查。它同样不会检查这些 token 是否真的表示原文。
    # 【易错点】这不是磁盘上的压缩比：token ID 若按 int32 存储，每个占 4 bytes，
    # 还未计词表等开销；R=2 时序列虽短，ID 数组反而可能比原 UTF-8 更大。
    # 这里优化的是模型计算位置的数量。空列表导致除以零，原演示只用非空文本。
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt5_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # 【版本边界】函数名来自原讲稿；实际固定加载的是 o200k_base 这套 encoding。
    # 它不是根据在线模型自动查询 tokenizer，也不等于整个聊天消息的序列化协议。
    # o200k_base 定义还包含预分词 regex、可合并字节串及特殊 token；学 BPE 时
    # 需要将“核心合并算法”与这些工程规则区分。
    # 定义：https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    return tiktoken.get_encoding("o200k_base")


def intro_to_tokenization():
    # 【背景：四个层次不要混用】
    # 可见字符（读者眼中的图形） → Unicode 码点序列 → UTF-8 字节序列 → token ID。
    # Unicode 给字符分配码点；UTF-8 规定如何把 Unicode 标量值表示为 1–4 个字节。
    # ASCII 字符使用一个字节，常见汉字使用三个字节，很多 emoji 使用四个字节。
    # token 可以对应一个字节、多个字符，或某个字符的一部分字节，没有固定长度。
    #
    # 【例子】对下面的字符串（计入空格与标点）：
    #   'Hello, 🌍! 你好!' 的 Python len 为 13，UTF-8 长度为 20 bytes。
    #   'a'   ：码点 97      → UTF-8 [97]。
    #   '你'  ：码点 20320   → UTF-8 [228,189,160]。
    #   '🌍'  ：码点 127757  → UTF-8 [240,159,140,141]。
    # 码点数、字节数、token 数是三个不同的量。课堂 01:07:25–01:07:44 的
    # bytes/token 演示也基于这一区分；本注释的长度已按当前源码实际计算核对。
    # 参考：https://docs.python.org/3/howto/unicode.html
    # 【输入范围】本讲 byte/BPE 假设文本可用 strict UTF-8 编码；Python str 可以
    # 人为包含孤立 surrogate 等特殊值，这类字符串不满足这个假设，会在编码时报错。
    text("Raw text is generally represented as Unicode strings.")
    string = "Hello, 🌍! 你好!"

    text("A language model places a probability distribution over sequences of tokens (usually represented by integer indices).")
    indices = [15496, 11, 995, 0]
    # 【易错点】上面的 indices 是“整数序列”的另一个示意，代码没有用 tokenizer
    # 从上一行字符串计算它；不要将它误读成该字符串的对应编码。真正的 roundtrip
    # 在 tokenization_examples() 里从同一 tokenizer 的 encode 结果开始验证。

    text("So we need a procedure that *encodes* strings into tokens.")
    text("We also need a procedure that *decodes* tokens back into strings.")
    text("A "), link(Tokenizer), text(" is a class that implements the encode and decode methods.")


def tokenization_examples():
    # 【观察方法】比较相同 tokenizer 下 'hello'、' hello'、中文、数字和换行；
    # 先预测边界，再观察实际 ID/字节。不同 tokenizer 的同一个整数可能表示不同内容。
    # 下方互动网址默认 encoder=gpt2，代码却用 o200k_base，因此不能直接对比 ID。
    text("To get a feel for how tokenizers work, play with this "), link(title="interactive site", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## Observations")
    text("- A word and its preceding space are part of the same token (e.g., \" world\").")
    text("- A word at the beginning and in the middle are represented differently (e.g., \"hello hello\").")
    text("- Numbers are tokenized into every few digits.")
    # 【解释】词前空格也是原文内容；把空格和词一起合并可以减少 token 数。
    # 同一个词在句首/句中得到不同 ID，意味着输入 embedding 查不同的行，但这
    # 不意味着模型训练后无法学到它们的关系。
    # 数字的分组还受预分词 regex 限制，不是整数值大小直接决定 ID，也不是所有
    # tokenizer 都按相同位数切分。分词影响模型看到的运算单位，但不单独决定算术能力。

    text("Here's the GPT-5 tokenizer from OpenAI (tiktoken) in action.")
    tokenizer = get_gpt5_tokenizer()  # @stepover
    string = "Hello, 🌍! 你好!"  # @inspect string

    text("Check that encode() and decode() roundtrip:")
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    
    # 【性质】roundtrip 保证 decode(encode(s))=s，通常不要求 encode(decode(ids))=ids。
    # 例如词表同时含 't'、'h'、'th'，两种 ID 列表都能还原 'th'，编码器却会按
    # 规则选定一种切分。模型输出的 token 序列与再次编码文本的结果不一定一样。
    text("Compression ratio: number of bytes per token")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    text("The larger the compression ratio, the shorter the sequence (good since attention is quadratic in sequence length).")
    text("One could increase compression ratio by increasing **vocabulary size** (number of possible token values increases), leading to sparsity.")
    vocabulary_size = tokenizer.n_vocab  # @inspect vocabulary_size
    # 【接口细节】tiktoken 的 n_vocab 实际是 max_token_value+1；如果编号有空洞，
    # 它不等于所有有效 token 条目的数量。不要假定 range(n_vocab) 中每个 ID 都可解码。
    # 【推导：为什么不能只追求压缩比】假设固定原文长度 S bytes，平均 R bytes/token，
    # 则 T≈S/R。dense attention 的序列交互项约 O(T²*d)，逐位置线性层约 O(T*d²)。
    # R 翻倍可让前一项约变为四分之一、后一项约变为一半；这不是整个训练提速四倍
    # 的保证，还取决于各部分占比与硬件利用率（回到 systems() 的瓶颈分析）。
    #
    # 扩大词表 V 则会增大 [V,d] embedding 和输出投影；若 d=1024，V 从 32000
    # 变成 128000，每张表从约 3277 万参数增至 1.31 亿参数。输入/输出绑权重时只
    # 存一张表，但生成全词表 logits 的计算仍在。训练中的罕见 token 也更难学好。
    # 所以 vocabulary sparsity 指许多词表项很少出现，不是说 ID 必须稀疏存储。
    # 固定 token 窗口下，R 高通常能装更多原文；不同语言的 R 不同，不能默认
    # “一个 token 等于一个单词”，也不能把英语的比例套到中文或代码上。

    text("Let's take a look at the actual vocabulary: "), link(title="vocab", url=get_local_url("var/gpt5_tokenizer_vocab.txt"))
    output_tokenizer(tokenizer, "var/gpt5_tokenizer_vocab.txt")  # @stepover


def output_tokenizer(tokenizer, path: str):
    """Write out the vocabulary of `tokenizer` to `path`, one per line."""
    # 【展示边界】这里是方便观察的文本导出，不是可无损恢复 tokenizer 的保存格式。
    # token_byte_values() 按字节排序返回普通可合并 token 的字节值，不按 token ID
    # 排列，也不包含完整的特殊 token 约定；因此“文件第 n 行 = ID n”是错误读法。
    # 单个 token 未必是完整 UTF-8，errors='replace' 只为显示，会把非法片段替换为 �。
    # token 自带换行时，写出的物理行数也不再等于 token 数；已有文件还会被直接跳过。
    # 查看精确 ID 时应使用 ID→bytes 接口，并对 bytes 使用 repr 等无损展示。
    # 接口依据：https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
    if not os.path.exists(path):
        vocab = [b.decode("utf-8", errors="replace") for b in tokenizer.token_byte_values()]
        with open(path, "w") as f:
            for token in vocab:
                f.write(token + "\n")


def character_tokenizer():
    # 【易错点】Python 逐个遍历 str 得到码点，不是用户感知的字素簇。
    # 例如 'é'（一个预组合码点）与 'e\u0301'（e + 组合重音，两个码点）外观近似，
    # len 却分别为 1 和 2；家庭 emoji 等也可能是多个码点组合成一个图形。
    # normalization 可以规范化某些等价表示，但若直接改写原文，就不能再声称
    # 逐码点精确 roundtrip 回原始输入。本教学 tokenizer 不做这类规范化。
    text("A Unicode string is a sequence of Unicode characters.")
    text("Each character can be converted into a code point (integer) via `ord`.")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    text("It can be converted back via `chr`.")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    text("Now let's build a `Tokenizer` and make sure it round-trips:")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # call ord @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # call chr @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("There are approximately 150K Unicode characters. "), link(title="Wikipedia", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    # 【澄清】“已分配字符数量”与“最大码点加一”不同。Unicode 码点空间到
    # U+10FFFF，已分配字符只占其中一部分；约 150K 是原讲稿的粗略数量级。
    # 本例直接使用码点作 ID，因此若要把 ID 当数组下标，需要覆盖最大 ID；
    # 这句话里的 lower bound 指这种索引方案的表大小下界，不是已观察字符种数。
    # 本例 max=127757，故下界为 127758，但 len(set(indices)) 只有 10。
    # 可以将出现过的字符重映射为连续 ID 来省空间，但又需处理未见字符和覆盖范围。
    # Unicode 版本会变化；不要把一个约数当永久词表大小。
    text("Problem 1: this is a very large vocabulary.")
    text("Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    text("This tokenizer is the worst of both worlds (large vocabulary, low compression ratio).")
    # 【自测】'你好' 在本类中是多少 token？解析：两个，ID 是 [20320,22909]；
    # UTF-8 则是六个字节，故此例 bytes/token=3。这里 character 的压缩比可以大于 1，
    # 仍不代表它对所有语言、规模和模型都是最佳方案。


def byte_tokenizer():
    # 【与前一方案的联系】character 给每个码点一个 ID；byte 改为给每种 byte 值
    # 一个 ID。词表缩小到 256，但一个字符可能展开为多个模型位置。
    text("Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.")
    text("The most common Unicode encoding is "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("Some Unicode characters are represented by one byte:")
    assert bytes("a", encoding="utf-8") == b"a"
    text("Others take multiple bytes:")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"
    # 【字面量】b'...' 表示 bytes；\xf0 是十六进制写法，数值为十进制 240。
    # 四段 \x.. 表示四个字节，不是字符串中十六个可见字符，也不是四个 Unicode 字符。

    text("Now let's build a `Tokenizer` and make sure it round-trips:")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("The vocabulary is nice and small: a byte can represent 256 values.")
    vocabulary_size = 256  # @inspect vocabulary_size
    text("What about the compression rate?")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    assert compression_ratio == 1
    # 因为一个 UTF-8 字节恰好一个 token，非空输入的分子分母必相等。
    # 对 'Hello, 🌍! 你好!'，ByteTokenizer 输出 20 个 token，而 CharacterTokenizer
    # 输出 13 个；差异完全来自表示方法，不是模型突然获得或失去了内容。
    text("The compression ratio is terrible, which means the sequences will be too long.")
    text("Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...")
    # 【边界】上下文限制还涉及位置设计、训练长度、KV cache 和实现，不只 T² 一项。
    # byte 方法的覆盖性很好，困难是计算成本；因此后续 tokenizer-free 架构要改变
    # 计算组织，而不只是把已有 Transformer 的输入机械换成字节。


def word_tokenizer():
    # 【背景】这里的“word”由 regex 近似定义，不是通用语言学分词器。英文空格
    # 也不能解决全部问题：缩写、标点、罕见长词、代码标识符、新人名都会影响切分。
    text("Another approach (closer to what was done classically in NLP) is to split strings into words.")
    string = "I'll say supercalifragilisticexpialidocious!"

    chunks = regex.findall(r"\w+|.", string)  # @inspect chunks
    # 【正则拆解】r'' 是 raw string，避免 Python 先解释反斜杠；\w+ 贪婪匹配
    # 一个或多个 word 字符，| 表示备选，. 匹配其余单个字符（默认不匹配换行）。
    # 本例得到 ["I", "'", "ll", " ", "say", " ",
    #           "supercalifragilisticexpialidocious", "!"]，共八块。
    # \w 不只包括英文字母，也涵盖数字、下划线及许多 Unicode 字符；连续中文可能
    # 整段匹配，不会自动按中文词义拆分。若输入含换行，这个示例 regex 会漏掉它，
    # 所以不要把演示代码当通用可逆 tokenizer。此函数也未真正实现整数词表与 decode。
    text("This regular expression keeps all alphanumeric characters together (words).")

    text("To turn this into a `Tokenizer`, we need to map these chunks into integers.")
    text("Then, we can build a mapping from each chunk into an integer.")

    text("What's good: each token is meaningful (since humans invented words).")

    vocabulary_size = "Number of distinct chunks in the training data"
    compression_ratio = get_compression_ratio(string, chunks)  # @inspect compression_ratio @stepover
    text("Compression ratio is good, but vocabulary size can be huge.")

    text("Moreover:")
    text("- Many words are rare and the model won't learn much about them.")
    text("- This doesn't obviously provide a fixed vocabulary size.")
    text("- New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.")
    # 【为什么 UNK 有问题】把不同未见词都换成一个 ID，会丢失它们原来的拼写；
    # decode 无法知道原先是哪一个词。评估时，把很多难预测的新词统一成 UNK，
    # 也改变了任务难度，可能给出具有误导性的较低 PPL。
    # 如果不断为新词增加 ID，模型的 embedding 和输出层又没有为这些新 ID 学过参数。
    # 【过渡】我们需要固定可控的词表，又想让罕见词能拆成已知片段，而常见词
    # 少占几个位置。byte-level BPE 就是这样的一个折中。


def bpe_tokenizer():
    # 【背景】subword（子词）让常见片段复用，避免给每个可能的完整词独立建表。
    # BPE 在 NLP 中有字符起步等不同变体；本文件具体演示的是 byte-level BPE，
    # 初始字母表覆盖全部 256 个 byte。不能把“无需 UNK 的字节覆盖性”归给任意
    # 一种名为 BPE 的 tokenizer，而不看它的初始字母表和兜底策略。
    # NLP 原论文：https://arxiv.org/abs/1508.07909
    # 配套教学实现：https://github.com/karpathy/minbpe
    text("## Byte Pair Encoding (BPE)")
    text("The BPE algorithm was introduced by Philip Gage in 1994 for data compression. "), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("It was adapted to NLP for neural machine translation. "), link(sennrich_2016)
    text("(Previously, papers had been using word-based tokenization.)")
    text("BPE was then used by GPT-2. "), link(gpt2_2019)

    text("Basic idea: *train* the tokenizer on raw text to construct a vocabulary tailored to the data.")
    text("Intuition: common sequences of bytes are represented by a single token, rare sequences are represented by many tokens.")

    text("Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.")
    # 【原理】每轮统计当前序列的相邻 token 对，挑频次最高的一对，为它新增 ID，
    # 然后替换所有不重叠的匹配。下一轮统计的对象已改变，必须基于合并后的序列。
    # 第一次 pair 是 byte+byte；后来可以是多字节 token+多字节 token，所以名字
    # 虽叫 Byte Pair Encoding，后续的“pair”不一定仅包含两个原始字节。
    #
    # 【为什么是 heuristic】它贪心选择当前频繁的局部模式，不直接优化语言模型
    # 的最终验证 loss，也不保证全局最短编码或得到语言学意义上的词根。
    # 频次近似反映潜在的长度收益，但相同 token 的重叠匹配会使“计数”与实际
    # 替换次数不同，见 merge()。词表增大与序列缩短是设计上的资源交换。
    #
    # 【训练/使用的区别】训练读取代表性语料，产出 vocab 和有序 merges；之后
    # encode 只应用规则，decode 只查字节表。与模型评估一样，研究 tokenizer 时
    # 也应在未参与词表训练的文本上检查压缩与覆盖，避免只在训练样本上自我验证。

    text("## Training the tokenizer")
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)
    # 【手算：空格用 ␠ 显示，方括号表示一个合并后的 token】
    # 初始文本：t h e ␠ c a t ␠ i n ␠ t h e ␠ h a t，共 18 个 ASCII byte/token。
    # 高频 pair 有 (t,h)、(h,e)、(e,␠)、(a,t)，各出现两次；本代码按首次出现
    # 顺序打破平局，因而第一轮选 (t,h)，而非按词典序选。这与课堂演示一致。
    #
    # 轮次   新规则                  vocab 新条目     当前序列
    # 1      (116,104) → 256         256: b'th'       [th] e ␠ c a t ␠ i n ␠ [th] e ␠ h a t
    # 2      (256,101) → 257         257: b'the'      [the] ␠ c a t ␠ i n ␠ [the] ␠ h a t
    # 3      (257,32)  → 258         258: b'the '     [the␠] c a t ␠ i n ␠ [the␠] h a t
    # 每轮有两次不重叠替换，长度分别为 16、14、12；旧条目仍留在词表里。
    # 最终 indices=[258,99,97,116,32,105,110,32,258,104,97,116]。
    # 词表大小=256+3=259，bytes/token=18/12=1.5。
    # 【课堂补充 01:13:20–01:15:21】这一段对应教师现场执行 train_bpe 的演示。
    # 本例词末空格被合并，是因为没有预分词限制，不应套用成所有 tokenizer 的规则。

    text("## Using the tokenizer")
    text("Now, given a new text, we can encode it.")
    tokenizer = BPETokenizer(params)  # @stepover
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    # 【手算新文本】'the quick brown fox' 共 19 个 ASCII bytes；依次应用已学规则，
    # 'the ' 变成一个 ID=258，其余部分仍是单字节，得到 16 个 token：
    # [258,113,117,105,99,107,32,98,114,111,119,110,32,102,111,120]。
    # q/u/i/c/k 等无需在训练样本出现，因为初始词表已含所有 byte 值。
    # encode 不会因新文本里某个 pair 频繁，就擅自新增 token 259。
    # decode 时把 vocab[258]=b'the ' 和其余 bytes 拼接，便恢复原文。
    # 【自测】删去所有旧的 t/h/e 字节 token，只保留合并项可行吗？
    # 解析：不行。新词仍可能需要这些单字节兜底，且编码从单字节序列开始。

    text("In Assignment 1, you will go beyond this in the following ways:")
    text("- encode() currently loops over all merges. Only loop over merges that matter.")
    text("- Detect and preserve special tokens (e.g., <|endoftext|>).")
    text("- Use pre-tokenization (e.g., the GPT-2 tokenizer regex).")
    text("- Try to make the implementation as fast as possible.")
    # 【工程差距 1：预分词】先按 regex 切成片段，再在各片段内训练/应用 BPE。
    # BPE 不跨片段边界合并，减少候选空间，同时表达对空格、标点、数字等的规则。
    # 它与 word tokenizer 不同：word 方法给完整片段一个 ID，预分词后的 BPE
    # 还会继续把片段拆成多个已知 token。训练和使用时应采用相同边界约定。
    # 把一个字符串随意切块分别 encode 后拼接，不一定等于整体 encode：例如这里
    # encode('t')+encode('he ') 是四个字节 ID，encode('the ') 却是 [258]。
    # 只有在确定不会跨越的边界处分块，才能据此设计等价的并行/流式处理。
    #
    # 【工程差距 2：特殊 token】如文档结束标记、对话角色边界，是协议约定的独立
    # ID，不应默认参加普通文本的 pair 合并。它们帮助模型区分文档/对话结构。
    # 文本恰好含 '<|endoftext|>' 时，是普通字面内容还是控制标记，必须由接口
    # 约定决定；不能悄悄改变用户文本语义。不同库有不同的允许/拒绝规则。
    #
    # 【工程差距 3：性能】设初始长度 L、合并规则数 M。教学版训练每轮重新扫描
    # 当前序列，粗略上界 O(M*L)；encode 也逐规则扫描，包括完全不在输入中的规则。
    # 改进方向是只维护受影响的邻接关系、复用相同片段的计数、只考虑当前可用
    # 规则等。正确性检查必须包含确切 ID 序列，单看 roundtrip 无法证明分词相同。
    # 本讲只解释动机与接口，不在这里展开作业的高性能实现。


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    # 【输入/输出】string 是用于学习规则的语料（这里缩小为一个字符串）；
    # num_merges 是合并次数，成功做 M 次后词表有 256+M 项，不含额外特殊 token。
    # 返回值是 tokenizer 参数，不是语言模型权重，也不是这段训练文本的 token 列表。
    # 想学习过程可配合 bpe_tokenizer() 的手算表逐轮对照。
    text("Start with the list of bytes of `string`.")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    # 【Python 易错点】bytes([x]) 生成“一个值为 x 的字节”；bytes(x) 却生成
    # x 个零字节，含义完全不同。b'th' 保存内容，256 只是内容的编号。
    # 初始词表包含未在语料出现的 byte；随着合并增加，0..255 的条目仍然保留。

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = count_adjacent_pairs(indices)  # @inspect counts @stepover
        # counts 的 key 是一对当前 token ID，value 是出现次数；不能只统计一次
        # 原始 byte pair 后始终复用，因新 token 会创造新的相邻对并消除旧对。

        # Find the most common pair
        pair = max(counts, key=counts.get)  # @inspect pair
        # 【平局规则】max 遍历字典的 key，用 counts.get(key) 的值比较；最大值
        # 并列时返回先遇到的 key。Python dict 保留插入顺序，而 count_adjacent_pairs
        # 从左到右首次插入 pair，故本例等价于“同频取最早出现者”。
        # 这只是本讲实现的确定性规则，不要假定所有 BPE 库或作业采用同样的规则。
        # 【边界】长度不足 2 时 counts 为空，max 会报错。请求过多合并也会遇到它。
        # 空文本即使 num_merges=0，最后的压缩比仍会除零。原代码用于有限的小演示，
        # 注释说明边界，保留原实现以便与课堂代码对照。

        # Merge that pair
        new_index = 256 + i  # @inspect new_index
        # 每轮使用一个从未用过的新 ID；词表旧条目和旧规则不删除。
        merges[pair] = new_index  # @inspect merges
        # 【可逆性的关键】新 ID 对应的内容是两个旧 token 的字节拼接。
        # 例如 pair=(256,101)，vocab[256]=b'th'、vocab[101]=b'e'，
        # 则新条目是 b'the'。这一步“合并内容”，下一步 merge() “替换序列编号”。
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices @stepover

    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    return BPETokenizerParams(vocab=vocab, merges=merges)
    # 【回看不变量】从原 UTF-8 bytes 开始，每次合并只改变分组，不改变字节次序
    # 和内容，所以解码训练/编码产生的合法序列时可以还原。这是 roundtrip 的
    # 结构性原因，不是小例子 assert 恰巧通过的巧合。


def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
    """Return a dictionary mapping each adjacent pair of tokens in `indices` to the number of times it occurs."""
    # 【代码导读】zip(indices, indices[1:]) 将原序列与“错开一位”的序列配对：
    # [a,b,c,d] → (a,b),(b,c),(c,d)，只统计相邻对，不统计所有两两组合。
    # defaultdict(int) 让未出现的 key 从 0 开始；重复 pair 累加其频次。
    # 注意滑窗允许重叠：[97,97,97] 统计 (97,97) 两次，merge 却只能替换一次。
    # 时间 O(L)；不同 pair 最多 L-1 种。indices[1:] 还会复制一个 O(L) 的列表，
    # 这是讲义清晰写法的额外空间成本，大规模实现需要留意。
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts


if __name__ == "__main__":
    main()
