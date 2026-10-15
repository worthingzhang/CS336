# 【第 12 讲中文自学增强：如何把“模型好不好”变成可信的证据】
# 学习目标：能从概率链式法则推导困惑度，识别分词器/语料造成的不可比；
# 能为考试、对话、智能体、推理与安全任务选择指标，并说明分数支持多大范围的结论；
# 能区分抽样误差、评判偏差、训练污染、构念有效性与现实代表性；能写出可复现评测协议。
# 先修：条件概率、对数、期望与方差；训练/验证/测试集；自回归模型和交叉熵。
# 阅读主线：what_is_good 提出测量目标 → 六类 benchmark 展示操作化方式 →
# realism 检查目标人群与工作流 → validity 检查证据可靠性 → 最后明确比较对象和规则。
# 与前后讲联系：第 10 讲推理系统决定预算/延迟，第 9、11 讲的损失曲线需要可靠测量；
# 第 13、14 讲的数据来源、过滤和混合应服务于这里先确定的目标，不能反过来按测试题调数据。
# 本文件保持原英文、代码和图片引用，新增内容均为 # 注释，供直接阅读 Python 原稿。
# 数字例子、公式推导、自测和设计建议若未标“课堂补充”，均为编者补充，非教师逐字原话。
# 原稿排行榜、模型成绩和样本数是课程时点的材料，不宣称为当前排名；版本差异在相邻处说明。
#
# 来源索引（2026-10-15 实际联网核验；原讲义是主题主轴，论文用于核对定义和边界）：
# [课] Spring 2026 课程 https://cs336.stanford.edu/
# [稿] https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py
# [字幕] 第三方公开录像字幕，未由教师审校；仅按标出的时间范围转述：
# https://jingzechen.github.io/assets/courses/stanford-cs336-s26/lectures/012/captions.source.srt
# [PPL] Hugging Face 官方文档 https://huggingface.co/docs/transformers/en/perplexity
# [Arena] 人类配对偏好 https://arxiv.org/abs/2403.04132
# [LC] 长度控制 AlpacaEval https://arxiv.org/abs/2404.04475
# [污染] 可交换性检验 https://arxiv.org/abs/2310.17623
# [质量] Platinum 数据审计 https://arxiv.org/abs/2502.03461
# [现实] GDPval https://arxiv.org/abs/2510.04374 ；MedHELM https://arxiv.org/abs/2505.23802
# 其余基准的一手论文 URL 保留在对应原行，相邻注释说明如何读，勿把榜单营销名当构念定义。
from edtrace import text, link, image
from lecture_util import post_link
from references import mmlu_2021

# 【整讲覆盖地图】
# “好”不是模型单独携带的一个常数。先指定用户、任务分布、允许工具和资源，再谈成功。
# 一张可信成绩单至少包含：测什么 → 抽哪些题 → 怎样运行 → 怎样判分 → 不确定性与边界。
# 同一个模型可以在考试正确率上胜出，却在限定时延的客服任务上因成本或误答率落败。
def main():
    text("## Lecture 12: evaluation")
    text("- So far: we've covered everything for training an LM (architecture, training, systems, scaling).")
    text("- Missing piece: what **data** do you train on?")
    text("- Data shapes model behavior (code? multilingual? DNA?).")
    text("- Before talking about data, need to talk about what behavior we want from a model.")

    text("**Evaluation**: given a model, how \"**good**\" is it?")

    what_is_good()

    perplexity()
    exam_benchmarks()
    chat_benchmarks()
    agentic_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()

    realism()
    validity()
    how_to_think_about_evaluation()

    text("Takeaways:")
    text("- There is no one true evaluation; choose the evaluation depending on what you're trying to measure.")
    text("- Clearly state the rules of the game (methods versus models versus agents).")
    text("- Considerations: difficulty, realism, validity.")


# 【从抽象构念到具体测量】
# construct（构念）是想研究而不能直接读出的属性，如事实可靠性、推理能力、用户满意度。
# metric（指标）是按规则从可观察行为得到的数，如正确率、胜率、单位成本成功次数。
# 操作化就是指定“用哪些任务和打分规则代表这个属性”；选择会决定研究被什么奖励。
# 编者的形式化：输入 x 来自目标分布 q，系统 S 在预算 b 下产生输出/轨迹 y，
# 指标 g(x,y) 给分，目标量是 μ=E_{x~q, y~S(·|x,b)}[g(x,y)]。
# 有限测试集只给估计 μ_hat；变更 q、b、S 的工具或 g，测量对象也随之变了。
# 构念有效性问“这个分数是否真的支持对目标能力的解释”，不是问名字听起来是否高级。
# 例：给回答加漂亮排版能提高偏好票，但未改善事实正确性；若目标是正确性，这就是混杂。
# 即使每次都打出同一分（可靠性高），也可能一直量错属性（有效性低）。
def what_is_good():
    text("Evaluation might appear to be a mechanical process:")
    text("1. Define some prompts")
    text("2. Send prompts to a model and get back responses")
    text("3. Compute accuracy")

    text("But actually, evaluation is a deep and important topic...")
    text("...which shapes the development of AI.")

    text("**Core challenge**: <font color=\"red\">abstract construct</font> → <font color=\"blue\">concrete metric</font>")

    text("Maybe a model is good if it does well on benchmarks...")
    link(title="Artificial Analysis", url="https://artificialanalysis.ai/")
    image("images/artificial-analysis.png", width=800)

    # 读下面几种排名时，每次先问纵轴测什么、横轴固定了什么、样本是谁提供的。
    # 质量与费用可构成 Pareto 前沿：若 A 更准也更便宜，B 被支配；若 A 更准更贵，选择依赖需求。
    # 不要把一次调用价格当总任务成本：多次尝试、长推理、检索和人工复核都可能计费。
    # 例如 A 每次 0.02 元、成功率 0.8，B 每次 0.01 元、成功率 0.5；
    # 在独立重试且能识别成功的理想模型中，每成功一次期望费用分别为 0.025 与 0.02 元。
    # 但真实用户可能只允许一次回答，或无法知道成功与否，此时这个费用指标不适用。
    text("Maybe a model is good if it does well on benchmarks and is cheap to run...")
    image("images/artificial-analysis-cost.png", width=800)

    text("Maybe a model is good if people prefer its responses...")
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)

    # 偏好、使用量与付费都能提供证据，但解释各异：偏好混合内容与风格，
    # 使用量还受价格、免费额度、产品入口、默认选项和用户群影响。销量不是能力的控制实验。
    # 自测：A 的使用量翻倍能否推出推理能力翻倍？答：不能；构念、量纲与混杂都不支持此推断。
    text("Maybe a model is good if people simply choose to use (and pay for) it...")
    link(title="OpenRouter", url="https://openrouter.ai/rankings")
    image("images/openrouter.png", width=600)


# 【从 token 概率到困惑度：先固定计量单位】
# 设 D 是测试 token 序列 x_1,…,x_N，模型对真实前缀作条件预测（teacher forcing）。
# p(D)=∏_{i=1}^N p(x_i|x_<i)，取负对数把连乘变成相加：
# L(D)=-(1/N)Σ_i ln p(x_i|x_<i)=-ln p(D)/N，这是每 token 交叉熵的样本估计。
# PPL(D)=exp(L(D))=p(D)^(-1/N)。用 log_2 时损失单位是 bit，PPL=2^L。
# 所以困惑度越低越好；归一化离散概率下 PPL≥1，任何真实 token 概率为 0 会使损失无穷大。
# 不要直接连乘几千个概率，易下溢；实际累计 log probability，再对平均负对数取指数。
# 手算：两个真实 token 的条件概率都是 1/4，则联合概率 1/16，
# L=ln 4≈1.3863，PPL=4。它相当于每步均匀面对 4 个候选时的不确定度，
# 不表示模型真的只考虑 4 个词，也不表示生成正确率为 25%。
#
# [PPL] 分词器和预处理是指标定义的一部分。即使同一原始字符串的概率恰好同为 1/16，
# 分成 2 个 token 的 PPL=4，分成 4 个 token 的 PPL=2；模型并未因此变“好一倍”。
# 跨分词器比较宜同时报固定原文的总 NLL、bits per byte（BPB）或 bits per character，
# 并统一原文、Unicode/空白处理、EOS、上下文长度与评分位置；BPB 也不能消除语料差异。
# 若上述字符串恰为 8 个 UTF-8 字节，BPB=-log_2(1/16)/8=0.5。
# 这里把规范 token 路径的码长作为可比较量；不要忽略不同编码/规范化改变了原始字节。
# 多个文档应先累计 NLL 再除以计分 token 总数。不能直接算各文档 PPL 的算术平均。
# 例：1-token 文档 PPL=2，3-token 文档 PPL=8，合并为 exp((ln2+3ln8)/4)=√32≈5.657，
# 不是 (2+8)/2=5。按文档等权平均损失则是另一种明确可选、但不同的指标。
#
# 上下文窗口也会改变条件概率：分块时丢弃上下文通常使任务更难；滑窗可保留更多上下文。
# 评分时不能把 padding 算作真实 token，也不能将 prompt 的损失混进仅评 response 的分母。
# 要说明首 token 如何处理、文档间是否重置上下文，并防止滑窗重叠位置重复计分。
def perplexity():
    text("- Recall: that a language model is a probability distribution **p(x)** over sequences of tokens.")
    text("- Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D.")

    text("- In pre-training, you minimize perplexity on the training set.")
    text("- The obvious thing is to measure perplexity on the test set.")
    text("- This is what people did traditionally in language modeling research.")

    # 同分布测试是希望 train/test 来自同一生成机制且样本相互隔离，不是只要文件名叫 test。
    # 测试损失衡量该语料的预测能力；新闻、百科、代码的频率结构不同，PPL 不能跨语料直接排名。
    # 分布外（OOD）评估把迁移本身作为问题，因此性能变化同时反映模型能力和分布偏移。
    text("Standard datasets:")
    text("- Penn Treebank (WSJ)")
    text("- WikiText-103 (Wikipedia)")
    text("- One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)")
    text("Classic paradigm: in-distribution evaluation: train on train split and evaluate on test split of some dataset.")
    text("Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 → 30.0) "), link("https://arxiv.org/abs/1602.02410")

    text("GPT-2:")
    text("- Trained on WebText (40GB text, websites linked from Reddit)")
    text("- Zero-shot on standard datasets (**out-of-distribution** evaluation)")
    # 图表阅读：同一行是同一规模模型，不同列却不全是 PPL；ACC 越高越好，PPL/BPB/BPC 越低越好。
    # 例如 1542M 在 PTB 的 35.76 低于表中旧 SOTA 的 46.54，在 1BW 的 42.16 却高于 21.8。
    # 这支持“迁移效果随数据集而异”，不支持把两列数字直接比大小，也不是只由参数量决定的因果实验。
    # LAMBADA 同时出现 PPL 和 ACC：给正确词更大概率与最终选对词相关，但不是同一个量。
    image("images/gpt2-perplexity.png", width=800)
    text("- Works better on small datasets (PTB) where transfer is helpful, but not larger datasets (1BW)")

    text("Perplexity is all you need (more faith than science):")
    text("- True distribution is t, model is p.")
    # 【原稿公式辨析；保留下一行英文】这里把 entropy 与 perplexity 简写混用了。
    # 在固定 token/上下文定义下，真实分布 t 的期望交叉熵为：
    # H(t,p)=-Σ_x t(x)ln p(x)=H(t)+Σ_x t(x)ln(t(x)/p(x))=H(t)+KL(t||p)。
    # KL≥0，因此交叉熵的最小值是 H(t)，困惑度的最小值是 exp(H(t))，不是 H(t)。
    # 用 bit 度量则最小困惑度是 2^{H(t)}；相等要求 p 与 t 匹配（条件模型在 t 支持的上下文上）。
    # 这里的 H 是同口径的每 token 熵/熵率；有限测试样本的损失会波动，不必始终高于总体 H(t)。
    # 课堂补充 · 00:09:46–00:11:36：讲者把这段当作推动语言建模研究的信念来讨论，
    # 而非已经证明的“损失下降必达通用智能”；上面的等式是编者按标准定义作的数学澄清。
    # 即使 p=t，若 t 是含谣言、错误解答的网页分布，忠实模仿 t 也不等于总能输出正确答案。
    # 从平均损失变好跳到所有任务都解决，还缺任务支持、可靠条件分布和可负担推理等前提。
    text("- Best possible perplexity is H(t) obtained iff p = t.")
    text("- If p = t, then solve all the tasks: p(solution | problem)")
    text("- So by pushing down on perplexity, we will eventually \"reach AGI\".")

    text("Perplexity is maybe more than you need:")
    text("- Example: *Stanford was founded in 1885*")
    text("- Perplexity penalizes prediction on all tokens, some (e.g., *founded*) of which might not be relevant")
    # 【原稿公式辨析；保留下一行英文】设 prompt=u，response=y_1,…,y_m。
    # 条件困惑度应为 p(y|u)^(-1/m)=exp(-(1/m)Σ_j ln p(y_j|u,y_<j))。
    # 下一行原式 p(y|u)^(1/m) 是几何平均条件概率，越高越好，恰是条件困惑度的倒数。
    # 实现时仍输入完整 prompt+response，但只对 response 的预测位置累计 NLL 和 token 数。
    # 课堂补充 · 00:12:38–00:13:06：条件化用于聚焦想测的回答 token，忽略无关前缀的预测成本。
    # 边界：固定参考答案的概率仍会惩罚正确同义表达；开放任务要区分“参考措辞”与“满足意图”。
    text("- Solution: measure conditional perplexity p(response | prompt)^(1/|response|)")

    # 完形填空把概率集中到特定空缺；多项选择则比较候选续写分数，最后通常报告选择正确率。
    # 候选 a 的原始分数可为 Σ_j log p(a_j|prompt,a_<j)，也有人除以候选长度。
    # 两者不等价：候选 A 概率 0.2、长 1，B 概率 0.09、长 2；
    # 按联合概率选 A，按每 token 几何平均概率则选 B（√0.09=0.3>0.2）。
    # 因此选项标签概率、完整答案概率、长度归一化、自由生成后解析，必须明确采用哪种协议。
    # LAMBADA 要用长篇上下文预测末词；HellaSwag 的难点在合理的情境续写与干扰选项。
    # “披着困惑度外衣”说明评分依赖似然，不意味着 benchmark 正确率数值等于 PPL。
    text("Some benchmarks are perplexity in disguise:")
    text("- Cloze tasks (fill in the blank): LAMBADA "), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=700)
    text("- Multiple choice sentence completion: HellaSwag "), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=500)

    # 为什么概率接口需要额外信任？对每个前缀，词表概率应非负且求和为 1。
    # 若提交者只对每个观察到的答案谎报 log_prob=0，就会得到 PPL=1，
    # 但不可能对多个互斥候选同时赋概率 1；这不是合法的语言模型分布。
    # 稳定的 log_softmax 可实现归一化，但仅收一个标量的远程评分接口未必能由评测方核查。
    # 输出答案再独立判分缩小了这类作弊空间；仍需防测试泄漏和判分器漏洞。
    text("**Warning** (if you're running a perplexity leaderboard):")
    text("- People submit `LM` and you compute `log_prob = LM(test_data)`")
    text("- You need to trust that the probabilities are valid (sum to 1)")
    text("- For downstream tasks, `response = LM(prompt)` and compute accuracy on `response`")

    text("Summary:")
    text("- Perplexity is still used heavily in language model development (smooth scaling laws)")
    text("- Still need benchmarks that capture real-world situations (for the non-believers)...")


# 【考试能控制问题，却不会自动控制测量偏差】
# 给定 n 道题，正确指示量 X_i∈{0,1}，准确率 p_hat=(Σ_i X_i)/n。
# 先规定作答方式（选字母还是生成解释）、few-shot 示例、选项排列、答案解析和推理预算。
# 四选一的均匀猜测基线是 1/4，十选一是 1/10；选项增加也会改变干扰结构和阅读成本。
# macro 平均先算各科准确率再等权平均；micro 平均把所有题混在一起，每题等权。
# 例：科目甲 10 题答对 9，乙 90 题答对 45；macro=(0.9+0.5)/2=70%，micro=54%。
# 两种数字都可正确，前者代表科目等权、后者代表题目等权；必须与目标任务权重对应。
#
# 【抽样误差：编者推导】若题目是从目标题库独立同分布抽取，X_i~Bernoulli(p)，
# Var(X_i)=p(1-p)，独立性使 Var(p_hat)=p(1-p)/n。
# 代入 p_hat 得标准误 SE≈√(p_hat(1-p_hat)/n)，大样本近似 95% 区间为 p_hat±1.96SE。
# 例：100 题答对 80，SE=0.04，近似区间 [0.7216,0.8784]；多 2 道题不足以自动宣称领先。
# 这里 95% 指重复抽样构造的区间约 95% 覆盖固定总体 p，并非该次区间中 p 有 95% 随机概率。
# 边界附近或 n 小时此正态近似不可靠；可用 Wilson 区间，避免“全对所以 SE=0”的错误确定性。
# 若题目就是固定完整集合且运行确定，80/100 是该集合的精确成绩；区间针对向更广题库推广。
# 同一文章衍生的题可能相关，应按文章/任务簇重采样，而非假装每题独立以压窄区间。
#
# 同题比较 A、B 应使用配对差 d_i=X_i^A-X_i^B：均值是差值，SE=s_d/√n。
# 例：100 题中 A 独对 12、B 独对 8、其余同时对/错，均值差 0.04；
# s_d²=(20-100×0.04²)/99≈0.2004，SE≈0.0448，区间约 [-0.0477,0.1277]。
# 这个例子无法支持稳定正提升。配对 bootstrap 可同时重采样同题的两个结果再算差，保留相关性。
# 题目抽样、模型随机采样、评委噪声是不同来源；重复同一题的生成不会增加新题的覆盖度。
# 检查许多模型/提示后只报最佳分会产生选择偏差；开发集调参后用独立保留测试集确认。
# 以上是抽样不确定性，不会校正污染、错误标签或人群分布偏差；更多题也不能自动治好这些偏差。
def exam_benchmarks():
    text("Exams are a useful way to test language models (as with humans):")
    text("- Have control over the subject and difficulty")
    text("- Design to have unambiguous correct answer, easy to grade")

    text("**Massive Multitask Language Understanding (MMLU)** "), link(mmlu_2021)
    text("- 57 subjects (e.g., math, US history, law, morality), multiple-choice")
    text("- \"collected by graduate and undergraduate students from freely available sources online\"")
    text("- Despite the name, MMLU is really about testing knowledge, not language understanding")
    text("- Evaluated on GPT-3 using few-shot prompting")
    image("images/mmlu.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu")
    link(title="HELM MMLU for visualizing predictions", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    # MMLU 的广科目覆盖有助于检查知识面，但科目、语言、考试文化都是样本选择。
    # 所谓 few-shot 是在测试提示内放少量示例，不等于本轮更新模型权重；示例仍可能显著改变得分。
    # 读原图应问具体题目测知识、题型识别还是多步计算；名称中的 understanding 不能替代证据。
    text("**MMLU-Pro** "), link("https://arxiv.org/abs/2406.01574")
    text("- Removed noisy/trivial questions from MMLU")
    text("- Expanded 4 choices to 10 choices")
    text("- Evaluated using chain of thought (gives model more of a chance)")
    # 这里同时改变了题目清理、选项数和 CoT 协议，不能把变化归因于某一个因素。
    # CoT 给出中间推理，通常允许更多输出 token；比较应固定或报告相应预算。
    # 原文“16% to 33%”沿用论文摘要表述，单独看这行不足以判断相对降幅还是百分点。
    # 报告时需回到对应模型的前后 accuracy：80%→60% 是降 20 个百分点、相对降 25%。
    # 不要凭这行措辞计算新的精确数值，更不要外推为所有模型今天都会下降相同比例。
    text("- Accuracy of models drop by 16% to 33% (not as saturated)")
    image("images/mmlu-pro.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu-pro")
    link(title="HELM MMLU-Pro for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    text("**Graduate-Level Google-Proof Q&A (GPQA)** "), link("https://arxiv.org/abs/2311.12022")
    text("- Questions written by 61 PhD contractors from Upwork")
    image("images/gpqa.png", width=700)
    text("- PhD experts achieve 65% accuracy")
    text("- Non-experts achieve 34% over 30 minutes with access to Google")
    text("- GPT-4 achieves 39%")
    link("https://llm-stats.com/benchmarks/gpqa")
    link(title="HELM GPQA for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    # GPQA 的“Google-Proof”是对被试条件下搜索帮助有限的描述，不是不可检索的数学保证。
    # 专家也非满分：学科匹配、题目歧义、时间和复核规则都会影响人类基线。
    # 原稿的专家、非专家与 GPT-4 数字必须连同当时协议理解；不能当作固定“人类上限”。
    # 课堂补充 · 00:26:26–00:27:19：污染可能来自题目引用的更早材料，而非直接下载测试文件；
    # 这提示我们核对来源链。正常学习学科知识与见过具体题目/答案仍应区分，不能混称作弊。
    text("**Humanity's Last Exam (HLE)** "), link("https://arxiv.org/abs/2501.14249")
    text("- 2500 questions: multimodal, many subjects, multiple-choice + short-answer")
    image("images/hle-examples.png", width=700)
    text("- Awarded $500K prize pool + co-authorship to question creators")
    text("- Filtered by frontier LLMs, multiple stages of review")
    # HLE 的筛题流程图应读成选择机制：先有候选题，再经过模型筛选和专家审查形成集合。
    # 若刻意保留当前强模型失败的题，评估的是这类困难题条件下的性能，不是随机用户问题平均水平。
    # 多模态题还依赖图像输入/分辨率；短答题需要明确等价答案与评分规则。
    # 题目变难能延缓饱和，却不能自动提高现实代表性；过难导致全模型近零，也难分辨进步。
    # 自测：某模型 HLE 高分能直接保证客户投诉处理更好吗？答：不能；任务分布和成功定义都不同。
    image("images/hle-pipeline.png", width=700)
    image("images/hle-results.png", width=600)
    link("https://llm-stats.com/benchmarks/hle")

    text("Summary:")
    text("- Trend towards harder questions as models improve and saturate existing benchmarks")
    text("- Multiple-choice format can be as difficult as one wants")
    text("- Does not capture real usage (open-ended, doesn't necessarily exist correct answer)")


# 【开放回答：把“更喜欢”与“更正确”分别检查】
# 沙拉建议没有唯一参考字符串，可以按相关性、事实性、可执行性和遵守约束分别打分。
# 配对比较让评委在同一请求上直接比较两份回答，通常比凭空给绝对分容易，但仍受评委影响。
# 课堂补充 · 00:37:15–00:37:42：用户偏好可能混合风格和正确性；流畅、自信不保证内容正确。
# 先定义 rubric（评分细则），给出各档标准和边界例子，再用人工核查判断细则是否被正确执行。
def chat_benchmarks():
    text("- So far, we've been evaluating on well-defined multiple-choice tasks.")
    text("- Most people don't ask multiple-choice exam questions to their AI assistant.")
    
    text("Example:")
    text("Prompt: *I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?*")
    text("Response: *Here’s a breakdown of herbs that work well (and some that don’t) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...")

    text("**Challenge**: how to evaluate an open-ended response?")

    text("**Chatbot Arena** "), link("https://arxiv.org/abs/2403.04132")
    text("Data collection:")
    text("- Random person from the Internet types in prompt")
    text("- They get response from two random (anonymized) models")
    text("- They rate which one is better")
    image("images/arena-beets.png", width=700)
    # 【读懂配对评分模型】ELO 差决定胜率：同分时预测胜率 1/2，A 高 400 分时为 10/11≈90.9%。
    # 可将每场胜负看作 Bernoulli 观测，通过最大化对数似然拟合各模型评分；
    # 所有评分同时加一个常数不改变胜率，因此需要固定参考点。ELO 差值不是准确率差值。
    # 该模型用单一标量概括偏好；不同用户群或任务可能出现 A 胜 B、B 胜 C、C 胜 A 的循环。
    # 平局、模型配对抽样和置信区间需另行约定；票数少、对手不同或用户相关会影响排名稳定性。
    # 真人自愿提交的请求具有使用价值，却不保证覆盖你的目标客户、语言和专业任务。
    text("Compute ELO rankings based on pairwise comparisons:")
    text("- Define model: p(A wins against B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))")
    text("- Fit this model to maximize probability of pairwise comparisons")
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)
    text("Properties:")
    text("- Real-world prompts (free for users, incentives to actually use it)")
    text("- But who are these people? biases? spammers?")
    text("- Binary preference but conflates style and correctness")
    text("- How does the human even assess correctness?  Prone to sycophancy?")
    text("- Feature: don't need to feed same prompts to all models (important because human is rating)")
    text("- Dynamic: incorporates new prompts and models over time")

    text("**AlpacaEval** (2023)"), link(title="leaderboard", url="https://tatsu-lab.github.io/alpaca_eval/")
    text("- 805 instructions from various sources")
    text("- Metric: win rate against baseline model (GPT-4 preview) as judged by GPT-4 preview (potential bias?)")
    text("- Problem: LLM judges favor longer responses, resulted in leaderboard gaming")
    # [LC] 长度控制先拟合偏好与长度差等变量的回归关系，再预测“双方长度相同”时的偏好。
    # 它不是把回答截成同样长，也不是胜率除以 token 数；校正依赖模型假设，不能消除所有偏差。
    # 评判还可能受答案位置、相似风格和评委能力影响；可交换 A/B 位置、隐去身份并抽样人工复核。
    # 课堂补充 · 00:38:21–00:41:17：与人类榜单相关是校验线索，不证明每个任务都可靠。
    # 两个榜单也可能共享长度/风格偏差；跨模型相关性高不保证单个回答判分正确。
    text("- Alpaca Eval 2.0 used regression to debias the metric "), link("https://arxiv.org/pdf/2404.04475")
    text("- How do we evaluate the metric?")
    text("- Correlation with Chatbot Arena (humans) is high:")
    image("https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png", width=500)
    image("images/alpacaeval-leaderboard.png", width=400)

    # WildBench 从真实对话筛选任务，再用任务专属 checklist 帮助自动评判。
    # “来自百万对话”描述来源池，实际 1024 个任务才是这个测试集；筛选本身会改变任务分布。
    # 开放任务报告应包含评委版本、提示、参考模型、平局规则与人工审计，而不只一个总胜率。
    # 自测：冗长回答赢得更多票是否说明事实更准？答：需要单独检查事实；长度可能是混杂因素。
    text("**WildBench** "), link("https://arxiv.org/pdf/2406.04770")
    text("- Sourced 1024 examples from 1M human-chatbot conversations")
    text("- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge")
    text("- Well-correlated with Chatbot Arena (seems to be the de facto sanity check)")
    image("images/wildbench.png", width=700)
    link(title="HELM WildBench for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")

    text("Summary:")
    text("- Challenge: how to evaluate open-ended responses?")
    text("- Pairwise comparisons between similar responses provide higher signal")
    text("- Beware of biases (both from humans and LLM judges)")
    text("- Checklist/rubric improves reliability (regardless of human or LLM judge)")


# 【智能体评估的对象是完整系统】
# 模型决定下一步输出，scaffold 负责提示、工具调用、记忆、重试和停止；环境返回观察与反馈。
# 所以成功率属于“模型+脚手架+工具+环境+预算”的组合，不宜单独归功于基础模型。
# 单位测试使结果可执行验证，但测试通过仅证明覆盖到的条件通过，不能证明满足全部用户意图。
# 报告须固定仓库/环境版本、可见文件、网络权限、工具、超时、重试次数和验证器。
# 纯模型对比要控制脚手架；产品选型可比较整套系统，但应披露各自的资源成本。
def agentic_benchmarks():
    text("Previously: evaluate what LMs say (chat)")
    text("Now: evaluate what LMs do (agents)")

    text("Agent = language model + agent scaffold (logic for deciding how to use the LM)")
    
    text("Consider tasks that require tool use (e.g., running code) and iterating over a period of time")

    text("**SWEBench** "), link("https://arxiv.org/abs/2310.06770")
    text("- 2294 tasks across 12 Python repositories")
    text("- Given codebase + issue description, submit a PR")
    # SWE-bench 从 issue 与代码仓库出发，系统产出补丁，再在指定环境中运行测试。
    # 原始集合和 Verified 是不同版本；下一行之后链接到 Verified 榜单，不表示原始任务数就是其规模。
    # 还要区分应从失败变成功的测试与原本通过且应保持通过的回归测试。
    # 若测试泄漏、环境已含答案或验证器不检查关键要求，分数会偏离真正修复能力。
    text("- Evaluation metric: unit tests")
    image("images/swebench.png", width=800)
    link("https://llm-stats.com/benchmarks/swe-bench-verified")

    text("**TerminalBench** "), link("https://arxiv.org/abs/2601.11868"), link(title="website", url="https://www.tbench.ai/")
    image("images/terminal-bench.png", width=700)
    text("- Computer terminal environments: simple and universal")
    text("- 229 tasks crowdsourced from 93 contributors, 89 tasks constitute Terminal-Bench 2.0")
    image("images/terminal-bench-human-time.png", width=600)
    image("images/terminal-bench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/terminal-bench")

    # Terminal-Bench 让代理在终端环境操作，版本、任务筛选和时间预算都会影响成功率。
    # 读人类耗时图时留意统计对象：人类解决耗时可作难度线索，却不是模型运行时间的换算系数。
    # CyBench 用隔离的 CTF 环境检查安全能力；首个解题团队的时间受参赛者和并行搜索影响，
    # 不是任务固有常数。是否提供子任务提示也会改变所测的自主规划能力。
    text("**CyBench** "), link("https://arxiv.org/abs/2408.08926")
    image("images/cybench.png", width=700)
    text("- 40 Capture the Flag (CTF) tasks")
    text("- Use first-solve time as a measure of difficulty")
    image("images/cybench-agent.png", width=700)
    image("images/cybench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/cybench")

    text("**MLEBench** "), link("https://arxiv.org/abs/2410.07095")
    text("- 75 Kaggle competitions (require training models, processing data, etc.)")
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=700)

    # MLE-bench 包括数据处理、训练和实验选择，因此 GPU、墙钟时间、数据权限与采样预算尤其重要。
    # 比赛分数/获奖门槛衡量给定竞赛上的结果，不自动等于可维护生产系统的工程质量。
    # 脚手架图中的规划、子代理和文件记忆各自增加能力，也可能增加调用量、延迟与失败途径。
    # 编者补充：pass@k 表示 k 次尝试中至少一次成功的机会，不等于系统能在无答案时选出成功那次。
    # 若每次独立成功率 p，则该机会为 1-(1-p)^k；共享失败模式或相关采样时不能套独立公式。
    # 可部署结果还需说明如何选最终答案，选择器和验证器同样属于系统。
    text("Agent scaffolds "), post_link("https://www.philschmid.de/agents-2.0-deep-agents")
    image("https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png", width=400)
    text("- Explicit planning: keep a todo list that gets checked off")
    text("- Hierarchical delegation: agents calling other sub-agents (clean context)")
    text("- Persistent memory: read/write files")
    text("- Extreme context engineering: explicit more instructions on process")

    text("Summary:")
    text("- Agents dramatically enhance the capability surface of language models")
    text("- Agent scaffolds are very important")
    text("- Evaluating agents = evaluating agent scaffold + language model")


# 【推理测量：减少知识依赖，不等于完全消除先验】
# ARC 给少量输入/输出网格示例，要求发现规则并推广；需结合颜色、位置、对象和变换关系。
# “每题不同”旨在避免直接记住具体答案；公开题训练、规则经验与空间先验仍可能有帮助。
# 原文“100% solvable”不应读作每位被试每题都答对；人类可解性依赖被试与试验条件。
# ARC-AGI-1/2 的静态网格与 ARC-AGI-3 的交互环境协议不同，不应把版本分数直接连成同尺度曲线。
# 交互任务还测探索、反馈利用和行动效率，而非只测一次输出答案。
#
# 【推理预算是成绩条件；编者补充】更多思考 token、采样或工具调用可能提高成绩，但有成本，
# 效果依任务难度和分配策略而变；不能把“更多 token 必然更强”当定理。
# 比较须记录最大输出/推理 token、尝试数、时间、工具与是否有验证器，最好报告质量—成本曲线。
# 固定预算的优劣和最高可达分数回答不同问题；结果须与实际用户能承担的延迟和费用对应。
def pure_reasoning_benchmarks():
    text("- All of the tasks so far require linguistic and world knowledge.")
    text("- Can we isolate **reasoning** from knowledge?")
    text("- Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts).")

    text("**ARC-AGI** "), link(title="website", url="https://arcprize.org/arc-agi")
    text("- 100\% solvable by humans, but challenging for AI")
    text("- Each task is unique, so memorization doesn't help.")

    text("- ARC-AGI-1 (2019): first iteration")
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)

    text("- ARC-AGI-2 (March 2025): more multi-step reasoning")
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)

    image("images/arc-agi-results.png", width=700)
    text("- Pretrained language models didn't move the needle")
    text("- Reasoning models (o1, o3) started making things take off")

    text("- ARC-AGI-3 (March 2026): interactive environments "), post_link("https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf")
    image("images/arc-agi-3.png", width=300)
    image("images/arc-agi-3-results.png", width=500)

    text("Summary:")
    text("- Goal is to disentangle reasoning from knowledge (difficult to do!)")
    text("- Constrained to human reasoning (not superhuman reasoning)")
    text("- Clearly exposes gaps in current models")


# 【安全指标必须说明威胁模型】
# 先写清攻击者能改变什么、想诱发什么行为、目标模型有什么防护、如何判定攻击成功。
# HarmBench 用指定有害行为测试拒绝与红队方法；AIR-Bench 将法规/政策来源映射成风险分类。
# 它们覆盖的是各自定义的风险范围，不是给所有现实情境颁发“安全证明”。
# 攻击成功率 ASR 的分母应明确是行为、提示还是重复攻击次数，并披露查询预算与评判器。
# 拒绝率高也可能来自过度拒绝正常请求；应同时评估有害请求的防御和正常请求的有用性。
# GCG 在概念上用梯度指导离散提示 token 的候选搜索；跨模型迁移是特定实验结果，并非处处成立。
# 本节目的是理解测量边界，不需复现攻击；上下文、法域、部署权限与受影响人群都会改变风险。
# 同样的网络能力可服务防御或攻击，能力分数与允许的用途、权限控制需要分别评估。
# 自测：未见一次越狱是否等于风险为零？答：只能说明给定攻击集与预算下未检出。
def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=400)
    text("What does safety mean for AI?")

    text("**HarmBench** "), link("https://arxiv.org/abs/2402.04249")
    text("- Based on 510 harmful behaviors that violate laws or norms")
    link(title="HarmBench on HELM", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="Example of safety failure", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    text("**AIR-Bench** "), link("https://arxiv.org/abs/2407.17436")
    text("- Based on regulatory frameworks and company policies")
    text("- Taxonomized into 314 risk categories, 5694 prompts")
    image("https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png", width=800)
    link(title="HELM AIR-Bench", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    text("Jailbreaking:")
    text("- Language models are trained to refuse harmful instructions")
    text("- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety "), link("https://arxiv.org/pdf/2307.15043")
    text("- Transfers from open-weight models (Llama) to closed models (GPT-4)")
    image("images/gcg-examples.png", width=800)

    text("What is safety?")
    text("- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries)")
    text("- Many risks are quite varied (hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking)")

    text("**Dual-use**: capable cybersecurity agents (Mythos) can be used to hack into a system or to do penetration testing")


# 【现实代表性与构念有效性分别回答什么】
# 构念有效性：指标是否支持“这个系统具备某种能力”的解释，例如单元测试是否真的检验修复。
# 现实代表性/生态有效性：任务、人群、工具、交互过程是否贴近想推广到的实际使用。
# 两者相关但不能互换：真实用户请求可被坏评委误判，严谨考试也可能偏离真实工作流。
# 课堂补充 · 01:05:23–01:05:58：真人提问仍需追问是否来自目标人群和目标用例。
# 编者建议把目标拆成用户群、语言、任务频率、难度、风险、时间与允许工具，按层报告结果。
# 基准分布 q 与部署分布 r 不同，通常不能保证 E_q[g]=E_r[g]；更大样本只会更精确估计前者。
# 若已知目标各层权重，可按层加权；缺少目标群体的数据时，重加权也不能凭空补出证据。
def realism():
    text("**Ecological validity**: how well does an evaluation capture real-world use?")
    text("- Exam benchmarks (e.g., GPQA) are far away from real-world use.")
    text("- Chatbot Arena prompts are from real people, but distribution is uncontrolled.")

    text("**GDPVal** (OpenAI) "), link("https://arxiv.org/pdf/2510.04374")
    text("- 44 occupations from top 9 sectors according to US GDP")
    text("- Tasks come from professionals with ~14 years of experience")
    # GDPval 以职业交付物更接近工作，但其职业选择和任务设计仍有范围。
    # “来自高 GDP 行业”不是“分数提升多少，GDP 就提升多少”，也不直接证明可替代整个职业。
    # 真实职业还包括需求澄清、团队协作、责任与长期维护，单项交付物只覆盖其中一部分。
    image("images/gdpval.png", width=700)

    text("**MedHELM** "), link("https://arxiv.org/abs/2505.23802")
    text("- Previous medical benchmarks were based on standardized exams")
    # [现实] MedHELM 的 121 指临床任务分类，不是只有 121 道测试题；论文套件包含 35 个 benchmark。
    # 分类强调临床工作流的多样性，不能用标准考试成绩替代每类临床任务的可靠性证据。
    # 课堂补充 · 01:06:37–01:07:34：通过医学考试与满足临床实际任务之间仍有距离。
    text("- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets")
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png", width=700)
    link(title="MedHELM", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    text("**Clio** (Anthropic) "), link("https://arxiv.org/abs/2412.13678")
    text("- Use language models to analyze real user data")
    text("- Share general patterns of what people are asking")
    image("images/clio-table4.png", width=700)

    # Clio 从真实使用日志提取聚合主题，帮助理解用户在做什么；它不是直接判定每次任务是否成功。
    # 聚合和隐私保护会限制可见细节；模型生成的分类/摘要本身也应检查误差与遗漏。
    # 课堂补充 · 01:07:39–01:08:43：真实请求流很有评估价值，但分析和公开会受到隐私约束。
    # 因此需保留授权、去标识化与抽样边界，不能因“更真实”就无条件公开原始对话。
    text("Unfortunately, realism and privacy are sometimes at odds with each other.")


# 【有效性审计：分数是否支持所声称的结论】
# 本节原稿重点是训练测试重叠与数据质量；前面的构念、代表性和评委问题也属于广义证据检查。
# 污染可能来自原题、答案、题源、改写副本、微调数据，或反复依据测试反馈选择模型。
# 学习相关知识是期望发生的泛化；见过具体测试实例则改变“对未见题泛化”的解释。
# 课堂补充 · 00:26:55–00:27:19：不能只搜索测试集文件名，还要检查题目的来源链。
# 污染存在不代表所有成绩都由记忆造成；无污染证据也不代表已证明不存在污染。
def validity():
    text("How do we know our evaluations are valid?")

    text("### Train-test overlap")
    text("- Machine learning 101: don't train on your test set")
    text("- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits")
    text("- Today: train on the Internet and don't tell people about your data")

    text("Route 1: try to infer train-test overlap from model")
    text("- Exploit exchangeability of data points "), link("https://arxiv.org/pdf/2310.17623")
    # [污染] 图中比较规范题目顺序和打乱题目顺序的拼接序列似然；打乱的是题目，不是题内词序。
    # 在模型与测试集独立、题目序列可交换的零假设下，规范顺序不应系统性地特别高似然。
    # 这不是说每次洗牌所得 logp 必须数值相等，而是以洗牌分布校验规范顺序是否异常。
    # 若题目按难度排序、共享文章或带连续索引，可交换性可能不成立，需先排除这些结构因素。
    # 课堂补充 · 01:09:44–01:10:27：模型偏好已发布的特定题序可能提示记忆，论文给出统计检验条件。
    # 未检出可能是统计功效不足，或模型只见过部分题目/改写；不能据此颁发“无污染证明”。
    image("images/contamination-exchangeability.png", width=500)

    text("Route 2: encourage reporting norms (e.g., people report confidence intervals)")
    text("- Model providers should report train-test overlap "), link("https://arxiv.org/abs/2410.08385")

    # 透明报告可以说明查重算法、阈值、查找范围、重叠比例与重叠/非重叠子集结果。
    # 精确字符串查重查不到所有语义改写；供应商声明、可审计数据与黑盒推断是不同强度的证据。
    # 课堂补充 · 01:10:41–01:11:13：讲者用报告置信区间类比，希望重叠报告也成为常规。
    # 置信区间量化抽样不确定性，重叠审计检查系统偏差，两者不能相互替代。
    text("Route 3: use fresh evals")
    text("- LiveCodeBench, UncheatableEval: scrape new webpages")
    text("- Timestamps aren't always safe due to copying either")

    # 新测试要检查内容首次出现的时间，而非当前网页的更新时间；复制旧代码也会获得新时间戳。
    # 还应记录模型版本与截止时间，防止评测期间更新或可用检索工具改变“未见过”的含义。
    text("Route 4: use private evals")
    text("- Companies use internal code bases that aren't on the Internet")
    text("- Use your personal writings")
    text("- Easiest for perplexity")

    # 私有测试减少公开训练泄漏的机会，但需防内部训练、日志回流和反复调参；私有也不保证代表性。
    # 困惑度只需私有文本与合法概率接口，不需给每题人工标答案；代价是它测预测而非全部使用目标。
    # 数据质量审计要检查问题是否充分、参考答案是否唯一正确、验证器是否覆盖要求、环境是否可复现。
    # 删改坏题会改变测试集合，必须报告版本和规则；不能只因为模型答错就删题。
    text("### Dataset quality")
    text("- Fixed up SWE-Bench to produce SWE-Bench Verified "), post_link("https://openai.com/index/introducing-swe-bench-verified/")
    text("- Create Platinum versions of benchmarks "), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)
    # Platinum 的动机是先修复含糊、缺条件或标签有误的题，让接近高分时的差异仍有解释力。
    # Verified 也不是永久无误的保证：复杂环境中的测试可漏掉关键行为，需继续审计。
    # 已核验下述论文摘要提到 TAU-bench 空回答也能通过的漏洞；不能把字幕中的近似转写当准确名称。
    # 编者建议保留无操作/空输出等简单基线：若它们高分，先查验证器是否测到了目标行为。
    text("- Problems with agentic benchmarks: insufficient test cases, trivial agent can solve task "), link("https://arxiv.org/abs/2507.02825")
    # 轨迹审计可查看是否读到答案、改动测试、绕过任务或因环境失败；总分无法解释这些机制。
    # 自动轨迹审阅也是一个会犯错的测量器，重要结论需回到工具输出、文件与验证器记录。
    # 课堂补充 · 01:15:08–01:15:41：运行或设计 benchmark 后应实际看输出，检查是否量到想量的东西。
    # 自测：增加到一万题能修好错误标签或泄漏吗？答：不能，方差可能下降，系统偏差仍在。
    text("- Docent: use LLM to inspect agent traces to detect problems "), post_link("https://transluce.org/introducing-docent")


# 【把目标写成可执行的评估协议】
# 采购要看目标工作流的质量、费用、延迟与失败损失；研究能力要说明构念与控制变量；
# 分析利益/危害要覆盖受影响群体和风险；开发诊断则需分项错误和可行动的反馈。
# 编者检查清单：目标人群与任务 → 数据来源/划分/版本 → 提示和输出协议 →
# 模型与脚手架版本 → 工具/推理/时间/重试预算 → 指标和判分器 →
# 样本数、分层/配对区间 → 污染与质量审计 → 允许推广的范围。
# 任务失败与环境/接口失败应分开记录，同时给出对最终用户有意义的端到端成功率。
# 不要默默丢弃超时题后只报剩余题的准确率；超时是否计失败应在运行前约定。
def how_to_think_about_evaluation():
    text("### What's the point of evaluation?")
    text("There is no one true evaluation; it depends on what question you're trying to answer.")
    text("1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots).")
    text("2. Researchers want to measure the raw capabilities of a model (e.g., intelligence).")
    text("3. We want to understand the benefits + harms of a model (for business and policy reasons).")
    text("4. Model developers want to get feedback to improve the model.")

    text("### What are we evaluating?")
    text("- Pre-foundation models, we evaluated **methods** (standardized train-test splits).")
    text("- Today, we're (mostly) evaluating **models/systems** (anything goes).")

    # 方法比较要固定数据、预算和评估规则，尽量隔离算法这个变化因素；否则无法解释提升由何而来。
    # 模型/系统比较允许训练数据等不同，因为用户采购的是最终系统，但仍需声明测试规则。
    # 课堂补充 · 01:16:41–01:17:26：讲者用训练 speedrun 说明方法评估与最终系统评估的区别。
    # 固定验证损失的速度赛还要统一硬件、精度、计时口径、数据与损失计算，否则“更快”含义会变。
    text("There are some exceptions...")
    text("- nanogpt speedrun: fixed data, compute time to get to a particular validation loss")
    image("images/karpathy-nanogpt-speedrun.png", width=600), post_link("https://x.com/karpathy/status/1846790537262571739")

    text("Evaluating methods encourage algorithmic innovation from researchers.")
    text("Evaluating models/systems is useful for downstream users.")

    # 【带解答的收束自测】
    # 1. A 困惑度低于 B，但 tokenizer 不同，能直接选 A 吗？不能；先统一原文与计量单位/上下文协议。
    # 2. 真实用户偏好票很多，是否就有构念有效性？不必然；若目标是事实正确性，还需独立核查事实。
    # 3. 私有题判分完全正确，是否代表所有用户？不必然；还要检查目标人群、工作流与任务权重。
    # 4. 同模型多试十次分数变高，能说模型权重更强吗？不能；这是资源和系统协议变化的结果。
    # 5. 应怎样决定下一讲收什么训练数据？先以目标任务和可信评测诊断缺口，再在训练数据上改进；
    #    保留独立测试和来源记录，避免把评测反馈直接变成测试题训练，造成虚假的泛化提升。
    text("Either way, we need to define the rules of the game!")


if __name__ == "__main__":
    main()
