# 第14讲中文自学增强：数据流水线与合成数据（Spring 2026）。
# 学习目标：能解释过滤分数代表什么，手算 Jaccard 与 MinHash 碰撞概率，
# 由单次碰撞推导 LSH 候选概率，并把混合比例换算为每个来源的重复训练次数。
# 先修：集合交并、条件概率、独立事件、伯努利均值与方差，以及语言模型交叉熵。
# 阅读主线：先决定“文本是什么”，再决定“保留什么”，然后消除冗余、分配 token 预算；
# 最后把同一套质量与污染控制用于教师模型生成的训练数据。
# 来源索引：本文件原英文/代码为主；共享缓存 sources/lecture_14.srt 为第三方字幕，
# 不是教师审校逐字稿。下文标“编者补充”的证明、算例和风险分析是教学推演。
# 课堂核对位置：00:31:35–00:32:54（Jaccard/MinHash）；00:39:10–00:39:22（band 结构）。
# 本轮复用原稿与缓存，不新增外部资料；原稿已有论文链接留在对应主题处供追溯。
# 自学建议：先算出两个示例集合的相似度，再读哈希实现；不要把候选概率当成删除阈值。
from dataclasses import dataclass
import numpy as np
import itertools
import mmh3
from edtrace.file_util import download_file
from edtrace import text, image, link
from lecture_13 import the_pile
from lecture_util import article_link, post_link
from references import dolma_2024, the_pile_2020, dclm_2024

def main():
    text("## Lecture 14: Data II")
    text("Last lecture:")
    text("- Live service (e.g., GitHub) → dump/crawl (e.g., GitHub Archive) → processed data (e.g., The Stack)")
    text("- Considerations: terms of service, copyright (licenses or fair use)")

    text("This lecture:")
    text("- Data pipeline: transformation, filtering, deduplication, mixing")
    text("- Mid-training + SFT: synthetic data")

    # Data pipeline
    transformation()
    filtering()
    deduplication()
    data_mixing()

    # Post-training data
    post_training_data()

    text("Summary:")
    text("- Filtering: train classifier (language id, quality, toxicity) for what good looks like")
    text("- Deduplication: hashing scales to large datasets for fuzzy matching")
    text("- Mixing: try mixtures at small scale, extrapolate to optimal mixture and large scale")
    text("- Applications: language identification, quality filtering, toxicity filtering")
    text("- Post-training data: looks like evaluations, use of synthetic data")
    text("- A lot of data work is domain-specific, looking at examples, etc.")


# 【文本转换：后续指标的输入契约】
# 编者补充：网页的 DOM 树含正文、导航、广告；PDF 主要描述页面上的排版位置。
# 转成一维 token 序列必须选择阅读顺序，表格跨列与公式上下标可能因此丢失。
# 例如“名称 价格 / A 10 / B 20”若列序错位，就可能变成错误的价格对应关系。
# OCR 错字还会降低重复文档的 shingle 重合度，使去重漏检；导航残留则会提高无关网页相似度。
# 因此不能只统计提取速度：应抽样对照原页面，检查正文保留、乱码、表格和数学符号。
# 原图所比较的抽取方案应理解为具体实验结果，不能推出某个工具在所有站点都最佳。
def transformation():
    text("Raw data does not come as text.")
    text("It is HTML, PDF (arxiv), or directories (code repositories).")

    text("HTML to text (main one):")
    text("- Remove boilerplate (e.g., navigation, ads) and extract content")
    text("- What about images, tables, etc.?")
    text("- Inherently lossy (need to linearize)")
    text("- Tools (rule-based): trafilatura, resiliparse, jusText, lynx, etc.")
    text("- Accuracy matters: "), link(dclm_2024)
    image("images/dclm-wet.png", width=300)

    text("FinePDFs "), post_link("https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog")
    image("https://huggingfacefw-finepdfsblog.hf.space/_astro/pdf-description.Cb49jXc6_Z17eX4E.webp", width=600)
    text("- Source: Common Crawl")
    text("- Recrawl truncated PDFs (since they are big)")
    text("- OCR (RolmOCR) using a VLM or Docling (make these run fast)")
    text("- Lots of cleanup and filtering")
    text("- A lot of layout information is missing")


# 【过滤：把目标分布变成可计算的代理分数】
# T 是“希望接近的文本样本”，R 是待筛选的大池；T′ 来自 R，不是复制 T。
# 关键前提是 T 足以代表训练目标，且标注/采样差异没有变成分类捷径。
# 例如正例全是排版整齐的百科，分类器可能只学会百科格式，错删有价值的口语或代码。
# 语言、质量、毒性是不同目标；“像正例”既不等于事实正确，也不等于对所有下游任务有用。
# 若多个过滤器串联，必须检查最终交集：逐项看起来合理的阈值可能共同删掉某类少数文本。
def filtering():
    text("Algorithmic building block:")
    text("- Given some **target data** T and lots of **raw data** R, find subset T' of R similar to T.")
    image("images/raw-target-schema.png", width=600)

    text("Applications:")
    text("- Language identification (English versus rest)")
    text("- Quality filtering (high quality versus low quality)")
    text("- Toxicity filtering (non-toxic versus toxic)")

    text("Desiderata for filtering algorithm:")
    text("- Generalize from the target data (want T and T' to be different)")
    text("- Extremely fast (have to run it on R, which is huge)")

    text("Survey paper on data selection "), link("https://arxiv.org/abs/2402.16827")

    text("General framework: Given target T and raw R, find subset of R similar to T")
    text("1. Estimate some model based on R and T and derive a scoring function")
    text("2. Keep examples in R based on their score")

# 编者补充：生成模型 p_T(x) 衡量目标语料下的似然；原始序列概率会随长度快速下降，
# 实际比较常用每 token 负对数似然 L=−Σ log p(x_i|x_<i)/m，困惑度 PPL=exp(L)。
# 因此需要统一 tokenizer 和长度处理；不同模型的 PPL 阈值不能直接互换。
# p(T|x) 是在给定训练正负例比例下的分类分数；训练集先验、域偏移与校准都会影响数值。
# 阈值调高通常提高精确率、降低召回率，但“精确”只相对于既定质量标签而言。
    text("Types of classifiers:")
    text("- Generative model of T (KenLM): score(x) = p_T(x)")
    text("- Simple classifier (fastText): score(x) = p(T | x)")
    text("To use: keep examples x with score(x) >= threshold (stochastically)")

    text("Model-based filtering?")
    text("- Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)")
    text("- Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]")

    text("Language identification:")
    text("- Goal: find text of a specific language (e.g., English)")
    text("- fastText language identification "), article_link("https://fasttext.cc/docs/en/language-identification.html")
    text("- Off-the-shelf classifier")
    text("- Supports 176 languages")
    text("- Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)")
    text("- Dolma keeps pages with p(English) >= 0.5 "), link(dolma_2024)

    text("OpenMathText "), link("https://arxiv.org/pdf/2310.06786")
    text("- Goal: curate large corpus of mathematical text from CommonCrawl")
    text("- Use rules to filter (e.g., contains latex commands)")
    text("- KenLM trained on ProofPile, keep if perplexity < 15000")
    text("- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math")
    text("- Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data")

    text("GPT-3 "), link("https://arxiv.org/pdf/2005.14165")  # Appendix A
    text("- Positives: samples from {Wikipedia, WebText2, Books1, Books2}")
    text("- Negatives: samples from CommonCrawl")
    text("Train linear classifier based on word features "), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    text("Keep documents stochastically based on score")
# 编者补充：这里 np.random.pareto 使用 Lomax 形式，X≥0 且 P(X>t)=(1+t)^−9。
# 若 score=q∈[0,1]，保留概率为 P(X>1−q)=(2−q)^−9：
# q=0 时为 1/512，q=1 时为 1；它偏好高分但仍保留少量低分样本，不是硬阈值。
# 这仅解释当前代码的采样规则，不能把 q 自动解释成真实质量概率。
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    text("LLaMA/RedPajama "), link("https://arxiv.org/pdf/2302.13971")
    text("- Positives: samples from pages **referenced** by Wikipedia")
    text("- Negatives: samples from CommonCrawl")
    text("- Keep documents that are classified positive")

    text("phi-1 "), link("https://arxiv.org/pdf/2306.11644")
    text("- Philosophy: really high quality data (textbooks) to train a small model (1.5B)")
    text("- Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data")
    R = "Python subset of the Stack"   # Raw data
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    text("- Train random forest classifier on T using output embedding from pretrained codegen model")
    text("- Select data from R that is classified positive by the classifier")
    text("Result on [HumanEval](https://huggingface.co/datasets/openai_humaneval):")
    text("- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)")
    text("- Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!")

    text("Toxicity filtering in Dolma "), link(dolma_2024)
    text("- Dataset: Jigsaw Toxic Comments dataset (2018) "), link(title="dataset", url="https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    text("- Project goal: help people have better discussions online "), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    text("- Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}")

# 【过滤与预算的联系】
# 固定训练 token 预算时，更严格过滤可能提高平均质量，却缩小独特数据池。
# 短训练尚未用尽高分文本；长训练若反复循环它们，新增低分但不同的文本可能更有价值。
# 因此阈值比较应控制训练预算，并记录保留量、重复次数、域分布及独立验证损失。
# phi-1 等结果属于对应训练设置的经验比较，不能仅凭步数较少推出质量提升的唯一原因。
    text("Scale-dependent effects of filtering:")
    text("- No single optimal threshold for filtering")
    text("- If training for longer, want more (lower quality) data")
    text("- If training for shorter, want less (higher quality) data")
    image("images/data-filtering-scale.png", width=800)

    text("Summary:")
    text("- Filtering is critical for building a good model")
    text("- Recipe: define target data (what good looks like), extrapolate to raw data")
    

# 【去重：先定义对象与相似，再谈速度】
# 整篇去重会保留包含重复段落的不同文档；段落去重更细，却可能破坏上下文。
# 标准化大小写、空白或标点能增加匹配，但也可能合并原本有意义的差别，如代码缩进。
# “近重复”不是语义等价：改写可语义相同但词面不同；模板可词面相似却数字结论不同。
# 哈希可避免全体两两比较的 O(N²) 成本，但超大桶与候选爆炸仍可能使最坏情况很昂贵。
def deduplication():
    text("Two types of duplicates:")
    text("- Exact duplicates (mirror sites, GitHub forks) "), link(title="Gutenberg mirrors", url="https://www.gutenberg.org/MIRRORS.ALL")
    text("- Near duplicates: same text differing by a few tokens")

    text("Examples of near duplicates:")
    text("- Terms of service and licenses "), link(title="MIT license", url="https://opensource.org/license/mit")
    text("- Formulaic writing (copy/pasted or generated from a template) "), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600)
    text("- Minor formatting differences in copy/pasting")

    text("Product description repeated 61,036 times in C4")
    text("'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!")
    link(title="example page", url="https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    text("Deduplication training data makes language models better "), link("https://arxiv.org/pdf/2107.06499")
    text("- Train more efficiently (because have fewer tokens)")
    text("- Avoid memorization (can mitigate copyright, privacy concerns)")

    text("Design space:")
    text("1. What is an item (sentence, paragraph, document)?")
    text("2. How to match (exact match, existence of common subitem, fraction of common subitems)?")
    text("3. What action to take (remove all, remove all but one)?")

    text("Key challenge:")
    text("- Deduplication is fundamentally about comparing items to other items")
    text("- Need linear time algorithms to scale")

    hash_functions()
    exact_deduplication()
    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    text("- Hash function h maps item to a hash value (integer or string)")
    text("- Hash value much smaller than item")
    text("- Hash collision: h(x) = h(y) for x ≠ y")

    text("Tradeoff between efficiency and collision resistance "),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    text("- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)")
    text("- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)")

    text("We will use MurmurHash:")
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    text("**Simple example**")
    text("1. Item: string")
    text("2. How to match: exact match")
    text("3. Action: remove all but one")

    # Original items
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # Compute hash -> list of items with that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

# 编者补充：此教学实现把“哈希值相同”直接当作完全相同；有限位 MurmurHash 会碰撞。
# 生产精确去重应在桶内比较原始/规范化内容，或用更长指纹并明确残余碰撞风险。
# 此外 sorted 的成本为 O(N log N)；MapReduce 式可分布执行不等于本段代码严格线性。
    # Keep one item from each group
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    text("- Pro: simple, clear semantics, high precision")
    text("- Con: does not deduplicate near duplicates")
    text("- This code is written in a MapReduce way, can easily parallelize and scale")

    text("**C4** "), link("https://arxiv.org/pdf/1910.10683v4")
    text("1. Item: 3-sentence spans")
    text("2. How to match: use exact match")
    text("3. Action: remove all but one")
    text("Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent")


# 【从文档到集合，再到紧凑签名】
# 编者补充：常把文档表示为连续 k 个词组成的 shingles 的集合。
# “a b a b”的二元组为 {“a b”,“b a”}，重复出现只计一次；集合 Jaccard 不保留词频。
# k 太小会把共享常用词的文档误判相似，太大则对少量插入/改写很敏感；须先固定分词与规范化。
# Jaccard 适合相对长度接近的词面比较：短文完全包含于长文时仍可能得低分，
# 例如 |A|=10、|B|=100 且 A⊂B，则 J=0.1；若任务是查抄袭片段，应另考虑 containment。
# 空集需约定：下面函数对两个空集会除以零，minhash 对空集也无最小值，应预先分流。
def jaccard_minhash():
    text("Let's now look at approximate set membership.")
    text("First we need a similarity measure.")

    text("### Jaccard similarity")
    text("Definition: Jaccard(A, B) = |A intersect B| / |A union B|")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

# 手算：A∩B={1,2,3}，A∪B={1,2,3,4,5}，故 s=3/5=0.6。
# 分母不是 |A|+|B|=8，因为交集元素不能在并集中重复计数。
    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    text("Definition: two documents are **near duplicates** if their Jaccard similarity >= threshold")

    text("Algorithmic challenge: find near duplicates in linear time")

    text("### MinHash")
    text("MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B)")

    text("Normally, you want different items to hash to different hashes")
    text("...but here, you want collision probability to depend on similarity")

# 【MinHash 等式为何成立】
# 编者补充：理想模型是在共同元素全集上取均匀随机排列 π，签名取集合中排名最前元素。
# 令 U=A∪B；U 中每个元素成为最前者的概率均为 1/|U|。
# 若最前者在 A∩B，它同时是 A、B 的最前者；若仅在一侧，两侧最前者不同。
# 所以 P[签名相同]=|A∩B|/|U|=s。两集合必须使用同一排列，否则此论证不成立。
# 代码取最小哈希值近似随机排列；有限哈希碰撞及哈希族非理想性使精确等式变成近似。
# 有符号哈希的负值也有固定顺序，不是概率证明的关键；均匀次序和碰撞处理才是关键。
    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    text("Characteristic matrix representation:")
    text("item | A | B", verbatim=True)
    text("1    | 1 | 1", verbatim=True)
    text("2    | 1 | 1", verbatim=True)
    text("3    | 1 | 1", verbatim=True)
    text("4    | 1 | 0", verbatim=True)
    text("5    | 0 | 1", verbatim=True)

    text("Random hash function induces a permutation over items")
    text("Look at which item is first in A and which item is first in B.")
    text("Each item has the same probability as being first (min)")
    text("- If 1, 2, 3 is first, then first in A = first in B.")
    text("- If 4, 5 is first, then first in A ≠ first in B.")

# 【估计值不是确定性保证】
# 用独立排列得到 m 个碰撞指示量 X_i∈{0,1}，均值 ŝ=ΣX_i/m 的期望为 s，
# 方差为 s(1−s)/m。s=0.6、m=100 时标准差约 0.049，远大于 0.01。
# 所以下面的 assert 是对特定 seed/实现结果的演示检查，不是任意随机试验必然成立的定理。
# 更多独立签名通常使估计更稳定；重用高度相关的哈希不能直接套用上述方差。
    # Verify MinHash approximates Jaccard as advertised
    n = 100  # Generate this many random hash functions
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]  # @stepover
    estimated_jaccard = len([m for m in matches if m]) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    text("Now we can hash our items, but a collision doesn't tell us Jaccard(A, B) > threshold.")


# 【为什么还需要 LSH】
# MinHash 缩短了每篇文档的表示，但若逐对比较签名，N 篇文档仍有约 N²/2 对。
# LSH 用签名分桶生成少量候选，再对候选算精确 Jaccard 或其他确认指标。
# 单个 MinHash 在 s=0.6 时仅以 0.6 概率碰撞，不能直接实现“s≥阈值才匹配”。
# 候选生成允许假阳性与漏检；最终阈值确认可去掉候选中的假阳性，却找不回遗漏的文档对。
def locality_sensitive_hashing():
    text("Locality sensitive hashing (LSH) "), link(title="book chapter", url="http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    text("Suppose we hash examples with just one MinHash function")
    text("P[A and B collide] = Jaccard(A, B)")
    text("On average, more similar items will collide, but very stochastic...")

    text("Goal: have A and B collide if Jaccard(A, B) > threshold")
    text("We have to somehow sharpen the probabilities...")

    text("Solution: use n hash functions")
    text("Break up into b bands of r hash functions each (n = b * r)")

    n = 12      # Number of hash functions
    b = 3       # Number of bands
    r = 4       # Number of hash functions per band
    text("Hash functions:")
    text("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", verbatim=True)

    text("Key: A and B collide if for *some* band, *all* its hash functions return same value")
    text("As we will see, the and-or structure of the bands sharpens the threshold")

    text("Given Jaccard(A, B), what is the probability that A and B collide?")

# 【AND/OR 概率逐步推导】
# 记真实 Jaccard 为 s；假定各行使用独立的理想 MinHash，band 的桶键无额外碰撞。
# 一个 band 内 r 行全相同是 AND，概率 s^r；该 band 不匹配的概率是 1−s^r。
# b 个独立 band 全不匹配为 (1−s^r)^b；取补集便得至少一个匹配概率 1−(1−s^r)^b。
# 此式是“给定相似度后成为候选”的概率，不是“已经碰撞后为真重复”的后验概率。
# 后者还依赖整个语料中相似文档对的比例；海量无关文档也可能产生许多假候选。
    def get_prob_collision(sim, b, r):  # @inspect sim @inspect b @inspect r
        prob_match = sim ** r                        # Probability that a fixed band matches  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches  @inspect prob_collision
        return prob_collision

    text("**Example**")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs @stepover

# 编者补充：s=0.8、b=5、r=10 时 s^r≈0.1074，候选概率≈0.4333。
# 固定 b 增大 r，会降低 0<s<1 的候选概率；固定 r 增大 b 会提高召回，也增加候选数。
# 若固定签名总长 n=br，则增大 r 同时减少 b，不能将两种独立调参描述混为一谈。
# 曲线始终是软概率曲线；“变陡”应看具体参数区间，不是所有点斜率都单调增加。
    text("Increasing r sharpens the threshold and moves the curve to the right (harder to match)")
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs @stepover

    text("Increasing b moves the curve to the left (easier to match)")
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs @stepover
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    text("Example setting "), link("https://arxiv.org/pdf/2107.06499"), text(": n = 9000, b = 20, r = 450")
    b = 20
    r = 450
    text("What is the threshold (where the phase transition happens)?")
# 编者补充：t=b^(−1/r) 是令每 band 成功率为 1/b 的常用转折位置近似，
# 不是硬阈值，也不是恰好 50% 召回点。b=20、r=450 时 t≈0.993365，
# 此处候选概率为 1−(19/20)^20≈0.6415；b 很大时才趋近 1−1/e≈0.6321。
# 若要精确的 50% 候选点，解 1−(1−s^r)^b=0.5，得 s=(1−2^(−1/b))^(1/r)。
    threshold = (1 / b) ** (1 / r)  # @inspect threshold

    text("Probability that a fixed band matches:")
    prob_match = (1 / b)  # @inspect prob_match
    text("Probability that A and B collide is a constant (≈ 1-1/e):")
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


# 【去重后如何成组：非传递性的陷阱】
# 编者补充：若确认后的近重复对作为图的边，连通分量可以用于选一个代表，但不保证组内每对相似。
# 例如 A={1,2}、B={1,2,3}、C={2,3}，J(A,B)=J(B,C)=2/3，J(A,C)=1/3。
# 阈值 0.6 时 A—B—C 连通，A 与 C 却不是近重复；删到只剩一个代表可能损失独特内容。
# 应按任务选择连通分量、代表约束或更严格的簇内检查，同时记录代表选择的质量规则。
def billion(x):
    return x * 10**9

def trillion(x):
    return x * 10**12


# 【混合：概率、token 数和重复次数是三个量】
# 编者补充：下文 p(s) 是来源 s 的 token 预算比例，p(s)≥0 且 Σ_s p(s)=1。
# 若按文档抽样，不同来源平均文档长度不同，则文档概率并不直接等于 token 比例。
# 总预算 D、来源独特 token 数 N_s 给出分配 D_s=p(s)D，平均重复次数 e_s=D_s/N_s。
# 均匀来源采样不等于均匀 token 采样；质量加权还应保留目标任务所需的领域覆盖。
def data_mixing():
    text("Recall that language models are trained on multiple data sources.")

    text("Datasets in Marin: "), link(title="token viewer", url="https://huggingface.co/spaces/marin-community/token-count-viewer")
    image("images/marin-token-viewer.png", width=800)

    text("The Pile "), link(the_pile_2020)
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)
    text("Key question: what distribution over the data sources should we use?")

    text("Example:")
    sources = {"Wikipedia", "CC", "GitHub"}
    p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}  # One possible data mixture

    text("Baselines:")
    text("- Vibes: set p(s) manually based on intuition (quite common)")
    text("- Uniform sampling: sample uniformly (p(s) ∝ 1)")
    text("- Proportional mixing: sample proportional to the number of tokens in a source (p(s) ∝ num_tokens(s))")

    text("Intuition: should upweight higher quality sources")
    text("However...")
    text("1. We want to ensure diversity (e.g., across incomparable sources: literature, code, papers)")
    text("2. Each source is finite, so if put too much weight on a small source, then need to epoch over it")
    
    text("This last point is important and a bit subtle.")
    text("Example:")
    source_token_counts = {
        "low": trillion(10),  # 10T tokens (abundant) @stepover
        "high": billion(10),  # 10B tokens (scarce) @stepover
    }
    p = {"low": 0.5, "high": 0.5}  # Naive data mixture
    train_tokens = trillion(1)  # Train for 1T tokens @stepover
    low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]  # @inspect low_num_epochs
    high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"]  # @inspect high_num_epochs
    text("50x epochs on high quality data...can lead to overfitting!")

# 手算本例：低质量来源分配 0.5T，除以 10T 得 0.05 遍；高质量来源分配 0.5T，
# 除以 10B 得 50 遍。后者没有得到 50 倍独特知识，只是反复训练同一批内容。
# 重复并非必然有害，但边际收益会衰减；是否过拟合应看独立验证集，而非只看训练损失。
    text("UniMax "), link("https://arxiv.org/abs/2304.09151")
    text("- Setting: balancing different languages for multilingual models")
    text("- Previous work: between uniform and proportional mixing (p(s) ∝ num_tokens(s)^α for α in [0, 1])")
    text("- Idea: sample sources uniformly but with a hard **cap** C on number of epochs for any source")
    text("- Specifically, p(s) * num_training_tokens ≤ C for all sources s")

# 【原稿公式的单位说明】
# 若 C 表示允许的最大 epoch 数，上方 p(s)*num_training_tokens≤C 少了来源大小因子。
# 应理解为 p(s)D/N_s≤C，即 p(s)D≤C N_s；左边 token 数、右边也应是 token 数。
# 当所有来源总容量 Σ_s C N_s 小于预算 D 时，硬上限下根本无可行混合，必须调整预算或上限。
# 回归混合在小训练上估计 p→验证损失，再搜索 p；搜索会放大回归误差，尤其在训练混合覆盖不足处。
# 用于挑选混合的验证集已参与开发，最终效果还需独立测试；反复挑最高分也会过拟合评测集。
    text("Regression-based mixing "), link("https://arxiv.org/abs/2407.01492"), link("https://arxiv.org/pdf/2602.12237")
    image("images/regmix.png", width=700)
    text("- Define distribution over mixtures `p` (e.g., Dirichlet) ")
    text("- Define regression method (e.g., linear, gradient boosted trees)")
    text("- Define target based on downstream evals (careful not to overfit!)")
    text("- Discrepancy between small and large scale (tradeoff cost and accuracy)")
    image("images/data-mixing-methods.png", width=700)
    text("Hope 1: regression model is accurate at minimizer 🙏")
    text("Hope 2: optimal data mixtures transfer from small to large scale 🙏")

    text("Hold on. There's at least one scale-dependent effect:")
    source_token_counts = {
        "low": trillion(10),  # 10T tokens (abundant) @stepover
        "high": billion(10),  # 10B tokens (scarce) @stepover
    }
    text("- If train small models on low token counts:")
    p = {"low": 0.1, "high": 0.9}  # More mass on high quality data
    text("- But if train large model on this mixture, we will epoch a ton on high quality data and overfit!")

    text("Simulated epoching "), link("https://arxiv.org/pdf/2501.11747")
    text("- General idea: make small scale look like large scale (general theme of this course)")
    text("- Instantiation: downsample all sources proportionally")
    small_run_tokens = billion(10)  # @stepover
    large_run_tokens = trillion(1)  # @stepover
    ratio = small_run_tokens / large_run_tokens  # @inspect ratio
    downsampled_source_token_counts = {s: count * ratio for s, count in source_token_counts.items()}  # @inspect downsampled_source_token_counts
# 【模拟重复的代数依据】
# 此处缩放比 q=D_small/D_large=10B/1T=0.01，各来源容量变成 qN_s。
# 保持同一混合 p 时，小实验的重复次数 pD_small/(qN_s)=pD_large/N_s，恰好等于大实验。
# 例如高质量来源从 10B 缩到 0.1B，若 p=0.9，小训练用 9B，仍是 90 遍。
# 它只匹配了重复次数；模型容量、优化动态、抽样后领域覆盖及泛化不会因此自动相同。
    text("- In this downsampled mixture, models that epoch too much won't look good.")
    text("- So the optimum will be more balanced.")
    p = {"low": 0.7, "high": 0.3}  # More mass on high quality data

    text("Summary:")
    text("- Problem: how to weight different data sources (e.g., Wikipedia, general, code)")
    text("- Regression-based mixing: estimate mixture → loss at small scale, optimize (analogous to scaling laws)")
    text("- Important consideration: epoching and overfitting (solution: cap or simulated)")


# 【合成数据：教师输出是训练信号，也需要质量控制】
# 编者补充：mid-training 常继续训练领域文本/推理轨迹；SFT 则用任务提示和期望回复训练条件行为。
# 真实问题+教师回复、真实仓库+生成缺陷、完全生成的问题，来源与可验证程度各不相同。
# 多次采样增加解法覆盖，但同一提示的 16 个回复不是 16 个独立问题；应按提示/来源划分数据。
# 教师能力强不保证教学信号最佳：解法风格、正确性、长度与学生容量都影响迁移。
# 原稿“过滤无益”等结论属于对应实验，不能推出所有合成数据都不需要答案验证。
def post_training_data():
    text("Recipe:")
    text("1. Define a set of environments")
    text("2. Define a set of tasks / prompts")
    text("3. Collect responses from a strong model (teacher)")

    text("OpenThoughts "), link("https://arxiv.org/abs/2506.04178")
    text("- 1.2M examples using QwQ-32B as a teacher")
    text("- Questions come from 27 human and synthetic sources (e.g., StackExchange, NuminaMath, Chemistry)")
    image("images/openthoughts-sources.png", width=500)
    text("- Sampling multiple (16) responses per prompt is helpful")
    text("- Better models aren't necessarily better teachers: QwQ-32B is a better teacher than DeepSeek-R1")
    text("- Answer filtering wasn't helpful")
    text("- Smaller high quality sources (e.g., OpenMath-2-Math) is better than large diverse sources")
    image("images/openthoughts-pipeline.png", width=600)

    text("SWE-smith "), link("https://arxiv.org/abs/2504.21798")
    image("images/swe-smith.png", width=500)
    text("- Given a repository, use LM to generate tasks (introduce bugs with LM)")
    text("- 128 GitHub repositories yields 50K tasks")

# 【软件任务的环境与泄漏边界】
# 仓库任务不仅是输入输出函数：依赖版本、文件布局、测试和工具交互都会改变成功条件。
# 无执行反馈的轨迹生成更便宜，却不能以“模型认为修好了”替代真实测试通过。
# 删除未来 git 提交可阻止代理直接查看答案，但不消除教师预训练已经见过补丁的可能。
# 课堂图中的无执行表现和 world model 解释是经验观察/解释，不是任意仓库可免执行的保证。
    text("SWE-Zero "), link("https://arxiv.org/abs/2604.01496")
    text("- SWE tasks have heavy dependencies (unlike math or coding contests)")
    text("- Setting up thousands of Docker images is an infrastructural nightmare")
    text("- Observation: strong models can solve many tasks without execution feedback")
    image("images/swezero-noexec.png", width=600)
    text("Key: strong models have internal \"world model\" of code semantics")
    text("- SWE-Zero: 300K agent trajectories that don't require repository-specific execution")
    text("- 150K GitHub PRs")
    text("- OpenHands scaffold, remove future git commits to prevent \"git hacking\" by agent")
    image("images/swezero-prompt.png", width=600)
    text("- Distilled from Qwen3-Coder-480B + filtering (try to execute anyway)")
    text("- SWE-Hero: 13K agent trajectories that do require execution feedback")
    image("images/swezero-results.png", width=700)

    text("SWE-rebench "), link("https://arxiv.org/pdf/2505.20411")
    text("- 21K interactive Python SWE tasks from 3.4K GitHub repositories")
    text("- 450K PRs from GitHub and GitHub Archive")
    text("- Used Qwen 2.5-72B-Instruct to install dependencies and assess PR quality")
    image("images/swe-rebench.png", width=600)

    text("SWE-ZERO-12M-trajectories "), link(title="data", url="https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories")
    text("- Scale SWE-Zero up to 12M agent trajectories")
    text("- Used the SWE-rebench-v2 tasks (32K executable tasks + 120K nonexecutable tasks)")
    text("- Ran mini-coder-1.7b (very small model, 50.4 pass@100), mini-swe-agent scaffold")
    text("- [Example](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories/viewer/default/train?row=5&conversation-viewer=0)")

# 【合成数据与评测污染：必须追溯完整来源链】
# 编者补充：评测题、参考解答或同一 PR 的修复若进入提示/教师上下文，生成回复仍携带答案；
# 改写题面、换变量名或让教师重述，不能把受污染样本变成独立训练数据。
# 风险链条包括：评测题→生成种子→教师轨迹→学生训练，以及评测反馈→过滤/混合参数选择。
# 软件任务应按仓库、时间、问题/PR 家族核查重叠，保留生成提示、教师版本、提交快照和过滤记录。
# 先划分训练/开发/测试，再生成与过滤；去重应跨这些分区检查，而不只在训练集内部运行。
# 文本去重可发现词面重合，但不能证明教师没见过测试题，也不能覆盖所有语义改写。
# 【带解答自测】
# 1. 两集合共享 3 项、并集 5 项，一次 MinHash 相同是否可删除？答：s=0.6，碰撞仅是随机候选证据。
# 2. band 有 4 行、共 3 个 band，s=0.6 时候选概率？答：1−(1−0.6^4)^3≈0.3406。
# 3. 10B 的来源在 1T 训练中占 20% 是几遍？答：0.2×1000B/10B=20 遍，不能只看 20% 很小。
# 4. 教师把测试题改写后再作答能用于训练吗？答：仍有评测来源泄漏，应隔离该题及其派生样本。
    text("Summary:")
    text("- Generating prompts: fully-synthetic, semi-synthetic (real environment + synthetic tasks), real (GitHub PRs)")
    text("- Responses: from capable models (that are also good teachers)")
    text("- Code environments are painful")
    text("- Lots of filtering and other details")


if __name__ == "__main__":
    main()
