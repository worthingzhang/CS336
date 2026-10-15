# 第13讲中文自学导读：数据从哪里来（Spring 2026 原稿增强）。
# 学习目标：能沿来源、快照、抽取、筛选、去重、混合、训练追踪一条数据；
# 能比较本讲语料的设计选择，并区分可访问、获许可、发布、实际用于训练四种证据。
# 先修：自回归语言模型的交叉熵、tokenization、训练/验证划分；本讲不要求爬虫实现经验。
# 阅读主线：先理解获取与权利边界，再看通用网页与专门来源，最后比较加工语料。
# 每个函数前的中文解释、手算和自测均是编者补充，除显式时间戳外不冒充课堂原话。
# 来源索引：原稿各相邻 link/article_link 为论文与数据卡入口；沿用原稿的历史数值，
# 不把它们当成今日实时规模或统一 tokenizer 下的可比测量。第三方字幕仅辅助核对：
# https://jingzechen.github.io/assets/courses/stanford-cs336-s26/lectures/013/captions.source.srt
# 本地缓存：控制目录 sources/lecture_13.srt（不是教师审校逐字稿）。
# 补充核验：https://commoncrawl.org/get-started （格式与处理链）；
# https://info.arxiv.org/help/license/index.html （论文许可与站点分发权的区别）。
# 保护期概念核验：美国法典17 USC 302（非统一75年）：
# https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section302&num=0&edition=prelim
# 法律段落是课程概念辨析，不将原稿中的案件进度或简化规则更新为当前法律结论。

from edtrace import text, image, link
from lecture_util import article_link
from references import dclm_2024, nemotron_cc_2024, olmo_2_2025, llama_3_2024, gpt2_2019, openwebtext_2019, gopher_2021, alpaca_2023


# 整讲地图与覆盖清单：动机/训练阶段 → 在线来源与获取限制 → 著作权 →
# Common Crawl/Wikipedia/GitHub/arXiv → BERT、WebText、CCNet、C4、GPT-3、
# The Pile、MassiveText、LLaMA、RefinedWeb/FineWeb、Dolma、DCLM、Nemotron、
# The Stack、CommonPile。比较时始终问：上游是什么、做了什么、保留了什么、如何验证？
# “原始”是相对处理链而言：HTML已经是服务器生成的表示，文本快照也不是现实世界本身。
# 本讲给数据地图，下一讲深入过滤、去重与混合；不要在此把某个阈值记成通用最佳值。
def main():
    text("## Lecture 13: Data I")
    text("Previous lectures: how to train a model *given data*")
    text("Next two lectures: *what data* should we train on?")

    motivation()

    # Origin of data
    raw_sources()    # What does data come from?
    copyright()      # What data can we use?

    # Sources of data
    common_crawl()   # Web crawl
    wikipedia()      # General knowledge
    github()         # Code
    arxiv()          # Research papers

    # Data from various models
    bert()                # Wikipedia, books (trained BERT) [2019]
    gpt2_webtext()        # pages based on Reddit links (trained GPT-2) [2019]
    ccnet()               # Filter Common Crawl based on Wikipedia [2019]
    t5_c4()               # Filter using rules (trained T5) [2019]

    gpt3()                # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
    the_pile()            # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
    gopher_massivetext()  # Filter using rules (trained Gopher) [2021]
    llama()               # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
    refinedweb()          # CommonCrawl (used to train Falcon) [2023]
    dolma()               # Lots of different sources [2024]
    dclm()                # Filtered using good quality classifier [2024]
    nemotron_cc()         # Lots of tokens [2024]
    the_stack()           # Code dataset
    common_pile()         # Properly licensed data

    text("### Summary")
    text("- Key lesson: Data does not fall from the sky. You have to work to get it.")
    text("- Live service → raw data → processed data (transformation, filtering, deduplication)")
    text("- Data is the key ingredient that differentiates language models")
    text("- Legal and ethical issues (e.g., copyright and privacy)")
    text("- Much of this pipeline is heuristic, many opportunities to improve!")


# 为何系统课之后讲数据？固定模型和算力仍可因训练分布变化而学到不同能力。
# 自回归目标 L = E_{x~q}[-Σ_t log pθ(x_t|x_<t)] 中，数据工程改变的是 q；
# 优化器把损失降得更低，并不能修复 q 中缺少某种语言、重复广告或答案泄漏的问题。
# 开放权重只使参数可下载，不自动公开逐文档数据、抓取日期、过滤代码和混合权重。
# 下方 OLMo 图应连读为“广覆盖 → 能力定向 → 行为对齐”，不是三种互不相干的模型。
# mid-training 可能提高数学/代码等内容占比；post-training 改变交互行为与偏好。
# “质量”依任务定义：聊天答案、代码、百科各有用途；阶段边界和 base 的命名并不统一。
# 自测：有模型架构和权重就能复现吗？不能，还缺数据版本、采样、顺序、训练设置等。
def motivation():
    text("**Data** is the most important thing to get right in training language models.")

    text("One justification: let's see what companies disclose.")
    text("Open-weight models (e.g., Llama 3 "), link(llama_3_2024), text(" have full transparency into architecture")
    text("...and even training procedures")
    text("...but basically no information on data.")
    image("images/llama3-data.png", width=700)
    
    text("Reasons for secrecy:")
    text("1. Competitive dynamics")
    text("2. Copyright liability")

    text("- Before foundation models, data work meant heavy annotation of labeled data for supervised learning.")
    text("- Now there's less annotation, but there's still a lot of curation and cleaning.")
    text("- Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems).")

    text("Stages of training:")
    text("1. Pre-training: train on raw text (e.g., documents from the web)")
    text("2. Mid-training: train more on high quality data to enhance capabilities")
    text("3. Post-training: train on chat transcripts or reinforcement learning")
    text("In practice, the lines are blurry and there could be more stages")
    text("...but the basic trend is throughout training, we go from")
    text("large amounts of lower quality data to")
    text("small amounts of high quality data.")

    text("Terminology:")
    text("- Base model: after pre-training + mid-training")
    text("- Instruct/chat model: after post-training")
    text("(Increasingly, base models are not released - e.g., Qwen3.5-397B-A17B is an instruct model.)")

    text("Example (OLMo from AI2) "), link(olmo_2_2025)
    text("1. **Pre-training**")
    image("images/olmo2-pretraining.png", width=600)
    text("2. **Mid-training**")
    image("images/olmo2-dolmino.png", width=600)
    text("3. **Post-training** "), link("https://arxiv.org/pdf/2411.15124")
    image("images/tulu.png", width=600)

    text("What are these datasets?  How are they chosen and processed?")


# 区分 Internet（网络基础设施）与 Web（HTTP 等提供的资源）；模型见到的是采样快照。
# 爬虫维护待访问队列与已访问集合，种子和链接图决定覆盖偏差，不是均匀抽样互联网。
# 例如搜索到商品页只能抓到初始 HTML，登录后评论或 JavaScript 加载内容未必在其中。
# robots.txt 是爬虫协作协议，不是作品授权书；访问成功也不能证明拥有训练许可。
# 验证码、登录、速率限制、ToS、版权分别约束技术可达性、服务使用和作品复制使用。
# 原稿影子图书馆的数量与争议是历史背景；可下载、免费标价和公有领域不是同义词。
# “未抓到”可能是抽样遗漏、动态页面或限制，不意味着内容在世界上不存在。
# 工程上应记录 URL、抓取时间、内容哈希、来源版本与许可依据，才能后续追踪或移除。
def raw_sources():
    text("One might often hear: *language models are trained on the entire Internet*.")
    text("Slightly more accurately, ~Internet~ public (world wide) web.")
    text("But this is not quite right either...")

    text("First, the web consists of a set of live servers that one can connect to:")
    text("`$ curl https://cs336.stanford.edu/`")

    text("You can't train on live servers.")
    text("A **crawler**:")
    text("- Discovers webpages (starting from a seed set)")
    text("- Downloads the discovered webpages")

    text("However, you can't download and train on all the webpages.")

    text("Dynamic content:")
    text("- Many sites these days are apps")
    text("- URL doesn't change")
    text("- Need to click buttons and submit forms to access content")
    text("- Examples: Discord, wandb")

    text("Authentication:")
    text("- Sometimes need login with an account (and pay usually)")
    text("- Example: Facebook, X, LinkedIn, NYTimes (huge content behind walled gardens)")

    text("Technical restrictions:")
    text("- Not allowed to download some content based on `robots.txt` ([example](https://www.nytimes.com/robots.txt)) (voluntary)")
    text("- Website might use Cloudflare to detect and block bot activity (present CAPTCHAs)")
    text("- Website might block certain IP addresses / countries")
    text("- Website might have rate limits")
    
    text("Legal restrictions:")
    text("- Terms of service (ToS) might prohibit downloading using bots")
    text("- You might not have a license to copy the webpages (for training)")

    text("Decline of consent "), link("https://arxiv.org/abs/2407.14933")
    text("- Examined restrictions (robots.txt, ToS) for URLs in common datasets (C4, RefinedWeb, Dolma)")
    text("- Restrictions have increased over time")
    image("images/decline-consent.png", width=700)

    text("When crawlers are not well-behaved:")
    image("images/anthropic-crawling.png", width=500)
    text("- Factors: ToS, robots.txt, server load (degrades service, costs website money)")
    text("- And then there is copyright (more later)...")

    text("Shadow libraries "), article_link("https://en.wikipedia.org/wiki/Shadow_library")
    text("- Technically part of the web")
    text("- Examples: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub")
    text("- Disregards copyright and bypasses paywalls (e.g., Elsevier)")
    text("- Received takedown orders, lawsuits, blocked in various countries")
    text("- Usually controls are circumvented, have servers in various countries")
    text("- Some argue this makes freely available what should be free")
    text("- From a legal perspective, this is piracy and copyright infringement")
    text("- LibGen has ~4M books (2019), Sci-Hub has ~88M papers (2022)")

    text("Summary:")
    text("- The Internet is huge")
    text("- Many technical and legal restrictions on what data one can access")


# 阅读边界：本节以美国框架解释概念；原稿保留课堂表述，不应直接当作个案法律意见。
# 版权针对原创表达；事实/思想与具体代码/文字表达要分开，集合的原创选择编排也可受保护。
# 原稿“75 years”是过度简化：保护期随作品年代、作者与雇佣创作等条件变化，不能统一套用。
# “登记后才能起诉”也有适用范围，不能概括所有外国作品；费用与案件状态均属时间敏感信息。
# 授权链至少分三层：作品权利人授权、数据集合/数据库许可、提供服务的合同条款。
# CC 并非单一许可证：BY、SA、NC、ND 等限制不同；写着 CC 不能直接推断任意商业训练可用。
# 合理使用四因素是综合判断，不是四项打分达到阈值；教育目的也不自动免除全部责任。
# 原稿把重实现思想列入例子，需先问是否复制受保护表达；未复制表达可能无需诉诸合理使用。
# 复制是权利分析的起点；许可、公有领域、法定例外与合理使用可能影响是否构成侵权，
# 因此“复制即违法”和“训练必然合理使用”都过强。获取副本与训练用途可以被分别评价。
# 下方案例只作原稿历史叙述；具体判决范围、证据、和解与终局裁判不可互相替代。
# 自测：数据集的开放许可证能洗净来源作品权利吗？不能，发布者不能授予自己没有的权利。
def copyright():
    text("What data is legal to use (for training)?")

    text("### Intellectual property law")
    text("- Goal: *incentivize* the creation of intellectual goods")
    text("- Types of intellectual property: copyright, patents, trademarks, trade secrets.")

    text("**Copyright law**:")
    text("- Goes back to 1709 in England (Statute of Anne), first time regulated by governments and courts "), article_link("https://en.wikipedia.org/wiki/Statute_of_Anne")
    text("- In United States, most recent: Copyright Act of 1976 "), article_link("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")
    text("- Copyright protection applies to *'original works of authorship fixed in any tangible medium of expression, now known or later developed, from which they can be perceived, reproduced, or otherwise communicated, either directly or with the aid of a machine or device'*")

    text("- Collections are not original works so hence not copyrightable (e.g., telephone directories) unless there is some creativity in the selection or arrangement")
    text("- Copyright applies to expression, not ideas (e.g., quicksort)")

    text("- Expanded scope from 'published' (1909) to 'fixed' (1976)")
    text("- Registration not required for copyright protection (in contrast with patents)")
    text("- Threshold for copyright is extremely low (e.g., your website is copyrighted)")

    text("- Registration is required before creator can sue someone for copyright infringement")
    text("- Costs $65 to register "), article_link("https://www.copyright.gov/about/fees.html")
    text("- Lasts for 75 years, and then the copyright expires and it becomes part of the public domain (works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)")

    text("Summary: *basically everything on the Internet are copyrighted.*")

    text("How to use a copyrighted work:")
    text("1. Get a license for it.")
    text("2. Appeal to the fair use clause.")

    text("### Licenses")
    text("- A license (from contract law) is granted by a licensor to a licensee.")
    text("- Effectively, 'a license is a promise not to sue'.")

    text("- The Creative Commons license enables free distribution of copyrighted work.")
    text("- Examples: Wikipedia, Open Courseware, Khan Academy, Free Music Archive, 307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.")
    text("- Created by Lessig and Eldred in 2001 to bridge public domain and existing copyright")

    text("Many model developers license data for training foundation models")
    text("- Google and Reddit "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- OpenAI and Shutterstock "), article_link("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    text("- OpenAI and StackExchange "), article_link("https://stackoverflow.co/company/press/archive/openai-partnership")

    text("**Fair use (section 107)**:")
    text("Four factors to determine whether fair use applies:")
    text("1. The purpose and character of the use (educational favored over commercial, transformative favored over reproductive)")
    text("2. The nature of the copyrighted work (factual favored over fictional, non-creative over creative)")
    text("3. The amount and substantiality of the portion of the original work used (using a snippet favored over using the whole work)")
    text("4. The effect of the use upon the market (or potential market) for the original work")

    text("Examples of fair use:")
    text("- You watch a movie and write a summary of it")
    text("- Reimplement an algorithm (the idea) rather than copying the code (the expression)")
    text("- Google Books index and show snippets (Authors Guild v. Google 2002-2013)")

    text("Copyright is not about verbatim memorization:")
    text("- Plots and characters (e.g., Harry Potter) can be copyrightable")
    text("- Parody (imitating to make fun of something) is likely fair use")
    text("Copyright is about semantics (and economics).")

    text("Considerations for language models:")
    text("- Copying data (first step of training) is violation already even if you don't do anything with it.")
    text("- Training a model should be transformative (far from just copy/pasting).")
    text("- Model should be about the general idea (e.g., wizards), not in the concrete expression (e.g., Harry Potter).")
    text("- Language models can definitely affect the market (writers, artists), regardless of copyright")

    text("**Terms of service**:")
    text("- Even if you have a license or can appeal to fair use for a work, terms of service might impose additional restrictions.")
    text("- Example: YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons.")

    text("### Lawsuits")
    text("The New York Times v. OpenAI (2023)")
    text("- Allegation: for training and reproducing NYT articles")

    text("Authors (Bartz, Graeber, ...) v. Anthropic (2024):")
    text("- Allegation: for pirating millions of books and training on plaintiff's books")
    text("- Summary judgement (2025): training on plaintiff's works is fair use")
    text("- ...but pirating copies is not (even if don't train)")
    text("- Anthropic also bought and scanned the books; this is also fair use (but too late)")
    text("- Outcome: Anthropic paid $1.5B to authors to settle")

    text("Authors (Kadrey, Silverman, ...) v. Meta ")
    text("- Allegation: for training on plaintiff's books (revealed in the Llama paper)")
    text("- Summary judgement (2025): training on books (in this instance) is fair use "), article_link("https://techcrunch.com/2025/06/25/federal-judge-sides-with-meta-in-lawsuit-over-training-ai-models-on-copyrighted-books/")
    text("- Allegation of torrenting books is still pending")

    text("Summary:")
    text("- So far training has been deemed fair use (for specific instances, but unclear in general)")
    text("- Pirating books is clearly illegal")
    text("- Still a very active, evolving area")


# 把 Common Crawl 当作上游快照仓库，而非可直接无差别喂给模型的干净训练集。
# 队列爬取图示：取 URL → 请求 → 存响应 → 解析链接 → 加入队列；实际还需域名限速、
# 规范化和重访策略。同一页可有查询参数、镜像和多次快照，所以 URL 不等于独立文档。
# 课堂核对（字幕 00:35:00–00:36:06）：WARC 到 WET 有损，抽取器会影响下游效果。
# WARC 是归档记录容器，常含 HTTP 响应/HTML，也有其他记录；不是只有裸 HTML。
# WET 保存抽取文本；补充还有 WAT（派生元数据），所以原稿“两种格式”只列本节重点。
# 示例：HTML 中导航“首页 新闻”和正文“水在标准大气压下约100°C沸腾”；好抽取器去导航留正文，
# 差抽取器可能保留菜单并丢失表格行列。文本一旦丢失结构，不能靠 WET 完整还原原网页。
# 处理链应画成 WARC → HTML解析/正文抽取 → 文本 → 语言识别/过滤/去重 → tokenization。
# 下方 DCLM 图比较的是特定设置的抽取效果，不证明某工具在所有网页上恒优。
# 自测：只拿 WET 能重新比较所有 HTML 抽取器吗？不能，需要保留相应 WARC/HTML。
def common_crawl():
    text("[Common Crawl](https://commoncrawl.org/) is a non-profit organization founded in 2007.")

    text("Statistics:")
    text("- Every ~month, run a web crawl (add 3-5 billion web pages)")
    text("- Crawls have some overlap but try to diversify")
    text("- 300 billion pages so far")

    text("- How many URLs are there? Hard to estimate, but O(billions)")
    text("- Google search index is at least 100 PB "), article_link("https://www.google.com/search/howsearchworks/how-search-works/organizing-information/")
    text("- [April 2026 Crawl](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available) has 2.19 billion pages (372.2 TB)")

    text("Crawling uses Apache Nutch "), article_link("https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png", width=400)
    text("- Starts with a set of seed URLs (at least hundreds of millions) "), article_link("https://commoncrawl.org/blog/march-2018-crawl-archive-now-available")
    text("- Pop a URL from the queue, download URL, and add hyperlinks to queue")

    text("Policies "), article_link("https://en.wikipedia.org/wiki/Web_crawler")
    text("- Selection policy: which pages to download?")
    text("- Politeness policy: respect robots.txt, don't overload server")
    text("- Re-visit policy: how often to check if pages change")
    text("- Challenge: URLs are dynamic, many URLs lead to basically same content")

    text("Two formats:")
    text("- WARC: raw HTTP response (e.g., HTML)")
    text("- WET: converted to text (lossy process)")

    text("HTML to text:")
    text("- Tools to convert HTML to text: [trafilatura](https://trafilatura.readthedocs.io/en/latest/), [resiliparse](https://resiliparse.chatnoir.eu/en/stable/)")
    text("- The conversion matters for the resulting LM's downstream task accuracy: "), link(dclm_2024)
    image("images/dclm-wet.png", width=300)


# 百科内容有可引用性、关注度与协作审校机制，因而是有用质量代理，但不是所有知识的全集。
# 倾向书面、可公开验证内容，口语、地方知识、新事件与低资源语言可能覆盖不足。
# 应使用带版本日期的 dump；抓取在线页与训练时看到的修订版本不是一回事。
# 投毒例子说明“来源声誉高”不保证每次快照正确：攻击者可利用修改与回退之间的时间窗。
# 因此来源可信度、快照完整性、异常检测和下游污染评估要分别做，不能只维护域名白名单。
def wikipedia():
    text("Let's now look at more specialized sources.")

    text("[Wikipedia](https://www.wikipedia.org/): free online encyclopedia")
    text("- [Random article](https://en.wikipedia.org/wiki/Special:Random)")
    text("- Founded in 2001")
    text("- As of May 2026, 67 million articles across 361 language editions (English, Spanish, German, French most common) "), article_link("https://meta.wikimedia.org/wiki/Wikipedia")

    text("What is the scope?")
    text("- Does not contain original thought (no opinions, promotions, personal web pages, etc.) "), article_link("https://en.wikipedia.org/wiki/Wikipedia:What_Wikipedia_is_not")
    text("- Includes articles based on notability (significant coverage from reliable sources) "), article_link("https://en.wikipedia.org/wiki/Wikipedia:Notability")

    text("Who writes the content?")
    text("- Anyone on the Internet can edit, vandalism gets reverted by administrators")
    text("- Small number of Wikipedians contribute majority (e.g., Steven Pruit with 5M edits) "), article_link("https://en.wikipedia.org/wiki/Steven_Pruitt")
    text("- Produce [periodic dumps](https://dumps.wikimedia.org/enwiki/) every few weeks (no need to crawl)")

    text("Aside: data poisoning attacks "), link("https://arxiv.org/pdf/2302.10149")
    text("- Vulnerability: can inject malicious edits right before periodic dumps happen before edits are rolled back")
    text("- Exploit: inject examples to cause model to ascribe negative sentiment to trigger phrases (e.g., iPhone) "), link("https://arxiv.org/pdf/2010.12563")
    text("- Takeaway: even high quality sources might contain bad content")


# 仓库快照包含文件，提交图包含演变历史，issue/PR 包含对话；三者需不同抽取与序列化。
# fork、依赖复制、生成代码、多个提交反复出现同文件，会使字节数严重高估独立信息量。
# 公开仓库没有许可证并不等于公有领域；宽松许可证也可能要求保留声明、署名等。
# 原稿“permissive 即 allowed”是简写，仍要核查文件级例外、依赖、秘密和个人信息。
# 原稿 GitHub Archive 的链接误指 arXiv S3；应与 github.com 的 git 数据和事件流区分。
# Software Heritage 保存源代码历史；其“source files”规模行疑有单位/量级问题，勿作为精确统计引用。
# 代码帮助推理在此是经验线索；要证明因果，需要控制模型、训练 token 数与其他数据再消融。
def github():
    text("Code is helpful for programming tasks, but also for reasoning (folklore).")

    text("[GitHub](https://github.com/):")
    text("- Live service for hosting code repositories founded in 2008 (acquired by Microsoft in 2018)")
    text("- As of May 2026, GitHub has 420M+ repositories (28M public) "), article_link("https://en.wikipedia.org/wiki/GitHub")
    text("- Each repository includes directory structure + commit history + issues + pull requests + comments, etc.")
    text("- Lots of duplicates (e.g., copied code, forks, etc.)")
    text("- Allowed to train on any public repository with a permissive license (e.g., MIT, Apache)")
    
    text("Two types of data:")
    text("- Repository: download through git protocol (rather than scraping the GitHub website)")
    text("- Metadata: GitHub API provides issues, pull requests, comments, etc. (hourly snapshots of event stream on [GitHub Archive](https://info.arxiv.org/help/bulk_data_s3.html))")

    text("[Software Heritage](https://www.softwareheritage.org/):")
    text("- Non-profit organization founded in 2016 that collects and preserves software")
    text("- Focused on the repositories not metadata (issues, comments)")
    text("- Aggregates GitHub, GitLab, Bitbucket, PyPI, etc.")
    text("- As of May 2026, there are 28.8M source files")


# 论文可有 PDF、LaTeX 源码、版本与元数据；选择输入表示本身就是数据加工决策。
# PDF抽取易打乱双栏与公式；LaTeX需去宏定义、注释和参考文献，但过度清洗会损伤数学结构。
# 站点可免费阅读、作者允许 arXiv 分发和作品允许第三方再利用是不同层次。
# 官方许可页列出多种许可；原稿“all rights reserved”是简化，默认分发许可不等于开放再利用。
# CC0 元数据政策也不意味着整篇论文正文都是 CC0。须保存逐项许可而非只看域名。
# 预印本不等于同行评审结论；技术语料的专业性与真实性仍需分别评价。
def arxiv():
    text("[arXiv](https://arxiv.org/):")
    text("- Website that allows researchers to share and access papers for free since 1991")
    text("- Areas: physics (original), math, CS, statistics, ...")
    text("- Has ~3M submissions "), article_link("https://arxiv.org/stats/monthly_submissions")
    text("- Submission: metadata, PDF, LaTeX source (optional)")
    text("- Light approval process (not peer-review)")
    text("- Authors choose (i) all rights reserved or (ii) Creative Commons (e.g., CC-BY)")
    text("- Metadata (title, abstract) is under a permissive license (CC0)")
    text("- Bulk download from [Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html), no need to crawl")


# BERT 使用双向上下文的掩码语言建模；本讲关注语料，不应把其目标混同于 GPT 的自回归目标。
# 连续文档使跨句指代、主题与篇章关系可学习；随机打散句子会保留局部语法却破坏长程结构。
# 例如“甲把钥匙交给乙。他随后离开。”需要上下句判断代词与事件，孤立第二句信息不足。
# BooksCorpus 与 Wikipedia 的组合补充长篇叙事和百科，但不代表覆盖所有写作风格。
def bert():
    link("https://arxiv.org/pdf/1810.04805")

    text("The BERT training data consists of:")
    text("- Wikipedia")
    text("- Books")
    books_corpus()

    text("- Important: sequences are documents rather than sentences")
    text("- Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)")


# 零售价只说明消费者价格，不自动授权大规模复制或训练。BooksCorpus 是从平台获取并加工的
# 特定语料，不等同于 Smashwords 整个平台；原版、后续重建版和下载状态需分别标明。
# 比较 word 与 token 数时必须固定分词器；书籍重复、同作者系列和章节切分也会影响有效规模。
def books_corpus():
    text("[Smashwords](https://www.smashwords.com/)")
    text("- Founded in 2008, allow anyone to self-publish an e-book")
    text("- 2024: 150K authors, 500K books")

    text("BooksCorpus "), link("https://arxiv.org/abs/1506.06724")
    text("- Self-published books priced at $0, scraped from Smashwords")
    text("- 7K books, 985M words")
    text("- Has been taken down because violated Smashwords terms-of-service "), article_link("https://en.wikipedia.org/wiki/BookCorpus")


# WebText 把 Reddit 用户投票当作间接质量信号：筛选的是被链接页面，不是主要训练 Reddit 评论。
# ≥3 karma 的门槛继承了 Reddit 用户群、话题和时间的偏好，受欢迎并不等于真实或全面。
# OpenWebText 是开放复现尝试，不是 GPT-2 私有语料的逐字节公开版本；抓取时间、失效链接、
# 语言识别和近重复规则会改变最终集合。复现算法不代表复现数据或模型效果。
def gpt2_webtext():
    text("WebText: dataset used to train GPT-2 "), link(gpt2_2019)
    text("- Contains pages that are outgoing links from Reddit posts with ≥ 3 karma (surrogate for quality)")
    text("- 8 million pages, 40GB text")

    text("OpenWebTextCorpus: open replication of WebText "), link(openwebtext_2019)
    text("- Extracted all the URLs from the Reddit submissions dataset")
    text("- Used Facebook's fastText classifier to filter out non-English")
    text("- Removed near duplicates")


# 三层操作各自回答不同问题：去重问“见过吗”，语言识别问“哪种语言”，质量评分问“像什么”。
# 课堂核对（字幕 00:48:03–00:49:22）：以百科训练的模型评分，质量代理不等于真值。
# KenLM 是 n-gram 语言模型；5-gram 用前至多4个词预测下一个词，不是五层神经网络。
# 编者推导：文档平均负对数似然 H=-(1/n)Σ log p(w_i|w_{i-4:i})，困惑度 PPL=exp(H)。
# 若四个词条件概率均为1/2，则 H=log2、PPL=2；均为1/4时 PPL=4，模型觉得更意外。
# 用 Wikipedia 训练的模型偏好百科式语言；较低困惑度是域相似度代理，不保证事实正确或无毒。
# 低资源语言的识别误差可能删除本来稀缺的数据；须按语言检查保留率而非只看总体均值。
def ccnet():
    text("CCNet "), link("https://arxiv.org/pdf/1911.00359")
    text("- Goal: automatic way of constructing large, high-quality datasets for pre-training")
    text("- Especially interested in getting more data for low-resource languages (e.g., Urdu)")

    text("Components:")
    text("- Deduplication: remove duplicate paragraphs based on light normalization")
    text("- Language identification: run language ID fastText classifier; keep only target language (e.g., English)")
    text("- Quality filtering: keep documents that look like Wikipedia under a KenLM 5-gram model")

    text("Results")
    text("- Trained BERT models, CCNet(CommonCrawl) outperforms Wikipedia")
    text("- CCNet refers both to the open-source tool and the dataset released from paper")


# C4 是 Common Crawl 的加工子集，T5 是用它训练的模型/框架；不能把上游和产物当作独立来源相加。
# 规则组合隐含目标分布：句末标点偏爱规范散文，“{”过滤会误删代码、数学或某些合法内容。
# 坏词列表可能误伤身份讨论、医疗教育和方言；语言检测分数也不是经过保证的真实后验概率。
# 编者手算：从1.4T到156B的账面保留比约11.1%，但只有 tokenizer/计数口径一致才可严格解释。
# 下方域名分布图用于观察哪些站点占比高；占比是采样结果，不能推出这些站点质量一定更高。
# WebText-like 的成绩来自特定基准；其17GB与40GB差异提示覆盖不全，也可能受抽取/时间影响。
# 自测：删除“坏词”的语料会消除毒性吗？不会，既有漏检，也可能删掉反歧视讨论。
def t5_c4():
    text("Colossal Clean Crawled corpus (C4) "), link("https://arxiv.org/pdf/1910.10683v4")

    text("Paper is more famous for Text-to-text Transfer Transformer (T5), which pushes the idea of putting all NLP tasks into one format")
    text("...but a major contribution was the C4 dataset.")

    text("Observation: Common Crawl is mostly not useful natural language")

    text("Started with one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)")

    text("Manual heuristics:")
    text("- Keep lines that end in punctuation and have >= 5 words")
    text("- Remove page with fewer than 3 sentences")
    text("- Removed page that contains any 'bad words' "), article_link("https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en")
    text("- Removed page containing '{' (no code), 'lorem ipsum', 'terms of use', etc.")
    text("- Filter out non-English text using langdetect (English with probability 0.99)")

    text("End result: 806 GB of text (156 billion tokens)")

    text("Analysis of C4 "), link("https://arxiv.org/pdf/2104.08758")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png", width=700)

    text("Bonus: WebText-like dataset")
    text("- Filtered to pages from OpenWebText links (links in Reddit posts with ≥ 3 karma)")
    text("- Used 12 dumps to get 17 GB text (WebText was 40 GB, suggesting CommonCrawl is incomplete)")
    text("- This improved on various NLP benchmarks (GLUE, SQuAD, etc.)")


# 质量分类器的正类来自已选语料，学习的是与该选择相似的特征，而非抽象的“真质量”。
# 负类也可能含好文档；可把分类器看作将网页分布推向目标分布的工具。
# 训练/测试重叠清理应跨来源检查：Wikipedia 中的材料也可能在网页镜像中再次出现。
# Books1/Books2 的具体来源未公开；不能由“Internet books”推断就是某个公开书库。
# 语料池大小与训练采样量分开记账：一个来源可被重复多轮采样，发布 token 数不等于模型见过数。
def gpt3():
    text("GPT-3 dataset "), link("https://arxiv.org/pdf/2005.14165")  # Section 2.2
    text("- Common Crawl (processed)")
    text("- WebText2 (WebText expanded with more links)")
    text("- (Mysterious) Internet-based books corpora (Books1, Books2)")
    text("- Wikipedia")

    text("Result: 570 GB (400 billion tokens)")

    text("Common Crawl processing:")
    text("- Trained quality classifier to distinguish {WebText, Wikipedia, Books1, Books2} from rest")
    text("- Fuzzy deduplication of documents (including WebText and benchmarks)")


# 22个组件体现领域覆盖思路：论文、代码、问答、网页和书籍给模型不同语言与任务结构。
# 下方组成图同时看组件名称和占比；混合权重可以不同于原始字节占比，不能只按面积推训练权重。
# Pile-CC 从 WARC 自行抽取，说明即便上游相同也能得到不同训练文本。
# 公开邮件不意味着没有个人信息；论文公开可读不意味着每篇都允许任意复制。
# 跨组件重叠需要联合去重，如网页转载论文、问答被博客转载；逐组件去重不足以消除重复。
# 825GB与275B的近似换算不是“每token恒等3字节”的定律，语言/代码/分词器都会改变比例。
def the_pile():
    text("The Pile "), link("https://arxiv.org/pdf/2101.00027")

    text("- In reaction to GPT-3, part of effort to produce open-source language models")
    text("- Grassroots effort with lots of volunteers contributing/coordinating on Discord")
    text("- Curated 22 high-quality domains")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)

    text("- 825 GB of text (~275B tokens)")
    text("- Pile-CC: Common Crawl, use WARC, jusText to convert into text (better than WET)")
    text("- PubMed Central: 5 million papers, mandated to be public for NIH funded work")
    text("- arXiv: preprint for research papers since 1991 (use latex)")
    text("- Enron emails: 500K emails from 150 users from Enron senior management, released during Enron investigation (2002) "), article_link("https://www.cs.cmu.edu/~enron/")

    project_gutenberg()
    books3()
    stackexchange()


# 公有领域书籍可提供长篇连贯叙事，年代与文体偏差也明显，不能独自覆盖现代知识。
# 版权清理有司法辖区范围，不能把一地公有领域结论自动推广全球；版本中的序言/插图也可有独立权利。
# PG-19 是选定集合/切分方案；评估长上下文时应按书划分，避免同书章节跨训练测试泄漏。
def project_gutenberg():
    text("[Project Gutenberg](https://www.gutenberg.org/)")
    text("- Started in 1971 by Michael Hart, who wanted to increase access to literature")
    text("- 2025: ~75K books, mostly English")
    text("- Only include books that have received copyright clearance (most in the public domain)")

    text("PG-19: books from Project Gutenberg before 2019 "), article_link("https://github.com/google-deepmind/pg19")


# Books3 是具体加工语料，与 BooksCorpus、GPT-3 Books1/2 不能仅凭名字认作同一集合。
# 这里的获取与版权争议揭示来源审计必要性；出现于某语料说明文档不证明任一模型完整使用过它。
# 下架状态也不抹去既有训练或镜像；研究记录需保存版本和来源链而非只给易失下载链接。
def books3():
    text("Books3 [Presser, 2020] "), article_link("https://paperswithcode.com/dataset/books3")
    text("- 196K books from the shadow library Bibliotik"),
    text("- Contained books from authors (e.g., Stephen King, Min Jin Lee, Zadie Smith) "), article_link("https://www.wired.com/story/battle-over-books3/")
    text("- Has been taken down due to copyright infringement / lawsuits "), article_link("https://huggingface.co/datasets/the_pile_books3")



# 问题、多个答案、投票、评论组成结构化对象，转文本时须保留角色与问答关联。
# 例如先写问题，再按分数写答案，能增加好答案可见性；分数仍可能受年代/曝光量影响。
# 预训练中看到 Q&A 不等于已经完成指令微调：训练目标、角色格式和样本权重可能不同。
# 用户标识替换不保证匿名，正文仍可含姓名、邮箱或可关联细节；许可证条件也需单独检查。
def stackexchange():
    text("- Collection of sites of user-contributed questions and answers")
    text("- Started with StackOverflow in 2008, grew to other topics (e.g., math, literature) "), link(title="sites", url="https://stackexchange.com/sites")
    text("- Use reputation points and badges to incentivize participation")
    text("- [Example](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)")

    text("- Q&A format is close to instruction tuning / real application")
    text("- Note: there is metadata (users, votes, comments, badges, tags) for filtering")
    text("- Data dumps in XML (anonymized, include metadata) "), link(title="link", url="https://archive.org/details/stackexchange")



# MassiveText 是总池，MassiveWeb 是其中网页组件，Gopher 是训练出来的模型。
# “80%词含字母”是针对噪声的启发式，可能误伤数学表格或非拉丁文字；不能脱离语言套用。
# SafeSearch 分类信号也会出错，不等于完美的毒性判定器；不同过滤器定义的“毒性”可能不同。
# 10.5TB、300B token 与12%涉及不同量纲；不能直接用300B除10.5TB推出12%。
# 原稿比例应理解为论文口径下的训练消耗说明；查精确使用量还需 tokenizer 和混合采样表。
def gopher_massivetext():
    text("MassiveText dataset used to train Gopher "), link(gopher_2021)
    text("The Gopher model is subsumed by Chinchilla (also never released), but the description of data is good")

    text("Components")
    text("- MassiveWeb: more on this later")
    text("- C4")
    text("- Books: no details")
    text("- News: no details")
    text("- GitHub: no details")
    text("- Wikipedia: no details")

    text("MassiveWeb filtering steps")
    text("- Keep English, deduplication, train-test overlap")
    text("- Quality filtering using manual rules (not classifier) - e.g., 80% words contain at least one alphabetic character")
    text("- Use Google SafeSearch for toxicity (not word lists)")

    text("Result: 10.5 TB of text (though Gopher only trained on 300B tokens - 12%)")


# 这是首代 LLaMA 的配方，不是后续 Llama 3 的训练数据披露。
# Wikipedia references 指百科引用指向的网页，不是把参考文献标题本身当成全部训练材料；
# 用这类页面训练分类器仍在引入百科编辑群体的选择偏好。
# 代码保留许可证，LaTeX 展开宏，问答按分数排序，说明同一清洗规则无法适用全部模态。
# 原稿1.2T等数值需结合模型尺寸/训练设置看，不能概括所有 LLaMA 模型消耗都相同。
# RedPajama 是配方复现，SlimPajama 是进一步清理子集，三者不可当独立无重叠来源混加。
def llama():
    text("Dataset for LLaMA "), link("https://arxiv.org/pdf/2302.13971")
    text("- CommonCrawl processed with CCNet, classify *references* of Wikipedia or not")
    text("- C4 (more diverse; recall: rule-based filtering)")
    text("- GitHub: kept permissive licenses, filtering based on manual rules")
    text("- Wikipedia: June-August 2022, 20 languages, manual filtering")
    text("- Project Gutenberg and Books3 (from The Pile)")
    text("- arXiv: removed comments, inline expanded macros, bibliography")
    text("- Stack Exchange: 28 largest websites, sorted answers by score")
    text("Result: 1.2T tokens")

    text("Reproduced by Together's RedPajama v1 "), link("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")
    text("Cerebras's [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama): 627B subset of RedPajama v1 by deduplication (MinHashLSH)")


# “web data is all you need”应读为特定模型/预算/评测下的实验证据，不是网页永远足够的定理。
# 先从 WARC 抽正文，再规则过滤，再用5-gram集合估计近重复；顺序会改变结果。
# 编者例子：A={a,b,c,d}, B={b,c,d,e}，Jaccard=|交集|/|并集|=3/5=0.6。
# MinHash 用短签名近似这一相似度，LSH找候选；哈希碰撞、阈值和粒度会带来误判/漏判。
# 不用机器学习过滤器也不会消除偏差：标点、词数、去重阈值都在选择分布。
# FineWeb 扩大快照覆盖并改进规则；邮箱/IP替换只是部分PII处理，不能保证没有姓名和隐私叙事。
# 600B发布子集、5T总池和15T另一语料的规模要分别记录，比较模型还需对齐训练预算。
def refinedweb():
    text("RefinedWeb "), link("https://arxiv.org/pdf/2306.01116") 
    text("- Point: web data is all you need")
    text("- [Examples](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("- trafilatura for HTML→text, extract content (WARC instead of WET files)")
    text("- Filtering: Gopher rules, avoid ML-based filtering to avoid biases")
    text("- Fuzzy deduplication using MinHash over 5-grams")
    text("Released 600B (out of 5T) tokens")

    text("FineWeb "), article_link("https://huggingface.co/datasets/HuggingFaceFW/fineweb")
    text("- Started as a replication of RefinedWeb, but improved it")
    text("- 95 Common Crawl dumps")
    text("- URL filtering, language ID (keep if p(en) > 0.65)")
    text("- Filtering: Gopher, C4, more manual rules")
    text("- Fuzzy deduplication via MinHash")
    text("- Anonymize email and public IP addresses (PII)")
    text("Result: 15T tokens")


# Dolma 强调多来源与可追踪处理管线；下方组成图对应“覆盖哪些域”，不直接回答“最优混合是什么”。
# Reddit 帖子与评论可独立抽取，若丢掉线程关系会损失上下文；PeS2o是论文加工资源而非arXiv同义词。
# Bloom filter以位数组和多哈希回答“可能见过/肯定未见过”；标准插入查询有假阳性，无假阴性。
# 因此去重可能误删少量新文本；若系统还做分片、删除或近似归一化，端到端性质另需验证。
# Toxicity 分类与规则各有误差，分来源/分人群保留率审计有助于发现偏差。
# 开放发布整体集合不替代原始文档许可，后面CommonPile将进一步讨论这一层次差异。
def dolma():
    text("Dolma "), link("https://arxiv.org/pdf/2402.00159")
    image("https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png", width=700)

    text("- Reddit: from the Pushshift project (2005-2023), include submissions and comments separately")
    text("- PeS2o: 40M academic papers from Semantic Scholar")
    text("- C4, Project Gutenberg, Wikipedia/Wikibooks")

    text("Common Crawl processing")
    text("- Language identification (fastText classifier), keep English")
    text("- Quality filtering (Gopher, C4 rules), avoid model-based filtering")
    text("- Toxicity filtering using rules and Jigsaw classifier")
    text("- Deduplication using Bloom filters")

    text("Result: 3T tokens")

# DataComp-LM 是受控数据筛选研究框架；DCLM-pool是候选池，baseline是特定筛选结果。
# 比较过滤方法应控制模型架构、token预算、训练超参和评测，避免把更多计算误认作数据更好。
# 图中的正类 OpenHermes/ELI5 更接近说明、问答与指令风格；RefinedWeb负类不是全都低质量。
# fastText 分类器学习词/短语等轻量特征，利于海量推断；它不逐条验证事实或进行复杂推理。
# 保留分数最高部分会改变领域和写作风格分布；分类器胜出只针对报告的评估任务与预算。
# 编者手算：3.8T/240T≈1.58%，是账面池到产物比例；不能视为“网页98%毫无价值”。
# 自测：用验证集答案作质量正例再评估同题会怎样？有污染风险，必须隔离选择与最终评测。
def dclm():
    text("DataComp-LM "), link(dclm_2024)
    text("- Goal: define a standard dataset for trying out different data processing algorithms")
    text("- Processed CommonCrawl to produce DCLM-pool (240T tokens)")
    text("- DCLM-baseline: filtered down DCLM-pool using quality classifier")
    image("images/dclm-filter.png", width=800)

    text("### Model-based filtering")
    text("Positive examples (200K):")
    text("- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5): mostly GPT-4 generated instruction data ([examples](https://huggingface.co/datasets/teknium/OpenHermes-2.5/viewer/default/train))")
    text("- [ELI5](https://www.reddit.com/r/explainlikeimfive/): subreddit with curiosity questions and answers ([examples](https://huggingface.co/datasets/sentence-transformers/eli5/viewer/pair/train))")
    text("Negative examples (200K):")
    text("- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("Result: 3.8T tokens")

    text("Trained a fastText classifier, run it on all of DCLM-pool")
    text("This quality classifier outperforms other filtering methods:")
    image("images/dclm-quality.png", width=600)


# 这里的矛盾是固定质量门槛与所需数据量：更高门槛可能提高单token价值，却减少覆盖与总量。
# 多个分类器集成可融合不同偏好，但相关错误不会因投票自动消失；需检查不同域的选择结果。
# 用大模型教育分数训练小模型称蒸馏：小模型拟合教师输出，不是获得独立人工真值。
# 重写低质文档与从高质文档生成QA是两种合成路径；可能修复噪声，也可能编造事实或删除少见细节。
# 应保留原文/生成文关联、教师版本及提示词，避免把生成token误报成新增独立知识。
# 结果图的优势依赖报告设置；6.3T总集、1.1T高质子集与其他模型训练消耗并非同一指标。
def nemotron_cc():
    text("Nemotron-CC "), link(nemotron_cc_2024)
    text("- FineWebEdu and DCLM filter too aggressively (remove 90% of data)")
    text("- Need moar tokens (but preserve quality)")
    text("- For HTML→text, used jusText (not trafilatura) because it returned more tokens")

    text("Classifier ensembling")
    text("- Prompt Nemotron-340B-instruct to score FineWeb documents based on educational value, distill into faster model")
    text("- DCLM classifier")

    text("Synthetic data rephrasing")
    text("- For low-quality data, use LM to rephrase")
    text("- For high-quality data, use LM to generate tasks (QA pairs, extract key information, etc.)")

    text("Result: 6.3T tokens (HQ subset is 1.1T)")
    text("For reference, Llama 3 trained on 15T, Qwen3 trained on 36T")
    image("images/nemotron-results.png", width=800)


# 代码采集链：仓库标识 → 固定修订/文件 → 许可证检测 → 过滤/去重 → 可训练文本。
# 文件多于独立文件揭示 fork 与历史复制造成的冗余；许可证自动检测可能漏掉文件级例外。
# The Stack v2 与 StarCoder2 的完整训练配方要分开：原稿此节后半也介绍模型训练所用的
# 额外来源/处理，不能把每项额外数据都视为代码归档本体的组成。
# LLVM是低层中间表示；源码与IR配对提供跨语言结构桥梁，但编译器版本/优化会影响IR形态。
# PR图示要读成结构化对象到序列：标题/描述、文件路径、上下文、diff与角色边界需要明确分隔。
# 缺少上下文时“+return x”不说明修改意图；保留全部上下文又增加成本，所以需要抽样与截断。
# 应隔离评测代码/题解及其fork，防止代码模型因记忆测试答案获得虚高成绩。
def the_stack():
    text("The Stack "), link("https://arxiv.org/pdf/2211.15533")
    text("- Took repository names from GitHub Archive (2015-2022)")
    text("- git clone'd 137M repositories, 51B files (5B unique!)")
    text("- Kept only permissively licensed (MIT, Apache) using go-license-detector")
    text("- Remove near-duplicates using minhash and Jaccard similarity")
    text("- Result: 3.1 TB of code")

    text("Stack v2 "), link("https://arxiv.org/abs/2402.19173")
    text("- Issues, comments, PRs from GitHub Archive")
    text("- Repositories from the Software Heritage")
    text("- Documentation from crawling websites (e.g., PyPI, npm, devdocs.io)")
    text("- Processing: remove binary files, malware, bot activity, deduplication, PII redaction, subsample PRs")
    text("- Pair source code (especially low-resource languages like Nim) with shared low-level intermediate language (LLVM)")
    text("- Include existing datasets (GSM8K, code contests, StackOverflow, arXiv, Wikipedia, OpenWebMath)")

    text("Pull requests:")
    text("- Linearize structured object to token sequence")
    text("- Add some inline context (e.g., file surrounding diff), subsample")
    image("images/stackv2-pr1.png", width=250), image("images/stackv2-pr2.png", width=400)


# CommonPile（连写）与 The Pile 是不同工程；目标是在许可约束下构建有竞争力的预训练语料。
# 下方来源图关注授权来源构成，结果图关注模型表现；两者结合才说明约束下的能力和缺口。
# 许可清晰仍需核验授权者是否有权授权；license laundering 是将无权授权的作品重新贴上
# 开放许可证，掩盖底层权利缺口；它不只发生于集合标签。
# 课堂核对（字幕 01:16:35–01:18:04）：成员权利、许可洗白与合成数据是不同问题。
# 集合许可只约束集合层面的权利，不能替代每篇文章、图片或代码的许可条件。
# 合成数据还要区分生成服务条款、输入作品权利与输出内容，不能仅因“模型生成”就判定清白。
# 8TB是数据量，能力比较还需训练token、架构与评测；不要从单图推断性能差完全由许可导致。
# 整讲自测：同一网页可否同时出现在C4、The Pile、Dolma？可以，它们共享上游且可能互相纳入。
# 若把三者混合，先追踪版本和重复，再设采样权重；下一讲的过滤、去重和混合正解决这些问题。
# 综合练习（编者）：网页池1000B token，清洗留40%，去重再留75%，则独立池约300B。
# 若训练消耗600B，并非获得600B独立信息，而是平均约2次曝光（实际取决于不均匀采样）。
# 交付数据说明应包含来源/日期/授权证据、加工代码与参数、去重范围、tokenizer及训练采样量。
def common_pile():
    text("Recall:")
    text("- Almost all data on the Internet is copyrighted.")
    text("- Some of it is permissively licensed.")
    text("- Fair use of copyrighted content is not settled.")

    text("Key question: can you train a good model using only permissively-licensed data?")

    text("CommonPile "), link("https://arxiv.org/pdf/2506.05209")
    image("images/commonpile.png", width=700)
    text("- Collected 8TB dataset of permissively licensed data")

    text("Subtleties:")
    text("- License laundering: redistribute copyrighted work under permissive license (hard to detect)")
    text("- Collection licenses (Dolma is ODC-By) doesn't extend to individual")
    text("- Synthetic data from LMs trained on unlicensed data is unclear")

    image("images/comma-results.png", width=700)
    text("- Can do decently, but tough to compete without more tokens")


if __name__ == "__main__":
    main()
