# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's Language Modeling from Scratch (CS336).

## Executable lectures

These are named `lecture_XX.py`.

### Setup

        uv sync
        git clone https://github.com/percyliang/edtrace

You can compile a lecture by running:

        python execute.py -m lecture_01

which generates a `var/traces/lecture_01.json` and caches any images as
appropriate.

To view it locally:

Load a local server to view at `http://localhost:5173?trace=var/traces/sample.json`:

        npm run --prefix=edtrace/frontend dev

Deploy to the main website:

        npm run --prefix=edtrace/frontend build
        git add assets
        git ci -am "<some message>"
        git push

## Non-executable lectures

These are named `lecture_XX.pdf`.

## 中文自学增强版：第 3–17 讲

Python 讲义的中文解释直接放在原文件的相邻注释中，保留原有英文和代码。
PDF 讲义请打开下表中的增强版：每组完整原页之后紧接中文讲解，书签可跳转到原页、讲解和自测。
原始 `lecture_XX.pdf` 保持不变。第 1、2 讲由独立任务维护，不属于本次增强范围。

| 讲次 | 主题 | 阅读入口 |
| --- | --- | --- |
| 03 | 架构与超参数 | [原页与中文讲解](lecture_03_annotated.pdf) |
| 04 | 注意力替代方案与 MoE | [原页与中文讲解](lecture_04_annotated.pdf) |
| 05 | GPU、TPU 与高效计算 | [原页与中文讲解](lecture_05_annotated.pdf) |
| 06 | GPU kernel 与 Triton | [原文件及中文注释](lecture_06.py) |
| 07 | 分布式并行的代码实现 | [原文件及中文注释](lecture_07.py) |
| 08 | 并行策略与大规模训练 | [原页与中文讲解](lecture_08_annotated.pdf) |
| 09 | 缩放律基础 | [原页与中文讲解](lecture_09_annotated.pdf) |
| 10 | 推理效率与服务 | [原文件及中文注释](lecture_10.py) |
| 11 | 缩放实验与超参数迁移 | [原页与中文讲解](lecture_11_annotated.pdf) |
| 12 | 模型评估 | [原文件及中文注释](lecture_12.py) |
| 13 | 数据来源与数据集 | [原文件及中文注释](lecture_13.py) |
| 14 | 过滤、去重、混合与合成数据 | [原文件及中文注释](lecture_14.py) |
| 15 | Midtraining、SFT 与偏好优化 | [原页与中文讲解](lecture_15_annotated.pdf) |
| 16 | 可验证奖励强化学习 | [原页与中文讲解](lecture_16_annotated.pdf) |
| 17 | 多模态模型 | [原文件及中文注释](lecture_17.py) |

中文内容侧重概念动机、背景、公式与形状推导、手算例子、前后联系、适用边界和带答案自测。
课堂转述保留时间定位，编者推导与原论文结论分别标注。使用的第三方字幕缓存不是教师审校的官方逐字稿；
术语和公式以原课件、官方录像和原论文交叉核对。来源及原文件指纹见 [provenance.json](selfstudy/provenance.json)。

### 重建与检查

`selfstudy/lecture_XX.json` 是增强 PDF 的可编辑构建源，不是另一本替代原稿的中文讲义。
在仓库根目录建立一个独立环境后，可以重建任意一讲：

```bash
python3 -m venv /tmp/cs336-pdf-build
/tmp/cs336-pdf-build/bin/pip install -r lectures/selfstudy/requirements.txt
/tmp/cs336-pdf-build/bin/python lectures/selfstudy/build_pdf.py --root . --lecture 3
```

构建器保留原页的内容流及透明度组，检查原文件字节不变、原页顺序/尺寸/文字/渲染像素一致，并检查中文解释、自测及书目文字没有缺失。
每讲的 `selfstudy/lecture_XX.build.json` 记录原页与增强版页码的映射；`selfstudy/reviews/` 保留工作区审阅记录。
整体验收结果见 [validation_summary.json](selfstudy/validation_summary.json)，独立 worktree 会话及实际 subagent 任务见 [execution.json](selfstudy/execution.json)。

Python 注释的静态验收命令如下，比较基准是本次增强前的本地提交：

```bash
python3 lectures/selfstudy/validate_python.py --root . --lecture 6
```

这会检查语法、AST、可执行 token、原始行的保留顺序及 diff 格式。
这些检查不等同于运行完整的 GPU/分布式讲义，也不代表已经重新生成并测试 edtrace 网页。
