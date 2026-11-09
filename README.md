# 校园 FAQ 智能问答（NLP 课程项目）

面向大连东软信息学院校园事务的 FAQ 意图分类与语义检索系统。核心链路完全由课程方法实现，BERT 微调用于效果提升，并提供一个可选的智能补答接口（不配置 API Key 也能完整运行）。

## 任务与数据

- 10 类校园意图：选课与教务、考试与成绩、图书馆、宿舍与住宿、食堂与生活、校园网络与IT、请假与考勤、奖助学金与资助、校园卡与缴费、毕业与档案
- 60 条知识库条目，每条包含 1 个标准问题、1 个答案和 5 种同义问法；答案结合学校实际情况整理，时间、流程类问题以官方通知为准
- 共 360 条问法，按“问法”切分：240 条训练 / 120 条测试；检索库只使用 60 条标准问题

## 方法

**基线（课程方法）**

- 检索：TF-IDF（unigram + bigram、sublinear tf）；PPMI 共现矩阵 + randomized SVD 100 维；手写 PyTorch Skip-gram 词向量
- 句子表示：词向量平均 + L2 归一化 + 余弦相似度
- 意图分类：两层 MLP（100 → 128 → 64 → 10，dropout 0.3，早停）

**提升**

- `bert-base-chinese` 微调意图分类（20 epoch，lr 3e-5，max_len 64）
- BERT 最后一层 attention-mask 平均池化作为句向量，用于知识库检索
- 统一评测 Accuracy、macro-F1、Recall@1、Recall@3、MRR

## 实验结果

| 检索方法 | Recall@1 | Recall@3 | MRR | 意图级 Recall@1 |
|---|---|---|---|---|
| TF-IDF | 0.342 | 0.558 | 0.467 | 0.533 |
| PPMI+SVD | 0.175 | 0.383 | 0.322 | 0.375 |
| Skip-gram | 0.233 | 0.442 | 0.367 | 0.400 |
| BERT 句向量 | **0.508** | **0.700** | **0.635** | **0.808** |

| 意图分类 | Accuracy | Macro-F1 |
|---|---|---|
| 词向量平均 + MLP（PPMI+SVD） | 0.500 | 0.502 |
| 词向量平均 + MLP（Skip-gram） | 0.400 | 0.393 |
| BERT 微调 | **0.850** | **0.849** |

补充实验包括平均池化与 `[CLS]` 池化对比、检索分数与意图概率融合系数扫描、意图过滤检索（预测意图 vs oracle 上界）、置信度分桶，以及 59 条 Top-1 错误的归因分析。

## 界面

Gradio 5 演示界面：卡片式布局、自定义主题与 CSS，支持四种检索方式切换、Top-3 结果与相似度展示、两种课程 MLP 对比；智能补答默认关闭，仅在勾选且置信度低于阈值时可选调用外部 OpenAI 兼容接口。

## 项目结构

```
nlp_final/
├─ data/faq.json           # 知识库与同义问法
├─ src/
│  ├─ common.py            # 数据加载、切分、分词、评测指标
│  ├─ run_baseline.py      # TF-IDF / PPMI+SVD / Skip-gram + MLP
│  ├─ run_bert.py          # BERT 微调与句向量检索
│  ├─ extra_eval.py        # 池化对比、融合、意图过滤、置信度分桶
│  ├─ error_analysis.py    # 错误案例归因
│  ├─ make_figures.py      # 报告图表
│  ├─ demo.py              # Gradio 演示
│  ├─ llm_fallback.py      # 可选智能补答接口
│  └─ make_report.py       # 生成实验报告 docx
├─ outputs/
│  ├─ figures/             # 实验结果图
│  └─ results/             # 评测指标 JSON
└─ requirements.txt
```

`models/`（BERT 预训练与微调权重、词向量、MLP 检查点）体积约 1.1 GB，未纳入仓库；运行训练脚本会自动生成。

## 运行

```bash
pip install -r requirements.txt

python src/run_baseline.py    # 课程方法基线
python src/run_bert.py        # BERT 微调与检索
python src/extra_eval.py      # 补充分析
python src/error_analysis.py  # 错误分析
python src/make_figures.py    # 生成图表
python src/demo.py            # 启动演示（127.0.0.1:7860）
```

可选智能补答环境变量（不配置也能正常演示）：

```powershell
$env:LLM_API_KEY="你的密钥"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus"
```

## 技术栈

Python 3.12 · PyTorch · Transformers · scikit-learn · jieba · Gradio · numpy · matplotlib
