# -*- coding: utf-8 -*-
"""生成期末大作业报告 docx。"""
import os
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "report"
OUT.mkdir(parents=True, exist_ok=True)
DOCX = OUT / "24001360412-周俊毅-自然语言处理大作业报告.docx"

doc = Document()

# ---------------- 页面与默认样式 ----------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
sec.different_first_page_header_footer = True  # 封面不显示页码

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
rpr = normal.element.get_or_add_rPr()
rfonts = rpr.find(qn("w:rFonts"))
if rfonts is None:
    rfonts = OxmlElement("w:rFonts")
    rpr.insert(0, rfonts)
rfonts.set(qn("w:eastAsia"), "宋体")


def set_font(run, size=12, bold=False, east="宋体", color=None, mono=False):
    run.font.name = "Consolas" if mono else "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    rf.set(qn("w:eastAsia"), "Consolas" if mono else east)
    return run


def add_text(text, indent=True, size=12, bold=False, align=None, space_after=6,
             line=1.5, east="宋体"):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    set_font(p.add_run(text), size=size, bold=bold, east=east)
    return p


def add_heading(text, level=1):
    p = doc.add_heading("", level=level)
    pf = p.paragraph_format
    pf.space_before = Pt(18 if level == 1 else 10)
    pf.space_after = Pt(8 if level == 1 else 4)
    size = {1: 16, 2: 14, 3: 12}[level]
    set_font(p.add_run(text), size=size, bold=True, east="黑体", color=(0, 0, 0))
    return p


def add_caption(text, before=2, after=10, keep_next=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = keep_next
    set_font(p.add_run(text), size=10.5, east="宋体")
    return p


def shade(cell, color="D9E2F3"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_table(headers, rows, widths=None, font_size=10.5, caption=None, first_col_left=False):
    if caption:
        add_caption(caption, before=8, after=4, keep_next=True)
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = 1
    t.autofit = False
    # 表头跨页重复 + 单行不跨页
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ""
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(str(h)), size=font_size, bold=True)
        shade(c)
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ""
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (first_col_left and j == 0) else WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(str(val)), size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_figure(name, caption, width_cm=15.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    add_caption(caption, before=2, after=10)


def add_code(code, size=9):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    pf.left_indent = Cm(0.4)
    tcPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F2F2F2")
    tcPr.append(shd)
    run = p.add_run()
    set_font(run, size=size, mono=True)
    for i, line in enumerate(code.strip("\n").split("\n")):
        if i:
            run.add_break()
        run.add_text(line)


def page_break():
    """在最后一个段落末尾插入分页符，避免额外空段落产生空白页。"""
    doc.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)


def add_field(paragraph, code, placeholder=""):
    run = paragraph.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = code
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = placeholder
    fld3 = OxmlElement("w:fldChar")
    fld3.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    run._r.append(txt)
    run._r.append(fld3)


# ---------------- 封面 ----------------
for _ in range(4):
    doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("《自然语言处理》期末大作业报告"), size=20, bold=True, east="黑体")
doc.add_paragraph()
for _ in range(2):
    doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.line_spacing = 1.4
set_font(p.add_run("基于课程方法与 BERT 微调的"), size=24, bold=True, east="黑体")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("校园 FAQ 智能问答系统"), size=24, bold=True, east="黑体")
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("——面向大连东软信息学院的意图分类、语义检索与智能补答"), size=14, east="宋体")

for _ in range(5):
    doc.add_paragraph()
for label, value in [("学　　号", "24001360412"), ("姓　　名", "周俊毅"),
                     ("课　　程", "自然语言处理"), ("报告类型", "期末大作业")]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.5
    set_font(p.add_run(f"{label}：{value}"), size=14)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(24)
set_font(p.add_run("2026 年 11 月"), size=14)
page_break()

# ---------------- 摘要 ----------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("摘　要"), size=16, bold=True, east="黑体")
add_text(
    "针对校园咨询信息分散、人工回复成本高的问题，本文面向大连东软信息学院构建了一个包含 10 类意图、60 条知识库条目、"
    "360 个问法的校园 FAQ 数据集，并实现了一个以课程方法为基线、BERT 微调为核心的问答系统。"
    "系统包含意图分类与 FAQ 检索两个子任务：分别使用 TF-IDF、PPMI+SVD 词向量、Skip-gram 词向量"
    "以及词向量平均 + MLP 作为课程方法基线，使用 bert-base-chinese 微调进行意图分类，"
    "并用微调编码器的平均池化向量做语义检索。实验表明，BERT 意图分类在 120 条测试问法上取得 "
    "85.0% 的准确率和 84.9% 的宏平均 F1，显著优于 PPMI+SVD+MLP 的 50.0% 和 Skip-gram+MLP 的 40.0%；"
    "检索任务中 BERT 的意图级 Recall@1 为 80.8%，条目级 Recall@1/Recall@3 分别为 50.8%/70.0%，"
    "均优于三种课程方法基线。系统还实现了基于置信度阈值的可选智能补答：仅当本地模型置信度低于阈值时，"
    "才将检索到的知识库内容作为上下文交给大模型组织答案，保证系统不依赖大模型也能完成核心问答。"
    "最后使用 Gradio 搭建了可视化演示界面，单次问答在 RTX 4060 上约 40–50 ms。"
)
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
set_font(p.add_run("关键词："), size=12, bold=True)
set_font(p.add_run("FAQ 检索；意图分类；PPMI；Skip-gram；BERT 微调；Gradio"), size=12)
page_break()

# ---------------- 目录 ----------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("目　录"), size=16, bold=True, east="黑体")
p = doc.add_paragraph()
add_field(p, 'TOC \\o "1-2" \\h \\z \\u', "（目录将在打开文档时自动更新，若未更新请右键选择“更新域”）")
page_break()

# ---------------- 1 问题背景 ----------------
add_heading("1  问题背景与任务定义", 1)
add_text(
    "高校校园里，教务、考试、图书馆、宿舍、食堂、网络、请假、奖助学金、校园卡和毕业档案等事务"
    "每天都会产生大量重复咨询。这类问题通常有固定答案，但学生提问方式口语化、表达差异大，"
    "仅靠关键词搜索或人工客服既容易漏检，也难以覆盖长尾问法。FAQ 问答系统可以把“标准问题—答案”"
    "组织成知识库，先用意图分类缩小范围，再用语义检索找到最相关的条目，最后直接把答案返回给学生。"
)
add_text(
    "本文将任务形式化为两个子任务：（1）意图分类，给定用户问题 q，预测其所属的 10 类校园意图之一；"
    "（2）FAQ 检索，给定 q，在 60 条标准问答中找出与 q 语义最接近的条目，并返回其答案。"
    "两个任务共享同一个知识库和同一套测试问法，但评测目标不同：分类只看意图是否正确，"
    "检索还要求命中具体条目，因此后者更难。"
)
add_text(
    "课程要求项目不能只依赖调用大模型。因此本文的整体设计是：课程上讲过的 TF-IDF、PPMI+SVD、"
    "Skip-gram 和 MLP 构成基线并负责可解释的核心链路；BERT 微调作为效果提升；大模型只在"
    "低置信度时作为可选智能补答，而且必须基于检索到的知识库作答。即使不配置任何大模型 API，"
    "系统仍然可以完成分类、检索和答案展示。"
)

# ---------------- 2 数据集 ----------------
add_heading("2  数据集构建", 1)
add_heading("2.1  构建动机与流程", 2)
add_text(
    "公开的中文 FAQ 数据集大多面向电商或客服场景，与校园事务的词汇、意图分布差异较大；"
    "课程也没有提供可直接使用的标注数据。因此本文自建了一个小规模、领域贴合、可解释的数据集。"
    "构建流程为：先梳理校园高频事务，确定 10 类意图；每类整理 6 条标准问答，共 60 条知识库；"
    "再对每条标准问题人工改写 5 个同义问法，覆盖口语化、书面化和关键词式表达，最终得到 360 个问法。"
)
add_table(
    ["意图类别", "知识库条目数", "问法总数", "训练问法", "测试问法"],
    [["选课与教务", 6, 36, 24, 12], ["考试与成绩", 6, 36, 24, 12],
     ["图书馆", 6, 36, 24, 12], ["宿舍与住宿", 6, 36, 24, 12],
     ["食堂与生活", 6, 36, 24, 12], ["校园网络与IT", 6, 36, 24, 12],
     ["请假与考勤", 6, 36, 24, 12], ["奖助学金与资助", 6, 36, 24, 12],
     ["校园卡与缴费", 6, 36, 24, 12], ["毕业与档案", 6, 36, 24, 12],
     ["合计", 60, 360, 240, 120]],
    widths=[4.0, 3.0, 2.6, 2.6, 2.6], caption="表 1  校园 FAQ 数据集构成"
)

add_heading("2.2  训练/测试切分", 2)
add_text(
    "切分以“问法”为单位：每条知识库保留标准问题与前 3 个同义问法进入训练集（每条 4 个问法，"
    "60 条共 240 条），剩余 2 个同义问法进入测试集（共 120 条）。检索阶段的知识库始终使用 60 条"
    "标准问题，测试问法不会出现在知识库中。这样既保证测试问法未参与训练，又模拟了“知识库已上线、"
    "用户用不同说法提问”的真实场景。需要说明的是，同一条目的标准问题会同时出现在训练集和知识库中，"
    "这会让模型见过目标条目的标准表述；该设定对分类和检索一视同仁，但也意味着本文的测试集更接近"
    "“同义问法泛化”而不是“全新条目泛化”，这是数据规模有限时的一种折中。"
)
add_figure("fig1_dataset.png", "图 1  校园 FAQ 数据集构成（每类 36 个问法，训练 24、测试 12）", 15.5)

# ---------------- 3 方法 ----------------
add_heading("3  方法设计", 1)
add_heading("3.1  系统总体流程", 2)
add_text(
    "系统分为离线与在线两部分。离线阶段：读取 faq.json，构建 60 条知识库；对全部问题与答案做 jieba 分词，"
    "训练 PPMI+SVD 与 Skip-gram 词向量；用训练集问法训练 MLP 和 BERT 分类器；预先计算知识库的 TF-IDF、"
    "词向量和 BERT 向量。在线阶段：用户输入问题后先分词，BERT 给出意图概率；若置信度低于阈值且启用了"
    "大模型，则把 Top-3 检索结果作为上下文交给大模型组织答案；否则直接返回本地检索到的答案。"
)
add_text(
    "预处理使用 jieba 精确模式分词，去掉标点和“的、了、吗”等虚词，保留“怎么、多少、哪里”等疑问词，"
    "因为它们在区分意图时仍有信息量。全部语料共得到 734 个词，作为 PPMI 和 Skip-gram 的词表。"
)

add_heading("3.2  课程方法基线", 2)
add_text(
    "（1）TF-IDF 检索。把知识库问题表示为词级 TF-IDF 向量，使用 unigram + bigram、sublinear_tf，"
    "查询与知识库之间用余弦相似度排序。它代表传统的词面匹配方法，优点是无需训练、可解释，"
    "缺点是无法处理“饭卡”与“校园卡”这类同义表达。"
)
add_text(
    "（2）PPMI+SVD 词向量。在分词后的语料上以窗口 3 统计共现矩阵 C，计算 PPMI 矩阵："
    "PPMI(i,j)=max(log2(C(i,j)·N/(C(i)·C(j))+ε), 0)，再用随机 SVD 降到 100 维，得到每个词的向量。"
    "句子向量取所有词向量的平均。该方法是课程中“共现矩阵—PPMI—SVD”流程的直接实现，"
    "本质上是对词向量的低秩分解。"
)
add_text(
    "（3）Skip-gram 词向量。用 PyTorch 手写 Skip-gram：窗口 2，用中心词预测上下文词，"
    "输出层为全词表 softmax，词向量维度 100，Adam 学习率 1e-3，训练 12 轮。句子向量同样取平均。"
    "由于语料只有 734 个词、约 1 万 token，Skip-gram 很难学到稳定的语义关系，这一基线主要用于"
    "说明“数据规模对分布式表示的影响”。"
)
add_text(
    "（4）词向量平均 + MLP 意图分类。把 PPMI+SVD 或 Skip-gram 的句向量（100 维）送入两层 MLP："
    "100→128→64→10，激活函数 ReLU，dropout 0.3，Adam 学习率 1e-3、权重衰减 1e-4，批大小 32。"
    "训练前对特征做标准化，按 85%/15% 划分训练与验证，早停耐心 25 轮，保存验证集准确率最高的参数。"
)

add_heading("3.3  BERT 微调", 2)
add_text(
    "预训练模型使用 bert-base-chinese（12 层、768 维隐状态）。意图分类时取 [CLS] 位置的表示，"
    "接 dropout 和 10 类线性分类头。训练 20 轮，最大长度 64，批大小 16，学习率 3e-5，"
    "warmup 比例 0.1，权重衰减 0.01，GPU 上使用混合精度。验证集从训练问法中分层抽取 10%（24 条），"
    "每个 epoch 记录验证损失与准确率。"
)
add_text(
    "检索时不再使用分类头，而是取编码器最后一层隐状态做 attention-mask 平均池化，得到句向量；"
    "知识库与查询分别编码后做 L2 归一化和余弦相似度排序。本文也对比了 [CLS] 池化：平均池化的"
    "Recall@1 为 50.8%，[CLS] 为 44.2%，平均池化更优，最终采用平均池化。"
)

add_heading("3.4  检索分数与意图概率融合", 2)
add_text(
    "由于分类和检索共享同一套意图标签，可以尝试把检索相似度与意图概率融合："
    "score(q,d)=cos(q,d)+α·P(intent(d)|q)。α 在验证集上从 {0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0} 中挑选。"
    "验证集上各 α 的意图级准确率均为 87.5%，无法区分，最终取 α=0；在测试集上 α 从 0.1 到 1.0 仅带来小幅提升"
    "（Recall@1 51.7%、Recall@3 73.3%、MRR 0.649），因此默认仍取 α=0，避免引入额外复杂度。"
    "这是一个负结果：BERT 检索本身已经较好地对齐了意图，额外的意图概率主要起“重排同类条目”的作用，"
    "而本文的难点恰恰是同一意图内的 6 条相似 FAQ，意图概率无法进一步区分它们。"
)

add_heading("3.5  智能补答设计", 2)
add_text(
    "智能补答模块只在两种条件下触发：用户勾选“启用智能补答”，且 BERT 意图置信度低于界面阈值（默认 0.70）。"
    "触发后，系统把 Top-3 检索到的“问题—答案”拼进提示词，要求大模型“仅依据知识库回答，"
    "资料不足时明确说未找到，不要编造”，温度设为 0.2、最多 400 token。API 采用 OpenAI 兼容格式，"
    "通过环境变量配置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL；未配置密钥时页面会提示，并直接返回"
    "本地 Top-1 答案。这样既保留了大模型的语言组织能力，又让它无法脱离知识库自由发挥，"
    "同时保证断网或无密钥时系统仍可用。"
)

# ---------------- 4 实验 ----------------
add_heading("4  实验与结果分析", 1)
add_heading("4.1  实验设置与评测指标", 2)
add_text(
    "实验环境为 Windows 11、Python 3.12、PyTorch 2.10（CUDA 12.6）、Transformers 4.57、"
    "scikit-learn 1.8、jieba 0.42，显卡为 NVIDIA GeForce RTX 4060 Laptop GPU。随机种子固定为 42。"
    "检索使用条目级 Recall@1、Recall@3、MRR，以及意图级 Recall@1（Top-1 条目的意图是否正确）；"
    "分类使用 Accuracy 和 Macro-F1。测试集为 120 条未参与训练的同义问法。"
)
add_table(
    ["方法", "类型", "Recall@1", "Recall@3", "MRR", "意图级 Recall@1"],
    [["TF-IDF", "词面匹配", "0.342", "0.558", "0.467", "0.533"],
     ["PPMI+SVD 平均", "课程词向量", "0.175", "0.383", "0.322", "0.375"],
     ["Skip-gram 平均", "课程词向量", "0.233", "0.442", "0.367", "0.400"],
     ["BERT 平均池化", "预训练微调", "0.508", "0.700", "0.635", "0.808"]],
    widths=[4.0, 3.2, 2.4, 2.4, 2.0, 3.2], caption="表 2  FAQ 检索结果（测试集 120 条问法，知识库 60 条）"
)
add_text(
    "从表 2 可以看出，BERT 在四项指标上均明显领先。TF-IDF 的条目级 Recall@1 只有 34.2%，"
    "主要失败在“饭卡/校园卡”“补考/缓考”这类词面不重合的问法；PPMI+SVD 与 Skip-gram 更低，"
    "原因是训练语料仅约 1 万 token，100 维词向量难以覆盖 689 个词的语义差异，平均池化又会进一步"
    "稀释信息。BERT 借助大规模预训练，在 240 条标注上微调后就能把意图级 Recall@1 提升到 80.8%，"
    "说明迁移学习在小样本场景下优势明显。"
)
add_figure("fig2_retrieval.png", "图 2  四种检索方法的 Recall@1、Recall@3 与意图级 Recall@1", 15.5)

add_heading("4.2  意图分类结果", 2)
add_table(
    ["方法", "特征/模型", "Accuracy", "Macro-F1"],
    [["PPMI+SVD + MLP", "100 维平均词向量 + 两层 MLP", "0.500", "0.502"],
     ["Skip-gram + MLP", "100 维平均词向量 + 两层 MLP", "0.400", "0.393"],
     ["BERT 微调", "bert-base-chinese + 线性分类头", "0.850", "0.849"]],
    widths=[4.2, 7.0, 2.8, 2.8], caption="表 3  意图分类结果（测试集 120 条）"
)
add_text(
    "BERT 的分类准确率比 PPMI+SVD+MLP 高 35.0 个百分点，比 Skip-gram+MLP 高 45.0 个百分点。"
    "值得注意的是，PPMI+SVD 的检索指标（R@1 17.5%）低于 Skip-gram（23.3%），但分类指标反而更高"
    "（50.0% 对 40.0%）。这说明在极小语料上，PPMI 的全局共现统计更稳定，而 Skip-gram 的局部窗口"
    "预测容易受稀疏样本影响；同时 MLP 能在一定程度上纠正句向量的噪声。"
)
add_figure("fig3_classification.png", "图 3  三种意图分类方法的 Accuracy 与 Macro-F1", 13.5)

add_heading("4.3  训练过程", 2)
add_text(
    "图 4(a) 是两种 MLP 的训练损失，均能稳定下降并早停；图 4(b) 是 BERT 的训练损失与验证准确率。"
    "BERT 在前 4 个 epoch 快速收敛，第 6 轮起验证准确率达到 87.5% 并保持，之后训练损失继续下降但验证准确率"
    "不再提升，出现轻微过拟合。最终报告使用第 20 轮模型（测试准确率 85.0%）。"
    "在 240 条训练样本上，20 轮、3e-5 学习率是经过一次调整后的配置：最初 5 轮、2e-5 时训练损失仍"
    "在 1.5 左右，属于明显欠拟合。"
)
add_figure("fig4_training.png", "图 4  MLP 与 BERT 的训练曲线", 16.0)

add_heading("4.4  混淆矩阵与错误分析", 2)
add_text(
    "图 5 是 BERT 的混淆矩阵。食堂与生活类取得 100% 的 F1，奖助学金与资助为 96%，"
    "说明这些类别的词汇辨识度较高；毕业与档案的 F1 最低（75%），主要与“离校手续、档案、三方协议”"
    "等条目相互接近有关。选课与教务、考试与成绩之间也有少量互错，因为二者都涉及“课程、成绩、补考”"
    "等校园学业词汇。"
)
add_figure("fig5_confusion.png", "图 5  BERT 意图分类混淆矩阵", 13.0)
add_text(
    "检索错误进一步说明了任务难点：120 条测试问法中有 59 条 Top-1 未命中，其中 36 条（61.0%）"
    "错到了同一意图内的其他条目，只有 23 条（39.0%）跨意图；59 条错误中有 23 条的正确条目仍排在"
    "Top-3 内。也就是说，系统大多数时候“大方向对了，只是具体条目选错”，而 Top-3 展示可以把"
    "这部分错误补救回来（条目级 Recall@3 达到 70.0%）。表 4 给出了几个典型例子。"
)
add_table(
    ["用户问法", "Top-1 检索到的条目", "正确条目", "排名", "错误类型"],
    [["怎么取消已经选上的课", "本学期的选课什么时候开始？", "选课后还能退课吗？", "2", "同意图错条目"],
     ["对成绩有疑问怎么办", "绩点怎么计算？专业排名在哪里看？", "成绩什么时候出、在哪里查？", "3", "同意图错条目"],
     ["可以修双学位吗", "毕业前要满足哪些条件？", "怎么申请转专业或辅修？", "7", "跨意图"],
     ["平均学分绩点是什么意思", "在哪里查专业的培养方案和学分要求？", "绩点怎么计算？专业排名在哪里看？", "7", "跨意图"]],
    widths=[3.3, 6.2, 5.0, 1.4, 2.8], font_size=9.5,
    caption="表 4  BERT 检索的典型错误案例"
)
add_figure("fig6_error_confidence.png", "图 6  BERT 检索错误构成与置信度分桶准确率", 16.0)
add_text(
    "图 6(b) 按 BERT 置信度分桶统计。置信度 ≥0.9 的问法占 85.0%，该区间准确率为 90.2%；"
    "≥0.7 时覆盖 90.8%、准确率 88.1%；≥0.5 时覆盖 96.7%、准确率 86.2%。置信度整体偏高"
    "（测试集平均 0.937），但高置信度区间的准确率也确实更高，说明输出概率具有一定的区分能力，"
    "可以作为“是否转人工/调用大模型”的依据。默认阈值取 0.70 时，约 9.2% 的问法会触发智能补答，"
    "既控制了成本，也覆盖了相对不确定的样本。"
)

add_heading("4.5  补充分析", 2)
add_text(
    "（1）池化方式：平均池化与 [CLS] 池化的条目级 Recall@1 分别为 50.8% 和 44.2%，MRR 为 0.635 和"
    "0.606，平均池化更优，最终选择平均池化。"
)
add_text(
    "（2）意图过滤检索：先用 BERT 预测意图，再只在该意图的 6 条知识库中检索。由于分类准确率 85.0%，"
    "这种硬过滤仅把条目级 Recall@1 从 50.8% 提升到 51.7%，但若使用真实意图过滤（上界），Recall@1 可升至"
    "59.2%、Recall@3 升至 85.8%。这说明限制候选范围方向正确，但硬过滤会放大分类错误；更稳妥的做法是"
    "把意图概率作为软约束（即 3.4 节的融合），而本文的融合系数在验证集上无法选出有效值，最终保留"
    "纯 BERT 检索。"
)
add_text(
    "（3）智能补答的作用边界：本地 BERT 已能覆盖大部分高置信度问题，智能补答主要处理低置信度长尾问法。"
    "由于提示词严格限定在 Top-3 知识库内容内，大模型不会凭空生成政策条款；代价是当知识库本身没有"
    "相关条目时，它也只能回答“暂未找到”。这符合课程“不能只依赖大模型”的要求：大模型是锦上添花，"
    "而不是系统的知识来源。"
)

# ---------------- 5 系统实现 ----------------
add_heading("5  系统实现与演示", 1)
add_heading("5.1  代码结构", 2)
add_table(
    ["文件", "作用"],
    [["data/faq.json", "大连东软信息学院校园 FAQ 数据集（10 类意图、60 条知识库、360 个问法）"],
     ["src/common.py", "数据加载、按问法切分、jieba 分词、检索/分类指标、绘图设置"],
     ["src/run_baseline.py", "TF-IDF、PPMI+SVD、Skip-gram 检索与 MLP 分类，保存基线结果"],
     ["src/run_bert.py", "BERT 意图分类微调、句向量检索、融合实验"],
     ["src/extra_eval.py", "池化对比、意图过滤、融合系数与置信度分桶分析"],
     ["src/error_analysis.py", "检索错误分类（同意图/跨意图）与典型案例导出"],
     ["src/make_figures.py", "生成报告所需全部图表"],
     ["src/demo.py", "Gradio 问答界面（本地模型 + 可选 LLM 智能补答）"],
     ["src/llm_fallback.py", "OpenAI 兼容接口的 LLM 调用与提示词构造"]],
    widths=[4.6, 11.0], font_size=10, first_col_left=True,
    caption="表 5  项目文件结构"
)

add_heading("5.2  演示界面", 2)
add_text(
    "演示界面使用 Gradio 5.49 实现，采用卡片式布局：顶部汇总数据集与核心指标，左侧为问题输入、检索方式"
    "和示例问题，右侧按“意图识别—知识库检索—智能补答”分层展示结果，并用进度条和标签突出置信度"
    "与意图一致性。左侧可选择 BERT/TF-IDF/PPMI+SVD/Skip-gram 四种检索方式，"
    "可勾选 LLM 智能补答并调整置信度阈值。所有本地模型在启动时一次性加载，"
    "单次问答（BERT 编码 + 分类 + 检索）在 RTX 4060 上约 40–50 ms。"
)
add_figure("fig7_demo_panel.png", "图 7  Gradio 演示界面：输入“正式成绩单在哪里打印”后的意图识别与 Top-3 检索结果", 13.5)
add_text(
    "以“正式成绩单在哪里打印”为例，BERT 以 0.995 的置信度判定为“考试与成绩”，Top-1 命中"
    "“正式成绩单在哪里打印？”，相似度 0.997；Top-3 也都是考试与成绩类条目，并返回“软件园校区 A5 二楼、"
    "需携带身份证”的真实答案。由于置信度高于默认阈值 0.70，页面显示智能补答“未触发”，"
    "并说明核心答案完全来自本地模型。"
)
add_text(
    "此外，在配置 DashScope 兼容接口的环境下对智能补答链路做了一次实测：把“正式成绩单在哪里打印”的 Top-3 "
    "知识库内容交给 qwen-plus，模型返回“正式成绩单可在软件园校区 A5 二楼打印，需携带身份证……”，"
    "与知识库一致，未编造额外信息。这说明“检索约束 + 大模型组织语言”的智能补答方式是可行的，"
    "但它在系统中始终是可选路径。"
)

add_heading("5.3  运行方式", 2)
add_text("在项目根目录 nlp_final 下依次执行：", indent=False)
add_code(
    """
conda activate yolo
python src/run_baseline.py      # 课程方法基线
python src/run_bert.py          # BERT 微调与检索
python src/extra_eval.py        # 补充分析
python src/error_analysis.py    # 错误分析
python src/make_figures.py      # 生成图表
python src/demo.py              # 启动 Gradio，浏览器访问 127.0.0.1:7860
"""
)
add_text("如需启用智能补答，在 PowerShell 中配置环境变量（不配置也能正常演示）：", indent=False)
add_code(
    """
$env:LLM_API_KEY="你的密钥"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus"
"""
)

# ---------------- 6 结论 ----------------
add_heading("6  结论与心得", 1)
add_text(
    "本文完成了一个从数据构建、课程方法基线、BERT 微调到可视化演示的完整校园 FAQ 问答系统。"
    "实验结果表明：在只有 240 条训练问法的小样本条件下，BERT 微调仍能取得 85.0% 的意图分类准确率，"
    "比课程方法中最好的 MLP 高 35.0 个百分点；检索的意图级 Recall@1 达到 80.8%，"
    "条目级 Recall@3 达到 70.0%。课程方法的价值在于提供了可解释、可复现的对照：TF-IDF 依赖词面重合，"
    "PPMI+SVD 与 Skip-gram 受语料规模限制，MLP 能部分纠正句向量噪声，这些现象与课堂上的结论一致。"
)
add_text(
    "通过这个项目，我对“表示学习需要数据”有了更直观的认识。同样的 100 维向量，PPMI+SVD 在 689 个词、"
    "约 1 万 token 的语料上只能达到 17.5% 的条目级 Recall@1，而 BERT 借助预训练语言知识，"
    "只用 240 条标注就能做到 50.8%。这也解释了为什么实际系统很少从零训练词向量，而是优先使用预训练模型。"
)
add_text(
    "同时，检索比分类更暴露问题。分类错一点可能只是相邻意图，检索错一条就会答非所问；而本文数据中"
    "同一意图内有 6 条语义相近的 FAQ，条目级 Top-1 只有 50.8%。错误分析显示 61.0% 的错误是“同意图错条目”，"
    "Top-3 能补救其中 23 条。因此演示界面选择展示 Top-3 而不是只给一个答案，这是从评测结果反推的产品设计。"
)
add_text(
    "系统还有明显不足：数据规模小、同义问法由人工改写、测试集按问法切分而非按条目切分、未接入真实用户日志、"
    "未训练对比学习句向量或 cross-encoder 重排。后续可以扩充知识库与真实问法，尝试 BM25+BERT 混合检索、"
    "用对比学习微调句向量，并在更大规模上验证融合策略。总体而言，本项目满足“不依赖大模型也能完成核心任务”"
    "的要求，大模型仅作为低置信度智能补答，并始终受检索结果约束。"
)

# ---------------- 参考文献 ----------------
add_heading("参考文献", 1)
refs = [
    "[1] Mikolov T, Chen K, Corrado G, et al. Efficient Estimation of Word Representations in Vector Space[C]. ICLR, 2013.",
    "[2] Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and their Compositionality[C]. NIPS, 2013.",
    "[3] Levy O, Goldberg Y. Neural Word Embedding as Implicit Matrix Factorization[C]. NIPS, 2014.",
    "[4] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]. NAACL, 2019.",
    "[5] 宗成庆. 统计自然语言处理（第 2 版）[M]. 北京: 清华大学出版社, 2013.",
    "[6] jieba 中文分词. https://github.com/fxsjy/jieba",
    "[7] Gradio Documentation. https://www.gradio.app/docs",
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.4
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    set_font(p.add_run(r), size=10.5)

# ---------------- 附录 ----------------
add_heading("附录 A  核心代码片段", 1)
add_text("A.1  PPMI 矩阵与句向量", indent=False, bold=True, space_after=2)
add_code(
    """
def ppmi_matrix(C, eps=1e-8):
    N = C.sum()
    S = C.sum(axis=0)
    pmi = np.log2(C * N / (S[:, None] * S[None, :] + eps) + eps)
    return np.maximum(pmi, 0).astype(np.float32)

def sentence_vectors(texts, vocab, vecs):
    out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
    for i, text in enumerate(texts):
        ids = [vocab[t] for t in tokenize(text) if t in vocab]
        if ids:
            out[i] = vecs[ids].mean(axis=0)
    return out
"""
)
add_text("A.2  MLP 分类器与 BERT 训练参数", indent=False, bold=True, space_after=2)
add_code(
    """
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(128, 64), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], n_classes))

args = TrainingArguments(num_train_epochs=20, per_device_train_batch_size=16,
                         learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1,
                         fp16=torch.cuda.is_available(), eval_strategy="epoch")
"""
)
add_text("A.3  检索分数与意图概率融合", indent=False, bold=True, space_after=2)
add_code(
    """
sim = cosine_scores(query_vec, kb_vec)          # (n_query, n_kb)
fused = sim + alpha * probs[:, kb_label_ids]    # probs: 意图概率
order = np.argsort(-fused, axis=1)
"""
)

# ---------------- 页码 ----------------
footer = sec.footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(fp.add_run("第 "), size=9)
add_field(fp, "PAGE", "1")
set_font(fp.add_run(" 页"), size=9)

# 打开文档时自动更新目录
settings = doc.settings.element
upd = OxmlElement("w:updateFields")
upd.set(qn("w:val"), "true")
settings.append(upd)

doc.save(DOCX)
print("已生成：", DOCX)
