# -*- coding: utf-8 -*-
"""Gradio 演示：校园 FAQ 智能问答。

核心链路完全由本地模型完成：
  BERT 微调意图分类 + BERT/PPMI/Skip-gram/TF-IDF 检索
LLM 仅作为低置信度时的可选智能补答（需自行配置 API Key）。
界面使用自定义主题与 CSS，结果以卡片形式渲染，便于课堂演示。
"""
import html
import os
import sys
import time
from pathlib import Path

os.environ["USE_TF"] = "0"

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from common import MODEL_DIR, build_dataset, cosine_scores, load_faq, l2_normalize, tokenize
from llm_fallback import llm_answer, llm_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

print("正在加载数据与模型 ...")
LABELS, ENTRIES = load_faq()
KB, TRAIN, TEST = build_dataset(ENTRIES)
KB_TEXTS = [x["question"] for x in KB]
KB_LABEL_IDS = np.array([LABELS.index(x["label"]) for x in KB])

# 词向量检索
PPMI = np.load(MODEL_DIR / "ppmi_svd.npz")
PPMI_VOCAB = {w: i for i, w in enumerate(PPMI["vocab"].tolist())}
PPMI_VECS = PPMI["vectors"]
SG = np.load(MODEL_DIR / "skipgram.npz")
SG_VOCAB = {w: i for i, w in enumerate(SG["vocab"].tolist())}
SG_VECS = SG["vectors"]


def sent_vec(text, vocab, vecs):
    ids = [vocab[t] for t in tokenize(text) if t in vocab]
    v = vecs[ids].mean(axis=0) if ids else np.zeros(vecs.shape[1], dtype=np.float32)
    return v.reshape(1, -1)


KB_PPMI = np.vstack([sent_vec(t, PPMI_VOCAB, PPMI_VECS) for t in KB_TEXTS])
KB_SG = np.vstack([sent_vec(t, SG_VOCAB, SG_VECS) for t in KB_TEXTS])

# TF-IDF 检索
TFIDF = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
KB_TFIDF = TFIDF.fit_transform(KB_TEXTS)

# 课程方法 MLP
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(128, 64), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_mlp(name):
    ckpt = torch.load(MODEL_DIR / f"{name}.pt", map_location=DEVICE, weights_only=False)
    m = MLP(ckpt["in_dim"], ckpt["n_classes"]).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt["mean"], ckpt["std"]


MLP_PPMI, PPMI_MEAN, PPMI_STD = load_mlp("mlp_ppmi")
MLP_SG, SG_MEAN, SG_STD = load_mlp("mlp_skipgram")

# BERT
BERT = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "bert_finetuned").to(DEVICE).eval()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR / "bert_finetuned")


@torch.no_grad()
def bert_encode(texts, batch_size=32):
    outs = []
    for start in range(0, len(texts), batch_size):
        enc = TOKENIZER(texts[start:start + batch_size], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(DEVICE)
        hidden = BERT.bert(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        outs.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


@torch.no_grad()
def bert_probs(text):
    enc = TOKENIZER([text], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
    return torch.softmax(BERT(**enc).logits, dim=1)[0].cpu().numpy()


def mlp_probs(model, mean, std, x):
    xs = torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return torch.softmax(model(xs), dim=1)[0].cpu().numpy()


KB_BERT = bert_encode(KB_TEXTS)
print(f"模型加载完成，设备：{DEVICE}，知识库 {len(KB)} 条")


def retrieve(text, method):
    if method == "TF-IDF":
        sim = (TFIDF.transform([text]) @ KB_TFIDF.T).toarray()[0]
    elif method == "PPMI+SVD":
        sim = cosine_scores(sent_vec(text, PPMI_VOCAB, PPMI_VECS), KB_PPMI)[0]
    elif method == "Skip-gram":
        sim = cosine_scores(sent_vec(text, SG_VOCAB, SG_VECS), KB_SG)[0]
    else:
        sim = cosine_scores(bert_encode([text]), KB_BERT)[0]
    order = np.argsort(-sim)[:3]
    return [(KB[i], float(sim[i])) for i in order]


# ----------------------- 页面渲染工具 -----------------------
def esc(text):
    return html.escape(str(text))


def pct(x):
    return f"{max(0.0, min(1.0, float(x))):.1%}"


def bar(score, thin=False):
    cls = "cf-bar cf-bar-thin" if thin else "cf-bar"
    return f'<div class="{cls}"><span style="width:{pct(score)}"></span></div>'


def tag(text, kind=""):
    return f'<span class="cf-tag{" cf-tag-" + kind if kind else ""}">{esc(text)}</span>'


EMPTY_STATE = """
<div class="cf-empty">
  <div class="cf-empty-icon">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#4F46E5"
         stroke-width="1.6" stroke-linecap="round">
      <circle cx="11" cy="11" r="6"></circle>
      <path d="m16 16 4 4"></path>
      <path d="M11 8v6M8 11h6" opacity=".35"></path>
    </svg>
  </div>
  <h3>输入问题，开始检索</h3>
  <p>系统先用本地 BERT 判断问句意图，再从 60 条校园知识库中检索最相关的 3 条答案。</p>
  <p class="cf-empty-note">核心答案不依赖外部接口 · 单次响应约 40–50 ms</p>
</div>
"""


def render_llm(use_llm, conf, threshold, answer, err, cfg):
    """生成智能补答区块的 HTML。"""
    status = "接口状态：已配置" if cfg else "接口状态：未配置 API Key"
    if not use_llm:
        return f"""
        <section class="cf-llm">
          <div class="cf-section-title"><span class="cf-dot cf-dot-gray"></span>智能补答
            <span class="cf-pill">未启用</span></div>
          <p>核心答案完全来自本地模型。展开左侧“高级设置”可启用低置信度智能补答。</p>
          <div class="cf-llm-foot">{status}</div>
        </section>"""
    if answer:
        return f"""
        <section class="cf-llm cf-llm-on">
          <div class="cf-section-title"><span class="cf-dot cf-dot-purple"></span>智能补答
            <span class="cf-pill cf-pill-purple">已触发</span></div>
          <p>置信度 {conf:.3f} 低于阈值 {threshold:.2f}，由智能补答接口基于 Top-3 知识库内容组织答案：</p>
          <div class="cf-llm-answer">{esc(answer)}</div>
          <div class="cf-llm-foot">{status}</div>
        </section>"""
    if err:
        return f"""
        <section class="cf-llm cf-llm-fail">
          <div class="cf-section-title"><span class="cf-dot cf-dot-red"></span>智能补答
            <span class="cf-pill cf-pill-red">调用失败</span></div>
          <p>置信度 {conf:.3f} 低于阈值 {threshold:.2f}，但调用未成功，已自动回退本地答案：{esc(err)}</p>
          <div class="cf-llm-foot">{status}</div>
        </section>"""
    return f"""
    <section class="cf-llm">
      <div class="cf-section-title"><span class="cf-dot cf-dot-blue"></span>智能补答
        <span class="cf-pill cf-pill-blue">未触发</span></div>
      <p>置信度 {conf:.3f} ≥ 阈值 {threshold:.2f}，不调用外部接口，核心答案来自本地检索。</p>
      <div class="cf-llm-foot">{status}</div>
    </section>"""


def render_result(query, method, probs, ppmi_probs, sg_probs, hits, elapsed_ms, llm_html):
    intent = LABELS[int(probs.argmax())]
    conf = float(probs.max())
    hit_html = []
    for rank, (item, score) in enumerate(hits, 1):
        tag_html = tag("意图一致", "ok") if item["label"] == intent else tag("相关条目")
        hit_html.append(f"""
        <article class="cf-hit{" cf-hit-top" if rank == 1 else ""}">
          <div class="cf-hit-head">
            <span class="cf-rank">{rank}</span>
            <span class="cf-hit-q">{esc(item["question"])}</span>
            <span class="cf-sim">{score:.3f}</span>
          </div>
          {bar(score, thin=True)}
          <p class="cf-answer">{esc(item["answer"])}</p>
          <div class="cf-tags">{tag_html}{tag("意图：" + item["label"])}</div>
        </article>""")
    return f"""
    <div class="cf-result">
      <div class="cf-result-head">
        <div class="cf-query">“{esc(query)}”</div>
        <div class="cf-meta">{tag(method, "soft")}<span class="cf-time">{elapsed_ms} ms</span></div>
      </div>

      <section class="cf-intent">
        <div class="cf-section-title"><span class="cf-dot"></span>意图识别
          <span class="cf-chip-local">本地 BERT 微调</span></div>
        <div class="cf-intent-row">
          <div class="cf-intent-name">{esc(intent)}</div>
          <div class="cf-conf"><b>{conf:.3f}</b><span>置信度</span></div>
        </div>
        {bar(conf)}
        <div class="cf-compare">
          <div class="cf-compare-item"><span>PPMI+SVD + MLP</span>
            <b>{esc(LABELS[int(ppmi_probs.argmax())])} · {float(ppmi_probs.max()):.3f}</b></div>
          <div class="cf-compare-item"><span>Skip-gram + MLP</span>
            <b>{esc(LABELS[int(sg_probs.argmax())])} · {float(sg_probs.max()):.3f}</b></div>
        </div>
      </section>

      <section class="cf-retrieval">
        <div class="cf-section-title"><span class="cf-dot"></span>知识库检索
          <span class="cf-chip-method">{esc(method)}</span></div>
        {"".join(hit_html)}
      </section>

      {llm_html}
    </div>
    """


def build_answer(query, method, use_llm, threshold):
    query = (query or "").strip()
    if not query:
        return EMPTY_STATE

    t0 = time.time()
    probs = bert_probs(query)
    conf = float(probs.max())

    ppmi_vec = sent_vec(query, PPMI_VOCAB, PPMI_VECS)
    sg_vec = sent_vec(query, SG_VOCAB, SG_VECS)
    mlp_p = mlp_probs(MLP_PPMI, PPMI_MEAN, PPMI_STD, ppmi_vec)
    mlp_s = mlp_probs(MLP_SG, SG_MEAN, SG_STD, sg_vec)

    hits = retrieve(query, method)

    cfg = llm_config()
    answer = err = None
    if use_llm and conf < threshold:
        answer, err = llm_answer(query, [h[0] for h in hits])
    llm_html = render_llm(use_llm, conf, threshold, answer, err, cfg)

    elapsed_ms = int((time.time() - t0) * 1000)
    return render_result(query, method, probs, mlp_p, mlp_s, hits, elapsed_ms, llm_html)


# ----------------------- 主题与样式 -----------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f5f7fb",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_shadow="none",
    block_radius="18px",
    input_background_fill="#f8fafc",
    input_border_color="#dfe5f0",
)

CUSTOM_CSS = """
/* ---------- 全局 ---------- */
html, body { background: #f5f7fb !important; }
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 26px 20px 38px !important;
  background:
    radial-gradient(900px 420px at 6% -6%, rgba(99, 102, 241, .13), transparent 62%),
    radial-gradient(760px 420px at 100% 2%, rgba(14, 165, 233, .10), transparent 58%),
    #f5f7fb !important;
}
.gradio-container, .gradio-container button, .gradio-container input,
.gradio-container textarea, .gradio-container label {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Microsoft YaHei", "Noto Sans SC", sans-serif !important;
}
.gradio-container footer { display: none !important; }

/* ---------- 顶部 ---------- */
.cf-hero {
  position: relative;
  overflow: hidden;
  padding: 30px 34px 26px;
  border-radius: 22px;
  background: linear-gradient(120deg, #1e1b4b 0%, #312e81 52%, #4338ca 100%);
  box-shadow: 0 20px 44px rgba(49, 46, 129, .26);
}
.cf-hero::before {
  content: ""; position: absolute; width: 440px; height: 440px; right: -130px; top: -190px;
  background: radial-gradient(circle, rgba(129, 140, 248, .38), transparent 66%);
}
.cf-hero::after {
  content: ""; position: absolute; width: 320px; height: 320px; left: -110px; bottom: -200px;
  background: radial-gradient(circle, rgba(56, 189, 248, .24), transparent 66%);
}
.cf-hero > * { position: relative; z-index: 1; }
.cf-hero-badge {
  display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
  border-radius: 999px; background: rgba(255, 255, 255, .12);
  border: 1px solid rgba(255, 255, 255, .18); color: #e0e7ff;
  font-size: 12px; letter-spacing: .4px;
}
.cf-hero-badge i {
  width: 7px; height: 7px; border-radius: 50%; background: #a5b4fc;
  box-shadow: 0 0 0 3px rgba(165, 180, 252, .25);
}
.cf-hero h1 {
  margin: 15px 0 8px; font-size: 31px; line-height: 1.25;
  color: #fff; font-weight: 750; letter-spacing: .5px;
}
.cf-hero p { margin: 0; color: rgba(255, 255, 255, .72); font-size: 14px; line-height: 1.7; }
.cf-hero-sub { margin-top: 8px !important; font-size: 12.5px !important; color: rgba(255, 255, 255, .52) !important; }
.cf-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.cf-stat {
  min-width: 122px; padding: 10px 16px; border-radius: 14px;
  background: rgba(255, 255, 255, .09); border: 1px solid rgba(255, 255, 255, .14);
  backdrop-filter: blur(6px);
}
.cf-stat b { display: block; font-size: 19px; color: #fff; font-weight: 700; font-variant-numeric: tabular-nums; }
.cf-stat span { font-size: 12px; color: rgba(255, 255, 255, .62); }

/* ---------- 输入卡片 ---------- */
.cf-card {
  gap: 12px !important;
  background: #fff !important;
  border: 1px solid #e7eaf3 !important;
  border-radius: 22px !important;
  padding: 22px !important;
  box-shadow: 0 14px 34px rgba(15, 23, 42, .07) !important;
}
.cf-card .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.cf-field-title {
  display: flex; align-items: baseline; gap: 8px;
  margin: 2px 0 2px; font-size: 13px; font-weight: 700; color: #1e293b;
}
.cf-field-title span { font-size: 12px; font-weight: 400; color: #94a3b8; }

.cf-input textarea {
  min-height: 104px !important;
  padding: 14px !important;
  border: 1px solid #dfe5f0 !important;
  border-radius: 14px !important;
  background: #f8fafc !important;
  font-size: 15px !important;
  line-height: 1.65 !important;
  color: #0f172a !important;
  box-shadow: none !important;
}
.cf-input textarea:focus {
  background: #fff !important;
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 4px rgba(99, 102, 241, .12) !important;
}
.cf-radio .wrap { display: grid !important; grid-template-columns: 1fr 1fr; gap: 8px !important; }
.cf-radio label {
  padding: 8px 10px !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 11px !important;
  background: #f8fafc !important;
  font-size: 12px !important;
  white-space: nowrap !important;
  transition: all .15s ease;
}
.cf-radio label:hover { border-color: #c7d2fe !important; }
.cf-radio label.selected {
  background: #eef2ff !important;
  border-color: #6366f1 !important;
  color: #3730a3 !important;
  font-weight: 600 !important;
}
.cf-adv {
  border: 1px dashed #e2e8f0 !important;
  border-radius: 14px !important;
  background: #fbfcfe !important;
}
.cf-adv > div[style*="display: block"] { padding: 10px 14px 8px !important; }
.cf-adv .form { gap: 10px !important; }
.cf-adv label { color: #475569 !important; font-size: 12.5px !important; }
.cf-adv input[type=number] { color: #4338ca !important; font-weight: 600 !important; }
.cf-adv-note { margin: 2px 0 4px; font-size: 11.5px; color: #94a3b8; line-height: 1.6; }
.cf-submit {
  margin-top: 2px;
  padding: 12px 18px !important;
  border: none !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
  color: #fff !important;
  font-size: 15px !important;
  font-weight: 650 !important;
  letter-spacing: 1px;
  box-shadow: 0 10px 24px rgba(79, 70, 229, .28) !important;
  transition: transform .15s ease, box-shadow .15s ease;
}
.cf-submit:hover {
  transform: translateY(-1px);
  box-shadow: 0 13px 28px rgba(79, 70, 229, .34) !important;
}
.cf-chip-row { display: grid !important; grid-template-columns: 1fr 1fr; gap: 8px !important; }
.cf-chip {
  justify-content: flex-start !important;
  padding: 8px 11px !important;
  border: 1px solid #e8edf5 !important;
  border-radius: 10px !important;
  background: #f8fafc !important;
  color: #475569 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-align: left !important;
  box-shadow: none !important;
  white-space: normal !important;
  line-height: 1.4 !important;
}
.cf-chip:hover {
  background: #f5f7ff !important;
  border-color: #c7d2fe !important;
  color: #4338ca !important;
}
.cf-output-card { min-height: 620px; }
.cf-output { width: 100%; }
.cf-output .prose { max-width: none !important; }

/* ---------- 结果卡片 ---------- */
.cf-result-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 12px; padding-bottom: 15px; margin-bottom: 18px;
  border-bottom: 1px solid #eef1f6;
}
.cf-query { font-size: 19px; font-weight: 700; color: #0f172a; line-height: 1.5; }
.cf-meta { display: flex; align-items: center; gap: 8px; flex-shrink: 0; padding-top: 3px; }
.cf-time { font-size: 12px; color: #94a3b8; font-variant-numeric: tabular-nums; }
.cf-tag {
  display: inline-flex; align-items: center; padding: 3px 9px;
  border-radius: 999px; border: 1px solid #e9eef5; background: #f1f5f9;
  color: #64748b; font-size: 12px; line-height: 1.5;
}
.cf-tag-ok { background: #ecfdf5; border-color: #d1fae5; color: #047857; }
.cf-tag-soft { background: #eef2ff; border-color: #e0e7ff; color: #4338ca; }

.cf-section-title {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 12px; font-size: 13px; font-weight: 700; color: #334155;
}
.cf-dot { width: 7px; height: 7px; border-radius: 50%; background: #4f46e5; box-shadow: 0 0 0 4px rgba(79, 70, 229, .12); }
.cf-dot-gray { background: #94a3b8; box-shadow: 0 0 0 4px rgba(148, 163, 184, .14); }
.cf-dot-purple { background: #8b5cf6; box-shadow: 0 0 0 4px rgba(139, 92, 246, .14); }
.cf-dot-blue { background: #3b82f6; box-shadow: 0 0 0 4px rgba(59, 130, 246, .13); }
.cf-dot-red { background: #ef4444; box-shadow: 0 0 0 4px rgba(239, 68, 68, .13); }
.cf-chip-local, .cf-chip-method {
  padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
  background: #ede9fe; color: #6d28d9;
}
.cf-chip-method { background: #eef2ff; color: #4338ca; }

.cf-intent {
  padding: 18px; margin-bottom: 18px;
  border: 1px solid #e6e9fb; border-radius: 18px;
  background: linear-gradient(135deg, #f5f6ff 0%, #fbfbfe 72%);
}
.cf-intent-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.cf-intent-name { font-size: 26px; font-weight: 750; color: #1e1b4b; line-height: 1.25; }
.cf-conf { text-align: right; }
.cf-conf b { display: block; font-size: 22px; color: #4338ca; line-height: 1; font-variant-numeric: tabular-nums; }
.cf-conf span { font-size: 11px; color: #94a3b8; }
.cf-bar { height: 8px; border-radius: 999px; background: #e8eaf3; overflow: hidden; }
.cf-bar span {
  display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #818cf8, #4f46e5);
}
.cf-bar-thin { height: 5px; background: #eef1f7; }
.cf-bar-thin span { background: linear-gradient(90deg, #a5b4fc, #6366f1); }
.cf-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
.cf-compare-item { padding: 10px 12px; border: 1px solid #edf0f7; border-radius: 12px; background: #fff; }
.cf-compare-item span { display: block; margin-bottom: 4px; font-size: 11px; color: #94a3b8; }
.cf-compare-item b { font-size: 12.5px; font-weight: 600; color: #334155; }

.cf-hit {
  margin-bottom: 10px; padding: 14px 16px;
  border: 1px solid #edf0f7; border-radius: 16px; background: #fff;
}
.cf-hit-top { border-color: #dde1fb; background: #fdfdff; box-shadow: 0 8px 22px rgba(79, 70, 229, .07); }
.cf-hit-head { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.cf-rank {
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  width: 22px; height: 22px; border-radius: 8px;
  background: #f1f5f9; color: #64748b; font-size: 12px; font-weight: 700;
}
.cf-hit-top .cf-rank { background: #4f46e5; color: #fff; }
.cf-hit-q { flex: 1; font-size: 14px; font-weight: 650; color: #1e293b; line-height: 1.5; }
.cf-sim { font-size: 12px; font-weight: 700; color: #4338ca; font-variant-numeric: tabular-nums; }
.cf-answer { margin: 8px 0 10px; font-size: 13px; line-height: 1.78; color: #475569; }
.cf-tags { display: flex; flex-wrap: wrap; gap: 6px; }

.cf-llm { padding: 16px; border: 1px solid #edf0f7; border-radius: 16px; background: #fbfcfe; }
.cf-llm-on { background: #fcfbff; border-color: #ece7fd; }
.cf-llm-fail { background: #fffafa; border-color: #fde8e8; }
.cf-llm p { margin: 0; font-size: 13px; line-height: 1.75; color: #64748b; }
.cf-pill { padding: 2px 9px; border-radius: 999px; background: #f1f5f9; color: #64748b; font-size: 11px; font-weight: 600; }
.cf-pill-blue { background: #eff6ff; color: #1d4ed8; }
.cf-pill-purple { background: #f5f3ff; color: #6d28d9; }
.cf-pill-red { background: #fef2f2; color: #b91c1c; }
.cf-llm-answer {
  margin-top: 10px; padding: 12px 14px; border-left: 3px solid #8b5cf6; border-radius: 0 12px 12px 0;
  background: #fff; color: #334155; font-size: 13px; line-height: 1.8;
}
.cf-llm-foot { margin-top: 10px; font-size: 11px; color: #94a3b8; }

/* ---------- 空状态与页脚 ---------- */
.cf-empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 560px; padding: 64px 24px; text-align: center;
}
.cf-empty-icon {
  display: flex; align-items: center; justify-content: center;
  width: 64px; height: 64px; margin-bottom: 18px;
  border-radius: 20px; background: #eef2ff;
}
.cf-empty h3 { margin: 0 0 8px; font-size: 19px; color: #1e293b; }
.cf-empty p { margin: 0; max-width: 380px; font-size: 13px; line-height: 1.75; color: #64748b; }
.cf-empty-note { margin-top: 10px !important; font-size: 12px !important; color: #94a3b8 !important; }
.cf-foot {
  max-width: 1180px; margin: 18px auto 0; padding: 0 4px;
  font-size: 12px; color: #94a3b8; text-align: center; line-height: 1.7;
}
"""

HERO = """
<div class="cf-hero">
  <div class="cf-hero-badge"><i></i>大连东软信息学院 · 自然语言处理 · 期末项目</div>
  <h1>校园 FAQ 智能问答</h1>
  <p>BERT 微调意图识别 + 语义检索，覆盖大连东软信息学院 10 类校园事务；低置信度时可选用智能补答，核心答案始终来自本地知识库。</p>
  <p class="cf-hero-sub">10 类意图 · 60 条知识库 · 360 条问法 · CPU/GPU 均可运行</p>
  <div class="cf-stats">
    <div class="cf-stat"><b>85.0%</b><span>意图分类准确率</span></div>
    <div class="cf-stat"><b>50.8%</b><span>条目 Recall@1</span></div>
    <div class="cf-stat"><b>70.0%</b><span>条目 Recall@3</span></div>
    <div class="cf-stat"><b>45 ms</b><span>单次响应耗时</span></div>
  </div>
</div>
"""

METHODS = ["BERT 向量检索", "TF-IDF", "PPMI+SVD", "Skip-gram"]
EXAMPLES = [
    "金智教务系统怎么查成绩",
    "正式成绩单在哪里打印",
    "软件园三期宿舍几点锁门",
    "食堂能用微信支付吗",
    "电费在哪里交",
    "东软智慧学工怎么请假",
    "毕业要修多少素质学分",
    "美团外卖能送到宿舍楼下吗",
]

with gr.Blocks(title="校园 FAQ 智能问答", theme=THEME, css=CUSTOM_CSS, fill_width=True) as demo:
    gr.HTML(HERO, container=False, padding=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["cf-card", "cf-input-card"]):
            gr.HTML('<div class="cf-field-title">你的问题<span>支持自然语言提问，回车即可检索</span></div>',
                    container=False, padding=False)
            query = gr.Textbox(show_label=False, placeholder="例如：正式成绩单在哪里打印？",
                               lines=3, elem_classes=["cf-input"])
            gr.HTML('<div class="cf-field-title">检索方式</div>', container=False, padding=False)
            method = gr.Radio(METHODS, value="BERT 向量检索", show_label=False,
                              elem_classes=["cf-radio"])
            with gr.Accordion("高级设置 · 智能补答（可选）", open=False, elem_classes=["cf-adv"]):
                use_llm = gr.Checkbox(label="低置信度时启用智能补答", value=False)
                threshold = gr.Slider(0.30, 0.95, value=0.70, step=0.05, label="补答触发阈值")
                gr.HTML('<div class="cf-adv-note">未配置 API Key 时自动回退本地答案，不影响核心问答。</div>',
                        container=False, padding=False)
            btn = gr.Button("开始检索", variant="primary", elem_classes=["cf-submit"])
            gr.HTML('<div class="cf-field-title">试试这些问题<span>点击直接提问</span></div>',
                    container=False, padding=False)
            chips = []
            for chunk in (EXAMPLES[:4], EXAMPLES[4:]):
                with gr.Row(elem_classes=["cf-chip-row"]):
                    for q in chunk:
                        chips.append((q, gr.Button(q, size="sm", elem_classes=["cf-chip"])))

        with gr.Column(scale=7, elem_classes=["cf-card", "cf-output-card"]):
            output = gr.HTML(EMPTY_STATE, container=False, padding=False, elem_classes=["cf-output"])

    for q, chip in chips:
        chip.click(lambda x=q: x, outputs=query).then(
            build_answer, inputs=[query, method, use_llm, threshold], outputs=output
        )

    btn.click(build_answer, inputs=[query, method, use_llm, threshold], outputs=output)
    query.submit(build_answer, inputs=[query, method, use_llm, threshold], outputs=output)

    gr.HTML(
        '<div class="cf-foot">核心链路由课程方法（PPMI+SVD / Skip-gram / 词向量平均 + MLP）与 BERT 微调实现；'
        '智能补答只在置信度低于阈值时可选启用。</div>',
        container=False, padding=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_api=False)
