# -*- coding: utf-8 -*-
"""公共工具：数据加载、数据集切分、中文分词、评测指标、绘图风格。"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import jieba
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "faq.json"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"

# 只过滤标点和极少数虚词，保留“怎么/多少/哪里”等疑问信息
STOPWORDS = set("的 了 吗 呢 吧 啊 呀 哦 是 在 有 和 与 及 或 就 都 也 还 我 你 他 她 它 们 这 那 个 一下".split())
TOKEN_RE = re.compile(r"[\u4e00-\u9fffa-zA-Z0-9]")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def tokenize(text: str):
    """jieba 分词 + 去标点/虚词。"""
    tokens = []
    for t in jieba.lcut(text.strip().lower()):
        t = t.strip()
        if t and TOKEN_RE.match(t) and t not in STOPWORDS:
            tokens.append(t)
    return tokens


def load_faq(path: Path = DATA_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    labels = [it["label"] for it in data["intents"]]
    entries = []
    for it in data["intents"]:
        for e in it["entries"]:
            entries.append({
                "id": e["id"],
                "label": it["label"],
                "question": e["question"],
                "answer": e["answer"],
                "paraphrases": e["paraphrases"],
            })
    return labels, entries


def build_dataset(entries, n_train_para: int = 3):
    """按问法切分：每条知识库保留标准问题 + 前 n_train_para 个同义问法作训练，
    其余同义问法作测试。知识库（检索候选）始终包含全部 60 条标准问题。

    返回:
      kb: 60 条 {id, label, question, answer}
      train: [(query, label, entry_id)]
      test:  [(query, label, entry_id)]
    """
    kb = [{"id": e["id"], "label": e["label"], "question": e["question"], "answer": e["answer"]} for e in entries]
    train, test = [], []
    for e in entries:
        train.append((e["question"], e["label"], e["id"]))
        for p in e["paraphrases"][:n_train_para]:
            train.append((p, e["label"], e["id"]))
        for p in e["paraphrases"][n_train_para:]:
            test.append((p, e["label"], e["id"]))
    return kb, train, test


def l2_normalize(x, eps=1e-8):
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / norm


def cosine_scores(A, B, eps=1e-8):
    """A: (n, d)，B: (m, d)，返回 (n, m) 余弦相似度。"""
    A = l2_normalize(A, eps)
    B = l2_normalize(B, eps)
    return A @ B.T


def retrieval_metrics(scores, query_labels, query_entry_ids, kb, ks=(1, 3)):
    """scores: (n_query, n_kb) 相似度矩阵（越大越相似）。

    返回 entry 级 Recall@k / MRR，以及 intent 级 Recall@1。
    """
    kb_ids = [x["id"] for x in kb]
    kb_labels = [x["label"] for x in kb]
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        hits = 0
        for i, eid in enumerate(query_entry_ids):
            top = order[i, :k]
            if eid in [kb_ids[j] for j in top]:
                hits += 1
        out[f"recall@{k}"] = hits / len(query_entry_ids)

    # MRR（正确条目的排名取倒数）
    rr = []
    for i, eid in enumerate(query_entry_ids):
        rank = np.where([kb_ids[j] == eid for j in order[i]])[0]
        rr.append(1.0 / (rank[0] + 1) if len(rank) else 0.0)
    out["mrr"] = float(np.mean(rr))

    # 意图级 Recall@1：Top-1 条目意图是否正确
    intent_hits = sum(1 for i, lab in enumerate(query_labels) if kb_labels[order[i, 0]] == lab)
    out["intent_recall@1"] = intent_hits / len(query_labels)
    return out


def classification_metrics(y_true, y_pred, labels, label_names=None):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    names = label_names or labels
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i in range(len(labels))}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": cm}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def data_stats(entries, train, test):
    """数据集统计，用于报告与图表。"""
    label_counts = Counter(e["label"] for e in entries)
    return {
        "n_intents": len(label_counts),
        "n_entries": len(entries),
        "n_questions": sum(1 + len(e["paraphrases"]) for e in entries),
        "n_train": len(train),
        "n_test": len(test),
        "n_kb": len(entries),
        "per_intent": dict(label_counts),
    }
