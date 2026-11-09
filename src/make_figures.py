# -*- coding: utf-8 -*-
"""根据实验结果生成报告图表。"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from common import FIG_DIR, RESULT_DIR, load_json, load_faq, build_dataset, setup_matplotlib, data_stats

plt = setup_matplotlib()
C1, C2, C3, C4 = "#3b6ea5", "#e08e3c", "#5b9b6f", "#b45454"


def fig_dataset():
    labels, entries = load_faq()
    kb, train, test = build_dataset(entries)
    stats = data_stats(entries, train, test)
    per = stats["per_intent"]
    names = list(per.keys())
    train_n = [24] * len(names)   # 每类 6 条 ×（标准问 + 3 同义问法）
    test_n = [12] * len(names)    # 每类 6 条 × 2 同义问法
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x, train_n, color=C1, label="训练集")
    ax.bar(x, test_n, bottom=train_n, color=C2, label="测试集")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("问题数量")
    ax.set_ylim(0, 40)
    ax.set_title("图 1  校园 FAQ 数据集构成（10 类意图 / 60 条知识库 / 360 个问法）", pad=32)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for i, (a, b) in enumerate(zip(train_n, test_n)):
        ax.text(i, a + b + 0.6, str(a + b), ha="center", fontsize=8, color="#444")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_dataset.png", dpi=200)
    plt.close()


def fig_retrieval():
    base = load_json(RESULT_DIR / "baseline.json")
    bert = load_json(RESULT_DIR / "bert.json")
    methods = [
        ("TF-IDF", base["methods"]["tfidf"]["retrieval"], C1),
        ("PPMI+SVD", base["methods"]["ppmi_svd"]["retrieval"], C2),
        ("Skip-gram", base["methods"]["skipgram"]["retrieval"], C3),
        ("BERT 微调", bert["retrieval"], C4),
    ]
    metrics = [("recall@1", "Recall@1"), ("recall@3", "Recall@3"), ("intent_recall@1", "意图级 Recall@1")]
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (name, r, color) in enumerate(methods):
        vals = [r[m] for m, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title("图 2  FAQ 检索方法对比（条目级 Recall / 意图级 Recall）")
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_retrieval.png", dpi=200)
    plt.close()


def fig_classification():
    base = load_json(RESULT_DIR / "baseline.json")
    bert = load_json(RESULT_DIR / "bert.json")
    methods = [
        ("PPMI+SVD + MLP", base["methods"]["mlp_ppmi"]["classification"], C1),
        ("Skip-gram + MLP", base["methods"]["mlp_skipgram"]["classification"], C2),
        ("BERT 微调", bert["classification"], C4),
    ]
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (name, m, color) in enumerate(methods):
        vals = [m[k] for k, _ in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title("图 3  意图分类方法对比")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_classification.png", dpi=200)
    plt.close()


def fig_training():
    base = load_json(RESULT_DIR / "baseline.json")
    bert = load_json(RESULT_DIR / "bert.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    for key, name, color in [("mlp_ppmi", "PPMI+SVD + MLP", C1), ("mlp_skipgram", "Skip-gram + MLP", C2)]:
        h = base["methods"][key]["train_history"]
        ax.plot(range(1, len(h["loss"]) + 1), h["loss"], color=color, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("训练损失")
    ax.set_title("(a) MLP 训练损失")
    ax.legend(fontsize=9)

    ax = axes[1]
    log = bert["train_log"]
    train_loss = [(x["epoch"], x["loss"]) for x in log if "loss" in x]
    evals = [(x["epoch"], x["eval_loss"], x["eval_accuracy"]) for x in log if "eval_accuracy" in x]
    ax.plot([e for e, _ in train_loss], [l for _, l in train_loss], color=C1, label="训练损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("训练损失", color=C1)
    ax.tick_params(axis="y", labelcolor=C1)
    ax2 = ax.twinx()
    ax2.plot([e for e, _, _ in evals], [a for _, _, a in evals], color=C4, marker="o", ms=3, label="验证准确率")
    ax2.set_ylabel("验证准确率", color=C4)
    ax2.tick_params(axis="y", labelcolor=C4)
    ax2.set_ylim(0, 1.0)
    ax.set_title("(b) BERT 训练损失与验证准确率")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="center right")

    plt.suptitle("图 4  训练曲线", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_training.png", dpi=200)
    plt.close()


def fig_confusion():
    bert = load_json(RESULT_DIR / "bert.json")
    labels = [x for x in bert["classification"]["per_class"].keys()]
    cm = np.array(bert["classification"]["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_xlabel("预测意图")
    ax.set_ylabel("真实意图")
    ax.set_title("图 5  BERT 意图分类混淆矩阵（测试集 120 条）")
    fig.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_confusion.png", dpi=200)
    plt.close()


def fig_error_and_confidence():
    err = load_json(RESULT_DIR / "error_analysis.json")["stats"]
    extra = load_json(RESULT_DIR / "extra.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    names = ["Top-1 正确", "错到同意图条目", "错到其他意图"]
    vals = [err["n"] - err["top1_error"], err["same_intent"], err["cross_intent"]]
    bars = ax.bar(names, vals, color=[C3, C2, C4])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=9)
    ax.set_ylabel("问题数")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("(a) BERT 检索 Top-1 结果构成")
    ax.text(0.98, 0.95, f"Top-3 命中：{err['gold_in_top3']} 条错误得以补救",
            transform=ax.transAxes, ha="right", fontsize=9, color="#555")

    ax = axes[1]
    buckets = extra["confidence_buckets"]
    keys = [k for k in buckets if k != ">=0.0"]
    x = np.arange(len(keys))
    cov = [buckets[k]["coverage"] for k in keys]
    acc = [buckets[k]["accuracy"] for k in keys]
    ax.bar(x - 0.18, cov, 0.36, color=C1, label="覆盖率")
    ax.bar(x + 0.18, acc, 0.36, color=C3, label="该区间准确率")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.08)
    ax.set_title("(b) 置信度阈值与分类准确率")
    ax.legend(fontsize=9)
    for i, (c, a) in enumerate(zip(cov, acc)):
        ax.text(i - 0.18, c + 0.02, f"{c:.2f}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=7.5)

    plt.suptitle("图 6  错误分析与置信度分析", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_error_confidence.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_dataset()
    fig_retrieval()
    fig_classification()
    fig_training()
    fig_confusion()
    fig_error_and_confidence()
    print("图表已生成到", FIG_DIR)
