# -*- coding: utf-8 -*-
"""错误分析：统计 BERT 检索 Top-1 错误中“同意图近义条目”与“跨意图”的比例，并导出案例。"""
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from common import (MODEL_DIR, RESULT_DIR, build_dataset, cosine_scores, load_faq, save_json, set_seed)
from run_bert import encode_texts

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    labels, entries = load_faq()
    kb, _, test = build_dataset(entries)
    kb_texts = [x["question"] for x in kb]
    kb_ids = [x["id"] for x in kb]
    kb_labels = [x["label"] for x in kb]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "bert_finetuned")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "bert_finetuned").to(DEVICE)
    kb_vec = encode_texts(model, tokenizer, kb_texts)
    q_vec = encode_texts(model, tokenizer, [x[0] for x in test])
    sim = cosine_scores(q_vec, kb_vec)

    stats = {"n": len(test), "top1_error": 0, "same_intent": 0, "cross_intent": 0, "gold_in_top3": 0}
    cases = []
    for i, (query, label, eid) in enumerate(test):
        order = np.argsort(-sim[i])
        if kb_ids[order[0]] == eid:
            continue
        stats["top1_error"] += 1
        top1 = order[0]
        same_intent = kb_labels[top1] == label
        stats["same_intent" if same_intent else "cross_intent"] += 1
        gold_rank = int(np.where(order == kb_ids.index(eid))[0][0]) + 1
        if gold_rank <= 3:
            stats["gold_in_top3"] += 1
        if len(cases) < 25:
            cases.append({
                "query": query,
                "gold_intent": label,
                "gold_question": kb[kb_ids.index(eid)]["question"],
                "pred_question": kb[top1]["question"],
                "pred_intent": kb_labels[top1],
                "score": float(sim[i, top1]),
                "gold_rank": gold_rank,
                "failure_type": "同意图错条目" if same_intent else "跨意图",
            })

    save_json({"stats": stats, "cases": cases}, RESULT_DIR / "error_analysis.json")
    print(stats)
    for c in cases[:8]:
        print(f"{c['query']} -> {c['pred_question']} (gold rank {c['gold_rank']}, {c['failure_type']})")


if __name__ == "__main__":
    main()
