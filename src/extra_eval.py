# -*- coding: utf-8 -*-
"""补充实验：BERT 池化方式对比、意图过滤检索、融合系数与置信度分析。"""
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from common import (MODEL_DIR, RESULT_DIR, build_dataset, cosine_scores, load_faq,
                    retrieval_metrics, save_json, set_seed)
from run_bert import MAX_LEN, encode_texts

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_cls(model, tokenizer, texts, batch_size=32):
    model.eval()
    outs = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(DEVICE)
        outs.append(model.bert(**enc).last_hidden_state[:, 0].cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def main():
    labels, entries = load_faq()
    kb, _, test = build_dataset(entries)
    kb_texts = [x["question"] for x in kb]
    kb_labels = np.array([labels.index(x["label"]) for x in kb])
    test_texts = [x[0] for x in test]
    test_labels = [x[1] for x in test]
    test_ids = [x[2] for x in test]
    test_y = np.array([labels.index(x[1]) for x in test])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "bert_finetuned")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "bert_finetuned").to(DEVICE)

    kb_mean = encode_texts(model, tokenizer, kb_texts)
    q_mean = encode_texts(model, tokenizer, test_texts)
    kb_cls = encode_cls(model, tokenizer, kb_texts)
    q_cls = encode_cls(model, tokenizer, test_texts)

    mean_ret = retrieval_metrics(cosine_scores(q_mean, kb_mean), test_labels, test_ids, kb)
    cls_ret = retrieval_metrics(cosine_scores(q_cls, kb_cls), test_labels, test_ids, kb)
    print("平均池化：", mean_ret)
    print("CLS 池化：", cls_ret)

    # 直接用保存的测试概率做融合与意图过滤分析
    probs = np.load(RESULT_DIR / "bert_test_probs.npy")
    sim = cosine_scores(q_mean, kb_mean)
    pred = probs.argmax(1)

    fusion = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        fused = sim + alpha * probs[:, kb_labels]
        fusion[alpha] = retrieval_metrics(fused, test_labels, test_ids, kb)

    # 意图过滤：候选限制在预测意图对应的 6 条知识库
    filtered_sim = np.full_like(sim, -1e9)
    for i, p in enumerate(pred):
        filtered_sim[i, kb_labels == p] = sim[i, kb_labels == p]
    filtered = retrieval_metrics(filtered_sim, test_labels, test_ids, kb)

    # 真实意图过滤（上界）
    oracle_sim = np.full_like(sim, -1e9)
    for i, y in enumerate(test_y):
        oracle_sim[i, kb_labels == y] = sim[i, kb_labels == y]
    oracle = retrieval_metrics(oracle_sim, test_labels, test_ids, kb)

    # 置信度分桶：高置信度时的准确率
    conf = probs.max(1)
    buckets = {}
    for lo in [0.0, 0.3, 0.5, 0.7, 0.9]:
        hi = 1.01
        m = (conf >= lo) & (conf < hi) if lo == 0.0 else (conf >= lo)
        if m.sum():
            buckets[f">={lo}"] = {"n": int(m.sum()), "coverage": float(m.mean()),
                                  "accuracy": float((pred[m] == test_y[m]).mean())}

    result = {
        "pooling": {"mean": mean_ret, "cls": cls_ret},
        "fusion_alpha_test": {str(k): v for k, v in fusion.items()},
        "intent_filter": {"predicted": filtered, "oracle": oracle},
        "confidence_buckets": buckets,
        "mean_test_confidence": float(conf.mean()),
    }
    save_json(result, RESULT_DIR / "extra.json")
    for k, v in result.items():
        print(k, v)


if __name__ == "__main__":
    main()
