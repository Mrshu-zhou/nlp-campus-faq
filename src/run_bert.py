# -*- coding: utf-8 -*-
"""BERT 提升方案：
1) bert-base-chinese 意图分类微调
2) 用微调后编码器的平均池化向量做 FAQ 检索
3) 检索分数 + 意图概率融合（融合系数在验证集上选择）
"""
import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"          # 环境里的 TensorFlow 与 Keras3 不兼容，禁用 TF 后端
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)

sys.path.append(str(Path(__file__).resolve().parent))
from common import (MODEL_DIR, RESULT_DIR, build_dataset, classification_metrics, cosine_scores,
                    load_faq, retrieval_metrics, save_json, set_seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_DIR = MODEL_DIR / "bert-base-chinese"
MAX_LEN = 64


class FaqDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding="max_length")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, y = eval_pred
    pred = logits.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=32):
    """取最后一层隐状态的平均池化作为句向量。"""
    model.eval()
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        hidden = model.bert(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (hidden * mask).sum(1) / mask.sum(1)
        outs.append(vec.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


@torch.no_grad()
def predict_probs(model, tokenizer, texts, batch_size=32):
    model.eval()
    outs = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        outs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    labels, entries = load_faq()
    kb, train, test = build_dataset(entries)
    label2id = {lab: i for i, lab in enumerate(labels)}

    train_texts = [x[0] for x in train]
    train_labels = [label2id[x[1]] for x in train]

    # 用索引切分，避免重复文本导致的错误
    idx = np.arange(len(train_texts))
    tr_idx, dev_idx = train_test_split(idx, test_size=0.1, random_state=42, stratify=train_labels)
    tr_texts = [train_texts[i] for i in tr_idx]
    tr_y = [train_labels[i] for i in tr_idx]
    dev_texts = [train_texts[i] for i in dev_idx]
    dev_y = [train_labels[i] for i in dev_idx]

    test_texts = [x[0] for x in test]
    test_y = [label2id[x[1]] for x in test]
    test_labels = [x[1] for x in test]
    test_ids = [x[2] for x in test]
    kb_texts = [x["question"] for x in kb]
    kb_labels = np.array([label2id[x["label"]] for x in kb])

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR, num_labels=len(labels)).to(DEVICE)

    train_ds = FaqDataset(tr_texts, tr_y, tokenizer)
    dev_ds = FaqDataset(dev_texts, dev_y, tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "bert_ckpt"),
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
                      compute_metrics=compute_metrics)
    trainer.train()

    model.save_pretrained(MODEL_DIR / "bert_finetuned")
    tokenizer.save_pretrained(MODEL_DIR / "bert_finetuned")

    # ---- 测试集分类 ----
    pred = trainer.predict(FaqDataset(test_texts, test_y, tokenizer))
    test_probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    test_pred = test_probs.argmax(1)
    cls = classification_metrics(test_y, test_pred.tolist(), list(range(len(labels))), labels)
    print("BERT 意图分类：acc=%.4f macro_f1=%.4f" % (cls["accuracy"], cls["macro_f1"]))

    # ---- 测试集检索 ----
    kb_vec = encode_texts(model, tokenizer, kb_texts)
    q_vec = encode_texts(model, tokenizer, test_texts)
    sim = cosine_scores(q_vec, kb_vec)
    ret = retrieval_metrics(sim, test_labels, test_ids, kb)
    print("BERT 检索：", ret)

    # ---- 融合：检索分数 + 意图概率 ----
    dev_vec = encode_texts(model, tokenizer, dev_texts)
    dev_probs = predict_probs(model, tokenizer, dev_texts)
    dev_sim = cosine_scores(dev_vec, kb_vec)
    dev_labels = [labels[y] for y in dev_y]
    dev_ids = [None] * len(dev_texts)
    # 验证集问法可能对应多个条目，无法做 entry 级 MRR；用意图级 R@1 选 alpha
    best_alpha, best_score = 0.0, -1
    fusion_curve = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        fused = dev_sim + alpha * dev_probs[:, kb_labels]
        order = fused.argmax(1)
        kb_lab = [labels[i] for i in kb_labels]
        acc = float(np.mean([kb_lab[o] == lab for o, lab in zip(order, dev_labels)]))
        fusion_curve[alpha] = acc
        if acc > best_score:
            best_score, best_alpha = acc, alpha
    print("融合系数验证集扫描：", fusion_curve, "最佳 alpha =", best_alpha)

    fused_test = sim + best_alpha * test_probs[:, kb_labels]
    fused_ret = retrieval_metrics(fused_test, test_labels, test_ids, kb)
    print("融合检索：", fused_ret)

    np.savez(MODEL_DIR / "bert_retrieval.npz", kb_texts=np.array(kb_texts), kb_vectors=kb_vec, labels=np.array(labels))
    np.save(RESULT_DIR / "bert_test_probs.npy", test_probs)

    result = {
        "classification": cls,
        "retrieval": ret,
        "fusion": {"alpha": best_alpha, "dev_curve": fusion_curve, "retrieval": fused_ret},
        "train_log": trainer.state.log_history,
    }
    save_json(result, RESULT_DIR / "bert.json")
    print("已保存 outputs/results/bert.json")


if __name__ == "__main__":
    main()
