# -*- coding: utf-8 -*-
"""课程方法基线：
1) TF-IDF + 余弦相似度检索
2) PPMI + SVD 词向量（句子向量取平均）
3) Skip-gram 词向量（句子向量取平均）
4) 词向量平均特征 + MLP 意图分类
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd

sys.path.append(str(Path(__file__).resolve().parent))
from common import (FIG_DIR, MODEL_DIR, RESULT_DIR, build_dataset, classification_metrics,
                    data_stats, load_faq, retrieval_metrics, save_json, set_seed, setup_matplotlib,
                    tokenize)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 3
DIM = 100


# ---------------- 词向量 ----------------
def build_cooccurrence(token_lists, window=WINDOW):
    vocab, corpus = {}, []
    for toks in token_lists:
        ids = []
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
            ids.append(vocab[t])
        corpus.append(ids)
    V = len(vocab)
    C = np.zeros((V, V), dtype=np.float32)
    for ids in corpus:
        for i, wid in enumerate(ids):
            for d in range(1, window + 1):
                if i - d >= 0:
                    C[wid, ids[i - d]] += 1
                if i + d < len(ids):
                    C[wid, ids[i + d]] += 1
    return C, vocab


def ppmi_matrix(C, eps=1e-8):
    N = C.sum()
    S = C.sum(axis=0)
    pmi = np.log2(C * N / (S[:, None] * S[None, :] + eps) + eps)
    return np.maximum(pmi, 0).astype(np.float32)


def train_ppmi_svd(token_lists, dim=DIM):
    C, vocab = build_cooccurrence(token_lists)
    W = ppmi_matrix(C)
    n_comp = min(dim, W.shape[0] - 1)
    U, S, _ = randomized_svd(W, n_components=n_comp, n_iter=7, random_state=42)
    vecs = U[:, :n_comp].astype(np.float32)
    print(f"[PPMI+SVD] 词表 {len(vocab)}，共现矩阵 {C.shape}，词向量 {vecs.shape}")
    return vocab, vecs


def train_skipgram(token_lists, dim=DIM, epochs=12, window=2, batch_size=512, seed=42):
    """手写 Skip-gram（softmax 版本），与课程 Word2vec.py 的思路一致。"""
    torch.manual_seed(seed)
    vocab = {}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)

    pairs = []
    for toks in token_lists:
        ids = [vocab[t] for t in toks]
        for i, center in enumerate(ids):
            for d in range(1, window + 1):
                if i - d >= 0:
                    pairs.append((center, ids[i - d]))
                if i + d < len(ids):
                    pairs.append((center, ids[i + d]))
    pairs = np.array(pairs, dtype=np.int64)
    print(f"[Skip-gram] 词表 {V}，训练样本对 {len(pairs)}")

    model = nn.Sequential(nn.Embedding(V, dim), nn.Linear(dim, V, bias=False)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = len(pairs)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        total, cnt = 0.0, 0
        for start in range(0, n, batch_size):
            batch = pairs[perm[start:start + batch_size]]
            centers = torch.tensor(batch[:, 0], device=DEVICE)
            contexts = torch.tensor(batch[:, 1], device=DEVICE)
            logits = model(centers)
            loss = loss_fn(logits, contexts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            cnt += len(batch)
        print(f"  epoch {epoch:2d}  loss={total / cnt:.4f}")

    vecs = model[0].weight.detach().cpu().numpy().astype(np.float32)
    return vocab, vecs


def sentence_vectors(texts, vocab, vecs):
    out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
    for i, text in enumerate(texts):
        ids = [vocab[t] for t in tokenize(text) if t in vocab]
        if ids:
            out[i] = vecs[ids].mean(axis=0)
    return out


def cosine_scores(A, B, eps=1e-8):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A @ B.T


# ---------------- MLP 分类器 ----------------
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


def train_mlp(X_train, y_train, n_classes, name, epochs=120, batch_size=32, lr=1e-3, patience=25):
    # 训练集标准化，参数随模型一起保存，供 demo 使用
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    Xs = (X_train - mean) / std

    X_tr, X_dev, y_tr, y_dev = train_test_split(
        Xs, y_train, test_size=0.15, random_state=42, stratify=y_train)

    model = MLP(Xs.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    Xdev_t = torch.tensor(X_dev, dtype=torch.float32, device=DEVICE)
    ydev_t = torch.tensor(y_dev, dtype=torch.long, device=DEVICE)

    history = {"loss": [], "dev_acc": []}
    best_acc, best_state, best_epoch = 0.0, None, -1
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        total, cnt = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            logits = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            cnt += len(idx)
        model.eval()
        with torch.no_grad():
            dev_acc = (model(Xdev_t).argmax(1) == ydev_t).float().mean().item()
        history["loss"].append(total / cnt)
        history["dev_acc"].append(dev_acc)
        if dev_acc > best_acc:
            best_acc, best_epoch = dev_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
        if epoch % 10 == 0:
            print(f"  [{name}] epoch {epoch:3d} loss={total / cnt:.4f} dev_acc={dev_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"  [{name}] 最佳 epoch {best_epoch}，dev_acc={best_acc:.4f}")
    ckpt = {"state_dict": best_state, "mean": mean, "std": std,
            "in_dim": int(Xs.shape[1]), "n_classes": int(n_classes), "history": history}
    torch.save(ckpt, MODEL_DIR / f"{name}.pt")
    return model, mean, std, history


def predict_mlp(model, X, mean, std):
    model.eval()
    Xs = torch.tensor((X - mean) / std, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(Xs), dim=1).cpu().numpy()
    return probs


# ---------------- 主流程 ----------------
def main():
    labels, entries = load_faq()
    kb, train, test = build_dataset(entries)
    stats = data_stats(entries, train, test)
    print("数据集：", stats)

    train_texts = [x[0] for x in train]
    test_texts = [x[0] for x in test]
    label2id = {lab: i for i, lab in enumerate(labels)}
    y_train = np.array([label2id[x[1]] for x in train])
    y_test = np.array([label2id[x[1]] for x in test])
    test_labels = [x[1] for x in test]
    test_ids = [x[2] for x in test]
    kb_texts = [x["question"] for x in kb]

    # 用于训练词向量的语料：全部问题 + 答案
    corpus_tokens = [tokenize(t) for t in train_texts + [e["question"] for e in entries] + [e["answer"] for e in entries]]

    results = {"dataset": stats, "methods": {}}

    # ---- 1. TF-IDF ----
    print("\n[1] TF-IDF 检索")
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=1)
    X_kb = tfidf.fit_transform(kb_texts)
    X_q = tfidf.transform(test_texts)
    scores = (X_q @ X_kb.T).toarray()
    r = retrieval_metrics(scores, test_labels, test_ids, kb)
    print("  ", r)
    results["methods"]["tfidf"] = {"retrieval": r, "n_features": int(X_kb.shape[1])}

    # ---- 2. PPMI + SVD ----
    print("\n[2] PPMI + SVD")
    vocab, vecs = train_ppmi_svd(corpus_tokens)
    np.savez(MODEL_DIR / "ppmi_svd.npz", vocab=np.array(list(vocab.keys())), vectors=vecs)
    kb_vec = sentence_vectors(kb_texts, vocab, vecs)
    q_vec = sentence_vectors(test_texts, vocab, vecs)
    r = retrieval_metrics(cosine_scores(q_vec, kb_vec), test_labels, test_ids, kb)
    print("  ", r)
    results["methods"]["ppmi_svd"] = {"retrieval": r, "vocab_size": len(vocab), "dim": int(vecs.shape[1])}

    # 训练集/测试集句向量，供 MLP 使用
    Xtr_ppmi = sentence_vectors(train_texts, vocab, vecs)
    Xte_ppmi = sentence_vectors(test_texts, vocab, vecs)

    # ---- 3. Skip-gram ----
    print("\n[3] Skip-gram")
    sg_vocab, sg_vecs = train_skipgram(corpus_tokens)
    np.savez(MODEL_DIR / "skipgram.npz", vocab=np.array(list(sg_vocab.keys())), vectors=sg_vecs)
    kb_vec_sg = sentence_vectors(kb_texts, sg_vocab, sg_vecs)
    q_vec_sg = sentence_vectors(test_texts, sg_vocab, sg_vecs)
    r = retrieval_metrics(cosine_scores(q_vec_sg, kb_vec_sg), test_labels, test_ids, kb)
    print("  ", r)
    results["methods"]["skipgram"] = {"retrieval": r, "vocab_size": len(sg_vocab), "dim": int(sg_vecs.shape[1])}

    Xtr_sg = sentence_vectors(train_texts, sg_vocab, sg_vecs)
    Xte_sg = sentence_vectors(test_texts, sg_vocab, sg_vecs)

    # ---- 4. MLP 意图分类 ----
    for name, Xtr, Xte in [("mlp_ppmi", Xtr_ppmi, Xte_ppmi), ("mlp_skipgram", Xtr_sg, Xte_sg)]:
        print(f"\n[4] MLP 分类：{name}")
        model, mean, std, history = train_mlp(Xtr, y_train, len(labels), name)
        probs = predict_mlp(model, Xte, mean, std)
        pred = probs.argmax(1)
        m = classification_metrics(y_test.tolist(), pred.tolist(), list(range(len(labels))), labels)
        m["confidence"] = float(probs.max(1).mean())
        results["methods"][name] = {"classification": m, "train_history": history}
        print("  acc=%.4f macro_f1=%.4f" % (m["accuracy"], m["macro_f1"]))

    results["labels"] = labels
    save_json(results, RESULT_DIR / "baseline.json")
    print("\n已保存 outputs/results/baseline.json")


if __name__ == "__main__":
    main()
