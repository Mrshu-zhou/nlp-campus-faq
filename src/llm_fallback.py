# -*- coding: utf-8 -*-
"""可选的大模型兜底模块。

设计原则：本地模型（PPMI+SVD / BERT）负责核心的分类与检索，
只有在置信度低于阈值时才调用 LLM，且 LLM 必须基于检索到的知识库内容作答。
未配置 API Key 时返回 None，不影响演示。
"""
import os

import requests

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_BASE = "https://api.openai.com/v1"


def llm_config():
    """读取环境变量，返回 (api_key, base_url, model) 或 None。"""
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        return None
    if os.getenv("LLM_BASE_URL"):
        base = os.getenv("LLM_BASE_URL")
    elif os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base = DASHSCOPE_BASE
    else:
        base = OPENAI_BASE
    model = os.getenv("LLM_MODEL") or ("qwen-plus" if "dashscope" in base else "gpt-4o-mini")
    return key, base.rstrip("/"), model


def build_prompt(query, contexts):
    lines = ["你是校园问答助手。请仅依据下面给出的知识库内容回答用户问题。",
             "如果知识库中没有足够信息，请回答“知识库中暂未找到相关答案，建议咨询学校相关部门”，不要编造。", "",
             "【知识库】"]
    for i, c in enumerate(contexts, 1):
        lines.append(f"{i}. 问：{c['question']}\n   答：{c['answer']}")
    lines += ["", f"【用户问题】{query}", "【回答】"]
    return "\n".join(lines)


def llm_answer(query, contexts, timeout=30):
    """返回 (answer_text, error_text)。未配置或失败时 answer_text 为 None。"""
    cfg = llm_config()
    if cfg is None:
        return None, "未配置 LLM_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY"
    key, base, model = cfg

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨的校园问答助手，只根据给定资料回答，不编造。"},
            {"role": "user", "content": build_prompt(query, contexts)},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    try:
        resp = requests.post(f"{base}/chat/completions", json=payload,
                             headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip(), None
    except Exception as e:  # 网络/额度/格式等问题都不应让页面崩溃
        return None, f"LLM 调用失败：{type(e).__name__}: {e}"
