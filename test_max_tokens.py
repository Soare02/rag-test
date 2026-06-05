import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import json
import math
import time

def test_score(query, doc_content, max_tokens):
    url = "http://localhost:1234/v1/responses"
    input_data = [
        {"role": "system", "content": "你是一个文档相关性判定助手。你必须只回答 'yes' 或 'no'，不要输出任何解释或前缀。"},
        {"role": "user", "content": f"用户问题：{query}\n参考文档：{doc_content}\n该文档是否相关，能回答该问题吗？请仅回答 'yes' 或 'no'。是否相关:"}
    ]
    
    payload = {
        "model": "qwen3-reranker-0.6b",
        "input": input_data,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 20,
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    
    start = time.time()
    resp = requests.post(url, json=payload, timeout=10)
    dur = time.time() - start
    
    if resp.status_code != 200:
        print(f"Error {resp.status_code}")
        return None, dur
        
    resp_data = resp.json()
    logprobs = None
    full_text = ""
    for out_item in resp_data.get("output", []):
        if out_item.get("type") == "message":
            for content_item in out_item.get("content", []):
                if content_item.get("type") == "output_text" and "logprobs" in content_item:
                    logprobs = content_item["logprobs"]
                    full_text = content_item.get("text", "")
                    break
                    
    score = 0.0
    if logprobs:
        yes_tokens = {"yes", "Yes", "YES"}
        no_tokens = {"no", "No", "NO"}
        for token_info in logprobs:
            token_str = token_info.get("token", "").strip()
            if token_str.lower() in {"yes", "no"}:
                top_logprobs = token_info.get("top_logprobs", [])
                yes_lp = -99.0
                no_lp = -99.0
                for tl in top_logprobs:
                    tl_token = tl.get("token", "").strip()
                    tl_val = float(tl.get("logprob", -99.0))
                    if tl_token in yes_tokens:
                        yes_lp = max(yes_lp, tl_val)
                    elif tl_token in no_tokens:
                        no_lp = max(no_lp, tl_val)
                        
                if yes_lp > -99.0 or no_lp > -99.0:
                    p_yes = math.exp(yes_lp) if yes_lp > -99.0 else 0.0
                    p_no = math.exp(no_lp) if no_lp > -99.0 else 0.0
                    if p_yes + p_no > 0:
                        score = round((p_yes / (p_yes + p_no)) * 10.0, 2)
                        break
    return score, dur, full_text

query = "魔渊为什么要盗走玄天镜"
doc = "三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。"

print("--- Testing with max_tokens = 10 ---")
score, dur, text = test_score(query, doc, 10)
print(f"Score: {score}, Time: {dur:.4f}s, Text: {repr(text)}")

print("\n--- Testing with max_tokens = 1 ---")
score, dur, text = test_score(query, doc, 1)
print(f"Score: {score}, Time: {dur:.4f}s, Text: {repr(text)}")
