import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import json
import math
import time

def test_score(model_name, query, doc_content):
    url = "http://localhost:1234/v1/responses"
    input_data = [
        {"role": "system", "content": "You are a document relevance evaluator. Answer ONLY with 'yes' or 'no'. Do not write anything else."},
        {"role": "user", "content": f"Query: {query}\nDocument: {doc_content}\nRelevant (yes/no):"}
    ]
    
    payload = {
        "model": model_name,
        "input": input_data,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 20,
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    start = time.time()
    resp = requests.post(url, json=payload, timeout=15)
    dur = time.time() - start
    
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return None, dur, ""
        
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
        for idx, token_info in enumerate(logprobs):
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
doc1 = "三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。"
doc2 = "东海之滨的村民们过着平静的生活，很少有外人来到这里打扰。"

print("=== Gemma-4 ===")
for doc in [doc1, doc2]:
    score, dur, text = test_score("google/gemma-4-e4b", query, doc)
    print(f"Score: {score}, Time: {dur:.4f}s, Text: {repr(text)}")
