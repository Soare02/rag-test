import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import json
import math

def get_logprobs_from_response(resp_data):
    # Search for type == "message" in output
    for out_item in resp_data.get("output", []):
        if out_item.get("type") == "message":
            for content_item in out_item.get("content", []):
                if content_item.get("type") == "output_text" and "logprobs" in content_item:
                    return content_item["logprobs"], content_item.get("text", "")
    return None, None

def calculate_score(logprobs, full_text):
    if not logprobs:
        return 0.0
        
    yes_tokens = {"yes", "Yes", "YES", "是"}
    no_tokens = {"no", "No", "NO", "否"}
    
    for idx, token_info in enumerate(logprobs):
        token_str = token_info.get("token", "").strip()
        if token_str in yes_tokens or token_str in no_tokens:
            print(f"  Found decision token '{token_str}' at index {idx}")
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
            
            print(f"  yes_lp: {yes_lp}, no_lp: {no_lp}")
            if yes_lp > -99.0 or no_lp > -99.0:
                p_yes = math.exp(yes_lp) if yes_lp > -99.0 else 0.0
                p_no = math.exp(no_lp) if no_lp > -99.0 else 0.0
                if p_yes + p_no > 0:
                    return p_yes / (p_yes + p_no)
            return 1.0 if token_str in yes_tokens else 0.0
            
    # Fallback to full text keyword matching
    for yes_t in yes_tokens:
        if yes_t.lower() in full_text.lower():
            return 1.0
    return 0.0

url = "http://localhost:1234/v1/responses"
query = "魔渊为什么要盗走玄天镜"
doc1 = "三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。"
doc2 = "东海之滨的村民们过着平静的生活，很少有外人来到这里打扰。"

# Prompt format
input_template = [
    {"role": "system", "content": "You are a document relevance evaluator. Answer ONLY with 'yes' or 'no'. Do not write anything else."},
    {"role": "user", "content": "Query: {query}\nDocument: {doc}\nRelevant (yes/no):"}
]

for doc_name, doc_text in [("Doc 1 (Relevant)", doc1), ("Doc 2 (Irrelevant)", doc2)]:
    print(f"\n================= {doc_name} =================")
    
    # Fill prompt
    user_content = input_template[1]["content"].format(query=query, doc=doc_text)
    input_data = [
        {"role": "system", "content": input_template[0]["content"]},
        {"role": "user", "content": user_content}
    ]
    
    payload = {
        "model": "qwen3-reranker-0.6b",
        "input": input_data,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 10,
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            logprobs, full_text = get_logprobs_from_response(resp.json())
            score = calculate_score(logprobs, full_text)
            print(f"Final Score: {score:.6f}")
        else:
            print(f"Failed with status: {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")
