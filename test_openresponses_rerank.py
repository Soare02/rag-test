import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import json
import math

def calculate_rerank_score(model_name, query, doc_content):
    url = "http://localhost:1234/v1/responses"
    
    # We can try structuring the input as a messages list, or a single prompt string
    # Let's test if messages list is supported:
    input_data = [
        {"role": "system", "content": "You are an assistant that judges document relevance. You must respond ONLY with the word 'Yes' or the word 'No'. Do not output any other text or reasoning."},
        {"role": "user", "content": f"Query: {query}\nDocument: {doc_content}\nIs the document relevant and does it answer the query? Answer Yes or No:"}
    ]
    
    payload = {
        "model": model_name,
        "input": input_data,
        "include": [
            "message.output_text.logprobs"
        ],
        "top_logprobs": 10,
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"Error status code: {resp.status_code}, body: {resp.text}")
            return None
        
        resp_data = resp.json()
        output = resp_data.get("output", [])
        if not output:
            return None
            
        content_items = output[0].get("content", [])
        if not content_items:
            return None
            
        logprobs_list = content_items[0].get("logprobs", [])
        full_text = content_items[0].get("text", "")
        
        print(f"[{model_name}] Full text generated: {repr(full_text)}")
        
        # Print tokens and their top logprobs for debugging
        for idx, token_info in enumerate(logprobs_list[:5]):
            t = token_info.get("token")
            lp = token_info.get("logprob")
            print(f"  Token {idx}: {repr(t)} (logprob={lp})")
            top_lp = token_info.get("top_logprobs", [])
            for tl in top_lp[:3]:
                print(f"    - {repr(tl.get('token'))}: {tl.get('logprob')}")
                
        # Find first Yes/No token
        yes_tokens = {"yes", "Yes", "YES", "是"}
        no_tokens = {"no", "No", "NO", "否"}
        
        for token_info in logprobs_list:
            token_str = token_info.get("token", "").strip()
            if token_str in yes_tokens or token_str in no_tokens:
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
                        return p_yes / (p_yes + p_no)
                return 1.0 if token_str in yes_tokens else 0.0
                
        # Fallback if no yes/no token was found
        for yes_t in yes_tokens:
            if yes_t.lower() in full_text.lower():
                return 1.0
        return 0.0
        
    except Exception as e:
        print(f"[{model_name}] Exception: {e}")
        return None

query = "魔渊为什么要盗走玄天镜"
doc1 = "三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。"
doc2 = "东海之滨的村民们过着平静的生活，很少有外人来到这里打扰。"

for model in ["qwen3-reranker-0.6b", "google/gemma-4-e4b"]:
    print(f"\n================= Testing model: {model} =================")
    print("--- Doc 1 (Relevant) ---")
    score1 = calculate_rerank_score(model, query, doc1)
    print(f"Doc 1 Score: {score1}")
    
    print("--- Doc 2 (Irrelevant) ---")
    score2 = calculate_rerank_score(model, query, doc2)
    print(f"Doc 2 Score: {score2}")
