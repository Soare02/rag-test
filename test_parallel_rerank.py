import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

class TestReranker:
    def __init__(self, model):
        self.model = model
        self.base_url = "http://localhost:1234/v1"

    def _score_single(self, query, doc_content):
        url = f"{self.base_url}/responses"
        input_data = [
            {"role": "system", "content": "You are a document relevance evaluator. Answer ONLY with 'yes' or 'no'. Do not write anything else."},
            {"role": "user", "content": f"Query: {query}\nDocument: {doc_content}\nRelevant (yes/no):"}
        ]
        
        payload = {
            "model": self.model,
            "input": input_data,
            "include": ["message.output_text.logprobs"],
            "top_logprobs": 10,
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code != 200:
                return 0.0
            
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
            
            if not logprobs:
                if "yes" in full_text.lower():
                    return 10.0
                return 0.0
                
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
                            return round((p_yes / (p_yes + p_no)) * 10.0, 2)
                    return 10.0 if token_str.lower() == "yes" else 0.0
            return 0.0
        except Exception as e:
            print(f"Exception: {e}")
            return 0.0

    def rerank_with_scores(self, query, documents):
        scored = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                (i, doc, executor.submit(self._score_single, query, doc))
                for i, doc in enumerate(documents)
            ]
            for i, doc, future in futures:
                score = future.result()
                scored.append((i, score, doc))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

query = "魔渊为什么要盗走玄天镜"
docs = [
    "三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。",
    "玄天镜是仙界的至宝，通体由玄天玉打造而成，能够照出世间一切虚妄 and 妖魔的本源。",
    "魔渊当年盗取玄天镜，并非为了力量，而是为了复活道侣，但由于天道反噬最终失败了。",
    "修真界共有四大门派：玄天宗、百花谷、御剑山庄、天机阁。玄天宗以剑法闻名。",
    "东海之滨的村民们过着日出而作、日落而息的平静生活，很少有外人来到这里打扰。"
]

print("Starting Parallel Reranking Test (timeout=30)...")
start_time = time.time()
reranker = TestReranker("qwen3-reranker-0.6b")
results = reranker.rerank_with_scores(query, docs)
end_time = time.time()

print(f"\nReranking completed in {end_time - start_time:.2f} seconds.")
print("\nResults:")
for rank, (idx, score, doc) in enumerate(results):
    print(f"Rank {rank+1} (Original index {idx}): Score {score} - {doc}")
