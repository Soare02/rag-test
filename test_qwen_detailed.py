import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import time

url = "http://localhost:1234/v1/responses"
payload1 = {
    "model": "qwen3-reranker-0.6b",
    "input": [
        {"role": "system", "content": "You are a helpful assistant. You must output ONLY 'Yes' or 'No'."},
        {"role": "user", "content": "问题：魔渊为什么要盗走玄天镜\n文档：三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。\n请问该文档是否能回答该问题？请仅回答 Yes 或 No。"}
    ],
    "include": ["message.output_text.logprobs"],
    "top_logprobs": 5,
    "temperature": 0.0,
    "max_tokens": 10
}

payload2 = {
    "model": "qwen3-reranker-0.6b",
    "input": [
        {"role": "system", "content": "You are a helpful assistant. You must output ONLY 'Yes' or 'No'."},
        {"role": "user", "content": "问题：魔渊为什么要盗走玄天镜\n文档：东海之滨的村民们过着平静的生活，很少有外人来到这里打扰。\n请问该文档是否能回答该问题？请仅回答 Yes 或 No。"}
    ],
    "include": ["message.output_text.logprobs"],
    "top_logprobs": 5,
    "temperature": 0.0,
    "max_tokens": 10
}

print("Querying Document 1 (Relevant)...")
start = time.time()
try:
    resp = requests.post(url, json=payload1, timeout=60)
    print(f"Doc 1 finished in {time.time() - start:.2f}s, status: {resp.status_code}")
    print(f"Doc 1 Text: {repr(resp.json()['output'][0]['content'][0]['text'])}")
except Exception as e:
    print(f"Doc 1 failed: {e}")

print("\nQuerying Document 2 (Irrelevant)...")
start = time.time()
try:
    resp = requests.post(url, json=payload2, timeout=60)
    print(f"Doc 2 finished in {time.time() - start:.2f}s, status: {resp.status_code}")
    print(f"Doc 2 Text: {repr(resp.json()['output'][0]['content'][0]['text'])}")
except Exception as e:
    print(f"Doc 2 failed: {e}")
