import requests
import json

payload = {
    "model": "qwen3-reranker-0.6b",
    "input": "问题：魔渊为什么要盗走玄天镜\n文档：三百年前，魔渊因贪图力量，盗走了玄天镜，企图统治三界。\n请问该文档是否能回答该问题？请仅回答 Yes 或 No。",
    "include": [
        "message.output_text.logprobs"
    ],
    "top_logprobs": 5
}

try:
    resp = requests.post("http://localhost:1234/v1/responses", json=payload)
    print("Status code:", resp.status_code)
    print("Response JSON:")
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print("Error querying responses:", e)
