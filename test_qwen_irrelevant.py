import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import json

url = "http://localhost:1234/v1/responses"
query = "魔渊为什么要盗走玄天镜"
doc = "东海之滨的村民们过着日出而作、日落而息的平静生活，很少有外人来到这里打扰。"

input_data = [
    {"role": "system", "content": "你是一个文档相关性判定助手。你必须只回答 'yes' 或 'no'，不要输出任何解释或前缀。"},
    {"role": "user", "content": f"用户问题：{query}\n参考文档：{doc}\n该文档是否相关，能回答该问题吗？请仅回答 'yes' 或 'no'。是否相关:"}
]

payload = {
    "model": "qwen3-reranker-0.6b",
    "input": input_data,
    "include": ["message.output_text.logprobs"],
    "top_logprobs": 10,
    "temperature": 0.0,
    "max_tokens": 50
}

try:
    resp = requests.post(url, json=payload, timeout=20)
    print("Status:", resp.status_code)
    resp_json = resp.json()
    print(json.dumps(resp_json, indent=2))
except Exception as e:
    print("Error:", e)
