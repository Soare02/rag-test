import requests
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

url = "http://127.0.0.1:8000/query"
payload = {
    "question": "魔渊为什么要盗走玄天镜？"
}

print(f"Sending query to {url}...")
try:
    resp = requests.post(url, json=payload, timeout=40)
    print("Status code:", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print("\nAnswer generated:")
        print(data.get("answer"))
        print("\nReranked Chunks:")
        for chunk in data.get("reranked_chunks", []):
            print(f"  Rank {chunk['rank']} | Original #{chunk['original_index']} | Score: {chunk['score']} | Source: {chunk['source']} | Content: {chunk['content'][:50]}...")
    else:
        print("Response body:", resp.text)
except Exception as e:
    print("Error querying API:", e)
