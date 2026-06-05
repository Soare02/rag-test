import sys
sys.stdout.reconfigure(encoding='utf-8')

from rag import LMStudioReranker, RERANK_MODEL, RERANK_BASE_URL
from langchain_core.documents import Document
import time

query = "魔渊为什么要盗走玄天镜"
docs = [
    Document(page_content="三百年前，魔渊盗走玄天镜是为了复活他的至爱道侣，这面镜子拥有逆转阴阳的神力。", metadata={"source": "doc1.txt"}),
    Document(page_content="玄天镜是仙界的至宝，通体由玄天玉打造而成，能够照出世间一切虚妄 and 妖魔的本源。", metadata={"source": "doc2.txt"}),
    Document(page_content="魔渊当年盗取玄天镜，并非为了力量，而是为了复活道侣，但由于天道反噬最终失败了。", metadata={"source": "doc3.txt"}),
    Document(page_content="修真界共有四大门派：玄天宗、百花谷、御剑山庄、天机阁。玄天宗以剑法闻名。", metadata={"source": "doc4.txt"}),
    Document(page_content="东海之滨的村民们过着日出而作、日落而息的平静生活，很少有外人来到这里打扰。", metadata={"source": "doc5.txt"}),
]

print(f"Loaded reranker model: {RERANK_MODEL} from {RERANK_BASE_URL}")
reranker = LMStudioReranker(model=RERANK_MODEL, base_url=RERANK_BASE_URL, top_n=3)

print("\n--- Running Reranker on 5 docs sequentially ---")
start = time.time()
scored_results = reranker.rerank_with_scores(query, docs)
dur = time.time() - start

print(f"Completed in {dur:.4f} seconds.")
for rank, (orig_idx, score, doc) in enumerate(scored_results):
    print(f"Rank {rank+1}: Original #{orig_idx+1} | Score: {score} | Content: {doc.page_content[:50]}...")
