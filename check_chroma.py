from rag import get_vectorstore
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

vectorstore = get_vectorstore()
question = "魔渊为什么要盗走玄天镜"

print("--- Querying vectorstore directly ---")
results = vectorstore.similarity_search_with_relevance_scores(question, k=6)
for i, (doc, score) in enumerate(results):
    print(f"Doc {i+1}:")
    print(f"  Score: {score}")
    print(f"  Source: {os.path.basename(doc.metadata.get('source', 'unknown'))}")
    print(f"  Content: {repr(doc.page_content[:150])}")
