import os
import sys
import math
from typing import Any, List, Optional, Generator, AsyncGenerator
from dotenv import load_dotenv
import chromadb
from openai import OpenAI as OpenAI_client, AsyncOpenAI
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    KeywordTableIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage, CompletionResponse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore

# Reuse the LMStudioReranker from rag.py for semantic reranking
from rag import LMStudioReranker, RERANK_MODEL, RERANK_BASE_URL

# Note: We don't use llama_index.llms.openai.OpenAI or
# llama_index.embeddings.openai.OpenAIEmbedding because they validate
# model names against an internal allowlist. Instead we use custom
# LMStudioEmbedding and DeepSeekLLM defined above.

load_dotenv()

# ===== Configuration =====
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"
EMBEDDING_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "deepseek-v4-flash"
LLM_API_BASE = "https://api.deepseek.com/v1"
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./documents"
LLAMA_STORAGE_DIR = "./llama_storage"
CHROMA_COLLECTION = "llamaindex"


# ===== 1. Custom Embedding (LM Studio via OpenAI-compatible API) =====
class LMStudioEmbedding(BaseEmbedding):
    """Custom embedding for LM Studio's non-OpenAI models.
    Bypasses model name validation in llama_index.embeddings.openai.
    """

    def __init__(
        self, model_name: str, api_base: str, api_key: str = "lm-studio", **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self._client = OpenAI_client(base_url=api_base, api_key=api_key)
        self._model = model_name

    @staticmethod
    def _clean(text: str) -> str:
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def _get_query_embedding(self, query: str) -> List[float]:
        response = self._client.embeddings.create(
            input=self._clean(query), model=self._model
        )
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            input=self._clean(text), model=self._model
        )
        return response.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        cleaned = [self._clean(t) for t in texts]
        response = self._client.embeddings.create(
            input=cleaned, model=self._model
        )
        return [item.embedding for item in response.data]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


# ===== 2. Custom LLM (DeepSeek via OpenAI-compatible API) =====
class DeepSeekLLM(LLM):
    """Custom LLM wrapper for DeepSeek via OpenAI-compatible API.
    Bypasses model name validation in llama_index.llms.openai.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._client = OpenAI_client(base_url=api_base, api_key=api_key)
        self._model_name = model
        self._temperature = temperature

    @property
    def metadata(self):
        from llama_index.core.base.llms.types import LLMMetadata

        return LLMMetadata(
            model_name=self._model_name,
            is_chat_model=True,
            context_window=128000,
        )

    def chat(self, messages: List[ChatMessage], **kwargs) -> "ChatResponse":
        from llama_index.core.base.llms.types import ChatResponse

        openai_messages = [
            {
                "role": m.role.value,
                "content": m.content.encode("utf-8", errors="replace").decode("utf-8")
                if m.content
                else "",
            }
            for m in messages
        ]
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
        )
        msg = response.choices[0].message
        return ChatResponse(
            message=ChatMessage(role=msg.role, content=msg.content),
        )

    def complete(
        self, prompt: str, **kwargs
    ) -> "CompletionResponse":
        from llama_index.core.base.llms.types import CompletionResponse

        safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": safe_prompt}],
            temperature=self._temperature,
        )
        return CompletionResponse(
            text=response.choices[0].message.content or "",
        )

    def stream_chat(
        self, messages: List[ChatMessage], **kwargs
    ) -> Generator["ChatResponse", None, None]:
        from llama_index.core.base.llms.types import ChatResponse

        openai_messages = [
            {
                "role": m.role.value,
                "content": m.content.encode("utf-8", errors="replace").decode("utf-8")
                if m.content
                else "",
            }
            for m in messages
        ]
        stream = self._client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=delta.content),
                    delta=delta.content,
                )

    def stream_complete(
        self, prompt: str, **kwargs
    ) -> Generator["CompletionResponse", None, None]:
        from llama_index.core.base.llms.types import CompletionResponse

        safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
        stream = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": safe_prompt}],
            temperature=self._temperature,
            stream=True,
        )
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_text += delta.content
                yield CompletionResponse(text=full_text, delta=delta.content)

    async def achat(
        self, messages: List[ChatMessage], **kwargs
    ) -> "ChatResponse":
        from llama_index.core.base.llms.types import ChatResponse

        async_client = AsyncOpenAI(
            base_url=self._client.base_url, api_key=self._client.api_key
        )
        openai_messages = [
            {
                "role": m.role.value,
                "content": m.content.encode("utf-8", errors="replace").decode("utf-8")
                if m.content
                else "",
            }
            for m in messages
        ]
        response = await async_client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
        )
        msg = response.choices[0].message
        return ChatResponse(
            message=ChatMessage(role=msg.role, content=msg.content),
        )

    async def acomplete(
        self, prompt: str, **kwargs
    ) -> "CompletionResponse":
        from llama_index.core.base.llms.types import CompletionResponse

        async_client = AsyncOpenAI(
            base_url=self._client.base_url, api_key=self._client.api_key
        )
        safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
        response = await async_client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": safe_prompt}],
            temperature=self._temperature,
        )
        return CompletionResponse(
            text=response.choices[0].message.content or "",
        )

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs
    ) -> AsyncGenerator["ChatResponse", None]:
        from llama_index.core.base.llms.types import ChatResponse

        async_client = AsyncOpenAI(
            base_url=self._client.base_url, api_key=self._client.api_key
        )
        openai_messages = [
            {
                "role": m.role.value,
                "content": m.content.encode("utf-8", errors="replace").decode("utf-8")
                if m.content
                else "",
            }
            for m in messages
        ]
        stream = await async_client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=delta.content),
                    delta=delta.content,
                )

    async def astream_complete(
        self, prompt: str, **kwargs
    ) -> AsyncGenerator["CompletionResponse", None]:
        from llama_index.core.base.llms.types import CompletionResponse

        async_client = AsyncOpenAI(
            base_url=self._client.base_url, api_key=self._client.api_key
        )
        safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
        stream = await async_client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": safe_prompt}],
            temperature=self._temperature,
            stream=True,
        )
        full_text = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_text += delta.content
                yield CompletionResponse(text=full_text, delta=delta.content)


# ===== 3. LlamaIndex Global Settings =====
def init_llama_settings():
    """Configure LlamaIndex global settings: embedding, LLM, and node parser."""
    Settings.embed_model = LMStudioEmbedding(
        model_name=EMBEDDING_MODEL,
        api_base=EMBEDDING_BASE_URL,
        api_key="lm-studio",
    )
    Settings.llm = DeepSeekLLM(
        model=LLM_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=LLM_API_BASE,
        temperature=0.7,
    )
    Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)


# ===== 3. Chroma Vector Store (separate collection from LangChain) =====
def get_chroma_vector_store():
    """Get LlamaIndex-compatible ChromaVectorStore in its own collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    return ChromaVectorStore(chroma_collection=collection)


# ===== 4. Document Loading =====
def load_documents(doc_dir=DOCS_DIR):
    """Load documents using LlamaIndex's SimpleDirectoryReader."""
    if not os.path.exists(doc_dir):
        return []
    reader = SimpleDirectoryReader(
        input_dir=doc_dir,
        recursive=True,
        required_exts=[".txt", ".md", ".pdf", ".docx"],
        filename_as_id=True,
    )
    return reader.load_data()


# ===== 5. Build All Three Indices =====
def build_indices(documents):
    """Build VectorStoreIndex, SummaryIndex, and KeywordTableIndex, then persist."""
    os.makedirs(LLAMA_STORAGE_DIR, exist_ok=True)

    # --- VectorStoreIndex (Chroma-backed) ---
    vector_dir = os.path.join(LLAMA_STORAGE_DIR, "vector")
    vector_store = get_chroma_vector_store()
    vector_storage = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=vector_storage, show_progress=True
    )
    os.makedirs(vector_dir, exist_ok=True)
    vector_index.storage_context.persist(persist_dir=vector_dir)

    # --- SummaryIndex (docstore-backed, persisted to disk) ---
    summary_dir = os.path.join(LLAMA_STORAGE_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_storage = StorageContext.from_defaults()
    summary_index = SummaryIndex.from_documents(
        documents, storage_context=summary_storage, show_progress=True
    )
    summary_index.storage_context.persist(persist_dir=summary_dir)

    # --- KeywordTableIndex (docstore-backed, persisted to disk) ---
    keyword_dir = os.path.join(LLAMA_STORAGE_DIR, "keyword")
    os.makedirs(keyword_dir, exist_ok=True)
    keyword_storage = StorageContext.from_defaults()
    keyword_index = KeywordTableIndex.from_documents(
        documents, storage_context=keyword_storage, show_progress=True
    )
    keyword_index.storage_context.persist(persist_dir=keyword_dir)

    return vector_index, summary_index, keyword_index


# ===== 6. Load or Build Indices =====
def load_or_build_indices():
    """Load persisted indices from disk, or build fresh from documents."""
    init_llama_settings()

    vector_dir = os.path.join(LLAMA_STORAGE_DIR, "vector")
    summary_dir = os.path.join(LLAMA_STORAGE_DIR, "summary")
    keyword_dir = os.path.join(LLAMA_STORAGE_DIR, "keyword")

    def index_exists(path):
        return os.path.exists(os.path.join(path, "docstore.json"))

    all_exist = all(index_exists(d) for d in [vector_dir, summary_dir, keyword_dir])

    if all_exist:
        print("加载已有 LlamaIndex ...")
        vector_store = get_chroma_vector_store()
        vector_storage = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=vector_dir
        )
        vector_index = load_index_from_storage(vector_storage)

        summary_storage = StorageContext.from_defaults(persist_dir=summary_dir)
        summary_index = load_index_from_storage(summary_storage)

        keyword_storage = StorageContext.from_defaults(persist_dir=keyword_dir)
        keyword_index = load_index_from_storage(keyword_storage)
    else:
        print("加载文档并创建 LlamaIndex ...")
        documents = load_documents()
        if not documents:
            raise FileNotFoundError(
                f"在 {DOCS_DIR} 目录下未找到文档且无持久化索引。"
                "请先运行 ingest 或向目录中添加文档。"
            )
        vector_index, summary_index, keyword_index = build_indices(documents)

    return vector_index, summary_index, keyword_index


# ===== 7. Hybrid Retriever (Dense + BM25 with RRF Fusion) =====
def get_hybrid_retriever(vector_index, top_k=6):
    """Build hybrid retriever with dense vector + BM25 sparse search and RRF fusion.

    Falls back to vector-only retrieval if BM25 is not installed.
    """
    vector_retriever = VectorIndexRetriever(
        index=vector_index, similarity_top_k=top_k
    )

    # Graceful BM25 fallback (Chinese docs may cause bm25s tokenizer to fail)
    bm25_available = False
    bm25_retriever = None
    try:
        from llama_index.retrievers.bm25 import BM25Retriever

        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index, similarity_top_k=top_k
        )
        bm25_available = True
    except Exception as e:
        print(
            f"BM25 not available ({e}). "
            "Falling back to vector-only retrieval."
        )

    class HybridRetriever(BaseRetriever):
        """Custom retriever combining dense vector and BM25 sparse results via RRF."""

        def __init__(self, vector_ret, bm25_ret, tk, use_bm25):
            super().__init__()
            self._vector_retriever = vector_ret
            self._bm25_retriever = bm25_ret
            self._top_k = tk
            self._use_bm25 = use_bm25
            self._rrf_k = 60

        def _retrieve(self, query):
            vector_results = self._vector_retriever.retrieve(query)

            if not self._use_bm25 or self._bm25_retriever is None:
                return vector_results[: self._top_k]

            bm25_results = self._bm25_retriever.retrieve(query)

            # Reciprocal Rank Fusion scoring
            scores = {}
            for rank, node in enumerate(vector_results):
                nid = node.node.node_id
                scores[nid] = scores.get(nid, 0) + 1.0 / (self._rrf_k + rank + 1)
            for rank, node in enumerate(bm25_results):
                nid = node.node.node_id
                scores[nid] = scores.get(nid, 0) + 1.0 / (self._rrf_k + rank + 1)

            # Deduplicate and assign RRF scores
            node_map = {}
            for n in vector_results:
                n.score = scores.get(n.node.node_id, 0.0)
                node_map[n.node.node_id] = n
            for n in bm25_results:
                n.score = scores.get(n.node.node_id, 0.0)
                node_map[n.node.node_id] = n

            sorted_nodes = sorted(
                node_map.values(), key=lambda n: n.score, reverse=True
            )[: self._top_k]
            return sorted_nodes

    return HybridRetriever(vector_retriever, bm25_retriever, top_k, bm25_available)


# ===== 8. RouterQueryEngine =====
def create_router_query_engine(vector_index, summary_index, keyword_index):
    """Create a RouterQueryEngine with 3 tools for intent-based routing."""
    vector_tool = QueryEngineTool(
        query_engine=vector_index.as_query_engine(similarity_top_k=6),
        metadata=ToolMetadata(
            name="vector_tool",
            description="用于语义搜索和事实性问答，基于含义和上下文相似度。",
        ),
    )
    summary_tool = QueryEngineTool(
        query_engine=summary_index.as_query_engine(),
        metadata=ToolMetadata(
            name="summary_tool",
            description="用于摘要、概括和跨文档信息综合。",
        ),
    )
    keyword_tool = QueryEngineTool(
        query_engine=keyword_index.as_query_engine(),
        metadata=ToolMetadata(
            name="keyword_tool",
            description="用于精确关键词匹配，按特定命名实体、术语或专有名词查找文档。",
        ),
    )
    return RouterQueryEngine.from_defaults(
        [vector_tool, summary_tool, keyword_tool],
        llm=Settings.llm,
        verbose=True,
        select_multi=False,
    )


# ===== 9. Hybrid Query for API Mode =====
def hybrid_query_with_details(question, vector_index, top_k=6):
    """Run hybrid retrieval + reranker and return (recalled_chunks_list, answer_text).

    Performs:
      1. Hybrid retrieval (dense + BM25, RRF fusion) → Top N
      2. Qwen3-Reranker semantic rescoring → Top N sorted
      3. Generation from top 3 documents

    Used by app.py in 'llamaindex' mode.
    """
    retriever = get_hybrid_retriever(vector_index, top_k=top_k)
    nodes = retriever.retrieve(question)

    recalled_chunks = []
    for i, node in enumerate(nodes):
        recalled_chunks.append({
            "index": i + 1,
            "content": node.node.text,
            "source": node.node.metadata.get("file_name", "unknown"),
            "score": node.score,
        })

    # Rerank: pass RRF-fused results through Qwen3-Reranker for semantic scoring
    reranker = LMStudioReranker(model=RERANK_MODEL, base_url=RERANK_BASE_URL, top_n=3)
    # Convert nodes to LangChain-compatible doc objects for the reranker
    class _DummyDoc:
        def __init__(self, text):
            self.page_content = text
    dummy_docs = [_DummyDoc(n.node.text) for n in nodes]
    scored = reranker.rerank_with_scores(question, dummy_docs)

    reranked_chunks = []
    top_docs_text = []
    for rank, (orig_idx, score, doc) in enumerate(scored):
        reranked_chunks.append({
            "rank": rank + 1,
            "original_index": orig_idx + 1,
            "content": doc.page_content,
            "source": recalled_chunks[orig_idx]["source"] if orig_idx < len(recalled_chunks) else "unknown",
            "score": "N/A (服务异常，已自动降级)" if score == -1.0 else score,
        })
        if rank < 3:
            top_docs_text.append(doc.page_content)

    # Build query engine using the reranked top-3 docs directly
    from llama_index.core.schema import TextNode
    top_nodes = [TextNode(text=t) for t in top_docs_text]
    response = Settings.llm.complete(
        f"根据以下上下文回答问题。\n\n上下文：\n{' '.join(top_docs_text)}\n\n问题：{question}\n\n请用中文简洁地回答。"
    )

    return recalled_chunks, reranked_chunks, str(response)


# ===== 9b. Router Query for API Mode (Multi-Route) =====
def router_query_with_details(question, vector_index, summary_index, keyword_index, top_k=6):
    """Route query to best index via LLMSingleSelector, then retrieve + rerank + generate.

    Returns (recalled_chunks, reranked_chunks, answer, selected_route_name).
    Used by app.py in 'llamaindex' mode to showcase multi-query routing.
    """
    from llama_index.core.selectors import LLMSingleSelector

    # 1. Route selection: LLM picks the best index based on query intent
    choices = [
        ("vector_tool", "用于语义搜索和事实性问答，基于含义和上下文相似度。"),
        ("summary_tool", "用于摘要、概括和跨文档信息综合。"),
        ("keyword_tool", "用于精确关键词匹配，按特定命名实体、术语或专有名词查找文档。"),
    ]
    selector = LLMSingleSelector.from_defaults(llm=Settings.llm)
    result = selector.select(
        choices=[desc for _, desc in choices],
        query=question,
    )
    selected_idx = result.selections[0].index
    selected_name = choices[selected_idx][0]

    # 2. Retrieve from the selected index
    indices = [vector_index, summary_index, keyword_index]
    index = indices[selected_idx]

    if selected_idx == 0:  # vector
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    elif selected_idx == 1:  # summary
        retriever = index.as_retriever()
    else:  # keyword
        retriever = index.as_retriever()

    nodes = retriever.retrieve(question)[:top_k]

    # 3. Build recalled_chunks (raw retrieval results)
    recalled_chunks = []
    for i, node in enumerate(nodes):
        recalled_chunks.append({
            "index": i + 1,
            "content": node.node.text,
            "source": node.node.metadata.get("file_name", "unknown"),
            "score": node.score,
        })

    # 4. Reranker: Qwen3-Reranker semantic rescoring
    reranker = LMStudioReranker(model=RERANK_MODEL, base_url=RERANK_BASE_URL, top_n=3)
    class _DummyDoc:
        def __init__(self, text):
            self.page_content = text
    dummy_docs = [_DummyDoc(n.node.text) for n in nodes]
    scored = reranker.rerank_with_scores(question, dummy_docs)

    reranked_chunks = []
    top_docs_text = []
    for rank, (orig_idx, score, doc) in enumerate(scored):
        reranked_chunks.append({
            "rank": rank + 1,
            "original_index": orig_idx + 1,
            "content": doc.page_content,
            "source": recalled_chunks[orig_idx]["source"] if orig_idx < len(recalled_chunks) else "unknown",
            "score": "N/A (服务异常，已自动降级)" if score == -1.0 else score,
        })
        if rank < 3:
            top_docs_text.append(doc.page_content)

    # 5. DeepSeek generation from top 3
    response = Settings.llm.complete(
        f"根据以下上下文回答问题。\n\n上下文：\n{' '.join(top_docs_text)}\n\n问题：{question}\n\n请用中文简洁地回答。"
    )

    return recalled_chunks, reranked_chunks, str(response), selected_name


# ===== 10. Ingest: Incremental Document Addition =====
def ingest():
    """Load documents from DOCS_DIR and add to existing indices (or build new)."""
    init_llama_settings()
    documents = load_documents()
    if not documents:
        print(f"在 {DOCS_DIR} 中未找到文档。")
        return

    vector_dir = os.path.join(LLAMA_STORAGE_DIR, "vector")
    if os.path.exists(os.path.join(vector_dir, "docstore.json")):
        # Load existing indices and insert new documents
        vi, si, ki = load_or_build_indices()
        for doc in documents:
            vi.insert(doc)
            si.insert(doc)
            ki.insert(doc)
        # Persist all three
        vi.storage_context.persist(persist_dir=vector_dir)
        si.storage_context.persist(persist_dir=os.path.join(LLAMA_STORAGE_DIR, "summary"))
        ki.storage_context.persist(persist_dir=os.path.join(LLAMA_STORAGE_DIR, "keyword"))
        print(f"已向已有索引中添加 {len(documents)} 个文档。")
    else:
        print(f"从 {len(documents)} 个文档构建索引...")
        build_indices(documents)
        print("索引构建完成。")


# ===== 11. CLI Entry Point =====
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        vi, si, ki = load_or_build_indices()
        engine = create_router_query_engine(vi, si, ki)
        print("LlamaIndex RAG 已就绪（RouterQueryEngine 启用）。输入 'quit' 退出。\n")
        while True:
            q = input("Query: ").strip()
            if q.lower() == "quit":
                break
            r = engine.query(q)
            print(f"Answer: {r}\n")
    else:
        print("用法: python llama_rag.py [ingest|query]")
