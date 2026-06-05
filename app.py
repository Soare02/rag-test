import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# 导入 rag.py 中的配置与函数
from rag import (
    get_vectorstore,
    LMStudioReranker,
    llm,
    format_docs,
    prompt,
    RERANK_MODEL,
    RERANK_BASE_URL,
    DOCS_DIR
)
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# LlamaIndex integration (optional; import errors gracefully handled)
try:
    from llama_rag import (
        load_or_build_indices as load_llama_indices,
        hybrid_query_with_details,
        router_query_with_details,
        init_llama_settings,
    )
    LLAMA_INDEX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LlamaIndex packages not available: {e}")
    LLAMA_INDEX_AVAILABLE = False

app = FastAPI(title="RAG Web Dashboard")

# 确保文档上传目录存在
os.makedirs(DOCS_DIR, exist_ok=True)

# 确保静态文件目录存在并挂载
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str
    mode: str = "langchain"  # "langchain" | "llamaindex"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h3>index.html 尚未创建</h3>")
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100)
):
    """处理文档拖拽上传，进行切片并存入 Chroma 数据库（去重）。"""
    file_path = os.path.join(DOCS_DIR, file.filename)
    
    # 写入本地临时文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Backend: Received upload for '{file.filename}', method={chunk_method}, size={chunk_size}, overlap={chunk_overlap}")
        # 根据后缀名选择合适的加载器
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
            return {
                "status": "error",
                "message": f"不支持的文件格式: {ext}。仅支持 .txt, .md, .pdf, .docx 格式。"
            }
            
        docs = loader.load()
        
        if chunk_method == "fixed":
            # 使用 CharacterTextSplitter 按固定长度切片
            splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            # 默认使用 RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
        chunks = splitter.split_documents(docs)
        
        vectorstore = get_vectorstore()
        
        # 修复已存在数据去重逻辑：比对已存在的 source 字段
        metadatas = vectorstore.get()["metadatas"]
        existing_sources = set()
        if metadatas:
            for m in metadatas:
                if m and "source" in m:
                    existing_sources.add(os.path.normpath(m["source"]))
                    
        # 过滤出未导入过的文本块
        norm_file_path = os.path.normpath(file_path)
        new_chunks = [c for c in chunks if os.path.normpath(c.metadata.get("source", "")) not in existing_sources]
        
        if not new_chunks:
            return {
                "status": "success",
                "message": f"文件 '{file.filename}' 已经存在且已处理过，无需重复导入。",
                "chunks_count": len(chunks),
                "added_count": 0
            }
            
        vectorstore.add_documents(new_chunks)
        
        return {
            "status": "success",
            "message": f"文件 '{file.filename}' 处理完成，切片并增量入库成功！",
            "chunks_count": len(chunks),
            "added_count": len(new_chunks)
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return {
            "status": "error",
            "message": f"处理文件失败: {str(e)}"
        }

# ===== LangChain RAG Query (existing, unchanged logic) =====
async def query_langchain(question: str) -> dict:
    """Original LangChain RAG pipeline: Chroma recall + Qwen3-Reranker + DeepSeek."""
    try:
        vectorstore = get_vectorstore()

        # 1. 粗回阶段：从向量库检索 k=6 个候选文本块
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        raw_docs = base_retriever.invoke(question)

        recalled_chunks = []
        for i, doc in enumerate(raw_docs):
            recalled_chunks.append({
                "index": i + 1,
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "未知"))
            })

        # 2. 精排阶段：使用 Qwen3-Reranker 逐条打分
        reranker = LMStudioReranker(model=RERANK_MODEL, base_url=RERANK_BASE_URL, top_n=3)
        scored_results = reranker.rerank_with_scores(question, raw_docs)

        reranked_chunks = []
        top_docs = []

        for rank, (orig_idx, score, doc) in enumerate(scored_results):
            reranked_chunks.append({
                "rank": rank + 1,
                "original_index": orig_idx + 1,
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "未知")),
                "score": "N/A (服务异常，已自动降级)" if score == -1.0 else score
            })
            if rank < 3:
                top_docs.append(doc)

        # 3. DeepSeek 生成回答
        context = format_docs(top_docs)
        messages = prompt.format_messages(context=context, question=question)

        response = llm.invoke(messages)
        answer = response.content

        return {
            "status": "success",
            "answer": answer,
            "recalled_chunks": recalled_chunks,
            "reranked_chunks": reranked_chunks
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"LangChain 查询失败: {str(e)}"
        }


# ===== LlamaIndex RAG Query =====
async def query_llamaindex(question: str) -> dict:
    """LlamaIndex RAG pipeline: multi-route query (LLM selects index) + Reranker + DeepSeek."""
    if not LLAMA_INDEX_AVAILABLE:
        return {
            "status": "error",
            "message": "LlamaIndex 包未安装。请运行: uv add llama-index-core llama-index-embeddings-openai llama-index-llms-openai llama-index-readers-file llama-index-retrievers-bm25 llama-index-vector-stores-chroma"
        }

    try:
        init_llama_settings()
        vector_index, summary_index, keyword_index = load_llama_indices()

        # Multi-route query: LLM selects best index → retrieve → reranker → generate
        recalled_chunks, reranked_chunks, answer, selected_route = router_query_with_details(
            question, vector_index, summary_index, keyword_index, top_k=6
        )

        return {
            "status": "success",
            "answer": answer,
            "recalled_chunks": recalled_chunks,
            "reranked_chunks": reranked_chunks,
            "selected_route": selected_route,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"LlamaIndex 查询失败: {str(e)}"
        }


# ===== Unified Query Endpoint =====
@app.post("/query")
async def query_rag(request: QueryRequest):
    """Unified query endpoint. Routes to LangChain or LlamaIndex based on mode."""
    question = request.question
    if not question.strip():
        return {"status": "error", "message": "问题不能为空"}

    if request.mode == "llamaindex":
        return await query_llamaindex(question)
    return await query_langchain(question)

