import sys
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from openai import OpenAI

load_dotenv()

# ===== Configuration =====
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"
EMBEDDING_BASE_URL = "http://localhost:1234/v1"
RERANK_MODEL = "qwen3-reranker-0.6b"
RERANK_BASE_URL = "http://localhost:1234/v1"
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./documents"

# ===== 1. Custom Embeddings (lmstudio OpenAI-compatible endpoint) =====
class LMStudioEmbeddings(Embeddings):
    """直接调用 lmstudio embedding API，避免 langchain_openai 的兼容问题。"""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")

    # 分批参数：本地 LM Studio + Qwen3-Embedding-0.6B 的安全档。
    # 不分批时 22MB 文档（~5 万 chunk）一次性 POST 会让服务端 OOM/超时，整批废掉。
    EMBED_BATCH_SIZE = 32
    EMBED_DIM = 1024  # Qwen3-Embedding-0.6B 输出维度，失败时占位向量需要对齐
    EMBED_TIMEOUT = 60  # 单批超时（秒），避免 server 卡死无限挂起
    EMBED_MAX_RETRIES = 3  # 单批失败重试次数，指数退避 1/2/4s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批 + 重试 + 单批容错的批量向量化。

        - 把一次 N 条的请求切成 ceil(N/BATCH) 次小请求，避免显存爆 / server 超时。
        - 单批 3 次重试仍失败 → 用零向量占位（Chroma 要求 texts 与 vectors 等长），
          这些块永远不会被检索命中，等同软删除，不影响其他块正常入库。
        - 每 20 批打印一次进度，22MB 这种长任务可观测。
        """
        import time

        total = len(texts)
        if total == 0:
            return []

        out: list[list[float]] = []
        failed = 0

        for i in range(0, total, self.EMBED_BATCH_SIZE):
            batch = texts[i:i + self.EMBED_BATCH_SIZE]
            last_err = None
            for attempt in range(self.EMBED_MAX_RETRIES):
                try:
                    resp = self.client.embeddings.create(
                        input=batch, model=self.model, timeout=self.EMBED_TIMEOUT
                    )
                    out.extend(item.embedding for item in resp.data)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < self.EMBED_MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
            else:
                # 三次都失败 → 占位，保证维度对齐不让 Chroma 写入崩
                print(f"  Embedding 批次 {i // self.EMBED_BATCH_SIZE + 1} "
                      f"重试 {self.EMBED_MAX_RETRIES} 次仍失败: {last_err}")
                out.extend([[0.0] * self.EMBED_DIM] * len(batch))
                failed += len(batch)

            done = min(i + self.EMBED_BATCH_SIZE, total)
            batch_idx = i // self.EMBED_BATCH_SIZE
            if batch_idx % 20 == 0 or done >= total:
                print(f"  embed 进度 {done}/{total} (失败 {failed})")

        return out

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


embeddings = LMStudioEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_BASE_URL)


# ===== 1.5 Custom Reranker (通过 Open Responses /v1/responses 接口计算 logprobs 实现精排) =====
class LMStudioReranker:
    """
    通过 LM Studio 的 /v1/responses 接口，利用 Qwen3-Reranker 模型
    提取 logprobs 来计算每个候选文档与查询的相关性连续得分 (0.0~10.0)。
    由于 LM Studio 并发处理 logprobs 时存在状态混淆的问题，这里采用顺序查询（单线程）。
    """

    def __init__(self, model: str, base_url: str, top_n: int = 3):
        self.model = model
        self.base_url = base_url
        self.top_n = top_n

    def _score_single(self, query: str, doc_content: str) -> float:
        """对单个文档计算相关性分数 (0.0 到 10.0)"""
        import requests
        import math

        url = f"{self.base_url.rstrip('/')}/responses"
        input_data = [
            {"role": "system", "content": "你是一个文档相关性判定助手。你必须只回答 'yes' 或 'no'，不要输出任何解释或前缀。"},
            {"role": "user", "content": f"用户问题：{query}\n参考文档：{doc_content}\n该文档是否相关，能回答该问题吗？请仅回答 'yes' 或 'no'。是否相关:"}
        ]

        payload = {
            "model": self.model,
            "input": input_data,
            "include": ["message.output_text.logprobs"],
            "top_logprobs": 20,
            "temperature": 0.0,
            "max_tokens": 10  # 少量生成，只要捕获到判定词即可
        }

        try:
            # timeout 15s：正常块 2-3s 足够；个别"毒块"会让 reranker 推理卡死，
            # 15s 后快速失败并由 rerank_with_scores 跳过该块（不拖垮整批）。
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                err_msg = resp.json().get("error", "") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                print(f"Reranker: 单个评分接口错误 {resp.status_code}: {err_msg}")
                return -1.0
            
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

            # 在生成的 token sequence 中寻找第一个 yes/no 判定 token
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
            
            if "yes" in full_text.lower():
                return 10.0
            return 0.0
        except Exception as e:
            print(f"Reranker: 单个评分网络异常: {e}")
            return -1.0

    def rerank_with_scores(self, query: str, documents: list) -> list:
        """对文档列表进行重排并返回每个文档的分数信息。
        返回格式: [(index, score, doc), ...] 按分数从高到低排列。

        容错策略（逐块独立）：
        - 单块评分失败（如某些文本让 reranker 模型推理卡死/超时）→ 该块标记 -1.0 排到最后，
          不影响其余块的正常精排（避免"一个毒块拖垮整批降级"）。
        - 仅当所有块都失败（reranker 服务整体不可用/未启动）→ 整批降级，保留原始顺序。
        """
        if not documents:
            return []

        scored = []
        error_count = 0

        for i, doc in enumerate(documents):
            score = self._score_single(query, doc.page_content)
            if score < 0.0:  # 该块接口异常或超时
                error_count += 1
                scored.append((i, -1.0, doc))  # 标记失败，不中断其余块
            else:
                scored.append((i, score, doc))

        # 仅当全部失败才视为服务整体异常，整批降级
        if error_count == len(documents):
            print("Reranker: 全部评分失败，触发整批 Fallback 降级（reranker 服务可能未启动）。")
            return [(i, -1.0, doc) for i, doc in enumerate(documents)]

        if error_count > 0:
            print(f"Reranker: {error_count}/{len(documents)} 块评分失败（超时/卡死），已降级排后，其余正常精排。")

        # 成功块按分降序，失败块(-1.0)自然排到最后
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def rerank(self, query: str, documents: list) -> list:
        """对文档列表进行重排，返回 top_n 个最相关文档。"""
        scored = self.rerank_with_scores(query, documents)
        return [item[2] for item in scored[:self.top_n]]



# ===== 2. Load & Split Documents =====
def load_and_split_docs(doc_dir: str = DOCS_DIR):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
    docs = []
    if not os.path.exists(doc_dir):
        return []
    for root, dirs, files in os.walk(doc_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            try:
                if ext in [".txt", ".md"]:
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                docs.extend(loader.load())
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
                
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)


# ===== 3. Build or Load Vectorstore =====
def get_vectorstore(doc_dir: str = DOCS_DIR):
    if os.path.exists(CHROMA_DIR) and os.path.exists(os.path.join(CHROMA_DIR, "chroma.sqlite3")):
        print("加载已有向量数据库...")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    else:
        print("加载文档并创建向量数据库...")
        chunks = load_and_split_docs(doc_dir)
        if not chunks:
            raise FileNotFoundError(
                f"在 {doc_dir} 目录下未找到任何支持的文档文件，请先放入文档。"
            )
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )


# ===== 4. LLM =====
llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.7,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ===== 5. RAG Chain =====
prompt = ChatPromptTemplate.from_template(
    "你是一个专业的问答助手。根据以下上下文回答问题。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{question}\n\n"
    "请用中文简洁地回答。"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ===== 6. Ingest: 增量添加新文档 =====
def ingest():
    """加载 documents/ 中的文档，添加到已有向量库（自动去重）。"""
    vectorstore = get_vectorstore()
    chunks = load_and_split_docs()
    metadatas = vectorstore.get()["metadatas"]
    existing_sources = set()
    if metadatas:
        for m in metadatas:
            if m and "source" in m:
                existing_sources.add(os.path.normpath(m["source"]))
                
    new_chunks = [c for c in chunks if os.path.normpath(c.metadata.get("source", "")) not in existing_sources]

    if not new_chunks:
        print("没有新文档需要添加。")
        return

    # Chroma 单次 upsert 硬上限约 5461 条（SQLite 绑定参数限制），超过会整批拒收。
    # 这里用 5000 留余量，分批写入；任何一批失败不会回滚已写入的批次。
    CHROMA_WRITE_BATCH = 5000
    total = len(new_chunks)
    written = 0
    for i in range(0, total, CHROMA_WRITE_BATCH):
        batch = new_chunks[i:i + CHROMA_WRITE_BATCH]
        try:
            vectorstore.add_documents(batch)
            written += len(batch)
            print(f"  写入进度 {written}/{total}")
        except Exception as e:
            print(f"  写入批次 {i // CHROMA_WRITE_BATCH + 1} 失败: {e}")

    print(f"成功添加 {written}/{total} 个新文本块到向量库（总切片数: {len(chunks)}）。")


# ===== 7. 根据命令行模式执行 =====
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
        sys.exit(0)

    # 查询模式
    vectorstore = get_vectorstore()
    # 粗回阶段：从向量库检索更多候选文档 (例如 k=10)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 精排阶段：使用 LM Studio Reranker 过滤并排序得到 Top 3
    reranker = LMStudioReranker(model=RERANK_MODEL, base_url=RERANK_BASE_URL, top_n=3)

    # 自定义检索 + 重排函数，使用 RunnableLambda 包装以接入 LCEL 链
    def retrieve_and_rerank(query: str):
        docs = base_retriever.invoke(query)
        return reranker.rerank(query, docs)

    retriever = RunnableLambda(retrieve_and_rerank)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG 系统已启动（输入 'quit' 退出）\n")
    while True:
        question = input("问题: ")
        if question.strip().lower() == "quit":
            break
        print("回答: ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")
