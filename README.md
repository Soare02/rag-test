# LangChain-L — LangChain & LlamaIndex RAG Playground

基于 **LangChain** 与 **LlamaIndex** 双框架的 RAG（检索增强生成）实验项目，支持多管道对比、混合检索与 LM Studio 本地模型集成。

---

## 功能特性

- **双 RAG 管道** — LangChain Chroma + Qwen3-Reranker 与 LlamaIndex RouterQueryEngine 并存，Web 页面一键切换对比
- **多路查询路由** — LlamaIndex 根据问题意图自动路由到语义搜索 / 摘要概括 / 关键词匹配
- **Reranker 精排** — 基于 Qwen3-Reranker 的 Logprobs 连续打分（0–10 分），不依赖生成文本判断
- **混合检索** — Vector + BM25 融合（RRF 算法），支持中文分词降级
- **本地模型集成** — 通过 LM Studio 调用本地 Embedding 与 Reranker 模型
- **Web 可视化控制台** — FastAPI + 纯前端，实时查看召回、精排分数与模型回答
- **Agent 系统** — 基于 `create_agent()` 的"圣地巡礼"规划助手，集成天气 / 景点 / 动漫场景查询工具

## 技术栈

| 类别 | 技术 |
|---|---|
| **RAG 框架** | LangChain (`rag.py`) / LlamaIndex (`llama_rag.py`) |
| **向量存储** | Chroma DB |
| **LLM** | DeepSeek-V4-Flash (远程 API) / Gemma-4-E4B (本地 lm-studio) |
| **Embedding** | Qwen3-Embedding-0.6B (本地 lm-studio) |
| **Reranker** | Qwen3-Reranker-0.6B (Logprobs 打分，本地) |
| **Web 框架** | FastAPI + Uvicorn |
| **包管理器** | `uv` |

## 快速开始

### 前置条件

- Python >= 3.14
- `uv` 包管理器
- LM Studio（运行本地模型），至少加载：
  - `text-embedding-qwen3-embedding-0.6b`
  - `qwen3-reranker-0.6b`

### 安装

```bash
# 克隆仓库
git clone https://github.com/Soare02/rag-test.git
cd rag-test

# 安装依赖
uv sync
```

### 环境变量

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 运行

**Web 控制台（推荐）：**
```bash
uv run python -m uvicorn app:app --port 8001 --host 127.0.0.1
# 浏览器打开 http://127.0.0.1:8001
```

**CLI 模式：**

```bash
# LangChain RAG
uv run python rag.py            # 交互式问答
uv run python rag.py ingest     # 增量导入文档

# LlamaIndex RAG
uv run python llama_rag.py query   # RouterQueryEngine 交互模式
uv run python llama_rag.py ingest  # 构建索引

# Agent
uv run python agent.py   # 圣地巡礼规划助手
```

## 项目结构

```
├── app.py                 # FastAPI Web 后端（双管道切换）
├── rag.py                 # LangChain RAG 管道
├── llama_rag.py           # LlamaIndex RAG 管道（多索引 + 路由）
├── agent.py               # Agent 系统
├── tools.py               # 自定义 @tool 工具函数
├── static/                # 前端资源
│   ├── index.html
│   ├── script.js
│   └── style.css
├── documents/             # 知识文档（.txt, .md, .pdf, .docx）
├── chroma_db/             # Chroma 向量数据库
├── llama_storage/         # LlamaIndex 持久化索引
├── LCEL.py / LCEL1.py / LCEL2.py  # LCEL 链示例
├── image.py               # 多模态示例
├── pyproject.toml         # 依赖配置
└── .env                   # API Key 配置（不提交）
```

## 管道对比

| 特性 | LangChain 模式 | LlamaIndex 模式 |
|---|---|---|
| 检索方式 | Chroma 向量检索 | RouterQueryEngine 路由 |
| 检索索引 | 单路向量 | 三索引（向量 / 摘要 / 关键词） |
| 精排 | Qwen3-Reranker (Logprobs) | Qwen3-Reranker (Logprobs) |
| 生成 | DeepSeek-V4-Flash | DeepSeek-V4-Flash |
| 路由策略 | 固定向量检索 | LLM 根据意图自动选择索引 |

## 容错机制

- **Reranker 单块容错**：个别文本块可能让 reranker 模型推理卡死/超时。系统对每个块**独立打分**，单块失败（标记 `-1.0`）只把该块排到末尾，不影响其余块的正常精排；仅当**所有块**都失败（reranker 服务未启动）才整批降级保留原始顺序。
- **路由空命中兜底**：LlamaIndex 模式下若路由选中的索引（如 keyword）未命中任何结果，自动回退到向量检索，保证生成始终有上下文（前端显示 `keyword_tool → vector_tool (兜底)`）。
- **文本截断**：送入 reranker 打分的文本截断到 1500 字（判相关性足够），展示与生成仍用原文，降低超时概率。
- **Embedding 分批 + 重试**：大文档录入时把 N 条文本切成 32 条/批串行调用 LM Studio，单批 60s 超时、失败指数退避重试 3 次；三次仍失败用零向量占位（保证 `texts` 与 `vectors` 数量对齐），不让单点故障拖垮整个 ingest。每 20 批打印进度，长任务可观测。
- **Chroma 写入分批**：Chroma 后端 SQLite 单次 `upsert` 有约 5461 条硬上限（绑定参数限制）。`ingest()` 与 `/upload` 都按 5000 条/批写入，避免大文档录入时撞 `Batch size of N is greater than max batch size of 5461` 错误。
- **召回密度自适应**：粗召回 `k=20`，给 reranker 足够大的候选池（小说类长文档库 ~2 万 chunk 时，k=6 容易漏掉相关 chunk）。

## 常见问题

**Q: 我有一份很大的文档（>10 MB），上传时 Web 页面超时怎么办？**
A: Web `/upload` 走的是同步 HTTP，浏览器默认会在几分钟内 timeout。建议把文档放到 `documents/` 目录后改用 CLI：`uv run python rag.py ingest`。CLI 没有 HTTP 超时限制，配合分批 embedding（32 条/批）和分批写入 Chroma（5000 条/批），一份 22 MB 的纯文本（约 1.9 万 chunk）在 RTX 4090 + 本地 0.6B embedding 上大约跑 30-60 分钟，期间 LM Studio 需要保持加载且关闭"自动卸载未使用的即时加载模型"。

**Q: 跑完 `rag.py ingest` 后，前端为什么还查不到新文档？**
A: FastAPI 服务在启动时缓存了 Chroma client 状态，外部进程（CLI ingest）写入的数据它感知不到。需要 **重启 uvicorn**（Ctrl+C 后重新 `uv run python -m uvicorn app:app --port 8001 --host 127.0.0.1`），新进程会重新打开 sqlite 读取最新数据。

**Q: 召回 Top-K 看不到我刚录入的长文档内容？**
A: 检查粗召回 `k`（`app.py: query_langchain` 中的 `search_kwargs={"k": 20}`）。在 ~2 万 chunk 量级的库里 k 太小（如 6）容易漏，相关 chunk 排到 50 名外都正常；增大 k 到 20-30 给 reranker 更大池子精排即可。代价是单次查询时间从 ~10s 增加到 ~40s。

**Q: 为什么 Reranker 显示"服务异常，已自动降级"？**
A: 分两种情况：(1) 个别块显示降级、其余正常 → 该块让 reranker 推理超时，属正常容错，不影响整体结果；(2) **全部**块都降级 → LM Studio 未启动或 Reranker 模型未加载，请确认 LM Studio Server 在 `http://localhost:1234` 运行并加载了 `qwen3-reranker-0.6b`。建议在 LM Studio 中关闭模型自动卸载，让 embedding 与 reranker 常驻显存。

**Q: LlamaIndex 模式显示"keyword_tool → vector_tool (兜底)"是报错吗？**
A: 不是。这是正常容错：LLM 把问题路由到了关键词索引但未命中，系统自动回退到向量检索，最终结果正确。

**Q: BM25 检索为什么不可用？**
A: 项目使用 `bm25s` 库，默认空格分词器对中文无效，会自动降级为纯向量检索。安装 `jieba` 后可启用中文分词支持。

**Q: LlamaIndex 索引如何重建？**
A: 删除 `llama_storage/` 目录后重新运行 `uv run python llama_rag.py ingest`。

## License

MIT
