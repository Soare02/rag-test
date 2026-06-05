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

## 常见问题

**Q: 为什么 Reranker 显示"服务异常，已自动降级"？**
A: LM Studio 未启动或 Reranker 模型未加载。请确保 LM Studio Server 在 `http://localhost:1234` 运行并加载了 `qwen3-reranker-0.6b`。

**Q: BM25 检索为什么不可用？**
A: 项目使用 `bm25s` 库，默认空格分词器对中文无效。安装 `jieba` 后可启用中文分词支持。

**Q: LlamaIndex 索引如何重建？**
A: 删除 `llama_storage/` 目录后重新运行 `uv run python llama_rag.py ingest`。

## License

MIT
