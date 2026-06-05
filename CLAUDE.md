# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Run a script**: `uv run python <script.py>` (e.g., `uv run python LCEL.py`)
- **Run all scripts**: `uv run python main.py`
- **RAG query mode**: `uv run python rag.py`
- **RAG ingest mode**: `uv run python rag.py ingest`
- **LlamaIndex ingest mode**: `uv run python llama_rag.py ingest`
- **LlamaIndex query mode**: `uv run python llama_rag.py query`
- **FastAPI web app**: `uv run python -m uvicorn app:app --port 8000 --host 127.0.0.1`
- **LlamaIndex API test**: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"your question\",\"mode\":\"llamaindex\"}"`
- **Add a dependency**: `uv add <package>`
- **Activate venv**: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)

## Project Structure

This is a **LangChain learning/playground project** that experiments with LCEL, agents, RAG, and multimodal patterns. Python 3.14 required; package manager is `uv`.

### Files

- **`LCEL.py`** — LCEL chain using `ChatDeepSeek` directly with `ChatPromptTemplate` and `StrOutputParser`
- **`LCEL1.py`** — Same LCEL pattern using the generic `init_chat_model()` factory (DeepSeek backend)
- **`LCEL2.py`** — LCEL chain targeting a local lm-studio endpoint (OpenAI-compatible, Gemma model)
- **`agent.py`** — An agent for "圣地巡礼" (anime pilgrimage) planning, using `create_agent()` with custom tools and DeepSeek + local Gemma models
- **`tools.py`** — `@tool`-decorated functions: `get_weather()` (wttr.in), `get_attraction()` (Tavily search), `get_time()`, `get_yg()`
- **`messages.py`** — Agent using explicit `SystemMessage`/`HumanMessage`/`AIMessage` with tool calling, streaming enabled
- **`image.py`** — Multimodal agent: passes an image URL via `HumanMessage` to a local vision-capable model
- **`rag.py`** — Full RAG pipeline: loads `.txt` files from `./documents/` → splits → embeds via local lm-studio → stores in Chroma (`./chroma_db/`) → retrieves with DeepSeek LLM. Two modes: default (interactive Q&A) and `ingest` (incremental document addition with dedup)
- **`llama_rag.py`** — LlamaIndex RAG pipeline alternative: multi-index system (VectorStoreIndex + SummaryIndex + KeywordTableIndex) with RouterQueryEngine for intent-based query routing, plus hybrid dense+BM25 retrieval with RRF fusion. Matching ingest/query CLI modes.
- **`app.py`** — FastAPI web dashboard. `/query` endpoint accepts `mode` parameter (`"langchain"` or `"llamaindex"`) to switch between LangChain and LlamaIndex backends.
- **`env.py`** — Small utility to test env var loading (prints `ARK_API_KEY`)
- **`main.py`** — Project entrypoint placeholder

### LLM Backends

All scripts access LLMs through one of two routes:
1. **DeepSeek remote API** (`deepseek-v4-flash`) — used via `init_chat_model()` or `ChatDeepSeek`, key from `DEEPSEEK_API_KEY` env var
2. **Local lm-studio** (`http://localhost:1234/v1`, OpenAI-compatible) — typically `google/gemma-4-e4b`, key set to `"lm-studio"`

### Key LangChain APIs Used

- `init_chat_model()` — model-agnostic factory (`langchain.chat_models`)
- `ChatDeepSeek` — DeepSeek-specific binding (`langchain_deepseek`)
- `create_agent()` — agent builder (`langchain.agents`)
- `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough` — LCEL core
- `DirectoryLoader`, `RecursiveCharacterTextSplitter`, `Chroma` — RAG pipeline
- `@tool` decorator — custom tool definition

### Common Pattern

```
load_dotenv() → init LLM → build chain/agent → invoke/stream → print output
```
