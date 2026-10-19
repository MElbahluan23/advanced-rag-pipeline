# 🏢 Qafela (قافلة) RAG — Expert Q&A System

A complete, end-to-end **Retrieval-Augmented Generation (RAG)** project for an insurance tech company knowledge base.

Two RAG pipelines — from a simple LangChain baseline to a production-grade system with LLM re-ranking and query rewriting — all built on a **synthetically generated knowledge base** created with `llama3.2` running locally via **Ollama** (free, no API key needed).

---

## 🗂️ Project Structure

```
insurellm-rag/
├── notebooks/
│   ├── 01_build_synthetic_data.ipynb   ← Generate the knowledge base with Llama-3.2-3B
│   └── 02_rag_implementations.ipynb    ← Full RAG walkthrough: Simple → Pro + Gradio UI
├── knowledge-base/                     ← Generated .md files (auto-created by notebook 01)
│   ├── company/
│   ├── employees/
│   ├── products/
│   └── contracts/
├── .env.example                        ← Template for API keys
├── requirements.txt
└── README.md
```

> Everything lives in the two notebooks — no separate `.py` files needed.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/insurellm-rag.git
cd insurellm-rag
pip install -r requirements.txt
```

### 2. Set up Ollama (for notebook 01)

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve   # starts the local server (runs in background)
```

No API key needed — runs 100% locally and free!

### 3. Set up your OpenAI key (for notebook 02)

```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

### 4. Run Notebook 01 — Build the Knowledge Base

Open **`notebooks/01_build_synthetic_data.ipynb`** and run all cells.

Generates ~80+ `.md` files in `knowledge-base/` using `llama3.2` via Ollama:
- 4 company pages (about, overview, culture, careers)
- 10 employee HR profiles
- 8 product spec sheets
- 16 client contracts

> ⚡ ~5-10 minutes depending on your machine.

### 4. Run Notebook 02 — RAG Implementations

Open **`notebooks/02_rag_implementations.ipynb`** and run it step by step.

Covers both pipelines end-to-end and launches a **Gradio UI** at the end.

---

## 📊 Simple vs Pro Implementation

| Feature | Simple | Pro |
|---|---|---|
| **Chunking** | Fixed-size (500 chars, LangChain) | LLM-generated (headline + summary + original text) |
| **Vector DB** | LangChain Chroma wrapper | Raw `chromadb.PersistentClient` |
| **Query** | Raw user question | LLM rewrites query from conversation context |
| **Retrieval** | One vector search | Two searches (original + rewritten), merged |
| **Re-ranking** | ❌ | ✅ LLM re-ranks by relevance |
| **Retry logic** | ❌ | ✅ Exponential backoff on all LLM calls |
| **Best for** | Learning / prototyping | Production |

---

## 🧠 How RAG Works

```
INGEST (run once per implementation):
  knowledge-base/*.md
    → Chunk text
    → Embed (OpenAI text-embedding-3-large)
    → Store in ChromaDB

ANSWER (every query):
  User Question
    → [Pro: LLM rewrites query]
    → Vector search ChromaDB
    → [Pro: dual search + LLM re-rank]
    → Top-K chunks → System Prompt → LLM → Answer
```

---

## ⚙️ Customize

### Use a different model for synthetic data

In `notebooks/01_build_synthetic_data.ipynb` (Step 2):
```python
MODEL_ID = "llama3.2"      # default — already installed via Ollama
# MODEL_ID = "mistral"     # ollama pull mistral
# MODEL_ID = "qwen2.5:14b" # ollama pull qwen2.5:14b (higher quality)
```

### Generate a knowledge base for your own company

Edit the `COMPANY` dict in notebook 01:
```python
COMPANY = {
    "name":    "YourCompany",
    "founded": 2020,
    "founder": "Jane Doe",
    "industry": "FinTech",
    "products": [...],
    "employee_roles": [...],
}
```

---

## 📦 Requirements

- Python 3.10+
- OpenAI API key (RAG pipeline)
- HuggingFace token (data generation)

---

## 📄 License

MIT
