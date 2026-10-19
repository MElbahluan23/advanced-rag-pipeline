# 🏢 Insurellm RAG — Expert Q&A System

A complete, end-to-end **Retrieval-Augmented Generation (RAG)** project for an insurance tech company knowledge base.

This project demonstrates two RAG pipelines — from a simple LangChain baseline to a production-grade system with LLM re-ranking and query rewriting — all built on a **synthetically generated knowledge base** created with GPT.

---

## 🗂️ Project Structure

```
insurellm-rag/
├── notebooks/
│   ├── 01_build_synthetic_data.ipynb   ← Generate the knowledge base with an LLM
│   └── 02_rag_implementations.ipynb    ← Full RAG walthrough: Simple → Pro
├── implementation/                     ← Simple RAG (LangChain)
│   ├── ingest.py                       ← Fixed-size chunking + Chroma
│   └── answer.py                       ← Retrieve → Prompt → LLM
├── pro_implementation/                 ← Pro RAG (raw OpenAI SDK + litellm)
│   ├── ingest.py                       ← LLM-powered chunking + raw ChromaDB
│   └── answer.py                       ← Rewrite → Dual search → Re-rank → LLM
├── knowledge-base/                     ← Generated .md files (git-ignored vectors)
│   ├── company/                        ← about.md, overview.md, culture.md, careers.md
│   ├── employees/                      ← One .md file per employee
│   ├── products/                       ← One .md file per product
│   └── contracts/                      ← One .md file per client contract
├── app.py                              ← Gradio UI (toggle Simple / Pro)
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/insurellm-rag.git
cd insurellm-rag
pip install -r requirements.txt
```

### 2. Set up your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add your API keys to `.env`

```bash
cp .env.example .env
```

| Key | Used for |
|-----|----------|
| `OPENAI_API_KEY` | RAG embeddings + answer pipeline |
| `HF_TOKEN` | Synthetic data generation (Llama via HuggingFace Inference API) |

Get your HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
Then accept the model license at [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### 4. Generate the knowledge base

Open and run **`notebooks/01_build_synthetic_data.ipynb`** top to bottom.

This generates ~80+ markdown files in `knowledge-base/` using **`meta-llama/Llama-3.2-3B-Instruct`**:
- 4 company pages
- 10 employee profiles
- 8 product spec sheets
- 16 client contracts

> ⚡ Takes ~3-5 minutes via the HuggingFace Inference API (free tier).

### 4. Run the RAG pipeline (notebook)

Open **`notebooks/02_rag_implementations.ipynb`** and run it step by step to understand both implementations.

### 5. Launch the Gradio app

```bash
python app.py
```

Starts a web UI at `http://localhost:7860` with a **Simple / Pro** toggle.

---

## 📊 Simple vs Pro Implementation

| Feature | Simple | Pro |
|---|---|---|
| **Chunking** | Fixed-size (500 chars) | LLM-generated (headline + summary + original text) |
| **Vector DB** | LangChain Chroma wrapper | Raw `chromadb.PersistentClient` |
| **Query** | Raw user question | LLM rewrites query from conversation context |
| **Retrieval** | One vector search | Two searches (original + rewritten), merged |
| **Re-ranking** | ❌ | ✅ LLM re-ranks by relevance |
| **Retry logic** | ❌ | ✅ Exponential backoff on all LLM calls |
| **Parallelism** | Single-threaded | Multiprocessing (3 workers during ingest) |
| **Best for** | Learning / prototyping | Production |

---

## 🧠 How RAG Works

```
INGEST (run once):
  Knowledge Base (.md files)
    → Chunk text
    → Embed chunks (OpenAI text-embedding-3-large)
    → Store in ChromaDB

ANSWER (every query):
  User Question
    → [Pro: rewrite query with LLM]
    → Vector search in ChromaDB
    → [Pro: merge dual search + LLM re-rank]
    → Top-K chunks injected into system prompt
    → LLM generates answer
```

---

## ⚙️ Configuration

### Change the synthetic data model

In `notebooks/01_build_synthetic_data.ipynb`:
```python
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"   # default — free HF Inference API
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # larger, better quality
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
```

### Change the RAG answer model

In `implementation/answer.py` and `implementation/ingest.py`:
```python
MODEL = "gpt-4.1-nano"   # or "gpt-4o", "gpt-4o-mini"
```

In `pro_implementation/`:
```python
MODEL = "openai/gpt-4.1-nano"   # or any litellm-supported model
# MODEL = "groq/llama-3.1-70b-versatile"   # faster via Groq
# MODEL = "anthropic/claude-3-haiku-20240307"
```

### Generate your own company's knowledge base

Edit the `COMPANY` config in `notebooks/01_build_synthetic_data.ipynb`:
```python
COMPANY = {
    "name":        "YourCompany",
    "founded":     2020,
    "founder":     "Jane Doe",
    "industry":    "FinTech",
    "products": [...],
    "employee_roles": [...],
}
```

---

## 📦 Requirements

- Python 3.10+
- OpenAI API key
- Optional: Groq API key (for faster inference in pro mode)

---

## 📄 License

MIT
