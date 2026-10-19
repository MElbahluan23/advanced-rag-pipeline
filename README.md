# Qafela (قافلة) RAG

An end-to-end retrieval-augmented generation project. A local Llama model writes a synthetic knowledge base for **Qafela (قافلة)**, an Egyptian insurance-tech company in Cairo, then two RAG pipelines answer questions about it and are scored on the same test set.

The founder and CEO in the generated data is **Mohamed Elbahluan**.

---

## Project structure

```
insurellm-rag/
├── notebooks/
│   ├── 01_build_synthetic_data.ipynb   # Knowledge base + evaluation tests
│   └── 02_rag_implementations.ipynb    # Simple RAG, pro RAG, then a comparison
├── knowledge-base/                     # Markdown written by notebook 01
│   ├── company/
│   ├── employees/
│   ├── products/
│   └── contracts/
├── evaluation/
│   └── tests.jsonl                     # One test question per line
├── .env.example
├── requirements.txt
└── README.md
```

Both pipelines live in the notebooks. There are no separate application modules.

---

## Quick start

### 1. Install

```bash
git clone git@github.com:MElbahluan23/insurellm-rag.git
cd insurellm-rag
pip install -r requirements.txt
```

### 2. Start Ollama for notebook 01

```bash
ollama pull llama3.2
ollama serve
```

Notebook 01 calls Llama locally. It does not need an API key.

### 3. Add an OpenAI key for notebook 02

```bash
cp .env.example .env
```

Put `OPENAI_API_KEY` in `.env`. Embeddings use OpenAI. The pro pipeline and the answer judge use LiteLLM, so a Groq key is optional if you keep the Groq model in notebook 02.

### 4. Run notebook 01

Open `notebooks/01_build_synthetic_data.ipynb` and run it from the top.

Steps 4–7 write the knowledge base:

| Folder | Files |
|---|---|
| `company/` | 4 pages: about, overview, culture, careers |
| `employees/` | 32 Egyptian employee profiles |
| `products/` | 8 product sheets |
| `contracts/` | 16 contracts with Egyptian insurers, amounts in EGP |

Step 10 reads those markdown files and writes `evaluation/tests.jsonl`. Each line is one question whose keywords and reference answer come from that file. A question is kept only when every keyword appears in the source document.

### 5. Run notebook 02

Open `notebooks/02_rag_implementations.ipynb` and run it in order.

1. **Simple RAG** splits documents on a fixed size, embeds them, and answers from one vector search.
2. **Pro RAG** asks the model to chunk each document, rewrites the question, searches twice, and re-ranks the chunks.
3. **Compare Simple vs Pro** loads `evaluation/tests.jsonl` and scores both pipelines.

`MAX_TESTS` at the bottom of notebook 02 defaults to 5. Raise it when you want the full file.

---

## How a test line is shaped

This matches `TestQuestion` in the course file `week5/evaluation/test.py`:

```json
{"question": "Who is the CEO of Qafela?", "keywords": ["Mohamed", "Elbahluan", "CEO"], "reference_answer": "Mohamed Elbahluan is the Co-Founder and CEO of Qafela.", "category": "direct_fact"}
```

The generator aims for the same mix as the course set: 70 `direct_fact`, 20 `temporal`, 20 `spanning`, 10 `comparative`, 10 `numerical`, 10 `relationship`, and 10 `holistic`.

---

## How the comparison is scored

The last cell of notebook 02 follows `week5/evaluation/eval.py`.

| Score | What it measures |
|---|---|
| Retrieval MRR | How high the first chunk containing each keyword is ranked |
| Retrieval nDCG | Whether relevant chunks sit near the top |
| Keyword coverage | Share of keywords found in the retrieved chunks |
| Accuracy | Judge score, 1–5, against the reference answer |
| Completeness | Judge score, 1–5, for covering the reference answer |
| Relevance | Judge score, 1–5, for answering only the question |

---

## Simple vs pro

| | Simple | Pro |
|---|---|---|
| Chunking | Fixed size, 500 characters | Model writes a headline, summary, and original text |
| Vector store | LangChain Chroma | Raw Chroma client |
| Query | The user question as written | Model rewrites the question from the conversation |
| Search | One search | Original question and rewritten question, merged |
| Ranking | Vector similarity | Model re-ranks the chunks |
| Retries | None | Exponential backoff |

---

## Flow

```
Notebook 01
  llama3.2 writes knowledge-base/**/*.md
  llama3.2 reads those files and writes evaluation/tests.jsonl

Notebook 02
  chunk and embed the markdown
  simple pipeline answers from one search
  pro pipeline rewrites, searches twice, and re-ranks
  both answers are scored with the test file
```

---

## Change the company

Edit `COMPANY` in notebook 01, Step 2, then rerun the notebook. Delete old files in `knowledge-base/` first so previous names do not stay mixed in.

```python
COMPANY = {
    "name": "Qafela (قافلة)",
    "founded": 2015,
    "founder": "Mohamed Elbahluan",
    "hq": "Cairo, Egypt",
    "employee_roles": [...],   # 32 people
    "clients_per_product": 2,  # 8 products x 2 = 16 contracts
}
```

To switch the local model:

```python
MODEL_ID = "llama3.2"
# MODEL_ID = "mistral"
# MODEL_ID = "qwen2.5:14b"
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with `llama3.2` for notebook 01
- An OpenAI API key for embeddings in notebook 02

## License

MIT
