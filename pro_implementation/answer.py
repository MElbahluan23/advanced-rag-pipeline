"""
Pro RAG Answer Pipeline
------------------------
Features:
  - LLM query rewriting (context-aware from conversation history)
  - Dual retrieval (original + rewritten query)
  - Result merging (deduplication)
  - LLM re-ranking
  - Retry with exponential backoff
"""

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# MODEL = "openai/gpt-4.1-nano"   # cheaper
MODEL           = "groq/openai/gpt-oss-120b"
DB_NAME         = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K     = 20
FINAL_K         = 10
wait            = wait_exponential(multiplier=1, min=10, max=240)

openai_client = OpenAI()
chroma        = PersistentClient(path=DB_NAME)
collection    = chroma.get_or_create_collection(COLLECTION_NAME)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


# ── Pydantic models ────────────────────────────────────────────────────────────

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# ── Query rewriting ────────────────────────────────────────────────────────────

@retry(wait=wait)
def rewrite_query(question, history=None):
    """Rewrite the user's question to be more specific and context-aware."""
    if history is None:
        history = []

    message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


# ── Retrieval ──────────────────────────────────────────────────────────────────

def fetch_context_unranked(question):
    """Run a single vector search and return Result objects."""
    query_vec = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[question]
    ).data[0].embedding
    results = collection.query(query_embeddings=[query_vec], n_results=RETRIEVAL_K)
    return [
        Result(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def merge_chunks(chunks, extra):
    """Merge two result lists, keeping unique chunks only."""
    merged   = chunks[:]
    existing = {chunk.page_content for chunk in chunks}
    for chunk in extra:
        if chunk.page_content not in existing:
            merged.append(chunk)
            existing.add(chunk.page_content)
    return merged


# ── Re-ranking ─────────────────────────────────────────────────────────────────

@retry(wait=wait)
def rerank(question, chunks):
    """Ask the LLM to re-rank chunks by relevance to the question."""
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
Rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all chunk ids you are provided with.
"""
    user_prompt = f"Question:\n\n{question}\n\nChunks:\n\n"
    for i, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {i + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    order    = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[i - 1] for i in order]


# ── Full retrieval pipeline ────────────────────────────────────────────────────

def fetch_context(original_question, history=None):
    """Full pro retrieval: rewrite → dual search → merge → re-rank → top-K."""
    rewritten = rewrite_query(original_question, history)
    print(f"  Rewritten query: {rewritten}")

    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten)
    chunks  = merge_chunks(chunks1, chunks2)

    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


# ── RAG message builder ────────────────────────────────────────────────────────

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    return (
        [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        + history
        + [{"role": "user", "content": question}]
    )


# ── Main answer function ───────────────────────────────────────────────────────

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """Answer a question using the full Pro RAG pipeline."""
    chunks   = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
