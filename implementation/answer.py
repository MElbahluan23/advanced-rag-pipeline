"""
Simple RAG Answer Pipeline
---------------------------
Retrieves relevant chunks from Chroma, injects them into a system prompt,
and calls OpenAI's chat model.
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL       = "gpt-4.1-nano"
DB_NAME     = str(Path(__file__).parent.parent / "vector_db")
RETRIEVAL_K = 10

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever   = vectorstore.as_retriever()
llm         = ChatOpenAI(temperature=0, model_name=MODEL)


def fetch_context(question: str) -> list[Document]:
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
