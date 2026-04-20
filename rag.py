"""
rag.py — Retrieval-Augmented Generation for Aime.

Loads knowledge.txt, chunks it, embeds it with OpenAI embeddings,
and persists the vector store to ./chroma_db.

On subsequent runs, the existing chroma_db is reused (no re-embedding).

Public API:
    get_relevant_context(query: str) -> str
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import OPENAI_API_KEY

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE   = os.path.join(_BASE_DIR, "knowledge.txt")
_CHROMA_DIR  = os.path.join(_BASE_DIR, "chroma_db")
_COLLECTION  = "aime_knowledge"
_TOP_K       = 3

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)


def _build_store() -> Chroma:
    """Load knowledge.txt, split, embed, and persist to chroma_db."""
    loader = TextLoader(_KNOWLEDGE, encoding="utf-8")
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings,
        collection_name=_COLLECTION,
        persist_directory=_CHROMA_DIR,
    )
    return store


def _load_store() -> Chroma:
    """Load an already-persisted chroma_db from disk."""
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=_embeddings,
        persist_directory=_CHROMA_DIR,
    )


def _get_store() -> Chroma:
    """Return the vector store, building it on first run."""
    if os.path.isdir(_CHROMA_DIR) and os.listdir(_CHROMA_DIR):
        return _load_store()
    return _build_store()


# Initialise once at import time so the first query has no latency spike.
_store: Chroma = _get_store()


def get_relevant_context(query: str) -> str:
    """
    Retrieve the top-k most relevant chunks from the knowledge base
    and return them as a single concatenated string.
    Returns an empty string if nothing relevant is found.
    """
    results = _store.similarity_search(query, k=_TOP_K)
    if not results:
        return ""
    return "\n\n".join(doc.page_content for doc in results)
