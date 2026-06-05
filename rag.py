"""
rag.py — Knowledge base loader.

knowledge.txt is small enough to inject in full as system-prompt context,
so no embeddings / vector store are needed.
"""

import os

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE = os.path.join(_BASE_DIR, "knowledge.txt")

try:
    with open(_KNOWLEDGE, "r", encoding="utf-8") as _f:
        _KNOWLEDGE_TEXT = _f.read().strip()
except FileNotFoundError:
    _KNOWLEDGE_TEXT = ""


def get_relevant_context(query: str) -> str:
    """Return the full knowledge base. `query` is accepted for API compatibility."""
    return _KNOWLEDGE_TEXT
