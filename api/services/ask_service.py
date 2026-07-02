"""Q&A over a processed document's PageIndex via the local Ollama chat model.

ChatEngine instances are cached per doc_id (loading a PageIndex parses JSON);
the cache is invalidated on delete.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from api.services import storage_service as storage

logger = logging.getLogger(__name__)

_engines: Dict[str, Any] = {}
_lock = threading.Lock()


def _get_engine(doc_id: str) -> Optional[Any]:
    with _lock:
        engine = _engines.get(doc_id)
    if engine is not None:
        return engine

    path = storage.pageindex_path(doc_id)
    if not path.is_file():
        return None

    from chatbot import ChatEngine  # lazy: pulls ollama client

    engine = ChatEngine(str(path))
    with _lock:
        _engines[doc_id] = engine
    return engine


def invalidate(doc_id: str) -> None:
    with _lock:
        _engines.pop(doc_id, None)


def ask(doc_id: str, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """Answer a question; returns {"answer", "sources": [...]}. Raises if no store.

    ``top_k`` (optional) overrides how many context nodes are retrieved for this question.
    """
    engine = _get_engine(doc_id)
    if engine is None:
        raise FileNotFoundError(f"PageIndex not found for document {doc_id}")

    result = engine.ask(question, top_k=top_k)

    sources: List[Dict[str, Any]] = []
    for item in result.get("sources") or []:
        node = engine.store.get_node(item.get("node_id")) if item.get("node_id") else None
        snippet = ((node or {}).get("content") or "").strip()[:240]
        sources.append(
            {
                "page_number": item.get("page_number"),
                "node_type": item.get("node_type"),
                "source": "pageindex",
                "snippet": snippet,
            }
        )

    # Return the answer as an array of lines (no embedded newlines for API consumers).
    answer_lines = [ln.rstrip() for ln in str(result.get("answer") or "").splitlines() if ln.strip()]
    return {"answer": answer_lines, "sources": sources}
