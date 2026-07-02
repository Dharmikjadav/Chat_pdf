"""Local, vector-free PageIndex-style storage for parsed PDF documents."""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["PageIndex"]

_PREFIX = {
    "document": "doc",
    "page": "page",
    "section": "sec",
    "text": "txt",
    "table": "tbl",
    "chart": "cht",
    "kpi": "kpi",
    "fact": "fct",
}

_CONTEXT_TYPES = ("text", "table", "chart", "section", "kpi", "fact")
_INDEXED_TYPES = ("text", "table", "chart", "section", "kpi", "fact")

# --- BM25 keyword retrieval tuning ----------------------------------------- #
_BM25_K1 = 1.5
_BM25_B = 0.75
_TITLE_BOOST = 3        # title tokens folded this many times into the doc term-freqs
_PHRASE_BOOST = 1.25    # multiplier when the full filtered query is a substring of content

# Grammatical glue + question words only. Finance words (revenue, profit, operations,
# total, income, …) are deliberately NOT stopwords — IDF down-weights them instead.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had has have
    he her him his how i if in into is it its me my no nor not of on once only or our out
    over own she so than that the their them then there these they this those to too us was
    we were what when where which while who whom why will with would you your your
    """.split()
)

# Acronym expansions: the acronym genuinely MEANS the phrase, so these are safe to use for
# exact metric/title matching as well as BM25 recall (both singular/plural where the doc varies).
_ACRONYM_SYNONYMS = {
    "eps": ["earnings", "earning", "per", "share"],
    "pat": ["profit", "after", "tax"],
    "ebitda": ["earnings", "before", "interest", "tax", "depreciation", "amortisation"],
    "capex": ["capital", "expenditure"],
    "yoy": ["year", "over", "year"],
}
# Semantic finance synonyms: used ONLY to widen BM25 recall, NOT for exact title matching
# (e.g. revenue!=income; expenditure!=investment for capex), to avoid mis-routing to a wrong metric.
_SEMANTIC_SYNONYMS = {
    "expenditure": ["investment", "spend", "spending"],
    "investment": ["expenditure"],
    "revenue": ["sales", "turnover"],
    "sales": ["revenue", "turnover"],
    "turnover": ["revenue", "sales"],
    "headcount": ["employees", "employee", "workforce"],
    "employees": ["headcount", "workforce"],
    "employee": ["headcount", "workforce"],
    "workforce": ["employees", "headcount"],
    "division": ["segment", "segments", "divisions"],
    "divisions": ["segment", "segments", "division"],
    "segment": ["division", "divisions", "segments"],
    "segments": ["division", "divisions", "segment"],
    "operating": ["operations", "operational"],
    "operations": ["operating", "operational"],
    "operational": ["operating", "operations"],
    "margin": ["margins"],
    "margins": ["margin"],
    "remuneration": ["compensation", "pay"],
    "compensation": ["remuneration", "pay"],
    "subsidiary": ["subsidiaries"],
    "subsidiaries": ["subsidiary"],
}
_QUERY_SYNONYMS = {**_ACRONYM_SYNONYMS, **_SEMANTIC_SYNONYMS}


def _as_set(value: Optional[Union[str, Iterable[str]]]) -> Optional[set]:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def _table_to_text(data: Any, title: Optional[str]) -> str:
    parts: List[str] = []
    if title:
        parts.append(str(title))
    if isinstance(data, dict):
        columns = data.get("columns") or []
        if columns:
            parts.append(" | ".join(str(column) for column in columns))
        for row in data.get("rows") or []:
            if isinstance(row, dict):
                parts.append(" | ".join("" if value is None else str(value) for value in row.values()))
            elif isinstance(row, (list, tuple)):
                parts.append(" | ".join("" if cell is None else str(cell) for cell in row))
            else:
                parts.append(str(row))
        if data.get("units"):
            parts.append(str(data["units"]))
        for footnote in data.get("footnotes") or []:
            parts.append(str(footnote))
    elif data is not None:
        parts.append(str(data))
    return "\n".join(parts).strip()


def _chart_to_text(data: Any, title: Optional[str]) -> str:
    parts: List[str] = []
    if title:
        parts.append(str(title))
    if isinstance(data, dict):
        if data.get("title"):
            parts.append(str(data["title"]))
        if data.get("chart_type"):
            parts.append(f"{data['chart_type']} chart")
        for axis_key in ("x_axis", "y_axis"):
            axis = data.get(axis_key)
            if isinstance(axis, dict) and axis.get("label"):
                parts.append(str(axis["label"]))
        for series in data.get("series") or []:
            if isinstance(series, dict):
                if series.get("name"):
                    parts.append(str(series["name"]))
                for point in series.get("points") or []:
                    if isinstance(point, dict):
                        segment = " ".join(
                            str(value)
                            for value in (point.get("label"), point.get("value"))
                            if value is not None
                        )
                        if segment:
                            parts.append(segment)
        for entry in data.get("legend") or []:
            if entry is not None:
                parts.append(str(entry))
        if data.get("notes"):
            parts.append(str(data["notes"]))
    elif data is not None:
        parts.append(str(data))
    return "\n".join(parts).strip()


class PageIndex:
    """In-memory document tree with keyword and page-based retrieval."""

    def __init__(self, path: Optional[str] = None):
        self.path = str(path) if path else None
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.roots: List[str] = []
        self._seq = 0
        self._doc_index: Dict[str, str] = {}
        self._page_index: Dict[str, str] = {}
        self._search_index: Optional[Dict[str, Any]] = None  # lazy BM25 index cache

    def __len__(self) -> int:
        return len(self.nodes)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    def document_names(self) -> List[str]:
        return list(self._doc_index.keys())

    def _next_id(self, node_type: str) -> str:
        self._seq += 1
        prefix = _PREFIX.get(node_type, node_type[:3])
        return f"{prefix}-{self._seq:06d}"

    def _unique_id(self, node_id: str) -> str:
        if node_id not in self.nodes:
            return node_id
        suffix = 2
        while f"{node_id}-{suffix}" in self.nodes:
            suffix += 1
        return f"{node_id}-{suffix}"

    def _require(self, node_id: Optional[str]) -> Dict[str, Any]:
        if node_id is None or node_id not in self.nodes:
            raise KeyError(f"parent node not found: {node_id}")
        return self.nodes[node_id]

    def _make_node(
        self,
        node_type: str,
        parent_id: Optional[str],
        document_name: Optional[str],
        page_number: Optional[int],
        title: Optional[str],
        content: Optional[str],
        metadata: Optional[Dict[str, Any]],
        node_id: Optional[str],
    ) -> Dict[str, Any]:
        chosen = self._unique_id(node_id) if node_id else self._next_id(node_type)
        node = {
            "id": chosen,
            "parent_id": parent_id,
            "document_name": document_name,
            "page_number": page_number,
            "node_type": node_type,
            "title": title,
            "content": content or "",
            "metadata": dict(metadata or {}),
            "children": [],
        }
        self.nodes[chosen] = node
        self._search_index = None  # invalidate cached BM25 index on mutation
        return node

    def _add_child(
        self,
        parent_id: str,
        node_type: str,
        title: Optional[str],
        content: Optional[str],
        page_number: Optional[int],
        metadata: Optional[Dict[str, Any]],
        node_id: Optional[str],
    ) -> str:
        parent = self._require(parent_id)
        document_name = parent["document_name"]
        page = page_number if page_number is not None else parent.get("page_number")
        node = self._make_node(node_type, parent_id, document_name, page, title, content, metadata, node_id)
        parent["children"].append(node["id"])
        return node["id"]

    def add_document(
        self,
        document_name: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a document root node (idempotent by document_name)."""
        if document_name in self._doc_index:
            return self._doc_index[document_name]
        node = self._make_node(
            "document", None, document_name, None, title or document_name, "", metadata, node_id
        )
        self.roots.append(node["id"])
        self._doc_index[document_name] = node["id"]
        return node["id"]

    def _resolve_document(self, document: str) -> str:
        if document in self.nodes and self.nodes[document]["node_type"] == "document":
            return document
        if document in self._doc_index:
            return self._doc_index[document]
        return self.add_document(document)

    def add_page(
        self,
        document: str,
        page_number: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create (or return existing) a page node under a document."""
        document_id = self._resolve_document(document)
        key = f"{document_id}#{page_number}"
        if key in self._page_index:
            return self._page_index[key]
        document_name = self.nodes[document_id]["document_name"]
        node = self._make_node(
            "page",
            document_id,
            document_name,
            page_number,
            title or f"Page {page_number}",
            content,
            metadata,
            node_id,
        )
        self.nodes[document_id]["children"].append(node["id"])
        self._page_index[key] = node["id"]
        return node["id"]

    def add_section(
        self,
        parent_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a section node under a page or document."""
        return self._add_child(parent_id, "section", title, content, page_number, metadata, node_id)

    def add_text_node(
        self,
        parent_id: str,
        content: str,
        title: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a text node under a parent."""
        return self._add_child(parent_id, "text", title, content, page_number, metadata, node_id)

    def add_table_node(
        self,
        parent_id: str,
        data: Any = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a table node; structured data is kept in metadata['data']."""
        meta = dict(metadata or {})
        if data is not None and "data" not in meta:
            meta["data"] = data
        text = content if content is not None else _table_to_text(data, title)
        return self._add_child(parent_id, "table", title, text, page_number, meta, node_id)

    def add_chart_node(
        self,
        parent_id: str,
        data: Any = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a chart node; structured data is kept in metadata['data']."""
        meta = dict(metadata or {})
        if data is not None and "data" not in meta:
            meta["data"] = data
        text = content if content is not None else _chart_to_text(data, title)
        return self._add_child(parent_id, "chart", title, text, page_number, meta, node_id)

    def add_kpi_node(
        self,
        parent_id: str,
        data: Any = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a KPI node; the KPI dict is kept in metadata['data']."""
        meta = dict(metadata or {})
        if data is not None and "data" not in meta:
            meta["data"] = data
        text = content if content is not None else (title or "")
        return self._add_child(parent_id, "kpi", title, text, page_number, meta, node_id)

    def add_fact_node(
        self,
        parent_id: str,
        content: str,
        title: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Create a leaf fact node (one metric/period/value); content is its searchable sentence."""
        return self._add_child(parent_id, "fact", title, content, page_number, metadata, node_id)

    def save(self, path: Optional[str] = None) -> str:
        """Persist the store to a JSON file."""
        target = path or self.path
        if not target:
            raise ValueError("no path provided to save()")
        out = Path(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "seq": self._seq, "roots": self.roots, "nodes": self.nodes}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.path = str(out)
        logger.info("PageIndex saved %d nodes to %s", len(self.nodes), out)
        return str(out)

    @classmethod
    def load(cls, path: str) -> "PageIndex":
        """Load a store from a JSON file."""
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        store = cls(path=str(source))
        store.nodes = data.get("nodes", {})
        store.roots = data.get("roots", [])
        store._seq = data.get("seq", 0)
        store._reindex()
        logger.info("PageIndex loaded %d nodes from %s", len(store.nodes), source)
        return store

    def _reindex(self) -> None:
        self._doc_index = {}
        self._page_index = {}
        for node in self.nodes.values():
            if node["node_type"] == "document":
                self._doc_index[node["document_name"]] = node["id"]
            elif node["node_type"] == "page" and node.get("page_number") is not None:
                self._page_index[f"{node['parent_id']}#{node['page_number']}"] = node["id"]

    @staticmethod
    def _terms(query: str) -> List[str]:
        """Raw tokenization (fallback when stopword filtering empties the query)."""
        return [token for token in re.split(r"\W+", str(query).lower()) if token]

    @staticmethod
    def _doc_terms(text: str) -> List[str]:
        """Document tokenizer: lowercase, keep digits and tokens of length >= 2.

        Collapses single-letter ampersand acronyms (R&D -> rd, S&P -> sp) so they survive as one
        meaningful token instead of being split into dropped 1-char pieces."""
        s = re.sub(r"\b([a-z])\s*&\s*([a-z])\b", r"\1\2", str(text).lower())
        return [t for t in re.split(r"\W+", s) if t and (t.isdigit() or len(t) >= 2)]

    @classmethod
    def _filtered_terms(cls, query: str) -> List[str]:
        """Query tokenizer for BM25 recall: drop stopwords, expand acronyms + semantic synonyms."""
        out: List[str] = []
        for token in cls._doc_terms(query):
            if token in _STOPWORDS:
                continue
            out.append(token)
            out.extend(_QUERY_SYNONYMS.get(token, ()))
        return out

    @classmethod
    def _core_terms(cls, query: str) -> List[str]:
        """Query terms for EXACT title/metric matching: raw tokens + acronym expansions only.

        Semantic synonyms (revenue<->sales, expenditure<->investment) are deliberately excluded
        here so they widen BM25 recall but cannot mis-route exact-title injection to a different
        metric (e.g. 'total revenue' must not exact-match a 'Total income' table)."""
        out: List[str] = []
        for token in cls._doc_terms(query):
            if token in _STOPWORDS:
                continue
            out.append(token)
            out.extend(_ACRONYM_SYNONYMS.get(token, ()))
        return out

    @staticmethod
    def _page_sort_key(node: Dict[str, Any]):
        page = node.get("page_number")
        return (page if isinstance(page, int) else 10**9, node["id"])

    def _ensure_search_index(self) -> Dict[str, Any]:
        """Build (once, then cache) the BM25 index over retrievable nodes."""
        if self._search_index is not None:
            return self._search_index

        postings: List[Dict[str, Any]] = []
        df: Dict[str, int] = {}
        total_len = 0
        for node in self.nodes.values():
            if node["node_type"] not in _INDEXED_TYPES:
                continue
            tf: Counter = Counter(self._doc_terms(node.get("content") or ""))
            for token in self._doc_terms(node.get("title") or ""):
                tf[token] += _TITLE_BOOST
            length = sum(tf.values())
            total_len += length
            for token in tf:
                df[token] = df.get(token, 0) + 1
            postings.append(
                {"id": node["id"], "tf": tf, "dl": length, "content_lc": (node.get("content") or "").lower()}
            )

        n_docs = len(postings)
        avgdl = (total_len / n_docs) if n_docs else 0.0
        idf = {token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}
        self._search_index = {"postings": postings, "idf": idf, "avgdl": avgdl, "n_docs": n_docs}
        return self._search_index

    @staticmethod
    def _bm25_score(posting: Dict[str, Any], q_terms: List[str], idf: Dict[str, float], avgdl: float) -> float:
        tf = posting["tf"]
        length_norm = _BM25_B * (posting["dl"] / avgdl if avgdl else 0.0)
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            term_idf = idf.get(term)
            if term_idf is None:
                continue
            score += term_idf * (freq * (_BM25_K1 + 1)) / (freq + _BM25_K1 * (1 - _BM25_B + length_norm))
        return score

    def search_keyword(
        self,
        query: str,
        limit: Optional[int] = 10,
        node_types: Optional[Union[str, Iterable[str]]] = None,
        document_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rank nodes by BM25 keyword relevance to the query."""
        q_terms = self._filtered_terms(query)
        if not q_terms:  # query was entirely stopwords -> fall back to raw tokens
            q_terms = self._terms(query)
        if not q_terms:
            return []

        index = self._ensure_search_index()
        idf, avgdl = index["idf"], index["avgdl"]
        type_filter = _as_set(node_types)
        phrase = " ".join(q_terms)
        use_phrase = len(q_terms) > 1

        scored: List[tuple] = []
        for posting in index["postings"]:
            node = self.nodes.get(posting["id"])
            if node is None:
                continue
            if type_filter and node["node_type"] not in type_filter:
                continue
            if document_name and node["document_name"] != document_name:
                continue
            score = self._bm25_score(posting, q_terms, idf, avgdl)
            if score <= 0:
                continue
            if use_phrase and phrase in posting["content_lc"]:
                score *= _PHRASE_BOOST
            scored.append((score, node))

        scored.sort(key=lambda item: (-item[0], self._page_sort_key(item[1])))
        results = [node for _, node in scored]
        return results[:limit] if limit else results

    def search_by_page(
        self,
        page_number: int,
        document_name: Optional[str] = None,
        node_types: Optional[Union[str, Iterable[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return all nodes on a given page."""
        type_filter = _as_set(node_types)
        results = [
            node
            for node in self.nodes.values()
            if node.get("page_number") == page_number
            and (not document_name or node["document_name"] == document_name)
            and (not type_filter or node["node_type"] in type_filter)
        ]
        results.sort(key=self._page_sort_key)
        return results

    def search_by_node_type(
        self,
        node_type: str,
        document_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all nodes of a given type."""
        results = [
            node
            for node in self.nodes.values()
            if node["node_type"] == node_type
            and (not document_name or node["document_name"] == document_name)
        ]
        results.sort(key=self._page_sort_key)
        return results

    def _breadcrumb(self, node: Dict[str, Any]) -> str:
        titles: List[str] = []
        current = node
        seen = set()
        while current.get("parent_id") and current["parent_id"] in self.nodes:
            if current["id"] in seen:
                break
            seen.add(current["id"])
            current = self.nodes[current["parent_id"]]
            if current.get("title"):
                titles.append(str(current["title"]))
        return " > ".join(reversed(titles))

    def get_context_for_question(
        self,
        question: str,
        max_nodes: int = 8,
        max_chars: int = 4000,
        document_name: Optional[str] = None,
        node_types: Optional[Union[str, Iterable[str]]] = None,
    ) -> str:
        """Assemble a cited context block for a question via keyword retrieval."""
        ranked = self.search_keyword(
            question,
            limit=max_nodes,
            node_types=node_types or _CONTEXT_TYPES,
            document_name=document_name,
        )
        if not ranked:
            return ""

        blocks: List[str] = []
        total = 0
        for node in ranked:
            page = node.get("page_number")
            citation = f"[{node['document_name']} p.{page}]" if page is not None else f"[{node['document_name']}]"
            crumb = self._breadcrumb(node)
            header = f"{citation} {crumb}".strip()
            block = f"{header}\n{node.get('content') or ''}".strip()
            if total + len(block) > max_chars and blocks:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n---\n\n".join(blocks)
