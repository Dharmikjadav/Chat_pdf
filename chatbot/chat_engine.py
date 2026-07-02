"""Offline chat engine: retrieve from a PageIndex store and answer via Ollama.

Retrieval is vector-free (PageIndex BM25 keyword search). The local LLM is
instructed to answer strictly from the retrieved context and to return a tight
structured answer (table/chart name + key:value + a single page reference).
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pageindex_store import PageIndex

logger = logging.getLogger(__name__)

__all__ = ["ChatEngine"]

DEFAULT_MODEL = "qwen3:8b"
_CONTEXT_TYPES = ("text", "table", "chart", "kpi", "fact")
_FULL_TABLE_RE = re.compile(
    r"\b(full|entire|whole|complete|all rows|all the data|all the values|"
    r"show (?:me )?the table|give (?:me )?the table|list all|full table)\b",
    re.IGNORECASE,
)
# request boilerplate stripped from a "full table" query so only the metric name drives retrieval
_TABLE_REQUEST_NOISE = re.compile(
    r"\b(give|show|tell|get|want|need|please|me|my|i|you|can|the|a|an|of|in|all|"
    r"full|entire|whole|complete|proper|exact|list|table|tables|data|detail|details|"
    r"row|rows|value|values|key|pair|pairs)\b",
    re.IGNORECASE,
)
_NO_CONTEXT_MSG = "I could not find anything relevant to that question in the document."
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PERIOD_RE = re.compile(r"\bfy ?\d{2,4}\b|\b20\d{2}(?:[-/]\d{2,4})?\b|\b\d{4}[-/]\d{2,4}\b", re.IGNORECASE)


def _period_variants(token: str) -> set:
    """Canonicalize a period token to the space-stripped lowercase forms it can appear as.

    Indian fiscal years are written many ways for the SAME year — 'FY25', 'FY2025', '2024-25',
    '2024-2025', 'year ended March 31, 2025'. They all denote fiscal-year-END 2025, so we map a
    token to its end year and emit every spelling, letting period matching survive notation drift.
    """
    t = re.sub(r"\s+", "", token.lower())
    end = None
    bare_year = None
    if t.startswith("fy"):
        m = re.search(r"\d{2,4}", t)
        if m:
            n = int(m.group(0))
            end = n if n >= 1900 else 2000 + n
    elif re.fullmatch(r"\d{4}[-/]\d{2,4}", t):
        b = re.split(r"[-/]", t)[1]
        end = int(b) if len(b) == 4 else 2000 + int(b)
    elif re.fullmatch(r"\d{4}", t):
        end = int(t)
        bare_year = t  # a calendar-year query ("2023") must also match bare "2023" in content
    if not end:
        return {t}
    yy = end % 100
    variants = {
        f"fy{yy:02d}", f"fy{end}", f"{end - 1}-{yy:02d}", f"{end - 1}-{end}",
        f"{end - 1}/{yy:02d}", f"march31,{end}", f"31,{end}",
    }
    if bare_year:
        variants.add(bare_year)
    return variants

_SYSTEM_PROMPT = (
    "You are an offline document assistant. Answer the user's QUESTION using ONLY the "
    "CONTEXT in the user message.\n"
    "\n"
    "Pick the answer style from the CONTEXT you actually use:\n"
    "- NUMERIC answer from a TABLE or CHART: line 1 is the table/chart name; then one "
    "'key: value' per line using the EXACT numbers, units and labels from the context. Give every "
    "key:value pair the question asks for; do not add commentary around the numbers.\n"
    "- TEXT answer from prose: write a COMPLETE, well-formed answer in clear sentences, using ALL the "
    "relevant information in the context. Be thorough and specific, but stay strictly within the "
    "context — do not add outside knowledge.\n"
    "- End the answer with a final line: (page N)  -- the 'page:' number of the CONTEXT block you used.\n"
    "- If the CONTEXT does not contain the answer, reply with exactly 'Not found in the document.' "
    "(no page line). Never guess from unrelated context.\n"
    "\n"
    "Hard rules:\n"
    "- Copy numbers, units, currency symbols and labels EXACTLY; never round, reformat, compute, or "
    "invent a value.\n"
    "- If several tables mention the metric, answer from the ONE whose title most exactly names the "
    "metric asked (it appears first in the CONTEXT). Prefer the reported figure over an 'adjusted' "
    "variant unless the question asks for adjusted. Do not mix values from different tables.\n"
    "- When the QUESTION names a specific year or period (e.g. FY25, FY2025, 2024-25), report ONLY "
    "that period's value, read by its column LABEL. FY25, FY2025 and 2024-25 all mean the same year; "
    "never substitute a different year's value. When no year is named, report the most recent fiscal "
    "year present (highest FY number / latest year-ended), labelled with that period.\n"
    "- Put each label and its value on ONE line as 'label: value'; never split them across lines.\n"
    "- Answer ONLY from the CONTEXT; never use outside knowledge or fabricate facts.\n"
    "- Do NOT write 'source:', the document name, or any citation other than the final (page N)."
)

_CLIENT = None
_CLIENT_HOST: Optional[str] = None


def _host() -> str:
    host = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    # 0.0.0.0 is a server bind address, not a connectable client target
    return host.replace("0.0.0.0", "127.0.0.1")


def _timeout() -> float:
    return float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))


def _num_ctx() -> int:
    return int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _import_ollama():
    import ollama  # noqa: PLC0415 - lazy so the module imports without the dep

    return ollama


def _get_client(ollama_mod):
    global _CLIENT, _CLIENT_HOST
    host = _host()
    if _CLIENT is None or _CLIENT_HOST != host:
        _CLIENT = ollama_mod.Client(host=host, timeout=_timeout())
        _CLIENT_HOST = host
    return _CLIENT


def _content(response: Any) -> str:
    try:
        return response["message"]["content"]
    except (TypeError, KeyError, IndexError):
        pass
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content
    return str(response)


def _strip_think(text: str) -> str:
    cleaned = _THINK_RE.sub("", text)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


class ChatEngine:
    """Answer questions about a single PageIndex store, grounded in its content."""

    def __init__(
        self,
        store: Union[str, Path, PageIndex],
        model: Optional[str] = None,
        top_k: int = 4,
        max_context_chars: int = 12000,
        temperature: float = 0.0,
    ):
        if isinstance(store, (str, Path)):
            self.store = PageIndex.load(str(store))
        else:
            self.store = store
        self.model = (model or os.getenv("OLLAMA_CHAT_MODEL") or DEFAULT_MODEL).strip()
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.temperature = temperature

    def _has_period(self, node: Dict[str, Any], periods: set) -> bool:
        data = (node.get("metadata") or {}).get("data")
        hay = ((node.get("content") or "") + " " + (str(data) if data else "")).lower().replace(" ", "")
        # `periods` is a set of canonical variant-sets; the node matches a period if ANY of its
        # spellings is present (so FY25 also matches '2024-25' / 'year ended March 31, 2025').
        return any(any(v in hay for v in variants) for variants in periods)

    def _title_terms(self, node: Dict[str, Any]) -> set:
        data = (node.get("metadata") or {}).get("data") or {}
        return set(self.store._doc_terms(str(data.get("title") or node.get("title") or "")))

    def _title_overlap(self, node: Dict[str, Any], query_terms: set) -> int:
        """How many query terms the table's TITLE matches (with an 'adjusted' penalty)."""
        title_terms = self._title_terms(node)
        if not title_terms:
            return 0
        score = len(title_terms & query_terms)
        data = (node.get("metadata") or {}).get("data") or {}
        low = str(data.get("title") or node.get("title") or "").lower()
        # Prefer reported over non-reported variants. The doc marks adjusted metrics with '#' or a
        # leading 'Adjusted ...'; do NOT penalise 'Earning per share (adjusted for bonus/split)',
        # which is the standard EPS, so only penalise '#' or titles that START with 'adjusted'.
        if "#" in low or low.strip().startswith("adjusted"):
            score -= 1
        return score

    @staticmethod
    def _chart_points(data: Dict[str, Any]) -> int:
        """Number of plotted data points that carry an actual VALUE.

        The stored n_points counts label-only points too, so process/flow diagrams misread as
        charts (each step a value-less point) report n_points>=1; we count only valued points."""
        n = 0
        for s in data.get("series") or data.get("data_points") or data.get("values") or []:
            if not isinstance(s, dict):
                if s not in (None, "", "null"):
                    n += 1
                continue
            for p in s.get("points") or s.get("data") or []:
                val = p.get("value") if isinstance(p, dict) else p
                if val not in (None, "", "null"):
                    n += 1
        return n

    def _is_empty_chart(self, node: Dict[str, Any]) -> bool:
        """A 'chart' node carrying no data points (a process/flow diagram misread as a chart)."""
        if node.get("node_type") != "chart":
            return False
        return self._chart_points((node.get("metadata") or {}).get("data") or {}) < 1

    @staticmethod
    def _node_conf(node: Dict[str, Any]) -> float:
        try:
            return float((node.get("metadata") or {}).get("confidence"))
        except (TypeError, ValueError):
            return 0.0

    def retrieve(
        self,
        question: str,
        document_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 over a wide pool, then surface the metric's clean table without displacing text.

        Short metric tables (a title + a row of bare numbers) score low in BM25, so for a PERIOD
        question we inject the period-matching exact-title table and float period nodes to the top.
        For a NON-period question we are conservative: BM25 order is preserved and only a table whose
        full multi-word title is literally named by the query is fronted — so descriptive questions
        ('main therapeutic areas') and wrong-metric synonyms ('total revenue' vs 'Total income') do
        NOT get hijacked by a tangential table. Exact matching uses CORE terms (raw + acronym only),
        never semantic synonyms. False-positive empty chart nodes are dropped.
        """
        k = top_k or self.top_k
        pool = max(k, 40)
        nodes = self.store.search_keyword(
            question,
            limit=pool,
            node_types=_CONTEXT_TYPES,
            document_name=document_name,
        )
        nodes = [n for n in nodes if not self._is_empty_chart(n)]
        if len(nodes) < 2:
            return nodes[:k]

        periods = {frozenset(_period_variants(p)) for p in _PERIOD_RE.findall(question)}
        core = set(self.store._core_terms(question))
        # BM25 position of each pooled node = its keyword-relevance rank (search_keyword is sorted).
        bm25_rank = {n.get("id"): i for i, n in enumerate(nodes)}

        def names_metric(node: Dict[str, Any]) -> bool:
            """The query literally names this table's whole (multi-word) metric title."""
            tterms = self._title_terms(node)
            return bool(tterms) and tterms <= core and len(tterms) >= 2

        def strongly_titled(node: Dict[str, Any]) -> bool:
            tterms = self._title_terms(node)
            return bool(tterms) and (len(tterms & core) >= 2 or names_metric(node))

        def line_match(node: Dict[str, Any]) -> bool:
            """A table with a single ROW/line containing all the query's core terms — i.e. the query
            names a line item inside the table (e.g. 'revenue from operations' in a P&L). Line-scoped
            so it is precise: it won't fire on descriptive questions or terms scattered across a page."""
            if node.get("node_type") != "table" or len(core) < 2:
                return False
            for line in (node.get("content") or "").lower().splitlines():
                if all(t in line for t in core):
                    return True
            return False

        # Inject exact-title / line-item tables BM25 may have ranked below the pool. For period
        # queries, a strong title or line-item match that also carries the period; for non-period
        # queries, only tables whose full title or a whole line the query names (keeps unrelated
        # tables out of the candidate set).
        have = {n.get("id") for n in nodes}
        for nid, n in self.store.nodes.items():
            if n["node_type"] != "table" or nid in have:
                continue
            if document_name and n.get("document_name") != document_name:
                continue
            if periods:
                inject = (strongly_titled(n) or line_match(n)) and self._has_period(n, periods)
            else:
                inject = names_metric(n) or line_match(n)
            if inject:
                nodes.append(n)
                have.add(nid)
                bm25_rank.setdefault(nid, len(bm25_rank))

        # A text/section node the query strongly NAMES (>=2 core title terms) that BM25 also ranked at
        # the very top is a genuine prose answer (e.g. 'Division-wise performance'). Such a heading is
        # often just a title, with the figures in sibling sub-sections under the same parent page; pull
        # those siblings in and keep them adjacent so the prose answer arrives complete.
        TEXT_PROTECT = 3
        pulled: Dict[str, tuple] = {}  # child id -> (heading bm25 rank, sibling order): tier-0, adjacent
        for n in list(nodes):
            if n.get("node_type") not in ("text", "section"):
                continue
            hr = bm25_rank.get(n.get("id"), 99)
            if hr >= TEXT_PROTECT or not strongly_titled(n):
                continue
            # Only a title-only "divider" heading (its data lives in sibling sub-sections) pulls
            # siblings; a content-bearing note already holds its own answer, so pulling its same-page
            # neighbours would just add unrelated notes.
            if len((n.get("content") or "").strip()) > len((n.get("title") or "").strip()) + 40:
                continue
            parent = self.store.nodes.get(n.get("parent_id"))
            if not parent:
                continue
            for order, cid in enumerate(parent.get("children") or []):
                child = self.store.nodes.get(cid)
                if not child or cid == n.get("id"):
                    continue
                if child.get("node_type") in ("text", "section") and child.get("page_number") == n.get("page_number"):
                    if cid not in have:
                        nodes.append(child)
                        have.add(cid)
                    bm25_rank.setdefault(cid, hr)
                    pulled.setdefault(cid, (hr, order))

        def _key(item):
            idx, node = item
            nid = node.get("id")
            ntype = node.get("node_type")
            is_table = ntype == "table"
            is_text = ntype in ("text", "section")
            # A strongly-titled text/section node BM25 ranked at the very top (or a sibling pulled in
            # beside it) is a genuine prose answer; it leads, then the period structured nodes follow
            # in their original order (so numeric questions keep fronting their clean metric table).
            protected_text = (
                is_text and bm25_rank.get(nid, 99) < TEXT_PROTECT and strongly_titled(node)
            ) or nid in pulled
            sib = pulled.get(nid, (bm25_rank.get(nid, idx), -1))  # (heading BM25 rank, sibling order)
            if periods:
                if protected_text:  # tier 0: the prose answer and its siblings, kept together
                    return (0, sib[0], sib[1])
                # tier 1: period structured nodes; tier 2: the rest by BM25.
                hit = self._has_period(node, periods)
                overlap = self._title_overlap(node, core) if is_table else 0
                # Rank period nodes first by how many of the query's core terms they actually contain
                # (coverage), then let the strongest BM25 matches (top k) lead, then the original
                # table-first / title-overlap fronting. Coverage rescues big, garbled financial-statement
                # nodes (the consolidated P&L / balance sheet, whose header text is mojibake but whose body
                # holds every line item) that BM25's length penalty otherwise buries under small dense tables.
                content_l = (node.get("content") or "").lower()
                coverage = sum(1 for t in core if t in content_l)
                return (1 if hit else 2, -coverage, 0 if idx < k else 1, 0 if is_table else 1, -overlap, idx)
            # no period: BM25 order; front a structured node the query names, a line item, or a
            # protected prose answer (its siblings kept adjacent).
            is_struct = ntype in ("table", "chart", "kpi")
            fronted = (is_struct and names_metric(node)) or line_match(node) or protected_text
            return (0 if fronted else 1, sib[0] if protected_text else idx, sib[1] if protected_text else 0)

        bm25_order = list(nodes)  # BM25 relevance order, before the rerank below
        ranked = [node for _, node in sorted(enumerate(nodes), key=_key)]
        result = ranked[:k]
        # Safety net: if the rerank still dropped a top-3 BM25 match, add it back (no displacement) so a
        # strongly-matching node is never lost outright from the context.
        present = {id(n) for n in result}
        for node in bm25_order[:3]:
            if id(node) not in present:
                result.append(node)
                present.add(id(node))
        return result

    @staticmethod
    def _column_header(col: Any) -> str:
        """Normalize a column header; vision tables store them as dicts / stringified dicts."""
        if isinstance(col, dict):
            return str(col.get("header") or col.get("name") or col.get("label") or "").strip()
        if isinstance(col, str):
            text = col.strip()
            if text.startswith("{") and ("header" in text or "row_header" in text):
                try:
                    obj = ast.literal_eval(text)
                    if isinstance(obj, dict):
                        return str(obj.get("header") or obj.get("row_header") or obj.get("name") or "").strip()
                except (ValueError, SyntaxError):
                    pass
            return text
        return "" if col is None else str(col)

    @staticmethod
    def _is_period(header: str) -> bool:
        return bool(re.fullmatch(r"(?i)\s*(fy ?\d{2,4}|20\d{2}([-/]\d{2,4})?|\d{4}[-/]\d{2,4})\s*", header or ""))

    @staticmethod
    def _render_table(node: Dict[str, Any], data: Dict[str, Any]) -> str:
        name = data.get("title") or node.get("title") or "Table"
        headers = [ChatEngine._column_header(c) for c in (data.get("columns") or [])]

        rows: List[List[str]] = []
        for row in (data.get("rows") or []):
            if isinstance(row, dict):
                cells = list(row.values())
            elif isinstance(row, (list, tuple)):
                cells = list(row)
            else:
                cells = [row]
            rows.append(["" if c is None else str(c) for c in cells])

        # Detect a leading label column: header empty/non-period AND every row's first cell is text.
        has_label_col = False
        if headers and rows:
            first_textual = all((r[0] and not re.search(r"\d", r[0])) for r in rows if r)
            has_label_col = (not headers[0] or not ChatEngine._is_period(headers[0])) and first_textual
        value_headers = headers[1:] if has_label_col else headers

        lines = [str(name)]
        budget = 4000
        for r in rows:
            label = r[0] if has_label_col else ""
            cells = r[1:] if has_label_col else r
            if value_headers:
                pairs = " | ".join(f"{h}: {v}" for h, v in zip(value_headers, cells) if (h or v))
            else:  # no headers at all -> fall back to plain cells
                pairs = " | ".join(v for v in cells if v)
            line = f"{label}: {pairs}" if label else pairs
            if line:
                lines.append(line)
            if sum(len(x) + 1 for x in lines) > budget:
                break

        if data.get("units"):
            lines.append(f"units: {data['units']}")
        for fn in (data.get("footnotes") or [])[:3]:
            lines.append(fn.get("text", "") if isinstance(fn, dict) else str(fn))
        return "\n".join(lines)[:budget]

    @staticmethod
    def _render_chart(node: Dict[str, Any], data: Dict[str, Any]) -> str:
        name = data.get("title") or node.get("title") or "Chart"
        lines = [str(name)]
        if data.get("chart_type"):
            lines.append(f"chart_type: {data['chart_type']}")
        for series in (data.get("series") or [])[:6]:
            if not isinstance(series, dict):
                continue
            if series.get("name"):
                lines.append(str(series["name"]))
            for point in (series.get("points") or [])[:20]:
                if isinstance(point, dict):
                    label, value = point.get("label"), point.get("value")
                    if label is not None or value is not None:
                        lines.append(f"{label}: {value}")
        return "\n".join(lines)[:1200]

    @staticmethod
    def _full_table_markdown(node: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Render the COMPLETE table (every row and column) as a markdown table — no truncation."""
        name = data.get("title") or node.get("title") or "Table"
        headers = [ChatEngine._column_header(c) for c in (data.get("columns") or [])]
        rows: List[List[str]] = []
        for row in (data.get("rows") or []):
            cells = list(row.values()) if isinstance(row, dict) else (list(row) if isinstance(row, (list, tuple)) else [row])
            rows.append(["" if c is None else str(c) for c in cells])
        width = max([len(headers)] + [len(r) for r in rows] or [0])
        if not width:
            return str(name)
        headers = (headers + [""] * (width - len(headers)))[:width]
        lines = [str(name)]
        if data.get("units"):
            lines.append(f"(units: {data['units']})")
        lines.append("| " + " | ".join(h or " " for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
        for r in rows:
            r = (r + [""] * (width - len(r)))[:width]
            lines.append("| " + " | ".join(c or " " for c in r) + " |")
        if node.get("page_number") is not None:
            lines.append(f"(page {node['page_number']})")
        return "\n".join(lines)

    @staticmethod
    def _full_table_keyvalue(node: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Render the COMPLETE table as proper 'key: value' lines — every cell, no truncation."""
        name = data.get("title") or node.get("title") or "Table"
        headers = [ChatEngine._column_header(c) for c in (data.get("columns") or [])]
        rows: List[List[str]] = []
        for row in (data.get("rows") or []):
            cells = list(row.values()) if isinstance(row, dict) else (list(row) if isinstance(row, (list, tuple)) else [row])
            rows.append(["" if c is None else str(c).strip() for c in cells])

        has_label = False
        if headers and rows:
            textual = sum(1 for r in rows if r and r[0] and not re.search(r"\d", str(r[0])))
            has_label = (not headers[0] or not ChatEngine._is_period(headers[0])) and textual >= 0.5 * len(rows)
        value_headers = headers[1:] if has_label else headers

        lines = [str(name)]
        if data.get("units"):
            lines.append(f"(units: {data['units']})")
        for r in rows:
            label = r[0] if has_label else ""
            cells = r[1:] if has_label else r
            if value_headers:
                for h, v in zip(value_headers, cells):
                    if v == "" and not h:
                        continue
                    key = (f"{label} {h}".strip() if label else h) or "value"
                    lines.append(f"{key}: {v}")
            else:
                joined = " | ".join(c for c in cells if c)
                lines.append(f"{label}: {joined}" if label else joined)
        if node.get("page_number") is not None:
            lines.append(f"(page {node['page_number']})")
        return "\n".join(lines)

    def _full_table_answer(self, node: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Cleanest full-table rendering by shape: a single-row metric table reads well as
        'key: value' lines; a multi-row grid (e.g. a 10-year highlights table) reads far better
        as an aligned table than as hundreds of 'RowLabel Col: value' lines."""
        if len(data.get("rows") or []) <= 1:
            return self._full_table_keyvalue(node, data)
        return self._full_table_markdown(node, data)

    def _best_table(self, question: str, document_name: Optional[str] = None):
        """The table to use for a 'full table' request. Strips request boilerplate ('give me the
        entire … table') so only the metric name drives retrieval, then returns the first table
        from the smart retriever (its line/title match fronts the right table above fact nodes)."""
        clean = _TABLE_REQUEST_NOISE.sub(" ", question).strip() or question
        for node in self.retrieve(clean, document_name=document_name):
            if node.get("node_type") == "table":
                return node
        hits = self.store.search_keyword(clean, limit=self.top_k, node_types="table", document_name=document_name)
        return hits[0] if hits else None

    def get_table(self, question: str, document_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the ENTIRE best-matching table for a question (deterministic, no LLM).

        Returns {title, page_number, units, columns, rows, records, markdown, keyvalue} or {error}.
        """
        node = self._best_table(question, document_name)
        if node is None:
            return {"error": "no table matched the question"}
        data = (node.get("metadata") or {}).get("data") or {}
        headers = [self._column_header(c) for c in (data.get("columns") or [])]
        rows = [
            (list(r.values()) if isinstance(r, dict) else list(r) if isinstance(r, (list, tuple)) else [r])
            for r in (data.get("rows") or [])
        ]
        return {
            "title": data.get("title") or node.get("title"),
            "page_number": node.get("page_number"),
            "units": data.get("units"),
            "columns": headers,
            "rows": rows,
            "records": [dict(zip(headers, r)) for r in rows],
            "markdown": self._full_table_markdown(node, data),
            "keyvalue": self._full_table_keyvalue(node, data),
        }

    def _format_context(self, nodes: List[Dict[str, Any]]):
        blocks: List[str] = []
        used: List[Dict[str, Any]] = []
        seen: set = set()
        total = 0
        for node in nodes:
            page = node.get("page_number")
            source = node.get("document_name") or node.get("source") or "unknown"
            ntype = node.get("node_type")
            data = (node.get("metadata") or {}).get("data")
            rendered = ""
            if ntype == "table" and isinstance(data, dict):
                rendered = self._render_table(node, data)
            elif ntype == "chart" and isinstance(data, dict):
                rendered = self._render_chart(node, data)
            if not rendered.strip():
                rendered = (node.get("content") or "").strip()
            # facts carry a trailing "on page N"; drop it so the page is cited once (final line), not per fact.
            rendered = re.sub(r"\s+on page\s+\d+\s*$", "", rendered, flags=re.IGNORECASE).strip()
            # skip a node that duplicates one already included (same shareholding fact on two pages, etc.).
            if ntype == "fact" and isinstance(data, dict):
                yr = re.search(r"(?:19|20)\d{2}|fy\s?\d{2,4}", str(data.get("period") or "").lower())
                sig = re.sub(r"\W+", "", f"{data.get('metric')}|{data.get('value')}|{yr.group(0) if yr else ''}".lower())
            else:
                sig = re.sub(r"\W+", "", rendered.lower())[:300]
            if sig and sig in seen:
                continue
            seen.add(sig)
            header = f"[{len(used) + 1}] {ntype} | source: {source}" + (f", page: {page}" if page is not None else "")
            block = f"{header}\n{rendered}"
            if total + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            used.append(node)
            total += len(block)
        return "\n\n".join(blocks), used

    @staticmethod
    def _sources(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for index, node in enumerate(nodes, 1):
            sources.append(
                {
                    "n": index,
                    "node_id": node.get("id"),
                    "node_type": node.get("node_type"),
                    "title": node.get("title"),
                    "document_name": node.get("document_name") or node.get("source"),
                    "page_number": node.get("page_number"),
                }
            )
        return sources

    def _call_llm(self, context: str, question: str) -> str:
        ollama_mod = _import_ollama()
        client = _get_client(ollama_mod)
        user_message = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using only the CONTEXT above, following the formatting rules exactly."
        )
        options = {"temperature": self.temperature, "num_ctx": _num_ctx()}
        num_gpu = os.getenv("OLLAMA_NUM_GPU")
        if num_gpu not in (None, ""):
            try:
                options["num_gpu"] = int(num_gpu)
            except ValueError:
                pass
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options=options,
        )
        return _strip_think(_content(response))

    @staticmethod
    def _is_nonanswer(answer: str) -> bool:
        """True when the model did not really answer: empty, the 'Not found' guardrail, or a degenerate
        repetition-loop / meta-commentary breakdown (the local model sometimes loops on messy context)."""
        a = (answer or "").strip().lower()
        if not a:
            return True
        if a.startswith("not found in the document"):
            return True
        meta = ("loop of repetition", "your message", "as an ai", "i cannot answer",
                "technical issue", "i am unable", "i'm unable", "repeated text", "unable to answer")
        return any(m in a for m in meta) and "(page" not in a

    def _top_result_answer(self, used: List[Dict[str, Any]]) -> str:
        """Surface the top retrieved result as the answer, so the user gets the most relevant retrieved
        content instead of 'Not found'. Walks the ranked list and returns the first node that has real
        content (skips an empty/degenerate top node). Reuses the renderers; no pipeline change."""
        for node in used:
            data = (node.get("metadata") or {}).get("data")
            ntype = node.get("node_type")
            if ntype == "table" and isinstance(data, dict):
                body = self._render_table(node, data)
            elif ntype == "chart" and isinstance(data, dict):
                body = self._render_chart(node, data)
            else:
                body = node.get("content") or ""
            body = body.strip()
            if not body:
                continue
            page = node.get("page_number")
            if page is not None and "(page" not in body[-16:].lower():
                body = f"{body}\n(page {page})"
            return body
        return ""

    def _ground_page(self, answer: str, used: List[Dict[str, Any]]) -> str:
        """Rewrite the trailing '(page N)' citation to the page of the retrieved SOURCE that actually
        contains the answer. The model picks the page from a context block and often guesses the wrong
        one; here the page is grounded in the source whose content best matches the answer (distinctive
        numbers dominate, words back it up). Conservative: only overrides when the model's cited page does
        NOT hold the answer but another retrieved page does — otherwise the model's citation is kept."""
        if not used:
            return answer
        m = re.search(r"\(\s*page\s+(\d+)", answer, re.IGNORECASE)
        claimed = int(m.group(1)) if m else None
        body = re.sub(r"\(\s*page[^)]*\)", " ", answer, flags=re.IGNORECASE)
        nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d+|\b\d{2,}\b", body) if len(n.replace(",", "")) >= 2]
        words = set(re.findall(r"[a-z]{4,}", body.lower()))

        def node_score(node: Dict[str, Any]) -> int:
            text = (node.get("content") or "").lower()
            text_ns = text.replace(",", "")
            return sum(5 if len(n) >= 4 else 1 for n in nums if n in text_ns) + sum(1 for w in words if w in text)

        page_score: Dict[Any, int] = {}
        for node in used:
            page = node.get("page_number")
            if page is None:
                continue
            page_score[page] = max(page_score.get(page, 0), node_score(node))
        if not page_score:
            return answer
        best_page = max(page_score, key=page_score.get)
        best_score = page_score[best_page]
        if best_score <= 0 or page_score.get(claimed, -1) >= best_score:
            return answer  # the cited page already holds the answer (or nothing matched) -> leave it
        stripped = re.sub(r"\s*\(\s*page[^)]*\)\s*$", "", answer, flags=re.IGNORECASE).rstrip()
        return f"{stripped}\n(page {best_page})"

    def ask(
        self,
        question: str,
        document_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Answer a question grounded in retrieved PageIndex context.

        ``top_k`` overrides how many ranked nodes are retrieved as context for THIS question
        (falls back to the engine default when None).
        """
        nodes = self.retrieve(question, document_name=document_name, top_k=top_k)
        if not nodes:
            logger.info("chat: no relevant nodes for question")
            return {"answer": _NO_CONTEXT_MSG, "sources": [], "context": "", "model": self.model}

        # "show me the full/entire X table" -> return the whole table as key:value (no LLM).
        if _FULL_TABLE_RE.search(question):
            node = self._best_table(question, document_name)
            if node is not None:
                data = (node.get("metadata") or {}).get("data") or {}
                return {
                    "answer": self._full_table_answer(node, data),
                    "sources": self._sources([node]),
                    "context": "",
                    "model": self.model,
                    "mode": "full_table",
                }

        context, used = self._format_context(nodes)
        sources = self._sources(used)

        try:
            answer = self._call_llm(context, question)
        except ImportError:
            logger.error("chat: ollama package not installed")
            return {
                "answer": "[error] ollama package not installed; run 'pip install ollama'",
                "sources": sources,
                "context": context,
                "model": self.model,
                "error": "ImportError",
            }
        except Exception as exc:  # noqa: BLE001 - report API/connection failure, do not crash
            name = type(exc).__name__
            if isinstance(exc, (ConnectionError, TimeoutError)) or any(
                token in name for token in ("Connect", "Timeout", "Transport")
            ):
                message = f"could not reach Ollama at {_host()}: {exc}"
            else:
                message = f"ollama request failed: {exc}"
            logger.exception("chat: LLM call failed")
            return {
                "answer": f"[error] {message}",
                "sources": sources,
                "context": context,
                "model": self.model,
                "error": name,
            }

        # When the model can't give a real answer (empty / "Not found" / a degenerate loop), surface the
        # top retrieved result instead of telling the user nothing. Only falls back if something was
        # retrieved; a genuinely empty retrieval still returns the no-context message.
        if self._is_nonanswer(answer):
            fallback = self._top_result_answer(used)
            answer = fallback or (answer.strip() and "Not found in the document.") or _NO_CONTEXT_MSG
        else:
            answer = self._ground_page(answer, used)  # cite the page that actually holds the answer
        return {"answer": answer, "sources": sources, "context": context, "model": self.model}


def _repl(store_path: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = ChatEngine(store_path)
    print(f"Loaded {len(engine.store)} nodes from {store_path} (model: {engine.model}).")
    print("Ask a question (blank line or Ctrl-C to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        result = engine.ask(question)
        print("\n" + result["answer"])
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                page = source["page_number"]
                tag = f"p.{page}" if page is not None else "p.?"
                print(f"  [{source['n']}] {source['document_name']} {tag} ({source['node_type']})")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m chatbot.chat_engine <pageindex.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(_repl(sys.argv[1]))
