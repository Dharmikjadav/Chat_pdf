"""Decompose validated table/chart/KPI nodes into searchable (metric, period, value) facts.

Each fact becomes a small ``fact`` PageIndex node whose content is a natural sentence
("Revenue from operations in FY16 was 120 (in million) on page 5"), so BM25 can match a
specific metric+period directly instead of only the flattened whole-table content.
"""

import ast
import re
from typing import Any, Dict, List, Optional

__all__ = ["generate_facts"]

_PERIOD_RE = re.compile(r"(?i)^\s*(fy ?\d{2,4}|20\d{2}([-/]\d{2,4})?|\d{4}[-/]\d{2,4})\s*$")
_MAX_FACTS_PER_NODE = 80
_LATEST_PERIODS = 8


def _header(col: Any) -> str:
    if isinstance(col, dict):
        return str(col.get("header") or col.get("name") or col.get("label") or "").strip()
    text = str(col or "").strip()
    if text.startswith("{") and ("header" in text or "row_header" in text):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return str(obj.get("header") or obj.get("row_header") or obj.get("name") or "").strip()
        except (ValueError, SyntaxError):
            pass
    return text


def _is_period(h: str) -> bool:
    return bool(_PERIOD_RE.match(h or ""))


def _fy_num(h: str) -> int:
    m = re.search(r"(\d{2,4})", h or "")
    if not m:
        return -1
    n = int(m.group(1))
    return n if n >= 1900 else 2000 + n


def _row_cells(row: Any) -> List[str]:
    if isinstance(row, dict):
        return [("" if v is None else str(v).strip()) for v in row.values()]
    if isinstance(row, (list, tuple)):
        return [("" if c is None else str(c).strip()) for c in row]
    return [("" if row is None else str(row).strip())]


def _fact_text(metric: str, period: Optional[str], value: str, unit: Optional[str], page: Any) -> str:
    parts = [str(metric).strip()]
    if period:
        parts.append(f"in {period}")
    parts.append(f"was {value}")
    if unit:
        u = str(unit).strip()
        if u.startswith("(") and u.endswith(")"):
            u = u[1:-1].strip()
        if u:
            parts.append(f"({u})")
    if page:
        parts.append(f"on page {page}")
    return " ".join(p for p in parts if p).strip()


def _table_facts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    headers = [_header(c) for c in (data.get("columns") or [])]
    rows = [_row_cells(r) for r in (data.get("rows") or [])]
    if not headers or not rows:
        return []

    first_textual = all((r and r[0] and not re.search(r"\d", r[0])) for r in rows if r)
    has_label = (not headers[0] or not _is_period(headers[0])) and first_textual
    unit = data.get("units")

    period_idx = sorted(
        [(i, h) for i, h in enumerate(headers) if _is_period(h)],
        key=lambda ih: _fy_num(ih[1]),
        reverse=True,
    )[:_LATEST_PERIODS]

    facts: List[Dict[str, Any]] = []
    if period_idx:
        for r in rows:
            metric = r[0].strip() if has_label and r else (data.get("title") or "")
            if not metric:
                continue
            for i, h in period_idx:
                if i < len(r) and r[i] and re.search(r"\d", r[i]):
                    facts.append({"metric": metric, "period": h.strip(), "value": r[i], "unit": unit})
                    if len(facts) >= _MAX_FACTS_PER_NODE:
                        return facts
    elif has_label:
        value_headers = headers[1:]
        for r in rows:
            metric = r[0].strip()
            if not metric:
                continue
            for h, v in zip(value_headers, r[1:]):
                if v and re.search(r"\d", v):
                    facts.append({"metric": metric, "period": h.strip() or None, "value": v, "unit": unit})
                    break
            if len(facts) >= _MAX_FACTS_PER_NODE:
                break
    return facts


def _chart_facts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    y = data.get("y_axis")
    unit = y.get("unit") if isinstance(y, dict) else data.get("units")
    title = data.get("title") or "Chart"
    facts: List[Dict[str, Any]] = []
    for s in data.get("series") or []:
        sname = s.get("name") if isinstance(s, dict) else None
        for p in (s.get("points") if isinstance(s, dict) else None) or []:
            if not isinstance(p, dict) or p.get("value") in (None, "", "null"):
                continue
            facts.append({
                "metric": str(sname or title),
                "period": str(p.get("label")) if p.get("label") else None,
                "value": str(p.get("value")),
                "unit": p.get("unit") or unit,
            })
            if len(facts) >= _MAX_FACTS_PER_NODE:
                return facts
    return facts


def _kpi_facts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for k in data.get("kpis") or []:
        if not isinstance(k, dict) or not k.get("metric") or k.get("value") in (None, ""):
            continue
        facts.append({
            "metric": str(k.get("metric")),
            "period": k.get("period"),
            "value": str(k.get("value")),
            "unit": k.get("unit"),
            "context": k.get("context"),
            "confidence": k.get("confidence"),
        })
    return facts


def generate_facts(
    node: Dict[str, Any],
    document_id: Optional[str] = None,
    company: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a list of fact dicts for one validated table/chart/kpi node."""
    ntype = node.get("type") or node.get("node_type")
    data = node.get("data") or {}
    page = node.get("page_number")
    conf = node.get("confidence")

    if ntype == "table":
        raw = _table_facts(data)
    elif ntype == "chart":
        raw = _chart_facts(data)
    elif ntype == "kpi":
        raw = _kpi_facts(data)
    else:
        return []

    facts: List[Dict[str, Any]] = []
    for f in raw:
        facts.append({
            "metric": f["metric"],
            "period": f.get("period"),
            "value": f["value"],
            "unit": f.get("unit"),
            "page": page,
            "source_type": ntype,
            "confidence": f.get("confidence", conf),
            "context": f.get("context"),
            "document_id": document_id,
            "company": company,
            "text": _fact_text(f["metric"], f.get("period"), f["value"], f.get("unit"), page),
        })
    return facts
