"""Deterministic parser for financial-statement text nodes.

Marker/OCR emits the Statement of Profit and Loss / Balance Sheet as newline-separated
text: a line item, an optional note number, then one value per reporting period, e.g.

    Revenue from operations
    30
     525,784.4
     484,968.5

These line items are not individually retrievable as flattened text, so this module
turns such a text node into a structured table (Particulars + one column per period) so
line-item questions ("revenue from operations") surface and render cleanly. It reuses the
ALREADY-extracted text — no vision, no PDF, fully offline and deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# A pure numeric token: optional parens (negatives), digits/commas, optional decimals.
_NUM = re.compile(r"^\(?\s*[\d,]+\.?\d*\s*\)?$")
_ROMAN = re.compile(r"^\([ivxlcdm]+\)$", re.IGNORECASE)
_YEAR = re.compile(r"(20\d{2})")
_STATEMENT_TITLES = (
    "statement of profit and loss",
    "balance sheet",
    "statement of cash flow",
    "statement of changes in equity",
)


def _is_value(line: str) -> bool:
    """A reported figure (has a comma, decimal or parentheses) — not a bare note number."""
    return bool(_NUM.match(line)) and any(c in line for c in ",.(")


def _is_note(line: str) -> bool:
    """A bare small integer note reference (e.g. '30', '34') — skip it."""
    return bool(_NUM.match(line)) and not any(c in line for c in ",.(")


def is_statement_text(node: Dict[str, Any]) -> Optional[str]:
    """Return the statement title if this text node is a financial statement, else None."""
    if node.get("node_type") != "text":
        return None
    content = node.get("content") or ""
    head = "\n".join(content.splitlines()[:4]).lower()
    for t in _STATEMENT_TITLES:
        if t in head:
            return content.splitlines()[0].strip()
    return None


def parse_statement(content: str) -> Optional[Dict[str, Any]]:
    """Parse statement text into {title, unit, columns, rows}; None if it doesn't look like one."""
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return None
    title = lines[0]
    if not any(t in title.lower() for t in _STATEMENT_TITLES):
        return None

    # period columns: first two distinct years in the header block → FY label (FY<yy>).
    # header_end marks the last such header line so row parsing skips the title/column block.
    years: List[str] = []
    unit = ""
    header_end = 0
    for i, ln in enumerate(lines[:16]):
        if any(u in ln.lower() for u in ("million", "billion", "crore", "lakh")):
            unit = ln.strip("` ")
        m = _YEAR.search(ln)
        if m:
            fy = "FY" + m.group(1)[2:]
            if fy not in years:
                years.append(fy)
            header_end = i
    if len(years) < 2:
        return None
    ncols = min(len(years), 2)
    cols = years[:ncols]

    rows: List[List[str]] = []
    label_parts: List[str] = []
    values: List[str] = []

    def flush():
        if label_parts and len(values) >= ncols:
            label = " ".join(label_parts).strip()
            if label and not _ROMAN.match(label):
                rows.append([label] + values[:ncols])

    for ln in lines[header_end + 1:]:
        if _is_value(ln):
            values.append(ln.strip().replace("(", "-").replace(")", ""))
            continue
        if _is_note(ln):
            continue  # note reference between label and values
        if _ROMAN.match(ln):
            continue  # section marker (I)/(II)…
        # a text line: it begins a new row, so flush the previous one first
        if values:
            flush()
            label_parts, values = [], []
        label_parts.append(ln)
    flush()

    # keep only rows that look like real line items (drop header echoes)
    rows = [r for r in rows if r[0].lower() not in ("particulars", "notes") and len(r[0]) > 2]
    if len(rows) < 3:
        return None

    # Indian statements list the current year FIRST; normalize period columns to oldest -> newest
    # so that "rightmost column = latest year" holds consistently with every other table.
    def _fyn(c):
        m = re.search(r"(\d{2,4})", c)
        if not m:
            return 0
        v = int(m.group(1))
        return v if v >= 1900 else 2000 + v

    order = sorted(range(ncols), key=lambda i: _fyn(cols[i]))
    cols = [cols[i] for i in order]
    rows = [[r[0]] + [r[1 + i] for i in order] for r in rows if len(r) >= 1 + ncols]
    return {"type": "table", "title": title, "unit": unit, "columns": ["Particulars"] + cols, "rows": rows}


def _searchable_text(parsed: Dict[str, Any]) -> str:
    """Flatten a parsed statement into line-item text so BM25 finds individual rows."""
    cols = parsed["columns"][1:]
    lines = [parsed["title"] + (f" ({parsed['unit']})" if parsed.get("unit") else "")]
    for row in parsed["rows"]:
        pairs = " | ".join(f"{c} {v}" for c, v in zip(cols, row[1:]))
        lines.append(f"{row[0]} — {pairs}")
    return "\n".join(lines)


def enrich_store_with_statements(store) -> int:
    """Add a structured companion TABLE node for each financial-statement TEXT node.

    Idempotent (skips '<id>-stmt' nodes already present). Returns how many were added.
    Reuses already-extracted text, so it is offline, deterministic and safe to re-run.
    """
    added = 0
    for node in list(store.nodes.values()):
        if not is_statement_text(node):
            continue
        parent_id = node.get("parent_id")
        if not parent_id or parent_id not in store.nodes:
            continue
        new_id = f"{node['id']}-stmt"
        if new_id in store.nodes:
            continue
        parsed = parse_statement(node.get("content") or "")
        if not parsed:
            continue
        store.add_table_node(
            parent_id,
            data=parsed,
            content=_searchable_text(parsed),
            title=parsed["title"],
            page_number=node.get("page_number"),
            metadata={"origin": "statement_parser", "source_text": node["id"]},
            node_id=new_id,
        )
        added += 1
    return added
