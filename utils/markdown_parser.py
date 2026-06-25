"""Parse Marker-generated Markdown into text, clean-table, and complex-table nodes."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["parse_markdown", "parse_markdown_file", "markdown_parser_node"]

_PAGE_BRACE_RE = re.compile(r"^\s*\{(\d+)\}-{2,}\s*$")
_PAGE_COMMENT_RE = re.compile(r"^\s*<!--\s*page[:\s]+(\d+)\s*-->\s*$", re.IGNORECASE)
_HTML_TABLE_OPEN_RE = re.compile(r"^\s*<table", re.IGNORECASE)
_SEP_CELL_RE = re.compile(r"^:?-+:?$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SPLIT_CELL_RE = re.compile(r"(?<!\\)\|")


def _slug(name: str) -> str:
    cleaned = _SLUG_RE.sub("-", str(name)).strip("-")
    return cleaned or "doc"


def _node_id(document_name: str, element_type: str, index: int) -> str:
    return f"{_slug(document_name)}-{element_type}-{index:04d}"


def _page_from_marker(line: str) -> Optional[int]:
    match = _PAGE_BRACE_RE.match(line)
    if match:
        return int(match.group(1)) + 1
    match = _PAGE_COMMENT_RE.match(line)
    if match:
        return int(match.group(1))
    return None


def _split_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    parts = _SPLIT_CELL_RE.split(stripped)
    return [part.strip().replace("\\|", "|") for part in parts]


def _is_separator_line(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_row(line)
    if not cells:
        return False
    matched = False
    for cell in cells:
        token = cell.strip()
        if not token or not _SEP_CELL_RE.match(token):
            return False
        matched = True
    return matched


def _looks_like_table_start(lines: List[str], index: int) -> bool:
    line = lines[index]
    if "|" not in line:
        return False
    nxt = lines[index + 1] if index + 1 < len(lines) else ""
    if _is_separator_line(nxt):
        return True
    if "|" in nxt and nxt.strip():
        return True
    return _is_separator_line(line)


def _collect_table(lines: List[str], index: int) -> Tuple[List[str], int]:
    block: List[str] = []
    total = len(lines)
    while index < total:
        line = lines[index]
        if not line.strip():
            break
        if _page_from_marker(line) is not None:
            break
        if "|" not in line and not _is_separator_line(line):
            break
        block.append(line)
        index += 1
    return block, index


def _collect_html_table(lines: List[str], index: int) -> Tuple[List[str], int]:
    block: List[str] = []
    total = len(lines)
    while index < total:
        block.append(lines[index])
        if "</table>" in lines[index].lower():
            index += 1
            break
        index += 1
    return block, index


def _normalize_headers(header: List[str]) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for position, raw in enumerate(header, 1):
        name = raw.strip() if raw and raw.strip() else f"column_{position}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _classify_table(block: List[str]) -> Dict[str, Any]:
    sep_idx = None
    for idx, line in enumerate(block):
        if _is_separator_line(line):
            sep_idx = idx
            break

    if sep_idx is None:
        return {"complex": True, "reason": "missing_separator", "data": {"columns": [], "rows": []}}
    if sep_idx == 0:
        return {"complex": True, "reason": "missing_header", "data": {"columns": [], "rows": []}}

    header_raw = _split_row(block[sep_idx - 1])
    if sep_idx > 1:
        return {"complex": True, "reason": "multi_row_header", "data": {"columns": header_raw, "rows": []}}
    if not header_raw or (len(header_raw) == 1 and header_raw[0] == ""):
        return {"complex": True, "reason": "empty_header", "data": {"columns": header_raw, "rows": []}}

    columns = _normalize_headers(header_raw)
    n_cols = len(columns)
    body = block[sep_idx + 1:]
    records: List[Dict[str, str]] = []

    for line in body:
        cells = _split_row(line)
        if len(cells) > n_cols:
            return {"complex": True, "reason": "ragged_columns", "data": {"columns": columns, "rows": []}}
        if len(cells) < n_cols:
            cells = cells + [""] * (n_cols - len(cells))
        records.append(dict(zip(columns, cells)))

    data = {"columns": columns, "rows": records, "n_rows": len(records), "n_cols": n_cols}
    return {"complex": False, "reason": None, "data": data}


def parse_markdown(
    text: str,
    document_name: str,
    source: str,
    *,
    page_number: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse Markdown text into text_nodes, table_nodes, complex_table_candidates."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    text_nodes: List[Dict[str, Any]] = []
    table_nodes: List[Dict[str, Any]] = []
    complex_table_candidates: List[Dict[str, Any]] = []
    counters = {"text": 0, "table": 0, "complex_table": 0}

    current_page = page_number
    buffer: List[str] = []

    def flush_text() -> None:
        if not buffer:
            return
        raw_text = "\n".join(buffer).strip("\n")
        buffer.clear()
        if not raw_text.strip():
            return
        counters["text"] += 1
        text_nodes.append(
            {
                "id": _node_id(document_name, "text", counters["text"]),
                "document_name": document_name,
                "page_number": current_page,
                "element_type": "text",
                "content": raw_text.strip(),
                "raw_text": raw_text,
                "source": source,
            }
        )

    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]

        marker_page = _page_from_marker(line)
        if marker_page is not None:
            flush_text()
            current_page = marker_page
            index += 1
            continue

        if not line.strip():
            flush_text()
            index += 1
            continue

        if _HTML_TABLE_OPEN_RE.match(line):
            flush_text()
            block, index = _collect_html_table(lines, index)
            raw_md = "\n".join(block)
            counters["complex_table"] += 1
            complex_table_candidates.append(
                {
                    "id": _node_id(document_name, "complex_table", counters["complex_table"]),
                    "document_name": document_name,
                    "page_number": current_page,
                    "element_type": "complex_table",
                    "data": {"columns": [], "rows": [], "reason": "html_table"},
                    "raw_markdown": raw_md,
                    "source": source,
                    "reason": "html_table",
                }
            )
            continue

        if "|" in line and _looks_like_table_start(lines, index):
            flush_text()
            block, index = _collect_table(lines, index)
            raw_md = "\n".join(block)
            verdict = _classify_table(block)
            if verdict["complex"]:
                counters["complex_table"] += 1
                payload = dict(verdict["data"])
                payload["reason"] = verdict["reason"]
                complex_table_candidates.append(
                    {
                        "id": _node_id(document_name, "complex_table", counters["complex_table"]),
                        "document_name": document_name,
                        "page_number": current_page,
                        "element_type": "complex_table",
                        "data": payload,
                        "raw_markdown": raw_md,
                        "source": source,
                        "reason": verdict["reason"],
                    }
                )
            else:
                counters["table"] += 1
                table_nodes.append(
                    {
                        "id": _node_id(document_name, "table", counters["table"]),
                        "document_name": document_name,
                        "page_number": current_page,
                        "element_type": "table",
                        "data": verdict["data"],
                        "raw_markdown": raw_md,
                        "source": source,
                    }
                )
            continue

        buffer.append(line)
        index += 1

    flush_text()

    return {
        "text_nodes": text_nodes,
        "table_nodes": table_nodes,
        "complex_table_candidates": complex_table_candidates,
    }


def parse_markdown_file(
    md_path: str,
    document_name: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Read a Marker Markdown file and parse it into nodes."""
    path = Path(md_path)
    if not path.is_file():
        raise FileNotFoundError(f"markdown file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_markdown(
        text,
        document_name=document_name or path.stem,
        source=source or str(path),
    )


def markdown_parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline node: parse state['marker_md_path'] into the three node channels."""
    md_path = state.get("marker_md_path")
    document_name = state.get("document_name")

    if not md_path:
        logger.error("markdown_parser_node: no marker_md_path in state")
        return {
            "text_nodes": [],
            "table_nodes": [],
            "complex_table_candidates": [],
            "errors": [{"stage": "parsing", "message": "no marker_md_path in state"}],
        }

    try:
        result = parse_markdown_file(md_path, document_name=document_name)
    except Exception as exc:
        logger.exception("markdown_parser_node failed on %s", md_path)
        return {
            "text_nodes": [],
            "table_nodes": [],
            "complex_table_candidates": [],
            "errors": [
                {
                    "stage": "parsing",
                    "message": f"markdown parse failed: {exc}",
                    "exception_type": type(exc).__name__,
                }
            ],
        }

    logger.info(
        "Parsed %s: text=%d tables=%d complex=%d",
        md_path,
        len(result["text_nodes"]),
        len(result["table_nodes"]),
        len(result["complex_table_candidates"]),
    )
    result["errors"] = []
    return result
