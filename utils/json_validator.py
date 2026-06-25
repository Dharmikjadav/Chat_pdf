"""Repair, normalize, and validate JSON produced by the vision/LLM stages.

Public API:
    repair_json(raw_text)        -> parsed object (dict/list/...) or None
    normalize_table_json(data)   -> canonical table dict
    normalize_chart_json(data)   -> canonical chart dict
    validate_node(node)          -> node guaranteed to carry
                                    type, page_number, source, confidence, data
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "repair_json",
    "normalize_table_json",
    "normalize_chart_json",
    "normalize_kpi_json",
    "validate_table",
    "validate_node",
]

DEFAULT_CONFIDENCE = 1.0


def _nkey(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _get(data: Any, names: List[str], default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    normalized = {_nkey(k): v for k, v in data.items()}
    for name in names:
        token = _nkey(name)
        if token in normalized and normalized[token] is not None:
            return normalized[token]
    for name in names:
        token = _nkey(name)
        if token in normalized:
            return normalized[token]
    return default


def _to_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def _to_page(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _norm_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _clamp_confidence(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number:
        return float(default)
    return max(0.0, min(1.0, number))


def _normalize_headers(header: List[Any]) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for position, raw in enumerate(header, 1):
        text = str(raw).strip() if raw is not None and str(raw).strip() else f"column_{position}"
        if text in seen:
            seen[text] += 1
            text = f"{text}_{seen[text]}"
        else:
            seen[text] = 1
        out.append(text)
    return out


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_block(text: str) -> Optional[str]:
    start = None
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start is None:
        return None

    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _try_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _json_repair_lib(text: str) -> Optional[Any]:
    try:
        import json_repair  # noqa: PLC0415 - lazy so the module imports without the dep
    except ImportError:
        return None
    try:
        obj = json_repair.repair_json(text, return_objects=True)
    except TypeError:
        try:
            obj = json_repair.loads(text)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None
    if obj is None or obj == "":
        return None
    return obj


def repair_json(raw_text: Any) -> Optional[Any]:
    """Parse text into a JSON object, repairing malformations via json-repair."""
    if raw_text is None:
        return None
    if isinstance(raw_text, (dict, list, int, float, bool)):
        return raw_text

    text = str(raw_text).strip()
    if not text:
        return None

    obj = _try_load(text)
    if obj is not None:
        return obj

    stripped = _strip_fences(text)
    if stripped != text:
        obj = _try_load(stripped)
        if obj is not None:
            return obj

    repaired = _json_repair_lib(stripped) or _json_repair_lib(text)
    if repaired is not None:
        return repaired

    for source in (stripped, text):
        block = _extract_json_block(source)
        if block:
            obj = _try_load(block)
            if obj is not None:
                return obj

    logger.warning("repair_json: could not parse text of length %d", len(text))
    return None


def _coerce_rows(raw_rows: Any, columns: List[str]) -> Tuple[List[str], List[List[Any]]]:
    columns = list(columns)
    rows: List[List[Any]] = []
    if not isinstance(raw_rows, list):
        return columns, rows

    if not columns:
        derived: List[str] = []
        for row in raw_rows:
            if isinstance(row, dict) and _get(row, ["cells", "values", "row", "data"]) is None:
                for key in row.keys():
                    if _nkey(key) not in {"rowheader", "header", "label"} and str(key) not in derived:
                        derived.append(str(key))
        columns = _normalize_headers(derived) if derived else []

    for row in raw_rows:
        if isinstance(row, dict):
            cells_value = _get(row, ["cells", "values", "row", "data"])
            if isinstance(cells_value, list):
                cells = [_norm_cell(cell) for cell in cells_value]
                row_header = _get(row, ["row_header", "header", "label"])
                if row_header is not None and len(cells) < len(columns):
                    cells = [_norm_cell(row_header)] + cells
            else:
                cells = [_norm_cell(row.get(column)) for column in columns]
        elif isinstance(row, (list, tuple)):
            cells = [_norm_cell(cell) for cell in row]
        else:
            cells = [_norm_cell(row)]
        rows.append(cells)

    # No header but cells present (e.g. column-less financial statements):
    # synthesize column names from the widest row so n_cols reflects the data.
    if not columns and rows:
        columns = [f"column_{i}" for i in range(1, max(len(c) for c in rows) + 1)]

    if columns:
        width = len(columns)
        max_width = max([width] + [len(cells) for cells in rows]) if rows else width
        if max_width > width:
            columns = columns + [f"column_{i}" for i in range(width + 1, max_width + 1)]
            width = max_width
        rows = [cells + [None] * (width - len(cells)) for cells in rows]

    return columns, rows


def normalize_table_json(data: Any) -> Dict[str, Any]:
    """Normalize any table-shaped JSON into a canonical table dict."""
    if isinstance(data, str):
        data = repair_json(data)

    if isinstance(data, list):
        data = {"rows": data}
    if not isinstance(data, dict):
        return {
            "type": "table",
            "title": None,
            "columns": [],
            "rows": [],
            "units": None,
            "footnotes": [],
            "n_rows": 0,
            "n_cols": 0,
        }

    title = _get(data, ["title", "caption", "name", "heading"])
    columns = _get(data, ["columns", "column", "headers", "header", "cols", "fields"])
    raw_rows = _get(data, ["rows", "records", "values", "body", "table", "data"])
    units = _get(data, ["units", "unit"])
    footnotes = _get(data, ["footnotes", "footnote"])

    columns = [_to_str_or_none(c) or "" for c in columns] if isinstance(columns, list) else []
    columns = _normalize_headers(columns) if columns else []
    columns, rows = _coerce_rows(raw_rows if isinstance(raw_rows, list) else [], columns)

    if footnotes is None:
        footnotes = []
    elif not isinstance(footnotes, list):
        footnotes = [footnotes]

    return {
        "type": "table",
        "title": _to_str_or_none(title),
        "columns": columns,
        "rows": rows,
        "units": _to_str_or_none(units),
        "footnotes": footnotes,
        "n_rows": len(rows),
        "n_cols": len(columns),
    }


_NUMISH_RE = re.compile(r"\(?[-+]?[\d,]+\.?\d*%?\)?")
_MERGED_NUM_RE = re.compile(r",\d{4,}")
_MERGED_FY_RE = re.compile(r"(?:fy\d{2,4}){2,}", re.IGNORECASE)
_PERIOD_HDR_RE = re.compile(r"(fy ?\d{2,4}|20\d{2}|q[1-4])", re.IGNORECASE)
_AUTO_COL_RE = re.compile(r"column_?\d+|col_?\d+|^_?\d+$", re.IGNORECASE)


def _row_cells(row: Any) -> List[str]:
    if isinstance(row, dict):
        return [str(v) for v in row.values()]
    if isinstance(row, (list, tuple)):
        return [str(c) for c in row]
    return [str(row)]


def validate_table(table: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Quality gate for a cheaply-extracted (canonical) table.

    Returns ``(ok, reasons)``. ``ok`` is False when the extraction looks broken — too few
    rows/cols, missing header, a financial table with no numbers, merged year/value strings
    (``FY22FY23`` / ``273,920284,870``), or mostly-empty rows — signalling the caller to fall
    back to vision extraction.
    """
    reasons: List[str] = []
    columns = table.get("columns") or []
    rows = table.get("rows") or []

    if len(rows) < 2:
        reasons.append("rows<2")
    if len(columns) < 2:
        reasons.append("cols<2")

    real_headers = [str(c).strip() for c in columns if str(c).strip()]
    auto = sum(1 for c in columns if _AUTO_COL_RE.fullmatch(str(c).strip()))
    if not real_headers or auto >= max(1, len(columns)):
        reasons.append("no_header")
    if sum(1 for c in columns if len(str(c).strip()) == 1) > max(1, len(columns) // 3):
        reasons.append("single_char_columns")

    cells = [c for row in rows for c in _row_cells(row)]
    nonempty = [c for c in cells if c.strip()]
    period_header = any(_PERIOD_HDR_RE.search(str(c)) for c in columns)
    if not nonempty:
        reasons.append("empty_body")
    elif period_header:
        numeric = sum(1 for c in nonempty if _NUMISH_RE.fullmatch(c.strip()))
        if numeric / len(nonempty) < 0.20:
            reasons.append("broken_financial_table")
        labelled = sum(1 for row in rows if _row_cells(row) and _row_cells(row)[0].strip())
        if rows and labelled / len(rows) < 0.6:
            reasons.append("missing_row_labels")

    if any(_MERGED_NUM_RE.search(c) or _MERGED_FY_RE.search(c.replace(" ", "")) for c in cells):
        reasons.append("merged_values")

    empty_rows = sum(1 for row in rows if not any(c.strip() for c in _row_cells(row)))
    if rows and empty_rows / len(rows) > 0.5:
        reasons.append("mostly_empty_rows")

    critical = {"rows<2", "cols<2", "no_header", "empty_body", "broken_financial_table",
                "missing_row_labels", "merged_values", "mostly_empty_rows"}
    return (not any(r in critical for r in reasons), reasons)


def _coerce_axis(value: Any) -> Dict[str, Optional[str]]:
    if isinstance(value, dict):
        return {
            "label": _to_str_or_none(_get(value, ["label", "name", "title", "text"])),
            "unit": _to_str_or_none(_get(value, ["unit", "units"])),
        }
    if isinstance(value, str):
        return {"label": _to_str_or_none(value), "unit": None}
    return {"label": None, "unit": None}


def _coerce_points(raw: Any) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        for label, value in raw.items():
            points.append({"label": _to_str_or_none(label), "value": _norm_cell(value), "unit": None})
        return points
    if not isinstance(raw, list):
        return points
    for point in raw:
        if isinstance(point, dict):
            points.append(
                {
                    "label": _to_str_or_none(_get(point, ["label", "category", "x", "name", "key"])),
                    "value": _norm_cell(_get(point, ["value", "y", "val", "amount", "count"])),
                    "unit": _to_str_or_none(_get(point, ["unit", "units"])),
                }
            )
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append({"label": _to_str_or_none(point[0]), "value": _norm_cell(point[1]), "unit": None})
        elif isinstance(point, (list, tuple)) and len(point) == 1:
            points.append({"label": None, "value": _norm_cell(point[0]), "unit": None})
        else:
            points.append({"label": None, "value": _norm_cell(point), "unit": None})
    return points


def _is_series_obj(value: Any) -> bool:
    return isinstance(value, dict) and _get(value, ["points", "data", "values", "series"]) is not None


def _one_series(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": _to_str_or_none(_get(value, ["name", "label", "title"])),
        "points": _coerce_points(_get(value, ["points", "data", "values", "series"])),
    }


def _coerce_series(raw: Any) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    if raw is None:
        return series
    if isinstance(raw, dict):
        if _is_series_obj(raw):
            series.append(_one_series(raw))
        elif raw and all(isinstance(v, (list, dict)) for v in raw.values()):
            for name, points in raw.items():
                series.append({"name": _to_str_or_none(name), "points": _coerce_points(points)})
        else:
            series.append({"name": None, "points": _coerce_points(raw)})
        return series
    if isinstance(raw, list):
        if any(_is_series_obj(item) for item in raw):
            for item in raw:
                if _is_series_obj(item):
                    series.append(_one_series(item))
                else:
                    name = _to_str_or_none(_get(item, ["name", "label"])) if isinstance(item, dict) else None
                    series.append({"name": name, "points": _coerce_points([item])})
        else:
            series.append({"name": None, "points": _coerce_points(raw)})
    return series


def normalize_chart_json(data: Any) -> Dict[str, Any]:
    """Normalize any chart-shaped JSON into a canonical chart dict."""
    if isinstance(data, str):
        data = repair_json(data)

    if isinstance(data, list):
        data = {"series": data}
    if not isinstance(data, dict):
        return {
            "type": "chart",
            "chart_type": None,
            "title": None,
            "x_axis": {"label": None, "unit": None},
            "y_axis": {"label": None, "unit": None},
            "series": [],
            "legend": [],
            "n_series": 0,
            "n_points": 0,
        }

    chart_type = _get(data, ["chart_type", "charttype", "kind", "graph_type", "plot_type"])
    if not chart_type:
        candidate = _get(data, ["type"])
        if isinstance(candidate, str) and candidate.strip().lower() not in ("chart", ""):
            chart_type = candidate

    series = _coerce_series(_get(data, ["series", "datasets", "data", "values"]))
    legend = _get(data, ["legend", "legends", "keys"])
    if legend is None:
        legend = []
    elif not isinstance(legend, list):
        legend = [legend]

    return {
        "type": "chart",
        "chart_type": _to_str_or_none(chart_type),
        "title": _to_str_or_none(_get(data, ["title", "caption", "name", "heading"])),
        "x_axis": _coerce_axis(_get(data, ["x_axis", "xaxis", "x", "horizontal_axis", "x_label"])),
        "y_axis": _coerce_axis(_get(data, ["y_axis", "yaxis", "y", "vertical_axis", "y_label"])),
        "series": series,
        "legend": [_to_str_or_none(item) for item in legend],
        "n_series": len(series),
        "n_points": sum(len(item["points"]) for item in series),
    }


def normalize_kpi_json(data: Any) -> Dict[str, Any]:
    """Normalize KPI-card JSON into a canonical {type, kpis:[...], n_kpis} dict."""
    if isinstance(data, str):
        data = repair_json(data)
    if isinstance(data, list):
        data = {"kpis": data}

    kpis_raw = _get(data, ["kpis", "kpi", "cards", "items", "metrics"]) if isinstance(data, dict) else None
    if kpis_raw is None and isinstance(data, dict) and _nkey("metric") in {_nkey(k) for k in data}:
        kpis_raw = [data]
    if not isinstance(kpis_raw, list):
        kpis_raw = []

    out: List[Dict[str, Any]] = []
    for k in kpis_raw:
        if not isinstance(k, dict):
            continue
        metric = _to_str_or_none(_get(k, ["metric", "label", "name", "kpi", "title"]))
        value = _to_str_or_none(_get(k, ["value", "figure", "amount", "number"]))
        if not metric and value is None:
            continue
        out.append({
            "metric": metric,
            "value": value,
            "unit": _to_str_or_none(_get(k, ["unit", "units", "currency"])),
            "period": _to_str_or_none(_get(k, ["period", "year", "fy", "quarter"])),
            "context": _to_str_or_none(_get(k, ["context", "note", "qualifier", "description"])),
            "confidence": _clamp_confidence(_get(k, ["confidence", "score"]), DEFAULT_CONFIDENCE),
        })
    return {"type": "kpi", "kpis": out, "n_kpis": len(out)}


def _resolve_type(node: Dict[str, Any], data: Any) -> str:
    for key in ("type", "element_type"):
        value = node.get(key)
        if isinstance(value, str):
            token = value.strip().lower()
            if "chart" in token:
                return "chart"
            if "table" in token:
                return "table"
            if token in ("text", "figure", "image", "heading", "form", "mixed", "other"):
                return token

    if isinstance(data, dict):
        data_type = data.get("type")
        if isinstance(data_type, str):
            if "chart" in data_type.lower():
                return "chart"
            if "table" in data_type.lower():
                return "table"
        if _get(data, ["series", "datasets"]) is not None or _get(data, ["x_axis", "y_axis", "xaxis", "yaxis"]) is not None:
            return "chart"
        if _get(data, ["columns", "rows", "headers"]) is not None:
            return "table"
    return "text"


def _table_valid(table: Dict[str, Any]) -> bool:
    return bool(table.get("columns")) and isinstance(table.get("rows"), list)


def _chart_valid(chart: Dict[str, Any]) -> bool:
    series = chart.get("series") or []
    return any(item.get("points") for item in series)


def validate_node(node: Any) -> Dict[str, Any]:
    """Normalize a node and guarantee type/page_number/source/confidence/data keys."""
    if not isinstance(node, dict):
        node = {"data": node}

    result: Dict[str, Any] = dict(node)

    raw_data = node.get("data")
    if raw_data is None:
        raw_data = node.get("content")
    if isinstance(raw_data, str) and raw_data.lstrip()[:1] in ("{", "["):
        repaired = repair_json(raw_data)
        if repaired is not None:
            raw_data = repaired

    node_type = _resolve_type(node, raw_data)
    result["type"] = node_type

    if node_type == "table":
        normalized = normalize_table_json(raw_data)
        valid = _table_valid(normalized)
        result["data"] = normalized
    elif node_type == "chart":
        normalized = normalize_chart_json(raw_data)
        valid = _chart_valid(normalized)
        result["data"] = normalized
    else:
        result["data"] = raw_data
        valid = raw_data is not None

    if isinstance(raw_data, dict) and raw_data.get("error"):
        valid = False

    result["page_number"] = _to_page(node.get("page_number", node.get("page")))
    result["source"] = node.get("source") or node.get("document_name")

    confidence = _get(node, ["confidence", "score", "prob", "probability"])
    if confidence is None and isinstance(raw_data, dict):
        confidence = _get(raw_data, ["confidence", "score"])
    result["confidence"] = _clamp_confidence(confidence, DEFAULT_CONFIDENCE if valid else 0.0)

    # Guarantee node_type (alias of type) and bbox on every node.
    result["node_type"] = node_type
    if result.get("bbox") is None:
        result["bbox"] = node.get("bbox")

    result["valid"] = bool(valid)
    return result
