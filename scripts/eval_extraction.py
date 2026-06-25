"""Accuracy harness: compare extracted.json against a hand-made ground_truth.json.

This is how you VERIFY the ~90-95% target on a small sample before a full run.

Ground-truth schema (page number -> expected content; every field optional):
{
  "1": {"text": "expected plain text of page 1"},
  "3": {
    "tables": [{"columns": ["Metric","FY23","FY22"],
                "rows": [["Revenue","438,857","417,000"]]}],
    "charts": [{"chart_type": "bar", "values": [10, 20, 30]}]
  }
}

Usage:
  python scripts/eval_extraction.py <extracted.json> <ground_truth.json>

Scoring (0-100%): text = char-overlap (difflib ratio); table = cell-match
fraction (whitespace/case-normalized); chart = chart_type match + value match
(2% numeric tolerance). Prints per-field and overall accuracy.
"""

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip().lower()


def _num(value: Any) -> Optional[float]:
    match = re.findall(r"-?\d[\d,]*\.?\d*", str(value))
    if not match:
        return None
    try:
        return float(match[0].replace(",", ""))
    except ValueError:
        return None


def load_extracted(path: str) -> Dict[int, Dict[str, List[Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = data.get("validated_nodes") or []
    if not nodes:
        nodes = (data.get("text_nodes") or []) + (data.get("table_nodes") or []) + (data.get("chart_nodes") or [])
    by_page: Dict[int, Dict[str, List[Any]]] = {}
    for node in nodes:
        page = node.get("page_number")
        if page is None:
            continue
        bucket = by_page.setdefault(int(page), {"text": [], "tables": [], "charts": []})
        node_type = node.get("type") or node.get("node_type") or node.get("element_type")
        data_field = node.get("data")
        if node_type == "text":
            content = data_field if isinstance(data_field, str) else node.get("content")
            if content:
                bucket["text"].append(str(content))
        elif node_type == "table":
            bucket["tables"].append(data_field if isinstance(data_field, dict) else {})
        elif node_type == "chart":
            bucket["charts"].append(data_field if isinstance(data_field, dict) else {})
    return by_page


def text_score(expected: str, got: str) -> float:
    return difflib.SequenceMatcher(None, _norm(expected), _norm(got)).ratio()


def table_score(gt: Dict[str, Any], ext: Dict[str, Any]) -> float:
    gt_grid = [[_norm(c) for c in gt.get("columns", [])]] + [[_norm(c) for c in r] for r in gt.get("rows", [])]
    ext_grid = [[_norm(c) for c in ext.get("columns", [])]]
    for row in ext.get("rows", []):
        cells = row if isinstance(row, list) else list(row.values())
        ext_grid.append([_norm(c) for c in cells])
    total = sum(len(r) for r in gt_grid) or 1
    match = 0
    for i, grow in enumerate(gt_grid):
        erow = ext_grid[i] if i < len(ext_grid) else []
        for j, cell in enumerate(grow):
            if j < len(erow) and erow[j] == cell:
                match += 1
    return match / total


def chart_score(gt: Dict[str, Any], ext: Dict[str, Any]) -> float:
    score = 0.0
    parts = 0
    if "chart_type" in gt:
        parts += 1
        score += 1.0 if _norm(gt["chart_type"]) == _norm(ext.get("chart_type")) else 0.0
    gt_vals = gt.get("values") or []
    if gt_vals:
        parts += 1
        ext_vals = []
        for series in ext.get("series") or []:
            for point in series.get("points") or []:
                num = _num(point.get("value"))
                if num is not None:
                    ext_vals.append(num)
        matched = 0
        considered = 0
        for value in gt_vals:
            target = _num(value)
            if target is None:
                continue
            considered += 1
            if any(abs(target - ev) <= max(1e-6, abs(target) * 0.02) for ev in ext_vals):
                matched += 1
        score += matched / max(1, considered)
    return score / parts if parts else 0.0


def _best(gt_item: Dict[str, Any], ext_items: List[Dict[str, Any]], scorer) -> float:
    return max((scorer(gt_item, e) for e in ext_items), default=0.0)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/eval_extraction.py <extracted.json> <ground_truth.json>", file=sys.stderr)
        return 2

    extracted = load_extracted(argv[0])
    ground_truth = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    categories: Dict[str, List[float]] = {"text": [], "table": [], "chart": []}
    for page_str, expected in ground_truth.items():
        page = int(page_str)
        ext = extracted.get(page, {"text": [], "tables": [], "charts": []})
        if "text" in expected:
            categories["text"].append(text_score(expected["text"], " ".join(ext["text"])))
        for table in expected.get("tables", []):
            categories["table"].append(_best(table, ext["tables"], table_score) if ext["tables"] else 0.0)
        for chart in expected.get("charts", []):
            categories["chart"].append(_best(chart, ext["charts"], chart_score) if ext["charts"] else 0.0)

    print("=" * 48)
    print(" Extraction accuracy")
    print("=" * 48)
    overall: List[float] = []
    for key in ("text", "table", "chart"):
        scores = categories[key]
        if scores:
            overall.extend(scores)
            print(f" {key:6s}: {sum(scores) / len(scores) * 100:5.1f}%  (n={len(scores)})")
        else:
            print(f" {key:6s}:   n/a  (n=0)")
    if overall:
        print("-" * 48)
        print(f" OVERALL: {sum(overall) / len(overall) * 100:5.1f}%  (n={len(overall)})")
    print("=" * 48)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
