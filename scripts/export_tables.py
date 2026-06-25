"""Export every table in a PageIndex store to CSV (one file per table) or a single JSON.

Usage:
    python scripts/export_tables.py <pageindex.json> --out tables_out [--json]
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pageindex_store import PageIndex  # noqa: E402


def col_name(c):
    if isinstance(c, dict):
        return str(c.get("header") or c.get("name") or c.get("label") or "")
    t = str(c).strip()
    if t.startswith("{") and "header" in t:
        try:
            return str(ast.literal_eval(t).get("header") or t)
        except (ValueError, SyntaxError):
            return t
    return t


def get_table(node):
    d = (node.get("metadata") or {}).get("data") or {}
    cols = [col_name(c) for c in (d.get("columns") or [])]
    rows = [(list(r.values()) if isinstance(r, dict) else list(r) if isinstance(r, (list, tuple)) else [r])
            for r in (d.get("rows") or [])]
    return {
        "id": node.get("id"),
        "page": node.get("page_number"),
        "title": d.get("title") or node.get("title"),
        "units": d.get("units"),
        "columns": cols,
        "rows": rows,
        "records": [dict(zip(cols, r)) for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store", help="path to pageindex.json")
    ap.add_argument("--out", default="tables_out", help="output directory")
    ap.add_argument("--json", action="store_true", help="write one tables.json instead of per-table CSVs")
    args = ap.parse_args()

    store = PageIndex.load(args.store)
    tables = [get_table(n) for n in store.nodes.values() if n["node_type"] == "table"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.json:
        path = out / "tables.json"
        path.write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {len(tables)} tables -> {path}")
        return

    for i, t in enumerate(tables, 1):
        safe = "".join(ch if ch.isalnum() else "_" for ch in (t["title"] or "table"))[:40]
        path = out / f"p{t['page'] or 0:03d}_{i:03d}_{safe}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if t["columns"]:
                w.writerow(t["columns"])
            for row in t["rows"]:
                w.writerow(row)
    print(f"wrote {len(tables)} CSV table(s) -> {out}/")


if __name__ == "__main__":
    main()
