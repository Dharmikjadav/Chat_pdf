"""One-shot: add structured companion tables for financial-statement text nodes in an
existing PageIndex store (so 'revenue from operations' etc. become retrievable) without
re-running ingestion. Backs the store up first. Idempotent.

Usage:  python scripts/enrich_statements.py <path/to/pageindex.json>
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageindex_store import PageIndex  # noqa: E402
from utils.statement_parser import enrich_store_with_statements  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/enrich_statements.py <pageindex.json>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not found: {path}")
        return 2

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup written: {backup}")
    else:
        print(f"backup already exists: {backup}")

    store = PageIndex.load(str(path))
    before = len(store.nodes)
    added = enrich_store_with_statements(store)
    store.save(str(path))
    print(f"nodes: {before} -> {len(store.nodes)}  (+{added} statement tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
