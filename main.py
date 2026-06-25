"""Terminal chatbot over an existing PageIndex store.

Usage:
    python main.py data/processed/sample/pageindex.json
    python main.py                      # auto-discovers a pageindex.json

Commands inside the chat:
    help      show available commands
    sources   show the sources cited for the last answer
    pages     list the pages held in the loaded store
    exit      quit
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from chatbot import ChatEngine

_DISCOVERY_PATTERNS = [
    "pageindex.json",
    "data/**/pageindex.json",
    "data/pageindex/*.json",
]


def _discover_stores() -> List[Path]:
    found: List[Path] = []
    for pattern in _DISCOVERY_PATTERNS:
        for path in sorted(Path.cwd().glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  help     show this help\n"
        "  sources  show the sources for the last answer\n"
        "  pages    list the pages in the loaded store\n"
        "  exit     quit\n"
        "Type anything else to ask a question."
    )


def _print_pages(engine: ChatEngine) -> None:
    pages = engine.store.search_by_node_type("page")
    if not pages:
        print("No pages in this store.")
        return
    by_document: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        by_document.setdefault(page.get("document_name") or "unknown", []).append(page)
    for document, items in by_document.items():
        print(f"\nDocument: {document}  ({len(items)} pages)")
        for page in sorted(items, key=lambda item: item.get("page_number") or 0):
            print(f"  page {page.get('page_number')}: {page.get('title') or ''}".rstrip())


def _format_source(source: Dict[str, Any]) -> str:
    parts: List[str] = []
    page = source.get("page_number")
    if page is not None:
        parts.append(f"page {page}")
    node_type = source.get("node_type")
    title = source.get("title")
    if title:
        parts.append(f"{node_type}: {title}")
    elif node_type:
        parts.append(str(node_type))
    return ", ".join(parts)


def _print_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        print("No sources yet. Ask a question first.")
        return
    print("\nSources for the last answer:")
    for source in sources:
        document = source.get("document_name") or "unknown"
        print(f"  [{source['n']}] {document} - {_format_source(source)}")


def _print_answer(result: Dict[str, Any]) -> None:
    print(f"\nBot: {result['answer']}")
    sources = result.get("sources") or []
    if sources:
        primary = _format_source(sources[0])
        if primary:
            print(f"Source: {primary}.")
        if len(sources) > 1:
            print(f"({len(sources)} sources retrieved - type 'sources' to see them all.)")


def _chat_loop(engine: ChatEngine, store_path: str) -> int:
    documents = ", ".join(engine.store.document_names()) or "(none)"
    print("=" * 56)
    print(" Offline PDF Chatbot")
    print("=" * 56)
    print(f" Store     : {store_path}")
    print(f" Nodes     : {len(engine.store)}")
    print(f" Documents : {documents}")
    print(f" Model     : {engine.model}")
    print(" Type 'help' for commands, 'exit' to quit.")
    print("=" * 56)

    last_sources: List[Dict[str, Any]] = []
    while True:
        try:
            line = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        command = line.lower()
        if command in ("exit", "quit"):
            break
        if command == "help":
            _print_help()
            continue
        if command == "pages":
            _print_pages(engine)
            continue
        if command == "sources":
            _print_sources(last_sources)
            continue

        result = engine.ask(line)
        last_sources = result.get("sources") or []
        _print_answer(result)

    print("Goodbye.")
    return 0


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Chat with an ingested PDF using its PageIndex store.",
    )
    parser.add_argument("store", nargs="?", default=None, help="path to a pageindex.json store")
    parser.add_argument("--model", default=None, help="Ollama model (default: qwen3:8b)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    from utils.logger import setup_logging

    setup_logging(level="DEBUG" if args.verbose else "WARNING")

    store_path = args.store
    if not store_path:
        candidates = _discover_stores()
        if len(candidates) == 1:
            store_path = str(candidates[0])
        elif not candidates:
            print("error: no PageIndex store found. Run ingest.py first, or pass a path.", file=sys.stderr)
            return 2
        else:
            print("error: multiple stores found, please specify one:", file=sys.stderr)
            for candidate in candidates:
                print(f"  {candidate}", file=sys.stderr)
            return 2

    if not Path(store_path).is_file():
        print(f"error: store not found: {store_path}", file=sys.stderr)
        return 2

    try:
        engine = ChatEngine(store_path, model=args.model)
    except Exception as exc:  # noqa: BLE001 - report load failure cleanly
        print(f"error: could not load store '{store_path}': {exc}", file=sys.stderr)
        return 1

    return _chat_loop(engine, store_path)


if __name__ == "__main__":
    sys.exit(main())
