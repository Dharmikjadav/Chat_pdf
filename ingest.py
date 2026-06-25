"""Command-line entrypoint: ingest a PDF through the offline pipeline.

Usage:
    python ingest.py input_pdfs/sample.pdf
    python ingest.py input_pdfs/sample.pdf --output-dir data/processed/sample
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _setup_logging(verbose: bool) -> None:
    from utils.logger import setup_logging

    setup_logging(level="DEBUG" if verbose else "INFO")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="Ingest a PDF into the offline PageIndex store.",
    )
    parser.add_argument("pdf_path", help="path to the input PDF")
    parser.add_argument("--output-dir", default=None, help="where to write artifacts")
    parser.add_argument("--document-name", default=None, help="logical document name")
    parser.add_argument("--max-pages", type=int, default=None, help="only process the first N pages (smoke test)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def _save_extracted_json(state: Dict[str, Any]) -> str:
    out_dir = Path(state["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_name": state.get("document_name"),
        "pdf_path": state.get("pdf_path"),
        "pageindex_path": state.get("pageindex_path"),
        "summary": state.get("final_output"),
        "page_metadata": state.get("page_metadata"),
        "page_classes": state.get("page_classes"),
        "text_nodes": state.get("text_nodes"),
        "table_nodes": state.get("table_nodes"),
        "chart_nodes": state.get("chart_nodes"),
        "validated_nodes": state.get("validated_nodes"),
        "errors": state.get("errors"),
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    path = out_dir / "extracted.json"
    path.write_text(blob, encoding="utf-8")
    # Additional flat alias requested in the spec (does not affect chatbot discovery).
    alias = Path("extracted_json") / "final_document.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.write_text(blob, encoding="utf-8")
    return str(path)


def _print_summary(state: Dict[str, Any], extracted_path: str) -> None:
    errors = state.get("errors") or []
    final = state.get("final_output") or {}

    print("\n" + "=" * 48)
    print(" Ingestion summary")
    print("=" * 48)
    print(f" Document        : {state.get('document_name')}")
    print(f" PDF             : {state.get('pdf_path')}")
    print(f" Pages processed : {len(state.get('page_metadata') or [])}")
    print(f" Text nodes      : {len(state.get('text_nodes') or [])}")
    print(f" Table nodes     : {len(state.get('table_nodes') or [])}")
    print(f" Chart nodes     : {len(state.get('chart_nodes') or [])}")
    print(f" Errors          : {len(errors)}")
    print(f" Status          : {final.get('status', 'unknown')}")
    print("-" * 48)
    print(f" Extracted JSON  : {extracted_path}")
    print(f" PageIndex store : {state.get('pageindex_path') or '(not saved)'}")
    print("=" * 48)

    if errors:
        print("\n First errors:")
        for err in errors[:5]:
            stage = err.get("stage", "?")
            message = err.get("message", "")
            page = err.get("page_number")
            location = f" (page {page})" if page is not None else ""
            print(f"  - [{stage}]{location} {message}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    _load_env()

    if args.max_pages is not None:
        os.environ["MAX_PAGES"] = str(args.max_pages)

    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_file():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    try:
        from graph.pipeline import run_pipeline
    except ImportError as exc:
        print(f"error: could not import pipeline: {exc}", file=sys.stderr)
        return 1

    kwargs: Dict[str, Any] = {}
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.document_name:
        kwargs["document_name"] = args.document_name

    try:
        state = run_pipeline(str(pdf_path), **kwargs)
    except ImportError as exc:
        print(f"error: missing dependency to run the pipeline: {exc}", file=sys.stderr)
        print("hint: install requirements (e.g. 'pip install -r requirements.txt')", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report any pipeline failure to the user
        logging.getLogger("ingest").exception("pipeline failed")
        print(f"error: pipeline failed: {exc}", file=sys.stderr)
        return 1

    extracted_path = _save_extracted_json(state)
    _print_summary(state, extracted_path)

    status = (state.get("final_output") or {}).get("status")
    return 0 if status in ("success", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
