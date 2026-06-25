"""Stage 4a: extract table crops — cheap-first PyMuPDF, vision fallback.

For each OpenCV-detected table crop:
  1. CHEAP: re-open the PDF page and run PyMuPDF ``page.find_tables()`` clipped to
     the crop's bbox (text-layer extraction; near-exact on digital tables).
  2. FALLBACK: if find_tables yields nothing usable (complex / scanned / failure),
     re-render the bbox at high DPI and send it to Qwen2.5-VL via Ollama.
Emits table nodes appended to ``table_nodes`` (same shape the rest of the pipeline
already consumes). Reuses fitz, render_region, extract_complex_table, normalize_table_json.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    import pymupdf as fitz  # type: ignore

from nodes.ollama_vision_node import extract_complex_table
from nodes.pymupdf_node import render_region
from utils.doc_cache import crop_key, load_cache, save_cache
from utils.json_validator import normalize_table_json, validate_table
from utils.table_extractors import close_cache, extract_table_pdfplumber

logger = logging.getLogger(__name__)

__all__ = ["process_table_crops"]

MIN_ROWS = 2
MIN_COLS = 2


def _vision_model() -> str:
    return os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b").strip()


def _use_pdfplumber() -> bool:
    return os.getenv("USE_PDFPLUMBER", "1").strip().lower() in {"1", "true", "yes", "on"}


def _cheap_table(pdf_path, page_number, bbox, src_dpi):
    """Try find_tables then pdfplumber; return (data, origin) only if it passes validate_table."""
    try:
        data = _find_tables_in_bbox(pdf_path, page_number, bbox, src_dpi)
        if data is not None and validate_table(normalize_table_json(data))[0]:
            return data, "pymupdf_find_tables"
    except Exception as exc:  # noqa: BLE001 - fall through to pdfplumber/vision
        logger.warning("find_tables failed on page %s: %s", page_number, exc)
    if _use_pdfplumber():
        data = extract_table_pdfplumber(pdf_path, page_number, bbox, src_dpi)
        if data is not None and validate_table(normalize_table_json(data))[0]:
            return data, "pdfplumber"
    return None, None


def _crop_dpi() -> int:
    return int(os.getenv("VISION_CROP_DPI", "300"))


def _page_dpi(state: Dict[str, Any], page_number: Optional[int]) -> int:
    for image in state.get("page_images") or []:
        if image.get("page_number") == page_number:
            return int(image.get("dpi") or 250)
    return int(os.getenv("RENDER_DPI", "250"))


def _find_tables_in_bbox(pdf_path: str, page_number: int, bbox: List[int], src_dpi: int) -> Optional[Dict[str, Any]]:
    """Try PyMuPDF text-layer table extraction within the bbox. None if no good table."""
    scale = 72.0 / float(src_dpi)
    x0, y0, x1, y1 = (c * scale for c in bbox)
    pad = 4.0
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        clip = fitz.Rect(min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad) & page.rect
        found = page.find_tables(clip=clip)
        tables = list(getattr(found, "tables", found) or [])
        if not tables:
            return None
        # pick the largest table by cell count
        best = max(tables, key=lambda t: (t.row_count * t.col_count))
        if best.row_count < MIN_ROWS or best.col_count < MIN_COLS:
            return None
        grid = best.extract()  # list[list[str|None]]
        if not grid or len(grid) < MIN_ROWS:
            return None
        header = [("" if c is None else str(c)) for c in grid[0]]
        rows = [[("" if c is None else str(c)) for c in r] for r in grid[1:]]
        return {"type": "table", "title": None, "columns": header, "rows": rows}
    finally:
        doc.close()


def process_table_crops(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract every table crop (cheap-first, vision fallback) -> table_nodes."""
    candidates = state.get("complex_table_candidates") or []
    if not candidates:
        return {"table_nodes": list(state.get("table_nodes") or []), "errors": []}

    pdf_path = state.get("pdf_path")
    document_name = state.get("document_name")
    model = _vision_model()
    crop_dpi = _crop_dpi()
    debug = os.getenv("DEBUG_REGIONS", os.getenv("OPENCV_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}

    extracted: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    cache = load_cache(state.get("cache_dir"))
    cache_dirty = False

    for cand in candidates:
        page_number = cand.get("page_number")
        bbox = cand.get("bbox")
        src_dpi = _page_dpi(state, page_number)
        confidence = cand.get("confidence")
        key = crop_key(page_number, bbox, "table")

        if key in cache:
            normalized, origin = cache[key]["data"], cache[key]["origin"]
        else:
            data: Optional[Dict[str, Any]] = None
            origin = None
            if pdf_path and bbox and page_number:
                data, origin = _cheap_table(pdf_path, page_number, bbox, src_dpi)

            if data is None:  # vision fallback
                image_path = cand.get("image_path")
                try:
                    if pdf_path and bbox and page_number and image_path:
                        hi = str(Path(image_path).with_name(f"{Path(image_path).stem}_hi.png"))
                        render_region(pdf_path, page_number, bbox, hi, src_dpi=src_dpi, dpi=crop_dpi)
                        image_path = hi
                except Exception as exc:  # noqa: BLE001 - use the original crop
                    logger.warning("hi-dpi re-render failed on page %s: %s", page_number, exc)
                result = extract_complex_table(image_path, model=model) if image_path else {"error": "no image"}
                if isinstance(result, dict) and "error" in result:
                    errors.append(
                        {"stage": "vision", "message": result["error"], "node_id": cand.get("id"),
                         "page_number": page_number, "exception_type": result.get("error_type")}
                    )
                    continue
                data = result if isinstance(result, dict) else None
                origin = "vision"
                if isinstance(result, dict) and isinstance(result.get("confidence"), (int, float)):
                    confidence = result.get("confidence")
            normalized = normalize_table_json(data)
            cache[key], cache_dirty = {"data": normalized, "origin": origin}, True

        extracted.append(
            {
                "id": cand.get("id"),
                "document_name": cand.get("document_name") or document_name,
                "page_number": page_number,
                "source": cand.get("source"),
                "element_type": "table",
                "data": normalized,
                "bbox": bbox,
                "confidence": confidence,
                "image_path": cand.get("image_path"),
                "metadata": {"origin": origin, "reason": cand.get("reason")},
            }
        )
        if debug:
            logger.info("table page=%s origin=%s cols=%s rows=%s", page_number, origin,
                        normalized.get("n_cols"), normalized.get("n_rows"))

    close_cache()
    if cache_dirty:
        save_cache(state.get("cache_dir"), cache)
    logger.info("process_table_crops: %d table(s), %d error(s)", len(extracted), len(errors))
    return {"table_nodes": list(state.get("table_nodes") or []) + extracted, "errors": errors}
