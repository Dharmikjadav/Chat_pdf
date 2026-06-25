"""Stage 4b: extract chart crops via Qwen2.5-VL into structured chart JSON.

Charts always go to the vision model (no cheap text-layer path). The crop is
re-rendered from the PDF at high DPI for accuracy, sent to extract_chart, then
normalized and flattened to the requested flat shape. Emits chart nodes appended
to ``chart_nodes``.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from nodes.ollama_vision_node import extract_chart
from nodes.pymupdf_node import render_region
from utils.doc_cache import crop_key, load_cache, save_cache
from utils.json_validator import normalize_chart_json

logger = logging.getLogger(__name__)

__all__ = ["process_chart_crops"]


def _vision_model() -> str:
    return os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b").strip()


def _crop_dpi() -> int:
    return int(os.getenv("VISION_CROP_DPI", "300"))


def _page_dpi(state: Dict[str, Any], page_number: Optional[int]) -> int:
    for image in state.get("page_images") or []:
        if image.get("page_number") == page_number:
            return int(image.get("dpi") or 200)
    return int(os.getenv("RENDER_DPI", "200"))


def _flatten_chart(norm: Dict[str, Any], page_number: Optional[int], source_crop: Optional[str]) -> Dict[str, Any]:
    x_axis = norm.get("x_axis") or {}
    y_axis = norm.get("y_axis") or {}
    return {
        "chart_type": norm.get("chart_type"),
        "title": norm.get("title"),
        "x_axis_label": x_axis.get("label"),
        "y_axis_label": y_axis.get("label"),
        "units": y_axis.get("unit") or x_axis.get("unit"),
        "legend": norm.get("legend") or [],
        "series": norm.get("series") or [],
        "page_number": page_number,
        "source_crop": source_crop,
    }


def process_chart_crops(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract every chart crop via the vision model -> chart_nodes."""
    candidates = state.get("chart_candidates") or []
    if not candidates:
        return {"chart_nodes": list(state.get("chart_nodes") or []), "errors": []}

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
        image_path = cand.get("image_path")
        key = crop_key(page_number, bbox, "chart")

        if key in cache:
            normalized, origin = cache[key], "cache"
        else:
            try:
                if pdf_path and bbox and page_number and image_path:
                    hi = str(Path(image_path).with_name(f"{Path(image_path).stem}_hi.png"))
                    render_region(pdf_path, page_number, bbox, hi, src_dpi=_page_dpi(state, page_number), dpi=crop_dpi)
                    image_path = hi
            except Exception as exc:  # noqa: BLE001 - use original crop
                logger.warning("hi-dpi re-render failed on page %s: %s", page_number, exc)

            result = extract_chart(image_path, model=model) if image_path else {"error": "no image"}
            if isinstance(result, dict) and "error" in result:
                errors.append(
                    {"stage": "vision", "message": result["error"], "node_id": cand.get("id"),
                     "page_number": page_number, "exception_type": result.get("error_type")}
                )
                continue
            normalized = normalize_chart_json(result if isinstance(result, dict) else None)
            cache[key], cache_dirty, origin = normalized, True, "vision"

        flat = _flatten_chart(normalized, page_number, image_path)
        extracted.append(
            {
                "id": cand.get("id"),
                "document_name": cand.get("document_name") or document_name,
                "page_number": page_number,
                "source": cand.get("source"),
                "element_type": "chart",
                "data": normalized,
                "bbox": bbox,
                "confidence": cand.get("confidence"),
                "image_path": image_path,
                "metadata": {"origin": origin, "reason": cand.get("reason"), "chart": flat},
            }
        )
        if debug:
            logger.info("chart page=%s origin=%s series=%s", page_number, origin, normalized.get("n_series"))

    if cache_dirty:
        save_cache(state.get("cache_dir"), cache)
    logger.info("process_chart_crops: %d chart(s), %d error(s)", len(extracted), len(errors))
    return {"chart_nodes": list(state.get("chart_nodes") or []) + extracted, "errors": errors}
