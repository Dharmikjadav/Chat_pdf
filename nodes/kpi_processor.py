"""Stage 4c: extract KPI-card crops via Qwen2.5-VL -> kpi_nodes (vision-only).

Mirrors the table/chart processors: re-render each KPI candidate at high DPI from the PDF,
send it to the vision model, normalize, and emit one KPI node per crop. Never raises.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from nodes.ollama_vision_node import extract_kpi
from nodes.pymupdf_node import render_region
from utils.doc_cache import crop_key, load_cache, save_cache
from utils.json_validator import normalize_kpi_json

logger = logging.getLogger(__name__)

__all__ = ["process_kpi_crops"]


def _vision_model() -> str:
    return os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b").strip()


def _crop_dpi() -> int:
    return int(os.getenv("VISION_CROP_DPI", "300"))


def _page_dpi(state: Dict[str, Any], page_number: Optional[int]) -> int:
    for image in state.get("page_images") or []:
        if image.get("page_number") == page_number:
            return int(image.get("dpi") or 250)
    return int(os.getenv("RENDER_DPI", "250"))


def _kpi_content(kpis: List[Dict[str, Any]]) -> str:
    lines = []
    for k in kpis:
        metric, value = k.get("metric"), k.get("value")
        if not metric:
            continue
        unit = f" {k['unit']}" if k.get("unit") else ""
        period = f" ({k['period']})" if k.get("period") else ""
        lines.append(f"{metric}: {value}{unit}{period}".strip())
    return "\n".join(lines)


def process_kpi_crops(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract every KPI-card crop -> kpi_nodes."""
    candidates = state.get("kpi_candidates") or []
    if not candidates:
        return {"kpi_nodes": list(state.get("kpi_nodes") or []), "errors": []}

    pdf_path = state.get("pdf_path")
    document_name = state.get("document_name")
    model = _vision_model()
    crop_dpi = _crop_dpi()
    extracted: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    cache = load_cache(state.get("cache_dir"))
    cache_dirty = False

    for cand in candidates:
        page_number = cand.get("page_number")
        bbox = cand.get("bbox")
        image_path = cand.get("image_path")
        src_dpi = _page_dpi(state, page_number)
        key = crop_key(page_number, bbox, "kpi")

        if key in cache:
            data = cache[key]
        else:
            try:
                if pdf_path and bbox and page_number and image_path:
                    hi = str(Path(image_path).with_name(f"{Path(image_path).stem}_hi.png"))
                    render_region(pdf_path, page_number, bbox, hi, src_dpi=src_dpi, dpi=crop_dpi)
                    image_path = hi
            except Exception as exc:  # noqa: BLE001 - use the original crop
                logger.warning("kpi hi-dpi re-render failed page %s: %s", page_number, exc)

            result = extract_kpi(image_path, model=model) if image_path else {"error": "no image"}
            if isinstance(result, dict) and "error" in result:
                errors.append({"stage": "vision", "message": result["error"], "node_id": cand.get("id"),
                               "page_number": page_number, "exception_type": result.get("error_type")})
                continue
            data = normalize_kpi_json(result)
            cache[key], cache_dirty = data, True

        if not data.get("kpis"):
            continue
        kpis = data["kpis"]
        extracted.append({
            "id": cand.get("id"),
            "document_name": cand.get("document_name") or document_name,
            "page_number": page_number,
            "source": cand.get("source"),
            "element_type": "kpi",
            "data": data,
            "content": _kpi_content(kpis),
            "title": kpis[0].get("metric"),
            "bbox": bbox,
            "confidence": cand.get("confidence"),
            "image_path": cand.get("image_path"),
            "metadata": {"origin": "vision", "reason": cand.get("reason")},
        })

    if cache_dirty:
        save_cache(state.get("cache_dir"), cache)
    logger.info("process_kpi_crops: %d kpi node(s), %d error(s)", len(extracted), len(errors))
    return {"kpi_nodes": list(state.get("kpi_nodes") or []) + extracted, "errors": errors}
