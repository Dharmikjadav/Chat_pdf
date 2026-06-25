"""Stage 1a: classify each page from fast PyMuPDF signals (no OpenCV, no re-open).

Classes: text_only | table_page | chart_page | kpi_page | image_heavy_page | mixed_page | scanned_page.
Downstream routing uses these: text_only pages skip OpenCV+vision (their text is
captured directly); scanned pages are OCR'd; table/chart/kpi/image/mixed pages go
through region detection + the appropriate extractor.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["classify_pages", "classify_page"]

SCANNED_MAX_CHARS = 50   # below this with no/poor text layer => treat as scanned
HIGH_DRAWINGS = 40       # vector lines/rects suggesting a table/chart grid
KPI_MAX_WORDS = 120      # KPI/infographic pages are short on words
IMAGE_HEAVY_AREA = 0.55  # raster coverage fraction => image-dominated page
IMAGE_HEAVY_MAX_CHARS = 400


def _text_threshold() -> int:
    return int(os.getenv("TEXT_PAGE_CHAR_THRESHOLD", "1600"))


def _kpi_num_ratio() -> float:
    try:
        return float(os.getenv("KPI_NUM_RATIO", "0.30"))
    except ValueError:
        return 0.30


def classify_page(
    char_count: int,
    image_count: int,
    has_text_layer: bool,
    drawing_count: int,
    text_threshold: Optional[int] = None,
    *,
    word_count: int = 0,
    numeric_token_ratio: float = 0.0,
    image_area_ratio: float = 0.0,
) -> str:
    """Return the page class from its signals."""
    threshold = text_threshold if text_threshold is not None else _text_threshold()

    # 1. No usable text layer -> scanned (or a vector grid with no text)
    if not has_text_layer or char_count < SCANNED_MAX_CHARS:
        if image_count >= 1:
            return "scanned_page"
        if drawing_count >= HIGH_DRAWINGS:
            return "table_page"
        return "scanned_page"

    text_heavy = char_count >= threshold
    many_drawings = drawing_count >= HIGH_DRAWINGS
    has_image = image_count >= 1

    # 2. lots of prose, nothing visual -> text only
    if text_heavy and not has_image and not many_drawings:
        return "text_only"
    # 3. a raster image dominating the page with little prose -> image heavy
    if has_image and image_area_ratio >= IMAGE_HEAVY_AREA and char_count <= IMAGE_HEAVY_MAX_CHARS:
        return "image_heavy_page"
    # 4. short page dominated by large numbers / KPI cards
    if not text_heavy and 0 < word_count <= KPI_MAX_WORDS and numeric_token_ratio >= _kpi_num_ratio():
        return "kpi_page"
    # 5. vector grid dominated, little prose, no raster image -> table page
    if many_drawings and not has_image and not text_heavy:
        return "table_page"
    # 6. raster image(s) with modest text -> chart/figure page
    if has_image and not text_heavy:
        return "chart_page"
    # 7. substantial prose alongside visuals -> mixed
    if text_heavy and (has_image or many_drawings):
        return "mixed_page"
    # 8. fallbacks
    if has_image or many_drawings:
        return "mixed_page"
    return "text_only"


def classify_pages(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify every page; returns {"page_classes": [{page_number, page_class, signals}]}."""
    threshold = _text_threshold()
    drawings = {pt.get("page_number"): pt.get("drawing_count", 0) for pt in state.get("page_texts") or []}
    debug = os.getenv("DEBUG_REGIONS", os.getenv("OPENCV_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}

    page_classes: List[Dict[str, Any]] = []
    for meta in state.get("page_metadata") or []:
        page = meta.get("page_number")
        drawing_count = int(drawings.get(page, 0) or 0)
        page_class = classify_page(
            char_count=int(meta.get("char_count") or 0),
            image_count=int(meta.get("image_count") or 0),
            has_text_layer=bool(meta.get("has_text_layer")),
            drawing_count=drawing_count,
            text_threshold=threshold,
            word_count=int(meta.get("word_count") or 0),
            numeric_token_ratio=float(meta.get("numeric_token_ratio") or 0.0),
            image_area_ratio=float(meta.get("image_area_ratio") or 0.0),
        )
        signals = {
            "char_count": meta.get("char_count"),
            "image_count": meta.get("image_count"),
            "drawing_count": drawing_count,
            "has_text_layer": meta.get("has_text_layer"),
            "word_count": meta.get("word_count"),
            "numeric_token_ratio": meta.get("numeric_token_ratio"),
            "image_area_ratio": meta.get("image_area_ratio"),
        }
        page_classes.append({"page_number": page, "page_class": page_class, "signals": signals})
        if debug:
            logger.info("page %s -> %s | %s", page, page_class, signals)

    counts: Dict[str, int] = {}
    for entry in page_classes:
        counts[entry["page_class"]] = counts.get(entry["page_class"], 0) + 1
    logger.info("classify_pages: %d pages -> %s", len(page_classes), counts)
    return {"page_classes": page_classes}
