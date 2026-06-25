"""Pure-Python easy-table extraction (pdfplumber), tried before the vision model.

Runs only on table-crop candidates, never on whole pages. The opened PDF is cached per
run and closed via :func:`close_cache` so 300-page documents do not re-open the file per crop.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["extract_table_pdfplumber", "close_cache"]

MIN_ROWS = 2
MIN_COLS = 2

_PDF_CACHE: Dict[str, Any] = {}


def _open(pdf_path: str):
    pdf = _PDF_CACHE.get(pdf_path)
    if pdf is None:
        import pdfplumber  # lazy: optional dependency

        pdf = pdfplumber.open(pdf_path)
        _PDF_CACHE[pdf_path] = pdf
    return pdf


def close_cache() -> None:
    """Close every cached pdfplumber handle (call once at end of a run)."""
    for pdf in _PDF_CACHE.values():
        try:
            pdf.close()
        except Exception:  # noqa: BLE001
            pass
    _PDF_CACHE.clear()


def extract_table_pdfplumber(
    pdf_path: str,
    page_number: int,
    bbox: List[int],
    src_dpi: int,
) -> Optional[Dict[str, Any]]:
    """Extract the largest table inside ``bbox`` via pdfplumber. None if nothing usable.

    ``bbox`` is in rendered-image pixels at ``src_dpi``; it is scaled to PDF points to
    crop the page. Output matches the find_tables shape so ``normalize_table_json`` consumes it.
    """
    try:
        scale = 72.0 / float(src_dpi)
        x0, y0, x1, y1 = (c * scale for c in bbox)
        pdf = _open(pdf_path)
        page = pdf.pages[page_number - 1]
        region = (
            max(0.0, min(x0, x1)),
            max(0.0, min(y0, y1)),
            min(float(page.width), max(x0, x1)),
            min(float(page.height), max(y0, y1)),
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            return None
        tables = page.within_bbox(region).extract_tables() or []
    except Exception as exc:  # noqa: BLE001 - any pdfplumber failure -> fall through
        logger.warning("pdfplumber failed on page %s: %s", page_number, exc)
        return None

    grids = [t for t in tables if t and len(t) >= MIN_ROWS]
    if not grids:
        return None
    best = max(grids, key=lambda g: len(g) * max((len(r) for r in g), default=0))
    if max((len(r) for r in best), default=0) < MIN_COLS:
        return None
    header = [("" if c is None else str(c).strip()) for c in best[0]]
    rows = [[("" if c is None else str(c).strip()) for c in r] for r in best[1:]]
    return {"type": "table", "title": None, "columns": header, "rows": rows}
