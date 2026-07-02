"""LangGraph workflow wiring the offline PDF -> PageIndex pipeline.

START
  -> pymupdf_process
  -> marker_process
  -> parse_marker_markdown
  -> process_complex_tables   (conditional: only if complex_table_candidates)
  -> process_charts           (conditional: only if chart_candidates)
  -> validate_all_json
  -> build_pageindex
  -> END

Each node returns only the state fields it owns. Vision stages are skipped
entirely when there is nothing for them to do.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graph.state import PDFPipelineState
from nodes.marker_node import marker_node
from nodes.ollama_vision_node import extract_chart, extract_complex_table
from pageindex_store import PageIndex
from utils.json_validator import validate_node
from utils.markdown_parser import markdown_parser_node
from utils.progress import report_progress

logger = logging.getLogger(__name__)

__all__ = ["build_pipeline", "run_pipeline"]


def _slug(name: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-")
    return cleaned or "doc"


def _vision_model() -> str:
    return os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b").strip()


def _page_image_path(state: PDFPipelineState, page_number: Optional[int]) -> Optional[str]:
    if page_number is None:
        return None
    for image in state.get("page_images") or []:
        if image.get("page_number") == page_number and image.get("image_path"):
            return image.get("image_path")
    # deferred page (render-on-demand): rasterize it now
    from nodes.pymupdf_node import ensure_page_image

    return ensure_page_image(state, page_number)


def _resolve_image(state: PDFPipelineState, candidate: Dict[str, Any]) -> Optional[str]:
    image_path = candidate.get("image_path")
    if image_path and Path(image_path).is_file():
        return image_path
    fallback = _page_image_path(state, candidate.get("page_number"))
    return fallback or image_path


def _pageindex_path(state: PDFPipelineState) -> str:
    return str(Path(state["output_dir"]) / "pageindex.json")


# --------------------------------------------------------------------------- #
# Pipeline nodes (each returns only the fields it owns)
# --------------------------------------------------------------------------- #
def pymupdf_process(state: PDFPipelineState) -> Dict[str, Any]:
    from nodes.pymupdf_node import pymupdf_node  # lazy: heavy deps (fitz, PIL)

    return pymupdf_node(state)


def marker_process(state: PDFPipelineState) -> Dict[str, Any]:
    return marker_node(state)


# --- Optimized flow wrappers (default, USE_MARKER=False) -------------------- #
def render_pages_with_pymupdf(state: PDFPipelineState) -> Dict[str, Any]:
    return pymupdf_process(state)


def classify_pages(state: PDFPipelineState) -> Dict[str, Any]:
    from nodes.page_classifier import classify_pages as _classify_pages

    total = len(state.get("page_metadata") or [])
    report_progress("page_classification", total, total, "classifying pages")
    return _classify_pages(state)


def _text_node(document_name, page, text, *, origin, page_class, idx=0, title=None) -> Dict[str, Any]:
    suffix = "text" if idx == 0 else f"s{idx:02d}"
    return {
        "id": f"{_slug(document_name)}-p{(page or 0):03d}-{suffix}",
        "document_name": document_name,
        "page_number": page,
        "source": document_name,
        "element_type": "text",
        "content": text,
        "title": title,
        "data": None,
        "bbox": None,
        "confidence": 1.0 if origin == "pymupdf_text" else 0.9,
        "metadata": {"origin": origin, "page_class": page_class, "section": title},
    }


def _split_sections(page) -> List[Tuple[Optional[str], str]]:
    """Split a page's prose into ordered (heading, content) sections using font cues.

    A line is treated as a heading when its font is clearly larger than the page's body font, or it
    is bold, short, and at least body size. Each section's content includes its heading line so the
    heading is searchable in both the boosted title and the body. A page with no detectable headings
    returns a single (None, full_text) section, so plain pages behave exactly as before."""
    try:
        info = page.get_text("dict")
    except Exception:  # noqa: BLE001 - fall back to flat text on any extraction issue
        return []
    lines: List[Tuple[float, bool, str]] = []
    for block in info.get("blocks") or []:
        if block.get("type") != 0:  # text blocks only
            continue
        for ln in block.get("lines") or []:
            spans = ln.get("spans") or []
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            size = max((float(s.get("size", 0.0)) for s in spans), default=0.0)
            bold = any(int(s.get("flags", 0)) & 16 for s in spans)
            lines.append((round(size, 1), bold, text))
    if not lines:
        return []
    from collections import Counter

    body = Counter(sz for sz, _, _ in lines).most_common(1)[0][0] or 1.0

    def is_heading(sz: float, bold: bool, text: str) -> bool:
        if len(text) > 110 or text.endswith((".", ",", ";", ":")) and len(text) > 70:
            return False
        return sz >= body * 1.15 or (bold and sz >= body and len(text) <= 70)

    sections: List[Tuple[Optional[str], List[str]]] = []
    cur_title: Optional[str] = None
    cur_body: List[str] = []
    for sz, bold, text in lines:
        if is_heading(sz, bold, text):
            if cur_title is not None or cur_body:
                sections.append((cur_title, cur_body))
            cur_title, cur_body = text, [text]
        else:
            cur_body.append(text)
    if cur_title is not None or cur_body:
        sections.append((cur_title, cur_body))

    raw: List[Tuple[Optional[str], str]] = []
    for title, body_lines in sections:
        content = "\n".join(body_lines).strip()
        if content:
            raw.append((title, content))

    # Merge fragments (labels, table cells, page numbers, signatures) into the substantial prose
    # section they belong to, so only real headings with real bodies become separate nodes. A page
    # made entirely of fragments (a table/KPI page) collapses back to a single node, as before.
    min_section = 160
    merged: List[Tuple[Optional[str], str]] = []
    buf_title: Optional[str] = None
    buf: List[str] = []
    for title, content in raw:
        if title is not None and len(content) >= min_section:  # a real section begins
            if buf:
                merged.append((buf_title, "\n".join(buf).strip()))
            buf_title, buf = title, [content]
        else:  # fragment -> attach to the section currently being accumulated
            if not buf and title is not None:
                buf_title = title
            buf.append(content)
    if buf:
        merged.append((buf_title, "\n".join(buf).strip()))
    return merged


def _section_is_prose(content: str) -> bool:
    """True if a section carries real words (headings/navigation/notes), not just table numbers.

    Used to recover meaningful text from chart/table pages (e.g. a 'What's inside' contents page or
    a chart caption) without duplicating the pure numeric cells already captured as table data."""
    words = content.split()
    if not words:
        return False
    alpha = sum(1 for w in words if sum(ch.isalpha() for ch in w) >= 2)
    return alpha >= 8 and alpha >= 0.3 * len(words)


def extract_text_only_pages(state: PDFPipelineState) -> Dict[str, Any]:
    """text_nodes: PyMuPDF text (split into heading sections) for text/mixed pages, OCR for scanned.

    Chart/table pages also contribute their real prose sections (TOC, headings, captions, notes) so
    navigational text is never lost; pure numeric table content is filtered out (kept as table data)."""
    import fitz  # lazy

    classes = {pc.get("page_number"): pc.get("page_class") for pc in state.get("page_classes") or []}
    texts = {pt.get("page_number"): pt.get("text", "") for pt in state.get("page_texts") or []}
    document_name = state.get("document_name")
    ocr_model = _vision_model()
    new_nodes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    pdf_path = state.get("pdf_path")
    doc = fitz.open(pdf_path) if pdf_path else None
    sectioned = 0
    try:
        total = len(state.get("page_metadata") or [])
        for meta in state.get("page_metadata") or []:
            page = meta.get("page_number")
            report_progress("page_classification", page, total, "extracting text")
            page_class = classes.get(page, "mixed_page")
            is_full_text = page_class in ("text_only", "mixed_page", "kpi_page", "image_heavy_page")
            is_data = page_class in ("table_page", "chart_page")
            if is_full_text or is_data:
                sections = _split_sections(doc.load_page(page - 1)) if (doc is not None and meta.get("has_text_layer")) else []
                if is_data:  # data page: keep only real prose, never duplicate pure table numbers
                    sections = [(t, c) for (t, c) in sections if _section_is_prose(c)]
                if is_full_text and len(sections) <= 1:
                    # plain page with no real headings -> one node for the whole page (unchanged behavior)
                    text = (texts.get(page) or "").strip()
                    if text:
                        new_nodes.append(_text_node(document_name, page, text, origin="pymupdf_text", page_class=page_class))
                elif sections:  # one node per heading (multi-section pages, or recovered chart/table prose)
                    if len(sections) > 1:
                        sectioned += 1
                    for i, (title, content) in enumerate(sections):
                        new_nodes.append(_text_node(document_name, page, content, origin="pymupdf_text",
                                                    page_class=page_class, idx=i, title=title))
            elif page_class == "scanned_page":
                image_path = _page_image_path(state, page)
                if not image_path:
                    errors.append({"stage": "vision", "message": "no page image for scanned OCR", "page_number": page})
                    continue

                from utils.surya_ocr import ocr_image

                text = ocr_image(image_path).strip()
                origin = "surya_ocr"
                if not text:  # Surya disabled/failed -> Qwen2.5-VL OCR
                    from nodes.ollama_vision_node import extract_text

                    result = extract_text(image_path, model=ocr_model)
                    if isinstance(result, dict) and "error" in result:
                        errors.append({"stage": "vision", "message": result["error"], "page_number": page, "exception_type": result.get("error_type")})
                        continue
                    text = (result if isinstance(result, str) else "").strip()
                    origin = "vision_ocr"
                if text:
                    new_nodes.append(_text_node(document_name, page, text, origin=origin, page_class=page_class))
            # table_page / chart_page: prose skipped; their data comes from crops
    finally:
        if doc is not None:
            doc.close()

    logger.info("extract_text_only_pages: %d text node(s) (%d page(s) split into sections), %d error(s)",
                len(new_nodes), sectioned, len(errors))
    return {"text_nodes": list(state.get("text_nodes") or []) + new_nodes, "errors": errors}


def process_table_crops(state: PDFPipelineState) -> Dict[str, Any]:
    from nodes.table_processor import process_table_crops as _ptc

    total = len(state.get("page_metadata") or [])
    report_progress("table_extraction", total, total, "extracting tables")
    return _ptc(state)


def process_chart_crops(state: PDFPipelineState) -> Dict[str, Any]:
    from nodes.chart_processor import process_chart_crops as _pcc

    total = len(state.get("page_metadata") or [])
    report_progress("chart_extraction", total, total, "extracting charts")
    return _pcc(state)


def _use_kpi() -> bool:
    return os.getenv("USE_KPI", "1").strip().lower() in {"1", "true", "yes", "on"}


def process_kpi_crops(state: PDFPipelineState) -> Dict[str, Any]:
    from nodes.kpi_processor import process_kpi_crops as _pkc

    total = len(state.get("page_metadata") or [])
    report_progress("chart_extraction", total, total, "extracting KPIs")
    return _pkc(state)


_REGION_NUM_RE = re.compile(r"^\(?[-+]?[\d,]+\.?\d*%?\)?$")


def _is_prose(words_in_region, region_w_pts: float) -> bool:
    """True if the region's text reads as flowing sentences (wide lines, no column gaps).

    Tables/charts lay text out in columns/labels (a big horizontal gap inside a line, or few words
    per line); a paragraph fills each line continuously. Such prose is already in the text layer."""
    lines: Dict[int, List[Any]] = {}
    for w in words_in_region:
        lines.setdefault(int(w[1] / 3.0), []).append(w)
    full, total = 0, 0
    for ws in lines.values():
        if len(ws) < 4:
            continue
        ws.sort(key=lambda w: w[0])
        if (ws[-1][2] - ws[0][0]) < 0.55 * region_w_pts:  # short line -> a label, not prose
            continue
        total += 1
        max_gap = max((ws[i + 1][0] - ws[i][2]) for i in range(len(ws) - 1))
        if max_gap < 0.08 * region_w_pts:  # no column gap -> continuous sentence
            full += 1
    return total >= 3 and full / total > 0.7


def _grid_shape(inb) -> Tuple[int, int, int]:
    """(columns, rows, numeric_count) of the words in a region — a numeric grid is a table."""
    if len(inb) < 4:
        return (0, 0, 0)
    ys = sorted((w[1] + w[3]) / 2 for w in inb)
    yh = sum(w[3] - w[1] for w in inb) / len(inb)
    nrow = 1 + sum(1 for i in range(1, len(ys)) if ys[i] - ys[i - 1] > 0.8 * yh)
    xs = sorted(w[0] for w in inb)
    span = (xs[-1] - xs[0]) or 1.0
    ncol = 1 + sum(1 for i in range(1, len(xs)) if xs[i] - xs[i - 1] > max(20.0, 0.05 * span))
    nn = sum(1 for w in inb if _REGION_NUM_RE.match(str(w[4]).strip()))
    return (ncol, nrow, nn)


def _data_regions(words, regions, dpi) -> List[Dict[str, Any]]:
    """Keep only regions carrying real numeric DATA; drop prose and decorative graphics.

    Uses the PDF text layer: data tables/charts sit over numeric labels in columns; a prose
    paragraph has wide continuous lines (already a text node); a decorative graphic has ~no words.
    A region detected as chart/visual that is actually a numeric GRID is re-typed to 'table' and its
    crop widened to the full row width, so the table extractor reads every column (the chart
    extractor would drop columns). KPI cards are kept; scanned pages are never filtered (caller).
    """
    sc = 72.0 / float(dpi or 200)
    kept: List[Dict[str, Any]] = []
    for r in regions:
        if r.get("type") == "kpi_card":
            kept.append(r)
            continue
        x0, y0, x1, y1 = [c * sc for c in r["bbox"]]
        inb = [w for w in words if w[0] >= x0 - 2 and w[2] <= x1 + 2 and w[1] >= y0 - 2 and w[3] <= y1 + 2]
        nw = len(inb)
        nn = sum(1 for w in inb if _REGION_NUM_RE.match(str(w[4]).strip()))
        if not (nn >= 2 and (nn / nw) >= 0.12 and not _is_prose(inb, (x1 - x0))):
            continue
        if r.get("type") in ("chart", "visual"):
            nc, nr, gnn = _grid_shape(inb)
            if nc >= 2 and nr >= 2 and gnn >= 3:  # a numeric grid is a table, not a chart
                row_words = [w for w in words if w[1] >= y0 - 2 and w[3] <= y1 + 2]
                r = dict(r, type="table")
                if row_words:
                    left = min(w[0] for w in row_words) / sc
                    right = max(w[2] for w in row_words) / sc
                    r["bbox"] = [max(0, int(left) - 6), r["bbox"][1], int(right) + 6, r["bbox"][3]]
        kept.append(r)
    return kept


def _column_gutter(words, page_width_pts) -> Optional[float]:
    """Return the split x (PDF points) for a genuine two-column page, else None.

    A two-column page has a NARROW empty vertical gutter near the centre with real content on BOTH
    sides. We merge word x-intervals and pick the widest central gap; require gutter 2-20% of width
    and content >=8% on each side (so single-column pages with an empty margin are NOT split)."""
    W = float(page_width_pts or 0)
    if not words or W <= 0:
        return None
    intervals = sorted((float(w[0]), float(w[2])) for w in words if w[2] > w[0])
    if not intervals:
        return None
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    lo, hi = W * 0.30, W * 0.70
    best = None
    for i in range(len(merged) - 1):
        g0, g1 = merged[i][1], merged[i + 1][0]
        if lo <= (g0 + g1) / 2 <= hi and (best is None or (g1 - g0) > best[1] - best[0]):
            best = (g0, g1)
    if not best or not (W * 0.02 <= best[1] - best[0] <= W * 0.20):
        return None
    left = sum(b - a for a, b in merged if b <= best[0])
    right = sum(b - a for a, b in merged if a >= best[1])
    if left < W * 0.08 or right < W * 0.08:
        return None
    return (best[0] + best[1]) / 2.0


def _detect_split_page(image_path, page, split_px, words, dpi, detect_fn, debug, debug_dir):
    """Detect regions on a two-column page by splitting it into left/right single-column halves and
    running the normal OpenCV detector + data-region filter on EACH half in the half's own coordinate
    space (so nothing crosses the gutter), then mapping boxes back to full-page coordinates."""
    from PIL import Image

    img = Image.open(image_path)
    W, H = img.size
    sc = 72.0 / float(dpi or 200)  # image px -> PDF points
    regions: List[Dict[str, Any]] = []
    gpu = False
    for x0, x1 in ((0, int(split_px)), (int(split_px), W)):
        if x1 - x0 < 40:
            continue
        x0_pt, x1_pt = x0 * sc, x1 * sc
        half_words = [(w[0] - x0_pt, w[1], w[2] - x0_pt, w[3]) + tuple(w[4:])
                      for w in words if x0_pt - 2 <= w[0] < x1_pt]
        tmp = str(Path(image_path).with_name(f"{Path(image_path).stem}_half{x0}.png"))
        try:
            img.crop((x0, 0, x1, H)).save(tmp)
            res = detect_fn(tmp, page_number=page, debug=debug, debug_dir=debug_dir)
            gpu = gpu or bool(res.get("gpu"))
            for r in _data_regions(half_words, res.get("regions", []), dpi):
                b = r["bbox"]
                regions.append(dict(r, bbox=[b[0] + x0, b[1], b[2] + x0, b[3]]))
        except Exception:  # noqa: BLE001 - a half that fails just contributes no regions
            logger.exception("split-half detection failed on page %s", page)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return {"page": page, "regions": regions, "gpu": gpu, "two_column": True}


def opencv_detect_regions(state: PDFPipelineState) -> Dict[str, Any]:
    """Stage 1b: detect table/chart/kpi/visual regions, then keep only data-bearing ones.

    Two-column pages are split at their gutter into single-column halves and run through the normal
    detector per half (so side-by-side tables are detected separately, like single-column pages),
    then boxes are mapped back to the full page. Single-column pages take the normal path unchanged."""
    try:
        from nodes.opencv_region_detector import detect_page_regions  # lazy: cv2
    except ImportError as exc:
        logger.warning("OpenCV unavailable, skipping region detection: %s", exc)
        return {
            "page_regions": [],
            "errors": [{"stage": "region_detection", "message": f"opencv unavailable: {exc}", "exception_type": "ImportError"}],
        }

    import fitz  # lazy

    debug = os.getenv("DEBUG_REGIONS", os.getenv("OPENCV_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    debug_dir = str(Path(state["output_dir"]) / "debug")
    class_by_page = {pc.get("page_number"): pc.get("page_class") for pc in state.get("page_classes") or []}
    meta_by_page = {m.get("page_number"): m for m in state.get("page_metadata") or []}
    pdf_path = state.get("pdf_path")

    page_regions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total = len(state.get("page_images") or [])
    dropped = 0
    doc = fitz.open(pdf_path) if pdf_path else None
    try:
        for image in state.get("page_images") or []:
            page = image.get("page_number")
            report_progress("opencv_region_detection", page, total, "detecting regions")
            # Routing rule: text_only pages skip OpenCV+vision (their text is captured directly).
            if class_by_page.get(page) == "text_only":
                page_regions.append({"page": page, "regions": [], "skipped": "text_only"})
                continue
            try:
                has_text = bool((meta_by_page.get(page) or {}).get("has_text_layer"))
                words = doc.load_page(page - 1).get_text("words") if (doc is not None and has_text) else None
                # Two-column page -> split at the gutter and detect each half as a single column.
                split_px = None
                if words:
                    gutter = _column_gutter(words, doc.load_page(page - 1).rect.width)
                    if gutter is not None:
                        split_px = int(gutter * (image.get("dpi") or 200) / 72.0)
                if split_px:
                    # split path detects + filters per half internally (in each half's coord space)
                    result = _detect_split_page(image["image_path"], page, split_px, words,
                                                image.get("dpi") or 200, detect_page_regions, debug, debug_dir)
                else:
                    result = detect_page_regions(image["image_path"], page_number=page, debug=debug, debug_dir=debug_dir)
                    # On a page with a text layer, drop prose/decorative regions (keep only data).
                    if words is not None and result.get("regions"):
                        before = len(result["regions"])
                        result["regions"] = _data_regions(words, result["regions"], image.get("dpi"))
                        dropped += before - len(result["regions"])
                page_regions.append(result)
            except Exception as exc:  # noqa: BLE001 - isolate per-page detection failures
                logger.exception("region detection failed on page %s", page)
                errors.append(
                    {"stage": "region_detection", "message": f"page {page}: {exc}", "page_number": page, "exception_type": type(exc).__name__}
                )
    finally:
        if doc is not None:
            doc.close()

    kept = sum(len(p.get("regions") or []) for p in page_regions)
    logger.info("opencv_detect_regions: %d data region(s) kept, %d non-data dropped across %d page(s)",
                kept, dropped, len(page_regions))
    return {"page_regions": page_regions, "errors": errors}


def _chart_color_frac(image_path: Optional[str]) -> float:
    """Fraction of saturated-colour pixels in a crop. Charts (bars/slices/segments) are colour-filled;
    financial tables are black text on white, so this cheaply flags chart-suspicious crops."""
    if not image_path:
        return 0.0
    try:
        import cv2  # lazy
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        colored = int(((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 50)).sum())
        return colored / float(img.shape[0] * img.shape[1])
    except Exception:  # noqa: BLE001 - colour check is best-effort
        return 0.0


def _words_in_bbox(words, bbox, dpi) -> int:
    """Count text-layer words inside a crop bbox (bbox is image px at ``dpi``; words are PDF points)."""
    if not words or not bbox:
        return 0
    sc = 72.0 / float(dpi or 200)
    x0, y0, x1, y1 = [c * sc for c in bbox]
    return sum(1 for w in words if w[0] >= x0 - 2 and w[2] <= x1 + 2 and w[1] >= y0 - 2 and w[3] <= y1 + 2)


def _crop_classifier_enabled() -> bool:
    """The per-crop vision classifier corrects OpenCV's chart/table label but costs one vision call per
    ambiguous crop (the bulk of ingest time). OFF by default (trust OpenCV routing, the fast pre-classifier
    behaviour); set CROP_CLASSIFIER=1 to re-enable it for more accurate chart/table routing at ~2x cost."""
    return os.getenv("CROP_CLASSIFIER", "0").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_crop_type(crop: Dict[str, Any], n_words: int, model: str, page_class: Optional[str] = None) -> str:
    """The TRUE type of a crop (table/chart/visual), so it is NAMED and routed correctly. The per-crop
    vision classifier is OFF by default (crops keep OpenCV's label — the fast pre-classifier behaviour).
    When re-enabled with CROP_CLASSIFIER=1, cheap skips still avoid a classifier call when the type is
    already clear, to bound cost:
      * an OpenCV 'table' on a table_page is almost certainly a table (charts live on chart/mixed/highlights
        pages), so trust the label and skip the call;
      * a big dense non-colourful region is clearly a table.
    'kpi_card' and other types are left untouched."""
    ctype = crop.get("type")
    if ctype not in ("table", "visual"):
        return ctype
    if not _crop_classifier_enabled():
        return ctype  # classifier off -> keep OpenCV's routing
    if ctype == "table" and page_class == "table_page":
        return ctype  # table region on a table-dense page -> trust OpenCV, skip the classifier call
    if n_words >= 80 and _chart_color_frac(crop.get("image_path")) < 0.10:
        return ctype  # clearly a dense table; don't spend a classifier call
    try:
        from nodes.ollama_vision_node import classify_region

        res = classify_region(crop["image_path"], model=model)
    except Exception:  # noqa: BLE001 - a classifier failure must not abort cropping; keep OpenCV's type
        return ctype
    if not isinstance(res, dict) or res.get("error"):
        return ctype
    verdict = str(res.get("type") or "").strip().lower()
    if verdict == "chart":
        return "chart"
    if verdict == "table":
        return "table"
    if verdict in ("figure", "image", "diagram"):
        return "visual"
    return ctype


def _rename_crops_by_type(crops: List[Dict[str, Any]], finals: List[str], document_name, page) -> int:
    """Renumber a page's crops by their FINAL type and rename the files on disk (two-phase, so a
    table_2->table_1 reshuffle never collides). Updates each crop's type/image_path/id in place;
    returns how many crops changed type."""
    counters: Dict[str, int] = {}
    plan: List[tuple] = []
    for crop, ftype in zip(crops, finals):
        counters[ftype] = counters.get(ftype, 0) + 1
        old_path = Path(crop["image_path"])
        new_path = old_path.with_name(f"{ftype}_{counters[ftype]}.png")
        new_id = f"{_slug(document_name)}-p{(page or 0):03d}-{ftype}-{counters[ftype]}"
        plan.append((crop, old_path, new_path, new_id, ftype))
    # phase 1: stage files that need a different name to a unique temp name
    for i, (crop, old_path, new_path, new_id, ftype) in enumerate(plan):
        crop["_stage"] = None
        if old_path.name != new_path.name and old_path.exists():
            tmp = old_path.with_name(f"__stage_p{page}_{i}.png")
            try:
                os.replace(str(old_path), str(tmp))
                crop["_stage"] = (tmp, new_path)
            except OSError:
                logger.warning("crop rename stage failed: %s", old_path)
    # phase 2: temp -> final; update records
    changed = 0
    for crop, old_path, new_path, new_id, ftype in plan:
        stage = crop.pop("_stage", None)
        if stage is not None:
            tmp, dst = stage
            try:
                os.replace(str(tmp), str(dst))
            except OSError:
                logger.warning("crop rename finalise failed: %s", dst)
        if ftype != crop.get("type"):
            changed += 1
        crop["type"] = ftype
        crop["image_path"] = str(new_path)
        crop["id"] = new_id
    return changed


def crop_regions(state: PDFPipelineState) -> Dict[str, Any]:
    """Stage 1c: crop detected regions and route them to extraction channels."""
    page_regions = state.get("page_regions") or []
    report_progress("region_cropping", len(page_regions), len(page_regions), "cropping regions")
    empty = {
        "region_crops": [],
        "complex_table_candidates": [],
        "chart_candidates": [],
        "visual_candidates": [],
        "kpi_candidates": [],
        "errors": [],
    }
    page_class_by_page = {
        pc.get("page_number"): pc.get("page_class") for pc in state.get("page_classes") or []
    }
    kpi_pages = {
        pc.get("page_number")
        for pc in state.get("page_classes") or []
        if pc.get("page_class") == "kpi_page"
    } if _use_kpi() else set()
    if not any(p.get("regions") for p in page_regions) and not kpi_pages:
        return empty

    try:
        from utils.region_cropper import crop_regions as _crop_regions  # lazy: cv2
    except ImportError as exc:
        logger.warning("OpenCV unavailable, skipping crops: %s", exc)
        empty["errors"] = [{"stage": "region_crop", "message": f"opencv unavailable: {exc}", "exception_type": "ImportError"}]
        return empty

    document_name = state.get("document_name")
    out_root = str(Path(state["output_dir"]) / "extracted_regions")
    vision_model = _vision_model()
    reclassified = 0
    import fitz  # lazy: text layer used by the dense-table guard
    doc = fitz.open(state.get("pdf_path")) if state.get("pdf_path") else None
    dpi_by_page = {im.get("page_number"): im.get("dpi") for im in state.get("page_images") or []}
    region_crops: List[Dict[str, Any]] = []
    table_c: List[Dict[str, Any]] = []
    chart_c: List[Dict[str, Any]] = []
    visual_c: List[Dict[str, Any]] = []
    kpi_c: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    try:
        for page_result in page_regions:
            if not page_result.get("regions"):
                continue
            page = page_result.get("page")
            image_path = _page_image_path(state, page)
            if not image_path:
                errors.append({"stage": "region_crop", "message": f"no page image for page {page}", "page_number": page})
                continue
            try:
                crops = _crop_regions(image_path, page_result, out_root, document_name=document_name, source=image_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("cropping failed on page %s", page)
                errors.append({"stage": "region_crop", "message": f"page {page}: {exc}", "page_number": page, "exception_type": type(exc).__name__})
                continue

            # Name + route each crop by its TRUE type (the vision classifier decides; dense tables skip it),
            # then rename the files so charts are chart_N.png and tables table_N.png.
            words = doc.load_page(page - 1).get_text("words") if (doc is not None and page) else None
            dpi = dpi_by_page.get(page)
            pclass = page_class_by_page.get(page)
            finals = [_resolve_crop_type(c, _words_in_bbox(words, c.get("bbox"), dpi), vision_model, pclass) for c in crops]
            reclassified += _rename_crops_by_type(crops, finals, document_name, page)
            region_crops.extend(crops)
            for crop in crops:
                ctype = crop["type"]
                candidate = {
                    "id": crop["id"],
                    "kind": ctype,
                    "document_name": document_name,
                    "page_number": crop["page_number"],
                    "source": crop["image_path"],
                    "image_path": crop["image_path"],
                    "confidence": crop["confidence"],
                    "bbox": crop["bbox"],
                    "reason": "opencv_region",
                }
                if ctype == "table":
                    table_c.append(candidate)
                elif ctype == "chart":
                    chart_c.append(candidate)
                elif ctype == "kpi_card" and _use_kpi():
                    kpi_c.append(candidate)
                elif ctype == "kpi_card":
                    continue
                else:
                    visual_c.append(candidate)
    finally:
        if doc is not None:
            doc.close()

    # KPI pages with no detected card -> send the whole page to the KPI extractor.
    have_kpi = {c["page_number"] for c in kpi_c}
    for page in sorted(kpi_pages - have_kpi):
        img = _page_image_path(state, page)
        if img:
            kpi_c.append({
                "id": f"{document_name}-p{page}-kpipage",
                "kind": "kpi_card",
                "document_name": document_name,
                "page_number": page,
                "source": img,
                "image_path": img,
                "confidence": 0.5,
                "bbox": None,
                "reason": "kpi_page",
            })

    logger.info(
        "crop_regions: %d crop(s) (tables=%d charts=%d kpis=%d visuals=%d); %d reclassified by vision",
        len(region_crops), len(table_c), len(chart_c), len(kpi_c), len(visual_c), reclassified,
    )
    return {
        "region_crops": region_crops,
        "complex_table_candidates": table_c,
        "chart_candidates": chart_c,
        "visual_candidates": visual_c,
        "kpi_candidates": kpi_c,
        "errors": errors,
    }


def parse_marker_markdown(state: PDFPipelineState) -> Dict[str, Any]:
    result = markdown_parser_node(state)

    document_name = state.get("document_name")
    source = state.get("marker_md_path") or document_name
    marker_charts: List[Dict[str, Any]] = []
    for index, image in enumerate(state.get("marker_images") or [], 1):
        marker_charts.append(
            {
                "id": f"{_slug(document_name)}-chart_candidate-{index:04d}",
                "kind": "chart",
                "document_name": document_name,
                "page_number": image.get("page_number"),
                "source": source,
                "image_path": image.get("path"),
                "reason": "marker_image",
            }
        )

    # Merge OpenCV-detected crops (already in state from crop_regions) with the
    # markdown/marker-derived candidates so both reach the vision stages.
    result["complex_table_candidates"] = (
        list(state.get("complex_table_candidates") or []) + (result.get("complex_table_candidates") or [])
    )
    result["chart_candidates"] = list(state.get("chart_candidates") or []) + marker_charts
    return result


def process_complex_tables(state: PDFPipelineState) -> Dict[str, Any]:
    model = _vision_model()
    document_name = state.get("document_name")
    errors: List[Dict[str, Any]] = []
    extracted: List[Dict[str, Any]] = []

    for candidate in state.get("complex_table_candidates") or []:
        image_path = _resolve_image(state, candidate)
        if not image_path:
            errors.append(
                {
                    "stage": "vision",
                    "message": "no image available for complex table",
                    "node_id": candidate.get("id"),
                    "page_number": candidate.get("page_number"),
                }
            )
            continue

        result = extract_complex_table(image_path, model=model)
        if isinstance(result, dict) and "error" in result:
            errors.append(
                {
                    "stage": "vision",
                    "message": result["error"],
                    "node_id": candidate.get("id"),
                    "page_number": candidate.get("page_number"),
                    "exception_type": result.get("error_type"),
                }
            )
            continue

        extracted.append(
            {
                "id": candidate.get("id"),
                "document_name": candidate.get("document_name") or document_name,
                "page_number": candidate.get("page_number"),
                "source": candidate.get("source"),
                "element_type": "table",
                "data": result if isinstance(result, dict) else None,
                "raw_markdown": candidate.get("raw_markdown"),
                "metadata": {"origin": "vision", "reason": candidate.get("reason")},
            }
        )

    table_nodes = list(state.get("table_nodes") or []) + extracted
    logger.info("process_complex_tables: %d extracted, %d error(s)", len(extracted), len(errors))
    return {"table_nodes": table_nodes, "errors": errors}


def process_charts(state: PDFPipelineState) -> Dict[str, Any]:
    model = _vision_model()
    document_name = state.get("document_name")
    errors: List[Dict[str, Any]] = []
    extracted: List[Dict[str, Any]] = []

    for candidate in state.get("chart_candidates") or []:
        image_path = _resolve_image(state, candidate)
        if not image_path:
            errors.append(
                {
                    "stage": "vision",
                    "message": "no image available for chart",
                    "node_id": candidate.get("id"),
                    "page_number": candidate.get("page_number"),
                }
            )
            continue

        result = extract_chart(image_path, model=model)
        if isinstance(result, dict) and "error" in result:
            errors.append(
                {
                    "stage": "vision",
                    "message": result["error"],
                    "node_id": candidate.get("id"),
                    "page_number": candidate.get("page_number"),
                    "exception_type": result.get("error_type"),
                }
            )
            continue

        extracted.append(
            {
                "id": candidate.get("id"),
                "document_name": candidate.get("document_name") or document_name,
                "page_number": candidate.get("page_number"),
                "source": candidate.get("source"),
                "element_type": "chart",
                "data": result if isinstance(result, dict) else None,
                "image_path": image_path,
                "metadata": {"origin": "vision", "reason": candidate.get("reason")},
            }
        )

    chart_nodes = list(state.get("chart_nodes") or []) + extracted
    logger.info("process_charts: %d extracted, %d error(s)", len(extracted), len(errors))
    return {"chart_nodes": chart_nodes, "errors": errors}


def _visual_description(classification: Any) -> str:
    if not isinstance(classification, dict):
        return str(classification)
    parts = [str(classification[key]) for key in ("type", "subtype", "title") if classification.get(key)]
    return " - ".join(parts) if parts else "visual region"


def process_visuals(state: PDFPipelineState) -> Dict[str, Any]:
    """Classify OpenCV 'large visual' crops via the Ollama vision classifier."""
    visual_candidates = state.get("visual_candidates") or []
    if not visual_candidates:
        return {"visual_nodes": list(state.get("visual_nodes") or []), "errors": []}

    from nodes.ollama_vision_node import classify_region

    model = _vision_model()
    document_name = state.get("document_name")
    nodes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for candidate in visual_candidates:
        image_path = _resolve_image(state, candidate)
        if not image_path:
            errors.append({"stage": "vision", "message": "no image for visual region", "node_id": candidate.get("id"), "page_number": candidate.get("page_number")})
            continue
        result = classify_region(image_path, model=model)
        if isinstance(result, dict) and "error" in result:
            errors.append({"stage": "vision", "message": result["error"], "node_id": candidate.get("id"), "page_number": candidate.get("page_number"), "exception_type": result.get("error_type")})
            continue
        nodes.append(
            {
                "id": candidate.get("id"),
                "document_name": candidate.get("document_name") or document_name,
                "page_number": candidate.get("page_number"),
                "source": candidate.get("source"),
                "element_type": "figure",
                "content": _visual_description(result),
                "data": result if isinstance(result, dict) else None,
                "image_path": image_path,
                "metadata": {"origin": "opencv_visual", "confidence": candidate.get("confidence")},
            }
        )

    visual_nodes = list(state.get("visual_nodes") or []) + nodes
    logger.info("process_visuals: %d classified, %d error(s)", len(nodes), len(errors))
    return {"visual_nodes": visual_nodes, "errors": errors}


def validate_all_json(state: PDFPipelineState) -> Dict[str, Any]:
    total = len(state.get("page_metadata") or [])
    report_progress("json_validation", total, total, "validating nodes")
    raw_nodes: List[Dict[str, Any]] = []
    raw_nodes.extend(state.get("text_nodes") or [])
    raw_nodes.extend(state.get("table_nodes") or [])
    raw_nodes.extend(state.get("chart_nodes") or [])
    raw_nodes.extend(state.get("visual_nodes") or [])

    validated = [validate_node(node) for node in raw_nodes]
    invalid = sum(1 for node in validated if not node.get("valid"))
    logger.info("validate_all_json: %d nodes (%d invalid)", len(validated), invalid)
    return {"validated_nodes": validated, "errors": []}


def _use_facts() -> bool:
    return os.getenv("USE_FACTS", "1").strip().lower() in {"1", "true", "yes", "on"}


def _build_facts(store, document_id, company) -> int:
    """Decompose every table/chart/kpi node in the store into searchable fact nodes."""
    if not _use_facts():
        return 0
    from utils.fact_generator import generate_facts

    count = 0
    for node in list(store.nodes.values()):
        if node.get("node_type") not in ("table", "chart", "kpi"):
            continue
        data = (node.get("metadata") or {}).get("data")
        if not isinstance(data, dict):
            continue
        page = node.get("page_number")
        shim = {
            "type": node["node_type"],
            "data": data,
            "page_number": page,
            "confidence": (node.get("metadata") or {}).get("confidence"),
        }
        try:
            facts = generate_facts(shim, document_id=document_id, company=company)
        except Exception:  # noqa: BLE001 - never let one node abort the build
            logger.exception("fact generation failed for node %s", node.get("id"))
            continue
        parent = node.get("parent_id") or node["id"]
        for f in facts:
            store.add_fact_node(
                parent,
                content=f["text"],
                title=f.get("metric"),
                page_number=page,
                metadata={"data": f, "source_node": node.get("id")},
            )
            count += 1
    return count


def build_pageindex(state: PDFPipelineState) -> Dict[str, Any]:
    document_name = state["document_name"]
    errors: List[Dict[str, Any]] = []
    total = len(state.get("page_metadata") or [])
    report_progress("pageindex_building", total, total, "building PageIndex")

    try:
        store = PageIndex()
        document_id = store.add_document(document_name)

        page_parents: Dict[Any, str] = {}
        for meta in sorted(
            state.get("page_metadata") or [],
            key=lambda item: item.get("page_number") or 0,
        ):
            page_number = meta.get("page_number")
            page_parents[page_number] = store.add_page(
                document_name,
                page_number,
                metadata={
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "has_text_layer": meta.get("has_text_layer"),
                    "source": meta.get("source"),
                },
            )

        def parent_for(page: Optional[int]) -> str:
            if page is None:
                return document_id
            if page not in page_parents:
                page_parents[page] = store.add_page(document_name, page)
            return page_parents[page]

        for node in state.get("validated_nodes") or []:
            page = node.get("page_number")
            parent = parent_for(page)
            node_type = node.get("type")
            title = node.get("title")
            node_id = node.get("id")
            meta = {
                "valid": node.get("valid"),
                "confidence": node.get("confidence"),
                "bbox": node.get("bbox"),
                "origin": (node.get("metadata") or {}).get("origin"),
            }
            data = node.get("data")

            if node_type == "table":
                store.add_table_node(
                    parent,
                    data=data if isinstance(data, dict) else None,
                    title=title,
                    page_number=page,
                    metadata=meta,
                    node_id=node_id,
                )
            elif node_type == "chart":
                store.add_chart_node(
                    parent,
                    data=data if isinstance(data, dict) else None,
                    title=title,
                    page_number=page,
                    metadata=meta,
                    node_id=node_id,
                )
            else:
                content = data if isinstance(data, str) else (node.get("content") or "")
                store.add_text_node(
                    parent,
                    content=content,
                    title=title,
                    page_number=page,
                    metadata=meta,
                    node_id=node_id,
                )

        for kn in state.get("kpi_nodes") or []:
            page = kn.get("page_number")
            store.add_kpi_node(
                parent_for(page),
                data=kn.get("data") if isinstance(kn.get("data"), dict) else None,
                content=kn.get("content"),
                title=kn.get("title"),
                page_number=page,
                metadata={
                    "valid": kn.get("valid", True),
                    "confidence": kn.get("confidence"),
                    "bbox": kn.get("bbox"),
                    "origin": (kn.get("metadata") or {}).get("origin"),
                },
                node_id=kn.get("id"),
            )

        # Structure financial-statement text into companion tables so individual line items
        # (e.g. "revenue from operations") are retrievable. Offline/deterministic; never fatal.
        try:
            from utils.statement_parser import enrich_store_with_statements

            n_stmt = enrich_store_with_statements(store)
            if n_stmt:
                logger.info("statement parser added %d structured statement tables", n_stmt)
        except Exception:  # noqa: BLE001 - enrichment is best-effort
            logger.exception("statement enrichment failed (continuing)")

        n_facts = _build_facts(store, document_name, state.get("company"))
        if n_facts:
            logger.info("fact generator added %d fact nodes", n_facts)

        path = _pageindex_path(state)
        store.save(path)
    except Exception as exc:  # noqa: BLE001 - surface storage failure, do not crash graph
        logger.exception("build_pageindex failed")
        return {
            "pageindex_path": None,
            "final_output": {"document_name": document_name, "status": "failed"},
            "errors": [
                {"stage": "storage", "message": f"build_pageindex failed: {exc}", "exception_type": type(exc).__name__}
            ],
        }

    validated = state.get("validated_nodes") or []
    accumulated_errors = state.get("errors") or []
    num_tables = sum(1 for node in validated if node.get("type") == "table")
    num_charts = sum(1 for node in validated if node.get("type") == "chart")
    status = "success" if not accumulated_errors else "partial"
    if not validated:
        status = "failed"

    final_output = {
        "document_name": document_name,
        "status": status,
        "num_pages": len(state.get("page_metadata") or []),
        "num_nodes": len(validated),
        "num_tables": num_tables,
        "num_charts": num_charts,
        "num_errors": len(accumulated_errors),
        "pageindex_path": path,
    }
    logger.info("build_pageindex: saved %s (%s)", path, status)
    return {"pageindex_path": path, "final_output": final_output, "errors": errors}


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #
def route_after_parse(state: PDFPipelineState) -> str:
    if state.get("complex_table_candidates"):
        return "process_complex_tables"
    if state.get("chart_candidates"):
        return "process_charts"
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_all_json"


def route_after_tables(state: PDFPipelineState) -> str:
    if state.get("chart_candidates"):
        return "process_charts"
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_all_json"


def route_after_charts(state: PDFPipelineState) -> str:
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_all_json"


# Optimized-flow routers (operate on crop candidate lists; return new node names)
def route_after_crops(state: PDFPipelineState) -> str:
    if state.get("complex_table_candidates"):
        return "process_table_crops"
    if state.get("chart_candidates"):
        return "process_chart_crops"
    if state.get("kpi_candidates"):
        return "process_kpi_crops"
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_json"


def route_after_table_crops(state: PDFPipelineState) -> str:
    if state.get("chart_candidates"):
        return "process_chart_crops"
    if state.get("kpi_candidates"):
        return "process_kpi_crops"
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_json"


def route_after_chart_crops(state: PDFPipelineState) -> str:
    if state.get("kpi_candidates"):
        return "process_kpi_crops"
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_json"


def route_after_kpi_crops(state: PDFPipelineState) -> str:
    if state.get("visual_candidates"):
        return "process_visuals"
    return "validate_json"


def _use_marker() -> bool:
    return os.getenv("USE_MARKER", "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def _timed(name: str, fn):
    """Wrap a node so it logs ``[timing] <stage> <s>`` and records it on stage_timings."""
    def wrapper(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = round(time.perf_counter() - start, 2)
        logger.info("[timing] %s %.2fs", name, elapsed)
        if isinstance(result, dict):
            result["stage_timings"] = [{"stage": name, "seconds": elapsed}]
        return result
    return wrapper


def build_pipeline():
    """Compile and return the LangGraph application.

    Default = optimized flow (PyMuPDF text + find_tables + vision). Set
    ``USE_MARKER=1`` to rebuild the original whole-document Marker graph.
    """
    from langgraph.graph import END, START, StateGraph  # lazy: optional dep

    if _use_marker():
        return _build_marker_pipeline(StateGraph, START, END)

    graph = StateGraph(PDFPipelineState)
    graph.add_node("render_pages_with_pymupdf", _timed("render_pages", render_pages_with_pymupdf))
    graph.add_node("classify_pages", _timed("classify_pages", classify_pages))
    graph.add_node("extract_text_only_pages", _timed("extract_text", extract_text_only_pages))
    graph.add_node("detect_regions_with_opencv", _timed("detect_regions", opencv_detect_regions))
    graph.add_node("crop_regions", _timed("crop_regions", crop_regions))
    graph.add_node("process_table_crops", _timed("tables", process_table_crops))
    graph.add_node("process_chart_crops", _timed("charts", process_chart_crops))
    graph.add_node("process_kpi_crops", _timed("kpis", process_kpi_crops))
    graph.add_node("process_visuals", _timed("visuals", process_visuals))
    graph.add_node("validate_json", _timed("validate", validate_all_json))
    graph.add_node("build_pageindex", _timed("build_pageindex", build_pageindex))

    graph.add_edge(START, "render_pages_with_pymupdf")
    graph.add_edge("render_pages_with_pymupdf", "classify_pages")
    graph.add_edge("classify_pages", "extract_text_only_pages")
    graph.add_edge("extract_text_only_pages", "detect_regions_with_opencv")
    graph.add_edge("detect_regions_with_opencv", "crop_regions")

    graph.add_conditional_edges(
        "crop_regions",
        route_after_crops,
        {
            "process_table_crops": "process_table_crops",
            "process_chart_crops": "process_chart_crops",
            "process_kpi_crops": "process_kpi_crops",
            "process_visuals": "process_visuals",
            "validate_json": "validate_json",
        },
    )
    graph.add_conditional_edges(
        "process_table_crops",
        route_after_table_crops,
        {
            "process_chart_crops": "process_chart_crops",
            "process_kpi_crops": "process_kpi_crops",
            "process_visuals": "process_visuals",
            "validate_json": "validate_json",
        },
    )
    graph.add_conditional_edges(
        "process_chart_crops",
        route_after_chart_crops,
        {
            "process_kpi_crops": "process_kpi_crops",
            "process_visuals": "process_visuals",
            "validate_json": "validate_json",
        },
    )
    graph.add_conditional_edges(
        "process_kpi_crops",
        route_after_kpi_crops,
        {
            "process_visuals": "process_visuals",
            "validate_json": "validate_json",
        },
    )
    graph.add_edge("process_visuals", "validate_json")
    graph.add_edge("validate_json", "build_pageindex")
    graph.add_edge("build_pageindex", END)

    return graph.compile()


def _build_marker_pipeline(StateGraph, START, END):
    """Original whole-document Marker graph (USE_MARKER=1)."""
    graph = StateGraph(PDFPipelineState)

    graph.add_node("pymupdf_process", pymupdf_process)
    graph.add_node("detect_regions", opencv_detect_regions)
    graph.add_node("crop_regions", crop_regions)
    graph.add_node("marker_process", marker_process)
    graph.add_node("parse_marker_markdown", parse_marker_markdown)
    graph.add_node("process_complex_tables", process_complex_tables)
    graph.add_node("process_charts", process_charts)
    graph.add_node("process_visuals", process_visuals)
    graph.add_node("validate_all_json", validate_all_json)
    graph.add_node("build_pageindex", build_pageindex)

    graph.add_edge(START, "pymupdf_process")
    graph.add_edge("pymupdf_process", "detect_regions")
    graph.add_edge("detect_regions", "crop_regions")
    graph.add_edge("crop_regions", "marker_process")
    graph.add_edge("marker_process", "parse_marker_markdown")

    graph.add_conditional_edges(
        "parse_marker_markdown",
        route_after_parse,
        {
            "process_complex_tables": "process_complex_tables",
            "process_charts": "process_charts",
            "process_visuals": "process_visuals",
            "validate_all_json": "validate_all_json",
        },
    )
    graph.add_conditional_edges(
        "process_complex_tables",
        route_after_tables,
        {
            "process_charts": "process_charts",
            "process_visuals": "process_visuals",
            "validate_all_json": "validate_all_json",
        },
    )
    graph.add_conditional_edges(
        "process_charts",
        route_after_charts,
        {
            "process_visuals": "process_visuals",
            "validate_all_json": "validate_all_json",
        },
    )
    graph.add_edge("process_visuals", "validate_all_json")
    graph.add_edge("validate_all_json", "build_pageindex")
    graph.add_edge("build_pageindex", END)

    return graph.compile()


def run_pipeline(pdf_path: str, *, progress_callback=None, **kwargs) -> Dict[str, Any]:
    """Build the graph and run it for a single PDF.

    ``progress_callback(stage, current_page, total_pages, message=None)`` is
    invoked throughout the run when provided (used by the API for live progress).
    """
    from graph.state import init_state
    from utils.progress import clear_progress_callback, report_progress, set_progress_callback

    if progress_callback is not None:
        set_progress_callback(progress_callback)
    app = build_pipeline()
    try:
        report_progress("uploaded", 0, 0, "starting")
        return app.invoke(init_state(pdf_path, **kwargs))
    finally:
        if progress_callback is not None:
            clear_progress_callback()
