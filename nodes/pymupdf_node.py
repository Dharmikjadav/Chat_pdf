"""Stage 1 node — PyMuPDF ingestion.

Opens the source PDF, records per-page metadata, and renders every page to a
PNG under ``<output_dir>/page_images/``. Output is a *partial* update to
:class:`graph.state.PDFPipelineState` (``page_metadata``, ``page_images``, and
any accumulated ``errors``).

Also exposes :func:`crop_region`, used by later stages to cut a table/chart
region out of a rendered page before sending it to the Ollama vision model.

The node never raises for recoverable problems: a page that fails to render is
logged as a :class:`graph.state.PipelineError` and the pipeline continues. Only
an unopenable PDF short-circuits the stage (still returning a valid state
update, never an exception).
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PyMuPDF historically imports as ``fitz``; newer releases also expose ``pymupdf``.
try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - alternate package name
    import pymupdf as fitz  # type: ignore

from PIL import Image

from graph.state import PDFPipelineState, PageImage, PageMetadata, PipelineError
from utils.progress import report_progress

logger = logging.getLogger(__name__)

__all__ = [
    "pymupdf_node",
    "crop_region",
    "render_region",
    "ensure_page_image",
    "PAGE_IMAGE_SUBDIR",
    "DEFAULT_DPI",
]

_NUMERIC_RE = re.compile(r"^\(?[-+]?[\d,]+\.?\d*%?\)?$")
_SCAN_CHARS = 50
_GRID_DRAWINGS = 40
_KPI_NUM_RATIO = 0.30
_KPI_MAX_WORDS = 120


def _numeric_ratio(words: List[Any]) -> float:
    if not words:
        return 0.0
    hits = sum(1 for w in words if len(w) > 4 and _NUMERIC_RE.match(str(w[4]).strip()))
    return hits / len(words)


def _image_area_ratio(page) -> float:
    try:
        page_area = float(page.rect.width) * float(page.rect.height)
        if page_area <= 0:
            return 0.0
        covered = 0.0
        for info in page.get_image_info():
            box = info.get("bbox")
            if box and len(box) == 4:
                covered += abs(box[2] - box[0]) * abs(box[3] - box[1])
        return min(1.0, covered / page_area)
    except Exception:  # noqa: BLE001 - image info is best-effort
        return 0.0


def _needs_render(meta: Dict[str, Any], drawing_count: int) -> bool:
    if not meta["has_text_layer"] or meta["char_count"] < _SCAN_CHARS:
        return True
    if meta.get("image_count", 0) >= 1:
        return True
    if drawing_count >= _GRID_DRAWINGS:
        return True
    if meta.get("numeric_token_ratio", 0.0) >= _KPI_NUM_RATIO and 0 < meta.get("word_count", 0) <= _KPI_MAX_WORDS:
        return True
    return False


def render_region(
    pdf_path: str,
    page_number: int,
    bbox: "BBox",
    output_path: str,
    *,
    src_dpi: int = 200,
    dpi: int = 300,
    pad: float = 2.0,
) -> str:
    """Re-render a page-image bbox region straight from the PDF at higher DPI.

    The crops produced during detection are at ``src_dpi`` (default 200); for the
    vision model a sharper crop improves accuracy. ``bbox`` is in ``src_dpi``
    pixel coordinates; it is converted to PDF points and re-rasterized at ``dpi``.

    Returns ``output_path``. Raises on a missing page / IO error (callers handle).
    """
    scale = 72.0 / float(src_dpi)  # src-pixel -> PDF points
    x0, y0, x1, y1 = (coord * scale for coord in bbox)
    clip = fitz.Rect(min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad)
    zoom = dpi / 72.0
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        clip = clip & page.rect  # clamp to page bounds
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out))
    finally:
        doc.close()
    return str(output_path)

#: Sub-directory (under the document's ``output_dir``) that holds page PNGs.
PAGE_IMAGE_SUBDIR = "page_images"

#: Default render resolution; overridable per-call or via ``RENDER_DPI`` env var.
DEFAULT_DPI = int(os.getenv("RENDER_DPI", "200"))

#: A pixel- or point-space bounding box: (x0, y0, x1, y1).
BBox = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pipeline_error(
    message: str,
    *,
    page_number: Optional[int] = None,
    exc: Optional[BaseException] = None,
) -> PipelineError:
    """Build a :class:`PipelineError` record for the ``ingestion`` stage."""
    err: PipelineError = {"stage": "ingestion", "message": message}
    if page_number is not None:
        err["page_number"] = page_number
    if exc is not None:
        err["exception_type"] = type(exc).__name__
    return err


def crop_region(
    image_path: str,
    bbox: BBox,
    output_path: str,
    *,
    scale: float = 1.0,
    padding: int = 0,
) -> str:
    """Crop ``bbox`` out of ``image_path`` and save it to ``output_path``.

    Args:
        image_path: Path to a previously rendered page image.
        bbox: ``(x0, y0, x1, y1)``. Corner order is normalised automatically.
        output_path: Where the cropped PNG is written (parents are created).
        scale: Multiplier applied to ``bbox`` before cropping. Use this when the
            box is in PDF points and the image was rendered at a higher DPI —
            e.g. ``scale = dpi / 72``. Defaults to ``1.0`` (box already in pixels).
        padding: Extra pixels added on every side after scaling.

    Returns:
        ``output_path``.

    Raises:
        ValueError: If the resulting region has zero or negative area.
        FileNotFoundError: If ``image_path`` does not exist.
        OSError: If the image cannot be read or written.
    """
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"crop source image not found: {src}")

    with Image.open(src) as img:
        img_w, img_h = img.size

        x0, y0, x1, y1 = (coord * scale for coord in bbox)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))

        left = max(0, int(round(left - padding)))
        top = max(0, int(round(top - padding)))
        right = min(img_w, int(round(right + padding)))
        bottom = min(img_h, int(round(bottom + padding)))

        if right <= left or bottom <= top:
            raise ValueError(
                f"empty crop for bbox={bbox} (scale={scale}, padding={padding}) "
                f"against image size {img.size}"
            )

        region = img.crop((left, top, right, bottom))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        region.save(out)

    return str(out)


def ensure_page_image(
    state: PDFPipelineState,
    page_number: int,
    *,
    dpi: Optional[int] = None,
) -> Optional[str]:
    """Return a rendered PNG path for ``page_number``, rendering on demand if deferred.

    Looks up ``state['page_images']``; if the page was not pre-rendered, rasterizes it once,
    records it on the state list, and returns the path. ``None`` only on render failure.
    """
    for img in state.get("page_images") or []:
        if img.get("page_number") == page_number and img.get("image_path"):
            return img["image_path"]

    dpi = dpi or DEFAULT_DPI
    image_dir = Path(state["output_dir"]) / PAGE_IMAGE_SUBDIR
    pad = max(4, len(str(len(state.get("page_metadata") or []) or page_number)))
    out_path = image_dir / f"page_{page_number:0{pad}d}.png"
    try:
        image_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(state["pdf_path"])
        try:
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
            pix.save(str(out_path))
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - render-on-demand is best-effort
        logger.exception("ensure_page_image failed for page %d", page_number)
        return None

    state.setdefault("page_images", []).append(
        {"page_number": page_number, "image_path": str(out_path), "dpi": dpi}
    )
    return str(out_path)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
def pymupdf_node(
    state: PDFPipelineState,
    *,
    dpi: Optional[int] = None,
) -> Dict[str, Any]:
    """Open the PDF, collect page metadata, and render each page to PNG.

    Args:
        state: Current pipeline state. Requires ``pdf_path`` and ``output_dir``;
            ``document_name`` is used as the page ``source`` when present.
        dpi: Render resolution. Falls back to ``DEFAULT_DPI``.

    Returns:
        A partial :class:`PDFPipelineState` update with ``page_metadata``,
        ``page_images`` and (append-only) ``errors``.
    """
    dpi = dpi or DEFAULT_DPI
    zoom = dpi / 72.0  # PDF user space is 72 points / inch
    matrix = fitz.Matrix(zoom, zoom)
    render_all = os.getenv("RENDER_ALL", "").strip().lower() in {"1", "true", "yes", "on"}

    pdf_path = state["pdf_path"]
    source = state.get("document_name") or pdf_path
    image_dir = Path(state["output_dir"]) / PAGE_IMAGE_SUBDIR

    page_metadata: List[PageMetadata] = []
    page_images: List[PageImage] = []
    page_texts: List[Dict[str, Any]] = []
    errors: List[PipelineError] = []

    # --- Open the document (fatal-but-graceful on failure) ---------------- #
    try:
        image_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - report any open/IO failure
        logger.exception("Failed to open PDF: %s", pdf_path)
        errors.append(
            _pipeline_error(f"could not open PDF '{pdf_path}': {exc}", exc=exc)
        )
        return {"page_metadata": [], "page_images": [], "page_texts": [], "errors": errors}

    try:
        page_count = doc.page_count
        max_pages = int(os.getenv("MAX_PAGES", os.getenv("MARKER_MAX_PAGES", "0")) or "0")
        if max_pages > 0:
            page_count = min(page_count, max_pages)
        report_progress("rendering_pages", 0, page_count, "rendering pages")
        pad = max(4, len(str(page_count)))  # zero-pad filenames consistently
        logger.info("Opened '%s' (%d pages) @ %d DPI", pdf_path, page_count, dpi)

        for index in range(page_count):
            page_number = index + 1  # 1-based, preserved end-to-end
            try:
                page = doc.load_page(index)
                rect = page.rect

                text = page.get_text("text")
                char_count = len(text)
                try:
                    drawing_count = len(page.get_drawings())
                except Exception:  # noqa: BLE001 - drawings optional
                    drawing_count = 0
                words = page.get_text("words")
                word_count = len(words)
                meta: PageMetadata = {
                    "page_number": page_number,
                    "width": float(rect.width),
                    "height": float(rect.height),
                    "rotation": int(page.rotation),
                    "has_text_layer": char_count > 0,
                    "char_count": char_count,
                    "source": source,
                    "image_count": len(page.get_images(full=False)),
                    "word_count": word_count,
                    "block_count": len(page.get_text("blocks")),
                    "numeric_token_ratio": round(_numeric_ratio(words), 4),
                    "image_area_ratio": round(_image_area_ratio(page), 4),
                }
                meta["needs_render"] = _needs_render(meta, drawing_count)
                page_metadata.append(meta)
                page_texts.append(
                    {"page_number": page_number, "text": text, "drawing_count": drawing_count}
                )
                report_progress("rendering_pages", page_number, page_count)

                # Render -> PNG only when a later stage needs the raster (render-on-demand for the rest).
                if render_all or meta["needs_render"]:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    image_path = image_dir / f"page_{page_number:0{pad}d}.png"
                    pix.save(str(image_path))
                    page_images.append(
                        {
                            "page_number": page_number,
                            "image_path": str(image_path),
                            "dpi": dpi,
                            "width": int(pix.width),
                            "height": int(pix.height),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - isolate per-page failures
                logger.exception("Failed on page %d of %s", page_number, pdf_path)
                errors.append(
                    _pipeline_error(
                        f"failed to process page {page_number}: {exc}",
                        page_number=page_number,
                        exc=exc,
                    )
                )
    finally:
        doc.close()

    logger.info(
        "PyMuPDF stage done: %d/%d pages rendered to %s (%d error(s))",
        len(page_images),
        len(page_metadata),
        image_dir,
        len(errors),
    )

    return {
        "page_metadata": page_metadata,
        "page_images": page_images,
        "page_texts": page_texts,
        "errors": errors,
    }
