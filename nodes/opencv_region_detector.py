"""OpenCV-only region detection on rendered page images (pre-Marker stage).

Detects table, chart, and large-visual regions on a page raster and returns
their bounding boxes with confidence scores:

    {
      "page": 5,
      "regions": [
        {"type": "table",  "bbox": [x1, y1, x2, y2], "confidence": 0.91},
        {"type": "chart",  "bbox": [x1, y1, x2, y2], "confidence": 0.78}
      ],
      "gpu": True
    }

Techniques: grayscale, adaptive threshold, morphology, contour detection,
line detection (Hough), and area / aspect-ratio filtering. The heavy per-pixel
ops (grayscale, morphology, Canny) run on the GPU via ``cv2.cuda`` when a
CUDA-enabled OpenCV build is present, and fall back to CPU otherwise.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["detect_page_regions", "detect_regions", "gpu_available"]

BBox = List[int]

DEFAULT_PARAMS: Dict[str, float] = {
    "min_area_ratio": 0.012,   # region must cover >= 1.2% of the page
    "max_area_ratio": 0.92,    # ignore near-full-page blobs
    "min_aspect": 0.08,        # drop thin slivers (w/h)
    "max_aspect": 16.0,
    "iou_merge": 0.30,         # de-dup overlap threshold
    "pad": 6,                  # pixels added around each kept box
    "table_min_lines": 2,      # need >= 2 horizontal and 2 vertical lines
}

_COLORS = {
    "table": (0, 180, 0),
    "chart": (220, 120, 0),
    "visual": (0, 140, 255),
    "kpi_card": (180, 0, 180),
}


def gpu_available() -> bool:
    """True when OpenCV was built with CUDA and a device is present."""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:  # noqa: BLE001 - cv2.cuda missing on CPU-only builds
        return False


class _Backend:
    """Dispatches the GPU-accelerable ops to CUDA when available, else CPU."""

    def __init__(self, use_gpu: bool):
        self.gpu = bool(use_gpu) and gpu_available()

    def to_gray(self, img: "np.ndarray") -> "np.ndarray":
        if self.gpu:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                return gray.download()
            except Exception:  # noqa: BLE001 - fall back permanently
                self.gpu = False
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def morph_open(self, binary: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":
        if self.gpu:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(binary)
                flt = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, kernel)
                return flt.apply(gpu_img).download()
            except Exception:  # noqa: BLE001
                self.gpu = False
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def canny(self, gray: "np.ndarray", low: int = 50, high: int = 150) -> "np.ndarray":
        if self.gpu:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(gray)
                detector = cv2.cuda.createCannyEdgeDetector(low, high)
                return detector.detect(gpu_img).download()
            except Exception:  # noqa: BLE001
                self.gpu = False
        return cv2.Canny(gray, low, high)


def _clamp(value: float, low: float = 0.0, high: float = 0.99) -> float:
    return max(low, min(high, value))


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _passes_geometry(area: int, w: int, h: int, page_area: float, params: Dict[str, float]) -> bool:
    ratio = area / page_area
    if ratio < params["min_area_ratio"] or ratio > params["max_area_ratio"]:
        return False
    if h == 0:
        return False
    aspect = w / float(h)
    return params["min_aspect"] <= aspect <= params["max_aspect"]


def _count_lines(line_crop: "np.ndarray", axis: str, min_frac: float) -> int:
    contours, _ = cv2.findContours(line_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = line_crop.shape[:2]
    count = 0
    for contour in contours:
        _, _, cw, ch = cv2.boundingRect(contour)
        if axis == "h" and cw >= min_frac * w:
            count += 1
        elif axis == "v" and ch >= min_frac * h:
            count += 1
    return count


def _axis_lines(edge_crop: "np.ndarray", w: int, h: int) -> Tuple[int, int]:
    min_len = int(0.4 * min(w, h))
    if min_len < 12:
        return 0, 0
    lines = cv2.HoughLinesP(edge_crop, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=10)
    h_long = v_long = 0
    if lines is not None:
        for line in lines[:300]:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) <= 3 and abs(x2 - x1) >= min_len:
                h_long += 1
            elif abs(x2 - x1) <= 3 and abs(y2 - y1) >= min_len:
                v_long += 1
    return h_long, v_long


def _detect_tables(backend: _Backend, bw: "np.ndarray", W: int, H: int, params: Dict[str, float]) -> List[Dict[str, Any]]:
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, W // 40), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, H // 40)))
    h_lines = backend.morph_open(bw, h_kernel)
    v_lines = backend.morph_open(bw, v_kernel)
    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(W * H)
    regions: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not _passes_geometry(area, w, h, page_area, params):
            continue
        h_count = _count_lines(h_lines[y:y + h, x:x + w], "h", 0.5)
        v_count = _count_lines(v_lines[y:y + h, x:x + w], "v", 0.5)
        if h_count < params["table_min_lines"] or v_count < params["table_min_lines"]:
            continue
        density = float(np.count_nonzero(grid[y:y + h, x:x + w])) / max(1, area)
        confidence = _clamp(0.45 + 0.05 * (h_count + v_count) + 0.4 * density)
        regions.append({"type": "table", "bbox": [x, y, x + w, y + h], "confidence": round(confidence, 3)})
    return regions


def _detect_blocks(
    backend: _Backend,
    gray: "np.ndarray",
    bw: "np.ndarray",
    tables: List[Dict[str, Any]],
    W: int,
    H: int,
    params: Dict[str, float],
) -> List[Dict[str, Any]]:
    edges = backend.canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = float(W * H)
    table_boxes = [r["bbox"] for r in tables]
    regions: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not _passes_geometry(area, w, h, page_area, params):
            continue
        bbox = [x, y, x + w, y + h]
        if any(_iou(bbox, tb) > 0.40 for tb in table_boxes):
            continue

        edge_density = float(np.count_nonzero(edges[y:y + h, x:x + w])) / max(1, area)
        count, _, stats, _ = cv2.connectedComponentsWithStats(bw[y:y + h, x:x + w], connectivity=8)
        comp_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
        if comp_areas.size:
            small_ratio = float(np.count_nonzero(comp_areas < 0.0006 * area)) / comp_areas.size
        else:
            small_ratio = 0.0
        # dense field of tiny components with little large-scale edge => text block
        if small_ratio > 0.85 and comp_areas.size > 120 and edge_density < 0.18:
            continue

        h_long, v_long = _axis_lines(edges[y:y + h, x:x + w], w, h)
        if h_long >= 1 and v_long >= 1 and edge_density <= 0.45:
            confidence = _clamp(0.50 + 0.08 * min(5, h_long + v_long) + 0.2 * min(edge_density, 0.3) / 0.3)
            regions.append({"type": "chart", "bbox": bbox, "confidence": round(confidence, 3)})
        else:
            confidence = _clamp(0.45 + 0.4 * min(area / page_area / 0.30, 1.0) + 0.15 * min(edge_density, 0.3) / 0.3)
            regions.append({"type": "visual", "bbox": bbox, "confidence": round(confidence, 3)})
    return regions


def _detect_kpi_cards(
    bw: "np.ndarray",
    tables: List[Dict[str, Any]],
    W: int,
    H: int,
    params: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Card-like rectangles: a sparse box holding a large headline glyph, not a dense grid."""
    page_area = float(W * H)
    table_boxes = [r["bbox"] for r in tables]
    closed = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, W // 30), max(8, H // 45))),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = area / page_area
        if ratio < 0.02 or ratio > 0.22 or h == 0:
            continue
        if not (0.6 <= w / float(h) <= 4.5):
            continue
        bbox = [x, y, x + w, y + h]
        if any(_iou(bbox, tb) > 0.30 for tb in table_boxes):
            continue
        sub = bw[y:y + h, x:x + w]
        fill = float(np.count_nonzero(sub)) / max(1, area)
        count, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        comps = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
        big = int(np.count_nonzero(comps > 0.01 * area)) if comps.size else 0
        if 0.02 <= fill <= 0.35 and big >= 1 and comps.size < 80:
            confidence = _clamp(0.40 + 0.10 * min(big, 3) + 0.20 * (1.0 - fill))
            regions.append({"type": "kpi_card", "bbox": bbox, "confidence": round(confidence, 3)})
    return regions


def _merge(regions: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
    priority = {"table": 4, "kpi_card": 3, "chart": 2, "visual": 1}
    ordered = sorted(regions, key=lambda r: (priority.get(r["type"], 0), r["confidence"]), reverse=True)
    kept: List[Dict[str, Any]] = []
    for region in ordered:
        if all(_iou(region["bbox"], k["bbox"]) <= iou_thresh for k in kept):
            kept.append(region)
    kept.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return kept


def _pad(bbox: BBox, pad: int, W: int, H: int) -> BBox:
    return [max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(W, bbox[2] + pad), min(H, bbox[3] + pad)]


def _annotate(img: "np.ndarray", regions: List[Dict[str, Any]]) -> "np.ndarray":
    out = img.copy()
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        color = _COLORS.get(region["type"], (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{region['type']} {region['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _resolve_use_gpu(use_gpu: Optional[bool]) -> bool:
    if use_gpu is not None:
        return use_gpu
    if os.getenv("OPENCV_USE_GPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return os.getenv("TORCH_DEVICE", "").strip().lower() == "cuda"


def detect_page_regions(
    image_path: str,
    page_number: Optional[int] = None,
    *,
    use_gpu: Optional[bool] = None,
    debug: bool = False,
    debug_dir: str = "debug",
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Detect table / chart / visual regions on one page image.

    Args:
        image_path: Path to a rendered page raster (PNG/JPG).
        page_number: 1-based page number, echoed in the result.
        use_gpu: Force GPU on/off; ``None`` auto-detects from env / CUDA.
        debug: When True, save an annotated overlay to ``debug_dir``.
        debug_dir: Directory for the annotated image.
        params: Overrides for :data:`DEFAULT_PARAMS`.

    Returns:
        ``{"page", "regions": [{"type", "bbox", "confidence"}], "gpu"}`` and,
        when ``debug``, an ``"annotated_path"`` key.
    """
    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"page image not found: {src}")
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"could not read image: {src}")

    H, W = img.shape[:2]
    settings = dict(DEFAULT_PARAMS)
    if params:
        settings.update(params)

    backend = _Backend(_resolve_use_gpu(use_gpu))
    gray = backend.to_gray(img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    tables = _detect_tables(backend, bw, W, H, settings)
    blocks = _detect_blocks(backend, gray, bw, tables, W, H, settings)
    kpi_cards = _detect_kpi_cards(bw, tables, W, H, settings) if settings.get("detect_kpi", 1) else []
    regions = _merge(tables + blocks + kpi_cards, settings["iou_merge"])

    pad = int(settings["pad"])
    for region in regions:
        region["bbox"] = _pad(region["bbox"], pad, W, H)

    result: Dict[str, Any] = {"page": page_number, "regions": regions, "gpu": backend.gpu}

    if debug:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = out_dir / f"page_{(page_number or 0):03d}_detected.png"
        cv2.imwrite(str(annotated_path), _annotate(img, regions))
        result["annotated_path"] = str(annotated_path)

    logger.info("region detect page=%s regions=%d gpu=%s", page_number, len(regions), backend.gpu)
    return result


detect_regions = detect_page_regions
