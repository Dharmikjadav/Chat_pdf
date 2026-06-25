"""Crop detected regions out of a page image and save them per page.

Layout written under ``output_dir``::

    extracted_regions/
    |- page_001/
    |   |- table_1.png
    |   |- chart_1.png
    |   `- visual_1.png
    `- page_002/
        `- table_1.png
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["crop_regions"]


def _slug(name: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "doc")).strip("-")
    return cleaned or "doc"


def crop_regions(
    page_image: str,
    detected_regions: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_dir: str,
    *,
    document_name: Optional[str] = None,
    page_number: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Crop each detected region from ``page_image`` and save it.

    Args:
        page_image: Path to the rendered page raster.
        detected_regions: Either a detector result dict ``{"page", "regions"}``
            or a plain list of region dicts.
        output_dir: Root directory; a ``page_<NNN>/`` sub-folder is created.
        document_name / page_number / source: Provenance for the crop records.

    Returns:
        A list of crop records: ``{id, type, page_number, bbox, confidence,
        image_path, document_name, source}``.
    """
    import cv2  # noqa: PLC0415 - lazy so the module imports without OpenCV

    if isinstance(detected_regions, dict):
        regions = detected_regions.get("regions") or []
        page = detected_regions.get("page", page_number)
    else:
        regions = list(detected_regions or [])
        page = page_number

    if not regions:
        return []

    img = cv2.imread(str(page_image))
    if img is None:
        raise ValueError(f"could not read page image: {page_image}")
    height, width = img.shape[:2]

    page_dir = Path(output_dir) / f"page_{(page or 0):03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    counters: Dict[str, int] = {}
    crops: List[Dict[str, Any]] = []
    for region in regions:
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        region_type = region.get("type", "visual")
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue

        counters[region_type] = counters.get(region_type, 0) + 1
        index = counters[region_type]
        out_path = page_dir / f"{region_type}_{index}.png"
        cv2.imwrite(str(out_path), img[y1:y2, x1:x2])

        crops.append(
            {
                "id": f"{_slug(document_name)}-p{(page or 0):03d}-{region_type}-{index}",
                "type": region_type,
                "page_number": page,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(region.get("confidence", 0.0)),
                "image_path": str(out_path),
                "document_name": document_name,
                "source": source,
            }
        )

    logger.info("cropped %d region(s) from page %s into %s", len(crops), page, page_dir)
    return crops
