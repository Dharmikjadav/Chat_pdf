"""Surya OCR for scanned pages/crops — lazy, GPU-serialized, never fatal.

Surya and Ollama share one GPU, so predictor construction is lazy (only on the first scanned
page) and every call holds a process-wide lock. Any failure returns '' so callers fall back to
the Qwen2.5-VL OCR path. Disable with ``USE_SURYA_OCR=0``; force CPU with ``SURYA_DEVICE=cpu``.
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["ocr_image", "surya_available"]

_LOCK = threading.Lock()
_PREDICTORS = None
_LOAD_FAILED = False


def surya_available() -> bool:
    if os.getenv("USE_SURYA_OCR", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    try:
        import surya  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _device() -> Optional[str]:
    return os.getenv("SURYA_DEVICE") or os.getenv("TORCH_DEVICE") or None


def _load():
    global _PREDICTORS, _LOAD_FAILED
    if _PREDICTORS is not None or _LOAD_FAILED:
        return _PREDICTORS
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        device = _device()
        foundation = FoundationPredictor(device=device) if device else FoundationPredictor()
        _PREDICTORS = (RecognitionPredictor(foundation), DetectionPredictor())
        logger.info("Surya OCR predictors loaded (device=%s)", device or "default")
    except Exception:  # noqa: BLE001 - model load can fail (VRAM, version drift)
        logger.exception("Surya OCR load failed; falling back to vision OCR")
        _LOAD_FAILED = True
        _PREDICTORS = None
    return _PREDICTORS


def ocr_image(image_path: str) -> str:
    """Return OCR text for a scanned page/crop, or '' on any failure (caller falls back)."""
    if not surya_available():
        return ""
    try:
        from PIL import Image

        with _LOCK:
            predictors = _load()
            if not predictors:
                return ""
            recognition, detection = predictors
            image = Image.open(image_path).convert("RGB")
            try:
                from surya.common.surya.schema import TaskNames

                results = recognition([image], task_names=[TaskNames.ocr_with_boxes], det_predictor=detection)
            except Exception:  # noqa: BLE001 - older/newer signature without task_names
                results = recognition([image], det_predictor=detection)
        if not results:
            return ""
        lines = getattr(results[0], "text_lines", None) or []
        return "\n".join((getattr(ln, "text", "") or "") for ln in lines).strip()
    except Exception:  # noqa: BLE001 - any OCR failure -> empty -> vision fallback
        logger.exception("Surya OCR failed for %s", image_path)
        return ""
