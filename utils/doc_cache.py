"""Per-document crop-extraction cache so re-runs skip already-extracted crops.

Keyed by (kind, page, bbox); stores the normalized extraction dict. A vision call is the
expensive step, so reusing it across re-runs of the same document is the main saving.
Disable with ``USE_CACHE=0``.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["use_cache", "load_cache", "save_cache", "crop_key"]


def use_cache() -> bool:
    return os.getenv("USE_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _path(cache_dir: str) -> Path:
    return Path(cache_dir) / "extract_cache.json"


def load_cache(cache_dir: Optional[str]) -> Dict[str, Any]:
    if not cache_dir or not use_cache():
        return {}
    path = _path(cache_dir)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt cache is ignored, not fatal
            logger.warning("ignoring unreadable cache at %s", path)
    return {}


def save_cache(cache_dir: Optional[str], cache: Dict[str, Any]) -> None:
    if not cache_dir or not use_cache():
        return
    try:
        path = _path(cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:  # noqa: BLE001 - caching is best-effort
        logger.exception("could not write cache to %s", cache_dir)


def crop_key(page_number: Any, bbox: Any, kind: str) -> str:
    raw = f"{kind}|{page_number}|{bbox}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
