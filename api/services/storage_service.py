"""Persistence facade: filesystem layout, status JSON mirror, progress math,
the SQLite registry, and full per-document artifact deletion.

Data layout (under ``API_DATA_DIR``, default ``data/``):
    uploads/<doc_id>/<filename>            uploaded PDF
    processed/<doc_id>/                     pipeline work dir (page_images, extracted_regions, debug, pageindex.json)
    extracted_json/<doc_id>/final_document.json
    pageindex_store/<doc_id>/pageindex.json final PageIndex (used by /ask)
    status/<doc_id>.json                    status mirror (spec-required)
    registry.db                             SQLite source of truth
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.db.local_store import LocalStore
from api.utils.file_utils import now_iso, remove_file_safe, rmtree_safe

DATA_DIR = Path(os.getenv("API_DATA_DIR", "data"))
UPLOADS = DATA_DIR / "uploads"
PROCESSED = DATA_DIR / "processed"
PAGE_IMAGES = DATA_DIR / "page_images"
EXTRACTED_REGIONS = DATA_DIR / "extracted_regions"
EXTRACTED_JSON = DATA_DIR / "extracted_json"
PAGEINDEX_STORE = DATA_DIR / "pageindex_store"
STATUS = DATA_DIR / "status"
LOGS = DATA_DIR / "logs"
REGISTRY_DB = DATA_DIR / "registry.db"

_ALL_DIRS = [UPLOADS, PROCESSED, PAGE_IMAGES, EXTRACTED_REGIONS, EXTRACTED_JSON, PAGEINDEX_STORE, STATUS, LOGS]

# Cumulative progress (%) at the START of each stage; the page fraction fills the band.
STAGE_ORDER = [
    "uploaded", "rendering_pages", "page_classification", "opencv_region_detection",
    "region_cropping", "table_extraction", "chart_extraction", "json_validation",
    "pageindex_building", "completed",
]
STAGE_START = {
    "uploaded": 0.0, "rendering_pages": 5.0, "page_classification": 40.0,
    "opencv_region_detection": 48.0, "region_cropping": 60.0, "table_extraction": 64.0,
    "chart_extraction": 80.0, "json_validation": 92.0, "pageindex_building": 96.0, "completed": 100.0,
}

_store: Optional[LocalStore] = None

_STATUS_FIELDS = (
    "doc_id", "filename", "status", "total_pages", "processed_pages", "current_page",
    "progress_percent", "stage", "error", "errors", "created_at", "updated_at",
)


def init_storage() -> LocalStore:
    """Create all data directories and the SQLite registry (idempotent)."""
    global _store
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    if _store is None:
        _store = LocalStore(str(REGISTRY_DB))
    return _store


def store() -> LocalStore:
    return _store if _store is not None else init_storage()


# --- per-document paths ---------------------------------------------------- #
def upload_dir(doc_id: str) -> Path:
    return UPLOADS / doc_id


def work_dir(doc_id: str) -> Path:
    return PROCESSED / doc_id


def extracted_json_path(doc_id: str) -> Path:
    return EXTRACTED_JSON / doc_id / "final_document.json"


def pageindex_path(doc_id: str) -> Path:
    return PAGEINDEX_STORE / doc_id / "pageindex.json"


def status_path(doc_id: str) -> Path:
    return STATUS / f"{doc_id}.json"


# --- progress math --------------------------------------------------------- #
def compute_progress(stage: str, current_page: int, total_pages: int) -> float:
    if stage == "completed":
        return 100.0
    start = STAGE_START.get(stage, 0.0)
    idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
    end = STAGE_START[STAGE_ORDER[idx + 1]] if idx + 1 < len(STAGE_ORDER) else 100.0
    frac = min(1.0, max(0.0, current_page / total_pages)) if total_pages else 0.0
    return round(min(100.0, max(0.0, start + (end - start) * frac)), 2)


# --- status JSON mirror ---------------------------------------------------- #
def _write_status_file(record: Dict[str, Any]) -> None:
    path = status_path(record["doc_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_status_file(doc_id: str) -> None:
    row = store().get(doc_id)
    if row:
        _write_status_file({key: row.get(key) for key in _STATUS_FIELDS})


# --- registry lifecycle ---------------------------------------------------- #
def register_upload(doc_id: str, filename: str) -> None:
    timestamp = now_iso()
    store().create(doc_id, filename, "uploaded", timestamp, timestamp)
    _sync_status_file(doc_id)


def set_processing(doc_id: str) -> None:
    store().update(doc_id, status="processing", stage="uploaded", updated_at=now_iso())
    _sync_status_file(doc_id)


def update_progress(doc_id: str, stage: str, current_page: int, total_pages: int, message: Optional[str] = None) -> None:
    current_page = int(current_page or 0)
    fields: Dict[str, Any] = {
        "status": "processing",
        "stage": stage,
        "current_page": current_page,
        "progress_percent": compute_progress(stage, current_page, total_pages),
        "processed_pages": min(current_page, total_pages) if total_pages else current_page,
        "updated_at": now_iso(),
    }
    if total_pages and total_pages > 0:
        fields["total_pages"] = int(total_pages)
    store().update(doc_id, **fields)
    _sync_status_file(doc_id)


def complete(doc_id: str, total_pages: int, errors: List[Dict[str, Any]]) -> None:
    total = int(total_pages or 0)
    store().update(
        doc_id, status="completed", stage="completed", total_pages=total,
        processed_pages=total, progress_percent=100.0, errors=errors or [], error=None,
        updated_at=now_iso(),
    )
    _sync_status_file(doc_id)


def fail(doc_id: str, error_message: str) -> None:
    store().update(doc_id, status="failed", error=str(error_message)[:2000], updated_at=now_iso())
    _sync_status_file(doc_id)


# --- final artifacts ------------------------------------------------------- #
def write_final_json(doc_id: str, state: Dict[str, Any]) -> str:
    payload = {
        "doc_id": doc_id,
        "document_name": state.get("document_name"),
        "pdf_path": state.get("pdf_path"),
        "pageindex_path": str(pageindex_path(doc_id)),
        "summary": state.get("final_output"),
        "page_metadata": state.get("page_metadata"),
        "page_classes": state.get("page_classes"),
        "text_nodes": state.get("text_nodes"),
        "table_nodes": state.get("table_nodes"),
        "chart_nodes": state.get("chart_nodes"),
        "validated_nodes": state.get("validated_nodes"),
        "errors": state.get("errors"),
    }
    path = extracted_json_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def copy_pageindex(doc_id: str) -> Optional[str]:
    src = work_dir(doc_id) / "pageindex.json"
    if not src.is_file():
        return None
    dst = pageindex_path(doc_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def delete_all_artifacts(doc_id: str) -> None:
    """Remove the uploaded PDF, page images, regions, JSON, PageIndex and status."""
    rmtree_safe(str(upload_dir(doc_id)))
    rmtree_safe(str(work_dir(doc_id)))          # page_images + extracted_regions + working pageindex + debug
    rmtree_safe(str(EXTRACTED_JSON / doc_id))
    rmtree_safe(str(PAGEINDEX_STORE / doc_id))
    rmtree_safe(str(PAGE_IMAGES / doc_id))      # spec top-level roots (if ever populated)
    rmtree_safe(str(EXTRACTED_REGIONS / doc_id))
    remove_file_safe(str(status_path(doc_id)))


# --- registry reads -------------------------------------------------------- #
def list_documents() -> List[Dict[str, Any]]:
    return store().list_all()


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    return store().get(doc_id)


def delete_document_row(doc_id: str) -> bool:
    return store().delete(doc_id)
