"""Document-level operations: registry <-> Pydantic mapping, listing, deletion."""

import logging
from typing import List, Optional

from api.schemas import Document, ProcessingError, ProgressResponse
from api.services import ask_service
from api.services import storage_service as storage

logger = logging.getLogger(__name__)


def _to_document(row: dict) -> Document:
    return Document(
        doc_id=row["doc_id"],
        filename=row["filename"],
        status=row["status"],
        total_pages=row.get("total_pages") or 0,
        processed_pages=row.get("processed_pages") or 0,
        progress_percent=row.get("progress_percent") or 0.0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row.get("error"),
    )


def list_documents() -> List[Document]:
    return [_to_document(row) for row in storage.list_documents()]


def get_document(doc_id: str) -> Optional[Document]:
    row = storage.get_document(doc_id)
    return _to_document(row) if row else None


def get_progress(doc_id: str) -> Optional[ProgressResponse]:
    row = storage.get_document(doc_id)
    if not row:
        return None

    total_pages = int(row.get("total_pages") or 0)
    processed_pages = int(row.get("processed_pages") or 0)

    if total_pages > 0:
        progress_percent = round((processed_pages / total_pages) * 100, 2)
    else:
        progress_percent = float(row.get("progress_percent") or 0.0)

    if row.get("status") == "completed":
        progress_percent = 100.0
        processed_pages = total_pages

    errors = []
    for e in row.get("errors") or []:
        if isinstance(e, dict):
            errors.append(
                ProcessingError(
                    stage=e.get("stage") or "unknown",
                    message=str(e.get("message", "")),
                    page_number=e.get("page_number"),
                )
            )

    return ProgressResponse(
        doc_id=row["doc_id"],
        status=row.get("status", "uploaded"),
        total_pages=total_pages,
        processed_pages=processed_pages,
        current_page=int(row.get("current_page") or processed_pages or 0),
        progress_percent=progress_percent,
        stage=row.get("stage") or row.get("status") or "uploaded",
        errors=errors,
    )


def delete_document(doc_id: str) -> bool:
    row = storage.get_document(doc_id)
    if not row:
        return False
    ask_service.invalidate(doc_id)
    storage.delete_all_artifacts(doc_id)
    storage.delete_document_row(doc_id)
    logger.info("deleted document %s and all artifacts", doc_id)
    return True
