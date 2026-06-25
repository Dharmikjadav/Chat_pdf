"""Background PDF processing via a ThreadPoolExecutor.

Each upload is processed off the request thread so /post_doc returns immediately.
The existing pipeline is called unchanged via run_pipeline(); progress is relayed
through the progress_callback into the storage layer after each page/stage.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from api.services import storage_service as storage

logger = logging.getLogger(__name__)

_MAX_WORKERS = int(os.getenv("API_MAX_WORKERS", "1"))  # 1 = serialize (single GPU)
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="pdf-proc")


def submit(doc_id: str, pdf_path: str) -> None:
    """Queue a document for background processing (returns immediately)."""
    _executor.submit(_process, doc_id, pdf_path)


def _process(doc_id: str, pdf_path: str) -> None:
    try:
        storage.set_processing(doc_id)

        def progress_callback(stage: str, current_page: int, total_pages: int, message: Optional[str] = None) -> None:
            try:
                storage.update_progress(doc_id, stage, current_page, total_pages, message)
            except Exception:  # noqa: BLE001 - never let progress writes kill the job
                logger.exception("progress update failed for %s", doc_id)

        from graph.pipeline import run_pipeline  # lazy: heavy pipeline deps

        state = run_pipeline(
            pdf_path,
            output_dir=str(storage.work_dir(doc_id)),
            progress_callback=progress_callback,
        )

        storage.write_final_json(doc_id, state)
        storage.copy_pageindex(doc_id)

        final = state.get("final_output") or {}
        total = int(final.get("num_pages") or len(state.get("page_metadata") or []))
        errors = state.get("errors") or []
        storage.complete(doc_id, total, errors)
        logger.info("processing complete for %s (%d pages, %d errors)", doc_id, total, len(errors))
    except Exception as exc:  # noqa: BLE001 - record failure, do not crash the worker
        logger.exception("processing failed for %s", doc_id)
        try:
            storage.fail(doc_id, str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("could not record failure for %s", doc_id)


def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)
