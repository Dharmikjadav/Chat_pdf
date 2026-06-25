"""Lightweight, thread-safe progress reporting for the extraction pipeline.

The pipeline stays decoupled from the API: nodes call :func:`report_progress`,
which is a no-op unless a caller (e.g. the FastAPI processing service) has
installed a callback for the current run via :func:`set_progress_callback`.

A ``contextvars.ContextVar`` is used so concurrent documents processed in
separate worker threads each see their own callback.
"""

import contextvars
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["set_progress_callback", "clear_progress_callback", "report_progress", "ProgressCallback"]

# signature: callback(stage, current_page, total_pages, message=None)
ProgressCallback = Callable[[str, int, int, Optional[str]], None]

_callback: "contextvars.ContextVar[Optional[ProgressCallback]]" = contextvars.ContextVar(
    "progress_callback", default=None
)


def set_progress_callback(callback: Optional[ProgressCallback]) -> None:
    """Install the progress callback for the current execution context."""
    _callback.set(callback)


def clear_progress_callback() -> None:
    """Remove any installed progress callback."""
    _callback.set(None)


def report_progress(stage: str, current_page: int = 0, total_pages: int = 0, message: Optional[str] = None) -> None:
    """Report progress to the installed callback (no-op if none is set)."""
    callback = _callback.get()
    if callback is None:
        return
    try:
        callback(stage, int(current_page or 0), int(total_pages or 0), message)
    except Exception:  # noqa: BLE001 - never let reporting break the pipeline
        logger.exception("progress callback raised")
