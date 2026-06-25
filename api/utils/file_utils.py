"""Filesystem helpers: UUIDs, PDF validation, streamed upload save, safe delete."""

import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

_CHUNK = 1024 * 1024  # 1 MiB


def new_doc_id() -> str:
    """Generate a fresh document id."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    """Strip path components and unsafe characters from an uploaded filename."""
    base = Path(str(filename or "document.pdf")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "document.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned


def is_pdf(filename: str, head: bytes) -> bool:
    """Validate that an upload is a PDF (extension + %PDF magic in the header)."""
    name_ok = str(filename or "").lower().endswith(".pdf")
    magic_ok = b"%PDF-" in (head or b"")[:1024]
    return name_ok and magic_ok


def save_upload(fileobj: BinaryIO, dest_path: str) -> int:
    """Stream an uploaded file object to ``dest_path``; returns bytes written."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        fileobj.seek(0)
    except (OSError, AttributeError):
        pass
    with open(dest, "wb") as out:
        while True:
            chunk = fileobj.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    return total


def rmtree_safe(path: str) -> None:
    """Remove a directory tree if it exists (ignore errors)."""
    target = Path(path)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def remove_file_safe(path: str) -> None:
    """Remove a file if it exists (ignore errors)."""
    target = Path(path)
    try:
        if target.is_file():
            target.unlink()
    except OSError:
        pass
