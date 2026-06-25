"""Stage 2 node: convert a PDF to Markdown via the Marker CLI (offline)."""

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph.state import MarkerImage, PDFPipelineState, PipelineError

logger = logging.getLogger(__name__)

__all__ = ["marker_node", "MARKER_OUTPUT_SUBDIR"]

MARKER_OUTPUT_SUBDIR = "marker_output"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_PAGE_RE = re.compile(r"page[_\-\s]?(\d+)", re.IGNORECASE)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a truthy/falsy environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device(requested: str) -> str:
    """Normalize the device and fall back to CPU if CUDA is unavailable.

    Prevents Marker from crashing with 'Torch not compiled with CUDA enabled'
    when TORCH_DEVICE=cuda but the installed torch is the CPU-only build.
    """
    device = (requested or "cpu").strip().lower()
    if device in ("cuda", "gpu"):
        try:
            import torch  # noqa: PLC0415 - only needed to probe CUDA
        except Exception as exc:  # noqa: BLE001 - torch missing: let Marker decide
            logger.warning("could not import torch to verify CUDA (%s); using '%s'", exc, device)
            return device
        if not torch.cuda.is_available():
            logger.warning(
                "TORCH_DEVICE=%s requested but CUDA is unavailable "
                "(torch is CPU-only or no GPU present); falling back to cpu",
                requested,
            )
            return "cpu"
        return "cuda"
    return device


def _marker_error(message: str, exc: Optional[BaseException] = None) -> PipelineError:
    """Build a PipelineError tagged to the marker stage."""
    err: PipelineError = {"stage": "marker", "message": message}
    if exc is not None:
        err["exception_type"] = type(exc).__name__
    return err


def _resolve_marker_cmd() -> Optional[str]:
    """Locate the marker_single executable (env override, PATH, or venv Scripts)."""
    override = os.getenv("MARKER_CMD") or os.getenv("MARKER_SINGLE_CMD")
    if override:
        return shutil.which(override) or (override if Path(override).is_file() else None)
    found = shutil.which("marker_single")
    if found:
        return found
    # Fall back to the current interpreter's Scripts/bin dir (venv not activated):
    # console scripts live next to python.exe at <venv>/Scripts/marker_single.exe.
    script_dir = Path(sys.executable).parent
    for name in ("marker_single.exe", "marker_single"):
        candidate = script_dir / name
        if candidate.is_file():
            return str(candidate)
    return None


def _page_from_name(name: str) -> Optional[int]:
    """Best-effort parse of a page number out of a Marker asset filename."""
    match = _PAGE_RE.search(name)
    return int(match.group(1)) if match else None


def _locate_doc_dir(output_dir: Path, stem: str) -> Optional[Path]:
    """Find the per-document folder Marker writes, falling back to a scan."""
    preferred = output_dir / stem
    if preferred.is_dir():
        return preferred
    markdowns = sorted(output_dir.rglob("*.md"), key=lambda p: len(str(p)))
    if markdowns:
        return markdowns[0].parent
    return None


def _locate_markdown(doc_dir: Path, stem: str) -> Optional[Path]:
    """Locate the generated Markdown file inside the document folder."""
    expected = doc_dir / f"{stem}.md"
    if expected.is_file():
        return expected
    candidates = sorted(doc_dir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def _locate_json(doc_dir: Path, stem: str) -> Optional[Path]:
    """Locate a JSON sidecar if Marker produced one (metadata or json format)."""
    for name in (f"{stem}.json", f"{stem}_meta.json"):
        candidate = doc_dir / name
        if candidate.is_file():
            return candidate
    others = sorted(doc_dir.rglob("*.json"), key=lambda p: len(str(p)))
    return others[0] if others else None


def _locate_images(doc_dir: Path, source: str) -> List[MarkerImage]:
    """Collect every extracted image asset as a MarkerImage record."""
    images: List[MarkerImage] = []
    for path in sorted(doc_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            images.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "page_number": _page_from_name(path.name),
                }
            )
    return images


def marker_node(state: PDFPipelineState) -> Dict[str, Any]:
    """Run marker_single on the input PDF and return located output paths."""
    errors: List[PipelineError] = []
    empty_update: Dict[str, Any] = {
        "marker_md_path": None,
        "marker_json_path": None,
        "marker_images": [],
        "errors": errors,
    }

    pdf_path = Path(state["pdf_path"])
    stem = pdf_path.stem
    source = state.get("document_name") or stem
    output_dir = Path(state["output_dir"]) / MARKER_OUTPUT_SUBDIR

    marker_cmd = _resolve_marker_cmd()
    if not marker_cmd:
        logger.error(
            "marker_single not found on PATH. Install marker-pdf or set MARKER_CMD."
        )
        errors.append(
            _marker_error(
                "marker_single command not found; install 'marker-pdf' or set MARKER_CMD"
            )
        )
        return empty_update

    if not pdf_path.is_file():
        logger.error("Input PDF does not exist: %s", pdf_path)
        errors.append(_marker_error(f"input PDF not found: {pdf_path}"))
        return empty_update

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.exception("Could not create marker output dir: %s", output_dir)
        errors.append(_marker_error(f"could not create output dir '{output_dir}': {exc}", exc))
        return empty_update

    output_format = os.getenv("MARKER_OUTPUT_FORMAT", "markdown")
    cmd: List[str] = [
        marker_cmd,
        str(pdf_path),
        "--output_dir",
        str(output_dir),
        "--output_format",
        output_format,
    ]
    if _env_bool("MARKER_FORCE_OCR"):
        cmd.append("--force_ocr")
    extra_args = os.getenv("MARKER_EXTRA_ARGS", "").split()
    cmd.extend(extra_args)

    timeout = int(os.getenv("MARKER_TIMEOUT", "1800"))
    device = _resolve_device(os.getenv("TORCH_DEVICE", "cuda"))
    run_env = {**os.environ, "TORCH_DEVICE": device}
    logger.info("Running marker_single (TORCH_DEVICE=%s): %s", device, " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=run_env,
        )
    except FileNotFoundError as exc:
        logger.exception("marker_single vanished before execution: %s", marker_cmd)
        errors.append(_marker_error(f"marker command not found at '{marker_cmd}'", exc))
        return empty_update
    except subprocess.TimeoutExpired as exc:
        logger.error("marker_single timed out after %ss on %s", timeout, pdf_path)
        errors.append(_marker_error(f"marker_single timed out after {timeout}s", exc))
        return empty_update
    except Exception as exc:
        logger.exception("Unexpected failure launching marker_single")
        errors.append(_marker_error(f"failed to launch marker_single: {exc}", exc))
        return empty_update

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        logger.error(
            "marker_single exited with code %s\nstderr:\n%s",
            proc.returncode,
            stderr_tail,
        )
        errors.append(
            _marker_error(
                f"marker_single failed (exit {proc.returncode}): {stderr_tail[-500:] or 'no stderr'}"
            )
        )
        return empty_update

    if not output_dir.is_dir():
        logger.error("marker output folder missing after run: %s", output_dir)
        errors.append(_marker_error(f"marker output folder not created: {output_dir}"))
        return empty_update

    doc_dir = _locate_doc_dir(output_dir, stem)
    if doc_dir is None:
        logger.error("No marker output document folder found under %s", output_dir)
        errors.append(_marker_error(f"no marker output found under: {output_dir}"))
        return empty_update

    md_path = _locate_markdown(doc_dir, stem)
    if md_path is None:
        logger.error("Marker produced no markdown file in %s", doc_dir)
        errors.append(_marker_error(f"markdown output not found in: {doc_dir}"))
        return empty_update

    json_path = _locate_json(doc_dir, stem)
    images = _locate_images(doc_dir, source)

    if json_path is None:
        logger.warning("No JSON sidecar found in %s (continuing)", doc_dir)
    if not images:
        logger.info("No extracted images found in %s", doc_dir)

    logger.info(
        "Marker stage done: md=%s json=%s images=%d",
        md_path,
        json_path,
        len(images),
    )

    return {
        "marker_md_path": str(md_path),
        "marker_json_path": str(json_path) if json_path else None,
        "marker_images": images,
        "errors": errors,
    }
