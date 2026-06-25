"""Ollama vision calls for complex tables, charts, and region classification.

Each public function returns one of three things:
  * a parsed JSON object (dict/list) when the model returns valid JSON,
  * the raw response string when the output is not valid JSON,
  * an error dict ``{"error", "error_type", "image_path", "model"}`` when the
    image is unreadable or Ollama cannot be reached (never raises).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from prompts.vision_prompts import (
    CHART_PROMPT,
    COMPLEX_TABLE_PROMPT,
    KPI_PROMPT,
    PAGE_OCR_PROMPT,
    REGION_CLASSIFIER_PROMPT,
)

logger = logging.getLogger(__name__)

__all__ = ["extract_complex_table", "extract_chart", "classify_region", "extract_kpi", "extract_text"]

DEFAULT_MODEL = "qwen2.5vl:7b"

VisionResult = Union[Dict[str, Any], list, str]

_CLIENT = None
_CLIENT_HOST: Optional[str] = None


def _host() -> str:
    host = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    # 0.0.0.0 is a server bind address, not a connectable client target
    return host.replace("0.0.0.0", "127.0.0.1")


def _timeout() -> float:
    return float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))


def _temperature() -> float:
    return float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))


def _num_ctx() -> int:
    return int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _options() -> Dict[str, Any]:
    options: Dict[str, Any] = {"temperature": _temperature(), "num_ctx": _num_ctx()}
    raw = os.getenv("OLLAMA_NUM_GPU")
    if raw not in (None, ""):
        try:
            options["num_gpu"] = int(raw)
        except ValueError:
            pass
    return options


def _import_ollama():
    import ollama  # noqa: PLC0415 - lazy so the module imports without the dep

    return ollama


def _get_client(ollama_mod):
    global _CLIENT, _CLIENT_HOST
    host = _host()
    if _CLIENT is None or _CLIENT_HOST != host:
        _CLIENT = ollama_mod.Client(host=host, timeout=_timeout())
        _CLIENT_HOST = host
    return _CLIENT


def _content(response: Any) -> str:
    try:
        return response["message"]["content"]
    except (TypeError, KeyError, IndexError):
        pass
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content
    return str(response)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_block(text: str) -> Optional[str]:
    start = None
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start is None:
        return None

    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _json_candidates(text: str) -> Iterator[str]:
    yield text
    fenced = _strip_fences(text)
    if fenced != text:
        yield fenced
    for source in (fenced, text):
        block = _extract_json_block(source)
        if block:
            yield block


def _parse_json(raw: Optional[str]) -> Optional[Any]:
    if not raw or not raw.strip():
        return None
    seen = set()
    for candidate in _json_candidates(raw):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _error(message: str, error_type: str, image_path: str, model: str) -> Dict[str, Any]:
    return {
        "error": message,
        "error_type": error_type,
        "image_path": image_path,
        "model": model,
    }


def _run_vision(image_path: str, prompt: str, model: str, *, json_format: bool = True) -> VisionResult:
    model = (model or "").strip()
    path = Path(image_path)

    if not path.is_file():
        logger.error("vision: image not found: %s", path)
        return _error(f"image not found: {path}", "FileNotFoundError", str(path), model)

    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        logger.exception("vision: could not read image %s", path)
        return _error(f"could not read image: {exc}", type(exc).__name__, str(path), model)

    try:
        ollama_mod = _import_ollama()
        client = _get_client(ollama_mod)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
            format="json" if json_format else "",
            options=_options(),
        )
        raw = _content(response)
    except ImportError:
        logger.error("vision: ollama package not installed")
        return _error(
            "ollama package not installed; run 'pip install ollama'",
            "ImportError",
            str(path),
            model,
        )
    except Exception as exc:  # noqa: BLE001 - report any API/connection failure
        name = type(exc).__name__
        if isinstance(exc, (ConnectionError, TimeoutError)) or any(
            token in name for token in ("Connect", "Timeout", "Transport")
        ):
            message = f"could not reach Ollama at {_host()}: {exc}"
        else:
            message = f"ollama request failed: {exc}"
        logger.exception("vision: Ollama call failed for %s", path)
        return _error(message, name, str(path), model)

    if not json_format:
        logger.info("vision: OCR text from %s (model=%s)", path, model)
        return (raw or "").strip()

    parsed = _parse_json(raw)
    if parsed is not None:
        logger.info("vision: parsed JSON from %s (model=%s)", path, model)
        return parsed

    logger.warning("vision: non-JSON response for %s (model=%s)", path, model)
    return raw


def extract_complex_table(image_path: str, model: str = DEFAULT_MODEL) -> VisionResult:
    """Extract a complex/broken table image into JSON via the Ollama vision model."""
    return _run_vision(image_path, COMPLEX_TABLE_PROMPT, model)


def extract_chart(image_path: str, model: str = DEFAULT_MODEL) -> VisionResult:
    """Extract a chart image into structured JSON via the Ollama vision model."""
    return _run_vision(image_path, CHART_PROMPT, model)


def classify_region(image_path: str, model: str = DEFAULT_MODEL) -> VisionResult:
    """Classify an image region (table/chart/figure/text/...) into JSON."""
    return _run_vision(image_path, REGION_CLASSIFIER_PROMPT, model)


def extract_kpi(image_path: str, model: str = DEFAULT_MODEL) -> VisionResult:
    """Extract KPI cards from a crop/page image into structured JSON."""
    return _run_vision(image_path, KPI_PROMPT, model)


def extract_text(image_path: str, model: str = DEFAULT_MODEL) -> VisionResult:
    """OCR a (scanned) page image into plain text via the Ollama vision model.

    Returns the transcribed text (str) on success, or an error dict on failure.
    """
    return _run_vision(image_path, PAGE_OCR_PROMPT, model, json_format=False)
