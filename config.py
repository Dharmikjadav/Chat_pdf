"""Central configuration for the offline PDF chatbot, loaded from env / .env."""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_dotenv()


def _str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _int(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                pass
    return int(default)


def _float(*names: str, default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return float(default)


def _bool(*names: str, default: bool) -> bool:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration (immutable snapshot of the environment)."""

    MARKER_OUTPUT_DIR: str
    PAGE_IMAGE_DIR: str
    PAGEINDEX_DIR: str
    OLLAMA_VISION_MODEL: str
    OLLAMA_CHAT_MODEL: str
    RENDER_DPI: int
    MAX_PAGES: int
    OLLAMA_BASE_URL: str
    TORCH_DEVICE: str
    OLLAMA_NUM_GPU: int
    OPENCV_USE_GPU: bool
    OPENCV_DEBUG: bool
    DEBUG_REGIONS: bool
    TEXT_PAGE_CHAR_THRESHOLD: int
    USE_MARKER: bool
    VISION_CROP_DPI: int
    VISION_VERIFY: bool
    USE_PDFPLUMBER: bool
    USE_SURYA_OCR: bool
    USE_CACHE: bool
    USE_FACTS: bool
    USE_KPI: bool
    RENDER_ALL: bool
    SURYA_DEVICE: str
    KPI_NUM_RATIO: float
    LOGS_DIR: str
    LOG_LEVEL: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            MARKER_OUTPUT_DIR=_str("MARKER_OUTPUT_DIR", default="./data/marker_output"),
            PAGE_IMAGE_DIR=_str("PAGE_IMAGE_DIR", "IMAGES_DIR", default="./data/page_images"),
            PAGEINDEX_DIR=_str("PAGEINDEX_DIR", default="./data/pageindex"),
            OLLAMA_VISION_MODEL=_str("OLLAMA_VISION_MODEL", default="qwen2.5vl:7b").strip(),
            OLLAMA_CHAT_MODEL=_str("OLLAMA_CHAT_MODEL", default="qwen3:8b").strip(),
            RENDER_DPI=_int("RENDER_DPI", default=200),
            MAX_PAGES=_int("MAX_PAGES", "MARKER_MAX_PAGES", default=0),
            OLLAMA_BASE_URL=_str("OLLAMA_BASE_URL", "OLLAMA_HOST", default="http://localhost:11434"),
            TORCH_DEVICE=_str("TORCH_DEVICE", default="cuda").strip(),
            OLLAMA_NUM_GPU=_int("OLLAMA_NUM_GPU", default=999),
            OPENCV_USE_GPU=_bool("OPENCV_USE_GPU", default=False),
            OPENCV_DEBUG=_bool("OPENCV_DEBUG", default=False),
            DEBUG_REGIONS=_bool("DEBUG_REGIONS", "OPENCV_DEBUG", default=False),
            TEXT_PAGE_CHAR_THRESHOLD=_int("TEXT_PAGE_CHAR_THRESHOLD", default=1600),
            USE_MARKER=_bool("USE_MARKER", default=False),
            VISION_CROP_DPI=_int("VISION_CROP_DPI", default=300),
            VISION_VERIFY=_bool("VISION_VERIFY", default=False),
            USE_PDFPLUMBER=_bool("USE_PDFPLUMBER", default=True),
            USE_SURYA_OCR=_bool("USE_SURYA_OCR", default=True),
            USE_CACHE=_bool("USE_CACHE", default=True),
            USE_FACTS=_bool("USE_FACTS", default=True),
            USE_KPI=_bool("USE_KPI", default=True),
            RENDER_ALL=_bool("RENDER_ALL", default=False),
            SURYA_DEVICE=_str("SURYA_DEVICE", "TORCH_DEVICE", default="cuda").strip(),
            KPI_NUM_RATIO=_float("KPI_NUM_RATIO", default=0.30),
            LOGS_DIR=_str("LOGS_DIR", default="./logs"),
            LOG_LEVEL=_str("LOG_LEVEL", default="INFO").upper(),
        )

    def ensure_dirs(self) -> None:
        """Create the output directories if they do not exist."""
        for path in (self.MARKER_OUTPUT_DIR, self.PAGE_IMAGE_DIR, self.PAGEINDEX_DIR, self.LOGS_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)


config = Config.from_env()


def get_config() -> Config:
    """Return the current configuration snapshot."""
    return config


def reload() -> Config:
    """Reload configuration from the environment (re-reads .env)."""
    global config
    _load_dotenv()
    config = Config.from_env()
    return config
