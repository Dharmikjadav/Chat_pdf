"""Project logging: writes to logs/pipeline.log (rotated) and the console."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from config import config as _config

    _LOGS_DIR = _config.LOGS_DIR
    _LEVEL_NAME = _config.LOG_LEVEL
except Exception:
    _LOGS_DIR = os.getenv("LOGS_DIR", "./logs")
    _LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()

LOG_FILE = Path(_LOGS_DIR) / "pipeline.log"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def _level(name: Optional[str]) -> int:
    return getattr(logging, str(name).upper(), logging.INFO)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger with a rotating file handler (and console)."""
    global _configured, LOG_FILE
    if _configured and not force:
        return logging.getLogger()

    target = Path(log_file) if log_file else LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE = target

    root = logging.getLogger()
    root.setLevel(_level(level or _LEVEL_NAME))

    for handler in list(root.handlers):
        if getattr(handler, "_pipeline_handler", False):
            root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    file_handler = RotatingFileHandler(target, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._pipeline_handler = True
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._pipeline_handler = True
        root.addHandler(console_handler)

    _configured = True
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger, ensuring logging has been configured."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
