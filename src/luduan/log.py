"""Centralised logging for Luduan.

Log file: ~/.config/luduan/luduan.log  (rotated at 2 MB, 3 backups kept)

Usage::

    from luduan.log import get_logger
    log = get_logger(__name__)
    log.info("hello")
    log.exception("something broke")   # includes full traceback
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path.home() / ".config" / "luduan"
LOG_PATH = _LOG_DIR / "luduan.log"

_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("luduan")
    root.setLevel(logging.DEBUG)

    # Rotating file handler — 2 MB per file, keep 3 backups
    fh = logging.handlers.RotatingFileHandler(
        LOG_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    # Keep name under the "luduan" hierarchy so the root handler captures it
    if not name.startswith("luduan"):
        name = f"luduan.{name}"
    return logging.getLogger(name)
