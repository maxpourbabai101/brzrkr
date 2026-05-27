"""Centralized logging configuration.

Use :func:`configure_logging` once at process start (run.py /
backtest_runner.py / tests). All other modules just call
``logging.getLogger(__name__)`` and inherit the root config.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

DEFAULT_LOG_PATH = Path("trading_enhancer.log")
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(
    level: str | int = "INFO",
    log_path: Path | str = DEFAULT_LOG_PATH,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backups: int = 5,
) -> logging.Logger:
    """Attach rotating file + console handlers to the root logger."""
    log_path = Path(os.getenv("TRADING_ENHANCER_LOG", log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Clear any pre‑existing handlers so re‑invocation is idempotent.
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)
    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backups
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root.addHandler(console_handler)

    root.debug("Logging initialised at level %s — file=%s", level, log_path)
    return root
