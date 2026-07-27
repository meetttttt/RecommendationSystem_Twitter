"""Application logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from floter_ai.config import LOG_DIR, LOG_PATH


LOGGER_NAME = "floter_ai"


def configure_logging() -> logging.Logger:
    """Configure API logging to stdout and logs/api.log."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_PATH)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Write one structured JSON log event."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))
