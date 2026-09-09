from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger("yolo")
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)


def get_logger() -> logging.Logger:
    """Return the global yolo logger."""
    return _LOGGER


def log_info(message: str, *args: Any) -> None:
    _LOGGER.info(message, *args)


def log_warning(message: str, *args: Any) -> None:
    _LOGGER.warning(message, *args)


def log_error(message: str, *args: Any) -> None:
    _LOGGER.error(message, *args)
