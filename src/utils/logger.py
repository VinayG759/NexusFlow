"""Centralised logging configuration for NexusFlow.

Usage::

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Agent started")
    logger.error("Something went wrong")

Every call to get_logger with the same *name* returns the same underlying
:class:`logging.Logger` instance (Python's logging module caches them), so
handlers are attached only once even when multiple modules import the same
logger name.

Log output destinations:
- Console  — INFO and above
- logs/nexusflow.log — INFO and above (all structured logs)
- logs/errors.log    — ERROR and above (failures only)
"""

import logging
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

_LOG_DIR = Path("logs")
_ALL_LOG = _LOG_DIR / "nexusflow.log"
_ERR_LOG = _LOG_DIR / "errors.log"
_FMT = "[%(asctime)s] [%(levelname)s] [%(name)s] — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Tracks whether the root NexusFlow logger has already been configured so that
# repeated calls to get_logger() never add duplicate handlers.
_configured: bool = False


def _configure_root_logger() -> None:
    """Attach handlers to the 'nexusflow' root logger exactly once."""
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    all_file_handler = logging.FileHandler(_ALL_LOG, encoding="utf-8")
    all_file_handler.setLevel(logging.INFO)
    all_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(_ERR_LOG, encoding="utf-8")
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    root = logging.getLogger("nexusflow")
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(all_file_handler)
    root.addHandler(error_file_handler)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    The logger is a child of the ``nexusflow`` root logger, so all handlers
    (console, nexusflow.log, errors.log) are inherited automatically.

    Args:
        name: Identifier for the logger, typically ``__name__`` of the calling
            module (e.g. ``"src.agents.orchestrator"``).

    Returns:
        A :class:`logging.Logger` instance ready to use.

    Example::

        logger = get_logger(__name__)
        logger.info("Task started: %s", task_id)
        logger.error("Agent '%s' failed: %s", agent_name, exc)
    """
    _configure_root_logger()
    return logging.getLogger(f"nexusflow.{name}")
