"""Structured logging setup using structlog.

Provides JSON output for production deployments and coloured console output
for local development.  All log entries include a timestamp, log level, and
logger name.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog and the stdlib root logger.

    Parameters
    ----------
    level:
        Root log level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Shared pre-chain: processors applied to *all* log entries before they
    # reach the final renderer.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Detect whether we are writing to an interactive terminal.
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    if is_tty:
        # Development: human-readable coloured console output.
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
        )
    else:
        # Production / CI: machine-readable JSON lines.
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "websockets", "duckdb", "h2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for *name*.

    Typical usage at module level::

        log = get_logger(__name__)
    """
    return structlog.get_logger(name)
