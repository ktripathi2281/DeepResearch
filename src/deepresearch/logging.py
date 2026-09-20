"""Structured (JSON) logging for Milestone 1.

Single dependency-free formatter using the stdlib. Emits one JSON object
per record with timestamp, level, logger, message, plus request context
when bound via the ``extra`` dict (e.g. request_id, stage, duration_ms).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach common structured context if present. Only these explicit
        # keys ever reach the log — prompts, documents, secrets, and
        # headers have no key here by design (see ADR-015).
        for key in (
            "request_id",
            "stage",
            "duration_ms",
            "path",
            "method",
            "status_code",
            "event",
            "status",
            "error_type",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
