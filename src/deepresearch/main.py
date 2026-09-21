"""FastAPI application — Milestones 1 + 15 + 18 + 20.

Endpoints:
- GET /health — liveness: "the process is alive". No DB, no provider,
  never fails when dependencies are down.
- GET /ready  — readiness: "the application can accept research work".
  Requires PostgreSQL reachability AND a valid provider selection
  (name known, required credential present). Never probes the model
  daemon itself: Ollama/a cloud endpoint may be down at boot and the
  app still starts; jobs then fail cleanly with safe errors.
- POST /api/research — start a research job (202 + pollable snapshot).
- GET /api/research/{request_id} — poll a research job.

M15 request tracing: every request gets an ``X-Request-ID`` (preserved
when the client supplies a usable one, generated otherwise), an
isolated observability context, and a completion log entry.
Health/readiness stay quiet (header only, no access log).

M18 CORS: the browser frontend runs on a different origin
(``CORS_ORIGINS``, default ``http://localhost:3000``); cookies are
never used, so credentials stay disabled.

M20 runtime: lifespan validates configuration at startup (fail fast
with a secret-free message), and on shutdown marks running jobs
failed-interrupted, closes provider HTTP clients, and disposes the
database engine.

No ingestion, retrieval, generation, or frontend logic here — the
research router in ``research_api`` is the only boundary.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deepresearch.config import ConfigurationError, get_settings
from deepresearch.db import check_connection, get_engine
from deepresearch.logging import configure_logging, get_logger
from deepresearch.observability import (
    REQUEST_ID_HEADER,
    get_current_trace,
    normalize_request_id,
    traced_request,
)
from deepresearch.research_api import close_default_resources
from deepresearch.research_api import router as research_router

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate early, shut down gracefully (see ADR-020)."""
    from deepresearch.providers import create_llm_provider

    try:
        settings.validate_for_runtime()
    except ConfigurationError as exc:
        logger.error("invalid configuration at startup: %s", exc)
        raise RuntimeError(f"invalid configuration: {exc}") from exc
    # Report exactly what the factory selects (single source of truth);
    # construction is lazy (no network), so this stays a cheap probe.
    # main.py never reads provider-specific settings attributes itself.
    selected = create_llm_provider(settings)
    try:
        provider_name, model_name = type(selected).__name__, selected.model_name
    finally:
        close = getattr(selected, "close", None)
        if callable(close):
            close()
    logger.info(
        "startup complete",
        extra={
            "service": settings.app_name,
            "env": settings.app_env,
            "provider": provider_name,
            "llm_model": model_name,
            "api_port": settings.api_port,
            "event": "startup",
            "status": "ok",
        },
    )
    yield
    interrupted = close_default_resources()
    logger.info(
        "shutdown complete",
        extra={"interrupted_jobs": interrupted, "event": "shutdown", "status": "ok"},
    )


app = FastAPI(title="DeepResearch API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(research_router)

QUIET_PATHS = frozenset({"/health", "/ready"})


@app.middleware("http")
async def trace_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
    start = time.perf_counter()
    with traced_request(request_id):
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        trace = get_current_trace()
        if trace is not None:
            trace.finish("ok" if response.status_code < 500 else "error")
        if request.url.path not in QUIET_PATHS:
            logger.info(
                "%s %s -> %s",
                request.method,
                request.url.path,
                response.status_code,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "event": "request_finished",
                    "status": "ok" if response.status_code < 500 else "error",
                },
            )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(Exception)
async def safe_unexpected_errors(request: Request, exc: Exception) -> JSONResponse:
    """User-safe 500: log detail server-side, never leak stack traces.

    The header is stamped here (not only in the tracing middleware)
    because unhandled exceptions propagate through ``call_next`` after
    the safe response is generated — without this, the 500 body would
    carry the request ID but the response headers would not.
    """
    request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error.",
                "type": type(exc).__name__,
                "request_id": request_id,
            }
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "env": settings.app_env}


def _readiness() -> bool:
    """True when the app can accept research work: DB reachable + provider selectable.

    Cheap by design: one ``SELECT 1`` plus a lazy provider construction
    (no network, no model loading). The model daemon itself is never
    probed — it may be down at boot; jobs then fail cleanly. No detail
    leaves this function: bodies stay ``{"status": ...}`` exactly, and
    diagnostics go to server logs only.
    """
    from deepresearch.llm import LLMConfigurationError
    from deepresearch.providers import create_llm_provider

    try:
        settings.validate_for_runtime()
    except ConfigurationError as exc:
        logger.warning("readiness: invalid configuration: %s", exc)
        return False
    try:
        provider = create_llm_provider(settings)
    except LLMConfigurationError as exc:
        logger.warning("readiness: provider selection failed: %s", exc)
        return False
    except Exception:  # noqa: BLE001 — readiness is boolean; detail stays in logs
        logger.exception("readiness: provider construction failed")
        return False
    try:
        provider.close()
    except Exception:  # noqa: BLE001 — best-effort cleanup on a boolean probe
        logger.exception("readiness: provider close failed")
    engine = get_engine(settings)
    try:
        return check_connection(engine)
    finally:
        engine.dispose()


@app.get("/ready")
def ready() -> JSONResponse:
    if _readiness():
        return JSONResponse(status_code=200, content={"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "not_ready"})
