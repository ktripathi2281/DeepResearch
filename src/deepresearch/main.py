"""FastAPI application — Milestones 1 + 15 + 18.

Endpoints:
- GET /health — liveness, no DB dependency.
- GET /ready  — readiness, checks PostgreSQL connectivity.
- POST /api/research — start a research job (202 + pollable snapshot).
- GET /api/research/{request_id} — poll a research job.

M15 request tracing: every request gets an ``X-Request-ID`` (preserved
when the client supplies a usable one, generated otherwise), an
isolated observability context, and a completion log entry.
Health/readiness stay quiet (header only, no access log).

M18 CORS: the browser frontend runs on a different origin
(``CORS_ORIGINS``, default ``http://localhost:3000``); cookies are
never used, so credentials stay disabled.

No ingestion, retrieval, generation, or frontend logic here — the
research router in ``research_api`` is the only boundary.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deepresearch.config import get_settings
from deepresearch.db import check_connection, get_engine
from deepresearch.logging import configure_logging, get_logger
from deepresearch.observability import (
    REQUEST_ID_HEADER,
    get_current_trace,
    normalize_request_id,
    traced_request,
)
from deepresearch.research_api import router as research_router

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(title="DeepResearch API", version="0.1.0")

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
    """User-safe 500: log detail server-side, never leak stack traces."""
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
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "env": settings.app_env}


@app.get("/ready")
def ready() -> JSONResponse:
    engine = get_engine(settings)
    try:
        ok = check_connection(engine)
    finally:
        engine.dispose()
    if ok:
        return JSONResponse(status_code=200, content={"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "not_ready"})
