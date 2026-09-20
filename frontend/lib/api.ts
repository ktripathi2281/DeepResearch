import type { ResearchJobSnapshot } from "../types/research";

/**
 * Thin HTTP client for the research API boundary.
 *
 * - Base URL comes from `NEXT_PUBLIC_API_BASE_URL` (default: local FastAPI).
 * - The request ID always comes from the server response; the client
 *   never generates its own ID (M15 behavior preserved).
 * - No retrieval/generation/verification logic lives here — the client
 *   only submits, polls, and surfaces typed errors.
 */

export const DEFAULT_POLL_INTERVAL_MS = 1000;
/** Local research can take minutes (retrieval + LLM + verification). */
export const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

export class ResearchApiError extends Error {
  readonly kind:
    | "unavailable"
    | "timeout"
    | "invalid_question"
    | "server_error"
    | "research_failed"
    | "unknown_job";
  readonly status: number | null;
  readonly requestId: string | null;

  constructor(
    kind: ResearchApiError["kind"],
    message: string,
    options?: { status?: number | null; requestId?: string | null; cause?: unknown },
  ) {
    super(message, { cause: options?.cause });
    this.name = "ResearchApiError";
    this.kind = kind;
    this.status = options?.status ?? null;
    this.requestId = options?.requestId ?? null;
  }
}

export function apiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
  return raw.trim().replace(/\/+$/, "") || "http://localhost:8000";
}

async function fetchJson(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<{ status: number; body: unknown }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response || typeof response.status !== "number") {
      throw new ResearchApiError(
        "unavailable",
        "The research backend returned an unreadable response.",
      );
    }
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    return { status: response.status, body };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ResearchApiError("timeout", "The request timed out.", { cause: error });
    }
    throw new ResearchApiError(
      "unavailable",
      "The research backend is unavailable. Start it with `uvicorn deepresearch.main:app`.",
      { cause: error },
    );
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

function errorFromBody(body: unknown): { message: string; requestId: string | null } {
  if (body && typeof body === "object" && "error" in body) {
    const err = (body as { error: { message?: unknown; request_id?: unknown } }).error;
    const message = typeof err?.message === "string" ? err.message : "Research request failed.";
    const requestId = typeof err?.request_id === "string" ? err.request_id : null;
    return { message, requestId };
  }
  return { message: "Research request failed.", requestId: null };
}

export interface SubmitOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

/** POST /api/research → 202 with the initial job snapshot. */
export async function submitResearch(
  question: string,
  options?: SubmitOptions,
): Promise<ResearchJobSnapshot> {
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { status, body } = await fetchJson(
    `${apiBaseUrl()}/api/research`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
    timeoutMs,
    options?.signal,
  );
  if (status === 422) {
    throw new ResearchApiError("invalid_question", "That question was rejected. Keep it non-empty and under 4000 characters.", {
      status,
    });
  }
  if (status === 202 && body && typeof body === "object") {
    return body as ResearchJobSnapshot;
  }
  const { message, requestId } = errorFromBody(body);
  throw new ResearchApiError("server_error", message, { status, requestId });
}

export interface PollOptions extends SubmitOptions {
  pollIntervalMs?: number;
  onSnapshot?: (snapshot: ResearchJobSnapshot) => void;
}

/**
 * Poll GET /api/research/{id} until the job completes or fails.
 * Resolves with the completed snapshot; rejects on failure states.
 */
export async function pollResearch(
  requestId: string,
  options?: PollOptions,
): Promise<ResearchJobSnapshot> {
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const pollIntervalMs = options?.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (Date.now() > deadline) {
      throw new ResearchApiError("timeout", "Research is still running past the timeout.", {
        requestId,
      });
    }
    const remaining = Math.max(1, deadline - Date.now());
    const { status, body } = await fetchJson(
      `${apiBaseUrl()}/api/research/${encodeURIComponent(requestId)}`,
      { method: "GET" },
      remaining,
      options?.signal,
    );
    if (status === 404) {
      throw new ResearchApiError("unknown_job", "Unknown research request.", {
        status,
        requestId,
      });
    }
    if (status !== 200 || !body || typeof body !== "object") {
      const { message } = errorFromBody(body);
      throw new ResearchApiError("server_error", message, { status, requestId });
    }
    const snapshot = body as ResearchJobSnapshot;
    options?.onSnapshot?.(snapshot);
    if (snapshot.job_status === "completed" && snapshot.result) {
      return snapshot;
    }
    if (snapshot.job_status === "failed") {
      const message = snapshot.error?.message ?? "Research request failed.";
      throw new ResearchApiError("research_failed", message, { requestId: snapshot.request_id });
    }
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
  }
}
