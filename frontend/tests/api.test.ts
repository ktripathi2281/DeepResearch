import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ResearchApiError,
  pollResearch,
  submitResearch,
} from "../lib/api";
import { jsonResponse, snapshot } from "./fixtures";

const fetchMock = vi.fn();

function mockFetchOnce(response: Response) {
  fetchMock.mockResolvedValueOnce(response);
}

describe("research API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
    delete process.env.NEXT_PUBLIC_API_BASE_URL;
  });

  function installFetch() {
    vi.stubGlobal("fetch", fetchMock);
    process.env.NEXT_PUBLIC_API_BASE_URL = "http://test-api";
  }

  it("submits a question and returns the snapshot with the server request ID", async () => {
    installFetch();
    const body = snapshot({ request_id: "server-id-1", job_status: "running", result: null });
    mockFetchOnce(jsonResponse(body, 202));

    const result = await submitResearch("What is RRF?");
    expect(result.request_id).toBe("server-id-1");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://test-api/api/research");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ question: "What is RRF?" });
  });

  it("maps 422 to an invalid-question error", async () => {
    installFetch();
    mockFetchOnce(jsonResponse({ detail: "too long" }, 422));
    await expect(submitResearch("x".repeat(4001))).rejects.toMatchObject({
      kind: "invalid_question",
    });
  });

  it("maps 500 envelopes to server errors with the request ID", async () => {
    installFetch();
    mockFetchOnce(
      jsonResponse({ error: { message: "Internal server error.", request_id: "r-9" } }, 500),
    );
    const error = await submitResearch("Q?").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ResearchApiError);
    expect(error).toMatchObject({ kind: "server_error", requestId: "r-9" });
  });

  it("maps fetch failure to backend-unavailable", async () => {
    installFetch();
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    await expect(submitResearch("Q?")).rejects.toMatchObject({ kind: "unavailable" });
  });

  it("maps abort to timeout", async () => {
    installFetch();
    fetchMock.mockImplementationOnce(
      (_url: string, init?: RequestInit) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );
    await expect(submitResearch("Q?", { timeoutMs: 20 })).rejects.toMatchObject({
      kind: "timeout",
    });
  });

  it("polls running snapshots until completed", async () => {
    installFetch();
    mockFetchOnce(jsonResponse(snapshot({ request_id: "r-1", job_status: "running", result: null })));
    const done = snapshot({ request_id: "r-1", job_status: "completed" });
    mockFetchOnce(jsonResponse(done));

    const seen: string[] = [];
    const final = await pollResearch("r-1", {
      pollIntervalMs: 1,
      onSnapshot: (s) => seen.push(s.job_status),
    });
    expect(final.job_status).toBe("completed");
    expect(final.result?.answer).toContain("Hybrid retrieval");
    expect(seen).toEqual(["running", "completed"]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://test-api/api/research/r-1");
  });

  it("rejects failed jobs with the server message", async () => {
    installFetch();
    mockFetchOnce(
      jsonResponse({
        request_id: "r-2",
        job_status: "failed",
        stages: [],
        result: null,
        error: { message: "Research request failed.", request_id: "r-2" },
      }),
    );
    await expect(pollResearch("r-2", { pollIntervalMs: 1 })).rejects.toMatchObject({
      kind: "research_failed",
      requestId: "r-2",
    });
  });

  it("rejects unknown jobs on 404", async () => {
    installFetch();
    mockFetchOnce(jsonResponse({ detail: "unknown research request" }, 404));
    await expect(pollResearch("nope", { pollIntervalMs: 1 })).rejects.toMatchObject({
      kind: "unknown_job",
    });
  });

  it("times out when the job never finishes", async () => {
    installFetch();
    // Fresh Response per call: a Response body can only be consumed once.
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse(snapshot({ job_status: "running", result: null }))),
    );
    await expect(pollResearch("r-3", { pollIntervalMs: 1, timeoutMs: 30 })).rejects.toMatchObject(
      {
        kind: "timeout",
        requestId: "r-3",
      },
    );
  });
});
