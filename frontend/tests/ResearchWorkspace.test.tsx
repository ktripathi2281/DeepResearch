import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ResearchWorkspace } from "../components/ResearchWorkspace";
import { jsonResponse, snapshot } from "./fixtures";

const fetchMock = vi.fn();

describe("ResearchWorkspace", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
    delete process.env.NEXT_PUBLIC_API_BASE_URL;
  });

  function install() {
    vi.stubGlobal("fetch", fetchMock);
    process.env.NEXT_PUBLIC_API_BASE_URL = "http://test-api";
    render(<ResearchWorkspace />);
  }

  async function ask(user: ReturnType<typeof userEvent.setup>, question = "What is RRF?") {
    await user.type(screen.getByLabelText("Research question"), question);
    await user.click(screen.getByRole("button", { name: "Research" }));
  }

  it("runs submit → progress → answer with real stage labels", async () => {
    install();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(snapshot({ request_id: "w-1", job_status: "running", result: null }), 202),
      )
      .mockResolvedValueOnce(
        jsonResponse(
          snapshot({
            request_id: "w-1",
            job_status: "running",
            result: null,
            stages: [{ name: "embedding", duration_ms: 5, success: true }],
          }),
        ),
      )
      .mockResolvedValueOnce(jsonResponse(snapshot({ request_id: "w-1" })));
    const user = userEvent.setup();
    await ask(user);

    // Progress shows a real backend stage label while running.
    await waitFor(() => {
      expect(screen.getByText("Preparing search")).toBeInTheDocument();
    });
    // Submit is disabled while the request runs.
    expect(screen.getByRole("button", { name: "Researching…" })).toBeDisabled();

    // Final answer with status, citation, evidence, and request ID.
    await waitFor(() => {
      expect(screen.getByTestId("status-banner")).toHaveTextContent("Answered");
    });
    expect(screen.getByTestId("answer-text")).toHaveTextContent("Hybrid retrieval combines");
    expect(screen.getByRole("button", { name: "Go to evidence 1" })).toBeInTheDocument();
    expect(screen.getByTestId("evidence-1")).toBeInTheDocument();
    expect(screen.getByTestId("details-request-id")).toHaveTextContent("w-1");
  });

  it("clicking a citation highlights the evidence", async () => {
    install();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(snapshot({ request_id: "w-2", job_status: "running", result: null }), 202),
      )
      .mockResolvedValueOnce(jsonResponse(snapshot({ request_id: "w-2" })));
    const user = userEvent.setup();
    await ask(user);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Go to evidence 1" })).toBeInTheDocument();
    });
    const scrollSpy = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollSpy;
    await user.click(screen.getByRole("button", { name: "Go to evidence 1" }));
    expect(screen.getByTestId("evidence-1")).toHaveClass("evidence-highlight");
    expect(scrollSpy).toHaveBeenCalled();
  });

  it("shows each answer status from the backend taxonomy", async () => {
    for (const [status, heading] of [
      ["insufficient_evidence", "Insufficient evidence"],
      ["conflicting_evidence", "Conflicting evidence"],
      ["no_evidence", "No evidence"],
    ] as const) {
      vi.unstubAllGlobals();
      fetchMock.mockReset();
      vi.stubGlobal("fetch", fetchMock);
      process.env.NEXT_PUBLIC_API_BASE_URL = "http://test-api";
      const { unmount } = render(<ResearchWorkspace />);
      fetchMock
        .mockResolvedValueOnce(
          jsonResponse(snapshot({ request_id: "w-s", job_status: "running", result: null }), 202),
        )
        .mockResolvedValueOnce(jsonResponse(snapshot({ request_id: "w-s", status })));
      const user = userEvent.setup();
      await ask(user, `Status probe ${status}?`);
      await waitFor(() => {
        expect(screen.getByTestId("status-banner")).toHaveTextContent(heading);
      });
      unmount();
    }
  });

  it("shows a user-safe error when the backend is down", async () => {
    install();
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    const user = userEvent.setup();
    await ask(user);
    await waitFor(() => {
      expect(screen.getByTestId("error-banner")).toBeInTheDocument();
    });
    const banner = screen.getByTestId("error-banner");
    expect(within(banner).getByText("The research backend is unavailable. Start it with `uvicorn deepresearch.main:app`.")).toBeInTheDocument();
    expect(banner.textContent).not.toMatch(/Traceback|stack/i);
  });

  it("shows the request ID on research failure", async () => {
    install();
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(snapshot({ request_id: "w-3", job_status: "running", result: null }), 202),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          request_id: "w-3",
          job_status: "failed",
          stages: [],
          result: null,
          error: { message: "Research request failed.", request_id: "w-3" },
        }),
      );
    const user = userEvent.setup();
    await ask(user);
    await waitFor(() => {
      expect(screen.getByTestId("error-banner")).toHaveTextContent("w-3");
    });
  });

  it("announces validation errors accessibly", async () => {
    install();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Research" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Enter a research question.");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
