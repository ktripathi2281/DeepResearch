import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ResearchDetails } from "../components/ResearchDetails";
import { researchResult } from "./fixtures";

describe("ResearchDetails", () => {
  it("displays the request ID and counts", () => {
    render(<ResearchDetails result={researchResult()} />);
    expect(screen.getByTestId("details-request-id")).toHaveTextContent("req-123");
    expect(screen.getByTestId("details-evidence-count")).toHaveTextContent("1 items");
    expect(screen.getByTestId("details-citation-count")).toHaveTextContent("1 valid");
  });

  it("displays verification state without private reasoning", () => {
    const { container } = render(<ResearchDetails result={researchResult()} />);
    expect(screen.getByTestId("details-verification")).toHaveTextContent("1 supported");
    // No chain-of-thought, prompt, or reasoning keys appear anywhere.
    expect(container.textContent).not.toMatch(/chain-of-thought|system prompt|reasoning/i);
  });

  it("surfaces invalid citation counts", () => {
    const result = researchResult({
      research_details: { ...researchResult().research_details, invalid_citation_count: 2 },
    });
    render(<ResearchDetails result={result} />);
    expect(screen.getByTestId("details-citation-count")).toHaveTextContent("2 invalid");
  });

  it("shows conflicts without resolving them", () => {
    const result = researchResult({
      status: "conflicting_evidence",
      conflicts: [
        {
          conflict_id: "conflict-1",
          conflict_type: "numeric_mismatch",
          description: "Evidence [1] states 2022 where evidence [2] states 2024.",
          citation_ids: [1, 2],
        },
      ],
    });
    render(<ResearchDetails result={result} />);
    expect(screen.getByText("Detected conflicts")).toBeInTheDocument();
    expect(screen.getByText(/states 2022 where evidence/)).toBeInTheDocument();
  });

  it("shows model names when the backend reports them", () => {
    render(<ResearchDetails result={researchResult()} />);
    expect(screen.getByText("llm: qwen3:4b")).toBeInTheDocument();
  });
});
