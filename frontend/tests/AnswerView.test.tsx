import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { AnswerStatus } from "../types/research";
import { StatusBanner } from "../components/StatusBanner";
import { AnswerView } from "../components/AnswerView";
import { researchResult } from "./fixtures";

describe("StatusBanner", () => {
  const cases: Array<[AnswerStatus, string]> = [
    ["answered", "Answered"],
    ["insufficient_evidence", "Insufficient evidence"],
    ["conflicting_evidence", "Conflicting evidence"],
    ["no_evidence", "No evidence"],
  ];
  it.each(cases)("renders the %s banner", (status, heading) => {
    render(<StatusBanner status={status} />);
    const banner = screen.getByTestId("status-banner");
    expect(banner).toHaveTextContent(heading);
    expect(banner).toHaveAttribute("role", "status");
  });

  it("does not rely on color alone", () => {
    render(<StatusBanner status="conflicting_evidence" />);
    // The heading text itself carries the meaning, independent of CSS color.
    expect(screen.getByText("Conflicting evidence")).toBeInTheDocument();
  });
});

describe("AnswerView", () => {
  const citations = researchResult().citations;

  it("renders valid citations as buttons that map to evidence", async () => {
    const onCitationClick = vi.fn();
    const user = userEvent.setup();
    render(
      <AnswerView
        answer="Hybrid retrieval combines vector and lexical search [1]."
        citations={citations}
        onCitationClick={onCitationClick}
      />,
    );
    const button = screen.getByRole("button", { name: "Go to evidence 1" });
    expect(button).toHaveTextContent("[1]");
    await user.click(button);
    expect(onCitationClick).toHaveBeenCalledWith(1);
  });

  it("renders invalid markers as plain text", () => {
    render(
      <AnswerView
        answer="A phantom source [99] is claimed."
        citations={citations}
        onCitationClick={() => {}}
      />,
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.getByTestId("answer-text")).toHaveTextContent("[99]");
  });

  it("renders duplicate citations as separate buttons to the same evidence", async () => {
    const onCitationClick = vi.fn();
    const user = userEvent.setup();
    render(
      <AnswerView
        answer="First [1] and again [1]."
        citations={citations}
        onCitationClick={onCitationClick}
      />,
    );
    const buttons = screen.getAllByRole("button", { name: "Go to evidence 1" });
    expect(buttons).toHaveLength(2);
    await user.click(buttons[1]);
    expect(onCitationClick).toHaveBeenCalledWith(1);
  });

  it("handles answers with no citations", () => {
    render(<AnswerView answer="No evidence was found." citations={[]} onCitationClick={() => {}} />);
    expect(screen.getByTestId("answer-text")).toHaveTextContent("No evidence was found.");
    expect(
      screen.getByText("No citations — this answer is not backed by listed evidence."),
    ).toBeInTheDocument();
  });
});
