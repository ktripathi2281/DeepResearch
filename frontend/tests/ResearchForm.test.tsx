import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ResearchForm } from "../components/ResearchForm";

describe("ResearchForm", () => {
  it("labels the question input", () => {
    render(<ResearchForm onSubmit={() => {}} loading={false} />);
    expect(screen.getByLabelText("Research question")).toBeInTheDocument();
  });

  it("rejects an empty question", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={onSubmit} loading={false} />);
    await user.click(screen.getByRole("button", { name: "Research" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Enter a research question.");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("rejects whitespace-only questions", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={onSubmit} loading={false} />);
    await user.type(screen.getByLabelText("Research question"), "   \n ");
    await user.click(screen.getByRole("button", { name: "Research" }));
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("accepts a valid question", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={onSubmit} loading={false} />);
    await user.type(screen.getByLabelText("Research question"), "What is hybrid retrieval?");
    await user.click(screen.getByRole("button", { name: "Research" }));
    expect(onSubmit).toHaveBeenCalledWith("What is hybrid retrieval?");
  });

  it("rejects questions above 4000 characters", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={onSubmit} loading={false} />);
    const box = screen.getByLabelText("Research question");
    // Type a short prefix, then set the long value directly (typing 4001 chars is slow).
    await user.type(box, "Q");
    const long = `Q${"x".repeat(4000)}`; // 4001 chars
    await user.clear(box);
    await user.click(box);
    await user.paste(long);
    await user.click(screen.getByRole("button", { name: "Research" }));
    expect(screen.getByRole("alert")).toHaveTextContent("over the 4000-character limit");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows remaining characters", async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={() => {}} loading={false} />);
    expect(screen.getByText("4000 characters remaining")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Research question"), "abc");
    expect(screen.getByText("3997 characters remaining")).toBeInTheDocument();
  });

  it("disables submission while loading", () => {
    render(<ResearchForm onSubmit={() => {}} loading={true} />);
    const button = screen.getByRole("button", { name: "Researching…" });
    expect(button).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("disabled until it finishes");
  });

  it("is keyboard accessible", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={onSubmit} loading={false} />);
    await user.type(screen.getByLabelText("Research question"), "Keyboard question?");
    await user.keyboard("{Enter}");
    // Enter inside a textarea is a newline, not a submit — the explicit button submits.
    expect(onSubmit).not.toHaveBeenCalled();
    await user.tab();
    expect(screen.getByRole("button", { name: "Research" })).toHaveFocus();
  });
});
