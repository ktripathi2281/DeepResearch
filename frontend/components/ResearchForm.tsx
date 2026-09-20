"use client";

import { useState } from "react";
import { MAX_QUESTION_CHARS } from "../types/research";

interface ResearchFormProps {
  onSubmit: (question: string) => void;
  loading: boolean;
}

function validate(question: string): string | null {
  const trimmed = question.trim();
  if (!trimmed) return "Enter a research question.";
  if (trimmed.length > MAX_QUESTION_CHARS) {
    const over = trimmed.length - MAX_QUESTION_CHARS;
    return `Question is ${over} character${over === 1 ? "" : "s"} over the ${MAX_QUESTION_CHARS}-character limit.`;
  }
  return null;
}

/**
 * Research question input. Client-side validation mirrors the backend
 * limit for fast feedback; the backend remains authoritative.
 */
export function ResearchForm({ onSubmit, loading }: ResearchFormProps) {
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);

  const remaining = MAX_QUESTION_CHARS - question.trim().length;

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const problem = validate(question);
    setError(problem);
    if (problem === null && !loading) {
      onSubmit(question.trim());
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate aria-label="Research question form">
      <label htmlFor="research-question" className="field-label">
        Research question
      </label>
      <textarea
        id="research-question"
        name="question"
        rows={4}
        value={question}
        disabled={loading}
        onChange={(event) => {
          setQuestion(event.target.value);
          if (error) setError(validate(event.target.value));
        }}
        placeholder="Ask a complex research question — DeepResearch investigates the indexed corpus and answers with evidence-backed citations."
        aria-describedby="question-hint question-error question-count"
        aria-invalid={error !== null}
        className="question-input"
      />
      <p id="question-hint" className="hint">
        Enter text normally; press Research to submit. Backend limit: {MAX_QUESTION_CHARS}{" "}
        characters.
      </p>
      <div className="form-row">
        <span
          id="question-count"
          className={remaining < 0 ? "count count-over" : "count"}
          aria-live="polite"
        >
          {remaining >= 0
            ? `${remaining} characters remaining`
            : `${-remaining} characters over limit`}
        </span>
        <button type="submit" disabled={loading} className="primary-button">
          {loading ? "Researching…" : "Research"}
        </button>
      </div>
      {loading && (
        <p className="hint" role="status">
          Research is running — submission is disabled until it finishes.
        </p>
      )}
      {error && (
        <p id="question-error" role="alert" className="error-text">
          {error}
        </p>
      )}
    </form>
  );
}
