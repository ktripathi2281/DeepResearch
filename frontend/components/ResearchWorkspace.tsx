"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ResearchJobSnapshot, StageInfo } from "../types/research";
import { ResearchApiError, pollResearch, submitResearch } from "../lib/api";
import { ResearchForm } from "./ResearchForm";
import { ProgressPanel } from "./ProgressPanel";
import { StatusBanner } from "./StatusBanner";
import { AnswerView } from "./AnswerView";
import { EvidencePanel } from "./EvidencePanel";
import { ResearchDetails } from "./ResearchDetails";
import { ErrorBanner } from "./ErrorBanner";

type Phase =
  | { name: "idle" }
  | { name: "running"; requestId: string; stages: StageInfo[] }
  | { name: "done"; snapshot: ResearchJobSnapshot }
  | { name: "error"; error: ResearchApiError };

function toApiError(error: unknown): ResearchApiError {
  if (error instanceof ResearchApiError) return error;
  return new ResearchApiError("unavailable", "The research backend is unavailable.", {
    cause: error,
  });
}

/**
 * Research request → research result. Owns the submit/poll lifecycle;
 * all domain knowledge (retrieval, generation, verification) stays in
 * the backend — this component only moves snapshots to the screen.
 */
export function ResearchWorkspace() {
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const [highlightedCitation, setHighlightedCitation] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const handleSubmit = useCallback(async (question: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setHighlightedCitation(null);
    setPhase({ name: "running", requestId: "", stages: [] });
    try {
      const initial = await submitResearch(question, { signal: controller.signal });
      setPhase({ name: "running", requestId: initial.request_id, stages: initial.stages });
      const completed = await pollResearch(initial.request_id, {
        signal: controller.signal,
        onSnapshot: (snapshot) =>
          setPhase({ name: "running", requestId: snapshot.request_id, stages: snapshot.stages }),
      });
      if (!completed.result) {
        throw new ResearchApiError("research_failed", "Research finished without a result.", {
          requestId: completed.request_id,
        });
      }
      setPhase({ name: "done", snapshot: completed });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      if (controller.signal.aborted) return;
      setPhase({ name: "error", error: toApiError(error) });
    }
  }, []);

  const handleCitationClick = useCallback((citationId: number) => {
    setHighlightedCitation(citationId);
    const target = document.getElementById(`evidence-${citationId}`);
    target?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, []);

  const running = phase.name === "running";
  const stages: StageInfo[] =
    phase.name === "running" ? phase.stages : phase.name === "done" ? phase.snapshot.stages : [];

  return (
    <div className="workspace">
      <ResearchForm onSubmit={handleSubmit} loading={running} />

      {running && <ProgressPanel stages={stages} running />}

      {phase.name === "done" && phase.snapshot.result && (
        <div className="result">
          <StatusBanner status={phase.snapshot.result.status} />
          <AnswerView
            answer={phase.snapshot.result.answer}
            citations={phase.snapshot.result.citations}
            onCitationClick={handleCitationClick}
          />
          <EvidencePanel
            evidence={phase.snapshot.result.evidence}
            highlightedCitation={highlightedCitation}
          />
          <ResearchDetails result={phase.snapshot.result} />
        </div>
      )}

      {phase.name === "error" && (
        <ErrorBanner
          error={phase.error}
          onDismiss={() => setPhase({ name: "idle" })}
        />
      )}
    </div>
  );
}
