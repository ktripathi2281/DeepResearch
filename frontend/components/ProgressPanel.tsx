"use client";

import type { StageInfo } from "../types/research";
import { orderStageNames, stageLabel } from "../lib/stages";

interface ProgressPanelProps {
  stages: StageInfo[];
  running: boolean;
}

/**
 * Research progress from real backend stage transitions.
 * Completed stages show recorded durations; while running, a final
 * "working" row indicates activity without inventing percentages.
 */
export function ProgressPanel({ stages, running }: ProgressPanelProps) {
  const byName = new Map<string, StageInfo>();
  for (const stage of stages) byName.set(stage.name, stage);
  const ordered = orderStageNames([...byName.keys()]);

  return (
    <section aria-label="Research progress" className="panel">
      <h2 className="panel-title">Research progress</h2>
      {ordered.length === 0 && running && (
        <p role="status" className="progress-working">
          Starting research…
        </p>
      )}
      {ordered.length === 0 && !running && (
        <p className="hint">No stages recorded.</p>
      )}
      {ordered.length > 0 && (
        <ol className="stage-list" aria-live="polite">
          {ordered.map((name) => {
            const stage = byName.get(name);
            return (
              <li key={name} className="stage-item">
                <span aria-hidden="true" className={stage?.success === false ? "stage-bad" : "stage-ok"}>
                  {stage?.success === false ? "✕" : "✓"}
                </span>{" "}
                <span>{stageLabel(name)}</span>{" "}
                <span className="stage-duration">
                  {stage ? `${stage.duration_ms} ms` : ""}
                </span>
              </li>
            );
          })}
          {running && (
            <li key="__working" className="stage-item">
              <span aria-hidden="true" className="stage-working-dot">
                ●
              </span>{" "}
              <span role="status">Working…</span>
            </li>
          )}
        </ol>
      )}
    </section>
  );
}
