# ADR-016: Evaluation framework (measure, don't declare winners)

Date: 2026-09-20 | Status: Accepted | Milestone: M16

## Context

M1–M15 built a pipeline with no measurement: whether hybrid beats
vector-only, or reranking earns its latency, was intuition. M16
must make such claims testable with versioned datasets, explicit
metric definitions, and reproducible experiment identities — using
fakes and deterministic checks so the suite needs no Ollama.

## Decision

- **Domain in `evaluation.py`, execution in `eval_runner.py`.**
  Cases (id, question, 8 explicit categories, difficulty,
  source-level ground truth, expected facts, expected status,
  abstain/injection markers), datasets (version + fixture corpus +
  cases, duplicate IDs rejected), experiment configs, and results
  (measured values, `None` for unavailable, separate
  errors/skips lists).
- **Relevance by `document.source`**, not UUIDs: fixtures stay
  readable and runnable without database archaeology. Retrieved
  units are first-appearance source order.
- **Retrieval metrics**: Recall@3/5/10 and MRR with explicit
  semantics — `None` without ground truth (never zero), `0.0`
  when ground truth exists but nothing was found, duplicate
  results deduplicated.
- **Answer metrics are deterministic approximations, labeled as
  such**: fact presence and citation completeness are
  case-insensitive substring checks (wording, not semantics);
  faithfulness/correctness read M12 verification rows (per-claim
  vs per-citation, invalid markers excluded from denominators);
  abstention/conflict read `AnswerStatus` (`no_evidence` /
  `insufficient_evidence` with clean citations;
  `conflicting_evidence` with sources kept). No judge model is
  introduced — semantic grading would be a separate,
  clearly-labeled metric.
- **Injection evaluation is structural**: markers retrieved as
  data + pipeline completion. It proves plumbing, not adversarial
  robustness — stated plainly in code and docs.
- **Latency from M15 traces**: per-case wall time plus recorded
  stage durations; P50/P95 by linear interpolation (`None` on
  empty). Real timings only, never fabricated.
- **Experiment identity is a sha256 fingerprint** of the canonical
  config (timestamp/notes excluded): different configs cannot
  collide into one identity. Results serialize deterministically
  (`sort_keys`, trailing newline) to `evals/results/` (git-ignored
  except committed snapshots).
- **Comparison = N experiments, reported side by side.**
  `dispatch_retrieval` maps vector/bm25/hybrid/hybrid_reranked to
  the existing implementations without touching them. The runner
  never ranks methods; interpretation belongs to the developer
  reading the report (per EVALUATION.md §4: measure, don't assume).
- **Per-case errors recorded, never raised**; missing providers
  produce skips with reasons; config/dataset version mismatch
  fails fast (comparing across datasets would be meaningless).

## Alternatives considered

- LLM-as-judge scoring: the obvious next metric, explicitly
  deferred — it needs its own calibration, cost, and
  nondeterminism analysis, not a silent addition.
- Chunk-UUID ground truth: precise but hostile to fixture
  authorship and cross-run reuse; source matching is the
  pragmatic level for a dev-scale dataset.
- A CLI runner now: deferred — the Python API (`load_dataset`,
  `run_evaluation`, `save_result`) is the interface; a CLI is
  wrapping, not new capability.

## Consequences

- The shipped `eval-dev-v1` (8 cases, 7 docs) is a development
  fixture, not a representative benchmark — stated in the dataset,
  docs, and tests. Real comparisons need the 50–100 case dataset
  from EVALUATION.md §2, which this framework is ready to load.
- M16+ experiments (chunk size, top-K, reranker on/off, model
  swaps) vary `ExperimentConfig` and diff identities instead of
  arguing from anecdotes.
