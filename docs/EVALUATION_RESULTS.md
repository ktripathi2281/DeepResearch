# DeepResearch — Live Evaluation Record (M21)

Measured on 2026-09-21 against the **real local pipeline** — not
fakes. The raw machine-readable result lives git-ignored at
`evals/results/m21-final-live.json` (repository policy: routine
results are not committed); this document is its human-readable
record. Nothing here is estimated or fabricated.

## Configuration

- Dataset: `eval-dev-v1` (8 cases, 7 documents, one case per
  category: single-document, multi-document, exact-lookup,
  semantic, multi-hop, no-answer, conflict, injection)
- Experiment identity: `7c75b06d2b77512d`
- Retrieval: `hybrid_reranked`, top-K 5, RRF (`rrf_k=60`),
  reranker on
- Embeddings: `BAAI/bge-small-en-v1.5` (v1) · Reranker:
  `BAAI/bge-reranker-base` · LLM: `ollama` / `qwen3:4b`
- Isolated database, fixture corpus ingested and embedded with the
  production code paths
- Errors: 0 · Skipped: 0 · Total wall time: ~7 minutes

## Measured results

| Metric | Value |
|---|---|
| Recall@3 / @5 / @10 | 1.0 / 1.0 / 1.0 |
| MRR | 0.93 |
| Answer correctness (expected-fact recall) | 0.86 |
| Citation completeness | 1.0 |
| Faithfulness | null (unavailable) |
| Citation correctness | null (unavailable) |
| Abstention rate | 0.0 |
| Conflict rate | 0.38 (3 of 8) |
| Latency P50 / P95 | 12.0 s / 60.3 s |

## Reading the numbers honestly

- **Retrieval is strong on this fixture**: every case's relevant
  sources appear in the top 3. With 7 documents this is easy;
  treat it as a pipeline-correctness signal, not a quality claim.
- **Faithfulness / citation correctness are null, not zero**: both
  derive from supported/unsupported verification tallies, and this
  run produced no supported rows — the live `qwen3:4b` verifier
  repeatedly returned empty output under its small `max_tokens`
  budget. `null` means "unevaluable here", and the framework
  correctly refuses to score what it cannot judge.
- **Abstention 0.0**: the single no-answer case did not abstain by
  the metric's definition on this run — a real gap on a single
  sample, not a trend.
- **Conflict 0.38**: the tiny shared corpus makes unrelated cases
  retrieve the intentionally disagreeing pair, so conflict fires
  beyond the conflict case. Fixture-scale cross-talk, not a
  detector failure (the detector's unit behavior is pinned by
  tests).
- **Latency is machine-specific** (dev machine, first-run model
  loads amortized): P50 ~12 s, P95 ~60 s per case end to end.

## Limitations (do not generalize)

- 8 cases cannot support general claims about retrieval quality,
  answer quality, or provider quality.
- Keyword/substring metrics approximate; there is no semantic
  judge.
- Verifier quality itself is unevaluated — and this run shows why
  that matters: the headline answer metrics depend on a judge that
  flaked on empty outputs.
- Latencies reflect one local machine on one day.
