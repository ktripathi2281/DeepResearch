# ADR-006: BM25 lexical retrieval (in-process Okapi)

Date: 2026-09-20 | Status: Accepted | Milestone: M6

## Context

M6 needs lexical retrieval over `Chunk.text` (FR-5, OPENCODE M6),
coexisting with M5 vector retrieval but never merged with it (M7 owns
fusion). ADR-001 already fixed in-Python BM25 over PostgreSQL persisted
chunks — no external search service.

## Decision

- **Direct Okapi BM25 implementation** (`src/deepresearch/bm25.py`,
  ~200 lines incl. lifecycle) instead of `rank-bm25`. Zero new
  dependencies; full control of the tokenizer contract below; every
  line reviewable. Standard parameters `k1=1.5`, `b=0.75` via
  `Settings.bm25_k1/bm25_b` so experiments vary them explicitly.
- **Tokenization: M3 `tokenize` (`deepresearch-regex-v1`) + lowercase,
  punctuation-only tokens dropped.** One pipeline for queries and
  corpus — a second incompatible tokenizer is explicitly rejected. No
  stemming/lemmatization: exact lexical behavior stays obvious, which
  is the point of keeping a lexical path beside the semantic one.
- **Scoring: textbook Okapi** with Lucene-style positive IDF
  (`ln(1 + (N−df+0.5)/(df+0.5))`), summing over unique query terms
  (no query-tf weight — repeated query terms score identically, by
  design). Zero-score docs are excluded, so no-match queries return
  `[]`; punctuation-only queries (no indexable terms) also return `[]`,
  while blank queries raise like M5.
- **Results reuse M5's `RetrievalResult`** with `method="bm25"`,
  sharing `validate_top_k` (default 5, cap 100, raise-never-clamp) and
  `document_id`/`document_type` filters applied at corpus load.
- **Lifecycle: snapshot index + explicit freshness.**
  `build_index` is pure (entries in, index out);
  `build_index_from_session` loads from PG;
  `BM25Index.is_fresh` compares the persisted chunk id-set (ids only,
  cheap) against the snapshot — sufficient because chunks are
  immutable (ADR-001). A provided stale index raises
  `StaleIndexError`; `BM25Retriever` holds one snapshot and
  auto-refreshes only when the corpus changed (rebuilds logged).
  Winner metadata is re-fetched by chunk id, so the index stores ids
  and statistics, not duplicated texts.
- **No schema change**: no BM25 columns, no stored scores — BM25 is a
  search concern, not persistence.
- **Why BM25 beside vectors:** lexical retrieval wins on exact terms
  (names, error codes, rare keywords) where embedding similarity
  blurs; vector retrieval wins on paraphrase and semantics where
  shared vocabulary is absent. M7 fuses the two because their failure
  modes are complementary, not overlapping.

## Alternatives considered

- `rank-bm25` package: maintained, but an extra dependency for a
  60-line algorithm, with its own tokenization expectations.
- PostgreSQL FTS instead of BM25: rejected by ADR-001 (in-Python keeps
  one retrieval codebase and deterministic behavior across backends —
  the full M6 suite runs on SQLite too).
- Stemming: rejected — obscures exact-match semantics for no
  M6-scale benefit.

## Consequences

- M6 leaves two independent paths (`retrieve` vector,
  `retrieve_bm25`/`BM25Retriever`); M7 fuses their
  higher-is-better scores without touching either implementation.
- EVALUATION BM25-only baselines run against this implementation.
