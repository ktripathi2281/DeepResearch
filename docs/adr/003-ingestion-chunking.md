# ADR-003: Ingestion parsing, tokenization, and chunking

Date: 2026-09-20 | Status: Accepted | Milestone: M3

## Context

M3 needs bytes → Document + Chunks with deterministic, configurable,
token-based chunking (defaults 800/120, decided pre-M1). Choices needed:
parsing libraries, tokenizer, hash input, title fallback, empty-input
semantics — without coupling to M4's embedding model or adding heavy
frameworks.

## Decision

- **Parsing: `pypdf` (PDF), `beautifulsoup4` + stdlib `html.parser`
  backend (HTML), stdlib regex (Markdown headings), stdlib decode
  (TXT).** `pypdf` is the maintained pure-Python PDF successor and was
  already transitively present; `beautifulsoup4` is the standard
  small HTML extractor and needs no `lxml` (uses `html.parser`).
  HTML `<script>/<style>/<noscript>/<template>` are stripped —
  document content stays untrusted data.
- **Structure:** PDF pages preserved per page (1-based, matching the
  `page >= 1` CHECK); Markdown `#{1,6}` headings tracked, H1 doubles as
  title fallback; HTML `h1–h6` tracked, `<title>`/H1 as fallback; TXT
  has no structure. A chunk inherits the page/section of the token
  where its window starts.
- **Tokenizer: local regex (`[\w']+` words, single punctuation),
  named `deepresearch-regex-v1`.** Deterministic, zero-dependency.
  Deliberately NOT the `bge-small-en-v1.5` tokenizer: M3 chunk
  boundaries must not shift when M4 picks embedding-library versions.
  Counts are "deepresearch tokens" — valid for relative chunking
  experiments (EVALUATION Experiment A); absolute model-token parity
  is explicitly out of scope.
- **Chunking: sliding window (target 800, overlap 120, overlap <
  target enforced via `ChunkingError`).** Sequential indices from 0,
  no empty chunks, pure function of (units, config). Per-chunk
  `metadata` records `{tokenizer, token_count}` so experiments can
  compare configurations from the DB.
- **Hash: `sha256("<canonical type>\\n<normalized text>")`.** The type
  prefix prevents a PDF and a TXT with identical text from colliding;
  the filename is excluded (renamed copies deduplicate). Same content
  + same config → same chunks → same hash, every time.
- **Titles:** explicit argument → parser title (MD H1 / HTML
  `<title>`/H1 / PDF metadata) → NULL. Never fabricated.
- **Empty-but-valid inputs** (blank TXT, text-less PDF) persist a
  chunk-less Document; corrupt inputs (bad PDF bytes, undecodable
  text, unknown type) raise `ParsingError`; bad chunk config raises
  `ChunkingError`. Nothing is silently swallowed.
- **Idempotency:** hash lookup first; duplicate returns existing rows
  with `duplicate=True` and zero writes. `IntegrityError` on insert
  rolls back and re-reads (covers concurrent-ingest races).

## Alternatives considered

- `tiktoken`: accurate model tokens, but adds a dependency and couples
  M3 to an OpenAI vocabulary unrelated to our local stack.
- `bge-small-en-v1.5` tokenizer now: would tie chunk boundaries to M4
  library versions — rejected for the coupling reason above.
- `lxml` backend for HTML: faster, but a compiled dependency for no
  M3-scale benefit.
- Separate `embeddings` writes in M3: rejected — M4 owns the first
  write to `chunks.embedding` per ADR-001.

## Consequences

- New runtime deps: `pypdf>=4.0`, `beautifulsoup4>=4.12` (both local,
  no network, no cloud).
- `Settings` gains `chunk_target_tokens` / `chunk_overlap_tokens`
  (800/120 defaults); M3 service takes explicit overrides for
  experiments.
- M4 embeds chunk text as-is; if model-token-exact chunking is ever
  needed, it arrives as a new tokenizer option + experiment, not a
  silent substitution.
