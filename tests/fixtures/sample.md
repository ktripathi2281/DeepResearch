# DeepResearch Sample Document

## Retrieval

Hybrid retrieval combines dense vector search with lexical BM25 matching.
Each method compensates for the other's blind spots.

## Reranking

A cross-encoder reranker reorders the fused candidate set.
Latency matters, so candidate counts stay bounded.
