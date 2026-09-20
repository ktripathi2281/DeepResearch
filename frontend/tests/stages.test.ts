import { describe, expect, it } from "vitest";
import { orderStageNames, stageLabel } from "../lib/stages";

describe("stage labels", () => {
  it("maps real backend stages to user-facing labels", () => {
    expect(stageLabel("vector_retrieval")).toBe("Searching evidence (semantic)");
    expect(stageLabel("bm25_retrieval")).toBe("Searching evidence (keyword)");
    expect(stageLabel("hybrid_fusion")).toBe("Combining search results");
    expect(stageLabel("reranking")).toBe("Reranking evidence");
    expect(stageLabel("generation")).toBe("Generating answer");
    expect(stageLabel("citation_verification")).toBe("Verifying citations");
  });

  it("passes unknown backend stages through instead of hiding them", () => {
    expect(stageLabel("future_custom_stage")).toBe("future_custom_stage");
  });

  it("orders stages by pipeline position, unknowns last", () => {
    expect(orderStageNames(["generation", "embedding", "zzz_custom"])).toEqual([
      "embedding",
      "generation",
      "zzz_custom",
    ]);
  });
});
