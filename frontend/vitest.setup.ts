import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// RTL auto-cleanup only registers itself when the global `afterEach`
// exists at import time; vitest runs here with `globals: false`, so
// register it explicitly to keep renders isolated between tests.
afterEach(() => {
  cleanup();
});
