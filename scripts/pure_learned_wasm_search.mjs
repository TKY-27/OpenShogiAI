#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { WasmBrowserEngine, initSync } from "../bindings/wasm/open_shogi_wasm.js";

// Deterministic Wasm harness for the pure_learned runtime profile. Loads one model artifact
// with an explicit expected SHA-256, runs one browser search under the profile, and prints
// the raw search response so Python tests can compare native and Wasm profile identities.
// Pass the literal string "null" as the expected hash to exercise the fail-closed path.

const [modelArgument, expectedSha256Argument, profileArgument] = process.argv.slice(2);
if (!modelArgument || !expectedSha256Argument || !profileArgument || process.argv.length !== 5) {
  throw new Error("usage: pure_learned_wasm_search.mjs MODEL EXPECTED_SHA256|null PROFILE");
}

const wasmBytes = readFileSync(
  resolve(import.meta.dirname, "../bindings/wasm/open_shogi_wasm_bg.wasm"),
);
initSync({ module: wasmBytes });

const modelBytes = readFileSync(resolve(modelArgument));
const expectedSha256 =
  expectedSha256Argument === "null" ? undefined : expectedSha256Argument;

const engine = new WasmBrowserEngine();
const modelSummary = JSON.parse(engine.loadModel(modelBytes, expectedSha256));
try {
  const response = JSON.parse(engine.search(profileArgument, "pure_learned", 1));
  process.stdout.write(`${JSON.stringify({ modelSummary, response })}\n`);
} finally {
  engine.free();
}
