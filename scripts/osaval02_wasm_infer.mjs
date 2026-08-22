#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { WasmOsaval02Model, initSync } from "../bindings/wasm/open_shogi_wasm.js";

const [modelArgument, corpusArgument] = process.argv.slice(2);
if (!modelArgument || !corpusArgument || process.argv.length !== 4) {
  throw new Error("usage: osaval02_wasm_infer.mjs MODEL CORPUS");
}

const wasmBytes = readFileSync(
  resolve(import.meta.dirname, "../bindings/wasm/open_shogi_wasm_bg.wasm"),
);
initSync({ module: wasmBytes });

const modelBytes = readFileSync(resolve(modelArgument));
const corpus = JSON.parse(readFileSync(resolve(corpusArgument), "utf8"));
if (corpus.schema !== "open_shogiai_osaval02_parity_corpus/v1") {
  throw new Error("parity corpus schema is incompatible");
}
if (
  corpus.provenancePolicy !==
  "synthetic positions only; no game records, evaluations, labels, or training examples"
) {
  throw new Error("parity corpus provenance policy is incompatible");
}

const model = new WasmOsaval02Model(modelBytes);
try {
  const fixtures = corpus.fixtures.map((fixture) => {
    if (
      fixture.provenance?.kind !== "synthetic-rule-state" ||
      fixture.provenance?.source !== "hand-authored-from-public-shogi-rules" ||
      fixture.provenance?.containsLabels !== false
    ) {
      throw new Error(`fixture ${fixture.fixtureId} provenance is not source-safe`);
    }
    const history = fixture.history === undefined ? undefined : JSON.stringify(fixture.history);
    const first = model.deterministicTest(fixture.sfen, history);
    const second = model.infer(fixture.sfen, history);
    if (first !== second) {
      throw new Error(`fixture ${fixture.fixtureId} repeated inference is not deterministic`);
    }
    const inference = JSON.parse(first);
    if (inference.positionSha256 !== fixture.positionSha256) {
      throw new Error(`fixture ${fixture.fixtureId} position hash does not match`);
    }
    return { fixtureId: fixture.fixtureId, inference };
  });
  process.stdout.write(
    `${JSON.stringify({
      schema: "open_shogiai_osaval02_wasm_parity/v1",
      modelIdentity: JSON.parse(model.identity()),
      fixtures,
    })}\n`,
  );
} finally {
  model.free();
}
