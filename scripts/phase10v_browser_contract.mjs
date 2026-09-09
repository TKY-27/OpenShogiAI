#!/usr/bin/env node
// Execute the actual wasm-bindgen web artifact in Node, never a native substitute.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve, join, dirname} from 'node:path';

const args = process.argv.slice(2);
const option = (name) => {
  const index = args.indexOf(name);
  assert(index >= 0 && args[index + 1], `missing ${name}`);
  return args[index + 1];
};
const positional = args.length === 3 && !args[0].startsWith('--');
const modulePath = resolve(positional ? join(args[0], 'open_shogi_wasm.js') : option('--module'));
const modelPath = resolve(positional ? args[1] : option('--model'));
const expectedHash = positional ? args[2] : option('--expected-sha256');
const reportPath = positional ? null : resolve(option('--report'));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const bytes = readFileSync(modelPath);
assert.equal(hash(bytes), expectedHash, 'test input identity must be explicit');
const wasmBytes = readFileSync(join(dirname(modulePath), 'open_shogi_wasm_bg.wasm'));
const wasm = await import(pathToFileURL(modulePath));
wasm.initSync({module: wasmBytes});
assert.equal(typeof wasm.WasmBrowserEngine, 'function');
const browser = new wasm.WasmBrowserEngine();
const checks = [];
const record = (name, check) => { check(); checks.push(name); };
const state = () => JSON.parse(browser.snapshot());
const start = state();
record('snapshot-contract', () => {
  assert.equal(start.schema, 'open_shogi_browser_snapshot/v1');
  assert.equal(start.buildClass, 'pure-only');
  assert.equal(start.board.length, 81);
  assert.equal(start.legalMoves.length, 30);
  assert.equal(start.evaluator.kind, 'model-required');
  assert.equal(start.openingBook, null);
});
record('missing-model-fails-closed', () => assert.throws(() => browser.search('eco', 'pure_learned', 1)));
record('hash-required', () => assert.throws(() => browser.loadModel(bytes)));
record('opening-unavailable', () => {
  assert.throws(() => browser.loadOpeningBook(new Uint8Array()));
  assert.throws(() => browser.unloadOpeningBook());
  assert.throws(() => browser.configureOpening('off', 0, 0n, 0));
});
const model = JSON.parse(browser.loadModel(bytes, expectedHash));
record('explicit-osaval03-identity', () => {
  assert.equal(model.modelFormat, 'OSAVAL03');
  assert.equal(model.artifactSha256, expectedHash);
  assert.equal(model.expectedHashVerified, true);
});
record('profile-depth-ceiling', () => {
  assert.throws(() => browser.searchWithTimeControl('eco', 'pure_learned', 1,
    JSON.stringify({schema: 'open_shogi_time_control/v1', depth: 64})));
});
record('prohibited-evaluator-rejected', () => {
  for (const evaluator of ['overall-champion', 'model', 'model-residual', 'model-composite']) {
    assert.throws(() => browser.search('eco', evaluator, 1));
  }
});
browser.playMove('7g7f');
const moved = state();
record('legal-move-contract', () => {
  assert.equal(moved.sideToMove, 'white');
  assert.deepEqual(moved.moves, ['7g7f']);
  assert.throws(() => browser.restore(start.sfen, '["7g7f","7g7f"]'));
  assert.deepEqual(state(), moved);
});
browser.restore(start.sfen, '["7g7f"]');
const result = JSON.parse(browser.searchWithTimeControl('eco', 'pure_learned', 3,
  JSON.stringify({schema: 'open_shogi_time_control/v1', nodes: 100, depth: 1})));
record('actual-wasm-legal-search', () => {
  assert.equal(result.schema, 'open_shogi_browser_search/v1');
  assert.equal(result.perspective, 'white');
  assert.equal(result.evaluator, 'pure_learned');
  assert(moved.legalMoves.some((move) => move.usi === result.bestMove));
  assert(result.runtimeProof);
  assert.equal(result.stats.handcraftedEvalCalls, 0);
  assert.equal(result.stats.residualEvalCalls, 0);
  assert.equal(result.stats.compositeEvalCalls, 0);
  assert.equal(result.stats.fallbackCount, 0);
});
const request = {
  schema: 'open_shogi_analysis/v1', positionSfen: moved.sfen, modelHash: expectedHash,
  evaluatorConfigHash: expectedHash, featureSchemaHash: expectedHash,
  evaluationSemanticsHash: expectedHash, searchOptionsHash: expectedHash,
  openingProfileHash: expectedHash, multiPv: 1,
};
record('analysis-hash-binding', () => {
  assert.throws(() => browser.analysisStart('eco', 'pure_learned', JSON.stringify({...request, modelHash: '0'.repeat(64)})));
});
record('analysis-worker-lifecycle', () => {
  assert.equal(JSON.parse(browser.analysisStart('eco', 'pure_learned', JSON.stringify(request))).event, 'started');
  assert.throws(() => browser.analysisRestart());
  assert.throws(() => browser.analysisStep(JSON.stringify({schema: request.schema, nodes: 100, maxDepth: 64, timestampMs: 10})));
  const step = JSON.parse(browser.analysisStep(JSON.stringify({schema: request.schema, nodes: 100, maxDepth: 1, timestampMs: 10})));
  assert.equal(step.event, 'updates');
  assert.equal(JSON.parse(browser.analysisWorkerFailed()).event, 'worker-failed');
  assert.equal(JSON.parse(browser.analysisRestart()).event, 'restarted');
  assert.equal(JSON.parse(browser.analysisStop()).event, 'stopped');
  browser.reset();
  assert.throws(() => browser.analysisStep('{}'));
});
record('failed-replacement-removes-old-model', () => {
  assert.throws(() => browser.loadModel(bytes, '0'.repeat(64)));
  assert.throws(() => browser.search('eco', 'pure_learned', 1));
});
browser.loadModel(bytes, expectedHash);
browser.unloadModel();
record('unload-closes-search', () => assert.throws(() => browser.search('eco', 'pure_learned', 1)));
browser.free();
const report = {
  schema: 'open_shogiai_phase10v_browser_contract/v1', status: 'PASS',
  runtime: 'actual-wasm-web-bindings-in-node', buildClass: 'pure-only', modelFormat: 'OSAVAL03',
  modelSha256: expectedHash, moduleSha256: hash(readFileSync(modulePath)), wasmSha256: hash(wasmBytes),
  checks, search: result,
};
if (reportPath) writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report));
