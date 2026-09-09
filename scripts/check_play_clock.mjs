#!/usr/bin/env node
// Measured Wasm clock/cancellation checks. Node workers are not browser UI evidence.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const clock = remaining => JSON.stringify({ schema: 'open_shogi_time_control/v1',
  blackTimeMs: remaining, whiteTimeMs: remaining, safetyMarginMs: 50 });
function verify(result, modelHash) {
  assert.equal(result.runtimeProof.model_sha256, modelHash);
  for (const key of ['handcrafted_eval_calls', 'residual_eval_calls', 'composite_eval_calls',
    'book_hits', 'teacher_calls', 'fallback_count']) assert.equal(result.runtimeProof[key], 0);
  if (result.outcome === 'evaluated') assert(result.runtimeProof.learned_eval_calls > 0);
  else assert.equal(result.runtimeProof.learned_eval_calls, 0);
}

if (!isMainThread) {
  const started = performance.now();
  const wasm = await import(pathToFileURL(workerData.modulePath));
  wasm.initSync({ module: readFileSync(join(dirname(workerData.modulePath), 'open_shogi_wasm_bg.wasm')) });
  const engine = new wasm.WasmBrowserEngine();
  engine.loadModel(readFileSync(workerData.leafPath), workerData.leafHash);
  const initializationMs = performance.now() - started;
  let sharedFlag;
  let probes = 0;
  globalThis.__openShogiPlayCancelled = () => { probes++; return Atomics.load(sharedFlag, 0) !== 0; };
  globalThis.__openShogiPlayProgress = raw => parentPort.postMessage({ event: 'progress', value: JSON.parse(raw) });
  parentPort.on('message', request => {
    try {
      sharedFlag = new Int32Array(request.cancelBuffer);
      probes = 0;
      engine.reset();
      if (request.white) engine.playMove('7g7f');
      if (request.sfen) engine.restore(request.sfen, '[]');
      const before = JSON.parse(engine.snapshot());
      const control = request.nodes ? JSON.stringify({ ...JSON.parse(clock(600_000)), nodes: request.nodes }) : clock(request.remaining);
      const requestStarted = performance.now();
      if (request.legacy) {
        const result = JSON.parse(engine.searchWithTimeControl(request.profile, 'pure_learned', 1, control));
        parentPort.postMessage({ event: 'done', value: { result }, wallMs: performance.now() - requestStarted, probes, before });
      } else {
        const prepared = JSON.parse(engine.playStart(request.profile, 'pure_learned', 1, control));
        parentPort.postMessage({ event: 'prepared', value: prepared, before });
        const value = prepared.done ? prepared : JSON.parse(engine.playRun());
        parentPort.postMessage({ event: 'done', value, wallMs: performance.now() - requestStarted, probes, before });
      }
    } catch (error) { parentPort.postMessage({ event: 'error', error: String(error?.stack ?? error) }); }
  });
  parentPort.postMessage({ event: 'ready', initializationMs });
} else {
  const [moduleArg, leafArg, leafHash, reportArg, matrix] = process.argv.slice(2);
  assert(reportArg, 'usage: check_play_clock.mjs MODULE LEAF SHA REPORT [--matrix]');
  assert(!matrix || matrix === '--matrix');
  const modulePath = resolve(moduleArg), leafPath = resolve(leafArg);
  assert.equal(hash(readFileSync(leafPath)), leafHash);
  const worker = new Worker(fileURLToPath(import.meta.url), { workerData: { modulePath, leafPath, leafHash } });
  let initializationMs;
  await new Promise((resolveReady, reject) => {
    worker.once('error', reject);
    worker.once('message', event => { assert.equal(event.event, 'ready'); initializationMs = event.initializationMs; resolveReady(); });
  });
  const evidence = [];
  async function run(request) {
    const cancelBuffer = new SharedArrayBuffer(4), flag = new Int32Array(cancelBuffer);
    const updates = [];
    let cancelSentMs = null, prepared;
    const origin = performance.now();
    return new Promise((resolveRun, reject) => {
      const timeout = setTimeout(() => finish(new Error('bounded Wasm test watchdog expired')), 22_000);
      let cancellationTimer;
      const onMessage = event => {
        if (event.event === 'error') return finish(new Error(event.error));
        if (event.event === 'prepared') {
          prepared = event.value;
          verify(prepared.result, leafHash);
          if (request.cancelAfterMs !== undefined) cancellationTimer = setTimeout(() => {
            cancelSentMs = performance.now() - origin;
            Atomics.store(flag, 0, 1);
          }, request.cancelAfterMs);
        }
        if (event.event === 'progress') {
          verify(event.value.result, leafHash);
          updates.push(event.value);
        }
        if (event.event === 'done') {
          verify(event.value.result, leafHash);
          if (event.value.result.bestMove) assert(event.before.legalMoves.some(move => move.usi === event.value.result.bestMove));
          finish(null, { request, ...event, prepared, updates, hostWallMs: performance.now() - origin,
            cancelSentMs, cancelLatencyMs: cancelSentMs === null ? null : performance.now() - origin - cancelSentMs });
        }
      };
      function finish(error, result) {
        clearTimeout(timeout); clearTimeout(cancellationTimer); worker.off('message', onMessage);
        if (error) reject(error); else { evidence.push(result); resolveRun(result); }
      }
      worker.on('message', onMessage);
      worker.postMessage({ ...request, cancelBuffer });
    });
  }
  try {
    const cancelled = await run({ profile: 'quality', remaining: 600_000, cancelAfterMs: 40 });
    assert.equal(cancelled.value.result.termination, 'cancelled');
    assert(cancelled.probes > 0);
    assert(cancelled.cancelLatencyMs < 250, 'cross-thread cancellation was not promptly observed');
    const expired = await run({ profile: 'quality', remaining: 0 });
    assert.equal(expired.value.result.outcome, 'time_limit_before_evaluation');
    const terminal = await run({ profile: 'balanced', remaining: 180_000, sfen: '4k4/3P1P3/4K4/9/9/9/9/9/9 w - 1' });
    assert.equal(terminal.value.result.outcome, 'no_legal_moves');
    assert.equal(terminal.value.result.bestMove, null);
    const samples = [];
    for (let i = 0; i < 3; i++) {
      const plain = await run({ profile: 'balanced', nodes: 1_500, legacy: true });
      const atomic = await run({ profile: 'balanced', nodes: 1_500 });
      for (const key of ['bestMove', 'scoreCp', 'depth', 'nodes', 'outcome']) assert.deepEqual(atomic.value.result[key], plain.value.result[key]);
      samples.push({ plainMs: plain.wallMs, atomicMs: atomic.wallMs, probes: atomic.probes });
    }
    if (matrix) for (const remaining of [180_000, 600_000]) for (const white of [false, true]) {
      for (const profile of ['balanced', 'quality']) {
        const measured = await run({ profile, remaining, white });
        assert(measured.value.timing.elapsedMs <= measured.value.timing.hardLimitMs + 100);
        assert(measured.wallMs < 10_000, 'ordinary opening response exceeded the preregistered 10s bound');
      }
    }
    const report = { schema: 'open_shogiai_wasm_play_clock_check/v1', status: 'PASS',
      runtime: 'actual-wasm-in-node-worker-threads; not browser UI verification',
      moduleHash: hash(readFileSync(modulePath)), wasmHash: hash(readFileSync(join(dirname(modulePath), 'open_shogi_wasm_bg.wasm'))),
      leafHash, initializationMs, samples, evidence };
    writeFileSync(reportArg, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, initializationMs, samples,
      tests: evidence.map(e => ({ request: e.request, move: e.value.result.bestMove,
        depth: e.value.result.depth, nodes: e.value.result.nodes, wallMs: e.wallMs,
        termination: e.value.result.termination, cancelLatencyMs: e.cancelLatencyMs })) }));
  } finally { await worker.terminate(); }
}
