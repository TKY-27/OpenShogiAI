#!/usr/bin/env node
// Post-export parity audit. Uses only explicit local artifacts and development positions.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, realpathSync, statSync, writeFileSync} from 'node:fs';
import {basename, dirname, join, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {performance} from 'node:perf_hooks';

const [moduleArg, modelArg, expectedHash, probeArg, reportArg, positionsArg, ...extra] = process.argv.slice(2);
assert(reportArg && extra.length === 0,
  'usage: check_evaluator_model.mjs MODULE MODEL SHA256 PROBE REPORT [DEVELOPMENT_POSITIONS_JSON]');
assert.match(expectedHash, /^[a-f0-9]{64}$/, 'expected model SHA-256 is required');
const scriptPath = fileURLToPath(import.meta.url);
const root = realpathSync(resolve(dirname(scriptPath), '..'));
const reportPath = resolve(reportArg);
assert(reportPath.startsWith(`${join(root, 'local')}${sep}`), 'report must be inside repository local/');
assert(!existsSync(reportPath), 'report already exists; choose a new immutable report path');
let existingParent = dirname(reportPath);
while (!existsSync(existingParent)) existingParent = dirname(existingParent);
assert(realpathSync(existingParent) === join(root, 'local')
  || realpathSync(existingParent).startsWith(`${join(root, 'local')}${sep}`), 'report ancestor escapes local/');
mkdirSync(dirname(reportPath), {recursive: true});
assert(realpathSync(dirname(reportPath)).startsWith(`${join(root, 'local')}${sep}`)
  || realpathSync(dirname(reportPath)) === join(root, 'local'), 'report directory escapes local/');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = {
  schema: 'open_shogiai_evaluator_export_audit/v1', status: 'RUNNING',
  runtime: 'actual-wasm-in-node', format: 'OSAVAL03', profile: 'pure_learned',
  positionsPurpose: 'development-regression-only', artifacts: {}, checks: [],
  parity: {roots: 0, children: 0, uniquePositions: 0, errors: 0, maximumCpDifference: 0,
    maximumWdlDifference: 0, incrementalFullEqual: true}, positions: [], searches: [],
  timing: {}, memory: [], errors: [],
  limitations: [
    'These development positions do not measure unseen-data strength or browser behavior.',
    'Full-evaluation timings include SFEN parsing, accumulator refresh, head inference and JSON/FFI overhead.',
    'Cached-head-only inference cost is not measured by the existing exported API.',
    'Native batch timing also includes process/model startup, successor generation and accumulator audits.',
    'Wasm linear memory is allocated capacity, not live model heap or browser-wide memory.',
  ],
};
let pure = null;
let browser = null;
const files = [];
function artifact(label, path, maximum) {
  const resolved = realpathSync(resolve(path));
  const info = statSync(resolved);
  assert(info.isFile() && info.size > 0 && info.size <= maximum, `${label}: invalid file size/type`);
  const bytes = readFileSync(resolved);
  assert.equal(bytes.length, info.size, `${label}: file changed during read`);
  const sha256 = digest(bytes);
  report.artifacts[label] = {name: basename(resolved), sha256, bytes: bytes.length};
  files.push({label, path: resolved, sha256});
  return {path: resolved, bytes};
}
function cp(value, label) {
  assert(Number.isSafeInteger(value) && Math.abs(value) <= 20_000, `${label}: invalid learned cp`);
  return value;
}
function sfen(value) {
  assert(typeof value === 'string' && value.length <= 512 && /^[a-zA-Z0-9/+* -]+$/.test(value), 'invalid SFEN text');
  const fields = value.split(' ');
  assert(fields.length === 4 && ['b', 'w'].includes(fields[1]) && /^[1-9][0-9]*$/.test(fields[3]), 'invalid SFEN fields');
  return value;
}
function usi(value) {
  assert(typeof value === 'string' && /^(?:[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i])$/.test(value), 'invalid legal move');
  return value;
}
function logits(value) {
  assert(Array.isArray(value) && value.length === 3 && value.every(Number.isFinite), 'invalid WDL logits');
  return value;
}
function evaluation(value) {
  assert.equal(value.schema, 'open_shogiai_phase10t_pure_runtime/v1');
  assert.equal(value.profile, 'pure_learned');
  assert.equal(value.model_format, 'OSAVAL03');
  assert.equal(value.model_sha256, expectedHash);
  assert.deepEqual(value.compiled_evaluators, ['osaval02', 'phase10t-a1', 'phase10v']);
  cp(value.cp, 'Wasm');
  logits(value.wdl_logits);
  return value;
}
function proof(value) {
  assert.equal(value.profile, 'pure_learned');
  assert.equal(value.profile_schema, 'open_shogiai_pure_learned_v3_profile/v1');
  assert.equal(value.model_sha256, expectedHash);
  assert.equal(value.evaluator_profile_schema_hash,
    'd2eec27887926ccc8a076552815cd54e34b85d6d23e65732ddba4989bf59c1e7');
  assert(Number.isSafeInteger(value.learned_eval_calls) && value.learned_eval_calls > 0, 'search lacks learned inference');
  for (const key of ['handcrafted_eval_calls', 'residual_eval_calls', 'composite_eval_calls',
    'book_hits', 'teacher_calls', 'fallback_count']) assert.equal(value[key], 0, `forbidden runtime counter: ${key}`);
}
const initial = 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL';
const positions = [
  ['initial-sente', `${initial} b - 1`],
  ['initial-gote', `${initial} w - 1`],
  ['sente-king-moved', 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B1K3R1/LNSG1GSNL w - 2'],
  ['gote-king-moved', 'lnsg1gsnl/1r1k3b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 2'],
  ['sente-pawn-hand', '4k4/9/9/9/9/9/9/9/4K4 b P 1'],
  ['gote-pawn-hand', '4k4/9/9/9/9/9/9/9/4K4 w p 1'],
  ['sente-promotion', '4k4/9/4P4/9/9/9/9/9/4K4 b - 1'],
  ['gote-promotion', '4k4/9/9/9/9/9/4p4/9/4K4 w - 1'],
  ['sente-capture-promotion', '4k4/9/4p4/4P4/9/9/9/9/4K4 b - 1'],
  ['gote-capture-promotion', '4k4/9/9/9/9/4p4/4P4/9/4K4 w - 1'],
].map(([id, position]) => ({id, sfen: position, source: 'built-in-diagnostic'}));

try {
  artifact('auditScript', scriptPath, 128 * 1024);
  const moduleFile = artifact('moduleJs', moduleArg, 2 * 1024 * 1024);
  const wasmFile = artifact('wasm', join(dirname(moduleFile.path), 'open_shogi_wasm_bg.wasm'), 32 * 1024 * 1024);
  const modelFile = artifact('model', modelArg, 64 * 1024 * 1024);
  const probeFile = artifact('nativeProbe', probeArg, 32 * 1024 * 1024);
  assert.equal(digest(modelFile.bytes), expectedHash, 'model SHA-256 mismatch');
  assert.equal(modelFile.bytes.subarray(0, 8).toString('ascii'), 'OSAVAL03');
  if (positionsArg) {
    // Final holdout selection is never part of this development command.
    assert(!/(?:holdout|sealed)/i.test(realpathSync(resolve(positionsArg))), 'sealed/holdout input is forbidden');
    const input = artifact('developmentPositions', positionsArg, 1024 * 1024);
    const value = JSON.parse(input.bytes.toString('utf8'));
    assert.deepEqual(Object.keys(value).sort(), ['positions', 'purpose', 'schema']);
    assert.equal(value.schema, 'open_shogiai_evaluator_audit_positions/v1');
    assert.equal(value.purpose, 'development');
    assert(Array.isArray(value.positions) && value.positions.length >= 1 && value.positions.length <= 128);
    for (const position of value.positions) {
      assert.deepEqual(Object.keys(position).sort(), ['id', 'sfen']);
      assert(typeof position.id === 'string' && /^[a-zA-Z0-9._-]{1,96}$/.test(position.id), 'invalid diagnostic id');
      positions.push({id: `data-${position.id}`, sfen: sfen(position.sfen), source: 'explicit-development-input'});
    }
  }
  for (const position of positions) sfen(position.sfen);
  assert.equal(new Set(positions.map(position => position.id)).size, positions.length, 'duplicate position IDs');
  const nativeStart = performance.now();
  const native = spawnSync(probeFile.path, [modelFile.path, expectedHash], {
    input: positions.map(position => JSON.stringify({reset: position.sfen, successors: true})).join('\n') + '\n',
    encoding: 'utf8', timeout: 120_000, maxBuffer: 32 * 1024 * 1024,
  });
  report.timing.nativeAuditBatchMs = performance.now() - nativeStart;
  assert(!native.error && native.status === 0 && native.signal === null,
    `native audit failed: ${native.error?.message ?? native.stderr?.slice(0, 512) ?? native.status}`);
  const rows = native.stdout.trim().split('\n').map(line => JSON.parse(line));
  assert.equal(rows.length, positions.length, 'native response count mismatch');
  const nativeWrongHash = spawnSync(probeFile.path, [modelFile.path, '0'.repeat(64)], {
    input: '', encoding: 'utf8', timeout: 5000, maxBuffer: 16_384,
  });
  assert(!nativeWrongHash.error && nativeWrongHash.status !== 0, 'native accepted wrong model hash');
  report.checks.push('native-model-hash-and-incremental-full-audit');
  const compileStart = performance.now();
  const wasm = await import(pathToFileURL(moduleFile.path).href);
  const exports = wasm.initSync({module: wasmFile.bytes});
  assert(exports.memory instanceof WebAssembly.Memory, 'Wasm memory export unavailable');
  report.timing.wasmModuleAndCompileMs = performance.now() - compileStart;
  const memory = stage => report.memory.push({stage, wasmLinearBytes: exports.memory.buffer.byteLength,
    nodeRssBytes: process.memoryUsage().rss});
  memory('compiled');
  const loadStart = performance.now();
  pure = wasm.PureEngine.new_with_format(modelFile.bytes, expectedHash, 'pure_learned', 'OSAVAL03');
  report.timing.wasmModelLoadMs = performance.now() - loadStart;
  memory('one-pure-model-loaded');
  assert.throws(() => wasm.PureEngine.new_with_format(modelFile.bytes, '0'.repeat(64), 'pure_learned', 'OSAVAL03'));
  assert.throws(() => wasm.PureEngine.new_with_format(modelFile.bytes, expectedHash, 'pure_learned', 'OSAT10A1'));
  assert.throws(() => wasm.PureEngine.new_with_format(modelFile.bytes, expectedHash, 'handcrafted', 'OSAVAL03'));
  assert.throws(() => pure.evaluate('not a valid sfen'));
  assert.throws(() => pure.search(positions[0].sfen, 0, 256));
  assert.throws(() => pure.search(positions[0].sfen, 3, 0));
  report.checks.push('wasm-explicit-format-profile-hash-and-invalid-input-rejection');
  browser = new wasm.WasmBrowserEngine();
  const identity = JSON.parse(browser.loadModel(modelFile.bytes, expectedHash));
  assert.equal(identity.artifactSha256, expectedHash);
  assert.equal(identity.modelFormat, 'OSAVAL03');
  assert.equal(identity.expectedHashVerified, true);
  assert.equal(identity.buildClass, 'pure-only');
  assert.equal(identity.evaluationMode, 'pure-value');
  const observed = new Set();
  const compare = (position, expectedCp, expectedWdl = null) => {
    const value = evaluation(JSON.parse(pure.evaluate(position)));
    const difference = Math.abs(value.cp - cp(expectedCp, 'native'));
    report.parity.maximumCpDifference = Math.max(report.parity.maximumCpDifference, difference);
    assert.equal(value.cp, expectedCp, 'native/Wasm rounded learned cp mismatch');
    if (expectedWdl !== null) {
      logits(expectedWdl);
      for (let index = 0; index < 3; index += 1) {
        const delta = Math.abs(value.wdl_logits[index] - expectedWdl[index]);
        report.parity.maximumWdlDifference = Math.max(report.parity.maximumWdlDifference, delta);
        assert(delta <= 1e-6 * Math.max(1, Math.abs(expectedWdl[index])), 'native/Wasm WDL mismatch');
      }
    }
    observed.add(position);
    return value;
  };
  const parityStart = performance.now();
  for (let index = 0; index < positions.length; index += 1) {
    const fixture = positions[index];
    const row = rows[index];
    assert.equal(row.sfen, fixture.sfen, 'native returned another position');
    assert.equal(row.model_sha256, expectedHash);
    assert.equal(row.incremental_full_equal, true);
    assert(Array.isArray(row.successors) && row.successors.length <= 1024);
    assert.equal(row.audited_successors, row.successors.length);
    const snapshot = JSON.parse(browser.restore(fixture.sfen, '[]'));
    assert.equal(snapshot.sfen, fixture.sfen);
    assert.equal(snapshot.evaluator.model.artifactSha256, expectedHash);
    assert.equal(snapshot.openingBook, null);
    assert.equal(snapshot.openingPolicy.profile, 'disabled');
    const legal = snapshot.legalMoves.map(move => usi(move.usi)).sort();
    assert.equal(new Set(legal).size, legal.length);
    assert.deepEqual(row.successors.map(child => usi(child.move)).sort(), legal, 'native/Wasm legal moves mismatch');
    const rootEval = compare(fixture.sfen, row.cp);
    report.parity.roots += 1;
    for (const child of row.successors) {
      sfen(child.sfen);
      assert.equal(child.sfen.split(' ')[1], fixture.sfen.split(' ')[1] === 'b' ? 'w' : 'b', 'child turn did not change');
      compare(child.sfen, child.child_cp, child.child_wdl_logits);
      report.parity.children += 1;
    }
    report.positions.push({...fixture, cp: rootEval.cp, children: row.successors.length});
  }
  report.parity.uniquePositions = observed.size;
  report.timing.wasmParityFullEvaluationsMs = performance.now() - parityStart;
  report.checks.push('exact-cp-parity-all-legal-children-kings-captures-promotions-drops', 'legal-set-and-side-to-move-parity');
  // A bounded repeated public API timing: this deliberately is not called a cached-head benchmark.
  const evaluationRepeats = 128;
  const evaluateStart = performance.now();
  for (let index = 0; index < evaluationRepeats; index += 1)
    evaluation(JSON.parse(pure.evaluate(positions[index % 10].sfen)));
  const evaluationMs = performance.now() - evaluateStart;
  report.timing.fullEvaluation = {calls: evaluationRepeats, totalMs: evaluationMs, meanMs: evaluationMs / evaluationRepeats};
  report.timing.cachedHead = {measured: false, reason: 'existing public evaluate API refreshes the accumulator'};
  memory('after-full-evaluation-audit-two-model-instances');
  for (const fixture of positions.slice(0, 2)) {
    const snapshot = JSON.parse(browser.restore(fixture.sfen, '[]'));
    const started = performance.now();
    const result = JSON.parse(browser.searchWithTimeControl('eco', 'pure_learned', 1,
      JSON.stringify({schema: 'open_shogi_time_control/v1', nodes: 256, depth: 3})));
    const wallMs = performance.now() - started;
    assert.equal(result.outcome, 'evaluated');
    assert.equal(result.perspective, snapshot.sideToMove);
    assert(snapshot.legalMoves.some(move => move.usi === result.bestMove), 'search chose an illegal move');
    assert(Number.isSafeInteger(result.nodes) && result.nodes > 0 && result.nodes <= 256);
    assert(Number.isSafeInteger(result.depth) && result.depth >= 0 && result.depth <= 3);
    proof(result.runtimeProof);
    assert.equal(result.runtimeProof.learned_eval_calls, result.stats.learnedEvalCalls);
    for (const field of ['handcraftedEvalCalls', 'residualEvalCalls', 'compositeEvalCalls', 'fallbackCount', 'osaval02InferenceErrors'])
      assert.equal(result.stats[field], 0);
    report.searches.push({id: fixture.id, wallMs, result});
  }
  report.checks.push('bounded-actual-wasm-search-legal-moves-positive-pure-proof-both-sides');
  memory('after-fixed-node-search');
  for (const file of files) assert.equal(digest(readFileSync(file.path)), file.sha256, `${file.label} changed during audit`);
  report.checks.push('all-input-identities-unchanged-after-audit');
  report.status = 'PASS';
} catch (error) {
  report.status = 'FAIL';
  report.parity.errors += 1;
  report.errors.push({name: error?.name ?? 'Error', message: String(error?.message ?? error).slice(0, 2048)});
  process.exitCode = 1;
} finally {
  pure?.free();
  browser?.free();
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, {flag: 'wx'});
  console.log(JSON.stringify({status: report.status, report: reportPath, modelSha256: expectedHash,
    parity: report.parity, fullEvaluation: report.timing.fullEvaluation, errors: report.errors}));
}
