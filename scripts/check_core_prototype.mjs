#!/usr/bin/env node
// Actual Wasm control/ablation contract, with explicit local model identities.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {resolve, dirname, join} from 'node:path';
import {pathToFileURL} from 'node:url';

const [moduleArg, leafArg, leafHash, controllerArg, controllerHash, reportArg] = process.argv.slice(2);
assert(reportArg, 'usage: check_core_prototype.mjs MODULE LEAF SHA CONTROLLER SHA REPORT');
const modulePath = resolve(moduleArg);
const hash = b => createHash('sha256').update(b).digest('hex');
const leaf = readFileSync(leafArg), controller = readFileSync(controllerArg);
assert.equal(hash(leaf), leafHash); assert.equal(hash(controller), controllerHash);
const bytes = readFileSync(join(dirname(modulePath), 'open_shogi_wasm_bg.wasm'));
const wasm = await import(pathToFileURL(modulePath));
wasm.initSync({module:bytes});
const engine = new wasm.WasmBrowserEngine();
const checks = [], searches = [];
assert.throws(() => engine.setComputeEnabled(true));
engine.loadModel(leaf, leafHash);
const identity = JSON.parse(engine.loadComputeModel(controller, controllerHash));
assert.deepEqual(identity, {schema:'open_shogiai_computation_identity/v1',
  artifactSha256:controllerHash,leafModelSha256:leafHash,expectedHashVerified:true});
checks.push('strict-controller-and-leaf-identities');
for (const enabled of [false,true]) {
  engine.reset(); engine.setComputeEnabled(enabled);
  const before=JSON.parse(engine.snapshot());
  const r=JSON.parse(engine.searchWithTimeControl('eco','pure_learned',1,JSON.stringify({
    schema:'open_shogi_time_control/v1',nodes:1500,depth:4,
  })));
  assert(before.legalMoves.some(m=>m.usi===r.bestMove));
  assert.equal(r.outcome,'evaluated');
  assert(r.runtimeProof.learned_eval_calls > 0);
  for(const field of ['handcrafted_eval_calls','residual_eval_calls','composite_eval_calls',
    'book_hits','teacher_calls','fallback_count']) assert.equal(r.runtimeProof[field],0);
  assert.equal(r.computeControl.modelSha256,controllerHash);
  assert.equal(r.computeControl.enabled,enabled);
  if(enabled) assert(r.computeControl.decisions>0 && r.computeControl.reorderedMoves>0);
  else assert.equal(r.computeControl.decisions,0);
  engine.playMove(r.bestMove);
  assert.equal(JSON.parse(engine.snapshot()).sideToMove,'white');
  searches.push(r);
}
checks.push('actual-wasm-on-off-legal-root-order-and-proof');
// A short real clock request tests both side selection and advisory target bounds.
engine.setComputeEnabled(true);
const clock=JSON.parse(engine.searchWithTimeControl('eco','pure_learned',1,JSON.stringify({
  schema:'open_shogi_time_control/v1',blackTimeMs:9000,whiteTimeMs:1500,
  blackIncrementMs:200,whiteIncrementMs:0,byoyomiMs:0,safetyMarginMs:50,
})));
assert.equal(clock.perspective,'white'); assert.equal(clock.timeControlMode,'clock');
assert(clock.computeControl.targetMs <= 100); // white: 1500/30*3 - 50
assert(clock.elapsedNs/1e6 < 200);
searches.push(clock); checks.push('clock-side-increment-and-independent-ceiling');
engine.restore('4k4/3P1P3/4K4/9/9/9/9/9/9 w - 1','[]');
assert.equal(JSON.parse(engine.snapshot()).terminal.kind,'no-legal-moves');
assert.throws(()=>engine.search('eco','pure_learned',1));
checks.push('no-legal-moves-terminal-without-search');
assert.throws(()=>engine.loadComputeModel(controller,'0'.repeat(64)));
assert.throws(()=>engine.setComputeEnabled(true));
engine.loadComputeModel(controller,controllerHash);
assert.throws(()=>engine.loadModel(leaf,'0'.repeat(64)));
assert.throws(()=>engine.setComputeEnabled(true));
checks.push('failed-replacement-revokes-policy');
engine.free();
const report={schema:'open_shogiai_core_wasm_contract/v1',status:'PASS',
  runtime:'actual-wasm-in-node',leafHash,controllerHash,moduleHash:hash(readFileSync(modulePath)),
  wasmHash:hash(bytes),checks,searches};
writeFileSync(reportArg,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({status:report.status,checks,searches:searches.map(r=>({
  bestMove:r.bestMove,depth:r.depth,nodes:r.nodes,elapsedMs:r.elapsedNs/1e6,compute:r.computeControl
}))}));
