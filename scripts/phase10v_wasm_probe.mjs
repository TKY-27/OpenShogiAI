import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const [directory, modelPath, hash, positionFile] = process.argv.slice(2);
const { PureEngine, initSync } = await import(pathToFileURL(resolve(directory, 'open_shogi_wasm.js')));
initSync({ module: readFileSync(resolve(directory, 'open_shogi_wasm_bg.wasm')) });
const bytes = readFileSync(modelPath);
const model = PureEngine.new_with_format(bytes, hash, 'pure_learned', 'OSAVAL03');
try {
  const positions = JSON.parse(readFileSync(positionFile, 'utf8'));
  const results = positions.map(sfen => ({ evaluation: JSON.parse(model.evaluate(sfen)), search: JSON.parse(model.search(sfen, 2, 64)) }));
  let rejected = 0;
  for (const [badBytes, badHash, profile, format] of [
    [new Uint8Array(), hash, 'pure_learned', 'OSAVAL03'],
    [bytes.subarray(0, bytes.length - 1), hash, 'pure_learned', 'OSAVAL03'],
    [bytes, '0'.repeat(64), 'pure_learned', 'OSAVAL03'],
    [bytes, hash, 'standard', 'OSAVAL03'],
    [bytes, hash, 'pure_learned', 'OSAVAL02'],
    [bytes, hash, 'pure_learned', 'auto'],
  ]) {
    try { const bad = PureEngine.new_with_format(badBytes, badHash, profile, format); bad.free(); }
    catch { rejected++; }
  }
  if (rejected !== 6) throw new Error('fail-closed Wasm model gate failed');
  process.stdout.write(JSON.stringify({results, rejected}) + '\n');
} finally { model.free(); }
