# Phase 10R-C1 OSAVAL02 Native/Wasm Parity Closure

Date: 2026-08-22 (Asia/Tokyo)
Status: passed; inference parity closed, no candidate promoted.

## Outcome

OSAVAL02 now has a normative closed format, deterministic Python float/int8 export and inspection,
an independent Python reference evaluator, one shared Rust native/Wasm loader and sparse inference
path, and an actual generated-Wasm parity gate. The work covers both frozen eligible architectures
and does not run training, Arena, self-play, or external dataset acquisition.

The 12-fixture corpus is synthetic and label-free. It covers the initial position, both sides to
move, captures, promotions, every drop type, checks, pins, a quiet middlegame, endgame, mate/near
mate, substantial hands, and explicit repetition/continuous-check history. Its SHA-256 is
`5f6b2a218e88da1e6e2106182e3897d5f201643844d8aa861444c4ae39c0e5d0`.

## Matrix result

| Variant | Weights | Bytes | Native/Wasm max abs | Python/native max abs |
| --- | --- | ---: | ---: | ---: |
| sparse pair policy WDL | float32 | 9,657,380 | 8.674e-19 | 5.551e-17 |
| factorized pair/triple policy score | float32 | 10,710,568 | 0 | 5.551e-17 |
| sparse pair policy WDL | int8 | 2,417,417 | 0 | 5.551e-17 |
| factorized pair/triple policy score | int8 | 2,680,714 | 0 | 5.551e-17 |

The acceptance tolerance is relative `1e-10`, absolute `1e-12`. Identities, feature and position
hashes, legal-move membership/order, discrete heads, integer scores, history, and terminal/search
semantics are exact. Each compiled runtime also produced byte-identical JSON on repeated inference.

The frozen fixture-only float/int8 thresholds are probability `1e-5`, policy logit `2e-6`, other
continuous heads `2.5e-5`, and calibrated score `1 cp`, with exact policy ordering. Observed maxima
were `6.928e-6`, `1.384e-6`, `1.983e-5`, and `0 cp`. These are runtime regression thresholds, not a
substitute for the separately required quantized-strength Arena gate.

## Safety and compatibility

All loaders reject corruption, incompatible hashes, invalid dimensions/dtypes/scales, non-finite
parameters, gaps/trailing sections, checksum failure, and oversized input. Native path loading uses
the stable regular-file boundary; Wasm accepts one bounded byte array. Mate-head output is never
converted to a search value, and non-mate values cannot enter the exact mate range.

OSAVAL01 remains supported for historical consumers. It is not accepted as OSAVAL02 and was not
silently migrated. The Phase 10R runner now observes the OSAVAL02 export/native/Wasm backend; it
still blocks campaign execution because the separate training backend and full approved-source
canonical/history leakage scan are outside this closure and remain incomplete.

## Frozen-control changes

The frozen manifest was expanded only for the directly affected OSAVAL02 normative spec/schema,
Python exporter/reference, Rust core and Wasm adapter, generated bindings, source-safe corpus,
parity executables/tests, Make gate, and runner receipt. Each changed frozen digest is recorded in
`configs/phase10r/frozen-controls.sha256`; the reason is either new OSAVAL02 normative/runtime
coverage or the minimal registry/test update needed to verify that coverage. No architecture,
target, normalization, dataset, holdout, Arena, self-play, or resource-budget control changed.

| Frozen path | New SHA-256 | Reason |
| --- | --- | --- |
| `Makefile` | `ebfd99d54226d802e404344dbd31b481209198ef150bad14217ee284acd83c61` | add the focused parity gate |
| `docs/model/OSAVAL02_FORMAT.md` | `cf16777edaf5554cb2d265e7ab8c7b8d081e16d08bbf2cf7c5163f3e680a0ac9` | normative artifact/runtime contract |
| `docs/model/osaval02-parity-corpus.schema.json` | `ab173dde34119eeff848bb2ac74aa661bd7610a1e7bc5027aabff6c536552987` | closed corpus schema |
| `artifacts/phase10r/osaval02-parity-report.json` | `5c73a545a9f0928b869f3c207c1f329e0f1d66dfb890dac85bac261f3e56c676` | freeze machine parity evidence |
| `training/open_shogi_training/phase10r.py` | `5d66e7431018897af670bddf270fa296e2b50fb60b74fcbfa75203d7fc5eddb0` | register directly affected controls |
| `training/open_shogi_training/phase10r_model.py` | `10c88a6df56969bee1c4ce979ff76189178a6379627864e5f44c384e1afe5f32` | Python export/inspect/reference runtime |
| `training/open_shogi_training/phase10r_run.py` | `46a36d090f6621bda3b6ef96da249d8dd49846b3fb42fc93c8b8eb745a0848e5` | observe the completed backend accurately |
| `engine/core/src/lib.rs` | `c97bd259289f11cd52615c59e54cc7326bb8f19369503b2f67d46261ad2b2fb5` | export OSAVAL02 public API |
| `engine/core/src/position.rs` | `db12f86a94b019a822a7fbbfeb4185003f57e2232d3512d3e3a2a824956bbe06` | share exact attack semantics |
| `engine/core/src/phase10r.rs` | `98b6fcc5f344f04e44f86b752453d932fbed5e911a5048c390db0a8f5d88d6d6` | strict Rust loader/features/inference |
| `engine/core/examples/osaval02_infer.rs` | `572213006ddf490994b7f1daa30f8f4f0a484960fec0bfedaece4d1f563ed977` | native parity executable |
| `engine/wasm/src/lib.rs` | `ba187752b07cf6b4fb59dd93be701f0fd0f34b6c7e2a30a78c9259f54bde1066` | Wasm adapter over shared evaluator |
| `scripts/osaval02_wasm_infer.mjs` | `d3041a6a0435af91a0b903563e6e5a4bd1fb1a11186a3a22b34a7b6b5c49eb0a` | actual generated-Wasm gate |
| `tests/fixtures/osaval02/parity-corpus.json` | `5f6b2a218e88da1e6e2106182e3897d5f201643844d8aa861444c4ae39c0e5d0` | source-safe parity corpus |
| `tests/python/models/test_osaval02.py` | `13962438b526e1b6334e4cc99c32fa007382fb9ecd4505c22f45f6c3d1c41ef7` | export/parser negative coverage |
| `tests/python/models/test_osaval02_parity.py` | `2a544015a7236756b2eb7609ba099db5433bf70e16d838cebac33983c71cca42` | four-way matrix parity coverage |
| `tests/python/test_phase10r_run.py` | `2aea855a1f1dcffa6e81b893e898f136563a469ecd27a86131f05e55b767a93e` | backend receipt regression |
| `bindings/wasm/open_shogi_wasm.d.ts` | `8b899de7246585f5c7ccafe0b5b2dde012eec0af12a8032d5d4e5741be9cf063` | deterministic generated API |
| `bindings/wasm/open_shogi_wasm.js` | `8d36745850bb91e90f93436a9c40d833686cca6c1232ab2c7c9346a59e109023` | deterministic generated adapter |
| `bindings/wasm/open_shogi_wasm_bg.wasm` | `f5a973474b1de3eea89578bbaa75ff4a6ad0f64fc23b4e150eb4f823ea18de3a` | deterministic generated runtime |
| `bindings/wasm/open_shogi_wasm_bg.wasm.d.ts` | `efb20b72f02808e77a8baed92fc80fa2f3e1c9b8ce0709f2cd49ed7259933068` | deterministic generated ABI types |

Machine-readable observations and exact deterministic artifact/binding hashes are in
`artifacts/phase10r/osaval02-parity-report.json`.

## Verification

- `make phase10r-osaval02-parity`: passed (Rust core/Wasm suites plus 28 focused Python cases).
- `make check`: passed (516 Python tests, complete Rust workspace tests, lint, formatting,
  repository/license/provenance audits, builds, and deterministic Wasm comparison).
- `scripts/build_wasm_web.sh check`, repeated twice after regeneration: passed.
- Phase 10R frozen validation: passed with 46 bound paths and 33 unchanged approved artifacts.
