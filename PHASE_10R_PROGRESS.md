# Phase 10R progress

Updated: 2026-08-29 (Asia/Tokyo)

## Current state

The replay/identity repair, complete approved-source leakage proof, OSAVAL02 backend, and exact
preflight prerequisites are present. The original 1M preparation exposed one frozen-control defect:
relative weights produced AobaZero/WCSC/Denryu `40%/40%/20%`, exceeding the explicit AobaZero 35%
maximum.

The user-authorized narrow reconciliation is implemented. The sole active pretraining mixture is
now `35%/45%/20%`; the invalid legacy preparation is preserved and marked
`REJECTED_MIXTURE_CONTROL_CONFLICT`; and a distinct `1m-mixture-v2` preparation contains exactly
1,000,000 examples with validated counts, hashes, repetition pressure, leakage, determinism, disk,
and RSS evidence.

Both exact 1M training rungs completed from `1m-mixture-v2`. Candidate evaluation originally
stopped fail-closed because the native and Wasm report wrappers exposed different `root.schema`
identities. The wrapper contract now has one canonical versioned identity,
`open_shogiai_osaval02_parity/v1`, and both existing candidates pass the unchanged OSAVAL02
numerical tolerances across Python, native, and actual Wasm execution.

No retraining, teacher labeling, hard-position selection, Arena, cross-play, self-play, final
holdout inspection, promotion, push, release, or deployment was performed while closing this gate.

## Preparation evidence

- Legacy path: `local/phase10r-data/phase10r-prepared/1m/`
- Legacy manifest file SHA-256:
  `298654d99e04383a8a63ccca837ff9095ecba45e2bb7aadf747262c8fe7d04c5`
- Legacy train SHA-256: `5ab54aaa8fc59c5083f4037e1c9cb4b811164f09c9da225d0eda413d837459c0`
- Replacement path: `local/phase10r-data/phase10r-prepared/1m-mixture-v2/`
- Replacement manifest file SHA-256:
  `9dddab33fa50ec070f267f43ea12ebab1c1931e16c8826219eb50b85d8061d2b`
- Replacement train SHA-256:
  `8748f5fdf8af7716a347f1faa94d0f4965afca009f368ce1a66463e0e917b161`
- Target and realized counts: AobaZero 350,000; WCSC 450,000; Denryu 200,000.
- Eligible-train epoch equivalents: AobaZero 44.7284; WCSC 1.2633; Denryu 17.4429.
- Maximum record occurrences: AobaZero 45; WCSC 2; Denryu 18.
- Same-seed reproduction: exact 1,000,000-row/2,077,794,499-byte train hash match.
- Effective cross-split groups: 0; protected public/internal/final holdouts excluded.
- Peak preparation RSS: 213,958,656 bytes; free disk after preparation: 278,063,026,176 bytes.

## Durable artifacts

- Canonical control: `configs/phase10r/dataset-mixture.yaml`
- Human reconciliation: `PHASE_10R_MIXTURE_RECONCILIATION_REPORT.md`
- Machine reconciliation: `artifacts/phase10r/phase10r-mixture-reconciliation.json`
- Existing replay/leakage proof: `artifacts/phase10r/phase10r-scan-completion-proof.json`
- Local immutable preparation receipt:
  `local/phase10r-runs/20260827T100448.829117Z-prepare.json`

## Candidate evaluation evidence

- Evaluation receipt: `local/phase10r-runs/20260829T092040.933643Z-evaluate.json`
- Receipt SHA-256: `81cadb096d6f4bc36635d29689391a538ef7c167381c0932a5cf98e7ebec66e4`
- Evaluation output: `local/phase10r-data/evaluations/1m/evaluation.json`
- Evaluation output SHA-256:
  `91eb25990541165c1408c18f44cd18b57c457e056eba1040422cf58650e8d995`
- Preparation manifest body SHA-256:
  `3e36bf596ddb25356abad276745c3b2b2cae2bf01a38894f65d1a362b6481ecf`
- `sparse-pair-policy-wdl` artifact SHA-256:
  `6b7f2f4dc0bb013992460cf475f5e2e1f220da7fb09542175be367821f296f63`
- `sparse-pair-policy-wdl` native/Wasm maximum absolute delta:
  `2.220446049250313e-16` (`12` fixtures; passed).
- `factorized-pair-triple-policy-score` artifact SHA-256:
  `d2c9af492cb67be35433662b75673efa2c6fece3f0ba2457e281185469364453`
- `factorized-pair-triple-policy-score` native/Wasm maximum absolute delta:
  `4.547473508864641e-13` (`12` fixtures; passed).
- Incremental native/Wasm parity suite: passed.
- Candidate evaluation status: passed.
- Expansion gate: `blocked_until_teacher_stages_and_arena`.

## Verification state

- Focused mixture/execution/freeze/runner tests: passed (`26` tests).
- Canonical configuration and pipeline sanity without stale hashes: passed.
- Versioned preparation self-validation: passed.
- Frozen-hash validation: passed (`62` paths).
- Leakage validation: passed (`528,570` effective records; zero effective cross-split groups).
- Full `make check`: passed (`556` Python tests, `344` Rust tests, lint, boundary, license,
  provenance, build, and deterministic Wasm binding check).
- Exact Phase 10R preflight: passed; no failures; preparation control, same-seed proof, leakage,
  OSAVAL02 bounded backend probe, disk, memory, and thermal controls all passed.
- Focused OSAVAL02 schema/parity suite: passed, including both completed 1M candidates, identical
  identity fields, missing/stale/unsupported schema rejection, real mismatch detection, and
  deterministic Wasm regeneration.
- Exact existing-candidate evaluation with all-source-held-out, cross-runtime, and incremental
  parity gates: passed.

## Next boundary

The existing 1M candidates have passed the candidate evaluation gate. Execution stops here. Teacher
dependent stages are `not_started`, and the broader expansion gate remains fail-closed until those
stages and Arena are completed under a separately resumed goal. Do not re-run training or candidate
evaluation when resuming from this boundary.
