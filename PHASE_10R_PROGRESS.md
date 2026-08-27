# Phase 10R progress

Updated: 2026-08-27 (Asia/Tokyo)

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

Production training has not started. No teacher labeling, Arena, cross-play, self-play, final
holdout evaluation, promotion, push, release, or deployment was performed.

## Preparation evidence

- Legacy path: `local/phase10r-data/phase10r-prepared/1m/`
- Legacy manifest file SHA-256:
  `298654d99e04383a8a63ccca837ff9095ecba45e2bb7aadf747262c8fe7d04c5`
- Legacy train SHA-256: `5ab54aaa8fc59c5083f4037e1c9cb4b811164f09c9da225d0eda413d837459c0`
- Replacement path: `local/phase10r-data/phase10r-prepared/1m-mixture-v2/`
- Replacement manifest file SHA-256:
  `dcbcfd7584f84bd03d9d948a5633e9b18bb3db06027d1f1a0bec2b2cd7a8e076`
- Replacement train SHA-256:
  `8748f5fdf8af7716a347f1faa94d0f4965afca009f368ce1a66463e0e917b161`
- Target and realized counts: AobaZero 350,000; WCSC 450,000; Denryu 200,000.
- Eligible-train epoch equivalents: AobaZero 44.7284; WCSC 1.2633; Denryu 17.4429.
- Maximum record occurrences: AobaZero 45; WCSC 2; Denryu 18.
- Same-seed reproduction: exact 1,000,000-row/2,077,794,499-byte train hash match.
- Effective cross-split groups: 0; protected public/internal/final holdouts excluded.
- Peak preparation RSS: 238,813,184 bytes; free disk after preparation: 283,567,955,968 bytes.

## Durable artifacts

- Canonical control: `configs/phase10r/dataset-mixture.yaml`
- Human reconciliation: `PHASE_10R_MIXTURE_RECONCILIATION_REPORT.md`
- Machine reconciliation: `artifacts/phase10r/phase10r-mixture-reconciliation.json`
- Existing replay/leakage proof: `artifacts/phase10r/phase10r-scan-completion-proof.json`
- Local immutable preparation receipt:
  `local/phase10r-runs/20260827T094227.338945Z-prepare.json`

## Verification state

- Focused mixture/execution/freeze/runner tests: passed (`25` tests).
- Canonical configuration and pipeline sanity without stale hashes: passed.
- Versioned preparation self-validation: passed.
- Frozen-hash validation: passed (`60` paths).
- Leakage validation: passed (`528,570` effective records; zero effective cross-split groups).
- Full `make check`: passed (`544` Python tests, `344` Rust tests, lint, boundary, license,
  provenance, build, and deterministic Wasm binding check).
- Exact Phase 10R preflight: passed; no failures; preparation control, same-seed proof, leakage,
  OSAVAL02 bounded backend probe, disk, memory, and thermal controls all passed.

## Next boundary

Once final validation and local commits are complete, production training is technically unblocked
at the preparation boundary. Training remains intentionally stopped until the existing Luna Goal is
explicitly resumed. The first resumed action must re-run exact preflight against the committed clean
tree before invoking any `train` command.
