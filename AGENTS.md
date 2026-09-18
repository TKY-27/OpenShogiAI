# OpenShogiAI working rules

- Start with `docs/status.md`, inspect Git status, and preserve existing user edits.
- Work inside this Git root. OpenShogiUI is a separate repository; do not modify it without
  task-specific authorization. Do not change global tools/settings or shared caches.
- Use `make check` for locked dependencies, boundary/license/provenance checks, Rust/Python
  format and lint, tests, native build and deterministic Wasm regeneration.
  `make pure-build` and `make frozen-smoke` verify the optional local OSAVAL03 baseline.
- Preserve legal-move, native/Wasm parity, model-input validation, rights and split-leakage
  checks. Never inspect or repartition final holdout contents during routine development.
- Generated data, models, logs and receipts belong in ignored `local/`; Cargo output belongs
  in `target/`. Never commit weights, teacher assets, secrets or machine-specific paths.
- Keep one comparison model plus only justified test/reproduction exceptions under
  `local/frozen/manifest.json`. Keep essential rights/partition metadata and unique sealed
  evaluation assets. Old campaigns do not require permanent retention of every artifact.
- Before deleting, verify actual paths, symlinks/mounts, references and open files; stay inside
  the repository and record paths/reasons/approximate sizes locally. Do not use repository-wide
  `git clean -xfd`, delete `.git`, rewrite history or delete unknown user files.
- Large training, release, OSUI integration and default promotion require explicit task scope.
  Closed campaign validation fails closed; historical controls are regression fixtures only.

- Development matches, strength evaluation and future production are always book-free.
  No fixed first moves, position-to-move tables, preloaded analysis or online teacher calls.
  Offline teacher/scenario learning is allowed; normal search transposition tables remain allowed.
  Pure learned play never mixes handcrafted nonterminal evaluation scores.
- Astra owns strength diagnosis/design, engine/teacher/training settings, adoption review and OSUI
  integration. A separately user-started Luna Max/max session executes the sealed finite contract.
  Do not spawn Luna as a child or use Astra for long monitoring. Stop preparation at
  `ready_for_luna`; return completed runs at `awaiting_astra_review` without retraining.
- `docs/status.md` links the sole current execution contract. Operational results may change;
  sealed experiment/source hashes may not. Reviewed startup-only operational revisions may be
  recorded separately by the formal resume command in the same run, preserving the original
  seal, failure and cumulative budgets; Luna may execute that verified recovery without reapproval.
  Learning approval does not authorize model publication,
  public-default promotion, main merge, deployment or paid compute.
- A single incomplete fixed-depth teacher label, including a root, is a local deferred task
  under the current recovery contract, never a fabricated label or automatic whole-run failure.
  Resume preserves cumulative attempts and lineage. Training requires the sealed coverage
  manifest; exhausted cohort budgets, systemic faults, integrity and resource failures remain
  stage stop conditions. A new acceptance policy requires Astra and a successor contract.
- For `defense-20260912-recovery-r3`, the explicit 2026-09-17 authorization supersedes
  the old whole-run focus ceilings and generation-exhaustion handoff. The hash-bound
  `itemwise-focus-v1` dataset admission in the existing run preserves the original seal
  and failures, closes generation/hard reanalysis, and admits valid independent targets
  to prepare/train. Missing focus is reported and excluded itemwise, not fabricated.
  Host memory pressure/swap alone is diagnostic; confirmed allocation failures use the
  finite fallback/resource-wait path. See `docs/status.md` for the sole current handoff.
- The approved post-training continuation for the same run is `optional-screen-v1`:
  preserve completed generation/prepare/training and the checkpoint-bound best1536.
  Optional teacher screens never block independent arena or development OSUI loading;
  missing/not-run screens remain unverified, not passing. Session 2 executes only the
  fixed evaluation remainder (plan A) in `docs/status.md` and `configs/evaluator-main.json`.
  It does not retrain, publish, promote, merge, or open the final holdout.
- The explicit 2026-09-18 R4-C1 authorization starts one new evaluator continuation round
  after the old Session 2 completed. The sole current handoff is now R4-C1 in docs/status.md;
  the old Session 2 restrictions above apply to the old defense run, not this new round.
  Luna resumes the preserved initial updates in local/runs/r4-c1/attempt-01, never step zero.
  Missing optional screens do not gate independent completion. Stop at awaiting_astra_review.
  C2/C3, publication, weight distribution, public-default promotion and main integration
  still require the user's separate explicit instruction.
