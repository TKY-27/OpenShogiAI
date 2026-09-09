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
