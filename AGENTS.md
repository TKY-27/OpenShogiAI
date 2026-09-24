# OpenShogiAI working rules

- `make check` is the full verification gate (dependencies, boundaries, license/provenance,
  format, lint, tests, native build, deterministic Wasm regeneration). Keep it green.
- Generated data, models, logs, checkpoints and receipts belong in ignored `local/`; Cargo
  output belongs in `target/`. Never commit weights, teacher assets, secrets or
  machine-specific paths. The final holdout stays sealed.
- Play and analysis are always book-free: no fixed first moves, position-to-move tables,
  preloaded analysis or online teacher calls in any runtime. Pure learned play never mixes
  handcrafted nonterminal evaluation. Offline teacher labeling for training is allowed.
- Preserve legal-move, native/Wasm parity, model-input validation, rights and
  split-leakage checks. Model publication flows through
  `configs/models/distribution.json` (rights and hashes) — see
  `docs/model/distribution.md`; never republish an unreviewed checkpoint.
- OpenShogiUI is a separate repository; coordinate API/binding changes with it and verify
  with its `npm run check` when the shared Wasm contract changes.
- Public claims stay factual: no strength ratings, tournament records or dan ranks that
  were not measured; development order is not a strength ranking.
