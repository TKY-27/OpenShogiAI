# Development

Run commands from the repository root. Setup is in [README](../README.md).

| Command | Purpose |
| --- | --- |
| `make check` | Locked environment, boundaries, licenses, provenance, links, format/lint, tests and builds |
| `make test` | Rust and Python regression suites, using fixtures |
| `make build` | Native workspace and byte-for-byte development Wasm regeneration |
| `make pure-build` | Native and Wasm with `pure-only`, under `target/pure/` |
| `make frozen-smoke` | Explicit local baseline hash/load/native/Wasm smoke; no Arena or training |
| `make format` / `make lint` | Rustfmt, Clippy and Ruff |
| `make phase3-validate-registry` | Validate source-rights catalog without acquisition |
| `make model-validate` | Validate legacy model/feature/training schemas |

Python modules under `training/open_shogi_training/` retain model, data, teacher and evidence
primitives with their regression tests. Historical `phase*` control files are compatibility
fixtures, not current execution plans. The completed campaign freeze entry points reject
execution; old documents and all-receipt preservation checks are not development gates.
Old campaign snapshots can be studied through Git history when specifically needed.
Optional model comparison commands require explicit model/data inputs; champion-gate commands
require an explicit config. Historical candidate paths are unavailable, not fallback defaults.

## Local storage

Use one generated-asset root, `local/`: `frozen/` holds the comparison manifest, minimal
reproduction, provenance and sealed evaluation originals; `runs/` holds future experimental
outputs; `teacher/` holds the optional external teacher; `tooling/` holds the pinned Wasm tool.
`maintenance/` contains only compact local cleanup/verification records. Cargo builds stay
in `target/`; `.venv/` and `.uv-cache/` are reusable Python infrastructure.

Never add local assets or trained weights to Git. Before removing a generated subtree,
confirm references, idle processes and file handles, containment, symlinks and mount boundaries.
Preserve unique evaluation inputs and their source-game/descendant partition identities.
Do not create archival copies of all obsolete build or experiment outputs. Record deletion
paths, reasons and approximate sizes locally. Git-history size is separate from worktree savings.

## Compatibility checks

`make check` checks generated development bindings against the locked toolchain. Pure-only
bindings are separate build outputs. The shared runtime tests cover model rejection, legal
moves, history, accumulators and native/Wasm contracts. The local baseline smoke tests run
only a small fixed search and explicit failure-on-bad-model checks; they do not remeasure strength.

No CI workflow is configured in this tree. Local passes do not claim remote CI, device/browser UI,
publication, training recovery or large-evaluation availability.
