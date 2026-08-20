# Architecture

OpenShogiAI is the AI-side repository. Browser presentation is a separate consumer and is not a
Cargo, Python, or npm dependency of this project.

```text
external USI teacher process
           |
           v
data tooling -> teacher adapter -> training/export -> OSAVAL01
      |                                  |
      +-> opening tooling                v
                                   model runtime
                                         |
engine/cli -> engine/usi -> engine/core <-+-> engine/wasm -> bindings/wasm
     |
     +-> Arena / self-play orchestration through versioned artifacts
```

## Rust

- `engine/core`: rules, legal moves, notation, game state, handcrafted evaluation,
  `OSAVAL01` inference, secure artifact I/O, and search.
- `engine/usi`: the bounded USI protocol boundary over `engine/core`.
- `engine/cli`: local tools, terminal play, Arena, model inspection, dataset replay, and
  opening-book integration.
- `engine/wasm`: the versioned WebAssembly adapter over `engine/core`.
- `bindings/wasm`: deterministic `wasm-bindgen` output. It is an engine interface, not a GUI.

Cargo dependency direction is one-way toward `engine/core`; no crate depends on UI code.

## Python

`training/open_shogi_training/` contains source-governed data tooling, external-process teacher
integration, model training/export, self-play/Arena orchestration, and bounded evaluation.
Teacher engines remain separate executables communicating only through USI. Generated data,
labels, checkpoints, models, games, and reports stay in ignored local storage.

## Interchange boundaries

SFEN, USI, CSA, `OSAVAL01`, arena reports, model registries, and browser-engine JSON use closed,
versioned, bounded formats documented in `docs/interfaces.md`. Untrusted files and process
output are validated for size, schema, identity, hashes, legality, and path containment as
applicable.

The generated Wasm bindings are the only physical artifact copied to OpenShogiUI. An integration
check compares those four files byte-for-byte; UI code never imports an engine-internal crate.

## Governed storage

`data/raw/`, `data/processed/`, `artifacts/`, `weights/`, and `local/` are ignored
except for explanatory documents. Teacher binaries and evaluation files belong under
`local/teacher/`. Candidate boundary checks reject accidental inclusion of private data,
trained weights, browser GUI source, secrets, or absolute machine paths.
