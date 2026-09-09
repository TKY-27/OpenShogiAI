# Architecture

OpenShogiAI is the AI-side repository. Browser presentation is a separate consumer and is not a
Cargo, Python, or npm dependency of this project.

```text
external USI teacher process
           |
           v
data tooling -> teacher adapter -> training/export -> OSAVAL01 / OSAVAL02 / OSAT10A1 / OSAVAL03
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
  `OSAVAL01` inference, shared native/Wasm `OSAVAL02` sparse inference, secure artifact I/O,
  time allocation, play/analysis resource
  coordination, persistent completed-depth analysis cache, opening-book validation, and search.
- `engine/usi`: the interruptible USI protocol boundary over `engine/core`, including complete
  clock mapping and immediate validated book moves.
- `engine/cli`: local tools, terminal play, Arena, model inspection, dataset replay, and
  opening-book integration. Its line-oriented analysis command is the minimal reference client
  for the UI protocol.
- `engine/wasm`: the versioned WebAssembly adapter over `engine/core`, with the same time-control,
  opening, and analysis lifecycle schemas.
- `bindings/wasm`: deterministic `wasm-bindgen` output. It is an engine interface, not a GUI.

Cargo dependency direction is one-way toward `engine/core`; no crate depends on UI code.

## Python

`training/open_shogi_training/` contains source-governed data tooling, external-process teacher
integration, model training/export, self-play/Arena orchestration, and bounded evaluation.
Teacher engines remain separate executables communicating only through USI. Generated data,
labels, checkpoints, models, games, and reports stay in ignored local storage.

## Interchange boundaries

SFEN, USI, CSA, `OSAVAL01`, `OSAVAL02`, `OSAT10A1`, `OSAVAL03`, arena reports, model registries, and browser-engine JSON use closed,
versioned, bounded formats documented in [interfaces](interfaces.md). Untrusted files and process
output are validated for size, schema, identity, hashes, legality, and path containment as
applicable.

Play and continuous analysis are separate logical `SearchEngine` instances. Compatible
transposition entries and completed root evidence may be reused, but recursive call stacks are
never serialized. Hosts enforce the versioned resource budget and may pause analysis slices while
play is active.

The generated Wasm bindings are the only physical artifact copied to OpenShogiUI. An integration
check compares those four files byte-for-byte; UI code never imports an engine-internal crate.

## Storage and local baseline

Generated assets live under ignored `local/`, builds under `target/`. Model loading always
requires an explicit format and expected hash for pure profiles. The local comparison manifest
is `local/frozen/manifest.json`; the public catalog is [model registry](../configs/models/registry.json).
See [model handling](model/handling.md) and [development](development.md).
