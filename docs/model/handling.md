# Models and local comparison

The default evaluator is built-in `handcrafted-experimental`, requiring no local weight.
The [registry](../../configs/models/registry.json) records a single comparison candidate;
it does not promote a champion. All trained weights remain ignored and distribution is
pending review; source licensing does not grant weight redistribution.

## Frozen baseline

`100k-w256-hard2` uses OSAVAL03, width 256, SHA-256
`859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480`, size 8,679,836 bytes.
Source commit: `0203a847dc662b4f32404b30340905a953f42865`.
The one canonical copy is `local/frozen/baseline/model.osaval03`; its profile and local manifest
bind format, hash and provenance. `metadata/` contains compact training/teacher/split/result
records. The historical training corpus had 115,993 labeled rows (107,763 train / 8,230 validation),
with 4,131 mate-only rows masked. These are row counts, not claims of unique new positions;
training exposures and lineage are separately recorded. No optimizer or restart checkpoint is retained.
Historical results and current limitations live in the ignored local run records.

```sh
make pure-build
make frozen-smoke
```

The smoke checks the explicit hash, native legal search and actual Wasm model/browser contract.
Without the local model it fails with a missing-model error; `make check` does not need it.
Builds are regenerated from source/lockfiles, not kept as large frozen build directories.

## Resolved hard4 reproduction

The no-legal-moves runtime-proof defect is fixed and covered by small native/pure/Wasm
regression fixtures. The obsolete hard4 weight and historical executable were removed;
its original request, failure identity and compact diagnosis remain in ignored local storage.
The useful model copies are the old hard2 comparator and r3 best, each with format/hash/lineage.
The retired active-run handoff lives in ignored `local/handoff/`.

## Supported formats

- OSAVAL01: dense value model, feature/model configs under `configs/features/` and `configs/models/`;
  the container is specified in [interfaces](../interfaces.md).
- [OSAVAL02](OSAVAL02_FORMAT.md): sparse pair/triple evaluator and explicit history semantics.
- [OSAT10A1](OSAT10A1_FORMAT.md): king-relative accumulator runtime.
- [OSAVAL03](OSAVAL03_FORMAT.md): current local W256 comparison container.

Legacy trained candidates were removed. Small synthetic fixtures and deterministic generators
exercise format/legality/parity without those weights. Historical registry identifiers do not
mean a removed model is currently available. Model loads fail closed; no format fallback or
handcrafted substitution is permitted inside the pure-only profile.
