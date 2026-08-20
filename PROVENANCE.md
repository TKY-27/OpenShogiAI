# Provenance

## Clean-history extraction

This candidate was extracted on 2026-08-21 (Asia/Tokyo) from the private source checkpoint:

- private source commit: `5451e02d35abc3efc1fcc29260cdb89acad1d416`;
- private source tree: `21bf0c1fbe02250855b66a96f2fe37bd042650d4`;
- private checkpoint tag: `private-pre-split-source`;
- split-manifest SHA-256:
  `aa3eec7afc3bc9dff5a330eb6c3b3a1ee503baa703073281a906bb3753c01731`;
- private bundle identifier: `OpenShogiAI-private-pre-split.bundle`;
- private bundle SHA-256:
  `60504e4fe09686f61f3184008c759a2c0ba43182f4538e67bf170b4717715730`.

Extraction used `git archive private-pre-split-source` followed by the validated AI mapping in
the private split manifest. The generated Wasm interface was relocated from
`web/src/generated/` to `bindings/wasm/` without byte changes.

The browser GUI, UI-only tests, npm workspace, private phase reports, private audit
instructions, generated/local evidence, teacher binaries/evaluation files, raw/processed data,
checkpoints, and trained model weights were intentionally excluded. Candidate-specific AGPL
metadata, documentation, and automated boundary checks were then added.

The private source history was not published, grafted, or made reachable from this candidate.
The source commit identifiers above are provenance identifiers in the private repository, not
commits in this clean history.

The versioned Wasm module was subsequently regenerated from the repository's own Rust source with
the locked Cargo graph and `wasm-bindgen 0.2.127`. Its current SHA-256 is
`49034d4d1cff1e004eceaf1e62b309d9882516e039457c09a758904217fd3805`; `make wasm-web-check`
reproduces and verifies the complete binding set.

## Independent implementation

Project engine code was independently implemented. Material rules, standards, protocol, and
algorithm references are recorded in `docs/references.md`. Third-party engines may participate
only as separate processes and do not contribute production engine code.

## Data lineage

The only approved data source is the exact 100-object AobaZero no-noise CSA slice listed in
`configs/data_source_objects/aobazero_no_noise.yaml`. The decision, evidence, transport
limitations, and Public Domain status are documented in `DATASET_CARD.md` and
`docs/source-audits/aobazero.md`. Dataset bytes are not included.

## Teacher and model lineage

Apery v2.0.0 may be used as a separately installed USI teacher. Its engine and evaluation
licenses remain GPL-3.0-only and MIT respectively. Teacher artifacts are not included.

The project-designed `value_v0` model starts from random initialization and exports the
versioned `OSAVAL01` format. Historical bounded runs and hashes are documented in
`MODEL_CARD.md`; no checkpoint or trained weight is included. Model-weight licensing remains
`pending-review`.

## Required future records

Every acquired object and released model must bind source identity, retrieval or creation time,
license evidence, byte size, SHA-256, format, configuration, seed, code identity, runtime
versions, transformation lineage, and output hashes. Unknown or ambiguous provenance fails
closed.
