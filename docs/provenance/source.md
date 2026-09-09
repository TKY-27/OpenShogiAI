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

## Current generated interface

The development bindings are generated from this repository's Rust source and Cargo.lock
with wasm-bindgen 0.2.127. Source paths in standard distributed bindings are remapped to
`/user`, `/cargo` and `/open-shogi` at compile time; no host account path is retained. `make wasm-web-check` verifies all four files byte-for-byte.
The current Wasm SHA-256 is
`8a07d7261447b23fcce1e72937b8190e7f54f7c34ee98fd45bfd009de7981962`.
`scripts/check_provenance.sh` also checks their identities and private-history exclusion.
Older generation hashes remain historical Git evidence, not current freeze gates.

## Engine, data and models

Production engine source is independently implemented; [references](../references.md) record
protocol and algorithm references. External teachers are separate processes, never linked
engine implementations. See [third-party notices](../../THIRD_PARTY.md).

Source-specific rights decisions and reservations live in the data registries and
[data handling](../data/handling.md). A historical accepted registry entry is scoped to its
exact source/version and is not a blanket confirmation of future third-party use.

The frozen OSAVAL03 comparison model derives from source commit
`0203a847dc662b4f32404b30340905a953f42865`, with random-initialization lineage and external
Apery labels. Its local manifest binds the model, teacher, datasets and split metadata;
weights are not included in Git and distribution remains pending review.

Future acquired objects/models must record source and rights evidence, retrieval/creation time,
size, hash, format/configuration, seed, code/runtime identity and transformation/partition
lineage. Unknown provenance fails closed.
