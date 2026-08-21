# Phase 10R-A data foundation report

Date: 2026-08-21 (Asia/Tokyo)
Branch: `codex/phase10r-data-foundation`
Scope: rights audit, bounded acquisition, source-preserving adapters, legality validation, overlap policy, and storage guardrails.

## Outcome

The Phase 10R data foundation is established without activating unresolved sources. The file-level registry covers 68 artifacts across 11 source families. Thirty-three artifacts are approved for raw-record training, one is local-only prior art, 26 are pending permission, one is denied, and seven are reserved holdouts. Two small approved archives were acquired under the 150 GiB free-space guard: WCSC32 and Denryu hardware-3. No full GCT, nodchip, QPD, or DL水匠15b release was downloaded.

No training, teacher relabeling, self-play, Arena, public-test inspection, holdout inspection, promotion, push, release, or deployment was performed.

## Preserved failed Phase 10 evidence

The failed Phase 10 offline-gate report is retained as the ignored gzip object `artifacts/phase10r/failed-phase10-offline-gate/campaign-report.json.gz`. The original uncompressed report SHA-256 is `7453b399eb17d690f59f71441c70d69ec5fe23344605ed17c457a24e82bc1935`; compressed evidence SHA-256 is `bfd190c223bbaf3fa54335780a67c4e1d00a9bea4ab79e9d816a5cebc1775ffb`. The report records `FAILED_OFFLINE_GATE` for both frozen pure candidates and a stop before Arena. Its dataset/config/model hashes remain bound in the preserved report; they were not mixed into Phase 10R data.

## Rights and source decisions

The authoritative registry is [`configs/phase10r/source-registry.yaml`](configs/phase10r/source-registry.yaml). The supporting audit is [`docs/data/PHASE10R_SOURCE_AUDIT.md`](docs/data/PHASE10R_SOURCE_AUDIT.md), with the permission dimensions in [`docs/data/PHASE10R_LICENSE_MATRIX.md`](docs/data/PHASE10R_LICENSE_MATRIX.md).

The approved scope is intentionally narrow:

* the existing exact 100-object AobaZero approval;
* WCSC1–29, WCSC31–32, with archive attribution retained; and
* the exact Denryu hardware-3 archive whose official page states unrestricted game-record use.

WCSC33–36 and recent Denryu TSEC archives are reserved before normalization. GCT, nodchip, Taya/Yaneura, QPD, Floodgate, the DL水匠15b derivative, and Lishogi remain pending. Bonanza is local-only prior art. Permission-request drafts are unsent in [`docs/data/permission-requests/`](docs/data/permission-requests/).

## Acquired and validated samples

| Artifact | Raw object | Archive inventory | Adapter / replay result |
| --- | --- | --- | --- |
| WCSC32 | 297,175 bytes; SHA-256 `d6b8ed2b4b971f488800835d2f8e5fe3a2026c5bf893b5c3e161c8a0d48eac03` | 293 entries; 519,128 uncompressed bytes; 279 CSA files after excluding metadata | CP932 CSA preserved; derived UTF-8/V3.0 view; 10/10 bounded games accepted by Rust; first file yielded 320 move-prefix records. |
| Denryu hardware-3 | 2,200,967 bytes; SHA-256 `3aa6b8d21ba0f586a9f8c04425425883453887a8cf403d0e598a4e11ee2ee79e` | 135 entries; 12,016,210 uncompressed bytes; 132 KIF files | CP932 KIF, raw evaluation/PV preserved; printed source-square converter; 1/1 bounded game accepted by Rust; 96 moves and terminal `千日手`. |

Normalized sample JSONL and derived validation outputs remain ignored below `local/phase10r-data`; none is committed. The exact acquisition/checksum/inventory summary is in [`artifacts/phase10r/data-foundation-manifest.json`](artifacts/phase10r/data-foundation-manifest.json).

## Implementation

* `phase10r_registry.py` validates exact decision states, permission dimensions, URLs, hashes, source/artifact indexes, and fail-closed approval rules.
* `phase10r_acquisition.py` resolves `OPENSHOGI_DATA_ROOT`, checks disk before and during streaming, resumes with ETag/Last-Modified, writes JSONL checkpoints, and verifies size/SHA before activation.
* `phase10r_archive.py` inventories ZIP members, rejects traversal/symlinks, bounds decompression, and refuses overwrites. LZH is not silently treated as an empty archive.
* `phase10r_adapters.py` and `phase10r_kif.py` preserve CSA/KIF/HCPE/HCPE3/PackedSfenValue semantics and route derived CSA/KIF views through the existing Rust rule engine.
* `phase10r_overlap.py` measures exact identity layers, keeps raw-record hashes out of per-position deduplication, quarantines missing identities, and assigns deterministic splits.

The format, overlap, storage, and holdout policies are documented in [`docs/data/PHASE10R_FORMAT_MATRIX.md`](docs/data/PHASE10R_FORMAT_MATRIX.md), [`docs/data/PHASE10R_OVERLAP_REPORT.md`](docs/data/PHASE10R_OVERLAP_REPORT.md), [`docs/data/PHASE10R_STORAGE_PLAN.md`](docs/data/PHASE10R_STORAGE_PLAN.md), and [`docs/data/PHASE10R_HOLDOUT_POLICY.md`](docs/data/PHASE10R_HOLDOUT_POLICY.md).

## Verification

Focused Phase 10R tests cover registry fail-closed behavior, CP932 KIF evaluation/PV preservation, KIF promotion/drop/terminal conversion, binary layout retention, safe ZIP extraction, deterministic overlap, and split assignment. The pre-change baseline and final `make check` both passed; the final run included repository boundaries, license/provenance checks, formatting, lint, all Rust tests, 475 Python tests, build, and WASM verification.
