# Phase 10R source audit

Status: completed as a bounded, file-level audit on 2026-08-21 (Asia/Tokyo).

The machine-readable source of truth is [`configs/phase10r/source-registry.yaml`](../../configs/phase10r/source-registry.yaml). It contains 68 artifact records across 11 source families, including exact archive names, direct URLs, revisions, formats, sizes when published, evidence URLs, permission dimensions, and a fail-closed decision. The registry deliberately distinguishes `approved`, `approved_local_only`, `pending_permission`, `denied`, and `reserved_holdout`.

## Decisions

| State | Count | Scope |
| --- | ---: | --- |
| `approved` | 33 | Existing exact AobaZero catalog, WCSC 1–29/31–32, and the exact Denryu hardware-3 archive; training only where the artifact permission says `approved`. |
| `approved_local_only` | 1 | Bonanza prior-art inspection; no third-party table import, production use, or redistribution. |
| `pending_permission` | 26 | GCT, nodchip, Taya/Yaneura, QPD, Floodgate filters, DL水匠15b derivative, most Denryu archives, Lishogi, and unpinned artifacts. |
| `denied` | 1 | Existing public evaluation/test material; retained as a non-training boundary. |
| `reserved_holdout` | 7 | WCSC 33–36 and recent Denryu TSEC archives. |

An artifact is not approved because it is reachable, has a repository license, or is mentioned in a README. For every approved row, the audit records raw training permission separately from derived-weight, redistribution, and commercial-use permission. A pending field remains pending; it is not inferred.

## Approved exact artifacts

The CSA page says game-record use is free but asks for the program name, game date, event name, or source when records are used in articles or publications. The audit preserves that attribution requirement for all WCSC archive entries. See the [official WCSC archive page](https://www.computer-shogi.org/kifu/kifu.html).

The approved WCSC archive list is:

`kifu1.lzh`, `kifu2.lzh`, `kifu3.lzh`, `kifu4.lzh`, `kifu5.lzh`, `kifu6.lzh`, `kifu7.lzh`, `kifu8.lzh`, `kifu9.lzh`, `kifu10.lzh`, `kifu11.lzh`, `kifu12.lzh`, `wcsc13_kifu.lzh`, `wcsc14_kifu.lzh`, `wcsc15_kifu.lzh`, `wcsc16_kifu.lzh`, `wcsc17.zip`, `wcsc18_kifu.zip`, `wcsc19_kifu.zip`, `wcsc20_kifu.zip`, `wcsc21_kifu.zip`, `wcsc22_kifu.zip`, `wcsc23_kifu.zip`, `wcsc24_kifu.zip`, `wcsc25_kifu.zip`, `wcsc26_kifu.zip`, `wcsc27_kifu.zip`, `wcsc28_kifu.zip`, `wcsc29_kifu.zip`, `wcsc31_kifu.zip`, and `wcsc32_kifu.zip`.

WCSC30 is not an omitted download: the official archive list identifies that tournament as cancelled. WCSC33–36 are registered as `reserved_holdout` and were not acquired or inspected.

The Denryu hardware-3 page explicitly says `棋譜利用は制限等ありませんので、ご自由にお使いください。` The exact `kifu_dr5hdw3.zip` archive is therefore approved for raw game-record training, subject to the required attribution and the separate derived-weight fields. The [official Denryu link collection](https://denryu-sen.jp/denryusen/dr_link/dr1_live.php) is retained as the archive index. Production, TSEC, hardware-1/2, and designated-position archives are not generalized from that sentence: they remain pending or holdout in the registry.

The existing exact 100-object AobaZero approval is carried forward without expansion. Live AobaZero indexes, daily records, and new revisions remain pending because their exact manifest/evidence differs from the existing provenance object.

## Pending and denied families

* The [GCT release article](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701) publishes names and formats such as HCPE/HCPE3, self-play, Floodgate, Taya, AobaZero, and Suisho material. It does not supply the complete data-use grant required for training, derived weights, or redistribution. No GCT bulk artifact was downloaded.
* The [nodchip Hao repository](https://huggingface.co/datasets/nodchip/shogi_hao_depth9/tree/main) and related Tanuki/Suisho repositories are recorded as exact repositories, but repository metadata is not treated as a data and derivative-rights grant. The approximately 320 GB Hao release was not downloaded.
* The [DL水匠15b knowledge-distilled README](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_DLSuisho15b/raw/main/README.md) documents a transformation of nodchip material and quality caveats. Its approximately 160億-position release remains pending; the approximately 855 GB bulk release was not downloaded.
* The [QPD notice](https://qhapaq.hatenablog.com/entry/2021/11/23/220251) asks users to disclose use when a trained evaluation function is entered in a tournament or published. That notice is preserved as evidence, but the full redistribution/commercial/derived-weight scope is still pending.
* Floodgate records require an explicit operator/program-level decision and an exact filter manifest. The existing public evaluation/test boundary remains denied and is not acquired.
* The [Lishogi API](https://lishogi.org/api) has documented rate limits, including a full-minute wait after HTTP 429. Rate limiting does not establish ML or redistribution rights; the terms and account authorization remain pending, so no export was performed.
* Bonanza is local-only historical prior art. The maintained source mirror records a commercial-use restriction; third-party `fv.bin`, `book.bin`, and `hash.bin` tables are not imported.

## Acquisition evidence

Two small approved archives were fetched under the 150 GiB free-space guard and retained only below the ignored `local/phase10r-data` root:

| Artifact | Exact bytes | SHA-256 | HTTP evidence |
| --- | ---: | --- | --- |
| `wcsc32_kifu.zip` | 297,175 | `d6b8ed2b4b971f488800835d2f8e5fe3a2026c5bf893b5c3e161c8a0d48eac03` | ETag `"ff96e06244d1d545d037702c170c1803"`; Last-Modified 2026-02-03. |
| `kifu_dr5hdw3.zip` | 2,200,967 | `3aa6b8d21ba0f586a9f8c04425425883453887a8cf403d0e598a4e11ee2ee79e` | ETag `"219587-62d9eedbe0540"`; Last-Modified 2025-02-08. |

The same objects were re-acquired through the Phase 10R resumable downloader into the ignored raw object store, so the command path, checksum verification, and JSONL checkpoint are exercised. No raw archive, normalized JSONL, teacher artifact, or model weight is tracked by Git.

## Preserved failed Phase 10 evidence

The failed Phase 10 offline-gate report is preserved byte-for-byte as a gzip object at `artifacts/phase10r/failed-phase10-offline-gate/campaign-report.json.gz` (ignored). The original uncompressed SHA-256 is `7453b399eb17d690f59f71441c70d69ec5fe23344605ed17c457a24e82bc1935`; the compressed evidence SHA-256 is `bfd190c223bbaf3fa54335780a67c4e1d00a9bea4ab79e9d816a5cebc1775ffb`. Its result remains `FAILED_OFFLINE_GATE`; no Arena, public test, holdout inspection, training, or promotion was started from that failure.
