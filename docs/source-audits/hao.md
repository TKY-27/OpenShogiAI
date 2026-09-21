# R4-C2 source decision (2026-09-20)

This decision applies only to the four pinned C2 shards. It does not rewrite the
closed Phase 10A audit or its storage/rights decisions.

Publisher: nodchip. Dataset: [shogi_hao_depth9](https://huggingface.co/datasets/nodchip/shogi_hao_depth9).
The dataset's own [README at revision 7bc19a9e880ea307a52c57f57ea6c752301b25bc](https://huggingface.co/datasets/nodchip/shogi_hao_depth9/blob/7bc19a9e880ea307a52c57f57ea6c752301b25bc/README.md)
declares `license: mit`, identifies training-data files, unshuffled PackedSfenValue,
and Hao depth 9 generation. This is dataset-specific publisher metadata, not an
engine-code license inferred onto different files. C2 accepts it for local training
and derived-weight preparation; retain publisher/MIT attribution with any future
approved distribution. No raw dataset is redistributed. No dataset-specific separate
license text or exact generating engine/weights hash was supplied; that limit remains explicit.
Future model publication additionally needs the user's GO and the model allowlist review.

`configs/evaluator-main.json` の `data_preparation` fixes revision, exact URLs, sizes, SHA-256,
local paths and parser. Four spread shards cover start batches 1695340981 (threads000/032),
1695606850 (001), 1695872823 (002). Total 1,231,381,720 bytes / 30,784,543 supplied
40-byte records. These are not 30M optimizer examples. No full 320GB archive acquisition.
Exact downloaded card/API and format-source snapshots remain in ignored `local/r4-c2-preparation/`.

Parser `r4_sources.py` decodes `<32shHHbB`: packed root, signed teacher Value, move16,
gamePly, root-side game result, uninterpreted padding. Result is not used as a good-move
label. Padding can be nonzero; it is not a corruption flag. Root board is not the
quiescence leaf; the engine stores its PV-leaf evaluation from root-side perspective.
The referenced nodchip engine [source](https://github.com/nodchip/tanuki-/tree/abccc2ba0fe1cc34e354399f275d27f732dc96e7)
and [format implementation](https://github.com/yaneurao/YaneuraOu/blob/master/source/extra/sfen_packer.cpp)
explain packing and USI conversion: trunc(100 * Value / 90). This is the format/scale
reference, not a claim that the generating binary was recovered. Raw Value is retained.
No probability calibration, fabricated MultiPV/bound, or mate certificate is inferred.
Large/mate-range values are excluded; no wholesale local D12 relabel.

Actual first 1,000 records decoded; 991 adjacent same-segment recorded moves reproduce
the next root with the existing rules helper (8 segment-boundary pairs excluded).
A synthetic standard-position decoder regression and native/Wasm legality checks
cover the execution boundary. Original game/history IDs are absent. Ply resets plus
initial-state families infer partitions; this cannot certify full series independence.
16-ply spacing, source-game exposure caps, symmetry-normalized cross-source exclusion,
and separate old-source validation reduce the measurable risks without merging positions
by piece count or opening name. Existing conflicting labels win; no unconditional averaging.
An ancestor may already have seen an equivalent board: new acquisition is not proven
new lifetime experience.

The independent GCT release was also checked: [publisher's release description](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701),
[hcpe3 folder](https://drive.google.com/drive/folders/1mykqtCNnUWR4jZKF5uLSQsD8-4vAqpmR).
`selfplay_gct-???.hcpe3.xz` is distinguished from AobaZero conversion, Floodgate and
mutual-position descendants. Current files are listed; reusable downstream/derived-weight
conditions remain unverified. GCT adoption and actual learning count are **0**, pending rights;
it does not stop the admitted Hao/replay round. Tanuki/Suisho multipart archives,
Knowledge_distilled descendants and other suggested tournaments were not adopted or downloaded.

## C3 reuse (2026-09-21)

C3 reuses the exact admitted C2 dataset; no additional external shards or GCT rows
are claimed. The original download plan is retained in the sealed C2 run and
`local/r4-c2-preparation/dataset/manifest.json`. The current configuration pins
that manifest and the publisher card. Inherited acquisition statistics describe
C2; `added_counterfactual_rows` describes only new C3 authored/Apery observations.

A local source review confirmed `evaluate_leaf` reverses leaf evaluation when
leaf/root side differs, and `PawnValue=90` explains the existing 100/90 conversion.
Apery MultiPV root scores are negated for child-side scalar labels; root boards
are not confused with search leaves. No new train/test-fitted scale is introduced.
Sibling ranking is restricted to one source/root and ignores differences at most
50 cp; cross-teacher absolute equivalence remains a hypothesis tested by separate
source losses. Original generating weights/history remain unknown.

## C4 reuse (2026-09-21)

C4 pins the completed C3 dataset manifest and keeps its input references, rather than
reacquiring the four shards. Fresh C4 data is generated locally with recorded OSAI actors
and the already audited, hash-bound offline Apery binary/evaluation assets. D8 broad and
D12 strong labels use direct root/child side-to-move scores; incomplete/bounded/mate claims
are not converted to scalar targets. Historical source IDs and inferred Hao sequences
remain visible in sampling/exposure accounting. No GCT, dlshogi or other new download is
admitted by adding this section. Source conditions and the separate publication GO remain.
