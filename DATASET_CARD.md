# Dataset Card

## Status and scope

Phase 3 approves one bounded source slice: the exact 100 AobaZero no-noise CSA objects listed
in `configs/data_source_objects/aobazero_no_noise.yaml`. The decision and pinned official
evidence are documented in `docs/source-audits/aobazero.md`. It does not approve other
AobaZero files, weights, bulk archives, dynamically discovered URLs, or Floodgate. Generated
raw and processed artifacts remain local and ignored by Git.

## Rights decision

The immutable official AobaZero README connects the official project page to its collected
game records and game samples, then assigns Public Domain status to project material other
than the named GPLv3 USI engine. The official sample index is separately hash-pinned and must
reference every object in the exact 100-object audited subset. The pinned page references 209
CSA samples; the approved catalog selects `w4745.csa` through `w4235.csa` in steps of five,
except the page's unavailable `w4495.csa`, `w4425.csa`, and `w4420.csa` references. For this
slice, machine-learning use and redistribution are approved. Public reachability by itself is
never treated as permission.

## Acquisition and transformation

The fail-closed downloader fetches a live robots policy first, verifies hash-pinned rights and
sample-index evidence, and then requests only exact catalog URLs at 0.25 requests/second with
concurrency one. Each completed object records URL, retrieval time, response validators,
format, size, SHA-256, rights evidence, and usage decisions in an append-only manifest. Raw
objects and evidence use content-addressed storage.

Normalization performs these steps:

1. verify the manifest, exact catalog metadata, content-addressed paths, evidence, size, and
   SHA-256 from no-follow file descriptors;
2. adapt only the audited AobaZero CSA dialect while retaining the source bytes;
3. replay every move through the independent Rust legality engine and emit canonical CSA,
   initial/position SFEN, USI moves, outcome, terminal reason, names, optional ratings/date,
   source URL/ID, and raw SHA-256;
4. write deterministic `phase3_game/v1` and `phase3_position/v1` gzip JSONL plus a dataset
   manifest and normalization report.

Invalid, corrupt, raw-duplicate, canonical-duplicate, extremely short, extremely long, and
position-cap cases are counted explicitly. Position duplication is reported within and across
games. The final state and configured final eight move-bearing positions are ineligible for
opening statistics, preventing resignation-tail oversampling.

## Splits and opening data

Train, validation, and test assignment is game-only and addition-stable. It hashes the public
split salt plus the canonical game SHA-256; all positions from one game inherit the same split.
The independent SQLite opening database consumes only `train` rows marked `eligible` and
stores state/move counts, actor and black/white outcome counts, ply sums, and source counts.
Query/export derives score and decisive win rates, side-specific rates, average plies, and
Wilson 95% intervals from those sufficient statistics. Minimum occurrence count is applied at
query/export time rather than deleting observations.

## Known limitations

- The sample is one self-play source and one recent exact slice, so it does not represent all
  openings, eras, engines, or human play.
- Player ratings are optional and may be absent. Source timestamps have no verified timezone.
- The official sample host is HTTP-only. Pinned evidence, hashes, and repeat verification
  detect later changes but cannot remove first-transfer on-path authenticity risk.
- CSA results such as resignation can depend on external conditions; the dataset preserves
  Rust's `verified`, `external_condition`, or `missing` result classification.
- No general license is asserted for future datasets or model weights.

## Observed Phase 3 sample

The completed local run included 100 of 100 acquired games with no exclusions and produced
15,488 positions: 72/18/10 games in train/validation/test and 14,588 eligible positions. The
100 raw hashes and 100 canonical hashes were unique. The acquisition manifest SHA-256 is
`4a02894ba3e0c5730bba42c37e4cda6193b625fa6d638401052d636817a1b267`; the normalized
dataset manifest SHA-256 is
`db2a286ebfa4edd01c67041fb55d33b0f5f8813d24a2203d7c5e55a36111e243`.

The full counts, artifact hashes, runtime versions, limitations, and reproducibility result
are recorded in `PHASE_3_REPORT.md`. The complete 100-object hash list remains in ignored local
manifests rather than being duplicated here.

## Phase 4 teacher-label view

Phase 4 selected 10,000 unique canonical board/side/hand states from the 14,588 eligible
positions while preserving each Phase 3 game's split. Cross-split duplicates were assigned by
the fixed priority `test`, `validation`, then `train`; same-priority duplicates were removed.
The resulting label population is:

| Dimension | Count |
| --- | ---: |
| Train / validation / test | 7,042 / 1,894 / 1,064 |
| Opening / middlegame / endgame | 2,257 / 3,923 / 3,820 |
| Centipawn / mate labels | 9,725 / 275 |
| Three / two / one recorded PV candidates | 9,578 / 294 / 128 |
| Completed / quarantined | 10,000 / 0 |

One or two PV candidates are retained only when the teacher emits a nonempty contiguous
MultiPV prefix. Every returned root and full PV is replayed by the independent Rust rules
engine, rank one must match `bestmove`, and roots must be unique. A short prefix means
incomplete teacher coverage, not necessarily that the position has fewer than three legal
moves. Of 422 short rows, 420 have returned-candidate count equal to the independent depth-one
legal-root count. Two rows each omit one additional legal root move: label line 4,363 omits
`5i5f`, and line 6,967 omits `2h2f`. Three repeat probes per row reproduced the same Apery
output. No score is fabricated for either omitted move. Bounded `lowerbound`/`upperbound`
scores are validated but never used as exact labels. The label loader keeps centipawn and mate
scores distinct, verifies all repeated provenance/config hashes, and does not expose the test
split to training.

The label file SHA-256 is
`195850b3ae7b1dce8a98185f2ba17f794a0200c92eafc672e11674d17d8cd3f4`.
The complete label and manifest files remain ignored local artifacts; no raw data or teacher
output is committed.

## Phase 6 bounded self-play and replay view

Phase 6 did not acquire a new external dataset and did not increase the teacher-label count.
One bounded generation ran 40 self-play games as 20 same-start, color-swapped pairs: ten pairs
from the normal initial position and ten pairs from approved Phase 3 train positions. Seed,
nodes, maximum plies, models, opening state, full artifact identities, and CSA identities are
bound by the plan and execution manifests. All 40 games completed; no game or attempt was
quarantined and no illegal move was recorded.

The derived position evidence contains 10,722 unique retained positions: 10,000 Phase 4
teacher positions plus 722 non-duplicate, non-protected self-play positions. It applied the
initial model's predictions to all 1,894 validation teacher positions. Derivation excluded
2,018 duplicate self-play rows and 20 protected start positions. The evidence artifact
SHA-256 is
`7eebf02480fc0f4242c48dec453af9903b76a43563b847071f51d32a79c2fed1`.

Hard-position extraction found 2,143 eligible unique positions and selected 500, all of which
already had Phase 4 teacher labels. The available new-label budget was zero because the exact
10,000 cap had already been reached. Apery therefore received no Phase 6 labeling request.

The replay buffer retained 10,590 unique entries under a 100,000-position capacity:

| Dimension | Count |
| --- | ---: |
| Train / validation / test | 7,706 / 1,820 / 1,064 |
| New-generation / older retained | 702 / 9,888 |
| Hard positions retained | 485 |
| Factual outcome targets: win / loss / draw | 5,300 / 5,252 / 38 |
| Deduplication deletions | 30 |

The replay manifest SHA-256 is
`eb2ec3a511d3bab2b19fc13fc149d799b0dbdc35d4663a178488aa7923799b60`.
Self-play rows use only replayed factual outcomes and retain generation, game, split, SFEN,
source manifest, and position identities. Unknown or max-plies outcomes are not converted into
training targets. The held-out test split remains final-evaluation-only.

Human-play CSA and decision logs are isolated as `pending_human_review`. Human moves are not
accepted as labels or added to replay automatically. All Phase 6 datasets and reports remain
ignored local artifacts and were not uploaded or published.

## 2026-08-21 bounded evaluator and opening views

The evaluator campaign did not acquire new data and did not expand the approved population.
Pure 2x128, residual 2x64, and residual 2x128 used the same 10,000 unique teacher positions,
fixed split assignment, seed, and manifests as the initial model. Training and model selection
read train/validation only. Exactly one selected model, residual 2x128, was evaluated on the
1,064-row held-out test split after selection. No test row entered a residual baseline decision,
training batch, checkpoint selection, move-ranking benchmark, opening statistic, or Arena start
selection.

The opening-book workflow reanalyzed those same 10,000 unique positions with the separately
predeclared Apery MultiPV 32 configuration. This created a wider teacher view, not 10,000 new
unique teacher-labeled positions. Its label SHA-256 is
`03f5375f09b167a12a4297cd2a404f0fb95a91edd2f6cdeabc955c14c7ce5ea0`, and its manifest SHA-256
is `57f70b734f8c3eab4bd6c0ae2c439eae3373baf4f1b304d4fb32a17bb0e726fb`.
All 10,000 completed, none was quarantined, and every returned root/PV was independently replayed
for legality. Of 10,000 rows, 8,105 returned 32 candidates. Short rows retained incomplete
coverage explicitly; 25,946 independently legal but unreturned roots received no fabricated
score.

The resulting v2 book joins only approved Phase 3 train-split opening counts to teacher-returned
legal candidates. It contains 424 canonical positions and 448 candidates through at most 40
plies. Generated labels, manifests, predictions, model artifacts, Arena records, and the book
remain ignored local evidence and were not published.
