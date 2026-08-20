# Phase 9 Report: Analysis, Time, Opening Book, and Strength Campaign

Report date: 2026-08-21 (Asia/Tokyo)

Branch: `codex/engine-analysis-book-strength`

Repository: final AI-only `OpenShogiAI`; `OpenShogiUI` was not modified.

## Completion status and champion identities

The phase implementation and bounded measurements are complete. Progression to a larger
self-play generation is not ready because no non-baseline evaluator passed the predeclared
overall-champion gate.

1. **Overall champion:** `handcrafted-experimental`.
2. **Neural-lineage champion:** `value-v0-f32-g1r4`.

Native play, USI `ModelKind=overall-champion`, Wasm defaults, and UI-facing engine defaults resolve
to the overall champion. A neural model is never promoted overall from a neural-only comparison.

The authoritative historical fixed-node Phase 5 comparison was handcrafted experimental versus
the initial neural evaluator: handcrafted won 35-3 with 2 draws, zero illegal moves, and zero
reported crashes. The neural score rate was 0.100 and the recorded paired-bootstrap interval was
0.025-0.1875. This was fixed-node evidence, not equal-wall-clock evidence. The later g1r4 neural
challenger beat g0 32-8, with Wilson 95% interval 0.6524-0.8950, and therefore became only the
neural-lineage champion.

## Equal-wall-clock evidence and overall gate

The exploratory matrix used 10 ms per move for both players, depth cap 8, 32 MiB hash per player,
alternating colors, 128 plies, opening disabled, and no illegal moves. Every candidate played
handcrafted experimental.

| Candidate | Finished / scheduled | Candidate W-L-D | Interpretation |
| --- | ---: | ---: | --- |
| handcrafted baseline | 17/20 | 0-2-15 | 3 capped games; no 20-game score claim |
| pure 2x64 initial | 20/20 | 0-20-0 | exploratory |
| pure 2x64 g1r4 | 20/20 | 0-20-0 | exploratory |
| pure 2x128 | 20/20 | 0-20-0 | exploratory |
| residual 2x64 | 20/20 | 0-20-0 | exploratory |
| composite g1r4 | 20/20 | 0-19-1 | score 0.025; Wilson-with-half-draws 0.0026-0.2005 |

Residual 2x128 alone entered the predeclared overall gate in
`configs/evaluation/overall_champion_gate.toml` (SHA-256
`dd5a6c19c7384762f750c7e95bcb9cac46f441eb4130b8e275d99cfc732e95aa`).
It finished all 40 paired games at 0 wins, 8 losses, and 32 draws. Candidate score was 0.400;
Wilson-with-half-draws 95% interval was 0.2634832568-0.5540410634. Candidate/incumbent measured
search time was 14,822/14,900 ms over 1,471/1,491 searches. There were zero illegal moves and no
process crash observed during the uninterrupted successful invocation. Both evaluators passed
both tactical cases, so tactical regressions were zero.

The candidate failed minimum decisive games (8 < 12), minimum score rate (0.400 < 0.550), and
minimum lower confidence bound (0.2635 < 0.500). Decision artifact SHA-256 is
`062e834722f8aa63c6f7f713258157863f0d9fc4f5b7a9fb8fbef5a6ea5e4048`; Arena report SHA-256 is
`b2b546af4344eb8461544cbf061163a12b0992727e5f837a9e81d5b5598638c3`; tactical report SHA-256 is
`05ce784dc477773ee944c6274fa3b64d95d95533c1c4ec1a7fccffa8554d26f1`.

## Time manager and hard-limit compliance

`open_shogi_time_control/v1` is shared by Rust, native play, USI mapping, and Wasm. It supports
casual no-clock, explicit move time, black/white remaining time, byoyomi, side-specific increment,
diagnostic nodes, depth, and infinite analysis. Native play flags are `--black-time-ms`,
`--white-time-ms`, `--byoyomi-ms`, `--black-increment-ms`, and `--white-increment-ms`; native AI
clock state is debited after each search and the applicable increment is added. Human-play config
records were versioned to `open_shogi_play_config/v3` so every clock field is identity-bound.

Deadlines use a monotonic clock and interruptible iterative deepening. Casual allocation is at
most 20,000 ms; the default 50 ms safety margin makes the search hard deadline 19,950 ms. The
engine always retains a legal root fallback. After the soft deadline it may finish early only when
the root and score are stable. Root changes, close candidate scores, score volatility, check,
tactical/mate danger, and difficult branching suppress early convergence and extend search toward
the hard deadline.

The deterministic fake-clock test disabled early convergence and observed `TimeLimit`, elapsed
not greater than 20,000 ms, and a legal move without sleeping. A real release-build smoke observed:

- `go movetime 200`: legal `5i6h` in 153.408 ms;
- casual `go`: legal `5i4h` in 3,904.232 ms.

The casual smoke demonstrates useful early completion rather than sleeping to 20 seconds. These
are local wall-clock observations, not portable performance promises.

## Continuous analysis and resource isolation

`open_shogi_analysis/v1` implements `start`, `stop`, `change-position`, `change-multi-pv`, bounded
renewed `step`, `worker-failed`, and `restart`. Updates contain completed depth, nodes, NPS,
centipawn or mate score, MultiPV lines, root statistics, timestamp, engine version, and every
compatibility identity.

`AnalysisCacheKey` binds canonical SFEN plus internal Zobrist hash, model hash,
evaluator-config hash, feature-schema hash, evaluation-semantics hash, search-options hash,
opening-profile hash, and MultiPV. Position switches cancel the prior root, publish a compatible
cached result immediately, preserve compatible TT/root evidence, and begin new iterative
deepening. Model, feature, evaluation-semantic, or search-option changes invalidate incompatible
entries. “Resume” does not mean restoring arbitrary recursive call stacks; only completed-depth
cache plus renewed search and compatible evidence are retained.

The deterministic same-root reuse benchmark searched 3,861 nodes initially and 124 on renewal,
with 94/94 second-search TT probes hitting. Tests also cover cancellation, position switch, cache
invalidation, MultiPV separation, worker failure/restart, and Wasm/native conformance.

Play and analysis own separate logical `SearchEngine` instances. `open_shogi_resource_budget/v1`
exposes play/analysis threads, separate hash allocations, pause-analysis-during-AI-turn, and
maximum aggregate memory. Search is currently single-threaded, so play threads are exactly one and
analysis threads are zero or one. The host-facing JSON schema and TypeScript type let a browser
host pause analysis slices while its play worker is active rather than running two unrestricted
searches.

The UI protocol is documented in `docs/protocol/ANALYSIS_PROTOCOL.md`. Machine-readable files are
`analysis-protocol.schema.json`, `time-control.schema.json`, `resource-budget.schema.json`, and
`analysis-types.ts`; `open-shogi-cli analysis` is the minimal reference client.

## Opening-book provenance, content, and behavior

No existing engine book, `standard_book.db`, commercial book, or license-unclear file was used.
The book uses only the approved AobaZero public-domain 100-game sample, Phase 3 train opening
statistics, and a fresh MultiPV 32 Apery view of the same approved 10,000 unique positions.
Source details and hashes are in `docs/provenance/OPENING_BOOK_V2.md`.

The bounded teacher run used one worker, four teacher threads, 1,024 MiB hash, 25,000 nodes, and a
16 GiB working-memory ceiling. Its nine-position benchmark had p95 11 ms and peak process-tree RSS
2,408,628,224 bytes. All 10,000 positions completed with zero quarantine. The label artifact
SHA-256 is `03f5375f09b167a12a4297cd2a404f0fb95a91edd2f6cdeabc955c14c7ce5ea0`;
manifest SHA-256 is `57f70b734f8c3eab4bd6c0ae2c439eae3373baf4f1b304d4fb32a17bb0e726fb`.

The serious book uses `open_shogi_opening_book/v2`, maximum 40 plies, minimum sample count 2,
maximum teacher loss 80 cp, and build version `openshogiai-book-v2-20260821-multipv32`. It contains
424 canonical positions and 448 candidates: 432 `ibisha`, 4 `ibisha-vs-furibisha`, and 12 other.
The deterministic gzip is 69,423 bytes, SHA-256
`3925fb49cbd9dfaf62f66cb92ebb06116a2cc63d05113ec4ee9490b951f34b85`; an independent rebuild was
byte-identical. The start-position strict move is `2g2f`, with sample count 17 and teacher score
93 cp at depth 4 / 25,112 nodes.

Book hits are validated for artifact/record checksum, schema, provenance, canonical position,
rule profile, sample threshold, teacher loss, and legal move. Maximum-strength mode selects the
strongest safe candidate deterministically; variety is not mixed into the default. A hit reports
`source=book`, returns immediately, consumes no normal search budget, and falls back to search for
absent, corrupt, unsafe, or illegal entries.

Strict-book versus no-book used 20 equal-clock games, all finished as draws and with zero illegal
moves. The book side had 10 hits over 580 decisions (1.72%, 0.5 hit/game), 570 searches, and 5,703
ms search time versus 580 searches and 5,803 ms without book. The measured counterfactual saving
was about 100 ms total, 5 ms/game, or 10 ms/hit under the 10 ms/move setup.

## Ibisya profiles

The default is `ibisha_strict`. Style is applied only in the opening book/policy layer through a
configurable opening ply; legal move generation is unchanged.

- `unrestricted`: strongest validated safe book candidate;
- `ibisha_preferred`: safe Ibisya/Ibisya-versus-Furibisha candidate first, then strongest safe
  unrestricted book candidate;
- `ibisha_strict`: only safe Ibisya/Ibisya-versus-Furibisha book candidate, otherwise unrestricted
  legal search.

Classification uses the opening sequence/position structure rather than a single rook-file test.
Opponent Furibisha and Ibisya-versus-Furibisha records are supported. Rook movement after the
opening is never made illegal.

Small exploratory profile Arenas showed preferred versus unrestricted at 1-2-15 among 18 finished
games (2 capped) and strict versus unrestricted at 1-2-16 among 19 finished games (1 capped).
Both sides recorded about 140 book hits and zero illegal moves. These results do not establish a
profile strength difference.

## Model campaign, labels, and self-play decision

The campaign trained at most one MPS worker at a time, below the two-worker and 16 GiB limits. It
used exactly 10,000 unique approved teacher positions, not the allowed 250,000 maximum. No new data
source was acquired and no test row was used for selection. The residual baseline was rebuilt
twice byte-identically at SHA-256
`b90db1c8b41a58fbc6d8d0e45f920da66fd3fb932ed3b1614495aa268162529d`.

| Run | Validation objective | Optimized MAE | Time | Peak RSS | Float32 SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| pure 2x128 | 0.6890455457 | 829.2856 cp | 115.263 s | 918,011,904 | `b42b09304fef239668c7fb5f0e5d652822e75daf715fdb843be52b4aeb410967` |
| residual 2x64 | 0.3901236407 | 747.1778 cp delta | 109.598 s | 915,259,392 | `cfe84ee0b74d48daa87af95cf70defddad9d991671adfe03368422056ac9a984` |
| residual 2x128 | 0.3896368305 | 746.1339 cp delta | 113.305 s | 906,117,120 | `b577f7c98dad15c76af21146831169881068f8bf5a67a35409ce15ba1a0dede1` |

2x64 has 150,722 training / 150,657 exported parameters and a 602,736-byte float artifact. 2x128
has 309,634 / 309,505 parameters and a 1,238,128-byte float artifact. Search-context inference was
about 13.8k calls/s for 2x64 and 6.95k calls/s for 2x128.

At 500 nodes and depth cap 4 on all 1,894 validation positions, teacher-bestmove agreement was:
19.905% handcrafted experimental, 10.348% initial pure 2x64, 9.979% g1r4 pure 2x64, 10.665% pure
2x128, 21.911% residual 2x64, 22.175% residual 2x128, and 20.961% composite g1r4. This is an
engine move-ranking benchmark; it is distinct from the training-only binary policy head.

Only residual 2x128 touched the held-out test after selection: objective `0.4406921195`, optimized
delta MAE 808.6469 cp, and auxiliary policy accuracy 0.619361. Raw final-score calibration and
full artifact identities are recorded in `MODEL_CARD.md`.

Because residual 2x128 failed the overall gate, the bounded larger self-play prerequisite was not
met. Phase 9 started **0 generations and 0 self-play games**; therefore there are no generation
promotion results, no replay-buffer expansion, and no automatic campaign beyond the requested
bounds.

## Commands

Native casual, clock, move-time, and diagnostic-node examples:

```sh
target/release/open-shogi-cli play --human black --profile overall-champion
target/release/open-shogi-cli play --human black --profile overall-champion \
  --black-time-ms 600000 --white-time-ms 600000 --byoyomi-ms 10000 \
  --black-increment-ms 0 --white-increment-ms 0 --safety-margin-ms 50
target/release/open-shogi-cli play --human black --profile overall-champion --movetime-ms 20000
target/release/open-shogi-cli play --human black --profile overall-champion --nodes 50000
target/release/open-shogi-cli analysis
target/release/open-shogi-cli opening-book verify \
  --book artifacts/opening/open-shogi-opening-v2.jsonl.gz
```

USI examples:

```text
usi
setoption name ModelKind value overall-champion
setoption name TimeSafetyMarginMs value 50
setoption name OpeningBookPath value artifacts/opening/open-shogi-opening-v2.jsonl.gz
setoption name OpeningProfile value ibisha_strict
setoption name OpeningMaxPlies value 40
isready
position startpos
go movetime 20000
go btime 600000 wtime 600000 byoyomi 10000 binc 0 winc 0
go nodes 50000
go
stop
quit
```

The actual book smoke emitted:

```text
info string source book profile ibisha_strict samples 17 teacher_cp 93 teacher_depth 4 teacher_nodes 25112 classification ibisha
bestmove 2g2f
```

Rebuild and gate commands are in `docs/reproducibility.md`. Analysis command examples and all
request fields are in `docs/protocol/ANALYSIS_PROTOCOL.md`.

## Verification

Final `make check` passed on Apple M5 / 24 GB, macOS 26.5.1 arm64, Rust 1.94.1, Python 3.12.13:

- repository boundary, license-scope, and provenance checks;
- Rustfmt, Ruff format, Clippy `-D warnings`, and Ruff lint;
- 157 CLI tests, 102 core tests plus 51 core integration/property tests, 25 USI tests, 9 Wasm
  tests, and 446 Python tests;
- workspace build and deterministic Wasm binding regeneration check.

Focused evidence also covered fake-clock 20-second hard deadline, real deadline smoke, clock
properties, native clock debit/increment, analysis cancellation/switch/invalidation/restart,
transposition reuse, corrupt/illegal book rejection, Ibisya policy behavior, overall-gate
derivation, tactical regression, and native/Wasm protocol schema conformance.

## Commits and remaining limitations

Logical implementation commits preceding this report:

- `6d2d97a` — analysis, time control, champion defaults, opening policy;
- `b52c9b0` — residual model target semantics;
- `079846e` — residual builder path canonicalization;
- `4dc402c` — predeclared overall-champion gate;
- `9dabc15` — distinct preferred opening behavior;
- `3c98691` — predeclared opening-book teacher run;
- `eb36e68` — full native clock fields, clock-state debit, resource schema, and numeric TT reuse
  evidence.

The report/documentation commit is necessarily listed by the final handoff rather than
self-referencing its own hash.

Remaining limitations:

- the 424-position book is reproducible but narrow; observed strict hit rate was only 1.72%;
- equal-wall-clock samples are small, and capped games are not silently relabeled as draws;
- residual validation diagnostics improved, but equal-time play remained below the incumbent;
- analysis search is single-threaded; cross-worker pausing is a host responsibility expressed by
  the resource protocol;
- cache persistence restores completed evidence, not recursive stacks;
- generated weights/books and the external teacher remain ignored local artifacts; model-weight
  licensing is still pending review;
- no larger self-play may begin without a future non-baseline overall-gate pass and explicit user
  approval for the next phase.
