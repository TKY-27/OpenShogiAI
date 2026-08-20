# Engine interfaces

## Compatibility policy

Text crossing a process, language, storage, or browser boundary must use a documented,
versioned representation. Rust struct layout and Zobrist values are internal implementation
details and are not persistent identifiers.

Unknown schema versions and malformed input are rejected. Parsers do not guess repairs for
untrusted data.

Play clocks use `open_shogi_time_control/v1` across Rust, native play, USI mapping, and Wasm.
Persistent analysis uses `open_shogi_analysis/v1`; see
[`protocol/ANALYSIS_PROTOCOL.md`](protocol/ANALYSIS_PROTOCOL.md).
Opening data uses `open_shogi_opening_book/v2`; see
[`protocol/OPENING_BOOK_FORMAT.md`](protocol/OPENING_BOOK_FORMAT.md). Native, USI, and Wasm book
hits report `source=book` and do not consume the normal search budget.

## Phase 1 rule boundary

`engine/core` owns:

- validated `Position`, `Piece`, `Hand`, `Move`, and `Game` values;
- legal move generation and make/unmake;
- deterministic but internal Zobrist hashes;
- SFEN positions, USI moves, CSA V3.0 records, and perft traversal.

Adapters own:

- standard input/output and command lifecycles;
- clocks, cancellation, tournament operations, and UI state;
- files, network acquisition, provenance, and cross-language schemas.

## Canonical SFEN

Canonical output has four single-space-separated fields:

```text
<board> <b|w> <hands|-> <one-based move number>
```

The board uses nine `/`-separated ranks, file `9` through `1`. Hand output is Black then White,
in `R B G S N L P` order, with a decimal count before a letter only when greater than one.
The parser requires this unique canonical hand ordering and rejects duplicate piece tokens,
promoted hand pieces, kings in hand, zero counts, invalid inventories, nifu, dead unpromoted
pieces, and missing or duplicate kings. Serialization always emits the same canonical order.

SFEN contains no repetition history. Loading SFEN begins a new `Game` history at that state.

## USI move text

- Normal: `7g7f`
- Promotion: `2b3c+`
- Drop: `P*5e`

Move parsing validates only the notation shape and board coordinates. Legality is validated
against a `Position` before application.

## CSA record text

The Phase 1 parser accepts bounded CSA V3.0 text needed for records. Other versions are
rejected until their specification is separately audited:

- version, player names, and `$` metadata;
- `PI` standard initial position and validated four-byte piece-removal entries;
- explicit `P1`–`P9`, `P+`, `P-`, and side-to-move position records;
- signed move lines and ignored time lines;
- comma-separated V3 multi-statements;
- typed standard `%` special endings.

Every move is replayed through the same Rust legality implementation used by search and UI.
Errors retain a source line number. The canonical writer emits an explicit UTF-8 declaration,
V3.0, a complete position, metadata, legal moves, and the typed ending. Because CSA has no
field for an initial SFEN move number, canonical output requires `initial_position.move_number`
to be one instead of silently discarding a different value.

`CsaGame::result_validation` distinguishes three states. Replay-proven `TSUMI`, fourfold
`SENNICHITE`, and `OUTE_SENNICHITE` are marked `Verified`. An earlier agreed repetition restart
and results that depend on clocks, resignation, player agreement, an entering-king declaration,
tournament limits, or other external evidence are preserved but marked `ExternalCondition`;
an absent terminal code is `Missing`. A source-ingestion pipeline must not relabel the latter
two states as rules-verified. The canonical writer replays and checks this classification so
its output remains parseable under the same contract.

Phase 3 retains source bytes and their SHA-256 separately; parsing and canonicalization never
replace provenance.

## Perft API

`perft` and `perft_divide` return `Result` and fail explicitly if a generated move cannot be
applied or any exact `u64` counter would overflow. The local CLI limits depth to five as a
defensive runtime bound; library callers remain responsible for choosing tractable positions
and depths.

## Phase 2 search boundary

`engine/core` exposes a deterministic search configuration whose evaluation terms and search
ladder switches can be changed independently. `SearchLimits` carries depth, node, and move-time
limits; a separate cancellation token permits `stop` without coupling the core to threads or
standard I/O. Node limits are exact, time and external cancellation are checked at node
boundaries, and a legal root fallback remains available when a bounded search stops before
completing depth one.

The transposition table is direct-mapped and bounded by an entry count. A Zobrist match is not
sufficient for a hit: the stored complete `Position` must also match, making collisions affect
replacement/performance rather than correctness. Byte and MiB conversion use the actual Rust
slot size instead of an estimated structure size.

Search scores are signed from the side-to-move perspective. Values beyond the documented mate
threshold encode mate distance internally; the USI adapter translates them to `score mate`.
Search does not persist `Position` hashes or Rust struct bytes as an external format.

## USI process boundary

Run the protocol adapter with:

```sh
cargo run --locked -p open-shogi-cli -- usi
```

The bounded parser supports `usi`, `isready`, `setoption`, `usinewgame`, `position startpos
moves`, `position sfen`, `go`, clock fields, `go movetime`, `go nodes`, `go depth`, `go
infinite`, `stop`, `gameover`, and `quit`. Only protocol output is written to stdout. Search
runs on one worker thread so input can cancel it; replacing a position or changing an option
cancels and joins the previous search before committing new state.

## Arena report v1 (legacy)

The arena atomically writes `arena-report.json` with schema
`phase2_arena_report/v1`. The exact top-level fields are:

```text
schema
run { seed, gameLimit, engine, gitCommit, startedAt, completedAt }
metrics {
  games, finishedGames, searchWins, draws, nodesPerSecond, averageDepth,
  ttHitRate, cutoffRate, pruningRate, millisecondsPerMove, peakMemoryBytes,
  illegalMoves
}
games[] { id, black, white, result, moves, csaPath }
```

Game results are `black_win`, `white_win`, `draw`, or `max_plies`. `cutoffRate` is beta-cutoff
nodes divided by searched nodes. `pruningRate` is the number of post-filter move candidates
skipped after a beta cutoff divided by all post-filter move candidates generated; TT-returned
nodes and horizon leaves that generate no moves are outside that denominator. UTC timestamps
use ISO-8601. A partial report has `completedAt: null`; peak resident memory is `null` when it
was not measured by a platform wrapper. The browser importer rejects unknown/missing fields,
non-finite or out-of-range metrics, duplicate game IDs, inconsistent aggregates, unsupported
player labels, non-ISO timestamps, files over 5 MiB, and more than 10,000 games.

`arena.state` is a private, versioned resume artifact rather than a public interchange format.
Resume requires the complete behavioral configuration signature to match and never guesses
through a malformed or mismatched state. One process holds an exclusive standard-library file
lock for the output directory, preventing concurrent writers from corrupting its state,
reports, or predictable temporary paths.

The arena and benchmark JSON schemas are metrics interchange artifacts, not complete
provenance manifests. The checked-in Phase 2 report is authoritative for the official
comparison and binds its clean Git commit, full resolved configuration, runtime versions,
dataset non-applicability, external peak-RSS measurement, and SHA-256 artifact hashes.

## Phase 4 teacher and model schemas

The external-teacher boundary uses closed, bounded JSON/YAML schemas:

- `phase4_teacher_config/v1` describes the executable, expected official identity, USI
  options, fixed-node limits, concurrency, timeouts, and local evidence paths;
- `phase4_teacher_install/v2` records the official archive/tree, binary, evaluation, license,
  and build identities while retaining a strict legacy-v1 reader;
- `phase4_teacher_selection/v2` binds every canonical Phase 3 source field used for labeling;
- `phase4_teacher_benchmark/v2` carries raw searches whose metrics and identity are recomputed
  by the verifier;
- `phase4_teacher_label/v1` is one append-only label row, and
  `phase4_teacher_label_manifest/v2` binds the exact selection, benchmark, label JSONL,
  quarantine, teacher, parser, and Rust legality verifier identities.

Label readers reject unknown fields, duplicate position IDs, source mismatches, noncanonical
SFEN, non-finite scores, malformed mate values, illegal best moves/PVs, and broken artifact
digests. A returned MultiPV list is a teacher-returned contiguous rank prefix; it is not a
claim that every legal root move was returned.

Model configs use closed TOML schema version 1. Checkpoints use
`phase4_value_checkpoint/v1`; prediction rows use `phase4_value_prediction/v1`; model metadata
uses `phase4_value_model_metadata/v1`. The production join verifies the Phase 3 row, split,
stage, source, outcome, label identity, and optional replay provenance before tensor creation.

## OSAVAL01 model boundary

`OSAVAL01` is the project-owned portable model container used by Python, Rust, and future
WebAssembly inference. Its header declares little-endian container/feature/architecture
versions, quantization, input/hidden dimensions, layer count, enabled feature mask, payload
length, and payload SHA-256. The bounded payload contains either finite float32 weights or
symmetric per-layer int8 weights with finite positive scales. Loaders reject unknown flags or
versions, trailing bytes, invalid dimensions, non-finite values, checksum mismatch, oversized
artifacts, symlinks, and files that change during a verified read.

Rust machine interfaces are:

```text
phase5_model_inspection/v1
phase5_model_inference/v1
phase5_handcrafted_inference/v1
```

Neural inference rows bind artifact and payload identities, architecture/quantization, input
index and canonical SFEN, score, and elapsed nanoseconds. Handcrafted rows use the closed
profiles `handcrafted-baseline` and `handcrafted-experimental`. Timing is observational; score
and identity fields are the deterministic comparison surface.

## Arena report v2 and state v4

New arena runs emit `phase2_arena_report/v2`; the web importer still accepts v1 explicitly.
V2 is a closed schema with top-level `{schema, run, metrics, games}`. `run` binds the JSON-safe
seed, game limit, engine and lowercase commit identity, timestamps, initial SFEN, maximum
plies, config SHA-256, fixed node or move-time budget, logical player A/B configurations, and
the complete opening artifact identity. Neural players expose full model artifact/payload
SHA-256, size, architecture, and quantization; non-neural fields are null rather than omitted.

Each game binds color-assigned player labels, result, move count, contained CSA path, CSA
SHA-256 and size, plus A/B search and neural counters. The combined A/B search count cannot
exceed `moves + 1`. Aggregate metrics repeat those logical-player counters and add A/B wins,
legacy `searchWins`, draws, rates, memory, and illegal moves. The writer and browser validator
recompute game counts, A/B/color classification, all exposed counter sums, and exactly three
six-decimal metrics from those counters: nodes/second, average depth, and milliseconds/move.
Because v2 does not expose raw TT-hit, cutoff, or pruning event counts, those three rates are
validated only as finite values in the inclusive 0–100% range. Seed values are limited to
JavaScript's exact integer range so a JSON number preserves identity across Rust and browsers.

The v2 engine field must be the exact version plus captured A/B labels and budget string. The
browser parses the canonical initial-SFEN syntax accepted by the Rust arena boundary, including
board width, kings, material limits, immobile unpromoted pieces, double pawns, canonical hands,
side to move, and move number 1. These are structural checks, not proof of reachability,
check-state validity, or complete shogi legality. The browser also does not read or hash the
referenced CSA, model, or opening bytes; Rust replay and pipeline artifact verification supply
that evidence.

The browser file-import boundary accepts at most 5 MiB, and its schema parser accepts at most
10,000 games. The CLI keeps matching 5 MiB output and 10,000-game limits. Additionally, before
an arena starts, it rejects a requested game count whose deterministic worst-case report size
cannot fit the 5 MiB limit; an imported report that actually fits remains valid up to 10,000
games.

Private `phase2_arena_state/v4` is LF-only and binds the full configuration signature plus
canonical CSA identity for every completed game. Total and per-player search elapsed values are
persisted in nanoseconds so interrupted runs resume without millisecond-rounding loss; public v2
report fields retain their existing millisecond names and semantics. Older states are
intentionally rejected because they lack sufficient immutable evidence. Resume replays CSA,
verifies assignments and results, and refuses model/opening changes, missing files, symlinks, or
inconsistent counters.

## Phase 6 generation schemas

Phase 6 uses closed JSON records connected by full artifact hashes:

- start and planning: `phase6_start_positions/v1`,
  `phase6_start_position_validation/v1`, `phase6_selfplay_plan/v1`,
  `phase6_paired_arena_plan/v1`, and `phase6_challenger_training_plan/v1`;
- execution: `phase6_paired_job_state/v1`, `phase6_selfplay_manifest/v1`, and
  `phase6_arena_execution_manifest/v1`;
- data derivation: `phase6_position_evidence/v1`, `phase6_hard_positions/v1`,
  `phase6_replay_candidates/v1`, and `phase6_replay_buffer_manifest/v1`;
- decision/publication: `phase6_paired_arena_results/v1`,
  `phase6_paired_arena_analysis/v1`, `phase6_promotion_decision/v1`, and
  `phase6_model_registry/v1`.

Plans contain complete argv/path/count/resource contracts; the executor does not accept an
open-ended command. Self-play consumes train-only starts, while arena consumes validation-only
starts. Position evidence is derived by legal CSA replay and factual Phase 3 joins. Replay
retains split/provenance/generation/deduplication/capacity/deletion evidence and never feeds test
rows into training. Promotion decisions are recomputed from the exact policy and analysis
before registry mutation; `promoted`, `rejected`, and `inconclusive` are all terminal outcomes.

`phase6_generation_pipeline_state/v1` and `phase6_generation_manifest/v1` are reserved internal
schemas exercised only by isolated tests. They are not integrated into the canonical CLI flow,
and the bounded Phase 6 generation did not emit either artifact. `finalize-generation` validates
the promotion lifecycle and publishes the next registry revision; it does not create a
generation manifest. A future pipeline integration requires a separate implementation and
validation decision.

The model registry is append-only by revision, requires contained nonsymlink `OSAVAL01`
artifacts with matching metadata, enforces parent/lifecycle/evidence order, and records one
champion plus an optional active challenger. Rust terminal play resolves `champion` and
`challenger` through this validated registry rather than trusting a path alias.

Human-play decision logs use `open_shogi_play_config/v3` followed by
`phase6_human_decision/v1` rows. The v3 configuration identity-binds the shared time-control
schema, complete initial clock/byoyomi/increment fields, safety margin, evaluator, opening policy,
and artifact identities. A `phase6_human_publication/v1` marker atomically binds the canonical CSA
and decision log; recovery revalidates both. The registry-facing intake artifact is
`phase6_pending_human_review/v1`, so human choices never become labels automatically.

## Phase 7 developer-evaluation schemas

Phase 7 uses a closed TOML `phase7_evaluation_config/v1` and these closed JSON artifacts:

- `phase7_official_evaluation_plan/v1` binds the clean lowercase Git commit, immutable Rust
  engine and build receipt, Phase 6 registry revision and champion model, teacher config, two
  color-complementary human-play commands, output paths, and the no-auto-training decision;
- `phase7_official_games/v1` binds both canonical CSA files and decision logs to a receipt-
  validated `export-csa-jsonl` replay and its exact SFEN/move sequence;
- `phase7_teacher_analysis/v1` binds one recorded move position, the human/AI decision row,
  pinned teacher identity/options/search result, Rust PV-legality evidence, and a conservative
  diagnosis;
- `phase7_evaluation_report/v1` binds the plan, games manifest, all analysis artifacts,
  teacher and legality-validator identities, counts, classifications, and completion status;
- `phase7_hard_examples/v1` binds the report, fixed thresholds, ranked selected rows, omitted
  count, `pending_human_review` status, and `autoTrainingEligible: false`.

The fixed plan has exactly two games, human black then human white, with opening disabled,
10,000 nodes, depth cap 8, and maximum 256 plies. Teacher searches use the separately installed
Apery 2.0.0 process at 25,000 nodes and MultiPV 3. All returned PV moves are replayed through
the receipt-bound Rust legality CLI.

Diagnostics keep teacher move-choice regret separate from root-evaluation disagreement.
Human moves never become hard examples. An AI move outside returned MultiPV has no exact regret
unless the returned score prefix proves a lower bound; categories ending in `_candidate` remain
triage hypotheses. Selection never changes labels, replay, checkpoints, model artifacts, or the
registry, and no selected row is automatically trainable.

## Phase 3 source and acquisition schemas

`configs/data_sources.yaml`, the exact object catalog, and the evidence catalog use closed
schema version 1. YAML aliases, duplicate keys, unknown/missing fields, oversized files,
unbounded node/string counts, unsafe paths/URLs, and unpinned non-robots evidence are rejected.
Approval, machine-learning use, redistribution, robots policy, rate/concurrency, exact hosts
and paths, object size, adapter, and evidence are independent required decisions.

The acquisition `manifest.jsonl` is append-only schema version 1. `completed` events contain
source/object identity, URL, retrieval time, SHA-256, size, response metadata,
content-addressed path, original format, license evidence snapshots, and usage decisions.
`partial` events contain source/object identity, URL, time, saved path, size, partial SHA-256,
and validators. Resume requires a syntactically strong ETag and a matching partial hash;
Last-Modified is retained as provenance but is not a Range validator.

## Rust CSA export boundary

`open-shogi-cli export-csa-jsonl` accepts a non-symlink directory, non-overwriting output path,
and `--max-games` from 1 through 10,000. Each sorted `.csa` candidate produces exactly one LF-
terminated `phase3_csa_export/v1` JSON object:

```text
ok       { schema, status, inputFile, normalizedCsa, initialSfen, positionSfens,
           usiMoves, blackName, whiteName, terminalReason, outcome, resultValidation }
rejected { schema, status, inputFile, reason }
```

Input CSA is limited to 1 MiB, each physical JSONL record to 32 MiB including LF, and total
output to 1 GiB. Accepted games are reparsed, legally replayed, and emitted with a position
sequence exactly one element longer than the USI move sequence. Only explicit draw results are
labeled draw; external-condition results retain that classification.

`repetition_outcome_from_moves(initial, moves)` classifies a complete legal history without
stopping at an earlier ordinary repetition. It exists for record validation and does not alter
the live `Game` terminal-state contract.

## Normalized dataset and opening schemas

The deterministic gzip artifacts use empty gzip filename, `mtime=0`, sorted JSON keys, compact
UTF-8 JSON, and one LF per row:

- `phase3_game/v1`: raw and canonical CSA identity, SFEN/USI sequence, outcome and validation,
  players/optional ratings, date/timezone, source/retrieval, rights decision, and game split;
- `phase3_position/v1`: game/raw identity, split, position index, current/next SFEN, move label,
  outcome, side, ply counts, eligibility, and terminal-tail flag;
- `phase3_normalization_report/v1`: input/inclusion/exclusion, split, short/long, and within/
  cross-game position-duplication statistics;
- `phase3_dataset_manifest/v1`: source decision, full normalization config, raw/canonical and
  evidence identities, counts, and artifact hashes.

`phase3_game_split/v1` computes SHA-256 over UTF-8 public salt, one NUL byte, and canonical game
hash bytes. The first 64 bits select a basis-point bucket. This is game-only and addition-
stable, so every position of one game remains in one split.

`phase3_opening_sqlite/v1` consumes only `train` + `eligible` position rows. It records full
state/move counts, actor wins/losses/draws/unknown, black/white wins, full and remaining ply
sums, and per-source counts. Queries and deterministic `phase3_opening_export/v1` derive score
rate, decisive win rate and Wilson 95% bounds, side-specific decisive rates, averages, and
apply `min-count` at query time. Dataset directories, SQLite files, and exports are published
atomically without overwriting an existing destination.

The portable dataset manifest deliberately does not infer repository or runtime state.
`PHASE_3_REPORT.md` is the authoritative external binding from its registry/catalog and
acquisition-manifest hashes to the clean implementation commits, runtime versions, observed
artifact hashes, and independent-output comparison.

## Phase 8 browser Worker boundary

The page creates one module Worker; only that Worker instantiates `engine/wasm` and runs search.
Requests and responses use closed `open_shogi_worker_response/v1` envelopes with a positive
integer request ID and one of `initialize`, `reset`, `load-model`, `unload-model`, `play-move`,
or `search`. Unknown keys, kinds, scalar types, and unbounded values are rejected on the
TypeScript boundary before Rust dispatch. Failures contain only a bounded public code and
message; successful protocol pseudo-kinds are invalid.

`open_shogi_browser_snapshot/v1` contains engine identity, canonical initial/current SFEN,
side and move number, exactly 81 position-bound board entries, exact ordered hand counters,
bounded legal moves and move history, terminal state, and evaluator/model summary. Board array
index zero is 9一 and index 80 is 1九; each nonempty entry repeats and must match its exact square.
Restore accepts at most 512 USI moves and 16 KiB of move-history JSON.

`open_shogi_browser_search/v1` records the fixed profile (`eco`, `balanced`, or `quality`),
selected evaluator, root perspective, best move, score, depth, nodes, elapsed time,
termination, statistics, and one to three ranked lines. The first line is the ordinary bounded
search result. Additional lines are smaller per-root comparison searches and are not a claim of
equal search depth. Search is cancelled by terminating the Worker; a new Worker is initialized
from the last accepted snapshot history.

`open_shogi_browser_model/v1` is a read-only summary of a successfully parsed `OSAVAL01`
artifact. Browser model input is 1 byte through 16 MiB. JavaScript computes the whole-artifact
SHA-256 and passes it as an expected identity; Rust recomputes that identity, validates the
container/payload checksums, architecture, finite values, activation, and quantization, and
retains the evaluator only after complete validation. Model bytes are not persisted or sent to
a network service.

## Error and output streams

Human CLI commands write successful results to stdout and diagnostics to stderr. The Phase 2
USI command loop reserves stdout exclusively for protocol messages. Debug formatting, the
arena resume state, and terminal board rendering are not stable machine formats.

Phase 3 machine outputs use the schemas above. Human-readable diagnostics are not stable
machine formats.
