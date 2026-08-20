# Reproducibility

## Versioned inputs

- `rust-toolchain.toml` selects Rust stable and required components.
- `.python-version`, `pyproject.toml`, and `uv.lock` select Python 3.12 and Python tools.
- `configs/project.toml` holds the project schema, default seed, resource budget, and storage
  paths.

## Standard verification

```sh
make check
```

The target uses project-local caches and locked dependency graphs. It runs environment checks,
format checks, linting, unit/property tests, and host production builds.

The deterministic WebAssembly build requires the Rust target and the pinned local
`wasm-bindgen-cli` described in the root README:

```sh
rustup target add wasm32-unknown-unknown
make wasm-build
```

## Experiment records

Each training, labeling, arena, and self-play run must record:

- a stable run identifier and UTC start time;
- Git commit and dirty-tree state;
- complete resolved configuration and its SHA-256;
- all random seeds;
- input dataset or position-manifest hashes;
- runtime, compiler, dependency, device, and OS versions;
- checkpoints, resume lineage, metrics, timing, and peak memory;
- hashes and schema versions of produced artifacts.

The test split must remain unavailable to model selection and training processes.

## Phase 2 deterministic evidence

Fixed-node search is the reproducible comparison mode. The seed controls every random player,
the arena alternates player colors by game ID, and the behavioral configuration is included
in the resume signature. Supply the current commit explicitly so the report never shells out
or guesses repository state:

```sh
cargo run --locked -p open-shogi-cli -- arena \
  --games 20 --player-a search --player-b random \
  --nodes 2000 --max-plies 128 --seed 20260729 \
  --git-commit "$(git rev-parse HEAD)" \
  --output-dir artifacts/arena
```

The commit value alone does not prove that the worktree was clean. The checked-in
`make phase2-arena` target refuses a dirty worktree before using `HEAD`; direct CLI callers
must record dirty status and a diff hash separately or run from a clean commit.

Use `--resume` only with the same output directory and complete behavioral configuration.
The command rejects a nonempty output directory without `--resume`, a missing resume
directory, a malformed state, noncontiguous game IDs, or a signature mismatch.

Wall-clock fields, nodes per second, and externally measured resident memory naturally vary
by machine and load. Reproducibility checks compare the ordered game identities, players,
results, move counts, and CSA bytes for the same fixed-node configuration; they do not require
timestamps or timing-derived metrics to be byte-identical.

The arena and benchmark JSON files carry portable metrics, not a complete provenance
manifest. `PHASE_2_REPORT.md` is the authoritative record for the checked-in Phase 2 run and
adds the full resolved configuration, clean/dirty status, runtime and host versions, dataset
non-applicability, external peak RSS, and SHA-256 hashes.

## Phase 3 deterministic data evidence

The registry, exact source/evidence catalogs, split salt, limits, normalizer, and opening
schema are versioned inputs. Network acquisition is intentionally separate from deterministic
processing: live `robots.txt`, retrieval timestamps, and response metadata can change, while
accepted raw bytes are fixed by their manifest SHA-256 identities.

Run acquisition only from a clean implementation commit. The checked-in targets refuse to
attribute generated evidence to a dirty tree:

```sh
make phase3-validate-registry
make phase3-dry-run
make phase3-acquire
make phase3-normalize
make phase3-build-opening
make phase3-export-opening
```

`phase3-acquire` is limited to the exact approved catalog and 100 objects. A new process is
required for every acquisition run so the live access and pinned rights evidence are fetched
again. A repeated run verifies local completed objects and should report 100 `skipped-url`
results without appending duplicate completion records.

To check normalization reproducibility without modifying the first output, use a second
ignored processed root with the same committed inputs and split salt:

```sh
make phase3-normalize PHASE3_PROCESSED_ROOT=data/processed/phase3-repro
make phase3-build-opening PHASE3_PROCESSED_ROOT=data/processed/phase3-repro
make phase3-export-opening PHASE3_PROCESSED_ROOT=data/processed/phase3-repro
```

Compare the dataset manifest's artifact SHA-256 values and the opening database/export
SHA-256 values. Deterministic gzip writers fix headers, ordering, JSON serialization, and
newlines. Game-level split assignment hashes the public salt and canonical game hash, so
adding future approved games cannot move existing games between train, validation, and test.

`PHASE_3_REPORT.md` records the clean implementation commit, registry/catalog hashes,
acquisition-manifest hash, observed counts, artifact hashes, tool/runtime versions, and the
result of the independent-output comparison. Raw and processed objects remain under ignored
paths and are never committed.

## Phase 4 teacher and training evidence

The teacher is an independent USI process under ignored `local/teacher/` storage. Recreate and
verify the pinned Apery installation without using global packages or administrator access:

```sh
./scripts/setup_teacher_apery.sh
cargo build --locked --release -p open-shogi-cli
PYTHONPATH=training uv run --frozen python -m open_shogi_training.labeling audit-label-set \
  --config configs/teacher/apery-v2.0.0.yaml \
  --positions data/processed/phase3/aobazero-no-noise-pd-sample100/positions-00000.jsonl.gz \
  --dataset-manifest data/processed/phase3/aobazero-no-noise-pd-sample100/manifest.json \
  --benchmark-report artifacts/phase4/teacher/benchmark-v3.json \
  --output-dir artifacts/phase4/teacher/labels-v2
```

Setup revalidates the official archive tree, exact binary, evaluation files, licenses, and
install manifest. Runtime code hashes and snapshots the verified executable before spawning
it. Selection, benchmark, label, quarantine, and manifest files have closed schemas and exact
SHA-256 bindings. Resume recomputes the existing JSONL prefix and never silently adopts a
nonempty label file without its original binding evidence. Returned best moves and every PV
move are replayed by the release Rust rule engine.

Model configuration is split across `configs/features/`, `configs/models/`, and
`configs/training/`. The production dataset join requires exactly 10,000 teacher labels,
recomputes stage and game-level split identity from Phase 3 records, and keeps test evaluation
an explicit operation separate from training and model selection. Typical bounded checks are:

```sh
make model-validate
make model-describe
make train-overfit
make train-smoke
PYTHONPATH=training uv run --frozen python -m open_shogi_training.models validate \
  --checkpoint artifacts/phase4/models/value-v0-initial-205dc84/best.pt \
  --labels artifacts/phase4/teacher/labels-v2/labels.jsonl \
  --label-manifest artifacts/phase4/teacher/labels-v2/manifest.json \
  --positions data/processed/phase3/aobazero-no-noise-pd-sample100/positions-00000.jsonl.gz \
  --dataset-manifest data/processed/phase3/aobazero-no-noise-pd-sample100/manifest.json
```

Checkpoints carry the resolved config hash, dataset identities, epoch/global step, optimizer
and RNG state, runtime identity, and bounded tensor tree. Resume rejects incompatible or
malformed state. Training logs record seed, device, thread settings, elapsed time, resident
memory, and MPS memory when available. `PHASE_4_REPORT.md` records the observed bounded runs,
artifact hashes, and exact validation/test measurements.

## Phase 5 model and arena evidence

`OSAVAL01` is a little-endian, checksummed inference artifact with a fixed magic value,
feature/architecture version, quantization kind, dimensions, feature mask, payload hash, and
bounded payload. Export is create-only and emits float32, symmetric per-layer int8, and a
metadata JSON identity. Python reference inference and Rust inspection/inference can be
rechecked with:

```sh
uv run --frozen pytest -q tests/python/models/test_cross_runtime.py
cargo run --locked -p open-shogi-cli -- model inspect \
  --model artifacts/phase4/models/value-v0-initial-205dc84/export/value_v0.f32.osaval
```

Arena comparison uses fixed nodes, paired color assignment, deterministic start identity, and
immutable model/opening bytes. Public reports use `phase2_arena_report/v2`; private resume state
uses `phase2_arena_state/v4` and is rejected unless the complete run signature and all existing
CSA identities still match. State v4 retains total and per-player search elapsed time in
nanoseconds; the public v2 report keeps its millisecond fields and performs the conversion only
when the completed report is rendered. `artifacts/phase5/arena/227680f/arena-manifest.json`
binds the 100 reports and 200 games used by `PHASE_5_REPORT.md`. Timing metrics are observations
rather than byte-reproducible outputs.

The exact historical Phase 5 verification command is recorded in `PHASE_5_REPORT.md`. It must be
run from the clean commit named by the frozen manifest with the ignored artifact tree available;
current code must not relabel the historical evidence.

## Phase 6 generation evidence

The bounded generation controller writes immutable start, self-play, evidence, replay,
training, arena, analysis, decision, and registry artifacts. Every operation revalidates its
upstream SHA-256 identities. Paired jobs are resumable at game-pair granularity; completed
reports and CSA files are rehashed and replayed, while invalid attempts are quarantined with
their observed identities. Concurrency is two workers, the configured ceiling is 8 GiB, and
the validated smoke generation contains exactly 40 self-play and 40 arena games.

The canonical operation order is:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.selfplay validate-config \
  --selfplay-config configs/selfplay/phase6_smoke.toml \
  --promotion-policy configs/generation/phase6_promotion.toml
PYTHONPATH=training uv run --frozen python -m open_shogi_training.selfplay --help
```

Then follow the exact subcommands and artifact paths recorded in `PHASE_6_REPORT.md`: build and
validate start positions, initialize the registry, plan/execute paired self-play, build factual
evidence, reuse already labeled hard positions, build replay, train/export/register the
challenger, plan/execute/collect/analyze paired arena, recompute the promotion decision, and
finalize the generation. At the 10,000-label cap, no supplemental teacher plan may be run.
Promotion is a deterministic function of the versioned policy and exact arena analysis; the
decision is rederived immediately before registry mutation. Human games remain isolated as
`phase6_pending_human_review/v1` and never enter training automatically.

The authoritative bounded run is `generation-1r4`, bound to clean commit
`1796d8750696ccddcd734506407b11c053e59e44`; its final registry is
`artifacts/phase6/registry-r4/model-registry.json`. A future generation must create fresh plans
from its own clean commit instead of relabeling or mixing the excluded r1/r2/r3 local roots.

## Phase 7 bounded developer evaluation

The fixed configuration is `configs/evaluation/phase7_official.toml`. Planning requires a clean
commit and resolves the active content-addressed Rust engine build plus the authoritative Phase 6
registry/champion. Each output root is create-only; use a new root rather than relabeling or
overwriting an earlier attempt.

```sh
make phase7-validate-config
make phase7-plan PHASE7_OUTPUT_ROOT=artifacts/phase7/<new-run> \
  PHASE7_GIT_COMMIT="$(git rev-parse --verify HEAD)"
make phase7-commands PHASE7_OUTPUT_ROOT=artifacts/phase7/<new-run>
# Run both exact interactive commands printed above.
make phase7-prepare-games PHASE7_OUTPUT_ROOT=artifacts/phase7/<new-run>
make phase7-analyze PHASE7_OUTPUT_ROOT=artifacts/phase7/<new-run>
make phase7-curate PHASE7_OUTPUT_ROOT=artifacts/phase7/<new-run>
make phase7-verify PHASE7_OUTPUT_ROOT=artifacts/phase7/<new-run>
```

`prepare-games` independently Rust-replays both CSA records and binds each move to the atomic
human-play decision log. `analyze` uses the pinned teacher and analyzes every recorded move
position within the configured process-tree memory and call ceilings; it does not sample or
silently skip positions. Verification reopens every referenced artifact, validates process
receipts and identities, rechecks teacher and Rust legality evidence, and rederives the report
and hard-example selection. Creating or executing a run requires the clean current HEAD recorded
by its plan. Read-only verification instead validates the immutable historical build receipt and
recorded commit directly, so a completed run remains verifiable after the repository advances.

The authoritative bounded local run is under
`artifacts/phase7/bounded-official-v4`, bound to clean commit
`a277e01e5923454e86153ae952eae91771a227ed`. Its report SHA-256 is
`e7a8e57ae321b3f623d3d59ce6cbcb6664d326602b6c417f06dc2209da2ec819`.
The two games were deliberately terminated by human resignation after exercising both side
paths. Reproducing the workflow does not require reproducing wall-clock time or the human input,
and these two outcomes are not strength evidence. The checked-in `PHASE_7_REPORT.md` records the
exact inputs, observations, hashes, and interpretation limits.
