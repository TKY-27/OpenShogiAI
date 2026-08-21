# Exact execution prompt — OpenShogiAI Phase 10

You are GPT-5.6 Luna. Use `reasoning=max`. Work only in the repository root containing this prompt.
Do not spawn any subagent. This is a bounded implementation and experiment-execution task, not an
architecture or policy-design task.

## Authority and first action

Start from the clean commit containing this prompt and the frozen Phase 10B controls. Create the
branch `codex/phase10-execution`. Do not push. Before any edit, acquisition, normalization, training,
teacher call, Arena, or self-play, run:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10 verify --root .
```

If verification fails, stop and report the exact mismatch. Do not regenerate hashes and do not
repair a frozen control. Read the following files completely:

- `PHASE_10_FROZEN_PLAN.md`
- `configs/phase10/dataset-mixture.yaml`
- `configs/phase10/curriculum.yaml`
- `configs/phase10/split-policy.yaml`
- `configs/phase10/experiment-matrix.yaml`
- `configs/phase10/resource-budget.yaml`
- `configs/phase10/statistical-gates.yaml`
- `configs/phase10/frozen-controls.sha256`
- `PHASE_9_REPORT.md`, `PHASE_10A_REPORT.md`, `MODEL_CARD.md`, and `DATASET_CARD.md`

The hashed files are authoritative. This prompt is self-contained so that missing chat context never
grants extra discretion.

## Decisions you may not change

You must not change, bypass, reinterpret, or regenerate any of the following:

- model architecture, feature set, parameter budgets, output scale, variant count, or eligibility;
- source acceptance, license decision, exact artifact subset, source role, sampling weight, maximum
  contribution, attribution requirement, or release constraint;
- split salt, split ratios, protected grouping, canonical identity, dedup priority, holdout identity,
  holdout inspection count, or source-held-out rule;
- dataset rung size, target semantics, loss kind/weight, early-stop rule, Arena size/configuration,
  statistical method, threshold, 48% self-play gate, or 55% final objective;
- resource limits, disk floor, time ceiling, or artifact-retention policy;
- overall champion, model registry, promotion policy, or promotion state;
- any file listed in `configs/phase10/frozen-controls.sha256`.

Do not promote a model and do not mutate a registry even if the 55% gate passes. Do not use a
residual or composite evaluator to satisfy a gate. Do not initialize from third-party weights. Do
not add handcrafted score at runtime. Do not inspect a public-test holdout.

If implementation reveals that a frozen decision is infeasible or ambiguous, stop and report a
blocked result. You have no authority to redesign it.

## Frozen source ledger

The only active external population is `aobazero-no-noise-exact100`, limited to the exact catalog
in `configs/data_source_objects/aobazero_no_noise.yaml`. It may supply legally replayed positions,
verified factual WDL, approved non-test teacher joins, ranking labels, validation, and
opening/start-position candidates. Raw AobaZero `v` annotations have unknown semantics and
perspective and must have value mask zero.

`aobazero-no-noise-w4745-sample` is already contained in exact-100. It is a format-audit fixture
with weight zero and may not become a second training copy.

The following artifacts are deferred, have weight/cap zero, and may not be downloaded or used:

- `aobazero-live-no-noise-sample-index`
- `aobazero-daily-public-records`
- `dlshogi-gct-hcpe3-selfplay`
- `dlshogi-gct-floodgate-hcpe3`
- `dlshogi-gct-taya36-hcpe3`
- `dlshogi-gct-model-taya36-hcpe3`
- `dlshogi-gct-aobazero-hcpe`
- `dlshogi-gct-floodgate-play-hcpe`
- `dlshogi-gct-taya36-rl-hcpe`
- `dlshogi-gct-yaneura36-rl-hcpe`
- `dlshogi-gct-suisho-teacher-hcpe`
- `nodchip-shogi-hao-depth9`
- `nodchip-tanuki-nnue-pytorch-2024-07-30-1`
- `tayayan-gokaku-36-sfen`
- `tayayan-gokaku-36-generated-5247`
- `tayayan-suisho-teacher-kif`
- `tayayan-suisho-teacher-psv`
- `qhapaq-qpd-train`

`dlshogi-public-evaluation-test` and `open-shogiai-floodgate` are rejected from training. The
dlshogi and Taya lineages are unavailable immutable public-test reservations with zero permitted
inspection in this execution.

Do not interpret public availability, a repository license, HF metadata, publication permission,
or a conditional attribution statement as permission. Do not approve a pending source.

## Frozen target semantics

- Canonical position identity is canonical SFEN board, hands, and side to move, with move number
  omitted. Do not mirror or rotate colors.
- All exported values use side-to-move perspective. Convert a black/white factual result only after
  legal replay establishes the current side. Unknown/max-plies outcomes are masked.
- WDL probabilities remain probabilities and use source-local Brier/cross-entropy. Never convert
  WDL to centipawns.
- HCPE3 visits, if a future plan ever authorizes them, would use masked policy cross-entropy with
  `visits/sum(visits)` and missing legal moves unknown. No HCPE3 artifact is active now. Do not fake
  policy pretraining with the binary policy-agreement diagnostic.
- Ordinary cp Huber loss is allowed only for the frozen OpenShogiAI Apery teacher lane, with its
  existing 3,000 cp clip. Do not combine external engine scales.
- Mate labels retain kind, sign, and distance. Implement signed mate-margin/order supervision. Do
  not convert mate to `±3000 cp` or any other fabricated cp target.
- Preserve FV_SCALE/source scale metadata. No unsupported conversion formula is permitted.
- One canonical training row may carry compatible separately masked labels. Never average
  incompatible source scores.

## Frozen data and split behavior

Split protected game/history/source groups before extracting positions. The history group is the
SHA-256 of canonical initial SFEN plus the first up-to-24 legal USI moves. Exact canonical duplicate
priority is:

`public test > final holdout > source-held-out > validation > train`.

Within one priority retain the lexicographically smallest artifact ID then record ID. Use one row
per canonical train position, cap each game at 2% of an epoch and each opening-prefix group at 5%.
Do not heuristically merge non-identical boards.

Preserve existing Phase 3 assignments. The legacy final test has 1,064 rows and has one prior
inspection. It is not available for implementation, training, validation, model/checkpoint choice,
calibration, start positions, or Arena. At most one additional inspection is allowed, only after one
pure finalist and every hyperparameter/artifact hash are frozen. Its result may not trigger
retraining or selection. If there is no finalist, do not inspect it.

Use the existing sequence/position-structure style classifier. Training reports Ibisya and
opponent-Furibisha. Arena construction needs at least 50 unique legal approved positions in each
group: general opening, Ibisya, opponent-Furibisha, and hard middle/endgame. The pilot, entry, and
final gates use 5, 20, and 50 distinct paired starts from every group. A start may appear in at most
one pair per Arena. If any group is short, stop; do not duplicate, synthesize, or borrow positions.

The repaired start-pool semantics are frozen and overlapping at eligibility time:

- `general_opening` is the umbrella predicate `position_index < 24`;
- Ibisya and opponent-Furibisha are authoritative classifier tags and may overlap the umbrella;
- `hard_middlegame_endgame` is the legacy later-position residual: `position_index >= 24` with an
  unclassified or Furibisha continuation; it remains exclusive of the two style tags;
- the central manifest contains one row per canonical position, while deterministic allocation
  assigns each reserved start to exactly one reporting group and prohibits cross-group reuse.

Before any execution, verify `artifacts/phase10/start-pool-manifest.json`,
`artifacts/phase10/start-pool-overlap-report.json`, and
`artifacts/phase10/start-pool-legality-report.json`. The reserve is 200 positions per group under
seed `20260821`, capped at five reserved positions per source game per assigned group. Do not
regenerate it from a pending source or admit any legacy-final-test/public-test identity.

## Exact architecture and experiment matrix

Use random initialization and float32 strength artifacts. Both candidates export the existing
`OSAVAL01` architecture v1: uniform-width fully connected ReLU trunk and one current-side scalar
value. Only the existing binary policy-agreement diagnostic head is training-only.

1. `pure-v1-2x128-control`
   - existing board planes, hand counts, side-to-move, king coordinates;
   - no pseudo-attack maps;
   - input 2,287; 2 hidden layers ×128; dropout 0.10; scale 1,200 cp;
   - 309,634 training and 309,505 exported parameters.
2. `pure-v1-3x256-attacks`
   - same inputs plus the already implemented deterministic pseudo-attack maps;
   - input 2,449; 3 hidden layers ×256; dropout 0.10; scale 1,200 cp;
   - 759,298 training and 759,041 exported parameters.

Do not add convolution, transformer, residual blocks, a move-policy architecture, a third variant,
or a runtime blend. Historical residual 2x128 is read-only diagnostic evidence.

Use seed `20260729`, one run per variant, equal pilot examples and steps. Pilot caps are 7,042 train,
1,894 validation, and 1,000 hard positions. Only the winning pilot may use all additional approved
non-test exact-100 positions, capped at 12,000 train and 2,000 hard positions. Maximum execution is
three new supervised runs and two Arena candidates.

## Exact curriculum

Implement only missing adapters/loss masks necessary to execute these frozen stages. Add focused
tests and preserve all current closed-schema, stable-file, hash, legality, and path checks.

1. Representation: verified factual WDL only; 2–12 epochs, patience 3, minimum delta 0.002.
2. Source value: AobaZero factual WDL only; raw `v` masked; 1–8 epochs, patience 2, delta 0.002.
3. Canonical calibration: approved Apery cp Huber 1.0, factual WDL 0.2, same-teacher candidate
   ranking 0.15, signed mate margin 0.15, binary policy-agreement diagnostic 0.05; 2–20 epochs,
   patience 3, delta 0.002.
4. Hard fine-tune: train-only top-20% teacher error, all sign disagreements/mates within cap,
   MultiPV gap ≤120 cp, Ibisya and opponent-Furibisha; hard rows ≤25% of epoch and ≤16/game;
   cp 1.0, WDL 0.2, ranking 0.25, mate margin 0.25; 1–6 epochs, patience 2, delta 0.001.
5. Offline gate and equal-wall-clock Arenas.
6. One bounded 200-game/100-pair self-play generation with at most 50,000 new train positions and
   factual WDL only, but only after the exact 48% gate passes. No teacher relabeling, no promotion.

Checkpoint selection uses validation only. Keep best and last only. Validate exact Python/Rust cp
agreement for every candidate.

## Exact gates

Offline gate for a pure candidate, on validation only:

- approved-teacher raw cp MAE ≤870.0;
- Pearson correlation ≥0.40;
- engine best-move agreement ≥0.18;
- mate-sign accuracy ≥0.90;
- exact Python/Rust integer-cp agreement;
- search-context throughput ≥3,000 evaluations/second;
- search slowdown ≤1.50× versus pure 2x128;
- complete source/style report.

Arena contract for every Arena: fixed `handcrafted-experimental` opponent, opening disabled,
paired colors, 10 ms/move each, depth cap 8, 32 MiB hash/player, 128 plies, seed `20260821`, and
equal start-group weights 25% general opening / 25% Ibisya / 25% opponent-Furibisha / 25% hard
middle/endgame. A capped or incomplete game is not a draw. Report Wilson 95% on half-draw score and
paired-color bootstrap.

- Pilot: 40 games/20 pairs, all finished, ≥8 decisive, score ≥0.30, side gap ≤0.20, zero illegal,
  crash, or tactical regression.
- Frozen self-play entry: 160 games/80 pairs, all finished, ≥32 decisive, score ≥0.48, Wilson lower
  ≥0.40, every start group ≥0.40, side gap ≤0.15, zero failures. Failure means stop and no self-play.
- Final pure objective: 400 games/200 pairs, all finished, ≥80 decisive, score ≥0.55, Wilson lower
  ≥0.50, every group ≥0.50, side gap ≤0.10, zero failures, pure runtime only. Passing records the
  objective but does not authorize promotion.

Apply the frozen tie break only after gate-compatible evidence: Arena score, validation best-move
agreement, approved-teacher MAE, then fewer exported parameters.

## Exact resource limits

Reference host is Apple M5 / 24 GB. Use one MPS training worker, one teacher worker, at most four
CPU threads and two data-loader workers. Batch size ≤512. Process-tree RSS ≤16 GiB; aggregate
working memory ≤20 GiB. Stop the stage on breach.

Keep filesystem free space ≥200 GiB and incremental Phase 10 use ≤32 GiB. The allocations are 2
GiB data, 8 GiB labels/indexes, 8 GiB checkpoints/exports, 8 GiB Arena/self-play, and 6 GiB
logs/headroom. Check before every stage and every additional GiB streamed. Do not materialize a
decompressed JSON copy. Do not download any full external candidate dataset. If the already
approved exact-100 local data is absent, only the repository's guarded exact-catalog acquisition is
allowed, within the 2 GiB data allocation.

One supervised run may take at most 3,600 seconds, entry Arena 3,600 seconds, and first self-play
generation 7,200 seconds. Do not start a job whose conservative estimate exceeds its limit.

## Execution order and required evidence

1. Inspect all applicable instructions, Git status, current diff, manifests, and ignored local data.
2. Verify frozen hashes and resource preflight. Publish a read-only preflight report.
3. Implement minimal Phase 10 adapters/validators/loss behavior and focused regression tests. Do not
   redesign adjacent code.
4. Reuse the frozen immutable start-pool artifacts. Build the remaining game/history/canonical
   training manifests and verify source/provenance bindings. Confirm zero legacy-final-test
   canonical/history overlap; unavailable public-test overlap must remain explicitly unmeasured,
   never reported as zero. Stop on a failed integrity gate.
5. Run only bounded parser/dataset/overfit smokes. Then execute the two pilot variants in order, one
   MPS worker at a time.
6. Apply offline gates. Run at most the eligible 40-game pilots. Choose at most one winner under the
   frozen tie break.
7. Expand the winner only if allowed, then run the 160-game gate. On failure, stop with no self-play.
8. On pass, run one bounded factual-WDL self-play generation, retrain only the same winning
   architecture within the third-run limit, and run the 400-game final objective if resources and
   time remain within the frozen bounds.
9. Inspect the legacy final holdout at most once and only for a fully hash-frozen pure finalist. Do
   not inspect any public test. Do not retrain from the result.
10. Run focused checks, then the project verification proportionate to changed code. Review the
    complete diff and generated evidence. Commit source/config-independent implementation and small
    reviewed reports only. Never commit data, labels, teacher files, checkpoints, weights, large
    Arena/self-play artifacts, or machine paths. Do not push.

## Stop conditions and final report

Stop immediately on any frozen-hash mismatch, pending-source requirement, split/holdout leakage,
insufficient style starts, unsupported target semantics, non-finite training state, resource breach,
gate failure that forbids the next stage, or need to change a frozen decision.

Your final report must state:

- branch and commit SHA(s);
- exact files changed and tests/checks run;
- input/output hashes, row counts, split/duplicate/style counts, resource peaks, and elapsed times;
- every stage and gate attempted, passed, failed, or not reached;
- candidate architecture/parameter/artifact identities and pure-runtime proof;
- Arena W-L-D, capped/incomplete counts, confidence results, group/color results, and failures;
- holdout inspection count (normally zero; never more than one) and public-test inspection count
  (must be zero);
- self-play games/positions (zero unless the 48% gate passed);
- an explicit statement that no source/license/split/threshold/architecture/promotion decision was
  changed, no model was promoted, no registry was mutated, and nothing was pushed;
- the exact next action for Sol or the user.
