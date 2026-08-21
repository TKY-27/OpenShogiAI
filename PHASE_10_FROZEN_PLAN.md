# Phase 10 — Frozen supervised-data curriculum

Frozen: 2026-08-21 (Asia/Tokyo)

Branch: `codex/phase10b-dataset-curriculum`

Objective: a pure learned evaluator that scores at least 55% against the fixed
`handcrafted-experimental` evaluator under the frozen equal-wall-clock Arena contract.

Narrow revision: 2026-08-21 (Arena start-pool repair only). `general_opening` is the broad
position-index-below-24 control population. Ibisya and opponent-Furibisha are descriptive tags
that may overlap that control population; the four eligibility groups are not mutually exclusive.
The immutable manifest stores one canonical row per position and assigns 200 globally distinct
reserved paired starts to each group under seed `20260821`. No model, target, source-license,
training, statistical-threshold, budget, promotion, or holdout-access control changed.

## Decision summary

This plan is deliberately fail-closed. Phase 10A audited 22 artifact records. Only the already
approved AobaZero exact-100 catalog and one member already contained in that catalog are approved.
No new external training artifact has complete training/derived-weight rights. Therefore:

- the active supervised population is the legally normalized exact-100 AobaZero games, their
  verified factual outcomes, and labels produced for those approved positions by the separately
  executed, approved OpenShogiAI Apery teacher configuration;
- `w4745.csa` is an audit fixture contained in the exact-100 set and has zero independent sampling
  weight, preventing a duplicate-source bias;
- all 18 pending-license records are deferred with zero sampling weight and zero maximum
  contribution;
- the dlshogi public evaluation test and OpenShogiAI Floodgate records are rejected from training;
  the former remains a named immutable public-test reservation;
- no third-party weights or external model initialization are allowed;
- residual and composite evaluators remain diagnostics and cannot pass any Phase 10 objective;
- no handcrafted score, residual, blend, or fallback may contribute to the runtime score of an
  eligible candidate.

This is not a claim that the small exact-100 population is sufficient to reach 55%. It is the only
curriculum currently compatible with the frozen provenance rules. The plan makes a bounded
supervised attempt, requires a 48% Arena gate before any self-play, and stops rather than silently
expanding to ambiguous data.

## Evidence that fixes the design

- Phase 5 fixed-node evidence: the initial pure 2x64 evaluator scored 10% against handcrafted
  experimental (`PHASE_9_REPORT.md`, `MODEL_CARD.md`).
- Phase 9 equal-wall-clock evidence: pure 2x64, replayed pure 2x64, and pure 2x128 each lost 0-20;
  residual 2x128 reached 40% over 40 games but used handcrafted score at runtime and failed the
  overall gate.
- Current validation move agreement is 10.665% for pure 2x128, 19.905% for handcrafted, and
  22.175% for residual 2x128. This supports a stronger pure feature/trunk candidate while keeping
  residual only as a diagnostic ceiling.
- Existing data comprises 100 unique AobaZero games, 15,488 positions, and a game-only split.
  The current teacher view contains 7,042 train, 1,894 validation, and 1,064 final-test positions.
  The v2 manifest hash is
  `edeab652bd825410f46bab3010d3e8a017f33cd281cd0f28329d8eaab2a2cb54`.
- The current pure 2x128 model has 309,505 exported parameters, used about 0.92 GB peak RSS while
  training, and delivered about 6.95k search-context evaluations/second. The M5 24 GB host can
  safely test one bounded sub-million-parameter candidate without broadening the runtime format.
- The Phase 10A overlap audit confirms AobaZero conversion lineage inside GCT and strong Taya
  derivative overlap. Unknown overlap is never interpreted as zero.

## Frozen source decisions

Weights below are stage-local sampling weights. A weight of zero means the artifact cannot emit a
training example. `cap` is the maximum fraction of an active stage. Holdout reservations have a
dedup priority above every train/validation source but remain unavailable until separately
authorized and hash-manifested.

| Audited artifact | Decision | Exact subset | Role | Weight / cap | Dedup priority | Attribution and release constraint |
| --- | --- | --- | --- | --- | ---: | --- |
| `aobazero-no-noise-exact100` | accepted | exact versioned 100-object catalog only | verified WDL pretraining; teacher ranking/calibration join; opening/start pool; game-split validation | active weights 1.0/1.0/1.0/0.25; cap 1.0 | 500 | pinned Public Domain scope only; no live-index expansion; generated weights remain pending review |
| `aobazero-no-noise-w4745-sample` | accepted as audit fixture | `w4745.csa`, already inside exact-100 | none independently | 0 / 0 | 0 | must never be counted as a second source or second copy |
| `aobazero-live-no-noise-sample-index` | deferred | none | none | 0 / 0 | 0 | current hash drift; every downstream right pending |
| `aobazero-daily-public-records` | deferred | none | none | 0 / 0 | 0 | no exact manifest or dataset-specific grant |
| `dlshogi-gct-hcpe3-selfplay` | deferred | none; wildcard is not a manifest | future policy/value lane only after a new freeze | 0 / 0 | 0 | reuse and derived-weight rights pending |
| `dlshogi-gct-floodgate-hcpe3` | deferred | none | none | 0 / 0 | 0 | Floodgate lineage and rights unresolved |
| `dlshogi-gct-taya36-hcpe3` | deferred | none | protected Taya lineage only | 0 / 0 | 0 | rights pending; high holdout-contamination risk |
| `dlshogi-gct-model-taya36-hcpe3` | deferred | none | protected Taya/model lineage only | 0 / 0 | 0 | model and derived-weight rights pending |
| `dlshogi-gct-aobazero-hcpe` | deferred | none | none | 0 / 0 | 0 | confirmed source-family duplication; exact crosswalk absent |
| `dlshogi-gct-floodgate-play-hcpe` | deferred | none | none | 0 / 0 | 0 | mixed Floodgate/tournament/local rights cannot be separated |
| `dlshogi-gct-taya36-rl-hcpe` | deferred | none | protected Taya lineage only | 0 / 0 | 0 | exact overlap unknown; rights pending |
| `dlshogi-gct-yaneura36-rl-hcpe` | deferred | none | none | 0 / 0 | 0 | exact identity, scale, perspective, and rights absent |
| `dlshogi-gct-suisho-teacher-hcpe` | deferred | none | none | 0 / 0 | 0 | publication permission is not a complete downstream grant |
| `nodchip-shogi-hao-depth9` | deferred | none; representative shard is evidence only | future source-local PSV value lane | 0 / 0 | 0 | dataset-file/derived-weight grant, FV_SCALE, perspective, and mate semantics unresolved |
| `nodchip-tanuki-nnue-pytorch-2024-07-30-1` | deferred | none; representative shard is evidence only | future source-local PSV value lane | 0 / 0 | 0 | same unresolved terms plus unknown Hao overlap |
| `tayayan-gokaku-36-sfen` | deferred | none; whole lineage reserved | unavailable external holdout reservation | 0 / 0 | 1000 | do not acquire or inspect until rights and exact manifest are approved |
| `tayayan-gokaku-36-generated-5247` | deferred | none; reserve with original family | unavailable external holdout reservation | 0 / 0 | 1000 | original/derivative are one protected family; rights pending |
| `tayayan-suisho-teacher-kif` | deferred | none | future source-local teacher lane | 0 / 0 | 0 | complete downstream rights pending; legal replay required |
| `tayayan-suisho-teacher-psv` | deferred | none | future source-local teacher lane | 0 / 0 | 0 | complete downstream rights and KIF/PSV crosswalk pending |
| `qhapaq-qpd-train` | deferred | none; archive not inspected | none | 0 / 0 | 0 | attribution is required for event/publication use; other rights and format pending |
| `dlshogi-public-evaluation-test` | rejected from training | none; lineage reserved | immutable external public test only | 0 / 0 | 1100 | never train/tune/select/start from it; acquisition needs a later rights decision |
| `open-shogiai-floodgate` | rejected | none | none | 0 / 0 | 0 | ML use, derived weights, and redistribution are denied |

The exact machine-readable version, including every prohibited role, is
`configs/phase10/dataset-mixture.yaml`.

## Target compatibility contract

### Perspective and identity

Every position is represented by canonical SFEN board, hands, and side to move with the move
number omitted. No mirror or color rotation is applied. This avoids silently transforming shogi
move labels and preserves the existing Rust/Python feature contract. All exported values are from
the current side-to-move perspective.

Black/white factual results become `+1/0/-1` only after legal replay establishes the current side.
If a source value's perspective is unknown, its value mask is zero. Raw provenance remains stored.

### HCPE3 MCTS targets

If an HCPE3 artifact is authorized in a future revision, each candidate retains raw move and visit
count. A source-local policy probability is `visits / sum(visits)` only when the sum is positive.
Missing legal moves are unknown and masked, not assigned zero target probability. Selected move,
playouts, temperature/noise metadata, and the complete candidate prefix remain attached. Policy
cross-entropy is a separate head/loss; visits are never treated as alpha-beta scores.

The frozen execution has no approved HCPE3 source, so policy-distribution pretraining is disabled.
The existing scalar policy-agreement head predicts only whether the recorded move equals teacher
best move; it is a diagnostic auxiliary target and is not relabeled as a move policy.

### WDL and value probabilities

Factual WDL and declared W/D/L probabilities retain their probability semantics. They use Brier or
cross-entropy losses in their own source lane. They do not become centipawns through a logistic or
other guessed conversion. Unknown/max-plies outcomes are masked.

### Alpha-beta scores, mate, and FV_SCALE

Ordinary centipawn regression is active only for the approved OpenShogiAI Apery teacher at its
frozen engine, options, node limit, and scale. The existing 3,000 cp clip is a robustness bound for
that lane, not a cross-engine conversion.

Mate labels retain kind, sign, and distance. They train a signed margin and same-source ordering;
they do not become `±3000 cp`. Runtime search discovers and encodes mate separately, so the static
evaluator exports no fabricated mate score.

HCPE/PSV/source eval integers retain source engine, config, declared perspective, and FV_SCALE.
Without an official formula they cannot join the canonical cp loss. The preferred future bridge is
to deduplicate approved positions and relabel them with the approved OpenShogiAI teacher, rather
than invent a universal centipawn formula.

### Duplicate, near-duplicate, and source bias controls

Splitting happens before position extraction at protected game/history/source-lineage level. Exact
canonical board duplicates inherit the highest protected split priority:

`public test > final holdout > source-held-out > validation > train`.

Only one training row exists per canonical position. Compatible labels attach as separately masked
objects. Incompatible source scores are not averaged. Within a split, the lexicographically smallest
artifact/record identity wins the row identity. Each game is capped at 2% of an epoch and each
24-ply opening-prefix group at 5%.

`history_group_id` hashes canonical initial SFEN and the first up-to-24 legal USI moves. Exact
histories and all descendants of protected public-test starts remain together. Non-identical boards
are reported as similar but are not merged by a heuristic distance threshold.

Arena starts apply the same priority before pool construction. Any history whose effective split is
the legacy final holdout is excluded, and any canonical identity observed anywhere in that holdout
is excluded even when a train/validation copy exists. Train/validation duplicates retain validation
priority. The final start manifest contains no exact or canonical duplicate row.

## Frozen curriculum

```text
approved exact-100 games
        |
        +-- verified factual WDL --> stage 1 representation / stage 2 source value
        |
        +-- approved non-test positions -- Apery teacher --> stage 3 cp/rank/mate calibration
                                                        |
                                                        +--> stage 4 hard-position fine-tune
                                                                     |
                                                                     v
validation-only offline gate --> 40-game pilot --> 160-game 48% gate
                                                        |
                                                        +-- pass --> one bounded self-play generation
                                                        +-- fail --> stop, no self-play
```

1. **Representation/policy pretraining.** Train trunk/value on verified factual WDL. True policy
   pretraining is frozen off because no approved policy distribution exists.
2. **Source-specific value learning.** Continue only the AobaZero factual-WDL lane. Raw `v` is
   excluded because meaning and perspective are unknown.
3. **Canonical score calibration.** Use only approved OpenShogiAI Apery labels. CP Huber, factual
   WDL, same-teacher candidate ranking, signed mate margin, and binary policy-agreement diagnostic
   retain separate masks. Train/validation are read; final test is not.
4. **Hard-position fine-tuning.** Select at most 2,000 train-only unique positions: high teacher
   error, sign disagreement, mate, close MultiPV gap (≤120 cp), Ibisya, and opponent-Furibisha.
   Hard rows are capped at 25% of an epoch and 16 positions per game.
5. **Equal-wall-clock screening.** Against fixed handcrafted experimental: opening off, paired
   colors, 10 ms/move, depth cap 8, 32 MiB hash/player, 128 plies. Starts are equally allocated
   among general opening, Ibisya, opponent-Furibisha, and hard middle/endgame groups.
6. **Self-play.** Disabled unless a pure candidate passes the frozen 160-game 48% gate. The first
   generation is limited to 200 paired games and 50,000 new train positions, factual WDL only,
   with no teacher relabeling and no promotion.

### Ibisya and opponent-Furibisha coverage

The existing sequence/position-structure classifier is authoritative; a single rook-file check is
not sufficient. Training reports both styles, caps Ibisya and unknown/other at 60% each, and targets
at least 10% opponent-Furibisha when available. For Arena eligibility, `general_opening` is the
umbrella control predicate `position_index < 24`; Ibisya and opponent-Furibisha remain classifier
tags and can overlap the umbrella. `hard_middlegame_endgame` is the later-position control predicate
`position_index >= 24` restricted to unclassified/Furibisha continuations, preserving the legacy
exclusive later-position residual. General-opening/style overlap is membership metadata, never a
duplicated manifest row. Arena allocation assigns every canonical start to exactly one reporting
group and prohibits reuse across groups within a campaign.

Arena construction requires 50 unique legal positions in each eligibility group. The immutable
reserve contains 200 distinct starts per group (800 globally canonical-distinct rows), which covers
the pilot/entry/final requirements of 5/20/50 and also preserves 200/group capacity for a 1,600-game
paired-color campaign without reuse. Selection uses seed `20260821`, allocates the scarce general
opening control first, and caps each source game at five reserved positions per assigned group. If
the approved population cannot supply the reserve under those controls, planning fails; it may not
duplicate starts, synthesize positions, or borrow a pending dataset.

## Frozen splits and holdout access

- Protected game/history/source groups are hashed addition-stably into 80% train, 10% validation,
  and 10% final holdout.
- Existing Phase 3 assignments remain authoritative. The 1,064-row legacy final test was inspected
  once for the frozen Phase 9 residual finalist. Phase 10 permits exactly one additional inspection,
  only for one completely frozen pure finalist after all training and Arena selection ends.
- No final-holdout result may select a model, hyperparameter, checkpoint, dataset mixture, or
  retraining decision.
- Taya original/derivative and dlshogi public evaluation lineages are immutable public-test
  reservations with zero allowed inspection in this plan. They are not authorized for download.
- Each future newly authorized source must reserve at least 10% of protected groups and support a
  source-held-out model trained with that entire source omitted. Activating such a source requires a
  new frozen-plan revision.
- Every holdout access appends holdout ID, artifact/model hashes, purpose, time, commit, and result
  hash to an immutable inspection log.

## Model and experiment matrix

Only two new supervised variants are justified:

| Variant | Inputs / trunk | Training / exported params | Role |
| --- | --- | ---: | --- |
| `pure-v1-2x128-control` | current 2,287 inputs; 2×128 ReLU | 309,634 / 309,505 | curriculum control |
| `pure-v1-3x256-attacks` | adds existing deterministic pseudo-attack maps; 3×256 ReLU | 759,298 / 759,041 | bounded capacity+tactical candidate |

Both use random initialization, a single current-side value output, and the existing `OSAVAL01`
v1 runtime. Pseudo-attack maps are deterministic position inputs, not a handcrafted score.
Architecture changes beyond this exact pair are prohibited during execution.

The pilot runs both variants on 7,042/1,894 train/validation positions. Only the winner may use all
additional unique approved, non-test exact-100 positions, capped at 12,000 train positions and
2,000 hard positions. One seed and one run per variant prevent an unjustified sweep. At most three
new supervised runs and two Arena candidates are permitted.

The historical residual 2x128 result is a read-only diagnostic. It cannot enter the matrix as an
eligible candidate, initialize a pure model, or satisfy the objective.

### Early stop and expansion

Stages use validation-only early stopping: representation 12 epochs/patience 3, source value 8/2,
calibration 20/3, hard fine-tune 6/2. The minimum improvement is fixed in `curriculum.yaml`. A model
that fails the offline gate does not enter Arena. A pilot that fails the 40-game screen does not
expand. No external source may be activated by an execution agent.

## Statistical gates

The score is `(wins + 0.5 × draws) / finished W-L-D games`; capped/incomplete games are never
silently draws. Wilson 95% on the half-draw score and paired-color bootstrap are both reported.

| Gate | Size | Required result |
| --- | ---: | --- |
| offline | 1,894 validation rows | raw approved-teacher MAE ≤870 cp, Pearson ≥0.40, engine best-move agreement ≥18%, mate-sign ≥90%, exact Python/Rust cp, ≥3k search-context eval/s, ≤1.5× search slowdown |
| pilot | 40 games / 20 pairs | score ≥30%, ≥8 decisive, all 40 finished, no illegal/crash/tactical regression, side gap ≤20% |
| frozen self-play entry | 160 games / 80 pairs | score ≥48%, Wilson lower ≥40%, ≥32 decisive, each start group ≥40%, side gap ≤15%, zero failures |
| final pure objective | 400 games / 200 pairs | score ≥55%, Wilson lower ≥50%, ≥80 decisive, each group ≥50%, side gap ≤10%, zero failures, pure runtime only |

Passing 48% permits one bounded self-play generation; it is not promotion. Passing 55% records that
the objective is met but still does not mutate the overall-champion registry in this execution.

## Resource freeze for Apple M5 / 24 GB

- one MPS training worker, one teacher worker, four CPU intra-op threads, two data-loader workers;
- process-tree RSS ≤16 GiB and aggregate working memory ≤20 GiB;
- batch size ≤512, exported parameters ≤800,000, runtime artifact ≤4 MiB;
- incremental Phase 10 disk ≤32 GiB and filesystem free space must remain ≥200 GiB;
- full external downloads and decompressed JSON materialization are prohibited;
- one supervised run ≤1 hour, entry Arena ≤1 hour, first self-play generation ≤2 hours;
- retain best+last checkpoints only; raw/processed data and weights remain ignored and uncommitted.

## Architecture and implementation blueprint

The `code-architect` workflow fixes the implementation at existing boundaries rather than adding a
new training framework.

| File/component | Purpose | Dependencies/data flow |
| --- | --- | --- |
| `configs/phase10/*.yaml` | closed frozen decisions | Phase 10A audit and existing manifests |
| `training/open_shogi_training/phase10.py` | offline validator and hash verifier | strict YAML loader, SHA-256, existing audit catalog |
| existing data normalization/replay | emit canonical position, game/history/source identities and masked target objects | Rust legal replay + Phase 10 split policy |
| existing model trainer/exporter | implement the frozen separated losses and exact two variants | current feature encoder, `OSAVAL01` v1 |
| existing paired Arena pipeline | create equal-time, paired, stratified reports | frozen start manifest and statistical gates |
| `prompts/LUNA_PHASE10_EXECUTION.md` | exact execution authority | all hashed controls; no decision authority |

Build order for the execution phase:

1. verify hashes, rights decisions, disk/memory preflight, and immutable split manifests;
2. implement only missing target masks/mate-margin and Phase 10 config adapters with focused tests;
3. verify `artifacts/phase10/start-pool-manifest.json`, its overlap/leakage report, and its
   all-position Rust legality report before any training or Arena work;
4. run bounded dataset/overfit smokes, then the two pilot supervised variants;
5. apply offline and 40-game gates; expand only the one winner if allowed;
6. run the 160-game gate; run one bounded self-play generation only on pass;
7. stop without final-test inspection unless a single finalist is frozen, and always stop without
   promotion or registry mutation.

## Frozen controls and change policy

`configs/phase10/frozen-controls.sha256` hashes this plan, all six Phase 10 YAML controls, the Luna
prompt, the validator, and authoritative Phase 10A/split evidence. Run:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10 verify --root .
```

Any content change requires intentionally regenerating the manifest, reviewing the complete diff,
and creating a new frozen-plan revision. The Luna execution prompt has no authority to change
architecture, source/license decisions, split identities, thresholds, gates, objective eligibility,
promotion policy, or frozen hashes.
