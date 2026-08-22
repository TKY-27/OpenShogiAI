# Model Card: `value_v0`

## Status

`value_v0` is OpenShogiAI's first project-designed neural position evaluator. It was trained
locally from deterministic random initialization; no third-party model architecture or weight
was copied, linked, or used as an initialization. The initial float32 model was registered as
`value-v0-f32-g0r4`. A one-epoch bounded replay run produced `value-v0-f32-g1r4`, which met the
configured 40-game neural-lineage promotion policy and is the `neural_lineage_champion`. It is
not the `overall_champion`; that role remains `handcrafted-experimental`.

Both models are experimental. They are not published, are not assigned a model-weight license,
and are not evidence of general playing strength or amateur-dan strength. Their registry license
status remains `pending-review`; `weights/LICENSE-PENDING.md` is authoritative.

## Intended use

The model is intended for local research on:

- compact deterministic evaluation in the independent Rust search engine;
- Python/Rust numerical agreement and float32/int8 export;
- bounded self-play, replay, arena, and model-registry workflows; and
- future WebAssembly-compatible inference.

It is not intended as a production service, a safety-critical decision system, a teacher for
human moves, or an automatic learner from human-play logs. Human games remain
`pending_human_review` until separately approved.

## Architecture and features

The versioned configuration is in:

- `configs/features/value_v0.toml`
- `configs/models/value_v0.toml`
- `configs/training/value_v0_full_initial.toml`

The exported `OSAVAL01` architecture version is 1:

| Property | Value |
| --- | ---: |
| Input dimension | 2,287 |
| Board planes | 2 sides x 14 piece kinds x 81 squares = 2,268 |
| Normalized hand counts | 14 |
| Side to move | 1 |
| Black/white king coordinates | 4 |
| Hidden trunk | 2 fully connected layers x 64 units |
| Activation | ReLU |
| Training dropout | 0.1 |
| Value output | 1, current-side-to-move perspective, scale 1,200 cp |
| Auxiliary output | 1 policy-agreement logit; training only, not exported |
| Training parameters | 150,722 |
| Exported parameters | 150,657 |
| Estimated exported operations / position | 150,785 |

Absolute-color board planes, hands, side to move, and king coordinates are enabled. Geometric
pseudo-attack planes are implemented but disabled in the default model. The feature schema fixes
piece order, square order, hand normalization, coordinate normalization, output perspective, and
feature flags. The exporter omits the training-only policy head.

User-editable settings include feature enablement, hidden depth/width, activation, dropout, loss
weights, ranking margin, learning rate, weight decay, batch size, epochs, sampling and phase
ratios, stage boundaries, teacher clipping/normalization, seed, device, and quantization. Closed
schema validation rejects unknown keys and invalid or incompatible values.

## Training data and objectives

The initial run used only the approved Phase 3 AobaZero slice and its 10,000 Apery labels:

- dataset manifest SHA-256:
  `db2a286ebfa4edd01c67041fb55d33b0f5f8813d24a2203d7c5e55a36111e243`;
- positions SHA-256:
  `d36cfc86757887f05a2d0c5ecc50e606a5836687cff32e757e268f8cf5cb627f`;
- labels SHA-256:
  `195850b3ae7b1dce8a98185f2ba17f794a0200c92eafc672e11674d17d8cd3f4`;
- label manifest v2 SHA-256:
  `edeab652bd825410f46bab3010d3e8a017f33cd281cd0f28329d8eaab2a2cb54`;
- train / validation / test: 7,042 / 1,894 / 1,064 positions; and
- opening / middlegame / endgame: 2,257 / 3,923 / 3,820 positions.

The objective combines clipped normalized teacher-value regression, factual game outcome,
recorded-move/teacher-bestmove agreement, and candidate ranking. Training reads only `train`;
validation selects checkpoints; the held-out test split requires the explicit `--final-test`
command. Mate and centipawn labels remain distinct at the data boundary.

The Phase 6 challenger additionally used the replay manifest SHA-256
`6b3df7838aaa6ce4b263de6f2d9d479d6cac5197ede6654ff50adf31d64a0f7d`.
It retained 10,590 unique entries (7,706 train, 1,820 validation, 1,064 test), including 702
new-generation and 9,888 older entries. New self-play rows use only factual current-side game
outcomes; they contain no fabricated teacher or policy labels. The goal-wide teacher-label count
remained exactly 10,000.

## Initial training and evaluation

The bounded proof runs were:

- tiny overfit: CPU, 32 train + 32 validation examples, 200 epochs / 400 steps,
  best validation loss `0.0012799606`, 5.228 s, peak RSS 430,718,976 bytes;
- smoke: MPS, 1 epoch / 2 train batches, validation loss `0.6647041222`, 2.182 s,
  peak RSS 825,409,536 bytes; and
- full initial: MPS, 50 epochs / 5,550 steps, interrupted after epoch 1 and resumed from
  `last.pt`, 107.545 s combined recorded execution time, peak RSS 911,671,296 bytes.

The full run's best checkpoint was epoch 1, not epoch 50. Later training loss continued to fall
while validation loss worsened, which is direct evidence of overfitting. Metrics below were
recomputed from the best checkpoint:

| Split | Examples | Total loss | Teacher MAE (cp) | Policy agreement |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1,894 | 0.6712710091 | 802.5032 | 0.641499 |
| Test | 1,064 | 0.6101716538 | 737.8941 | 0.619361 |

Initial best-checkpoint identity:

- checkpoint SHA-256:
  `28e28869397bbaa225195a2c816f28149c75321574822d9f2f0323cee6d5b50d`;
- combined config SHA-256:
  `e8a7413e4a26fd5c2f89c9bea2bf1cb6babf970a54af2f2a6f843b2b4d44d833`;
- seed: `20260729`; and
- runtime: Python 3.12.13, PyTorch 2.11.0, deterministic algorithms, MPS, four
  intra-op threads and ten inter-op threads.

The experiment records clean Git commit `205dc84fe34f1984506f4b49d882d657a756cad3`
and `gitDirty=false`, plus an exact model-code SHA-256. The ignored experiment manifest remains
the authority for the code and artifact identities actually executed.

## Export, quantization, and runtime agreement

`OSAVAL01` is a project-specific little-endian, closed, checksummed format. Rust validates file
size, magic, format/feature/architecture versions, feature flags, tensor dimensions, payload
SHA-256, finite values, and quantization metadata before inference. Explicit neural selection
fails closed on missing or corrupt models.

Phase 10R also has a parity-qualified `OSAVAL02` contract for its two frozen sparse candidate
architectures. It is not a promoted/default model and does not change the historical OSAVAL01
claims in this card. See [`docs/model/OSAVAL02_FORMAT.md`](docs/model/OSAVAL02_FORMAT.md) and
`PHASE_10R_OSAVAL02_PARITY_REPORT.md` for its exact scope and limitations.

Initial exports:

| Artifact | Bytes | File SHA-256 | Payload SHA-256 |
| --- | ---: | --- | --- |
| float32 | 602,736 | `7a7192b3f5a849ca895452fe89d14a1f770c5416ee10888fef295af805e95d2e` | `df2a73d3261acf13b1fa52f06d4c467e310550690d475ace8564cb0a63e5987d` |
| symmetric per-layer int8 | 151,164 | `515898914f34ac7e3d47f3849653973947fe3ed3d9d3b31ba084c50789e7e317` | `e94ed6432303aba81be13b975d6595ba4d11785270d3498bbcbcbbbe344516eb` |

Across all 1,894 validation positions:

- PyTorch checkpoint versus Python float32 export: maximum/mean/p95 absolute error
  `0 / 0 / 0 cp`;
- Python export versus Rust: exact integer-cp agreement for float32 and int8;
- float32 versus int8: maximum `2 cp`, mean `0.409187 cp`, p95 `1 cp`;
- Rust float32 reported latency: mean 77,129 ns, p95 103,500 ns; and
- Rust int8 reported latency: mean 83,179 ns, p95 85,750 ns.

Int8 reduces the artifact to about one quarter of float32 size, but was about 7.85% slower in
this bounded scalar Rust measurement. Quantization therefore provides a size benefit, not a
demonstrated speed benefit on this machine.

## Bounded playing evidence

All Phase 5 comparisons used 40 games, 20 paired start positions, alternating colors, 500 nodes
per move, depth cap 4, 16 MiB hash, and 128 plies. There were zero illegal moves and zero
crashes. The most important negative result is explicit: handcrafted experimental beat the
initial neural float model 35-3 with 2 draws. The neural model's score rate was 0.100 in that
comparison. This first model is substantially weaker than the handcrafted evaluator under the
tested conditions.

Float32 versus int8 was 18-14 with 2 draws and 6 max-plies results (float score 0.55; paired
bootstrap interval 0.425-0.6625). Int8 opening-off versus opening-on was 21-14 with 3 draws and
2 max-plies results (opening-off score 0.5875; interval 0.475-0.700); the book supplied 12 moves.
Neither comparison establishes a reliable quality difference at this sample size.

The Phase 6 one-epoch MPS challenger used 14,748 train and 3,714 validation rows for 231 steps,
completed in 11.012 s, and peaked at 1,288,306,688 bytes RSS. Its best validation loss was
`0.7228932487`. Export identities are:

| Artifact | Bytes | File SHA-256 | Payload SHA-256 |
| --- | ---: | --- | --- |
| challenger float32 | 602,736 | `ef43319301e400b650045f86427981de6530a960c966613a2c77542814b0b684` | `a1ba58a91bb7e02fddb34db0f0a31118bd08c29bf531b9a77f57b755f613a670` |
| challenger int8 | 151,164 | `ed7ec02534289ef297004cd0c8a160dfe51e982789a7d85397ae258eae836d89` | `ef9348e9f69e4736442f2a7062786af62fb8ab96a4b4416743d3487acb34585e` |

Against generation 0 in the bounded promotion arena, the challenger scored 32 wins and 8
losses in 40 decisive games, with zero illegal moves/crashes. Its score rate was 0.8 and the
recorded Wilson 95% interval was 0.6524-0.8950. It scored 20-0 on normal starts and 12-8 on the
validation start set. All configured promotion gates passed, so it was promoted locally. This
supports that exact bounded promotion decision only; it does not overturn the Phase 5 evidence
that this model family is weak against the handcrafted evaluator.

## Bounded 2026-08-21 evaluator campaign

The follow-up campaign reused exactly the approved 10,000 teacher-labeled positions. It trained
one model at a time on MPS with seed `20260729`, 50 epochs, and no test-split access during
selection. Residual targets are defined as:

```text
final_score = handcrafted_experimental_score + learned_delta
```

The residual baseline was independently built twice with identical SHA-256
`b90db1c8b41a58fbc6d8d0e45f920da66fd3fb932ed3b1614495aa268162529d`.
Because an absolute game-result target is not a delta, that auxiliary loss is masked for residual
runs. Teacher delta regression, candidate ranking, and the training-only policy-agreement head
remain enabled. The pure-neural controls remain intact.

| Run | Training / exported parameters | Validation objective | Optimized MAE (cp) | Time (s) | Peak RSS (bytes) | Float32 artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pure 2x64 control | 150,722 / 150,657 | 0.6712710091 | 802.5032 | 107.545 | 911,671,296 | `7a7192b3...` |
| pure 2x128 | 309,634 / 309,505 | 0.6890455457 | 829.2856 | 115.263 | 918,011,904 | `b42b0930...` |
| residual 2x64 | 150,722 / 150,657 | 0.3901236407 | 747.1778 delta | 109.598 | 915,259,392 | `cfe84ee0...` |
| residual 2x128 | 309,634 / 309,505 | 0.3896368305 | 746.1339 delta | 113.305 | 906,117,120 | `b577f7c9...` |

Float32 sizes are 602,736 bytes for 2x64 and 1,238,128 bytes for 2x128. Int8 sizes are 151,164
and 310,396 bytes. The full artifact hashes are in `PHASE_9_REPORT.md`; generated weights remain
ignored and unpublished.

Calibration below uses the 1,845 validation rows with ordinary centipawn teacher scores and the
unclipped final exported score. `teacher = intercept + slope * model`; it is distinct from the
clipped training objective.

| Model | Raw MAE (cp) | RMSE (cp) | Mean model bias (cp) | Slope | Intercept (cp) | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pure 2x128 | 967.03 | 1,752.78 | -119.01 | 0.8718 | 132.93 | 0.2223 |
| residual 2x64 | 867.39 | 1,379.01 | -137.63 | 1.0543 | 132.75 | 0.6444 |
| residual 2x128 | 866.45 | 1,377.45 | -126.82 | 1.0568 | 121.11 | 0.6448 |

Only residual 2x128 was evaluated on the held-out test after selection. Its optimized test
objective was `0.4406921195`, optimized delta MAE was 808.6469 cp, and auxiliary policy accuracy
was 0.619361. On the 1,037 ordinary-cp test rows its raw final-score MAE was 872.39 cp, RMSE was
1,341.20 cp, calibration slope/intercept were 0.8919/110.65 cp, and Pearson r was 0.6454.

Move-ranking evidence is a direct engine benchmark, not the training-only binary policy head.
Each engine searched all 1,894 validation positions at 500 nodes and depth cap 4 and was compared
with Apery's recorded best move. The test split was not read.

| Evaluator | Matches | Accuracy |
| --- | ---: | ---: |
| handcrafted baseline | 377 | 19.905% |
| handcrafted experimental | 377 | 19.905% |
| pure 2x64 initial | 196 | 10.348% |
| pure 2x64 neural-lineage champion | 189 | 9.979% |
| pure 2x128 | 202 | 10.665% |
| residual 2x64 | 415 | 21.911% |
| residual 2x128 | 420 | 22.175% |
| composite 50/50 with g1r4 | 397 | 20.961% |

Equal-wall-clock exploratory Arenas used 10 ms/move, depth cap 8, 32 MiB hash, alternating
colors, 128 plies, opening disabled, and zero illegal moves. Against handcrafted experimental:

| Candidate | Games finished | Candidate W-L-D | Notes |
| --- | ---: | ---: | --- |
| handcrafted baseline | 17/20 | 0-2-15 | 3 max-plies; incomplete for a 20-game score claim |
| pure 2x64 initial | 20/20 | 0-20-0 | exploratory |
| pure 2x64 g1r4 | 20/20 | 0-20-0 | exploratory |
| pure 2x128 | 20/20 | 0-20-0 | exploratory |
| residual 2x64 | 20/20 | 0-20-0 | exploratory |
| composite g1r4 | 20/20 | 0-19-1 | exploratory |
| residual 2x128 | 40/40 | 0-8-32 | predeclared overall gate; score 0.400, Wilson 95% 0.2635-0.5540 |

Search-context neural inference throughput was about 13.8k calls/s for 2x64 models and 6.95k
calls/s for 2x128 models. These are Arena observations, not standalone hardware specifications.
Residual 2x128 passed both tactical cases with no regression but failed the gate's decisive-game,
score-rate, and lower-confidence-bound thresholds. Therefore `handcrafted-experimental` remains
the overall champion. The prerequisite for the larger self-play campaign was not met: zero new
generations and zero new self-play games were started.

## Limitations

- Training data is a small, homogeneous 100-game self-play slice with only 10,000 teacher
  labels; it is not representative of human play or broad shogi distributions.
- Teacher analysis used 25,000 nodes and at most three returned PVs. Two short-PV rows omit one
  independently legal root move, so unreturned moves have unknown teacher scores.
- The separate opening-book teacher view used MultiPV 32 on the same 10,000 positions; it was not
  substituted into model training, validation, or test data.
- Validation selected epoch 1 and the 50-epoch run overfit strongly.
- Phase 5 and Phase 6 arenas contain only 40 games per comparison and show material color/start
  sensitivity; all strength conclusions are provisional.
- The promotion opponent was another weak neural model, not a strong external engine or human.
- No multi-generation or large-scale training campaign was run, and no current Elo, rank, or
  amateur-dan estimate is justified.
- The equal-wall-clock matrix is small. Max-plies games are reported as capped rather than silently
  converted to draws, and only the predeclared residual 2x128 gate is promotion evidence.
- MPS memory readings are snapshots plus process-lifetime peak RSS, not a full memory trace.
- Weight licensing remains pending review. The source-code AGPL-3.0-only license does not
  license generated weights by implication.

## Reproduction entry points

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

The exact historical Phase 5 arena verification command and clean-commit precondition are in
`PHASE_5_REPORT.md`. A current checkout must not be used to relabel that frozen evidence.

Generated checkpoints, exports, reports, and registries stay in ignored local storage and are
not committed or published.
