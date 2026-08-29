# Phase 10R teacher-binding repair

Frozen: 2026-08-29 (Asia/Tokyo)

Scope: resolve only the `no completed teacher-bound candidate` prerequisite and freeze the next
bounded execution. This repair performs no teacher search, production labeling, calibration
training, hard-example selection, Arena, self-play, holdout inspection, promotion, push, or
deployment.

## Decision: case 3

The execution sequence skipped the already-frozen teacher-binding stages.

This is not case 1. The two 1M training summaries and receipts contain only
`representation_policy_pretraining` and `source_specific_wdl_value_pretraining`; both explicitly
record `teacher_dependent_stages: not_started`. The 1M preparation contains exactly AobaZero,
WCSC, and Denryu factual-record rows and no Apery score, ranking, mate, or teacher-identity field.
The exported OSAVAL02 artifacts are therefore pretraining-only candidates. Their existing weights
must never be relabeled as teacher-derived.

This is not case 4. The frozen plan requires stage 3
`packed_sfen_value_and_ranking` and stage 4 `approved_teacher_calibration` before active-learning
hard-example fine-tuning. The select-hard provenance requirement protects that design and must not
be removed.

Case 3 is more precise than case 2 because the teacher stages were already present in the frozen
curriculum. `PHASE_10R_EXECUTION_IMPLEMENTATION_RATIONALE.md` says that the runner implemented only
stages 1 and 2, while `prompts/LUNA_PHASE10R_EXECUTION.md` proceeded from `train` and `evaluate`
directly to `select-hard`. No command existed between them to create the required bound candidate.

## Immutable pretraining parents

The exact parents are frozen in `configs/phase10r/teacher-binding.yaml`.

| Variant | Stage-2 checkpoint SHA-256 | Pretraining OSAVAL02 SHA-256 |
| --- | --- | --- |
| `sparse-pair-policy-wdl` | `6a5048ae1e7d230dbb073d701fad6ac7acab6d9dc9637480282c5e0bc8ece48f` | `6b7f2f4dc0bb013992460cf475f5e2e1f220da7fb09542175be367821f296f63` |
| `factorized-pair-triple-policy-score` | `47c43cd44f6258973d3262931d96d46e4e36f55b456249647fe086c4253c5ed8` | `d2c9af492cb67be35433662b75673efa2c6fece3f0ba2457e281185469364453` |

Both parent artifacts remain at their current paths. Teacher-derived checkpoints and artifacts use
the distinct versioned `teacher-bound-v1` child directory. A child artifact whose SHA-256 equals
its pretraining parent is rejected.

The two failed select-hard receipts remain immutable local evidence:

- `20260829T102425.480772Z-select-hard.json`: SHA-256
  `1d6776f1804a5a4fbec7be4ec95b91aa04b4bd7882307ed0ddfa5368b568afaf`;
  the teacher installation was not yet available.
- `20260829T105218.398890Z-select-hard.json`: SHA-256
  `d163c07d2061d7469e734ebd334793e6f448d78f306fb5d3f9b4f91fbf34a686`;
  the exact teacher was restored, but no completed teacher-bound candidate existed.

## Canonical teacher-binding identity

The canonical binding identity is SHA-256
`781a45570ce6c88e29e4c9f7f3acb96e3981bb74d7da6fb10ff80cdd76f51cbf`. It is the canonical JSON
hash of the closed `teacher_binding_identity` object in
`configs/phase10r/teacher-binding.yaml`, excluding only its own hash field. It binds all of the
following together:

- Apery 2.0.0 binary SHA-256
  `8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403`;
- KKP SHA-256 `422b23bced817ecb3430adf1d2621f5a7934263b4e46673ab80cf34633537fa5`;
- KPP SHA-256 `4906c48c201a102ec02217216929c20f73ab364e79be26e6213a02c04e454805`;
- book disabled, `Eval_Dir=eval/20190617`, `Eval_Hash=256`, `MultiPV=3`, `Threads=4`,
  `USI_Hash=1024`, and ponder disabled;
- 25,000 nodes, MultiPV 3, four threads, 1,024 MiB hash, and concurrency one;
- current-side-to-move root score perspective, canonical OpenShogiAI centipawns, separate search
  mate namespace, and the frozen signed-log score transform;
- the exact target-semantics file; and
- the complete Phase 4 labels-v2 manifest, label JSONL, benchmark, and their hashes.

Changing any one of these fields produces a different identity and stops validation.

## Smallest teacher-binding stage

No new teacher call is authorized. The existing complete 10,000-row labels-v2 artifact already
uses the exact canonical teacher and exhausts the 1M rung's frozen 10,000-label cap. Calibration
must join it to the current 1M prepared rows by canonical SFEN and retain only rows whose old and
current protected splits agree:

| Use | Rows | Detail |
| --- | ---: | --- |
| stage-3 train | 6,570 | 6,407 cp and 163 mate rows |
| stage-4 validation | 1,880 | 1,831 cp and 49 mate rows |
| excluded Phase 4 test | 1,064 | never used for fitting |
| absent from current preparation | 467 | no current lineage; excluded |
| protected-split mismatch | 19 | excluded fail-closed |

No public, internal, source-held-out, or final-holdout row may enter the calibration input.

Stage 3 resumes each exact stage-2 checkpoint, runs one deterministic epoch at batch size 128,
learning rate `0.0001`, weight decay `0.0001`, and seed `20260729`, and uses only the teacher's
returned contiguous MultiPV prefix for source-local pairwise/listwise ranking. The primary may use
available cp and mate targets; unavailable targets remain masked. PackedSfenValue remains disabled.

Stage 4 fits one positive monotonic affine calibration on the 1,831 validation cp rows. The pair
candidate fits Apery cp from WDL log-odds; the primary fits Apery cp from the inverse transformed
direct score. The fit is frozen before select-hard and may not be refit after selection or Arena.

Every child must then pass exact parent/label/teacher hashes, finite positive calibration, zero
forbidden-split rows, zero new teacher calls, OSAVAL02 Python/native/Wasm parity,
incremental/full-recompute/unmake parity, frozen source-held-out and ECE regression limits, zero
new tactical failures, and its existing throughput floor. No failed child may be represented as
teacher-bound.

## Lineage and gate behavior

`docs/model/phase10r-candidate-lineage.schema.json` defines the closed durable manifest.
`open_shogi_training.phase10r_lineage` validates the live teacher fingerprint, prepared-data and
pretraining parents, exact label artifacts, calibration-input counts, stage outputs, versioned
OSAVAL02 metadata, byte-exact equality between stage-3 checkpoint tensors and exported float32
weights, the stage-4 affine receipt, parity receipt, and all acceptance gates. Missing lineage is
not an error during inspection, but a present malformed or drifted lineage stops closed.

`select-hard` now distinguishes the original provenance gate from later selection work. With no
completed lineage it retains the exact `no completed teacher-bound candidate` stop. Once at least
one lineage validates, that gate has passed; the command still stops before selection because
implementing or executing hard-example selection is outside this repair. This prevents a success
receipt from claiming that selection occurred.

## Frozen-control changes and rationale

- `configs/phase10r/teacher-binding.yaml`: new exact case-3 parent, teacher, label, calibration,
  and acceptance control. It changes no architecture, source approval, split, holdout, threshold,
  resource ceiling, Arena/self-play rule, or promotion policy.
- `docs/model/phase10r-candidate-lineage.schema.json`: closed candidate-lineage interchange.
- `training/open_shogi_training/phase10r_lineage.py`: fail-closed live validators.
- `training/open_shogi_training/phase10r.py`: includes the new control and evidence in the frozen
  hash set.
- `training/open_shogi_training/phase10r_run.py`: adds read-only completed-lineage validation.
- `training/open_shogi_training/phase10r_campaign.py`: makes select-hard consume validated lineage
  without weakening its provenance gate.
- focused tests prove the positive path and identity, label, parent, parity, and pretraining-only
  rejection paths.
- `prompts/LUNA_PHASE10R_TEACHER_BINDING_EXECUTION.md`: freezes the only authorized next execution.
- both Phase 10R hash registries are refreshed after review, with this section as the per-change
  rationale.

## Next execution boundary

The next execution is unblocked at the specification and input-provenance boundary: all exact
parents, teacher files, label artifacts, joins, counts, hyperparameters, outputs, and gates are
frozen and locally available. Luna must implement only the two missing bounded preparation and
calibration commands described by the execution prompt, execute them resumably, validate the
lineage, and retry select-hard. It must stop at any other backend gap and must not improvise the
selection algorithm or enter labeling, Arena, or self-play.
