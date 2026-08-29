# Luna Max execution prompt — Phase 10R teacher binding

You are Luna Max executing only the bounded Phase 10R case-3 teacher-binding repair.

Repository: the current OpenShogiAI checkout.

Required branch: `codex/phase10r-execution`

Subagents are forbidden. Do not spawn another agent.

## Authority boundary

Read `PHASE_10R_TEACHER_BINDING_REPAIR.md`, `configs/phase10r/teacher-binding.yaml`,
`docs/model/phase10r-candidate-lineage.schema.json`, the original frozen plan/curriculum, and
`prompts/LUNA_PHASE10R_EXECUTION.md` before acting. The teacher-binding YAML is the sole machine
control for this execution.

Do not change architecture, features, model matrix, source approvals, prepared data, protected
splits or holdouts, target or score semantics, teacher identity, label artifacts, label budget,
stage order, hyperparameters, thresholds, resource ceilings, Arena/self-play rules, or promotion
policy. Do not publish, push, deploy, promote, inspect a final/public/internal holdout, call the
teacher, create labels, run `label-hard`, run Arena/cross-play/self-play, or overwrite either 1M
pretraining checkpoint or artifact.

## Initial exact commands

```sh
repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"
test "$(git branch --show-current)" = codex/phase10r-execution
test -z "$(git status --porcelain)"
make phase10r-validate
PYTHONPATH=training uv run --frozen python -c 'from pathlib import Path; from open_shogi_training.phase10r_lineage import load_teacher_binding_control; control = load_teacher_binding_control(Path(".")); print(control["teacher_binding_identity"]["identity_sha256"])'
```

The last command must print exactly
`781a45570ce6c88e29e4c9f7f3acb96e3981bb74d7da6fb10ff80cdd76f51cbf`. Stop immediately on a
wrong branch, dirty tree, failed hash, changed identity, or missing local artifact.

## Missing bounded interface to implement

If absent, implement only these two `open_shogi_training.phase10r_run` commands under the frozen
control. Add focused tests before running them.

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run prepare-teacher-binding --root . --scale 1m
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run calibrate-teacher --root . --scale 1m --variant sparse-pair-policy-wdl --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run calibrate-teacher --root . --scale 1m --variant factorized-pair-triple-policy-score --resume
```

`prepare-teacher-binding` is bounded and must make zero teacher calls. It must verify all hashes in
the control, join the exact labels-v2 rows to current prepared train/validation rows by canonical
SFEN, enforce identical protected splits, and atomically publish immutable train/validation JSONL
plus an `open_shogiai_phase10r_teacher_calibration_input/v1` manifest. The manifest must record:

- train 6,570; validation 1,880; validation cp 1,831;
- excluded Phase 4 test 1,064, absent 467, split mismatch 19;
- zero forbidden-split rows and zero new teacher calls;
- the exact control, teacher-binding identity, preparation-manifest, label-manifest, labels, and
  output hashes.

Reject duplicate SFEN, duplicate label position IDs, noncanonical SFEN, any teacher/config/score
perspective drift, any MultiPV move outside the current Rust legal-root mask, and any count or hash
mismatch. Never read public, internal, source-held-out, or final-holdout content.

Each `calibrate-teacher` command must:

1. load, hash, and retain the exact stage-2 parent named in the control without modifying it;
2. write only below the variant's versioned `teacher-bound-v1` child directory;
3. run one resumable deterministic stage-3 epoch with batch 128, learning rate `0.0001`, weight
   decay `0.0001`, and seed `20260729` over the 6,570 approved train rows;
4. keep PackedSfenValue disabled and mask every unavailable target;
5. fit the frozen positive monotonic affine stage-4 calibration only on the 1,831 validation cp
   rows, using WDL log-odds for the pair candidate and inverse transformed direct score cp for the
   primary;
6. make the completed stage-3 checkpoint record the exact parent-checkpoint SHA-256,
   calibration-input-manifest SHA-256, and teacher-binding identity SHA-256; publish stage 4 as a
   closed `open_shogiai_phase10r_teacher_calibration/v1` receipt binding that checkpoint, the 1,831
   fit rows, affine parameters, and zero teacher calls;
7. export a new non-overwriting float32 OSAVAL02 artifact from the exact stage-3 checkpoint weights,
   with the stage-4 affine parameters, whose dataset/training-manifest hash is the exact
   calibration-input-manifest file SHA-256 and whose training reference is exactly
   `phase10r-1m-{variant}-teacher-bound-v1`;
8. run Python/native/generated-Wasm OSAVAL02 parity and incremental/full-recompute/unmake parity;
9. enforce every acceptance gate in `teacher-binding.yaml`; and
10. atomically publish `candidate-lineage.json` conforming to the closed schema only after all
   gates pass.

One trainer is the only heavy worker. Run the variants sequentially. Preserve resumable state on
resource pressure or failure. A failed or partial output must never have `status: completed` and
must never satisfy the lineage validator.

## Exact post-calibration validation and retry

After both variant commands finish, run exactly:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run validate-teacher-binding --root . --scale 1m
make phase10r-validate
make phase10r-sanity
make phase10r-memory
make check
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run select-hard --root . --scale 1m
```

The validation receipt must name at least one fully validated teacher-bound candidate. Do not run
`select-hard` before that receipt passes. The final command is a retry of the next existing gate,
not authority to implement or run production teacher labeling. The bounded Sol repair deliberately
stops select-hard after teacher-binding authorization; if it reports that selection execution is
outside the repair, preserve that receipt and stop. Do not improvise a selection algorithm, weaken
the lineage requirement, or continue to `label-hard`.

## Final report

Report the clean commit, calibration-input manifest and SHA-256, each immutable parent checkpoint
and artifact SHA-256, each teacher-bound checkpoint/artifact/lineage SHA-256, affine parameters,
all parity and acceptance results, the validate-teacher-binding receipt, the select-hard retry
receipt, zero teacher calls, and the exact next stop reason. Do not push.
