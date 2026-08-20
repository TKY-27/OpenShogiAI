# Evaluation

Phase 2 establishes the Rust arena and benchmark boundary. Phase 7 adds a deliberately small
developer-evaluation workflow around the promoted Phase 6 champion. Generated games, teacher
analyses, and reports remain under ignored `artifacts/` storage.

The fixed official configuration is `configs/evaluation/phase7_official.toml`:

- two human/champion games, one for each human side;
- 10,000 nodes, depth cap 8, maximum 256 plies, and no opening book;
- pinned Apery 2.0.0 analysis at 25,000 nodes and MultiPV 3;
- a 4,096 MiB process-tree ceiling and at most 512 teacher calls;
- no automatic training admission.

Create a plan only from a clean commit. The plan binds the immutable Rust executable/build
receipt, Phase 6 registry revision and champion model, teacher configuration, exact game argv,
paths, and resource limits. Print and run the two interactive commands, then derive and verify
the evidence:

```sh
make phase7-validate-config
make phase7-plan
make phase7-commands
# Run the two printed human-play commands.
make phase7-prepare-games
make phase7-analyze
make phase7-curate
make phase7-verify
```

Planning and execution authorize the native engine against the clean current HEAD. The final
read-only verifier uses the plan's immutable build receipt and recorded commit instead, allowing
completed evidence to remain verifiable after later source or documentation commits.

Every CSA is replayed through the receipt-bound Rust CLI. Every recorded move position is then
analyzed by the separately installed teacher, and every returned teacher PV is replayed through
the same Rust legality boundary. The workflow keeps move-choice regret and root-evaluation
disagreement separate. Categories ending in `_candidate` are triage hypotheses, not causal
proof. A move outside the returned MultiPV cannot receive an exact regret value, so the
classifier remains conservative.

Human moves are observations only. Hard examples require an AI move, no opening-book choice,
an evidence-backed failure-candidate category, and the configured severity threshold. Even
selected rows are emitted with `status: pending_human_review` and
`autoTrainingEligible: false`; Phase 7 never appends labels, replay rows, or training inputs.

The authoritative bounded local run is recorded in `PHASE_7_REPORT.md`. Its two games were
ended deliberately by immediate human resignation after exercising both color paths, so its
results are pipeline evidence and must not be interpreted as a playing-strength measurement.

For arena metrics, `cutoffRate` and `pruningRate` remain distinct: the former counts beta-cutoff
nodes, while the latter measures the share of generated, post-filter move candidates skipped by
those cutoffs. The browser Evaluation Lab is a read-only arena-report viewer; it does not replay
rules, teacher searches, or Phase 7 evidence.
