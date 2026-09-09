# Current direction

The first cleanup stage is complete on the `codex/oss-minimal-freeze` branch. The former
Phase 10–10V campaigns and Sunday deadline are closed. Their immutable receipts do not
validate this reorganized tree, and their all-artifact preservation rules are superseded.
This file is the only current-state and next-session entry point.

## Baseline and limits

The local comparison baseline is `100k-w256-hard2` (`OSAVAL03`, W256), SHA-256
`859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480`.
It is frozen for comparison; iterative W256/W512 training or enlargement is not the default plan.
The built-in default remains `handcrafted-experimental`. No model promotion or weight
publication is authorized. See [model handling](model/handling.md).

Historical 400-game results against handcrafted: equal nodes score 0.175, lower bound 0.13625;
equal wall clock score 0.1525, lower bound 0.11875. These are retained summaries, not rerun
results or proof of amateur-dan strength. W256 hard4 stopped twice at `p0039-0` with
`error: pure runtime proof failed`; the coordinator reported a player exiting before a complete
response. The cause remains unresolved. Its minimal local reproduction is retained.

## Next work and authorization

Next: a small prototype of the preferred core design, minimal automated verification, then
integration that permits real play in OpenShogiUI (OSUI). Stable wins against amateur first-dan
are not a prototype acceptance requirement. Large-scale training begins only after the user
judges the prototype promising and explicitly approves it. Public release and default-model
promotion require separate approval too. None starts as part of cleanup.

Originality may come from the combined evaluator, search and time-allocation design.
A non-neural evaluator is a candidate, not an adopted design or a claimed world first.
Astra owns implementation and important decisions. Luna Max handles execution, management,
monitoring and mechanical work for an approved large training run; Astra is not a permanent monitor.

Allocate time from the whole game clock (for example, three minutes sudden death), not a
fixed number of seconds per move. A public OSUI build may use an opening book; research
results without a book must remain distinct. External teacher delegation during play and
hand-written evaluation injected into nonterminal positions are not authorized.

Data is not restricted to AobaZero, but third-party terms are not automatically cleared.
Track unique new positions separately from total training exposures. Prevent overlap between
source games, descendants and split components; repetitions alone do not prove improvement.
Do not inspect, train on or repartition the final holdout. Rights and split checks remain mandatory.

## Handoff

No repository-cleanup blocker prevents a small prototype. The exact design is still to be chosen
in that work. Before using the frozen engine for OSUI play, investigate the hard4 runtime-proof
failure and run the UI repository's actual integration checks. Historical large evaluation
derivatives were removed; they are unavailable until deliberately reconstructed and verified.
Small protected evaluation originals and split identifiers remain local and unopened.
Build and test commands are in [development](development.md). Validation of this cleanup is
recorded below; local detailed outputs are under `local/maintenance/`.

## Cleanup validation

- `make check`: passed; 381 Rust tests and 878 Python tests, no skips. Includes format/lint,
  repository/license/provenance checks, document references and standard native/Wasm builds.
- `make pure-build` and `make frozen-smoke`: passed with the baseline SHA above; native search
  and actual Wasm in Node exercised. This is not browser GUI or complete-game validation.
- One saved hard4 request reproduced exit 2 and `pure runtime proof failed` using the retained
  historical binary. Root cause unresolved; no full Arena rerun or strength reassessment.
- Installed Apery identity/source/license/evaluation verification passed after its duplicate
  download archive was removed. No teacher labeling or new data acquisition ran.
- Project build products were pruned with Cargo after verification; dependency caches and the
  Python environment remain. Standard native/Wasm build was checked again after that pruning.

The local cleanup record is `local/maintenance/cleanup.json`. No push, merge, release, deployment,
model promotion, training or OpenShogiUI modification was performed.
