# Phase 4–6 Decision Log

This log records conservative implementation decisions made under the approved bounded
Phase 4–6 scope. Observed measurements belong in the corresponding phase reports; this file
records why the implementation took a particular path.

## Teacher and labels

- **Teacher:** Apery Rust v2.0.0 was selected from its official repository and release because
  it is versioned, USI-compatible, supports fixed-node search and MultiPV, builds on arm64
  macOS without administrator access, and ships its evaluation files with explicit evidence.
  It remains an ignored local executable and communicates with OpenShogiAI only through USI.
- **Build compatibility:** the official release lock does not compile with Rust 1.94 because
  its old `num-bigint` conflicts with the stabilized integer `div_ceil` API. The setup process
  may create a separately recorded local compatibility lock, but it must not edit or import
  teacher source into the OpenShogiAI production engine.
- **Analysis budget:** the resource benchmark authorized 25,000 nodes, MultiPV 3, one teacher
  process, and four teacher threads. Labeling proceeded through 10, 100, 1,000, and 10,000
  positions. The last number is a hard goal-wide cap, not a suggested batch size.
- **Short MultiPV output:** Apery can return a nonempty contiguous MultiPV prefix shorter than
  the requested three even when the independent rules engine finds additional legal root
  moves (notably an inferior non-promoting alternative beside a promoting move). A short
  result is accepted only as incomplete teacher coverage: every returned root and full PV
  must replay legally, ranks must be contiguous from one, rank one must match `bestmove`, and
  roots must be unique. Omitted moves receive no fabricated score. Reports distinguish the
  requested and returned counts and quantify positions where legal-root count is larger.
  Bounded lower/upper-bound search lines are never treated as exact labels.
- **Integrity:** labels, selection, benchmark, teacher identity, and source positions are bound
  by hashes. Existing append-only artifacts must be revalidated rather than rebaselined on
  resume, and every teacher move/PV must replay legally through the independent Rust rules
  boundary.

## Initial model

- **Architecture:** the first original value model is a small configurable MLP with two
  64-unit hidden layers, ReLU, dropout 0.1 during training, one value output, and one
  training-only auxiliary policy-agreement head. It starts from the repository seed and
  random initialization; no third-party weights are used.
- **Default features:** absolute-color board planes, normalized hands, side to move, and both
  king coordinates produce 2,287 inputs. Pseudo-attack planes are implemented but disabled by
  default so the first Rust/Wasm-oriented model stays small and easy to audit.
- **Data isolation:** training uses only the train split. Validation selects checkpoints; the
  test split is available only through an explicit final-test command. Phase 6 replay examples
  carry factual game outcomes and have no fabricated teacher or policy target.
- **Device:** use PyTorch MPS when it is built, available, and stable for the configured
  operations. CPU fallback is valid and must be recorded; MPS availability is not a success
  criterion.
- **Export:** `OSAVAL01` is a small, checksummed, closed binary format implemented independently
  in Python and Rust. It carries the feature/architecture versions and supports float32 and
  symmetric per-layer int8 weights without adding a large Rust inference runtime.

## Search, arena, and local play

- **Fail closed:** an explicitly requested neural evaluator must not silently fall back after
  a load/configuration failure. Missing-model fallback is allowed only when the caller
  explicitly selects the handcrafted evaluator.
- **Immutable evidence:** model/opening bytes are loaded once and hashed from those same bytes.
  Arena resume trusts neither mutable paths nor state summaries: CSA identity, assignment, and
  replayed result must agree before a game is reused.
- **Bounded comparisons:** Phase 5 comparisons use fixed nodes, alternating colors, fixed
  seeds, and at least 40 games per reported comparison. Results at this scale are provisional
  and do not justify broad playing-strength claims.

## One bounded generation

- **Generation size:** Phase 6 uses exactly 40 games: ten normal-start color pairs and ten
  approved-start-set color pairs, with two workers and an 8 GiB configured ceiling.
- **Teacher cap:** because Phase 4 already reached 10,000 labels, this generation may reuse
  already-labeled hard positions but must not request another teacher label. New self-play
  examples train only against their factual current-side game outcome.
- **Promotion:** `promoted`, `rejected`, and `inconclusive` are first-class outcomes. A
  challenger is never promoted merely to prove the pipeline; all configured evidence,
  legality, crash, group, side-balance, and confidence conditions must pass.
- **Human games:** human moves are written to a pending-review evidence stream and never become
  automatic training labels.

## Reproducibility and review

- Local dependency, teacher, data, and model caches are accelerators only. Locked dependency
  resolution, official-source hashes, manifests, and setup commands must be sufficient to
  reconstruct or reject missing/corrupt state.
- High- and medium-priority defects, and findings that affect playing strength, user
  experience, security, data provenance, or evidence integrity, remain mandatory fixes.
  A low-priority finding may be deferred when its practical improvement is small, it affects
  neither playing strength nor user experience, and the proposed change creates a material
  regression or compatibility risk. Every such deferral records the residual effect and a
  concrete condition for reopening it.
- Independent review is limited to two passes for the current change set: the completed
  discovery review and one final review of stable post-fix bytes. After findings from the
  final pass are resolved and verified, work proceeds to the integration gates and bounded
  phase tasks rather than starting another review cycle. The final pass is a gate for
  material findings: if it finds none, minor residual observations are recorded and do not
  start another fix-and-review loop. If it finds a material defect, that defect is fixed and
  verified directly, without a third independent review pass.

### Review findings deliberately deferred under that rule

- **Log/quarantine path spelling:** artifact contents, outcomes, and hashes remain verified,
  but terminal-attempt log and quarantine references are not forced into one additional
  retry/recovery filename grammar. Reopen if these paths become a cleanup or trust boundary.
- **Start-validation process receipt:** each start is independently parsed and replayed by the
  Rust rules engine, but the validation-result schema is not revised solely to add a second
  per-command process receipt. Reopen if exact process provenance becomes a promotion gate.
- **Retry-crash side attribution:** the overall retry-crash count and promotion crash limit
  remain exact. The current v1 side breakdown places a pair-level retry count on the pair's
  first result row even though the failed process may not have reached either game, so that
  field must not be interpreted as side-specific crash evidence. Reopen when side-specific
  crash telemetry becomes a decision input, using a versioned results/analysis schema rather
  than a silent semantic change.
- **Normalization-report semantics in Rust:** the report artifact remains hash/size bound,
  while its underlying games, positions, counts, splits, provenance, and legality are
  independently validated. Rust does not duplicate the complete Python report-statistics
  implementation. Reopen when the normalization-report schema is formally shared or revised
  across runtimes.
- **POSIX shell descriptors above nine:** ShellCheck reports `SC3023` for the retained setup
  descriptors numbered 10–17. The supported macOS `/bin/sh` accepts them, and moving the
  simultaneously retained archive, source, build, and lock authorities into the portable
  0–9 range would materially increase cleanup and identity-regression risk for no current
  playing-strength or user-facing benefit. Reopen if the setup script must support a shell
  that restricts file descriptors to 0–9; at that point, move the retained-authority workflow
  into one bounded Python process instead of renumbering the shell transaction in place.
