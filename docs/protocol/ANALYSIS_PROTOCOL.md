# OpenShogiAI analysis protocol

Version: `open_shogi_analysis/v1`

This protocol controls a logical analysis engine that is distinct from play search. A host starts
one canonical SFEN root, repeatedly schedules bounded work slices, consumes completed-depth
updates, and stops or changes the root at any time. Repeated slices are the implementation of
“infinite” analysis: the logical session remains active until `stop`, while each individual slice
is bounded and interruptible.

The machine-readable request/response definitions are in
[`analysis-protocol.schema.json`](analysis-protocol.schema.json). Shared play-clock requests use
`open_shogi_time_control/v1`, defined by
[`time-control.schema.json`](time-control.schema.json). TypeScript types and a minimal
worker-transport client are in [`analysis-types.ts`](analysis-types.ts).

## Lifecycle

1. `start` supplies a canonical SFEN, `multiPv`, and every immutable compatibility hash.
2. The engine cancels the prior root, clears TT state only if evaluation/search semantics changed,
   and immediately publishes a matching cached update when one exists.
3. The host schedules `step` slices. Each completed iterative-deepening depth yields one `update`.
4. `change-position` or another `analysisStart` repeats step 2 for the new canonical root.
5. `change-multi-pv` selects a distinct cache key without changing legal move generation.
6. `stop` cooperatively cancels work but retains compatible completed cache and TT entries.
7. After a host worker failure, record `worker-failed`, create/restart the worker, call `restart`,
   display the returned cache update, and resume bounded slices.

Native reference transport:

```sh
cargo run --locked -p open-shogi-cli -- analysis
```

It accepts one JSON command per input line and emits one JSON event per output line. The Wasm
adapter exposes `analysisStart`, `analysisStep`, `analysisStop`, `analysisWorkerFailed`, and
`analysisRestart`. Browser work is cooperative so the host can pause the analysis worker while the
separate play worker is choosing a move.

## Cache identity

`AnalysisCacheKey` contains:

- canonical SFEN and the engine-local Zobrist hash (the SFEN protects persistent identity from
  hash collisions);
- model hash;
- evaluator-config hash;
- feature-schema hash;
- evaluation-semantics hash;
- search-options hash;
- opening-profile hash;
- MultiPV value.

All supplied hashes are lowercase SHA-256 hexadecimal. A changed model, feature schema,
evaluation meaning, or search option selects an incompatible cache namespace and clears reusable
TT state. A changed position, opening profile, or MultiPV selects a different cache entry while
retaining TT data when search/evaluation semantics remain compatible.

Each cache entry stores the latest fully completed depth, nodes, NPS, score, optional mate score,
MultiPV lines, per-root move statistics, caller-provided Unix timestamp in milliseconds, engine
version, and model hash. An update identifies whether it came from `cache` or renewed `search`.

## Resume semantics

“Resume” means immediate display of the last compatible completed-depth cache entry followed by a
new iterative-deepening run that reuses valid transposition and root evidence. OpenShogiAI does not
serialize or restore suspended recursive call stacks. Partially searched depths are not published
as completed cache entries.

## Resource isolation

`open_shogi_resource_budget/v1` exposes play threads, analysis threads, separate hash allocations,
pause-analysis-during-AI-turn, and maximum aggregate memory. Current search is single-threaded, so
play threads must be one and analysis threads must be zero or one. Hash allocations must fit the
aggregate memory ceiling. Play and analysis own different `SearchEngine` instances; a coordinator
can deny analysis slices while play is active.

## Conformance

Conformance is covered by core position-switch, cancellation, invalidation, worker-restart, cache,
and transposition-reuse tests; the native line protocol test; and the Wasm protocol lifecycle test.
Unknown schemas, unknown JSON fields, noncanonical positions, invalid hashes, out-of-range MultiPV,
and unbounded slice requests fail closed.
