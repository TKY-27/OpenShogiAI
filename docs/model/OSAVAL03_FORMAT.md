# OSAVAL03 runtime and format

Each perspective rotates White by 180 degrees, encodes own/enemy piece kind at its displacement
from that perspective's king, the oriented king square, unary hand counts, and relative side
to move. Promoted pieces have distinct identities. There are **8,427** collision-free features:
`17*17*28 + 81 + 2*7*18 + 2`. History is adjudicated by exact search rules rather than being
invented from a standalone SFEN.

A shared sparse learned table and bias produce two accumulators. The trunk input is
`[clip(us), clip(them), clip(us)*clip(them)]`, with clipping to `[0,1]`. The product term learns
interactions between latent features without a quadratic piece-pair table. A 16-unit clipped
layer feeds four learned outputs: direct side-to-move cp, then loss/draw/win auxiliary logits.
The scalar is primary and never reconstructed from WDL. Runtime rounds ties away from zero
and saturates to +/-20,000cp; search mate scores remain in their separate namespace.

No policy indexing is added. Existing OSAVAL02 policy encoding remains tested for legacy
comparison only; OSAVAL03 ordering uses the engine's legal search heuristics. Adding a policy
head would require a new explicit experiment with equal-clock benefit, outside this two-variant
freeze. No handcrafted evaluation, residual, composite, teacher, fallback or opening book is
reachable in the pure-only artifact.

## Incremental runtime

Non-king moves update only from/to piece features, captured/dropped hand ordinal and side-to-move
features. A moved king refreshes only its own perspective. The opponent perspective remains
incremental even when the other king moves. Table and bias values lie on an exact Q20 grid (multiples of 2^-20); float64
accumulators sum these bounded values exactly. This prevents cancellation from losing small
features beside large accepted weights. Off-grid model values fail loading. Clipped inputs
are cast to float32 before pair multiplication. Saved parent states restore unmake exactly;
floating-point inverse additions are never used for undo. Full refresh is the parity
oracle; tests exercise long legal lines plus capture/drop/promotion/king moves.

The target cached-head budget is <=10 microseconds for W256 and <=20 microseconds for W512 on
this host, with O(width) non-king updates. These are engineering targets, not substitute strength
gates. `phase10v_probe` reports full versus cached inference plus 128-node and 50ms legal search
behavior. Timing targets are historical and are not new measurements. Tracing is disabled by default;
an explicit maximum of 10,000 records captures the actual static/qsearch/PV-horizon evaluations.
Only real qsearch/PV leaf records enter the corresponding leaf distribution categories.

## OSAVAL03 wire contract

OSAVAL02 cannot unambiguously describe the sparse accumulator, interaction products, direct cp
semantics and two dynamic widths. One successor is introduced; OSAT10A1 and OSAVAL02 remain
explicitly selected immutable comparison formats, never auto-detected fallback paths.

Header: 44 bytes, little endian `<8s6IQf>`:
magic `OSAVAL03`, version3, feature_schema1, feature_count8427, width256/512, hidden16,
heads4, random seed u64, cp scale1. Payload is row-major float32 in this exact order:

1. table `[8427,W]`, bias `[W]`
2. hidden weight `[3W,16]`, hidden bias `[16]`
3. head weight `[16,4]`, head bias `[4]`

A 32-byte SHA256 trailer authenticates **header and payload**. External SHA256 binds the entire
artifact. Loaders reject unknown dimensions/scale/schema/version, wrong size/checksum,
trailing bytes, nonfinite or excessive parameters, off-grid accumulator tensors, and invalid
expected hash. Table and bias Q20 encoding is part of feature_schema1, not an optional loader
repair. Training uses quantization-aware straight-through gradients; exported values are
verified after quantization. Python and the
same native/Wasm core implement this contract. Learned scalar values never borrow teacher
mate-score constants.
