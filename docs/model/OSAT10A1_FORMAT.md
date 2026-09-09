# OSAT10A1 runtime and format

The frozen a1 architecture uses the project-owned `OSAT10A1` container introduced in
`93d2ec1303c46000cdebd4c868f18b3ab8b51c9c`. It is **not OSAVAL02**. OSAVAL02 remains the
closed Phase 10R c0 pair/pair-triple format described in `OSAVAL02_FORMAT.md`; no loader
aliases one format to the other and neither falls back to OSAVAL01.

Native and Wasm a1 use one implementation in `engine/core/src/phase10t.rs`. The diagnostic
crate re-exports this implementation. The Python exporter/reference remains
`training/open_shogi_training/phase10t_model.py`. The versioned playing contract is
`configs/runtime/pure_learned-a1-v1.json`; the legacy OSAVAL02 profile remains unchanged.

## Container and inference

The artifact is little-endian `44-byte header || float32 tensors || 32-byte SHA-256(payload)`.
The complete artifact SHA-256 is mandatory at the playing boundary, so the header and payload
are both bound to the caller's expected model identity. No trailing or optional data is accepted.

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | `OSAT10A1` |
| 8 | u32 | Format version 1 |
| 12 | u32 | Feature schema version 1 |
| 16 | u32 | 8433 sparse features |
| 20 | u32 | 128 accumulator channels |
| 24 | u32 | 32 hidden units |
| 28 | u32 | Four heads: direct cp, loss/draw/win logits |
| 32 | u64 | Random initialization seed |
| 40 | f32 | Exactly 1.0 score scale |

Tensor order is table `[8433,128]`, accumulator bias `[128]`, hidden weight `[256,32]`,
hidden bias `[32]`, head weight `[32,4]`, head bias `[4]`. Tensor values must be finite;
unknown versions, dimensions, scales, truncated/oversized data and checksum failures reject loading.
Only float32 is supported; projected quantization is not an accepted format.
Every parameter must also have absolute magnitude at most 1,000,000. This conservative
load-time bound keeps worst-case accumulator and head intermediates finite in float32
(less than 5e24 even with 512 feature contributions). The identical Python/native/Wasm
check rejects checksum-valid extreme finite tensors before inference instead of silently
saturating overflow. This defensive runtime validation repair does not change wire dimensions.

Features use both king-relative perspectives and authoritative board/hand state. Accumulators
are concatenated side-to-move first, passed through ReLU, then the 32-unit affine/ReLU layer
and four affine outputs. Direct cp rounds ties away from zero and clamps to ±28999.
WDL logits never supply mate scores. Search owns exact terminal/rule handling and mate distance.

OSAVAL02 history semantics are shared: unavailable history means repetition one and no
continuous-check owner; known histories carry repetition 1..4 and mutually exclusive relative
check ownership. A standalone SFEN cannot prove its preceding game history. Game/search APIs
must retain authoritative history rather than inferring it from the SFEN move number.
