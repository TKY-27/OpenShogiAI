# OSAVAL02 Sparse Evaluator Format and Runtime Contract

Status: normative for Phase 10R-C1, frozen 2026-08-22.

OSAVAL02 is the only Phase 10R sparse evaluator artifact. It is a closed, little-endian,
hash-bound container for the two candidates already frozen in
`configs/phase10r/model-matrix.yaml`. This document defines the artifact, feature encoding,
inference outputs, rejection behavior, and native/browser parity contract. It does not authorize
training, change the Phase 10R model matrix, or change the historical OSAVAL01 contract.

## 1. Container

An artifact is exactly `4064-byte header || tensor payload || 32-byte SHA-256`. The trailing digest
is SHA-256 over every preceding byte. No optional section, alignment padding, compression,
extension record, or trailing byte is permitted. The defensive maximum is 16 MiB.

All integers and IEEE-754 binary32 values are little-endian. Reserved and unused bytes must be
zero. Text fields are nonempty printable ASCII followed by NUL padding; embedded NULs and nonzero
padding are invalid. The 64-byte training-run reference is the sole exception: a printable
reference may consume all 64 bytes and therefore has no padding byte.

| Offset | Bytes | Type | Required value or meaning |
| ---: | ---: | --- | --- |
| 0 | 8 | bytes | `OSAVAL02` |
| 8 | 4 | u32 | format version `2` |
| 12 | 4 | u32 | byte-order marker `0x01020304` |
| 16 | 4 | u32 | header bytes `4064` |
| 20 | 8 | u64 | exact complete artifact length |
| 28 | 4 | u32 | architecture: `1` pair, `2` pair/triple |
| 32 | 4 | u32 | representation: `0` float32, `1` int8 |
| 36 | 4 | u32 | exact tensor count for the architecture |
| 40 | 4 | u32 | closed head bit field |
| 44 | 4 | u32 | exact parameter count |
| 48 | 4 | u32 | policy classes `13689` |
| 52 | 4 | u32 | scalar inputs `64` |
| 56 | 4 | u32 | trunk input `48` or `56` |
| 60 | 4 | u32 | trunk hidden width `128` |
| 64 | 8 | u64 | sparse hash seed `20260729` |
| 72 | 4 | u32 | triple cap `0` or `256` |
| 76 | 4 | u32 | signed hashing flag `1` |
| 80 | 4 | f32 | positive finite calibration scale |
| 84 | 4 | f32 | finite calibration bias |
| 88 | 4 | f32 | WDL epsilon in `(0, 0.01]` |
| 92 | 4 | u32 | maximum non-mate score `28999` |
| 96 | 32 | bytes | feature-schema SHA-256 |
| 128 | 32 | bytes | model-matrix SHA-256 |
| 160 | 32 | bytes | target-semantics SHA-256 |
| 192 | 32 | bytes | input-normalization SHA-256 |
| 224 | 32 | bytes | nonzero dataset/training-manifest SHA-256 |
| 256 | 32 | bytes | tensor-payload SHA-256 |
| 288 | 32 | bytes | selected quantization-config SHA-256 |
| 320 | 32 | bytes | generated move-index mapping SHA-256 |
| 352 | 32 | ASCII | exporter version |
| 384 | 40 | ASCII | exact lowercase hexadecimal Git commit |
| 424 | 64 | ASCII | training-run reference |
| 488 | 4 | u32 | tensor-table offset `512` |
| 492 | 4 | u32 | descriptor bytes `112` |
| 496 | 8 | u64 | payload offset `4064` |
| 504 | 8 | u64 | reserved zero |
| 512 | variable | descriptors | exact closed tensor table |

The eight compatibility hashes are raw 32-byte digests. Their frozen lowercase hexadecimal forms
are:

- feature schema: `fb5d69c96ae45ed308ee18ab7fd16d4fbefe0f5778e0b0ee2144879bcc7881df`;
- architecture: `50a6873b521f389c766010a0ed83fa5d2399d18a05a860fec78ef35f523dfb6b`;
- targets: `bafe7ba97319fa12d0c7a8ef3e3634fe033926fa77fcb781b67cd8ab12dda3bb`;
- normalization: `b19dea246a17e059010418f2ac04b5a75fb1e21a30ba050159e539a9f5f69185`;
- float32 config: `4ab7ec4932dd595a80b3ccddcf46f0fe42fe318c62558d79ca91da3218580327`;
- int8 config: `315c92526d38149c67d5fb8b97b88a177b7ee87932910ebdd20e42479d3c7041`;
- move mapping: `096b227cadae6e585977b688495f261b163ebd7d3b4d2cc7276555a1b297de2a`.

## 2. Tensor descriptors and architectures

Each 112-byte descriptor contains: 48-byte name, u32 dtype, u32 rank, four u32 dimensions, u64
payload-relative offset, u64 encoded length, u64 element count, f32 scale, i32 zero point, u32
flags, and four zero bytes. Dtype is `1` for float32 or `2` for int8. Dimensions after `rank` are
one. Offsets are contiguous from zero in the exact order below. Flags are `1`; zero point is zero.

| Tensor | Pair shape | Pair/triple shape |
| --- | ---: | ---: |
| `king_piece_embeddings` | 367416×4 | 367416×4 |
| `king_hand_embeddings` | 43092×4 | 43092×4 |
| `pair_hash_embeddings` | 65536×8 | 65536×8 |
| `triple_hash_embeddings` | absent | 32768×8 |
| `scalar_projection.weight` | 32×64 | 32×64 |
| `scalar_projection.bias` | 32 | 32 |
| `trunk.0.weight` | 128×48 | 128×56 |
| `trunk.0.bias` | 128 | 128 |
| `trunk.1.weight` | 128×128 | 128×128 |
| `trunk.1.bias` | 128 | 128 |
| `value_heads.weight` | 8×128 | 9×128 |
| `value_heads.bias` | 8 | 9 |
| `policy.move_embeddings` | 13689×16 | 13689×16 |
| `policy.context_weight` | 16×128 | 16×128 |
| `policy.context_bias` | 16 | 16 |
| `policy.move_offset` | 16 | 16 |
| `policy.log_temperature` | 1 | 1 |

The pair candidate has 2,413,321 parameters and exact artifact sizes 9,657,380 bytes (float32) and
2,417,417 bytes (int8). The pair/triple candidate has 2,676,618 parameters and exact sizes
10,710,568 bytes and 2,680,714 bytes respectively. Any other count, shape, order, dtype, byte
length, or head combination is incompatible.

Float tensors store finite binary32 values and scale `1`. Int8 tensors use independent symmetric
per-tensor quantization: `scale = binary32(max(abs(weight))/127)`, with `1` for an all-zero tensor;
values are round-to-nearest, ties-to-even, clamped to `[-127,127]`; dequantization is
`signed_i8 * scale`. NaN, infinity, nonpositive scale, nonzero zero point, and `-128` produced by an
exporter are invalid.

## 3. Feature and policy contract

Input is one strictly parsed SFEN position and four bounded history facts: availability,
repetition count `1..4`, and mutually exclusive current/opponent continuous-check ownership.
Unavailable history must use repetition one and both check flags false.

Board and hand owners are relative to the current side to move. Board piece kinds use the frozen
order pawn, lance, knight, silver, gold, bishop, rook, king, promoted pawn, promoted lance,
promoted knight, promoted silver, horse, dragon. Hand kinds use pawn, lance, knight, silver, gold,
bishop, rook. Squares use SFEN scan order (rank a through i, file 9 through 1).

King-piece and king-hand rows use the exact Cartesian indices implemented in the hash-bound
feature specification. Pair and bounded triple keys encode relative owners, piece kinds, squares,
signed king offsets, attack/pin/king-zone flags, and the fixed feature category. Keys are hashed as
SHA-256 of `OpenShogiAI/phase10r/features/v1\0 || LE-u64(20260729) || key`; the little-endian first
u32 selects the row modulo table size and bit zero of byte four selects sign. Sparse rows are
sorted before accumulation and checksum emission. Triple candidates are category/square sorted,
deduplicated, and capped at 256. Embedding sums are divided by emitted-row count.

The 64 scalars cover relative piece/hand counts, attack maps, king zones, legal mobility,
check/pin/promotion/material/history facts, sparse row counts, drop count, and constant bias. Each
is rounded to binary32 before inference and feature hashing. The feature checksum hashes every
tagged sparse `(u32 row, i8 sign)` and each scalar binary32 byte sequence.

Policy indices are the frozen bijection: normal moves are
`(from_index * 81 + to_index) * 2 + promotion_bit`; drops begin at 13,122 in `RBGSNLP` order. The
move-table mapping hash above is mandatory. Only legal moves produced by the shared rule core are
scored. Policy output is sorted by descending logit then ascending move index.

## 4. Output heads and score semantics

Both variants emit WDL softmax probabilities, mate-kind softmax, transformed mate distance,
clamped log variance plus variance, and policy logits. The pair/triple candidate also emits its
direct transformed score. The pair candidate derives its scalar from
`ln((win + epsilon)/(loss + epsilon))`.

For the direct head, inverse score transform is
`sign(x) * expm1(min(abs(x),1) * log1p(3000))`. The selected raw score then receives the stored
positive affine calibration and ties-away integer rounding, and is clamped to `[-28999,28999]`.
Perspective is always the current side to move. Uncertainty is never mixed into the score.

Mate classification and distance are auxiliary outputs only. Converting them into a search score
is forbidden. Search owns mate values: a terminal checked position with no legal move reports
`-30000`; a nonchecked no-legal-move position reports zero. This separation prevents a learned
mate label from colliding with exact search mate semantics.

## 5. Validation, portability, and parity

Python inspection/export, native Rust loading, and browser Wasm loading fail closed before
inference. Rejection includes wrong magic/version/endian, unknown architecture or representation,
wrong hashes/dimensions/heads/counts, malformed text, bad descriptors, gaps/overlap/trailing data,
truncation, size overflow, checksum mismatch, invalid scales, and non-finite tensors or outputs.
Native path loading additionally requires one stable regular nonsymlink file read under the
repository's anchored-file boundary. Browser loading accepts one bounded immutable byte array.

The native and generated-Wasm parity report envelope uses the exact shared schema identifier
`open_shogiai_osaval02_parity/v1`. The envelope has exactly `schema`, `modelIdentity`, and
`fixtures` at its root. A missing schema, either legacy runtime-specific schema, any other version,
or an incomplete model identity is incompatible and must be rejected before comparing inference
values.

The committed source-safe parity corpus contains synthetic rule states only and explicitly denies
game records, evaluations, labels, and training examples. Each fixture binds the canonical
three-field position SHA-256 and category/provenance metadata. Python, native Rust, and actual
generated Wasm must agree exactly on identity, hashes, discrete outputs, legal-move membership and
ordering, score integers, and terminal semantics. Finite floating outputs use relative tolerance
`1e-10` and absolute tolerance `1e-12`. Repeated inference in each compiled runtime must serialize
identically.

The deterministic fixture-only float32/int8 deviation gate is separately frozen at: WDL/mate
probabilities `1e-5` absolute, policy logits `2e-6` absolute, transformed score/mate distance/
uncertainty values `2.5e-5` absolute, calibrated score `1 cp`, and exact policy ordering. This is a
runtime quantization regression threshold, not the separate strength/Arena release gate required
by the frozen ablation plan.

OSAVAL01 remains a historical supported format for its existing consumers. It is not an alias,
fallback, migration source, or valid substitute for OSAVAL02. OSAVAL02 does not silently load an
OSAVAL01 artifact, and this phase does not promote either Phase 10R candidate to default runtime
champion.

## 6. Teacher-bound candidate lineage

The container's dataset/training-manifest hash and training-run reference identify an export, but
they are not independently sufficient evidence that a candidate used the frozen teacher. A Phase
10R artifact is teacher-bound only when a completed sibling
`open_shogiai_phase10r_candidate_lineage/v1` manifest validates against
`docs/model/phase10r-candidate-lineage.schema.json` and the live fail-closed validator. That
manifest binds the immutable pretraining parent, teacher binary and evaluation files, options and
search budget, score semantics, exact label manifest and rows, stage-3 checkpoint, stage-4 affine
calibration, candidate artifact, and Python/native/Wasm plus incremental parity receipts.

For a teacher-bound export, offset 224 contains the calibration-input-manifest file SHA-256 and
offset 424 contains `phase10r-1m-{variant}-teacher-bound-v1`. The original stage-2 artifacts retain
their existing preparation-manifest identity and remain pretraining-only. Missing, partial,
unknown, drifted, or failed lineage cannot be inferred from weights or repaired by metadata-only
mutation; it is rejected before select-hard authorization.
