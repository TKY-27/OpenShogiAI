# Phase 10A — Format compatibility report

このレポートは、公式形式定義に基づく bounded reader の実装結果である。外部artifact本体は、既存承認のAobaZero `w4745.csa` 以外を取得していない。形式fixtureは parser regression test 用にメモリ上で作成したもので、学習データではない。

## Adapter matrix

| format | adapter | external sample | fixture result | exact position identity | label policy |
|---|---|---|---|---|---|
| CSA / AobaZero CSA | `parse_csa_sample(..., strict_aobazero=True)` + existing `adapt_aobazero_csa` | `w4745.csa`: 19,762 bytes, 109 moves | pass | no; move-prefix hash only | played move、raw annotation、`v`、terminalを保持。勝敗・score perspectiveは推測しない |
| KIF | `parse_kif_sample` | no external file | 1 numbered move | no; KIF move-prefix hash only | Japanese move textとraw suffixを保持 |
| HCPE | `parse_hcpe_sample` | no external file | 1 record / 38 bytes | yes for 32-byte HCP | eval int16、bestMove16、raw/result mappingを分離 |
| HCPE3 | `parse_hcpe3_sample` | no external file | 1 game / 1 ply / 2 candidates / 50 bytes | no for per-ply prefix without replay | selected move、eval、candidate move/visit/probability、result/game-infoを保持 |
| PackedSfenValue | `parse_packed_sfen_value_sample` | no external file | 1 record / 40 bytes | yes for 32-byte PackedSfen | score、move、game ply、raw game resultを保持 |
| packed SFEN / binpack alias | same PackedSfen reader, `format_name="binpack"` | nodchip/tanuki candidate format | same 40-byte fixture | yes for source encoding only | PSV fieldをHCPE score/WDLへ自動変換しない |
| QPD source-specific archive | not implemented | archive not downloaded | not run | unknown | termsと内部schemaを取得するまでpending |

## Official layout evidence

- [DeepLearningShogi `csa_to_hcpe3.py`](https://raw.githubusercontent.com/TadaoYamaoka/DeepLearningShogi/master/dlshogi/utils/csa_to_hcpe3.py) defines the HCPE3 header, move info, and move-visit dtypes.
- [cshogi `_cshogi.pyx`](https://raw.githubusercontent.com/TadaoYamaoka/cshogi/master/cshogi/_cshogi.pyx) defines the 32-byte HCP, 38-byte HCPE, and 40-byte PackedSfenValue fields.
- [YaneuraOu / DeepLearningShogi format wiki](https://github.com/yaneurao/YaneuraOu/wiki/%E3%81%B5%E3%81%8B%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E5%AD%A6%E7%BF%92%E6%89%8B%E9%A0%86/0b69360878a8b4f693764ef7e1dd963c004e5fa5) describes HCPE's 38-byte layout and the source result codes.

第三者parserのコードをproductionへコピーせず、今回のreaderはPython標準ライブラリの`struct`、`hashlib`、bounded text handlingだけで実装した。既存のAobaZero CSA envelope adapterは、認証済みのPhase 3 dialect validationに限定して再利用している。

## Source-preserving normalized schema

schema IDは `phase10a_normalized_record/v1`。各normalized recordは以下の最上位領域を持つ。

```text
schema
record_id
source_artifact: source_id, artifact_id, format
position_identity: namespace, digest_sha256, exact, encoding/note
history_identity: namespace, digest_sha256, exact, availability
labels:
  played_move
  best_move
  policy_distribution       # raw visit count + derived probability, if present
  wdl / result               # raw code/terminal plus optional unambiguous name
  raw_source_score
  source_score_semantics
  score_perspective
  mate_representation
search: nodes, playouts, depth
provenance: raw_record_sha256, retrieval_scope, parser
license_decision
```

HCPE/PSVの整数scoreを同じ尺度へ揃えず、CSAの`v`を勝率やcentipawnへ変換せず、mateを通常scoreへ変換しない。HCPE3のcandidate visitsは raw `visits` と計算上の `probability`を併記し、source policyを失わない。`score_perspective`はソースが明示しない限り `unknown` である。

## Bounded and defensive behavior

- 既定sample上限は64 MiB、既定normalized record上限は100,000。
- HCPE3はmove count 512、candidate count 512を超えると拒否する。
- HCPE/PSVのpartial record、HCPE3のtruncated header/move/visit、KIF/CSAのmoveなし入力を拒否する。
- CSA/KIFはboard replayを行わない。したがってcross-format exact overlapの前提となるcanonical board keyはまだ生成しない。
- QPDは内部terms・schemaを読んでいないため、推測実装を追加していない。

## Verification

`tests/python/data/test_external_audit.py` に、CSA annotation保持、KIF text保持、HCPE 38-byte、HCPE3 variable visits、PackedSfenValue/binpack、exact overlap、split contamination、truncated inputの9テストを追加した。

実sampleの件数とbytesは [`artifacts/phase10a/sample-normalization-report.json`](../../artifacts/phase10a/sample-normalization-report.json) に固定した。
