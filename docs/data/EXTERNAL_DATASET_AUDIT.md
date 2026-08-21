# Phase 10A — 外部将棋データ監査

監査日: 2026-08-21 (Asia/Tokyo)

ブランチ: `codex/phase10a-external-data-audit`
目的: 公式著者・プロジェクトの公開データについて、権利、容量、形式、系譜、重複リスクを bounded sample の範囲で記録する。

このPhaseでは学習、自己対局、Arena、teacher relabeling、大規模データ取得、重み作成、push/releaseを行っていない。ダウンロードしたデータ本体は既存承認範囲の `w4745.csa` だけであり、解析後に削除した。全artifactの機械可読台帳は [`configs/data_sources_external.yaml`](../../configs/data_sources_external.yaml) にある。

## 結論

| 判定 | artifactレコード | 要点 |
|---|---:|---|
| approved | 2 | OpenShogiAIが既に承認していたAobaZero no-noiseの厳密な100-object範囲と、その中のbounded sample `w4745.csa` のみ |
| pending | 18 | GCT、nodchip/tanuki、たややん、Qhapaq、追加AobaZeroの権利条件またはexact manifestが不足 |
| denied | 2 | 公開評価testはimmutable holdout、Floodgateは既存provenanceで未検証かつML不許可 |

`approved` は新しい権利判断ではなく、既存の [`configs/data_sources.yaml`](../../configs/data_sources.yaml) と [`DATASET_CARD.md`](../../DATASET_CARD.md) の厳密な100-object範囲を再確認したもの。AobaZeroの公開indexを見て範囲を拡張することはしていない。

## 公式ソースと権利判断

### AobaZero

既存の承認証拠は、固定commitの公式READMEにある「USI engine aobaz belongs to GPL v3. Others are in the public domain.」という記載と、固定hashの公式sample indexである。今回確認した `w4745.csa` は既存catalog内にあるため、学習利用と再配布は既存方針の範囲で approved とした。一方、derived weights、商用利用、attributionの明示条件はこの監査で新たに推測していない。

- [AobaZero official project](http://www.yss-aya.com/aobazero/)
- [AobaZero pinned README](https://raw.githubusercontent.com/kobanium/aobazero/5eb944165300d5b88924c917a147e80d9d173eed/README_en.md)
- [AobaZero current sample index](http://www.yss-aya.com/aobazero/no_noise/sample.html)

current sample index は `23f45efcaecd760e1c25a99d2b920a17ad28bcd92c1ada05a41556471a105bf5` で、既存の固定hash `178b6bfdd4a128eef14bc1ae7d057fae3781230cef9dc5cc3e2351291e15f9ad` と異なる。この差分は既存catalogの権利判断を自動更新する根拠にはせず、live indexを pending とした。

### GCT / dlshogi

Tadao Yamaoka氏の [GCTデータセット公開記事](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701) は、`hcpe3/selfplay_gct-???.hcpe3.xz`、Floodgate系、たややん互角局面系、AobaZero converted HCPE、`gct/hcpe/...`、`suisho/hcpe` などの内容とGoogle Driveの場所を示す。しかし「公開した」という記述、Discord上の公開許可、由来説明だけでは、各artifactについて必要な training / derived weights / redistribution / commercial use の全条件は確定しない。従って全GCT候補を pending とした。

記事に示されたGoogle DriveフォルダのUI集計は約84GBだったが、これは個別ファイルの正確な合計やSHA-256ではない。full downloadは行っていない。

### nodchip / tanuki NNUE teacher-data

公式HF datasetページでは、`shogi_hao_depth9` と `tanuki-.nnue-pytorch-2024-07-30.1` の形式が YaneuraOu PackedSfenValue と説明され、各usedStorageは約320GBだった。HFの `cardData.license: mit` はdataset repository metadataであり、release asset / data rowsへの権利を自動的に確定する証拠としては扱わない。従って両方とも pending、かつPhase 10Aの容量予算外とした。

- [nodchip/shogi_hao_depth9](https://huggingface.co/datasets/nodchip/shogi_hao_depth9)
- [nodchip/tanuki-.nnue-pytorch-2024-07-30.1](https://huggingface.co/datasets/nodchip/tanuki-.nnue-pytorch-2024-07-30.1)
- [DeepLearningShogi official format/wiki reference](https://github.com/yaneurao/YaneuraOu/wiki/%E3%81%B5%E3%81%8B%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E5%AD%A6%E7%BF%92%E6%89%8B%E9%A0%86/0b69360878a8b4f693764ef7e1dd963c004e5fa5)

### たややん氏の互角局面集・公開教師データ

- [36SFEN / 互角局面集の公式紹介](https://yaneuraou.yaneu.com/2020/11/12/tayayan-36sfens/)
- [Tadao Yamaoka氏による5247局面版と比較](https://tadaoyamaoka.hatenablog.com/entry/2021/09/20/222018)
- [たややん氏の水匠教師データ案内](https://note.com/tayayan_ts/n/nacdfae58b4ef?magazine_key=m024e7c154df0)

公開ページとDrive fileの存在は確認したが、正式な再利用・商用利用・derived weights条件は確認できなかった。36SFENと5247局面版は、GCTの `taya36` 系artifactに由来する可能性が公式記事上で明示されているため、rightsが確定しても公開test/holdoutとして先に隔離する。

### Qhapaq / Qhapaq Pretty Daabi

- [Qhapaq公式の教師データ公開記事](https://qhapaq.hatenablog.com/entry/2021/11/23/220251)
- [Qhapaq dataset release](https://github.com/qhapaq-49/qhapaq-bin/releases/tag/dataset)
- [QPD_train.7z release asset](https://github.com/qhapaq-49/qhapaq-bin/releases/download/dataset/QPD_train.7z)

`QPD_train.7z` について、学習した評価関数を大会に出す・公開する場合のデータ利用 attribution を求める条件は公式記事にある。一方、7z内部のexact termsを監査しておらず、再配布、商用利用、derived weightsの範囲は未確定なので pending とした。archive本体は取得していない。

## ストレージ事前監査

| 項目 | 値 |
|---|---:|
| 監査開始時の空き | 345,509,875,712 bytes (約321.77 GiB) |
| 下限 | 214,748,364,800 bytes (200 GiB) |
| 下限までのheadroom | 130,761,510,912 bytes (約121.78 GiB) |
| Phase 10Aの安全なローカル予算 | 1,073,741,824 bytes (1 GiB) |
| bounded data本体 | 19,762 bytes |
| 公式evidence snapshot | 28,677 bytes |
| 合計 | 48,439 bytes |

既知の容量（取得せずに公式API/UIまたは既存catalogから記録した値）は以下の通り。`decompressed` と `normalized` の推定は、exact manifestなしでは範囲または unknown とした。

| artifact | compressed / source bytes | decompressed | normalized estimate |
|---|---:|---:|---:|
| AobaZero exact100 | 上限 104,857,600 | 104,857,600以下 | 4,646,400–12,390,400 bytes（Phase 3の15,488 positions） |
| AobaZero `w4745.csa` | 19,762 | 19,762 | 153,410 bytes（109 normalized recordsのJSON配列） |
| GCT Drive folder | 約84,000,000,000（UI集計） | unknown; 84GB以上 | 150,000,000,000–400,000,000,000（500M positions × 300–800） |
| nodchip Hao depth9 | 320,002,979,440 | 320,002,979,440 | 2,400,022,345,800–6,400,059,588,800 |
| nodchip tanuki | 320,002,292,200 | 320,002,292,200 | 2,400,017,191,500–6,400,045,844,000 |
| QPD_train.7z | 11,304,930 | unknown | unknown |
| Taya 36SFEN | 384,535 | 384,535 | 587,100–1,562,800 |
| たややん教師data | unknown | unknown | 45,000,000,000–120,000,000,000（約150M positions × 300–800） |

候補artifactの既知または観測値を単純合計すると約724,016,980,867 bytesだが、これは取得許可・exact file list・圧縮展開率を意味しない。full artifactはPhase 10Aの予算内に入らないため、取得していない。

## bounded sampleの結果

### AobaZero `w4745.csa`

既存承認sourceの1ファイルだけを bounded sample として取得し、hash、形式検証、source-preserving normalizationを行った。

| 項目 | 値 |
|---|---|
| bytes / SHA-256 | 19,762 / `6a08c7a9f519ce81505bbdf7c51905781413ddf8955a955b1b5b1d61a2703aad` |
| parser | `adapt_aobazero_csa` のstrict validation + Phase 10A CSA reader |
| normalized records | 109 move-prefix records |
| annotations / comments | 109 / 33 |
| terminal | `%TORYO`（勝者の推測はしていない） |
| source datetime | `2026-07-17T01:57:19` |
| raw score | `v`を原値として保持、単位・perspectiveはunknown |
| position identity | CSA move-prefix hash、board replayなしなので `exact: false` |

実レコードは学習用datasetへ昇格させず、解析後に削除した。`sample-normalization-report.json` は件数、hash、label保持方針のみを保存し、データ本体を含まない。

## 形式と正規化

実装は [`training/open_shogi_training/data/external_audit.py`](../../training/open_shogi_training/data/external_audit.py) にある。公式レイアウトを `struct` と標準ライブラリで読むだけで、第三者parser実装はproduction codeへコピーしていない。

- CSA / KIF: move prefix と raw annotation/move suffixを保持。board replayをしないため、position identityはexactではない。
- HCPE: 38-byte `HuffmanCodedPosAndEval` を読み、32-byte HCP、int16 eval、bestMove16、gameResultを保持。
- HCPE3: 36-byte header、move info、可変長 candidate visitsを読み、visit countと計算した確率を併記する。
- PackedSfenValue / binpack: 40-byte recordを読み、packed SFEN、score、move、game ply、raw game resultを保持。

共通schemaは `phase10a_normalized_record/v1`。`played_move`、`best_move`、`policy_distribution`、`wdl`/`result`、`raw_source_score`、`source_score_semantics`、`score_perspective`、`mate_representation`、`nodes/playouts/depth`、source/artifact/record IDs、raw hash、license decisionを別フィールドに置く。unknown scoreをcentipawnやside-to-moveの共通値へ変換していない。

形式ごとのfixture parse結果は [`docs/data/FORMAT_COMPATIBILITY_REPORT.md`](FORMAT_COMPATIBILITY_REPORT.md)、詳細な機械可読結果は [`artifacts/phase10a/sample-normalization-report.json`](../../artifacts/phase10a/sample-normalization-report.json) にある。

## 重複・漏洩の要点

詳細は [`docs/data/DATASET_OVERLAP_REPORT.md`](DATASET_OVERLAP_REPORT.md) にある。

- GCT記事は `aobazero/hcpe` を明記するため、AobaZero recordsのGCT内重複リスクは lineage-confirmed、exact countは未測定。
- GCT `taya36` / `selfplay-unique-900xx/...902xx` はたややん互角局面由来と明記され、Taya holdoutとの contamination riskが高い。
- たややん原版と5247局面版について、公式比較記事はHCPで1,895/1,957、24手目までの同一局面経路を含めると1,718/1,957を報告している。
- AobaZero CSA、HCPE3、PackedSfenを同じexact identityとして比較するには合法手順再生または共通canonical board keyが必要であり、今回のbounded sampleでは行っていない。
- 公開評価test datasetはtrainingに使わず、Taya 36SFEN/5247派生版も少なくともholdout候補として固定する。

## Sol向けの未解決事項

1. 各artifactについて、著者が明示した training、derived weights、redistribution、commercial use、attribution を分離した許諾表を確定する。
2. GCT/HF/Driveから exact file manifest、個別byte size、SHA-256、archive内termsを取得する。ただしその前に容量予算と保管先を決定する。
3. AobaZero approved 100-object範囲、GCT内AobaZero converted data、Qhapaq/Taya-derived dataのsource-level crosswalkを作る。
4. 形式ごとに canonical board identity と game/history identity を定義し、positionではなくgame/history単位でtrain/validationを分離する。
5. 公開test/holdoutのimmutable manifestを先に固定し、候補データの学習利用をdeny-by-defaultで管理する。

Phase 10Aはデータ採用・学習・昇格を決めない。上記を後続Solのdataset-design decisionへの入力として引き渡す。
