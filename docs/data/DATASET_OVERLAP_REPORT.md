# Phase 10A — Dataset overlap / leakage report

監査日: 2026-08-21

対象: 公式ページで確認できた外部artifactと、既存OpenShogiAI provenance
方針: exact identityが得られた場合だけ件数を測定し、公開記事のlineage記述をexact overlap件数へ昇格させない。

## 測定結果の分類

| 関係 | exact count | 判定 | 根拠 / 制限 |
|---|---:|---|---|
| AobaZero `w4745.csa` bounded sample内部 | 0 cross-source | 測定対象外 | CSA move-prefix identityは `exact: false`。同一sampleの109 prefixは形式検証用 |
| AobaZero records → GCT `aobazero/hcpe` | unknown | lineage-confirmed risk | [GCT公式記事](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701) がAobaZero棋譜のHCPE変換を明記。GCT本体未取得のためexact hash照合なし |
| AobaZero exact100 → GCT一般selfplay | unknown | pending | GCTの個別manifestとsource record IDsがない |
| たややん原版1,957 → Tadao派生5,247 | 1,895 / 1,957 (HCP); 1,718 / 1,957 (24手目同一経路を考慮) | measured in official source | [公式比較記事](https://tadaoyamaoka.hatenablog.com/entry/2021/09/20/222018) の報告値。今回その外部ファイルを再取得して再計算していない |
| たややん原版/派生 → GCT `taya36` | unknown | high leakage risk | [GCT公式記事](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701) がたややん互角局面からの生成を明記 |
| たややん原版/派生 → GCT `selfplay-unique-900xx/...902xx` | unknown | high leakage risk | たややん互角局面のGCT RL dataと明記。exact artifact listなし |
| たややん原版/派生 → QPD | unknown | unresolved | QPDのsource compositionが未監査。archive本体未取得 |
| GCT → QPD | unknown | unresolved | source-level manifestなし |
| nodchip Hao → tanuki NNUE data | unknown | unresolved | 両方PackedSfenValueだが、record/game IDsとhash manifestを取得していない |
| OpenShogiAI Phase 3 train/validation/test | 0 reported cross-split exact canonical positions | existing split evidence | `DATASET_CARD.md`のgame-only splitとunique raw/canonical countsに基づく既存結果。Phase 10Aで再計算していない |
| public evaluation test → any training split | 0 by policy | denied | 公開testはimmutable holdout。artifact location/licenseが未確定でも学習へ入れない |

「unknown」は重複がないという意味ではない。artifact本体、record ID、canonical board keyがないため測定できていないという意味である。

## 何をexact overlapと数えるか

[`external_audit.py`](../../training/open_shogi_training/data/external_audit.py) の `measure_exact_overlap` は、次の条件を満たすidentityだけを比較する。

1. position identityにdigestがある。
2. adapterが `exact: true` と明示している。
3. namespaceが同一である。

HCPEの32-byte HCPとPackedSfenValueの32-byte PackedSfenは、バイト長が同じでもencodingが異なる可能性があるため、自動的に同一namespaceへ寄せていない。CSA/KIF/HCPE3のmove prefixも、合法手順を再生していない段階ではexact board identityではない。形式fixtureでは同じPackedSfen recordを2つのsourceに渡したとき1件のexact overlapを検出するが、これはexternal datasetの重複結果ではなくreader regression testである。

## Game overlapとAobaZero duplication

GCTの公式記事には次のsource-specific artifactが列挙されている。

- `aobazero/hcpe`: AobaZero棋譜のHCPE変換
- `hcpe3/selfplay_gct???_taya36.hcpe3.xz`: たややん互角局面からGCT生成
- `gct/hcpe/selfplay-unique-900xx/...902xx`: たややん互角局面のGCT RL data
- `gct/hcpe/play-001`: Floodgate、電竜戦、local recordsの混在

したがって、AobaZero recordのGCT内重複は「あり得る」ではなく、記事のlineage記述によって少なくともsource family levelでは確認済みである。ただし、元のCSA object IDとGCT converted row hashのcrosswalkがないため、重複件数は `unknown` とする。GCTを将来取得する場合、最初にrecord/game ID、raw hash、converted HCP hashをmanifest化し、AobaZero exact100のhash集合と比較する必要がある。

## Train / validation / test contamination risk

外部データを追加する場合は position random splitを禁止し、次の順序で分割する。

1. raw artifact IDとgame/history identityを先に確定する。
2. source lineage（AobaZero, Taya, Floodgate, GCT, QPD, teacher model）でprotected groupsを作る。
3. 同じgame、同じhistory、同じcanonical board positionの行を同一splitへ割り当てる。
4. Taya 36SFEN、Tadao派生5247、dlshogi public evaluation testをimmutable holdout候補として先に除外する。
5. holdout除外後にだけtrain/validationの容量と分布を検討する。

既存OpenShogiAI Phase 3はgame-only addition-stable splitで、test rowをteacher trainingやselectionへ渡さない設計になっている。この不変条件を外部データにも継承する。既存のPhase 4–6/2026 evaluatorのlocal evidenceを外部データの許諾や重複測定の証拠には流用しない。

## Immutable holdouts

| holdout候補 | 理由 | 学習利用 |
|---|---|---|
| たややん 36SFEN | 後続GCT `taya36`と直接の系譜関係があり、外部評価・漏洩監査に使える | deny |
| Tadao派生5247 | 原版と公式報告の高い重複があり、別splitへ散らすと漏洩する | deny until explicit design decision |
| dlshogi public evaluation test | 公開testの意味を壊さない | deny |
| 後続Solが指定する公開test | test manifestをhash固定して学習から隔離 | deny |

## 未測定のまま残したもの

- GCT Driveの個別file一覧、archive展開後のrecord countとhash
- GCT converted AobaZero HCPEと既存100-object CSAのboard-canonical crosswalk
- Taya KIF/PSVとGCT/QPDのexact position/game overlap
- QPD archive内部のrecord schemaとsource lineage
- nodchip Hao/tanukiの全shard相互hash・position overlap
- exact public test artifactのbytes、hash、format

これらは、license decisionとstorage budgetが確定する前に取得・変換しない。今回のpending判定を「重複なし」と解釈してはならない。
