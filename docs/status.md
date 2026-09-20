# R4-C3 — Astra preparation

唯一の人間向け入口はこの文書、機械向け設定は `configs/evaluator-main.json`。
2026-09-21の明示承認はC3の診断・修正・生成・学習準備・ローカル開発統合と作業枝pushまで。
C4、公開、外部preview、重み配布、公開既定昇格、main統合、有料資源は別GO。
Lunaはユーザーが別セッションで起動する。棋力成功やユーザーへの安定勝利は未確認。

## 初期モデルとC2診断

初期・主要比較は防御best1536
`8c1c875038b74dc475c356d50c635c2e22dcab7aaa201d7d9aea7188e668b35a`。
C1 best512 `6b49c3361194011c0c8ac114491dcdb6c67a3b4f4cc6c17262a867572aefb3a6`
は対防御3分12勝12敗、10分4勝4敗で未採用。番号を棋力順位にしない。

C2 `local/runs/r4-c2/attempt-01` はattempt3、step8192、延べ2,096,579例、
validation_patienceで正常完了。pool2,067,381、seen1,850,644、最大露出3。
Haoのseen/露出は1,575,287/1,616,553、独自Apery157,440/362,109、
r3 replay117,917/117,917。best0は防御と同一8,679,836 bytes。
任意screen18/19、Arena18勝14敗、export・開発統合完了という履歴は保持する。
原封印 `79dff50b1d7a1c7466d7d0984dfbed5f85af92460c9d5235091ac8285b378dcd` と
運用改訂 `096e86ea…` は変更しない。C2完了時の文書もignored領域に保存した。

保存checkpointから全6tensorのoptimizer step8192と有限・非ゼロ勾配を確認。
更新後export `f2c413d2932816cf20a139ed8e3e0d61c5a474cbecafd43676bae1eb819eadea` は
実tensorも変化し、量子化で更新が消えていない。全層の変更要素数は
2,152,514 / 256 / 12,288 / 16 / 64 / 4。初期モデルの誤exportではない。
混合validation損失は1.07853→0.87676、Haoは1.06775→0.78968。
一方、replayのgeneralは0.85612→0.90151、opening0.96269→0.98188、
defense1.09411→1.10508、attack_end1.55672→1.58822。
すべての更新後検証でgeneralの1.03倍guardに抵触し、best0が残った。
固定標本のreplay trainも悪化し、単純な「未知データだけの過学習」とは断定しない。
教師間の尺度・目的・混合比による干渉が主仮説。容量限界は未証明。

C2 Arenaは両側の重み・probe・profile・設定が同一、controller OFF、乱数による
着手選択なし。pair8の同一局面14手目で、実時計/探索時間の差からdepth5の4f4g+と
depth4の2c2dへ分岐した。18勝14敗を学習改善の証拠にしない。
OSUIの独立C2項目を除き、古いC2選択/URLは防御への別名として表示する。
C2の学習済みcheckpointは診断資産として保持。
根拠は `local/r4-c3-preparation/c2-diagnosis.json`、
`c2-identical-arena-divergence.json`、`search-diagnosis.json`。

## C3の一つの学習仮説

W256/OSAVAL03の全層を防御から継続し、replay75%・Hao25%にする。
replay内general/opening/defense/attack_endは20/40/20/20%。
従来のcp/600 scalar損失に、同一教師・同一rootの兄弟候補の順位損失を0.25加える。
差50cp以内は複数許容、要求marginは600cpで上限。子局面は子の手番視点。
両方が実batchへ投入された対だけを比較し、欠測対を0点にしない。
既存の防御・駒得・終盤を保持しながら、静かな準備と応手後の価値差を学ぶ仮説である。
これが実着手を改善するかは本比較待ちで、損失改善だけを棋力向上と呼ばない。

[出典・形式・利用条件](source-audits/hao.md)。今回の外部新規ダウンロードは0。
C2の4つのhash固定Hao shard、独自Apery、防御/r3 replayを再利用する。
Haoは元局面に対するPV末端のroot視点評価、Value→cpは100/90。
正確な生成binary/weightと元対局の完全な独立性は未保証。D9全件をD12再解析しない。
GCTは独立hcpe3を識別済みだが派生利用条件未確認で採用0。未取得値を実績にしない。

相居飛車、棒銀、対四間/三間/中飛車、嬉野流、端、角交換の合法な自作14系列を
先後反転し各8root、trainの一部は現モデル候補の子と強い応手もD12/2M nodesで解析。
ユーザー棋譜ではない。272解析、raw1,087行から重複・対称・split衝突を除き714行追加。
既存splitと保護済み検証全キーを優先し、45衝突キーを除外。final holdout未開封。

| 出典 | train | validation | development |
|---|---:|---:|---:|
| 独自Apery生成 | 183,486 | 8,890 | 8,586 |
| r3 replay | 300,000 | 上段に含む | 上段に含む |
| Hao D9再利用 | 1,583,895 | 8,192 | 8,192 |
| C3自作counterfactual | 431 | 94 | 189 |
| 合計 | 2,067,812 | 17,176 | 16,967 |

兄弟候補対はtrain115,893、validation250、development295。
manifestのsupplied_records/new_unique_positions等の継承値は**C2取得時の統計**。
C3の追加はadded_counterfactual_rows、総poolはunique_positionsを読む。
元系列不明なHaoのfamilyは推定。ユーザー診断と自作developmentを最終未知試験とは呼ばない。

AdamW LR1e-5→1e-6、warmup128、weight_decay0.01、有効batch256、microbatch32。
最大16巡/16,384更新、各巡262,144例以下、検証512更新間隔、patience6。
露出上限replay8/Hao3、系列2,048、batch内4。少数shard先頭の循環で埋めない。
patienceはreplay60%・Hao30%coverage以後。optimizer/RNG/order/countsを再開で保持。
全runと各候補の出典・群・系列・露出分布はcheckpoint/training.jsonへ保存する。
祖先モデルからの生涯露出は不明。
16巡の実プールdry samplerは2,776,376露出、候補対237,219組。
少ない層が先に上限へ達するため、75/25は各巡の要求quotaであり固定実比率ではない。
実累積はreplay1,727,811/Hao1,048,565、新自作431例はseen431/露出3,253。
このdry値は本学習実績ではない。本runは実countsを記録する。

## 選抜・固定比較・開発用登録

incumbentは防御。従来のguard付きbestとは別に、更新後のraw最小validation目的値を
trained_candidateとして保存する。0.2%はpatience専用で、保存の足切りにしない。
固定形式/seedの実tensor変更とexport差を必要とし、metadataだけの差を候補にしない。
選抜候補は一つ。本比較までに勝つ候補を探し直すことはしない。

共通の既存book-free探索で対防御32局（3分24・10分8）、先後交換。
自力startpos4局は別枠。静かな序中盤/棒銀/対振り/嬉野流/攻め終盤のdevelopmentを含み、
自作4familyは結果を見る前に指定。draw=.5、max-plies/無効は未採点、期限違反は不利結果。
同一決定的対局を独立試行として水増ししない。family単位bootstrapで扱う。
推奨基準は両時計score>.5、片側95%下限>.5、各層>=.5、4群scalar悪化<=3%。
これはユーザー/初段への安定勝利の証明ではない。ユーザー対局後に採否判断する。
任意screen欠測・best0・低棋力・採用保留は独立arena/export/UIの停止条件にしない。

750/3000msの短い診断で予算による候補変化を確認。未完了反復の誤採用や詰み証明を
捨てる不具合は今回特定できず、Rust探索・詰み実装は変更しない。
非終端手作業加点、初手固定、戦法別応手、教師呼出し、訓練ノイズは対局へ入れない。
controller OFFのまま完全な選択重みを使う。

Lunaは学習後にaudit→固定arena→export/descriptor更新→loopback起動→実ブラウザーを
同じrunnerで実行する。編集範囲はignored C3 descriptorとrun証拠だけ。
失敗時は旧descriptorを復元。ソース設計をLunaへ残さない。
R4-C3（比較候補・未採用）、防御、C1/r3/W256を「最新←→開発初期」で表示。
解析も同じ一覧を維持し、大規模解析UXは学習後に扱う。
legacy `awaiting_astra_review` はdevelopment/result.json PASSならユーザー対局待ち。
対局するためだけのAstra再起動は不要。

## 実行と引継ぎ

実cwdはOpenShogiAI Git root。封印run内execution_cwdにも絶対パスを記録する。
設定のoperationsが正式コマンドと保持/復旧範囲の正本。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/r4-c3/attempt-01 </dev/null
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/r4-c3/attempt-01
```

Astraは実データ初期更新→保存→正常pause→別プロセスresume→追加更新→export、
さらに少数短時間対局と実ブラウザーまで確認してからready_for_lunaで止める。
その初期更新は本runで継承する。rehearsalは本Arena/採用実績へ加算しない。
途中の全体best0・任意screen欠測・更新候補不採用はfixtureでも通す。
現時点の実試験結果とrun identityは以下の完了記録へ追記する。

暦日wall limitなし。pressure/swapは診断のみ。実OOMはmicrobatch32→16→8→4→2→1、
有限process retry/resource wait。同run resume、stop/pauseを尊重する。
ディスク保存不能、広範な破損/漏洩、非有限値、必須runtime不整合は保護停止。
旧全root成功必須/固定focus絶対上限/C3未承認/統合にAstra必須の指示は本契約へ更新済み。
