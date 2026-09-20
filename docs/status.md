# R4-C2 — Astra準備・最終実行確認中

人間向け正本はこの文書、機械向け正本は `configs/evaluator-main.json`。
今回の明示承認（2026-09-20）はC2一ラウンドとローカル開発登録まで。
C2の初期更新・別プロセスresume・後学習経路の実証を終えてから `ready_for_luna` とする。
Lunaは別セッションでユーザーが起動する。互いを起動しない。
**棋力向上・ユーザーへの安定勝利・対初段は未検証。** C1採用保留を維持する。

## 継承した成果と診断

初期重み・主比較は防御best1536
`8c1c875038b74dc475c356d50c635c2e22dcab7aaa201d7d9aea7188e668b35a`。
C1 best512 `6b49c3361194011c0c8ac114491dcdb6c67a3b4f4cc6c17262a867572aefb3a6`
は3分12勝12敗、10分4勝4敗。ユーザー評価も同程度。番号による昇格はしない。

| 実績 | best時点の見た局面 / 延べ例 | 全学習の見た局面 / 延べ例 |
|---|---:|---:|
| 防御best1536 | 230,762 / 392,514 | 315,253 / 785,028 |
| C1 best512 | 90,997 / 130,128 | 215,635 / 516,416 |

防御の学習プールは独自Apery教師付き183,486局面と、以前の独自Apery生成r3 replay
300,000局面。現在の防御プールを「すべてAobaZero」とは呼ばない。
C1の学習プールにはこれに由来する483,451局面、AobaZero 8,016局面、
Apery D12 focus 53局面（既存35再ラベル＋18新規）が入った。
C1全体の実optimizer投入はreplay387,312例、新追加側129,104例。
新追加側8,069局面は16回反復。bestまでと全体を混同しない。
祖先W256等の全生涯露出は今回確定しておらず、新取得局面を祖先未見と断言しない。
根拠は既存runのmanifest/training.jsonと `local/r4-c1-review/training-review.json`。

C1では0.2%改善条件が保存とpatienceを兼ね、step1024の損失最小1.252780を
保存対象にできなかった（閾値1.252689）。C2は**観測上の最小の保存**と
**0.2%の意味ある改善によるpatience更新**を分離した。損失最小は棋力最良とは限らない。

既知の攻め継続、棒銀系受け、受けから攻め、端の詰みの4合法局面を750/3000msで
比較した。端の5手詰は両方232 nodesで証明。時間を増やすと評価の変わる局面もあり、
全弱点を評価器だけのせいにはしない。通常探索は静かな合法手を含む既存alpha-beta/PVS、
連続王手のAND/OR証明、合法合駒・打歩詰・反復履歴・絶対期限/stopを再利用する。
この診断で新たな探索不具合は特定できず、今回エンジン/runtimeは変更しない。
共通探索の変更効果を学習効果と取り違える対照差は発生しない。
`local/r4-c2-preparation/search-diagnosis.json` は既知例の診断であり未知局面の合格証ではない。

## データと反復制御

[出典判断・形式・条件](source-audits/hao.md)。nodchip Hao depth9の4つの離れたshardを
SHA-256照合して取得済み。1,231,381,720 bytes、提供30,784,543 records。
16手間隔の候補1,801,041例から、出典横断の実局面・合法対称キー、保護済みsplitと照合。
重複/除外120,337例。検証・開発側はhash固定抽出で各8,192件に限定する。

| 出典 | train局面 | validation局面 | development局面 |
|---|---:|---:|---:|
| 新取得Hao D9 | 1,583,895 | 8,192 | 8,192 |
| 防御/r3 replay | 483,486 | 8,890 | 8,586 |
| 合計 | 2,067,381 | 17,082 | 16,778 |

Haoユニーク採用総数1,600,279は提供件数やoptimizer延べ例数ではない。
元系列IDのないPSVは手数resetと初期実局面から系列/派生familyを推定し、同じfamilyを
split間へ割り振らない。trainのfamily93,947。ただし完全な系列独立性は保証しない。
駒数・戦型だけの機械的統合はしない。元の検証局面キーは抽出外も全てtrainから保護。
既存の未開封final holdoutは開かない。旧独立生成由来の検証も併用する。

dlshogi/GCTの独立 `hcpe3/selfplay_gct-???.hcpe3.xz` を現物一覧まで確認したが、
派生重み等の利用条件未確認のため採用0。AobaZero変換版を別出典として水増ししない。
Tanuki/Suisho巨大multipart、Knowledge_distilled等は今回は未採用。
取得済み・採用済み・実学習済みは別に報告する。

主案はW256 scalar全層の防御モデル継続、AdamW LR2e-5→2e-6、warmup256、
weight decay0.01、有効batch256、microbatch32、controller OFF。非終端手作業評価なし。
新側75%・replay25%、replay内general/opening/defense/attack_endは25/30/20/25%。
各層で未使用を優先し、1例の上限は新側2回・replay3回、1系列512例、同batch内4例。
枯渇層は反復で埋めず、残る有効例へ進み、全枯渇なら正常完結する。

1巡の計画上限262,144例、最大12巡/12,288更新。検証512更新間隔、patience4。
patience消費はreplayの50%・Haoの60%を見てから。破損/非有限はそれ以前も保護する。
4群のreplay損失が初期値の1.03倍以内のcheckpointだけ保存候補にする。
全件走査の追加基盤は使わず、実samplerのcounts/系列/出典分布、最近の新規増分、
実プールcoverageとbest時点の集計をcheckpoint/training.jsonへ保存する。
名前変更や再開でcountsをリセットしない。

実samplerを12巡だけdry実行し、延べ2,911,283例、seen1,921,165局面、
Haoは全1,583,895局面を見て最大2回、replay最大3回を確認した。
これはoptimizer実行ではない。学習実績には加算しない。
`local/r4-c2-preparation/sampler-simulation.json` に実分布を保存。

## 固定評価と停止

保存候補はvalidation最良1つだけ。防御best1536との先後交換24局×3分＋8局×10分を
同じruntime/clock/loadで比較。4局の初期局面デモは別で、独立多数局へ水増ししない。
引分0.5、maxplies/invalidは未採点、時間/期限違反は不利益として既存の固定集計を使う。
両時計score>0.5、family-cluster片側95%下限>0.5、各層score>=0.5、4群のscalar退行<=3%。
境界を満たしても自動採用・対初段認定・公開はしない。僅差は不確実。

開発screenは未知別系列各群2局面＋既知11回帰、各要求はD16/2Mの有限教師観測。
攻め継続/重大ミス/棒銀・端/終盤を分け、詰み誤認を合格扱いしない。
少数screenは記述的で、旧大規模screenの必要件数を満たすとは主張しない。
未完了深度・局所欠測・not_runは依存ラベルだけ欠測扱い。独立Arena/export/登録を止めない。
広範教師障害・不正合法手・破損/漏洩・保存不能は技術異常として証拠を保持する。
早期停止は正常完結し、最良を比較・登録する。弱ければ防御既定を維持する。

通常pressure/swapは診断のみ。実割当失敗は既存microbatch32→16→8→4→2→1、
実効batch不変の勾配蓄積、最大3プロセス再試行と資源待ちへ進む。
暦日上限なし。ユーザーpause、資源待ち、正常early stop、技術異常を区別する。
他アプリ終了・OS保護解除・状態ファイル手編集は禁止。

## 唯一のLuna入口と開発登録

cwdはOpenShogiAIのGitルート。初期prefixの正式確認後、このresumeだけを実行する。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/r4-c2/attempt-01 </dev/null
```

`status` / `pause` は同じmoduleとrunパス。復帰も同じresume。step0へ戻さない。
取得/prepareが必要な環境では、同設定の固定4shardだけを
`PYTHONPATH=training uv run --frozen python -m open_shogi_training.r4_sources configs/evaluator-main.json`
で取得・検証できる。今回は既に準備済みなので再生成不要。
`seal`はAstra準備で一度だけ実施済みになるまで、この入口を実行しない。

正式経路はgenerate検証→prepare snapshot→train→audit/screen→固定arena→integrate。
`integrate`がcheckpoint-bound bestを監査し、実Wasm/profile/hashを照合する。
Lunaの更新範囲は **`local/core-prototype/r4c2.json` とrun内のignored結果のみ**。
OSUIソース/設定の任意変更は禁止。reviewed OSUIソースhashはsealへ固定する。
登録失敗は旧descriptorへ戻し、失敗と新重みを残す。無言fallbackしない。

OSUIはloopback **http://127.0.0.1:5175/#/match** で起動・再利用する。
既存5174のユーザー用プロセスは止めない。新登録のdefaultは防御のまま。
「最新←→開発初期」の順にR4-C2/C1/防御/r3/W256を選択し、照合完了後に対局。
一対局一構成、途中切替禁止、旧応答破棄、モデル別TT/Worker分離を維持する。
高品質でも既存の3分/10分時計とhard stopを守る。完全な選抜重みを使用する。

実ブラウザー検証は`OpenShogiUI/scripts/verify-development-candidate.mjs` を自動実行し、
URL/title/console/実寸盤面、Worker identity、先後応手、stop/再対局、USIと診断保存、
小画面を確認する。棋譜・局面・両時計・消費時間・探索終了理由・profile/hashを自動保存。
診断欄展開時に盤面が潰れる高さ依存を今回修正した。
Browser専用skillは利用不可のため、frontend-testing-debugging skillの代替手順として
既存Playwrightと実Chromeを使用する。

`development/result.json` とbrowser結果がPASSなら、そのままユーザー対局待ちで終了。
機械の既存終端名 `awaiting_astra_review` は互換維持で、統合のためだけのAstra再起動は不要。
`rehearsal`コマンドは保存済み初期更新をexportし、30秒先後2局・任意screen未実施・
同じ登録/ブラウザー経路を小さく通す。最後に暫定descriptorを戻す。
本Arena完了/棋力採用のreceiptは作らない。

## 保持範囲と後工程

代表W256/r3/防御/C1/C2と対応runtime/profile/hash/来歴、sealed split、bestと有限resume
checkpoint、復旧証拠を保持。今回不明ファイルや旧枝の削除はしない。
開始時OSAI未追跡 `.playwright-mcp/` は保護。OSUIは開始時clean。
開始HEADはOSAI95e33e1/OSUI0d04f27、両方codex/core-prototype。

本番一モデル制約を撤回。将来の公開は権利確認した代表allowlistで、重み/optimizer/
元データを通常Gitへ入れない。OSUIの公開allowlistは未承認の空配列でfail closed。
ローカル複数モデルbuildは構造試験で、権利fixtureやローカル成果を配布承認にしない。
公開名OSAI R4はGO後の採用構成のみ。現段階はR4-C2比較候補。

解析モードでの同一覧・identity付き実解析、OSUI全体の時計/読込/入力/終局/保存/操作性
改善は学習後のAstra工程。このプロンプトの再送時は完了runを確認し再学習せず対応する。
C3、main merge、外部preview、公開、モデル配布、有料資源は別の明示GOまで停止。
