# OSAI R4-C1 — 本学習への引継ぎ

2026-09-18。人間向け正本はこの文書、実行設定は
[`configs/evaluator-main.json`](../configs/evaluator-main.json)。
対象は **`local/runs/r4-c1/attempt-01`**、候補名 **R4-C1**。
**ready_for_luna / requested_pause、step 9、execution attempt 3**。
長い生成・本学習・本比較は稼働していない。ユーザーが別に起動する
**Luna Max / max** が同じrunを正式resumeする。Astraの子として起動しない。

元seal・失敗・更新を保持し、終了済み工程は繰り返さない。
C1終了後はAstraレビュー、実ブラウザー、ユーザー対局を挟む。
C2は結果を受けた明示継続指示後だけ。C3は具体的な残存弱点と確認がある場合だけ。
公開表示名の予定はOSAI R4だが、現在は開発候補の準備段階。
デプロイ・公開URL/トンネル・重み配布・公開既定変更・main統合・final holdoutは禁止。

## 比較基準と旧第2セッション

旧 `defense-20260912-recovery-r3` はattempt17で固定評価を終了し、
`awaiting_astra_review`。best1536以外の新しい採用候補はなかった。
32局のストレス試験＋4局の初期局面実演は保存済み、35局checkmate、1局は
max-plies到達で未採点。3分16勝8敗、10分4勝3敗＋未採点1。
任意screenは未実施であり、旧採用条件の全面達成ではない。
旧runのgeneration/prepare/train/arenaを今回再実行していない。
旧結果とmanifestは `local/runs/defense-20260912/recovery-r3/` に保持。

C1の比較相手・初期重みは、ユーザーが明らかに強いと評価した**防御best1536**。
r3（step6144）や旧W256に勝つだけでは採用しない。
過去run名の末尾r3はモデルr3と別物。

| 資産 | SHA-256 |
| --- | --- |
| 比較基準 / 防御best1536 | `8c1c875038b74dc475c356d50c635c2e22dcab7aaa201d7d9aea7188e668b35a` |
| 当該run / run.json | `4b32461b9a7b51b5142236e105a60bf526ce8d8209495883cc1a1c51b1479fb1` |
| 元seal | `735f67ab848be6888f731c38e0349cea2429d26258dc504ee37bd25567e7a2e2` |
| C1 dataset manifest | `41eda6ac25760d041bc76f19e59338ac09bddc4335ae7c5953d9a4bba06cc3fe` |
| step9 resume checkpoint | `81316d8197caa6856451a4317f1e6b799bd14f770b9b4ca3cf0adb103004b96a` |
| step9 診断用runtime export | `4d30fba6245b1c85c4b0d7bc44d28970ab03bfa8021b90bd85403f17d31718d4` |
| pure Wasm | `2cd578b65a2a0afef1f095437260ccad366d8bba7464e2d02adca6a7b9ae556f` |

checkpointはoptimizer/RNG/samplerを含む約40MBの状態で、exportは約8.7MBの評価器。
変換関係は `local/r4-c1-preparation/prefix-nine.json`、実物は同ディレクトリの
`prefix-step000009.osaval03`。**bestはまだstep0＝防御基準**。
未検証の初期更新をbestや完成したC1と呼ばない。

## 診断・実装

ユーザー棋譜は未提供。保存済み旧Arenaの負けから4区分各1局面を診断し、
通常予算と追加予算のroot候補/PV/深さ/ノード/実時間/終了理由を保存した。
最初の追加予算が通常実時間を十分超えなかった2例は元記録を残して30秒/55秒で補足。
追加時間で教師D8の候補へ変わる例と、手が変わらない例の両方があった。
防御偏重だけを原因とは断定しない。教師も有限深度の参考値。
証拠は `local/r4-c1-preparation/diagnosis/`。これらはdevelopment専用。

既存の連続王手探索を、反復深さ・時間/ノード上限・実cancel・履歴付き反復判定に対応させ、
通常探索へ接続。通常時は最大5手、最大2,048ノードか全体ノードの1/32、
目標時間の1/20か20ms以内を使い、絶対期限の残りを超えない。
合法な全応手を調べ、打ち歩詰めを除外。履歴依存の証明に局面だけのTTは使わない。
未完了はunknown。証明済みPV/詰み距離を不完全反復で上書きしない。
詰めろ・必至・非王手の攻め全般を解決したという意味ではない。
1/9筋・先後の実Wasm例を確認。盤外indexバグは今回確認されず、一律の駒移動禁止は追加していない。
評価器の構造・特徴はOSAVAL03 W256を維持、全scalar層を継続学習、制御器はOFF。

## データ・学習

既存防御datasetの分割を維持し、公開AobaZeroの**歴史的train72局・validation18局だけ**を
同一raw hashで再取得した。holdout10局は取得していない。
[来歴・権利・ラベル監査](source-audits/aobazero.md)で上流のPublic Domainと
pre-move root側の探索勝率を確認。`600*log(v/(1-v))`の上流尺度で変換し、root値を
子や末端へ流用しない。Aperyとの完全な尺度校正を主張せず、新規比率を25%に制限。
旧train系列12rootsにD12/2Mノード/各1attempt、最大60ラベルの限定解析を行い、
実際53局面を直接ラベル化（欠測0）。子もその局面を独立に解析。
実戦の失敗局面そのものをtrainへ移していない。

| 分割 | 既存replay | 公開新規 | focus | 合計unique |
| --- | ---: | ---: | ---: | ---: |
| train | 483,451 | 8,016 | 53 | 491,520 |
| validation | 210,300 | 1,826 | 0 | 212,126 |
| development_test | 204,593 | 0 | 0 | 204,593 |

focus53のうち35は既存trainの再ラベル、18は新規。全体の新規uniqueは9,860。
分割衝突の新規対称キー1,280を除外し、旧evaluationをtrainに移さない。
系列・派生元・symmetryキー・source hashは各rows/manifestに残る。
戦法の網羅性は未検証。C1は不足していた序盤の重みを増やし、飛車・銀の連携、
角交換、端への圧力を既存系列と公開対局で補う。勝ち側だけを抽出しない。
分類不能な手順も維持し、runtimeに戦法認識表や対策表を持ち込まない。

更新時はreplay75% / new25%。replay内は一般25%・序盤30%・防御20%・攻撃終盤25%。
1epoch32,276 unique（replay24,207/new8,069）、epoch内重複なし、最大16epochs＝516,416露出。
最大2,048更新、validation256更新ごと、patience6、改善幅0.2%。
batch256/microbatch32/CPU4threads、LR5e-5→5e-6、warmup128、gradient norm5。
新ラウンドのoptimizer/scheduleは再設定し、同一runのresumeでは正確に復元。
選択値はvalidationのreplay75%＋new25%損失。4区分すべてのreplay損失≤初期の1.03を守る。
development_testはbest選択に使わず、最後の開発確認専用。保護された最終holdoutは未開封。
Mac M5 / 24GiBで実データの複数更新・保存・復帰が成立。9更新で2,304露出。
短い事前測定から本学習所要時間やピークメモリの保証まではしない。

## 実行・復旧・比較

cwdはOpenShogiAIのこのGitルート。実際の絶対cwd、コード・データ・重み・runtime・seedは
ignored `run.json` に固定されている。元実装commitは`3e83f59`。
再生成した通常Wasmの検査用hash記載だけを修正する運用改訂を正式登録した。
`approved-operation.json` → `operations/ceb37c8e7c871443363468d259121962a438db525a4859e5c9acc33d5ad293f9.json`。
改訂は元commit内のWasm hashへの置換だけを許し、学習コード/条件/データ/元sealは変えない。
通常resumeがこの改訂を照合する。手編集・再seal・step0への戻しは不要。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/r4-c1/attempt-01 </dev/null
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/r4-c1/attempt-01
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run pause local/runs/r4-c1/attempt-01 </dev/null
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run stop local/runs/r4-c1/attempt-01
```

最初のresumeだけを起動し、statusで同じrunを確認する。重複起動しない。
順序は完了済み取得/prepareを照合してskip → step9からtrain → export/runtime audit・
development scalar → 任意screenの残存証拠集計 → 固定Arena → `awaiting_astra_review`。
通常pressure/swapは診断値。実allocation失敗のみmicrobatch32→16→8→4→2→1、
最大3回のprocess再試行と資源待ち。ディスク残20GiB、破損・漏洩・非有限・違法手・保存失敗は保護停止。
暦日期限、旧focus上限、任意screen欠測を全体停止へ戻さない。STOPを自動無視しない。
取得90局・限定教師生成は完了済みで、追加取得・深い全再解析はない。

主比較は同じ修正runtimeで防御基準対C1 best。固定32ストレス局（3分24/10分8）、先後対。
各時計得点>0.5、family-cluster片側95%下限>0.5、各区分得点≥0.5、4区分scalar損失≤1.03。
初期局面からの時計/先後4実演は別集計で、反復した決定的対局を独立標本に水増ししない。
引分0.5、max-plies/無効局は未採点、時間/絶対期限違反は不利な事象として保存。
勝つまで再試行せず、有限の比較を保存して採否はAstraへ返す。
任意screenはretained_only。未実施・欠測はnull/unknownのまま、独立arenaや開発用読込を妨げない。
初段との対応、独立した外部対照での強さ、C1の改善はまだ未検証。

## 事前確認と開発画面

実データprepare→4更新保存→別processで4更新→正式stop→運用改訂の正式resumeで1更新→正式pause。
元seal、generate/prepare完了receipt、sampler offsetが保持され、step0再開なし。
step8とstep9 exportでnative/Wasm 10roots/300children、評価差0、有限出力・入力拒否・pure証拠PASS。
step8対防御基準の先後1組は各30秒・64plyで保存、両局max-plies未採点、違法手/絶対期限違反0。
これは経路確認であり、大規模評価完了の証拠ではない。
任意screen欠測の注入試験とR4最終集計、正確なmixed-source resume、二重出力防止もテスト済み。
証拠一式は `local/r4-c1-preparation/`、引継ぎcheckpointはrun内 `fit/resume.json`。

OSUIは開発用の防御基準/r3比較を保持。C1は未完了なので選択肢にまだ追加しない。
Playwright実ブラウザーで、モデルhash/Worker/Wasm一致、合法応手、1/9筋と先後の詰み、
3分/10分×標準/高品質、実cancel（約26msで確定合法手保持）、停止/再設定、診断downloadを確認。
通常の開始局面で3分は約0.40秒、10分は約1.42秒、絶対上限内。
実画面でも3分標準・10分高品質を操作し、盤面描画を確認した。
全棋譜・結果・PV・各AI手のclock/終了理由/model/runtime IDを既存の「診断ログを保存」でローカル保存できる。

ローカル試用: `http://127.0.0.1:5174/#/match`、選択名「防御学習候補」。
再起動が必要ならOpenShogiUIのGitルートで `npm run dev -- --host 127.0.0.1 --port 5174`。
今回のURLはloopbackのみ。公開選択は引き続き`model:null`。
Luna終了後に同じ依頼をAstraへ渡し、C1候補追加・採否レビュー・ユーザー対局へ進む。

最終検証: `make check` PASS（Python1,167件、Rust・format/lint・来歴・native/build・
決定的Wasm再生成を含む）、`make pure-build` PASS、`make frozen-smoke` PASS。
OSUIはローカル構成を明示した `npm run check` PASS（234件、型・lint・a11y・境界・
来歴・一構成build）。公開用`model:null`のままの通常buildは意図どおり拒否される。
ローカル構造検証の指定は
`OPENSHOGI_RELEASE_CONFIG=local/r4-c1-preparation/release-selection.local.json npm run check`。
runnerが古い所有checkpoint8件・305,063,214 bytesを削除し、latest step8/9とbestを保持。
削除台帳はrun内 `fit/cleanup.jsonl`。旧比較・rights/split・再現用W256は参照があり保持。
両repoともmainと使用中の未merge作業枝だけで、削除可能な旧枝はない。
