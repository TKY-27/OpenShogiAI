# 防御学習候補の後学習評価と開発用対局

正本はこの文書と `configs/evaluator-main.json` の既存実行設定。
対象runは `local/runs/defense-20260912/recovery-r3`。
生成・prepare・学習は完了済みで、再実行しない。
今回のコード変更は `4faa4f2`、実行用の最終ファイルhashは
同runの `approved-operation.json` を正本とする。
第2セッションは **A: best1536の固定評価のみ**。追加学習・モデル総当たりは行わない。
第3のAstraが結果と実戦をレビューする。初段到達・公開採用は未検証。

**現在はattempt16 / arena / ready_for_luna / arena_probe_complete。**
必須audit・development-test完了。固定scheduleの最初の先後1組は候補1勝1敗、
34手/48手で双方詰み決着、absolute deadline違反0。残りは34局。
保存済みcandidate-reviewはruntime PASS、scalar維持PASS、screen未実施、
Arena未完了、公開昇格なしを区別する。supervisor/stage/leaseは解放済み。
この1組はgeneral/central_spaceのgame1502・ply84からの開発局面試験で、
独立2標本/初期局面勝率/初段試験ではない。先後相関を保持する。

## モデルと完了済み学習

attempt15で累計3072更新、6 epochs、785,028露出へ到達し、正常にpatience終了。
選択はvalidationだけで行い、bestは1536更新。

| 資産 | SHA-256 |
| --- | --- |
| 防御候補 best1536 / OSAVAL03 | `8c1c875038b74dc475c356d50c635c2e22dcab7aaa201d7d9aea7188e668b35a` |
| 最終3072の評価時export identity | `07398aae415413b87e1bb1f5af9d2a3b40f157eab573ac4a7066ba3b8b63fbfe` |
| 最終3072の再開checkpoint | `710cc729ea57e2f12d0e00704e29f37d460e6e1551644952990ba1d5b4a5f860` |
| 比較用r3 best6144 / OSAVAL03 | `cd07f2a202f6e781afcb6a8af3c7a198c4203373e4505fe089f8c7f05eefd983` |
| dataset manifest | `67289ab62f78a1cb8f54af25367e27df729c7d3fc96cd6ad9bb573203a2dda42` |

最終checkpointには最終parameter/optimizer/samplerと、選択済みbestのexport bytesを別々に保存。
checkpoint内 `best_model_bytes` と `fit/best.osaval03`（8,679,836 bytes）は完全一致した。
run名のrecovery-r3は旧モデルr3の名称ではない。
元run/seal、生成receipt、旧失敗、学習identityは維持する。
運用の承認は `approved-operation.json` が指すhash付き既存形式のrevisionで固定する。
学習identity内の旧operationを維持し、後学習revisionに書き換えない。

生成済み3,243 trajectories / 181,519元行。有効trainは新規183,486＋r3 replay300,000
=483,486局面、実使用315,253局面。validation210,300、development_test204,593。
focusは62,220要求中61,957完了、欠測263。
独立scalar smooth L1回帰で、候補対ranking lossは使っていない。
欠測focusと依存候補は前処理で除外し、0点や引分で補完しない。
元split/family、左右対称重複、r3とのsplit漏洩、権利/来歴、mate除外を維持。
今回見たdevelopment_testを独立最終テストに再分類しない。final holdoutは開いていない。

## 任意screenと必須検査

attempt15の `audit-0.log` は必須actual-Wasmモデルaudit PASSの後、
`defense_evaluation.screen` → `_teacher_observation` → `_ledger_observation` の
`DeferredTaskError` で停止した。新OOM/EBADFではない。

失敗taskはgame914 / development_test / child ply89 / `6a6b+`、要求は **D16**。
2,000,762 observed nodes、最終MultiPV1/2/3はD15/D15/D14。
途中rank1のD16、rank2のD16 upperbound、seldepth25はD16完了を意味しない。
生成のD12とは別要求であり、深度を下げて採用しない。
元task、stdout tail、元state/logはローカル証拠に保持。

通常screenはtyped depth未達・timeout・確認できる有限予算消尽だけを局面ごとに
`missing`として保存し、次へ進む。未知例外、教師protocol/資源異常、
モデル不整合、違法手、保存失敗を握り潰さない。
今回の `optional-screen-v1` は追加取得を `retained_only` として明示的に停止。
完了/欠測/未実施、group/family/plyと分母を別々に報告する。
1件でも欠測/未実施があれば `screen_pass=null`。部分screenを全面合格にしない。
今回の追加取得は75ケース（development64＋既知回帰11）とも未実施。
元attempt15では最初のcase途中にchild欠測1件、case完了0件、他74case未着手。
現在の未実施数は今回の取得方針を表し、元の欠測taskを成功や無かった事にはしない。

必須auditはOSAVAL03形式・hash・入力拒否・合法手・有限値・incremental/fullとnative/Wasm
一致・pure証拠で、既存同runtimeのroots10/children300、差分0、PASSをhash再照合して再利用。
実行順は完了済み生成/prepare/trainを検証してskip → 必須audit再照合＋development scalar評価
→ 固定Arena → candidate-review。
exportは学習時に既に原子的保存済み。開発用読込はruntime PASSで可能であり、
screenや棋力合格を必要条件にしない。
reviewはruntime、screen、scalar、Arena、公開採用を別記する。任意screen未取得は
後段の停止理由にならず、旧採用基準の全面達成・公開許可を意味しない。

## 固定評価と第2セッション

cwdはこのOpenShogiAI Gitルート。別のユーザー起動Luna Max/maxが以下を実行する。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/defense-20260912/recovery-r3 </dev/null
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/defense-20260912/recovery-r3 </dev/null
```

正式resumeはlease・残留process・seal・承認code・receipt・SQLite・manifest・checkpoint・
model/runtime hashを照合する。Lunaは設定を変更/再承認/再sealしない。
今回の短い確認は同scheduleの最初の先後1組を
`resume ... --pause-after-arena-games 2`で実行し、残りは通常resumeへ渡す。
保存済みtask ID/plan/trace/receiptを再検証してskipし、既済局を重複計上しない。
2局probeに固定Arena完了receiptを作らない。

対象はbest1536一つ、比較r3、同じ封印済みnative core_probe、controller OFF、book OFF、
1thread、depth上限64、qdepth4、TT2MiB。対局時の教師・固定初手・手作業加点なし。
3分24局、10分8局の開発開始局面試験と、初期局面からの完全自力4局を分ける。
開始局面は4group/4familyの16根、各先後交換。
初期局面4局は3分/10分×先後の4条件だけ。同じ決定的対局の反復で局数を増やさない。
色交換局は関連標本、系列ごとのbootstrapを用い、独立32試料とは呼ばない。

評価は最大256ply/局、既存累積Arena予算86,400秒、process failure再試行は各局1回まで。
元sealed criterionのmaxplies drawという旧文言は保持するが、現行operationsと実装は
未決着をmax_plies_unscoredとして扱う。この運用を優先し、採点を捏造しない。
暦日の経過でrunを終了しない。完了済みgameの所要時間は再開時も予算に算入。
勝敗に応じた早期終了はしない。256ply未決着は未採点のまま固定task処理済みとして
レビューへ進める。引分/負けへ変換せず、全局有効という基準は不成立のまま。
真のruntime/clock/保存破損は影響工程を保護して停止し、弱めて再実行しない。

r3比較基準は全32 stress局＋4実演の完了、両時計得点率>50%、
family-cluster one-sided95%下限>50%、adverse attemptなし。
scalar general/attack lossはr3比<=1.03。
任意screenが未取得なら旧screen基準は未検証であり、`meets_frozen_criteria=false`。
結果にかかわらず有限処理後は `awaiting_astra_review`へ返し、Lunaの追加学習判断は不要。
A/B分岐はここでAに確定。負けてもLunaがBへ変更しない。

## 診断と独立性の限界

validation lossは1.145243→1.078256（5.85%減）、MAE976.24→930.24cp。
防御600cp超過大評価は24.008%→24.104%、序盤24.515%→24.505%でほぼ横ばい。
development_test204,593局面ではloss1.257077→1.177906（6.30%減）、
MAE996.46→948.69cp。general/attackのr3比<=1.03も達成。
防御600cp超過大評価33.36%→31.83%は改善したが、序盤23.93%→24.06%は横ばい。
欠測focusはdevelopment_test100/21,109で除外済み、これを全未知局面への保証としない。
回帰誤差改善やbestが中間だった事実だけから、棋力改善/過学習を断定しない。
初期局面と既知の構築棒銀3局面で各モデル10k-node診断を行い、
防御3局面の選択手は同じだった。rootは全合法手を探索候補へ入れ、
LMR/null/futilityの一律排除はない。静止探索の非王手時は捕獲・成りに限られるが、
今回の証拠でその変更の効果は確定できず、探索/評価器の推測修正は加えていない。
玉の安全、静かな受け、受けから反撃、終盤の強さは本Arenaと第3実戦で引き続き確認する。

補助対照は既存handcrafted_experimental評価（探索/合法手コードは共有、教師/重みは別）。
50ms/手、最大256ply、初期局面先後1組で候補0勝2敗、94/97plyで詰み。
合法手、pure/handcrafted証拠分離、保存と別起動での棋譜再生検証PASS。
soft超過最大0.919ms、hard2秒違反0。3分時計/独立エンジン/初段試験とは呼ばない。
既存binaryはSHA固定、現HEAD再build provenanceは未確認という制約を保持。
この2局は本Arenaに加算しない。再実行は保存済み2結果の検証だけを行う。

```sh
PYTHONPATH=training uv run --frozen python scripts/compare_evaluator_opponent.py --run local/runs/defense-20260912/recovery-r3 --output local/post-training-evidence/handcrafted-smoke
```

初段対応の対照は未確保。[K-Shogi開発元](https://www.studiok-i.net/kshogi.html)は
Core i5 3GHz/4core条件でLv20=初段+を目安にするが、Windows専用で当環境の校正はない。
Aperyは学習教師と同じなので独立初段対照にしない。
初段の暫定到達条件は根拠ある対照/段位既知人間に初期局面から3分主・10分補助、
20局以上・先後同数・得点率80%以上。引分、先後、関連性、標本不確かさを併記する。
対照/局数/判定は結果を見る前に固定し、候補選択・追加学習と分離する。
これは公式認定でも全初段への保証でもない。人間待ちで他工程を止めない。

## OSUI開発候補

確認済みURL: http://127.0.0.1:5175/#/core-prototype
「防御学習候補」を選ぶ。再読込時の既定は従来のr3。
開発descriptorは `local/core-prototype/defense.json`、旧r3 descriptorも維持。
本番設定はmodel:nullのままで、選択済み1モデルのみ出力する仕組みを維持する。

再起動はOpenShogiUIリポジトリで:

```sh
npm run dev -- --host 127.0.0.1 --port 5175 --strictPort
```

Browser専用skillは利用可能一覧とskill検索で見つからなかったため、検出した
Playwright SKILL.mdを読み、実Chromeで3分/10分×標準/高品質×先後の8ケース、16応手を確認。
モデル取得8,679,836 bytes/SHA、実Worker/Wasm/pure推論が一致、全応手合法。
探索343–2,966ms、hard以内。停止53ms、未確定盤面保持、時計停止/再開・再対局PASS。
console errors/warnings 0。少数序盤テストであり完局/棋力試験には数えない。
OSUIの `output/playwright/summary.json`、`browser-observations.json`、画面2枚をローカル保存。
OSUI commit `8237914`、既存draft PR #1更新済み。

## 資源・検証・公開境界

通常pressure/swap/RSSだけでは計算を止めない。空きdisk60GiBは保存余裕として維持。
実確保失敗だけ既存逐次microbatch32→16→8→4→2→1、完全checkpointへ戻す。
今回Aでは学習を再開しない。OOM process再試行は同taskで最大3回、予算をresetしない。
資源待ちはsupervisorのresource_waitで保持し、最低60秒後の回復を確認して自動復帰。
予算枯渇は保留を維持し無限再試行しない。ユーザーSTOPは自動解除しない。
OS保護、他アプリ、swap設定を変更しない。保存不能/破損/NaN/未知SIGKILLはOOM扱いしない。

OSAI関連191 tests、ruff、boundary/license/provenance/docs検査PASS。
process leaseテストに一度タイミング失敗があり、対象80 testsと関連191 testsの再実行はPASS。
`make check`はXcodeライセンス未同意で起動BLOCKED。ライセンス/権限を変更していない。
今回は封印済みruntimeのhash/auditを再利用して実行し、現HEAD全面build PASSとは報告しない。
OSUI `npm run check`（234 tests含む）、integration:ai、r3単一モデル本番build PASS。
main merge、deploy、重み公開、既定昇格、枝/モデル資産削除は行わない。
