# OSAI R4-C1 — Astraレビュー済み / ユーザー対局待ち

2026-09-20。人間向け正本はこの文書、終了済み実行設定は
[`configs/evaluator-main.json`](../configs/evaluator-main.json)。対象は
**`local/runs/r4-c1/attempt-01`**。技術上の採用判断は**保留**。
直前最良の**防御best1536を開発既定・比較基準として保持**し、best512を
**「R4-C1（比較候補・未採用）」**として開発OSUIで対局可能にした。

Lunaの正式resumeは正常完了（execution attempt 4、step2032、16 epochs）。
runの既存schemaは`awaiting_astra_review / complete`を終端とするため、
`state.json`、seal、checkpoint、candidate-review、raw Arenaは書き換えていない。
この文書がAstraレビュー済み・ユーザー確認待ちの引継ぎである。
採用条件未達はrun異常終了ではなく、任意screen欠測で`needs_astra`へ戻さない。
C2/C3、新run、追加更新・再学習・重いArena、Luna起動、promotion、公開既定変更、
重み配布、デプロイ、main統合、final holdout開封は今回行っていない。
C1を次ラウンド初期重みに自動採用しない。ユーザー対局と明示継続指示を待つ。

## 正本と構成

初期重み・実Arena比較相手は同じ防御best1536。run名末尾recovery-r3と
モデルr3（step6144）を混同しない。対象exportはlatest2032ではなくbest512。

| 資産 | identity / SHA-256 |
| --- | --- |
| 開発既定 `defense-20260912-recovery-r3-best-step1536` | `8c1c875038b74dc475c356d50c635c2e22dcab7aaa201d7d9aea7188e668b35a` |
| 比較候補 `r4-c1-attempt-01-best-step512` | `6b49c3361194011c0c8ac114491dcdb6c67a3b4f4cc6c17262a867572aefb3a6` |
| 共通pure Wasm | `2cd578b65a2a0afef1f095437260ccad366d8bba7464e2d02adca6a7b9ae556f` |
| 共通JS | `907da1421263a2cbc621297095d13ccd3f707942b1e6f09bd7a2b2512efeee2e` |
| pure_learned-v3 profile | `d2eec27887926ccc8a076552815cd54e34b85d6d23e65732ddba4989bf59c1e7` |
| 実Arena native probe | `bd1794fb91bbe3c9f01c0eb06e5745c5907fd263a38dd9136ce1e3fa9a4aee0b` |
| run.json | `4b32461b9a7b51b5142236e105a60bf526ce8d8209495883cc1a1c51b1479fb1` |
| 元seal | `735f67ab848be6888f731c38e0349cea2429d26258dc504ee37bd25567e7a2e2` |
| dataset manifest | `41eda6ac25760d041bc76f19e59338ac09bddc4335ae7c5953d9a4bba06cc3fe` |
| 最終resume checkpoint2032 | `556922529d5e62096b47515c558b8cb5c52c487267606b92e3b7cd0e7a7ec1b3` |

両モデル実体は各8,679,836 bytes、OSAVAL03 W256、scalar評価、制御器OFF、定跡OFF。
checkpoint2032内の`best_step=512`と`best_model_bytes`を照合し、exportとの完全一致を確認。
feature/shared層とscalar headを更新し、未使用WDL headは不変。制御器は学習していない。
既存model-auditはPASS（10 roots、300 children、306 unique、native/Wasm評価差0、
入力拒否・合法手・純学習経路）。今回audit・学習・全Arenaを再実行していない。
学習の最大RSS約2.07GiB、allocation failure/retry/resource-waitなし。
終了process/stage/leaseを確認し、起動中の対局・他worktreeを変更していない。

## Arenaの採否

主比較は同一runtime・seed20260918・定跡なしの16開始局面×先後対。
開始局面は既存development_test由来で、各対は同一SFENとrootからの履歴を使う。
持ち時間は3分/10分切れ負け、時計と絶対期限を各手で記録する。

| 集計 | C1勝 | C1敗 | 引分 | 未採点 | 得点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 主比較3分 | 12 | 12 | 0 | 0 | 12/24 = 0.5 |
| 主比較10分 | 4 | 4 | 0 | 0 | 4/8 = 0.5 |
| 初期局面実演3分（別枠） | 2 | 0 | 0 | 0 | 採否の主得点に含めない |
| 初期局面実演10分（別枠） | 1 | 0 | 0 | 1 | 採否の主得点に含めない |

得点は`(勝ち + 0.5×引分) / 採点局数`。全36タスク実施、35局採点・未対局0。
一般4/8、序盤4/8、防御3/8、攻撃終盤5/8。
事前条件の各時計得点>0.5、family-cluster片側95%下限>0.5、各区分≥0.5を満たさない。
4系列（局数8/12/8/4）の平均[0.5,0.5,0.375,0.75]を等重みで10,000回再標本化した
5%点は0.4375。等系列平均0.53125はゲーム平均0.5と別量であり、4系列しかない。
先後・同系列の依存を無視した35独立二値試行の推定は行わない。
同率は同一棋力や弱体化の証明ではない。集計実バグは見つからず、規則・raw結果を保持。

未採点は初期局面実演`demonstration-01-black`（C1先手10分）の256ply打切り。
全256手のrules結果は`game_end=None`、初期込み257局面の盤面/手番/持駒キーは全て一意。
千日手見逃しの証拠はなく、両玉入玉・合法手あり、残時計6.364秒/10.304秒。
Arenaは自動入玉・持将棋採点を実装していない。自己評価+3957は勝ちの証明ではない。
未採点を引分・勝ちに変えず、追加対局で埋めない。`arena_complete=false`と
`adverse_attempts=1`はこの打切りを含む保守的採否フラグで、実行異常とは区別する。

## 実学習量とbest選抜

available train491,520、validation212,126、development_test204,593。
train内訳は旧replay483,451＋公開8,016＋focus53。公開元はtrain72局、validation18局のみ。
focusは12 roots＋直接解析した41 childrenで、35例は既存再ラベル、18例が新規。
本当に新しいtrain局面は8,034、validation込み新規uniqueは9,860。
旧防御生成train183,486にはroot43,106・候補130,142・deviation3,551・recovery6,687があり、
旧r3 replay300,000例も継承。元trajectory台帳件数は独立例数や実際のsampling済み対局数ではない。
来歴と分割は既存manifestを保持（新規分割衝突1,280除外、旧evaluation流入0、holdout未開封）。
元データの全走査・再生成・全再hashは不要だった。[権利・尺度監査](source-audits/aobazero.md)を維持。

各epoch32,276＝replay24,207＋new8,069、epoch内重複なし、16epochs、127updates/epoch
（batch256、最後20例）で**516,416 exposures**。51万の新規局面ではない。
全runで実使用unique215,635＝replay207,566＋new8,069。replayは387,312露出、
newは各16回で129,104露出。未使用replay275,885、replayの露出範囲0〜10回。
**best512自身までは130,128露出・90,997 unique**であり、全run量と区別する。
newの群内訳は一般1,856・序盤725・防御13・攻撃終盤5,475。
公開attack_endは主に手数による粗い分類で、戦法や攻撃品質の網羅を意味しない。

選抜値は`0.75×(一般.25+序盤.30+防御.20+攻撃終盤.25のreplay validation損失)
+ 0.25×new validation損失`。全replay群≤初期1.03、256更新ごと、改善幅0.2%。
step0→256→512の損失は1.284028→1.260197→1.255200。
最小raw損失はstep1024の1.252780だが、更新閾値1.252689を満たさずbest512を保持。
最終2032は1.252981。sampleしたbatch平均train lossは0.873→0.600へ低下したが、
固定train全体の再評価ではない。後半の無価値や過学習をbest512だけから断定しない。
事後的なbest選び直しは行わない。development_testは選抜に使っていない。

| scalar指標（best512対初期） | 一般 | 序盤 | 防御 | 攻撃終盤 |
| --- | ---: | ---: | ---: | ---: |
| development_test損失の変化 | +1.733% | -0.220% | +0.454% | +2.018% |

全群3%ガード内。new validation損失は1.831806→1.667165、replay加重損失は
1.101436→1.117878。新sourceへの適合と既存domainの小幅悪化は確認できた。
教師尺度/domain差、scalar選抜と実戦のずれ、新規防御13例やsampling不足との因果は未検証。
任意分野screen、人間/初段検証はNOT RUNであり、合格扱いも新たな必須gate化もしない。

## 攻防の既知例と実ブラウザー

既存`evaluation-11`はC1両色勝ち。同一履歴の後手でC1は`P*5e`、基準は`G*8h`を選び、
C1は駒交換・飛車打ちから5手詰めへ進んだ。先手では受けから1筋攻めへ移り3手詰め。
一方`evaluation-10`はC1両色負けで、後手は自己評価+5390から攻め切れず逆転された。
防御区分の負けを防御忘却だけでは説明できない。棒銀系pair14は1勝1敗で改善未確認。

実Chromeでも同じ既知4局面を両モデルで各1探索した。
攻め継続例の分岐を再現し、棒銀受けは両者`2b2c`、攻守転換は両者`1d1c+`、
9筋の寄せは両者同じ5手詰めを証明。特定手の相違・一致だけで優劣を認定しない。
これらは診断済みdevelopment例で未知評価ではない。新たな教師呼出しは行っていない。

Browser専用plugin/skillは利用不可のため、`frontend-testing-debugging`手順に従いPlaywrightを使用。
`http://127.0.0.1:5174/#/match`で81マスの実描画、console error/warning0、
1280×900/390×844表示、モデル選択・取得bytes/SHA・Worker/Wasm identity・推論proofを確認。
初期局面からC1後手の通常3分対局を46手・詰みまで操作し、全手合法、時計違反0。
これは動作確認1局でありArenaへ加算しない。1/9筋と先後の短い詰み3例、短時計の時間切れ、
読込中開始禁止・503失敗明示/無言fallbackなし・再試行・対局中/停止中切替禁止・
終了/再対局後切替・古い非同期応答拒否・Worker破棄を確認。切替後は11生成/10終了で1個のみ。
実cancelは約27.6msで合法候補を保持。stop前の消費も再開後の確定手へ合算する。

新旧×3/10分×標準/高品質×先後、計16の短い初応手確認は全て合法・絶対上限内。
C1先手は3分0.389秒、10分1.374〜1.419秒。
`7g7f`後のC1後手は3分0.828〜0.831秒、10分7.377〜7.928秒
（基準2.452〜2.991秒、絶対上限17.95秒）。初手停止・無制限長考はないが、
10分後手応答の遅さは残る。攻防4局面には目標時間を超えて絶対期限まで探索した例もある。
時計実装・profile・engine/pure Wasmを変更して症状を隠していない。

## ローカル試用と保存

開発既定は「防御学習候補」。C1を試すときは「R4-C1（比較候補・未採用）」を選び、
照合済み表示後に対局開始。モデルと標準/高品質は別設定。
「モデルと読込の詳細 (開発用)」→「診断ログを保存」で完全hash/runtime/profile、
棋譜・結果・持ち時間・品質・先後、全確定手の消費時間/残時計、AI探索終了理由/PVを保存。
46手/46時間記録/23AI探索のdownload実物を照合した。外部送信・巨大探索ログはない。
設定へ戻る前に保存する。毎手の手入力は不要。

OpenShogiUIのGitルートでの再起動手順（5174が未使用の場合）:

```sh
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

loopbackのみ。ローカルdescriptorはOSAIの`local/core-prototype/defense.json`と`r4c1.json`。
元runのbest実体を直接参照し、重複exportを作っていない。公開選択は`release-model.json`の
`model:null`を維持。通常buildは選択未確定として拒否する。
ローカル構造確認だけに`OPENSHOGI_RELEASE_CONFIG=local/r4-c1-preparation/release-selection.local.json`
を使い、出力を防御best1536の1構成に限定した。C1重み・比較UI・precacheは本番出力に混入しない。

## 検証・保全・次の判断

`make check` PASS（Python1,167件、Rust、format/lint、来歴、native、決定的Wasm再生成）。
OSUIのローカル構成付き`npm run check` PASS（235件、型/lint/a11y/境界/来歴/1構成build）。
`npm run integration:ai`は通常解析用Wasmの取り込み漏れを検出し、OSAI commit
`3e83f59cd351dff733e8db390f5ae1304f6921ab`の正本へ同期後PASS。
通常解析Wasmの新hashは`6ad83e2a0631021e08206e505432d940852e94ad46acac6a7b781af1a5de22e6`。
実ブラウザーの解析画面でも81マス・`7g7f`着手を確認。C1/防御のpure runtimeは不変で、
元Arena結果を別runtimeの成績として流用していない。reset確認の「投了する」誤表示も局所修正。

開始時のユーザー変更は`docs/status.md`のLuna完了報告で、内容を保持して今回のレビューへ更新。
原文/diffは`local/r4-c1-review/status.before.md`/`initial-user-diff.patch`へ退避。
両repoとも既存`codex/core-prototype`、既存draft PR #1を使い、別branch/main mergeなし。
最良/C1重み、checkpoint、run証拠、権利/分割metadata、既存W256/r3参照資産を保持する。
使用終了した今回の一構成`dist/`（404 files、16,176,458 bytes）だけ削除した。
symlink・使用中handleなしを確認し、一覧・容量・理由は`local/r4-c1-review/cleanup.json`へ記録。
レビュー根拠は`local/r4-c1-review/`のtraining-review、arena-review、browser-runtime、
downloaded-game、browser-ui、各checkログ。以前の準備・復旧証拠は`local/r4-c1-preparation/`。

次回候補は以下の順。今回は設定・新run・新gateを作らない。

1. ユーザー棋譜で、攻めを継続できない局面と自陣を守れない局面を切り分ける。
   今回のevaluation-10のような攻め切り失敗を、防御区分の名前だけで処方しない。
2. 新sourceへの適合とreplay悪化のトレードオフ、実新規防御13例、粗いattack_end分類を踏まえ、
   攻撃構築/受け/終盤の対象とsamplingの不足を見直す。尺度差や戦法網羅は未検証のまま明示する。
3. scalar proxy/0.2%選抜閾値と実戦・時計のずれを次ラウンド設計で検討する。
   C1の閾値やbestを事後変更せず、10分後手初応手の長さもユーザー対局で確認する。
