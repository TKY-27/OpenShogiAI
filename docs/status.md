# 現在の状態と再開契約

2026-09-12、停止した防御強化runの復旧コードを検証中。本学習は未開始。
親は `defense-20260912-main-r1`、後継は **`defense-20260912-recovery-r3`**。
唯一の現行機械契約は [configs/evaluator-main.json](../configs/evaluator-main.json)。
後継のseal・移行・実resume検証が完了するまでは実行しない。
本書が人間向けの正本で、過去工程はGit履歴に残す。

## 原因・保全・来歴

親runは211 trajectoryを完了した後、game 211 / ply 116で停止した。
11,934行、候補35,338、deviation 2,059、recovery 2,034、focus欠測20。
全211 shard/receiptのhash一致を確認。これは行数でありunique局面数ではない。
群別完了数はgeneral54 / opening54 / defense52 / attack_end51。
未完了game211の58レコード・116手・乱数状態はcheckpointとして保全し、完走に数えない。
壊れた末尾はなく、正常データを再生成しない。optimizerはまだ存在する工程ではない。
親runのstate、契約、失敗JSONとprefix、小さなhash inventoryは
`local/runs/defense-20260912/recovery-evidence-r2/`に保存した。run全体は複製していない。

実教師Apery 2.0.0のbinary SHAは
`8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403`。
元局面はD12/2Mで1.286秒、D12/32Mで18.442秒の診断再現でも未完了。
32M応答は最終D11、32,001,478 nodes。途中D12のrank3はmate -7 upperbound。
bestmoveは返ったが、D12の完全な確定評価集合ではない。timeout・資源超過ではない。
Apery実ソースの停止・completed_depth・最終全rank再出力を照合した。
旧設計の問題は、正しいラベル拒否をrun全体の致命的終了へ伝播したこと。

既存全root/D16集合とprefix58集合を監査し不整合0。短MultiPV445集合もnative合法手数に一致。
旧成功stdoutは未保存なので、新方式で再取得したと主張せず旧code・採用条件付きで継承する。
証拠は `local/runs/defense-20260912/recovery-evidence/` の
`teacher-211-diagnostic.json`、`parent-label-compatibility.json`、`prefix-label-compatibility.json`。
診断再実行は本taskの追加採用試行ではない。元2M/32M消費は移行時に原子的に取り込む。

## 有限復旧と品質

D12の全rank exact値、同じ要求深度、ノード上限未到達、合法PV先頭、bestmove整合を要求。
必要候補数はmin(3, native合法手数)。depthとseldepth、cpとmate、boundを混同しない。
後続boundは古いexact値を無効化する。合法な終局・宣言はnative規則照合で別型として扱う。
低深度結果やbestmoveだけの値を主学習へ入れず、未完了rootでrolloutを推測継続しない。

通常要求2M、深度未達の難局面要求32M、task合計2試行・要求34M。
検索60秒、ready5秒、stop/quit各1秒。終了処理込みの試行予算75秒、task累積150秒。通信障害は該当教師をclose/handshakeし同じ2Mで再試行する。
worker復旧はstartup10秒/ready5秒、永続累積3回。再起動で予算を戻さない。ノード数はgo要求の上限であり、
教師の停止検査による実ノード超過も診断に保持し、壁時計で有限化する。
難局面全体は8,192試行・43,200秒（12時間）以内。
旧359難局面試行・観測2,205.209秒を引き継ぐ。未確定難局面試行は75秒を予約消費したまま復旧する。

rootは永続ledgerのtaskとして保留し、他trajectoryを進める。難queueは防御/攻撃終盤を優先し、
他群と交互に処理する。最大ply数に対応する有限巡回のみ。D16の依存候補群はshard確定前に
同じledger予算内で処理し、未達なら型付きmask。親rootや別候補の値で埋めない。
SQLite取引、ply checkpoint、確定shard/receiptにより再実行時の重複採用を防ぐ。
初期局面・全着手履歴・sampler乱数を保持し、deviationも履歴付きでnative再生する。

直近32個の異なるtaskで24失敗以上かつ8trajectory以上にまたがる故障は生成段階停止。
同一局面の反復だけでは全系故障に昇格しない。合法手不整合、データ/権利/split/identity異常、
資源危険は安全停止。全体の上限72時間、空き60GiB、所有RSS12GiB、swap追加増分512MiB、メモリ空き指標50%以上、
有効出力なし30分。期限は親の開始時刻を継承する。heartbeatだけで進捗とみなさない。

学習開始は検証済みmanifestとcoverageを基準とし、全root無欠測を要求しない。
各family240完了以上、保留root合計96以下・各family8以下。
focusは元の完了率95%以上・欠測200以下を維持し、20要求以上の群/ply帯/branchにも95%を要求。
旧focus欠測20はdeviation5・recovery15であり、依存候補もmaskされる。
不足時は同じfamily/splitに各16 variant、合計192 trajectoryまで補充し、難queueも有限処理する。
保留・解消・欠測の群/family/ply帯/branchをqueueとcoverageへ記録する。
補充後も不足なら学習段階をAstraへ返す。一般群で防御不足を隠さない。

最低train40万unique、新規train15万unique、train各群1万、validation/development各群128は維持。
不足manifestはhash別の版として保存し、補充後に新manifestを作る。学習に渡す版を固定し、
prepare/train/audit入口でもidentity・coverageを検査する。進行中のデータを黙って差し替えない。

## 学習方針・権限

防御・序盤重点、攻撃/終盤維持。r3 OSAVAL03 W256全層継続、controller OFF。
一般30%/序盤20%/防御30%/攻撃終盤20%、旧r3 train replay最大30万行。
初期r3はstep6144、SHA
`cd07f2a202f6e781afcb6a8af3c7a198c4203373e4505fe089f8c7f05eefd983`。
CPU4thread、batch256、最大12,288 updates/8巡/1行8露出。評価・選抜条件は機械契約を維持。
12自作familyの派生・先後反転は同一splitに固定し、独立人間棋譜とは数えない。
旧split対称重複と既知診断除外、権利metadataを維持し、最終holdoutは未開封。

Astraはコード/採用条件/契約と短試験を担当。ユーザーが次に開くLuna Max/maxが本学習を実行する。
本学習は承認済み。契約内の復旧・保留・補充・段階進行は再承認不要。
Luna子エージェントやAstra長時間監視を使わず、実行中のコード変更・再sealをしない。
契約外のモデル/ラベル設計はAstraへ返し、全工程終了は`awaiting_astra_review`。
定跡なし、第三者教師はoffline学習のみ、pure非終端評価に手作り点を混ぜない。
本番一モデル、開発比較選択の方針を維持。本作業はOSUIを変更しない。
main merge、公開既定昇格、重み配布、deploy、有料計算、force-push、履歴書換えは行わない。

## 承認された資源基準の移行

r2の最初のprobeは教師起動前に旧swap増分上限で停止した。旧基準1,500,313,026 bytesに対し
実測4,191,221,186 bytesで、元D12問題とは別の停止だった。メモリ空き指標は80%。
ユーザーの明示承認によりr3は実測4,191,221,186 bytesを基準とし、追加増分512MiB・
メモリ空き指標50%以上に制限する。元の基準・r2停止証拠を保持し、自動再基準化はしない。
元72時間期限、全task試行履歴、RSS12GiB、ディスク空き60GiBは維持。
`resource_epoch`が承認・旧基準・実測・停止証拠をhash-bindする。

## 操作

cwdはOpenShogiAI Gitルート。後継の実行正本は
`local/runs/defense-20260912/recovery-r3/run.json`、同所の`seal.json`がcode/run identity。
`state.json`、`data/tasks.sqlite3`、`data/recovery-queue.json`、`data/generation-progress.json`を引き継ぐ。
親 `main/` の`needs_astra`を直接解除しない。学習準備を一からやり直す運転ではない。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/defense-20260912/recovery-r3
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run start local/runs/defense-20260912/recovery-r3
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run stop local/runs/defense-20260912/recovery-r3
```

`start`がresumeコマンド。Astraの一度だけの準備はseal→migrate→probe。
Lunaは実施済み移行をやり直さず、ready状態からstartする。
検証とGit配送の最終結果は実resume確認後に本書へ追記する。
