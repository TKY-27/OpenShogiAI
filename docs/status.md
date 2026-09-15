# 現在の状態と再開契約

2026-09-15、対象は **`defense-20260912-recovery-r3`**（親 `defense-20260912-main-r1`）。
現在の実停止はattempt 4の **`needs_astra / swap_limit`**。
543 trajectory / 30,571 rows、accepted 45,067 task / deferred 21 / hard 10、running 0。
最新の未完了cursorはgame554 / ply61 / 部分31行。件数から再開位置を算出しない。
SQLite整合・確定receipt/hash・task identity/payload・未完了cursorのnative再生を再確認した。
本学習は未開始。r3・比較重み・既存生成データ・封印済みrun/sealは保持する。

本書が人間向けの正本。科学的条件は [configs/evaluator-main.json](../configs/evaluator-main.json)
とsealed run、運用policyは承認済み`operations/` revisionを参照する。
**継承した72時間の絶対期限（2026-09-15 14:15:44 JST）は超過している。無承認で延長せず、実再開試験・Luna継続は保留。**
以下の2026-09-12の実績は過去の証拠であり、今回の実再開PASSではない。

## 資源監視・再開の運用改訂（macos-attempt-v1）

attempt 4の比較は`17,997,758,464 - 4,191,221,186 > 512 MiB`。
基準はattempt開始値ではなく9月12日承認のr3 resource epoch。元親値1,500,313,026も保持されている。
標本時刻1789444062.332198、stage PID19124起動から約0.169秒、所有RSS26,148,864 bytes。
stage起動直後の標本であり、教師の起動を示す証拠はない。中断中のMac全体の増加が比較へ入っていた。
OSAI消費量とは断定しない。attempt 4の新規出力はない。

計測元の意味はApple公開実装で確認した。
[sysctl](https://github.com/apple-oss-distributions/system_cmds/blob/main/sysctl/sysctl.c)の
`vm.swapusage used`は現在使用量で、`total`や累積swapoutではない。表示MはMiB、小数2桁丸め
（旧bytes値にも約0.005 MiBの表示精度限界がある）。
[vm_stat](https://github.com/apple-oss-distributions/system_cmds/blob/main/vm_stat/vm_stat.c)の
Swapoutsはboot中の累積ページ数であり、実機は16,384 bytes/page。圧縮セグメントのswap活動量で、
RSSやswap使用量と同じ量ではない。boot識別とpage sizeを伴う差分だけを使う。
[pressure sysctl](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c)
は内部enumではなくdispatch flags（1 normal / 2 warning / 4 critical）を返す。
[memory_pressure](https://github.com/apple-oss-distributions/system_cmds/blob/main/memory_pressure/memory_pressure.c)
の`-Q`はquietな照会であり、圧力生成オプションを使わない。空き割合は当該指標として扱う。

45秒・4標本の実測はswap16,980 MiBで一定、swapout差分0 pages、pressure normal、
メモリ空き指標70%、ディスク空き約144 GiB（`shutil.disk_usage`/`df`）。OSAI教師/nativeプロセスなし、lease/残留groupなし。
これは観測時点の状態であり、次のresumeでは必ず再計測する。

- 履歴のrun初期snapshotは不変。起動直前のattempt baselineはID・boot・時刻・元/単位を付け、
  attempt記録へ一度だけ保存する。同attemptで取り直して上限を回避しない。
- 起動前は15秒間隔、初回標本＋3連続の安全な差分（通常約45秒）、最大5標本。
  待機間隔の合計は最大60秒、各取得25秒以内を要求し、取得時間込みでも最大185秒。
  normal、空き指標55%以上、空きdisk62 GiB以上、所有RSS9.6 GiB以下、
  swap使用量増加/swapout活動各1 MiB/s以下を要求する。計測失敗・古い標本・boot/counter不連続は保留。
- 実行中は15秒間隔。critical pressure、空き指標20%未満、disk60 GiB未満、所有RSS12 GiB超は即停止。
  warning/空き指標50%未満、swap活動8 MiB/s超は2連続で停止。
  attempt増分512 MiB超かつ使用量増加1 MiB/s超も2連続で停止する。単に高swapというだけでは停止しない。
  通常の保存待ちは最大80秒、criticalは即時停止処理。正式pauseの待機は最大120秒。
  generateで監督自身が送った終了信号だけは、保存cursor/資産検証後に再開可能とする。
  所有RSSはsupervisorとstage groupの共有メモリを含む近似合計で、ホストswapをOSAIへ帰属させない。
- 待機は既存schemaの`stopped / resource_wait`（実行中停止理由は`resource_wait:詳細`）。
  finite admissionが回復しなければ終了し、新task/attemptを増やさない。正式resumeで再試行できる。
  自動再開は行わない。ユーザー停止は`stopped / requested_stop`、正式pause完了は
  `ready_for_luna / requested_pause`で区別し、ユーザーのresumeまで停止する。
- 旧swap_limitだけのneeds_astraは、attempt結果一致・cleanup・lease・process・code/policy・
  資産/SQLite/cursor・現在資源の検証後に正式resumeで回復する。他原因の重大停止は解除しない。
  強制中断後のrunningも残留writerなしを検証して同cursorから再開する。
- 運用改訂を`approve-operations`でcommitとpolicyへhash-bindし、元run/sealは編集しない。
  資源の累積ピーク・各標本・待機時間・停止理由と旧attemptを保持する。
  期限、task試行/難queue会計、教師深度/nodes/Threads/Hash、学習条件、評価条件は変更しない。

運転code commit: `946be86`（主修正`b55916a`）。承認済みoperation revision SHA-256:
`2b112c409d97cad5761f032767ca5c000033c2df47b3da05f4b0ac3c6a1c5085`。
`approved-operation.json`が次のresumeへ渡す正本。attempt 4のstate中の旧revisionは停止履歴として保持する。
証拠は`local/runs/defense-20260912/resource-evidence/`。
今回の実再開2回・新規有効出力増加は**未実施（絶対期限超過）**。
資源注入/監督/resume回帰51件、既存launcher/resumeと合わせて125件PASS。旧baseline、高swap安全/危険、実page size、短いノイズと
継続負荷、boot/counter不連続、計測不能、有限待機/回復、期限/attempt基準保持、
旧swap停止限定解除、他重大停止/二重起動拒否、ユーザー停止、後段stage移行、前bootの履歴を保持した新attempt開始を検証した。
正式resumeも別プロセス・stdin=/dev/nullで実行し、`wall_limit`の事前拒否を確認。
旧state・attempt 4・run/seal・生成データは不変、attempt 5は作られていない。
この拒否確認を実生成の再開PASSとは扱わない。
最終code `946be86`の`make check`はPython1,099件・Rust・format/lint・locked依存・
権利/境界/provenance・native build・決定的Wasm照合までPASS。
ログは`resource-evidence/make-check-946be86.log`。
本学習・実生成2回のpause/resume・OSUI・main merge・昇格・deployは未実施。

## 起動EBADFの原因と今回の検証

完全な元ログはPythonの `init_sys_streams`、`OSError: [Errno 9] Bad file descriptor`、
`<no Python frame>`。生成moduleの実行前で、旧D12未達とは別件。
元の実起動は `uv run --frozen python -m open_shogi_training.evaluator_run start ...`、
同じGitルートで `tty: true`。旧`start`（623d203の865行）と`_run_stage`（1213行）はstdin未指定で、
CLI端末のFD0をsupervisor、stageへ暗黙継承していた。

macOSでは制御PTYのsession owner終了後、保持中slave FDの`fcntl(F_GETFD)`は成功しても
`fstat`がEBADFになる。これを別Pythonへ継承すると元ログと同じfatal/exit1を再現した。
CPython 3.12.13の`Python/pylifecycle.c:2580`でstdinを構築し、
`Modules/_io/fileio.c:443,453-454`のfstat/EBADF分岐から、`pylifecycle.c:2636`のfatalへ至る。
元PTY所有PID・実際の失効イベントは記録されていないため、そのPIDまで実測したとはしない。

修正は共通`_launch`で毎回stdinをDEVNULL、stdout/stderrを新しく開いたappend logへ結ぶ。
親はspawn後に自身のlog handleを閉じ、子はdup2された独立FDを使う。
leaseだけをその生存process treeへ継承し、再開時はlockを新規openする。FD番号を再開資産にしない。
USI教師とnative replayのstdin/stdout PIPE・読み出しは変更していない。
診断は元tracebackとcleanup失敗を別々に保持する。STOPとexit1を正常pauseへ変換せず、
専用終了code75と保存成功を確認する。結果保存失敗や未知のEBADFはneeds_astraに残す。

運転code commit: `e7edb5026a0976856ef0b9d6e7d84b3b02a35d9a`。
承認済み運転revision SHA-256:
`0e54f1be931f51ac1685f26d1a813690924333de2973b4febdbd9636386476b9`。
元のrun/seal/使用commitは不変で、今回の実行は`attempts/000001.json`、`000002.json`へ紐付く。

| 別プロセスの実行 | trajectory | rows | accepted task | supervisor / stage PID | resumeからpause完了 |
| --- | --- | --- | --- | --- | --- |
| attempt 1 | 212 → 213（game213確定） | 11,966 → 12,022 | 75 → 207 | 5628 / 5636 | 35.876秒 |
| attempt 2 | 213 → 214（game214確定） | 12,022 → 12,062 | 207 → 301 | 6010 / 6017 | 35.800秒 |

両回とも同じuv/interpreter（Python 3.12.13、`.venv/bin/python3`）、cwd、設定、
DEVNULL stdinの非対話CLIを使い、正式pauseでprocess/lease解放後に次のCLIを起動した。
新112手/56行と80手/40行は保存receipt/hashとnative全着手再生を再確認。
旧212組のshard/receipt、旧75 accepted task、元deferred taskと2M/32M消費履歴、
game211の58行/116手/乱数checkpointは不変。deferredは全体停止にならず後続へ進んだ。
第1pauseのgame214/ply0を第2resumeが完走し、現在の次cursorはgame215/ply0。
hard共有累積は360 → 362試行、2,207.048 → 2,215.622秒。元期限・swap基準・retriesは不変。
ログは元failure prefixを保持して追記され、新しいEBADFなし。

最終状態はsupervisor/stageとも非生存、lease非保持、`lsof +D`でrun内open fileなし、fit未作成。
4監督標本のメモリ空き最小56%、swap最大3,746,100,674 bytes、所有RSS最大1,303,478,272 bytes。
`make check`はPython1,048件、Rust、format/lint、権利/境界/provenance、native build、
決定的Wasm再生成照合までPASS。失効PTY/closed stdin・closed log・spawn直前EBADF、
承認後code drift、cursor破損、不正pause、結果保存失敗、二重起動、後段の共通launcher経路を検証。
テスト用データは隔離fixtureで、本番データに混ぜていない。

証拠は `local/runs/defense-20260912/ebadf-evidence/` の元ログ・元起動command、
`round-1.json`、`round-2.json`、`processes-1.json`、`processes-2.json`、`make-check.log`。
最終 `final-verification.json` SHA-256:
`e0c95bb770ab1ac5b2a2a0a456bd5fa4b97b1269decae30ef42d34778274ec63`。
PTY隔離再現は `local/ebadf-diagnostics/receipt.json` と実CPythonソースに保持。
本学習・大規模Arena・OSUI・main統合・公開・deployは実行していない。

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
検索60秒、ready5秒、stop/quit各1秒。終了処理込みの試行予算75秒、task累積150秒。
契約の120秒表記は検索待ち2回の合計で、通信・終了処理込みの値は75/150秒。通信障害は該当教師をclose/handshakeし同じ2Mで再試行する。
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
資源危険は安全停止。以下の資源数値は封印時の履歴で、現行監視は冒頭の運用profileを適用する。
全体の上限72時間、空き60GiB、所有RSS12GiB、swap追加増分512MiB、メモリ空き指標50%以上、
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
停止中の起動・log管理だけの修正は、Astraがcommit後に運転revisionを承認し、
同じrunのattemptへ記録する。検証済みresume/pauseはLunaが再承認なしで実行できる。
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
この段落は旧policyの履歴。現在の運用判定は上記macos-attempt-v1を承認revision経由で適用する。
`resource_epoch`が承認・旧基準・実測・停止証拠をhash-bindする。

## 操作

cwdはOpenShogiAI Gitルート。後継の実行正本は
`local/runs/defense-20260912/recovery-r3/run.json`、同所の`seal.json`がcode/run identity。
`state.json`、`data/tasks.sqlite3`、`data/recovery-queue.json`、`data/generation-progress.json`を引き継ぐ。
親 `main/` の`needs_astra`を直接解除しない。学習準備を一からやり直す運転ではない。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/defense-20260912/recovery-r3 </dev/null
```

確認は同moduleの`status`、正式pauseは以下。再開コマンドは毎回同一で、別実験・全データ複製・再sealは不要。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run pause local/runs/defense-20260912/recovery-r3 </dev/null
```

resumeはlease取得、残留process確認、承認済みcode/input hash、確定shard/receipt、
SQLite整合・未完了cursorのnative再生を検証し、元state/log prefixと運転revisionを
`attempts/`に保存して状態遷移と起動を行う。元`run.json`と`seal.json`は不変。
Astraの`approve-operations`は運転code review後の一回の準備で、stateをreadyに書き換えない。
Lunaは承認済みhashを使い、別commitの運転codeを自動承認しない。
教師・データ意味・学習・評価条件の変更は、この運転復旧の対象外。
資源監視の運用方針のみmacos-attempt-v1への今回の明示承認範囲で更新する。
データ書込みのEBADF、未知のstage失敗、診断/cleanup保存失敗を安全な再試行に分類しない。
pauseは専用終了codeと全process/lease解放、保存済み進捗を確認してreadyへ戻す。

## 前回の内部probeと不変の実験identity

- code commit: `623d203898d7487921e0599d8500c946978e3544`
- sealed run SHA-256: `507e535f946756a3bb43243450df68fd58c11cc95e7883659de5574fe2637152`
- label recovery契約revision: 2。r3は承認済みresource epochを含む後継実行identity。
- 復旧実装commit: `e9fba70caa425d53bd0db6cb359cb6eb1df15bd3`

実probeはgame211/ply116を2試行消費済みdeferredのまま扱い、その後game212を確定。
新しい防御群 `opposed_silver_pressure` / development_test / variant17は63plyでnative終局、
32行・96候補・deviation6・recovery6、focus欠測0。生成処理10.895秒。
全完了件数は212、延べ行11,966。新規教師task75 accepted、旧root1 deferred。
既存211 shardと親snapshotの全hashが不変。未完了58行を件数へ混ぜていない。

主経路の次の未着手IDは213。0..210と212のreceiptは再利用され、211はledgerの
`f8f2dff6f48b86f6090e48053e82e31f5213e3da0e16506a1232e034499a3d01`
に2M/32Mの2試行と116手の履歴を保持する。
難局面共有累積は360試行、観測2,207.048秒（旧359試行を含む）。
新shard SHAは `7044458baa54f4286521f91c0eee291b50fc6aec2a78f03bc211f5a04eb06d74`。

実probe後はsupervisor/stageとも終了、lease解放、fit未作成、学習未開始。
監督標本のメモリ空き指標最小77%、swap最大4,014,798,274 bytesで承認基準内。
最終実証は `local/runs/defense-20260912/recovery-evidence-r2/final-verification.json`、
SHA `e5150a14b66199e2b2dd1eaefa2159173878b7990428ab34a15273820c73bb26`。
`recovery-evidence-r2`という証拠フォルダ名は保持しており、実証対象runは明示的にr3。

最終`make check`はPython1,030件、Rust、format/lint、権利/境界/provenance、native build、
決定的Wasm再生成照合までPASS。実SIGKILL、commit直前/直後の実process exit137、
再開時の重複/試行reset防止、複数task未達・次task進行、bound/不完全MultiPV、
通信/再接続/終了race、群別不足、有限補充、manifest退避中断、局所保留付き段階進行を試験。
一時I/OのEBUSY/EAGAIN/EINTRだけ最大3回とし、EIO等は隠さず停止する。
全障害ゼロや本学習の正常終了率は保証しない。今回の局所未完了が全体停止へ伝播しないことを
実教師・実resume・確定shardで確認した。

変更は関連枝 `codex/core-prototype` のみへ通常pushし、既存draft PR #1を更新済み。
main統合・公開・重み配布・force-pushは実施しない。Lunaが読むのは本書と上記sealed runのみ。
