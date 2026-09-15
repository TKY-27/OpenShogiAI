# 現在の状態と再開契約

2026-09-15、対象は **`defense-20260912-recovery-r3`**（親 `defense-20260912-main-r1`）。
**現在は`ready_for_luna / requested_pause`、attempt 6。別プロセス2回の通常resumeで有効出力の増加を確認し、正常pause済み。**
本書が人間向けの正本。科学的条件は [configs/evaluator-main.json](../configs/evaluator-main.json)
と不変のsealed run、通常resumeの運用条件は同runの`approved-operation.json`が指すrevisionを使う。

## 今回の期限廃止と再開確認

ユーザーの2026-09-15の明示承認により、このrunに限って**固定暦日期限を廃止する**。
旧期限は`state.began_at = 1789190144.143338`（親runから継承）＋
`resources.maximum_wall_seconds = 259200`秒、2026-09-15 14:15:44.143338 JST。
`_resource_admission`、stage起動直前、supervisor監視ループの3経路で同じ値を参照していた。
旧attempt 4の終了理由`needs_astra / swap_limit`と、今回の起動前拒否`wall_limit`は別である。
旧期限拒否ではattempt 5、教師、出力は発生していない。

新運用`finite-work-v1`は次の絶対期限や代替の総時間上限を設けない。
中断・修正待ち・ユーザーpause・resource_wait・プロセス不在を計算消費とは数えない。
`began_at`は履歴として保持し、時計巻戻しでも開始時刻との大小だけで拒否しない。
カレンダー経過は診断に別表示。過去の総実稼働秒は**不明（null）**とし、0や暦日差に置換しない。
教師task・hard queue・評価の既存実消費、有限試行、nodes/timeout、生成/補充目標、
12,288 updates等の学習上限、評価計画、無進捗30分・資源安全条件は維持する。
有限仕事量の完了または既定の停止条件までであり、無限探索や品質基準の緩和ではない。

移行は既存`approve-operations --abolish-calendar-limit`でAstraが一度だけ適用する。
運転code commit/hash、元seal、元calendar上限、前operation参照、承認理由を同runの
`operations/`へ記録する。原子的な参照更新の前後で中断しても再実行で同revisionを再利用する。
元run/seal・旧承認・失敗・DB/payloadを保持し、通常resumeとsupervisor/child/後段stageは
stateに記録した同じ運用revisionを検証する。Lunaの追加フラグ・JSON手編集・再sealは不要。
科学的設定、教師品質、採用条件、sampling/split、coverage、学習率/更新数/評価は変更しない。

`macos-attempt-v1`の既存資源修正とDEVNULL共通launcherは現codeで有効。
旧swap epochとの差だけでは停止せず、15秒間隔の実圧力・swap使用量/活動率・所有RSS・
diskを確認する。起動は初回＋3連続安全差分、最大5標本の有限待機。
normal・空き指標55%以上・disk62 GiB以上・RSS9.6 GiB以下・swap各1 MiB/s以下が回復条件。
実行中はcritical/空き20%未満・disk60 GiB未満・RSS12 GiB超で即停止、
warning/空き50%未満・swap活動8 MiB/s超等は2連続で停止する。
危険/計測不成立はresource_wait。旧swap停止の原本はそのまま保存し、最新の起動保留を別表示する。
重大故障や破損は解除せず、手動pauseは次の明示resumeまで保持する。

起動診断は同moduleの`diagnose`（読み取り専用）。lease/process、承認code/seal/input、
移行、receipt/SQLite/cursor、期限、現在資源をまとめ、依存条件で調べられないものは未確認とする。
`status`は`latest_resume`と旧attemptのstate/reasonを分ける。通常resumeの最新結果は
`last-resume.json`へ保存し、旧attempt-resultを起動拒否で上書きしない。

修正前の実確認：543 trajectory / 30,571 rows、accepted 45,067 / deferred 21 / hard 10。
最新有効receiptはgame553、最新未完了はgame554 / ply61 / 部分31行。
hard共有累積963試行 / 4,526.367442550106秒。SQLite・全receipt/hash・task identity/payload・
全未完了cursorのnative再生は正常。件数から再開位置を算出せず、ledgerから継続する。
family coverage 43–47件は必要240件未満。accepted task数をcoverageや行数と混同しない。
prepare/train/audit/arena/exportは未開始。現在のcoverageを合格扱いにしない。

### 実再開の結果（PASS）

運転code commit **`7c212f2e838146bdf3212f4ee8a5b36abfb66a53`**。
旧operation `2b112c40…`から新operation
`25b025bc84fb34a380e38742c20b72d13026e5fa017e45a3d5d0c199ab70d4ff`へ正式移行した。
`finite-work-v1`と`macos-attempt-v1`を同時に検証・採用する。

| 別プロセスの通常resume | 新規確定receipt | trajectory | rows | accepted task | supervisor / stage PID |
| --- | --- | --- | --- | --- | --- |
| attempt 5 | game230、132ply / 66行 | 543 → 544 | 30,571 → 30,637 | 45,067 → 45,082 | 43490 / 43503 |
| attempt 6 | game251、176ply / 88行 | 544 → 545 | 30,637 → 30,725 | 45,082 → 45,102 | 44602 / 44615 |

両回とも同じcwd・uv/Python・stdin=/dev/null・上記通常resumeを別CLIプロセスから実行し、
正式pause完了後に次回を起動した。追加probeフラグは使っていない。
起動資源確認は45.596秒 / 46.293秒。短い出力確認とpauseだけで本生成全量は実行しない。
永続難queueから未完了game230とgame251を継続して完了した。元部分行を既存完走行へ重複加算しない。
game554/ply61/部分31行のcheckpoint hashは不変。旧543組（1,086ファイル）のshard/receipt hash、
元45,067 accepted task、run/seal、旧停止attempt、開始日時、retriesを保持した。
両新receiptのhash/行数、全132/176着手と最終局面をnativeで再生確認した。

現在 **545 trajectory / 30,725 rows、accepted 45,102 / deferred 22 / hard 7 / running 0**。
hard共有累積は963 → 965 → 966試行、4,526.367 → 4,531.574 → 4,558.645秒。
未達taskを局所deferredへ移し、教師深度や採用条件を緩めず継続した。
family coverageは**44–47 / 必要240**で不合格。元のcoverage関数で確認し、
診断出力だけを証拠フォルダへ保存した。prepare/train/audit/arena/exportは未開始で、fitも存在しない。

4監督標本はすべてsafe、メモリ空き指標最小62%、所有RSS最大1,517,273,088 bytes、
disk空き最小154,096,541,696 bytes。最終はsupervisor/stage非生存・残留groupなし・lease解放。
次のresumeは改めて有限資源admissionを行う。恒久的な資源充足を保証するものではない。
現在の起動阻害条件は解消済み。coverage不足は残りの有限生成・再解析・補充で判定する次工程の条件である。

最終運転codeの`make check`は**Python1,122件、Rust、format/lint、locked依存、
権利/境界/provenance、native build、決定的Wasm照合までPASS**。
追加23件と既存起動/資源/resumeを合わせ143件PASS。時計の前進/巻戻し、旧swap回復、
全stageへの運用伝播、元sealと予算保持、承認参照の原子的公開前中断と再実行、
破損/二重起動/危険資源拒否、ユーザーpause、最新起動結果のlease内保存を検証した。
既存D12未達・EBADF・resource_wait/boot変更・coverage不成立/成立fixtureの回帰も通過した。

証拠は`local/runs/defense-20260912/deadline-evidence/`。
`final-verification.json` SHA-256:
`c43a5af1d3f7f17b536091e3ba40d77affcb5ec7877953a619081a604a48bb45`。
全体checkログは`make-check-7c212f2.log`。原本・運用承認・実出力・診断をローカル保持し、Gitへ入れない。

## Lunaの正式操作

cwdはOpenShogiAI Gitルート。別の新規Luna Max/maxセッションが使う実在コマンド：

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/defense-20260912/recovery-r3 </dev/null
```

正常pause：

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run pause local/runs/defense-20260912/recovery-r3 </dev/null
```

本生成/学習/評価の長時間監視はAstraで行わず、Luna子エージェントを起動しない。
全工程終了は`awaiting_astra_review`。OSUI変更・main merge・公開・モデル昇格・
重み配布・有料計算・force-push・ブランチ整理は対象外。

以下は過去の修正・有限契約・実行証拠。旧期限維持の記述は当時の履歴であり、
対象runの現在の暦日運用は上記の明示承認`finite-work-v1`が置き換える。

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
現在は冒頭の明示承認によりmacos-attempt-v1とfinite-work-v1を適用する。
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
