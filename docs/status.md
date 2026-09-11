# 現在の状態と次工程

2026-09-11、`evaluator-20260910-main-r3` の学習・export監査・固定対局は完了し、
Astra/xhighによるOSAI/OSUIの実ブラウザー工程も **PASS**。開発版でr3と旧基準を
選んで対局でき、ローカル本番相当previewでは一構成だけを使用することを確認した。
学習監督の終端status `awaiting_astra_browser` は再起動防止のため保持し、既存stateの
`astra_browser.status=PASS` とhash付き結果参照にbrowser完了を記録した。新たな学習state名や
再sealは導入していない。追加学習・教師生成・大規模Arena・Luna監視は再開しない。
この文書を現在の状態と次工程の正本とする。

- r3 bestは **step 6144、OSAVAL03 W256、旧制御器OFF**。
  評価器SHA-256は `cd07f2a202f6e781afcb6a8af3c7a198c4203373e4505fe089f8c7f05eefd983`。
  正本は `local/runs/evaluator-20260910/main-r3/fit/best.osaval03`。
  最終更新は8192だがbestとは異なる。保存された8192の再開checkpointに含まれる
  `best_step=6144` と `best_model_bytes` が、このruntime exportと完全一致することを確認した。
  exportの作り直しや再学習は不要だった。
- ユニーク局面はtrain **435,147**、validation **22,039**、development test **25,667**。
  学習の延べ使用は2,096,940件、1局面の最大露出は5回。旧凍結W256、必要な再開資産、
  原本の評価記録、未開封最終holdoutを保持している。
- 既存evaluationは **32戦32勝**。相手は同じ封印済みnative engineで動く凍結W256 hard2
  (`859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480`)。
  両者制御器OFF・定跡OFF、3分24局／10分8局、先後各16局、depth64・qdepth4・TT2MiB・1thread。
  全32局は詰みで終了し、無効局・異常attempt・絶対期限違反は0。
  開始局面は同runのdevelopment testに属する16本の自前trajectoryの16手目。
  train/validationの選択とは分離されているが、offline開発評価と同じ集合由来であり、
  外部の独立した人間棋譜や最終holdoutによる検証ではない。
  初期局面からのdemonstration **8局は6勝2敗**で、32局の評価成績へ加算しない。
  旧基準に対するこの条件での改善を示すが、初段棋力・他の相手・ブラウザー動作は認定しない。
- export監査はnative／実Wasm-in-NodeでPASS、10根局面と合法子300の評価差0。
  40局の小型receipt署名とtrain/audit/arena完了資産hashを照合済み。
  封印codeは `a2bc10d1bdba22caae70b8078bcad3ea19812068`、run SHA-256は
  `8f279bae1d0e4a980afb29e40bc7df8984b621cb3718934cff6d39b2ebce3ebf`。
  今回の監査時点で封印対象のOSAI source hashは全一致し、engine／探索／時計の意味は変更していない。
  元のrun、runtime、評価成績のidentityを保持し、再sealしていない。
- OSUI `41ec5a6f4bbf418bf83c15b74ae948202cc7c5a4` で実装・実画面確認を完了。
  開発 `http://127.0.0.1:5176/#/match` はr3候補をローカル既定とし、旧基準W256を選べる。
  旧試作は同じW256に制御器をONにした構成なので、別モデルとして重複登録せず開発用の別項目にした。
  r3はcontrollerなし/OFF固定。品質は同一重みの探索方針で、別モデルではない。
  読込中再選択はAbort/世代ID/Worker破棄で保護し、対局・停止中は選択不可。
  明示終了/reset、HTTP失敗・hash不一致の開始禁止と同じ候補の再試行を実確認。
  正常読込後のWorkerは1個、失敗後0個。旧→r3→旧・同モデルreloadで実hashとidentityが追従する。
- JS/Wasm/評価器/旧制御器の実バイトを初期化時にSHA照合し、検証済みJS bytesをBlob moduleとして実行。
  Wasm loadModel identityと実探索のpure proofを診断へ保持する。毎手hash計算なし。
  共通Wasm SHA-256 `0df3ce43b275137bf18ccbf1372c4dacdc99645db6b7b3630669b8528252ad1f`、
  JS `907da1421263a2cbc621297095d13ccd3f707942b1e6f09bd7a2b2512efeee2e`、
  旧制御器 `66110bae4ef5fbedd6a3c5537813f0fae5b9276b576a745b2364b4a093141b63`。
  r3には旧制御器を流用していない。内容hash付きURL/no-storeで提供し、Service Workerは登録なし。
- 本番選択 `release-model.json` は `model: null` の未採用状態を維持。指定なしbuild、
  複数/形式不整合/hash不一致の小型試験、非productionモードのbuildは明確に拒否する。
  ignoredローカル一時指定でr3の評価器・Wasm・JSの3構成要素だけを出力し、実previewで推論確認。
  旧モデル、checkpoint、source map、旧標準Wasm、selector、解析upload/lab、比較routeは出力に含まれない。
  Viteのpublic自動コピーを無効にし既存UI資産をallowlistで出力、dist再生成・Workerを含む全出力を監査。
  preview `http://127.0.0.1:4176/#/match` でも実hash/推論がr3と一致し、旧選択のstorage/queryを無視。
  旧資産やsourceの不存在URLはpreviewのSPA fallback HTMLを返すが、重みやsource bytesは取得できない。
  標準/高品質・時計/先後は維持。本番採用・公開既定変更ではない。
- OSUI230 testsとformat/lint/type/accessibility/boundary/license/provenance/駒資産audit・一構成buildがPASS。
  OSAI docs/boundary/license/provenanceとnative/Wasm凍結smokeがPASS。engine sourceは未変更のため、
  完了済み学習や全Rust/Python試験の再実行、同一Wasmの再ビルドは行っていない。
  本番境界・runtime/UI統合の独立レビューで重大な追加指摘なし。実画面はDesktop Chrome、
  小viewport390×844。81マス盤面、モデル選択、console例外0、横はみ出し0を確認した。
- r3監視heartbeatは完了に伴い削除済みというユーザー報告。今回のprocess確認でもr3の
  supervisor・学習・評価processは0で、監視を再開していない。
  **promotionなし、公開既定変更なし、main merge・本番deploy・Release・重み公開なし**。

## browserの時計・序盤・終局結果

r3は3分/10分×標準/高品質×先後の8条件、各2回の序盤探索を実Workerで測定。
後手の初回は7六歩後。初手の固定や定跡・手の禁止は使わず、同じ残り対局時計で配分した。
初回/次回の実待ち時間（秒）は以下。準備時間161–197msは別計測で時計へ加算しない。

| 時計 | 品質 | AI手番 | 初回 | 次回 | 序盤のAI着手 |
| --- | --- | --- | ---: | ---: | --- |
| 3分 | 標準 | 先手 | 0.456 | 1.012 | 2g2f → 7g7f |
| 3分 | 標準 | 後手 | 0.593 | 0.658 | 8c8d → 5a4b |
| 3分 | 高品質 | 先手 | 0.476 | 0.926 | 2g2f → 7g7f |
| 3分 | 高品質 | 後手 | 0.557 | 0.625 | 8c8d → 5a4b |
| 10分 | 標準 | 先手 | 1.742 | 5.283 | 2g2f → 2f2e |
| 10分 | 標準 | 後手 | 4.061 | 3.567 | 5a4b → 8c8d |
| 10分 | 高品質 | 先手 | 1.710 | 4.915 | 2g2f → 2f2e |
| 10分 | 高品質 | 後手 | 3.814 | 3.247 | 5a4b → 8c8d |

r3全16探索はstable終了、初回hard limitは3分5.350秒/10分17.950秒、全探索が上限内。
目標時間は局面の安定性により1.798–9.064秒で変化し、常に使い切る実装ではない。
旧比較の同時計代表4条件・8探索は初回3分0.755/0.873秒、10分3.273/5.532秒、
次回まで含め最大5.847秒、全stable/上限内。目標・絶対停止時刻・実探索・待ち時間・
準備時間・remaining clock・full model/runtime identityは `browser-clock.json` に保存。
重いbuildや学習と時計比較を並行していない。

初手約1分・高品質の過剰長考は今回の代表局面で再現せず、engine時計の追加修正は不要だった。
r3先手は歩を進めるが、後手では早い `5a4b` の玉移動が残る。桂跳ねは今回の短い序盤では
観測しなかった。玉移動の妥当性や棋力不足の解消は認定せず、ユーザー棋譜の厳密再現や
教師追加診断は未実施。学習課題と実装不具合を混同せず、着手の禁止・定跡・再学習は行っていない。

残り0/50/200ms、先後/品質、1手詰・終盤・詰み・非王手合法手なしの18実Worker fixtureがPASS。
通常UIで探索中停止→再開→人間着手→AI応手→投了→再対局、対局中選択拒否を確認。
別の初期局面からの通常UI対局は、操作試験用の決定的な合法手を相手として17手目の詰みで完了。
この対局やユーザー由来開発局面を32戦の独立棋力証拠へ加算しない。

証拠は `local/runs/evaluator-20260910/diagnosis/astra-browser-r3/` 内の
`browser-complete.json`、`identity-audit.json`、実画面PNG、console/networkと小型JSON。
Browser plugin/skillはセッションに存在せず、Playwright実ブラウザーを使用した。
MCP unsafe実行環境のfilesystem import不足とfault-injection復旧の不安定さは記録し、
同じUI経路のnetwork障害・本番preview検証を既存Playwright/Chromeの別processで完了した。
mockや画像生成を実画面証拠として使っていない。他ブラウザーや物理モバイル端末は未確認。

開発サーバー5176とローカルpreview4176は稼働中。再起動はOSUIの既存READMEに記載した
`npm run dev -- --host 127.0.0.1 --port 5176 --strictPort`、
`npm run preview -- --host 127.0.0.1 --port 4176 --strictPort`。
OSUIの全check/buildには同READMEのignored一時release指定が必要。本番選択は未決定を維持する。
ローカル証拠は約1.2MiB、単一buildは約16MiB、空き約229GiB。重みの追加export/複製登録なし。
凍結基準/r3 best/必要な再開資産/評価原本/未開封holdoutは保持した。
browser合格とpromotionは別で、次の学習・公開・モデル昇格へ自動で進まない。

## 履歴：2026-09-10の本学習準備・復旧

以下は当時の記録を保持したもの。「現在」「起動中」「未開始」「残工程」、PID、
heartbeatのACTIVE記述はその時点の状態であり、現在の運用指示ではない。
現在の完了状態・残工程は上記を参照する。

2026-09-10、ユーザーは序盤評価・時計の修正、評価器を含む本学習、新規データの
取得・生成、開発用OpenShogiUIでの比較・再評価を一工程として承認した。
学習開始の再承認は不要。公開既定モデルへの昇格、mainへの試作統合、本番公開、
重みの公開、有料計算資源の契約は今回の範囲外。
現在は修正・実教師による短い学習/再開/export試験と実行監督の重要レビューを完了した。
初回runは327trajectory後に安全停止し、契約修正後の別run
`evaluator-20260910-main-r2` を起動したが、479trajectory後にswap増加上限で安全停止した。
全完了データとログを保持し、r2の479件を再ハッシュ・seed・split・件数照合したうえで、
同じキャンペーンの単一教師worker継続run `evaluator-20260910-main-r3` をsealした。
r3は479件を変更なしのhardlinkで再利用し、重みcheckpointなし・`prepared` から既存経路で起動した。
前回は開始直前のpageoutが3,134,456から3,135,181へ+725ページ増加したことだけで起動を保留したが、
今回の再開ゲートではこの単発deltaを単独の拒否条件にせず、実測page sizeを用いた短期の複合測定で再判定した。
起動前4点約30秒はpressure level=1、pageout/swapout delta=0、swap使用量delta=0、OSAIプロセスなし、
空き容量約292.5 GiBだった。教師1並列のcanaryで479→506 raw/receiptへ実増加し、pressure level=1、
pageout/swapout/swap使用量のcanary delta=0、OSAI owned RSS最大約1.06 GiB（teacher約0.95 GiB）で安定している。
現在はr3の教師処理を継続中で、重み更新・学習後評価はまだ始まっていない。
凍結W256と未開封最終holdoutを保持し、棋力向上・初段への安定勝ちは認定していない。
この文書を進捗と次工程の正本とする。詳細は小さな機械可読記録として
`local/runs/evaluator-20260910/` と既存 `local/core-prototype/` に保持する。

### 修正・本学習の実施記録

- 開始時の照合: OSAI `7e4b84c`、OSUI `852875e`、双方 `codex/core-prototype`。
  両PR #1はdraft・未merge、remote CI/reviewなし。既存Viteは5176で起動中。
  OSUIの時計・中断・途中局面復元修正は `a136e92` としてcommit/push済み。
- 長考の原因: 10分を残り30手で割り目標20秒、3倍の60秒を上限とする配分と、
  時計付き探索で安定候補の早期終了を無効にする経路を修正。
  高品質にも同じ時計・絶対停止を適用し、安定性に応じた早期終了を許可。
  Wasm同期探索中の共有atomic中断、host watchdog、古い応答拒否を実装。
- 実ブラウザー: 3/10分×標準/高品質×先後の8条件を凍結W256/OFFで確認。
  初期先手は3分0.870–0.887秒、10分3.732–3.860秒、7六歩後の後手は
  3分1.004–1.011秒、10分6.792–7.071秒。全てstable終了でhard cap内。
  初期3分hard 5.350秒、10分17.950秒、対局前準備174–230msを別計測。
  `diagnosis/browser-clock-baseline.json` に記録。短い8試験は棋力評価ではない。
  追加の実Workerブラウザー18ケース（残り0/50/200ms、先後/品質、1手詰、終盤、
  詰み/非王手合法手なし終局）も成功。途中SFENの開始手数を1と仮定するUI復元を修正。
  通常画面でも探索中停止→再開→人間着手→AI応手→投了→再対局を確認。
- 通常 `#/match` は旧handcrafted-onlyの標準snapshot、旧試作は3分eco固定だった。
  開発用 `#/core-prototype` に明示的な学習モデル・3/10分・品質選択を実装し、
  通常開発画面にも相違を表示。公開既定モデルは昇格していない。
  実棋譜は未取得。初期局面等6条件のON/OFF診断を実棋譜の再現とは扱わない。
- 玉移動・成り・打ち・持駒等を含む10局面と合法子300のnative/Wasm評価が完全一致。
  差分/全再計算一致。凍結評価器と教師の大きな判断差は残り、モデル弱さの改善対象。
  手作業の序盤規則や外部対局委託は導入していない。
- 主候補はW256/OSAVAL03を維持しscalar評価経路の全層を再学習。旧W256は凍結保存。
  旧制御器は着手変更確率を学んだだけで品質改善期待ではなく、新評価器ではOFF。
  新制御器を採用した/成功したとは扱わない。
- 実教師preflight-v2: 新規14trajectory、1,288 plies、2,563ユニーク局面
  (train2,086/validation284/development test193)、24更新・延べ2,982件・最大2露出。
  11更新で停止/再開し連続実行とparameter/RNG/order/exposureがbit一致。
  更新step24 export `866bc01b…0296` はnative/Wasm306局面差0・pure proof成功。
  小標本validationは悪化しbestは旧評価器のまま。強化とは認定せず、本学習へ進める。
- 本学習の固定設定は [`configs/evaluator-main.json`](../configs/evaluator-main.json)。
  新規2,048trajectory、最大192手、2 teacher workers、25,000 nodes。
  25万–70万ユニーク局面を見込み、実数を報告。最大8,192更新/8周、CPU4threads。
  計画・起動/再開/停止・保持・復旧条件は [development.md](development.md) に集約。
  初回run ID `evaluator-20260910-main`、保存先 `local/runs/evaluator-20260910/main`。
  `d8ca51f27b40ada81df7699b0177c20848712a5a` でseal済み、同commitをpush済み。
  run SHA-256 `60132bc849be461fcbd261ffe9ac23b0e631059ad5dda567a52cd3a5953c94ce`。
  既知診断と全preflight由来16,980個の対称局面hashを
  本runから除外する。凍結holdout原本は開いていない。
- 独立した重要実装/レビューは `gpt-6-astra` / `xhigh`。配布元調査は
  `gpt-5.6-luna` / `max` で実施。本runも同Lunaへ実引継ぎし、Lunaが起動済み。
  初回supervisor PID `96926`、generate stage PID/PGID `96928`。起動時の生存を確認。
  最初の正常進捗は2→22 trajectory、親の追加確認時40/2,048 trajectory。
  重複除去後ユニーク数・本runの延べ学習件数はまだ未確定（重み更新未開始）。
  15秒間隔の監督scriptがprogress/RSS/swap/容量を記録し、障害後の回収を完了した。
  supervisorのcode/config封印、子孫回収、二重起動、データ再照合の指摘は修正済み。
  未観測子を専用PGIDで回収してから親を終了処理し、監督消失後の残留も再起動前に検査。
  実教師2trajectory/16手の追加試験は終了後の残留processゼロ。
- 初回runの停止: 327trajectory（41,115手・標本root20,644）を保存後、
  `teacher bestmove is not a normal USI move` で `needs_astra`。重み更新はゼロ。
  原本に失敗時SFEN/raw応答がなく、fresh349/350は正常終了したため厳密再現は未達。
  構成した先後入玉宣言2局面では、固定教師の正規 `bestmove win` を同parserが
  拒否することを実確認した。証拠は `diagnosis/teacher-terminal-contract/`。
  今回generatorだけtyped終了を許可しnative CSA条件で検証する修正を完了。
  通常adapterの厳格契約、不正応答拒否、scalarラベルへの終端混入防止を維持する。
  新run `evaluator-20260910-main-r2` は全327件をhash照合したprefixから起動済み。
  関連Python60件・Rust入玉境界2件・Clippy/Ruff成功。先後実教師smokeは
  宣言終端各1件/通常scalar観測0件・4process残留0。更新診断器の306局面一致も成功。
  独立Astra/xhighレビューで阻害なし、旧85,494 observationの読取互換が完全一致。
  旧run/ログ/設定を保全、既知復旧診断も含む18,672対称hashを新除外ファイルに固定。
  規模・seed・学習率・採用閾値は不変、24h上限の起点も初回開始時刻を維持する。
  新code `4f684626464456be6b417d3e56ac8969355357fe` をcommit/push・seal済み。
  新run SHA-256 `4ac7ea429ed1ea6703a7d7867857bd8b876714f99345f2e639b98972e8911ca6`。
  Luna / maxが実起動し、新supervisor PID `24006`、stage PID `24009`。
  327→337 trajectoryの正常増加を確認。旧・新runの両lease下で654 raw/receipt
  ファイルをhash一致のhardlinkで引継ぎ、原本の書換え・ラベル再生成はゼロ。
  `main-r2/continuation-import.json` に実行script/入力/出力hashと元の期限を記録。
- r2の資源停止: 479trajectory保存後、global used swapが5.960GiBへ増加し、
  初回baseline 1,348,993,024 bytesから4.704GiB増で `swap_limit`。
  全479 raw/receiptのhash・元game/seed/splitを再照合し不一致0。60,209手、
  記録root30,229、教師特殊終端0。ユニーク数は重複除去前なので未確定。
  `diagnosis/resource-stop-preservation.json` に保全検査を保存。
  supervisor/全子は回収済み、leaseなし、重み更新ゼロ。停止時owned RSS 0.806GiB
  だけから原因が本run以外と断定しない。通常の学習/protocolエラーは追加されていない。
  停止後5秒の一時的なnormal/書出しゼロは、60秒の確認では再現しなかった。
  Luna/maxの60.201秒4点測定はpressure `[2,1,2,2]`、swapout +456,895 pages、
  pageout +516 pagesで不合格。`diagnosis/resource-recovery-samples.json` に保存。
  swapoutのbyte換算7,485,767,680は非圧縮ページ換算で、実ディスク書込量とは異なる。
  他appの内容やprocess帰属は調査せず、停止・メモリ強制解放も行っていない。
  条件付きr3案は独立Astra/xhighレビュー済み: 60秒以上・4点以上すべてpressure=1、
  各区間swapout/pageout増加ゼロ、逆行/測定失敗なし、開始直前にも再確認した場合のみ
  teacher workersを2→1へ減らす一度限りの復旧を認める。4GiB上限はその新資源区間
  の追加swap上限へ意味を変更するため、旧/新baselineと差を明記する。通算増加上限
  を維持したとは扱わない。RSS16GiB/空き80GiB/学習条件/採用基準/元24h期限は維持。
  60秒・4点のbounded測定は通過したため、元479件を再照合してr3継続manifestを作成した。
  ただし開始直前のpageout増加(+725 pages)で前回はr3を未起動のまま保持した。
  今回は単発pageoutを停止根拠にせず、実測page size 16,384 bytesで4点約30秒の再測定を実施。
  pressure level=1、pageout/swapout delta=0、swap使用量delta=0、OSAI起動前プロセス0、
  空き容量約292.5 GiBを確認したため、元のcampaign deadlineを維持してr3を一度だけ復旧した。
  起動後のteacher=1 canaryはsupervisor PID `2232`、generate stage PID `2234`、teacher PID `2248`。
  約30秒で479→506 raw/receipt、teacher RSS約995,440 KiB、owned RSS約1,137,442,816 bytes、
  pressure level=1、pageout/swapout/swap使用量delta=0。プロセス存在だけでなく実データ増加を確認し、
  通常監視へ移行した。r3のinitial swap baselineは`2,240,869,826` bytesで、元の24h期限は変更していない。
  r3で再度swap停止したら自動再基準化・再起動は禁止。
- 検証: Rust392件、Python928件、OSUI210件とnative/Wasmビルドが成功。
  最後の監督/USI修正には関連73件と実process回帰を追補し、Ruffと独立Astra/xhighレビュー済み。
  更新step24の再export照合と凍結pure smokeも成功。全体試験を変更なく反復していない。
- 継続: 既存heartbeat `openshogiai-main-r2-monitor` は同じautomation IDのまま、対象path/state/run IDを
  r3へ更新しACTIVE化した。r2は再起動せず、r2/r3の二重monitorも作成していない。
  通常は30分間隔の読み取り専用監視とし、stage遷移・実進捗・checkpoint・異常時だけ短く確認する。
  単発pageoutや既存swap非ゼロでは停止せず、pressure、継続swap/pageout、available/compressed/wired、
  OSAI RSS trend、CPU、disk freeを組み合わせる。圧迫中は他appを終了せず、OSAI自身の再現可能な持続増加時だけ安全停止してAstraへ戻す。
  旧mainは再起動しない。期限 `1789067056.291158` 到達時は延長せずcampaignを終了保存。
  常時監督は実script、Lunaは固定運用、Astraは重要な失敗・学習後判断と実ブラウザーを担当。
  変更なしでは通知せず、`awaiting_astra_browser` から候補identity/固定40対局を判断し、
  開発候補選択へ載せて8時計条件・18fixture・通常対局を再確認する。
  ブラウザーfixtureは `diagnosis/browser-clock-fixtures.html` とOSUIのignored
  `.playwright-mcp/clock-contract.html`、既知旧モデル実測は `diagnosis/browser-clock-baseline.json`。
  残工程は本学習、export監査、固定対局、Astra判断、候補の実ブラウザー再評価。
  コード/設定変更・再sealはせず、status/start/stopはdevelopment.mdの実コマンドを使う。
- 容量: 起動後空き約355GiB、監督対象RSS約1.39GiB、swap増加なし。
  比較証拠と再開側の資産を照合し、重複した連続実行preflight出力84,824,747 bytes
  を削除。削除前hash/参照/開放確認は `preflight-v2/cleanup-whole-preflight.json`。

以下は前回試作までの履歴。上記の現在runや採用判断と混同しない。

## main とブランチ

整理済み `3afecc2` と完全に同じtreeを内容統合した `3c91bab`、および閉鎖済みcampaignの
履歴を新mainの認証に使わない回帰試験 `8670880` を `origin/main` にpush済み。
古い未公開コミットに含まれた途中成果物や端末固有の履歴を新たに公開しないため、
private history全体のfast-forwardではなく内容統合を選んだ。
試作の基点は `8670880c692b3040b7974160ef42aa114d33aa36`。

`codex/phase10r-sparse-10m`、`codex/phase10v-leaf-evaluator`、
`codex/phase10v-sunday-execution`、`codex/pure-learned-pre-selfplay` のlocal枝を削除した。
すべて整理済み先端の祖先で、必要な最終treeのmain反映を確認してから削除した。
`codex/oss-minimal-freeze` は未公開履歴の唯一のlocal保存先として残した。
remote側には指定旧枝が存在しなかった。別worktreeや他人の枝は操作していない。
両repoにworkflow・main保護規則・公開連携は確認されず、remote CIは未設定。
試作は両repoの `origin/codex/core-prototype` へpush済み。
レビュー用draftは [OSAI PR #1](https://github.com/TKY-27/OpenShogiAI/pull/1) と
[OSUI PR #1](https://github.com/TKY-27/OpenShogiUI/pull/1)。engine実装commitは `db4fa8e`、
UI commitは `852875e`。両PRとも未merge、自動mergeなし、remote check/reviewは未設定・未提出。
独立ローカルレビューと検証の結果は以下に記録する。試作PRはmainへmergeしない。

## hard4 停止の確定原因と修正

保存された元requestを元binaryで実行すると `pure runtime proof failed`、exit 2を再現した。
最後の着手後は「王手ではないが合法手なし」。Gameがこれを終局に分類せず、探索は
評価呼出しゼロで戻り、正の学習評価回数を要求する監査に拒否されていた。
[CSA大会規則](https://www.computer-shogi.org/wcsc36/rule.pdf)の合法手なしの負けに従い、
`NoLegalMoves` を追加した。詰み、合法手なし、千日手、連続王手、評価前cancel/期限切れと
通常評価を `SearchOutcome` で区別する。通常局面のpure proofは引き続き正の学習評価を
要求し、終局を通すための無意味な評価、Handcrafted/教師fallbackは加えていない。

修正後、元requestはexit 0、`no_legal_moves`、着手なし、規則上の負け -30000、
探索評価ゼロで正常応答。同じprocessの非終端対照では学習評価6回を確認した。
Game/探索/proof/CSA/native/USI/Wasmの小さい回帰試験を追加した。
元の凍結binaryと今回の実行identityは別物で、旧400局成績を今回の成績に移していない。
元request・元binary・修正後binaryのhashは `local/core-prototype/diagnosis/` に記録した。
大規模Arenaは再実行していない。凍結manifestと未開封評価原本はそのまま保持している。

## 一つの新中核と学習

仮説は「次の深さで着手判断が変わる見込みを学習し、その見込みを候補への追加探索順序と
現在の一手に使う時間の両方へ結び付ける」。10特徴のlogistic predictorをRust探索へ組み込み、
各完了深さの評価差・変化・候補計算量・前回最善手などからroot候補を並べ替え、
対局全体の残り時間から得たsoft budgetを調整する。直近の深さごとの実計算時間から、
次の深さが配分内で終わるかも判断する。合法候補を保持し、危険な枝を学習で枝刈りしない。
終局、絶対上限、残り時計、stopは学習から独立して守る。局面の段階名で時間を固定しない。

近い既存方式は [Russell/Wefaldのvalue of computation](https://www.ijcai.org/Proceedings/89-1/Papers/053.pdf)
と [Stockfishの探索安定性に基づく時間管理](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)。
本試作は自エンジンの探索traceから学習した一つの判断変化推定をroot順序と時間配分で共有する。
それらの完全再現や世界初の主張ではない。深い探索の選択やalpha-betaの境界値は、
正解・詰み証明ではなく不確かな参照として扱う。

葉評価器は凍結W256 hard2のままで、追加学習していない。学習したのは思考制御だけ。
ON/OFFとも同じ共通探索・葉評価器を使う。identityの正本は
[`configs/core-prototype.json`](../configs/core-prototype.json)。

- W256 SHA-256: `859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480`
- 思考制御 SHA-256: `66110bae4ef5fbedd6a3c5537813f0fae5b9276b576a745b2364b4a093141b63`
- 自前生成の合法な対局trajectory 24本、1本最大8局面、取得182局面、重複除去後182。
  使用できる完了深さの組がない2局面を除き、学習123/開発validation32/開発test25局面。
  元trajectoryで16/4/4に事前分割、同一局面の跨ぎと凍結split hashとの重複はゼロ。
  同じ初期局面を共有する短いtrajectoryであり、広い独立棋譜集とは異なる。
- 13,028候補例、全batch更新300回、延べ候補例3,908,400回、延べ局面36,900回。
  checkpoint 1個、resume state 1個。別途preflightは修正前2局面・修正後2局面。
- M5/24GBで取得約46.68秒、学習約0.22秒。外部archive・教師取得、既存ラベル総再生成なし。
  入力・手番符号・cp尺度・詰み相当値除外を確認し、正規化と特徴設定に評価集合を使わない。
- 開発testのlog lossは学習モデル0.08339、単純な役割別学習頻度0.08299。
  最善手変更riskのBrierも0.31949対0.26358で劣る。全候補一律priorには勝るが、
  有用な自信推定や棋力向上を実証したとは言えない。学習不成立を周回数で隠さず、
  今回は一設定・一学習で停止する。評価失敗例を学習へ戻していない。

## 検証とローカル対局

- 全体検証は388 Rust tests成功。Pythonは初回879成功・2失敗で、標準Wasm再生成と
  閉鎖campaign履歴試験の整理後、失敗箇所を含む3 tests成功。format/lint、境界・権利・
  来歴・文書検査、native build、標準Wasmの決定的再生成、pure native/Wasm buildを実施。
  同じ全体検証は変更なく繰り返していない。
- 実WasmのNode検証で厳密なモデルhash、ON/OFF、合法手、root順序変更、pure proof、
  先後の残時計・加算、独立上限、終局、失敗したモデル交換によるcontroller失効を確認。
- OSUIはdev-only Worker/route/4資産allowlistを実装。公開既定モデル、overall_champion、
  既存標準binding snapshotを変更せず、production buildから試作routeと資産を除外する。
  重みはGitへ含めず、端末内だけで読み込む。
- OSUI全体checkの200 tests成功後、実ブラウザーで見つけた `stable` 終了理由の
  検証不整合を修正し、対象21 testsとtypecheckを確認した。

最小の実時間比較は同一native/葉評価器、各20秒切れ負け、先後交換2局のみ。
ONは先手133手・後手134手で2勝、無効局・未完・クラッシュ・期限超過は0。
子process CPU計68.495秒で、両局約0.981/0.995 CPU秒毎wall秒。
学習/ビルド/他のAI探索と同時実行していない。これは小標本の動作比較で、統計的優位や
棋力認定には使わない。凍結比較とは「同じW256を修正済み共通runtimeでOFF」の意味で、
旧binaryの成績ではない。実行前のrepo内symlink拒否1件は0局の事前失敗として別記録。

実ブラウザー（Codex内蔵、実Wasm Worker）ではモデル読込、先後、合法な応手、ON/OFF、
停止・残時計付き再開、再対局/制御切替後の古い応答抑止、投了を確認した。
3分切れ負けの開始から4手目の人間側時間切れまで進行し、0秒で負け・着手停止を確認。
ON先手の実消費は974/1276ms、学習目標4053.0/5104.4ms、予測risk 0.283/0.359。
OFF先手は770ms、学習判断・並べ替え0。別局面で配分が変わる実動作確認で、ブラウザー値を
公平な勝率比較としては扱わない。初回開発中のHMR旧state表示エラーとstable応答拒否は
保存し修正、完成後の確認ではconsole error/warnなし。
ブラウザーでの詰み終局、他browser/実機、長時間耐久は未実施。詰みはnativeの2局、
時計期限・加算/秒読み・cancel競合は高速回帰も併用して確認した。

独立Astra/xhighレビューは1回。子孫の合法手なし勝ちをpure USIが `score mate` と誤表示する
指摘を修正し、mate値だけから詰みを断定せず `info string rule_score` で報告する。
対象pure USI 7 tests、実USIの非終端局面から `4c5c` → `rule_score 29999`、pure再build成功。
この表示修正で比較用core_probe/ブラウザーWasmのbinary hashは変わっていない。

起動手順は [development](development.md#local-computation-control-prototype)。
`http://127.0.0.1:5176/#/core-prototype` で先後とON/OFFを選び、3分切れ負け・定跡offで対局する。
折りたたみには直前AI手の実消費時間、学習目標、予測risk、順序変更回数とasset hashを表示する。
既存の標準artifact用 `integration:ai` とこのpure dev経路は別契約であり、
古い標準snapshotを更新済みと報告しない。

## 保存容量と配布境界

作業前後の物理使用量（du、GiB換算）はOSAI 2.70 → 3.86、
OSUI 0.022 → 0.118。
OSAI localの増加は約9.2 MiB、主な増分はCargo検証用build出力と
OSUIの必要dependency。巨大データ取得やcheckpoint蓄積はしていない。
比較の重複trace/receiptを約1.89MBの検証済みgzipと約10.7KBの集計へ整理し、
同一controllerの余分なコピーを削除した。元の凍結資産、最良controllerとresume、
一つのcohort、ラベル/実測binary、圧縮証拠を保持する。容量記録は
`local/core-prototype/storage.json`、削除根拠は比較receipt内。

push前にソース・新規blobを検査し、標準Wasmに埋め込まれた端末パスをbuild時remapで除去した。
再生成hashと来歴を更新し、標準native/Wasm parity 13 testsも成功した。
重量物・私有資産・認証情報を含めず、ソースの試作branchとレビューPRだけを送信済み。
production buildからローカル試作とモデルendpointが除外されることを確認した。

## 承認境界

今回の修正・評価器の本学習・開発用UI再評価は承認済み。学習の再承認待ちには戻さない。
既定モデル昇格、試作PRのmain統合、本番公開・モデル配布は今回実行しない。
閉鎖済みPhase 10–10V campaignや初段勝率ゲートは統合の停止条件として復活させない。
