# 現在の状態と次工程

2026-09-12、序盤・受け・反撃の本学習契約をAstraが準備中。
今回のrun IDは **`defense-20260912-main-r1`**。r3とは別runで、本学習は未起動。
学習は承認済みで、再承認は不要。短い経路・停止再開・export確認と重要レビュー後に
`ready_for_luna`へ固定する。正本はこの文書一つとし、過去の工程記録はGit履歴に残す。

唯一の現行機械契約は [configs/evaluator-main.json](../configs/evaluator-main.json)。
seal後の実行正本は `local/runs/defense-20260912/main/run.json`、状態は同じ場所の
`state.json`、code commitとrun hashは`seal.json`。名前から古いr3へ切り替えない。
`awaiting_astra_review`なら再準備・再学習せず、下記のAstraレビュー工程へ進む。

## 固定方針と分担

開発対戦・評価・将来の本番すべて定跡なし。初手固定、局面と正解手の表、
過去解析の対局へのプリロード、対局中の外部教師委託は使わない。通常探索のTTは許容。
序盤の教師判断や訓練シナリオは重みを学ばせるofflineデータに限る。
非終端のpure評価には手作業の点を混ぜず、玉移動・桂跳ね・特定手を一律に禁止しない。

Astraは原因解析・設計・評価器／探索／教師／学習設定・重要レビュー・採用判断・OSUI統合。
ユーザーが別途起動するLuna Max/maxセッションは固定済み取得／生成・学習・定型評価・
監視・事前定義の復旧を実行する。Luna子エージェントやAstraの長時間監視は使用しない。
準備終端は`ready_for_luna`、実行終端は`awaiting_astra_review`。
本番公開、重み配布、公開既定昇格、main merge、有料計算資源契約は行わない。

## 確認した原因と変更

ユーザーの棋譜は未提供のため、合法な自作棒銀系列22plyを診断した。
ユーザー実戦の厳密再現ではなく、既知の開発用回帰である。
11根局面・422合法子でr3の差分accumulator／全再計算と出力は完全一致。
玉移動・持ち駒・銀交換の実装不整合を示す証拠はなかった。

主探索にはLMR／futility／null-move pruningはなく、完了深さで全root候補を列挙する。
非王手の静止探索はcapture／promotionのみなので、深さ外の静かな受けは学習評価に依存する。
ある局面は20k nodesで角交換、100kで玉移動となった。各子の同じ深さ4・全窓の自前探索では
角交換-9／玉移動+3cp、独立した教師深さ20の子再評価では+395／-394cpだった。
候補欠落より評価と有限の探索範囲に問題があるとの診断を支持するが、
教師も深さ24を完了できない局面があり、絶対正解や初段棋力とは扱わない。
root/PV/探索量の小型JSON観測を追加した。探索アルゴリズム・時計は変更しない。

教師Apery 2.0.0はnode打切り時に未完root値を完了深さ名義で再出力する場合があり、
旧node-onlyラベルには未完MultiPVが混ざる可能性を実stdoutと教師ソースで確認した。
新経路は1thread、毎局面Clear_Hash、完了深さ12（重点16）、rank／深さ／node上限／bestmoveの
厳密照合を要求する。深さ未達だけ2M→同じ深さ32Mへ1回復旧し、拒否されたstdoutを保持。
rank不整合・不正手・hash不一致・非数値等は再試行で隠さず停止する。
最初の短preflightは8/12系列で8M復旧上限に達し正しく停止。問題局面は深さ16を
9,766,294 nodes／8.273秒、教師RSS約0.948GiBで完了したため、深さを下げず32Mを固定した。
次の試験では別の回復局面が32Mでも深さ16未完となった。任意の重点D16だけ、
同深さ32Mでも未完なら型付き欠測として保持し、基礎D12への代入や同record候補の流用はしない。
重点完了率95%以上かつ欠測200件以下を生成・prepareで検査し、超過は`needs_astra`。
群／先後／段階／branch別の欠測偏りも保存する。必須root D12とその他の不正は停止のまま。
診断と3回のpreflightで観測した3,538 SFEN、旧除外を含む32,636対称キーを
`local/runs/defense-20260912/exclusions.json`へ統合して契約にhash-bindした。
新runのtrain／validation／development testから除き、独立な試験例とは数えない。

診断証拠は `local/runs/defense-20260912/diagnosis/diagnosis-summary.json` と
`recovery-cap32m.json`。初回失敗は `local/runs/defense-20260912/preflight/` に保持。

## 学習・評価契約

主構成は **r3 OSAVAL03 W256の全層継続学習**、scalar smooth L1（cp/600）、
Q20／f64の厳密accumulator、CPU4thread。制御器はOFF。
W256再学習・制御器OFFを新機構の成功とは呼ばない。

一般30%／序盤20%／防御30%／攻撃・終盤20%を一巡内重複なしでsampling。
一般replayは旧r3 trainのみ最大300,000行で、古いラベル品質の限界を持つ忘却防止用途。
r3 validation／development testをtrainへ戻さず、旧全splitと新生成の対称重複を除く。
12の事前定義scenario familyから最大3,072 trajectory、各192ply。先後反転と確率的派生は
同family／同splitに固定し、12 familyを3,072独立人間棋譜とは数えない。
分類は局面整理のためのもので加点減点ではない。棒銀以外の飛車圧力・歩交換・角交換、
接触前から突破後まで、実r3探索候補の再解析と教師応手後の回復を含める。
通常根／候補子／誤候補／回復、ユニーク局面／trajectory／延べ露出を別々に報告する。

目標は約40万～120万ユニーク局面。fit前にtrain40万以上、新規train15万以上、
各train群1万以上、validation／developmentの各群128以上を実manifestで検査する。
最大12,288 updates／8巡／1行8露出、batch256、LR0.0001→0.00001、warmup128。
validation256updates毎、patience6、相対改善0.002。一般／攻撃validation lossがr3の1.03倍を
超えるcheckpointは最良候補にしない。best export1個と直近2個の整合するresume stateを保持。
optimizer／乱数／sampler順序・offset・露出／best bytesを一緒に復元する。

学習指標で選んだ **1候補だけ** をr3と比較する。未知64根の100k-node move screenは
4群16系列ずつ、同じ強さの教師による候補子再解析で妥当手の幅と400cp重大誤りを測る。
既に負けている根と、選択手で詰み負けにしたケースを分ける。既知診断11根は別集計。
序盤12～40ply／防御16～96／一般32～128／攻撃終盤80～191を事前固定する。

定跡なし実時間対局は指定局面から32局（3分24／10分8）、先後交換、同一runtime、
depth64／qdepth4／TT2MiB／1thread／制御OFF。4群から計16の異なるtrajectoryを採るが、
信頼区間は共通family単位で計算する。完全初期局面からは時計×先後の4局だけを別記録し、
同一の決定的対局を反復して独立証拠にしない。打切り引分、無効局、時間切れ、失敗attemptを
保存し、成功した時点の早期打切りはしない。詳細採用条件は機械契約に固定。
未知scalar群／move screen／r3対局の条件を合わせてAstraが判断する。
人間初段への安定勝ちは別の実戦確認が必要で、旧32勝を新成績に流用しない。

入力は既存の教師・権利／split metadata・r3正本と新しい自前生成のみ。
追加公開archiveの取得は不要。教師のバイナリ／別配布評価資産／ライセンス／hashを保存し、
最終holdoutは未開封。資源上限は72時間の有限実行枠（終了予告ではない）、
空き60GiB／所有RSS12GiB／swap増分2GiB、stalled30分。監督は実scriptが担当する。
再開時も元の期限・資源基準・retry数を維持し、資源停止の自動再基準化はしない。

## Git・容量・検証

開始時はOSAI `d70e98d`、OSUI `41ec5a6`、双方`codex/core-prototype`でclean。
両remoteをfetch/pruneし、draft PR #1が未merge・auto-mergeなし・remote CI未設定と確認。
使用中worktreeは各1つ、学習／教師processなし。対象の旧remote枝は存在しなかった。

残存したlocal `codex/oss-minimal-freeze` のtreeはmainへ内容統合した`3c91bab`と完全一致。
必要source `0203a847…`へのlocal tag `baseline-source-osaval03-20260912`を残して枝を削除した。
現行枝／main／PRは保持。hard4の不要重みと旧binary、旧preflight再開checkpoint等を
合計94,663,622 bytes削除。r3 best／旧hard2／再利用データと再開資産／失敗記録／未開封原本は保持。
実path・symlink／mount・open filesとhash、削除理由は
`local/maintenance/defense-20260912/cleanup.json`に記録した。

OSUIは定跡読込・復元・手順選択とBrowserPlayの永続解析preloadを削除し、Workerでも拒否。
新しい4 bindingsを一括同期し、実Chromeで見つかった追加statsのparser不一致を修正。
233 testsを含む全checkと実Chromeの通常探索／再対局、390px幅、IndexedDB open0を確認。
r3 matchのidentityは同じ、3分標準先手の新snapshotで初手待ち527msとstop時計保持を実確認。
本番は`release-model.json: model=null`の未採用を維持し、一時ローカル指定で一構成buildを検査。
新重みのUI追加、全3/10分・標準/高品質・先後の本確認は学習後のAstra工程。

短試験v3は12系列完了、D16は239/240件（99.583%）、欠測は攻撃群・先手・中盤の回復1件。
train1,092（新836＋r3 replay256）、validation807、development922で実r3を8更新。
一括とstep3→STOP→再開のparameter／optimizer／RNG／sampler／offset／露出／bestが一致。
native/Wasmは20根・769子でcp/WDL差0。145.4秒は旧9完了rawの検証再利用を含む
残3系列生成＋prepare＋短学習等の所要時間で、12系列を一から生成した速度ではない。
自己RSS最大0.672GiB、子0.949GiB。親のstaging削除安全helper追加だけ途中source差があり、
該当削除分岐は未実行。学習・教師・sampling sourceは一致し、差を専用receiptに保持。
証拠は`preflight-v3/final-verification.json`と`resume-equivalence.json`／`export-audit.json`。
使用終了したcheckpoint3個と重複exportを122,936,201 bytes削除、step8再現用一組と
失敗・入力・照合receiptを保持。冒頭と合わせ217,599,823 bytes（約207.5MiB）削減。
本学習と同じbatch256／CPU4threadでも1更新＋小validationを1.008秒、RSS0.744GiBで確認。
この速度を大規模dataset全体へ外挿しない。batch試験の重複重み等67,484,400 bytesも削除し、
今回の削除総量は285,084,223 bytes（約271.9MiB）、終了時の空きは約222GiB。
最終`make check`はPython987件、Rust、format/lint、権利・境界・provenance、native build、
決定的Wasm再生成照合がPASS。最後のstatus検査追加後もrunner44件とlintがPASS。
`make pure-build`／`make frozen-smoke`もPASS。commit/push/sealは確定後に記録する。

## Lunaへの引継ぎと学習後のAstra工程

準備完了後は契約の`operations.start/status/stop/resume`と[development](development.md)を使う。
コード・学習設定・Git枝を実行中に変えず、運転結果だけを更新する。
`needs_astra`は事前定義外の復旧をせず戻す。全工程完了は`awaiting_astra_review`。

同じ依頼が再送されたら今回のrun identityとstateを確認し、Lunaの実manifest、
checkpoint選抜、未知群、r3対局、独立性、攻撃退行をレビューする。未達は未達と報告し再訓練しない。
最良候補を開発OSUIへ追加してr3比較を残し、本番一構成・未昇格・定跡なしを保持。
実ブラウザーでモデルhash／序盤／棒銀診断／通常対局／3分10分／両品質／先後／stop／再対局／終局を確認する。
旧r3の開発32勝は旧hard2相手・同run development start由来で、初段や独立外部棋力の証明ではない。
r3 bestはstep6144、SHA `cd07f2a202f6e781afcb6a8af3c7a198c4203373e4505fe089f8c7f05eefd983`。
旧比較hard2はSHA `859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480`。
