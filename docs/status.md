# 現在の状態と次工程

小規模の学習済み思考制御を `codex/core-prototype` で実装し、隣接OpenShogiUIの
同名ブランチへローカル開発用として統合した。ユーザーが実対局で評価する段階であり、
棋力向上・初段への安定勝ち・研究上の新規性は認定していない。
この文書を進捗と次工程の正本とする。詳細は小さな機械可読記録として
`local/core-prototype/` に保持する。

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
試作PRはmainへmergeしない。

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
重量物・私有資産・認証情報を含めず、ソースの試作branchとレビューPRだけを送る。
production buildからローカル試作とモデルendpointが除外されることを確認した。

## 次の明示承認

ユーザーがこの試作を実際に対局して評価する。弱くても動作確認の範囲を超えて強化認定しない。
大規模学習、W256/W512の反復大型化、既定モデル昇格、試作PRのmain統合、本番公開・
モデル配布は実行しない。有望と判断されても自己承認で次工程へ進まない。
閉鎖済みPhase 10–10V campaignや初段勝率ゲートは統合の停止条件として復活させない。
