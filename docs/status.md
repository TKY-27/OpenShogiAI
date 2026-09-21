# R4-C4 — 本runの実経路確認中

人間向け正本はこの文書、機械向け正本は `configs/evaluator-main.json`。
2026-09-21の明示承認によりC4を構築。Astraは実装と短い本run prefixまで、
長時間の生成・学習・比較はユーザーが別途開くLuna Max/maxが同じrunを継続する。
内部世代ごとの再承認・設計変更は不要。C5、公開、外部preview、重み配布、
公開既定昇格、main統合、有料資源契約は別GO。初段到達・ユーザーへの安定勝ちは未確認。

## 引き継ぐ実体と診断

C3 `local/runs/r4-c3/attempt-01` は実際には完了済み。旧status文書の準備状態は古く、
state/resultを優先した。code `d6b0bac43bd5d5ed950d68abe0f8b6744ae2dc06`、
16 epochs / step10850 / exposures2,776,376 / seen1,532,482。
C4初期重み・最初の生成元はC3 trained_candidate step9216、
`839a9f93398d2c989ca823ed01d88dc1c32e706baa32d5c61b17296114b9b262`。
この候補時点のexposuresは2,358,590。validation best
`09c9ba641548c2cb536cc39c7fcf000deaec8c31ade35d2714d32a6030188a98`
とは区別する。対防御18勝14敗、one-sided lower0.4464は固定採用条件未達。

防御best1536
`8c1c875038b74dc475c356d50c635c2e22dcab7aaa201d7d9aea7188e668b35a`
とC3を不変の主要対照として保護する。C3を学習の出発点に使うことは公開採用ではない。
既存C3の約153万実見は主に再利用で、シリーズへの新規追加とは数えない。
旧W256/r3/防御/C1/C3、C2実験とresume資産も保持する。同一重み・runtimeのC2は防御の別名。

既存24件の `local/r4-c3-preparation/search-diagnosis.json` とC3学習履歴を再利用。
新しい短い診断は `local/r4-c4-preparation/diagnosis-before.json` / `diagnosis-after.json`。
ユーザー棋譜の新規取得はなく、自作の合法な角交換・棒銀・中飛車系列を使用した。
750ms→3000msで深さが増えても同じ候補を選ぶ例があり、追加探索だけで誤評価が解消する
証拠はない。同じrootと直接子を新たにsingle-PV D12で比べた棒銀例では、C3は
9g9fを750/3000msとも選び、静的なroot視点も9g9f=+51、8h5e=-64と前者を好む。
同条件の直接子教師はそれぞれ-917/-115cpで逆順だった。差分/全計算は一致したため、
この例は符号・差分の破損より、受けを誤って好む評価の学習課題を支持する。
中飛車・角交換にも83–105cpの候補差があり、深さの異なるroot値を子へ流用しない。
教師探索はMultiPV数でも変わる推定値であり、旧MultiPV3と新single-PV値を混ぜない。
根拠は `local/r4-c4-preparation/child-diagnosis/report.json`。深ければ必ず正解とは扱わない。
現在探索にLMR/null/futility reductionはなく、削減率を下げる実験は不要。
C3で調べた手番・root/leaf・持駒/成り/打ち・差分更新を再総監査せず、
既存の合法手・差分・native/Wasm・終局・完了反復/取消の回帰を維持する。

序盤の遅さには実測で探索時計の原因があった。10分初期局面は候補2g2fが安定後も
奇数/偶数深さの費用差を過小予測し、深さ7まで約9.46秒使った。
直近2反復の増加率で次反復を予測する修正後は同じ2g2fを深さ6・約1.11秒で確定。
3分は約0.30→0.34秒。これは少数のnative計測で、全局面の高速化や棋力向上の証明ではない。
絶対期限、通常の詰み証明、最後の完了反復、合法な停止結果は変えない。
初手固定・序盤一律短時間・手作業評価・低ノード化は導入しない。
ブラウザーの取得/初期化/Worker準備と探索時間は別々に既存診断へ記録する。

ponderは今回OFF。既存ブラウザーは共有atomic取消で同期Wasmを本当に止められるが、
相手番から次手番へ探索器/TTと時計を引き継ぐ公開経路がなく、play開始で探索器を作る。
探索再利用関数があるだけではhit/miss・応答世代・時計の正しさを保証しない。
初手の実測遅延は時計修正で改善し、追加状態管理の費用に対する利益は未測定なので導入しない。
主要学習・評価・OSUIはponder OFF、対局中外部教師/思考サーバー利用0。

## 一つの有限計画

| 工程 | 固定上限・分岐 |
|---|---|
| 本run prefix | 新規16局、64手上限、D8広域/D12重要2root、128更新上限。実更新後の途中からLunaが再開 |
| 内部世代 | G01–G03を実行。前世代の新規有効trainが100,000以上かつ採用または連続不採用3未満ならG04/G05へ。最大5 |
| 生成 | 各8,192局・192手上限、累計24,576–40,960局。各世代のseed/生成元を固定 |
| 相手 | 半分は生成元同士、1/4はC3/防御を交互、1/4はオフラインApery（欠測時の自前着手は明示記録） |
| 開始 | 初期局面＋合法自作14系列、先後反転。相居飛車・棒銀・三間/四間/中飛車・嬉野流・端・角交換。runtime定跡ではない |
| 分岐 | 訓練だけ8手ごと96手未満。AB候補を同じ1024nodesの子探索で比較、最良との差80cp以内からseed選択。訪問数をpolicy教師にしない |
| 広域解析 | 2手ごと、同一Aperyの完全D8・200,000nodes上限 |
| 強い解析 | 食い違い・価値変化・層別ランダム、各局最大8root（32手未満2、32–95手4、96手以後2を予約）。完全D12・2M nodes上限、現候補/教師候補/別候補の直接子評価と2手の応手継続 |
| task有限性 | 各20秒、interruptionを含む累積2attemptまで。深さ未達/timeoutは局所missing。worker異常終了は累積3件後の次の異常でsystemic停止 |
| 学習 | OSAVAL03/W256全層、AdamW LR6e-6→6e-7、warmup128、batch256、micro32→16→8→4→2→1、weight_decay0.01 |
| 更新上限 | 世代あたり24 epochs / 24,576更新、検証1024更新ごと、patience6、min_delta0.2%。全体上限123,008更新（prefix込み） |
| 停滞 | 2世代不採用後はLR半分、replay比率を増加。旧生成元から次の新経験で学習。最低3世代後の3連続不採用なら追加世代を行わず固定比較/登録まで進む |

盤面目安は生成3–6M、既知祖先と重複除去した有効新規label1.5–3.5M、
重要root上限196,608–327,680。実数はmanifest、raw行数はtrajectory、
実際に学習へ入った数はcheckpoint countsで区別する。達成件数だけを成功・停止条件にしない。
終局/任意192手打切り/教師missingを区別し、打切りは `max_plies_unscored`。
勝敗を全手の正解ラベルにせず、root値を全子へ複製しない。mate/bound/未完了はcp回帰に入れない。

既存[出典監査](source-audits/hao.md)のApery、Hao、r3 replayを使用し、新規外部取得は0。
dlshogi/GCTの独立配布は派生利用条件未確認のため採用0。出典数を増やすことを新経験と呼ばない。
主仮説はC3からの価値蒸留＋同一教師・同条件の直接子間の順位損失0.25。
cp/600 SmoothL1、50cp以内の僅差は順位を要求せず、marginは600cpで上限。
ExItの「探索で得た判断を次の評価器へ学習する」という発想は参照するが、
[原論文](https://arxiv.org/abs/1705.08439)と同じ成果・MCTS相当を主張しない。

各世代に旧replay最大320,000、過去世代の新規由来最大120,000をhash標本で混ぜる。
通常の抽出枠は旧replay20%・現世代D8 50%・D12 25%・過去5%。
2世代停滞後は35/35/25/5%。枠不足は水増しせず、実比率・実損失寄与を記録する。
各枠内general/opening/defense/attack_endは20/40/20/20。
群は開始系列由来の代理分類で、全手を精密な戦術分類したとは扱わない。
旧例8/現例16露出まで、元系列8,192露出・batch内4、各epoch524,288例まで。
Hao等の旧出典別、再利用/再label/既知範囲新規、系列・群別の実見とexposuresを保存する。
「1M×10巡」は1M局面・10M exposuresであり、10M新規とは呼ばない。

開始系列のsplitは生成前固定、同一系列と先後/対称変換を移動させない。
C3 manifest・旧guardのhash・以前のC4観測（未label含む）をSQLite所有索引で照合する。
異split衝突を除外、同split再labelは新規と数えない。過去の未記録履歴は未知と明記。
既存最終holdoutは未開封。最終候補固定後にのみ、別の自作4系列から128本を生成し、
既知盤面を除いた32rootを新しい最終確認に使用する。系列数4の統計的限界は残る。
この結果を見て再選抜/再学習しない。不足はmissingとして開発比較と登録を続ける。

## 選抜と固定比較

incumbent、generation actor、更新trained_candidate、validation best、公開採用を分離。
prefix更新はG01の初期学習に継承するが、最初の生成元は指定のC3。
世代間でvalidation損失の数値を直接ランキングしない。
更新後候補は同世代objectiveで少数保存し、tensor/量子化本体の差・native/Wasmを検査。
全体best0でも別の更新候補を比較する。旧重みをC4と改名しない。

各世代は現在actorに対し固定32評価局（3分24/10分8）、別枠初期局面4局。
同じ新runtime、全体時計、ponder OFF、先後ペア、訓練とは時間を分離。
内部actor更新は全予定完了・異常0・各時計score>=0.5・平均>=0.55・系列bootstrap
片側95%下限>=0.45・全scalar群の損失1.03倍以内。これは公開採用条件ではない。
不採用/正常early stop/任意screen欠測でも次世代・成果保存を進める。
最終選択は最後の内部採用候補、採用0なら最後の実更新候補を比較用として残す。
全世代の候補・incumbent・最新resumeを保護し、選択を固定する。

最終はC3、防御それぞれ64評価局（3分48/10分16）＋初期局面4局。
新規最終root不足時だけ固定開発32局＋4局へ戻し、新規最終未確認と明記する。
先後/同系列/決定的重複を独立標本と水増ししない。正式引分は0.5、手数打切りは未採点。
不正応手/時計異常/プロセス異常/未完了を別に記録。勝率を見て局数を追加しない。
領域scalar/強い応手の順位/過大評価、時計・詰み・合法性を保存するが、
自己対局Eloや教師一致率だけで初段とは宣言しない。ユーザー対局待ちで工程は止めない。

## 実行と停止

実在Git root: `OpenShogiAI`。対応UIは隣接した独立Git root `OpenShogiUI`。
封印には絶対execution_cwd、code、教師binary/eval、入力manifest、runtime、UIファイルを保存する。
機械向け計画のoperationsが実コマンド。Lunaはsealや旧データ生成をやり直さず、次を実行する。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/r4-c4/attempt-01
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/r4-c4/attempt-01 </dev/null
```

状態確認は同じstatus。ユーザーが停止を求めたら:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run pause local/runs/r4-c4/attempt-01 </dev/null
```

既存lease/resume/ledgerを利用し、run/世代/cwd/開始時刻/入出力を照合して二重起動を防ぐ。
全optimizer更新をcheckpoint化、乱数・sampler・scheduler相当step・累積countsを復元する。
生成は各着手をSQLiteへcommit、shard→receipt中断もcursorから復元する。
同じtask失敗のattemptを起動でゼロにしない。全件missingゼロは工程gateにしない。
圧迫・swap・旧baseline差は診断だけ。実OOMのみmicrobatch縮小、有限再試行、資源待ち。
ディスク20GiB未満、改竄・漏洩・非有限値・保存失敗・systemic障害は範囲を保護して停止する。
カレンダー期限は適用しない。sleep/wake/修正時間を計算予算にしない。

Lunaが最後に行うexport→ignored開発registry→Wasm/profile照合→loopback起動→実ブラウザーQA
はrunnerの一部。変更先はC4 descriptorとignored証拠だけ、UIのコード判断は不要。
R4-C4は比較候補・未採用と表示し、表示順は正確に「最新←→開発初期」。
一構成だけロード、hash/取得bytes/Worker/Wasm identity照合、対局中切替禁止、古い応答破棄を維持。
3分/10分、標準/高品質、PC/390px、先後、stop/再対局、棋譜・診断保存を既存検証器で確認する。
解析モードの大規模改修は本学習後。最終 `awaiting_astra_review` は互換ラベルであり、
`development/result.json` PASSならユーザー対局待ち。統合だけのAstra再起動は不要。

## 計測・検証証拠

短い本runの準備測定: 192手上限の2局で219観測盤面、214有効scalar行、16重要root、
14.74秒。教師部分はD8 110task/1.80秒、D12 106task/1.16秒、残りは生成/再生/保存等。
別の64手1局は54盤面/37行/4.80秒。件数は重複除去前で本学習には混ぜない。
旧索引作成＋2局prepareは27.7秒、新規train126/validation50、train全8,324行。
小標本からの直列外挿は3–5世代の生成/再解析で約2–4日だが、局面の難しさ・新しい候補・
ホスト負荷を含め2–8日程度の幅を見る。新しいD12の第2候補追加分を含む本prefixで再計測する。
学習・比較は別加算し、未測定を完了日時にしない。

本run prefix、別プロセスresume、正式post-training rehearsal、ブラウザー、最終Git identityは
実行後にこの節へ追記する。それまではreadyとは扱わない。
