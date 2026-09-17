# 防御学習の現在状態と正式再開

対象は `defense-20260912-recovery-r3`、2026-09-17の明示承認に基づく
既存データ採用revision **`itemwise-focus-v1`**。元run/sealと生成receiptは不変。
現在の実行状態は同runの`state.json`、正式運用identityは`approved-operation.json`、
採用条件は`data/admission.json`、学習進捗は`fit/resume.json`から辿る。
**現在はattempt14、train / ready_for_luna / requested_pause。実更新14回まで完了し、
別プロセス再開で7回の増分を確認した。process/lease解放済み。Lunaはtrainから続ける。**

## 原因と今回の採用

attempt 11の最終例外は`finite generation coverage exhausted: coverage_exhausted`。
`stage_exit_1`は分類で、実際のcoverage未達は`focus_total`だけだった。
全12familyは266–272完走、各240要件を達成していた。
生成済み3,243 games / 181,519元行、accepted task398,524 / deferred task270、
未完了cursor21件。これらは別の単位で、元行数をunique学習局面数とは呼ばない。
難局面消費6,204試行 / 30,925.722506秒と残った保留証拠を維持する。

focusは62,220要求中61,957完了、欠測263（deviation132 / recovery131）。
99.5773%は任意教師信号の完了率で、棋力や正解率ではない。
旧欠測200上限・完了率・層別focusゲートは今回の対象runでは品質レポートへ移した。
新しい欠測件数上限を置かない。元の失敗・旧coverage・元snapshotは
`admission-evidence/`へhash付き保存し、単なる資源設定変更として扱わない。

学習は独立局面のvalue回帰。欠測focusと同局面の依存候補を入力作成前に除外し、
有効なroot・独立候補・ラベル済みfocusは保持する。欠測を0点/引分で埋めず、
不完全D12や必須root不在からvalueを作らない。mate/終局の既存除外、symmetry重複、
元split・family系列、r3全splitとの漏洩防止と未開封holdoutを維持する。

`data/dataset/manifest.json`は採用revisionのhash、split別unique件数・群別/信号別件数、
重複・除外理由、focus欠測のfamily/split/局面種別偏りを記録する。
欠測はattack/end群167、defense30、general40、opening26。
中盤128・終盤131・序盤4と難しい中終盤へ偏り、評価の限界として残る。

現在の生成・hard再解析予算は明示承認により消費済みとして閉じた。
正式resumeは再検証した既存成果の`generate-complete.json`を確認してprepare/trainへ進み、
追加gamesやqueue解消を要求しない。学習12,288更新・8巡/行8露出、教師・学習率・
sampling・r3比較・固定評価計画は変更しない。生成予算と学習/評価予算は別である。

## 確定データ

prepare manifest SHA-256: `67289ab62f78a1cb8f54af25367e27df729c7d3fc96cd6ad9bb573203a2dda42`。

| split | 新規unique | r3 replay | 有効unique合計 | focus欠測/要求（別単位） |
| --- | ---: | ---: | ---: | ---: |
| train | 183,486 | 300,000 | 483,486 | 84 / 19,626 |
| validation | 210,300 | 0 | 210,300 | 79 / 21,485 |
| development_test | 204,593 | 0 | 204,593 | 100 / 21,109 |

focus欠測263はすべて異なる局面で、既ラベルfocusとの重複0。
欠測に依存して除外した候補観測はtrain25 / validation19 / development23。
派生候補を含む777,446観測から、mate54,976・検証済終局1を学習valueから除外。
重複/除外観測124,090、cross-split競合key2,625、source cap除外0。
その他の防御・一般・攻撃・終盤を保持し、元のunique/split/群別最小要件も達成した。
詳細の教師信号別件数とfamily達成数はmanifestと`data/recovery-coverage.json`を参照。

## 実行と資源

学習コードはcommit `7e0db4e`（変更本体`9fec207`）で実学習前に固定。
CPU4thread、実効batch256、逐次microbatch32、勾配累積のsum lossを実効有効例数で割る。
worker/prefetchを追加せず、mmapを利用。
バッチ依存演算はなく、元batchとの勾配/AdamWは許容誤差内、同じ実行形態での再開は完全一致を試験した。実確保失敗だけ32→16→8→4→2→1へ有限半減し、
失敗した勾配を捨てて同じoptimizer更新を作り直す。欠測はこの前に除外済み。

各成功optimizer更新のparameter/optimizer/sampler/RNG/露出をcheckpointへ保存してから
次へ進む。hash付きimmutableファイル→resume参照の順で原子的に公開し、2状態を保持。
毎更新の保存はI/Oを増やすが、OOM後の成功更新の二重実行を防ぐ。
optimizer途中のOOMは直前の完全checkpointへ戻る。成功更新を保存できない場合、
書込失敗・破損・NaN/Inf・未知のexit1/SIGKILLは一般OOMへ変換しない。

全MacのRAM率・pressure・swap総量/増分・所有RSSだけでは実行を止めない。
pressureはDarwin dispatch flag（1=normal,2=warning,4=critical）で、値の大小だけで推測しない。
遅い進行を旧30分timeoutで打ち切らない。空きdisk60GiBの書込み余裕は維持する。
実OOMは専用終了code76と証拠を保存し、supervisor内の`resource_wait`へ移る。
同じcheckpointの自動process再試行は最大3回、attemptやresumeで予算を戻さない。
失敗から最低60秒後、空き指標5ポイント改善またはpressure正常化とdisk余裕を確認して復帰。
計測不能は待機を継続する。再試行枯渇は待機を維持し、無限再実行しない。
ユーザーSTOPは自動解除しない。暦日期限なしの`finite-work-v1`は継続。
OS設定・swapファイル・他アプリは変更しない。

prepare中の停止では、旧非協調処理がsupervisorにSIGTERMされる経路も実測した（attempt12）。
失敗原本を残し、supervisorによる停止と残留processなしを確認できたprepareのみ正式復旧する。
trainの未知の異常停止や無関係なneeds_astraは解除しない。

## 実学習・別プロセス再開の確認（PASS）

通常resume、stdin=/dev/nullの別CLIプロセスを2回使った。probe/stubは使っていない。

| attempt | optimizer step | sampler offset / 露出 | 最新batch loss | 終了 |
| --- | ---: | ---: | ---: | --- |
| 13 | 0 → 7 | 0 → 1,792 | 1.2004268095 | 正常pause |
| 14 | 7 → 14（+7） | 1,792 → 3,584 | 1.1605539173 | 正常pause |

全6パラメータ群のAdamW step=14、parameter有限、全split target有限。
再開前後のorder/RNG一致、countsは前回offsetからの続きだけ増加、最大露出1。
checkpoint identity、parametersの実変化、optimizer進捗と保存を実ファイルで確認した。
この14更新は本runの上限に含まれ、次回はstep14の続きから進む。
初期r3 validationの有効母数210,300とfocus欠測79/21,485を
`local/admission-evidence/validation-quality.json`へ併記した。
候補の固定評価/選抜/export/auditはまだ実行せず、棋力改善は主張しない。

checkpoint:
`local/runs/defense-20260912/recovery-r3/fit/checkpoint-000014-84cbbf8d9a6cae0af1c198978e645b845bc5bb5121a7e15885d0ddeeb04c101a.pt`

SHA-256: `84cbbf8d9a6cae0af1c198978e645b845bc5bb5121a7e15885d0ddeeb04c101a`。
現在の参照は同所`resume.json`。

運用revision SHA-256: `10d72b25886e63da2934a6a9ac8699d6b12c61e11c9da8651f30acabb6af5da6`。
採用revision SHA-256: `6ad2e014e408e7af29fe66ebf2ff3a95dfdd38d1679000959bbc05d129f83b2f`。
学習code commit: `7e0db4e6f92277e1107849f9efdac0bda69c4d1c`。
後続の運転文書・テストだけのcommitは学習code identityを変えない。

元snapshotとattempt14-pause snapshotは完全一致し、生成receipt・元行・task・cursor・
消費予算・ledger内容は変わっていない。元run/seal hashも一致。
最終statusはsupervisor/stage非生存、残留groupなし、lease非保持。
実OOMは誘発していない。圧迫値・確保失敗・checkpoint復帰は模擬テスト、実runは通常資源下。
証拠 `local/admission-evidence/final-verification.json` SHA-256:
`0181fc522fbc1d5abb01993e42fb6af0943479a791d12ca164e7cbacd959325f`。

## Lunaの正式操作

cwdはこのOpenShogiAI Gitルート。別の新規Luna Max/maxセッションが以下を実行する。
モデル設定やJSON手編集、追加生成、再sealは不要。stdinは`/dev/null`。

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run resume local/runs/defense-20260912/recovery-r3 </dev/null
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/defense-20260912/recovery-r3 </dev/null
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run pause local/runs/defense-20260912/recovery-r3 </dev/null
```

正式resumeはlease、元seal、承認code、全receipt、SQLiteとcursor、固定済みmanifest、
checkpoint hashを検証する。Lunaは本学習・候補選抜・固定評価・export/auditを継続し、
終了時は`awaiting_astra_review`へ返す。Astraの短い確認更新も12,288上限に算入する。
定跡なし、防御重点と攻撃維持、開発用新旧選択、本番一モデル・未公開を維持。
Luna子起動、OSUI変更、main merge、force-push、重み公開、deploy、有料計算は行わない。

## 検証

最終コードの関連232 tests、ruff、
boundary/license/provenance/docs検査PASS。
全体Python検証は1,093 PASS / 22 FAIL / 23 ERROR。
失敗はXcodeライセンス未同意によるビルド/開発ツール起動の環境阻害。
`make check`自体も同理由で起動不可。ライセンス承認や権限変更は行っていない。
本runは封印済みnative/Wasm runtimeをhash照合して使用する。全体build PASSとは報告しない。
検証ログ・短い実行証拠は`local/admission-evidence/`に保持する。
