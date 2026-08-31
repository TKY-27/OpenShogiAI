# Phase 10R Model Selection Report

Status: formally selected exactly one architecture. Selection is closed for this Goal; no post-selection large-scale training was started.

## Decision

Selected: sparse-pair-policy-wdl.

Rejected: factorized-pair-triple-policy-score. It passed mandatory correctness gates, but lost the direct equal-control head-to-head, ran materially slower, and was larger. Its offline policy/ranking and score-correlation advantages do not override the earlier deterministic criteria.

Decision order: mandatory correctness/tactical gates, aggregate equal-wall-clock score versus handcrafted-experimental, direct candidate head-to-head, robustness, throughput/latency, then offline metrics/model size. The direct head-to-head was conclusive, so the two-additional-seed escalation was not triggered and no Sol review was used.

| Candidate | Aggregate vs handcrafted (1,400 games) | Direct 1 s head-to-head | Selection |
| --- | ---: | ---: | --- |
| sparse-pair-policy-wdl | 81-0-1319; 0.05785714286 | 293-5-92; 0.7576923077; paired [0.7173913043, 0.7971576227] | selected |
| factorized-pair-triple-policy-score | 78-0-1322; 0.05571428571 | reverse result: 92-5-293; 0.2423076923 | rejected |

## Scope, frozen state, and implementation

- Repository: OpenShogiAI; branch: codex/phase10r-execution.
- Expected initial HEAD: b9514593ce2bf9f30626853d4572f70e32684100; initial worktree was verified clean.
- Runtime measurement HEAD: b8fa5c2ca26bb7ab2ad6789468585bb7f0764cb8.
- Frozen controls manifest: configs/phase10r/frozen-controls.sha256 (SHA-256 6231465fdb8feb1355d64710732570d594f5ed5a8079ef7de1f62aa6f5734200, 7116 bytes); all 68 entries match after the gate repair.
- Implementation hash manifest: configs/phase10r/phase10r-implementation.sha256 (SHA-256 976fed970046bcc79a7490af8432f3ca664db8f8419efc068ddb59e36a8146ad, 7126 bytes); all entries match after the gate repair.
- Implementation commits after the expected HEAD: d0e1b8502f9c6306fcce6fcfd4ccebcfd6b42203, bff7ae18d51a4d48215b0c5845bfcd6f2fc95005, b7bfc7419cdb892152b482b00f1df2cbbd1096a7, 714e07b56c9bff4f3b54b9fc06ce5bbf69fbfb8e, 222aa63dd518330ddec1c2efe70a9d1746f5396a, 101662232299558230595a340565410ba1d71445, 8a7b601c308742e87044b1dbdfd1ef5e1b8f2cb3, and b8fa5c2ca26bb7ab2ad6789468585bb7f0764cb8.

The production OSAVAL02 search/tactical adapter is shared across native Rust, CLI/USI, and Wasm. It strictly loads OSAVAL02, supports both frozen candidate variants, preserves current-side-to-move score/mate semantics, and has no OSAVAL01 fallback.
- CLI: target/release/open-shogi-cli (SHA-256 54a8aeae7a1148c6cadc64fe72cb6d43a7d5da651901297ee61cfba5d7e581fc, 4069056 bytes).
- Generated Wasm: bindings/wasm/open_shogi_wasm_bg.wasm (SHA-256 1e4291d611fa029d1414d88b434fb80b1fbc73407d2c019a99f76861adc1fb0c, 714258 bytes).
- Frozen OSAVAL02 core source SHA-256: b23219a7481aa5d72ec887e89fd7e724588b56159ebbc65d82e4b29a164000cc.
- Frozen Wasm source binding SHA-256: e525a5800b3def8cfbc9026cd24ca9822a235fb2ec35b6970a11c118dfdbcf4d.

## Data and teacher identity

- Preparation manifest: local/phase10r-data/phase10r-prepared/1m-mixture-v2/preparation-manifest.json (SHA-256 9dddab33fa50ec070f267f43ea12ebab1c1931e16c8826219eb50b85d8061d2b, 8860 bytes); manifest identity SHA-256 3e36bf596ddb25356abad276745c3b2b2cae2bf01a38894f65d1a362b6481ecf; file SHA-256 9dddab33fa50ec070f267f43ea12ebab1c1931e16c8826219eb50b85d8061d2b.
- Calibration input manifest: local/phase10r-data/phase10r-teacher-binding/1m/calibration-input-manifest-v2.json (SHA-256 9661f52684bffb1ecfb63a13e59e76e1d57cf9dc4c2b181140fe0fbdc3f395d0, 1153 bytes).
- Calibration train JSONL: local/phase10r-data/phase10r-teacher-binding/1m/calibration-train.jsonl (SHA-256 bc80546693b1e3e8a1b9fcd9c5f3ae5b5b630812d5e2bd33c69bb0bf0cc09d37, 21447182 bytes).
- Calibration validation JSONL: local/phase10r-data/phase10r-teacher-binding/1m/calibration-validation.jsonl (SHA-256 2ce60cbb87195da3ee68d32a8ae8514ff16ac434741ee8717b63b7c8ca34e62c, 5916984 bytes).
- Source-held-out JSONL: local/phase10r-data/phase10r-prepared/1m-mixture-v2/base-source_held_out.jsonl (SHA-256 1112a5f1a93002e5e3ce2ec0371b5a3adbb4c4cd5b01957fc4512e1d653091eb, 107659473 bytes); validation JSONL: local/phase10r-data/phase10r-prepared/1m-mixture-v2/base-validation.jsonl (SHA-256 029916f33af13572d2b8045508ad8eaa8005afa8298f8f355fd3903a5aede204, 109805427 bytes).
- Frozen Arena start pool: artifacts/phase10/start-pool-manifest.json (SHA-256 491a3d54d3d67002fc03fc3644c16b405f5b1dbb662e88e4af260bd45757a1f1, 1289935 bytes); schema open_shogiai_phase10_start_pool/v2; 800 starts, 200 per group; seed 20260821.
- Teacher binding config SHA-256: d587e8d06d86af6dc10cd1a844532f29912e4f11da53f7e659e2a8a0f131d9ff; teacher identity SHA-256: 781a45570ce6c88e29e4c9f7f3acb96e3981bb74d7da6fb10ff80cdd76f51cbf.
- Apery 2.0.0: book disabled, eval 20190617, eval hash 256, MultiPV 3, Threads 4, USI Hash 1024 MiB, 25,000 baseline nodes, concurrency 1; binary SHA-256 8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403; KKP 422b23bced817ecb3430adf1d2621f5a7934263b4e46673ab80cf34633537fa5; KPP 4906c48c201a102ec02217216929c20f73ab364e79be26e6213a02c04e454805.
- Existing teacher labels: 10,000. Calibration used 6,570 train rows, 1,880 validation rows, and 1,831 CP-fit rows; exclusions were 1,064 phase-4 test, 467 absent, and 19 split-mismatch rows; new teacher calls: zero.

## 1M parent identity and Stage 3/4 readiness

| Candidate | Parent checkpoint SHA-256 | Parent OSAVAL02 SHA-256 | Parent params / bytes | Teacher-bound artifact SHA-256 |
| --- | --- | --- | ---: | --- |
| sparse-pair-policy-wdl | 6a5048ae1e7d230dbb073d701fad6ac7acab6d9dc9637480282c5e0bc8ece48f | 6b7f2f4dc0bb013992460cf475f5e2e1f220da7fb09542175be367821f296f63 | 2,413,321 / 9,657,380 | 41ce44a93219b0bcee0270677104c379029999050524fe99c1a560bb5cc05060 |
| factorized-pair-triple-policy-score | 47c43cd44f6258973d3262931d96d46e4e36f55b456249647fe086c4253c5ed8 | d2c9af492cb67be35433662b75673efa2c6fe3f0ba2457e281185469364453 | 2,676,618 / 10,710,568 | 41ee0b8f901d7a37bb692073268a96058dc185198663a38a63a7517096343c28 |

| Candidate | Stage 3 steps / rows | Last loss | Ranking | Score | Mate | Stage 4 scale / bias | Fit rows / MSE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sparse-pair-policy-wdl | 52 / 6570 | 17.83121109 | 748.9108887 | — | 6.347250938 | 197.0466309 / 131.4604797 | 1831 / 2870466.85 | passed |
| factorized-pair-triple-policy-score | 52 / 6570 | 12.89728355 | 531.4847412 | 10.20113754 | 7.015503883 | 85.92930603 / -305.3045654 | 1831 / 2787252.661 | passed |

Both receipts report lineage passed; immutable parent; forbidden split rows 0; new teacher calls 0; Python/native/Wasm parity passed; incremental/full-recompute/unmake parity passed; zero new tactical failures; source-held-out regression maximum 0.0; ECE regression within the ±0.01 gate; and throughput floor passed.
- Sparse Stage 3/4 receipt: local/phase10r-runs/20260829T164346.831537Z-calibrate-teacher.json (SHA-256 1e89db2cde6708b037a61af93539e3dc1bcfdcae8cca40ac0d204dc29b371ded).
- Factorized Stage 3/4 receipt: local/phase10r-runs/20260829T171123.491795Z-calibrate-teacher.json (SHA-256 a86ff31a70c7098f1fbc34e0d7fda49d33841afbc294b246ca93233e28834afb).

## Source-held-out metrics

| Candidate | Rows | Policy NLL / top-1 | WDL Brier / NLL | ECE |
| --- | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl | 54142 | 6.301727721 / 0.0757005515 | 0.1684365393 / 0.861420416 | 0.1167036229 |
| factorized-pair-triple-policy-score | 54142 | 5.312692486 / 0.07266358623 | 0.1765488639 / 0.8879345422 | 0.1563934344 |

Source-held-out regressions versus each immutable parent were 0.0 for policy NLL, policy top-1, WDL Brier, and WDL NLL for both candidates. Sparse child: policy NLL 6.3017277213, top-1 0.0757005515, WDL Brier 0.1684365393, WDL NLL 0.8614204160, ECE 0.1167036229. Factorized child: policy NLL 5.3126924863, top-1 0.0726635862, WDL Brier 0.1765488639, WDL NLL 0.8879345422, ECE 0.1563934344.

## Offline validation and runtime measurements

Offline measurements use the frozen 1,880-row calibration validation split and OSAVAL02 float32 artifacts; no final holdout was opened.

| Candidate | Policy top-1 / top-3 / NLL | Teacher top-1 / top-3 recall / exact set | Ranking Spearman (n) | WDL Brier / NLL | Value MSE / MAE | ECE | Mate sign (n) | Score Pearson / Spearman / MAE cp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl | 0.08829787234 / 0.1962765957 / 7.312450175 | 0.09468085106 / 0.1875886525 / 0.08031914894 | 0.08386737782 (1857) | 0.2348077568 / 1.361844161 | 0.9994843295 / 0.8201334786 | 0.2713382057 | 0.693877551 (49) | 0.3330170195 / 0.1667266945 / 1014.836701 |
| factorized-pair-triple-policy-score | 0.09893617021 / 0.204787234 / 5.956453288 | 0.0914893617 / 0.1914893617 / 0.08085106383 | 0.08996808679 (1857) | 0.2425487229 / 1.456054746 | 1.196750163 / 0.9177597829 | 0.2962436174 | 0.387755102 (49) | 0.3696856991 / 0.1723889907 / 940.8678318 |

| Candidate | Inference n | Mean / p50 / p95 / p99 / max ns | Throughput/s | Stage 4 search NPS / floor |
| --- | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl | 1880 | 213200.0968 / 216583 / 262587.2 / 276609.25 / 336458 | 4690.42939 | 4414 / 3500 |
| factorized-pair-triple-policy-score | 1880 | 369052.2303 / 369125 / 454518.75 / 474321.14 / 519500 | 2709.643562 | 3349 / 3000 |

### Float/int8 parity

| Candidate | Rows | WDL max delta | Score max delta cp | Logit max delta | Policy top-1 agreement | Exact policy order | Mate-class agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl | 1880 | 0.02502606629 | 23 | 9.623786509 | 0.975 | 0.145212766 | 0.9813829787 |
| factorized-pair-triple-policy-score | 1880 | 0.02500338876 | 2175 | 5.168596239 | 0.9627659574 | 0.1457446809 | 0.9914893617 |

### Model size and memory

| Candidate | Parameters | Float32 / int8 artifact bytes | Offline float32 / int8 peak RSS bytes | Start-pool float32 peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl | 2,413,321 | 9,657,380 / 2,417,417 | 36,372,480 / 19,480,576 | 29,949,952 |
| factorized-pair-triple-policy-score | 2,676,618 | 10,710,568 / 2,680,714 | 39,174,144 / 20,414,464 | 32,555,008 |

### Frozen start-pool robustness

| Candidate | Group | Rows | Mean expected WDL | Mean score cp | Latency mean / p50 / p95 / p99 ns |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl | general_opening | 200 | 0.1277256331 | 202.36 | 259926.26 / 260250 / 282846.45 / 299275.84 |
| sparse-pair-policy-wdl | hard_middlegame_endgame | 200 | 0.1426678781 | 408.665 | 203625.6 / 207250 / 249983.65 / 272477.17 |
| sparse-pair-policy-wdl | ibisha | 200 | -0.04521173826 | 47.57 | 222447.725 / 229833.5 / 268062.8 / 285736.49 |
| sparse-pair-policy-wdl | opponent_furibisha | 200 | -0.2271149683 | -256.645 | 203134.2 / 208125 / 251779.2 / 266422.25 |
| factorized-pair-triple-policy-score | general_opening | 200 | 0.1151835641 | -159.795 | 445657.955 / 448313 / 477241.65 / 491638.75 |
| factorized-pair-triple-policy-score | hard_middlegame_endgame | 200 | 0.2012618366 | 510.345 | 354674.36 / 360687.5 / 419606.6 / 434698.75 |
| factorized-pair-triple-policy-score | ibisha | 200 | -0.02572448755 | 85.86 | 385025.4 / 390104 / 468825 / 495546.17 |
| factorized-pair-triple-policy-score | opponent_furibisha | 200 | -0.1787165167 | 103.465 | 353212.715 / 358687 / 426209 / 444305.84 |

## Arena measurements

Controls: frozen start manifest SHA-256 491a3d54d3d67002fc03fc3644c16b405f5b1dbb662e88e4af260bd45757a1f1; seed 20260821; reversed colors; opening book disabled; depth cap 8; 32 MiB hash per player; max 128 plies; pure-value OSAVAL02; one concurrent game. The requested 400/800/200 counts are total games represented by 200/400/100 paired starts, each pair using the same start with colors reversed.

max_plies is a fixed, explained legal termination. It is included in total games and excluded from the W-D-L denominator, exactly as recorded by finished_wld_games. There were 10 such head-to-head games, zero illegal moves, and zero unexplained crashes.

Raw Arena aggregate: local/phase10r-selection/arena-results.json (SHA-256 89f4f3771197fb66a1c6fb9cf1103706eb7dbf4c678afde9d94c7890d6920ae6, 1083043 bytes). It contains the 1,600 pair-report references and hashes; the independent validation sweep found 1,600/1,600 valid reports and 3,200/3,200 games.

| Matchup | ms | Games / pairs | W-D-L | Capped | Finished WDL | Score | Wilson 95% CI | Paired bootstrap 95% CI | Elo | Search NPS mean / p50 / p95 | Search latency mean / p50 / p95 ms | Illegal / unexplained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| sparse-pair-policy-wdl-vs-handcrafted-250ms | 250 | 400 / 200 | 28-0-372 | 0 | 400 | 0.07 | [0.0488716267, 0.09930894643] | [0.045, 0.0975] | -449.3539634 | 6061.809715 / 6000.071401 / 7051.357201 | 241.3310082 / 243.0454545 / 250.0911255 | 0 / 0 |
| sparse-pair-policy-wdl-vs-handcrafted-1000ms | 1000 | 800 / 400 | 47-0-753 | 0 | 800 | 0.05875 | [0.04446627494, 0.07725108331] | [0.0425, 0.07625] | -481.8788473 | 6153.440842 / 6063.879855 / 7328.24717 | 935.4288635 / 944.831117 / 998.936 | 0 / 0 |
| sparse-pair-policy-wdl-vs-handcrafted-5000ms | 5000 | 200 / 100 | 6-0-194 | 0 | 200 | 0.03 | [0.01382031434, 0.06389429245] | [0.01, 0.055] | -603.8601918 | 5727.61938 / 5915.826844 / 7193.960393 | 4515.948189 / 4596.926471 / 4796.751782 | 0 / 0 |
| factorized-pair-triple-policy-score-vs-handcrafted-250ms | 250 | 400 / 200 | 27-0-373 | 0 | 400 | 0.0675 | [0.04680146916, 0.09642666544] | [0.045, 0.0925] | -456.1380271 | 3076.79269 / 3178.672067 / 4068.589876 | 243.3187806 / 244.604878 / 250.7813131 | 0 / 0 |
| factorized-pair-triple-policy-score-vs-handcrafted-1000ms | 1000 | 800 / 400 | 41-0-759 | 0 | 800 | 0.05125 | [0.03800087404, 0.06878816736] | [0.03625, 0.06625] | -506.9831677 | 3192.265433 / 3255.606998 / 4058.002119 | 944.2150768 / 953.3455882 / 1000.158065 | 0 / 0 |
| factorized-pair-triple-policy-score-vs-handcrafted-5000ms | 5000 | 200 / 100 | 10-0-190 | 0 | 200 | 0.05 | [0.0273826456, 0.08957814814] | [0.025, 0.08] | -511.5014404 | 3508.151935 / 3497.265036 / 4211.061108 | 4555.011407 / 4628.76638 / 4854.314082 | 0 / 0 |
| sparse-pair-policy-wdl-vs-factorized-pair-triple-policy-score-1000ms | 1000 | 400 / 200 | 293-5-92 | 10 | 390 | 0.7576923077 | [0.7127869954, 0.7975706511] | [0.7173913043, 0.7971576227] | 198.0502707 | 5728.472539 / 5665.193114 / 6869.931489 | 958.0368797 / 967.8686869 / 991.9975347 | 0 / 0 |

### Arena results by opening group

| Matchup | Group | Games | W-D-L | Capped | Finished WDL | Score | Wilson 95% CI | Paired bootstrap 95% CI | Elo |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| sparse-pair-policy-wdl-vs-handcrafted-250ms | general_opening | 82 | 0-0-82 | 0 | 82 | 0 | [0, 0.04475062369] | [0, 0] | -1720.403312 |
| sparse-pair-policy-wdl-vs-handcrafted-250ms | hard_middlegame_endgame | 110 | 9-0-101 | 0 | 110 | 0.08181818182 | [0.04364064628, 0.1482179178] | [0.03636363636, 0.1363636364] | -420.0315457 |
| sparse-pair-policy-wdl-vs-handcrafted-250ms | ibisha | 112 | 10-0-102 | 0 | 112 | 0.08928571429 | [0.04922153008, 0.1565895764] | [0.03571428571, 0.1517857143] | -403.4400687 |
| sparse-pair-policy-wdl-vs-handcrafted-250ms | opponent_furibisha | 96 | 9-0-87 | 0 | 96 | 0.09375 | [0.05010810821, 0.1686533069] | [0.04166666667, 0.15625] | -394.1106973 |
| sparse-pair-policy-wdl-vs-handcrafted-1000ms | general_opening | 190 | 1-0-189 | 0 | 190 | 0.005263157895 | [0.0009296815657, 0.02920555811] | [0, 0.01578947368] | -910.5847217 |
| sparse-pair-policy-wdl-vs-handcrafted-1000ms | hard_middlegame_endgame | 208 | 16-0-192 | 0 | 208 | 0.07692307692 | [0.04790093362, 0.1212890777] | [0.04326923077, 0.1153846154] | -431.6724984 |
| sparse-pair-policy-wdl-vs-handcrafted-1000ms | ibisha | 208 | 11-0-197 | 0 | 208 | 0.05288461538 | [0.02978328086, 0.09220161749] | [0.02403846154, 0.08653846154] | -501.2294164 |
| sparse-pair-policy-wdl-vs-handcrafted-1000ms | opponent_furibisha | 194 | 19-0-175 | 0 | 194 | 0.09793814433 | [0.06359812601, 0.1478917158] | [0.05670103093, 0.1443298969] | -385.7137791 |
| sparse-pair-policy-wdl-vs-handcrafted-5000ms | general_opening | 28 | 0-0-28 | 0 | 28 | 0 | [6.938893904e-18, 0.1206433048] | [0, 0] | -1720.403312 |
| sparse-pair-policy-wdl-vs-handcrafted-5000ms | hard_middlegame_endgame | 72 | 2-0-70 | 0 | 72 | 0.02777777778 | [0.007651021904, 0.09574175221] | [0, 0.06944444444] | -617.6272177 |
| sparse-pair-policy-wdl-vs-handcrafted-5000ms | ibisha | 50 | 1-0-49 | 0 | 50 | 0.02 | [0.003539259272, 0.1049544359] | [0, 0.06] | -676.078432 |
| sparse-pair-policy-wdl-vs-handcrafted-5000ms | opponent_furibisha | 50 | 3-0-47 | 0 | 50 | 0.06 | [0.02061497035, 0.1621709169] | [0, 0.12] | -477.9906413 |
| factorized-pair-triple-policy-score-vs-handcrafted-250ms | general_opening | 82 | 0-0-82 | 0 | 82 | 0 | [0, 0.04475062369] | [0, 0] | -1720.403312 |
| factorized-pair-triple-policy-score-vs-handcrafted-250ms | hard_middlegame_endgame | 110 | 6-0-104 | 0 | 110 | 0.05454545455 | [0.02523582705, 0.1139178608] | [0.01818181818, 0.1] | -495.5528356 |
| factorized-pair-triple-policy-score-vs-handcrafted-250ms | ibisha | 112 | 7-0-105 | 0 | 112 | 0.0625 | [0.03060192234, 0.1234142563] | [0.02678571429, 0.1071428571] | -470.4365036 |
| factorized-pair-triple-policy-score-vs-handcrafted-250ms | opponent_furibisha | 96 | 14-0-82 | 0 | 96 | 0.1458333333 | [0.0889020578, 0.2300181503] | [0.08333333333, 0.21875] | -307.0743267 |
| factorized-pair-triple-policy-score-vs-handcrafted-1000ms | general_opening | 190 | 0-0-190 | 0 | 190 | 0 | [0, 0.01981752946] | [0, 0] | -1720.403312 |
| factorized-pair-triple-policy-score-vs-handcrafted-1000ms | hard_middlegame_endgame | 208 | 15-0-193 | 0 | 208 | 0.07211538462 | [0.04418678595, 0.1155622028] | [0.03846153846, 0.1057692308] | -443.78642 |
| factorized-pair-triple-policy-score-vs-handcrafted-1000ms | ibisha | 208 | 9-0-199 | 0 | 208 | 0.04326923077 | [0.02292847167, 0.08017438148] | [0.01442307692, 0.07692307692] | -537.8442268 |
| factorized-pair-triple-policy-score-vs-handcrafted-1000ms | opponent_furibisha | 194 | 17-0-177 | 0 | 194 | 0.08762886598 | [0.0554302234, 0.1358414092] | [0.05154639175, 0.1288659794] | -407.009738 |
| factorized-pair-triple-policy-score-vs-handcrafted-5000ms | general_opening | 28 | 0-0-28 | 0 | 28 | 0 | [6.938893904e-18, 0.1206433048] | [0, 0] | -1720.403312 |
| factorized-pair-triple-policy-score-vs-handcrafted-5000ms | hard_middlegame_endgame | 72 | 4-0-68 | 0 | 72 | 0.05555555556 | [0.02181421598, 0.1343201597] | [0.01388888889, 0.1111111111] | -492.1795686 |
| factorized-pair-triple-policy-score-vs-handcrafted-5000ms | ibisha | 50 | 3-0-47 | 0 | 50 | 0.06 | [0.02061497035, 0.1621709169] | [0, 0.12] | -477.9906413 |
| factorized-pair-triple-policy-score-vs-handcrafted-5000ms | opponent_furibisha | 50 | 3-0-47 | 0 | 50 | 0.06 | [0.02061497035, 0.1621709169] | [0, 0.12] | -477.9906413 |
| sparse-pair-policy-wdl-vs-factorized-pair-triple-policy-score-1000ms | general_opening | 82 | 61-1-15 | 5 | 77 | 0.7987012987 | [0.6959560314, 0.8730589353] | [0.7051282051, 0.8866666667] | 239.417367 |
| sparse-pair-policy-wdl-vs-factorized-pair-triple-policy-score-1000ms | hard_middlegame_endgame | 110 | 82-0-27 | 1 | 109 | 0.752293578 | [0.6635947907, 0.823814718] | [0.6697247706, 0.8272727273] | 192.9800353 |
| sparse-pair-policy-wdl-vs-factorized-pair-triple-policy-score-1000ms | ibisha | 112 | 82-3-24 | 3 | 109 | 0.7660550459 | [0.6783654541, 0.8356300277] | [0.6875, 0.8409090909] | 206.058518 |
| sparse-pair-policy-wdl-vs-factorized-pair-triple-policy-score-1000ms | opponent_furibisha | 96 | 68-1-26 | 1 | 95 | 0.7210526316 | [0.6236306075, 0.8012922994] | [0.6526315789, 0.7916666667] | 164.977879 |

## Gate and selection interpretation

1. Mandatory correctness/tactical gates: both candidates passed. Both have exact parent/data/teacher bindings, immutable parent artifacts, zero new teacher calls, no forbidden split rows, passed lineage, Python/native/Wasm OSAVAL02 parity, incremental/full/unmake parity, source-held-out non-regression, ECE gate, zero new tactical failures, and search throughput floors.
2. Aggregate equal-wall-clock score versus handcrafted: sparse 81-0-1319, score 0.05785714285714286; factorized 78-0-1322, score 0.055714285714285716. Both are below the frozen promotion gate, so neither is represented as promoted against handcrafted; this Goal is a relative architecture selection.
3. Direct candidate head-to-head: sparse 293-5-92, 390 finished W-D-L games plus 10 controlled max-ply games, score 0.7576923076923077, Wilson 95% CI [0.7127869954, 0.7975706511], paired-color bootstrap 95% CI [0.7173913043, 0.7971576227], Elo 198.0502706832. This resolves selection before extra seeds.
4. Robustness: sparse led factorized in every opening group at 1 second: general opening 61-1-15 plus 5 caps (score 0.7987012987), ibisha 82-3-24 plus 3 caps (0.7660550459), opponent furibisha 68-1-26 plus 1 cap (0.7210526316), and hard middlegame/endgame 82-0-27 plus 1 cap (0.7522935780). All per-time-control and per-opening results are tabulated above.
5. Throughput/latency: sparse offline float32 inference throughput was 4690.4293899/s versus factorized 2709.6435622/s in the captured v5 run; Stage 4 search NPS was 4,414 versus 3,349; sparse is 2,413,321 parameters versus 2,676,618.
6. Offline metrics/model size: factorized led policy top-1/top-3, policy NLL, ranking Spearman, score correlation, and score MAE; sparse led WDL Brier/NLL, value MSE/MAE, ECE, and mate-sign accuracy. These lower-priority metrics do not overturn the direct head-to-head.

## Receipts and durable evidence

- Machine-readable manifest: PHASE_10R_MODEL_SELECTION_MANIFEST.json, SHA-256 b6b373329718755aa7ad569b195ea261c8968a9a8ea0380f815490b88bcfebcf.
- Selected configuration: PHASE_10R_SELECTED_ARCHITECTURE.yaml, SHA-256 b2923df5addeb247d4c8969bb0295a7af42309a125067ac4504eb115156ce8d4.
- Existing run-receipt inventory: 50 JSON receipts under local/phase10r-runs (19 passed, 31 blocked), all individually hashed in the machine-readable manifest. Blocked receipts are preserved history and are not selection evidence.
- The first Arena attempt had transient report-flush/partial capped-pair failures; those reports were recoverably quarantined. The final current Arena result is the only counted campaign and was independently validated.

## Explicit non-actions and next execution

- Final holdout: not opened.
- overall_champion: not mutated.
- Post-selection large-scale training: not started.
- Push, merge, release, and deploy: not performed.
- Sol usage: none; no Sol delegation was needed.

Exact next large-scale Luna execution command (recorded but intentionally not executed in this Goal):

    PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run prepare --root . --scale 10m
    PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run train --root . --scale 10m --variant sparse-pair-policy-wdl --resume

The raw offline and Arena result files are ignored local evidence; their paths, sizes, SHA-256 values, aggregate metrics, and Arena report-hash inventory are bound by PHASE_10R_MODEL_SELECTION_MANIFEST.json.
