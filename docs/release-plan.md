# Release plan and handoff (2026-09-24, awaiting user audit)

**状態: ユーザー手動監査待ち。このセッションは commit / push / tag / Release /
main統合 / branch・worktree削除 / Cloudflare接続を一切行っていない。**
監査後の Git確定・配布は Luna が、最後の Cloudflare設定はユーザーが行う
（OpenShogiUI の docs/deploy-cloudflare.md）。

## 1. 成果物の所在

| 対象 | 場所 |
|---|---|
| 公開モデル C4 G02（OSAI R4）重み | `local/runs/r4-c4/attempt-01/generations/G02/fit/trained_candidate.osaval03`（SHA-256 `9466a7e8cf11b7d165b325edd9a5a421bdbaa4bed550940afcf33c9faf3bfd0f`、8,679,836 bytes。実在物と発行済みmanifestで照合済み） |
| 配布manifest（正本） | `configs/models/distribution.json`（追跡対象。権利・hash・asset名の一元管理） |
| リリース組み立てスクリプト | `scripts/build_release_assets.py`（検証して `local/release/models-v1/` へ生成） |
| 暫定配布物（監査前ビルド） | `local/release/models-v1/`（10資産＋権利記録＋SHA256SUMS＋release-manifest＋README） |
| 権利・出典の根拠 | `docs/model/distribution.md`、`docs/source-audits/`、各 `rights-*.json` |
| 容量整理の監査記録 | `local/release/cleanup-2026-09-24.md` |
| 旧C4実行記録（旧status.md） | `local/handoff/r4-c4-status.md`（ツリーからは移動。Git履歴に旧版が残る） |
| OSUI 統合済み正本 | `../OpenShogiUI`（作業ツリーに未コミット変更として統合） |
| UI改善の元branch | `codex/osui-mobile-consent` @ `0c2ed84`（別worktree。未取り込み差分ゼロ — 61ファイルを本体へバイト同一適用済み） |

## 2. 今回の未コミット変更範囲

**OpenShogiAI（branch `codex/core-prototype` @ `e5721eb`）**
- 新規: `README.ja.md`、`.github/`（CONTRIBUTING/SECURITY/テンプレート群）、
  `configs/models/distribution.json`、`scripts/build_release_assets.py`、
  `docs/model/distribution.md`、`docs/release-plan.md`（本文書）
- 更新: `README.md`（全面書換）、`AGENTS.md`（18行の常設ルールへ縮小）、
  `docs/model/license-pending.md`（CC BY 4.0決定を反映）、
  `docs/development.md`・`docs/model/handling.md`（旧status.md参照の解消）、
  `Makefile`（help文）、`training/open_shogi_training/phase10*.py`（終了メッセージ文言のみ）
- 削除: docs/status.md（→`local/handoff/`）、CONTRIBUTING.md・SECURITY.md
  （→`.github/`へ移動）、docs/model/osaval02-features.md・
  docs/model/osaval02-parity-corpus.schema.json（参照0の廃止ドキュメント）
- 削除済みの実データは `local/release/cleanup-2026-09-24.md` の表参照。

**OpenShogiUI（branch `codex/core-prototype` @ `cab4895`）**
- worktree `0c2ed84` の61ファイルを適用（UI改善2回分: モバイル盤/持ち駒/時計、
  移動marker、解析モデル選択、同意・収集API/D1、OGP/ロゴ、静的モデル配信）
- 追加実装: release-model.json（6モデル公開allowlist、既定 r4c4＝OSAI R4）、
  release-assets.json（固定release資産map）、scripts/fetch-release-models.mjs
  （hash検証付き取得＋`target/pure/bindings`ミラー）、
  scripts/build-cloudflare.mjs / deploy-cloudflare.mjs
  （`OSAI_D1_DATABASE_ID`/`PUBLIC_SITE_URL`→`wrangler.deploy.jsonc`生成、
  収集ON profile、cron `17 3 * * *`、D1 `openshogiai-games`）、
  `wrangler.jsonc`（Worker名 `openshogiai`）、収集のbuild時profile切替
  （`OSUI_COLLECTION_PROFILE`）、運営者・連絡先の一元化（`project.config.json`）、
  プライバシー文言の確定、scripts/verify-collection-browser.mjs
- 文書: README.md（全面書換）+ README.ja.md、docs/development.md、
  docs/deploy-cloudflare.md、AGENTS.md、.github/、ARCHITECTURE.md更新
- 削除: PHASE_1/2_REPORT、docs/design-qa.md・publication-readiness.md
  （→`OpenShogiUI/local/handoff/`）、`collection-policy.json`（build時profileへ統合）、
  ルート`SECURITY.md`（→`.github/`）

## 3. 検証済み項目（このセッションの実測）

- OSAI `make check` PASS（依存/境界/権利/来歴/文書リンク/format/lint/
  Pythonテスト/native build/決定的Wasm再生成）。
- OSUI `npm run check` PASS（format/lint/typecheck/vitest/a11y/境界/ライセンス/
  来歴/素材監査/本番ビルド＋6モデルallowlist検証）。※最終再実行中の結果は
  ユーザー監査時に最新ログで確認のこと。
- C4 G02 identity: 重み・engine.js（907da142）・wasm（ab7fcf0e）が
  `selection.json`・`c4-result.json`・`development/model-audit.json` と一致。
- 暫定配布物を空の `/tmp` ディレクトリへ展開し: SHA256SUMS 全OK、
  USIバイナリ実起動（usiok/readyok/bestmove 9g9f）、
  実Wasmでのnative/Wasm parity監査 PASS（10 root/300子/306局面、cp・WDL差0）。
- OSUIコールドビルド: `local/model-assets` を空にして `/tmp` ミラーから
  14資産を取得・hash検証・ビルド成功（dist 45MB、418ファイル、最大6.5MB —
  Free制限25MiB/ファイル・20,000ファイル内。静的資産リクエストは無料無制限）。
- 実ブラウザー: UI回帰 PASS（320/390/844×390/1280、盤・時計・成り/打ち・
  モデル重複押下・解析・privacy両言語・resize・投了・再対局、console error 0、POST 0）。
  収集E2E PASS（同意→実C4対局開始→投了→`/api/games` 1回・201受付表示・
  D1行確認、同意拒否→POST 0）。ローカルD1の受領/冪等/不正入力/期限削除は
  `npm run test:collection:local` PASS。
- 環境依存の未実測: 実モバイル機の熱/電池/性能、実CDN/SNSのOGP取得、
  本番Cloudflareでの動作（ユーザー設定後に確認）。

## 4. ユーザー監査の着眼点（短い一覧）

1. 秘密/プライバシー: 両repoの追跡ファイル・`dist/`・権利JSONにMacの絶対パス、
   実名・個人メール無しを確認済み。Git履歴の著者メールは
   `269872094+TKY-27@users.noreply.github.com` のみ（両repo）。
   公開履歴に残る旧status.md等の内容に問題無いか最終確認。
2. ライセンス: `docs/model/distribution.md` のCC BY 4.0判断と出典クレジット、
   `rights-*.json` の中身。第三者データ自体は再配布していないこと。
3. 学習成果の表示: OSAI R4＝C4 G02であること、対C3採用条件未達（0.5625/0.5625）・
   序盤弱点・人間初段未検証が事実として残っていること（README・distribution.md・UI表示）。
4. UI/API: 同意文言・拒否時POST 0・収集対象限定・30日削除・問い合わせ先表示。
5. 削除一覧: `local/release/cleanup-2026-09-24.md`（実削除分とLuna待ち分）。
6. 配布手順: OpenShogiUI の docs/deploy-cloudflare.md の入力値と順序。

## 5. Lunaが確定する要素（ユーザー監査後）

- 両repoの最終commit（ユーザー修正を保護した上で再検証: `make check` /
  `npm run check`、必要なら再build。Astraの暫定hash不一致だけでユーザー修正を
  巻き戻さない）。
- 提案tag: OpenShogiAI `models-v1`（配布用Release）＋両repoソフトウェアtag `v0.1.0`
  （既存tagと衝突なし。モデル世代名OSAI R4とは別物）。
- Release資産: `python3 scripts/build_release_assets.py` を最終commitから再生成し、
  `local/release/models-v1/` の全ファイルをRelease `models-v1` へ。
  `release-manifest.json`/`SHA256SUMS.txt` は生成物（未来のcommit SHAの自己参照は要求しない）。
- OSUI側は `release-assets.json` の固定参照がそのままRelease実物と一致するか
  公開URLでhash確認 → `npm run build:cloudflare` がGitHub URLから通ることを確認。
- commit名義: repo-local設定のみで `git config user.name TKY-27` /
  `git config user.email 269872094+TKY-27@users.noreply.github.com`
  （global設定は変更しない）。
- 旧branch/worktree削除: 統合commit後に `git -C ../OpenShogiUI worktree remove
  ../OpenShogiUI-mobile-consent`（node_modules 307MB含む）と
  `codex/osui-mobile-consent` branch削除。OpenShogiAI-private-source（2.4GB、
  remote無し）・final-campaign（2.0GB）等の削除判断は
  `local/release/cleanup-2026-09-24.md` のLuna待ちリスト参照。

## 6. 推奨commit分割とコマンド例（実行禁止・提示のみ）

```sh
# OpenShogiAI（監査後、ユーザー修正を含めて）
git -C ~/Projects/OpenShogiAI status          # 未コミット範囲を確認
git -C ~/Projects/OpenShogiAI add README.md README.ja.md AGENTS.md .github/ \
  configs/models/distribution.json scripts/build_release_assets.py docs/ Makefile training/
git -C ~/Projects/OpenShogiAI commit -m "Prepare the reviewed two-repository release"
git -C ~/Projects/OpenShogiAI push origin codex/core-prototype
# （main統合・tag models-v1 と v0.1.0・Release作成は Luna が別途実施）

# OpenShogiUI
git -C ~/Projects/OpenShogiUI add -A
git -C ~/Projects/OpenShogiUI commit -m "Integrate mobile UI, collection, and static model delivery for release"
git -C ~/Projects/OpenShogiUI push origin codex/core-prototype
```

Cloudflare設定は OpenShogiUI の docs/deploy-cloudflare.md の手順1〜6
（ユーザー実施。Build変数 `OSAI_D1_DATABASE_ID` / `PUBLIC_SITE_URL`）。
