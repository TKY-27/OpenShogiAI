# OpenShogiAI（OSAI）

[English version](README.md)

OpenShogiAIは、将棋エンジンと機械学習の個人研究プロジェクトです。Rustで実装した
エンジンを、ネイティブとWebAssemblyの両方のランタイムで動かせます。定跡を使わない
Book-freeな対局を、独自設計の学習評価ネットワーク（OSAVAL03）で行い、公開モデルを
作った学習・評価ツール一式も同梱しています。

ブラウザーで対局・解析するためのWebアプリは別リポジトリ
[OpenShogiUI（OSUI）](https://github.com/TKY-27/OpenShogiUI)です。
このリポジトリ（[OpenShogiAI](https://github.com/TKY-27/OpenShogiAI)）は
エンジン・モデル形式・学習・モデル配布を担当します。

作者は[TKY-27](https://github.com/TKY-27)。

## 特徴

- 将棋の完全なルール実装：合法手生成、王手・成り・打ち・持駒、千日手・ステイルメイト
  処理、SFEN/USI/CSA入出力。
- 同一コードからの2つのランタイム：ネイティブUSIコマンドと、
  [OSUI](https://github.com/TKY-27/OpenShogiUI)が使う純粋なWebAssemblyモジュール。
- 純粋学習評価：OSAVAL03形式は玉相対のクリップペア蓄積ネットワーク（W256/W512）に
  直接cp値と勝敗のヘッドを持ちます。`pure-only`ランタイムは評価関数のフォールバックを
  一切許さず、棋力は学習済み重みだけに依存します。
- 全対局が定跡なし：開発対局でも本番でも、定跡・初手固定・局面表・外部エンジン
  探索を行いません。
- ネイティブ/Wasm一致：決定的なWasm再生成と、ランタイム間の評価一致検査が
  `make check`に含まれます。
- 学習パイプライン：自己対局生成、オフラインの固定深さ教師ラベリング、
  分割・来歴管理付きデータセット準備、再開可能な有限最適化実行。

## モデル

代表的な学習済み重みを[Releases](https://github.com/TKY-27/OpenShogiAI/releases)
（タグ `models-v1`）からCC BY 4.0で配布しています。最新世代 **OSAI R4** と、
それ以前の開発世代が含まれます。世代は新しい順に並んでいるだけで、強さの順位では
ありません。人間の段位に対する正式な検証は行っていません。

- リリースの中身：重みファイル（OSAVAL03）、共有ブラウザーエンジンランタイム、
  ランタイムプロファイル、モデルごとの権利・出典記録、Apple Silicon版USIバイナリ、
  チェックサム。
- 重みのライセンスと学習データの出典：
  [docs/model/distribution.md](docs/model/distribution.md)（英語）。
- 形式仕様：[docs/model/OSAVAL03_FORMAT.md](docs/model/OSAVAL03_FORMAT.md)（英語）。

ブラウザーで対局する：[OpenShogiUI](https://github.com/TKY-27/OpenShogiUI)。

## ビルドと実行

必要環境：安定版Rust（1.89以上）、`wasm32-unknown-unknown`ターゲット、
wasm-bindgen CLI 0.2.127、Python 3.12と[uv](https://docs.astral.sh/uv/)、GNU Make、
実Wasm検査用のNode.js。

```sh
git clone https://github.com/TKY-27/OpenShogiAI.git
cd OpenShogiAI
./scripts/bootstrap_macos.sh        # 任意：ローカルツールの確認と導入
uv sync --locked --group dev
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked --root local/tooling/wasm-bindgen-0.2.127
make build
```

既定（ハンドクラフト評価）のエンジンを起動：

```sh
cargo run --locked -p open-shogi-cli -- usi
```

学習モデルをピュアUSIモードで起動（重みはリリースから）：

```sh
make pure-build
target/pure/release/open-shogi-cli usi \
  --model osai-r4.osaval03 \
  --model-sha256 9466a7e8cf11b7d165b325edd9a5a421bdbaa4bed550940afcf33c9faf3bfd0f \
  --model-format OSAVAL03 --profile pure_learned
```

検証一式（フォーマット、lint、テスト、ネイティブビルド、決定的Wasm再生成、
ライセンス/境界/来歴検査）：

```sh
make check
```

リリースのmacOSバイナリは未署名のローカルビルドです。Gatekeeperの説明は
リリースノートを参照するか、上の手順でソースからビルドしてください。

## ドキュメント

- [開発ガイド](docs/development.md) — コマンド、保存場所、検証。
- [USI / ShogiHome 対局ガイド](docs/usi-shogihome.md)（英語）— 公開済み
  OSAI R4 重みを Linux のネイティブ USI 対局エンジンとして登録・対局する手順。
- [アーキテクチャ](docs/architecture.md)と[インターフェース契約](docs/interfaces.md)。
- [ルール](docs/rules.md) — 実装している将棋ルール。
- [モデル扱い](docs/model/handling.md)、
  [OSAVAL03形式](docs/model/OSAVAL03_FORMAT.md)、
  [配布と重みライセンス](docs/model/distribution.md)。
- [データの権利と来歴](docs/data/handling.md)、[出典監査](docs/source-audits/)。
- [ライセンス範囲](docs/license-scope.md)と[サードパーティ通知](THIRD_PARTY.md)。
- [参考文献](docs/references.md) — 参照した規則・アルゴリズム・先行実装
  （Apery、YaneuraOu系の資料など）。

## コントリビュート

Issueとプルリクエストを歓迎します。詳細は
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)。個人が保守するプロジェクトのため、
継続更新・回答期限・全PRのマージは保証しませんが、質の高い議論と貢献は積極的に
検討します。AIによる貢献は、AstraまたはFableと同等以上の能力を持つエージェントによるものに限ります。

## ライセンス

プロジェクト独自部分のソースコードは**AGPL-3.0-only**（[LICENSE](LICENSE)）です。
依存ライブラリ・データセット・第三者素材・モデル重みはそれぞれの条件に従います。
[ライセンス範囲](docs/license-scope.md)と[サードパーティ通知](THIRD_PARTY.md)、
公開重みのCC BY 4.0と出典記録は
[docs/model/distribution.md](docs/model/distribution.md)を参照してください。

連絡先：X [@ANAg2bGOD](https://x.com/ANAg2bGOD)（DM）。
