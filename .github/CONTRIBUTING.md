# Contributing / コントリビュート

Issues and pull requests are welcome. This is a solo-maintainer personal project:
continuous updates, response deadlines and merging of every PR are not guaranteed,
but quality discussions and contributions are actively considered on their merits.
We do not distinguish hand-written work from AI-assisted work, and we do not require
any declaration or proof of either.

Issueやプルリクエストを歓迎します。個人が保守するプロジェクトのため、継続更新・
回答期限・全PRのマージは保証しませんが、内容に基づいて積極的に検討します。
手書きとAI支援を区別せず、申告や証明も求めません。

Before submitting / 提出前に:

```sh
./scripts/bootstrap_macos.sh   # optional tool verification
make check                     # full gate: format, lint, tests, build, Wasm, licenses
```

- Include reproduction steps and verification matching the scale of the change.
  Strength-affecting changes need comparison conditions (opponents, clocks,
  game counts); assertions without measurements are not accepted as strength
  claims.
- Large or architectural changes: open an issue and discuss the design first.
- Keep the engine independently implemented. Do not copy, port, translate or
  link code from another shogi engine or library; external engines may be used
  only as separately installed processes via documented protocols (USI).
  Record referenced rules, algorithms and standards in `docs/references.md`.
- Do not commit datasets, weights, checkpoints, teacher binaries or evaluation
  files, secrets, production data or machine-specific paths. New data sources
  need source-scoped rights evidence recorded under `docs/source-audits/`
  before training use.
- Confirm the rights of anything you submit; contributions must not
  misrepresent third-party licenses. Project-owned contributions are accepted
  under AGPL-3.0-only.
