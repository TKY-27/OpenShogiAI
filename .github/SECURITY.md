# Security Policy

OpenShogiAI is research software by a solo maintainer. No formal security
support or compatibility commitment is made.

## Reporting a vulnerability

Please do **not** post vulnerability details in public issues. Use GitHub's
private vulnerability reporting on this repository when available, or contact
X [@ANAg2bGOD](https://x.com/ANAg2bGOD) by direct message with a minimal,
non-sensitive description.

脆弱性の詳細を公開Issueへ投稿しないでください。GitHubの非公開セキュリティ報告
（利用可能になった場合）か、X [@ANAg2bGOD](https://x.com/ANAg2bGOD) のDMへ、
秘密を含まない最小の説明をお送りください。

## Scope notes

CSA/SFEN/USI input, downloaded data, teacher output, model files, manifests,
registries, Wasm messages and generated artifacts are untrusted inputs.
Preserve size limits, closed schemas, path containment, stable-file identity,
hashes, legality checks and resource bounds when changing parsers or runners.

Teacher executables and evaluation files stay under ignored `local/teacher/`.
Bootstrap and setup scripts must not require administrator privileges,
silently install software, or pipe downloaded content into a shell.
