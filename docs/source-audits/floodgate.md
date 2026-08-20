# Floodgate source audit

Review date: 2026-07-29 (Asia/Tokyo)

## Decision

Floodgate exposes individual CSA records and annual archives publicly, but public access is
not evidence that bulk acquisition, machine-learning use, or redistribution is authorized.
The reviewed official pages did not provide a sufficiently explicit file-scoped rights grant
for those activities.

The adapter entry therefore remains `enabled: false`, `approved: false`,
`machine_learning_allowed: false`, and `redistributable: false`. It has no catalog, and the
downloader rejects it before any network request or output-directory creation. Annual archive
paths are explicitly denied.

## Official pages reviewed

- Floodgate service and record archive: <https://wdoor.c.u-tokyo.ac.jp/shogi/>
- Shogi-server project: <https://github.com/shogi-server/shogi-server>
- Shogi-server documentation: <https://shogi-server.sourceforge.jp/>

The service page's heading identifies the public section simply as:

> 棋譜

That establishes availability, not data rights. No statement reviewed on 2026-07-29
comprehensively authorized bulk copying, machine learning, and redistribution of the record
corpus.

Verdict: **denied pending an explicit, attributable rights statement for the intended use**.
