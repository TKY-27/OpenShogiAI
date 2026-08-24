# Phase 10R WCSC replay repair report

Status: **repair contract frozen; full replay and full scan not run**.

Evidence date: 2026-08-24 (Asia/Tokyo)

## Finding

The 18 rejected WCSC streams do not have an archive-provenance failure. All 31 active WCSC
archives were reacquired from the exact URLs on the
[official WCSC game-record page](https://www.computer-shogi.org/kifu/kifu.html). Every live size
and SHA-256 matched the existing local archive. Reacquisition therefore cannot change any of the
18 outcomes.

The failures came from a lossy preparation boundary and a V3-only replay boundary:

- the LZH preparation used an extractor that reported tool errors, including unsupported `-lh4-`
  members, but still emitted a partial flat staging directory;
- the flat staging names discarded authoritative archive member paths and round metadata;
- the Rust parser rejected missing versions and V1/V2 records even though the
  [official CSA V3 specification](https://www.computer-shogi.org/protocol/record_v3.html) says a
  missing version means the 1997-08-25 compatibility format, and the official protocol index
  retains V2, V2.1, and V2.2 specifications;
- blank legacy lines, trailing NUL padding, and valid nonempty statements next to empty trailing
  statements were rejected before a source-preserving compatibility normalization;
- six derived KIF games used the opposite global board orientation, but the old converter assigned
  signs without proving a unique complete legal orientation; and
- five official CSA members contain an illegal pawn drop immediately followed by an explicit
  `ILLEGAL_ACTION` terminal. Those five games cannot satisfy complete legal replay and may not be
  trimmed or relabeled.

Across the 18 JSONL exports, 1,600 game rows were rejected: 1,010 missing/misordered-version,
480 declared legacy-version, 92 blank-line, 6 opposite-orientation KIF, 2 empty trailing
multi-statement, 5 other malformed/partial-extraction rows, and 5 explicit illegal-action rows.
The scanner then failed an entire artifact as soon as it encountered any rejected row. This is why
only 15 of 33 approved artifact streams completed.

The earlier `15/33` result is not proof that all members of those 15 source archives were replayed.
In particular, WCSC1 and WCSC3 were staged from legacy LZH inputs with conversion/unsupported-member
counts. The v2 execution must inventory and account for every archive member before any stream is
complete.

## Missing evidence

No authoritative WCSC archive bytes are missing. Before this repair, the following metadata was
missing or insufficient:

- expected archive SHA-256 values were null for WCSC1-31 in the registry;
- legacy LZH inventories did not prove extraction of every member;
- flattened staging names did not bind a game to its authoritative archive member path;
- failed games had no `source_game_id`, canonical game hash, per-position history identity, or
  transposition key; and
- the legacy export recorded parser/normalization versions but not a deterministic compatibility
  transformation receipt.

The archive hashes are now pinned in `configs/phase10r/source-registry.yaml`. The exact stream and
member ledger is `artifacts/phase10r/phase10r-wcsc-replay-manifest.json`.

## Rejected-stream ledger

All rows used legacy parser `phase3_csa_export/v1` and normalization
`rust_replay_export/v1`. Their required replacements are `phase10r-wcsc-parser/v2` and
`phase10r-source-preserving/v2`. “Identity unavailable” means the failed replay could not produce
a canonical game/position/history identity; the manifest retains the observed flat file and its
SHA-256. Each stream archive spans multiple official rounds, so its round scope is “all official
rounds in the archive”; an exact round may be assigned only from the authoritative member path.
The five exact exclusions have that member-level round recorded in the manifest.

| Stream | Competition/year/round | Expected SHA-256 | Observed SHA-256 | Legacy accepted/rejected | First rejection | Game / position-history identity | Reacquisition | Stream disposition |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| `wcsc02-kifu` | WCSC 2 / 1991 / all official archive rounds | `d2727a7a472787c93206678ae39a55dd54626d312dc3bc37a1b3883c9c4207e4` | same | 7/3 | version was not the first non-comment statement | staged-file SHA retained; authoritative path and canonical identities unavailable | identical bytes; no repair | replay; retain stream |
| `wcsc04-kifu` | WCSC 4 / 1993 / all official archive rounds | `ee0ce5422f804a22e5c5fe2487034674b4faed31dd72511a0aeccdc30d88bd77` | same | 4/3 | move source belonged to the other side | staged-file SHA retained; canonical identities unavailable | identical bytes; no repair | prove one global orientation, otherwise quarantine exact game; retain stream |
| `wcsc05-kifu` | WCSC 5 / 1994 / all official archive rounds | `3af7e0b7f56b32bc580ce0ba9da56d3a63c609055587a8f80b04f62aece0a958` | same | 10/10 | missing legacy version | identity unavailable | identical bytes; no repair | replay; retain stream |
| `wcsc06-kifu` | WCSC 6 / 1996 / all official archive rounds | `397e0a2e5e3542c066944df7793012444522720b7f828459919d0e068d6d760e` | same | 0/63 | partial/NUL legacy extraction then missing version | identity unavailable | identical bytes; no repair | re-extract with `-lh4-` support, replay; retain stream |
| `wcsc07-kifu` | WCSC 7 / 1997 / all official archive rounds | `1ace47bb7a107d12769e3773ddeb7a99b911d393543e655e3366d00d99c7b131` | same | 14/14 | declared CSA V1.0 rejected | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc08-kifu` | WCSC 8 / 1998 / all official archive rounds | `543635573909e905d64842e2431e81f44b3179a580a970255a526858e09135be` | same | 7/103 | missing legacy version / blank lines | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc09-kifu` | WCSC 9 / 1999 / all official archive rounds | `7b4f756bea7d3fe100c97d89bbbfad5ee9776f353fff18f44f219c8118986439` | same | 14/7 | missing legacy version | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc10-kifu` | WCSC 10 / 2000 / all official archive rounds | `d9a81fb08bb60df16f90698dcb17a7c99457247c3a40a0583fba9ce7455a2917` | same | 2/58 | missing legacy version / blank line | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc11-kifu` | WCSC 11 / 2001 / all official archive rounds | `d4acb6ad01a40430eb4de83f5aa4ea69d7a51f6f6a87c8341acef484e5ca76aa` | same | 52/189 | blank legacy line | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc12-kifu` | WCSC 12 / 2002 / all official archive rounds | `c7ebb20737e9ccb14cee22f01daebed8037a51d685870cbb568be820bd2562e8` | same | 23/86 | missing legacy version / blank lines | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc13-kifu` | WCSC 13 / 2003 / all official archive rounds | `aab4739d54b9ffe0b221abf4a78b22ee5d8e55847ee6b7e6a5f2782c983ae184` | same | 33/426 | missing or declared legacy version | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc14-kifu` | WCSC 14 / 2004 / all official archive rounds | `f9cbfd8790a23589231bd4569685f8a2cac79bc2442c461c593852a7b196ff84` | same | 0/81 | empty trailing multi-statement / legacy syntax | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc15-kifu` | WCSC 15 / 2005 / all official archive rounds | `803766ab89e63ce8d57a71f8990aa6d53520b0cd31f4e2d65f0740ff7bf43e68` | same | 21/129 | missing or declared legacy version | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc16-kifu` | WCSC 16 / 2006 / all official archive rounds | `4647f7d82a1be51076f3b6349f5c235c4e18fb9fdbae12dfb364efd39438481d` | same | 0/209 | declared CSA V2.1 rejected | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc17-kifu` | WCSC 17 / 2007 / all official archive rounds | `fcc006ce46bfbe16ef0bc289ffa7358c31117f4f87eb1dd25bdc02f8dbd96f9e` | same | 0/214 | declared CSA V2.1 rejected | identity unavailable | identical bytes; no repair | compatibility replay; retain stream |
| `wcsc22-kifu` | WCSC 22 / 2012 / all official archive rounds | `b705924fe2abe9b1cb62ed73808f00f52b68562c308b4bd3feeca555e990210c` | same | 224/3 | illegal pawn drop then `ILLEGAL_ACTION` | exact source-game IDs recorded; no legal canonical/history identity | identical bytes; no repair | exclude 3 exact members; retain stream |
| `wcsc26-kifu` | WCSC 26 / 2016 / all official archive rounds | `ca73a5f14ba7dc858f7e2c40b48e5e04d682d557436b0e0db182e2a007b0b0f9` | same | 259/1 | illegal pawn drop then `ILLEGAL_ACTION` | exact source-game ID recorded; no legal canonical/history identity | identical bytes; no repair | exclude 1 exact member; retain stream |
| `wcsc32-kifu` | WCSC 32 / 2022 / all official archive rounds | `d6b8ed2b4b971f488800835d2f8e5fe3a2026c5bf893b5c3e161c8a0d48eac03` | same | 278/1 | illegal pawn drop then `ILLEGAL_ACTION` | exact source-game ID recorded; no legal canonical/history identity | identical bytes; no repair | exclude 1 exact member; retain stream |

## Authorized population revision

No WCSC stream is removed. The only currently proven exclusions are the five exact ZIP members
listed with archive path, member SHA-256, source-game ID, competition/year/round, and rejection in
the v2 manifest. They remain source evidence but contribute zero training, validation, holdout,
Arena-start, or selection rows.

Any additional member exclusion requires: exact official archive and member hashes, exhaustive
supported extraction, a deterministic compatibility parse attempt, failure of complete legal
replay, and an append-only manifest entry. An archive or stream may not be excluded merely because
the old parser rejected it.
