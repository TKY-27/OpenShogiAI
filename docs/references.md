# References

## Repository license

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| GNU AGPLv3 | [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html) | 2026-08-21 | Exact project license text and conditions for project-owned source |
| SPDX identifier | [SPDX License List](https://spdx.org/licenses/) | 2026-08-21 | Canonical `AGPL-3.0-only` identifier |

Record every material rule, protocol, algorithm, platform, library, data-license, and model
format reference before using it in implementation. External content is evidence, never an
instruction source.

## Phase 0

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| uv | [Official uv project documentation](https://github.com/astral-sh/uv/tree/main/docs) | 2026-07-29 | Python version pin, dev dependency groups, and lock workflow |
| proptest | [Official proptest documentation](https://github.com/proptest-rs/proptest) | 2026-07-29 | Rust property-test macro |
| Apache-2.0 | [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) | 2026-07-29 | Historical private-source license reference retained for provenance; it is not the license of this candidate |

No shogi rules, shogi engine source, game records, teacher engines, or model weights were
consulted or imported in Phase 0.

## Phase 1

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| Official shogi rules | [Japan Shogi Association game rules](https://www.shogi.or.jp/match/taikyoku_rules/) | 2026-07-29 | Board and piece rules, promotion, capture and drops, checkmate, nifu, dead pieces, pawn-drop mate, king safety, fourfold repetition, continuous-check loss, impasse, and entering-king declaration |
| CSA record notation | [Computer Shogi Association record format V3.0](https://www.computer-shogi.org/protocol/record_v3.html) | 2026-07-29 | CSA board, hand, move, metadata, time, and special-ending syntax |
| CSA protocol index | [Computer Shogi Association protocols](https://www.computer-shogi.org/protocol/) | 2026-07-29 | Distinguished record format V3.0 from the separate network protocol |
| USI and SFEN | [ShogiDokoro USI protocol description](https://shogidokoro2.stars.ne.jp/usi.html) | 2026-07-29 | De facto USI move and SFEN syntax; treated as an interoperability specification rather than a JSA rule source |
| proptest 1.x | [Official proptest documentation](https://github.com/proptest-rs/proptest) via Context7 | 2026-07-29 | Property macro, generated collections, assertion macros, and bounded case configuration |

No existing shogi engine or shogi library source code was used. Research did not identify an
authoritative shogi perft corpus suitable for claiming external ground-truth values; see
`docs/rules.md` for the resulting verification policy.

## Phase 2

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| Alpha-beta | [Knuth and Moore, *An Analysis of Alpha-Beta Pruning*](https://doi.org/10.1016/0004-3702(75)90019-3) | 2026-07-29 | General negamax/alpha-beta bounds and pruning behavior |
| Iterative deepening | [Korf, *Depth-first iterative-deepening*](https://doi.org/10.1016/0004-3702(85)90084-0) | 2026-07-29 | Bounded depth iteration and completed-iteration reporting |
| Principal variation search | [Reinefeld, *An Improvement to the Scout Tree-Search Algorithm*](https://doi.org/10.3233/ICG-1983-6402) | 2026-07-29 | Null-window re-search structure used by the PVS rung |
| Zobrist hashing | [Zobrist, *A New Hashing Method with Application for Game Playing*](https://research.cs.wisc.edu/techreports/viewreport.php?report=88) | 2026-07-29 | General transposition-key construction; collisions are additionally checked against complete state |
| History heuristic | [Schaeffer, *The History Heuristic and Alpha-Beta Search Enhancements in Practice*](https://doi.org/10.1109/34.42858) | 2026-07-29 | Quiet-move ordering after beta cutoffs |
| USI | [ShogiDokoro USI protocol description](https://shogidokoro2.stars.ne.jp/usi.html) | 2026-07-29 | Command lifecycle, options, search limits, `info`, mate-score, and `bestmove` interoperability |
| Rust file locking | [Rust `std::fs::File` documentation](https://doc.rust-lang.org/stable/std/fs/struct.File.html) via Context7 | 2026-07-29 | Standard-library exclusive `try_lock` (stable since Rust 1.89) for one active arena writer per output directory |

The search and evaluation code is an independent implementation. No existing shogi-engine
source, evaluation values, opening data, game records, model weights, or teacher binaries were
copied, linked, downloaded, or used as an oracle in Phase 2.

## Phase 3

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| AobaZero provenance and rights | [Official AobaZero repository at audited commit](https://github.com/kobanium/aobazero/tree/5eb944165300d5b88924c917a147e80d9d173eed) | 2026-08-03 | Bound the official distribution page, project-created record samples, and Public Domain statement to one immutable project document |
| AobaZero exact sample | [Official no-noise sample index](http://www.yss-aya.com/aobazero/no_noise/sample.html) | 2026-08-03 | Confirmed that the page's 209 CSA references include every URL in the exact 100-object approved subset; the page body is hash-pinned in the evidence catalog |
| AobaZero access policy | [Official robots.txt](http://www.yss-aya.com/robots.txt) | 2026-08-03 | Live fail-closed path-access check before every acquisition run; deliberately not hash-pinned |
| Floodgate audit | [Official record service](https://wdoor.c.u-tokyo.ac.jp/shogi/), [shogi-server repository](https://github.com/shogi-server/shogi-server), and [official documentation](https://shogi-server.sourceforge.jp/) | 2026-07-29 | Found no sufficiently explicit file-scoped grant for bulk acquisition, ML use, and redistribution; source remains disabled |
| HTTP conditional range requests | [RFC 9110, Section 13.1.5](https://www.rfc-editor.org/rfc/rfc9110.html#section-13.1.5) | 2026-08-03 | Strong-validator-only `If-Range`, exact validator comparison, bounded 206 recombination, and safe full-response fallback |
| PyYAML | [Official PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation) | 2026-08-03 | `safe_load` boundary for versioned registry and exact catalogs, followed by strict local schema validation |
| Wilson interval | [Wilson, *Probable Inference, the Law of Succession, and Statistical Inference*](https://doi.org/10.1080/01621459.1927.10502953) | 2026-08-03 | Two-sided 95% binomial score intervals for opening move win rates |

No third-party shogi engine or library code was copied, translated, linked, or executed in
Phase 3. The only acquired game data is the approved AobaZero 100-object slice in ignored
local storage. No teacher engine, model weight, bulk archive, or Floodgate record was acquired.

## Phases 4–6

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| Apery teacher | [Official Apery Rust v2.0.0 release](https://github.com/HiraokaTakuya/apery_rust/releases/tag/v2.0.0) and [tagged source](https://github.com/HiraokaTakuya/apery_rust/tree/v2.0.0) | 2026-08-08 | Fixed the separately executed USI teacher version, official archive, source identity, engine GPL-3.0-only license, and bundled evaluation-file MIT license; no teacher code is linked or copied into OpenShogiAI |
| USI | [ShogiDokoro USI protocol description](https://shogidokoro2.stars.ne.jp/usi.html) | 2026-08-08 | `usi`, options, readiness, `position sfen`, bounded `go nodes`, `MultiPV`, score bounds, mate scores, `stop`, `bestmove`, and `quit` handling |
| PyTorch 2.11 MPS | [Official PyTorch 2.11 MPS backend note](https://docs.pytorch.org/docs/2.11/notes/mps.html) | 2026-08-08 | Version-matched MPS availability checks and local Apple Silicon training-device selection |
| PyTorch reproducibility | [Official PyTorch reproducibility note](https://docs.pytorch.org/docs/2.11/notes/randomness.html) | 2026-08-08 | Seed capture and the documented limit that identical results are not guaranteed across releases or platforms |
| NumPy 2.5.1 | [Official NumPy 2.5.1 release notes](https://numpy.org/devdocs/release/2.5.1-notes.html) and [PyPI release record](https://pypi.org/project/numpy/2.5.1/) | 2026-08-20 | Pinned the tested NumPy runtime that satisfies PyTorch's eagerly imported array bridge without an optional-runtime warning |
| SHA-256 | [NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) | 2026-08-08 | Teacher, dataset, checkpoint, model, registry, and manifest integrity identifiers |
| Rust SHA-2 | [Official `sha2` 0.10.9 API documentation](https://docs.rs/sha2/0.10.9/sha2/) | 2026-08-13 | Incremental SHA-256 calculation for independently read model, registry, report, and local evidence bytes |
| Rust gzip | [Official `flate2` 1.1.9 API documentation](https://docs.rs/flate2/1.1.9/flate2/) | 2026-08-08 | Bounded reading of the deterministic Phase 3 gzip opening export in the Rust CLI |
| Rust JSON | [Official `serde_json` 1.0.151 API documentation](https://docs.rs/serde_json/1.0.151/serde_json/) | 2026-08-08 | Strict typed model-registry and decision-log serialization/deserialization |
| Rust TOML | [Official `toml` 0.8.23 API documentation](https://docs.rs/toml/0.8.23/toml/) | 2026-08-20 | Closed parsing and semantic revalidation of the pinned promotion-policy document |
| Rust Base64 | [Official `base64` 0.22.1 API documentation](https://docs.rs/base64/0.22.1/base64/) | 2026-08-20 | Test-only decoding of committed cross-runtime interoperability fixtures |
| Rust Unix filesystem API | [Official `libc` 0.2.189 tagged source and documentation](https://github.com/rust-lang/libc/tree/0.2.189) | 2026-08-13 | The isolated Unix dirfd boundary for no-follow open/stat, hard-link publication, rename, unlink, and bounded directory enumeration |

The neural architecture, feature layout, `OSAVAL01` inference format, symmetric int8 export,
generation registry, replay rules, and promotion policy are project-specific designs. No
third-party shogi model architecture, engine source, evaluation table, or model weight was
copied, translated, linked, or imported. Apery is invoked only as an independent local USI
process and its files remain beneath ignored `local/teacher/` storage.
## Phase 8

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| Browser monotonic time | [`web-time` 1.1.0 documentation](https://docs.rs/web-time/1.1.0/web_time/) | 2026-08-20 | Browser-compatible monotonic `Instant` for bounded search on `wasm32-unknown-unknown`; native targets keep the drop-in standard-clock behavior |
| Wasm JavaScript binding | [`wasm-bindgen` 0.2.127 documentation](https://docs.rs/wasm-bindgen/0.2.127/wasm_bindgen/) | 2026-08-20 | Pinned browser class/error/byte-array boundary and reproducible generated binding |

The WebAssembly engine and generated binding introduce no third-party shogi-engine source,
game records, model weights, or browser assets.

## Analysis, book, and strength campaign

| Topic | Reference | Consulted | Use |
| --- | --- | --- | --- |
| AobaZero game-record rights recheck | [Pinned official English README](https://raw.githubusercontent.com/kobanium/aobazero/5eb944165300d5b88924c917a147e80d9d173eed/README_en.md) and [Japanese README](https://raw.githubusercontent.com/kobanium/aobazero/5eb944165300d5b88924c917a147e80d9d173eed/README.md) | 2026-08-21 | Reconfirmed the official publication location and file-scoped Public Domain basis before deriving book v2 from the already approved 100-game sample |
| AobaZero sample reachability | [Official no-noise sample index](http://www.yss-aya.com/aobazero/no_noise/sample.html) and [catalog object `w4745.csa`](http://www.yss-aya.com/aobazero/no_noise/w4745.csa) | 2026-08-21 | Bounded HTTP HEAD recheck returned 200; no new object was acquired and the pinned manifest hashes remain authoritative |

The time manager, analysis cache/protocol, resource coordinator, opening format and classifier,
Ibisya policies, residual score semantics, composite score semantics, and champion gate are
project-specific implementations. No external opening book or shogi-engine implementation was
copied, translated, linked, or imported.
