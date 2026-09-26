# Running the OSAI R4 learned engine as a USI opponent (Linux / ShogiHome)

This guide describes the supported path for playing the published **OSAI R4**
learned engine (`osai-r4.osaval03`) against another engine in a desktop USI
GUI, using the Linux desktop version of [ShogiHome](https://github.com/sunfish-shogi/shogihome)
as the tested reference. The engine plays book-free learned shogi: no fixed
opening moves, no handcrafted evaluation, no remote services.

What is claimed here is **native USI interoperability** — engine registration,
options, clocks, and play inside the GUI. It is not a claim about CSA network
play, and it is not a strength rating.

## Supported and tested environment

| Component | Status |
| --- | --- |
| Linux x86_64 (glibc), Rust stable + standard linker/C toolchain | Supported; built and exercised by the repository's Linux CI job |
| macOS arm64 | Supported for building and local subprocess testing (where this guide's commands were also verified) |
| Windows, Linux/musl or other libc variants, BSD | Untested — build from source and treat results as unverified |
| ShogiHome 1.29.0 | Version whose option handling (`USI_Hash`, `USI_Ponder` reserved defaults) and clock translation (`btime + binc` = time usable on the move) this adapter is matched against |

Rust toolchain requirements come from `rust-toolchain.toml` (stable channel).
`Cargo.lock` is pinned; build with `--locked`. Building and running the engine
binary needs only Rust plus a normal linker/C toolchain — no PyTorch, Node,
wasm-bindgen, or Python. The packaging script below additionally uses
`python3` (to read the manifest), and `curl` and `git` for the weight download;
on a machine without them, build with the plain cargo command and place the
weights next to the binary manually.

## Minimal build from a fresh clone

```sh
git clone https://github.com/TKY-27/OpenShogiAI.git
cd OpenShogiAI

# Native-only release build (no Wasm/training tooling required).
cargo build --locked --release -p open-shogi-cli --no-default-features --features pure-only

# Assemble the launch package under local/usi-r4/ (fetches the fixed public
# weights only if no verified local copy exists, verifies the SHA-256, writes
# the launcher, checksums, and usage text).
./scripts/build_usi_r4_package.sh
```

The package is written to `local/usi-r4/` (a Git-ignored location):

```
osai-r4-usi.sh      launcher to register in ShogiHome
open-shogi-cli      native engine binary
osai-r4.osaval03    fixed public OSAI R4 weights (SHA-256 verified on every load)
SHA256SUMS          checksums of the files above
PACKAGE_IDENTITY.txt source revision, model hash, supported options
USAGE.txt           short local usage text
```

The weights are the ones recorded in `configs/models/distribution.json`
(selection `r4c4`, asset `osai-r4.osaval03`, release tag `models-v1`). The
launcher passes the expected hash to the engine, and the engine refuses to
start on any mismatch. Note that the native binary previously published in the
`models-v1` release is an older adapter build; rebuild from source as above
instead of reusing it.

## Registering in ShogiHome

Menu and button names below were verified against ShogiHome 1.29.0. If your
version differs, expect the same flow under slightly different names.

1. Make the launcher executable once: `chmod +x osai-r4-usi.sh`.
2. In the main window, open **設定 → エンジン設定...** (**Settings → Engines...**,
   `Cmd/Ctrl+.`).
3. Click **追加** (**Add**) and select the file `osai-r4-usi.sh` — the
   launcher script, not the binary. ShogiHome starts scripts directly, so no
   shell command needs to be pasted into the path field.
4. The engine appears as `OpenShogiAI … pure_learned <model hash prefix>`.
   Set its options in the same dialog (defaults are correct):
   - `USI_Ponder`: **false** (keep it off; see below).
   - `Threads`: **1** (the engine runs exactly one search worker).
   - `USI_Hash`: 32 MiB is the default and a reasonable starting budget.
5. In the game dialog, choose this engine for one or both sides, pick a clock
   (for example 10 minutes + 5 seconds Fisher or byoyomi), and start.

To enable protocol logging for problem reports, turn on ShogiHome's USI log in
**設定 → アプリ設定...** (**Settings → App Settings...**), 開発者向け
(**Developer**) tab, 「USI通信ログを出力」 (**Record USI communication log**).
The engine itself writes nothing to stdout except USI lines.

## Supported options and protocol behavior

| Option | Values | Behavior |
| --- | --- | --- |
| `USI_Hash` (alias `Hash`) | spin 1..1024, default 32 MiB | Applied to the next search's transposition table via the engine's real table sizing. This budget is the TT size, **not** the process's total memory (observed RSS for a short match move is larger). |
| `USI_Ponder` | check, default **false** | Advertised false so the GUI does not invent `true`. Pondering is not implemented; keep it false. |
| `Threads` | spin 1..1, default 1 | One search worker. Other values are declined with a diagnostic, not accepted silently. |
| `RuntimeProfile` | combo, only `pure_learned` | The model and its verified hash are immutable at runtime. |

- Clocks follow the USI definition the GUIs implement: `btime + binc` (or
  `wtime + winc`) is the time usable on the current move, and byoyomi is
  current-move time. Zero base time with a positive increment is a valid turn
  and gets its increment as usable time.
- `go mate <ms|infinite>` is parsed and answered with `checkmate
  notimplemented`; the engine stays usable. No mate solver is built in.
- `go ponder` is held without pondering: `ponderhit` starts the ordinary
  search, and `stop` ends the request (the GUI discards that answer per USI).
  With `USI_Ponder false` the GUI never sends it.
- A repeated `stop` while idle produces no stale `bestmove`; `go infinite`
  publishes exactly one completion after `stop`; a replaced position supersedes
  any running search.
- The engine keeps its previous valid state on malformed options or invalid
  positions and answers with bounded `info string` diagnostics.
- EarlyPonder (ponderhit carrying clocks) is handled.

## Trial match conditions to record

When playing trial games, record both sides' clocks, hash budgets, hardware,
and the engine identity line. The opponent need not have identical resources;
the match conditions are simply part of the result record. Do not tune the
model against a specific opponent — the published weights are fixed by hash.

## Verification performed and outstanding scope

- Unit and lifecycle tests for both adapter feature classes
  (`cargo test --locked -p open-shogi-usi --no-default-features --features
  pure-only` and the default-feature suite).
- An end-to-end subprocess suite driving the real launcher with the real R4
  weights through the ShogiHome-like handshake, including stop/ponder/mate
  flows, reuse across games, and a run from an unrelated working directory
  under a path containing spaces (`scripts/verify_usi_r4_native.py`). On
  macOS this ran with ShogiHome 1.29.0 as the behavioral reference; the
  adapter is matched against its published option and clock behavior.
- A hosted Linux CI job that builds the package and runs the same suite on
  Ubuntu x86_64 (see `.github/workflows/native-usi.yml`).

**Outstanding:** driving an actual ShogiHome desktop window (engine
registration click-through and a GUI-played game) has not been validated in
this delivery — no Linux desktop environment was available, and no GUI
automation was substituted for it. The subprocess suite reproduces the exact
native transport contract (spawn, reserved options, clock translation,
stop/ponder lifecycle, score parsing) verified against the ShogiHome 1.29.0
source, but it is not a GUI pass. Registration steps above follow the
published 1.29.0 UI and still deserve one manual check at first use.

