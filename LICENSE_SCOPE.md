# License Scope

## Project-owned source

Project-owned source code and documentation in this clean-history candidate are licensed under
the GNU Affero General Public License, version 3 only (`SPDX-License-Identifier:
AGPL-3.0-only`). `LICENSE` contains the unmodified license text published by GNU.

This project license applies to the Rust, Python, shell, configuration, test, and documentation
files in this repository unless a file or directory carries a more specific notice.

## Not relicensed

This repository does not change or absorb the terms of:

- Rust and Python dependencies;
- generated `wasm-bindgen` interface files and their relevant upstream notices;
- the separately executed Apery engine or its evaluation files;
- the approved AobaZero sample records and other external datasets;
- generated model weights, whose distribution terms remain pending review; or
- any other external asset identified in `THIRD_PARTY.md`, source registries, or audit records.

No third-party image, icon set, font, engine binary, evaluation file, dataset object, checkpoint,
or trained model weight is included in this source tree.

## Distribution checklist

Before distributing a build or artifact, preserve this license and all applicable dependency
notices, publish corresponding source as required by AGPL-3.0-only, and independently confirm the
distribution decision for every external dataset or model weight included with the artifact.
