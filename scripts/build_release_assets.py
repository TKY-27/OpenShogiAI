#!/usr/bin/env python3
"""Assemble the reviewed model distribution into an ignored staging directory.

Reads the tracked publication manifest (configs/models/distribution.json),
verifies every referenced artifact byte-for-byte, writes the per-model
distribution-rights records consumed by the OpenShogiUI release build, and
collects the release assets under local/release/<release_tag>/.

The staging directory is the local mirror used to test asset fetching and
builds before the GitHub release exists. Nothing here touches the network,
Git refs, or files outside local/release/.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "configs" / "models" / "distribution.json"
STAGING_BASE = ROOT / "local" / "release"
RIGHTS_DIR = ROOT / "local" / "release" / "rights"

RELEASE_README = """\
# OpenShogiAI model distribution ({tag})

Trained evaluation weights (OSAVAL03) for the OpenShogiAI pure-learned runtime,
plus the shared browser engine runtime, the runtime profile, and an Apple
Silicon macOS USI command-line build.

- Weights license: CC BY 4.0 (attribution: TKY-27 / OpenShogiAI). Training-data
  sources and their conditions are recorded per model in `release-manifest.json`
  and in each `rights-*.json`.
- The engine runtime pair is shared by all models in this release.
- `SHA256SUMS.txt` lists every asset; verify before use:
  `shasum -a 256 --check SHA256SUMS.txt`.
- The macOS binary is an unsigned local build. Remove the quarantine attribute
  after review (`xattr -d com.apple.quarantine open-shogi-cli-macos-arm64`) or
  build from source with `make pure-build`.
- Model generations are listed newest to earliest; this is development order,
  not a strength ranking. No human-shodan rating is claimed.

Source code: https://github.com/TKY-27/OpenShogiAI
Web application: https://github.com/TKY-27/OpenShogiUI
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(spec: dict) -> None:
    path = ROOT / spec["path"]
    if not path.is_file():
        raise SystemExit(f"missing artifact: {spec['path']}")
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise SystemExit(
            f"size mismatch for {spec['path']}: manifest {spec['bytes']}, actual {size}"
        )
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise SystemExit(
            f"hash mismatch for {spec['path']}: manifest {spec['sha256']}, actual {actual}"
        )


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["schema"] != "open_shogiai_model_distribution_manifest/v1":
        raise SystemExit("unsupported distribution manifest schema")
    tag = manifest["release_tag"]
    artifacts: list[tuple[dict, str]] = [
        (manifest["engine"]["js"], "engine"),
        (manifest["engine"]["wasm"], "engine"),
        (manifest["profile"], "profile"),
        (manifest["native_cli"], "cli"),
    ]
    for model in manifest["models"]:
        artifacts.append((model["weights"], f"model:{model['selection']}"))

    for spec, _role in artifacts:
        verify_artifact(spec)

    RIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    lineage = manifest["baseline_lineage_sources"]
    for model in manifest["models"]:
        rights = {
            "schema": "open_shogi_model_distribution/v1",
            "modelSha256": model["weights"]["sha256"],
            "selection": model["selection"],
            "publicName": model["public_name"],
            "license": "CC-BY-4.0",
            "attribution": "TKY-27 / OpenShogiAI",
            "trainingAllowed": True,
            "derivedWeightsAllowed": True,
            "redistributionAllowed": True,
            "commercialUseAllowed": True,
            "sources": lineage + [s for s in model.get("extra_sources", [])],
            "note": (
                "Conditions verified against the recorded source audits and the "
                "2026-09-24 publication policy; see docs/model/distribution.md. "
                "Third-party dataset terms apply to the datasets, not to these weights."
            ),
        }
        (RIGHTS_DIR / f"{model['selection']}.json").write_text(
            json.dumps(rights, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rights_path = ROOT / model["rights_path"]
        if rights_path.resolve() != (RIGHTS_DIR / f"{model['selection']}.json").resolve():
            raise SystemExit(f"rights path mismatch for {model['selection']}")

    staging = STAGING_BASE / tag
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    index_entries = []
    sums: list[str] = []
    seen_assets: set[str] = set()
    for spec, role in artifacts:
        asset = spec["asset"]
        if asset in seen_assets:
            raise SystemExit(f"duplicate asset name: {asset}")
        seen_assets.add(asset)
        shutil.copyfile(ROOT / spec["path"], staging / asset)
        actual = sha256_file(staging / asset)
        if actual != spec["sha256"]:
            raise SystemExit(f"staged copy mismatch for {asset}")
        sums.append(f"{actual}  {asset}")
        index_entries.append(
            {
                "name": asset,
                "role": role,
                "sha256": spec["sha256"],
                "bytes": spec["bytes"],
                "source_path": spec["path"],
            }
        )

    for model in manifest["models"]:
        rights_name = f"rights-{model['selection']}.json"
        shutil.copyfile(RIGHTS_DIR / f"{model['selection']}.json", staging / rights_name)
        content = (staging / rights_name).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        sums.append(f"{digest}  {rights_name}")

    (staging / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    release_manifest = {
        "schema": "open_shogiai_release_assets/v1",
        "release_tag": tag,
        "repository": "TKY-27/OpenShogiAI",
        "engine": {name: manifest["engine"][name]["sha256"] for name in ("js", "wasm")},
        "profile": manifest["profile"]["sha256"],
        "models": [
            {
                "selection": model["selection"],
                "publicName": model["public_name"],
                "generation": model["generation"],
                "releaseId": model["release_id"],
                "asset": model["weights"]["asset"],
                "sha256": model["weights"]["sha256"],
                "bytes": model["weights"]["bytes"],
                "rights": f"rights-{model['selection']}.json",
                "provenance": model["provenance_note"],
            }
            for model in manifest["models"]
        ],
        "assets": index_entries,
        "note": "Generation order is newest to earliest; not a strength ranking.",
    }
    (staging / "release-manifest.json").write_text(
        json.dumps(release_manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(RELEASE_README.format(tag=tag), encoding="utf-8")
    print(f"staged {len(index_entries)} assets + rights records -> {staging}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
