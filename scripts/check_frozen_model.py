"""Verify the optional local comparison model with bounded native and actual-Wasm calls."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)


def main() -> None:
    catalog = json.loads((ROOT / "configs/models/registry.json").read_text())
    model = catalog["models"][0]
    relative = Path(model["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("model path must remain inside this repository")
    path = ROOT / relative
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError("model path must not use symlinks")
    if path.stat().st_size != model["bytes"]:
        raise ValueError("frozen model size mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]:
        raise ValueError("frozen model SHA-256 mismatch")
    profile = ROOT / model["profile"]
    if hashlib.sha256(profile.read_bytes()).hexdigest() != model["profile_sha256"]:
        raise ValueError("frozen profile SHA-256 mismatch")
    frozen = json.loads((ROOT / "local/frozen/manifest.json").read_text())
    if frozen["baseline"]["model"]["sha256"] != model["sha256"]:
        raise ValueError("local manifest and comparison registry disagree")
    binary = ROOT / "target/pure/release/open-shogi-cli"
    if not binary.is_file():
        raise SystemExit("Run make pure-build first")
    result = subprocess.run(
        [
            str(binary),
            "pure",
            "--model",
            str(path),
            "--model-sha256",
            model["sha256"],
            "--model-format",
            "OSAVAL03",
            "--profile",
            "pure_learned",
            "--sfen",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "--nodes",
            "64",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    native = json.loads(result.stdout)
    proof = native["proof"]
    if (
        proof["model_sha256"] != model["sha256"]
        or proof["learned_eval_calls"] <= 0
        or not native["best_move"]
        or any(proof[key] != 0 for key in FORBIDDEN)
    ):
        raise ValueError("native pure runtime proof failed")
    wasm = subprocess.run(
        [
            "node",
            "scripts/phase10v_browser_contract.mjs",
            "target/pure/bindings",
            str(path),
            model["sha256"],
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    contract = json.loads(wasm.stdout)
    if contract["status"] != "PASS" or contract["modelSha256"] != model["sha256"]:
        raise ValueError("Wasm model contract failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": model["id"],
                "native_best_move": native["best_move"],
                "wasm": contract["runtime"],
                "strength_evaluated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
