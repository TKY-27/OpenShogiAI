"""Bounded Phase 10V artifact audit. No labeling, Arena, self-play or holdout access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .phase10t_pure_build import (
    EXCLUDED_SOURCES,
    Audit,
    scan_prohibited,
    source_inventory,
    wasm_sections,
)

POSITIONS = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "4k4/9/9/9/4P4/9/9/9/4K4 b - 1",
    "4k4/9/9/9/4P4/9/9/9/4K4 w - 1",
    "4k4/9/9/9/9/9/9/9/4K4 b Pp 1",
    "4k4/9/2+p6/9/4+B4/9/9/9/4K4 w 2Pr 1",
]
FORBIDDEN = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_proof(value: dict, model_hash: str) -> None:
    proof = value["proof"]
    if (
        proof["model_sha256"] != model_hash
        or type(proof["learned_eval_calls"]) is not int
        or proof["learned_eval_calls"] <= 0
        or proof["profile"] != "pure_learned"
        or proof["profile_schema"] != "open_shogiai_pure_learned_v3_profile/v1"
    ):
        raise ValueError("unbound or empty learned runtime proof")
    if any(type(proof[name]) is not int or proof[name] != 0 for name in FORBIDDEN):
        raise ValueError("prohibited evaluator was called")
    if not value.get("best_move") or value["nodes"] <= 0:
        raise ValueError("short legal search returned no move")


def audit(
    root: Path, output: Path, *, build: bool = True, trained_models: tuple[Path, ...] = ()
) -> dict:
    from .phase10v_model import Phase10VModel

    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "local/phase10v-validation") or output.exists():
        raise ValueError("audit needs a fresh directory under local/phase10v-validation")
    if shutil.disk_usage(root).free < 80 * 1024**3:
        raise ValueError("80 GiB hard free-space floor")
    output.mkdir(parents=True)
    inventory = source_inventory(root)
    env = {**os.environ, "CARGO_TARGET_DIR": str(root / "target/phase10v-pure")}
    commands = []

    def run(args: list[str], *, success: bool = True, stdin: str | None = None) -> str:
        start = time.monotonic()
        process = subprocess.run(
            args,
            cwd=root,
            env=env,
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        text = process.stdout
        log = output / f"command-{len(commands):03d}.log"
        log.write_text(text)
        commands.append(
            {
                "argv": [a.replace(str(root), ".") for a in args],
                "returncode": process.returncode,
                "elapsed_seconds": time.monotonic() - start,
                "log": str(log.relative_to(root)),
                "sha256": digest(log),
            }
        )
        if (process.returncode == 0) != success:
            raise ValueError(f"unexpected command exit: {args[0]} (see {log.name})")
        return text

    if build:
        run(["./scripts/build_phase10v_pure.sh"])
    base = root / "target/phase10v-pure"
    binary = base / "release/open-shogi-cli"
    raw = base / "wasm32-unknown-unknown/release/open_shogi_wasm.wasm"
    wasm = base / "bindings/open_shogi_wasm_bg.wasm"
    symbols = run(["nm", "-C", str(binary)])
    if "open_shogi_core::phase10v::" not in symbols:
        raise ValueError("OSAVAL03 inference symbol absent")
    scan_prohibited(symbols.encode(), "native defined/undefined symbols")
    for path in (raw, wasm):
        scan_prohibited(path.read_bytes(), path.name)
        sections = wasm_sections(path.read_bytes())
        if not sections:
            raise ValueError("invalid Wasm artifact")
    dependencies = list((base / "release/deps").glob("open_shogi_core-*.d"))
    dependencies += list((base / "wasm32-unknown-unknown/release/deps").glob("open_shogi_core-*.d"))
    if len(dependencies) < 2:
        raise ValueError("missing native/Wasm source dependency evidence")
    for dependency in dependencies:
        text = dependency.read_text()
        if any(f"/src/{name}" in text for name in EXCLUDED_SOURCES):
            raise ValueError("prohibited source compiled into pure core")
    cfg = run(
        [
            "cargo",
            "rustc",
            "--locked",
            "-p",
            "open-shogi-core",
            "--no-default-features",
            "--features",
            "pure-only",
            "--",
            "--print",
            "cfg",
        ]
    )
    if 'feature="pure-only"' not in cfg or 'feature="handcrafted"' in cfg:
        raise ValueError("incorrect pure core feature selection")
    build_messages = run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "open-shogi-core",
            "--no-default-features",
            "--features",
            "pure-only",
            "--message-format=json",
        ]
    )
    libraries = []
    for line in build_messages.splitlines():
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("target", {}).get("name") == "open_shogi_core"
        ):
            libraries.extend(Path(p) for p in message["filenames"] if p.endswith(".rlib"))
    if len(libraries) != 1:
        raise ValueError("ambiguous pure core library identity")
    for symbol in ("EvaluationConfig", "NeuralEvaluator", "OpeningBookV2", "OpeningPolicy"):
        probe = output / f"forbidden-{symbol}.rs"
        probe.write_text(f"pub use open_shogi_core::{symbol};\n")
        rejection = run(
            [
                "rustc",
                "--edition=2024",
                "--crate-type=lib",
                "--crate-name=boundary_probe",
                str(probe),
                "--extern",
                f"open_shogi_core={libraries[0]}",
                "-L",
                f"dependency={base / 'release/deps'}",
                "--out-dir",
                str(output),
            ],
            success=False,
        )
        if "unresolved import" not in rejection or symbol not in rejection:
            raise ValueError("prohibited import failed for an unrelated reason")
    run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "open-shogi-core",
            "--example",
            "phase10u_replay",
            "--no-default-features",
            "--features",
            "pure-only",
        ]
    )
    oracle = base / "release/examples/phase10u_replay"
    positions = output / "positions.json"
    positions.write_text(json.dumps(POSITIONS))
    variants = []
    models = []
    for width in (256, 512):
        model = Phase10VModel.random(seed=20260908, width=width)
        model.head_weight[:, 0] *= 10
        model.head_bias[0] = 37.25
        path = output / f"diagnostic-{width}.osaval03"
        path.write_bytes(model.to_bytes())
        models.append(path)
    models.extend(path.resolve() for path in trained_models)
    for model_index, path in enumerate(models):
        model = Phase10VModel.read(path)
        width = model.width
        model_hash = digest(path)
        options = [
            "--model",
            str(path),
            "--model-sha256",
            model_hash,
            "--model-format",
            "OSAVAL03",
            "--profile",
            "pure_learned",
        ]
        native = []
        for sfen in POSITIONS:
            actual = json.loads(
                run(
                    [
                        str(binary),
                        "pure",
                        *options,
                        "--sfen",
                        sfen,
                        "--depth",
                        "2",
                        "--nodes",
                        "64",
                        "--leaf-trace-limit",
                        "64",
                    ]
                )
            )
            check_proof(actual, model_hash)
            if actual["proof"]["evaluator_profile_schema_hash"] != digest(
                root / "configs/runtime/pure_learned-v3.json"
            ):
                raise ValueError("runtime profile hash mismatch")
            if actual["cp"] != model.search_score(sfen):
                raise ValueError("Python/native direct cp mismatch")
            for leaf in actual["leaf_trace"]:
                if leaf["cp"] != model.search_score(leaf["sfen"]):
                    raise ValueError("actual incremental search-leaf/full Python cp mismatch")
            run(
                [str(oracle)],
                stdin=json.dumps(
                    {"initial_sfen": sfen, "moves": [actual["best_move"]], "max_plies": 256}
                ),
            )
            native.append(actual)
        observed = json.loads(
            run(
                [
                    "node",
                    str(root / "scripts/phase10v_wasm_probe.mjs"),
                    str(base / "bindings"),
                    str(path),
                    model_hash,
                    str(positions),
                ]
            )
        )
        for left, right in zip(native, observed["results"], strict=True):
            evaluation = right["evaluation"]
            if left["cp"] != evaluation["cp"] or any(
                abs(a - b) > 1e-5
                for a, b in zip(left["wdl_logits"], evaluation["wdl_logits"], strict=True)
            ):
                raise ValueError("native/Wasm score or WDL mismatch")
            check_proof(right["search"], model_hash)
            for key in ("score", "best_move", "nodes", "depth"):
                if left[key] != right["search"][key]:
                    raise ValueError(f"native/Wasm equal-node search mismatch: {key}")
        run([str(binary), "pure"], success=False)
        wrong_format = [str(binary), "pure", *options]
        wrong_format[wrong_format.index("OSAVAL03")] = "auto"
        run(wrong_format, success=False)
        missing = [str(binary), "pure", *options]
        missing[missing.index(str(path))] = str(output / "missing.osaval03")
        run(missing, success=False)
        for name, data in (
            ("empty", b""),
            ("truncated", path.read_bytes()[:-1]),
            ("corrupt", path.read_bytes()[:-32] + b"0" * 32),
        ):
            malformed = output / f"{model_index}-{width}-{name}.bin"
            malformed.write_bytes(data)
            run(
                [
                    str(binary),
                    "pure",
                    "--model",
                    str(malformed),
                    "--model-sha256",
                    digest(malformed),
                    "--profile",
                    "pure_learned",
                    "--model-format",
                    "OSAVAL03",
                ],
                success=False,
            )
        # Reuse the bounded transport helper, without any Phase10T architecture gates.
        usi_audit = Audit(root, output / f"usi-{model_index}")
        usi = usi_audit.usi([str(binary), "usi", *options])
        if "usiok" not in usi or "readyok" not in usi:
            raise ValueError("pure USI startup failed")
        usi_proof = next(
            json.loads(line.removeprefix("info string pure-proof "))
            for line in usi.splitlines()
            if line.startswith("info string pure-proof ")
        )
        usi_move = next(
            line.split()[1] for line in usi.splitlines() if line.startswith("bestmove ")
        )
        check_proof({"proof": usi_proof, "best_move": usi_move, "nodes": 1}, model_hash)
        run(
            [str(oracle)],
            stdin=json.dumps({"initial_sfen": POSITIONS[0], "moves": [usi_move], "max_plies": 256}),
        )
        for command in usi_audit.commands:
            command["argv"] = [a.replace(str(root), ".") for a in command["argv"]]
            command["log"] = str((usi_audit.output / command["log"]).relative_to(root))
            commands.append(command)
        browser = json.loads(
            run(
                [
                    "node",
                    str(root / "scripts/phase10v_browser_contract.mjs"),
                    str(base / "bindings"),
                    str(path),
                    model_hash,
                ]
            )
        )
        variants.append(
            {
                "width": width,
                "model": str(path.relative_to(root)),
                "sha256": model_hash,
                "bytes": path.stat().st_size,
                "native": native,
                "wasm": observed,
                "browser": browser,
                "usi_proof": usi_proof,
                "campaign_counted": False,
            }
        )
    if source_inventory(root) != inventory:
        raise ValueError("source changed during audit; run again after integration")
    result = {
        "schema": "open_shogiai_phase10v_artifact_audit/v1",
        "status": "PASS",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_hashes": inventory,
        "commands": commands,
        "variants": variants,
        "artifacts": {
            name: {"path": str(path.relative_to(root)), "sha256": digest(path)}
            for name, path in (("native", binary), ("wasm_raw", raw), ("wasm", wasm))
        },
        "prohibited_modules_absent": True,
        "actual_wasm_tested": True,
        "production_strength_proven": False,
        "campaign_started": False,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trained-model", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = audit(args.root, args.output, trained_models=tuple(args.trained_model))
    print(json.dumps({"status": result["status"], "commands": len(result["commands"])}))


if __name__ == "__main__":
    main()
