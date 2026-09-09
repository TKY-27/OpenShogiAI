"""Reproducible, opt-in production artifact audit. Never trains or accesses holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import selectors
import struct
import subprocess
import sys
import time
from pathlib import Path

PROHIBITED = (
    "open_shogi_core::evaluation::",
    "open_shogi_core::champion::",
    "open_shogi_core::opening::",
    "residual::",
    "composite::",
    "open_shogi_core::neural::",
    "NeuralEvaluator",
    "TeacherAdapter",
    "Ibisya",
    "evaluate_handcrafted",
    "overall_champion_evaluation",
)
EXCLUDED_SOURCES = {"evaluation.rs", "champion.rs", "opening.rs", "neural.rs"}
FORBIDDEN_COUNTERS = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)

NEGATIVE_IMPORTS = [
    "overall_champion_evaluation",
    "OpeningBookV2",
    "EvaluationConfig",
    "OpeningPolicy",
    "NeuralEvaluationMode",
    "NeuralEvaluator",
]
NEGATIVE_CASES = [
    "empty",
    "truncated",
    "corrupt",
    "wrong_schema",
    "unsupported_version",
    "unsupported_quantization",
    "nan_weights",
    "extreme_finite_weights",
    "missing",
    "hash",
    "profile",
]


def portable_text(value: str, root: Path) -> str:
    """Project paths take precedence over the containing home path; raw logs stay intact."""
    normalized = value.replace(str(root), "${REPO}").replace(str(Path.home()), "${HOME}")
    for placeholder in ("REPO", "HOME"):
        normalized = normalized.replace(
            "file" + "://${" + placeholder + "}", "${" + placeholder + "_URI}"
        )
    return normalized


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_prohibited(data: bytes, label: str) -> dict:
    hits = [word for word in PROHIBITED if word.encode() in data]
    hits += [
        match.decode(errors="replace")
        for match in re.findall(
            rb"open_shogi_core(?:10evaluation|8champion|7opening|6neural)", data
        )
    ]
    if hits:
        raise ValueError(f"{label}: prohibited implementation markers: {hits}")
    return {"label": label, "bytes": len(data), "prohibited_hits": hits}


def verify_counters(value: dict) -> dict:
    """Missing counters fail; integer zero cannot be substituted by booleans."""
    for key in FORBIDDEN_COUNTERS:
        if type(value.get(key)) is not int or value[key] != 0:
            raise ValueError(f"invalid pure-only runtime counter {key}: {value.get(key)!r}")
    if type(value.get("learned_eval_calls")) is not int or value["learned_eval_calls"] <= 0:
        raise ValueError("learned_eval_calls must be positive")
    return {key: value[key] for key in (*FORBIDDEN_COUNTERS, "learned_eval_calls")}


def wasm_sections(data: bytes) -> list[dict]:
    if data[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("invalid Wasm header")
    offset = 8
    result = []
    while offset < len(data):
        kind = data[offset]
        offset += 1
        size = 0
        for shift in range(0, 35, 7):
            if offset >= len(data):
                raise ValueError("truncated Wasm section length")
            byte = data[offset]
            offset += 1
            size |= (byte & 127) << shift
            if byte < 128:
                break
        else:
            raise ValueError("oversized Wasm section length")
        end = offset + size
        if end > len(data):
            raise ValueError("truncated Wasm section")
        payload = data[offset:end]
        scan_prohibited(payload, f"Wasm section {kind}")
        row = {"id": kind, "size": size, "sha256": hashlib.sha256(payload).hexdigest()}
        row.update(decode_section(kind, payload))
        result.append(row)
        offset = end
    if not any(row["id"] == 10 for row in result):
        raise ValueError("Wasm artifact has no code section")
    return result


class WasmReader:
    def __init__(self, data: bytes):
        self.data, self.offset = data, 0

    def number(self) -> int:
        value = 0
        for shift in range(0, 35, 7):
            byte = self.byte()
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("invalid Wasm integer")

    def byte(self) -> int:
        if self.offset >= len(self.data):
            raise ValueError("truncated Wasm field")
        value = self.data[self.offset]
        self.offset += 1
        return value

    def text(self) -> str:
        size = self.number()
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("truncated Wasm string")
        value = self.data[self.offset : end].decode("utf8")
        self.offset = end
        return value

    def limits(self) -> None:
        flags = self.number()
        if flags not in (0, 1, 2, 3):
            raise ValueError("unsupported Wasm memory limits")
        self.number()
        if flags & 1:
            self.number()


def decode_section(kind: int, payload: bytes) -> dict:
    reader = WasmReader(payload)
    if kind == 0:
        name = reader.text()
        return {"custom_name": name}
    if kind == 2:
        imports = []
        for _ in range(reader.number()):
            module, name, descriptor = reader.text(), reader.text(), reader.byte()
            imports.append({"module": module, "name": name, "kind": descriptor})
            if descriptor == 0:
                reader.number()
            elif descriptor == 1:
                reader.byte()
                reader.limits()
            elif descriptor == 2:
                reader.limits()
            elif descriptor == 3:
                reader.byte()
                reader.byte()
            else:
                raise ValueError("unsupported Wasm import kind")
        return {"imports": imports}
    if kind == 7:
        return {
            "exports": [
                {"name": reader.text(), "kind": reader.byte(), "index": reader.number()}
                for _ in range(reader.number())
            ]
        }
    if kind in (3, 10, 11):
        return {
            {3: "function_count", 10: "code_count", 11: "data_segment_count"}[kind]: reader.number()
        }
    return {}


class Audit:
    def __init__(self, root: Path, output: Path):
        self.root, self.output = root, output
        output.mkdir(parents=True, exist_ok=False)
        self.commands: list[dict] = []
        self.env = dict(os.environ, CARGO_TARGET_DIR=str(output / "target"))

    def run(self, args: list[str], *, success: bool = True, stdin: str | None = None) -> str:
        result = subprocess.run(
            args,
            cwd=self.root,
            env=self.env,
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
        log = self.output / f"command-{len(self.commands):03}.log"
        log.write_text(result.stdout)
        self.commands.append(
            {"argv": args, "returncode": result.returncode, "log": log.name, "sha256": digest(log)}
        )
        if (result.returncode == 0) != success:
            raise ValueError(f"unexpected command result {result.returncode}: {args}; see {log}")
        return result.stdout

    def negative_imports(self) -> list[str]:
        directory = self.output / "negative-imports"
        (directory / "src").mkdir(parents=True)
        (directory / "Cargo.toml").write_text(
            '[package]\nname="pure-import-test"\nversion="0.0.0"\nedition="2024"\n'
            "[workspace]\n[dependencies]\nopen-shogi-core={path="
            + json.dumps(str(self.root / "engine/core"))
            + ',default-features=false,features=["pure-only"]}\n'
        )
        forbidden = NEGATIVE_IMPORTS
        for symbol in forbidden:
            (directory / "src/main.rs").write_text(
                f"use open_shogi_core::{symbol}; fn main() {{}}\n"
            )
            output = self.run(
                ["cargo", "check", "--offline", "--manifest-path", str(directory / "Cargo.toml")],
                success=False,
            )
            if "unresolved import" not in output or symbol not in output:
                raise ValueError(f"negative import failed for unrelated reason: {symbol}")
        manifest = directory / "Cargo.toml"
        manifest.write_text(
            manifest.read_text().replace('features=["pure-only"]', 'features=["handcrafted"]')
        )
        (directory / "src/main.rs").write_text(
            "#![allow(unused_imports)]\nuse open_shogi_core::{"
            + ",".join(forbidden)
            + "}; fn main() {}\n"
        )
        # Separate target: full-control depfiles must never contaminate pure source evidence.
        previous = self.env["CARGO_TARGET_DIR"]
        self.env["CARGO_TARGET_DIR"] = str(self.output / "full-import-target")
        try:
            self.run(["cargo", "check", "--offline", "--manifest-path", str(manifest)])
        finally:
            self.env["CARGO_TARGET_DIR"] = previous
        return forbidden

    def usi(self, args: list[str]) -> str:
        process = subprocess.Popen(
            args,
            cwd=self.root,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            b"usi\nisready\nsetoption name RuntimeProfile value pure_learned\n"
            b"position startpos\ngo depth 1 nodes 32\n"
        )
        process.stdin.flush()
        output = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 30
        try:
            while b"\nbestmove " not in output:
                if time.monotonic() >= deadline:
                    raise ValueError("USI bounded search timed out")
                if selector.select(timeout=1):
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        raise ValueError("USI exited before bestmove")
                    output.extend(chunk)
            process.stdin.write(b"quit\n")
            process.stdin.flush()
            tail, _ = process.communicate(timeout=10)
            output.extend(tail)
            if process.returncode != 0:
                raise ValueError("USI failed")
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait()
            log = self.output / f"command-{len(self.commands):03}.log"
            log.write_bytes(output)
            self.commands.append(
                {
                    "argv": args,
                    "returncode": process.returncode,
                    "log": log.name,
                    "sha256": digest(log),
                }
            )
        return output.decode()

    def runtime(self) -> dict:
        from open_shogi_training.phase10r_model import HistoryFacts
        from open_shogi_training.phase10t_model import Phase10TModel, Phase10TModelError

        model = Phase10TModel.random(20260907)
        model.head_weight[:, 0] *= 10000
        model.head_bias[0] *= 10000
        model_path = self.output / "synthetic-a1.bin"
        model_hash = model.write(model_path)
        binary = str(self.output / "target/release/open-shogi-cli")
        base = [
            "--model",
            str(model_path),
            "--model-sha256",
            model_hash,
            "--profile",
            "pure_learned",
        ]
        positions = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/9/9/4P4/9/9/9/4K4 b - 1",
            "4k4/9/9/9/4P4/9/9/9/4K4 w - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b Pp 1",
        ]
        default_history = {
            "available": False,
            "repetitionCount": 1,
            "continuousCheckByUs": False,
            "continuousCheckByThem": False,
        }
        fixtures = [{"sfen": sfen, "history": default_history} for sfen in positions]
        for count, owner in (
            (1, "neither"),
            (2, "neither"),
            (3, "neither"),
            (4, "neither"),
            (2, "us"),
            (3, "them"),
        ):
            fixtures.append(
                {
                    "sfen": positions[2 if owner == "us" else 1],
                    "history": {
                        "available": True,
                        "repetitionCount": count,
                        "continuousCheckByUs": owner == "us",
                        "continuousCheckByThem": owner == "them",
                    },
                }
            )
        native = []
        references = []
        for fixture in fixtures:
            sfen, history = fixture["sfen"], fixture["history"]
            result = json.loads(
                self.run(
                    [
                        binary,
                        "pure",
                        *base,
                        "--sfen",
                        sfen,
                        "--nodes",
                        "32",
                        "--depth",
                        "1",
                        "--history-json",
                        json.dumps(history),
                    ]
                )
            )
            verify_counters(result["proof"])
            verify_registrations(result["compiled_evaluators"])
            if result["best_move"] is None:
                raise ValueError("synthetic non-terminal search produced no move")
            facts = HistoryFacts(
                history["available"],
                history["repetitionCount"],
                history["continuousCheckByUs"],
                history["continuousCheckByThem"],
            )
            score, logits = model.evaluate(sfen, facts)
            cp = quantize_cp(score)
            reference = {"cp": cp, "wdl_logits": [float(x) for x in logits]}
            verify_parity(reference, result)
            references.append(reference)
            native.append(result)
        data = model_path.read_bytes()
        variants = {
            "empty": b"",
            "truncated": data[:100],
            "corrupt": data[:-1] + bytes([data[-1] ^ 1]),
        }
        for name, offset, replacement in (
            ("wrong_schema", 12, struct.pack("<I", 999)),
            ("unsupported_version", 8, struct.pack("<I", 999)),
            ("unsupported_quantization", 40, struct.pack("<f", 2.0)),
            ("nan_weights", 44, struct.pack("<f", math.nan)),
            ("extreme_finite_weights", 44, struct.pack("<f", 3.4028234663852886e38)),
        ):
            changed = bytearray(data)
            changed[offset : offset + 4] = replacement
            changed[-32:] = hashlib.sha256(changed[44:-32]).digest()
            variants[name] = bytes(changed)
        bad_models = []
        for name, content in variants.items():
            try:
                Phase10TModel.from_bytes(content)
            except Phase10TModelError:
                pass
            else:
                raise ValueError(f"Python accepted invalid model: {name}")
            path = self.output / f"invalid-{name}.bin"
            path.write_bytes(content)
            bad_models.append({"name": name, "path": str(path), "sha256": digest(path)})
        for command in ("pure", "usi"):
            for bad in bad_models:
                self.run(
                    [
                        binary,
                        command,
                        "--model",
                        bad["path"],
                        "--model-sha256",
                        bad["sha256"],
                        "--profile",
                        "pure_learned",
                    ],
                    success=False,
                )
            self.run(
                [
                    binary,
                    command,
                    "--model",
                    str(self.output / "missing.bin"),
                    "--model-sha256",
                    model_hash,
                    "--profile",
                    "pure_learned",
                ],
                success=False,
            )
            self.run(
                [
                    binary,
                    command,
                    "--model",
                    str(model_path),
                    "--model-sha256",
                    "0" * 64,
                    "--profile",
                    "pure_learned",
                ],
                success=False,
            )
            self.run([binary, command, *base[:-1], "standard"], success=False)
        usi = self.usi([binary, "usi", *base])
        if "readyok" not in usi or "bestmove " not in usi:
            raise ValueError("USI readiness or legal search failed")
        proofs = [
            json.loads(line.split("pure-proof ", 1)[1])
            for line in usi.splitlines()
            if line.startswith("info string pure-proof ")
        ]
        if not proofs:
            raise ValueError("USI runtime proof missing")
        for proof in proofs:
            verify_counters(proof)
        invalid_histories = [
            default_history | {"repetitionCount": 0},
            default_history | {"available": True, "repetitionCount": 5},
            default_history | {"repetitionCount": 2},
            default_history
            | {"available": True, "continuousCheckByUs": True, "continuousCheckByThem": True},
        ]
        for history in invalid_histories:
            self.run([binary, "pure", *base, "--history-json", json.dumps(history)], success=False)
            facts = HistoryFacts(
                history["available"],
                history["repetitionCount"],
                history["continuousCheckByUs"],
                history["continuousCheckByThem"],
            )
            try:
                facts.validate()
            except ValueError:
                pass
            else:
                raise ValueError("Python accepted invalid history")
        request = self.output / "wasm-request.json"
        request.write_text(
            json.dumps(
                {
                    "fixtures": fixtures,
                    "invalidHistories": invalid_histories,
                    "model": str(model_path),
                    "hash": model_hash,
                    "invalid": bad_models,
                }
            )
        )
        (self.output / "package.json").write_text('{"type":"module"}\n')
        harness = self.output / "wasm-parity.mjs"
        harness.write_text("""
import fs from 'node:fs';
import {PureEngine, initSync} from './wasm-one/open_shogi_wasm.js';
initSync({module: fs.readFileSync(new URL('./wasm-one/open_shogi_wasm_bg.wasm', import.meta.url))});
const req = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const bytes = fs.readFileSync(req.model);
const model = new PureEngine(bytes, req.hash, 'pure_learned');
const outputs = req.fixtures.map(f => ({
  evaluate: JSON.parse(model.evaluate_history(f.sfen, JSON.stringify(f.history))),
  search: JSON.parse(model.search(f.sfen, 1, 32))}));
let invalidHistoryCount = 0;
for (const history of req.invalidHistories) {
  let failed = false;
  try { model.evaluate_history(req.fixtures[0].sfen, JSON.stringify(history)); }
  catch { failed = true; }
  if (!failed) throw Error('invalid history accepted');
  invalidHistoryCount++;
}
model.free();
let failures = [];
function reject(name, bytes, hash, profile) {
  let rejected = false;
  try { let m = new PureEngine(bytes, hash, profile); m.free(); } catch { rejected = true; }
  if (!rejected) throw Error('invalid model accepted: ' + name);
  failures.push(name);
}
for (const bad of req.invalid) {
  reject(bad.name, fs.readFileSync(bad.path), bad.sha256, 'pure_learned');
}
reject('hash', bytes, '0'.repeat(64), 'pure_learned');
reject('profile', bytes, req.hash, 'standard');
console.log(JSON.stringify({outputs, failures, invalidHistoryCount}));
""")
        wasm = json.loads(self.run(["node", str(harness), str(request)]))
        for reference, result in zip(references, wasm["outputs"], strict=True):
            verify_parity(reference, result["evaluate"])
            verify_registrations(result["evaluate"]["compiled_evaluators"])
            verify_counters(result["search"]["proof"])
            if result["search"]["best_move"] is None:
                raise ValueError("Wasm legal search failed")
        return {
            "model_sha256": model_hash,
            "model_path": str(model_path.relative_to(self.root)),
            "model_provenance": "deterministic random; no training; synthetic positions only",
            "fixtures": fixtures,
            "invalid_histories": invalid_histories,
            "native": native,
            "python": references,
            "wasm": wasm,
            "usi_proofs": proofs,
            "negative_model_cases": [row["name"] for row in bad_models]
            + ["missing", "hash", "profile"],
        }

    def build(self) -> dict:
        packages = ("open-shogi-cli", "open-shogi-wasm")
        graphs = {}
        for package in packages:
            features = ["--no-default-features", "--features", "pure-only"]
            graphs[package] = self.run(
                ["cargo", "tree", "--locked", "-p", package, "-e", "features", *features]
            )
            previous = self.env["CARGO_TARGET_DIR"]
            self.env["CARGO_TARGET_DIR"] = str(self.output / "incompatible-target")
            try:
                for arguments, message in (
                    (["--no-default-features"], "select exactly one"),
                    (["--features", "pure-only"], "mutually exclusive"),
                ):
                    output = self.run(
                        ["cargo", "check", "--locked", "-p", package, *arguments], success=False
                    )
                    if message not in output:
                        raise ValueError("feature rejection failed for unrelated reason")
            finally:
                self.env["CARGO_TARGET_DIR"] = previous
        metadata = json.loads(
            self.run(
                [
                    "cargo",
                    "metadata",
                    "--locked",
                    "--format-version",
                    "1",
                    "--no-default-features",
                    "--features",
                    "open-shogi-cli/pure-only,open-shogi-wasm/pure-only",
                ]
            )
        )
        for node in metadata["resolve"]["nodes"]:
            if any(
                f"#{package}@" in node["id"]
                for package in (
                    "open-shogi-core",
                    "open-shogi-cli",
                    "open-shogi-usi",
                    "open-shogi-wasm",
                )
            ) and node["features"] != ["pure-only"]:
                raise ValueError("package metadata exposes non-pure features")
        linkmap = self.output / "native.map"
        linker_arg = f"-Wl,-map,{linkmap}" if sys.platform == "darwin" else f"-Wl,-Map,{linkmap}"
        self.run(
            [
                "cargo",
                "rustc",
                "--locked",
                "--release",
                "-p",
                "open-shogi-cli",
                "--no-default-features",
                "--features",
                "pure-only",
                "--",
                "-C",
                "strip=none",
                "-C",
                f"link-arg={linker_arg}",
            ]
        )
        cfg = self.run(
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
            raise ValueError("incorrect core rustc cfg")
        binary = self.output / "target/release/open-shogi-cli"
        symbols = self.run(["nm", "-C", str(binary)])
        if "open_shogi_core::phase10t::" not in symbols:
            raise ValueError("native symbol dump is stripped or not Rust-demangled")
        scan_prohibited(symbols.encode(), "native nm defined and undefined")
        scan_prohibited(linkmap.read_bytes(), "native link map")
        self.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "open-shogi-wasm",
                "--target",
                "wasm32-unknown-unknown",
                "--no-default-features",
                "--features",
                "pure-only",
            ]
        )
        raw = self.output / "target/wasm32-unknown-unknown/release/open_shogi_wasm.wasm"
        raw_sections = wasm_sections(raw.read_bytes())
        if not any(row.get("custom_name") == "name" for row in raw_sections):
            raise ValueError("unstripped Wasm name section required")
        bindgen = self.root / "local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen"
        generated = []
        for suffix in ("wasm-one", "wasm-two"):
            if suffix == "wasm-two":
                before = digest(raw)
                self.run(
                    [
                        "cargo",
                        "clean",
                        "-p",
                        "open-shogi-wasm",
                        "--release",
                        "--target",
                        "wasm32-unknown-unknown",
                    ]
                )
                self.run(
                    [
                        "cargo",
                        "build",
                        "--locked",
                        "--release",
                        "-p",
                        "open-shogi-wasm",
                        "--target",
                        "wasm32-unknown-unknown",
                        "--no-default-features",
                        "--features",
                        "pure-only",
                    ]
                )
                if digest(raw) != before:
                    raise ValueError("Wasm recompilation is not deterministic")
            directory = self.output / suffix
            self.run(
                [
                    str(bindgen),
                    "--target",
                    "web",
                    "--typescript",
                    "--out-dir",
                    str(directory),
                    str(raw),
                ]
            )
            generated.append({p.name: digest(p) for p in sorted(directory.iterdir())})
        if generated[0] != generated[1]:
            raise ValueError("Wasm binding regeneration is not deterministic")
        final = self.output / "wasm-one/open_shogi_wasm_bg.wasm"
        depfiles = [
            path
            for path in (self.output / "target").rglob("*.d")
            if path.name.startswith(
                ("open_shogi_core", "open_shogi_cli", "open_shogi_usi", "open_shogi_wasm")
            )
        ]
        if not depfiles:
            raise ValueError("missing core source dependency evidence")
        for depfile in depfiles:
            names = set(re.findall(r"[A-Za-z_]+\.rs", depfile.read_text()))
            if any(
                path in depfile.read_text()
                for path in (
                    "engine/cli/src/dataset.rs",
                    "engine/cli/src/arena.rs",
                    "engine/wasm/src/full.rs",
                )
            ):
                raise ValueError("development adapter source compiled into pure artifact")
            if names & EXCLUDED_SOURCES:
                raise ValueError(f"prohibited source compiled: {names & EXCLUDED_SOURCES}")
        return {
            "cargo_features": ["pure-only"],
            "graphs": graphs,
            "metadata": metadata,
            "rustc_cfg": cfg,
            "native_sha256": digest(binary),
            "compiled_evaluator_registrations": ["osaval02", "phase10t-a1"],
            "native_link_map_sha256": digest(linkmap),
            "wasm_raw_sha256": digest(raw),
            "wasm_sha256": digest(final),
            "wasm_raw_sections": raw_sections,
            "wasm_sections": wasm_sections(final.read_bytes()),
            "deterministic_wasm_regeneration": generated,
            "source_depfiles": {str(p.relative_to(self.output)): digest(p) for p in depfiles},
        }


def quantize_cp(score: float) -> int:
    return max(-28999, min(28999, int(math.copysign(math.floor(abs(score) + 0.5), score))))


def verify_proof_identity(proof: dict, model_hash: str, profile_hash: str) -> None:
    if (
        proof.get("model_sha256") != model_hash
        or proof.get("evaluator_profile_schema_hash") != profile_hash
        or proof.get("profile") != "pure_learned"
        or proof.get("profile_schema") != "open_shogiai_pure_learned_a1_profile/v1"
    ):
        raise ValueError("runtime proof identity mismatch")


def verify_registrations(value: list) -> None:
    if value != ["osaval02", "phase10t-a1"]:
        raise ValueError("unexpected compiled evaluator registrations")


def verify_parity(reference: dict, actual: dict) -> None:
    if actual["cp"] != reference["cp"]:
        raise ValueError("a1 cp parity mismatch")
    for expected, observed in zip(reference["wdl_logits"], actual["wdl_logits"], strict=True):
        if not math.isfinite(observed) or abs(expected - observed) > 1e-5:
            raise ValueError("a1 WDL parity mismatch")


def source_inventory(root: Path) -> dict:
    output = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-co",
            "--exclude-standard",
            "--",
            "engine",
            "training",
            "scripts",
            "configs/runtime",
            "Cargo.toml",
            "Cargo.lock",
        ],
        cwd=root,
        text=True,
    )
    return {
        name: digest(root / name)
        for name in sorted(set(output.splitlines()))
        if (root / name).is_file()
    }


def verify_evidence(root: Path, receipt: dict) -> None:
    if (
        receipt.get("schema") != "open_shogiai_phase10t_pure_artifact_audit/v1"
        or receipt.get("status") != "passed"
    ):
        raise ValueError("pure artifact audit did not pass")
    if (
        receipt.get("git_status") != ""
        or re.fullmatch(r"[0-9a-f]{40}", receipt.get("git_commit", "")) is None
    ):
        raise ValueError("audit must identify a clean committed runtime")
    if not receipt.get("source_hashes") or source_inventory(root) != receipt["source_hashes"]:
        raise ValueError("runtime source changed since artifact audit")
    artifacts = receipt.get("artifacts", {})
    if set(artifacts) != {"native", "wasm_raw", "wasm", "model"}:
        raise ValueError("required artifacts missing")
    for item in artifacts.values():
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts or digest(root / path) != item["sha256"]:
            raise ValueError("artifact changed or path is not repository-relative")
    if set(receipt.get("negative_imports", [])) != set(NEGATIVE_IMPORTS):
        raise ValueError("missing negative import evidence")
    profile_hash = digest(root / "configs/runtime/pure_learned-a1-v1.json")
    if receipt.get("evaluator_profile_schema_hash") != profile_hash:
        raise ValueError("profile schema digest mismatch")
    if receipt.get("cargo_features") != ["pure-only"]:
        raise ValueError("incorrect feature class")
    verify_registrations(receipt["compiled_evaluator_registrations"])
    if not receipt.get("source_depfiles") or not receipt.get("commands"):
        raise ValueError("missing source or command audit evidence")
    output_directory = Path(receipt["output_directory"])
    if output_directory.is_absolute() or ".." in output_directory.parts:
        raise ValueError("invalid audit output path")
    if digest(root / output_directory / "native.map") != receipt["native_link_map_sha256"]:
        raise ValueError("native link map changed")
    generated = receipt["deterministic_wasm_regeneration"]
    expected_names = {
        "open_shogi_wasm.js",
        "open_shogi_wasm.d.ts",
        "open_shogi_wasm_bg.wasm",
        "open_shogi_wasm_bg.wasm.d.ts",
    }
    if len(generated) != 2 or generated[0] != generated[1] or set(generated[0]) != expected_names:
        raise ValueError("invalid deterministic Wasm evidence")
    for directory, expected in zip(("wasm-one", "wasm-two"), generated, strict=True):
        current = {
            path.name: digest(path)
            for path in (root / output_directory / directory).iterdir()
            if path.is_file()
        }
        if current != expected:
            raise ValueError("generated Wasm files changed")
    for name, sha256 in receipt["source_depfiles"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid source depfile path")
        if digest(root / output_directory / path) != sha256:
            raise ValueError("source depfile changed")
    verified_fields = set()
    for command in receipt["commands"]:
        log = Path(command["log"])
        if log.is_absolute() or ".." in log.parts:
            raise ValueError("invalid command log path")
        if digest(root / output_directory / log) != command["sha256"]:
            raise ValueError("command log digest mismatch")
        text = portable_text((root / output_directory / log).read_text(), root)
        argv = command["argv"]
        if argv[:2] == ["cargo", "tree"]:
            package = argv[argv.index("-p") + 1]
            if text != receipt["graphs"][package]:
                raise ValueError("feature graph does not match recorded command")
            verified_fields.add(package)
        elif argv[:2] == ["cargo", "metadata"]:
            if json.loads(text) != receipt["metadata"]:
                raise ValueError("metadata does not match recorded command")
            verified_fields.add("metadata")
        elif argv[-2:] == ["--print", "cfg"]:
            if text != receipt["rustc_cfg"]:
                raise ValueError("rustc cfg does not match recorded command")
            verified_fields.add("cfg")
    if verified_fields != {"open-shogi-cli", "open-shogi-wasm", "metadata", "cfg"}:
        raise ValueError("feature graph or compiler cfg evidence missing")
    for kind, section_key in (("wasm_raw", "wasm_raw_sections"), ("wasm", "wasm_sections")):
        if wasm_sections((root / artifacts[kind]["path"]).read_bytes()) != receipt[section_key]:
            raise ValueError("Wasm section audit mismatch")
    runtime = receipt["runtime"]
    if runtime["model_sha256"] != artifacts["model"]["sha256"]:
        raise ValueError("model digest mismatch")
    if set(runtime["wasm"]["failures"]) != set(NEGATIVE_CASES) - {"missing"}:
        raise ValueError("missing Wasm negative loading evidence")
    if set(runtime["negative_model_cases"]) != set(NEGATIVE_CASES):
        raise ValueError("missing negative model evidence")
    if len(runtime["native"]) != 10 or len(runtime["wasm"]["outputs"]) != 10:
        raise ValueError("missing runtime parity coverage")
    if runtime["wasm"].get("invalidHistoryCount") != 4 or len(runtime["invalid_histories"]) != 4:
        raise ValueError("missing invalid history evidence")
    if not runtime["usi_proofs"]:
        raise ValueError("missing USI proof")
    for proof in runtime["usi_proofs"]:
        verify_counters(proof)
        verify_proof_identity(proof, runtime["model_sha256"], profile_hash)
    for reference, native, wasm in zip(
        runtime["python"], runtime["native"], runtime["wasm"]["outputs"], strict=True
    ):
        verify_counters(native["proof"])
        verify_counters(wasm["search"]["proof"])
        verify_proof_identity(native["proof"], runtime["model_sha256"], profile_hash)
        verify_proof_identity(wasm["search"]["proof"], runtime["model_sha256"], profile_hash)
        verify_registrations(native["compiled_evaluators"])
        verify_registrations(wasm["evaluate"]["compiled_evaluators"])
        verify_parity(reference, native)
        verify_parity(reference, wasm["evaluate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    audit = Audit(root, args.output.resolve())
    evidence = {"schema": "open_shogiai_phase10t_pure_artifact_audit/v1", "status": "failed"}
    try:
        evidence["output_directory"] = str(audit.output.relative_to(root))
        evidence["evaluator_profile_schema_hash"] = digest(
            root / "configs/runtime/pure_learned-a1-v1.json"
        )
        evidence["git_commit"] = audit.run(["git", "rev-parse", "HEAD"]).strip()
        evidence["git_status"] = audit.run(["git", "status", "--porcelain"])
        if evidence["git_status"]:
            raise ValueError("formal artifact audit requires a clean committed worktree")
        evidence["source_hashes"] = source_inventory(root)
        evidence.update(audit.build())
        evidence["negative_imports"] = audit.negative_imports()
        evidence["runtime"] = audit.runtime()
        evidence["artifacts"] = {
            key: {"path": str(path.relative_to(root)), "sha256": digest(path)}
            for key, path in {
                "native": audit.output / "target/release/open-shogi-cli",
                "wasm_raw": audit.output
                / "target/wasm32-unknown-unknown/release/open_shogi_wasm.wasm",
                "wasm": audit.output / "wasm-one/open_shogi_wasm_bg.wasm",
                "model": audit.output / "synthetic-a1.bin",
            }.items()
        }
        candidate = evidence | {"status": "passed", "commands": audit.commands}
        candidate = json.loads(portable_text(json.dumps(candidate), root))
        verify_evidence(root, candidate)
        evidence["status"] = "passed"
    finally:
        evidence["commands"] = audit.commands
        (audit.output / "audit.json").write_text(
            portable_text(json.dumps(evidence, indent=2), root) + "\n"
        )


if __name__ == "__main__":
    main()
