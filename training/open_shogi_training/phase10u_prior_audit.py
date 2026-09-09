"""Bounded exact-model pure native/Wasm audit; never launches Arena or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import phase10t_pure_build as base
from .phase10r_model import HistoryFacts, infer_osaval02, load_osaval02
from .phase10t_model import Phase10TModel

SCHEMA = "open_shogiai_phase10u_prior_adapter_audit/v1"
EXPECTED_MODELS = {
    "prior-1m": "41ce44a93219b0bcee0270677104c379029999050524fe99c1a560bb5cc05060",
    "a1-100k": "09f02be8c821417cc9a1edf0d6058bce9904ab41533997c11b300f6cce19ec22",
    "hard1": "88a1e7d46aac5584e535f39d157ea60003ce77c264ff2ee9a1654d87651f119c",
}
NEGATIVES = {
    "wrong_magic",
    "cross_format",
    "truncated",
    "corrupt",
    "wrong_schema",
    "hash",
    "profile",
}


def model_identity(root: Path, entrant: dict) -> dict:
    path = Path(entrant["model"]["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("model path must be repository relative")
    data = (root / path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != EXPECTED_MODELS.get(entrant["id"]) or sha != entrant["model"]["sha256"]:
        raise ValueError("frozen exact model identity mismatch")
    prior = entrant["id"] == "prior-1m"
    if prior:
        payload = load_osaval02(root / path).weight_payload_sha256
    else:
        Phase10TModel.from_bytes(data)
        payload = data[-32:].hex()
    profile = Path(
        "configs/runtime/pure_learned-v1.json"
        if prior
        else "configs/runtime/pure_learned-a1-v1.json"
    )
    return {
        "id": entrant["id"],
        "path": str(path),
        "sha256": sha,
        "payload_sha256": payload,
        "format": "OSAVAL02" if prior else "OSAT10A1",
        "profile_path": str(profile),
        "profile_sha256": base.digest(root / profile),
        "evaluator_schema_hash": base.digest(root / profile),
        "feature_schema_hash": data[96:128].hex() if prior else None,
        "profile_schema": json.loads((root / profile).read_text())["schema"],
    }


def verify_identity(proof: dict, identity: dict) -> None:
    base.verify_counters(proof)
    if (
        proof.get("profile") != "pure_learned"
        or proof.get("model_sha256") != identity["sha256"]
        or proof.get("evaluator_profile_schema_hash") != identity["profile_sha256"]
        or proof.get("profile_schema") != identity["profile_schema"]
    ):
        raise ValueError("format-specific proof identity mismatch")


def close_tree(expected, actual, *, rel_tol=1e-5, abs_tol=1e-5) -> None:
    """Compare independent inference including every policy logit, with float tolerance."""
    if isinstance(expected, dict):
        if "osaval02" in expected:
            rel_tol, abs_tol = 1e-10, 1e-12
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("inference fields mismatch")
        for key in expected:
            close_tree(expected[key], actual[key], rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError("inference shape mismatch")
        for left, right in zip(expected, actual, strict=True):
            close_tree(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (float, int))
            or not math.isfinite(actual)
            or not math.isclose(expected, actual, rel_tol=rel_tol, abs_tol=abs_tol)
        ):
            raise ValueError("inference numeric parity mismatch")
    elif expected != actual:
        raise ValueError("inference parity mismatch")


def runtime(audit: base.Audit, identities: list[dict], fixtures: list[dict]) -> list[dict]:
    binary = str(audit.output / "target/release/open-shogi-cli")
    harness = audit.output / "prior-parity.mjs"
    harness.write_text("""
import fs from 'node:fs';
import {PureEngine, initSync} from './wasm-one/open_shogi_wasm.js';
initSync({module:fs.readFileSync(new URL('./wasm-one/open_shogi_wasm_bg.wasm',import.meta.url))});
const req=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const bytes=fs.readFileSync(req.path);
const engine=PureEngine.new_with_format(bytes,req.sha256,'pure_learned',req.format);
const outputs=req.fixtures.map(f=>({
 evaluate:JSON.parse(engine.evaluate_history(f.sfen,JSON.stringify(f.history))),
 search:JSON.parse(engine.search(f.sfen,1,32))}));
engine.free();
const failures=[];
for(const bad of req.invalid){
 let rejected=false;
 try {
  const e=PureEngine.new_with_format(
   fs.readFileSync(bad.path),bad.sha256,bad.profile,bad.format);
  e.free();
 }
 catch {rejected=true;}
 if(!rejected) throw Error('invalid model accepted: '+bad.name);
 failures.push(bad.name);
}
console.log(JSON.stringify({outputs,failures}));
""")
    rows = []
    for identity in identities:
        path = audit.root / identity["path"]
        prior = identity["format"] == "OSAVAL02"
        model = load_osaval02(path) if prior else Phase10TModel.from_bytes(path.read_bytes())
        arguments = [
            "--model",
            str(path),
            "--model-sha256",
            identity["sha256"],
            "--profile",
            "pure_learned",
            "--model-format",
            identity["format"],
        ]
        native, references, native_logs = [], [], []
        for fixture in fixtures:
            result = json.loads(
                audit.run(
                    [
                        binary,
                        "pure",
                        *arguments,
                        "--depth",
                        "1",
                        "--nodes",
                        "32",
                        "--sfen",
                        fixture["sfen"],
                        "--history-json",
                        json.dumps(fixture["history"]),
                    ]
                )
            )
            history = fixture["history"]
            facts = HistoryFacts(
                history["available"],
                history["repetitionCount"],
                history["continuousCheckByUs"],
                history["continuousCheckByThem"],
            )
            if prior:
                moves = [row["move"] for row in result["inference"]["osaval02"]["legalMoves"]]
                inference = infer_osaval02(model, fixture["sfen"], moves, facts)
                reference = {"cp": inference["score"]["calibratedCp"], "osaval02": inference}
            else:
                score, logits = model.evaluate(fixture["sfen"], facts)
                reference = {
                    "cp": base.quantize_cp(score),
                    "wdl_logits": [float(x) for x in logits],
                }
            close_tree(reference, result["inference"])
            verify_identity(result["proof"], identity)
            native_logs.append(audit.commands[-1]["log"])
            native.append(result)
            references.append(reference)
        data = path.read_bytes()
        wrong_schema = bytearray(data)
        offset = 96 if prior else 12
        wrong_schema[offset : offset + 4] = b"\xff" * 4
        if prior:
            wrong_schema[-32:] = hashlib.sha256(wrong_schema[:-32]).digest()
        variants = {
            "wrong_magic": b"BADMAGIC" + data[8:],
            "truncated": data[:100],
            "corrupt": data[:-1] + bytes([data[-1] ^ 1]),
            "wrong_schema": bytes(wrong_schema),
        }
        invalid = []
        for name, content in variants.items():
            bad = audit.output / f"{identity['id']}-{name}.bin"
            bad.write_bytes(content)
            invalid.append(
                {
                    "name": name,
                    "path": str(bad),
                    "sha256": base.digest(bad),
                    "profile": "pure_learned",
                    "format": identity["format"],
                }
            )
        valid = {
            "path": str(path),
            "sha256": identity["sha256"],
            "profile": "pure_learned",
            "format": identity["format"],
        }
        invalid.extend(
            [
                valid | {"name": "cross_format", "format": "OSAT10A1" if prior else "OSAVAL02"},
                valid | {"name": "hash", "sha256": "0" * 64},
                valid | {"name": "profile", "profile": "standard"},
            ]
        )
        for command in ("pure", "usi"):
            for bad in invalid:
                audit.run(
                    [
                        binary,
                        command,
                        "--model",
                        bad["path"],
                        "--model-sha256",
                        bad["sha256"],
                        "--profile",
                        bad["profile"],
                        "--model-format",
                        bad["format"],
                    ],
                    success=False,
                )
        usi = audit.usi([binary, "usi", *arguments])
        usi_log = audit.commands[-1]["log"]
        proofs = [
            json.loads(line.split("pure-proof ", 1)[1])
            for line in usi.splitlines()
            if line.startswith("info string pure-proof ")
        ]
        if not proofs:
            raise ValueError("USI proof missing")
        for proof in proofs:
            verify_identity(proof, identity)
        request = audit.output / f"{identity['id']}-request.json"
        request.write_text(
            json.dumps(identity | {"path": str(path), "fixtures": fixtures, "invalid": invalid})
        )
        wasm = json.loads(audit.run(["node", str(harness), str(request)]))
        wasm_log = audit.commands[-1]["log"]
        for reference, output in zip(references, wasm["outputs"], strict=True):
            close_tree(reference, output["evaluate"]["inference"])
            verify_identity(output["search"]["proof"], identity)
        arena_request = {
            "initial_sfen": fixtures[0]["sfen"],
            "moves": [],
            "depth": 1,
            "nodes": 4,
            "movetime_ms": None,
            "hard_timeout_ms": 30000,
            "hash_mb": 32,
        }
        arena_output = audit.run(
            [binary, "arena-player", *arguments], stdin=json.dumps(arena_request) + "\n"
        )
        arena_lines = [json.loads(line) for line in arena_output.splitlines()]
        arena_log = audit.commands[-1]["log"]
        if len(arena_lines) != 2:
            raise ValueError("bounded player protocol must emit readiness and one response")
        verify_player(arena_lines, arena_request, identity)
        rows.append(
            {
                "identity": identity,
                "fixtures": fixtures,
                "python": references,
                "native": native,
                "native_logs": native_logs,
                "usi_log": usi_log,
                "wasm_log": wasm_log,
                "arena_player": arena_lines,
                "arena_request": arena_request,
                "arena_log": arena_log,
                "wasm": wasm,
                "usi_proofs": proofs,
                "native_negative_cases": sorted(NEGATIVES),
            }
        )
    return rows


def verify_player(lines: list[dict], request: dict, identity: dict) -> None:
    ready, response = lines
    if (
        ready.get("ready") is not True
        or ready.get("model_format") != identity["format"]
        or ready.get("model_sha256") != identity["sha256"]
    ):
        raise ValueError("player readiness identity mismatch")
    base.verify_registrations(ready["compiled_evaluators"])
    verify_identity(response["proof"], identity)
    if (
        response.get("requested_controls") != request
        or response.get("best_move") is None
        or response.get("model_format") != identity["format"]
        or response.get("model_sha256") != identity["sha256"]
        or response.get("threads") != 1
        or response.get("hash_mb") != 32
    ):
        raise ValueError("bounded player transport evidence mismatch")
    timing = response["deadline"]
    if (
        timing.get("clock") != "monotonic"
        or timing.get("hard_compliant") is not True
        or timing.get("hard_timeout_ms") != request["hard_timeout_ms"]
        or timing.get("hard_budget_ns") != request["hard_timeout_ms"] * 1_000_000
        or not 0 <= timing["setup_elapsed_ns"] <= timing["search_start_ns"]
        or not timing["setup_elapsed_ns"] + timing["search_elapsed_ns"]
        <= timing["elapsed_ns"]
        <= timing["hard_budget_ns"]
    ):
        raise ValueError("bounded player deadline evidence mismatch")


def verify_evidence(root: Path, receipt: dict) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "passed":
        raise ValueError("prior adapter audit has not passed")
    ref = receipt["pure_artifact_audit"]
    path = Path(ref["path"])
    if path.is_absolute() or ".." in path.parts or base.digest(root / path) != ref["sha256"]:
        raise ValueError("pure artifact audit binding mismatch")
    certified = json.loads((root / path).read_text())
    base.verify_evidence(root, certified)
    if (
        receipt["git_commit"] != certified["git_commit"]
        or receipt["artifacts"] != certified["artifacts"]
    ):
        raise ValueError("source or artifact certification mismatch")
    if len(receipt["models"]) != 3 or {r["identity"]["id"] for r in receipt["models"]} != set(
        EXPECTED_MODELS
    ):
        raise ValueError("missing exact entrant certification")

    def recorded(name: str) -> str:
        matches = [c for c in receipt["commands"] if c["log"] == name]
        if len(matches) != 1 or matches[0]["returncode"] != 0:
            raise ValueError("successful runtime command missing")
        log_path = Path(name)
        if log_path.is_absolute() or ".." in log_path.parts:
            raise ValueError("invalid runtime log path")
        return (root / path.parent / log_path).read_text()

    for row in receipt["models"]:
        for native, log in zip(row["native"], row["native_logs"], strict=True):
            if json.loads(recorded(log)) != native:
                raise ValueError("native runtime output binding mismatch")
        if json.loads(recorded(row["wasm_log"])) != row["wasm"]:
            raise ValueError("Wasm runtime output binding mismatch")
        proofs = [
            json.loads(line.split("pure-proof ", 1)[1])
            for line in recorded(row["usi_log"]).splitlines()
            if line.startswith("info string pure-proof ")
        ]
        if proofs != row["usi_proofs"]:
            raise ValueError("USI runtime output binding mismatch")
        identity = row["identity"]
        if [json.loads(line) for line in recorded(row["arena_log"]).splitlines()] != row[
            "arena_player"
        ]:
            raise ValueError("player transport output binding mismatch")
        verify_player(row["arena_player"], row["arena_request"], identity)
        expected = model_identity(root, {"id": identity["id"], "model": identity})
        if identity != expected:
            raise ValueError("model or profile binding changed")
        if row["fixtures"] != certified["runtime"]["fixtures"]:
            raise ValueError("history parity fixtures mismatch")
        if (
            set(row["native_negative_cases"]) != NEGATIVES
            or set(row["wasm"]["failures"]) != NEGATIVES
        ):
            raise ValueError("missing format-specific negative evidence")
        if not row["usi_proofs"]:
            raise ValueError("missing USI runtime evidence")
        for proof in row["usi_proofs"]:
            verify_identity(proof, identity)
        if len(row["python"]) != len(row["fixtures"]):
            raise ValueError("incomplete parity evidence")
        for reference, native, wasm in zip(
            row["python"], row["native"], row["wasm"]["outputs"], strict=True
        ):
            for output in (native, wasm["evaluate"]):
                if output.get("model_format") != identity["format"]:
                    raise ValueError("runtime model format mismatch")
                base.verify_registrations(output["compiled_evaluators"])
                close_tree(reference, output["inference"])
            for output in (native, wasm["search"]):
                verify_identity(output["proof"], identity)
                if output["best_move"] is None:
                    raise ValueError("bounded legal search produced no move")
    for command in receipt["commands"]:
        log = Path(command["log"])
        if (
            log.is_absolute()
            or ".." in log.parts
            or base.digest(root / path.parent / log) != command["sha256"]
        ):
            raise ValueError("audit command log binding mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    audit = base.Audit(root, args.output.resolve())
    evidence = {"schema": SCHEMA, "status": "failed"}
    try:
        commit = audit.run(["git", "rev-parse", "HEAD"]).strip()
        status = audit.run(["git", "status", "--porcelain"])
        if status:
            raise ValueError("formal audit requires clean committed worktree")
        certified = {
            "schema": "open_shogiai_phase10t_pure_artifact_audit/v1",
            "git_commit": commit,
            "git_status": status,
            "output_directory": str(audit.output.relative_to(root)),
            "source_hashes": base.source_inventory(root),
            "evaluator_profile_schema_hash": base.digest(
                root / "configs/runtime/pure_learned-a1-v1.json"
            ),
        }
        certified.update(audit.build())
        certified["negative_imports"] = audit.negative_imports()
        certified["runtime"] = audit.runtime()
        certified["artifacts"] = {
            key: {"path": str(path.relative_to(root)), "sha256": base.digest(path)}
            for key, path in {
                "native": audit.output / "target/release/open-shogi-cli",
                "wasm_raw": audit.output
                / "target/wasm32-unknown-unknown/release/open_shogi_wasm.wasm",
                "wasm": audit.output / "wasm-one/open_shogi_wasm_bg.wasm",
                "model": audit.output / "synthetic-a1.bin",
            }.items()
        }
        certified.update(status="passed", commands=list(audit.commands))
        certified = json.loads(base.portable_text(json.dumps(certified), root))
        base.verify_evidence(root, certified)
        base_path = audit.output / "pure-build.json"
        base_path.write_text(json.dumps(certified, indent=2) + "\n")
        entrants = json.loads((root / "configs/phase10u/arena-rerun-plan.json").read_text())[
            "entrants"
        ]
        evidence.update(
            git_commit=commit,
            artifacts=certified["artifacts"],
            pure_artifact_audit={
                "path": str(base_path.relative_to(root)),
                "sha256": base.digest(base_path),
            },
            models=runtime(
                audit, [model_identity(root, e) for e in entrants], certified["runtime"]["fixtures"]
            ),
            commands=audit.commands,
            status="passed",
        )
        evidence = json.loads(base.portable_text(json.dumps(evidence), root))
        verify_evidence(root, evidence)
    except Exception:
        evidence["status"] = "failed"
        raise
    finally:
        evidence["commands"] = audit.commands
        (audit.output / "adapter-audit.json").write_text(
            base.portable_text(json.dumps(evidence, indent=2), root) + "\n"
        )


if __name__ == "__main__":
    main()
