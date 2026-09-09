"""Hash-pinned search leaves, external teacher labels and raw-verified training streams.

Only metadata about calibration/final-holdout positions is read. No sealed position
stream is opened. A completed receipt is published last; interrupted outputs remain
unapproved evidence and cannot be resumed as a completed dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .phase10r_model import parse_sfen
from .phase10t import safe_path, sha256
from .phase10u_execution import successor_sfen
from .phase10v_targets import TARGET_SCHEMA, example_from_mapping

DATASET_SCHEMA = "open_shogiai_phase10v_dataset/v1"
COLLECTION_SCHEMA = "open_shogiai_phase10v_collection/v1"
COLLECTION_ARTIFACTS = {"request", "roots", "split_guard", "data_policy", "searches", "leaves"}
DATASET_ARTIFACTS = COLLECTION_ARTIFACTS | {
    "collection",
    "label_request",
    "teacher_identity",
    "validator_identity",
    "teacher_raw",
    "teacher_config",
    "train",
    "validation",
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def position_hash(sfen: str) -> str:
    parsed = parse_sfen(sfen)
    ranks = []
    names = ("P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R")
    for rank in range(9):
        encoded, empty = [], 0
        for piece in parsed.board[rank * 9 : (rank + 1) * 9]:
            if piece is None:
                empty += 1
            else:
                if empty:
                    encoded.append(str(empty))
                    empty = 0
                name = names[piece.kind]
                encoded.append(name if piece.side == 0 else name.lower())
        if empty:
            encoded.append(str(empty))
        ranks.append("".join(encoded))
    hands = []
    for side in (0, 1):
        for name in "RBGSNLP":
            count = parsed.hands[side]["PLNSGBR".index(name)]
            if count:
                hands.append(
                    (str(count) if count > 1 else "") + (name if side == 0 else name.lower())
                )
    state = f"{'/'.join(ranks)} {'b' if parsed.side_to_move == 0 else 'w'} {''.join(hands) or '-'}"
    if state != " ".join(sfen.split()[:3]):
        raise ValueError("noncanonical SFEN cannot bypass position split hashes")
    return hashlib.sha256(state.encode()).hexdigest()


def _progress(output: Path, filename: str, row: dict) -> None:
    with (output / filename).open("ab") as stream:
        stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _distribution(rows: list, policy: dict, *, enforce: bool = True) -> dict:
    if policy.get("schema") != "open_shogiai_phase10v_data_policy/v1":
        raise ValueError("frozen stage data policy required")
    stage_rows = policy.get("stage_rows")
    if type(stage_rows) is not int or stage_rows < 1:
        raise ValueError("positive frozen stage row minimum required")
    if enforce and sum(row["split"] == "train" for row in rows) < stage_rows:
        raise ValueError("stage distribution gate failed: training rows below stage_rows")
    requirements = policy["requirements"]
    if set(requirements) != {"train", "validation"}:
        raise ValueError("stage policy must govern both development splits")
    counts = {split: {"leaf_kinds": {}, "position_categories": {}} for split in requirements}
    allowed = {
        "quiet_middlegame",
        "tactical_middlegame",
        "mate_boundary",
        "endgame",
        "approved_external",
        "teacher_candidate",
        "crossplay_leaf",
    }
    for row in rows:
        p = row["provenance"]
        categories = p.get("position_categories")
        if (
            not isinstance(categories, list)
            or not categories
            or len(set(categories)) != len(categories)
            or not set(categories) <= allowed
        ):
            raise ValueError("hash-bound source position categories required")
        leaf_counts = counts[row["split"]]["leaf_kinds"]
        leaf_counts[p["leaf_kind"]] = leaf_counts.get(p["leaf_kind"], 0) + 1
        for category in categories:
            category_counts = counts[row["split"]]["position_categories"]
            category_counts[category] = category_counts.get(category, 0) + 1
    for split, dimensions in requirements.items():
        if set(dimensions) != {"leaf_kinds", "position_categories"}:
            raise ValueError("both actual leaf and source-stratum quotas are required")
        for dimension, minima in dimensions.items():
            if not isinstance(minima, dict) or not minima:
                raise ValueError("empty distribution gate is forbidden")
            for name, minimum in minima.items():
                if type(minimum) is not int or minimum < 1:
                    raise ValueError("distribution quotas must be positive integer rows")
                if enforce and counts[split][dimension].get(name, 0) < minimum:
                    raise ValueError(f"stage distribution gate failed: {split}/{dimension}/{name}")
    return counts


def limits(root: Path) -> None:
    if shutil.disk_usage(root).free < 80 * 1024**3:
        raise ValueError("80 GiB hard free-space floor")
    cutoff = datetime.fromisoformat("2026-09-13T06:00:00+09:00")
    if datetime.now(cutoff.tzinfo) >= cutoff:
        raise ValueError("Sunday labeling/training cutoff reached")


def artifact(root: Path, name: str, expected_hash: str) -> Path:
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("an independently pinned artifact SHA256 is required")
    path = safe_path(root, name)
    if not path.is_file() or sha256(path) != expected_hash:
        raise ValueError(f"artifact identity changed: {name}")
    return path


def _write(output: Path, name: str, value: object) -> dict:
    path = output / name
    with path.open("xb") as stream:
        stream.write(canonical(value))
    return {"name": name, "expected_hash": sha256(path)}


def _copy(output: Path, name: str, source: Path) -> dict:
    with (output / name).open("xb") as stream:
        stream.write(source.read_bytes())
    return {"name": name, "expected_hash": sha256(output / name)}


def _json(root: Path, ref: dict) -> object:
    return json.loads(artifact(root, **ref).read_bytes())


def _new_output(root: Path, output: Path) -> Path:
    root = root.resolve(strict=True)
    resolved = output.absolute()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("output must be inside the project root") from error
    output = safe_path(root, str(relative))
    if output.exists():
        raise ValueError("new immutable output directory required")
    output.mkdir(parents=True)
    return output


def _guard_indexes(guard: dict) -> tuple[dict, dict]:
    games, positions = {}, {}
    for split in ("train", "validation", "calibration", "final_holdout"):
        entries = guard.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError("split guard must contain all four nonempty metadata inventories")
        for item in entries:
            component, state, source = (
                item.get(k) for k in ("component_id", "position_sha256", "source_sha256")
            )
            if not component or not all(
                isinstance(h, str) and len(h) == 64 for h in (state, source)
            ):
                raise ValueError("split guard lacks component/position/source identity")
            if component in games and games[component] != (split, source):
                raise ValueError("source component crosses original splits")
            if state in positions and positions[state] != split:
                raise ValueError("position crosses original splits")
            games[component], positions[state] = (split, source), split
    return games, positions


def _root_binding(row: dict, games: dict, positions: dict) -> None:
    split, p = row["split"], row["provenance"]
    if split not in {"train", "validation"} or p.get("original_split") != split:
        raise ValueError("forbidden or reassigned original split")
    if not p.get("source_game_id") or p.get("approved") is not True:
        raise ValueError("approved original source game required")
    if games.get(p.get("component_id")) != (split, p.get("source_sha256")):
        raise ValueError("source component is not bound to pinned split metadata")
    if positions.get(position_hash(row["sfen"])) != split:
        raise ValueError("root is absent from the approved original split inventory")


def _check_state(sfen: str, split: str, inventory: dict, observed: dict) -> None:
    state = position_hash(sfen)
    for mapping in (inventory, observed):
        if state in mapping and mapping[state] != split:
            raise ValueError("search/teacher position leaks across split boundary")
    observed[state] = split


def _collect_rows(roots: list, searches: list, guard: dict, model_hash: str) -> list:
    from .phase10v_audit import check_proof

    if len(roots) != len(searches):
        raise ValueError("search receipt count does not match root inventory")
    games, inventory = _guard_indexes(guard)
    seen, observed, rows = set(), {}, []
    source_games = {}
    for row, receipt in zip(roots, searches, strict=True):
        game = row["provenance"]["source_game_id"]
        if game in source_games and source_games[game] != row["split"]:
            raise ValueError("source game crosses original splits")
        source_games[game] = row["split"]
        _root_binding(row, games, inventory)
        check_proof(receipt, model_hash)
        if receipt.get("sfen") != row["sfen"]:
            raise ValueError("search receipt has another root")
        if receipt.get("model_format") != "OSAVAL03":
            raise ValueError("search receipt is not the frozen model format")
        for leaf in receipt["leaf_trace"]:
            if leaf.get("model_sha256") != model_hash:
                raise ValueError("leaf trace model identity drift")
            kind = {"quiescence_leaf": "quiescence_leaf", "pv_horizon_leaf": "pv_leaf"}.get(
                leaf["kind"]
            )
            if kind is None:
                continue
            if type(leaf.get("ply")) is not int or leaf["ply"] < 0:
                raise ValueError("leaf trace ply is invalid")
            _check_state(leaf["sfen"], row["split"], inventory, observed)
            state = position_hash(leaf["sfen"])
            if state in seen:
                continue
            seen.add(state)
            rows.append(
                {
                    "sfen": leaf["sfen"],
                    "split": row["split"],
                    "provenance": {
                        **row["provenance"],
                        "leaf_kind": kind,
                        "search_receipt_sha256": digest(receipt),
                        "root_position_sha256": position_hash(row["sfen"]),
                        "engine_model_sha256": model_hash,
                    },
                }
            )
    if not rows:
        raise ValueError("no actual eligible search leaves collected")
    return rows


def collect(root: Path, request: dict, output: Path) -> dict:
    if output.exists():
        raise ValueError("new immutable output directory required")
    if request.get("schema") != "open_shogiai_phase10v_leaf_request/v1":
        raise ValueError("unsupported leaf collection request")
    if type(request["nodes"]) is not int or not 1 <= request["nodes"] <= 1_000_000:
        raise ValueError("bounded collection node budget required")
    if type(request["leaf_limit"]) is not int or not 1 <= request["leaf_limit"] <= 10_000:
        raise ValueError("bounded leaf trace limit required")
    limits(root)
    binary, model = (artifact(root, **request[k]) for k in ("binary", "model"))
    roots, guard = (_json(root, request[k]) for k in ("roots", "split_guard"))
    policy = _json(root, request["data_policy"])
    if (
        not isinstance(roots, list)
        or not 1 <= len(roots) <= 100_000
        or len(roots) * request["leaf_limit"] > 2_000_000
    ):
        raise ValueError("root batch must contain 1..100000 approved roots")
    games, positions = _guard_indexes(guard)
    for row in roots:
        _root_binding(row, games, positions)
    output = _new_output(root, output)
    searches = []
    for row in roots:
        limits(root)
        for key in ("binary", "model"):
            artifact(root, **request[key])
        result = subprocess.run(
            [
                str(binary.resolve()),
                "pure",
                "--model",
                str(model.resolve()),
                "--model-sha256",
                request["model"]["expected_hash"],
                "--model-format",
                "OSAVAL03",
                "--profile",
                "pure_learned",
                "--sfen",
                row["sfen"],
                "--depth",
                "8",
                "--nodes",
                str(request["nodes"]),
                "--leaf-trace-limit",
                str(request["leaf_limit"]),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        search = json.loads(result.stdout)
        _progress(output, "search-progress.jsonl", {"root": row, "search": search})
        searches.append(search)
    rows = _collect_rows(roots, searches, guard, request["model"]["expected_hash"])
    _distribution(rows, policy)
    limits(root)
    refs = {
        key: _write(output, f"{key}.json", value)
        for key, value in (("request", request), ("searches", searches), ("leaves", rows))
    }
    for key in ("roots", "split_guard", "data_policy"):
        refs[key] = _copy(output, f"{key}.json", artifact(root, **request[key]))
    receipt = {
        "schema": COLLECTION_SCHEMA,
        "request_sha256": digest(request),
        "rows": len(rows),
        "artifacts": refs,
        "teacher_calls": 0,
    }
    _write(output, "receipt.json", receipt)
    return receipt


def _verify_collection(folder: Path, receipt: dict) -> dict:
    if receipt.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("a completed leaf collection receipt is required")
    if set(receipt["artifacts"]) != COLLECTION_ARTIFACTS:
        raise ValueError("unexpected collection artifact; sealed streams cannot be read")
    evidence = {k: _json(folder, ref) for k, ref in receipt["artifacts"].items()}
    request = evidence["request"]
    if receipt["request_sha256"] != digest(request):
        raise ValueError("collection request identity changed")
    # Copied raw source inventories must match the original independently frozen bytes.
    for key in ("roots", "split_guard", "data_policy"):
        if receipt["artifacts"][key]["expected_hash"] != request[key]["expected_hash"]:
            raise ValueError("copied source/split inventory differs from request")
    reconstructed = _collect_rows(
        evidence["roots"],
        evidence["searches"],
        evidence["split_guard"],
        request["model"]["expected_hash"],
    )
    _distribution(reconstructed, evidence["data_policy"])
    if reconstructed != evidence["leaves"] or len(reconstructed) != receipt["rows"]:
        raise ValueError("leaf rows differ from raw search evidence")
    return evidence


def _target(leaf: dict, raw: dict) -> dict | None:
    if raw["sfen"] != leaf["sfen"] or not 1 <= len(raw["candidates"]) <= 3:
        raise ValueError("teacher observation does not bind this leaf")
    candidates = raw["candidates"]
    if [c["multipv"] for c in candidates] != list(range(1, len(candidates) + 1)):
        raise ValueError("observed MultiPV ranks are not contiguous")
    if any(not c["pv"] for c in candidates) or candidates[0]["pv"][0] != raw["bestmove"]:
        raise ValueError("teacher bestmove/PV disagreement")
    if len({c["pv"][0] for c in candidates}) != len(candidates):
        raise ValueError("teacher repeated candidate")
    if (
        raw["legality"]["returned_candidates"] != len(candidates)
        or raw["legality"].get("requested_multipv") != 3
    ):
        raise ValueError("raw legal-replay coverage is incomplete")
    row = {
        "schema": TARGET_SCHEMA,
        "sfen": leaf["sfen"],
        "split": leaf["split"],
        "score_perspective": "side_to_move",
        "score": candidates[0]["score"],
        "wdl": None,
        "candidates": [
            {
                "move": c["pv"][0],
                "child_sfen": successor_sfen(leaf["sfen"], c["pv"][0]),
                "score": c["score"],
                "score_perspective": "parent_side_to_move",
                "replay_receipt_sha256": digest(raw),
            }
            for c in candidates
        ],
        "provenance": {**leaf["provenance"], "teacher_receipt_sha256": digest(raw)},
    }
    example = example_from_mapping(row, expected_split=leaf["split"])
    if example.cp is None and not example.ranking_pairs:
        return None  # No numeric score or ranking pair: retained only in raw evidence.
    return row


def label(root: Path, request: dict, output: Path) -> dict:
    from .labeling.config import load_teacher_config
    from .labeling.fingerprint import fingerprint_teacher
    from .labeling.legality import RustLegalityValidator
    from .labeling.usi import USIEngine

    if request.get("schema") != "open_shogiai_phase10v_label_request/v1":
        raise ValueError("unsupported external teacher request")
    limits(root)
    collection_path = artifact(root, **request["collection_receipt"])
    collection = json.loads(collection_path.read_bytes())
    evidence = _verify_collection(collection_path.parent, collection)
    config_path = artifact(root, **request["teacher_config"])
    config = load_teacher_config(config_path)
    teacher_identity = fingerprint_teacher(config, root).identity_record()
    if digest(teacher_identity) != request["teacher_identity_sha256"]:
        raise ValueError("external teacher fingerprint changed")
    if (
        type(request["nodes"]) is not int
        or request["nodes"] not in (100000, 400000)
        or config.multipv != 3
    ):
        raise ValueError("unsupported teacher controls")
    output = _new_output(root, output)
    rows, raw = [], []
    with USIEngine(config, root) as teacher, RustLegalityValidator(root) as validator:
        validator_identity = validator.identity.as_dict()
        for leaf in evidence["leaves"]:
            limits(root)
            result = teacher.analyze_with_retry(leaf["sfen"], nodes=request["nodes"])
            _progress(
                output,
                "teacher-progress.jsonl",
                {
                    "status": "observed_before_replay",
                    "sfen": leaf["sfen"],
                    "bestmove": result.bestmove,
                    "candidates": [c.as_dict() for c in result.candidates],
                    "teacher_identity_sha256": request["teacher_identity_sha256"],
                },
            )
            coverage = validator.validate(
                leaf["sfen"],
                result.bestmove,
                [list(c.pv) for c in result.candidates],
                configured_multipv=3,
            )
            teacher_row = {
                "sfen": leaf["sfen"],
                "bestmove": result.bestmove,
                "candidates": [c.as_dict() for c in result.candidates],
                "teacher_identity_sha256": request["teacher_identity_sha256"],
                "validator_identity_sha256": digest(validator_identity),
                "legality": asdict(coverage),
            }
            _progress(
                output, "teacher-progress.jsonl", {"status": "legally_replayed", "raw": teacher_row}
            )
            raw.append(teacher_row)
            row = _target(leaf, teacher_row)
            if row is not None:
                rows.append(row)
        if fingerprint_teacher(config, root).identity_record() != teacher_identity:
            raise ValueError("teacher fingerprint changed during labeling")
    refs = {
        k: _copy(output, ref["name"], artifact(collection_path.parent, **ref))
        for k, ref in collection["artifacts"].items()
    }
    refs.update(
        {
            k: _write(output, f"{k}.json", value)
            for k, value in (
                ("collection", collection),
                ("label_request", request),
                ("teacher_identity", teacher_identity),
                ("validator_identity", validator_identity),
                ("teacher_raw", raw),
            )
        }
    )
    refs["teacher_config"] = _copy(output, "teacher-config.yaml", config_path)
    for split in ("train", "validation"):
        path = output / f"{split}.jsonl"
        with path.open("xb") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(canonical(row) + b"\n")
        refs[split] = {"name": path.name, "expected_hash": sha256(path)}
    receipt = {
        "schema": DATASET_SCHEMA,
        "artifacts": refs,
        "rows": len(rows),
        "masked_mate_only": len(raw) - len(rows),
        "teacher_calls": len(raw),
    }
    _write(output, "pending-receipt.json", receipt)
    # Reconstruct the two streams from raw evidence before allowing their consumption.
    verify_training_inputs(
        output / "train.jsonl",
        output / "validation.jsonl",
        output / "pending-receipt.json",
        sha256(output / "pending-receipt.json"),
    )
    (output / "pending-receipt.json").rename(output / "receipt.json")
    return receipt


def verify_training_inputs(
    train_path: Path,
    validation_path: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> dict:
    """Verify frozen receipt bytes, raw search/teacher rows and every target, before training."""
    receipt_path = artifact(receipt_path.parent, receipt_path.name, expected_receipt_sha256)
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("schema") != DATASET_SCHEMA:
        raise ValueError("raw-verified Phase10V dataset receipt required")
    folder, refs = receipt_path.parent, receipt["artifacts"]
    if set(refs) != DATASET_ARTIFACTS:
        raise ValueError("unexpected dataset artifact; sealed streams cannot be read")
    evidence = {
        k: _json(folder, ref)
        for k, ref in refs.items()
        if k not in {"train", "validation", "teacher_config"}
    }
    collection = evidence["collection"]
    collected = _verify_collection(folder, collection)
    request = evidence["label_request"]
    config_path = artifact(folder, **refs["teacher_config"])
    if sha256(config_path) != request["teacher_config"]["expected_hash"]:
        raise ValueError("raw teacher controls differ from label request")
    if request["collection_receipt"]["expected_hash"] != digest(collection):
        raise ValueError("label request does not bind collection receipt")
    if request["teacher_identity_sha256"] != digest(evidence["teacher_identity"]):
        raise ValueError("label request does not bind external teacher")
    if request["nodes"] not in (100000, 400000):
        raise ValueError("teacher node control changed")
    raw = evidence["teacher_raw"]
    if len(raw) != len(collected["leaves"]) or len(raw) != receipt["teacher_calls"]:
        raise ValueError("teacher row coverage is incomplete")
    expected = {"train": [], "validation": []}
    _, inventory = _guard_indexes(collected["split_guard"])
    observed = {}
    for leaf, teacher in zip(collected["leaves"], raw, strict=True):
        if teacher["teacher_identity_sha256"] != request["teacher_identity_sha256"] or teacher[
            "validator_identity_sha256"
        ] != digest(evidence["validator_identity"]):
            raise ValueError("raw teacher/replay identity drift")
        row = _target(leaf, teacher)
        _check_state(leaf["sfen"], leaf["split"], inventory, observed)
        for candidate in teacher["candidates"]:
            state = leaf["sfen"]
            for move in candidate["pv"]:
                state = successor_sfen(state, move)
                _check_state(state, leaf["split"], inventory, observed)
        if row is not None:
            expected[row["split"]].append(row)
    for split, supplied in (("train", train_path), ("validation", validation_path)):
        path = artifact(folder, **refs[split])
        if path.resolve() != supplied.resolve():
            raise ValueError("training stream is not the receipt-bound path")
        actual = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        if actual != expected[split] or not actual:
            raise ValueError("training targets differ from raw observations or split is empty")
    distribution = _distribution(
        expected["train"] + expected["validation"],
        collected["data_policy"],
        enforce=True,
    )
    count = sum(map(len, expected.values()))
    if receipt["rows"] != count or receipt["masked_mate_only"] != len(raw) - count:
        raise ValueError("dataset counts do not match raw evidence")
    return {
        "dataset_sha256": expected_receipt_sha256,
        "train_sha256": refs["train"]["expected_hash"],
        "validation_sha256": refs["validation"]["expected_hash"],
        "root_evidence_sha256": digest(collection),
        "rows": count,
        "distribution": distribution,
        "teacher_identity_sha256": request["teacher_identity_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "label"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request_path = artifact(args.request.parent, args.request.name, args.request_sha256)
    request = json.loads(request_path.read_bytes())
    print(
        json.dumps(
            (collect if args.command == "collect" else label)(args.root, request, args.output)
        )
    )


if __name__ == "__main__":
    main()
