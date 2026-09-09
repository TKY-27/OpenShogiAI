"""Fail-closed execution surface for the frozen Phase 10S campaign.

The controls live in ``configs/phase10s`` and are intentionally not interpreted by
this module.  This module only binds those controls to immutable receipts, the
existing Phase 10R model/checkpoint code, and the existing native Arena binary.
New data and temporary exports are written below ``local/phase10s-runs``; frozen
Phase 10R artifacts are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from open_shogi_training.phase10s import (
    Phase10SValidationError,
    validate_phase10s,
)

RUNS_DIRECTORY: Final = Path("local/phase10s-runs")
REPORT_JSON: Final = RUNS_DIRECTORY / "PHASE10S_EXECUTION_REPORT.json"
REPORT_MD: Final = RUNS_DIRECTORY / "PHASE10S_EXECUTION_REPORT.md"
REQUIRED_BRANCH: Final = "codex/pure-learned-pre-selfplay"
REQUIRED_ANCESTOR: Final = "b81efc487ee3911d1c3e66d5ba868c3d4fe08199"
START_POOL: Final = Path("artifacts/phase10/start-pool-manifest.json")
START_POOL_SHA256: Final = "491a3d54d3d67002fc03fc3644c16b405f5b1dbb662e88e4af260bd45757a1f1"
SELECTED_MODEL: Final = Path(
    "local/phase10r-data/checkpoints/phase10r/1m/sparse-pair-policy-wdl/"
    "teacher-bound-v1/teacher-bound-v1.osaval02"
)
SELECTED_MODEL_SHA256: Final = "41ce44a93219b0bcee0270677104c379029999050524fe99c1a560bb5cc05060"
MINIMUM_FREE_BYTES: Final = 150 * 1024**3
TRAINING_SEED: Final = 20_260_729
FROZEN_ARENA_COUNTERS: Final = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)
CHECKPOINTS: Final = (
    {
        "id": "10m-stage1-best",
        "path": Path(
            "local/phase10r-data/checkpoints/phase10r/10m/sparse-pair-policy-wdl/"
            "representation_policy_pretraining/best.pt"
        ),
        "sha256": "ccf0b99a66f9b822d83a833a3dd81789336ef8f2f8dc6877c3ec259211b24279",
    },
    {
        "id": "10m-stage1-last",
        "path": Path(
            "local/phase10r-data/checkpoints/phase10r/10m/sparse-pair-policy-wdl/"
            "representation_policy_pretraining/last.pt"
        ),
        "sha256": "ee01016d174f740db87124c161675d6dbaae077b63043b5b291cc993ab035288",
    },
    {
        "id": "10m-stage2-best",
        "path": Path(
            "local/phase10r-data/checkpoints/phase10r/10m/sparse-pair-policy-wdl/"
            "source_specific_wdl_value_pretraining/best.pt"
        ),
        "sha256": "ae05b31b503bf80f3d0741e397caef3e8a7b6a6e68252c600787f2404be06cb2",
    },
    {
        "id": "10m-stage2-last-rejected",
        "path": Path(
            "local/phase10r-data/checkpoints/phase10r/10m/sparse-pair-policy-wdl/"
            "source_specific_wdl_value_pretraining/last.pt"
        ),
        "sha256": "926de4c73bed9343fdc4fd6f8d76b4adbc65b2229a3edeae2f375d3c6e79016d",
    },
)


class Phase10SRunError(RuntimeError):
    """Raised when a frozen execution boundary cannot be satisfied."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise Phase10SRunError(f"refusing to overwrite immutable Phase 10S file: {path}") from error


def _append_event(root: Path, event: Mapping[str, Any]) -> None:
    path = root / RUNS_DIRECTORY / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _receipt_path(root: Path, command: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / RUNS_DIRECTORY / f"{stamp}-{command}.json"


def _base_receipt(root: Path, command: str, argv: Sequence[str]) -> dict[str, Any]:
    try:
        git = {
            "branch": _git(root, "branch", "--show-current"),
            "commit": _git(root, "rev-parse", "HEAD"),
            "dirty": _git(root, "status", "--porcelain"),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        git = {"branch": None, "commit": None, "dirty": None, "error": str(error)}
    return {
        "schema": "open_shogi_ai_phase10s_run_receipt/v1",
        "command": command,
        "argv": list(argv),
        "started_at_utc": _utc_now(),
        "repository": root.name,
        "git": git,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": platform.machine(),
            "pid": os.getpid(),
        },
    }


def _finish_receipt(root: Path, command: str, argv: Sequence[str], receipt: dict[str, Any]) -> Path:
    receipt["ended_at_utc"] = _utc_now()
    rss_unit = 1 if sys.platform == "darwin" else 1024
    receipt["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit)
    path = _receipt_path(root, command)
    _write_new_json(path, receipt)
    _append_event(
        root,
        {
            "schema": "open_shogi_ai_phase10s_event/v1",
            "event": "receipt_written",
            "timestamp_utc": receipt["ended_at_utc"],
            "command": command,
            "receipt": str(path.relative_to(root)),
            "receipt_sha256": _sha256(path),
            "argv": list(argv),
        },
    )
    return path


def _disk_receipt(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    return {
        "path": str(root),
        "free_bytes": int(usage.free),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "passed": usage.free >= MINIMUM_FREE_BYTES,
    }


def _resource_receipt(root: Path) -> dict[str, Any]:
    return {
        "disk": _disk_receipt(root),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase10SRunError(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise Phase10SRunError(f"JSON root is not an object: {path}")
    return value


def _ensure_regular(path: Path, *, expected_sha256: str | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise Phase10SRunError(f"required regular file is unavailable: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise Phase10SRunError(f"hash mismatch: {path}")


def _hash_manifest(root: Path) -> dict[str, str]:
    path = root / "configs/phase10s/frozen-controls.sha256"
    _ensure_regular(path)
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in result:
            raise Phase10SRunError("Phase 10S frozen hash manifest is malformed")
        result[name] = digest
    if len(result) != 9:
        raise Phase10SRunError("Phase 10S frozen hash inventory changed")
    return result


def _process_inventory() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase10SRunError(f"cannot inspect active processes: {error}") from error
    patterns = (
        "phase10s_run",
        "phase10r_run",
        "phase10r_campaign",
        "open-shogi-cli arena",
        "cross-play",
        "crossplay",
        "self-play",
        "selfplay",
        "teacher",
        "open_shogi_training",
    )
    found: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if any(pattern in command.lower() for pattern in patterns):
            # The process inventory must not report this exact one-shot command's
            # shell/uv wrappers as a pre-existing campaign.
            if "phase10s_run preflight" in command:
                continue
            found.append({"pid": pid, "command": command})
    return found


def _validate_execution_boundary(root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    try:
        validation = validate_phase10s(root, require_branch=True)
    except (Phase10SValidationError, OSError, ValueError) as error:
        raise Phase10SRunError(f"frozen Phase 10S validation failed: {error}") from error
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    if branch != REQUIRED_BRANCH:
        raise Phase10SRunError(f"wrong branch: {branch}")
    if _git(root, "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"):
        raise Phase10SRunError("required Phase 10S ancestor is absent")
    dirty = _git(root, "status", "--porcelain")
    if require_clean and dirty:
        raise Phase10SRunError("worktree is not clean")
    return {
        "validation": validation,
        "branch": branch,
        "commit": commit,
        "ancestor": REQUIRED_ANCESTOR,
        "dirty": dirty,
        "frozen_hashes": _hash_manifest(root),
    }


def _run_command(
    command: Sequence[str], *, cwd: Path, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Phase10SRunError(
            f"command failed to start or timed out: {' '.join(command)}"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-8_000:]
        raise Phase10SRunError(
            f"command exited {completed.returncode}: {' '.join(command)}\n{detail}"
        )
    return completed


def _start_pool(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / START_POOL
    _ensure_regular(path, expected_sha256=START_POOL_SHA256)
    manifest = _load_json(path)
    if manifest.get("schema") != "open_shogiai_phase10_start_pool/v2":
        raise Phase10SRunError("start pool schema changed")
    positions = manifest.get("positions")
    if not isinstance(positions, list) or len(positions) != 800:
        raise Phase10SRunError("frozen start pool size changed")
    for position in positions:
        if (
            not isinstance(position, Mapping)
            or not isinstance(position.get("sfen"), str)
            or not isinstance(position.get("positionId"), str)
            or not isinstance(position.get("assignedGroup"), str)
        ):
            raise Phase10SRunError("start pool position identity is incomplete")
    return manifest, [dict(position) for position in positions]


def _ensure_cli(root: Path) -> Path:
    binary = root / "target/release/open-shogi-cli"
    if not binary.is_file() or binary.is_symlink():
        _run_command(["cargo", "build", "--locked", "--release", "-p", "open-shogi-cli"], cwd=root)
    _ensure_regular(binary)
    return binary


def _run_preflight(root: Path, argv: Sequence[str]) -> dict[str, Any]:
    receipt = _base_receipt(root, "preflight", argv)
    boundary = _validate_execution_boundary(root)
    processes = _process_inventory()
    if processes:
        raise Phase10SRunError(f"pre-existing campaign processes are active: {processes}")
    disk = _disk_receipt(root)
    if not disk["passed"]:
        raise Phase10SRunError("free disk is below the frozen 150 GiB floor")
    runtime_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/python/test_pure_learned_profile.py",
    ]
    runtime = _run_command(runtime_command, cwd=root, timeout=900)
    receipt.update(
        {
            "status": "passed",
            "exit_status": 0,
            "boundary": boundary,
            "process_inventory": processes,
            "disk": disk,
            "pure_learned_runtime_isolation": {
                "command": "PYTHONPATH=training uv run --frozen python -m pytest -q "
                "tests/python/test_pure_learned_profile.py",
                "status": "passed",
                "stdout_tail": runtime.stdout[-4_000:],
            },
            "rejected_10m": {
                "model": str(
                    root / "local/phase10r-data/checkpoints/phase10r/10m/sparse-pair-policy-wdl/"
                    "sparse-pair-policy-wdl.osaval02"
                ),
                "sha256": _sha256(
                    root / "local/phase10r-data/checkpoints/phase10r/10m/sparse-pair-policy-wdl/"
                    "sparse-pair-policy-wdl.osaval02"
                ),
                "promotion_allowed": False,
                "default": False,
                "selfplay_seed_allowed": False,
            },
            "resource": _resource_receipt(root),
        }
    )
    return receipt


def _canonical_sfen(sfen: str) -> str:
    fields = sfen.split()
    if len(fields) == 4:
        return " ".join(fields[:3])
    if len(fields) == 3:
        return sfen
    raise Phase10SRunError(f"unexpected SFEN field count: {sfen}")


def _sfen_board(sfen: str) -> dict[tuple[int, int], str]:
    board: dict[tuple[int, int], str] = {}
    rows = sfen.split()[0].split("/")
    if len(rows) != 9:
        raise Phase10SRunError("invalid SFEN board")
    for row, encoded in enumerate(rows):
        column = 0
        index = 0
        while index < len(encoded):
            character = encoded[index]
            if character.isdigit():
                column += int(character)
            else:
                if column >= 9:
                    raise Phase10SRunError("invalid SFEN row")
                if character == "+":
                    index += 1
                    if index >= len(encoded) or encoded[index].isdigit():
                        raise Phase10SRunError("invalid promoted SFEN piece")
                    character = "+" + encoded[index]
                board[(column, row)] = character
                column += 1
            index += 1
        if column != 9:
            raise Phase10SRunError("invalid SFEN row width")
    return board


def _usi_square(value: str) -> tuple[int, int]:
    if len(value) != 2 or value[0] not in "123456789" or value[1] not in "abcdefghi":
        raise Phase10SRunError(f"invalid USI square: {value}")
    return 9 - int(value[0]), ord(value[1]) - ord("a")


def _move_is_irreversible(sfen: str, move: str) -> bool:
    if "*" in move or "+" in move:
        return True
    if len(move) < 4:
        raise Phase10SRunError(f"invalid USI move: {move}")
    source = _usi_square(move[:2])
    target = _usi_square(move[2:4])
    board = _sfen_board(sfen)
    piece = board.get(source, "").upper()
    return piece in {"P", "L", "N"} or target in board


def _recurrence_facts(sfens: Sequence[str], moves: Sequence[str]) -> dict[str, Any]:
    canonical = [_canonical_sfen(str(sfen)) for sfen in sfens]
    seen: dict[str, list[int]] = defaultdict(list)
    repeat_events: list[dict[str, Any]] = []
    irreversible_prefix = [0]
    for index, move in enumerate(moves):
        irreversible_prefix.append(
            irreversible_prefix[-1] + int(_move_is_irreversible(str(sfens[index]), str(move)))
        )
    for index, state in enumerate(canonical):
        prior = seen[state]
        if prior:
            previous = prior[-1]
            repeat_events.append(
                {
                    "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                    "first_visit_ply": previous,
                    "repeat_visit_ply": index,
                    "visit_number": len(prior) + 1,
                    "reversible_interval": irreversible_prefix[index]
                    == irreversible_prefix[previous],
                }
            )
        prior.append(index)
    counts = Counter(canonical)
    return {
        "distinct_positions": len(counts),
        "repeated_position_states": sum(count > 1 for count in counts.values()),
        "repeat_visits": sum(count - 1 for count in counts.values() if count > 1),
        "twofold_visits": sum(count >= 2 for count in counts.values()),
        "threefold_visits": sum(count >= 3 for count in counts.values()),
        "maximum_visit_count": max(counts.values(), default=0),
        "reversible_cycles": sum(bool(event["reversible_interval"]) for event in repeat_events),
        "repetitions": repeat_events,
    }


def _pair_metadata(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for pair_receipt_path in sorted(
        (root / "local/phase10r-diagnostic-arena").rglob("pair-receipt.json")
    ):
        receipt = _load_json(pair_receipt_path)
        report = receipt.get("report")
        if not isinstance(report, Mapping) or not isinstance(report.get("path"), str):
            raise Phase10SRunError(
                f"historical pair receipt lacks report identity: {pair_receipt_path}"
            )
        report_path = root / str(report["path"])
        _ensure_regular(report_path, expected_sha256=str(report.get("sha256")))
        report_value = _load_json(report_path)
        for game in report_value.get("games", []):
            csa_path = pair_receipt_path.parent / str(game["csaPath"])
            result[str(csa_path.resolve())] = {
                "group": receipt.get("assignedGroup"),
                "position_id": receipt.get("positionId"),
                "pair_index": receipt.get("pairIndex"),
                "sfen": receipt.get("sfen"),
                "game_id": game.get("id"),
                "black": game.get("black"),
                "white": game.get("white"),
                "report": str(report_path.relative_to(root)),
                "csa_sha256": game.get("csaSha256"),
            }
    return result


def _run_analyze_max_plies(root: Path, argv: Sequence[str]) -> dict[str, Any]:
    boundary = _validate_execution_boundary(root)
    binary = _ensure_cli(root)
    csa_files = sorted((root / "local/phase10r-diagnostic-arena").rglob("*.csa"))
    if not csa_files:
        raise Phase10SRunError("no preserved CSA evidence was found")
    metadata = _pair_metadata(root)
    groups: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    aggregate = Counter()
    games: list[dict[str, Any]] = []
    (root / RUNS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase10s-csa-", dir=root / RUNS_DIRECTORY
    ) as temporary:
        input_dir = Path(temporary) / "input"
        input_dir.mkdir()
        for index, source in enumerate(csa_files):
            target = input_dir / f"game-{index:06d}.csa"
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        output = Path(temporary) / "export.jsonl"
        _run_command(
            [
                str(binary),
                "export-csa-jsonl",
                "--input-dir",
                str(input_dir),
                "--output",
                str(output),
                "--max-games",
                str(len(csa_files)),
            ],
            cwd=root,
            timeout=1_800,
        )
        lines = output.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(csa_files):
            raise Phase10SRunError("CSA replay output count does not match preserved files")
        for index, line in enumerate(lines):
            row = json.loads(line)
            if row.get("status") != "ok":
                raise Phase10SRunError(f"preserved CSA replay failed for {csa_files[index]}")
            source = csa_files[index]
            facts = _recurrence_facts(row["positionSfens"], row["usiMoves"])
            source_metadata = metadata.get(str(source.resolve()), {})
            group = str(source_metadata.get("group", "unknown"))
            groups[group] += 1
            black = str(source_metadata.get("black", ""))
            white = str(source_metadata.get("white", ""))
            if "pure_learned" in black:
                colors["candidate_black"] += 1
            elif "pure_learned" in white:
                colors["candidate_white"] += 1
            aggregate["games"] += 1
            aggregate["max_plies_games"] += int(len(row["usiMoves"]) >= 128)
            aggregate["repeat_visits"] += int(facts["repeat_visits"])
            aggregate["twofold_visits"] += int(facts["twofold_visits"])
            aggregate["threefold_visits"] += int(facts["threefold_visits"])
            aggregate["reversible_cycles"] += int(facts["reversible_cycles"])
            games.append(
                {
                    "source": str(source.relative_to(root)),
                    "group": group,
                    "pair_index": source_metadata.get("pair_index"),
                    "game_id": source_metadata.get("game_id"),
                    "black": black,
                    "white": white,
                    "csa_sha256": _sha256(source),
                    "position_recurrence": facts,
                    "evaluation_magnitude": {
                        "serialized_in_csa": False,
                        "runtime_proof_calls_available": True,
                        "source_receipt": source_metadata.get("report"),
                    },
                }
            )
    receipt = _base_receipt(root, "analyze-max-plies", argv)
    receipt.update(
        {
            "status": "passed",
            "exit_status": 0,
            "boundary": boundary,
            "binary": {"path": str(binary.relative_to(root)), "sha256": _sha256(binary)},
            "preserved_csa_files": len(csa_files),
            "replayed_csa_files": len(games),
            "aggregate": dict(aggregate),
            "start_groups": dict(sorted(groups.items())),
            "colors": dict(sorted(colors.items())),
            "position_recurrence_definition": (
                "exact canonical SFEN including side-to-move and hands, move number removed"
            ),
            "twofold_definition": "a canonical position visited at least twice in one replay",
            "threefold_definition": (
                "a canonical position visited at least three times in one replay"
            ),
            "reversible_cycle_definition": (
                "a repeated canonical position with no pawn/lance/knight move, drop, promotion, "
                "or capture in the interval"
            ),
            "evaluation_magnitude_note": {
                "arena_csa_does_not_store_score_cp": True,
                "per_game_runtime_evaluation_magnitude": "not serialized by the preserved receipt",
                "no_reinterpretation_of_capped_games": True,
            },
            "games": games,
            "resource": _resource_receipt(root),
        }
    )
    return receipt


def _load_checkpoint_model(root: Path, checkpoint: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    from open_shogi_training.phase10r_model import VARIANT_PAIR
    from open_shogi_training.phase10r_training import Phase10RModel

    path = root / str(checkpoint["path"])
    _ensure_regular(path, expected_sha256=str(checkpoint["sha256"]))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - backend error text is receipt evidence
        raise Phase10SRunError(f"cannot load preserved checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("variant_id") != VARIANT_PAIR:
        raise Phase10SRunError(f"preserved checkpoint is not the frozen pair variant: {path}")
    model = Phase10RModel(VARIANT_PAIR, seed=TRAINING_SEED).to("cpu")
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise Phase10SRunError(f"preserved checkpoint tensor identity failed: {path}") from error
    model.eval()
    return model, payload


def _fit_calibration(
    model: Any, validation_path: Path, *, maximum_rows: int = 2_048
) -> dict[str, Any]:
    import torch

    from open_shogi_training.phase10r_training import Phase10RExample

    xs: list[float] = []
    ys: list[float] = []
    source_counts: Counter[str] = Counter()
    model.eval()
    with validation_path.open(encoding="utf-8") as handle, torch.no_grad():
        for line in handle:
            if len(xs) >= maximum_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            example = Phase10RExample.from_mapping(row)
            example.validate()
            if not example.wdl_mask or example.wdl is None:
                continue
            output = model.forward_example(example)
            probabilities = torch.softmax(output["values"][:3], dim=0)
            epsilon = 1.0e-6
            raw = math.log(
                (float(probabilities[2].cpu()) + epsilon)
                / (float(probabilities[0].cpu()) + epsilon)
            )
            target = float(example.wdl - 1)
            if not math.isfinite(raw):
                raise Phase10SRunError("non-finite intermediate calibration input")
            xs.append(raw)
            ys.append(target)
            source_counts[example.source] += 1
    if len(xs) < 64:
        raise Phase10SRunError("intermediate calibration population is too small")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0.0:
        raise Phase10SRunError("intermediate calibration population has no variance")
    scale = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    bias = mean_y - scale * mean_x
    if not math.isfinite(scale) or not math.isfinite(bias) or scale <= 0.0:
        raise Phase10SRunError("intermediate calibration is not positive monotonic")
    return {
        "schema": "open_shogiai_phase10s_calibration/v1",
        "population": "non_final_validation",
        "rows": len(xs),
        "source_counts": dict(sorted(source_counts.items())),
        "calibration_scale": scale,
        "calibration_bias": bias,
        "positive_monotonic": True,
        "frozen_before_arena": True,
    }


def _write_calibration(root: Path, checkpoint_sha256: str, calibration: Mapping[str, Any]) -> Path:
    path = root / RUNS_DIRECTORY / "calibration" / f"{checkpoint_sha256}.json"
    if path.exists() or path.is_symlink():
        existing = _load_json(path)
        if existing != dict(calibration):
            raise Phase10SRunError(f"calibration identity changed: {path}")
    else:
        _write_new_json(path, calibration)
    return path


def _evaluate_checkpoint(
    root: Path, model: Any, *, path: Path, expected_rows: int, data_root: Path
) -> dict[str, Any]:
    from open_shogi_training.phase10r_campaign import _evaluate_file

    return _evaluate_file(path, expected_rows=expected_rows, model=model, data_root=data_root)


def _run_checkpoint_offline(root: Path, argv: Sequence[str]) -> dict[str, Any]:
    boundary = _validate_execution_boundary(root)
    manifest_path = (
        root / "local/phase10r-data/phase10r-prepared/10m-mixture-v2/preparation-manifest.json"
    )
    manifest = _load_json(manifest_path)
    manifest_sha256 = str(manifest["manifest_sha256"])
    validation_path = (
        root / "local/phase10r-data/phase10r-prepared/10m-mixture-v2/base-validation.jsonl"
    )
    held_out_path = (
        root / "local/phase10r-data/phase10r-prepared/10m-mixture-v2/base-source_held_out.jsonl"
    )
    _ensure_regular(validation_path)
    _ensure_regular(held_out_path)
    held_out_rows = int(manifest["files"]["base-source_held_out.jsonl"]["rows"])
    data_root = root / "local/phase10r-data"
    baseline_evaluation = _load_json(data_root / "evaluations/1m/evaluation.json")
    baseline_candidates = baseline_evaluation.get("candidates")
    if not isinstance(baseline_candidates, list):
        raise Phase10SRunError("selected 1M source-held-out baseline is unavailable")
    baseline = next(
        (
            item["source_held_out"]
            for item in baseline_candidates
            if item.get("variant") == "sparse-pair-policy-wdl"
        ),
        None,
    )
    if not isinstance(baseline, Mapping):
        raise Phase10SRunError("selected 1M pair source-held-out baseline is unavailable")
    candidates: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        model, payload = _load_checkpoint_model(root, checkpoint)
        held_out = _evaluate_checkpoint(
            root, model, path=held_out_path, expected_rows=held_out_rows, data_root=data_root
        )
        regression = {
            "policy_nll": float(held_out["policy_nll"]) - float(baseline["policy_nll"]),
            "policy_top1": float(baseline["policy_top1"]) - float(held_out["policy_top1"]),
            "wdl_brier": float(held_out["wdl_brier"]) - float(baseline["wdl_brier"]),
            "wdl_nll": float(held_out["wdl_nll"]) - float(baseline["wdl_nll"]),
            "calibration_ece": float(held_out["calibration_ece"])
            - float(baseline["calibration_ece"]),
        }
        source_held_out_gate_passed = max(regression.values()) <= 0.01
        calibration = _fit_calibration(model, validation_path)
        calibration_path = _write_calibration(root, str(checkpoint["sha256"]), calibration)
        candidates.append(
            {
                "id": checkpoint["id"],
                "checkpoint": str(checkpoint["path"]),
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_manifest_sha256": payload.get("manifest_sha256"),
                "stream_index": payload.get(
                    "stream_index", payload.get("sampler", {}).get("cursor")
                ),
                "source_held_out": held_out,
                "source_held_out_baseline": baseline,
                "source_held_out_regressions": regression,
                "source_held_out_gate_passed": source_held_out_gate_passed,
                "calibration": calibration,
                "calibration_path": str(calibration_path.relative_to(root)),
                "temporary_osaval02_export": "deferred_until_arena_intermediates",
            }
        )
    receipt = _base_receipt(root, "checkpoint-offline", argv)
    receipt.update(
        {
            "status": "passed",
            "exit_status": 0,
            "boundary": boundary,
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": manifest_sha256,
            "validation": {
                "path": str(validation_path.relative_to(root)),
                "sha256": _sha256(validation_path),
                "final_holdout_accessed": False,
            },
            "source_held_out": {
                "path": str(held_out_path.relative_to(root)),
                "sha256": _sha256(held_out_path),
                "rows": held_out_rows,
                "final_holdout_accessed": False,
            },
            "source_held_out_baseline": {
                "scale": "selected_1m_control",
                "metrics": dict(baseline),
                "final_holdout_accessed": False,
            },
            "preserved_checkpoint_evaluations": candidates,
            "full_source_held_out_table_reproduced": True,
            "training_batch_loss_used_for_selection": False,
            "resource": _resource_receipt(root),
        }
    )
    return receipt


def _runtime_proofs(value: object) -> list[Mapping[str, Any]]:
    proofs: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if value.get("profile") == "pure_learned":
            proofs.append(value)
        for child in value.values():
            proofs.extend(_runtime_proofs(child))
    elif isinstance(value, list):
        for child in value:
            proofs.extend(_runtime_proofs(child))
    return proofs


def _candidate_is_a(game: Mapping[str, Any], candidate_label: str) -> bool:
    black = str(game.get("black", ""))
    white = str(game.get("white", ""))
    return (candidate_label in black) or (candidate_label in white)


def _candidate_result(game: Mapping[str, Any], candidate_label: str) -> str:
    result = str(game.get("result"))
    if result == "draw" or result == "max_plies":
        return result
    candidate_black = candidate_label in str(game.get("black", ""))
    candidate_white = candidate_label in str(game.get("white", ""))
    if not candidate_black and not candidate_white:
        raise Phase10SRunError("candidate is absent from Arena game")
    winner = "black" if result == "black_win" else "white" if result == "white_win" else None
    if winner is None:
        raise Phase10SRunError(f"unknown Arena result: {result}")
    return (
        "win"
        if (winner == "black" and candidate_black) or (winner == "white" and candidate_white)
        else "loss"
    )


def _paired_bootstrap_lower(pair_scores: Sequence[float], *, seed: int) -> float | None:
    if not pair_scores:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    size = len(pair_scores)
    for _ in range(20_000):
        total = sum(pair_scores[rng.randrange(size)] for _ in range(size))
        means.append(total / size)
    means.sort()
    return means[max(0, int(0.025 * len(means)) - 1)]


def _wilson_lower(wins: int, draws: int, losses: int) -> float | None:
    finished = wins + draws + losses
    if finished == 0:
        return None
    # The frozen score treats a draw as one half point.  This is the normal
    # approximation used only for the diagnostic receipt; paired bootstrap is
    # retained as the primary paired gate.
    successes = wins + 0.5 * draws
    n = float(finished)
    z = 1.959963984540054
    phat = successes / n
    denominator = 1.0 + z * z / n
    center = phat + z * z / (2.0 * n)
    spread = z * math.sqrt(max(0.0, phat * (1.0 - phat) / n + z * z / (4.0 * n * n)))
    return (center - spread) / denominator


def _aggregate_arena_reports(
    root: Path,
    *,
    rung_id: str,
    pair_root: Path,
    pair_count: int,
    candidate_label: str,
    candidate_model_sha256: str,
    start_positions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    wins = draws = losses = excluded = illegal = crashes = 0
    pair_scores: list[float] = []
    proofs: list[Mapping[str, Any]] = []
    report_paths: list[str] = []
    for pair_index in range(pair_count):
        report_path = pair_root / f"pair-{pair_index:04d}" / "arena-report.json"
        _ensure_regular(report_path)
        report = _load_json(report_path)
        if report.get("schema") != "phase2_arena_report/v2":
            raise Phase10SRunError(f"Arena report schema changed: {report_path}")
        games = report.get("games")
        if not isinstance(games, list) or len(games) != 2:
            raise Phase10SRunError(f"Arena pair is not exactly two games: {report_path}")
        pair_points: list[float] = []
        for game in games:
            if not isinstance(game, Mapping):
                raise Phase10SRunError(f"Arena game record is malformed: {report_path}")
            result = _candidate_result(game, candidate_label)
            if result == "max_plies":
                excluded += 1
            elif result == "win":
                wins += 1
                pair_points.append(1.0)
            elif result == "draw":
                draws += 1
                pair_points.append(0.5)
            elif result == "loss":
                losses += 1
                pair_points.append(0.0)
            else:
                raise Phase10SRunError(f"unknown candidate result: {result}")
            if int(game.get("moves", 0)) > 128:
                raise Phase10SRunError(f"Arena game exceeded the frozen ply cap: {report_path}")
        metrics = report.get("metrics", {})
        illegal += int(metrics.get("illegalMoves", 0))
        proofs.extend(_runtime_proofs(report))
        if len(pair_points) == 2:
            pair_scores.append(sum(pair_points) / 2.0)
        report_paths.append(str(report_path.relative_to(root)))
    candidate_proofs = [
        proof for proof in proofs if proof.get("model_sha256") == candidate_model_sha256
    ]
    for proof in proofs:
        for counter in FROZEN_ARENA_COUNTERS:
            if int(proof.get(counter, 0)) != 0:
                raise Phase10SRunError(f"prohibited pure-learned counter is nonzero: {counter}")
    if (
        not candidate_proofs
        or sum(int(proof.get("learned_eval_calls", 0)) for proof in candidate_proofs) <= 0
    ):
        raise Phase10SRunError("candidate pure-learned Arena proof has no learned calls")
    finished = wins + draws + losses
    score = (wins + 0.5 * draws) / finished if finished else None
    aggregate = {
        "schema": "open_shogiai_phase10s_arena_summary/v1",
        "status": "passed" if illegal == 0 and crashes == 0 else "failed",
        "rung": rung_id,
        "controls": {
            "games": pair_count * 2,
            "pairs": pair_count,
            "movetime_ms": 10,
            "depth_cap": 8,
            "hash_mib_per_player": 32,
            "max_plies": 128,
            "paired_reversed_colors": True,
            "opening_book": False,
            "opening_style": "unrestricted",
        },
        "candidate_model_sha256": candidate_model_sha256,
        "candidate_label": candidate_label,
        "start_pool_sha256": START_POOL_SHA256,
        "start_positions": [
            {
                "pair_index": index,
                "position_id": position.get("positionId"),
                "assigned_group": position.get("assignedGroup"),
                "sfen": position.get("sfen"),
            }
            for index, position in enumerate(start_positions[:pair_count])
        ],
        "reports": report_paths,
        "results": {
            "wdl": {"wins": wins, "draws": draws, "losses": losses},
            "finished_wld_games": finished,
            "max_plies_exclusions": excluded,
            "score_rate": score,
            "wilson_lower_95": _wilson_lower(wins, draws, losses),
            "paired_bootstrap_lower_95": _paired_bootstrap_lower(pair_scores, seed=TRAINING_SEED),
            "paired_complete_pairs": len(pair_scores),
            "illegal_moves": illegal,
            "unexplained_crashes": crashes,
        },
        "runtime_proof": {
            "proof_count": len(proofs),
            "candidate_proof_count": len(candidate_proofs),
            "prohibited_counters_zero": all(
                int(proof.get(counter, 0)) == 0
                for proof in proofs
                for counter in FROZEN_ARENA_COUNTERS
            ),
            "learned_eval_calls": sum(
                int(proof.get("learned_eval_calls", 0)) for proof in candidate_proofs
            ),
        },
    }
    return aggregate


def _arena_command(
    *,
    binary: Path,
    root: Path,
    pair_dir: Path,
    position: Mapping[str, Any],
    pair_index: int,
    candidate_model: Path,
    candidate_model_sha256: str,
    opponent: str,
    opponent_model: Path | None,
    opponent_model_sha256: str | None,
    git_commit: str,
    resume: bool,
) -> list[str]:
    sfen = str(position["sfen"]).strip()
    if len(sfen.split()) == 3:
        sfen = f"{sfen} 1"
    command = [
        str(binary),
        "arena",
        "--games",
        "2",
        "--player-a",
        "pure_learned",
        "--a-model",
        str(candidate_model),
        "--a-model-sha256",
        candidate_model_sha256,
        "--a-opening-profile",
        "unrestricted",
        "--player-b",
        opponent,
        "--b-opening-profile",
        "unrestricted",
        "--movetime-ms",
        "10",
        "--a-depth",
        "8",
        "--b-depth",
        "8",
        "--a-hash-mb",
        "32",
        "--b-hash-mb",
        "32",
        "--max-plies",
        "128",
        "--seed",
        str(TRAINING_SEED + pair_index),
        "--git-commit",
        git_commit,
        "--sfen",
        sfen,
        "--output-dir",
        str(pair_dir.relative_to(root)),
    ]
    if opponent_model is not None:
        command.extend(
            ["--b-model", str(opponent_model), "--b-model-sha256", str(opponent_model_sha256)]
        )
    if resume:
        command.append("--resume")
    return command


def _run_arena_rung(
    root: Path,
    *,
    rung_id: str,
    pair_count: int,
    candidate_model: Path,
    candidate_model_sha256: str,
    opponent: str,
    opponent_model: Path | None = None,
    opponent_model_sha256: str | None = None,
    resume: bool,
) -> dict[str, Any]:
    boundary = _validate_execution_boundary(root)
    manifest, positions = _start_pool(root)
    binary = _ensure_cli(root)
    git_commit = _git(root, "rev-parse", "HEAD")
    _ensure_regular(candidate_model, expected_sha256=candidate_model_sha256)
    if opponent_model is not None:
        _ensure_regular(opponent_model, expected_sha256=opponent_model_sha256)
    pair_root = root / RUNS_DIRECTORY / "arena" / rung_id / "pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    for pair_index in range(pair_count):
        pair_dir = pair_root / f"pair-{pair_index:04d}"
        report = pair_dir / "arena-report.json"
        if report.is_file() and not report.is_symlink():
            continue
        command = _arena_command(
            binary=binary,
            root=root,
            pair_dir=pair_dir,
            position=positions[pair_index],
            pair_index=pair_index,
            candidate_model=candidate_model,
            candidate_model_sha256=candidate_model_sha256,
            opponent=opponent,
            opponent_model=opponent_model,
            opponent_model_sha256=opponent_model_sha256,
            git_commit=git_commit,
            resume=resume and pair_dir.exists(),
        )
        _run_command(command, cwd=root)
        if not report.is_file():
            raise Phase10SRunError(f"Arena did not produce a report: {report}")
        if pair_index % 10 == 0 or pair_index + 1 == pair_count:
            print(
                json.dumps(
                    {
                        "command": "arena",
                        "rung": rung_id,
                        "completed_pairs": pair_index + 1,
                        "total_pairs": pair_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    aggregate = _aggregate_arena_reports(
        root,
        rung_id=rung_id,
        pair_root=pair_root,
        pair_count=pair_count,
        candidate_label=f"m-{candidate_model_sha256[:12]}",
        candidate_model_sha256=candidate_model_sha256,
        start_positions=positions,
    )
    aggregate["boundary"] = boundary
    aggregate["binary"] = {"path": str(binary.relative_to(root)), "sha256": _sha256(binary)}
    aggregate["manifest"] = {
        "path": str(START_POOL),
        "sha256": START_POOL_SHA256,
        "schema": manifest.get("schema"),
    }
    aggregate["resource"] = _resource_receipt(root)
    summary_path = root / RUNS_DIRECTORY / "arena" / rung_id / "aggregate.json"
    if summary_path.exists() or summary_path.is_symlink():
        existing = _load_json(summary_path)
        if existing != aggregate:
            raise Phase10SRunError(f"Arena aggregate changed on resume: {summary_path}")
    else:
        _write_new_json(summary_path, aggregate)
    return aggregate


def _run_arena(root: Path, argv: Sequence[str], rung: str, resume: bool) -> dict[str, Any]:
    if rung != "selected-1m-vs-handcrafted-baseline":
        raise Phase10SRunError(f"unsupported frozen Arena rung: {rung}")
    aggregate = _run_arena_rung(
        root,
        rung_id=rung,
        pair_count=400,
        candidate_model=root / SELECTED_MODEL,
        candidate_model_sha256=SELECTED_MODEL_SHA256,
        opponent="handcrafted-experimental",
        resume=resume,
    )
    receipt = _base_receipt(root, "arena", argv)
    receipt.update(
        {
            "status": aggregate["status"],
            "exit_status": 0 if aggregate["status"] == "passed" else 2,
            "rung": rung,
            "arena": aggregate,
            "resource": _resource_receipt(root),
        }
    )
    return receipt


def _export_intermediate(
    root: Path, checkpoint: Mapping[str, Any], *, calibration: Mapping[str, Any], git_commit: str
) -> tuple[Path, str]:
    from open_shogi_training.phase10r_model import (
        VARIANT_PAIR,
        parse_osaval02,
        serialize_osaval02,
    )

    model, _ = _load_checkpoint_model(root, checkpoint)
    output = root / RUNS_DIRECTORY / "intermediate-models" / f"{checkpoint['id']}.osaval02"
    manifest_path = (
        root / "local/phase10r-data/phase10r-prepared/10m-mixture-v2/preparation-manifest.json"
    )
    manifest = _load_json(manifest_path)
    artifact = serialize_osaval02(
        model.export_tensors(),
        variant_id=VARIANT_PAIR,
        quantization="float32",
        dataset_manifest_sha256=str(manifest["manifest_sha256"]),
        training_run_reference=f"phase10s-{checkpoint['id']}",
        git_commit=git_commit,
        calibration_scale=float(calibration["calibration_scale"]),
        calibration_bias=float(calibration["calibration_bias"]),
    )
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    if output.exists() or output.is_symlink():
        _ensure_regular(output, expected_sha256=artifact_sha256)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as handle:
                handle.write(artifact)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise Phase10SRunError(
                f"refusing to overwrite intermediate export: {output}"
            ) from error
    parsed = parse_osaval02(output.read_bytes())
    if parsed.variant_id != VARIANT_PAIR or parsed.artifact_sha256 != artifact_sha256:
        raise Phase10SRunError("temporary OSAVAL02 export failed identity validation")
    return output, artifact_sha256


def _latest_receipt(root: Path, command: str) -> Path | None:
    paths = sorted((root / RUNS_DIRECTORY).glob(f"*-{command}.json"))
    return paths[-1] if paths else None


def _run_arena_intermediates(root: Path, argv: Sequence[str], resume: bool) -> dict[str, Any]:
    boundary = _validate_execution_boundary(root)
    offline_path = _latest_receipt(root, "checkpoint-offline")
    if offline_path is None:
        raise Phase10SRunError("checkpoint-offline must pass before intermediate Arena")
    offline = _load_json(offline_path)
    if offline.get("status") != "passed":
        raise Phase10SRunError("checkpoint-offline did not pass")
    calibrations = {
        str(candidate["id"]): candidate["calibration"]
        for candidate in offline.get("preserved_checkpoint_evaluations", [])
    }
    results: list[dict[str, Any]] = []
    git_commit = _git(root, "rev-parse", "HEAD")
    for checkpoint in CHECKPOINTS[:3]:
        calibration = calibrations.get(str(checkpoint["id"]))
        if not isinstance(calibration, Mapping):
            raise Phase10SRunError(f"calibration is missing for {checkpoint['id']}")
        export, export_sha256 = _export_intermediate(
            root, checkpoint, calibration=calibration, git_commit=git_commit
        )
        aggregate = _run_arena_rung(
            root,
            rung_id=str(checkpoint["id"]),
            pair_count=100,
            candidate_model=export,
            candidate_model_sha256=export_sha256,
            opponent="pure_learned",
            opponent_model=root / SELECTED_MODEL,
            opponent_model_sha256=SELECTED_MODEL_SHA256,
            resume=resume,
        )
        aggregate["parent_checkpoint"] = str(checkpoint["path"])
        aggregate["parent_checkpoint_sha256"] = checkpoint["sha256"]
        aggregate["temporary_export"] = str(export.relative_to(root))
        aggregate["temporary_export_sha256"] = export_sha256
        results.append(aggregate)
    activation: list[dict[str, Any]] = []
    for aggregate in results:
        score = aggregate["results"].get("score_rate")
        offline_candidate = next(
            item
            for item in offline["preserved_checkpoint_evaluations"]
            if item["id"] == aggregate["rung"]
        )
        offline_ok = bool(offline_candidate.get("source_held_out_gate_passed", False))
        activation.append(
            {
                "checkpoint": aggregate["rung"],
                "score_vs_selected_1m": score,
                "score_at_least_0_45": score is not None and score >= 0.45,
                "offline_gates_passed": offline_ok,
                "promising_handcrafted_rung_authorized": bool(
                    score is not None and score >= 0.45 and offline_ok
                ),
            }
        )
    passed = not any(item["promising_handcrafted_rung_authorized"] for item in activation)
    receipt = _base_receipt(root, "arena-intermediates", argv)
    receipt.update(
        {
            "status": "passed" if passed else "blocked",
            "exit_status": 0 if passed else 2,
            "boundary": boundary,
            "intermediate_arenas": results,
            "activation": activation,
            "stop_before_training": passed,
            "stop_reason": (
                "no preserved checkpoint passed both the >=45% cross-play score and offline gates"
                if passed
                else None
            ),
            "resource": _resource_receipt(root),
        }
    )
    return receipt


def _find_latest_aggregate(root: Path) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    arena_root = root / RUNS_DIRECTORY / "arena"
    if arena_root.is_dir():
        for path in sorted(arena_root.glob("*/aggregate.json")):
            aggregates.append(_load_json(path))
    return aggregates


def _write_final_report(root: Path) -> tuple[Path, Path]:
    boundary = _validate_execution_boundary(root)
    receipts: dict[str, list[str]] = defaultdict(list)
    for path in sorted((root / RUNS_DIRECTORY).glob("*-*.json")):
        if path.name in {REPORT_JSON.name} or path.name.startswith("PHASE10S_EXECUTION_REPORT"):
            continue
        try:
            value = _load_json(path)
        except Phase10SRunError:
            continue
        command = value.get("command")
        if isinstance(command, str):
            receipts[command].append(str(path.relative_to(root)))
    aggregates = _find_latest_aggregate(root)
    intermediate_receipt_path = _latest_receipt(root, "arena-intermediates")
    intermediate_receipt = (
        _load_json(intermediate_receipt_path) if intermediate_receipt_path else None
    )
    stop_reason = None
    if isinstance(intermediate_receipt, Mapping):
        stop_reason = intermediate_receipt.get("stop_reason")
    if stop_reason is None:
        stop_reason = "Phase 10S diagnostics have not reached a review-gate decision"
    report: dict[str, Any] = {
        "schema": "open_shogi_ai_phase10s_execution_report/v1",
        "status": "review_gate",
        "boundary": boundary,
        "frozen_controls": {
            "hash_manifest": boundary["frozen_hashes"],
            "start_pool": {"path": str(START_POOL), "sha256": START_POOL_SHA256},
        },
        "preserved_artifacts": _load_json(root / "configs/phase10s/preservation-manifest.json")[
            "artifacts"
        ],
        "rejected_10m": {
            "status": "rejected_for_promotion_and_selfplay",
            "default": False,
            "selfplay_seed_allowed": False,
            "promotion_allowed": False,
        },
        "unique_first_supervised_contract": {
            "eligible_unique_factual_rows": 375_505,
            "maximum_factual_exposures": 751_010,
            "maximum_exposures_per_row": 2,
            "status": "not_started_before_activation_gate",
        },
        "checkpoint_lineage_and_calibration_receipts": receipts.get("checkpoint-offline", []),
        "arena_receipts": receipts.get("arena", []) + receipts.get("arena-intermediates", []),
        "arena_aggregates": aggregates,
        "crossplay_receipts": receipts.get("crossplay", []),
        "selfplay_receipts": receipts.get("selfplay", []),
        "runtime_proofs": {
            "pure_learned_profile_test": receipts.get("preflight", []),
            "prohibited_call_counters": "zero required for every pure-learned Arena entrant",
        },
        "resources": {
            "final": _resource_receipt(root),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
        },
        "stop_reason": stop_reason,
        "forbidden_actions_not_performed": [
            "pure_selfplay_before_frozen_entry_gate",
            "final_holdout_inspection",
            "overall_champion_promotion",
            "merge",
            "push",
            "release",
            "deploy",
            "OpenShogiUI modification",
        ],
        "exact_next_command": (
            "Review this receipt at the frozen Phase 10S review gate; no further execution is "
            "authorized until a new supervised candidate passes the required offline and "
            "cross-play gates."
        ),
        "receipt_inventory": {key: value for key, value in sorted(receipts.items())},
    }
    _write_new_json(root / REPORT_JSON, report)
    markdown = "\n".join(
        [
            "# Phase 10S Execution Report",
            "",
            f"Status: `{report['status']}`",
            "",
            f"Stop reason: {stop_reason}",
            "",
            f"Branch: `{boundary['branch']}`; commit: `{boundary['commit']}`.",
            "",
            "The rejected 10M model remains non-default, non-promotable, and ineligible as a "
            "self-play seed.",
            "",
            "No final holdout was inspected and no self-play/promotion/release action was "
            "performed.",
            "",
            "See the JSON report for immutable receipt paths, hashes, source metrics, runtime "
            "proofs, and resource evidence.",
            "",
        ]
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (root / REPORT_MD).open("xb") as handle:
            handle.write(markdown.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise Phase10SRunError(f"refusing to overwrite immutable report: {REPORT_MD}") from error
    return root / REPORT_JSON, root / REPORT_MD


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "analyze-max-plies", "checkpoint-offline"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("."))
    arena = subparsers.add_parser("arena")
    arena.add_argument("--root", type=Path, default=Path("."))
    arena.add_argument("--rung", required=True)
    arena.add_argument("--resume", action="store_true")
    intermediates = subparsers.add_parser("arena-intermediates")
    intermediates.add_argument("--root", type=Path, default=Path("."))
    intermediates.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, default=Path("."))
    return parser


def _dispatch(arguments: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    root = arguments.root.resolve()
    if root.name != "OpenShogiAI":
        raise Phase10SRunError(f"unexpected repository root: {root}")
    if arguments.command == "preflight":
        return _run_preflight(root, argv)
    if arguments.command == "analyze-max-plies":
        return _run_analyze_max_plies(root, argv)
    if arguments.command == "checkpoint-offline":
        return _run_checkpoint_offline(root, argv)
    if arguments.command == "arena":
        return _run_arena(root, argv, arguments.rung, arguments.resume)
    if arguments.command == "arena-intermediates":
        return _run_arena_intermediates(root, argv, arguments.resume)
    if arguments.command == "report":
        receipt = _base_receipt(root, "report", argv)
        json_path, markdown_path = _write_final_report(root)
        receipt.update(
            {
                "status": "passed",
                "exit_status": 0,
                "report_json": str(json_path.relative_to(root)),
                "report_markdown": str(markdown_path.relative_to(root)),
            }
        )
        return receipt
    raise AssertionError(f"unknown command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    raise SystemExit("Closed campaign: see docs/status.md; use current development commands")


if __name__ == "__main__":
    raise SystemExit(main())
