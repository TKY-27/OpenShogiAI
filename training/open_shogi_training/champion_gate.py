"""Predeclared equal-wall-clock overall-champion promotion gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_shogi_training.labeling.artifacts import read_regular_bytes
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    artifact_ref,
    contained_path,
    ensure_contained_directory,
    load_json_and_ref,
    require_clean_head,
    validate_identifier,
    validate_relative_path,
    write_json_new,
)

SCHEMA = "open_shogi_overall_champion_gate/v1"
SUITE_SCHEMA = "open_shogi_tactical_suite/v1"
TACTICAL_REPORT_SCHEMA = "open_shogi_tactical_gate_report/v1"
DECISION_SCHEMA = "open_shogi_overall_champion_decision/v1"


@dataclass(frozen=True, slots=True)
class GateConfig:
    sha256: str
    games: int
    movetime_ms: int
    depth: int
    hash_mib: int
    max_plies: int
    seed: int
    candidate_id: str
    candidate_kind: str
    model_path: str
    model_sha256: str
    model_size: int
    incumbent_id: str
    confidence_level: float
    minimum_decisive_games: int
    minimum_score_rate: float
    minimum_wilson_lower: float
    maximum_illegal_moves: int
    maximum_crashes: int
    maximum_tactical_regressions: int
    suite_path: str
    suite_sha256: str
    tactical_nodes: int
    tactical_timeout_seconds: int


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        root = Path(options.project_root).resolve(strict=True)
        config = load_gate_config(contained_path(root, validate_relative_path(options.config)))
        if options.command == "preflight":
            result = preflight(
                root,
                config,
                engine=options.engine,
                output_dir=options.output_dir,
            )
        elif options.command == "tactical-run":
            result = run_tactical_suite(
                root,
                config,
                engine=options.engine,
                output=options.output,
            )
        else:
            result = verify_gate(
                root,
                config,
                engine=options.engine,
                arena_report=options.arena_report,
                tactical_report=options.tactical_report,
                output=options.output,
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    )
    return 0


def load_gate_config(path: Path) -> GateConfig:
    raw, digest = read_regular_bytes(path, max_bytes=64 * 1024)
    try:
        value = tomllib.loads(raw.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot parse overall-champion gate config: {error}") from error
    root = _closed(
        value,
        "gate config",
        {"schema", "arena", "candidate", "incumbent", "thresholds", "tactical"},
    )
    if root["schema"] != SCHEMA:
        raise ValueError(f"gate config schema must be {SCHEMA}")
    arena = _closed(
        root["arena"],
        "arena",
        {
            "games",
            "paired_colors",
            "movetime_ms",
            "depth",
            "hash_mib",
            "max_plies",
            "seed",
            "opening_enabled",
        },
    )
    if arena["paired_colors"] is not True or arena["opening_enabled"] is not False:
        raise ValueError("overall gate requires paired colors with opening disabled")
    candidate = _closed(
        root["candidate"], "candidate", {"id", "kind", "model_path", "model_sha256", "model_size"}
    )
    incumbent = _closed(root["incumbent"], "incumbent", {"id", "kind"})
    thresholds = _closed(
        root["thresholds"],
        "thresholds",
        {
            "confidence_level",
            "minimum_decisive_games",
            "minimum_score_rate",
            "minimum_wilson_lower",
            "maximum_illegal_moves",
            "maximum_crashes",
            "maximum_tactical_regressions",
        },
    )
    tactical = _closed(
        root["tactical"], "tactical", {"suite_path", "suite_sha256", "nodes", "timeout_seconds"}
    )
    candidate_id = _identifier(candidate["id"], "candidate.id")
    incumbent_id = _identifier(incumbent["id"], "incumbent.id")
    candidate_kind = _text(candidate["kind"], "candidate.kind")
    if candidate_kind not in {"neural", "residual", "composite"}:
        raise ValueError("candidate.kind must be neural, residual, or composite")
    if (
        incumbent_id != "handcrafted-experimental"
        or incumbent.get("kind") != "handcrafted-experimental"
    ):
        raise ValueError("overall gate incumbent must be handcrafted-experimental")
    confidence = _number(thresholds["confidence_level"], "thresholds.confidence_level")
    if confidence != 0.95:
        raise ValueError("overall gate v1 supports exactly 95% Wilson confidence")
    sha = _sha(candidate["model_sha256"], "candidate.model_sha256")
    suite_sha = _sha(tactical["suite_sha256"], "tactical.suite_sha256")
    games = _integer(arena["games"], "arena.games", 2, 10_000)
    if games % 2:
        raise ValueError("arena.games must be even for paired colors")
    config = GateConfig(
        sha256=digest.sha256,
        games=games,
        movetime_ms=_integer(arena["movetime_ms"], "arena.movetime_ms", 1, 20_000),
        depth=_integer(arena["depth"], "arena.depth", 1, 64),
        hash_mib=_integer(arena["hash_mib"], "arena.hash_mib", 1, 1024),
        max_plies=_integer(arena["max_plies"], "arena.max_plies", 1, 512),
        seed=_integer(arena["seed"], "arena.seed", 0, 9_007_199_254_740_991),
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        model_path=validate_relative_path(_text(candidate["model_path"], "candidate.model_path")),
        model_sha256=sha,
        model_size=_integer(candidate["model_size"], "candidate.model_size", 1, 256 * 1024 * 1024),
        incumbent_id=incumbent_id,
        confidence_level=confidence,
        minimum_decisive_games=_integer(
            thresholds["minimum_decisive_games"], "thresholds.minimum_decisive_games", 1, games
        ),
        minimum_score_rate=_rate(thresholds["minimum_score_rate"], "thresholds.minimum_score_rate"),
        minimum_wilson_lower=_rate(
            thresholds["minimum_wilson_lower"], "thresholds.minimum_wilson_lower"
        ),
        maximum_illegal_moves=_integer(
            thresholds["maximum_illegal_moves"], "thresholds.maximum_illegal_moves", 0, games
        ),
        maximum_crashes=_integer(
            thresholds["maximum_crashes"], "thresholds.maximum_crashes", 0, games
        ),
        maximum_tactical_regressions=_integer(
            thresholds["maximum_tactical_regressions"],
            "thresholds.maximum_tactical_regressions",
            0,
            1000,
        ),
        suite_path=validate_relative_path(_text(tactical["suite_path"], "tactical.suite_path")),
        suite_sha256=suite_sha,
        tactical_nodes=_integer(tactical["nodes"], "tactical.nodes", 1, 1_000_000_000),
        tactical_timeout_seconds=_integer(
            tactical["timeout_seconds"], "tactical.timeout_seconds", 1, 120
        ),
    )
    if (
        config.maximum_illegal_moves != 0
        or config.maximum_crashes != 0
        or config.maximum_tactical_regressions != 0
    ):
        raise ValueError(
            "overall champion gate must fail on any illegal move, crash, or tactical regression"
        )
    _validate_bound_artifacts(path.parents[2], config)
    return config


def preflight(root: Path, config: GateConfig, *, engine: str, output_dir: str) -> dict[str, Any]:
    commit = require_clean_head(root)
    engine_ref = _checked_ref(root, engine, maximum_bytes=256 * 1024 * 1024)
    model_ref = _model_ref(root, config)
    output = validate_relative_path(output_dir)
    command = [
        engine_ref.path,
        "arena",
        "--games",
        str(config.games),
        "--player-a",
        config.candidate_kind,
        "--a-model",
        model_ref.path,
        "--player-b",
        "handcrafted-experimental",
        "--a-depth",
        str(config.depth),
        "--b-depth",
        str(config.depth),
        "--a-hash-mb",
        str(config.hash_mib),
        "--b-hash-mb",
        str(config.hash_mib),
        "--movetime-ms",
        str(config.movetime_ms),
        "--max-plies",
        str(config.max_plies),
        "--seed",
        str(config.seed),
        "--git-commit",
        commit,
        "--output-dir",
        output,
    ]
    return {
        "schema": "open_shogi_overall_champion_preflight/v1",
        "configSha256": config.sha256,
        "gitCommit": commit,
        "engine": engine_ref.as_dict(),
        "model": model_ref.as_dict(),
        "pairedEqualWallClock": True,
        "command": command,
    }


def run_tactical_suite(
    root: Path, config: GateConfig, *, engine: str, output: str
) -> dict[str, Any]:
    commit = require_clean_head(root)
    engine_ref = _checked_ref(root, engine, maximum_bytes=256 * 1024 * 1024)
    model_ref = _model_ref(root, config)
    suite_ref, cases = _load_suite(root, config)
    results = []
    for case in cases:
        incumbent = _usi_search(root, root / engine_ref.path, case["sfen"], config, model=None)
        candidate = _usi_search(root, root / engine_ref.path, case["sfen"], config, model=model_ref)
        expected = case["expectedMoves"]
        incumbent["passed"] = incumbent["bestmove"] in expected
        candidate["passed"] = candidate["bestmove"] in expected
        results.append({**case, "incumbent": incumbent, "candidate": candidate})
    regressions = sum(
        row["incumbent"]["passed"] and not row["candidate"]["passed"] for row in results
    )
    payload = {
        "schema": TACTICAL_REPORT_SCHEMA,
        "configSha256": config.sha256,
        "gitCommit": commit,
        "engine": engine_ref.as_dict(),
        "suite": suite_ref.as_dict(),
        "candidate": {
            "id": config.candidate_id,
            "kind": config.candidate_kind,
            "model": model_ref.as_dict(),
        },
        "incumbent": {"id": config.incumbent_id, "kind": "handcrafted-experimental"},
        "nodesPerCase": config.tactical_nodes,
        "cases": results,
        "incumbentPassed": sum(row["incumbent"]["passed"] for row in results),
        "candidatePassed": sum(row["candidate"]["passed"] for row in results),
        "regressions": regressions,
    }
    relative = validate_relative_path(output)
    ensure_contained_directory(root, str(Path(relative).parent))
    write_json_new(contained_path(root, relative), payload)
    return {
        "schema": "open_shogi_tactical_gate_command/v1",
        "report": artifact_ref(root, relative).as_dict(),
        "regressions": regressions,
    }


def verify_gate(
    root: Path,
    config: GateConfig,
    *,
    engine: str,
    arena_report: str,
    tactical_report: str,
    output: str,
) -> dict[str, Any]:
    commit = require_clean_head(root)
    engine_ref = _checked_ref(root, engine, maximum_bytes=256 * 1024 * 1024)
    model_ref = _model_ref(root, config)
    arena, arena_ref = load_json_and_ref(
        root, validate_relative_path(arena_report), maximum_bytes=16 * 1024 * 1024
    )
    tactical, tactical_ref = load_json_and_ref(
        root, validate_relative_path(tactical_report), maximum_bytes=4 * 1024 * 1024
    )
    _validate_tactical_report(tactical, config, commit, engine_ref, model_ref)
    counts, timing = _validate_arena_report(arena, config, commit, model_ref)
    score_rate = (counts["wins"] + counts["draws"] / 2) / config.games
    lower, upper = _wilson(score_rate, config.games)
    decisive = counts["wins"] + counts["losses"]
    checks = {
        "minimumDecisiveGames": decisive >= config.minimum_decisive_games,
        "minimumScoreRate": score_rate >= config.minimum_score_rate,
        "minimumWilsonLower": lower >= config.minimum_wilson_lower,
        "illegalMoves": counts["illegalMoves"] <= config.maximum_illegal_moves,
        "crashes": config.maximum_crashes == 0,
        "tacticalRegressions": tactical["regressions"] <= config.maximum_tactical_regressions,
    }
    promoted = all(checks.values())
    decision = {
        "schema": DECISION_SCHEMA,
        "configSha256": config.sha256,
        "gitCommit": commit,
        "candidate": {
            "id": config.candidate_id,
            "kind": config.candidate_kind,
            "model": model_ref.as_dict(),
        },
        "incumbent": {"id": config.incumbent_id, "kind": "handcrafted-experimental"},
        "arenaReport": arena_ref.as_dict(),
        "tacticalReport": tactical_ref.as_dict(),
        "engine": engine_ref.as_dict(),
        "pairedEqualWallClock": True,
        "games": config.games,
        "wins": counts["wins"],
        "losses": counts["losses"],
        "draws": counts["draws"],
        "decisiveGames": decisive,
        "scoreRate": score_rate,
        "confidenceInterval": {
            "method": "wilson-score-with-half-draws",
            "level": 0.95,
            "lower": lower,
            "upper": upper,
        },
        "illegalMoves": counts["illegalMoves"],
        "crashes": 0,
        "crashEvidence": "complete arena report with every predeclared game",
        "tacticalRegressions": tactical["regressions"],
        "timing": timing,
        "checks": checks,
        "promoted": promoted,
        "overallChampion": config.candidate_id if promoted else config.incumbent_id,
    }
    relative = validate_relative_path(output)
    ensure_contained_directory(root, str(Path(relative).parent))
    write_json_new(contained_path(root, relative), decision)
    return {
        "schema": "open_shogi_overall_champion_verify/v1",
        "decision": artifact_ref(root, relative).as_dict(),
        "promoted": promoted,
        "overallChampion": decision["overallChampion"],
    }


def _validate_arena_report(
    report: object, config: GateConfig, commit: str, model: ArtifactRef
) -> tuple[dict[str, int], dict[str, Any]]:
    if not isinstance(report, dict) or report.get("schema") != "phase2_arena_report/v2":
        raise ValueError("arena report schema is invalid")
    run = report.get("run")
    metrics = report.get("metrics")
    games = report.get("games")
    if not isinstance(run, dict) or not isinstance(metrics, dict) or not isinstance(games, list):
        raise ValueError("arena report structure is invalid")
    budget = run.get("budget")
    if (
        run.get("gameLimit") != config.games
        or run.get("gitCommit") != commit
        or run.get("maxPlies") != config.max_plies
        or run.get("seed") != config.seed
        or budget != {"kind": "movetime_ms", "value": config.movetime_ms}
        or run.get("opening", {}).get("enabled") is not False
    ):
        raise ValueError("arena report differs from the predeclared equal-wall-clock config")
    player_a = run.get("playerA")
    player_b = run.get("playerB")
    if (
        not isinstance(player_a, dict)
        or not isinstance(player_b, dict)
        or player_a.get("evaluatorKind") != config.candidate_kind
        or player_a.get("modelArtifactSha256") != model.sha256
        or player_a.get("modelArtifactSize") != model.size
        or player_b.get("evaluatorKind") != "handcrafted-experimental"
        or player_b.get("modelArtifactSha256") is not None
    ):
        raise ValueError("arena player identities differ from the gate")
    if (
        len(games) != config.games
        or metrics.get("games") != config.games
        or metrics.get("finishedGames") != config.games
    ):
        raise ValueError("arena did not finish every predeclared game")
    label_a = player_a.get("label")
    label_b = player_b.get("label")
    wins = losses = draws = 0
    for index, game in enumerate(games):
        if not isinstance(game, dict) or game.get("id") != index:
            raise ValueError("arena game order is invalid")
        expected_black, expected_white = (
            (label_a, label_b) if index % 2 == 0 else (label_b, label_a)
        )
        if game.get("black") != expected_black or game.get("white") != expected_white:
            raise ValueError("arena is not a paired color-reversal schedule")
        result = game.get("result")
        if result == "draw":
            draws += 1
        elif result == "black_win":
            wins += game["black"] == label_a
            losses += game["black"] == label_b
        elif result == "white_win":
            wins += game["white"] == label_a
            losses += game["white"] == label_b
        else:
            raise ValueError("arena contains an unfinished or unsupported result")
    illegal = metrics.get("illegalMoves")
    if isinstance(illegal, bool) or not isinstance(illegal, int) or illegal < 0:
        raise ValueError("arena illegal move count is invalid")
    if (metrics.get("playerAWins"), metrics.get("playerBWins"), metrics.get("draws")) != (
        wins,
        losses,
        draws,
    ):
        raise ValueError("arena aggregate results differ from game rows")
    timing = {
        "movetimeMs": config.movetime_ms,
        "candidateSearchElapsedMs": metrics.get("playerASearchElapsedMs"),
        "candidateSearches": metrics.get("playerASearches"),
        "incumbentSearchElapsedMs": metrics.get("playerBSearchElapsedMs"),
        "incumbentSearches": metrics.get("playerBSearches"),
    }
    return {"wins": wins, "losses": losses, "draws": draws, "illegalMoves": illegal}, timing


def _validate_tactical_report(
    report: object, config: GateConfig, commit: str, engine: ArtifactRef, model: ArtifactRef
) -> None:
    if (
        not isinstance(report, dict)
        or report.get("schema") != TACTICAL_REPORT_SCHEMA
        or report.get("configSha256") != config.sha256
        or report.get("gitCommit") != commit
        or report.get("engine") != engine.as_dict()
        or report.get("candidate", {}).get("model") != model.as_dict()
        or report.get("candidate", {}).get("kind") != config.candidate_kind
        or report.get("incumbent", {}).get("id") != config.incumbent_id
        or report.get("nodesPerCase") != config.tactical_nodes
        or isinstance(report.get("regressions"), bool)
        or not isinstance(report.get("regressions"), int)
    ):
        raise ValueError("tactical report identity is invalid")


def _load_suite(root: Path, config: GateConfig) -> tuple[ArtifactRef, list[dict[str, Any]]]:
    reference = artifact_ref(root, config.suite_path, maximum_bytes=1024 * 1024)
    if reference.sha256 != config.suite_sha256:
        raise ValueError("tactical suite SHA-256 differs from the predeclared config")
    value, _ = load_json_and_ref(root, config.suite_path, maximum_bytes=1024 * 1024)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "cases"}
        or value.get("schema") != SUITE_SCHEMA
    ):
        raise ValueError("tactical suite schema is invalid")
    cases = value.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 256:
        raise ValueError("tactical suite case count is invalid")
    validated = []
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"id", "sfen", "expectedMoves"}:
            raise ValueError(f"tactical case {index} schema is invalid")
        identity = _identifier(case["id"], f"tactical case {index}.id")
        sfen = _text(case["sfen"], f"tactical case {index}.sfen", maximum=4096)
        moves = case["expectedMoves"]
        if (
            identity in seen
            or not isinstance(moves, list)
            or not moves
            or any(not isinstance(move, str) or not move for move in moves)
        ):
            raise ValueError(f"tactical case {index} identity or moves are invalid")
        seen.add(identity)
        validated.append({"id": identity, "sfen": sfen, "expectedMoves": moves})
    return reference, validated


def _usi_search(
    root: Path, engine: Path, sfen: str, config: GateConfig, *, model: ArtifactRef | None
) -> dict[str, Any]:
    process = subprocess.Popen(
        [str(engine), "usi"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=root,
    )
    if process.stdin is None or process.stdout is None:
        raise ValueError("cannot open USI subprocess pipes")
    client = _LineClient(process, config.tactical_timeout_seconds)
    try:
        client.send("usi")
        client.until("usiok")
        client.send(f"setoption name USI_Hash value {config.hash_mib}")
        client.send(f"setoption name MaxDepth value {config.depth}")
        if model is None:
            client.send("setoption name ModelKind value overall-champion")
        else:
            semantics = {
                "neural": "pure-value",
                "residual": "residual",
                "composite": "composite-50-50",
            }[config.candidate_kind]
            client.send("setoption name ModelKind value neural-float")
            client.send(f"setoption name ModelSemantics value {semantics}")
            client.send(f"setoption name ModelPath value {model.path}")
        client.send("isready")
        client.until("readyok")
        client.send(f"position sfen {sfen}")
        client.send(f"go nodes {config.tactical_nodes}")
        bestmove, info = client.bestmove()
        client.send("quit")
        return {"bestmove": bestmove, "lastInfo": info}
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


class _LineClient:
    def __init__(self, process: subprocess.Popen[bytes], timeout_seconds: int) -> None:
        self.process = process
        self.timeout_seconds = timeout_seconds
        self.buffer = b""

    def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write((line + "\n").encode())
        self.process.stdin.flush()

    def until(self, expected: str) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            line = self._line(deadline)
            if line == expected:
                return line

    def bestmove(self) -> tuple[str, str | None]:
        deadline = time.monotonic() + self.timeout_seconds
        info = None
        while True:
            line = self._line(deadline)
            if line.startswith("info "):
                info = line
            if line.startswith("bestmove "):
                move = line.split()[1]
                if move == "resign":
                    raise ValueError("tactical search resigned")
                return move, info

    def _line(self, deadline: float) -> str:
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("USI tactical search timed out")
            assert self.process.stdout is not None
            ready, _, _ = select.select([self.process.stdout.fileno()], [], [], remaining)
            if not ready:
                raise ValueError("USI tactical search timed out")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise ValueError("USI process ended before its response")
            self.buffer += chunk
            if len(self.buffer) > 1024 * 1024:
                raise ValueError("USI response exceeds its byte bound")
        raw, self.buffer = self.buffer.split(b"\n", 1)
        if len(raw) > 64 * 1024:
            raise ValueError("USI line exceeds its byte bound")
        return raw.rstrip(b"\r").decode()


def _wilson(rate: float, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _validate_bound_artifacts(root: Path, config: GateConfig) -> None:
    _model_ref(root, config)
    suite = artifact_ref(root, config.suite_path, maximum_bytes=1024 * 1024)
    if suite.sha256 != config.suite_sha256:
        raise ValueError("tactical suite identity differs from gate config")


def _model_ref(root: Path, config: GateConfig) -> ArtifactRef:
    reference = artifact_ref(root, config.model_path, maximum_bytes=256 * 1024 * 1024)
    if reference.sha256 != config.model_sha256 or reference.size != config.model_size:
        raise ValueError("candidate model identity differs from gate config")
    return reference


def _checked_ref(root: Path, relative: str, *, maximum_bytes: int) -> ArtifactRef:
    return artifact_ref(root, validate_relative_path(relative), maximum_bytes=maximum_bytes)


def _closed(value: object, context: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{context} keys mismatch: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _text(value: object, context: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\0" in value:
        raise ValueError(f"{context} is invalid")
    return value


def _identifier(value: object, context: str) -> str:
    result = _text(value, context, 96)
    validate_identifier(result, context)
    return result


def _integer(value: object, context: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{context} must be {minimum}..={maximum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return float(value)


def _rate(value: object, context: str) -> float:
    result = _number(value, context)
    if not 0 <= result <= 1:
        raise ValueError(f"{context} must be in [0, 1]")
    return result


def _sha(value: object, context: str) -> str:
    result = _text(value, context, 64)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenShogiAI overall-champion gate")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--engine", default="target/release/open-shogi-cli")
    preflight_parser.add_argument("--output-dir", default="local/runs/overall-gate/arena")
    tactical_parser = subparsers.add_parser("tactical-run")
    tactical_parser.add_argument("--engine", default="target/release/open-shogi-cli")
    tactical_parser.add_argument("--output", default="local/runs/overall-gate/tactical-report.json")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--engine", default="target/release/open-shogi-cli")
    verify_parser.add_argument(
        "--arena-report", default="local/runs/overall-gate/arena/arena-report.json"
    )
    verify_parser.add_argument(
        "--tactical-report", default="local/runs/overall-gate/tactical-report.json"
    )
    verify_parser.add_argument("--output", default="local/runs/overall-gate/decision.json")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
