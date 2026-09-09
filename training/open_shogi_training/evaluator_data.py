"""Resumable, game-partitioned teacher trajectories for the current evaluator run.

The teacher is offline only. Every move is replayed by our native legal engine.
Raw observations are retained; mate scores never become scalar centipawns.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import selectors
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from .labeling.config import load_teacher_config
from .labeling.usi import USIEngine
from .phase10r_model import parse_sfen
from .phase10v_data import position_hash
from .phase10v_model import sparse_features

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
SCHEMA = "open_shogiai_evaluator_data/v1"
MAX_FEATURES = 48


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def partition(game: int, seed: int) -> str:
    # Assigned from the original trajectory identity, before labels/features/normalization.
    bucket = int(hashlib.sha256(f"trajectory:{seed}:{game}".encode()).hexdigest()[:8], 16) % 20
    return "validation" if bucket == 0 else "development_test" if bucket == 1 else "train"


def symmetry_keys(sfen: str) -> set[str]:
    """Canonical SFEN hashes including left/right and color/180-degree symmetries."""
    p = parse_sfen(sfen)
    names = ("P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R")
    keys = set()
    for rotate in (False, True):
        for mirror in (False, True):
            board = [None] * 81
            for piece in p.board:
                if piece is None:
                    continue
                square = 80 - piece.square if rotate else piece.square
                rank, file = divmod(square, 9)
                square = rank * 9 + (8 - file if mirror else file)
                name = names[piece.kind]
                board[square] = name.lower() if piece.side ^ rotate else name
            ranks = []
            for rank in range(9):
                text, empty = "", 0
                for piece in board[rank * 9 : rank * 9 + 9]:
                    if piece is None:
                        empty += 1
                    else:
                        text += (str(empty) if empty else "") + piece
                        empty = 0
                ranks.append(text + (str(empty) if empty else ""))
            hand = ""
            for side in (0, 1):
                for name in "RBGSNLP":
                    count = p.hands[side ^ rotate]["PLNSGBR".index(name)]
                    if count:
                        hand += (str(count) if count > 1 else "") + (name.lower() if side else name)
            state = f"{'/'.join(ranks)} {'w' if p.side_to_move ^ rotate else 'b'} {hand or '-'}"
            keys.add(hashlib.sha256(state.encode()).hexdigest())
    return keys


class Replay:
    """One owned native process, bounded requests and no silent rules fallback."""

    def __init__(self, root: Path, config: dict):
        self.process = subprocess.Popen(
            [
                str(root / config["replay_path"]),
                str(root / config["leaf_path"]),
                config["leaf_sha256"],
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def ask(self, **request: object) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(30):
            raise RuntimeError("native replay made no progress for 30 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native replay exited: " + self.process.stderr.read()[-2000:])
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=2)
        self.selector.close()


def teacher_config(root: Path, config: dict):
    original = load_teacher_config(root / config["teacher_config_path"])
    return replace(
        original,
        binary_sha256=config["teacher_binary_sha256"],
        threads=1,
        hash_mb=64,
        multipv=3,
        nodes=config["teacher_nodes"],
        options=(
            ("Book_Enable", False),
            ("Eval_Dir", "eval/20190617"),
            ("Eval_Hash", 16),
            ("USI_Ponder", False),
        ),
    )


def _generate_group(root_name: str, output_name: str, config: dict, games: list[int]) -> list[dict]:
    root, output = Path(root_name), Path(output_name)
    reports = []
    replay = Replay(root, config)
    try:
        with USIEngine(teacher_config(root, config), root, isolate_process_group=False) as teacher:
            for game in games:
                target = output / "games" / f"{game:06d}.json.gz"
                receipt = target.with_suffix(".receipt.json")
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if digest(target) != saved["sha256"]:
                        raise ValueError("completed trajectory corrupted")
                    reports.append(saved)
                    continue
                rng = random.Random(config["seed"] + game)
                began = time.monotonic()
                state = replay.ask(reset=START, successors=True)
                records, moves = [], []
                for ply in range(config["max_plies"]):
                    if (output / "STOP").exists():
                        raise InterruptedError("requested stop; completed trajectories retained")
                    if state["terminal"] != "None":
                        break
                    result = teacher.analyze_with_retry(state["sfen"])
                    legal = {c["move"]: c for c in state["successors"]}
                    if any(c.pv[0] not in legal for c in result.candidates):
                        raise ValueError("teacher proposed illegal move")
                    if ply % config["sample_stride"] == 0:
                        candidates = [
                            dict(
                                c.as_dict(),
                                child_sfen=legal[c.pv[0]]["sfen"],
                                child_terminal=legal[c.pv[0]]["terminal"],
                            )
                            for c in result.candidates
                        ]
                        record = {
                            "sfen": state["sfen"],
                            "ply": ply,
                            "candidates": candidates,
                            "teacher_elapsed_ms": result.elapsed_ms,
                            "deviation": None,
                        }
                        # Offline coverage of the current weak evaluator's actual preferred child.
                        weak = min(legal.values(), key=lambda c: (c["child_cp"], c["move"]))
                        if (
                            weak["terminal"] == "None"
                            and ply % config["deviation_stride"] == 0
                            and weak["move"] not in {c.pv[0] for c in result.candidates}
                        ):
                            child_result = teacher.analyze_with_retry(weak["sfen"])
                            record["deviation"] = {
                                "move": weak["move"],
                                "sfen": weak["sfen"],
                                "score": child_result.primary.score.as_dict(),
                                "candidates": [c.as_dict() for c in child_result.candidates],
                                "teacher_elapsed_ms": child_result.elapsed_ms,
                            }
                        records.append(record)
                    # Data generation only: varied strong continuations, never a runtime preset.
                    weights = [0.6, 0.3, 0.1] if ply < 48 else [0.92, 0.06, 0.02]
                    choice = rng.choices(result.candidates, weights[: len(result.candidates)])[
                        0
                    ].pv[0]
                    moves.append(choice)
                    state = replay.ask(movement=choice, successors=True)
                raw = {
                    "schema": SCHEMA,
                    "game": game,
                    "seed": config["seed"] + game,
                    "split": partition(game, config["seed"]),
                    "moves": moves,
                    "end": state["terminal"],
                    "records": records,
                }
                # An interrupted old temporary output is replaced, never a completed receipt.
                atomic(target, gzip.compress(encoded(raw), mtime=0))
                saved = {
                    "game": game,
                    "sha256": digest(target),
                    "rows": len(records),
                    "plies": len(moves),
                    "elapsed_s": time.monotonic() - began,
                    "split": raw["split"],
                }
                atomic(receipt, encoded(saved))
                reports.append(saved)
                atomic(output / f"worker-{games[0]}.json", encoded(saved))
    finally:
        replay.close()
    return reports


def generate(root: Path, output: Path, config: dict) -> dict:
    for path_key, sha_key in [
        ("replay_path", "replay_sha256"),
        ("leaf_path", "leaf_sha256"),
        ("teacher_config_path", "teacher_config_sha256"),
    ]:
        if digest(root / config[path_key]) != config[sha_key]:
            raise ValueError(f"identity changed: {path_key}")
    output.mkdir(parents=True, exist_ok=True)
    identity = output / "generation.json"
    if identity.exists() and identity.read_bytes() != encoded(config):
        raise ValueError("generation config changed; new run required")
    atomic(identity, encoded(config))
    start = config.get("first_game", 0)
    expected = {f"{game:06d}.json.gz" for game in range(start, start + config["games"])}
    if any(p.name not in expected for p in (output / "games").glob("*.json.gz")):
        raise ValueError("unexpected trajectory outside the sealed generation range")
    groups = [
        list(range(start + i, start + config["games"], config["workers"]))
        for i in range(config["workers"])
    ]
    all_reports = []
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [
            pool.submit(_generate_group, str(root), str(output), config, g) for g in groups if g
        ]
        for future in as_completed(futures):
            try:
                all_reports.extend(future.result())
            except BaseException:
                (output / "STOP").touch()
                raise
    report = {
        "schema": SCHEMA,
        "games": len(all_reports),
        "sampled_roots": sum(r["rows"] for r in all_reports),
        "plies": sum(r["plies"] for r in all_reports),
        "config_sha256": digest(identity),
    }
    if {r["game"] for r in all_reports} != set(range(start, start + config["games"])):
        raise ValueError("generation did not complete its exact trajectory set")
    atomic(output / "generation-complete.json", encoded(report))
    return report


def _observations(game: dict):
    for index, row in enumerate(game["records"]):
        primary = row["candidates"][0]
        yield row["sfen"], primary["score"], "root", row["ply"], index, None
        for c in row["candidates"]:
            if c["child_terminal"] != "None":
                continue
            yield (
                c["child_sfen"],
                {"kind": c["score"]["kind"], "value": -c["score"]["value"]},
                "candidate",
                row["ply"] + 1,
                index,
                c["pv"][0],
            )
        if row["deviation"]:
            d = row["deviation"]
            yield d["sfen"], d["score"], "deviation", row["ply"] + 1, index, d["move"]


def prepare(root: Path, output: Path, config: dict, excluded_sfens: list[str]) -> dict:
    """Two passes remove ALL cross-split duplicate/symmetry states, then encode unique rows."""
    if not (output / "generation-complete.json").exists():
        raise ValueError("generation has not completed")
    if (output / "generation.json").read_bytes() != encoded(config):
        raise ValueError("preparation generation config changed")
    start = config.get("first_game", 0)
    expected_games = set(range(start, start + config["games"]))
    actual_paths = {p.name for p in (output / "games").glob("*.json.gz")}
    if actual_paths != {f"{game:06d}.json.gz" for game in expected_games}:
        raise ValueError("source trajectory set incomplete or contains unexpected games")
    guard_path = root / config["split_guard_path"]
    if digest(guard_path) != config["split_guard_sha256"]:
        raise ValueError("split guard changed")
    exclusions_sha256 = hashlib.sha256(encoded(sorted(excluded_sfens))).hexdigest()
    development_keys = set()
    if config.get("development_exclusions_path"):
        path = root / config["development_exclusions_path"]
        if digest(path) != config["development_exclusions_sha256"]:
            raise ValueError("development exclusions changed")
        development_keys = set(json.loads(path.read_text())["symmetry_keys"])
    dataset = output / "dataset"
    if (dataset / "manifest.json").exists():
        report = json.loads((dataset / "manifest.json").read_text())
        if (
            report.get("split_guard_sha256") != digest(guard_path)
            or report.get("generation_sha256") != digest(output / "generation.json")
            or report.get("exclusions_sha256") != exclusions_sha256
        ):
            raise ValueError("prepared dataset configuration/exclusions changed")
        for ref in report["artifacts"]:
            if digest(dataset / ref["path"]) != ref["sha256"]:
                raise ValueError("prepared dataset corrupted")
        for ref in report["source_games"]:
            if digest(output / "games" / ref["path"]) != ref["sha256"]:
                raise ValueError("prepared source trajectory corrupted")
        return report
    guard = json.loads(guard_path.read_text())
    excluded = {
        r["position_sha256"]
        for rows in guard.values()
        if isinstance(rows, list)
        for r in rows
        if isinstance(r, dict) and "position_sha256" in r
    }
    for sfen in excluded_sfens:
        excluded.update(symmetry_keys(sfen))
    excluded.update(development_keys)
    owners, conflicts, raw_count, raw_mates = {}, set(), 0, 0
    games = []
    for path in sorted((output / "games").glob("*.json.gz")):
        receipt = json.loads(path.with_suffix(".receipt.json").read_text())
        if digest(path) != receipt["sha256"]:
            raise ValueError("raw game identity mismatch")
        game = json.loads(gzip.decompress(path.read_bytes()))
        if (
            game["game"] not in expected_games
            or game["seed"] != config["seed"] + game["game"]
            or game["split"] != partition(game["game"], config["seed"])
            or path.name != f"{game['game']:06d}.json.gz"
        ):
            raise ValueError("source trajectory split changed")
        games.append(
            {
                "path": path.name,
                "sha256": receipt["sha256"],
                "game": game["game"],
                "split": game["split"],
            }
        )
        for sfen, score, *_ in _observations(game):
            raw_count += 1
            if score["kind"] != "cp":
                raw_mates += 1
                continue
            keys = symmetry_keys(sfen)
            key = min(keys)
            if keys & excluded:
                conflicts.add(key)
            if key in owners and owners[key] != game["split"]:
                conflicts.add(key)
            owners[key] = game["split"]
    if {g["game"] for g in games} != expected_games:
        raise ValueError("source trajectory set incomplete")
    dataset.mkdir(exist_ok=True)
    counts = {
        s: sum(owner == s and key not in conflicts for key, owner in owners.items())
        for s in ("train", "validation", "development_test")
    }
    arrays, streams = {}, {}
    with ExitStack() as stack:
        for split, count in counts.items():
            arrays[split] = {
                "features": np.lib.format.open_memmap(
                    dataset / f"{split}-features.npy",
                    mode="w+",
                    dtype="uint16",
                    shape=(count, 2, MAX_FEATURES),
                ),
                "lengths": np.lib.format.open_memmap(
                    dataset / f"{split}-lengths.npy", mode="w+", dtype="uint8", shape=(count, 2)
                ),
                "targets": np.lib.format.open_memmap(
                    dataset / f"{split}-targets.npy", mode="w+", dtype="float32", shape=(count,)
                ),
            }
            streams[split] = stack.enter_context(
                gzip.open(dataset / f"{split}-rows.jsonl.gz", "wb")
            )
        used, offsets, distributions = set(), dict.fromkeys(counts, 0), {s: {} for s in counts}
        for ref in games:
            game = json.loads(gzip.decompress((output / "games" / ref["path"]).read_bytes()))
            split = game["split"]
            for sfen, score, kind, ply, index, move in _observations(game):
                if score["kind"] != "cp":
                    continue
                key = min(symmetry_keys(sfen))
                if key in used or key in conflicts:
                    continue
                value = score["value"]
                if type(value) is not int or abs(value) > 28999:
                    raise ValueError("teacher cp outside observed namespace")
                used.add(key)
                black, white, stm = sparse_features(sfen)
                own, other = (white, black) if stm else (black, white)
                if max(len(own), len(other)) > MAX_FEATURES:
                    raise ValueError("sparse feature capacity exceeded")
                n = offsets[split]
                for side, features in enumerate((own, other)):
                    arrays[split]["features"][n, side, : len(features)] = features
                    arrays[split]["lengths"][n, side] = len(features)
                arrays[split]["targets"][n] = np.clip(value, -20000, 20000)
                row = {
                    "sfen": sfen,
                    "score": score,
                    "score_perspective": "side_to_move",
                    "kind": kind,
                    "game": game["game"],
                    "raw_record": index,
                    "move": move,
                    "ply": ply,
                    "position_sha256": position_hash(sfen),
                    "symmetry_key": key,
                }
                streams[split].write(encoded(row) + b"\n")
                offsets[split] += 1
                stage = "opening" if ply < 40 else "middle" if ply < 110 else "end"
                for label in (stage, kind, "saturated" if abs(value) > 20000 else "unsaturated"):
                    distributions[split][label] = distributions[split].get(label, 0) + 1
    for group in arrays.values():
        for array in group.values():
            array.flush()
    if offsets != counts:
        raise ValueError("unique dataset count mismatch")
    label_summary = {}
    for split, group in arrays.items():
        targets = group["targets"]
        label_summary[split] = {
            "count": len(targets),
            "distinct_cp": len(np.unique(targets)),
            "minimum": float(targets.min()) if len(targets) else None,
            "maximum": float(targets.max()) if len(targets) else None,
            "standard_deviation": float(targets.std()) if len(targets) else None,
            "zero_labels": int((targets == 0).sum()),
        }
    artifacts = [
        {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(dataset.iterdir())
        if p.is_file()
    ]
    report = {
        "schema": SCHEMA,
        "unique_positions": counts,
        "raw_observations": raw_count,
        "mate_observations_masked": raw_mates,
        "excluded_conflict_keys": len(conflicts),
        "duplicate_or_excluded_observations": raw_count - raw_mates - sum(counts.values()),
        "source_games": games,
        "distributions": distributions,
        "label_summary": label_summary,
        "artifacts": artifacts,
        "split_guard_sha256": digest(guard_path),
        "generation_sha256": digest(output / "generation.json"),
        "exclusions_sha256": exclusions_sha256,
        "sealed_holdout_opened": False,
        "excluded_preflight_symmetry_keys": len(development_keys),
        "symmetry_cross_split_overlap": 0,
        "independence_limit": (
            "Own trajectories share the initial position and teacher; "
            "no claim of independent human games."
        ),
    }
    atomic(dataset / "manifest.json", encoded(report))
    return report
