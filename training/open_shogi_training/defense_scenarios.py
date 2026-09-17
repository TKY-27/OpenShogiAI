"""Offline scenario lineage and verified r3 replay for the defense campaign.

Nothing in this module is imported by the playing engine. Prefixes create training
situations; labels, including quiet replies and compensating attacks, come from USI.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import selectors
import shutil
import subprocess
from pathlib import Path

import numpy as np

GROUPS = {"general": 0, "opening": 1, "defense": 2, "attack_end": 3}
SPLITS = ("train", "validation", "development_test")


def _remove_incomplete_dataset(root: Path, output: Path) -> None:
    """Rebuild only this run's incomplete staging, refusing links/mounts/open files."""
    from .evaluator_data import atomic, encoded

    target = output / "dataset-building"
    if not target.exists() and not target.is_symlink():
        return
    if (
        output.is_symlink()
        or target.is_symlink()
        or not target.is_dir()
        or not output.resolve().is_relative_to(root.resolve() / "local")
        or target.resolve().parent != output.resolve()
    ):
        raise ValueError("unsafe incomplete campaign dataset")
    device = output.stat().st_dev
    paths = [target, *target.rglob("*")]
    for path in paths:
        if path.is_symlink() or path.stat().st_dev != device or os.path.ismount(path):
            raise ValueError("link or mount in incomplete campaign dataset")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError("shared hardlink in incomplete campaign dataset")
    opened = subprocess.run(
        ["lsof", "+D", str(target)], capture_output=True, text=True, check=False
    )
    if opened.returncode != 1 or opened.stdout.strip() or opened.stderr.strip():
        raise ValueError("cannot establish closed incomplete campaign dataset")
    atomic(
        output / "dataset-staging-cleanup.json",
        encoded(
            {
                "path": str(target.relative_to(root)),
                "reason": "incomplete owned dataset staging; regenerated from verified inputs",
                "bytes": sum(p.stat().st_size for p in paths if p.is_file()),
                "symlinks_mounts_open_files": False,
            }
        ),
    )
    shutil.rmtree(target)


def default_families() -> list[dict]:
    """Separate source families; all stochastic/color variants keep their split.

    Names describe setup intentions rather than claiming a canonical book line.
    All prefixes are checked with native legal replay before generation.
    """
    lines = [
        ("general", "double_rook_pawns", "2g2f 8c8d 2f2e 8d8e 7g7f 3c3d"),
        ("general", "bishop_diagonals", "7g7f 3c3d 2g2f 4c4d 6i7h 8c8d"),
        ("general", "central_space", "5g5f 5c5d 7g7f 3c3d 6i5h 4a5b"),
        (
            "opening",
            "rook_silver_coordination",
            "2g2f 3c3d 2f2e 8c8d 3i3h 4a3b 3h2g 8d8e 2g2f 5a4b",
        ),
        ("opening", "left_structure", "7g7f 8c8d 6g6f 3c3d 7i7h 4a3b 7h6g 5a4b"),
        ("opening", "central_rook", "5g5f 3c3d 7g7f 8c8d 2h5h 4a3b 5f5e 5a4b"),
        (
            "defense",
            "rook_edge_silver",
            "2g2f 8c8d 2f2e 8d8e 3i3h 4a3b 3h2g 3c3d 2g2f 7a6b 2f1e 5a4b",
        ),
        (
            "defense",
            "fourth_file_rook_pressure",
            "7g7f 8c8d 6g6f 8d8e 7i7h 3c3d 7h6g 7a6b 2h6h 6c6d 3i4h 6b6c",
        ),
        (
            "defense",
            "opposed_silver_pressure",
            "7g7f 8c8d 6i7h 8d8e 2g2f 7a7b 2f2e 7b8c 3i3h 8c8d 3h2g 8d9e",
        ),
        ("attack_end", "bishop_exchange", "7g7f 3c3d 8h2b+ 3a2b 2g2f 8c8d 2f2e 8d8e"),
        ("attack_end", "rook_pawn_exchange", "2g2f 3c3d 2f2e 4c4d 2e2d 2c2d 2h2d 8c8d"),
        ("attack_end", "opposite_rook_exchange", "7g7f 8c8d 6i7h 8d8e 6g6f 8e8f 8g8f 8b8f"),
    ]
    return [
        {"id": name, "group": group, "split": SPLITS[i % 3], "moves": moves.split()}
        for i, (group, name, moves) in enumerate(lines)
    ]


def validate_campaign(config: dict) -> dict:
    campaign = config["defense_campaign"]
    families = campaign["families"]
    if not families or len({f["id"] for f in families}) != len(families):
        raise ValueError("defense families must have unique nonempty identities")
    for family in families:
        if (
            not isinstance(family["id"], str)
            or not family["id"]
            or family["group"] not in GROUPS
            or family["split"] not in SPLITS
            or not isinstance(family["moves"], list)
            or any(not isinstance(m, str) for m in family["moves"])
        ):
            raise ValueError("invalid defense family")
    if len({tuple(f["moves"]) for f in families}) != len(families):
        raise ValueError("identical prefixes must share one family")
    for key in (
        "probe_nodes",
        "probe_stride",
        "relabel_nodes",
        "relabel_depth",
        "variants_per_family",
    ):
        if type(campaign[key]) is not int or campaign[key] <= 0:
            raise ValueError(f"invalid defense {key}")
    if type(config.get("teacher_depth")) is not int or config["teacher_depth"] <= 0:
        raise ValueError("defense campaign requires completed fixed-depth teacher labels")
    if config["games"] > len(families) * campaign["variants_per_family"]:
        raise ValueError("trajectory count exceeds frozen family variants")
    for source in ("generated", "r3_replay"):
        cap = campaign["source_row_caps"][source]
        if type(cap) is not int or cap < 1:
            raise ValueError("source caps must be positive train row ceilings")
    if (
        type(campaign["maximum_unlabeled_focus"]) is not int
        or campaign["maximum_unlabeled_focus"] < 0
        or not 0 < campaign["minimum_focus_completion_rate"] <= 1
    ):
        raise ValueError("invalid optional focus completeness limits")
    return campaign


def assignment(config: dict, game: int) -> tuple[dict, int]:
    families = config["defense_campaign"]["families"]
    index = game - config.get("first_game", 0)
    return families[index % len(families)], index // len(families)


def rotate_sfen(sfen: str) -> str:
    """Rotate board 180 degrees and exchange colors, including hands and turn."""
    board, turn, _hands, ply = sfen.split()
    ranks = []
    for rank in reversed(board.split("/")):
        pieces = []
        i = 0
        while i < len(rank):
            if rank[i] == "+":
                pieces.append("+" + rank[i + 1].swapcase())
                i += 2
            else:
                pieces.append(rank[i].swapcase())
                i += 1
        ranks.append("".join(reversed(pieces)))
    from .phase10r_model import parse_sfen

    parsed = parse_sfen(sfen)
    rotated_hands = ""
    for side in (0, 1):
        for piece in "RBGSNLP":
            count = parsed.hands[1 - side]["PLNSGBR".index(piece)]
            if count:
                rotated_hands += (str(count) if count > 1 else "") + (
                    piece.lower() if side else piece
                )
    return f"{'/'.join(ranks)} {'w' if turn == 'b' else 'b'} {rotated_hands or '-'} {ply}"


def rotate_move(move: str) -> str:
    def square(value: str) -> str:
        return str(10 - int(value[0])) + chr(ord("a") + ord("i") - ord(value[1]))

    if "*" in move:
        return move[:2] + square(move[2:4])
    return square(move[:2]) + square(move[2:4]) + move[4:]


def verify_prefixes(root: Path, config: dict) -> dict:
    from .evaluator_data import START, Replay, digest, encoded

    campaign = validate_campaign(config)
    replay = Replay(root, config)
    outcomes = []
    try:
        for family in campaign["families"]:
            for rotated in (False, True):
                initial = rotate_sfen(START) if rotated else START
                state = replay.ask(reset=initial, successors=True)
                for movement in family["moves"]:
                    movement = rotate_move(movement) if rotated else movement
                    if movement not in {child["move"] for child in state["successors"]}:
                        raise ValueError(f"illegal training prefix {family['id']}: {movement}")
                    state = replay.ask(movement=movement, successors=True)
                    if state.get("error"):
                        raise ValueError("native rejected training prefix")
                outcomes.append(
                    {"family": family["id"], "rotated": rotated, "final_sfen": state["sfen"]}
                )
    finally:
        replay.close()
    return {
        "families_sha256": hashlib.sha256(encoded(campaign["families"])).hexdigest(),
        "replay_sha256": digest(root / config["replay_path"]),
        "verified": outcomes,
    }


class R3Probe:
    """Actual frozen r3 search, bounded nodes, controller/book/teacher absent."""

    def __init__(self, root: Path, config: dict):
        from .evaluator_data import digest

        campaign = config["defense_campaign"]
        if digest(root / campaign["probe_path"]) != campaign["probe_sha256"]:
            raise ValueError("r3 probe identity changed")
        self.model_sha = config["leaf_sha256"]
        self.nodes = campaign["probe_nodes"]
        self.process = subprocess.Popen(
            [
                str(root / campaign["probe_path"]),
                str(root / config["leaf_path"]),
                self.model_sha,
                "probe",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def search(self, initial: str, moves: list[str], expected: str) -> dict:
        from .evaluator_arena import _validate_response

        self.process.stdin.write(
            json.dumps(
                {
                    "sfen": initial,
                    "moves": moves,
                    "depth": 64,
                    "nodes": self.nodes,
                    "control": False,
                }
            )
            + "\n"
        )
        self.process.stdin.flush()
        if not self.selector.select(30):
            raise RuntimeError("r3 diagnostic probe stalled")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("r3 diagnostic probe exited")
        result = json.loads(line)
        _validate_response(result, self.model_sha, nodes=self.nodes)
        if result["sfen"].split()[:3] != expected.split()[:3]:
            raise ValueError("r3 probe/replay position mismatch")
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=2)
        self.selector.close()


def row_group(game: dict, ply: int) -> str:
    # Organizational buckets only; these never modify a target or runtime score.
    if ply < len(game["prefix_moves"]):
        return "opening"
    if ply >= 110:
        return "attack_end"
    return game["group"]


def replay_source(
    root: Path, config: dict, blocked: set[str] | None = None
) -> tuple[Path, dict, dict, set[str], list[int]]:
    """Validate old bytes and all old split identities; use only old train rows."""
    from .evaluator_data import digest, symmetry_keys

    campaign = config["defense_campaign"]
    reference = campaign["replay_dataset"]
    dataset = root / reference["path"]
    if not dataset.resolve().is_relative_to(root.resolve()) or any(
        path.is_symlink() for path in (dataset, *dataset.parents)
    ):
        raise ValueError("replay dataset must be an owned repository directory")
    manifest_path = dataset / "manifest.json"
    if digest(manifest_path) != reference["manifest_sha256"]:
        raise ValueError("r3 replay manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for ref in manifest["artifacts"]:
        path = dataset / ref["path"]
        if path.parent != dataset or path.is_symlink() or digest(path) != ref["sha256"]:
            raise ValueError("r3 replay artifact changed")
    for ref in manifest["source_games"]:
        path = dataset.parent / "games" / ref["path"]
        if (
            path.parent != dataset.parent / "games"
            or path.is_symlink()
            or digest(path) != ref["sha256"]
        ):
            raise ValueError("r3 raw source identity changed")
    all_keys, owners, train = set(), {}, []
    for split in SPLITS:
        count = 0
        with gzip.open(dataset / f"{split}-rows.jsonl.gz", "rt") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                keys = symmetry_keys(row["sfen"])
                key = min(keys)
                if row.get("symmetry_key") != key or key in owners:
                    raise ValueError("r3 replay has duplicate or inconsistent split identity")
                owners[key] = split
                all_keys.update(keys)
                if split == "train" and not keys.intersection(blocked or set()):
                    train.append(
                        (hashlib.sha256((str(config["seed"]) + key).encode()).digest(), index)
                    )
                count += 1
        if count != manifest["unique_positions"][split]:
            raise ValueError("r3 replay row count mismatch")
    selected = sorted(
        index for _, index in sorted(train)[: campaign["source_row_caps"]["r3_replay"]]
    )
    provenance = {
        "path": str(dataset.relative_to(root)),
        "manifest_sha256": digest(manifest_path),
        "source_games": manifest["source_games"],
        "artifacts": manifest["artifacts"],
        "old_split_counts": manifest["unique_positions"],
        "selected_train_rows": len(selected),
        "old_validation_or_development_reused": 0,
    }
    return dataset, manifest, provenance, all_keys, selected


def admission_identity(output: Path) -> dict | None:
    """Bind the explicit itemwise revision without changing the original generation seal."""
    from .evaluator_data import digest

    path = output / "admission.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("dataset admission is missing or linked")
    revision = json.loads(path.read_text())
    if (
        revision.get("schema") != "open_shogiai_dataset_admission/v1"
        or revision.get("policy") != "itemwise-focus-v1"
        or revision.get("generation_sha256") != digest(output / "generation.json")
    ):
        raise ValueError("dataset admission identity changed")
    return {"policy": revision["policy"], "sha256": digest(path)}


def validate_manifest_admission(output: Path, manifest: dict) -> None:
    if manifest.get("dataset_admission") != admission_identity(output):
        raise ValueError("prepared dataset admission changed")


def prepare_campaign(root: Path, output: Path, config: dict, excluded_sfens: list[str]) -> dict:
    from .evaluator_data import _prepare, atomic, digest, encoded

    validate_campaign(config)
    quality = focus_gate(output, config)
    admission = admission_identity(output)
    from .evaluator_data import symmetry_keys

    guard_path = root / config["split_guard_path"]
    if digest(guard_path) != config["split_guard_sha256"]:
        raise ValueError("split guard changed")
    blocked = {
        r["position_sha256"]
        for rows in json.loads(guard_path.read_text()).values()
        if isinstance(rows, list)
        for r in rows
        if isinstance(r, dict) and "position_sha256" in r
    }
    for sfen in excluded_sfens:
        blocked.update(symmetry_keys(sfen))
    if config.get("development_exclusions_path"):
        path = root / config["development_exclusions_path"]
        if digest(path) != config["development_exclusions_sha256"]:
            raise ValueError("development exclusions changed")
        blocked.update(json.loads(path.read_text())["symmetry_keys"])
    replay, _, provenance, excluded, selected = replay_source(root, config, blocked)
    dataset = output / "dataset"
    if (dataset / "manifest.json").exists():
        report = json.loads((dataset / "manifest.json").read_text())
        validate_manifest_admission(output, report)
        if report.get("replay_source") != provenance:
            raise ValueError("prepared replay source changed")
        if report.get("generation_sha256") != digest(output / "generation.json"):
            raise ValueError("prepared campaign generation changed")
        if (
            report.get("exclusions_sha256")
            != hashlib.sha256(encoded(sorted(excluded_sfens))).hexdigest()
        ):
            raise ValueError("prepared campaign exclusions changed")
        if report.get("split_guard_sha256") != digest(root / config["split_guard_path"]):
            raise ValueError("prepared campaign split guard changed")
        for ref in report["artifacts"]:
            if digest(dataset / ref["path"]) != ref["sha256"]:
                raise ValueError("prepared dataset corrupted")
        for ref in report["source_games"]:
            if digest(output / "games" / ref["path"]) != ref["sha256"]:
                raise ValueError("prepared source trajectory corrupted")
        return report
    generated = _prepare(
        root,
        output,
        config,
        excluded_sfens,
        extra_excluded_keys=excluded,
        dataset_name="dataset-generated",
    )
    source = output / "dataset-generated"
    building = output / "dataset-building"
    _remove_incomplete_dataset(root, output)
    building.mkdir()
    counts = dict(generated["unique_positions"])
    counts["train"] += len(selected)
    distributions = json.loads(json.dumps(generated["distributions"]))
    n = generated["unique_positions"]["train"]
    for split in SPLITS:
        for kind in ("features", "lengths", "targets", "groups"):
            old = np.load(source / f"{split}-{kind}.npy", mmap_mode="r")
            target = np.lib.format.open_memmap(
                building / f"{split}-{kind}.npy",
                mode="w+",
                dtype=old.dtype,
                shape=(counts[split], *old.shape[1:]),
            )
            target[: len(old)] = old
            if split == "train":
                if kind == "groups":
                    target[n:] = GROUPS["general"]
                else:
                    original = np.load(replay / f"train-{kind}.npy", mmap_mode="r")
                    for offset in range(0, len(selected), 4096):
                        indexes = selected[offset : offset + 4096]
                        target[n + offset : n + offset + len(indexes)] = original[indexes]
            target.flush()
        with gzip.open(building / f"{split}-rows.jsonl.gz", "wb") as destination:
            with gzip.open(source / f"{split}-rows.jsonl.gz", "rb") as stream:
                shutil.copyfileobj(stream, destination)
            if split == "train":
                chosen = set(selected)
                with gzip.open(replay / "train-rows.jsonl.gz", "rt") as stream:
                    for index, line in enumerate(stream):
                        if index in chosen:
                            row = json.loads(line)
                            row.update(
                                group="general",
                                source="r3_replay",
                                family=f"r3:{row['game']}",
                                source_row=index,
                            )
                            destination.write(encoded(row) + b"\n")
    distributions["train"]["general"] = distributions["train"].get("general", 0) + len(selected)
    artifacts = [
        {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(building.iterdir())
    ]
    summary = {}
    for split in SPLITS:
        labels = np.load(building / f"{split}-targets.npy", mmap_mode="r")
        summary[split] = {
            "count": len(labels),
            "distinct_cp": len(np.unique(labels)),
            "minimum": float(labels.min()) if len(labels) else None,
            "maximum": float(labels.max()) if len(labels) else None,
            "standard_deviation": float(labels.std()) if len(labels) else None,
            "zero_labels": int((labels == 0).sum()),
        }
    report = {
        **generated,
        "unique_positions": counts,
        "artifacts": artifacts,
        "label_summary": summary,
        "distributions": distributions,
        "replay_source": provenance,
        "generated_unique_positions": generated["unique_positions"],
        "source_train_rows": {"generated": n, "r3_replay": len(selected)},
        "group_ids": GROUPS,
        "source_row_caps": config["defense_campaign"]["source_row_caps"],
        "all_old_split_symmetries_excluded_from_generation": True,
        "old_eval_rows_in_train": 0,
        "trajectory_lineage": "family split is shared by all stochastic and color variants",
        "focus_quality": quality,
        "dataset_admission": admission,
    }
    atomic(building / "manifest.json", encoded(report))
    if dataset.exists():
        raise ValueError("unexpected existing campaign dataset")
    os.replace(building, dataset)
    return report


def focus_quality(games) -> dict:
    """Count requested optional observations once, including typed missing labels."""
    totals = {"requested": 0, "completed": 0, "unlabeled": 0}
    strata = {"group": {}, "family": {}, "split": {}, "side": {}, "ply_stage": {}, "branch": {}}
    excluded_candidates = {}
    positions = {"requested": set(), "completed": set(), "unlabeled": set()}
    for game in games:
        for row in game["records"]:
            deviation = row.get("deviation")
            recovery = deviation.get("recovery") if deviation else None
            missing_positions = set()
            for branch, observation, ply in (
                ("deviation", deviation, row["ply"] + 1),
                ("recovery", recovery, row["ply"] + 2),
            ):
                if not isinstance(observation, dict):
                    continue
                missing = observation.get("status") in (
                    "unlabeled_incomplete_depth",
                    "unlabeled_deferred",
                )
                if missing:
                    if "score" in observation or "candidates" in observation:
                        raise ValueError(
                            "unlabeled optional observation contains a fabricated label"
                        )
                    if not observation.get("failure_receipt") and not observation.get(
                        "ledger_task"
                    ):
                        raise ValueError(
                            "unlabeled optional observation lacks retained failure evidence"
                        )
                elif "score" not in observation and "terminal_outcome" not in observation:
                    raise ValueError("optional observation has neither a label nor a typed failure")
                state = "unlabeled" if missing else "completed"
                totals["requested"] += 1
                totals[state] += 1
                # SFEN move counters are not distinct positions. These are quality units,
                # independent of dataset symmetry deduplication and split admission.
                position = " ".join(observation["sfen"].split()[:3])
                positions["requested"].add(position)
                positions[state].add(position)
                if missing:
                    missing_positions.add(position)
                labels = {
                    "group": row_group(game, ply),
                    "family": game.get("family", "unspecified"),
                    "split": game.get("split", "unspecified"),
                    "side": observation["sfen"].split()[1],
                    "ply_stage": "opening" if ply < 40 else "middle" if ply < 110 else "late",
                    "branch": branch,
                }
                for category, label in labels.items():
                    counts = strata[category].setdefault(
                        label, {"requested": 0, "completed": 0, "unlabeled": 0}
                    )
                    counts["requested"] += 1
                    counts[state] += 1
            split = game.get("split", "unspecified")
            excluded_candidates[split] = excluded_candidates.get(split, 0) + sum(
                candidate.get("child_terminal") == "None"
                and " ".join(candidate["child_sfen"].split()[:3]) in missing_positions
                for candidate in row.get("candidates", [])
            )
    return {
        **totals,
        "completion_rate": totals["completed"] / totals["requested"]
        if totals["requested"]
        else None,
        "by": strata,
        "unique_positions": {key: len(value) for key, value in positions.items()},
        "duplicate_unlabeled_observations": totals["unlabeled"] - len(positions["unlabeled"]),
        "missing_positions_also_labeled": len(positions["unlabeled"] & positions["completed"]),
        "excluded_candidate_observations_by_split": excluded_candidates,
        "denominator": "requested optional observations, not retry attempts; root D12 excluded",
    }


def focus_gate(output: Path, config: dict) -> dict:
    from .evaluator_data import atomic, digest, encoded

    def games():
        for path in sorted((output / "games").glob("*.json.gz")):
            receipt_path = path.with_suffix(".receipt.json")
            if config.get("recovery_policy") and not receipt_path.exists():
                continue  # Output-before-receipt is not committed label evidence.
            receipt = json.loads(receipt_path.read_text())
            if digest(path) != receipt["sha256"]:
                raise ValueError("focus quality source identity changed")
            game = json.loads(gzip.decompress(path.read_bytes()))
            for row in game["records"]:
                deviation = row.get("deviation")
                for observation in (deviation, deviation.get("recovery") if deviation else None):
                    if isinstance(observation, dict) and observation.get("status") in (
                        "unlabeled_incomplete_depth",
                        "unlabeled_deferred",
                    ):
                        if observation.get("ledger_task"):
                            _validate_deferred_observation(output, observation, config)
                            continue
                        ref = output / observation["failure_receipt"]
                        if (
                            not ref.resolve().is_relative_to((output / "failures").resolve())
                            or not ref.is_file()
                        ):
                            raise ValueError("missing retained optional-label failure receipt")
                        if digest(ref) != observation["failure_sha256"]:
                            raise ValueError("optional-label failure receipt changed")
                        evidence = json.loads(ref.read_text())
                        if (
                            evidence["error_type"] != "USIIncompleteDepthError"
                            or evidence["sfen"] != observation["sfen"]
                        ):
                            raise ValueError("optional-label failure evidence mismatch")
            yield game

    report = focus_quality(games())
    campaign = config["defense_campaign"]
    report["maximum_unlabeled"] = campaign["maximum_unlabeled_focus"]
    report["minimum_completion_rate"] = campaign["minimum_focus_completion_rate"]
    report["legacy_passed"] = report["unlabeled"] <= campaign["maximum_unlabeled_focus"] and (
        report["completion_rate"] is None
        or report["completion_rate"] >= campaign["minimum_focus_completion_rate"]
    )
    admission = admission_identity(output)
    report["dataset_admission"] = admission
    report["passed"] = admission is not None or report["legacy_passed"]
    atomic(output / "focus-quality.json", encoded(report))
    if not report["passed"] and not config.get("recovery_policy"):
        raise ValueError("optional focus completeness gate failed; Astra review required")
    return report


def _validate_deferred_observation(output: Path, observation: dict, config: dict) -> None:
    import sqlite3

    if not config.get("recovery_policy") or observation.get("status") != "unlabeled_deferred":
        raise ValueError("ledger missing label requires the recovery acceptance contract")
    path = output / "tasks.sqlite3"
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing-label ledger missing or linked")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT identity,status,attempts FROM tasks WHERE id=?", (observation["ledger_task"],)
        ).fetchone()
        counters = dict(db.execute("SELECT name,value FROM counters"))
    if row is None:
        raise ValueError("missing-label task not found")
    identity, status, attempts = json.loads(row[0]), row[1], json.loads(row[2])
    if (
        identity["sfen"] != observation["sfen"]
        or identity["branch"] not in ("deviation", "recovery")
        or status != "deferred"
        or not (
            len(attempts) == 2
            or (
                len(attempts) == 1
                and (
                    sum(
                        a.get("elapsed_s", config["recovery_policy"]["maximum_attempt_seconds"])
                        for a in attempts
                    )
                    + config["recovery_policy"]["maximum_attempt_seconds"]
                    > config["recovery_policy"].get("maximum_task_seconds", float("inf"))
                    or counters.get("hard_attempts", 0)
                    >= config["recovery_policy"]["maximum_hard_attempts"]
                    or counters.get("hard_seconds", 0)
                    + config["recovery_policy"]["maximum_attempt_seconds"]
                    > config["recovery_policy"]["maximum_hard_seconds"]
                )
            )
        )
        or identity["teacher"] != config["teacher_binary_sha256"]
        or identity["teacher_config"] != config["teacher_config_sha256"]
    ):
        raise ValueError("missing-label task identity or exhausted budget mismatch")
