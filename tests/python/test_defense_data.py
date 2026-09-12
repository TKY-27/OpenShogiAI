"""Frozen offline lineage, actual-search deviations, replay exclusion and bounded retry."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from open_shogi_training import evaluator_data as data
from open_shogi_training.defense_scenarios import (
    GROUPS,
    SPLITS,
    assignment,
    default_families,
    rotate_move,
    rotate_sfen,
    validate_campaign,
)
from open_shogi_training.labeling.usi import (
    USICandidate,
    USIIncompleteDepthError,
    USIProtocolError,
    USIScore,
    USISearchResult,
)
from open_shogi_training.phase10u_execution import successor_sfen
from open_shogi_training.phase10v_model import sparse_features


def campaign():
    return {
        "families": default_families(),
        "variants_per_family": 256,
        "probe_nodes": 4000,
        "probe_stride": 12,
        "relabel_nodes": 2_000_000,
        "relabel_depth": 16,
        "maximum_unlabeled_focus": 200,
        "minimum_focus_completion_rate": 0.95,
        "source_row_caps": {"generated": 900_000, "r3_replay": 300_000},
    }


def positions():
    result, keys = [], set()
    for family in default_families():
        sfen = data.START
        for move in family["moves"]:
            sfen = successor_sfen(sfen, move)
            key = min(data.symmetry_keys(sfen))
            if key not in keys:
                result.append(sfen)
                keys.add(key)
    return result


def test_family_variants_and_color_transforms_keep_split():
    config = {"games": 3072, "teacher_depth": 12, "first_game": 100, "defense_campaign": campaign()}
    validate_campaign(config)
    for game in range(100, 3172):
        family, variant = assignment(config, game)
        assert family == config["defense_campaign"]["families"][(game - 100) % 12]
        assert variant == (game - 100) // 12
    assert {(f["group"], f["split"]) for f in default_families()} == {
        (group, split) for group in GROUPS for split in SPLITS
    }
    for sfen in positions():
        assert data.symmetry_keys(sfen) == data.symmetry_keys(rotate_sfen(sfen))
        assert rotate_sfen(rotate_sfen(sfen)) == sfen
    for move in ("7g7f", "8h2b+", "P*4e"):
        assert rotate_move(rotate_move(move)) == move
    config["defense_campaign"]["families"][1]["moves"] = config["defense_campaign"]["families"][0][
        "moves"
    ]
    with pytest.raises(ValueError, match="identical prefixes"):
        validate_campaign(config)


def write_replay(root: Path, sfens: list[str]):
    folder = root / "old/data/dataset"
    folder.mkdir(parents=True)
    split_rows = {"train": sfens[:2], "validation": sfens[2:3], "development_test": sfens[3:4]}
    for split, states in split_rows.items():
        features = np.zeros((len(states), 2, 48), dtype=np.uint16)
        lengths = np.zeros((len(states), 2), dtype=np.uint8)
        with gzip.open(folder / f"{split}-rows.jsonl.gz", "wb") as stream:
            for i, sfen in enumerate(states):
                black, white, stm = sparse_features(sfen)
                for side, active in enumerate((white, black) if stm else (black, white)):
                    features[i, side, : len(active)] = active
                    lengths[i, side] = len(active)
                stream.write(
                    data.encoded(
                        {
                            "sfen": sfen,
                            "game": i,
                            "ply": 10,
                            "kind": "root",
                            "score": {"kind": "cp", "value": 20},
                            "symmetry_key": min(data.symmetry_keys(sfen)),
                        }
                    )
                    + b"\n"
                )
        np.save(folder / f"{split}-features.npy", features)
        np.save(folder / f"{split}-lengths.npy", lengths)
        np.save(folder / f"{split}-targets.npy", np.full(len(states), 20, dtype=np.float32))
    raw = folder.parent / "games/000000.json.gz"
    data.atomic(raw, gzip.compress(data.encoded({"original": True})))
    manifest = {
        "unique_positions": {s: len(rows) for s, rows in split_rows.items()},
        "source_games": [
            {"path": raw.name, "sha256": data.digest(raw), "game": 0, "split": "train"}
        ],
        "artifacts": [{"path": p.name, "sha256": data.digest(p)} for p in sorted(folder.iterdir())],
    }
    data.atomic(folder / "manifest.json", data.encoded(manifest))
    return folder


def setup_generated(root: Path):
    states = positions()
    old = write_replay(root, states)
    guard = root / "guard.json"
    data.atomic(guard, data.encoded({}))
    config = {
        "seed": 83,
        "teacher_depth": 12,
        "games": 3,
        "split_guard_path": guard.name,
        "split_guard_sha256": data.digest(guard),
        "defense_campaign": campaign(),
    }
    config["defense_campaign"]["families"] = [default_families()[i] for i in (0, 4, 8)]
    config["defense_campaign"]["replay_dataset"] = {
        "path": str(old.relative_to(root)),
        "manifest_sha256": data.digest(old / "manifest.json"),
    }
    config["defense_campaign"]["source_row_caps"] = {"generated": 1, "r3_replay": 1}
    output = root / "local/generated"
    data.atomic(output / "generation.json", data.encoded(config))
    data.atomic(output / "generation-complete.json", data.encoded({"games": 3}))
    for game in range(3):
        family, variant = assignment(config, game)
        rows = []
        for sfen in [*states[:4], *states[4 + game * 2 : 6 + game * 2]]:
            rows.append(
                {
                    "sfen": sfen,
                    "ply": 20,
                    "deviation": None,
                    "candidates": [
                        {"score": {"kind": "cp", "value": 80}, "child_terminal": "Some(Checkmate)"}
                    ],
                }
            )
        raw = {
            "game": game,
            "seed": config["seed"] + game,
            "split": family["split"],
            "family": family["id"],
            "group": family["group"],
            "variant": variant,
            "prefix_moves": family["moves"],
            "initial_sfen": data.START,
            "records": rows,
        }
        path = output / "games" / f"{game:06d}.json.gz"
        data.atomic(path, gzip.compress(data.encoded(raw)))
        data.atomic(path.with_suffix(".receipt.json"), data.encoded({"sha256": data.digest(path)}))
    return config, output, old, states


def test_campaign_preparation_excludes_old_eval_and_deduplicates_replay(tmp_path):
    config, output, old, states = setup_generated(tmp_path)
    report = data.prepare(tmp_path, output, config, [])
    assert report["unique_positions"] == {"train": 2, "validation": 2, "development_test": 2}
    assert report["source_train_rows"] == {"generated": 1, "r3_replay": 1}
    assert report["source_cap_removed"] == 1
    assert report["old_eval_rows_in_train"] == 0
    assert report["replay_source"]["manifest_sha256"] == data.digest(old / "manifest.json")
    assert all("family" in ref for ref in report["source_games"])
    seen = set()
    for split in SPLITS:
        with gzip.open(output / "dataset" / f"{split}-rows.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        groups = np.load(output / "dataset" / f"{split}-groups.npy")
        assert groups.tolist() == [GROUPS[row["group"]] for row in rows]
        for row in rows:
            assert row["symmetry_key"] not in seen
            seen.add(row["symmetry_key"])
            assert row["symmetry_key"] not in {min(data.symmetry_keys(s)) for s in states[2:4]}
    assert data.prepare(tmp_path, output, config, []) == report
    target = old.parent / "games/000000.json.gz"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw source identity"):
        data.prepare(tmp_path, output, config, [])


def test_preflight_exclusions_also_remove_old_train_replay(tmp_path):
    config, output, _, states = setup_generated(tmp_path)
    report = data.prepare(tmp_path, output, config, states[:2])
    assert report["source_train_rows"]["r3_replay"] == 0
    assert report["replay_source"]["old_validation_or_development_reused"] == 0


def test_recovery_has_its_own_side_to_move_teacher_value():
    game = {
        "records": [
            {
                "sfen": data.START,
                "ply": 10,
                "candidates": [{"score": {"kind": "cp", "value": 80}, "child_terminal": "done"}],
                "deviation": {
                    "sfen": data.START,
                    "move": "7g7f",
                    "score": {"kind": "cp", "value": -500},
                    "recovery": {
                        "sfen": data.START,
                        "reply": "3c3d",
                        "score": {"kind": "cp", "value": 430},
                    },
                },
            }
        ]
    }
    rows = list(data._observations(game))
    assert [(score["value"], kind, ply) for _, score, kind, ply, *_ in rows] == [
        (80, "root", 10),
        (-500, "deviation", 11),
        (430, "recovery", 12),
    ]


@pytest.mark.parametrize("recoverable", [True, False])
def test_only_incomplete_depth_gets_one_fixed_ceiling_retry_with_evidence(tmp_path, recoverable):
    candidate = USICandidate(1, USIScore("cp", 20), ("7g7f",), 12, 12, 100)
    result = USISearchResult("7g7f", (candidate,), 1)
    calls = []

    class Teacher:
        def __init__(self):
            self.search_diagnostics = {"stdout_tail": "incomplete root output"}

        def analyze_with_retry(self, sfen, *, nodes, depth):
            calls.append((sfen, nodes, depth))
            if len(calls) == 1:
                raise (USIIncompleteDepthError if recoverable else USIProtocolError)("first error")
            return result

    config = {"teacher_depth": 12, "teacher_nodes": 2_000_000, "teacher_retry_nodes": 8_000_000}
    kwargs = dict(
        root=tmp_path,
        output=tmp_path / "local/out",
        config=config,
        game=0,
        ply=0,
        branch="root",
        moves=[],
        records=[],
    )
    state = {"sfen": data.START, "terminal": "None", "successors": [{"move": "7g7f"}]}
    if recoverable:
        assert data._teacher_observation(Teacher(), state, **kwargs)[0] is result
        assert [call[1:] for call in calls] == [(2_000_000, 12), (8_000_000, 12)]
    else:
        with pytest.raises(USIProtocolError):
            data._teacher_observation(Teacher(), state, **kwargs)
        assert len(calls) == 1
    failure = json.loads(next((tmp_path / "local/out/failures").glob("*.json")).read_text())
    assert failure["stdout_tail"] == "incomplete root output"


@pytest.mark.parametrize("incomplete_recovery", [False, True])
def test_generation_labels_actual_r3_choice_and_teacher_reply_without_changing_mainline(
    tmp_path, monkeypatch, incomplete_recovery
):
    from open_shogi_training import defense_scenarios

    def native_state(sfen):
        moves = ["7g7f", "2g2f", "5g5f"] if sfen.split()[1] == "b" else ["3c3d", "8c8d", "5c5d"]
        successors = [
            {
                "move": move,
                "sfen": successor_sfen(sfen, move),
                "terminal": "None",
                "child_cp": -100 if move == "2g2f" else 0,
            }
            for move in moves
        ]
        return {"sfen": sfen, "terminal": "None", "successors": successors}

    class Native:
        def __init__(self, *_):
            self.sfen = data.START

        def ask(self, **request):
            self.sfen = (
                request["reset"]
                if "reset" in request
                else successor_sfen(self.sfen, request["movement"])
            )
            return native_state(self.sfen)

        def close(self):
            pass

    class Search:
        def __init__(self, *_):
            pass

        def search(self, initial, moves, expected):
            assert initial == expected == data.START and moves == []
            return {"best_move": "5g5f", "score": 120, "nodes": 4000}

        def close(self):
            pass

    queries = []

    class Teacher:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def analyze_with_retry(self, sfen, *, nodes, depth):
            queries.append((sfen, nodes, depth))
            if incomplete_recovery and sfen.split()[1] == "b" and sfen != data.START:
                raise USIIncompleteDepthError("bounded recovery incomplete")
            # Recovery has the original player's turn but its pawn has already moved.
            moves = ["7g7f", "2g2f"] if sfen.split()[1] == "b" else ["3c3d", "8c8d"]
            candidates = tuple(
                USICandidate(i + 1, USIScore("cp", 100 - i * 20), (move,), depth, depth, 10)
                for i, move in enumerate(moves)
            )
            return USISearchResult(moves[0], candidates, 1)

    # The fake native needs only legal successors of roots being labeled, not a full move generator.
    original_state = native_state

    def safe_state(sfen):
        if sfen.split()[1] == "b" and sfen != data.START:
            moves = ["7g7f", "2g2f"]
            return {
                "sfen": sfen,
                "terminal": "None",
                "successors": [
                    {"move": m, "sfen": successor_sfen(sfen, m), "terminal": "None", "child_cp": 0}
                    for m in moves
                ],
            }
        return original_state(sfen)

    native_state = safe_state
    monkeypatch.setattr(data, "Replay", Native)
    monkeypatch.setattr(data, "USIEngine", Teacher)
    monkeypatch.setattr(data, "teacher_config", lambda *_: None)
    monkeypatch.setattr(defense_scenarios, "R3Probe", Search)
    config = {
        "seed": 1,
        "teacher_depth": 12,
        "teacher_nodes": 2_000_000,
        "teacher_retry_nodes": 32_000_000,
        "max_plies": 1,
        "sample_stride": 1,
        "deviation_stride": 12,
        "defense_campaign": campaign(),
    }
    output = tmp_path / "local/data"
    data._generate_group(str(tmp_path), str(output), config, [0])
    path = output / "games/000000.json.gz"
    raw = json.loads(gzip.decompress(path.read_bytes()))
    assert raw["moves"] == ["2g2f"]  # Offline family prefix, not probe/static preferred move.
    deviation = raw["records"][0]["deviation"]
    assert deviation["move"] == "5g5f"  # Actual search, not the static child_cp minimum 2g2f.
    assert deviation["recovery"]["reply"] == "3c3d"
    if incomplete_recovery:
        assert deviation["recovery"]["status"] == "unlabeled_incomplete_depth"
        assert "score" in deviation and "score" not in deviation["recovery"]
    assert [depth for _, _, depth in queries] == (
        [12, 16, 16, 16] if incomplete_recovery else [12, 16, 16]
    )
    data._generate_group(str(tmp_path), str(output), config, [0])
    assert len(queries) == (
        4 if incomplete_recovery else 3
    )  # Completed trajectory is not regenerated on resume.


def test_optional_budget_mask_keeps_failure_receipts_and_root_remains_strict(tmp_path):
    class Teacher:
        def analyze_with_retry(self, *_args, **_kwargs):
            raise USIIncompleteDepthError("fixed depth not completed")

    kwargs = dict(
        root=tmp_path,
        output=tmp_path / "local/data",
        config={"teacher_depth": 16, "teacher_nodes": 2_000_000, "teacher_retry_nodes": 32_000_000},
        game=0,
        ply=10,
        branch="deviation",
        moves=[],
        records=[],
    )
    result, missing = data._optional_focus_observation(
        Teacher(), {"sfen": data.START, "terminal": "None"}, **kwargs
    )
    assert result is None and missing["status"] == "unlabeled_incomplete_depth"
    assert missing["requested_depth"] == 16 and missing["node_ceiling"] == 32_000_000
    assert "score" not in missing and missing["recovery"] == "not_attempted"
    assert (kwargs["output"] / missing["failure_receipt"]).exists()
    assert len(list((kwargs["output"] / "failures").glob("*.json"))) == 2
    with pytest.raises(USIIncompleteDepthError):
        data._teacher_observation(
            Teacher(), {"sfen": data.START, "terminal": "None"}, **{**kwargs, "branch": "root"}
        )


def test_masked_child_has_no_low_depth_candidate_substitute_but_completed_root_is_retained():
    child = successor_sfen(data.START, "7g7f")
    marker = {
        "status": "unlabeled_incomplete_depth",
        "sfen": child,
        "failure_receipt": "f.json",
        "move": "7g7f",
    }
    row = {
        "sfen": data.START,
        "ply": 10,
        "candidates": [
            {
                "score": {"kind": "cp", "value": 12},
                "child_terminal": "None",
                "child_sfen": child,
                "pv": ["7g7f"],
            }
        ],
        "deviation": marker,
    }
    records = [
        row,
        {
            "sfen": child,
            "ply": 11,
            "candidates": [{"score": {"kind": "cp", "value": 30}, "child_terminal": "done"}],
            "deviation": None,
        },
    ]
    assert [(s, k) for s, _, k, *_ in data._observations({"records": records})] == [
        (data.START, "root"),
        (child, "root"),
    ]
    # A recovery failure does not erase the completed deviation's independently requested D16 value.
    row["deviation"] = {
        "sfen": child,
        "score": {"kind": "cp", "value": -40},
        "move": "7g7f",
        "recovery": marker,
    }
    assert [k for _, _, k, *_ in data._observations({"records": [row]})] == ["root", "deviation"]


def test_focus_quality_counts_observations_not_retries_and_reports_missingness():
    from open_shogi_training.defense_scenarios import focus_quality

    completed = {"sfen": data.START, "score": {"kind": "cp", "value": 30}}
    missing = {
        "sfen": data.START,
        "status": "unlabeled_incomplete_depth",
        "failure_receipt": "failure.json",
        "recovery": "not_attempted",
    }
    raw = {
        "group": "defense",
        "prefix_moves": [],
        "records": [{"ply": 50, "deviation": dict(completed)} for _ in range(19)]
        + [{"ply": 62, "deviation": missing}],
    }
    report = focus_quality([raw])
    assert (
        report["requested"],
        report["completed"],
        report["unlabeled"],
        report["completion_rate"],
    ) == (20, 19, 1, 0.95)
    assert report["by"]["group"]["defense"]["unlabeled"] == 1
    assert report["by"]["side"]["b"]["requested"] == 20
    assert report["by"]["ply_stage"]["middle"]["requested"] == 20
    missing["score"] = {"kind": "cp", "value": 0}
    with pytest.raises(ValueError, match="fabricated label"):
        focus_quality([raw])


def test_focus_gate_enforces_completed_cohort_rate_and_absolute_cap(tmp_path):
    from open_shogi_training.defense_scenarios import focus_gate

    output = tmp_path / "local/data"
    error = output / "failures/missing.json"
    data.atomic(error, data.encoded({"error_type": "USIIncompleteDepthError", "sfen": data.START}))
    marker = {
        "sfen": data.START,
        "status": "unlabeled_incomplete_depth",
        "failure_receipt": "failures/missing.json",
        "failure_sha256": data.digest(error),
    }
    completed = {"sfen": data.START, "score": {"kind": "cp", "value": 10}}
    raw = {
        "group": "defense",
        "prefix_moves": [],
        "records": [{"ply": 50, "deviation": completed} for _ in range(19)]
        + [{"ply": 50, "deviation": marker}],
    }
    path = output / "games/000000.json.gz"
    data.atomic(path, gzip.compress(data.encoded(raw)))
    data.atomic(path.with_suffix(".receipt.json"), data.encoded({"sha256": data.digest(path)}))
    config = {"defense_campaign": campaign()}
    assert focus_gate(output, config)["completion_rate"] == 0.95
    config["defense_campaign"]["minimum_focus_completion_rate"] = 0.96
    with pytest.raises(ValueError, match="completeness gate failed"):
        focus_gate(output, config)
    config["defense_campaign"].update(minimum_focus_completion_rate=0.95, maximum_unlabeled_focus=0)
    with pytest.raises(ValueError, match="completeness gate failed"):
        focus_gate(output, config)
    error.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        focus_gate(output, config)


def test_incomplete_dataset_cleanup_refuses_nested_symlink(tmp_path):
    from open_shogi_training.defense_scenarios import _remove_incomplete_dataset

    output = tmp_path / "local/run/data"
    staging = output / "dataset-building"
    staging.mkdir(parents=True)
    protected = tmp_path / "protected"
    protected.write_text("retain")
    (staging / "nested").symlink_to(protected)
    with pytest.raises(ValueError, match="link or mount"):
        _remove_incomplete_dataset(tmp_path, output)
    assert protected.read_text() == "retain"
    assert staging.exists()
