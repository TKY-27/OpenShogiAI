"""Move-screen regressions: phase selection and catastrophic candidate moves stay visible."""

import gzip
import json
from types import SimpleNamespace

import pytest
from open_shogi_training import defense_evaluation as evaluation
from open_shogi_training.evaluator_data import atomic, digest, encoded
from open_shogi_training.evaluator_ledger import DeferredTaskError, Ledger, task_identity
from open_shogi_training.labeling.usi import USIIncompleteDepthError, USIResourceError


def test_root_selection_uses_phase_and_distinct_trajectories(tmp_path):
    rows = []
    for index, (group, ply) in enumerate(
        (("general", 60), ("opening", 20), ("defense", 40), ("attack_end", 120))
    ):
        rows.extend(
            [
                {
                    "game": index,
                    "group": group,
                    "kind": "root",
                    "ply": ply,
                    "sfen": f"valid-{index}",
                },
                {
                    "game": index,
                    "group": group,
                    "kind": "root",
                    "ply": 200,
                    "sfen": f"wrong-phase-{index}",
                },
            ]
        )
    path = tmp_path / "development_test-rows.jsonl.gz"
    path.write_bytes(gzip.compress(b"\n".join(encoded(r) for r in rows)))
    atomic(
        tmp_path / "manifest.json",
        encoded(
            {
                "artifacts": [{"path": path.name, "sha256": digest(path)}],
                "source_games": [{"game": i, "split": "development_test"} for i in range(4)],
            }
        ),
    )
    selected = evaluation.select_roots(tmp_path, 42, per_group=1)
    assert len(selected) == 4
    assert all(row["sfen"].startswith("valid-") for row in selected)


def test_candidate_mate_loss_is_not_excluded_as_an_already_lost_root(tmp_path, monkeypatch):
    run = tmp_path / "local/run"
    run.mkdir(parents=True)
    atomic(run / "run.json", encoded({"fixture": True}))
    atomic(run / "fit/best.osaval03", b"fixture")
    atomic(run / "known.json", encoded({"roots": []}))
    row = {"sfen": "root b - 1", "group": "defense", "family": "test", "game": 0, "ply": 30}
    monkeypatch.setattr(evaluation, "select_roots", lambda *_: [row])

    class Probe:
        def __init__(self, _root, config):
            self.move = "bad" if "best.osaval03" in config["leaf_path"] else "safe"

        def search(self, *_):
            return {"best_move": self.move}

        def close(self):
            pass

    class Replay:
        def __init__(self, *_):
            pass

        def ask(self, *, reset, successors):
            assert successors is True
            if reset != row["sfen"]:
                return {"sfen": reset, "terminal": "None", "successors": [{"move": "reply"}]}
            return {
                "sfen": row["sfen"],
                "terminal": "None",
                "successors": [
                    {"move": move, "sfen": move, "terminal": "None"} for move in ("safe", "bad")
                ],
            }

        def close(self):
            pass

    class Teacher:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class Candidate:
        def __init__(self, move, kind="cp", value=100):
            self.pv = [move]
            self.score = SimpleNamespace(kind=kind, value=value)

        def as_dict(self):
            return {"pv": self.pv, "score": vars(self.score)}

    def observe(_teacher, state, **_):
        assert "successors" in state
        candidates = (
            [Candidate("safe"), Candidate("bad")]
            if state["sfen"] == row["sfen"]
            else [Candidate("reply", "mate", 5)]
            if state["sfen"] == "bad"
            else [Candidate("reply", "cp", -100)]
        )
        return SimpleNamespace(candidates=candidates, primary=candidates[0]), None

    monkeypatch.setattr(evaluation, "R3Probe", Probe)
    monkeypatch.setattr(evaluation, "Replay", Replay)
    monkeypatch.setattr(evaluation, "USIEngine", Teacher)
    monkeypatch.setattr(evaluation, "teacher_config", lambda *_: None)
    monkeypatch.setattr(evaluation, "_teacher_observation", observe)
    config = {
        "seed": 1,
        "generation": {"leaf_path": "r3", "defense_campaign": {}},
        "evaluation": {
            "screen_teacher_depth": 16,
            "screen_probe_nodes": 100000,
            "regression_positions_path": str((run / "known.json").relative_to(tmp_path)),
            "regression_positions_sha256": digest(run / "known.json"),
        },
    }
    report = evaluation.screen(tmp_path, run, config)
    case = json.loads((run / "move-screen/case-000.json").read_text())
    assert case["eligible"] is True
    assert case["regret_cp"] == {"r3": 0, "candidate": 30100}
    assert report["groups"]["defense"]["candidate"]["major_errors_400cp"] == 1
    assert report["screen_pass"] is False


@pytest.mark.parametrize(
    "failure", ["depth", "timeout", "budget", "integrity", "resource", "illegal"]
)
def test_optional_screen_miss_continues_but_mandatory_failures_stop(tmp_path, monkeypatch, failure):
    run = tmp_path / "local/run"
    run.mkdir(parents=True)
    atomic(run / "run.json", encoded({"fixture": True}))
    atomic(run / "fit/best.osaval03", b"fixture")
    atomic(run / "known.json", encoded({"roots": []}))
    # The original failed request is game914, development_test, child ply89, D16.
    original = "1n1R4l/1+P1sP2k1/2p3ns1/3p1p1pp/4p4/1BPP1PglP/1P1G3P1/6SR1/KNS2G1NL b BLg4p 89"
    rows = [
        dict(sfen=sfen, group="general", family="central_space", game=game, ply=88)
        for game, sfen in ((914, original), (2474, "next b - 1"))
    ]
    monkeypatch.setattr(evaluation, "select_roots", lambda *_: rows.copy())
    config = {
        "seed": 1,
        "generation": {
            "leaf_path": "r3",
            "teacher_binary_sha256": "teacher",
            "teacher_config_sha256": "config",
            "defense_campaign": {"families": [{"group": "general", "split": "development_test"}]},
        },
        "evaluation": {
            "screen_teacher_depth": 16,
            "screen_probe_nodes": 100000,
            "regression_positions_path": str((run / "known.json").relative_to(tmp_path)),
            "regression_positions_sha256": digest(run / "known.json"),
        },
    }

    class Probe:
        def __init__(self, _root, configuration):
            self.candidate = "best.osaval03" in configuration["leaf_path"]

        def search(self, *_):
            if self.candidate and failure == "integrity":
                raise ValueError("model identity changed")
            return {"best_move": "illegal" if failure == "illegal" else "6a6b+"}

        def close(self):
            pass

    class Replay:
        def __init__(self, *_):
            pass

        def ask(self, *, reset, successors):
            return {
                "sfen": reset,
                "terminal": "None",
                "successors": [{"move": "6a6b+", "sfen": "child w - 90", "terminal": "None"}],
            }

        def close(self):
            pass

    class Teacher:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    visited = []

    def observe(_teacher, *, state, config, game, ply, branch, moves, output, **_):
        visited.append((game, ply, branch))
        if game == 914 and branch == "screen-child":
            if failure == "resource":
                raise USIResourceError("allocation failed")
            ledger = Ledger(output)
            key, _saved = ledger.task(task_identity(config, game, ply, branch, state, moves))
            if failure != "budget":
                ledger.begin(key, 2_000_000)
                ledger.finish(
                    key,
                    "hard",
                    evidence={
                        "outcome": "USITimeoutError"
                        if failure == "timeout"
                        else "USIIncompleteDepthError",
                        "stdout_tail": (
                            "info depth 14 seldepth 22 multipv 3 score cp -4496 nodes 2000762\n"
                        ),
                    },
                )
            else:
                ledger.finish(key, "deferred")
            ledger.close()
            cause = (
                DeferredTaskError("task wall budget exhausted")
                if failure == "budget"
                else USIIncompleteDepthError("D16 incomplete")
            )
            raise DeferredTaskError(key) from cause
        candidate = SimpleNamespace(pv=["6a6b+"], score=SimpleNamespace(kind="cp", value=-100))
        candidate.as_dict = lambda: {"pv": candidate.pv, "score": vars(candidate.score)}
        return SimpleNamespace(candidates=[candidate], primary=candidate), None

    monkeypatch.setattr(evaluation, "R3Probe", Probe)
    monkeypatch.setattr(evaluation, "Replay", Replay)
    monkeypatch.setattr(evaluation, "USIEngine", Teacher)
    monkeypatch.setattr(evaluation, "teacher_config", lambda *_: None)
    monkeypatch.setattr(evaluation, "_teacher_observation", observe)
    if failure in {"integrity", "resource", "illegal"}:
        with pytest.raises(USIResourceError if failure == "resource" else ValueError):
            evaluation.screen(tmp_path, run, config)
        assert not (run / "move-screen/result.json").exists()
        return
    result = evaluation.screen(tmp_path, run, config)
    first = json.loads((run / "move-screen/case-000.json").read_text())
    assert first["status"] == "missing" and first["regret_cp"] is None
    assert first["missing_observations"][0]["requested_depth"] == 16
    assert first["missing_observations"][0]["ply"] == 89
    assert (2474, 89, "screen-child") in visited
    assert result["completed"] == result["missing"] == 1
    assert result["groups"]["general"]["eligible"] == 1
    assert result["groups"]["general"]["excluded_lost_or_mate"] == 0
    assert result["screen_pass"] is None
    # Retained-only mode must accept new receipts and never relaunch runtime or teacher.
    monkeypatch.setattr(evaluation, "R3Probe", lambda *_: pytest.fail("probe started"))
    monkeypatch.setattr(
        evaluation, "USIEngine", lambda *_args, **_kwargs: pytest.fail("teacher started")
    )
    retained = evaluation.screen(tmp_path, run, {**config, "_optional_screen": "retained_only"})
    assert retained["completed"] == retained["missing"] == 1


def test_retained_only_reports_unrun_denominators_without_starting_teacher(tmp_path, monkeypatch):
    run = tmp_path / "local/run"
    run.mkdir(parents=True)
    atomic(run / "run.json", b"{}")
    atomic(run / "fit/best.osaval03", b"fixture")
    atomic(run / "known.json", encoded({"roots": []}))
    row = dict(sfen="root b - 1", group="defense", family="test", game=0, ply=30)
    monkeypatch.setattr(evaluation, "select_roots", lambda *_: [row])
    monkeypatch.setattr(evaluation, "R3Probe", lambda *_: pytest.fail("probe started"))
    monkeypatch.setattr(evaluation, "Replay", lambda *_: pytest.fail("replay started"))
    monkeypatch.setattr(
        evaluation, "USIEngine", lambda *_args, **_kwargs: pytest.fail("teacher started")
    )
    config = {
        "seed": 1,
        "_optional_screen": "retained_only",
        "generation": {"defense_campaign": {}},
        "evaluation": {
            "screen_teacher_depth": 16,
            "screen_probe_nodes": 100000,
            "regression_positions_path": str((run / "known.json").relative_to(tmp_path)),
            "regression_positions_sha256": digest(run / "known.json"),
        },
    }
    report = evaluation.screen(tmp_path, run, config)
    assert report["status"] == "not_run" and report["not_run"] == 1
    assert report["completed"] == report["missing"] == 0
    assert report["groups"]["defense"]["candidate"]["mean_regret_cp"] is None
    assert report["screen_pass"] is None
    assert not (run / "move-screen/case-000.json").exists()
    atomic(run / "fit/best.osaval03", b"changed-model")
    with pytest.raises(ValueError, match="plan changed"):
        evaluation.screen(tmp_path, run, config)


def test_partial_screen_cannot_pass_even_when_observed_cases_improve():
    cases = [
        {
            "group": group,
            "eligible": True,
            "status": "completed",
            "regret_cp": {"r3": 500, "candidate": 0},
        }
        for group in evaluation.GROUPS
        for _ in range(8)
    ]
    cases.append({"group": "defense", "eligible": False, "status": "missing", "regret_cp": None})
    report = evaluation.summarize(cases)
    assert report["opening_defense_improved"] and report["general_attack_preserved"]
    assert report["screen_pass"] is None
