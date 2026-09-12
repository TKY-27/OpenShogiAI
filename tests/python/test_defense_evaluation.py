"""Move-screen regressions: phase selection and catastrophic candidate moves stay visible."""

import gzip
import json
from types import SimpleNamespace

from open_shogi_training import defense_evaluation as evaluation
from open_shogi_training.evaluator_data import atomic, digest, encoded


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
