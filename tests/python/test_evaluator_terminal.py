"""Typed offline terminal observations, native validation and retained failure evidence."""

import gzip
import json

import pytest
from open_shogi_training import evaluator_data as data
from open_shogi_training.labeling.usi import (
    USICandidate,
    USIProtocolError,
    USIScore,
    USISearchResult,
    USITerminalResult,
)
from open_shogi_training.phase10u_execution import successor_sfen

BLACK_WIN = "1p7/KRRBBPPPP/NN7/9/9/9/9/9/8k b 2P 1"
WHITE_WIN = "K8/9/9/9/9/9/nn7/krrbbpppp/1P7 w p 2"


def state(sfen, *, terminal="None", win=True):
    black = sfen.split()[1] == "b"
    points = 28 if black else 27
    return {
        "sfen": sfen,
        "terminal": terminal,
        "teacher_resign_eligible": False,
        "teacher_declaration": {
            "rule": "csa_28_27",
            "side": "black" if black else "white",
            "minimum_camp_pieces": 10,
            "required_points": points,
            "points": points if win else None,
            "result": "win" if win else "invalid_loss",
        },
    }


@pytest.mark.parametrize("sfen", [BLACK_WIN, WHITE_WIN])
def test_win_requires_verified_native_side_and_threshold(sfen):
    result = USITerminalResult("win", "bestmove win", 1)
    original = state(sfen)
    assert data._validated_teacher_terminal(result, original, 40)["kind"] == "terminal"
    for field, value in [
        ("result", "invalid_loss"),
        ("side", "wrong"),
        ("points", 26),
        ("rule", "unknown"),
    ]:
        changed = {
            **original,
            "teacher_declaration": {**original["teacher_declaration"], field: value},
        }
        with pytest.raises(ValueError, match="not a verified native"):
            data._validated_teacher_terminal(result, changed, 40)


def test_resign_requires_native_no_legal_move_termination():
    result = USITerminalResult("resign", "bestmove resign", 1)
    native = state(data.START, win=False)
    with pytest.raises(ValueError, match="disagrees with native"):
        data._validated_teacher_terminal(result, native, 0)
    native.update(terminal="Some(NoLegalMoves { loser: White })", teacher_resign_eligible=True)
    assert (
        data._validated_teacher_terminal(result, native, 0)["validation"] == "native_no_legal_moves"
    )


def install_generator(monkeypatch, *, reject=False):
    candidates = tuple(
        USICandidate(i + 1, USIScore("cp", 100 - i * 10), (move,), 8, 9, 20)
        for i, move in enumerate(["7g7f", "5g5f", "6g6f"])
    )
    successors = [
        {
            "move": c.pv[0],
            "sfen": successor_sfen(data.START, c.pv[0]),
            "terminal": "None",
            "child_cp": i,
        }
        for i, c in enumerate(candidates)
    ]
    successors.append({**state(WHITE_WIN), "move": "2g2f", "child_cp": -100})
    initial = {**state(data.START, win=False), "successors": successors}
    later = {**state(BLACK_WIN, win=not reject), "successors": []}

    class Native:
        def __init__(self, *_):
            self.closed = False

        def ask(self, **request):
            return initial if "reset" in request else later

        def close(self):
            self.closed = True

    class Teacher:
        def __init__(self, *_, **options):
            self.search_diagnostics = {
                "bestmove_line": "bestmove win",
                "stdout_tail": "bestmove win\n",
            }
            assert options == {"isolate_process_group": False, "allow_terminal_outcomes": True}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def analyze_with_retry(self, sfen):
            if sfen == data.START:
                return USISearchResult(candidates[0].pv[0], candidates, 2)
            return USITerminalResult("win", "bestmove win", 1)

    monkeypatch.setattr(data, "Replay", Native)
    monkeypatch.setattr(data, "USIEngine", Teacher)
    monkeypatch.setattr(data, "teacher_config", lambda *_: None)
    return {"seed": 20260915, "max_plies": 4, "sample_stride": 2, "deviation_stride": 2}


def test_deviation_win_is_masked_while_mainline_continues_and_root_preserves_prefix(
    tmp_path, monkeypatch
):
    config = install_generator(monkeypatch)
    output = tmp_path / "local/data"
    reports = data._generate_group(str(tmp_path), str(output), config, [0])
    raw = json.loads(gzip.decompress((output / "games/000000.json.gz").read_bytes()))
    assert len(raw["moves"]) == 1 and len(raw["records"]) == 2
    assert raw["records"][0]["candidates"][0]["score"] == {"kind": "cp", "value": 100}
    deviation = raw["records"][0]["deviation"]
    assert deviation["terminal_outcome"]["sfen"] == WHITE_WIN and "score" not in deviation
    assert raw["terminal_outcome"]["sfen"] == BLACK_WIN
    assert raw["terminal_outcome"]["ply"] == 1  # Retained even off the sampling stride.
    observations = list(data._observations(raw))
    assert len(observations) == 6
    assert sum(score["kind"] == "terminal" for _, score, *_ in observations) == 2
    assert reports[0]["teacher_terminal_outcomes"] == {"root_win": 1, "deviation_win": 1}
    assert data._generate_group(str(tmp_path), str(output), config, [0]) == reports

    guard = tmp_path / "guard.json"
    data.atomic(guard, data.encoded({}))
    preparation = {
        **config,
        "first_game": 0,
        "games": 1,
        "split_guard_path": guard.name,
        "split_guard_sha256": data.digest(guard),
    }
    data.atomic(output / "generation.json", data.encoded(preparation))
    data.atomic(output / "generation-complete.json", data.encoded({"games": 1}))
    report = data.prepare(tmp_path, output, preparation, [])
    assert report["teacher_terminal_observations_masked"] == {"win": 2, "resign": 0}
    assert report["mate_observations_masked"] == 0
    assert sum(report["unique_positions"].values()) == 4
    assert report["raw_observations"] == 6
    assert report["duplicate_or_excluded_observations"] == 0


def test_unverified_win_fails_closed_with_prefix_and_exact_failure_context(tmp_path, monkeypatch):
    config = install_generator(monkeypatch, reject=True)
    output = tmp_path / "local/data"
    with pytest.raises(ValueError, match="not a verified native"):
        data._generate_group(str(tmp_path), str(output), config, [0])
    assert not list((output / "games").glob("*.receipt.json"))
    receipt_path = next((output / "failures").glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert (receipt["game"], receipt["ply"], receipt["branch"], receipt["sfen"]) == (
        0,
        1,
        "root",
        BLACK_WIN,
    )
    assert receipt["bestmove_line"] == "bestmove win" and receipt["stdout_tail"] == "bestmove win\n"
    prefix = receipt_path.parent / receipt["prefix"]["path"]
    assert data.digest(prefix) == receipt["prefix"]["sha256"]
    assert len(json.loads(gzip.decompress(prefix.read_bytes()))["records"]) == 1
    original = receipt_path.read_bytes()
    with pytest.raises(ValueError):
        data._generate_group(str(tmp_path), str(output), config, [0])
    assert (
        receipt_path.read_bytes() == original
        and len(list((output / "failures").glob("*.json"))) == 2
    )


def test_malformed_teacher_failure_retains_actual_request_and_raw_response(tmp_path):
    class Teacher:
        def analyze_with_retry(self, _sfen):
            error = USIProtocolError("malformed bestmove")
            error.bestmove_line = "bestmove 0000"
            error.stdout_tail = "info string before failure\nbestmove 0000\n"
            raise error

    output = tmp_path / "local/data"
    with pytest.raises(USIProtocolError):
        data._teacher_observation(
            Teacher(),
            state(data.START, win=False),
            root=tmp_path,
            output=output,
            config={},
            game=35,
            ply=88,
            branch="deviation",
            moves=[],
            records=[],
        )
    report = json.loads(next((output / "failures").glob("*.json")).read_text())
    assert report["game"] == 35 and report["ply"] == 88 and report["branch"] == "deviation"
    assert report["sfen"] == data.START and report["bestmove_line"] == "bestmove 0000"
