"""Failure injection at persistent teacher-label and trajectory boundaries."""

from pathlib import Path

import pytest
from open_shogi_training.evaluator_data import _ledger_observation
from open_shogi_training.evaluator_ledger import DeferredTaskError, Ledger
from open_shogi_training.labeling.usi import (
    USICandidate,
    USIIncompleteDepthError,
    USIScore,
    USISearchResult,
)


def test_attempt_reservation_survives_worker_death(tmp_path):
    ledger = Ledger(tmp_path)
    key, _ = ledger.task({"game": 211, "ply": 116})
    ledger.begin(key, 2_000_000)
    ledger.close()  # killed after durable reservation, before teacher response
    ledger = Ledger(tmp_path)
    ledger.begin(key, 32_000_000)
    ledger.close()
    ledger = Ledger(tmp_path)
    with pytest.raises(DeferredTaskError):
        ledger.begin(key, 32_000_000)
    assert [a["nodes"] for a in ledger.get(key)["attempts"]] == [2_000_000, 32_000_000]
    ledger.close()


def test_checkpoint_and_accepted_commit_survive_restart(tmp_path):
    ledger = Ledger(tmp_path)
    key, _ = ledger.task({"game": 211})
    ledger.checkpoint(211, {"moves": ["7g7f"], "records": [{"ply": 0}], "rng": [1]})
    ledger.begin(key, 2)
    ledger.finish(key, "accepted", result={"label": 12})
    ledger.close()  # killed before shard/receipt commit
    ledger = Ledger(tmp_path)
    assert ledger.task({"game": 211})[0] == key
    assert ledger.get(key)["result"] == {"label": 12}
    assert ledger.load_checkpoint(211)["moves"] == ["7g7f"]
    assert ledger.summary()["accepted"] == 1
    ledger.close()


def test_depth_failure_does_not_block_next_task_and_is_finite(tmp_path):
    config = {
        "teacher_binary_sha256": "teacher",
        "teacher_config_sha256": "config",
        "teacher_depth": 12,
        "teacher_nodes": 2_000_000,
        "teacher_retry_nodes": 32_000_000,
        "recovery_policy": {"maximum_worker_restarts": 3},
    }
    state = {"sfen": "failed", "successors": [{"move": "7g7f"}], "terminal": "None"}

    class Teacher:
        def __init__(self):
            self.calls = []

        def analyze(self, sfen, **kwargs):
            self.calls.append((sfen, kwargs["nodes"]))
            if sfen == "failed":
                raise USIIncompleteDepthError("depth12 missing")
            return USISearchResult(
                "7g7f", (USICandidate(1, USIScore("cp", 1), ("7g7f",), 12, 12, 1),), 1
            )

    teacher = Teacher()
    kwargs = dict(
        root=Path(tmp_path),
        output=tmp_path,
        config=config,
        game=211,
        ply=116,
        branch="root",
        moves=[],
        records=[],
    )
    for _ in range(4):
        with pytest.raises(DeferredTaskError):
            _ledger_observation(teacher, state, **kwargs)
    result, terminal = _ledger_observation(
        teacher, {**state, "sfen": "normal"}, **{**kwargs, "game": 212}
    )
    assert result.primary.depth == 12 and terminal is None
    assert teacher.calls == [("failed", 2_000_000), ("failed", 32_000_000), ("normal", 2_000_000)]
    ledger = Ledger(tmp_path)
    assert ledger.summary()["deferred"] == 1 and ledger.summary()["accepted"] == 1
    ledger.close()


def test_distinct_task_outage_gate(tmp_path):
    ledger = Ledger(tmp_path)
    for game in range(8):
        for ply in range(4):
            key, _ = ledger.task({"game": game, "ply": ply})
            ledger.begin(key, 2)
            ledger.finish(key, "deferred")
    with pytest.raises(RuntimeError, match="health gate"):
        ledger.check_health()
    ledger.close()


def test_generation_continues_and_pauses_after_new_committed_game(tmp_path, monkeypatch):
    import json

    from open_shogi_training import evaluator_data as data

    source = tmp_path / "source"
    source.write_bytes(b"identity")
    config = {
        "games": 3,
        "workers": 1,
        "teacher_nodes": 2,
        "recovery_policy": {"maximum_supplemental_games": 0},
    }
    for path_key, sha_key in (
        ("replay_path", "replay_sha256"),
        ("leaf_path", "leaf_sha256"),
        ("teacher_config_path", "teacher_config_sha256"),
    ):
        config[path_key], config[sha_key] = source.name, data.digest(source)
    output = tmp_path / "data"
    calls = []

    def worker(root, output_name, config, games):
        game = games[0]
        calls.append(game)
        if game == 0:
            return []  # deferred root, never claims a completed trajectory
        report = {"game": game, "rows": 1, "plies": 1}
        path = Path(output_name) / "games" / f"{game:06d}.json.receipt.json"
        data.atomic(path, data.encoded(report))
        return [report]

    monkeypatch.setattr(data, "_generate_group", worker)
    report = data.generate(tmp_path, output, config, pause_after_new_games=1)
    assert calls == [0, 1]
    assert report["status"] == "paused" and report["games"] == 1
    assert not (output / "generation-complete.json").exists()
    assert json.loads((output / "recovery-queue.json").read_text())["generation_sha256"]


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("real_process", [False, True])
def test_generator_interrupted_receipt_commit_resumes_without_duplicate_labels(
    tmp_path, monkeypatch, after_commit, real_process
):
    import gzip
    import json

    from open_shogi_training import evaluator_data as data

    config = {
        "seed": 9,
        "max_plies": 2,
        "sample_stride": 1,
        "deviation_stride": 10,
        "teacher_binary_sha256": "teacher",
        "teacher_config_sha256": "config",
        "teacher_depth": 12,
        "teacher_nodes": 2,
        "teacher_retry_nodes": 32,
        "recovery_policy": {"maximum_worker_restarts": 3},
    }

    class Replay:
        def __init__(self, *args):
            self.moves = []

        def ask(self, **request):
            if "reset" in request:
                self.moves = []
            if "movement" in request:
                self.moves.append(request["movement"])
            return {
                "sfen": f"s{len(self.moves)}",
                "terminal": "None",
                "successors": [
                    {
                        "move": "7g7f",
                        "sfen": f"s{len(self.moves) + 1}",
                        "terminal": "Draw",
                        "child_cp": 0,
                    }
                ],
            }

        def close(self):
            pass

    calls = []

    class Teacher:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def analyze(self, sfen, **kwargs):
            calls.append(sfen)
            return USISearchResult(
                "7g7f", (USICandidate(1, USIScore("cp", 1), ("7g7f",), 12, 12, 1),), 1
            )

    monkeypatch.setattr(data, "Replay", Replay)
    monkeypatch.setattr(data, "USIEngine", Teacher)
    monkeypatch.setattr(data, "teacher_config", lambda *args: None)
    original_atomic = data.atomic

    def interrupted(path, value):
        if path.name.endswith("receipt.json"):
            if after_commit:
                original_atomic(path, value)
            if real_process:
                import os

                os._exit(137)
            raise SystemExit("injected worker death at output commit")
        original_atomic(path, value)

    monkeypatch.setattr(data, "atomic", interrupted)
    if real_process:
        import multiprocessing

        process = multiprocessing.get_context("fork").Process(
            target=data._generate_group, args=(str(tmp_path), str(tmp_path), config, [0])
        )
        process.start()
        process.join(timeout=10)
        assert process.exitcode == 137
    else:
        with pytest.raises(SystemExit):
            data._generate_group(str(tmp_path), str(tmp_path), config, [0])
    monkeypatch.setattr(data, "atomic", original_atomic)
    result = data._generate_group(str(tmp_path), str(tmp_path), config, [0])
    raw = json.loads(gzip.decompress((tmp_path / "games/000000.json.gz").read_bytes()))
    assert result[0]["rows"] == 2
    assert raw["moves"] == ["7g7f", "7g7f"]
    assert [row["ply"] for row in raw["records"]] == [0, 1]
    assert calls == ([] if real_process else ["s0", "s1"])
    ledger = Ledger(tmp_path)
    assert ledger.summary()["accepted"] == 2
    assert ledger.summary()["retried"] == 0
    ledger.close()


def test_hard_queue_budget_is_persistent_and_refunds_only_finished_time(tmp_path):
    policy = {
        "hard_nodes": 32,
        "maximum_attempt_seconds": 60,
        "maximum_hard_attempts": 3,
        "maximum_hard_seconds": 120,
        "inherited_hard_attempts": 1,
        "inherited_hard_seconds": 20.0,
    }
    ledger = Ledger(tmp_path)
    first, _ = ledger.task({"game": 1})
    ledger.begin(first, 32, policy=policy)
    ledger.finish(first, "deferred", evidence={"elapsed_s": 2})
    second, _ = ledger.task({"game": 2})
    ledger.begin(second, 32, policy=policy)
    ledger.close()  # interrupted search retains all reserved 60 seconds
    ledger = Ledger(tmp_path)
    third, _ = ledger.task({"game": 3})
    with pytest.raises(DeferredTaskError):
        ledger.begin(third, 32, policy=policy)
    assert ledger.get(third)["status"] == "deferred"
    counters = dict(ledger.db.execute("SELECT name,value FROM counters"))
    assert counters == {"hard_attempts": 3, "hard_seconds": 82}
    ledger.close()


def test_sigkill_after_attempt_reservation_does_not_reset_budget(tmp_path):
    import multiprocessing
    import os
    import signal

    def killed_worker():
        ledger = Ledger(tmp_path)
        key, _ = ledger.task({"game": 211, "ply": 116})
        ledger.begin(key, 2_000_000)
        os.kill(os.getpid(), signal.SIGKILL)

    process = multiprocessing.get_context("fork").Process(target=killed_worker)
    process.start()
    process.join(timeout=10)
    assert process.exitcode == -signal.SIGKILL
    ledger = Ledger(tmp_path)
    key, saved = ledger.task({"game": 211, "ply": 116})
    assert saved["attempts"][0]["outcome"] == "interrupted"
    ledger.begin(key, 32_000_000)
    with pytest.raises(DeferredTaskError):
        ledger.begin(key, 32_000_000)
    assert len(ledger.get(key)["attempts"]) == 2
    ledger.close()


def test_atomic_retries_only_transient_io(tmp_path, monkeypatch):
    import errno

    from open_shogi_training import evaluator_data as data

    replace = data.os.replace
    calls = []

    def busy_once(source, target):
        calls.append(target)
        if len(calls) == 1:
            raise OSError(errno.EBUSY, "temporary busy")
        replace(source, target)

    monkeypatch.setattr(data.os, "replace", busy_once)
    data.atomic(tmp_path / "receipt", b"committed")
    assert (tmp_path / "receipt").read_bytes() == b"committed"
    assert len(calls) == 2

    def corrupt_io(*args):
        raise OSError(errno.EIO, "permanent IO failure")

    monkeypatch.setattr(data.os, "replace", corrupt_io)
    with pytest.raises(OSError) as failure:
        data.atomic(tmp_path / "receipt", b"rejected")
    assert failure.value.errno == errno.EIO
    assert (tmp_path / "receipt").read_bytes() == b"committed"


def test_shared_hard_and_task_wall_budget_survive_reopen(tmp_path):
    policy = {
        "hard_nodes": 32,
        "maximum_hard_attempts": 1,
        "maximum_hard_seconds": 150,
        "maximum_attempt_seconds": 75,
        "maximum_task_seconds": 150,
    }
    ledger = Ledger(tmp_path)
    key, _ = ledger.task({"game": 1})
    ledger.begin(key, 32, policy=policy)
    ledger.close()
    ledger = Ledger(tmp_path)
    other, _ = ledger.task({"game": 2})
    with pytest.raises(DeferredTaskError, match="hard queue"):
        ledger.begin(other, 32, policy=policy)
    assert ledger.get(other)["status"] == "deferred"
    assert (
        ledger.db.execute("SELECT value FROM counters WHERE name='hard_seconds'").fetchone()[0]
        == 75
    )
    normal, _ = ledger.task({"game": 3})
    ledger.begin(normal, 2, policy=policy)
    ledger.finish(normal, "hard", evidence={"elapsed_s": 80})
    with pytest.raises(DeferredTaskError, match="task wall"):
        ledger.begin(normal, 2, policy=policy)
    assert len(ledger.get(normal)["attempts"]) == 1
    ledger.close()


def test_worker_handshake_failure_uses_persistent_recovery_slots(tmp_path):
    from open_shogi_training.labeling.usi import USIProcessError

    config = {
        "teacher_binary_sha256": "teacher",
        "teacher_config_sha256": "config",
        "teacher_depth": 12,
        "teacher_nodes": 2,
        "teacher_retry_nodes": 32,
        "recovery_policy": {"maximum_worker_restarts": 3},
    }

    class Teacher:
        starts = 0

        def analyze(self, *args, **kwargs):
            raise USIProcessError("worker died")

        def close(self):
            pass

        def start(self):
            self.starts += 1
            if self.starts == 1:
                raise USIProcessError("transient handshake EOF")

    teacher = Teacher()
    with pytest.raises(DeferredTaskError):
        _ledger_observation(
            teacher,
            {"sfen": "sfen", "successors": [{"move": "7g7f"}], "terminal": "None"},
            root=tmp_path,
            output=tmp_path,
            config=config,
            game=1,
            ply=0,
            branch="root",
            moves=[],
            records=[],
        )
    ledger = Ledger(tmp_path)
    assert teacher.starts == 2
    assert (
        ledger.db.execute("SELECT value FROM counters WHERE name='worker_restarts'").fetchone()[0]
        == 2
    )
    assert ledger.summary()["hard"] == 1
    ledger.close()
