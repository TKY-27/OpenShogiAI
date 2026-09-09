from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import open_shogi_training.labeling.execution as execution_module
import open_shogi_training.labeling.usi as usi_module
import pytest
from open_shogi_training.labeling.execution import (
    RuntimeTreeSnapshot,
    RuntimeTreeSnapshotError,
)
from open_shogi_training.labeling.usi import (
    USIEngine,
    USIProcessError,
    USIProtocolError,
    USIResourceError,
    USIRetryError,
    parse_info_line,
)

from .helpers import make_fake_project


def test_rss_monitor_preserves_a_process_exit_racing_with_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitingProcess:
        pid = 123
        returncode: int | None = None
        wait_calls = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            assert timeout == usi_module._RSS_EXIT_GRACE_SECONDS
            self.wait_calls += 1
            self.returncode = 7
            return self.returncode

    process = ExitingProcess()
    monkeypatch.setattr(
        usi_module,
        "_teacher_process_tree_rss_bytes",
        lambda _pid, *, retained_identities: (None, {}),
    )
    monitor = usi_module._RssMonitor(
        process,
        process_start_identity="process-start",
        limit_bytes=1,
        poll_ms=10,
    )

    monitor._run()
    monitor.check()

    assert process.wait_calls == 1
    assert process.returncode == 7


@pytest.mark.parametrize("unavailable", [True, False])
def test_live_teacher_rss_monitor_fails_closed_and_kills_on_unavailable_or_over_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unavailable: bool,
) -> None:
    config, _, _ = make_fake_project(tmp_path)

    def measurement(leader_pid: int, *, retained_identities=None):
        assert retained_identities is not None
        if unavailable:
            return None, {}
        identity = usi_module._read_process_start_identity(leader_pid)
        assert identity is not None
        return (config.benchmark.max_peak_rss_mib + 1) * 1024 * 1024, {leader_pid: identity}

    monkeypatch.setattr(usi_module, "_teacher_process_tree_rss_bytes", measurement)
    expected = "unavailable" if unavailable else "exceeded its hard limit"

    for _ in range(3):
        engine = USIEngine(config, tmp_path)
        with pytest.raises(USIResourceError, match=expected):
            engine.start()

        assert engine.pid is None


def test_teacher_rss_retains_an_escaped_reparented_descendant_and_rejects_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ps_output = b"100 1 100 10\n200 1 200 20\n300 1 300 30\n"

    class Completed:
        returncode = 0
        stdout = ps_output

    monkeypatch.setattr(usi_module.subprocess, "run", lambda *args, **kwargs: Completed())
    identities = {100: "identity-100", 200: "identity-200", 300: "identity-300-new"}
    monkeypatch.setattr(
        usi_module,
        "_read_process_start_identity",
        lambda process_id: identities.get(process_id),
    )

    rss, identities = usi_module._teacher_process_tree_rss_bytes(
        100,
        retained_identities={
            100: "identity-100",
            200: "identity-200",
            300: "identity-300-old",
        },
    )

    assert rss == 30 * 1024
    assert identities == {
        100: "identity-100",
        200: "identity-200",
    }


def test_process_identity_binds_pid_group_session_and_high_resolution_start() -> None:
    identity = usi_module._read_process_start_identity(os.getpid())

    assert identity is not None
    fields = identity.split(":")
    assert fields[0] in {"darwin", "linux"}
    assert int(fields[1]) == os.getpid()
    assert int(fields[2]) == os.getpgid(0)
    assert int(fields[3]) == os.getsid(0)
    if fields[0] == "darwin":
        assert len(fields) == 6
        assert len(fields[5]) == 6


def test_adapter_runs_full_handshake_options_newgame_nodes_multipv_and_quit(
    tmp_path: Path,
) -> None:
    command_log = tmp_path / "commands.log"
    config, _, _ = make_fake_project(
        tmp_path,
        mode="stderr",
        extra_arguments=["--command-log", str(command_log)],
    )
    engine = USIEngine(config, tmp_path)

    result = engine.analyze_with_retry("state b - 1")
    assert result.bestmove == "7g7f"
    assert result.primary.score.as_dict() == {"kind": "cp", "value": 42}
    assert result.candidates[1].score.as_dict() == {"kind": "mate", "value": -3}
    assert [candidate.multipv for candidate in result.candidates] == [1, 2, 3]
    assert len(engine.stderr_tail.encode()) <= config.protocol_limits.max_stderr_bytes
    engine.close()

    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0] == "usi"
    assert "isready" in commands
    assert "usinewgame" in commands
    assert "position sfen state b - 1" in commands
    assert "go nodes 20" in commands
    assert commands[-1] == "quit"
    assert "setoption name MultiPV value 3" in commands
    assert "setoption name Threads value 1" in commands
    assert "setoption name USI_Hash value 16" in commands
    assert not list((tmp_path / "bin").glob(".open-shogi-exec.*"))


def test_teacher_execution_uses_verified_private_copy_and_rejects_source_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, executable, _ = make_fake_project(tmp_path)
    marker = tmp_path / "unverified-executable-ran"
    real_popen = usi_module.subprocess.Popen
    swapped = False

    def swap_before_process(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        if not swapped:
            swapped = True
            executable.write_text(
                f"#!{sys.executable}\nfrom pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(usi_module.subprocess, "Popen", swap_before_process)
    engine = USIEngine(config, tmp_path)

    with pytest.raises(USIProcessError, match="identity drifted during process creation"):
        engine.start()

    assert not marker.exists()
    assert engine.pid is None
    assert not list((tmp_path / "bin").glob(".open-shogi-exec.*"))


def test_teacher_snapshot_creation_rejects_source_path_swap_while_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, executable, _ = make_fake_project(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(executable.read_bytes())
    replacement.chmod(0o755)
    real_read = usi_module.os.read
    swapped = False

    def swap_after_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            os.replace(replacement, executable)
        return chunk

    monkeypatch.setattr(usi_module.os, "read", swap_after_read)
    engine = USIEngine(config, tmp_path)

    with pytest.raises(USIProcessError, match="executable changed"):
        engine.start()

    assert engine.pid is None
    assert not list((tmp_path / "bin").glob(".open-shogi-exec.*"))


def test_runtime_tree_rejects_eval_source_swap_during_same_descriptor_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = make_fake_project(tmp_path)
    source = tmp_path / config.eval_files[0].path
    replacement = tmp_path / "replacement-eval.bin"
    replacement.write_bytes(source.read_bytes())
    real_pread = execution_module.os.pread
    swapped = False

    def swap_after_read(descriptor: int, count: int, offset: int) -> bytes:
        nonlocal swapped
        chunk = real_pread(descriptor, count, offset)
        if chunk and not swapped:
            swapped = True
            os.replace(replacement, source)
        return chunk

    monkeypatch.setattr(execution_module.os, "pread", swap_after_read)

    with pytest.raises(
        RuntimeTreeSnapshotError,
        match=r"runtime source (?:changed while copied|path changed)",
    ):
        RuntimeTreeSnapshot.create(
            project_root=tmp_path,
            working_directory=config.cwd,
            files=tuple((item.path, item.sha256, 1024) for item in config.eval_files),
            storage_directory=tmp_path / "local/teacher/runtime-snapshots",
        )


def test_runtime_tree_detects_private_eval_snapshot_mutation(tmp_path: Path) -> None:
    config, _, _ = make_fake_project(tmp_path)
    runtime = RuntimeTreeSnapshot.create(
        project_root=tmp_path,
        working_directory=config.cwd,
        files=tuple((item.path, item.sha256, 1024) for item in config.eval_files),
        storage_directory=tmp_path / "local/teacher/runtime-snapshots",
    )
    snapshot_path = runtime._files[0][0]
    snapshot_path.chmod(0o600)
    snapshot_path.write_bytes(b"mutated")

    with pytest.raises(RuntimeTreeSnapshotError, match="snapshot file changed"):
        runtime.assert_unchanged()
    runtime.close()


def test_runtime_tree_close_never_deletes_a_relinked_foreign_directory(tmp_path: Path) -> None:
    config, _, _ = make_fake_project(tmp_path)
    runtime = RuntimeTreeSnapshot.create(
        project_root=tmp_path,
        working_directory=config.cwd,
        files=tuple((item.path, item.sha256, 1024) for item in config.eval_files),
        storage_directory=tmp_path / "local/teacher/runtime-snapshots",
    )
    original = runtime.private_directory.with_name("owned-runtime-tree")
    runtime.private_directory.rename(original)
    runtime.private_directory.mkdir()
    marker = runtime.private_directory / "foreign-do-not-delete"
    marker.write_bytes(b"foreign")

    with pytest.raises(RuntimeTreeSnapshotError, match="changed before retirement"):
        runtime.close()

    assert marker.read_bytes() == b"foreign"
    assert original.is_dir()


def test_runtime_tree_close_releases_descriptors_after_unseal_failure_without_deleting_foreign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = make_fake_project(tmp_path)
    runtime = RuntimeTreeSnapshot.create(
        project_root=tmp_path,
        working_directory=config.cwd,
        files=tuple((item.path, item.sha256, 1024) for item in config.eval_files),
        storage_directory=tmp_path / "local/teacher/runtime-snapshots",
    )
    descriptors = [descriptor for _, descriptor, _ in runtime._files]
    original = runtime.private_directory.with_name("owned-runtime-after-unseal-failure")
    runtime.private_directory.rename(original)
    runtime.private_directory.mkdir()
    marker = runtime.private_directory / "foreign-do-not-delete"
    marker.write_bytes(b"foreign")

    monkeypatch.setattr(
        runtime,
        "unseal",
        lambda: (_ for _ in ()).throw(OSError("forced unseal failure")),
    )
    with pytest.raises(OSError, match="forced unseal failure"):
        runtime.close()

    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert marker.read_bytes() == b"foreign"
    assert original.is_dir()


def test_runtime_tree_successful_close_does_not_accumulate_retired_directories(
    tmp_path: Path,
) -> None:
    config, _, _ = make_fake_project(tmp_path)
    storage = tmp_path / "local/teacher/runtime-snapshots"
    for _ in range(3):
        runtime = RuntimeTreeSnapshot.create(
            project_root=tmp_path,
            working_directory=config.cwd,
            files=tuple((item.path, item.sha256, 1024) for item in config.eval_files),
            storage_directory=storage,
        )
        runtime.close()

    assert tuple((storage / ".open-shogi-retired").iterdir()) == ()


def test_teacher_runs_from_private_eval_tree_and_ignores_later_source_mutation(
    tmp_path: Path,
) -> None:
    config, _, _ = make_fake_project(tmp_path)
    engine = USIEngine(config, tmp_path)
    engine.start()
    runtime = engine._runtime_snapshot
    assert runtime is not None
    assert runtime.cwd.is_relative_to(runtime.private_directory)
    for item in config.eval_files:
        assert (runtime.private_directory / "tree" / item.path).read_bytes() == (
            tmp_path / item.path
        ).read_bytes()
    (tmp_path / config.eval_files[0].path).write_bytes(b"changed-after-start")

    assert engine.analyze("state b - 1").bestmove == "7g7f"
    engine.close()


def test_crash_is_restarted_and_retried_once(tmp_path: Path) -> None:
    marker = tmp_path / "crashed.marker"
    command_log = tmp_path / "commands.log"
    config, _, _ = make_fake_project(
        tmp_path,
        mode="crash-once",
        extra_arguments=[
            "--marker",
            str(marker),
            "--command-log",
            str(command_log),
        ],
        max_retries=1,
    )
    engine = USIEngine(config, tmp_path)

    result = engine.analyze_with_retry("state b - 1")
    engine.close()

    assert result.bestmove == "7g7f"
    assert engine.restart_count == 1
    assert command_log.read_text(encoding="utf-8").splitlines().count("usi") == 2


def test_malformed_output_and_oversized_stdout_are_rejected(tmp_path: Path) -> None:
    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    config, _, _ = make_fake_project(malformed_root, mode="malformed", max_retries=1)
    with (
        USIEngine(config, malformed_root) as engine,
        pytest.raises(USIProtocolError, match="score kind"),
    ):
        engine.analyze_with_retry("state b - 1")

    overflow_root = tmp_path / "overflow"
    overflow_root.mkdir()
    config, _, _ = make_fake_project(
        overflow_root,
        mode="overflow",
        max_retries=0,
        max_stdout_line_bytes=512,
    )
    with (
        USIEngine(config, overflow_root) as engine,
        pytest.raises(USIProtocolError, match="line exceeds"),
    ):
        engine.analyze_with_retry("state b - 1")


def test_timeout_stops_and_kills_the_entire_teacher_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    config, _, _ = make_fake_project(
        tmp_path,
        mode="timeout-child",
        extra_arguments=["--child-pid", str(child_pid_path)],
        max_retries=0,
        search_ms=100,
    )
    engine = USIEngine(config, tmp_path)

    with pytest.raises(USIRetryError, match="after 1 attempts"):
        engine.analyze_with_retry("state b - 1")
    assert engine.pid is None
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while _process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_exists(child_pid), "teacher descendant survived process-group cleanup"


def test_score_parser_keeps_cp_and_mate_distinct_and_discards_bounds() -> None:
    cp = parse_info_line(
        "info depth 5 seldepth 7 nodes 20 multipv 1 score cp -12 pv 7g7f",
        expected_multipv=1,
    )
    mate = parse_info_line(
        "info depth 5 seldepth 7 nodes 20 multipv 1 score mate 3 pv 7g7f",
        expected_multipv=1,
    )
    assert cp is not None and cp.score.as_dict() == {"kind": "cp", "value": -12}
    assert mate is not None and mate.score.as_dict() == {"kind": "mate", "value": 3}
    bounded = parse_info_line(
        "info depth 5 seldepth 7 nodes 20 multipv 1 score cp 1 lowerbound nps 20 time 2 pv 7g7f",
        expected_multipv=1,
    )
    assert bounded is None


def test_info_parser_accepts_bounded_multi_move_refutation_and_currline_fields() -> None:
    candidate = parse_info_line(
        "info currline 2 7g7f 3c3d refutation 2g2f 8c8d "
        "depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f 3c3d",
        expected_multipv=1,
    )

    assert candidate is not None
    assert candidate.pv == ("7g7f", "3c3d")

    without_cpu = parse_info_line(
        "info currline 7g7f 3c3d depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f 3c3d",
        expected_multipv=1,
    )
    assert without_cpu is not None
    assert without_cpu.pv == ("7g7f", "3c3d")


@pytest.mark.parametrize(
    "line",
    [
        "info currline 0 7g7f depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f",
        "info currline cpu 7g7f depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f",
        "info refutation depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f",
        "info refutation bad depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f",
        "info currmove bad depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f",
    ],
)
def test_info_parser_rejects_malformed_variable_move_fields(line: str) -> None:
    with pytest.raises(USIProtocolError, match=r"malformed|USI move|integer"):
        parse_info_line(line, expected_multipv=1)


def test_info_parser_rejects_an_oversized_variable_move_field() -> None:
    moves = " ".join(["7g7f"] * 1_025)

    with pytest.raises(USIProtocolError, match="move-count bound"):
        parse_info_line(
            f"info refutation {moves} depth 5 seldepth 7 nodes 20 score cp 12 pv 7g7f",
            expected_multipv=1,
        )


def test_teacher_signal_refuses_a_reused_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(usi_module, "_read_process_start_identity", lambda _pid: "new")
    monkeypatch.setattr(
        usi_module.os,
        "killpg",
        lambda process_group, requested_signal: signals.append((process_group, requested_signal)),
    )

    usi_module._signal_process_group(
        43_210,
        signal.SIGTERM,
        expected_start_identity="old",
    )

    assert signals == []


def test_valid_bounded_interim_lines_are_ignored_but_final_exact_ranks_are_required(
    tmp_path: Path,
) -> None:
    success_root = tmp_path / "success"
    success_root.mkdir()
    config, _, _ = make_fake_project(success_root, mode="bounded-then-good")
    with USIEngine(config, success_root) as engine:
        result = engine.analyze_with_retry("state b - 1")
    assert result.primary.score.as_dict() == {"kind": "cp", "value": 42}

    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    config, _, _ = make_fake_project(failure_root, mode="bounded-only")
    with (
        USIEngine(config, failure_root) as engine,
        pytest.raises(USIProtocolError, match="no complete unbounded"),
    ):
        engine.analyze_with_retry("state b - 1")


def test_later_lower_depth_exact_lines_replace_stale_higher_depth_ranks(
    tmp_path: Path,
) -> None:
    config, _, _ = make_fake_project(tmp_path, mode="later-lower-exact")

    with USIEngine(config, tmp_path) as engine:
        result = engine.analyze_with_retry("state b - 1")

    assert result.bestmove == "2g2f"
    assert [candidate.pv[0] for candidate in result.candidates] == ["2g2f", "7g7f", "5g5f"]
    assert [candidate.depth for candidate in result.candidates] == [7, 6, 6]
    assert [candidate.score.value for candidate in result.candidates] == [55, 45, 35]


@pytest.mark.parametrize(("mode", "expected_ranks"), [("one-rank", 1), ("two-ranks", 2)])
def test_forced_positions_accept_available_contiguous_multipv_prefix(
    tmp_path: Path,
    mode: str,
    expected_ranks: int,
) -> None:
    config, _, _ = make_fake_project(tmp_path, mode=mode)

    with USIEngine(config, tmp_path) as engine:
        result = engine.analyze_with_retry("state b - 1")

    assert len(result.candidates) == expected_ranks
    assert [candidate.multipv for candidate in result.candidates] == list(
        range(1, expected_ranks + 1)
    )


def test_noncontiguous_multipv_ranks_are_rejected(tmp_path: Path) -> None:
    config, _, _ = make_fake_project(tmp_path, mode="gapped-ranks")

    with (
        USIEngine(config, tmp_path) as engine,
        pytest.raises(USIProtocolError, match="contain a gap"),
    ):
        engine.analyze_with_retry("state b - 1")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("isolate", [True, False])
def test_teacher_group_isolation_is_explicit_and_default_remains_private(tmp_path, isolate):
    config, _, _ = make_fake_project(tmp_path)
    engine = (
        USIEngine(config, tmp_path)
        if isolate
        else USIEngine(config, tmp_path, isolate_process_group=False)
    )
    try:
        engine.start()
        assert os.getpgid(engine.pid) == (engine.pid if isolate else os.getpgid(0))
        assert engine.analyze_with_retry("state b - 1").bestmove == "7g7f"
    finally:
        engine.close()


def test_inherited_teacher_close_never_signals_the_stage_group(tmp_path, monkeypatch):
    config, _, _ = make_fake_project(tmp_path)
    engine = USIEngine(config, tmp_path, isolate_process_group=False)
    engine.start()
    pid = engine.pid
    monkeypatch.setattr(usi_module.os, "killpg", lambda *_args: pytest.fail("parent group killed"))
    monkeypatch.setattr(engine, "_send", lambda _line: None)  # Force the close escalation path.
    engine.close()
    assert engine.pid is None and not _process_exists(pid)


def test_inherited_teacher_rss_stop_signals_only_the_verified_teacher(tmp_path, monkeypatch):
    config, _, _ = make_fake_project(tmp_path)
    engine = USIEngine(config, tmp_path, isolate_process_group=False)
    engine.start()
    process = engine._process
    identity = engine._process_start_identity
    monitor = engine._rss_monitor
    # Stop just its monitor thread so this test invokes the hard-limit path deterministically.
    monitor.close()
    sent = []
    real_kill = os.kill

    def kill(pid, sig):
        sent.append(pid)
        real_kill(pid, sig)

    monkeypatch.setattr(usi_module.os, "kill", kill)
    monkeypatch.setattr(usi_module.os, "killpg", lambda *_args: pytest.fail("parent group killed"))
    try:
        monitor._fail("forced RSS limit", {process.pid: identity, os.getpid(): "parent"})
        process.wait(timeout=3)
        with pytest.raises(USIResourceError, match="forced RSS"):
            monitor.check()
        assert sent == [process.pid]
    finally:
        engine.close()


def test_inherited_teacher_signal_refuses_a_reused_pid(monkeypatch):
    monkeypatch.setattr(usi_module, "_read_process_start_identity", lambda _pid: "replacement")
    monkeypatch.setattr(usi_module.os, "kill", lambda *_args: pytest.fail("reused PID killed"))
    monkeypatch.setattr(usi_module.os, "killpg", lambda *_args: pytest.fail("group killed"))
    usi_module._signal_teacher_process(
        43210, signal.SIGKILL, expected_start_identity="original", isolate_process_group=False
    )
