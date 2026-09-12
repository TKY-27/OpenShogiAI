"""Fresh-process regressions for detached runner standard-stream ownership."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from open_shogi_training import evaluator_run as runner

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("failure", ["closed_log", "pre_exec_ebadf"])
def test_launcher_does_not_swallow_failure_before_child_start(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    log = tmp_path / "stage.log"
    log.write_text("retained failure history\n")
    popen = subprocess.Popen
    calls = []

    def inject(*args, **kwargs):
        calls.append(kwargs["stdout"])
        if failure == "closed_log":
            kwargs["stdout"].close()
            return popen(*args, **kwargs)
        raise OSError(errno.EBADF, "injected immediately before process creation")

    monkeypatch.setattr(runner.subprocess, "Popen", inject)
    expected = ValueError if failure == "closed_log" else OSError
    with (tmp_path / "lease").open("w+b") as lease, pytest.raises(expected):
        runner._launch(
            tmp_path,
            [sys.executable, "-c", "raise AssertionError('must not start')"],
            log.name,
            lease.fileno(),
        )
    assert len(calls) == 1
    assert calls[0].closed
    assert log.read_text() == "retained failure history\n"


def test_launcher_reopens_append_log_and_retains_only_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    log = tmp_path / "stage.log"
    log.write_text("previous attempt\n")
    code = """
import json, os, sys
lease, unrelated = map(int, sys.argv[1:])
try:
    os.fstat(unrelated)
except OSError:
    leaked = False
else:
    leaked = True
print(json.dumps({"stdin": sys.stdin.read(), "lease": os.fstat(lease).st_ino,
                  "leaked": leaked}), flush=True)
print("child stderr", file=sys.stderr, flush=True)
"""
    with (tmp_path / "lease").open("w+b") as lease:
        unrelated = os.open(tmp_path / "unrelated", os.O_CREAT | os.O_RDWR, 0o600)
        os.set_inheritable(unrelated, True)
        try:
            for _ in range(2):
                process = runner._launch(
                    tmp_path,
                    [sys.executable, "-c", code, str(lease.fileno()), str(unrelated)],
                    log.name,
                    lease.fileno(),
                )
                # _launch has closed its log object. The child's independent
                # descriptor must still allow both stdout and stderr writes.
                assert process.wait(timeout=10) == 0
        finally:
            os.close(unrelated)
        expected_inode = os.fstat(lease.fileno()).st_ino
    lines = log.read_text().splitlines()
    assert lines[0] == "previous attempt"
    assert len(lines) == 5
    assert lines[2] == lines[4] == "child stderr"
    for line in (lines[1], lines[3]):
        assert json.loads(line) == {"stdin": "", "lease": expected_inode, "leaked": False}


@pytest.mark.parametrize("revoked_pty", [False, True], ids=["closed-stdin", "revoked-pty"])
def test_launcher_survives_invalid_inherited_stdin_in_new_process(tmp_path, revoked_pty):
    if revoked_pty and sys.platform != "darwin":
        pytest.skip("controlling-tty revocation reproduces the Darwin startup failure")
    script = textwrap.dedent(
        """
        import errno, fcntl, json, os, select, subprocess, sys, termios
        from pathlib import Path
        from open_shogi_training import evaluator_run as runner

        root = Path(sys.argv[1])
        runner.ROOT = root
        revoked = sys.argv[2] == "True"
        original_stdin = os.dup(0)
        descriptors = []
        leader = None
        try:
            if revoked:
                master, slave = os.openpty()
                ready_read, ready_write = os.pipe()
                release_read, release_write = os.pipe()
                descriptors.extend((master, slave, ready_read, ready_write,
                                    release_read, release_write))
                leader_code = (
                    "import fcntl,os,sys,termios;"
                    "fcntl.ioctl(0,termios.TIOCSCTTY,0);"
                    "os.write(int(sys.argv[1]),b'1');"
                    "os.read(int(sys.argv[2]),1)"
                )
                leader = subprocess.Popen(
                    [sys.executable, "-c", leader_code,
                     str(ready_write), str(release_read)],
                    stdin=slave, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    pass_fds=(ready_write, release_read), start_new_session=True,
                )
                assert select.select([ready_read], [], [], 5)[0], "tty owner did not start"
                assert os.read(ready_read, 1) == b"1"
                os.dup2(slave, 0)
                os.fstat(0)  # Healthy at inheritance, before the tty owner exits.
                os.write(release_write, b"1")
                _, leader_error = leader.communicate(timeout=5)
                assert leader.returncode == 0, leader_error
                assert fcntl.fcntl(0, fcntl.F_GETFD) >= 0
            else:
                os.close(0)
            try:
                os.fstat(0)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                raise AssertionError("inherited stdin was not invalidated")

            # This child receives exactly the old implicit stdin inheritance.
            if revoked:
                control = subprocess.run(
                    [sys.executable, "-c", "print('unexpected stage start')"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True, timeout=5,
                )
                assert control.returncode == 1
                assert b"init_sys_streams" in control.stderr
                assert b"OSError: [Errno 9] Bad file descriptor" in control.stderr
                assert b"<no Python frame>" in control.stderr
                assert not control.stdout
                (root / "control.log").write_bytes(control.stderr)

            with (root / "lease").open("w+b") as lease:
                # A closed fd 0 may be reused by open(lease). Close it again
                # after duplicating the lease so _launch sees a truly closed 0.
                lease_fd = fcntl.fcntl(lease.fileno(), fcntl.F_DUPFD_CLOEXEC, 32)
                descriptors.append(lease_fd)
            if not revoked:
                try:
                    os.close(0)
                except OSError as error:
                    assert error.errno == errno.EBADF
            child = runner._launch(
                root,
                [sys.executable, "-c",
                 "import sys; assert sys.stdin.read() == ''; print('stage reached')"],
                "stage.log", lease_fd,
            )
            assert child.wait(timeout=5) == 0
            assert (root / "stage.log").read_text() == "stage reached\\n"
            print(json.dumps({"revoked": revoked, "child_exit": child.returncode}))
        finally:
            os.dup2(original_stdin, 0)
            os.close(original_stdin)
            if leader is not None and leader.poll() is None:
                leader.kill()
                leader.communicate(timeout=5)
            for descriptor in descriptors:
                os.close(descriptor)
        """
    )
    environment = dict(os.environ, PYTHONPATH=str(PROJECT / "training"))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(revoked_pty)],
        cwd=PROJECT,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"revoked": revoked_pty, "child_exit": 0}
