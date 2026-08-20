"""Bounded, restartable USI subprocess adapter for black-box teachers."""

from __future__ import annotations

import contextlib
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from open_shogi_training.labeling.config import TeacherConfig, resolve_config_path
from open_shogi_training.labeling.execution import (
    ExecutableSnapshot,
    ExecutableSnapshotError,
    RuntimeTreeSnapshot,
    RuntimeTreeSnapshotError,
)
from open_shogi_training.labeling.process_identity import read_process_identity

_USI_MOVE_RE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")
_EOF: Final = object()
_INTEGER_INFO_KEYS = frozenset(
    {
        "depth",
        "seldepth",
        "time",
        "nodes",
        "multipv",
        "currmovenumber",
        "hashfull",
        "nps",
        "tbhits",
        "cpuload",
    }
)
_MOVE_INFO_KEYS = frozenset({"currmove", "refutation", "currline"})
_INFO_FIELD_KEYS = _INTEGER_INFO_KEYS | _MOVE_INFO_KEYS | frozenset({"score", "pv", "string"})
_MAX_INFO_SEQUENCE_MOVES: Final = 1_024
_MAX_TEACHER_BINARY_BYTES: Final = 512 * 1024 * 1024
# A process can exit between the process-table snapshot and Popen.poll().
# Keep the grace bounded, then still fail closed if the teacher remains alive.
_RSS_EXIT_GRACE_SECONDS: Final = 0.05


class USIError(RuntimeError):
    """Base error carrying a bounded teacher stderr tail."""

    category = "usi_error"

    def __init__(self, message: str, *, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


class USITimeoutError(USIError):
    category = "timeout"


class USIProcessError(USIError):
    category = "process"


class USIResourceError(USIProcessError):
    category = "memory_limit"


class USIProtocolError(USIError):
    category = "protocol"


class USIRetryError(USIError):
    category = "retries_exhausted"

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_category: str,
        stderr_tail: str,
    ) -> None:
        super().__init__(message, stderr_tail=stderr_tail)
        self.attempts = attempts
        self.last_category = last_category


@dataclass(frozen=True, slots=True)
class USIScore:
    kind: str
    value: int

    def as_dict(self) -> dict[str, int | str]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class USICandidate:
    multipv: int
    score: USIScore
    pv: tuple[str, ...]
    depth: int
    seldepth: int
    nodes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "multipv": self.multipv,
            "score": self.score.as_dict(),
            "pv": list(self.pv),
            "depth": self.depth,
            "seldepth": self.seldepth,
            "nodes": self.nodes,
        }


@dataclass(frozen=True, slots=True)
class USISearchResult:
    bestmove: str
    candidates: tuple[USICandidate, ...]
    elapsed_ms: int

    @property
    def primary(self) -> USICandidate:
        return self.candidates[0]


@dataclass(frozen=True, slots=True)
class USIIdentity:
    name: str
    author: str | None
    declared_options: tuple[str, ...]


class _StdoutReader:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        max_line_bytes: int,
        max_queue_lines: int,
    ) -> None:
        self._stream = stream
        self._max_line_bytes = max_line_bytes
        self._queue: queue.Queue[str | object] = queue.Queue(max_queue_lines)
        self._error: str | None = None
        self._error_lock = threading.Lock()
        self._eof = threading.Event()
        self._thread = threading.Thread(target=self._run, name="usi-stdout", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def read(self, timeout: float) -> str:
        self._raise_reader_error()
        try:
            value = self._queue.get(timeout=max(timeout, 0.001))
        except queue.Empty:
            self._raise_reader_error()
            if self._eof.is_set():
                raise USIProcessError("teacher stdout closed") from None
            raise
        self._raise_reader_error()
        if value is _EOF:
            raise USIProcessError("teacher stdout closed")
        if not isinstance(value, str):
            raise AssertionError("stdout queue contained an invalid value")
        return value

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            while True:
                raw = self._stream.readline(self._max_line_bytes + 1)
                if not raw:
                    self._eof.set()
                    self._offer(_EOF)
                    return
                if len(raw) > self._max_line_bytes:
                    self._set_error(f"teacher stdout line exceeds {self._max_line_bytes} bytes")
                    continue
                if not raw.endswith(b"\n"):
                    self._set_error("teacher stdout ended with an incomplete line")
                    continue
                raw = raw[:-1]
                if raw.endswith(b"\r"):
                    raw = raw[:-1]
                try:
                    line = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    self._set_error("teacher stdout is not valid UTF-8")
                    continue
                self._offer(line)
        except (OSError, ValueError) as error:
            self._set_error(f"cannot drain teacher stdout: {error}")
            self._eof.set()
            self._offer(_EOF)

    def _offer(self, value: str | object) -> None:
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            self._set_error("bounded teacher stdout queue overflowed")

    def _set_error(self, message: str) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = message

    def _raise_reader_error(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise USIProtocolError(error)


class _StderrTail:
    def __init__(self, stream: BinaryIO, *, max_bytes: int) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._tail = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="usi-stderr", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def text(self) -> str:
        with self._lock:
            raw = bytes(self._tail)
        return raw.decode("utf-8", errors="replace")

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            while chunk := self._stream.read(4_096):
                with self._lock:
                    self._tail.extend(chunk)
                    overflow = len(self._tail) - self._max_bytes
                    if overflow > 0:
                        del self._tail[:overflow]
        except (OSError, ValueError):
            return


class _RssMonitor:
    """Continuously enforce the configured teacher process-tree RSS ceiling."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        process_start_identity: str,
        limit_bytes: int,
        poll_ms: int,
    ) -> None:
        self._process = process
        self._process_start_identity = process_start_identity
        self._limit_bytes = limit_bytes
        self._poll_seconds = poll_ms / 1_000
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._failure: str | None = None
        self._peak = 0
        self._observed_identities = {process.pid: process_start_identity}
        self._thread = threading.Thread(target=self._run, name="usi-rss", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._poll_seconds * 4))
        # A descendant may create a new session and later be reparented. Keep
        # ownership of every sampled start identity so normal shutdown cannot
        # leak such a process after the teacher leader has exited.
        self._kill_observed_descendants()

    def check(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise USIResourceError(failure)

    def _run(self) -> None:
        while not self._stop.is_set() and self._process.poll() is None:
            try:
                rss_bytes, process_identities = _teacher_process_tree_rss_bytes(
                    self._process.pid,
                    retained_identities=self._observed_identities,
                )
            except USIResourceError as error:
                self._fail(str(error), self._observed_identities)
                return
            if rss_bytes is None or not process_identities:
                if self._process.poll() is None:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._process.wait(timeout=_RSS_EXIT_GRACE_SECONDS)
                    if self._process.poll() is None:
                        self._fail(
                            "teacher RSS measurement became unavailable while the "
                            "process was alive",
                            self._observed_identities,
                        )
                return
            self._observed_identities.update(process_identities)
            with self._lock:
                self._peak = max(self._peak, rss_bytes)
            if rss_bytes > self._limit_bytes:
                self._fail(
                    "teacher process-tree RSS exceeded its hard limit: "
                    f"observed={rss_bytes}, limit={self._limit_bytes}",
                    self._observed_identities,
                )
                return
            self._stop.wait(self._poll_seconds)

    def _fail(self, message: str, process_identities: dict[int, str]) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = message
        if self._process.poll() is None:
            _signal_process_group(
                self._process.pid,
                signal.SIGKILL,
                expected_start_identity=self._process_start_identity,
            )
        for process_id, expected_identity in process_identities.items():
            if process_id in {self._process.pid, os.getpid()}:
                continue
            if process_id > 1:
                if _read_process_start_identity(process_id) != expected_identity:
                    continue
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(process_id, signal.SIGKILL)

    def _kill_observed_descendants(self) -> None:
        for process_id, expected_identity in tuple(self._observed_identities.items()):
            if process_id in {self._process.pid, os.getpid()} or process_id <= 1:
                continue
            if _read_process_start_identity(process_id) != expected_identity:
                continue
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(process_id, signal.SIGKILL)


class USIEngine:
    """One persistent USI process with strict protocol and process-group cleanup."""

    def __init__(self, config: TeacherConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve(strict=True)
        self.executable = resolve_config_path(
            self.project_root, config.executable, field="teacher.executable"
        )
        self.cwd = resolve_config_path(self.project_root, config.cwd, field="teacher.cwd")
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: _StdoutReader | None = None
        self._stderr: _StderrTail | None = None
        self._process_group: int | None = None
        self._process_start_identity: str | None = None
        self._snapshot: ExecutableSnapshot | None = None
        self._runtime_snapshot: RuntimeTreeSnapshot | None = None
        self._rss_monitor: _RssMonitor | None = None
        self.identity: USIIdentity | None = None
        self.restart_count = 0

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def stderr_tail(self) -> str:
        return "" if self._stderr is None else self._stderr.text()

    def __enter__(self) -> USIEngine:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> USIIdentity:
        if self._process is not None:
            if self._process.poll() is None and self.identity is not None:
                return self.identity
            self.close()
        if not self.cwd.is_dir() or self.cwd.is_symlink():
            raise USIProcessError(f"teacher cwd is not a non-symlink directory: {self.cwd}")
        runtime: RuntimeTreeSnapshot | None = None
        try:
            runtime = RuntimeTreeSnapshot.create(
                project_root=self.project_root,
                working_directory=self.config.cwd,
                files=tuple(
                    (item.path, item.sha256, 2 * 1024 * 1024 * 1024)
                    for item in self.config.eval_files
                ),
                storage_directory=self.project_root / "local/teacher/runtime-snapshots",
            )
            snapshot = ExecutableSnapshot.create(
                self.executable,
                temporary_directory=runtime.executable_storage,
                max_bytes=_MAX_TEACHER_BINARY_BYTES,
                expected_sha256=self.config.runtime_binary_sha256,
            )
            runtime.seal()
        except (ExecutableSnapshotError, RuntimeTreeSnapshotError) as error:
            if runtime is not None:
                runtime.close()
            raise USIProcessError(f"cannot pin teacher runtime: {error}") from error
        command = [str(self.executable), *self.config.arguments]
        try:
            process = subprocess.Popen(
                command,
                executable=snapshot.executable_path,
                cwd=runtime.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                pass_fds=snapshot.pass_fds(),
                start_new_session=True,
            )
        except BaseException as error:
            runtime.unseal()
            snapshot.close()
            runtime.close()
            if isinstance(error, OSError):
                raise USIProcessError(f"cannot start teacher: {error}") from error
            raise
        if process.stdin is None or process.stdout is None or process.stderr is None:
            _terminate_unowned_process(process)
            runtime.unseal()
            snapshot.close()
            runtime.close()
            raise USIProcessError("teacher pipes were not created")
        process_start_identity = _read_process_start_identity(process.pid)
        if process_start_identity is None:
            _terminate_unowned_process(process)
            runtime.unseal()
            snapshot.close()
            runtime.close()
            raise USIProcessError("cannot bind teacher process start identity")
        try:
            snapshot.assert_snapshot_unchanged()
            snapshot.assert_source_unchanged()
            runtime.assert_unchanged()
        except (ExecutableSnapshotError, RuntimeTreeSnapshotError) as error:
            _terminate_unowned_process(process)
            runtime.unseal()
            snapshot.close()
            runtime.close()
            raise USIProcessError(
                f"teacher runtime identity drifted during process creation: {error}"
            ) from error
        self._process = process
        self._process_group = process.pid
        self._process_start_identity = process_start_identity
        self._snapshot = snapshot
        self._runtime_snapshot = runtime
        self._stdout = _StdoutReader(
            process.stdout,
            max_line_bytes=self.config.protocol_limits.max_stdout_line_bytes,
            max_queue_lines=self.config.protocol_limits.max_stdout_queue_lines,
        )
        self._stderr = _StderrTail(
            process.stderr,
            max_bytes=self.config.protocol_limits.max_stderr_bytes,
        )
        self._stdout.start()
        self._stderr.start()
        self._rss_monitor = _RssMonitor(
            process,
            process_start_identity=process_start_identity,
            limit_bytes=self.config.benchmark.max_peak_rss_mib * 1024 * 1024,
            poll_ms=self.config.benchmark.rss_poll_ms,
        )
        self._rss_monitor.start()
        try:
            self._send("usi")
            identity = self._read_usi_identity()
            missing = sorted(set(self.config.option_map) - set(identity.declared_options))
            if missing:
                raise self._protocol_error(f"teacher did not declare configured options: {missing}")
            for name, value in sorted(self.config.option_map.items()):
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                self._send(f"setoption name {name} value {rendered}")
            self._ready()
            self._assert_executable_unchanged()
            self.identity = identity
            return identity
        except BaseException:
            self.close()
            raise

    def new_game(self) -> None:
        self._ensure_started()
        self._send("usinewgame")
        self._ready()

    def analyze(self, sfen: str, *, nodes: int | None = None) -> USISearchResult:
        """Analyze one independent SFEN, issuing stop before a timeout failure."""

        self._ensure_started()
        requested_nodes = self.config.nodes if nodes is None else nodes
        if isinstance(requested_nodes, bool) or not isinstance(requested_nodes, int):
            raise ValueError("nodes must be an integer")
        if not 1 <= requested_nodes <= 10_000_000_000:
            raise ValueError("nodes is outside the supported bound")
        _validate_sfen_command(sfen)
        self.new_game()
        self._send(f"position sfen {sfen}")
        started = time.monotonic()
        self._send(f"go nodes {requested_nodes}")
        deadline = started + self.config.timeouts.search_ms / 1_000
        candidates: dict[int, USICandidate] = {}
        lines = 0
        try:
            while True:
                line = self._readline(deadline, "search")
                lines += 1
                if lines > self.config.protocol_limits.max_search_lines:
                    raise self._protocol_error("teacher search exceeded the output-line bound")
                if line.startswith("info "):
                    parsed = parse_info_line(line, expected_multipv=self.config.multipv)
                    if parsed is not None:
                        candidates[parsed.multipv] = parsed
                    continue
                if line.startswith("bestmove "):
                    elapsed_ms = max(0, round((time.monotonic() - started) * 1_000))
                    result = _finish_search(
                        line,
                        candidates,
                        expected_multipv=self.config.multipv,
                        elapsed_ms=elapsed_ms,
                    )
                    self._assert_executable_metadata_unchanged()
                    return result
                raise self._protocol_error(f"unexpected search output: {line!r}")
        except USITimeoutError:
            self._send_stop_after_timeout()
            raise

    def analyze_with_retry(self, sfen: str, *, nodes: int | None = None) -> USISearchResult:
        """Restart and retry crashes/timeouts; reject malformed protocol immediately."""

        attempts = 0
        last_error: USIError | None = None
        stderr_tails: list[str] = []
        while attempts <= self.config.labeling.max_retries:
            attempts += 1
            try:
                if self._process is None or self._process.poll() is not None:
                    if last_error is not None:
                        self.restart_count += 1
                    self.start()
                return self.analyze(sfen, nodes=nodes)
            except USIProtocolError:
                self.close()
                raise
            except (USIProcessError, USITimeoutError) as error:
                last_error = error
                if error.stderr_tail:
                    stderr_tails.append(error.stderr_tail)
                elif self.stderr_tail:
                    stderr_tails.append(self.stderr_tail)
                self.close()
                if attempts > self.config.labeling.max_retries:
                    break
        if last_error is None:
            raise AssertionError("retry loop ended without an error")
        tail = "\n--- restart ---\n".join(stderr_tails)[
            -self.config.protocol_limits.max_stderr_bytes :
        ]
        raise USIRetryError(
            f"teacher {last_error.category} failure after {attempts} attempts: {last_error}",
            attempts=attempts,
            last_category=last_error.category,
            stderr_tail=tail,
        ) from last_error

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._send("stop")
        deadline = time.monotonic() + self.config.timeouts.stop_ms / 1_000
        while True:
            line = self._readline(deadline, "stop")
            if line.startswith("bestmove "):
                return
            if not line.startswith("info "):
                raise self._protocol_error(f"unexpected stop output: {line!r}")

    def close(self) -> None:
        process = self._process
        stdout = self._stdout
        stderr = self._stderr
        process_group = self._process_group
        process_start_identity = self._process_start_identity
        snapshot = self._snapshot
        self._snapshot = None
        runtime = self._runtime_snapshot
        self._runtime_snapshot = None
        rss_monitor = self._rss_monitor
        self._rss_monitor = None
        if process is None:
            if snapshot is not None:
                snapshot.close()
            if runtime is not None:
                runtime.close()
            if rss_monitor is not None:
                rss_monitor.close()
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    self._send("quit")
                    process.wait(timeout=self.config.timeouts.quit_ms / 1_000)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired, USIError):
                    pass
            if process_group is not None and process.poll() is None:
                _signal_process_group(
                    process_group,
                    signal.SIGTERM,
                    expected_start_identity=process_start_identity,
                )
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(1.0, self.config.timeouts.quit_ms / 1_000))
            if process_group is not None and process.poll() is None:
                _signal_process_group(
                    process_group,
                    signal.SIGKILL,
                    expected_start_identity=process_start_identity,
                )
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()
            if stdout is not None:
                stdout.join(1.0)
            if stderr is not None:
                stderr.join(1.0)
            if rss_monitor is not None:
                rss_monitor.close()
            self._process = None
            self._stdout = None
            self._stderr = None
            self._process_group = None
            self._process_start_identity = None
            self.identity = None
            if runtime is not None:
                runtime.unseal()
            if snapshot is not None:
                snapshot.close()
            if runtime is not None:
                runtime.close()

    def _read_usi_identity(self) -> USIIdentity:
        deadline = time.monotonic() + self.config.timeouts.startup_ms / 1_000
        name: str | None = None
        author: str | None = None
        options: set[str] = set()
        lines = 0
        while True:
            line = self._readline(deadline, "usi handshake")
            lines += 1
            if lines > self.config.protocol_limits.max_search_lines:
                raise self._protocol_error("teacher handshake exceeded the output-line bound")
            if line == "usiok":
                if name is None:
                    raise self._protocol_error("teacher omitted id name")
                return USIIdentity(name, author, tuple(sorted(options)))
            if line.startswith("id name "):
                if name is not None:
                    raise self._protocol_error("teacher repeated id name")
                name = _safe_protocol_text(line.removeprefix("id name "), "id name")
                continue
            if line.startswith("id author "):
                if author is not None:
                    raise self._protocol_error("teacher repeated id author")
                author = _safe_protocol_text(line.removeprefix("id author "), "id author")
                continue
            if line.startswith("option "):
                option_name = _parse_option_name(line)
                if option_name in options:
                    raise self._protocol_error(f"teacher repeated option {option_name!r}")
                options.add(option_name)
                continue
            raise self._protocol_error(f"unexpected USI handshake output: {line!r}")

    def _ready(self) -> None:
        self._send("isready")
        deadline = time.monotonic() + self.config.timeouts.ready_ms / 1_000
        while True:
            line = self._readline(deadline, "isready")
            if line == "readyok":
                return
            if line.startswith("info string "):
                continue
            raise self._protocol_error(f"unexpected isready output: {line!r}")

    def _send_stop_after_timeout(self) -> None:
        try:
            if self._process is None or self._process.poll() is not None:
                return
            self._send("stop")
            deadline = time.monotonic() + self.config.timeouts.stop_ms / 1_000
            while True:
                line = self._readline(deadline, "stop after timeout")
                if line.startswith("bestmove "):
                    return
        except USIError:
            return

    def _send(self, line: str) -> None:
        process = self._process
        if any(character in line for character in "\r\n\x00"):
            raise ValueError("USI command contains a line delimiter")
        if process is None or process.stdin is None:
            raise USIProcessError("teacher is not running")
        self._check_rss_monitor()
        return_code = process.poll()
        if return_code is not None:
            self._check_rss_monitor()
            raise USIProcessError(
                f"teacher exited with status {return_code}", stderr_tail=self.stderr_tail
            )
        try:
            process.stdin.write(line.encode("utf-8") + b"\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self._check_rss_monitor()
            raise USIProcessError(
                f"cannot write teacher command: {error}", stderr_tail=self.stderr_tail
            ) from error
        self._check_rss_monitor()

    def _readline(self, deadline: float, context: str) -> str:
        stdout = self._stdout
        process = self._process
        if stdout is None or process is None:
            raise USIProcessError("teacher is not running")
        self._check_rss_monitor()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise USITimeoutError(
                f"teacher timed out during {context}", stderr_tail=self.stderr_tail
            )
        try:
            line = stdout.read(remaining)
        except queue.Empty as error:
            self._check_rss_monitor()
            return_code = process.poll()
            if return_code is not None:
                raise USIProcessError(
                    f"teacher exited with status {return_code} during {context}",
                    stderr_tail=self.stderr_tail,
                ) from error
            raise USITimeoutError(
                f"teacher timed out during {context}", stderr_tail=self.stderr_tail
            ) from error
        except USIError as error:
            self._check_rss_monitor()
            if not error.stderr_tail:
                error.stderr_tail = self.stderr_tail
            raise
        self._check_rss_monitor()
        if not line:
            raise self._protocol_error("teacher emitted an empty stdout line")
        return line

    def _check_rss_monitor(self) -> None:
        monitor = self._rss_monitor
        if monitor is None:
            return
        try:
            monitor.check()
        except USIResourceError as error:
            error.stderr_tail = self.stderr_tail
            raise

    def _ensure_started(self) -> None:
        if self._process is None or self._process.poll() is not None or self.identity is None:
            self.start()

    def _protocol_error(self, message: str) -> USIProtocolError:
        return USIProtocolError(message, stderr_tail=self.stderr_tail)

    def _assert_executable_unchanged(self) -> None:
        snapshot = self._snapshot
        runtime = self._runtime_snapshot
        if snapshot is None or runtime is None:
            raise USIProcessError("teacher runtime snapshot is unavailable")
        try:
            snapshot.assert_snapshot_unchanged()
            snapshot.assert_source_unchanged()
            runtime.assert_unchanged()
        except (ExecutableSnapshotError, RuntimeTreeSnapshotError) as error:
            raise USIProcessError(f"teacher runtime identity drifted: {error}") from error

    def _assert_executable_metadata_unchanged(self) -> None:
        snapshot = self._snapshot
        runtime = self._runtime_snapshot
        if snapshot is None or runtime is None:
            raise USIProcessError("teacher runtime snapshot is unavailable")
        try:
            snapshot.assert_snapshot_metadata_unchanged()
            snapshot.assert_source_metadata_unchanged()
            runtime.assert_unchanged()
        except (ExecutableSnapshotError, RuntimeTreeSnapshotError) as error:
            raise USIProcessError(f"teacher runtime identity drifted: {error}") from error


def parse_info_line(line: str, *, expected_multipv: int) -> USICandidate | None:
    """Parse a complete scored PV info line, preserving cp versus mate."""

    tokens = line.split(" ")
    if not tokens or tokens[0] != "info" or any(not token for token in tokens):
        raise USIProtocolError("malformed info line")
    if len(tokens) >= 2 and tokens[1] == "string":
        return None
    integers: dict[str, int] = {}
    score: USIScore | None = None
    score_is_bounded = False
    pv: tuple[str, ...] | None = None
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in _INTEGER_INFO_KEYS:
            if token in integers or index + 1 >= len(tokens):
                raise USIProtocolError(f"malformed or duplicate info {token}")
            integers[token] = _parse_nonnegative_int(tokens[index + 1], token)
            index += 2
            continue
        if token == "score":
            if score is not None or index + 2 >= len(tokens):
                raise USIProtocolError("malformed or duplicate info score")
            kind = tokens[index + 1]
            if kind not in {"cp", "mate"}:
                raise USIProtocolError(f"unsupported USI score kind {kind!r}")
            value = _parse_signed_int(tokens[index + 2], f"score {kind}")
            score = USIScore(kind, value)
            index += 3
            if index < len(tokens) and tokens[index] in {"lowerbound", "upperbound"}:
                score_is_bounded = True
                index += 1
            continue
        if token == "pv":
            if pv is not None or index + 1 >= len(tokens):
                raise USIProtocolError("malformed or duplicate info pv")
            moves = tokens[index + 1 :]
            if any(_USI_MOVE_RE.fullmatch(move) is None for move in moves):
                raise USIProtocolError("info pv contains a malformed USI move")
            pv = tuple(moves)
            index = len(tokens)
            continue
        if token == "currmove":
            if index + 1 >= len(tokens) or _USI_MOVE_RE.fullmatch(tokens[index + 1]) is None:
                raise USIProtocolError("malformed info currmove")
            index += 2
            continue
        if token in {"refutation", "currline"}:
            index = _consume_move_sequence(tokens, index, token)
            continue
        raise USIProtocolError(f"unknown info token {token!r}")

    has_label_token = score is not None or pv is not None or "multipv" in integers
    if not has_label_token:
        return None
    required = {"depth", "seldepth", "nodes"}
    missing = sorted(required - set(integers))
    if missing or score is None or pv is None:
        raise USIProtocolError(f"scored PV info is incomplete; missing={missing}")
    rank = integers.get("multipv", 1)
    if expected_multipv > 1 and "multipv" not in integers:
        raise USIProtocolError("MultiPV search info omitted multipv rank")
    if not 1 <= rank <= expected_multipv:
        raise USIProtocolError(f"multipv rank {rank} exceeds configured MultiPV")
    if score_is_bounded:
        return None
    return USICandidate(
        multipv=rank,
        score=score,
        pv=pv,
        depth=integers["depth"],
        seldepth=integers["seldepth"],
        nodes=integers["nodes"],
    )


def _consume_move_sequence(tokens: list[str], index: int, field: str) -> int:
    """Consume UCI/USI variable-length refutation and currline fields.

    ``currline`` may begin with a positive decimal CPU number.  A non-decimal
    first token is the first move; ``0`` is not a valid USI CPU identifier. Both
    fields then carry one or more moves and end at the next recognized ``info``
    field. Treating them as single-token fields mis-parses valid engine
    diagnostics as unknown tokens.
    """

    cursor = index + 1
    if field == "currline" and cursor < len(tokens):
        possible_cpu = tokens[cursor]
        if possible_cpu.isascii() and possible_cpu.isdecimal():
            if _parse_nonnegative_int(possible_cpu, "currline cpu") < 1:
                raise USIProtocolError("info currline cpu integer must be at least one")
            cursor += 1
    start = cursor
    while cursor < len(tokens) and tokens[cursor] not in _INFO_FIELD_KEYS:
        if _USI_MOVE_RE.fullmatch(tokens[cursor]) is None:
            raise USIProtocolError(f"info {field} contains a malformed USI move")
        cursor += 1
        if cursor - start > _MAX_INFO_SEQUENCE_MOVES:
            raise USIProtocolError(f"info {field} exceeds the move-count bound")
    if cursor == start:
        raise USIProtocolError(f"malformed info {field}")
    return cursor


def _finish_search(
    bestmove_line: str,
    candidates: dict[int, USICandidate],
    *,
    expected_multipv: int,
    elapsed_ms: int,
) -> USISearchResult:
    tokens = bestmove_line.split(" ")
    if len(tokens) not in {2, 4} or tokens[0] != "bestmove":
        raise USIProtocolError("malformed bestmove line")
    bestmove = tokens[1]
    if _USI_MOVE_RE.fullmatch(bestmove) is None:
        raise USIProtocolError("teacher bestmove is not a normal USI move")
    if len(tokens) == 4 and (tokens[2] != "ponder" or _USI_MOVE_RE.fullmatch(tokens[3]) is None):
        raise USIProtocolError("malformed bestmove ponder suffix")
    observed_ranks = set(candidates)
    if not observed_ranks:
        raise USIProtocolError("teacher emitted no complete unbounded MultiPV rank")
    contiguous_ranks = set(range(1, max(observed_ranks) + 1))
    if observed_ranks != contiguous_ranks:
        raise USIProtocolError(
            f"teacher MultiPV ranks contain a gap: observed={sorted(observed_ranks)}"
        )
    ordered = tuple(candidates[rank] for rank in sorted(candidates))
    if ordered[0].pv[0] != bestmove:
        raise USIProtocolError("bestmove disagrees with MultiPV rank one")
    first_moves = [candidate.pv[0] for candidate in ordered]
    if len(set(first_moves)) != len(first_moves):
        raise USIProtocolError("MultiPV candidates repeat a first move")
    return USISearchResult(bestmove=bestmove, candidates=ordered, elapsed_ms=elapsed_ms)


def _parse_option_name(line: str) -> str:
    tokens = line.split(" ")
    if len(tokens) < 5 or tokens[:2] != ["option", "name"] or "type" not in tokens[2:]:
        raise USIProtocolError("malformed option declaration")
    type_index = tokens.index("type", 2)
    if type_index <= 2:
        raise USIProtocolError("option declaration omitted its name")
    return _safe_protocol_text(" ".join(tokens[2:type_index]), "option name")


def _safe_protocol_text(value: str, name: str) -> str:
    if not value or len(value) > 1_024 or any(character in value for character in "\r\n\x00"):
        raise USIProtocolError(f"invalid {name}")
    return value


def _parse_nonnegative_int(value: str, name: str) -> int:
    parsed = _parse_signed_int(value, name)
    if parsed < 0:
        raise USIProtocolError(f"info {name} must be nonnegative")
    return parsed


def _parse_signed_int(value: str, name: str) -> int:
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]{0,18})", value):
        raise USIProtocolError(f"info {name} is not a bounded integer")
    parsed = int(value)
    if not -(2**63) <= parsed <= 2**63 - 1:
        raise USIProtocolError(f"info {name} exceeds signed 64-bit range")
    return parsed


def _validate_sfen_command(sfen: str) -> None:
    if not isinstance(sfen, str) or len(sfen) > 2_048:
        raise ValueError("sfen must be a bounded string")
    if any(character in sfen for character in "\r\n\x00"):
        raise ValueError("sfen contains a command delimiter")
    fields = sfen.split(" ")
    if len(fields) != 4 or fields[1] not in {"b", "w"}:
        raise ValueError("sfen must contain four fields and a valid side")


def _terminate_unowned_process(process: subprocess.Popen[bytes]) -> None:
    """Reap a process that failed before it became owned by ``USIEngine``."""

    if process.poll() is None:
        start_identity = _read_process_start_identity(process.pid)
        if start_identity is not None:
            _signal_process_group(
                process.pid,
                signal.SIGKILL,
                expected_start_identity=start_identity,
            )
        else:
            # Popen still owns this unreaped child, while its process-group ID
            # cannot be authenticated.  Kill only the child in that case.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
    if process.poll() is None:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


def _signal_process_group(
    process_group: int,
    requested_signal: signal.Signals,
    *,
    expected_start_identity: str,
) -> None:
    if process_group <= 1:
        return
    if _read_process_start_identity(process_group) != expected_start_identity:
        return
    try:
        if os.getpgid(process_group) != process_group:
            return
        os.killpg(process_group, requested_signal)
    except (PermissionError, ProcessLookupError):
        return


def _read_process_start_identity(process_id: int) -> str | None:
    return read_process_identity(process_id)


def _teacher_process_tree_rss_bytes(
    leader_pid: int,
    *,
    retained_identities: Mapping[int, str] | None = None,
) -> tuple[int | None, dict[int, str]]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
            env={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "LC_ALL": "C",
                "LANG": "C",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise USIResourceError("cannot measure teacher process-tree RSS") from error
    if completed.returncode != 0 or len(completed.stdout) > 8 * 1024 * 1024:
        raise USIResourceError("cannot measure teacher process-tree RSS")
    processes: dict[int, tuple[int, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise USIResourceError("teacher process-tree RSS row is malformed")
        try:
            pid, ppid, pgid, rss_kib = (int(field) for field in fields[:4])
        except ValueError:
            raise USIResourceError("teacher process-tree RSS row is malformed") from None
        if pid <= 0 or ppid < 0 or pgid < 0 or rss_kib < 0 or pid in processes:
            raise USIResourceError("teacher process-tree RSS row is invalid")
        processes[pid] = (ppid, pgid, rss_kib)
    selected = {pid for pid, (_, pgid, _) in processes.items() if pgid == leader_pid}
    if leader_pid in processes:
        selected.add(leader_pid)
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in processes.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    # Preserve ownership of already sampled descendants after they escape the
    # process group or are reparented. PID reuse is rejected by the sampled
    # process-start identity before the process contributes to RSS or signals.
    for pid, expected_identity in (retained_identities or {}).items():
        row = processes.get(pid)
        if row is not None and _read_process_start_identity(pid) == expected_identity:
            selected.add(pid)
    if not selected:
        return None, {}
    identities: dict[int, str] = {}
    for pid in selected:
        identity = _read_process_start_identity(pid)
        if identity is None:
            raise USIResourceError("teacher process identity became unavailable")
        identities[pid] = identity
    return sum(processes[pid][2] for pid in selected) * 1024, identities
