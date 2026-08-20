"""Independent replay of teacher moves through the repository's Rust USI CLI."""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from open_shogi_training.labeling.execution import (
    ExecutableSnapshot,
    ExecutableSnapshotError,
)
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    contained_path,
    ensure_contained_directory,
)
from open_shogi_training.selfplay.engine_receipt import (
    resolve_active_engine_build,
)

_MAX_CLI_BYTES: Final = 64 * 1024 * 1024
_MAX_LINE_BYTES: Final = 64 * 1024
_MAX_RESPONSE_LINES: Final = 1_024
_START_TIMEOUT_SECONDS: Final = 30.0
_READY_TIMEOUT_SECONDS: Final = 10.0
_CLOSE_TIMEOUT_SECONDS: Final = 2.0
_ERROR_PREFIX: Final = "info string error "
_USI_MOVE_RE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")


class _ReceiptLoader(Protocol):
    def __call__(self, root: Path) -> tuple[ArtifactRef, ArtifactRef]: ...


def _load_validated_build_receipt(root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    engine, reference, _ = resolve_active_engine_build(root)
    return engine, reference


class LegalityValidationError(RuntimeError):
    """Raised when OpenShogiAI rejects or cannot replay a teacher sequence."""


@dataclass(frozen=True, slots=True)
class LegalityValidatorIdentity:
    path: str
    sha256: str
    size: int
    build_receipt: ArtifactRef
    reported_name: str
    reported_author: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "build_receipt": self.build_receipt.as_dict(),
            "reported_name": self.reported_name,
            "reported_author": self.reported_author,
        }


@dataclass(frozen=True, slots=True)
class LegalityCoverage:
    requested_multipv: int
    returned_candidates: int
    legal_root_count: int | None

    @property
    def has_additional_legal_roots(self) -> bool:
        return (
            self.legal_root_count is not None and self.legal_root_count > self.returned_candidates
        )


class RustLegalityValidator:
    """Persistent, bounded USI session backed by ``target/release/open-shogi-cli``.

    The executable path is intentionally fixed below the supplied repository root.
    Callers must build that exact workspace binary before labeling; no external
    shogi library or teacher process participates in this check.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        receipt_loader: _ReceiptLoader = _load_validated_build_receipt,
    ) -> None:
        self._root = project_root.resolve(strict=True)
        self._path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: queue.Queue[bytes | None] = queue.Queue(maxsize=_MAX_RESPONSE_LINES)
        self._stderr = bytearray()
        self._stderr_lock = threading.Lock()
        self._reader_threads: list[threading.Thread] = []
        self._snapshot: ExecutableSnapshot | None = None
        self._build_receipt_sha256: str | None = None
        self._receipt_loader = receipt_loader
        self.identity: LegalityValidatorIdentity | None = None

    def start(self) -> LegalityValidatorIdentity:
        if self._process is not None:
            raise LegalityValidationError("Rust legality validator is already started")
        self._stdout = queue.Queue(maxsize=_MAX_RESPONSE_LINES)
        self._stderr.clear()
        self._reader_threads.clear()
        try:
            engine_ref, build_receipt_ref = self._receipt_loader(self._root)
            self._path = contained_path(self._root, engine_ref.path, must_exist=True)
            if self._path.resolve(strict=True) != self._path:
                raise LegalityValidationError(
                    "Rust legality validator path must not traverse symlinks"
                )
            self._build_receipt_sha256 = build_receipt_ref.sha256
            snapshot = ExecutableSnapshot.create(
                self._path,
                temporary_directory=ensure_contained_directory(
                    self._root, "local/runtime-snapshots"
                ),
                max_bytes=_MAX_CLI_BYTES,
                expected_sha256=engine_ref.sha256,
            )
        except OSError as error:
            raise LegalityValidationError(
                "missing target/release/open-shogi-cli; build the repository's release CLI first"
            ) from error
        except ExecutableSnapshotError as error:
            raise LegalityValidationError(
                f"cannot pin target/release/open-shogi-cli: {error}"
            ) from error
        self._snapshot = snapshot
        digest_hex = snapshot.identity.sha256
        try:
            assert self._path is not None
            process = subprocess.Popen(
                [str(self._path), "usi"],
                executable=snapshot.executable_path,
                cwd=self._root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=snapshot.pass_fds(),
                start_new_session=True,
            )
        except BaseException as error:
            snapshot.close()
            self._snapshot = None
            if isinstance(error, OSError):
                raise LegalityValidationError(
                    f"cannot start Rust legality validator: {error}"
                ) from error
            raise
        self._process = process
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self.close()
            raise LegalityValidationError("Rust legality validator pipes were not created")
        try:
            snapshot.assert_snapshot_unchanged()
            snapshot.assert_source_unchanged()
        except ExecutableSnapshotError as error:
            self.close()
            raise LegalityValidationError(
                "Rust legality validator changed during process creation"
            ) from error
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            name="rust-legality-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name="rust-legality-stderr",
            daemon=True,
        )
        self._reader_threads = [stdout_thread, stderr_thread]
        for thread in self._reader_threads:
            thread.start()
        try:
            self._send("usi")
            reported_name: str | None = None
            reported_author: str | None = None
            for _ in range(_MAX_RESPONSE_LINES):
                line = self._receive(_START_TIMEOUT_SECONDS)
                if line.startswith("id name "):
                    reported_name = line.removeprefix("id name ")
                elif line.startswith("id author "):
                    reported_author = line.removeprefix("id author ")
                elif line.startswith(_ERROR_PREFIX):
                    raise LegalityValidationError(f"Rust legality validator startup failed: {line}")
                elif line == "usiok":
                    break
            else:
                raise LegalityValidationError(
                    "Rust legality validator exceeded handshake line limit"
                )
            if reported_name is None or not reported_name.startswith("OpenShogiAI "):
                raise LegalityValidationError(
                    "target/release/open-shogi-cli reported an unexpected USI identity"
                )
            self.assert_binary_unchanged()
            assert self._path is not None
            identity = LegalityValidatorIdentity(
                path=self._path.relative_to(self._root).as_posix(),
                sha256=digest_hex,
                size=snapshot.identity.size,
                build_receipt=build_receipt_ref,
                reported_name=reported_name,
                reported_author=reported_author,
            )
            self.identity = identity
            return identity
        except BaseException:
            self.close()
            raise

    def validate(
        self,
        root_sfen: str,
        bestmove: str,
        pvs: list[list[str]],
        *,
        configured_multipv: int,
    ) -> LegalityCoverage:
        """Replay the best move and every complete PV from the supplied root."""

        if self._process is None or self.identity is None:
            raise LegalityValidationError("Rust legality validator is not started")
        self._assert_binary_metadata_unchanged()
        if not 1 <= len(pvs) <= configured_multipv:
            raise LegalityValidationError("candidate count is outside configured MultiPV")
        if any(not pv for pv in pvs):
            raise LegalityValidationError("candidate PV sequence is empty")
        if pvs[0][0] != bestmove:
            raise LegalityValidationError("bestmove disagrees with MultiPV rank one")
        if len({pv[0] for pv in pvs}) != len(pvs):
            raise LegalityValidationError("candidate root moves are not distinct")
        legal_root_count = None
        if len(pvs) < configured_multipv:
            legal_root_count = self._root_legal_move_count(root_sfen)
            if legal_root_count < len(pvs):
                raise LegalityValidationError(
                    "returned MultiPV roots exceed the independent Rust legal-root count "
                    f"({legal_root_count})"
                )
        sequences = [("bestmove", [bestmove])]
        sequences.extend((f"multipv {index}", pv) for index, pv in enumerate(pvs, start=1))
        for name, moves in sequences:
            if not moves:
                raise LegalityValidationError(f"{name} sequence is empty")
            command = f"position sfen {root_sfen} moves {' '.join(moves)}"
            if len(command.encode("utf-8")) > _MAX_LINE_BYTES:
                raise LegalityValidationError(
                    f"{name} replay command exceeds {_MAX_LINE_BYTES} bytes"
                )
            self._send(command)
            self._send("isready")
            errors: list[str] = []
            for _ in range(_MAX_RESPONSE_LINES):
                line = self._receive(_READY_TIMEOUT_SECONDS)
                if line.startswith(_ERROR_PREFIX):
                    errors.append(line.removeprefix(_ERROR_PREFIX))
                elif line == "readyok":
                    break
            else:
                raise LegalityValidationError(f"{name} replay exceeded response line limit")
            if errors:
                detail = "; ".join(errors)[:2_048]
                raise LegalityValidationError(f"OpenShogiAI rejected {name}: {detail}")
        self.assert_binary_unchanged()
        return LegalityCoverage(
            requested_multipv=configured_multipv,
            returned_candidates=len(pvs),
            legal_root_count=legal_root_count,
        )

    def _root_legal_move_count(self, root_sfen: str) -> int:
        self.assert_binary_unchanged()
        snapshot = self._snapshot
        if snapshot is None:
            raise LegalityValidationError("Rust legality validator has no executable snapshot")
        try:
            completed = subprocess.run(
                [
                    str(self._path),
                    "perft",
                    "--depth",
                    "1",
                    "--sfen",
                    root_sfen,
                    "--divide",
                ],
                executable=snapshot.executable_path,
                cwd=self._root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                pass_fds=snapshot.pass_fds(),
                timeout=_READY_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LegalityValidationError(
                "cannot count root legal moves through OpenShogiAI perft"
            ) from error
        if len(completed.stdout) > 1024 * 1024 or len(completed.stderr) > _MAX_LINE_BYTES:
            raise LegalityValidationError("OpenShogiAI perft output exceeded its bound")
        try:
            output = completed.stdout.decode("utf-8")
            error_output = completed.stderr.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LegalityValidationError("OpenShogiAI perft emitted non-UTF-8") from error
        if completed.returncode != 0:
            raise LegalityValidationError(
                f"OpenShogiAI could not parse root SFEN: {error_output.strip()[:2_048]}"
            )
        lines = output.splitlines()
        if not lines or not lines[-1].startswith("depth 1 nodes "):
            raise LegalityValidationError("OpenShogiAI perft summary is malformed")
        summary_fields = lines[-1].split()
        if len(summary_fields) != 14 or summary_fields[:3] != ["depth", "1", "nodes"]:
            raise LegalityValidationError("OpenShogiAI perft summary is malformed")
        try:
            reported_nodes = int(summary_fields[3])
        except ValueError as error:
            raise LegalityValidationError("OpenShogiAI perft node count is malformed") from error
        divide_moves: set[str] = set()
        for line in lines[:-1]:
            fields = line.split()
            if (
                len(fields) != 2
                or fields[1] != "1"
                or _USI_MOVE_RE.fullmatch(fields[0]) is None
                or fields[0] in divide_moves
            ):
                raise LegalityValidationError("OpenShogiAI perft divide row is malformed")
            divide_moves.add(fields[0])
        if reported_nodes != len(divide_moves):
            raise LegalityValidationError("OpenShogiAI perft count disagrees with divide rows")
        self.assert_binary_unchanged()
        return reported_nodes

    def assert_binary_unchanged(self) -> None:
        """Rehash the private snapshot and reject source-path identity drift."""

        snapshot = self._snapshot
        if snapshot is None:
            raise LegalityValidationError("Rust legality validator has no executable snapshot")
        try:
            snapshot.assert_snapshot_unchanged()
            snapshot.assert_source_unchanged()
        except ExecutableSnapshotError as error:
            raise LegalityValidationError(
                "Rust legality validator changed during labeling"
            ) from error

    def _assert_binary_metadata_unchanged(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            raise LegalityValidationError("Rust legality validator has no executable snapshot")
        try:
            snapshot.assert_snapshot_metadata_unchanged()
            snapshot.assert_source_metadata_unchanged()
        except ExecutableSnapshotError as error:
            raise LegalityValidationError(
                "Rust legality validator changed during labeling"
            ) from error

    def close(self) -> None:
        process = self._process
        self._process = None
        self.identity = None
        snapshot = self._snapshot
        self._snapshot = None
        self._build_receipt_sha256 = None
        if process is None:
            if snapshot is not None:
                snapshot.close()
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b"quit\n")
                    process.stdin.flush()
                process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                try:
                    # This is the trusted repository CLI and is not allowed to
                    # delegate work. Signal the owned Popen child instead of an
                    # unauthenticated numeric PGID that could have been reused.
                    process.terminate()
                    process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
                except (OSError, subprocess.TimeoutExpired):
                    with suppress(OSError):
                        process.kill()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
        try:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with suppress(OSError):
                        stream.close()
            for thread in self._reader_threads:
                thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)
            self._reader_threads.clear()
        finally:
            if snapshot is not None:
                snapshot.close()

    def __enter__(self) -> RustLegalityValidator:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(self, line: str) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise LegalityValidationError(
                f"Rust legality validator exited unexpectedly: {self._stderr_tail()}"
            )
        try:
            process.stdin.write(line.encode("utf-8") + b"\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise LegalityValidationError(
                f"cannot write to Rust legality validator: {self._stderr_tail()}"
            ) from error

    def _receive(self, timeout: float) -> str:
        try:
            raw = self._stdout.get(timeout=timeout)
        except queue.Empty as error:
            raise LegalityValidationError(
                f"Rust legality validator timed out: {self._stderr_tail()}"
            ) from error
        if raw is None:
            raise LegalityValidationError(
                f"Rust legality validator closed stdout: {self._stderr_tail()}"
            )
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LegalityValidationError("Rust legality validator emitted non-UTF-8") from error

    def _read_stdout(self, stream: object) -> None:
        reader = stream
        try:
            while True:
                raw = reader.readline(_MAX_LINE_BYTES + 1)  # type: ignore[attr-defined]
                if not raw:
                    break
                if len(raw) > _MAX_LINE_BYTES or not raw.endswith(b"\n"):
                    self._put_stdout(None)
                    return
                self._put_stdout(raw.rstrip(b"\r\n"))
        except (OSError, ValueError):
            pass
        self._put_stdout(None)

    def _read_stderr(self, stream: object) -> None:
        reader = stream
        try:
            while chunk := reader.read(4096):  # type: ignore[attr-defined]
                with self._stderr_lock:
                    self._stderr.extend(chunk)
                    if len(self._stderr) > _MAX_LINE_BYTES:
                        del self._stderr[: len(self._stderr) - _MAX_LINE_BYTES]
        except (OSError, ValueError):
            pass

    def _put_stdout(self, value: bytes | None) -> None:
        try:
            self._stdout.put(value, timeout=1)
        except queue.Full:
            with suppress(queue.Empty):
                self._stdout.get_nowait()
            with suppress(queue.Full):
                self._stdout.put_nowait(None)

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr).decode("utf-8", errors="replace")[-2_048:]
