#!/usr/bin/env python3
"""Bounded end-to-end verification of the R4 USI launcher package.

Drives the launcher exactly the way ShogiHome's native transport does: reserved
options before `isready`, asynchronous `go` with bounded waits for `bestmove`,
stop/gate lifecycle, mate-search decline, ponder hold, and reuse across games.
Standard library only; suitable for a local run or a hosted CI job.
"""

from __future__ import annotations

import argparse
import queue
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BESTMOVE_TIMEOUT_S = 30.0
IDLE_WINDOW_S = 0.4


class ProtocolError(AssertionError):
    """A protocol expectation failed; carries the transcript so CI logs stay readable."""


class EngineSession:
    """One launcher subprocess with line-oriented, bounded protocol reads."""

    def __init__(self, launcher: Path, cwd: Path) -> None:
        self.process = subprocess.Popen(
            [str(launcher)],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.transcript: list[str] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip("\n")
            self.transcript.append(f"< {line}")
            self.lines.put(line)

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.transcript.append(f"> {command}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def expect_line(self, prefix: str, timeout: float = BESTMOVE_TIMEOUT_S) -> str:
        """Scans the stream for the next line with the prefix, like a USI GUI does."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                line = self.lines.get(timeout=max(remaining, 0.01))
            except queue.Empty as error:
                raise ProtocolError(
                    f"timed out waiting for a line starting with {prefix!r}; transcript tail:\n"
                    + "\n".join(self.transcript[-25:])
                ) from error
            if line.startswith(prefix):
                return line

    def expect_silence(self, prefix: str, window: float) -> None:
        deadline = time.monotonic() + window
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                line = self.lines.get(timeout=max(remaining, 0.01))
            except queue.Empty:
                return
            if line.startswith(prefix):
                raise ProtocolError(
                    f"unexpected {line!r} within the idle window; transcript tail:\n"
                    + "\n".join(self.transcript[-25:])
                )

    def expect_bestmove(self, timeout: float = BESTMOVE_TIMEOUT_S) -> str:
        return self.expect_line("bestmove ", timeout)

    def wait_for_exit(self, timeout: float = 15.0) -> int:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self.process.kill()
            raise ProtocolError("engine did not exit after quit") from error

    def stderr_text(self) -> str:
        assert self.process.stderr is not None
        try:
            return self.process.stderr.read()
        except (OSError, ValueError):
            return ""

    def max_rss_mib(self) -> float:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        if sys.platform == "darwin":
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)  # macOS reports bytes.
        return float(usage.ru_maxrss) / 1024.0  # Linux reports KiB.

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)


def shogihome_handshake(engine: EngineSession) -> None:
    """Registration exactly as ShogiHome does it, including reserved options."""
    engine.send("usi")
    identity = None
    options: list[str] = []
    while True:
        line = engine.expect_line("")
        if line.startswith("id name "):
            identity = line
            continue
        if line.startswith(("id author ", "info ")):
            continue
        if line == "usiok":
            break
        if line.startswith("option name "):
            options.append(line)
            continue
        raise ProtocolError(f"unexpected line during handshake: {line!r}")
    if identity is None or "pure_learned" not in identity:
        raise ProtocolError(f"engine identity does not advertise the pure profile: {identity!r}")
    joined = "\n".join(options)
    for expected in (
        "option name USI_Hash type spin default 32 min 1 max 1024",
        "option name USI_Ponder type check default false",
        "option name Threads type spin default 1 min 1 max 1",
        "option name RuntimeProfile type combo default pure_learned var pure_learned",
    ):
        if expected not in joined:
            raise ProtocolError(f"missing advertised option: {expected!r}")
    engine.send("setoption name USI_Hash value 32")
    engine.send("setoption name USI_Ponder value false")
    engine.send("setoption name Threads value 1")
    engine.send("setoption name RuntimeProfile value pure_learned")
    engine.send("setoption name NotAnOsaiOption value 7")
    engine.send("isready")
    engine.expect_line("readyok")
    engine.send("usinewgame")


def play_one_move(engine: EngineSession, position: str, go: str) -> tuple[str, float]:
    engine.send(position)
    engine.send(go)
    started = time.monotonic()
    line = engine.expect_line("bestmove ")
    elapsed = time.monotonic() - started
    movement = line.split()[1]
    if movement in ("resign", "win"):
        raise ProtocolError(f"engine produced a non-move response to a normal go: {line!r}")
    return movement, elapsed


def run_suite(launcher: Path, cwd: Path) -> list[str]:
    facts: list[str] = []
    engine = EngineSession(launcher, cwd)
    try:
        shogihome_handshake(engine)

        movement, elapsed = play_one_move(
            engine, "position startpos", "go btime 60000 wtime 60000 binc 5000 winc 5000"
        )
        facts.append(f"initial position bestmove {movement} in {elapsed:.2f}s")
        score_lines = [
            line for line in engine.transcript if line.startswith("< ") and " score cp " in line
        ]
        if not score_lines:
            raise ProtocolError("no centipawn score evidence was emitted for the normal go")
        facts.append(f"centipawn score evidence from real search: {score_lines[-1].strip()[:80]}")

        movement, elapsed = play_one_move(
            engine,
            "position startpos moves 7g7f 3c3d 2g2f 8c8d",
            "go btime 15000 wtime 15000 byoyomi 1000",
        )
        facts.append(f"history position bestmove {movement} in {elapsed:.2f}s")

        movement, elapsed = play_one_move(
            engine, "position startpos", "go btime 0 wtime 0 binc 1000 winc 1000"
        )
        facts.append(f"zero base clock with increment still yielded {movement} in {elapsed:.2f}s")

        engine.send("setoption name USI_Hash value 16")
        engine.send("position startpos")
        engine.send("go movetime 200")
        engine.expect_line("info string hash 16MiB ")
        line = engine.expect_bestmove()
        facts.append(f"16 MiB TT request applied and answered: {line}")

        engine.send("position sfen 3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1")
        bestmoves_before = sum(1 for entry in engine.transcript if entry.startswith("< bestmove"))
        engine.send("go infinite")
        engine.expect_silence("bestmove ", IDLE_WINDOW_S)
        bestmoves_now = sum(1 for entry in engine.transcript if entry.startswith("< bestmove"))
        if bestmoves_now != bestmoves_before:
            raise ProtocolError(
                "an infinite search published bestmove before stop; transcript:\n"
                + "\n".join(engine.transcript[-25:])
            )
        engine.send("stop")
        line = engine.expect_bestmove()
        if not any(" score mate -0" in entry for entry in engine.transcript):
            raise ProtocolError(
                "the checkmated root was not reported with the losing score mate -0"
            )
        facts.append(
            f"infinite search at a terminal root held until stop, reported as mate -0: {line}"
        )

        engine.send("position startpos")
        engine.send("go infinite")
        engine.expect_silence("bestmove ", IDLE_WINDOW_S)
        engine.send("stop")
        engine.expect_bestmove()
        engine.send("stop")
        engine.expect_silence("bestmove ", IDLE_WINDOW_S)
        facts.append("idle repeated stop produced no stale completion")

        engine.send("go mate 5000")
        engine.expect_line("checkmate notimplemented")
        engine.send("go mate infinite")
        engine.expect_line("checkmate notimplemented")
        engine.send("isready")
        engine.expect_line("readyok")
        facts.append("go mate declined with checkmate notimplemented, engine stays usable")

        engine.send("position startpos")
        engine.send("go ponder")
        engine.expect_line("info string error")
        engine.send("ponderhit btime 30000 wtime 30000 binc 500 winc 500")
        line = engine.expect_bestmove()
        movement = line.split()[1]
        if movement in ("resign", "win") or not movement:
            raise ProtocolError(f"ponderhit completion was not a real move: {line!r}")
        facts.append("go ponder held, ponderhit started the real search")

        engine.send("position startpos")
        engine.send("go ponder")
        engine.expect_line("info string error")
        engine.send("stop")
        engine.expect_bestmove()
        engine.send("gameover win")
        engine.send("isready")
        engine.expect_line("readyok")
        facts.append("unhit go ponder answered stop and the game loop continued")

        engine.send("position startpos moves 7g7f 3c3d")
        engine.send("go btime 10000 wtime 10000 binc 1000 winc 1000")
        engine.expect_bestmove()
        engine.send("quit")
        code = engine.wait_for_exit()
        if code != 0:
            raise ProtocolError(f"engine exited with code {code}")
        stderr = engine.stderr_text()
        if stderr.strip():
            raise ProtocolError(f"unexpected stderr traffic: {stderr!r}")
        facts.append(f"clean exit after reuse; child peak RSS ~{engine.max_rss_mib():.0f} MiB")
    finally:
        engine.close()
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launcher", type=Path, help="path to osai-r4-usi.sh")
    parser.add_argument(
        "--with-spaced-copy",
        action="store_true",
        help="additionally run the suite from a copy of the package under a path with spaces",
    )
    arguments = parser.parse_args()
    launcher = arguments.launcher.resolve()
    if not launcher.is_file():
        print(f"FAIL launcher not found: {launcher}", file=sys.stderr)
        return 2

    try:
        facts = run_suite(launcher, launcher.parent)
        print(f"PASS {launcher} (cwd = package dir)")
        for fact in facts:
            print(f"  - {fact}")
        if arguments.with_spaced_copy:
            with tempfile.TemporaryDirectory(prefix="osai qa ") as spaced:
                copy_dir = Path(spaced) / "package copy"
                shutil.copytree(launcher.parent, copy_dir)
                copied_launcher = copy_dir / launcher.name
                copied_launcher.chmod(0o755)
                facts = run_suite(copied_launcher, Path(tempfile.gettempdir()))
                print(f"PASS {copied_launcher} (path with spaces, unrelated cwd)")
                for fact in facts:
                    print(f"  - {fact}")
    except ProtocolError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
