"""One finite, immutable evaluator run; completion returns to Astra without promotion.

The detached supervisor measures actual progress and resources. Its exclusive lease is
inherited by each stage, so losing the supervisor cannot start a second writer.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .evaluator_data import atomic, digest, encoded, generate, prepare
from .labeling.process_identity import read_process_identity

ROOT = Path(__file__).resolve().parents[2]
MODULE = "open_shogi_training.evaluator_run"
SCHEMA = "open_shogiai_evaluator_run/v1"
STAGES = ("generate", "prepare", "train", "audit", "arena")
RUNTIME_SOURCES = {
    "replay": "target/release/examples/position_audit",
    "probe": "target/release/examples/core_probe",
    "module": "target/pure/bindings/open_shogi_wasm.js",
    "wasm": "target/pure/bindings/open_shogi_wasm_bg.wasm",
}


def inside(path: str | Path, *, exists: bool = True) -> Path:
    value = Path(path)
    value = value if value.is_absolute() else ROOT / value
    if not value.is_relative_to(ROOT):
        raise ValueError("path outside repository")
    for parent in (value, *value.parents):
        if parent == ROOT.parent:
            break
        if parent.is_symlink():
            raise ValueError("run paths must not traverse symlinks")
    resolved = value.resolve(strict=exists)
    if not resolved.is_relative_to(ROOT):
        raise ValueError("resolved path outside repository")
    return resolved


def _json(path: Path) -> dict:
    path = inside(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024 * 1024:
        raise ValueError("invalid bounded JSON file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON contract must be an object")
    # Also rejects Infinity produced by an oversized JSON exponent.
    encoded(value)
    return value


def _number(value, low: float, high: float, name: str, *, integer: bool = True) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and type(value) is not int)
    ):
        raise ValueError(f"invalid bounded {name}")


def _hash(value) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("invalid artifact hash")
    return value


def _validate_config(config: dict) -> None:
    encoded(config)
    if config.get("schema") != SCHEMA:
        raise ValueError("unknown run contract")
    if not isinstance(config.get("run_id"), str) or not re.fullmatch(
        r"[a-zA-Z0-9._-]{1,96}", config["run_id"]
    ):
        raise ValueError("invalid run ID")
    output = inside(config["output"], exists=False)
    if not output.is_relative_to(ROOT / "local" / "runs"):
        raise ValueError("run output must remain under local/runs")
    _number(config["seed"], 0, 2**63 - 1, "seed")
    generation = config["generation"]
    for name, low, high in (
        ("first_game", 0, 10**9),
        ("games", 1, 65536),
        ("workers", 1, 8),
        ("max_plies", 1, 512),
        ("sample_stride", 1, 512),
        ("deviation_stride", 1, 512),
        ("teacher_nodes", 1, 10_000_000),
    ):
        _number(generation[name], low, high, name)
    _hash(generation["leaf_sha256"])
    _hash(generation["teacher_binary_sha256"])
    if generation.get("development_exclusions_path"):
        _hash(generation["development_exclusions_sha256"])
    training = config["training"]
    if training["device"] != "cpu":
        raise ValueError("only the measured CPU training route is enabled")
    for name, maximum in (
        ("threads", 16),
        ("batch_size", 4096),
        ("max_steps", 1_000_000),
        ("max_epochs", 1024),
        ("validation_every", 1_000_000),
        ("patience", 1000),
        ("warmup_steps", 1_000_000),
    ):
        _number(training[name], 1, maximum, name)
    _number(training["learning_rate"], 1e-12, 1, "learning_rate", integer=False)
    _number(
        training["minimum_learning_rate"],
        0,
        training["learning_rate"],
        "minimum_learning_rate",
        integer=False,
    )
    _number(
        training["minimum_relative_improvement"],
        0,
        1,
        "minimum_relative_improvement",
        integer=False,
    )
    _number(training["gradient_norm_limit"], 1e-12, 1e6, "gradient_norm_limit", integer=False)
    resources = config["resources"]
    for name, low, high, integer in (
        ("maximum_wall_seconds", 1, 86400, True),
        ("free_space_floor_gib", 1, 1024, False),
        ("maximum_process_rss_gib", 1, 20, False),
        ("maximum_swap_growth_gib", 0, 16, False),
        ("stalled_seconds", 1, 86400, True),
        ("maximum_retries", 0, 3, True),
        ("monitor_interval_seconds", 0.1, 60, False),
    ):
        _number(resources[name], low, high, name, integer=integer)
    for name, expected in (
        ("depth", 64),
        ("max_plies", 256),
        ("paired_3minute_games", 24),
        ("paired_10minute_games", 8),
        ("startpos_demonstration_games", 8),
    ):
        if config["evaluation"][name] != expected:
            raise ValueError("evaluation must retain the reviewed finite plan")
    exclusions = config["excluded_development_sfens"]
    if (
        not isinstance(exclusions, list)
        or len(exclusions) > 2048
        or any(not isinstance(sfen, str) or not 1 <= len(sfen) <= 512 for sfen in exclusions)
    ):
        raise ValueError("invalid development regression exclusions")


def _code_identity() -> dict:
    scope = [
        "engine",
        "training",
        "scripts",
        "Cargo.lock",
        "pyproject.toml",
        "uv.lock",
        "configs/runtime",
        "configs/evaluator-main.json",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *scope],
        cwd=ROOT,
        text=True,
    )
    if dirty:
        raise ValueError("commit the stable run code/config, including new files, before sealing")
    paths = subprocess.check_output(["git", "ls-files", "--", *scope], cwd=ROOT, text=True)
    files = {name: digest(inside(name)) for name in paths.splitlines()}
    if str(Path(__file__).resolve().relative_to(ROOT)) not in files:
        raise ValueError("the actual runner must belong to the sealed code commit")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": files,
    }


def _reference(path: Path, expected: str | None = None) -> dict:
    path = inside(path)
    if not path.is_file():
        raise ValueError("artifact must be a regular file")
    sha = digest(path)
    if expected is not None and sha != _hash(expected):
        raise ValueError(f"artifact changed: {path.relative_to(ROOT)}")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha}


def verify(run: Path) -> dict:
    run = inside(run)
    receipt = _json(run / "seal.json")
    if receipt.get("schema") != "open_shogiai_evaluator_seal/v1":
        raise ValueError("missing or unknown seal receipt")
    if digest(inside(run / "run.json")) != _hash(receipt["run_sha256"]):
        raise ValueError("sealed run.json changed; preserve assets and create a new run")
    config = _json(run / "run.json")
    _validate_config(config)
    if inside(config["output"]) != run or receipt["run_id"] != config["run_id"]:
        raise ValueError("sealed run location/identity mismatch")
    if receipt["code_commit"] != config["code"]["commit"]:
        raise ValueError("sealed commit identity mismatch")
    for name, expected in config["code"]["files"].items():
        _reference(inside(name), expected)
    if set(config["runtime"]) != set(RUNTIME_SOURCES):
        raise ValueError("incomplete runtime snapshot")
    for ref in config["runtime"].values():
        path = inside(ref["path"])
        if path.parent != run / "runtime":
            raise ValueError("runtime must be copied inside this run")
        _reference(path, ref["sha256"])
    for ref in config["inputs"].values():
        _reference(inside(ref["path"]), ref["sha256"])
    return config


def seal(config_path: Path) -> dict:
    config_path = inside(config_path)
    config = _json(config_path)
    _validate_config(config)
    code = _code_identity()
    run = inside(config["output"], exists=False)
    if run.exists():
        raise ValueError("run already exists; use status/start to resume its immutable contract")
    generation = config["generation"]
    inputs = {"source_config": _reference(config_path)}
    for name, path_key, hash_key in (
        ("leaf", "leaf_path", "leaf_sha256"),
        ("teacher_config", "teacher_config_path", "teacher_config_sha256"),
        ("split_guard", "split_guard_path", "split_guard_sha256"),
        ("development_exclusions", "development_exclusions_path", "development_exclusions_sha256"),
    ):
        if path_key not in generation:
            if name == "development_exclusions":
                continue
            raise ValueError(f"missing {path_key}")
        inputs[name] = _reference(inside(generation[path_key]), generation.get(hash_key))
        generation[hash_key] = inputs[name]["sha256"]
    sources = {name: _reference(inside(path)) for name, path in RUNTIME_SOURCES.items()}
    run.parent.mkdir(parents=True, exist_ok=True)
    # This temporary directory is exclusively created here and never named by a sealed run.
    staging = Path(tempfile.mkdtemp(prefix=f".{run.name}.sealing-", dir=run.parent))
    try:
        (staging / "runtime").mkdir()
        runtime = {}
        for name, ref in sources.items():
            source = inside(ref["path"])
            destination = staging / "runtime" / source.name
            shutil.copy2(source, destination)
            if digest(destination) != ref["sha256"] or digest(source) != ref["sha256"]:
                raise ValueError("runtime changed during snapshot copy")
            runtime[name] = {
                "path": str((run / "runtime" / source.name).relative_to(ROOT)),
                "sha256": ref["sha256"],
            }
        generation.update(
            seed=config["seed"],
            replay_path=runtime["replay"]["path"],
            replay_sha256=runtime["replay"]["sha256"],
        )
        config["training"].update(
            seed=config["seed"],
            initial_model=generation["leaf_path"],
            initial_sha256=generation["leaf_sha256"],
        )
        config.update(code=code, runtime=runtime, inputs=inputs, sealed_at=time.time())
        for ref in inputs.values():
            _reference(inside(ref["path"]), ref["sha256"])
        for path, expected in code["files"].items():
            _reference(inside(path), expected)
        atomic(staging / "run.json", encoded(config))
        receipt = {
            "schema": "open_shogiai_evaluator_seal/v1",
            "run_id": config["run_id"],
            "run_sha256": digest(staging / "run.json"),
            "code_commit": code["commit"],
        }
        atomic(staging / "seal.json", encoded(receipt))
        atomic(
            staging / "state.json",
            encoded(
                {
                    "status": "prepared",
                    "run_id": config["run_id"],
                    "pid": None,
                    "run_sha256": receipt["run_sha256"],
                }
            ),
        )
        if run.exists():
            raise ValueError("run appeared during sealing; nothing was replaced")
        staging.rename(run)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only unpublished files created by this seal attempt.
    return {
        "run_id": config["run_id"],
        "run": str(run.relative_to(ROOT)),
        "status": "prepared",
        "code_commit": code["commit"],
        "run_sha256": receipt["run_sha256"],
    }


@contextlib.contextmanager
def _lease(run: Path, inherited_fd: int | None = None):
    path = inside(run / "operation.lock", exists=False)
    if inherited_fd is None:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    else:
        descriptor = inherited_fd
        observed, expected = os.fstat(descriptor), path.stat()
        if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("inherited lease belongs to another run")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield descriptor
    finally:
        # Do not LOCK_UN: children inherit the same open-file description.
        os.close(descriptor)


def _process_table() -> dict[int, tuple[int, int, str, int]]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss=,stat=,pgid="],
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if len(result.stdout) > 8 * 1024 * 1024:
        raise ValueError("process inventory exceeds bound")
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("invalid process inventory")
        pid, parent, rss, group = (*(int(value) for value in fields[:3]), int(fields[4]))
        if min(pid, parent, rss, group) < 0 or pid == 0 or pid in rows:
            raise ValueError("invalid process inventory values")
        rows[pid] = parent, rss * 1024, fields[3], group
    return rows


def _stage_group_snapshot(process, expected_identity: str | None = None) -> dict:
    """Authenticate the still-unreaped direct child before inspecting its group.

    Do not call Popen.poll/send_signal/wait here: poll reaps an exited leader.
    Holding its zombie reserves the numeric PID/PGID until all group members have
    stopped. Darwin no longer exposes a zombie's start identity; its direct PPID
    and private PGID plus our unreaped Popen ownership are the remaining anchor.
    """
    if process.returncode is not None:
        raise RuntimeError("stage leader was reaped before group cleanup")
    rows = _process_table()
    leader = rows.get(process.pid)
    if leader is None or leader[0] != os.getpid() or leader[3] != process.pid:
        raise RuntimeError("cannot authenticate the unreaped stage process group")
    if not leader[2].startswith("Z"):
        identity = read_process_identity(process.pid)
        if identity is None:
            # The leader can exit between ps and the identity read. Reinspect it
            # without reaping; any other unexplained identity loss fails closed.
            rows = _process_table()
            leader = rows.get(process.pid)
            if (
                leader is None
                or leader[0] != os.getpid()
                or leader[3] != process.pid
                or not leader[2].startswith("Z")
            ):
                raise RuntimeError("active stage process identity unavailable")
        elif expected_identity is not None and identity != expected_identity:
            raise RuntimeError("stage process identity changed")
    return rows


def _sample_owned(process, owned: dict[int, str]) -> tuple[int, dict[int, str]]:
    rows = _stage_group_snapshot(process, owned.get(process.pid))
    identities = {}
    for pid, (_, _, state, group) in rows.items():
        if group != process.pid or state.startswith("Z"):
            continue
        identity = read_process_identity(pid)
        if identity is not None:
            # read_process_identity binds its current group as well as start time.
            if identity.split(":")[2] != str(process.pid):
                raise RuntimeError("stage member left the isolated process group")
            identities[pid] = identity
    # Charge every live member from the authenticated group inventory, including
    # short-lived helpers that exit before their individual identity can be read.
    return sum(
        row[1] for row in rows.values() if row[3] == process.pid and not row[2].startswith("Z")
    ), identities


def _signal_owned(owned: dict[int, str], requested_signal: signal.Signals) -> list[int]:
    signalled = []
    for pid, identity in owned.items():
        if pid <= 1 or pid == os.getpid() or read_process_identity(pid) != identity:
            continue
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, requested_signal)
            signalled.append(pid)
    return signalled


def _cleanup(process, owned: dict[int, str], *, grace_seconds: float = 20) -> dict:
    expected_identity = owned.get(process.pid)
    signalled = set()

    def active_members():
        rows = _stage_group_snapshot(process, expected_identity)
        return {
            pid: row
            for pid, row in rows.items()
            if row[3] == process.pid and not row[2].startswith("Z")
        }

    # Let STOP-aware stages flush a checkpoint without reaping their leader.
    deadline = time.monotonic() + grace_seconds
    active = active_members()
    while active and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        active = active_members()
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        if not active:
            break
        # The leader is still our unreaped child, so this PGID cannot be reused.
        # killpg includes children created after the inventory and before the signal.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, requested_signal)
            signalled.update(active)
        deadline = time.monotonic() + 3
        while active and time.monotonic() < deadline:
            time.sleep(0.05)
            active = active_members()
    if active:
        raise RuntimeError("stage group did not stop after bounded cleanup")
    # No live group member can fork now. Reap only after this verified empty state.
    process.wait(timeout=1)
    return {
        "signalled_pids": sorted(signalled),
        "remaining_processes": {},
        "inventory_errors": [],
        "group_cleanup_before_reap": True,
    }


def _register_stage_group(run: Path, stage: str) -> dict:
    if _residual_stage_group(run) is not None:
        raise RuntimeError("unproven residual stage group prevents child launch")
    pid, group, session = os.getpid(), os.getpgid(0), os.getsid(0)
    identity = read_process_identity(pid)
    if pid != group or pid != session or identity is None:
        raise RuntimeError("stage must start in its own authenticated process group/session")
    receipt = {
        "schema": "open_shogiai_evaluator_stage_process/v1",
        "run_sha256": digest(run / "run.json"),
        "stage": stage,
        "pid": pid,
        "pgid": group,
        "process_identity": identity,
        "registered_at": time.time(),
        "status": "active",
    }
    # This must precede generation, native probes, torch or any other child launch.
    atomic(run / "stage-process.json", encoded(receipt))
    return receipt


def _residual_stage_group(run: Path) -> dict | None:
    path = run / "stage-process.json"
    if not path.exists():
        return None
    receipt = _json(path)
    pid = receipt.get("pid")
    if (
        receipt.get("schema") != "open_shogiai_evaluator_stage_process/v1"
        or receipt.get("run_sha256") != digest(run / "run.json")
        or type(pid) is not int
        or pid <= 1
        or receipt.get("pgid") != pid
        or not isinstance(receipt.get("process_identity"), str)
        or receipt.get("status") not in ("active", "cleaned")
    ):
        raise ValueError("invalid persistent stage group receipt")
    if receipt["status"] == "cleaned":
        return None
    members = [process_id for process_id, row in _process_table().items() if row[3] == pid]
    if members:
        # Without the original supervisor's unreaped Popen, do not authorize a
        # numeric-group signal. Preserve evidence and forbid another writer.
        return {**receipt, "remaining_pids": sorted(members)}
    return None


def _mark_group_cleaned(run: Path, process) -> None:
    path = run / "stage-process.json"
    if not path.exists():
        return  # The stage exited before registering, hence before creating children.
    receipt = _json(path)
    if receipt["pid"] != process.pid:
        if receipt["status"] != "cleaned":
            raise RuntimeError("stage group receipt belongs to another active process")
        return
    receipt.update(status="cleaned", cleaned_at=time.time(), returncode=process.returncode)
    atomic(path, encoded(receipt))


def status(run: Path) -> dict:
    value = _json(run / "state.json") if (run / "state.json").exists() else {}
    value["process_alive"] = bool(value.get("process_identity")) and (
        read_process_identity(value.get("pid")) == value["process_identity"]
    )
    value["stage_alive"] = bool(value.get("stage_identity")) and (
        read_process_identity(value.get("stage_pid")) == value["stage_identity"]
    )
    group = _residual_stage_group(run)
    if group is not None:
        value["stage_group"] = group
        value["stage_alive"] = read_process_identity(group["pid"]) == group["process_identity"]
    try:
        with _lease(run):
            value["lease_held"] = False
    except BlockingIOError:
        value["lease_held"] = True
    if value["lease_held"] and not value["process_alive"]:
        value["supervision"] = "supervisor_missing; retained stage lease prevents another writer"
    value["completed_trajectories"] = len(list((run / "data" / "games").glob("*.receipt.json")))
    if (run / "fit" / "progress.json").exists():
        value["training"] = _json(run / "fit" / "progress.json")
    if (run / "data" / "dataset" / "manifest.json").exists():
        value["unique_positions"] = _json(run / "data" / "dataset" / "manifest.json")[
            "unique_positions"
        ]
    if (run / "fit" / "training.json").exists():
        summary = _json(run / "fit" / "training.json")
        value.update(training_status=summary["status"], best_sha256=summary["best_sha256"])
    return value


def _clear_stop(run: Path) -> None:
    for path in (run / "STOP", run / "data" / "STOP", run / "arena" / "STOP"):
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_size)):
            raise ValueError("unexpected STOP marker")
        path.unlink(missing_ok=True)


def start(run: Path) -> dict:
    config = verify(run)
    try:
        with _lease(run) as lease:
            current = _json(run / "state.json")
            residual = _residual_stage_group(run)
            if residual is not None:
                current.update(
                    status="needs_astra",
                    reason="residual_stage_group_unproven",
                    residual_group=residual,
                )
                atomic(run / "state.json", encoded(current))
                return current
            if current.get("status") == "awaiting_astra_browser":
                for stage in STAGES:
                    _verify_completion(run, stage, config)
                return current
            _clear_stop(run)
            with inside(run / "supervisor.log", exists=False).open("ab") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        MODULE,
                        "work",
                        str(run.relative_to(ROOT)),
                        "--lease-fd",
                        str(lease),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(lease,),
                )
            return {
                "status": "starting",
                "pid": process.pid,
                "run_id": config["run_id"],
                "log": str((run / "supervisor.log").relative_to(ROOT)),
            }
    except BlockingIOError:
        current = status(run)
        if not current["process_alive"] and (current["stage_alive"] or current.get("stage_group")):
            stop(run)
            current["status"] = "supervisor_missing_stop_requested"
        return current


def stop(run: Path) -> dict:
    for folder in (run, run / "data", run / "arena"):
        if not folder.exists():
            continue
        path = inside(folder / "STOP", exists=False)
        if path.exists() and (not path.is_file() or path.stat().st_size):
            raise ValueError("unexpected STOP marker")
        path.touch()
    return {
        "status": "stop_requested",
        "retention": "completed data, best model and latest resume states",
    }


def _swap_bytes() -> int:
    output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=3)
    found = re.search(r"used\s*=\s*([0-9.]+)([MG])", output)
    if found is None:
        raise ValueError("swap measurement unavailable")
    return int(float(found[1]) * (1024**2 if found[2] == "M" else 1024**3))


def _progress_signature(run: Path, stage: str) -> tuple:
    folder = {
        "generate": run / "data" / "games",
        "prepare": run / "data" / "dataset",
        "train": run / "fit",
        "audit": run,
        "arena": run / "arena",
    }[stage]
    ignored = {
        "state.json",
        "stage-process.json",
        "supervisor.log",
        "monitor.jsonl",
        "operation.lock",
    }
    observed = []
    for path in folder.rglob("*") if folder.exists() else []:
        if path.name in ignored or path.name.endswith((".log", ".tmp")):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # Atomic publication and bounded checkpoint pruning are normal.
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("unexpected symlink in stage output")
        if stat.S_ISREG(info.st_mode):
            observed.append((info.st_size, info.st_mtime_ns))
    return (
        len(observed),
        sum(size for size, _ in observed),
        max((modified for _, modified in observed), default=0),
    )


def _dataset(run: Path, config: dict) -> dict:
    if not (run / "data" / "dataset" / "manifest.json").exists():
        raise ValueError("completed dataset manifest is required")
    manifest = _json(run / "data" / "dataset" / "manifest.json")
    for name, directory in (
        ("artifacts", run / "data" / "dataset"),
        ("source_games", run / "data" / "games"),
    ):
        for ref in manifest[name]:
            path = Path(ref["path"])
            if path.is_absolute() or len(path.parts) != 1:
                raise ValueError("dataset references must be local file names")
            inside(directory / path)  # Reject links before prepare reads the referenced bytes.
    return prepare(ROOT, run / "data", config["generation"], config["excluded_development_sfens"])


def _training_summary(run: Path, config: dict) -> dict:
    result = _json(run / "fit" / "training.json")
    expected = {
        "code_commit": config["code"]["commit"],
        "run_sha256": digest(run / "run.json"),
        "dataset_sha256": digest(run / "data" / "dataset" / "manifest.json"),
    }
    if result["status"] != "complete" or result["identity"] != expected:
        raise ValueError("training completion identity mismatch")
    _reference(run / "fit" / "best.osaval03", result["best_sha256"])
    resume = _json(run / "fit" / "resume.json")
    if resume["step"] != result["step"]:
        raise ValueError("final checkpoint step differs from training completion")
    checkpoint = inside(run / "fit" / resume["path"])
    if checkpoint.parent != run / "fit":
        raise ValueError("checkpoint escapes fit directory")
    _reference(checkpoint, resume["sha256"])
    return result


def _audit_report(run: Path, config: dict) -> dict:
    value = _json(run / "model-audit.json")
    if (
        value.get("schema") != "open_shogiai_evaluator_export_audit/v1"
        or value.get("runtime") != "actual-wasm-in-node"
        or value.get("parity", {}).get("roots", 0) < 10
        or value.get("parity", {}).get("children", 0) < 1
        or value.get("parity", {}).get("maximumCpDifference") != 0
        or len(value.get("searches", [])) != 2
        or value.get("status") != "PASS"
        or value.get("format") != "OSAVAL03"
        or value.get("profile") != "pure_learned"
        or value.get("errors") != []
        or value.get("parity", {}).get("errors") != 0
    ):
        raise ValueError("existing model audit is not a successful export contract")
    paths = {
        "auditScript": ROOT / "scripts/check_evaluator_model.mjs",
        "moduleJs": inside(config["runtime"]["module"]["path"]),
        "wasm": inside(config["runtime"]["wasm"]["path"]),
        "nativeProbe": inside(config["runtime"]["replay"]["path"]),
        "model": run / "fit" / "best.osaval03",
    }
    for name, path in paths.items():
        _reference(path, value["artifacts"][name]["sha256"])
    return value


def _arena_complete(result: dict) -> None:
    if result.get("status") == "stopped":
        raise InterruptedError("arena stopped with retained game receipts")
    if (
        result.get("status") != "complete"
        or result.get("planned_games") != 40
        or result.get("summary", {}).get("all_planned_complete") is not True
    ):
        raise ValueError("arena is failed or incomplete; preserve its evidence for Astra")
    # Losing the comparison is still a valid completed experiment, never a reason to rerun it.


def _completion_artifacts(run: Path, stage: str) -> list[dict]:
    paths = {
        "generate": [
            run / "data" / "generation.json",
            run / "data" / "generation-complete.json",
            *sorted((run / "data" / "games").glob("*.json.gz")),
            *sorted((run / "data" / "games").glob("*.receipt.json")),
        ],
        "prepare": [run / "data" / "dataset" / "manifest.json"],
        "train": [
            run / "fit" / "training.json",
            run / "fit" / "resume.json",
            run / "fit" / "best.osaval03",
        ],
        "audit": [run / "model-audit.json", run / "development-test.json"],
        "arena": [run / "arena" / "arena.json", run / "arena" / "plan.json"],
    }[stage]
    if stage == "arena":
        for attempt in _json(run / "arena" / "arena.json")["attempts"]:
            paths.extend(
                inside(ref["path"])
                for ref in [attempt["trace"], *attempt.get("stderr_artifacts", [])]
                if ref is not None
            )
    return [_reference(path) for path in paths]


def _verify_completion(run: Path, stage: str, config: dict) -> dict:
    receipt = _json(run / f"{stage}-complete.json")
    if (
        receipt.get("schema") != "open_shogiai_evaluator_stage/v1"
        or receipt.get("stage") != stage
        or receipt.get("run_sha256") != digest(run / "run.json")
    ):
        raise ValueError("completion receipt belongs to another run/stage")
    if receipt["artifacts"] != _completion_artifacts(run, stage):
        raise ValueError("completed stage artifact changed")
    if stage in ("prepare", "train", "audit", "arena"):
        _dataset(run, config)
    if stage in ("train", "audit", "arena"):
        _training_summary(run, config)
    if stage in ("audit", "arena"):
        _audit_report(run, config)
    if stage == "arena":
        _arena_complete(receipt["result"])
        if receipt["result"] != _json(run / "arena" / "arena.json"):
            raise ValueError("arena summary differs from its completion receipt")
    return receipt["result"]


def _run_stage(
    run: Path, stage: str, config: dict, state: dict, lease: int
) -> tuple[int, str | None]:
    process, owned, failure = None, {}, None
    limits = config["resources"]
    try:
        with inside(run / f"{stage}-{state['retries'].get(stage, 0)}.log", exists=False).open(
            "ab"
        ) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "stage",
                    str(run.relative_to(ROOT)),
                    stage,
                    "--lease-fd",
                    str(lease),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                start_new_session=True,
                pass_fds=(lease,),
            )
            state.update(stage_pid=process.pid, stage_identity=read_process_identity(process.pid))
            atomic(run / "state.json", encoded(state))
            signature, last_progress = None, time.monotonic()
            while True:
                rows = _stage_group_snapshot(process, state["stage_identity"])
                if rows[process.pid][2].startswith("Z"):
                    break
                if (run / "STOP").exists():
                    failure = "requested_stop"
                    break
                current = _progress_signature(run, stage)
                if current != signature:
                    signature, last_progress = current, time.monotonic()
                rss, owned = _sample_owned(process, owned)
                free, swap = shutil.disk_usage(run).free, _swap_bytes()
                metric = {
                    "at": time.time(),
                    "stage": stage,
                    "stage_pid": process.pid,
                    "rss_bytes": rss,
                    "free_bytes": free,
                    "swap_bytes": swap,
                    "progress": signature,
                }
                with inside(run / "monitor.jsonl", exists=False).open("ab") as monitor:
                    monitor.write(encoded(metric) + b"\n")
                state["owned_processes"] = {str(pid): identity for pid, identity in owned.items()}
                atomic(run / "state.json", encoded(state))
                if time.time() - state["began_at"] > limits["maximum_wall_seconds"]:
                    failure = "wall_limit"
                elif rss > limits["maximum_process_rss_gib"] * 1024**3:
                    failure = "memory_limit"
                elif free < limits["free_space_floor_gib"] * 1024**3:
                    failure = "space_limit"
                elif (
                    swap - state["initial_swap_bytes"] > limits["maximum_swap_growth_gib"] * 1024**3
                ):
                    failure = "swap_limit"
                elif time.monotonic() - last_progress > limits["stalled_seconds"]:
                    failure = "no_actual_progress"
                if failure:
                    break
                time.sleep(limits["monitor_interval_seconds"])
            if failure is None and (run / "STOP").exists():
                failure = "requested_stop"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        failure = f"supervision_error:{type(error).__name__}:{error}"
    finally:
        if process is not None:
            if failure:
                with contextlib.suppress(OSError, ValueError):
                    stop(run)
            try:
                cleanup = _cleanup(process, owned, grace_seconds=20 if failure else 0)
                state["cleanup"] = cleanup
                if cleanup["remaining_processes"]:
                    failure = "owned_process_cleanup_incomplete"
                else:
                    _mark_group_cleaned(run, process)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                failure = f"cleanup_error:{type(error).__name__}:{error}"
    return (
        process.returncode if process is not None and process.returncode is not None else -1,
        failure,
    )


def work(run: Path, inherited_fd: int | None = None) -> dict:
    with _lease(run, inherited_fd) as lease:
        config = verify(run)
        residual = _residual_stage_group(run)
        if residual is not None:
            state = _json(run / "state.json")
            state.update(
                status="needs_astra",
                reason="residual_stage_group_unproven",
                residual_group=residual,
            )
            atomic(run / "state.json", encoded(state))
            return state
        prior = _json(run / "state.json")
        began = prior.get("began_at", time.time())
        _number(began, 0, time.time(), "began_at", integer=False)
        initial_swap = prior.get("initial_swap_bytes")
        initial_swap = _swap_bytes() if initial_swap is None else initial_swap
        _number(initial_swap, 0, 1024**5, "initial_swap_bytes")
        retries = prior.get("retries", {})
        if not isinstance(retries, dict) or any(stage not in STAGES for stage in retries):
            raise ValueError("invalid persisted retries")
        for count in retries.values():
            _number(count, 0, config["resources"]["maximum_retries"], "retry count")
        state = {
            "run_id": config["run_id"],
            "pid": os.getpid(),
            "process_identity": read_process_identity(os.getpid()),
            "began_at": began,
            "initial_swap_bytes": initial_swap,
            "retries": retries,
            "status": "running",
        }
        if state["process_identity"] is None:
            raise RuntimeError("supervisor process identity unavailable")
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_handlers:
            signal.signal(sig, lambda _signum, _frame: stop(run))
        try:
            atomic(run / "state.json", encoded(state))
            for stage in STAGES:
                verify(run)
                if (run / "STOP").exists():
                    state.update(status="stopped", reason="requested_stop")
                    break
                if (run / f"{stage}-complete.json").exists():
                    _verify_completion(run, stage, config)
                    continue
                while True:
                    state.update(stage=stage, status="running", stage_started_at=time.time())
                    state.pop("reason", None)
                    atomic(run / "state.json", encoded(state))
                    returncode, failure = _run_stage(run, stage, config, state, lease)
                    if failure or returncode:
                        if (
                            failure is None
                            and returncode in (-signal.SIGTERM, -signal.SIGINT)
                            and retries.get(stage, 0) < config["resources"]["maximum_retries"]
                        ):
                            retries[stage] = retries.get(stage, 0) + 1
                            atomic(run / "state.json", encoded(state))
                            continue
                        state.update(
                            status="stopped" if failure == "requested_stop" else "needs_astra",
                            reason=failure or f"stage_exit_{returncode}",
                        )
                        break
                    _verify_completion(run, stage, config)
                    break
                if state["status"] != "running":
                    break
            else:
                state.update(
                    status="awaiting_astra_browser", stage="complete", completed_at=time.time()
                )
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            state.update(status="needs_astra", reason=f"{type(error).__name__}:{error}")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            atomic(run / "state.json", encoded(state))
        return state


def _arena_config(run: Path, config: dict) -> dict:
    generation = config["generation"]
    model = run / "fit" / "best.osaval03"
    return {
        **config["evaluation"],
        "seed": config["seed"],
        "probe_path": config["runtime"]["probe"]["path"],
        "probe_sha256": config["runtime"]["probe"]["sha256"],
        "baseline_path": generation["leaf_path"],
        "baseline_sha256": generation["leaf_sha256"],
        "candidate_path": str(model.relative_to(ROOT)),
        "candidate_sha256": digest(model),
        "dataset_path": str((run / "data" / "dataset").relative_to(ROOT)),
    }


def stage_run(run: Path, stage: str, inherited_fd: int | None = None) -> dict:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    with _lease(run, inherited_fd):
        config = verify(run)
        if (run / "STOP").exists():
            raise InterruptedError("stop requested before stage launch")
        if (run / f"{stage}-complete.json").exists():
            return _verify_completion(run, stage, config)
        _register_stage_group(run, stage)
        generation = config["generation"]
        if stage == "generate":
            result = generate(ROOT, run / "data", generation)
            if result.get("games") != generation["games"]:
                raise ValueError("generation stopped before the planned trajectory count")
        elif stage == "prepare":
            result = prepare(ROOT, run / "data", generation, config["excluded_development_sfens"])
        else:
            _dataset(run, config)
            if stage == "train":
                from .evaluator_training import train

                result = train(
                    run / "data" / "dataset",
                    run / "fit",
                    config["training"],
                    {
                        "code_commit": config["code"]["commit"],
                        "run_sha256": digest(run / "run.json"),
                        "dataset_sha256": digest(run / "data" / "dataset" / "manifest.json"),
                    },
                )
                if result["status"] != "complete":
                    raise InterruptedError("training stopped with coherent resume checkpoint")
            elif stage == "audit":
                import torch

                from .evaluator_training import arrays, evaluate
                from .phase10v_model import Phase10VModel, torch_parameters

                _training_summary(run, config)
                torch.set_num_threads(config["training"]["threads"])
                model, report = run / "fit" / "best.osaval03", run / "model-audit.json"
                if not report.exists():
                    subprocess.run(
                        [
                            "node",
                            str(ROOT / "scripts/check_evaluator_model.mjs"),
                            str(inside(config["runtime"]["module"]["path"])),
                            str(model),
                            digest(model),
                            str(inside(config["runtime"]["replay"]["path"])),
                            str(report),
                        ],
                        cwd=ROOT,
                        check=True,
                        timeout=180,
                    )
                _audit_report(run, config)
                test = arrays(run / "data" / "dataset", "development_test")
                metrics = {
                    name: evaluate(
                        torch_parameters(Phase10VModel.read(path)),
                        test,
                        config["training"]["batch_size"],
                    )
                    for name, path in (
                        ("baseline", inside(generation["leaf_path"])),
                        ("candidate", model),
                    )
                }
                atomic(run / "development-test.json", encoded(metrics))
                result = {
                    "model_sha256": digest(model),
                    "audit_sha256": digest(report),
                    "unknown_development_metrics": metrics,
                }
            else:
                from .evaluator_arena import run_arena

                _training_summary(run, config)
                _audit_report(run, config)
                result = run_arena(ROOT, run / "arena", _arena_config(run, config))
                _arena_complete(result)
        verify(run)
        if (run / "STOP").exists():
            raise InterruptedError("stop requested before completion publication")
        receipt = {
            "schema": "open_shogiai_evaluator_stage/v1",
            "stage": stage,
            "run_sha256": digest(run / "run.json"),
            "result": result,
            "artifacts": _completion_artifacts(run, stage),
        }
        atomic(run / f"{stage}-complete.json", encoded(receipt))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "start", "stop", "status", "work", "stage"))
    parser.add_argument("path")
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--lease-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    path = inside(args.path)
    if args.command == "seal":
        result = seal(path)
    elif args.command == "stage":
        result = stage_run(path, args.stage, args.lease_fd)
    elif args.command == "work":
        result = work(path, args.lease_fd)
    else:
        result = {"start": start, "stop": stop, "status": status}[args.command](path)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
