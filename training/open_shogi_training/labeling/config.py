"""Closed, hashed configuration for Phase 4 USI teacher labeling."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from open_shogi_training.labeling.artifacts import ArtifactError, read_regular_bytes

CONFIG_SCHEMA: Final = "phase4_teacher_config/v1"
MAX_CONFIG_BYTES: Final = 64 * 1024
MAX_CONFIG_NODES: Final = 2_048
MAX_CONFIG_DEPTH: Final = 16
MAX_CONFIG_STRING: Final = 4_096
MAX_LABEL_POSITIONS: Final = 10_000
APERY_V2_BINARY_SHA256: Final = "8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403"
APERY_V2_LEGACY_CONFIG_SHA256: Final = (
    "50ecf1b0a3395c0c00208b2d4bad12bb5d16c1f2ec49b8f66fa234a8ab847ba3"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED_OPTIONS = frozenset({"MultiPV", "Threads", "USI_Hash"})


class TeacherConfigError(ValueError):
    """Raised when a teacher configuration is malformed or unsafe."""


class _ClosedSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects aliases and duplicate mapping keys."""

    def compose_node(self, parent: object, index: object) -> object:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise TeacherConfigError(
                f"YAML aliases are not allowed at line {event.start_mark.line + 1}"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, "expected a mapping node", node.start_mark)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise TeacherConfigError(
                    f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True, slots=True)
class EvalFileConfig:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    startup_ms: int
    ready_ms: int
    search_ms: int
    stop_ms: int
    quit_ms: int

    def as_dict(self) -> dict[str, int]:
        return {
            "startup_ms": self.startup_ms,
            "ready_ms": self.ready_ms,
            "search_ms": self.search_ms,
            "stop_ms": self.stop_ms,
            "quit_ms": self.quit_ms,
        }


@dataclass(frozen=True, slots=True)
class ProtocolLimitConfig:
    max_stdout_line_bytes: int
    max_stdout_queue_lines: int
    max_search_lines: int
    max_stderr_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_stdout_line_bytes": self.max_stdout_line_bytes,
            "max_stdout_queue_lines": self.max_stdout_queue_lines,
            "max_search_lines": self.max_search_lines,
            "max_stderr_bytes": self.max_stderr_bytes,
        }


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    seed: str
    max_positions: int
    max_positions_per_game: int
    opening_end_basis_points: int
    middlegame_end_basis_points: int
    max_input_rows: int
    max_input_compressed_bytes: int
    max_input_uncompressed_bytes: int
    max_input_line_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "max_positions": self.max_positions,
            "max_positions_per_game": self.max_positions_per_game,
            "opening_end_basis_points": self.opening_end_basis_points,
            "middlegame_end_basis_points": self.middlegame_end_basis_points,
            "max_input_rows": self.max_input_rows,
            "max_input_compressed_bytes": self.max_input_compressed_bytes,
            "max_input_uncompressed_bytes": self.max_input_uncompressed_bytes,
            "max_input_line_bytes": self.max_input_line_bytes,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    node_candidates: tuple[int, ...]
    positions: int
    max_p95_ms: int
    max_peak_rss_mib: int
    require_peak_rss: bool
    rss_poll_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "node_candidates": list(self.node_candidates),
            "positions": self.positions,
            "max_p95_ms": self.max_p95_ms,
            "max_peak_rss_mib": self.max_peak_rss_mib,
            "require_peak_rss": self.require_peak_rss,
            "rss_poll_ms": self.rss_poll_ms,
        }


@dataclass(frozen=True, slots=True)
class LabelingRunConfig:
    max_retries: int
    max_quarantined: int
    manifest_interval: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_retries": self.max_retries,
            "max_quarantined": self.max_quarantined,
            "manifest_interval": self.manifest_interval,
        }


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    name: str
    version: str
    executable: str
    arguments: tuple[str, ...]
    cwd: str
    install_manifest: str | None
    binary_sha256: str | None
    eval_dir: str
    eval_files: tuple[EvalFileConfig, ...]
    options: tuple[tuple[str, str | int | bool], ...]
    nodes: int
    multipv: int
    threads: int
    hash_mb: int
    concurrency: int
    reference_host_memory_gib: int
    working_memory_limit_gib: int
    timeouts: TimeoutConfig
    protocol_limits: ProtocolLimitConfig
    selection: SelectionConfig
    benchmark: BenchmarkConfig
    labeling: LabelingRunConfig

    @property
    def option_map(self) -> dict[str, str | int | bool]:
        options = dict(self.options)
        options["MultiPV"] = self.multipv
        options["Threads"] = self.threads
        options["USI_Hash"] = self.hash_mb
        return options

    @property
    def runtime_binary_sha256(self) -> str:
        """Return the mandatory execution pin without changing the v1 config digest.

        The completed Apery label set records the historical YAML ``null`` value, so
        changing that serialized field would sever its immutable config identity.  No
        other teacher may use the legacy exception, and every execution path consumes
        this effective non-null pin.
        """

        if self.binary_sha256 is not None:
            return self.binary_sha256
        if (self.name, self.version) == (
            "Apery",
            "2.0.0",
        ) and self.sha256 == APERY_V2_LEGACY_CONFIG_SHA256:
            return APERY_V2_BINARY_SHA256
        raise TeacherConfigError(
            "teacher.binary_sha256 may be null only for the exact immutable Apery 2.0.0 "
            "legacy config"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CONFIG_SCHEMA,
            "teacher": {
                "name": self.name,
                "version": self.version,
                "executable": self.executable,
                "arguments": list(self.arguments),
                "cwd": self.cwd,
                "install_manifest": self.install_manifest,
                "binary_sha256": self.binary_sha256,
                "eval_dir": self.eval_dir,
                "eval_files": [item.as_dict() for item in self.eval_files],
                "options": dict(self.options),
                "nodes": self.nodes,
                "multipv": self.multipv,
                "threads": self.threads,
                "hash_mb": self.hash_mb,
                "concurrency": self.concurrency,
                "reference_host_memory_gib": self.reference_host_memory_gib,
                "working_memory_limit_gib": self.working_memory_limit_gib,
            },
            "timeouts": self.timeouts.as_dict(),
            "protocol_limits": self.protocol_limits.as_dict(),
            "selection": self.selection.as_dict(),
            "benchmark": self.benchmark.as_dict(),
            "labeling": self.labeling.as_dict(),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_teacher_config(path: Path) -> TeacherConfig:
    """Load a bounded closed YAML config without aliases or duplicate keys."""

    raw = _read_bounded_config(path)
    try:
        payload = yaml.load(raw, Loader=_ClosedSafeLoader)
    except TeacherConfigError:
        raise
    except yaml.YAMLError as error:
        raise TeacherConfigError(f"invalid teacher YAML: {error}") from error
    _validate_node_budget(payload)
    root = _closed_mapping(
        payload,
        "config",
        {"schema", "teacher", "timeouts", "protocol_limits", "selection", "benchmark", "labeling"},
    )
    if root["schema"] != CONFIG_SCHEMA:
        raise TeacherConfigError(f"schema must be {CONFIG_SCHEMA!r}")

    teacher = _closed_mapping(
        root["teacher"],
        "teacher",
        {
            "name",
            "version",
            "executable",
            "arguments",
            "cwd",
            "install_manifest",
            "binary_sha256",
            "eval_dir",
            "eval_files",
            "options",
            "nodes",
            "multipv",
            "threads",
            "hash_mb",
            "concurrency",
            "reference_host_memory_gib",
            "working_memory_limit_gib",
        },
    )
    timeouts = _closed_mapping(
        root["timeouts"],
        "timeouts",
        {"startup_ms", "ready_ms", "search_ms", "stop_ms", "quit_ms"},
    )
    limits = _closed_mapping(
        root["protocol_limits"],
        "protocol_limits",
        {
            "max_stdout_line_bytes",
            "max_stdout_queue_lines",
            "max_search_lines",
            "max_stderr_bytes",
        },
    )
    selection = _closed_mapping(
        root["selection"],
        "selection",
        {
            "seed",
            "max_positions",
            "max_positions_per_game",
            "opening_end_basis_points",
            "middlegame_end_basis_points",
            "max_input_rows",
            "max_input_compressed_bytes",
            "max_input_uncompressed_bytes",
            "max_input_line_bytes",
        },
    )
    benchmark = _closed_mapping(
        root["benchmark"],
        "benchmark",
        {
            "node_candidates",
            "positions",
            "max_p95_ms",
            "max_peak_rss_mib",
            "require_peak_rss",
            "rss_poll_ms",
        },
    )
    labeling = _closed_mapping(
        root["labeling"],
        "labeling",
        {"max_retries", "max_quarantined", "manifest_interval"},
    )

    options_raw = _mapping(teacher["options"], "teacher.options")
    options: list[tuple[str, str | int | bool]] = []
    for key, value in options_raw.items():
        name = _safe_text(key, "teacher.options key", max_length=128)
        if name in _RESERVED_OPTIONS:
            raise TeacherConfigError(f"teacher.options must not redefine reserved option {name!r}")
        if not isinstance(value, (str, int, bool)):
            raise TeacherConfigError(
                f"teacher.options.{name} must be a string, integer, or boolean"
            )
        if isinstance(value, str):
            _safe_text(value, f"teacher.options.{name}", max_length=1_024, allow_empty=True)
        options.append((name, value))

    arguments_raw = _sequence(teacher["arguments"], "teacher.arguments", maximum=32)
    arguments = tuple(
        _safe_text(item, f"teacher.arguments[{index}]", max_length=1_024, allow_empty=True)
        for index, item in enumerate(arguments_raw)
    )
    eval_raw = _sequence(teacher["eval_files"], "teacher.eval_files", maximum=64)
    if not eval_raw:
        raise TeacherConfigError("teacher.eval_files must not be empty")
    eval_files: list[EvalFileConfig] = []
    seen_eval_paths: set[str] = set()
    for index, item in enumerate(eval_raw):
        entry = _closed_mapping(item, f"teacher.eval_files[{index}]", {"path", "sha256"})
        eval_path = _safe_relative_path(entry["path"], f"teacher.eval_files[{index}].path")
        if eval_path in seen_eval_paths:
            raise TeacherConfigError(f"duplicate eval path: {eval_path}")
        seen_eval_paths.add(eval_path)
        eval_files.append(
            EvalFileConfig(
                path=eval_path,
                sha256=_sha256(entry["sha256"], f"teacher.eval_files[{index}].sha256"),
            )
        )

    timeout_config = TimeoutConfig(
        startup_ms=_bounded_int(timeouts["startup_ms"], "timeouts.startup_ms", 1, 600_000),
        ready_ms=_bounded_int(timeouts["ready_ms"], "timeouts.ready_ms", 1, 600_000),
        search_ms=_bounded_int(timeouts["search_ms"], "timeouts.search_ms", 1, 3_600_000),
        stop_ms=_bounded_int(timeouts["stop_ms"], "timeouts.stop_ms", 1, 60_000),
        quit_ms=_bounded_int(timeouts["quit_ms"], "timeouts.quit_ms", 1, 60_000),
    )
    limit_config = ProtocolLimitConfig(
        max_stdout_line_bytes=_bounded_int(
            limits["max_stdout_line_bytes"],
            "protocol_limits.max_stdout_line_bytes",
            256,
            1_048_576,
        ),
        max_stdout_queue_lines=_bounded_int(
            limits["max_stdout_queue_lines"],
            "protocol_limits.max_stdout_queue_lines",
            8,
            65_536,
        ),
        max_search_lines=_bounded_int(
            limits["max_search_lines"],
            "protocol_limits.max_search_lines",
            16,
            1_000_000,
        ),
        max_stderr_bytes=_bounded_int(
            limits["max_stderr_bytes"],
            "protocol_limits.max_stderr_bytes",
            256,
            1_048_576,
        ),
    )
    selection_config = SelectionConfig(
        seed=_safe_text(selection["seed"], "selection.seed", max_length=256),
        max_positions=_bounded_int(
            selection["max_positions"], "selection.max_positions", 1, MAX_LABEL_POSITIONS
        ),
        max_positions_per_game=_bounded_int(
            selection["max_positions_per_game"],
            "selection.max_positions_per_game",
            1,
            MAX_LABEL_POSITIONS,
        ),
        opening_end_basis_points=_bounded_int(
            selection["opening_end_basis_points"],
            "selection.opening_end_basis_points",
            1,
            9_998,
        ),
        middlegame_end_basis_points=_bounded_int(
            selection["middlegame_end_basis_points"],
            "selection.middlegame_end_basis_points",
            2,
            9_999,
        ),
        max_input_rows=_bounded_int(
            selection["max_input_rows"], "selection.max_input_rows", 1, 1_000_000
        ),
        max_input_compressed_bytes=_bounded_int(
            selection["max_input_compressed_bytes"],
            "selection.max_input_compressed_bytes",
            1,
            1_073_741_824,
        ),
        max_input_uncompressed_bytes=_bounded_int(
            selection["max_input_uncompressed_bytes"],
            "selection.max_input_uncompressed_bytes",
            1,
            4_294_967_296,
        ),
        max_input_line_bytes=_bounded_int(
            selection["max_input_line_bytes"],
            "selection.max_input_line_bytes",
            256,
            16_777_216,
        ),
    )
    if selection_config.opening_end_basis_points >= selection_config.middlegame_end_basis_points:
        raise TeacherConfigError("selection opening boundary must be below the middlegame boundary")

    candidates_raw = _sequence(
        benchmark["node_candidates"], "benchmark.node_candidates", maximum=16
    )
    candidates = tuple(
        _bounded_int(value, f"benchmark.node_candidates[{index}]", 1, 10_000_000_000)
        for index, value in enumerate(candidates_raw)
    )
    if not candidates or tuple(sorted(set(candidates))) != candidates:
        raise TeacherConfigError(
            "benchmark.node_candidates must be non-empty, unique, and ascending"
        )
    benchmark_config = BenchmarkConfig(
        node_candidates=candidates,
        positions=_bounded_int(benchmark["positions"], "benchmark.positions", 1, 100),
        max_p95_ms=_bounded_int(benchmark["max_p95_ms"], "benchmark.max_p95_ms", 1, 3_600_000),
        max_peak_rss_mib=_bounded_int(
            benchmark["max_peak_rss_mib"], "benchmark.max_peak_rss_mib", 1, 20_000
        ),
        require_peak_rss=_required_bool(
            benchmark["require_peak_rss"], "benchmark.require_peak_rss"
        ),
        rss_poll_ms=_bounded_int(benchmark["rss_poll_ms"], "benchmark.rss_poll_ms", 10, 5_000),
    )
    run_config = LabelingRunConfig(
        max_retries=_bounded_int(labeling["max_retries"], "labeling.max_retries", 0, 10),
        max_quarantined=_bounded_int(
            labeling["max_quarantined"], "labeling.max_quarantined", 1, MAX_LABEL_POSITIONS
        ),
        manifest_interval=_bounded_int(
            labeling["manifest_interval"], "labeling.manifest_interval", 1, 1_000
        ),
    )

    result = TeacherConfig(
        name=_safe_text(teacher["name"], "teacher.name", max_length=128),
        version=_safe_text(teacher["version"], "teacher.version", max_length=64),
        executable=_safe_relative_path(teacher["executable"], "teacher.executable"),
        arguments=arguments,
        cwd=_safe_relative_path(teacher["cwd"], "teacher.cwd"),
        install_manifest=_optional_relative_path(
            teacher["install_manifest"], "teacher.install_manifest"
        ),
        binary_sha256=_optional_sha256(teacher["binary_sha256"], "teacher.binary_sha256"),
        eval_dir=_safe_relative_path(teacher["eval_dir"], "teacher.eval_dir"),
        eval_files=tuple(eval_files),
        options=tuple(sorted(options)),
        nodes=_bounded_int(teacher["nodes"], "teacher.nodes", 1, 10_000_000_000),
        multipv=_bounded_int(teacher["multipv"], "teacher.multipv", 1, 500),
        threads=_bounded_int(teacher["threads"], "teacher.threads", 1, 256),
        hash_mb=_bounded_int(teacher["hash_mb"], "teacher.hash_mb", 1, 16_384),
        concurrency=_bounded_int(teacher["concurrency"], "teacher.concurrency", 1, 1),
        reference_host_memory_gib=_bounded_int(
            teacher["reference_host_memory_gib"],
            "teacher.reference_host_memory_gib",
            1,
            1_024,
        ),
        working_memory_limit_gib=_bounded_int(
            teacher["working_memory_limit_gib"],
            "teacher.working_memory_limit_gib",
            1,
            1_024,
        ),
        timeouts=timeout_config,
        protocol_limits=limit_config,
        selection=selection_config,
        benchmark=benchmark_config,
        labeling=run_config,
    )
    if result.working_memory_limit_gib >= result.reference_host_memory_gib:
        raise TeacherConfigError("teacher.working_memory_limit_gib must leave host memory headroom")
    if result.nodes not in result.benchmark.node_candidates:
        raise TeacherConfigError("teacher.nodes must appear in benchmark.node_candidates")
    if result.benchmark.positions > result.selection.max_positions:
        raise TeacherConfigError("benchmark.positions must not exceed selection.max_positions")
    # Resolve the legacy exception while loading so an unpinned generic teacher is
    # rejected before any filesystem access or subprocess creation.
    if len(result.runtime_binary_sha256) != 64:
        raise AssertionError("resolved runtime binary pin must be a SHA-256")
    return result


def resolve_config_path(project_root: Path, value: str, *, field: str) -> Path:
    """Resolve a validated path beneath the root without traversing symlinks."""

    root = project_root.resolve(strict=True)
    lexical = root / value
    cursor = root
    for component in Path(value).parts:
        cursor /= component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as error:
            raise TeacherConfigError(f"cannot inspect {field}: {error}") from error
        if stat.S_ISLNK(mode):
            raise TeacherConfigError(f"{field} must not traverse symlinks")
    candidate = lexical.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise TeacherConfigError(f"{field} escapes the project root") from error
    return candidate


def _read_bounded_config(path: Path) -> bytes:
    try:
        raw, _ = read_regular_bytes(path, max_bytes=MAX_CONFIG_BYTES)
    except ArtifactError as error:
        raise TeacherConfigError(f"cannot read teacher config: {error}") from error
    return raw


def _validate_node_budget(value: object) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_CONFIG_NODES:
            raise TeacherConfigError(f"teacher config exceeds {MAX_CONFIG_NODES} nodes")
        if depth > MAX_CONFIG_DEPTH:
            raise TeacherConfigError(f"teacher config exceeds depth {MAX_CONFIG_DEPTH}")
        if isinstance(item, str) and len(item) > MAX_CONFIG_STRING:
            raise TeacherConfigError("teacher config contains an oversized string")
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)


def _closed_mapping(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    result = _mapping(value, name)
    actual = set(result)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys, key=str)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise TeacherConfigError(f"{name} keys are invalid ({', '.join(details)})")
    if not all(isinstance(key, str) for key in result):
        raise TeacherConfigError(f"{name} keys must be strings")
    return result


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TeacherConfigError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise TeacherConfigError(f"{name} must be a sequence")
    if len(value) > maximum:
        raise TeacherConfigError(f"{name} exceeds {maximum} entries")
    return value


def _safe_text(
    value: object,
    name: str,
    *,
    max_length: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TeacherConfigError(f"{name} must be a string")
    if not allow_empty and not value:
        raise TeacherConfigError(f"{name} must not be empty")
    if len(value) > max_length:
        raise TeacherConfigError(f"{name} exceeds {max_length} characters")
    if any(character in value for character in "\r\n\x00"):
        raise TeacherConfigError(f"{name} contains a command delimiter")
    return value


def _safe_relative_path(value: object, name: str) -> str:
    text = _safe_text(value, name, max_length=1_024)
    path = Path(text)
    if path.is_absolute() or text.startswith("~") or any(part == ".." for part in path.parts):
        raise TeacherConfigError(f"{name} must be a project-relative non-traversing path")
    if any(part in {"", "."} for part in path.parts):
        raise TeacherConfigError(f"{name} contains an ambiguous path component")
    return path.as_posix()


def _optional_relative_path(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _safe_relative_path(value, name)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TeacherConfigError(f"{name} must be a lowercase SHA-256")
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TeacherConfigError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise TeacherConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _required_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TeacherConfigError(f"{name} must be a boolean")
    return value
