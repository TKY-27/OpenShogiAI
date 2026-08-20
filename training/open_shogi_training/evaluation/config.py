"""Closed Phase 7 evaluation configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from open_shogi_training.labeling.artifacts import ArtifactError, read_regular_bytes
from open_shogi_training.selfplay.common import canonical_sha256, validate_identifier
from open_shogi_training.selfplay.common import validate_relative_path as _relative_path

CONFIG_SCHEMA: Final = "phase7_evaluation_config/v1"
MAX_CONFIG_BYTES: Final = 64 * 1024


class EvaluationConfigError(ValueError):
    """Raised when the Phase 7 configuration is malformed or unbounded."""


@dataclass(frozen=True, slots=True)
class OfficialModeConfig:
    run_id: str
    profile: str
    human_sides: tuple[str, ...]
    nodes: int
    depth: int
    max_plies: int
    opening_enabled: bool
    opening_max_plies: int

    def as_dict(self) -> dict[str, object]:
        return {
            "runId": self.run_id,
            "profile": self.profile,
            "humanSides": list(self.human_sides),
            "nodes": self.nodes,
            "depth": self.depth,
            "maxPlies": self.max_plies,
            "openingEnabled": self.opening_enabled,
            "openingMaxPlies": self.opening_max_plies,
        }


@dataclass(frozen=True, slots=True)
class TeacherAnalysisConfig:
    config_path: str
    nodes: int

    def as_dict(self) -> dict[str, object]:
        return {"configPath": self.config_path, "nodes": self.nodes}


@dataclass(frozen=True, slots=True)
class DiagnosisConfig:
    move_regret_cp: int
    evaluation_disagreement_cp: int
    hard_example_cp: int
    max_hard_examples: int

    def as_dict(self) -> dict[str, int]:
        return {
            "moveRegretCp": self.move_regret_cp,
            "evaluationDisagreementCp": self.evaluation_disagreement_cp,
            "hardExampleCp": self.hard_example_cp,
            "maxHardExamples": self.max_hard_examples,
        }


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    memory_limit_mib: int
    max_teacher_calls: int

    def as_dict(self) -> dict[str, int]:
        return {
            "memoryLimitMiB": self.memory_limit_mib,
            "maxTeacherCalls": self.max_teacher_calls,
        }


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    official: OfficialModeConfig
    teacher: TeacherAnalysisConfig
    diagnosis: DiagnosisConfig
    resources: ResourceConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CONFIG_SCHEMA,
            "official": self.official.as_dict(),
            "teacher": self.teacher.as_dict(),
            "diagnosis": self.diagnosis.as_dict(),
            "resources": self.resources.as_dict(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load one bounded TOML document and reject every unrecognized field."""

    try:
        raw, _ = read_regular_bytes(path, max_bytes=MAX_CONFIG_BYTES)
        value = tomllib.loads(raw.decode("utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise EvaluationConfigError(f"cannot load evaluation config: {error}") from error
    root = _closed(
        value, "evaluation config", {"schema", "official", "teacher", "diagnosis", "resources"}
    )
    if root["schema"] != CONFIG_SCHEMA:
        raise EvaluationConfigError(f"schema must be {CONFIG_SCHEMA!r}")
    official_raw = _closed(
        root["official"],
        "official",
        {
            "run_id",
            "profile",
            "human_sides",
            "nodes",
            "depth",
            "max_plies",
            "opening_enabled",
            "opening_max_plies",
        },
    )
    teacher_raw = _closed(root["teacher"], "teacher", {"config_path", "nodes"})
    diagnosis_raw = _closed(
        root["diagnosis"],
        "diagnosis",
        {
            "move_regret_cp",
            "evaluation_disagreement_cp",
            "hard_example_cp",
            "max_hard_examples",
        },
    )
    resources_raw = _closed(
        root["resources"], "resources", {"memory_limit_mib", "max_teacher_calls"}
    )

    run_id = _text(official_raw["run_id"], "official.run_id", maximum=96)
    try:
        validate_identifier(run_id, "official.run_id")
    except ValueError as error:
        raise EvaluationConfigError(str(error)) from error
    profile = _text(official_raw["profile"], "official.profile", maximum=32)
    if profile != "champion":
        raise EvaluationConfigError("official.profile must be 'champion'")
    sides = tuple(
        _text(item, f"official.human_sides[{index}]", maximum=8)
        for index, item in enumerate(
            _sequence(official_raw["human_sides"], "official.human_sides", maximum=2)
        )
    )
    if sides != ("black", "white"):
        raise EvaluationConfigError("official.human_sides must be exactly ['black', 'white']")
    opening_enabled = official_raw["opening_enabled"]
    if opening_enabled is not False:
        raise EvaluationConfigError("official.opening_enabled must be false")
    official = OfficialModeConfig(
        run_id=run_id,
        profile=profile,
        human_sides=sides,
        nodes=_integer(official_raw["nodes"], "official.nodes", 1, 1_000_000_000),
        depth=_integer(official_raw["depth"], "official.depth", 1, 64),
        max_plies=_integer(official_raw["max_plies"], "official.max_plies", 1, 512),
        opening_enabled=False,
        opening_max_plies=_integer(
            official_raw["opening_max_plies"], "official.opening_max_plies", 1, 512
        ),
    )
    teacher = TeacherAnalysisConfig(
        config_path=_path(teacher_raw["config_path"], "teacher.config_path"),
        nodes=_integer(teacher_raw["nodes"], "teacher.nodes", 1, 10_000_000_000),
    )
    diagnosis = DiagnosisConfig(
        move_regret_cp=_integer(
            diagnosis_raw["move_regret_cp"], "diagnosis.move_regret_cp", 1, 1_000_000
        ),
        evaluation_disagreement_cp=_integer(
            diagnosis_raw["evaluation_disagreement_cp"],
            "diagnosis.evaluation_disagreement_cp",
            1,
            1_000_000,
        ),
        hard_example_cp=_integer(
            diagnosis_raw["hard_example_cp"], "diagnosis.hard_example_cp", 1, 1_000_000
        ),
        max_hard_examples=_integer(
            diagnosis_raw["max_hard_examples"], "diagnosis.max_hard_examples", 1, 512
        ),
    )
    resources = ResourceConfig(
        memory_limit_mib=_integer(
            resources_raw["memory_limit_mib"], "resources.memory_limit_mib", 128, 8_192
        ),
        max_teacher_calls=_integer(
            resources_raw["max_teacher_calls"], "resources.max_teacher_calls", 1, 1_024
        ),
    )
    maximum_possible_calls = len(sides) * official.max_plies
    if resources.max_teacher_calls < maximum_possible_calls:
        raise EvaluationConfigError(
            "resources.max_teacher_calls must cover every possible official-game move"
        )
    return EvaluationConfig(
        official=official,
        teacher=teacher,
        diagnosis=diagnosis,
        resources=resources,
    )


def _closed(value: object, context: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EvaluationConfigError(f"{context} must contain exactly {sorted(keys)}")
    if any(not isinstance(key, str) for key in value):
        raise EvaluationConfigError(f"{context} keys must be strings")
    return value


def _text(value: object, context: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise EvaluationConfigError(f"{context} must be a bounded single-line string")
    return value


def _path(value: object, context: str) -> str:
    text = _text(value, context, maximum=1_024)
    try:
        return _relative_path(text)
    except ValueError as error:
        raise EvaluationConfigError(f"{context}: {error}") from error


def _integer(value: object, context: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise EvaluationConfigError(f"{context} must be in {minimum}..{maximum}")
    return value


def _sequence(value: object, context: str, *, maximum: int) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list) or len(value) > maximum:
        raise EvaluationConfigError(f"{context} must be a list with at most {maximum} entries")
    return value
