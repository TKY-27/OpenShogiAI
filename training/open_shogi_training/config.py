"""Typed loading for the repository-level project configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated Phase 0 settings shared by future training commands."""

    schema_version: int
    name: str
    version: str
    default_seed: int
    reference_host_memory_gib: int
    working_memory_limit_gib: int


def load_project_config(path: Path) -> ProjectConfig:
    """Load and validate the stable subset of ``configs/project.toml``.

    Raises:
        ValueError: If a required field has an invalid type or the memory limit is unsafe.
        OSError: If the configuration cannot be read.
        tomllib.TOMLDecodeError: If the configuration is not valid TOML.
    """

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    schema_version = _required_int(raw, "schema_version")
    project = _required_table(raw, "project")
    reproducibility = _required_table(raw, "reproducibility")
    resources = _required_table(raw, "resources")

    name = _required_string(project, "name")
    version = _required_string(project, "version")
    default_seed = _required_int(reproducibility, "default_seed")
    reference_memory = _required_positive_int(resources, "reference_host_memory_gib")
    working_memory = _required_positive_int(resources, "working_memory_limit_gib")

    if working_memory >= reference_memory:
        message = "working_memory_limit_gib must leave headroom below reference_host_memory_gib"
        raise ValueError(message)

    return ProjectConfig(
        schema_version=schema_version,
        name=name,
        version=version,
        default_seed=default_seed,
        reference_host_memory_gib=reference_memory,
        working_memory_limit_gib=working_memory,
    )


def _required_table(table: dict[str, Any], key: str) -> dict[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return value


def _required_string(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_positive_int(table: dict[str, Any], key: str) -> int:
    value = _required_int(table, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value
