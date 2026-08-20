from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.evaluation.config import (
    EvaluationConfigError,
    load_evaluation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/evaluation/phase7_official.toml"


def test_checked_in_phase7_config_is_closed_bounded_and_hashable() -> None:
    config = load_evaluation_config(CONFIG_PATH)

    assert config.official.human_sides == ("black", "white")
    assert config.official.profile == "champion"
    assert config.official.opening_enabled is False
    assert config.official.nodes == 10_000
    assert config.teacher.nodes == 25_000
    assert config.resources.memory_limit_mib == 4_096
    assert config.resources.max_teacher_calls == 512
    assert len(config.sha256) == 64


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('profile = "champion"', 'profile = "challenger"', "champion"),
        (
            'human_sides = ["black", "white"]',
            'human_sides = ["white", "black"]',
            "exactly",
        ),
        ("opening_enabled = false", "opening_enabled = true", "must be false"),
        ("max_teacher_calls = 512", "max_teacher_calls = 511", "cover every"),
    ],
)
def test_config_rejects_nonofficial_or_unbounded_modes(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    candidate = tmp_path / "evaluation.toml"
    candidate.write_text(CONFIG_PATH.read_text(encoding="utf-8").replace(old, new))

    with pytest.raises(EvaluationConfigError, match=message):
        load_evaluation_config(candidate)


def test_config_rejects_unknown_fields_and_path_traversal(tmp_path: Path) -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(source.replace("[resources]", "mystery = 1\n\n[resources]"))
    with pytest.raises(EvaluationConfigError, match="exactly"):
        load_evaluation_config(unknown)

    traversal = tmp_path / "traversal.toml"
    traversal.write_text(
        source.replace(
            'config_path = "configs/teacher/apery-v2.0.0.yaml"',
            'config_path = "../teacher.yaml"',
        )
    )
    with pytest.raises(EvaluationConfigError, match=r"parent|travers"):
        load_evaluation_config(traversal)


def test_config_rejects_symlink_and_duplicate_key(tmp_path: Path) -> None:
    symlink = tmp_path / "evaluation.toml"
    symlink.symlink_to(CONFIG_PATH)
    with pytest.raises(EvaluationConfigError, match=r"symbolic|symlink"):
        load_evaluation_config(symlink)

    duplicate = tmp_path / "duplicate.toml"
    source = CONFIG_PATH.read_text()
    schema_line = 'schema = "phase7_evaluation_config/v1"'
    duplicate.write_text(source.replace(schema_line, f"{schema_line}\n{schema_line}", 1))
    with pytest.raises(EvaluationConfigError, match="cannot load"):
        load_evaluation_config(duplicate)
