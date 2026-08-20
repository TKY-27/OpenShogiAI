from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.labeling.config import TeacherConfigError, load_teacher_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_checked_in_apery_config_is_closed_bounded_and_hashable() -> None:
    config = load_teacher_config(PROJECT_ROOT / "configs/teacher/apery-v2.0.0.yaml")

    assert config.name == "Apery"
    assert config.version == "2.0.0"
    assert config.nodes == 25_000
    assert config.multipv == 3
    assert config.concurrency == 1
    assert config.benchmark.node_candidates == (10_000, 25_000)
    assert config.option_map["USI_Hash"] == 1024
    assert config.option_map["Threads"] == 4
    assert len(config.sha256) == 64


def test_config_rejects_duplicate_keys_aliases_and_reserved_option(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema: one\nschema: two\n", encoding="utf-8")
    with pytest.raises(TeacherConfigError, match="duplicate YAML key"):
        load_teacher_config(duplicate)

    alias = tmp_path / "alias.yaml"
    alias.write_text("schema: &value one\nteacher: *value\n", encoding="utf-8")
    with pytest.raises(TeacherConfigError, match="aliases are not allowed"):
        load_teacher_config(alias)

    source = (PROJECT_ROOT / "configs/teacher/apery-v2.0.0.yaml").read_text(encoding="utf-8")
    reserved = tmp_path / "reserved.yaml"
    reserved.write_text(
        source.replace("  options:\n", "  options:\n    MultiPV: 99\n"),
        encoding="utf-8",
    )
    with pytest.raises(TeacherConfigError, match="reserved option"):
        load_teacher_config(reserved)


def test_config_rejects_unbenchmarked_nodes_and_path_traversal(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/teacher/apery-v2.0.0.yaml").read_text(encoding="utf-8")
    unbenchmarked = tmp_path / "unbenchmarked.yaml"
    unbenchmarked.write_text(source.replace("  nodes: 25000", "  nodes: 30000"), encoding="utf-8")
    with pytest.raises(TeacherConfigError, match="must appear"):
        load_teacher_config(unbenchmarked)

    traversal = tmp_path / "traversal.yaml"
    traversal.write_text(
        source.replace(
            "  executable: local/teacher/build/apery-v2.0.0/",
            "  executable: ../",
        ),
        encoding="utf-8",
    )
    with pytest.raises(TeacherConfigError, match="non-traversing"):
        load_teacher_config(traversal)
