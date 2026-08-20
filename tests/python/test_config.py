from pathlib import Path

import pytest
from open_shogi_training import load_project_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_config_loads_with_safe_memory_headroom() -> None:
    config = load_project_config(PROJECT_ROOT / "configs" / "project.toml")

    assert config.schema_version == 1
    assert config.name == "OpenShogiAI"
    assert config.working_memory_limit_gib < config.reference_host_memory_gib


def test_config_rejects_memory_limit_without_headroom(tmp_path: Path) -> None:
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        """
schema_version = 1
[project]
name = "OpenShogiAI"
version = "0.0.0"
[reproducibility]
default_seed = 1
[resources]
reference_host_memory_gib = 24
working_memory_limit_gib = 24
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must leave headroom"):
        load_project_config(config_path)
