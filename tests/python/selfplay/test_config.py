from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.selfplay.common import ContractError
from open_shogi_training.selfplay.config import load_generation_policy, load_selfplay_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_repository_phase6_configs_are_strict_and_bounded() -> None:
    config = load_selfplay_config(PROJECT_ROOT / "configs" / "selfplay" / "phase6_smoke.toml")
    policy = load_generation_policy(
        PROJECT_ROOT / "configs" / "generation" / "phase6_promotion.toml"
    )

    assert config.run.games == 40
    assert config.run.normal_start_pairs + config.run.start_set_pairs == 20
    assert config.resources.workers * config.resources.memory_per_worker_mib <= 8192
    assert policy.evidence.minimum_games >= 40
    assert policy.thresholds.promote_wilson_lower >= 0.5


def test_config_rejects_unknown_keys_and_unsafe_worker_memory(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        (PROJECT_ROOT / "configs" / "selfplay" / "phase6_smoke.toml")
        .read_text(encoding="utf-8")
        .replace("workers = 2", "workers = 4")
        .replace("memory_per_worker_mib = 3072", "memory_per_worker_mib = 4096")
        .replace("games = 40", "games = 40\nunknown = 1"),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="keys mismatch"):
        load_selfplay_config(path)


def test_config_rejects_nonpaired_game_partition(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        (PROJECT_ROOT / "configs" / "selfplay" / "phase6_smoke.toml")
        .read_text(encoding="utf-8")
        .replace("normal_start_pairs = 10", "normal_start_pairs = 11"),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="must equal half"):
        load_selfplay_config(path)
