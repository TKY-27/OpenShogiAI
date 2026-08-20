from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import open_shogi_training.data.__main__ as cli_module
import pytest
from open_shogi_training.data.__main__ import main
from open_shogi_training.selfplay.common import ArtifactRef

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "data_sources.yaml"


def test_dry_run_cli_lists_exact_100_without_network_or_disk_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = tuple(tmp_path.iterdir())

    result = main(
        [
            "dry-run",
            "--registry",
            str(REGISTRY_PATH),
            "--source",
            "aobazero-no-noise",
            "--limit",
            "100",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["mode"] == "dry-run"
    assert payload["count"] == 100
    assert payload["objects"][0]["filename"] == "w4745.csa"
    assert payload["objects"][-1]["filename"] == "w4235.csa"
    assert tuple(tmp_path.iterdir()) == before


def test_dry_run_cli_refuses_unapproved_floodgate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "dry-run",
            "--registry",
            str(REGISTRY_PATH),
            "--source",
            "floodgate",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "disabled" in captured.err


def test_acquire_cli_requires_explicit_sample_only() -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "acquire",
                "--registry",
                str(REGISTRY_PATH),
                "--source",
                "aobazero-no-noise",
                "--output",
                "/tmp/not-used",
            ]
        )

    assert error.value.code == 2


def test_validate_registry_cli_reports_gates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(["validate-registry", "--registry", str(REGISTRY_PATH)])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["sources"] == [
        {
            "source_id": "aobazero-no-noise",
            "enabled": True,
            "approved": True,
            "catalog_objects": 100,
            "evidence_objects": 5,
        },
        {
            "source_id": "floodgate",
            "enabled": False,
            "approved": False,
            "catalog_objects": 0,
            "evidence_objects": 0,
        },
    ]


def test_normalize_cli_passes_bounded_config_without_echoing_split_salt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_normalize(*args: object, **kwargs: object) -> object:
        observed["args"] = args
        observed["kwargs"] = kwargs
        output = tmp_path / "processed" / "aobazero-no-noise-sample"
        return SimpleNamespace(
            output_dir=output,
            manifest_path=output / "manifest.json",
            report_path=output / "normalization-report.json",
            included_games=100,
            included_positions=1234,
            excluded_games=0,
        )

    monkeypatch.setattr(cli_module, "normalize_aobazero_dataset", fake_normalize)
    engine = ArtifactRef(
        "local/builds/open-shogi-cli/" + "1" * 64 + "/open-shogi-cli",
        "1" * 64,
        1,
    )
    receipt = ArtifactRef(
        "local/build-receipts/open-shogi-cli/" + "2" * 64 + ".json",
        "2" * 64,
        1,
    )
    monkeypatch.setattr(
        cli_module,
        "_resolve_normalization_engine",
        lambda *_args: (engine, receipt),
    )
    split_salt = "phase3-public-reproducibility-salt"
    result = main(
        [
            "normalize",
            "--registry",
            str(REGISTRY_PATH),
            "--source",
            "aobazero-no-noise",
            "--manifest",
            str(tmp_path / "raw" / "manifest.jsonl"),
            "--acquisition-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--dataset-id",
            "aobazero-no-noise-sample",
            "--split-salt",
            split_salt,
            "--cli",
            "target/release/open-shogi-cli",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert split_salt not in captured.out
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    config = kwargs["config"]
    assert isinstance(config, cli_module.NormalizationConfig)
    assert config.max_games == 100
    assert config.max_positions == 100_000
    assert config.split_policy.salt == split_salt
    assert kwargs["engine"] == engine
    assert kwargs["engine_build_receipt"] == receipt
    assert kwargs["repository_root"] == PROJECT_ROOT


def test_normalize_cli_rejects_an_alias_that_differs_from_the_immutable_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias_path = tmp_path / "target/release/open-shogi-cli"
    alias_path.parent.mkdir(parents=True)
    alias_path.write_bytes(b"mutable alias")
    engine = ArtifactRef(
        "local/builds/open-shogi-cli/" + "1" * 64 + "/open-shogi-cli",
        "1" * 64,
        1,
    )
    receipt = ArtifactRef(
        "local/build-receipts/open-shogi-cli/" + "2" * 64 + ".json",
        "2" * 64,
        1,
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_active_engine_build",
        lambda *_args, **_kwargs: (engine, receipt, {}),
    )

    with pytest.raises(ValueError, match="alias differs"):
        cli_module._resolve_normalization_engine(
            tmp_path,
            Path("target/release/open-shogi-cli"),
        )

    with pytest.raises(ValueError, match="fixed operator alias"):
        cli_module._resolve_normalization_engine(tmp_path, Path("local/other-engine"))
