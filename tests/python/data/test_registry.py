from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from open_shogi_training.data.downloader import AcquisitionError, plan_acquisition
from open_shogi_training.data.registry import (
    MAX_REGISTRY_BYTES,
    RegistryError,
    load_source_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "data_sources.yaml"


def _copy_registry_tree(tmp_path: Path) -> Path:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    audits = tmp_path / "docs" / "source-audits"
    audits.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "docs" / "source-audits" / "aobazero.md", audits)
    shutil.copy(PROJECT_ROOT / "docs" / "source-audits" / "floodgate.md", audits)
    return tmp_path / "configs" / "data_sources.yaml"


def test_repository_registry_approves_only_exact_aobazero_sample() -> None:
    registry = load_source_registry(REGISTRY_PATH)
    aobazero = registry.get("aobazero-no-noise")
    floodgate = registry.get("floodgate")

    assert aobazero.enabled is True
    assert aobazero.approved is True
    assert aobazero.adapter == "aobazero_csa"
    assert len(aobazero.catalog) == 100
    expected_numbers = [
        number for number in range(4745, 4234, -5) if number not in {4495, 4425, 4420}
    ]
    assert [item.object_id for item in aobazero.catalog] == [
        f"w{number}" for number in expected_numbers
    ]
    assert [item.filename for item in aobazero.catalog] == [
        f"w{number}.csa" for number in expected_numbers
    ]
    assert [item.url for item in aobazero.catalog] == [
        f"http://www.yss-aya.com/aobazero/no_noise/w{number}.csa" for number in expected_numbers
    ]
    assert len(aobazero.evidence_catalog) == 5
    assert all(
        item.sha256 is not None
        for item in aobazero.evidence_catalog
        if item.url != aobazero.robots_url
    )
    assert floodgate.enabled is False
    assert floodgate.approved is False
    assert floodgate.catalog == ()


def test_dry_run_is_exactly_bounded_and_does_not_create_files(tmp_path: Path) -> None:
    source = load_source_registry(REGISTRY_PATH).get("aobazero-no-noise")
    before = tuple(tmp_path.iterdir())

    plan = plan_acquisition(source, limit=100)

    assert len(plan.objects) == 100
    assert tuple(tmp_path.iterdir()) == before
    with pytest.raises(AcquisitionError, match="between 1 and 100"):
        plan_acquisition(source, limit=101)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda text: text.replace(
                "schema_version: 1",
                "schema_version: 1\nschema_version: 1",
                1,
            ),
            "duplicate key",
        ),
        (
            lambda text: text.replace(
                "schema_version: 1",
                "schema_version: 1\nunexpected: true",
                1,
            ),
            "keys mismatch",
        ),
        (
            lambda text: text.replace(
                "    enabled: true\n    approved: true",
                "    enabled: false\n    approved: true",
                1,
            ),
            "approved sources must be enabled",
        ),
        (
            lambda text: text.replace(
                "    enabled: true",
                "    enabled: 1",
                1,
            ),
            "enabled must be a boolean",
        ),
        (
            lambda text: text.replace(
                "    license: Public Domain",
                "    license: Pending",
                1,
            ),
            "decided license",
        ),
        (
            lambda text: text.replace(
                "    machine_learning_allowed: true",
                "    machine_learning_allowed: false",
                1,
            ),
            "permit machine learning",
        ),
    ],
)
def test_registry_rejects_ambiguous_or_unsafe_configuration(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    path = _copy_registry_tree(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(mutation(text), encoding="utf-8")  # type: ignore[operator]

    with pytest.raises(RegistryError, match=message):
        load_source_registry(path)


def test_registry_rejects_aliases_before_construction(tmp_path: Path) -> None:
    path = _copy_registry_tree(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "allowed_hosts:\n      - www.yss-aya.com",
            "allowed_hosts: &hosts\n      - www.yss-aya.com",
            1,
        ).replace(
            "allowed_hosts:\n      - wdoor.c.u-tokyo.ac.jp",
            "allowed_hosts: *hosts",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="aliases are not allowed"):
        load_source_registry(path)


def test_registry_rejects_oversized_yaml_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "oversized.yaml"
    path.write_bytes(b"x" * (MAX_REGISTRY_BYTES + 1))

    with pytest.raises(RegistryError, match="exceeds"):
        load_source_registry(path)


@pytest.mark.parametrize(
    "filename",
    ["aobazero_no_noise.yaml", "aobazero_evidence.yaml"],
)
def test_registry_rejects_symlinked_catalog_files(tmp_path: Path, filename: str) -> None:
    path = _copy_registry_tree(tmp_path)
    catalog = path.parent / "data_source_objects" / filename
    outside = tmp_path / f"outside-{filename}"
    catalog.rename(outside)
    catalog.symlink_to(outside)

    with pytest.raises(RegistryError, match="must not traverse a symlink"):
        load_source_registry(path)


def test_data_url_policy_does_not_inherit_evidence_hosts() -> None:
    source = load_source_registry(REGISTRY_PATH).get("aobazero-no-noise")

    with pytest.raises(RegistryError, match="host is not approved"):
        source.validate_url("http://raw.githubusercontent.com/aobazero/no_noise/w4745.csa")
