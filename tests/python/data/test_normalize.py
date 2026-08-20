import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import open_shogi_training.data.normalize as normalize
import pytest
from open_shogi_training.data.gzip_jsonl import iter_jsonl_gzip
from open_shogi_training.data.manifest import (
    CompletedObject,
    EvidenceSnapshot,
    ManifestStore,
)
from open_shogi_training.data.normalize import (
    NormalizationConfig,
    normalize_aobazero_dataset,
)
from open_shogi_training.data.registry import (
    CatalogObject,
    DataSource,
    EvidenceObject,
    LicenseEvidence,
)
from open_shogi_training.data.splits import SplitPolicy, assign_game_split
from open_shogi_training.selfplay.common import ArtifactRef
from open_shogi_training.selfplay.execution import CommandOutcome


def _raw_record(*, moves: int, comment: str, terminal: bool = True) -> bytes:
    lines = [
        "'20260717_015719_w4745.txt",
        f"'{comment}",
        "N+Black",
        "N-White",
        "PI",
        "+",
    ]
    for index in range(moves):
        move = "+7776FU" if index % 2 == 0 else "-3334FU"
        lines.append(f"{move},v=0.{index:02d},r=0.5")
    if terminal:
        lines.append("%TORYO")
    return ("\n".join(lines) + "\n").encode()


def _source(urls: list[str], evidence: LicenseEvidence) -> DataSource:
    return DataSource(
        source_id="aobazero",
        name="Synthetic AobaZero fixture",
        official_base="https://example.test/data/",
        enabled=True,
        approved=True,
        license="Synthetic-Test-License",
        license_evidence=(evidence,),
        robots_checked=date(2026, 7, 1),
        terms_checked=date(2026, 7, 1),
        max_requests_per_second=1.0,
        concurrency=1,
        allowed_paths=("/data/",),
        denied_paths=(),
        redistributable=True,
        machine_learning_allowed=True,
        last_reviewed=date(2026, 7, 1),
        allowed_hosts=("example.test",),
        allow_insecure_http=False,
        max_object_bytes=1_048_576,
        user_agent="OpenShogiAI synthetic tests",
        catalog_path=None,
        evidence_catalog_path=None,
        adapter="aobazero_csa",
        robots_url="https://example.test/data/robots.txt",
        robots_policy="allowed",
        catalog=tuple(
            CatalogObject(
                object_id=f"catalog-{index}",
                url=url,
                filename=f"{index}.csa",
                data_format="csa",
                compression="none",
            )
            for index, url in enumerate(urls)
        ),
        evidence_catalog=(
            EvidenceObject(
                evidence_id="license",
                url=evidence.url,
                max_bytes=1_024,
            ),
        ),
    )


def _write_acquisition_fixture(root: Path) -> tuple[DataSource, list[CompletedObject]]:
    root.mkdir(parents=True)
    evidence_bytes = b"Synthetic fixture permission.\n"
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_path = root / "evidence" / "sha256" / evidence_sha256[:2] / evidence_sha256
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(evidence_bytes)
    evidence = LicenseEvidence(
        url="https://example.test/data/LICENSE",
        local_path="evidence/license.txt",
        quote="Synthetic fixture permission.",
    )
    snapshot = EvidenceSnapshot(
        evidence_id="license",
        url=evidence.url,
        retrieved_at="2026-07-29T00:00:00Z",
        sha256=evidence_sha256,
        size=len(evidence_bytes),
        content_type="text/plain",
        object_path=evidence_path.relative_to(root).as_posix(),
    )

    base = _raw_record(moves=11, comment="base")
    canonical_duplicate = _raw_record(moves=11, comment="different ignored comment")
    invalid = _raw_record(moves=3, comment="missing terminal", terminal=False)
    rust_rejected = _raw_record(moves=4, comment="rust rejected")
    second = _raw_record(moves=9, comment="position cap")
    payloads = (base, canonical_duplicate, invalid, rust_rejected, second)
    for payload in payloads:
        digest = hashlib.sha256(payload).hexdigest()
        path = root / "objects" / "sha256" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    urls = [f"https://example.test/data/{index}.csa" for index in range(1, 7)]
    source = _source(urls, evidence)
    definitions = [
        ("catalog-0", urls[0], "0.csa", base),
        ("catalog-1", urls[1], "1.csa", base),
        (
            "catalog-2",
            urls[2],
            "2.csa",
            canonical_duplicate,
        ),
        ("catalog-3", urls[3], "3.csa", invalid),
        ("catalog-4", urls[4], "4.csa", rust_rejected),
        ("catalog-5", urls[5], "5.csa", second),
    ]
    records = [
        CompletedObject(
            source_id=source.source_id,
            object_id=object_id,
            url=url,
            retrieved_at="2026-07-29T01:02:03Z",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            content_type="application/x-csa",
            etag=f'"{object_id}"',
            last_modified="Wed, 29 Jul 2026 01:02:03 GMT",
            object_path=(
                Path("objects")
                / "sha256"
                / hashlib.sha256(payload).hexdigest()[:2]
                / hashlib.sha256(payload).hexdigest()
            ).as_posix(),
            original_filename=filename,
            data_format="csa",
            compression="none",
            license=source.license,
            license_evidence=source.license_evidence,
            evidence_snapshots=(snapshot,),
            redistributable=source.redistributable,
            machine_learning_allowed=source.machine_learning_allowed,
        )
        for object_id, url, filename, payload in definitions
    ]
    records[5] = replace(
        records[5],
        evidence_snapshots=(replace(snapshot, retrieved_at="2026-07-30T00:00:00Z"),),
    )
    store = ManifestStore(root)
    for record in records:
        store.append_completed(record)
    return source, records


def _fake_exporter(stage_dir: Path, output_path: Path, max_games: int) -> None:
    assert max_games <= 100
    rows = []
    for path in sorted(stage_dir.glob("*.csa")):
        csa = path.read_text(encoding="utf-8")
        if "'rust rejected" in csa:
            # Adapter removes comments, so use the four-move fixture as the selected rejection.
            move_count = sum(
                len(line) == 7 and line.startswith(("+", "-")) for line in csa.splitlines()
            )
            if move_count == 4:
                rows.append(
                    {
                        "schema": "phase3_csa_export/v1",
                        "status": "rejected",
                        "inputFile": path.name,
                        "reason": "synthetic illegal replay",
                    }
                )
                continue
        move_count = sum(
            len(line) == 7 and line.startswith(("+", "-")) for line in csa.splitlines()
        )
        if move_count == 4:
            rows.append(
                {
                    "schema": "phase3_csa_export/v1",
                    "status": "rejected",
                    "inputFile": path.name,
                    "reason": "synthetic illegal replay",
                }
            )
            continue
        positions = [f"repeated-state b - {index + 1}" for index in range(move_count + 1)]
        rows.append(
            {
                "schema": "phase3_csa_export/v1",
                "status": "ok",
                "inputFile": path.name,
                "normalizedCsa": "'CSA encoding=UTF-8\n" + csa,
                "initialSfen": positions[0],
                "positionSfens": positions,
                "usiMoves": ["7g7f"] * move_count,
                "blackName": "Black",
                "whiteName": "White",
                "terminalReason": "TORYO",
                "outcome": "white_win",
                "resultValidation": "external_condition",
            }
        )
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_normalizes_deduplicates_caps_and_reproduces_gzip(tmp_path) -> None:
    acquisition_root = tmp_path / "acquisition"
    source, records = _write_acquisition_fixture(acquisition_root)
    config = NormalizationConfig(
        dataset_id="aobazero",
        split_policy=SplitPolicy(salt="open-shogi-phase3-aobazero-v1"),
        max_games=10,
        max_positions=20,
    )

    first = normalize_aobazero_dataset(
        acquisition_root / "manifest.jsonl",
        acquisition_root=acquisition_root,
        processed_root=tmp_path / "processed-first",
        source=source,
        config=config,
        exporter=_fake_exporter,
    )
    second = normalize_aobazero_dataset(
        acquisition_root,
        acquisition_root=acquisition_root,
        processed_root=tmp_path / "processed-second",
        source=source,
        config=config,
        exporter=_fake_exporter,
    )

    assert first.included_games == 1
    assert first.included_positions == 12
    assert first.excluded_games == 5
    assert first.games_path.read_bytes() == second.games_path.read_bytes()
    assert first.positions_path.read_bytes() == second.positions_path.read_bytes()

    games = list(iter_jsonl_gzip(first.games_path))
    positions = list(iter_jsonl_gzip(first.positions_path))
    game = games[0]
    assert game["rawCsa"] == (acquisition_root / records[0].object_path).read_text(encoding="utf-8")
    assert game["rawObject"]["sha256"] == records[0].sha256
    assert game["normalizedCsa"].startswith("'CSA encoding=UTF-8\nV3.0\n")
    assert game["sourceDateTime"] == "2026-07-17T01:57:19"
    assert game["sourceTimeZone"] is None
    assert game["flags"] == {"long": False, "short": True}
    assert game["split"] == assign_game_split(game["canonicalSha256"], config.split_policy)
    assert game["licenseDecision"]["evidenceSnapshots"][0]["sha256"]
    assert sum(position["terminalTail"] for position in positions) == 9
    assert (
        sum(position["terminalTail"] and position["moveUsi"] is not None for position in positions)
        == 8
    )
    assert sum(position["eligible"] for position in positions) == 3
    assert {position["gameId"] for position in positions} == {game["gameId"]}
    assert {position["split"] for position in positions} == {game["split"]}

    report = json.loads(first.report_path.read_text())
    assert report["exclusionCounts"] == {
        "duplicate_canonical": 1,
        "duplicate_raw": 1,
        "incomplete": 1,
        "position_cap_reached": 1,
        "rust_rejected": 1,
    }
    assert report["positions"]["total"] == 12
    assert report["positions"]["unique"] == 1
    assert report["positions"]["withinGameDuplicateOccurrences"] == 11
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["config"]["split"]["salt"] == config.split_policy.salt
    assert manifest["evidenceSnapshots"][0]["sha256"]
    assert set(manifest["artifacts"]) == {
        "games-00000.jsonl.gz",
        "normalization-report.json",
        "positions-00000.jsonl.gz",
    }

    cross_config = NormalizationConfig(
        dataset_id="aobazero-cross",
        split_policy=config.split_policy,
        max_games=10,
        max_positions=30,
    )
    cross = normalize_aobazero_dataset(
        acquisition_root,
        acquisition_root=acquisition_root,
        processed_root=tmp_path / "processed-cross",
        source=source,
        config=cross_config,
        exporter=_fake_exporter,
    )
    cross_report = json.loads(cross.report_path.read_text())
    cross_manifest = json.loads(cross.manifest_path.read_text())
    assert cross.included_games == 2
    assert cross_report["positions"]["crossGameDuplicateStates"] == 1
    assert cross_report["positions"]["crossGameDuplicateOccurrences"] == 10
    assert len(cross_manifest["evidenceSnapshots"]) == 2
    assert {item["retrieved_at"] for item in cross_manifest["evidenceSnapshots"]} == {
        "2026-07-29T00:00:00Z",
        "2026-07-30T00:00:00Z",
    }

    with pytest.raises(FileExistsError):
        normalize_aobazero_dataset(
            acquisition_root,
            acquisition_root=acquisition_root,
            processed_root=tmp_path / "processed-first",
            source=source,
            config=config,
            exporter=_fake_exporter,
        )


def test_command_runner_boundary_uses_immutable_exporter_refs(tmp_path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    stage = tmp_path / "stage"
    stage.mkdir()
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

    class FakeRunner:
        def __init__(self, root: Path, **kwargs: object) -> None:
            observed["root"] = root
            observed["init"] = kwargs

        def run(self, command: object, **kwargs: object) -> CommandOutcome:
            observed["command"] = command
            observed["kwargs"] = kwargs
            stdout_path = str(kwargs["stdout_path"])
            stderr_path = str(kwargs["stderr_path"])
            stdout_file = tmp_path / stdout_path
            stderr_file = tmp_path / stderr_path
            stdout_file.write_bytes(b"")
            stderr_file.write_bytes(b"")
            return CommandOutcome(
                0,
                False,
                False,
                False,
                0,
                "process_tree_ps_short_lived_no_sample",
                ArtifactRef(stdout_path, hashlib.sha256(b"").hexdigest(), 0),
                ArtifactRef(stderr_path, hashlib.sha256(b"").hexdigest(), 0),
            )

    monkeypatch.setattr(normalize, "CommandRunner", FakeRunner)
    normalize._run_rust_exporter(
        tmp_path,
        engine,
        receipt,
        stage,
        tmp_path / "export.jsonl",
        7,
        12,
    )

    assert observed["command"] == {
        "kind": "engine_dataset_export",
        "argv": [
            engine.path,
            "export-csa-jsonl",
            "--input-dir",
            "stage",
            "--output",
            "export.jsonl",
            "--max-games",
            "7",
        ],
        "timeoutSeconds": 12,
    }
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["expected_executable"] == engine
    assert kwargs["engine_build_receipt"] == receipt
    assert kwargs["memory_limit_mib"] == 1_024
    assert observed["init"] == {
        "require_clean_repository": True,
        "require_engine_build_receipt": True,
    }
    assert not (tmp_path / "export.jsonl.stdout.log").exists()
    assert not (tmp_path / "export.jsonl.stderr.log").exists()


def test_production_normalization_routes_the_complete_export_through_immutable_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition_root = tmp_path / "acquisition"
    source, _ = _write_acquisition_fixture(acquisition_root)
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
    calls: list[tuple[object, dict[str, object]]] = []

    class FakeRunner:
        def __init__(self, root: Path, **_kwargs: object) -> None:
            assert root == tmp_path

        def run(self, command: object, **kwargs: object) -> CommandOutcome:
            typed_kwargs = dict(kwargs)
            calls.append((command, typed_kwargs))
            assert isinstance(command, dict)
            argv = command["argv"]
            assert isinstance(argv, list)
            assert argv[0] == engine.path
            stage = tmp_path / argv[argv.index("--input-dir") + 1]
            output = tmp_path / argv[argv.index("--output") + 1]
            _fake_exporter(stage, output, int(argv[argv.index("--max-games") + 1]))
            stdout_path = str(kwargs["stdout_path"])
            stderr_path = str(kwargs["stderr_path"])
            (tmp_path / stdout_path).write_bytes(b"")
            (tmp_path / stderr_path).write_bytes(b"")
            empty_sha = hashlib.sha256(b"").hexdigest()
            return CommandOutcome(
                0,
                False,
                False,
                False,
                0,
                "process_tree_ps_short_lived_no_sample",
                ArtifactRef(stdout_path, empty_sha, 0),
                ArtifactRef(stderr_path, empty_sha, 0),
                engine_build_receipt=receipt,
            )

    monkeypatch.setattr(normalize, "CommandRunner", FakeRunner)
    result = normalize_aobazero_dataset(
        acquisition_root / "manifest.jsonl",
        acquisition_root=acquisition_root,
        processed_root=tmp_path / "processed-production",
        source=source,
        config=NormalizationConfig(
            dataset_id="aobazero",
            split_policy=SplitPolicy(salt="open-shogi-phase3-production-route"),
            max_games=10,
            max_positions=20,
        ),
        repository_root=tmp_path,
        engine=engine,
        engine_build_receipt=receipt,
    )

    assert result.included_games == 1
    assert result.included_positions == 12
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["expected_executable"] == engine
    assert kwargs["engine_build_receipt"] == receipt


def test_export_boundary_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "export.jsonl"
    path.write_text(
        '{"schema":"phase3_csa_export/v1","status":"rejected",'
        '"inputFile":"a.csa","inputFile":"a.csa","reason":"bad"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        normalize._read_export_rows(path, {"a.csa"})

    assert normalize._MAX_EXPORT_LINE_BYTES == 32 * 1024 * 1024
    assert normalize._MAX_EXPORT_BYTES == 128 * 1024 * 1024


def test_raw_hash_is_deduplicated_only_after_integrity_verification(tmp_path, monkeypatch) -> None:
    acquisition_root = tmp_path / "acquisition"
    source, _ = _write_acquisition_fixture(acquisition_root)
    original = normalize._read_verified_raw
    calls = 0

    def fail_second_duplicate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic duplicate corruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(normalize, "_read_verified_raw", fail_second_duplicate)
    config = NormalizationConfig(
        dataset_id="aobazero-recovery",
        split_policy=SplitPolicy(salt="public-recovery"),
        max_games=10,
        max_positions=20,
    )

    result = normalize_aobazero_dataset(
        acquisition_root,
        acquisition_root=acquisition_root,
        processed_root=tmp_path / "processed",
        source=source,
        config=config,
        exporter=_fake_exporter,
    )

    report = json.loads(result.report_path.read_text())
    assert report["exclusionCounts"]["corrupt_raw_object"] == 1
    assert report["exclusionCounts"].get("duplicate_raw", 0) == 0


def test_dataset_directory_publication_never_replaces_existing_empty_directory(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "artifact").write_text("source")
    destination.mkdir()

    with pytest.raises(FileExistsError):
        normalize._rename_directory_without_overwrite(source, destination)

    assert (source / "artifact").read_text() == "source"
    assert list(destination.iterdir()) == []


def test_manifest_record_must_match_exact_catalog_metadata(tmp_path: Path) -> None:
    acquisition_root = tmp_path / "acquisition"
    source, records = _write_acquisition_fixture(acquisition_root)
    tampered = replace(records[0], original_filename="1.csa")

    with pytest.raises(ValueError, match="exact catalog entry"):
        normalize._validate_completed_records(
            [normalize._completed_record_view(tampered)],
            source,
            acquisition_root=acquisition_root,
        )


def test_missing_manifest_never_publishes_empty_dataset(tmp_path: Path) -> None:
    evidence = LicenseEvidence(
        url="https://example.test/data/LICENSE",
        local_path="docs/source-audits/example.md",
        quote="Synthetic fixture permission.",
    )
    source = _source(["https://example.test/data/1.csa"], evidence)
    processed_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="manifest"):
        normalize_aobazero_dataset(
            tmp_path / "missing" / "manifest.jsonl",
            acquisition_root=tmp_path / "missing",
            processed_root=processed_root,
            source=source,
            config=NormalizationConfig(
                dataset_id="aobazero",
                split_policy=SplitPolicy(salt="public"),
            ),
            exporter=_fake_exporter,
        )

    assert not (processed_root / "aobazero").exists()
    assert list(processed_root.iterdir()) == []


def test_initial_sample_game_cap_cannot_exceed_100() -> None:
    with pytest.raises(ValueError, match="max_games"):
        NormalizationConfig(
            dataset_id="aobazero",
            split_policy=SplitPolicy(salt="public"),
            max_games=101,
        )
