from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import open_shogi_training.labeling.legality as legality_module
import open_shogi_training.labeling.pipeline as pipeline_module
import open_shogi_training.models.dataset as model_dataset_module
import pytest
from open_shogi_training.labeling.artifacts import compact_json_bytes
from open_shogi_training.labeling.fingerprint import fingerprint_teacher
from open_shogi_training.labeling.legality import (
    LegalityCoverage,
    LegalityValidatorIdentity,
)
from open_shogi_training.labeling.pipeline import (
    LabelingError,
    audit_labeling_output,
    migrate_label_manifest_v2,
    run_labeling,
)
from open_shogi_training.labeling.schema import iter_teacher_labels
from open_shogi_training.labeling.usi import (
    USICandidate,
    USIIdentity,
    USIScore,
    USISearchResult,
)
from open_shogi_training.selfplay.common import ArtifactRef

from .helpers import (
    make_fake_project,
    phase3_position,
    selection_for,
    write_authorized_benchmark,
    write_phase3_dataset,
)

_TEST_BUILD_RECEIPT = ArtifactRef(
    path="local/build-receipts/open-shogi-cli.json",
    sha256="1" * 64,
    size=1,
)


@pytest.fixture(autouse=True)
def _inject_test_legality_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fake-script tests separate from the production native receipt boundary."""

    class TestRustLegalityValidator(legality_module.RustLegalityValidator):
        def __init__(self, project_root: Path) -> None:
            def load_test_receipt(root: Path) -> tuple[ArtifactRef, ArtifactRef]:
                payload = (root / "target/release/open-shogi-cli").read_bytes()
                return (
                    ArtifactRef(
                        path="target/release/open-shogi-cli",
                        sha256=hashlib.sha256(payload).hexdigest(),
                        size=len(payload),
                    ),
                    _TEST_BUILD_RECEIPT,
                )

            super().__init__(
                project_root,
                receipt_loader=load_test_receipt,
            )

    monkeypatch.setattr(pipeline_module, "RustLegalityValidator", TestRustLegalityValidator)


def test_label_output_lock_uses_directory_authority_not_replaceable_lock_entry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "labels"
    with pipeline_module._OutputLock(output):
        (output / ".labeling.lock").write_bytes(b"foreign-do-not-delete")
        with (
            pytest.raises(LabelingError, match="another labeling process"),
            pipeline_module._OutputLock(output),
        ):
            raise AssertionError("unreachable")

    assert (output / ".labeling.lock").read_bytes() == b"foreign-do-not-delete"


def test_label_output_lock_rejects_authority_directory_replacement(tmp_path: Path) -> None:
    output = tmp_path / "labels"
    moved = tmp_path / "labels-original"
    with (
        pytest.raises(LabelingError, match="ancestor changed"),
        pipeline_module._OutputLock(output),
    ):
        output.rename(moved)
        output.mkdir()
        with (
            pytest.raises(LabelingError, match="another labeling process"),
            pipeline_module._OutputLock(output),
        ):
            raise AssertionError("unreachable")


def test_immutable_evidence_publication_preserves_a_foreign_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "manifest.v1.evidence.json"
    publish = pipeline_module.publish_regular_at

    def publish_after_foreign_wins(*args: object, **kwargs: object) -> None:
        evidence.write_bytes(b"foreign-do-not-delete")
        publish(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "publish_regular_at", publish_after_foreign_wins)
    with pytest.raises(LabelingError, match="different bytes"):
        pipeline_module._publish_immutable_evidence(evidence, b"owned")

    assert evidence.read_bytes() == b"foreign-do-not-delete"


def test_immutable_evidence_recovers_from_an_orphaned_partial_temp(tmp_path: Path) -> None:
    evidence = tmp_path / "manifest.v1.evidence.json"
    orphan = tmp_path / ".manifest.v1.evidence.json.deadbeef.pending"
    orphan.write_bytes(b"partial")

    pipeline_module._publish_immutable_evidence(evidence, b"complete")

    assert evidence.read_bytes() == b"complete"
    assert orphan.read_bytes() == b"partial"


def test_staged_10_then_100_resumes_without_relabeling_or_reset(tmp_path: Path) -> None:
    command_log = tmp_path / "commands.log"
    config, _, _ = make_fake_project(
        tmp_path,
        max_positions=100,
        extra_arguments=["--command-log", str(command_log)],
    )
    rows = [
        phase3_position(
            game=f"game-{index // 10}",
            index=index % 10,
            sfen=f"state-{index} {'b' if index % 2 == 0 else 'w'} - {index + 1}",
            split=("test", "validation", "train")[index % 3],
            full_plies=30,
        )
        for index in range(100)
    ]
    positions, dataset_manifest = write_phase3_dataset(tmp_path, rows)
    selection = selection_for(config, positions, dataset_manifest)
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    output = tmp_path / "labels"

    first = run_labeling(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        target_completed=10,
    )
    assert first.status == "staged"
    assert first.completed == 10
    assert first.pending == 90
    first_labels = list(iter_teacher_labels(first.labels_path))
    assert len(first_labels) == 10

    second = run_labeling(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        target_completed=100,
    )
    assert second.status == "complete"
    assert second.completed == 100
    assert second.pending == 0
    labels = list(iter_teacher_labels(second.labels_path))
    assert len(labels) == 100
    assert len({label["position_id"] for label in labels}) == 100
    assert labels[0]["score"] == {"kind": "cp", "value": 42}
    assert labels[0]["candidates"][1]["score"] == {"kind": "mate", "value": -3}
    assert command_log.read_text(encoding="utf-8").splitlines().count("go nodes 20") == 100

    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["progress"]["status"] == "complete"
    assert manifest["progress"]["target_completed"] == 100
    assert len(manifest["progress"]["completed_position_ids"]) == 100
    assert manifest["artifacts"]["labels.jsonl"]["records"] == 100


def test_exhausted_position_failure_is_quarantined_and_not_retried_on_resume(
    tmp_path: Path,
) -> None:
    command_log = tmp_path / "commands.log"
    config, _, _ = make_fake_project(
        tmp_path,
        mode="fail-token",
        max_positions=10,
        max_retries=0,
        extra_arguments=[
            "--fail-token",
            "FAIL",
            "--command-log",
            str(command_log),
        ],
    )
    rows = [
        phase3_position(
            game=f"game-{index}",
            index=0,
            sfen=f"state-{index} b - 1",
        )
        for index in range(10)
    ]
    positions, dataset_manifest = write_phase3_dataset(tmp_path, rows)
    initial = selection_for(config, positions, dataset_manifest)
    first_id = initial.positions[0].position_id
    for row in rows:
        if row["gameId"] == initial.positions[0].game_id:
            row["sfen"] = "FAIL b - 1"
            break
    positions.unlink()
    positions, dataset_manifest = write_phase3_dataset(tmp_path, rows)
    selection = selection_for(config, positions, dataset_manifest)
    assert selection.positions[0].position_id == first_id
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    output = tmp_path / "labels"

    result = run_labeling(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        target_completed=9,
    )
    assert result.status == "completed_with_quarantine"
    assert result.completed == 9
    assert result.quarantined == 1
    quarantine_before = result.quarantine_path.read_bytes()
    go_before = command_log.read_text(encoding="utf-8").splitlines().count("go nodes 20")

    resumed = run_labeling(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        target_completed=9,
    )
    assert resumed.quarantine_path.read_bytes() == quarantine_before
    assert command_log.read_text(encoding="utf-8").splitlines().count("go nodes 20") == go_before
    quarantine = json.loads(quarantine_before)
    assert quarantine["position_id"] == first_id
    assert quarantine["error_category"] == "protocol"


def test_resume_rejects_label_bytes_that_differ_from_recorded_full_digest(
    tmp_path: Path,
) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    labels = output / "labels.jsonl"
    tampered = labels.read_bytes().replace(b'"created_at":"2026', b'"created_at":"2025', 1)
    assert tampered != labels.read_bytes()
    labels.write_bytes(tampered)

    with pytest.raises(LabelingError, match="recorded full digest"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )


def test_resume_rejects_source_drift_even_if_manifest_digest_is_rewritten(
    tmp_path: Path,
) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    labels_path = output / "labels.jsonl"
    label = json.loads(labels_path.read_text(encoding="utf-8"))
    label["source_id"] = "forged-source"
    payload = compact_json_bytes(label) + b"\n"
    labels_path.write_bytes(payload)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "records": 1,
    }
    manifest["artifacts"]["labels.jsonl"] = recorded
    manifest["binding"]["labels"] = recorded
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(LabelingError, match="source provenance disagrees"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )


@pytest.mark.parametrize("field", ["selection_sha256", "benchmark_sha256"])
def test_resume_rejects_manifest_label_binding_drift(tmp_path: Path, field: str) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["binding"][field] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(LabelingError, match="binding identity differs"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )


def test_labeling_refuses_to_silently_rebaseline_a_legacy_v1_manifest(
    tmp_path: Path,
) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    labels_before = (output / "labels.jsonl").read_bytes()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "phase4_teacher_label_manifest/v1"
    manifest["selection"] = selection_for(
        config,
        positions,
        dataset_manifest,
    ).legacy_summary()
    del manifest["binding"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    manifest_before = manifest_path.read_bytes()
    with pytest.raises(LabelingError, match="explicit migrate-label-manifest-v2"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )

    assert (output / "labels.jsonl").read_bytes() == labels_before
    assert manifest_path.read_bytes() == manifest_before


def test_read_only_audit_binds_legacy_manifest_without_changing_evidence(
    tmp_path: Path,
) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "phase4_teacher_label_manifest/v1"
    manifest["selection"] = selection_for(
        config,
        positions,
        dataset_manifest,
    ).legacy_summary()
    del manifest["binding"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    evidence_paths = [
        output / name for name in ("labels.jsonl", "quarantine.jsonl", "manifest.json")
    ]
    evidence_before = {path: path.read_bytes() for path in evidence_paths}

    result = audit_labeling_output(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
    )

    assert result.manifest_schema == "phase4_teacher_label_manifest/v1"
    assert result.completed == 1
    assert result.selection_sha256 == manifest["selection"]["sha256"]
    assert result.benchmark_sha256 == manifest["benchmark"]["sha256"]
    assert result.labels_sha256 == manifest["artifacts"]["labels.jsonl"]["sha256"]
    assert {path: path.read_bytes() for path in evidence_paths} == evidence_before


def test_explicit_10k_v1_migration_reaudits_without_teacher_and_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = make_fake_project(
        tmp_path,
        max_positions=10_000,
        max_input_rows=10_000,
        max_input_compressed_bytes=32 * 1024 * 1024,
        max_input_uncompressed_bytes=64 * 1024 * 1024,
    )
    rows = [
        phase3_position(
            game=f"migration-game-{index}",
            index=0,
            sfen=f"migration-state-{index} b - 1",
        )
        for index in range(10_000)
    ]
    positions, dataset_manifest = write_phase3_dataset(tmp_path, rows)
    selection = selection_for(config, positions, dataset_manifest)
    assert len(selection.positions) == 10_000
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    fingerprint = fingerprint_teacher(config, tmp_path)
    reported = USIIdentity(
        name="Fake Teacher 1.0",
        author="Test Suite",
        declared_options=tuple(sorted(config.option_map)),
    )
    search = USISearchResult(
        bestmove="7g7f",
        candidates=tuple(
            USICandidate(
                multipv=index,
                score=USIScore(kind="cp", value=10 - index),
                pv=(move,),
                depth=1,
                seldepth=1,
                nodes=config.nodes,
            )
            for index, move in enumerate(("7g7f", "2g2f", "5g5f"), start=1)
        ),
        elapsed_ms=1,
    )
    labels = {
        position.position_id: pipeline_module._label_record(
            position,
            search,
            teacher=fingerprint.teacher_record(reported),
            selection=selection,
            config=config,
        )
        for position in selection.positions
    }
    output = tmp_path / "labels"
    output.mkdir()
    labels_path = output / "labels.jsonl"
    labels_path.write_bytes(
        b"".join(
            compact_json_bytes(labels[position.position_id]) + b"\n"
            for position in selection.positions
        )
    )
    quarantine_path = output / "quarantine.jsonl"
    quarantine_path.write_bytes(b"")
    validator_path = tmp_path / "target/release/open-shogi-cli"
    validator_bytes = validator_path.read_bytes()
    validator_identity = LegalityValidatorIdentity(
        path="target/release/open-shogi-cli",
        sha256=hashlib.sha256(validator_bytes).hexdigest(),
        size=len(validator_bytes),
        build_receipt=_TEST_BUILD_RECEIPT,
        reported_name="OpenShogiAI test validator",
        reported_author="Test Suite",
    )
    progress = pipeline_module._Progress(
        completed=labels,
        quarantined={},
        legality_coverage={
            position.position_id: LegalityCoverage(
                requested_multipv=config.multipv,
                returned_candidates=config.multipv,
                legal_root_count=None,
            )
            for position in selection.positions
        },
    )
    manifest_path = output / "manifest.json"
    pipeline_module._write_manifest(
        manifest_path,
        labels_path,
        quarantine_path,
        selection=selection,
        config=config,
        fingerprint=fingerprint,
        benchmark_path=benchmark,
        benchmark_sha256=hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        progress=progress,
        reported_identity={"name": reported.name, "author": reported.author},
        validator_identity=validator_identity,
        target_completed=10_000,
    )
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy["schema"] = "phase4_teacher_label_manifest/v1"
    legacy["selection"] = selection.legacy_summary()
    del legacy["binding"]
    manifest_path.write_bytes(compact_json_bytes(legacy) + b"\n")
    legacy_bytes = manifest_path.read_bytes()
    legacy_sha256 = hashlib.sha256(legacy_bytes).hexdigest()
    labels_before = labels_path.read_bytes()
    quarantine_before = quarantine_path.read_bytes()

    class CountingValidator:
        validations = 0
        unchanged_checks = 0
        drift_identity = False

        def __init__(self, root: Path, *, receipt_loader=None) -> None:
            loaded = None if receipt_loader is None else receipt_loader(root)
            self.loaded_engine = None if loaded is None else loaded[0]
            self.loaded_receipt = None if loaded is None else loaded[1]

        def start(self) -> LegalityValidatorIdentity:
            if self.loaded_engine is not None:
                assert self.loaded_engine == ArtifactRef(
                    path=validator_identity.path,
                    sha256=validator_identity.sha256,
                    size=validator_identity.size,
                )
                assert self.loaded_receipt == validator_identity.build_receipt
            if type(self).drift_identity:
                return LegalityValidatorIdentity(
                    path=validator_identity.path,
                    sha256=validator_identity.sha256,
                    size=validator_identity.size,
                    build_receipt=validator_identity.build_receipt,
                    reported_name="OpenShogiAI drifted validator",
                    reported_author=validator_identity.reported_author,
                )
            return validator_identity

        def validate(
            self,
            _sfen: str,
            _bestmove: str,
            candidates: list[list[str]],
            *,
            configured_multipv: int,
        ) -> LegalityCoverage:
            type(self).validations += 1
            return LegalityCoverage(configured_multipv, len(candidates), None)

        def assert_binary_unchanged(self) -> None:
            type(self).unchanged_checks += 1

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline_module, "RustLegalityValidator", CountingValidator)

    result = migrate_label_manifest_v2(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        expected_legacy_manifest_sha256=legacy_sha256,
    )

    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_path = output / "manifest.v1.evidence.json"
    assert result.audit.completed == 10_000
    assert result.audit.quarantined == 0
    assert result.legacy_manifest_path == evidence_path
    assert migrated["schema"] == "phase4_teacher_label_manifest/v2"
    assert migrated["migration"] == {
        "schema": "phase4_teacher_label_manifest_migration/v1",
        "legacy_manifest": {
            "path": evidence_path.name,
            "sha256": legacy_sha256,
            "size": len(legacy_bytes),
        },
        "legacy_schema": "phase4_teacher_label_manifest/v1",
        "legacy_updated_at": legacy["updated_at"],
    }
    assert evidence_path.read_bytes() == legacy_bytes
    assert labels_path.read_bytes() == labels_before
    assert quarantine_path.read_bytes() == quarantine_before
    assert CountingValidator.validations == 20_000
    assert CountingValidator.unchanged_checks == 2

    # The production training consumer must accept the exact manifest emitted by
    # migration. The temporary test repository has no production Rust binary at
    # this module's fixed repository root, so its executable identity was already
    # exercised above and only that environment-specific lookup is substituted.
    monkeypatch.setattr(
        model_dataset_module,
        "_validate_legality_validator",
        lambda _raw, *, repository_root, verify_disk: None,
    )
    model_dataset_module._validate_label_manifest_v2(
        migrated,
        label_manifest_path=manifest_path,
        labels_path=labels_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        labels=tuple(labels[position.position_id] for position in selection.positions),
        labels_sha256=hashlib.sha256(labels_before).hexdigest(),
        labels_size=len(labels_before),
        dataset_manifest_sha256=hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
        dataset_manifest_size=dataset_manifest.stat().st_size,
        positions_sha256=hashlib.sha256(positions.read_bytes()).hexdigest(),
        positions_size=positions.stat().st_size,
        expected_teacher_labels=10_000,
        repository_root=tmp_path,
    )
    CountingValidator.drift_identity = True
    with pytest.raises(ValueError, match="legality validator identity differs"):
        model_dataset_module._validate_label_manifest_v2(
            migrated,
            label_manifest_path=manifest_path,
            labels_path=labels_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            labels=tuple(labels[position.position_id] for position in selection.positions),
            labels_sha256=hashlib.sha256(labels_before).hexdigest(),
            labels_size=len(labels_before),
            dataset_manifest_sha256=hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
            dataset_manifest_size=dataset_manifest.stat().st_size,
            positions_sha256=hashlib.sha256(positions.read_bytes()).hexdigest(),
            positions_size=positions.stat().st_size,
            expected_teacher_labels=10_000,
            repository_root=tmp_path,
        )
    CountingValidator.drift_identity = False

    migrated_before = manifest_path.read_bytes()
    migrate_label_manifest_v2(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        expected_legacy_manifest_sha256=legacy_sha256,
    )
    assert manifest_path.read_bytes() == migrated_before
    assert evidence_path.read_bytes() == legacy_bytes
    assert CountingValidator.validations == 40_000


def test_read_only_audit_rejects_an_unmanifested_artifact_tail(tmp_path: Path) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    labels_path = output / "labels.jsonl"
    labels_path.write_bytes(labels_path.read_bytes() + labels_path.read_bytes())

    with pytest.raises(LabelingError, match="exact manifest-bound artifact"):
        audit_labeling_output(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
        )


def test_resume_refuses_nonempty_labels_when_their_manifest_is_missing(tmp_path: Path) -> None:
    config, positions, dataset_manifest, benchmark, output = _completed_single_label(tmp_path)
    labels_path = output / "labels.jsonl"
    labels_before = labels_path.read_bytes()
    (output / "manifest.json").unlink()

    with pytest.raises(LabelingError, match="refusing to adopt label artifacts"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )

    assert labels_path.read_bytes() == labels_before


def test_selection_and_benchmark_checkpoint_exists_before_first_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=1)
    positions, dataset_manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    selection = selection_for(config, positions, dataset_manifest)
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    benchmark_sha256 = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    output = tmp_path / "labels"
    real_append = pipeline_module.append_jsonl_record
    checked = False

    def assert_checkpoint(
        path: Path,
        value: dict[str, object],
        *,
        max_line_bytes: int,
    ) -> int:
        nonlocal checked
        if path.name == "labels.jsonl" and not checked:
            checkpoint = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            assert checkpoint["selection"]["sha256"] == selection.selection_sha256
            assert checkpoint["benchmark"]["sha256"] == benchmark_sha256
            assert checkpoint["artifacts"]["labels.jsonl"]["records"] == 0
            checked = True
        return real_append(path, value, max_line_bytes=max_line_bytes)

    monkeypatch.setattr(pipeline_module, "append_jsonl_record", assert_checkpoint)
    result = run_labeling(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        target_completed=1,
    )

    assert result.completed == 1
    assert checked


def test_teacher_pv_rejected_by_rust_cli_is_never_appended(tmp_path: Path) -> None:
    config, _, _ = make_fake_project(
        tmp_path,
        mode="illegal-second-pv",
        max_positions=1,
    )
    positions, dataset_manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    selection = selection_for(config, positions, dataset_manifest)
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    output = tmp_path / "labels"

    with pytest.raises(LabelingError, match="OpenShogiAI rejected multipv 2"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )
    assert (output / "labels.jsonl").read_bytes() == b""


def test_short_multipv_perft_uses_the_same_verified_rust_cli_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = make_fake_project(tmp_path, mode="one-rank", max_positions=1)
    positions, dataset_manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="forced-one b - 1")],
    )
    selection = selection_for(config, positions, dataset_manifest)
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    output = tmp_path / "labels"
    validator_path = tmp_path / "target/release/open-shogi-cli"
    marker = tmp_path / "unverified-validator-ran"
    real_run = legality_module.subprocess.run
    swapped = False

    def swap_before_perft(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        command = args[0] if args else kwargs.get("args")
        if not swapped and isinstance(command, list) and len(command) > 1 and command[1] == "perft":
            swapped = True
            validator_path.write_text(
                f"#!{sys.executable}\nfrom pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            validator_path.chmod(0o755)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(legality_module.subprocess, "run", swap_before_perft)

    with pytest.raises(LabelingError, match="validator changed during labeling"):
        run_labeling(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=dataset_manifest,
            benchmark_report_path=benchmark,
            output_dir=output,
            target_completed=1,
        )

    assert not marker.exists()
    assert (output / "labels.jsonl").read_bytes() == b""
    assert not list(tmp_path.glob(".open-shogi-exec.*"))
    assert not list((tmp_path / "local/runtime-snapshots").glob(".open-shogi-exec.*"))


@pytest.mark.parametrize(
    ("sfen", "expected_additional_roots"),
    [("forced-one b - 1", 0), ("many-moves b - 1", 2)],
)
def test_short_multipv_records_independent_rust_root_coverage(
    tmp_path: Path,
    sfen: str,
    expected_additional_roots: int,
) -> None:
    config, _, _ = make_fake_project(tmp_path, mode="one-rank", max_positions=1)
    positions, dataset_manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen=sfen)],
    )
    selection = selection_for(config, positions, dataset_manifest)
    benchmark = write_authorized_benchmark(tmp_path, config, selection)
    result = run_labeling(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=tmp_path / "labels",
        target_completed=1,
    )
    manifest_record = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    coverage = manifest_record["binding"]["candidate_coverage"]
    assert result.completed == 1
    assert coverage["requested_multipv"] == 3
    assert coverage["returned_candidate_counts"] == {"1": 1, "2": 0, "3": 0}
    assert coverage["short_rows"] == 1
    assert coverage["additional_legal_root_moves"] == expected_additional_roots
    assert coverage["short_rows_with_additional_legal_roots"] == (
        1 if expected_additional_roots else 0
    )


def _completed_single_label(
    root: Path,
) -> tuple[object, Path, Path, Path, Path]:
    config, _, _ = make_fake_project(root, max_positions=1)
    positions, dataset_manifest = write_phase3_dataset(
        root,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    selection = selection_for(config, positions, dataset_manifest)
    benchmark = write_authorized_benchmark(root, config, selection)
    output = root / "labels"
    result = run_labeling(
        config=config,
        project_root=root,
        positions_path=positions,
        dataset_manifest_path=dataset_manifest,
        benchmark_report_path=benchmark,
        output_dir=output,
        target_completed=1,
    )
    assert result.completed == 1
    return config, positions, dataset_manifest, benchmark, output
