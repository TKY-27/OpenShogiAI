from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from open_shogi_training.labeling.benchmark import (
    BenchmarkError,
    _read_process_tree_rss,
    run_benchmark,
    verify_benchmark_report,
)
from open_shogi_training.labeling.fingerprint import fingerprint_teacher

from .helpers import (
    make_fake_project,
    phase3_position,
    selection_for,
    write_authorized_benchmark,
    write_phase3_dataset,
)


def test_benchmark_measures_candidates_and_authorizes_configured_highest_budget(
    tmp_path: Path,
) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=10, nodes=20)
    positions, manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    output = tmp_path / "benchmark-report.json"

    result = run_benchmark(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=manifest,
        output_path=output,
    )

    assert result.selected_nodes == 20
    assert result.report["labeling_authorized"] is True
    assert [budget["nodes"] for budget in result.report["budgets"]] == [10, 20]
    assert all(budget["completed"] == 1 for budget in result.report["budgets"])
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["resources"]["concurrency"] == 1
    assert stored["resources"]["usi_hash_mb"] == 16
    assert stored["resources"]["threads"] == 1

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_benchmark(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=manifest,
            output_path=output,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_resource", "resource settings"),
        ("summary", "summaries or gate decision"),
        ("teacher_binding", "selection/teacher binding"),
        ("teacher", "teacher files/options"),
        ("raw_type", "reported_nodes"),
        ("undersearched", "below the requested budget"),
        ("gate", "summaries or gate decision"),
    ],
)
def test_benchmark_verification_recomputes_closed_raw_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=1, nodes=20)
    positions, manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    selection = selection_for(config, positions, manifest)
    path = write_authorized_benchmark(tmp_path, config, selection)
    report = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "unknown_resource":
        report["resources"]["untrusted"] = True
    elif mutation == "summary":
        report["budgets"][0]["elapsed_ms"]["p95"] += 1
    elif mutation == "teacher_binding":
        report["budgets"][0]["searches"][0]["teacher_identity_sha256"] = "0" * 64
    elif mutation == "teacher":
        report["teacher"]["binary"]["sha256"] = "0" * 64
    elif mutation == "raw_type":
        report["budgets"][0]["searches"][0]["reported_nodes"] = True
    elif mutation == "undersearched":
        report["budgets"][0]["searches"][0]["reported_nodes"] = 1
    elif mutation == "gate":
        report["budgets"][0]["passes"] = False
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkError, match=message):
        verify_benchmark_report(
            path,
            config=config,
            selection=selection,
            fingerprint=fingerprint_teacher(config, tmp_path),
        )


def test_legacy_benchmark_rows_are_strictly_recomputed_for_completed_evidence(
    tmp_path: Path,
) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=1, nodes=20)
    positions, manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    selection = selection_for(config, positions, manifest)
    path = write_authorized_benchmark(tmp_path, config, selection)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema"] = "phase4_teacher_benchmark/v1"
    report["selection_sha256"] = selection.legacy_selection_sha256
    del report["teacher_identity_sha256"]
    for budget in report["budgets"]:
        for row in budget["searches"]:
            del row["canonical_state_sha256"]
            del row["teacher_identity_sha256"]
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    verified, _ = verify_benchmark_report(
        path,
        config=config,
        selection=selection,
        fingerprint=fingerprint_teacher(config, tmp_path),
    )
    assert verified["labeling_authorized"] is True


def test_benchmark_rejects_teacher_identity_drift_after_a_retry(tmp_path: Path) -> None:
    marker = tmp_path / "crashed.marker"
    config, _, _ = make_fake_project(
        tmp_path,
        mode="crash-identity-drift",
        extra_arguments=["--marker", str(marker)],
        max_positions=1,
        max_retries=1,
        nodes=20,
    )
    positions, manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )

    with pytest.raises(BenchmarkError, match="identity changed"):
        run_benchmark(
            config=config,
            project_root=tmp_path,
            positions_path=positions,
            dataset_manifest_path=manifest,
            output_path=tmp_path / "benchmark.json",
        )


def test_benchmark_records_retry_exhaustion_as_its_closed_root_cause(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "crashed.marker"
    config, _, _ = make_fake_project(
        tmp_path,
        mode="crash-once",
        extra_arguments=["--marker", str(marker)],
        max_positions=1,
        max_retries=0,
        nodes=20,
    )
    positions, manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    output = tmp_path / "benchmark.json"

    result = run_benchmark(
        config=config,
        project_root=tmp_path,
        positions_path=positions,
        dataset_manifest_path=manifest,
        output_path=output,
    )

    failed_row = result.report["budgets"][0]["searches"][0]
    assert failed_row["status"] == "failed"
    assert failed_row["error_category"] == "process"
    selection = selection_for(config, positions, manifest)
    verified, _ = verify_benchmark_report(
        output,
        config=config,
        selection=selection,
        fingerprint=fingerprint_teacher(config, tmp_path),
    )
    assert verified["labeling_authorized"] is True


def test_benchmark_rss_measurement_sums_process_group_and_descendant_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = b"100 1 100 10\n101 100 100 20\n102 101 200 30\n999 1 999 900\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )

    assert _read_process_tree_rss(100) == 60 * 1024
