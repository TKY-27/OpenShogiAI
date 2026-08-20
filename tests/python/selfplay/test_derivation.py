from __future__ import annotations

from pathlib import Path

from open_shogi_training.selfplay.common import (
    ArtifactRef,
    canonical_json_bytes,
    canonical_sha256,
)
from open_shogi_training.selfplay.config import SelfPlayConfig
from open_shogi_training.selfplay.derivation import build_position_evidence
from open_shogi_training.selfplay.evidence import (
    build_replay_buffer_manifest,
    build_replay_candidates,
    extract_hard_positions,
)
from open_shogi_training.selfplay.execution import CommandOutcome
from open_shogi_training.selfplay.planning import (
    ModelSpec,
    build_selfplay_plan,
)

from .conftest import (
    make_phase2_pair_report,
    tiny_model_bytes,
    write_completed_attempt_receipt,
    write_initial_registry_ref,
    write_ref,
    write_selfplay_config_ref,
    write_validated_phase3_starts,
)

TRAIN_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"
VALIDATION_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 1"


class _ExportRunner:
    def __init__(self, root: Path, starts_by_directory: dict[str, str], player_label: str) -> None:
        self.root = root
        self.starts_by_directory = starts_by_directory
        self.player_label = player_label
        self.calls = 0

    def run(
        self,
        command: object,
        *,
        stdout_path: str,
        stderr_path: str,
        resume: bool = False,
        expected_executable: ArtifactRef | None = None,
        engine_build_receipt: ArtifactRef | None = None,
        memory_limit_mib: int = 4_096,
        receipt_path: str | None = None,
    ) -> CommandOutcome:
        assert not resume
        assert expected_executable is not None
        del engine_build_receipt, memory_limit_mib, receipt_path
        argv = command["argv"]
        input_dir = argv[argv.index("--input-dir") + 1]
        output = argv[argv.index("--output") + 1]
        assert (self.root / output).parent.is_dir()
        initial = self.starts_by_directory[input_dir]
        fields = initial.split(" ")
        next_side = "w" if fields[1] == "b" else "b"
        next_sfen = " ".join((*fields[:1], next_side, fields[2], "2"))
        rows = [
            _export_row("game-000001.csa", initial, next_sfen, "black_win", self.player_label),
            _export_row("game-000002.csa", initial, next_sfen, "white_win", self.player_label),
        ]
        payload = b"".join(canonical_json_bytes(row) for row in rows)
        write_ref(self.root, output, payload)
        stdout = write_ref(self.root, stdout_path, b"")
        stderr = write_ref(self.root, stderr_path, b"")
        self.calls += 1
        return CommandOutcome(0, False, False, False, 1, "process_tree_ps_rss_sum", stdout, stderr)


def _export_row(
    file_name: str, initial: str, next_sfen: str, outcome: str, player_label: str
) -> dict[str, object]:
    return {
        "schema": "phase3_csa_export/v1",
        "status": "ok",
        "inputFile": file_name,
        "normalizedCsa": "V3.0\n%TORYO\n",
        "initialSfen": initial,
        "positionSfens": [initial, next_sfen],
        "usiMoves": ["7g7f"],
        "blackName": player_label,
        "whiteName": player_label,
        "terminalReason": "resign",
        "outcome": outcome,
        "resultValidation": "external_condition",
    }


def _build_completed_selfplay(
    tmp_path: Path, config: SelfPlayConfig
) -> tuple[dict[str, object], ArtifactRef, dict[str, object], ArtifactRef]:
    engine = write_ref(tmp_path, "target/release/open-shogi-cli", b"engine\n")
    starts, starts_ref, validation_ref, _, _ = write_validated_phase3_starts(
        tmp_path,
        engine_ref=engine,
        train_sfen=TRAIN_SFEN,
        validation_sfen=VALIDATION_SFEN,
    )
    model_bytes = tiny_model_bytes()
    model = write_ref(tmp_path, "weights/champion.osaval", model_bytes)
    registry = write_initial_registry_ref(
        tmp_path,
        champion_ref=model,
        quantization="float32",
    )
    config_ref = write_selfplay_config_ref(tmp_path)
    plan = build_selfplay_plan(
        generation_id="generation-0001",
        champion=ModelSpec("champion-v0", model, "neural"),
        engine=engine,
        model_registry=registry,
        git_commit="a399407",
        config=config,
        config_ref=config_ref,
        start_positions=starts,
        start_positions_ref=starts_ref,
        validation_ref=validation_ref,
    )
    plan_ref = write_ref(tmp_path, "artifacts/selfplay-plan.json", canonical_json_bytes(plan))
    attempts: list[dict[str, object]] = []
    for job in plan["jobs"]:
        output_dir = job["outputDir"]
        csa = (
            write_ref(
                tmp_path, f"{output_dir}/games/game-000001.csa", f"{job['jobId']}-a".encode()
            ),
            write_ref(
                tmp_path, f"{output_dir}/games/game-000002.csa", f"{job['jobId']}-b".encode()
            ),
        )
        report = make_phase2_pair_report(
            job=job,
            model_a_ref=model,
            model_a_bytes=model_bytes,
            model_b_ref=model,
            model_b_bytes=model_bytes,
            csa_refs=csa,
        )
        report["games"][0]["moves"] = 1
        report["games"][1]["moves"] = 1
        report_ref = write_ref(
            tmp_path,
            f"{output_dir}/arena-report.json",
            canonical_json_bytes(report),
        )
        stdout = write_ref(tmp_path, f"artifacts/logs/{job['jobId']}.stdout", b"")
        stderr = write_ref(tmp_path, f"artifacts/logs/{job['jobId']}.stderr", b"")
        command_receipt = write_completed_attempt_receipt(
            tmp_path,
            plan=plan,
            job=job,
            stdout=stdout,
            stderr=stderr,
            report=report_ref,
            csa=csa,
        )
        attempts.append(
            {
                "jobId": job["jobId"],
                "attempt": 1,
                "status": "completed",
                "returnCode": 0,
                "timedOut": False,
                "outputLimitExceeded": False,
                "memoryLimitExceeded": False,
                "peakRssBytes": 1,
                "rssMeasurement": "process_tree_ps_rss_sum",
                "stdout": stdout.as_dict(),
                "stderr": stderr.as_dict(),
                "report": report_ref.as_dict(),
                "csa": [reference.as_dict() for reference in csa],
                "quarantine": None,
                "failureCategory": None,
                "completedAt": "2026-08-08T00:00:01Z",
                "commandReceipt": command_receipt.as_dict(),
            }
        )
    execution: dict[str, object] = {
        "schema": "phase6_selfplay_manifest/v1",
        "generationId": "generation-0001",
        "plan": plan_ref.as_dict(),
        "planSha256": plan["planSha256"],
        "status": "completed",
        "gameCountPlanned": 40,
        "jobsPlanned": 20,
        "jobsCompleted": 20,
        "jobsQuarantined": 0,
        "gamesCompleted": 40,
        "gamesQuarantined": 0,
        "quarantinedAttempts": 0,
        "attempts": attempts,
    }
    execution["manifestSha256"] = canonical_sha256(execution)
    execution_ref = write_ref(
        tmp_path,
        "artifacts/selfplay-manifest.json",
        canonical_json_bytes(execution),
    )
    return plan, plan_ref, execution, execution_ref


def test_completed_selfplay_derives_factual_train_outcomes_used_by_replay(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    plan, plan_ref, execution, execution_ref = _build_completed_selfplay(tmp_path, selfplay_config)
    labels_ref = write_ref(tmp_path, "artifacts/labels.jsonl", b"")
    predictions_ref = write_ref(tmp_path, "artifacts/model-predictions.jsonl", b"")
    dataset_ref = ArtifactRef.from_dict(plan["datasetManifest"], "plan.datasetManifest")
    starts_by_directory = {f"{job['outputDir']}/games": str(job["sfen"]) for job in plan["jobs"]}
    model_sha = str(plan["champion"]["artifact"]["sha256"])
    player_label = f"search:neural:d6:h64:tt-on:book-off:m-{model_sha[:12]}"
    runner = _ExportRunner(tmp_path, starts_by_directory, player_label)

    evidence = build_position_evidence(
        repository_root=tmp_path,
        selfplay_plan=plan,
        selfplay_plan_ref=plan_ref,
        selfplay_manifest=execution,
        selfplay_manifest_ref=execution_ref,
        teacher_labels_ref=labels_ref,
        model_predictions_ref=predictions_ref,
        dataset_manifest_ref=dataset_ref,
        generation_ordinal=1,
        teacher_source_generation_id="generation-0",
        export_root="artifacts/evidence-export",
        timeout_seconds=30,
        runner=runner,
    )
    evidence_ref = write_ref(
        tmp_path,
        "artifacts/evidence.json",
        canonical_json_bytes(evidence),
    )
    hard = extract_hard_positions(
        evidence,
        evidence_ref=evidence_ref,
        config=selfplay_config,
        labels_before=10_000,
    )
    hard_ref = write_ref(tmp_path, "artifacts/hard.json", canonical_json_bytes(hard))
    candidates = build_replay_candidates(
        evidence,
        evidence_ref=evidence_ref,
        hard_positions=hard,
        hard_positions_ref=hard_ref,
        config=selfplay_config,
        repository_root=tmp_path,
    )
    candidates_ref = write_ref(
        tmp_path,
        "artifacts/replay-candidates.json",
        canonical_json_bytes(candidates),
    )
    replay = build_replay_buffer_manifest(
        candidates,
        candidates_ref=candidates_ref,
        config=selfplay_config,
        config_ref=write_selfplay_config_ref(tmp_path),
        repository_root=tmp_path,
    )

    assert runner.calls == 20
    assert evidence["derivation"]["counts"]["selfplayPositions"] == 40
    assert hard["budget"]["newTeacherLabelsSelected"] == 0
    assert replay["counts"]["newRetained"] > 0
    assert replay["counts"]["splits"] == {
        "train": replay["counts"]["retained"],
        "validation": 0,
        "test": 0,
    }
    assert all(entry["outcomeTarget"] in {-1, 0, 1} for entry in replay["entries"])
    assert all(entry["origins"][0]["sourcePly"] == 0 for entry in replay["entries"])
    assert all(entry["origins"][0]["sourceGameId"] for entry in replay["entries"])
