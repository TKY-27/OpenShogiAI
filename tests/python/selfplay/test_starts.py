from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import open_shogi_training.labeling.execution as labeling_execution
import open_shogi_training.selfplay.execution as selfplay_execution
import open_shogi_training.selfplay.starts as starts_module
import pytest
from open_shogi_training.data.aobazero import adapt_aobazero_csa
from open_shogi_training.models.phase5_arena import _load_starts
from open_shogi_training.selfplay import planning as planning_module
from open_shogi_training.selfplay.common import ArtifactRef, ContractError, canonical_json_bytes
from open_shogi_training.selfplay.execution import CommandOutcome
from open_shogi_training.selfplay.phase3 import ValidatedPhase3Artifacts
from open_shogi_training.selfplay.planning import (
    StartPositionSet,
    parse_start_positions,
    start_position_identity,
    validate_start_position_validation,
)
from open_shogi_training.selfplay.starts import (
    _adapt_phase3_game_csa,
    build_start_positions_from_phase3,
    execute_start_position_validation,
    revalidate_phase3_source_with_engine,
)

from .conftest import (
    write_ref,
    write_validated_phase3_starts,
)

TRAIN_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 17"
VALIDATION_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 31"
_REAL_AOBA_RAW = (
    "' AobaZero acquisition preamble\n"
    "N+black\n"
    "N-white\n"
    "$EVENT:fixture\n"
    "PI\n"
    "+\n"
    "+7776FU,v=12\n"
    "T1\n"
    "-3334FU,'pv annotation\n"
    "T2\n"
    "%TORYO\n"
)
_REAL_AOBA_CANONICAL = (
    "'CSA encoding=UTF-8\n"
    "V3.0\n"
    "N+black\n"
    "N-white\n"
    "$EVENT:fixture\n"
    "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n"
    "P2 * -HI *  *  *  *  * -KA * \n"
    "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n"
    "P4 *  *  *  *  *  *  *  *  * \n"
    "P5 *  *  *  *  *  *  *  *  * \n"
    "P6 *  *  *  *  *  *  *  *  * \n"
    "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n"
    "P8 * +KA *  *  *  *  * +HI * \n"
    "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n"
    "P+\n"
    "P-\n"
    "+\n"
    "+7776FU\n"
    "-3334FU\n"
    "%TORYO\n"
)


class _ValidationRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[list[str]] = []

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
        self.commands.append(argv)
        stdout = write_ref(self.root, stdout_path, b"depth 0 nodes 1\n")
        stderr = write_ref(self.root, stderr_path, b"")
        return CommandOutcome(0, False, False, False, 1, "process_tree_ps_rss_sum", stdout, stderr)


def _phase3_revalidation_game() -> dict[str, object]:
    raw = _REAL_AOBA_RAW.encode("utf-8")
    return {
        "rawObject": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        },
        "rawCsa": _REAL_AOBA_RAW,
        "normalizedCsa": _REAL_AOBA_CANONICAL,
        "canonicalSha256": hashlib.sha256(_REAL_AOBA_CANONICAL.encode()).hexdigest(),
        "initialSfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "usiMoves": ["7g7f", "3c3d"],
        "positionCount": 3,
        "players": {
            "black": {"name": "black"},
            "white": {"name": "white"},
        },
        "terminalReason": "TORYO",
        "outcome": "white_win",
        "resultValidation": "external_condition",
    }


def _phase3_export_row(game: dict[str, object], *, status: str = "ok") -> dict[str, object]:
    return {
        "schema": "phase3_csa_export/v1",
        "status": status,
        "inputFile": "game-000001.csa",
        "normalizedCsa": game["normalizedCsa"],
        "initialSfen": game["initialSfen"],
        "positionSfens": [game["initialSfen"], game["initialSfen"], game["initialSfen"]],
        "usiMoves": game["usiMoves"],
        "blackName": "black",
        "whiteName": "white",
        "terminalReason": game["terminalReason"],
        "outcome": game["outcome"],
        "resultValidation": game["resultValidation"],
    }


def _run_phase3_revalidation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    game: dict[str, object],
    export_status: str = "ok",
) -> list[bytes]:
    manifest_ref = write_ref(
        tmp_path,
        "data/phase3/manifest.json",
        canonical_json_bytes({"config": {"exporterTimeoutSeconds": 30}}),
    )
    positions_ref = write_ref(tmp_path, "data/phase3/positions.jsonl.gz", b"positions\n")
    engine_ref = write_ref(tmp_path, "local/builds/engine/open-shogi-cli", b"engine")
    receipt_ref = write_ref(tmp_path, "local/build-receipts/engine.json", b"receipt\n")
    starts = StartPositionSet(manifest_ref, positions_ref, ())
    artifacts = ValidatedPhase3Artifacts((game,), ())
    observed_inputs: list[bytes] = []

    monkeypatch.setattr(starts_module, "require_clean_head", lambda *_: "a" * 40)
    monkeypatch.setattr(
        starts_module,
        "validate_engine_build_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(starts_module, "validate_phase3_artifacts", lambda **_: artifacts)

    class Runner:
        def __init__(self, root: Path, *, require_clean_repository: bool) -> None:
            assert root == tmp_path
            assert require_clean_repository is True

        def run(
            self,
            command: object,
            *,
            stdout_path: str,
            stderr_path: str,
            expected_executable: ArtifactRef,
            engine_build_receipt: ArtifactRef,
            memory_limit_mib: int,
            receipt_path: str,
        ) -> CommandOutcome:
            argv = command["argv"]
            input_dir = str(argv[argv.index("--input-dir") + 1])
            output_path = str(argv[argv.index("--output") + 1])
            observed_inputs.append((tmp_path / input_dir / "game-000001.csa").read_bytes())
            write_ref(
                tmp_path,
                output_path,
                canonical_json_bytes(_phase3_export_row(game, status=export_status)),
            )
            stdout = write_ref(tmp_path, stdout_path, b"")
            stderr = write_ref(tmp_path, stderr_path, b"")
            command_receipt = write_ref(tmp_path, receipt_path, b"command receipt\n")
            assert expected_executable == engine_ref
            assert engine_build_receipt == receipt_ref
            assert memory_limit_mib == 1_024
            return CommandOutcome(
                0,
                False,
                False,
                False,
                1,
                "process_tree_ps_rss_sum",
                stdout,
                stderr,
                command_receipt,
            )

    monkeypatch.setattr(starts_module, "CommandRunner", Runner)
    revalidate_phase3_source_with_engine(
        repository_root=tmp_path,
        start_positions=starts,
        engine_ref=engine_ref,
        engine_build_receipt=receipt_ref,
        git_commit="a" * 40,
    )
    return observed_inputs


def test_phase3_rust_revalidation_stages_real_aobazero_adaptation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = _phase3_revalidation_game()

    observed = _run_phase3_revalidation_fixture(tmp_path, monkeypatch, game=game)

    adapted = adapt_aobazero_csa(_REAL_AOBA_RAW.encode("utf-8")).csa.encode("utf-8")
    assert observed == [adapted]
    assert observed[0].startswith(b"V3.0\n")
    assert observed[0] != _REAL_AOBA_CANONICAL.encode()
    assert b"AobaZero acquisition preamble" not in observed[0]
    assert b",v=" not in observed[0] and b",'" not in observed[0]


def test_phase3_rust_revalidation_rejects_raw_identity_tamper() -> None:
    game = _phase3_revalidation_game()
    game["rawCsa"] = str(game["rawCsa"]) + "'tampered\n"

    with pytest.raises(ContractError, match="raw CSA bytes differ"):
        _adapt_phase3_game_csa(game, index=1)


def test_phase3_rust_revalidation_rejects_adaptation_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = _phase3_revalidation_game()
    adapted = adapt_aobazero_csa(_REAL_AOBA_RAW.encode("utf-8"))
    monkeypatch.setattr(
        starts_module,
        "adapt_aobazero_csa",
        lambda _raw: replace(adapted, raw_sha256="0" * 64),
    )

    with pytest.raises(ContractError, match="adapter raw identity mismatch"):
        _adapt_phase3_game_csa(game, index=1)


def test_phase3_rust_revalidation_rejects_exporter_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = _phase3_revalidation_game()

    with pytest.raises(ContractError, match="rejected a Phase 3 raw CSA"):
        _run_phase3_revalidation_fixture(
            tmp_path,
            monkeypatch,
            game=game,
            export_status="rejected",
        )


def test_phase3_start_builder_is_deterministic_split_safe_and_rust_validated(
    tmp_path: Path,
) -> None:
    engine_ref = write_ref(tmp_path, "target/release/open-shogi-cli", b"engine")
    starts, starts_ref, _, manifest_ref, positions_ref = write_validated_phase3_starts(
        tmp_path,
        engine_ref=engine_ref,
        train_sfen=TRAIN_SFEN,
        validation_sfen=VALIDATION_SFEN,
    )

    first = build_start_positions_from_phase3(
        repository_root=tmp_path,
        positions_ref=positions_ref,
        dataset_manifest_ref=manifest_ref,
        seed=20260808,
        train_count=10,
        validation_count=10,
    )
    second = build_start_positions_from_phase3(
        repository_root=tmp_path,
        positions_ref=positions_ref,
        dataset_manifest_ref=manifest_ref,
        seed=20260808,
        train_count=10,
        validation_count=10,
    )

    assert first == second
    assert parse_start_positions(first) == starts
    assert first["selection"]["excludedCrossSplitStates"] == 1
    assert {row["split"] for row in first["positions"]} == {"train", "validation"}
    assert all(str(row["sfen"]).endswith(" 1") for row in first["positions"])
    runner = _ValidationRunner(tmp_path)
    validation = execute_start_position_validation(
        repository_root=tmp_path,
        start_positions=first,
        start_positions_ref=starts_ref,
        engine_ref=engine_ref,
        git_commit="a399407",
        output_root="artifacts/start-validation",
        timeout_seconds=30,
        runner=runner,
        now=lambda: "2026-08-08T00:00:00Z",
    )
    validate_start_position_validation(
        validation,
        start_positions_ref=starts_ref,
        engine_ref=engine_ref,
        positions=parse_start_positions(first),
        repository_root=tmp_path,
    )
    assert len(runner.commands) == 20
    assert all(command[1:4] == ["perft", "--depth", "0"] for command in runner.commands)

    recovered = execute_start_position_validation(
        repository_root=tmp_path,
        start_positions=first,
        start_positions_ref=starts_ref,
        engine_ref=engine_ref,
        git_commit="a399407",
        output_root="artifacts/start-validation",
        timeout_seconds=30,
        runner=runner,
        now=lambda: "2026-08-08T00:00:01Z",
    )
    assert all("recovery-002" in result["stdout"]["path"] for result in recovered["results"])


def test_phase5_start_loader_requires_phase3_origin_and_exact_rust_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_ref = write_ref(tmp_path, "target/release/open-shogi-cli", b"engine")
    receipt_ref = write_ref(tmp_path, "artifacts/engine-build-receipt.json", b"receipt\n")
    _, starts_ref, validation_ref, dataset_ref, positions_ref = write_validated_phase3_starts(
        tmp_path,
        engine_ref=engine_ref,
        engine_build_receipt=receipt_ref,
        train_sfen=TRAIN_SFEN,
        validation_sfen=VALIDATION_SFEN,
    )
    verified: list[tuple[ArtifactRef, ArtifactRef, str]] = []
    revalidated: list[tuple[ArtifactRef, ArtifactRef, str, int]] = []

    def verify_receipt(
        _root: Path,
        receipt: ArtifactRef,
        *,
        expected_engine: ArtifactRef,
        expected_git_commit: str,
    ) -> None:
        verified.append((receipt, expected_engine, expected_git_commit))

    monkeypatch.setattr(
        "open_shogi_training.selfplay.engine_receipt.validate_engine_build_receipt",
        verify_receipt,
    )

    def revalidate_sfens(
        root: Path,
        *,
        positions,
        engine_ref: ArtifactRef,
        engine_build_receipt: ArtifactRef,
        git_commit: str,
    ) -> None:
        verify_receipt(
            root,
            engine_build_receipt,
            expected_engine=engine_ref,
            expected_git_commit=git_commit,
        )
        revalidated.append((engine_ref, engine_build_receipt, git_commit, len(positions.positions)))

    monkeypatch.setattr(planning_module, "_revalidate_start_sfens", revalidate_sfens)
    loaded = _load_starts(
        tmp_path,
        tmp_path / starts_ref.path,
        tmp_path / validation_ref.path,
        engine_ref=engine_ref,
        build_receipt_ref=receipt_ref,
    )
    assert loaded[1:] == (starts_ref, validation_ref, dataset_ref, positions_ref)
    assert verified == [(receipt_ref, engine_ref, "a399407")]
    assert revalidated == [(engine_ref, receipt_ref, "a399407", 20)]

    other_engine = write_ref(tmp_path, "target/release/other-engine", b"other")
    with pytest.raises(ContractError, match="different engine"):
        _load_starts(
            tmp_path,
            tmp_path / starts_ref.path,
            tmp_path / validation_ref.path,
            engine_ref=other_engine,
            build_receipt_ref=receipt_ref,
        )

    forged = json.loads((tmp_path / starts_ref.path).read_bytes())
    row = forged["positions"][0]
    row["sourceGameSha256"] = "f" * 64
    row["positionId"] = start_position_identity(
        row["sourceGameSha256"], row["positionIndex"], row["sfen"]
    )
    forged_ref = write_ref(tmp_path, starts_ref.path, canonical_json_bytes(forged))
    validation = json.loads((tmp_path / validation_ref.path).read_bytes())
    validation["startPositions"] = forged_ref.as_dict()
    forged_validation_ref = write_ref(
        tmp_path,
        validation_ref.path,
        canonical_json_bytes(validation),
    )
    with pytest.raises(ContractError, match="Phase 3"):
        _load_starts(
            tmp_path,
            tmp_path / forged_ref.path,
            tmp_path / forged_validation_ref.path,
            engine_ref=engine_ref,
            build_receipt_ref=receipt_ref,
        )


def test_forged_legal_evidence_cannot_bypass_receipt_bound_rust_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_ref = write_ref(tmp_path, "target/release/open-shogi-cli", b"engine")
    receipt_ref = write_ref(tmp_path, "artifacts/engine-build-receipt.json", b"receipt\n")
    starts, starts_ref, validation_ref, _, _ = write_validated_phase3_starts(
        tmp_path,
        engine_ref=engine_ref,
        engine_build_receipt=receipt_ref,
        train_sfen=TRAIN_SFEN,
        validation_sfen=VALIDATION_SFEN,
    )
    forged_validation = json.loads((tmp_path / validation_ref.path).read_bytes())
    assert all(row["legal"] is True for row in forged_validation["results"])

    monkeypatch.setattr(
        "open_shogi_training.selfplay.engine_receipt.validate_engine_build_receipt",
        lambda *args, **kwargs: None,
    )

    class Snapshot:
        executable_path = tmp_path / "local/runtime-snapshots/verified-engine"
        closed = False

        def pass_fds(self) -> tuple[int, ...]:
            return ()

        def assert_snapshot_unchanged(self) -> None:
            return None

        def assert_source_unchanged(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    snapshot = Snapshot()
    monkeypatch.setattr(
        labeling_execution.ExecutableSnapshot,
        "create",
        staticmethod(lambda *args, **kwargs: snapshot),
    )
    commands: list[list[object]] = []

    def reject_impossible_sfen(argv, **kwargs):
        kwargs["launch_guard"]()
        commands.append(argv)
        kwargs["launch_guard"]()
        return (
            b"",
            b"too many Pawn pieces: found 19, maximum is 18",
            2,
            False,
            False,
            False,
            1,
            "process_tree_ps_rss_sum",
        )

    monkeypatch.setattr(selfplay_execution, "_run_bounded_process", reject_impossible_sfen)
    with pytest.raises(ContractError, match="too many Pawn pieces"):
        validate_start_position_validation(
            forged_validation,
            start_positions_ref=starts_ref,
            engine_ref=engine_ref,
            engine_build_receipt=receipt_ref,
            positions=starts,
            repository_root=tmp_path,
            runtime_authorization=True,
        )

    assert len(commands) == 1
    assert commands[0][1:4] == ["perft", "--depth", "0"]
    assert snapshot.closed is True
