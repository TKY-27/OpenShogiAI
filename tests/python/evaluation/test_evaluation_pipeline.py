from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from open_shogi_training.evaluation import pipeline as evaluation_pipeline
from open_shogi_training.evaluation.config import (
    DiagnosisConfig,
    EvaluationConfig,
    OfficialModeConfig,
    ResourceConfig,
    TeacherAnalysisConfig,
)
from open_shogi_training.evaluation.pipeline import (
    EvaluationError,
    _assert_exact_game_directory,
    _derive_hard_examples,
    _expected_plan_games,
    _load_and_validate_decision_log,
    _parse_export_rows,
    _recorded_play_config_sha256,
    _validate_decision_event,
    _validate_durable_engine_build_receipt,
)
from open_shogi_training.selfplay.common import ArtifactRef, artifact_ref, write_json_new

INITIAL_SFEN = "lnsgkgsnl/1r5b1/p1pppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1"
NEXT_SFEN = "lnsgkgsnl/1r5b1/p1pppp1pp/6p2/9/2P3P2/PP1PPP1PP/1B5R1/LNSGKGSNL w - 2"


def _config(*, max_hard_examples: int = 2) -> EvaluationConfig:
    return EvaluationConfig(
        official=OfficialModeConfig(
            run_id="phase7-test",
            profile="champion",
            human_sides=("black", "white"),
            nodes=10_000,
            depth=8,
            max_plies=256,
            opening_enabled=False,
            opening_max_plies=24,
        ),
        teacher=TeacherAnalysisConfig("configs/teacher/test.yaml", 25_000),
        diagnosis=DiagnosisConfig(150, 200, 150, max_hard_examples),
        resources=ResourceConfig(4_096, 512),
    )


def _plan() -> dict[str, object]:
    engine = ArtifactRef("local/builds/engine/open-shogi-cli", "e" * 64, 123)
    registry = ArtifactRef("artifacts/registry.json", "r" * 64, 456)
    return {
        "registry": registry.as_dict(),
        "registryRevision": 7,
        "champion": {
            "modelId": "champion-v1",
            "artifact": {
                "path": "artifacts/model.osaval",
                "sha256": "a" * 64,
                "size": 42,
            },
            "payloadSha256": "b" * 64,
            "architectureVersion": "1",
            "quantization": "float32",
        },
        "games": _expected_plan_games(
            config=_config(),
            output_root="artifacts/phase7/test",
            engine=engine,
            registry=registry,
        ),
    }


def _play_config() -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "phase6_human_play_config/v1",
        "configSha256": "0" * 64,
        "humanSide": "black",
        "budgetKind": "nodes",
        "budgetValue": 10_000,
        "depth": 8,
        "initialSfen": INITIAL_SFEN,
        "maxPlies": 256,
        "profile": "champion",
        "modelId": "champion-v1",
        "modelArtifactSha256": "a" * 64,
        "modelPayloadSha256": "b" * 64,
        "architectureVersion": 1,
        "quantization": "float32",
        "registrySha256": "r" * 64,
        "registryRevision": 7,
        "openingArtifactSha256": None,
        "openingArtifactSize": None,
        "openingMaxPlies": 24,
        "transpositionEntries": 16_384,
        "engineName": "OpenShogiAI",
        "engineVersion": "0.1.0",
    }
    record["configSha256"] = _recorded_play_config_sha256(record)
    return record


def _human_event(config_sha: str) -> dict[str, object]:
    return {
        "schema": "phase6_human_decision/v1",
        "ply": 1,
        "actor": "human",
        "modelId": "champion-v1",
        "modelArtifactSha256": "a" * 64,
        "modelPayloadSha256": "b" * 64,
        "configSha256": config_sha,
        "sfenBefore": INITIAL_SFEN,
        "moveUsi": "7g7f",
        "nodes": 0,
        "elapsedMs": 0,
        "depth": 0,
        "pv": [],
        "scoreCp": None,
        "openingBook": False,
    }


def _export_row(name: str) -> dict[str, object]:
    return {
        "schema": "phase3_csa_export/v1",
        "status": "ok",
        "inputFile": name,
        "normalizedCsa": "V2.2\nN+human\nN-OpenShogiAI\n%TORYO\n",
        "initialSfen": INITIAL_SFEN,
        "positionSfens": [INITIAL_SFEN, NEXT_SFEN],
        "usiMoves": ["7g7f"],
        "blackName": "human",
        "whiteName": "OpenShogiAI",
        "terminalReason": "TORYO",
        "outcome": "white_win",
        "resultValidation": "verified",
    }


def test_fixed_plan_has_exact_color_swapped_commands_and_no_opening() -> None:
    plan = _plan()
    games = plan["games"]

    assert [game["humanSide"] for game in games] == ["black", "white"]
    assert all("--opening" not in game["command"]["argv"] for game in games)
    assert all(game["command"]["argv"][0].startswith("local/builds/") for game in games)
    assert games[0]["csaPath"].endswith("official-black.csa")
    assert games[1]["decisionLogPath"].endswith("official-white.decisions.jsonl")


def test_exact_game_directory_accepts_only_the_planned_regular_entries(tmp_path: Path) -> None:
    plan = _plan()
    plan["outputRoot"] = "artifacts/phase7/test"
    directory = tmp_path / "artifacts/phase7/test/games"
    directory.mkdir(parents=True)
    (directory / ".open-shogi-cleanup").mkdir()
    for game in plan["games"]:
        for key in ("csaPath", "decisionLogPath"):
            (directory / Path(game[key]).name).write_text("evidence", encoding="utf-8")

    _assert_exact_game_directory(tmp_path, plan)

    (directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(EvaluationError, match="differs from the fixed plan"):
        _assert_exact_game_directory(tmp_path, plan)

    (directory / "unexpected.txt").unlink()
    (directory / ".open-shogi-cleanup/retained.tmp").write_text("retained", encoding="utf-8")
    with pytest.raises(EvaluationError, match="cleanup directory"):
        _assert_exact_game_directory(tmp_path, plan)


def test_durable_receipt_validation_does_not_reauthorize_current_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = ArtifactRef("local/builds/engine/open-shogi-cli", "e" * 64, 123)
    receipt = ArtifactRef("local/build-receipts/receipt.json", "b" * 64, 456)
    document = {"schema": "open_shogi_engine_build_receipt/v2"}
    observed: dict[str, object] = {}

    def load_document(root: Path, reference: ArtifactRef) -> object:
        assert root == tmp_path
        assert reference == receipt
        return document

    def validate_document(
        raw: object,
        *,
        expected_engine: ArtifactRef,
        expected_git_commit: str,
    ) -> dict[str, object]:
        observed.update(
            raw=raw,
            expected_engine=expected_engine,
            expected_git_commit=expected_git_commit,
        )
        return dict(document)

    monkeypatch.setattr(evaluation_pipeline, "load_json_artifact", load_document)
    monkeypatch.setattr(
        evaluation_pipeline, "validate_engine_build_receipt_document", validate_document
    )
    monkeypatch.setattr(
        evaluation_pipeline,
        "validate_engine_build_receipt",
        lambda *args, **kwargs: pytest.fail("durable validation reauthorized execution"),
    )

    assert (
        _validate_durable_engine_build_receipt(
            tmp_path,
            receipt,
            expected_engine=engine,
            expected_git_commit="a" * 40,
        )
        == document
    )
    assert observed == {
        "raw": document,
        "expected_engine": engine,
        "expected_git_commit": "a" * 40,
    }


def test_human_play_config_digest_has_a_stable_cross_runtime_vector() -> None:
    record = _play_config()

    assert (
        record["configSha256"] == "b58f1b5ed24e28e9eb309990fb4b2fa3547abb7dae8ce789193c586843bb8ef7"
    )


def test_decision_log_is_bound_to_rust_replay_and_config(tmp_path: Path) -> None:
    root = tmp_path
    plan = _plan()
    game = plan["games"][0]
    config = _play_config()
    event = _human_event(str(config["configSha256"]))
    path = root / "artifacts/phase7/test/games/official-black.decisions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in (config, event)
        )
    )
    reference = artifact_ref(root, path.relative_to(root).as_posix())

    observed_config, events = _load_and_validate_decision_log(
        root,
        reference,
        game_plan=game,
        plan=plan,
        export=_export_row("official-black.csa"),
    )
    assert observed_config == config
    assert events == [event]

    tampered = _export_row("official-black.csa")
    tampered["usiMoves"] = ["2g2f"]
    with pytest.raises(EvaluationError, match="differs"):
        _load_and_validate_decision_log(
            root,
            reference,
            game_plan=game,
            plan=plan,
            export=tampered,
        )


def test_ai_and_human_decision_evidence_is_closed() -> None:
    plan = _plan()
    human = _human_event(str(_play_config()["configSha256"]))
    invalid_human = {**human, "nodes": 1}
    with pytest.raises(EvaluationError, match="human move carries"):
        _validate_decision_event(invalid_human, context="decision", plan=plan)

    ai = {
        **human,
        "actor": "ai",
        "nodes": 500,
        "elapsedMs": 10,
        "depth": 4,
        "pv": ["2g2f"],
        "scoreCp": 20,
    }
    with pytest.raises(EvaluationError, match="inconsistent"):
        _validate_decision_event(ai, context="decision", plan=plan)


def test_rust_export_rows_reject_duplicate_names_and_unknown_fields() -> None:
    rows = [_export_row("official-black.csa"), _export_row("official-white.csa")]
    encoded = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)
    assert set(_parse_export_rows(encoded)) == {
        "official-black.csa",
        "official-white.csa",
    }

    duplicate = [rows[0], copy.deepcopy(rows[0])]
    duplicate_bytes = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in duplicate
    )
    with pytest.raises(EvaluationError, match="duplicated"):
        _parse_export_rows(duplicate_bytes)

    rows[1]["unexpected"] = True
    with pytest.raises(EvaluationError, match="schema"):
        _parse_export_rows(
            b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)
        )


def test_jsonl_parser_rejects_duplicate_keys_and_missing_final_lf() -> None:
    good = _export_row("official-white.csa")
    row = json.dumps(good, separators=(",", ":"))
    duplicate_key = row[:-1] + ',"schema":"phase3_csa_export/v1"}'
    other = json.dumps(_export_row("official-black.csa"), separators=(",", ":"))

    with pytest.raises(EvaluationError, match="duplicate JSON key"):
        _parse_export_rows((other + "\n" + duplicate_key + "\n").encode())
    with pytest.raises(EvaluationError, match="LF-terminated"):
        _parse_export_rows((other + "\n" + row).encode())


def test_hard_examples_are_ranked_capped_and_never_automatically_trainable(
    tmp_path: Path,
) -> None:
    root = tmp_path
    analysis_refs = []
    for index, severity in enumerate((175, 300, 220), start=1):
        row = {
            "gameId": "official-black",
            "ply": index,
            "actor": "ai",
            "sfen": f"{INITIAL_SFEN[:-1]}{index}",
            "source": {"csa": {}, "decisionLog": {}},
            "diagnosis": {
                "kind": "search_failure_candidate",
                "severityCp": severity,
                "hardExample": True,
            },
        }
        relative = f"artifacts/phase7/test/analyses/row-{index}.json"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(path, row)
        analysis_refs.append(artifact_ref(root, relative))
    report_ref = ArtifactRef("artifacts/phase7/test/evaluation-report.json", "f" * 64, 1)
    report = {
        "runId": "phase7-test",
        "analysis": [reference.as_dict() for reference in analysis_refs],
    }

    hard = _derive_hard_examples(
        config=_config(max_hard_examples=2),
        repository_root=root,
        report_ref=report_ref,
        report=report,
    )

    assert hard["status"] == "pending_human_review"
    assert hard["autoTrainingEligible"] is False
    assert hard["selected"] == 2
    assert hard["omitted"] == 1
    assert [example["severityCp"] for example in hard["examples"]] == [300, 220]
    expected_id = hashlib.sha256(
        b"phase7_hard_example/v1\0official-black\0"
        + (2).to_bytes(8, "big")
        + b"\0"
        + str(hard["examples"][0]["sfen"]).encode()
    ).hexdigest()
    assert hard["examples"][0]["exampleId"] == expected_id
