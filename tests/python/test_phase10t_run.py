from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
from open_shogi_training.labeling.usi import USICandidate, USIScore, USISearchResult
from open_shogi_training.phase10t_run import (
    Phase10TRunError,
    _analyze_teacher_label,
    _dry_run,
    _label_manifest_row,
    _pure_build_audit,
    _resume_label_prefix,
    _safe_path,
    _teacher_eligible_row,
    _validate_label_result,
)


def test_dry_run_has_no_external_work() -> None:
    args = argparse.Namespace(stage="diagnostics", rung="micro")
    result = _dry_run(args, Path("."))
    assert result["status"] == "dry_run"
    assert result["teacher_calls"] == 0
    assert result["arena_games"] == 0
    assert result["holdout_access"] == "forbidden"
    assert result["writes"] == []


def test_runner_rejects_protected_paths() -> None:
    with pytest.raises(Phase10TRunError, match="protected"):
        _safe_path(Path("/tmp/root"), Path("local/final_holdout/data.jsonl"))


def test_pure_build_status_alone_never_opens_execution(tmp_path: Path) -> None:
    control = tmp_path / "configs/phase10t/pure-build.json"
    control.parent.mkdir(parents=True)
    control.write_text(json.dumps({"status": "implementation_verified"}))
    assert _pure_build_audit(tmp_path)["passed"] is False


def test_pure_build_rejects_missing_artifact_evidence(tmp_path: Path) -> None:
    control = tmp_path / "configs/phase10t/pure-build.json"
    control.parent.mkdir(parents=True)
    receipt = tmp_path / "empty-audit.json"
    receipt.write_text(json.dumps({"status": "passed"}))
    control.write_text(
        json.dumps(
            {
                "status": "implementation_verified",
                "evidence": {
                    "path": receipt.name,
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                },
            }
        )
    )
    assert _pure_build_audit(tmp_path)["passed"] is False
    receipt.write_text("{}")
    assert "hash mismatch" in _pure_build_audit(tmp_path)["reason"]


def _teacher_result(*moves: str) -> USISearchResult:
    return USISearchResult(
        bestmove=moves[0],
        candidates=tuple(
            USICandidate(
                multipv=index,
                score=USIScore("cp", 0),
                pv=(move,),
                depth=1,
                seldepth=1,
                nodes=100_000,
            )
            for index, move in enumerate(moves, start=1)
        ),
        elapsed_ms=1,
    )


def test_teacher_label_allows_short_multipv_when_all_roots_are_legal() -> None:
    row = {"legal_moves": ["7g7f"]}
    _validate_label_result(row, _teacher_result("7g7f"))


def test_teacher_label_accepts_short_multipv_with_uncovered_legal_roots() -> None:
    row = {"legal_moves": ["7g7f", "2g2f"]}
    _validate_label_result(row, _teacher_result("7g7f"))


def test_teacher_selection_excludes_terminal_rows_without_inventing_a_root() -> None:
    terminal = {
        "raw_targets": {"terminal_position": True},
        "legal_moves": None,
    }
    assert _teacher_eligible_row(terminal) is False


def test_teacher_selection_rejects_inconsistent_nonterminal_row() -> None:
    nonterminal = {
        "raw_targets": {"terminal_position": False},
        "legal_moves": None,
    }
    with pytest.raises(Phase10TRunError, match="non-terminal train row"):
        _teacher_eligible_row(nonterminal)


def test_teacher_label_never_retries_valid_partial_multipv() -> None:
    row = {"sfen": "startpos", "legal_moves": ["7g7f", "2g2f", "8h2b+"]}

    class Teacher:
        def __init__(self) -> None:
            self.calls = 0
            self.restarts = 0

        def analyze_with_retry(self, sfen: str, *, nodes: int) -> USISearchResult:
            assert sfen == "startpos"
            assert nodes == 100_000
            self.calls += 1
            return (
                _teacher_result("7g7f")
                if self.calls == 1
                else _teacher_result("7g7f", "2g2f", "8h2b+")
            )

        def close(self) -> None:
            self.restarts += 1

        def start(self) -> None:
            self.restarts += 1

    teacher = Teacher()
    stats = {"teacher_validation_retries": 0}
    result, retries = _analyze_teacher_label(teacher, row, 100_000, retry_stats=stats)
    assert result.bestmove == "7g7f"
    assert retries == 0
    assert teacher.calls == 1
    assert teacher.restarts == 0
    assert stats["teacher_validation_retries"] == 0


def _legacy_label() -> tuple[dict, dict]:
    row = {
        "sfen": "startpos",
        "source": "wcsc",
        "legal_moves": ["7g7f", "2g2f"],
        "wdl": 2,
        "wdl_mask": True,
    }
    label = {
        "schema": "open_shogiai_phase10t_teacher_label/v1",
        "index": 0,
        "sfen": "startpos",
        "represented_position": "startpos",
        "source": "wcsc",
        "split": "train",
        "wdl": 2,
        "wdl_mask": True,
        "teacher": {
            "name": "Apery",
            "nodes": 100_000,
            "multipv": 3,
            "bestmove": "7g7f",
            "primary_score_kind": "cp",
            "primary_score_value": 0,
            "elapsed_ms": 1,
            "candidates": [candidate.as_dict() for candidate in _teacher_result("7g7f").candidates],
        },
    }
    return row, label


def test_legacy_label_gets_observed_masks_without_rewriting() -> None:
    row, label = _legacy_label()
    original = json.dumps(label)
    manifest = _label_manifest_row(0, row, label)
    assert json.dumps(label) == original
    assert manifest["observed_targets"]["observed_candidate_count"] == 1
    assert manifest["observed_targets"]["requested_candidate_count"] == 3
    assert manifest["observed_targets"]["ranking_mask"] is False
    assert manifest["observed_targets"]["policy_mask"] is False
    assert manifest["observed_targets"]["value_mask"] is True
    assert manifest["observed_targets"]["wdl"] == 2


@pytest.mark.parametrize("mutation", ["score", "root", "empty_pv", "wdl", "bool_score", "rank"])
def test_legacy_label_rejects_corrupt_identity_or_candidates(mutation: str) -> None:
    row, label = _legacy_label()
    if mutation == "score":
        label["teacher"]["primary_score_value"] = 1
    elif mutation == "root":
        label["teacher"]["candidates"][0]["pv"] = ["9a9b"]
    elif mutation == "empty_pv":
        label["teacher"]["candidates"][0]["pv"] = []
    elif mutation == "wdl":
        label["wdl"] = 0
    elif mutation == "bool_score":
        label["teacher"]["candidates"][0]["score"]["value"] = False
    else:
        label["teacher"]["candidates"][0]["multipv"] = 2
    with pytest.raises(Phase10TRunError):
        _label_manifest_row(0, row, label)


def _prefix_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, bytes]:
    from open_shogi_training import phase10t_run

    row, label = _legacy_label()
    label["teacher"]["candidates"][0]["pv"] = ["7g7f", "3c3d"]
    attempt = tmp_path / "local/phase10t-runs/labels-100k/attempt-0001"
    attempt.mkdir(parents=True)
    raw = (json.dumps(label) + "\n").encode()
    (attempt / "labels.jsonl").write_bytes(raw)
    teacher_identity = {"binary": {"sha256": "a" * 64}, "threads": 4, "multipv": 3}
    monkeypatch.setattr(phase10t_run, "_teacher_identity", lambda root: teacher_identity)
    receipt = {
        "status": "failed",
        "nodes": 100_000,
        "positions_completed": 1,
        "labels_path": (attempt / "labels.jsonl").relative_to(tmp_path).as_posix(),
        "labels_sha256": hashlib.sha256(raw).hexdigest(),
        "teacher": teacher_identity,
    }
    (attempt / "receipt.json").write_text(json.dumps(receipt))
    return row, receipt, raw


def test_prefix_reuses_exact_bytes_and_replays_complete_pv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, _, raw = _prefix_fixture(tmp_path, monkeypatch)

    class Validator:
        def __init__(self):
            self.calls = []

        def validate(self, sfen, bestmove, pvs, *, configured_multipv):
            self.calls.append((sfen, bestmove, pvs, configured_multipv))

    validator = Validator()
    resumed, manifest, path = _resume_label_prefix(
        tmp_path,
        "100k",
        100_000,
        [row, row],
        validator=validator,
    )
    assert resumed == raw
    assert len(manifest) == 1
    assert path is not None
    assert validator.calls == [("startpos", "7g7f", [["7g7f", "3c3d"]], 3)]


def test_prefix_teacher_change_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, receipt, _ = _prefix_fixture(tmp_path, monkeypatch)
    receipt["teacher"]["threads"] = 1
    path = tmp_path / "local/phase10t-runs/labels-100k/attempt-0001/receipt.json"
    path.write_text(json.dumps(receipt))
    from open_shogi_training import phase10t_run

    monkeypatch.setattr(phase10t_run, "_teacher_identity", lambda root: {"threads": 4})
    with pytest.raises(Phase10TRunError, match="teacher identity changed"):
        _resume_label_prefix(tmp_path, "100k", 100_000, [row, row], validator=object())


def test_prefix_illegal_later_pv_move_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_shogi_training.labeling.legality import LegalityValidationError

    row, _, _ = _prefix_fixture(tmp_path, monkeypatch)

    class Validator:
        def validate(self, *args, **kwargs):
            raise LegalityValidationError("illegal later PV move")

    with pytest.raises(LegalityValidationError, match="later PV"):
        _resume_label_prefix(tmp_path, "100k", 100_000, [row, row], validator=Validator())


def test_campaign_examples_rejects_changed_observed_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_shogi_training import phase10t_run

    row, label = _legacy_label()
    label["observed_targets"] = _label_manifest_row(0, row, label)["observed_targets"]
    label["observed_targets"]["policy_mask"] = True
    monkeypatch.setattr(phase10t_run, "_load_train_rows", lambda root, count: [row])
    with pytest.raises(Phase10TRunError, match="target masks changed"):
        phase10t_run._campaign_examples(tmp_path, [label], expected_nodes=100_000)


def test_zero_candidates_are_rejected() -> None:
    result = USISearchResult(bestmove="7g7f", candidates=(), elapsed_ms=1)
    with pytest.raises(Phase10TRunError, match="candidate count"):
        _validate_label_result({"legal_moves": ["7g7f"]}, result)


@pytest.mark.parametrize("fault", ["receipt_json", "label_json", "count", "bool_count", "newline"])
def test_matching_failed_prefix_corruption_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    row, receipt, raw = _prefix_fixture(tmp_path, monkeypatch)
    parent = tmp_path / "local/phase10t-runs/labels-100k"
    attempt = parent / "attempt-0002"
    attempt.mkdir()
    labels_path = attempt / "labels.jsonl"
    if fault == "label_json":
        raw = b"{corrupt label\n"
    elif fault == "newline":
        raw = raw.rstrip(b"\n")
    labels_path.write_bytes(raw)
    receipt["labels_path"] = labels_path.relative_to(tmp_path).as_posix()
    receipt["labels_sha256"] = hashlib.sha256(raw).hexdigest()
    if fault == "count":
        receipt["positions_completed"] = 0
    elif fault == "bool_count":
        receipt["positions_completed"] = True
    (attempt / "receipt.json").write_text(
        "{corrupt receipt" if fault == "receipt_json" else json.dumps(receipt)
    )
    with pytest.raises(Phase10TRunError, match=r"corrupt|incomplete"):
        _resume_label_prefix(tmp_path, "100k", 100_000, [row, row], validator=object())


def test_irrelevant_complete_attempt_still_allows_valid_failed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, _, raw = _prefix_fixture(tmp_path, monkeypatch)
    attempt = tmp_path / "local/phase10t-runs/labels-100k/attempt-0002"
    attempt.mkdir()
    (attempt / "receipt.json").write_text(json.dumps({"status": "complete", "nodes": 100_000}))

    class Validator:
        def validate(self, *args, **kwargs):
            pass

    resumed, manifest, path = _resume_label_prefix(
        tmp_path,
        "100k",
        100_000,
        [row, row],
        validator=Validator(),
    )
    assert resumed == raw
    assert len(manifest) == 1
    assert "attempt-0001" in path
