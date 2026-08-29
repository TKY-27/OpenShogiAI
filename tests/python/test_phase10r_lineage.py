from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from open_shogi_training.phase10r_campaign import Phase10RCampaignError, select_hard
from open_shogi_training.phase10r_lineage import (
    Phase10RLineageError,
    completed_teacher_bound_candidates,
    load_teacher_binding_control,
    validate_candidate_lineage,
)
from open_shogi_training.phase10r_model import VARIANT_PAIR, VARIANT_PRIMARY
from open_shogi_training.phase10r_training import (
    CHECKPOINT_SCHEMA,
    Phase10RModel,
    export_osaval02_artifact,
)

ROOT = Path(__file__).resolve().parents[2]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _ref(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def test_live_teacher_binding_control_proves_case3_and_exact_parents() -> None:
    control = load_teacher_binding_control(ROOT)

    assert control["repair_case"] == "execution_sequence_skipped_existing_teacher_binding_stage"
    assert control["teacher_binding_identity"]["identity_sha256"] == (
        "781a45570ce6c88e29e4c9f7f3acb96e3981bb74d7da6fb10ff80cdd76f51cbf"
    )
    assert control["calibration"]["expected_rows"] == {
        "train": 6570,
        "validation": 1880,
        "validation_cp_for_affine_fit": 1831,
    }
    assert control["calibration"]["label_budget"]["new_teacher_calls"] == 0
    candidates = completed_teacher_bound_candidates(ROOT, "1m")
    assert {candidate["variant_id"] for candidate in candidates} == {
        VARIANT_PAIR,
        VARIANT_PRIMARY,
    }
    assert all(
        candidate["teacher_binding"]["identity_sha256"]
        == control["teacher_binding_identity"]["identity_sha256"]
        for candidate in candidates
    )


def test_candidate_lineage_json_schema_is_closed_and_requires_teacher_evidence() -> None:
    schema = json.loads((ROOT / "docs/model/phase10r-candidate-lineage.schema.json").read_text())

    assert schema["additionalProperties"] is False
    assert schema["properties"]["status"]["const"] == "completed"
    assert schema["properties"]["binding_version"]["const"] == "teacher-bound-v1"
    assert "teacher_binding" in schema["required"]
    assert "parity_receipt" in schema["properties"]["candidate"]["required"]


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        ([], "not authorized without a completed teacher-bound candidate"),
        ([{"status": "completed"}], "gate passed, but hard-example selection execution"),
    ],
)
def test_select_hard_distinguishes_teacher_binding_from_selection_execution(
    monkeypatch: pytest.MonkeyPatch, candidates: list[dict[str, str]], message: str
) -> None:
    monkeypatch.setattr(
        "open_shogi_training.phase10r_campaign._manifest_identity",
        lambda root, scale: (Path("manifest.json"), {}, "1" * 64),
    )
    monkeypatch.setattr(
        "open_shogi_training.phase10r_campaign._teacher_identity",
        lambda root: {"name": "Apery", "version": "2.0.0"},
    )
    monkeypatch.setattr(
        "open_shogi_training.phase10r_lineage.completed_teacher_bound_candidates",
        lambda root, scale: candidates,
    )

    with pytest.raises(Phase10RCampaignError, match=message):
        select_hard(Path("."), "1m", git_commit="a" * 40)


@pytest.fixture
def valid_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object]]:
    preparation = _write(tmp_path, "local/preparation.json", b"preparation\n")
    receipt = _write(tmp_path, "local/train-receipt.json", b"receipt\n")
    checkpoint = _write(tmp_path, "local/stage2.pt", b"stage2 checkpoint\n")
    label_manifest = _write(tmp_path, "artifacts/labels/manifest.json", b"manifest\n")
    labels = _write(tmp_path, "artifacts/labels/labels.jsonl", b"label\n")
    benchmark = _write(tmp_path, "artifacts/labels/benchmark.json", b"benchmark\n")
    control_file = _write(tmp_path, "configs/phase10r/teacher-binding.yaml", b"synthetic control\n")
    calibration_train = _write(tmp_path, "local/calibration/train.jsonl", b"train row\n")
    calibration_validation = _write(
        tmp_path, "local/calibration/validation.jsonl", b"validation row\n"
    )
    identity_sha256 = "1" * 64
    calibration_value = {
        "schema": "open_shogiai_phase10r_teacher_calibration_input/v1",
        "status": "passed",
        "binding_version": "teacher-bound-v1",
        "scale": "1m",
        "control_sha256": control_file["sha256"],
        "teacher_binding_identity_sha256": identity_sha256,
        "preparation_manifest_sha256": "2" * 64,
        "label_manifest_sha256": label_manifest["sha256"],
        "labels_sha256": labels["sha256"],
        "rows": {"train": 6570, "validation": 1880, "validation_cp": 1831},
        "excluded_rows": {
            "phase4_test": 1064,
            "absent_from_current_preparation": 467,
            "split_mismatch": 19,
        },
        "new_teacher_calls": 0,
        "files": {"train": calibration_train, "validation": calibration_validation},
    }
    calibration = _write(
        tmp_path, "local/calibration/input-manifest.json", _json_bytes(calibration_value)
    )
    parent_model = Phase10RModel(VARIANT_PAIR, seed=8)
    parent_path = tmp_path / "local/pretraining.osaval02"
    export_osaval02_artifact(
        parent_model,
        parent_path,
        quantization="float32",
        dataset_manifest_sha256="2" * 64,
        training_run_reference=f"phase10r-1m-{VARIANT_PAIR}-pretraining",
        git_commit="a" * 40,
    )
    parent_artifact = _ref(tmp_path, parent_path)
    model = Phase10RModel(VARIANT_PAIR, seed=9)
    output_directory = (
        tmp_path / "local/phase10r-data/checkpoints/phase10r/1m" / VARIANT_PAIR / "teacher-bound-v1"
    )
    output_directory.mkdir(parents=True)
    stage3_path = output_directory / "stage3.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "variant_id": VARIANT_PAIR,
            "manifest_sha256": calibration["sha256"],
            "stage_id": "packed_sfen_value_and_ranking",
            "completed": True,
            "parent_checkpoint_sha256": checkpoint["sha256"],
            "teacher_binding_identity_sha256": identity_sha256,
            "model_state": model.state_dict(),
        },
        stage3_path,
    )
    stage3 = _ref(tmp_path, stage3_path)
    stage4_value = {
        "schema": "open_shogiai_phase10r_teacher_calibration/v1",
        "status": "passed",
        "variant_id": VARIANT_PAIR,
        "input_manifest_sha256": calibration["sha256"],
        "stage3_checkpoint_sha256": stage3["sha256"],
        "teacher_binding_identity_sha256": identity_sha256,
        "method": "positive_monotonic_affine",
        "fit_rows": 1831,
        "calibration_scale": 1000.0,
        "calibration_bias": 0.0,
        "new_teacher_calls": 0,
    }
    stage4 = _write(
        tmp_path,
        stage3_path.with_name("calibration.json").relative_to(tmp_path).as_posix(),
        _json_bytes(stage4_value),
    )
    candidate_path = (
        tmp_path
        / "local/phase10r-data/checkpoints/phase10r/1m"
        / VARIANT_PAIR
        / "teacher-bound-v1"
        / f"{VARIANT_PAIR}.osaval02"
    )
    export = export_osaval02_artifact(
        model,
        candidate_path,
        quantization="float32",
        dataset_manifest_sha256=calibration["sha256"],
        training_run_reference=f"phase10r-1m-{VARIANT_PAIR}-teacher-bound-v1",
        git_commit="a" * 40,
    )
    candidate = {
        "path": candidate_path.relative_to(tmp_path).as_posix(),
        "sha256": export["sha256"],
        "bytes": export["bytes"],
    }
    parity_value = {
        "schema": "open_shogiai_phase10r_candidate_parity/v1",
        "status": "passed",
        "variant_id": VARIANT_PAIR,
        "artifact_sha256": candidate["sha256"],
        "python_native_wasm": "passed",
        "incremental_full_recompute_unmake": "passed",
    }
    parity = _write(
        tmp_path,
        f"local/phase10r-data/checkpoints/phase10r/1m/{VARIANT_PAIR}/teacher-bound-v1/parity.json",
        _json_bytes(parity_value),
    )
    control = {
        "scale": "1m",
        "pretraining": {
            "preparation_manifest": {
                "path": preparation["path"],
                "file_sha256": preparation["sha256"],
                "manifest_sha256": "2" * 64,
            },
            "variants": [
                {
                    "variant_id": VARIANT_PAIR,
                    "training_receipt": {
                        "path": receipt["path"],
                        "sha256": receipt["sha256"],
                    },
                    "stage2_checkpoint": {
                        "path": checkpoint["path"],
                        "sha256": checkpoint["sha256"],
                    },
                    "stage2_artifact": {
                        "path": parent_artifact["path"],
                        "sha256": parent_artifact["sha256"],
                    },
                }
            ],
        },
        "teacher_binding_identity": {
            "identity_sha256": identity_sha256,
            "labels": {
                "manifest": {
                    "path": label_manifest["path"],
                    "sha256": label_manifest["sha256"],
                },
                "rows": {
                    "path": labels["path"],
                    "sha256": labels["sha256"],
                    "records": 10000,
                },
                "benchmark": {
                    "path": benchmark["path"],
                    "sha256": benchmark["sha256"],
                },
            },
        },
        "calibration": {
            "output": {
                "directory_template": "local/phase10r-data/checkpoints/phase10r/"
                "{scale}/{variant}/teacher-bound-v1"
            }
        },
    }
    monkeypatch.setattr(
        "open_shogi_training.phase10r_lineage.load_teacher_binding_control",
        lambda root: control,
    )
    parent = {
        "preparation_manifest": preparation,
        "training_receipt": receipt,
        "stage2_checkpoint": checkpoint,
        "stage2_artifact": parent_artifact,
    }
    lineage = {
        "schema": "open_shogiai_phase10r_candidate_lineage/v1",
        "status": "completed",
        "binding_version": "teacher-bound-v1",
        "scale": "1m",
        "variant_id": VARIANT_PAIR,
        "created_at_utc": "2026-08-29T12:00:00Z",
        "git_commit": "a" * 40,
        "parent": parent,
        "teacher_binding": {
            "identity_sha256": identity_sha256,
            "control": control_file,
            "label_manifest": label_manifest,
            "labels": labels,
            "benchmark": benchmark,
            "calibration_input_manifest": calibration,
        },
        "stages": [
            {
                "order": 3,
                "stage_id": "packed_sfen_value_and_ranking",
                "status": "passed",
                "input_manifest": calibration,
                "output": stage3,
            },
            {
                "order": 4,
                "stage_id": "approved_teacher_calibration",
                "status": "passed",
                "input_manifest": calibration,
                "output": stage4,
            },
        ],
        "candidate": {
            "artifact": candidate,
            "osaval02": {
                "variant_id": VARIANT_PAIR,
                "dataset_manifest_sha256": calibration["sha256"],
                "training_run_reference": f"phase10r-1m-{VARIANT_PAIR}-teacher-bound-v1",
            },
            "parity_receipt": parity,
        },
        "acceptance": {
            "status": "passed",
            "parent_immutable": True,
            "new_teacher_calls": 0,
            "forbidden_split_rows": 0,
            "calibration_scale": 1000.0,
            "calibration_bias": 0.0,
            "python_native_wasm_parity": "passed",
            "incremental_parity": "passed",
            "source_held_out_regression_maximum": 0.005,
            "calibration_ece_regression": 0.001,
            "new_tactical_failures": 0,
            "throughput_floor_passed": True,
        },
    }
    lineage_path = (
        tmp_path
        / "local/phase10r-data/checkpoints/phase10r/1m"
        / VARIANT_PAIR
        / "teacher-bound-v1/candidate-lineage.json"
    )
    lineage_path.write_bytes(_json_bytes(lineage))
    return tmp_path, lineage_path, lineage


def test_completed_candidate_lineage_validates_every_bound_artifact(
    valid_lineage: tuple[Path, Path, dict[str, object]],
) -> None:
    root, path, lineage = valid_lineage

    assert validate_candidate_lineage(root, path) == lineage


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("teacher", "teacher identity"),
        ("label", "labels changed"),
        ("parity", "parity receipt"),
        ("parity-extra", "parity receipt keys"),
        ("acceptance", "acceptance gates"),
    ],
)
def test_candidate_lineage_rejects_identity_and_gate_drift(
    valid_lineage: tuple[Path, Path, dict[str, object]], mutation: str, message: str
) -> None:
    root, path, original = valid_lineage
    lineage = copy.deepcopy(original)
    if mutation == "teacher":
        lineage["teacher_binding"]["identity_sha256"] = "f" * 64
    elif mutation == "label":
        lineage["teacher_binding"]["labels"]["sha256"] = "e" * 64
    elif mutation == "parity":
        lineage["candidate"]["parity_receipt"]["sha256"] = "d" * 64
    elif mutation == "parity-extra":
        parity_path = root / lineage["candidate"]["parity_receipt"]["path"]
        parity = json.loads(parity_path.read_text())
        parity["unbound_note"] = "must be rejected"
        lineage["candidate"]["parity_receipt"] = _write(
            root,
            parity_path.relative_to(root).as_posix(),
            _json_bytes(parity),
        )
    else:
        lineage["acceptance"]["new_teacher_calls"] = 1
    path.write_bytes(_json_bytes(lineage))

    with pytest.raises(Phase10RLineageError, match=message):
        validate_candidate_lineage(root, path)


def test_candidate_lineage_rejects_metadata_only_rebinding(
    valid_lineage: tuple[Path, Path, dict[str, object]],
) -> None:
    root, path, lineage = valid_lineage
    parent_path = root / lineage["parent"]["stage2_artifact"]["path"]
    stage3_path = root / lineage["stages"][0]["output"]["path"]
    checkpoint = torch.load(stage3_path, map_location="cpu", weights_only=True)
    candidate_path = root / lineage["candidate"]["artifact"]["path"]
    parent_model = Phase10RModel(VARIANT_PAIR, seed=8)
    checkpoint["model_state"] = parent_model.state_dict()
    torch.save(checkpoint, stage3_path)
    lineage["stages"][0]["output"] = _ref(root, stage3_path)
    stage4_path = root / lineage["stages"][1]["output"]["path"]
    stage4 = json.loads(stage4_path.read_text())
    stage4["stage3_checkpoint_sha256"] = lineage["stages"][0]["output"]["sha256"]
    lineage["stages"][1]["output"] = _write(
        root, stage4_path.relative_to(root).as_posix(), _json_bytes(stage4)
    )
    candidate_path.unlink()
    export = export_osaval02_artifact(
        parent_model,
        candidate_path,
        quantization="float32",
        dataset_manifest_sha256=lineage["teacher_binding"]["calibration_input_manifest"]["sha256"],
        training_run_reference=f"phase10r-1m-{VARIANT_PAIR}-teacher-bound-v1",
        git_commit="a" * 40,
    )
    lineage["candidate"]["artifact"] = {
        "path": candidate_path.relative_to(root).as_posix(),
        "sha256": export["sha256"],
        "bytes": export["bytes"],
    }
    parity_path = root / lineage["candidate"]["parity_receipt"]["path"]
    parity = json.loads(parity_path.read_text())
    parity["artifact_sha256"] = export["sha256"]
    lineage["candidate"]["parity_receipt"] = _write(
        root, parity_path.relative_to(root).as_posix(), _json_bytes(parity)
    )
    path.write_bytes(_json_bytes(lineage))

    assert parent_path.is_file()
    with pytest.raises(Phase10RLineageError, match="only changed pretraining metadata"):
        validate_candidate_lineage(root, path)
