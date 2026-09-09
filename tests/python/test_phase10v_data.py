"""Real collector/label schema with mocked external processes; never long teacher work."""

import json
from types import SimpleNamespace

import pytest
from open_shogi_training import phase10v_data as data
from open_shogi_training.labeling.legality import LegalityCoverage
from open_shogi_training.labeling.usi import USICandidate, USIScore, USISearchResult

TRAIN = "4k4/9/9/9/4P4/9/9/9/4K4 b - 1"
VALIDATION = "4k4/9/9/9/5P3/9/9/9/4K4 b - 1"


def ref(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, indent=2))
    return {"name": name, "expected_hash": data.sha256(path)}


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "limits", lambda _root: None)
    roots = []
    guard = {"train": [], "validation": [], "calibration": [], "final_holdout": []}
    for split, sfen in (("train", TRAIN), ("validation", VALIDATION)):
        provenance = {
            "approved": True,
            "source": "approved-external",
            "position_categories": ["approved_external"],
            "source_game_id": split,
            "component_id": split,
            "source_sha256": "a" * 64,
            "original_split": split,
        }
        roots.append({"split": split, "sfen": sfen, "provenance": provenance})
        guard[split].append(
            {
                "component_id": split,
                "position_sha256": data.position_hash(sfen),
                "source_sha256": "a" * 64,
            }
        )
    for split, marker in (("calibration", "d"), ("final_holdout", "e")):
        guard[split] = [
            {"component_id": split, "position_sha256": marker * 64, "source_sha256": "f" * 64}
        ]
    request = {
        "schema": "open_shogiai_phase10v_leaf_request/v1",
        "nodes": 16,
        "leaf_limit": 8,
        "binary": ref(tmp_path, "binary", "mock"),
        "model": ref(tmp_path, "model", "mock-model"),
        "roots": ref(tmp_path, "roots-input.json", roots),
        "split_guard": ref(tmp_path, "guard-input.json", guard),
        "data_policy": ref(
            tmp_path,
            "policy.json",
            {
                "schema": "open_shogiai_phase10v_data_policy/v1",
                "stage_rows": 1,
                "requirements": {
                    split: {
                        "leaf_kinds": {"quiescence_leaf": 1},
                        "position_categories": {"approved_external": 1},
                    }
                    for split in ("train", "validation")
                },
            },
        ),
    }
    model_hash = request["model"]["expected_hash"]

    def run(argv, **kwargs):
        sfen = argv[argv.index("--sfen") + 1]
        receipt = {
            "sfen": sfen,
            "model_format": "OSAVAL03",
            "best_move": "5e5d",
            "nodes": 16,
            "proof": {
                "model_sha256": model_hash,
                "profile": "pure_learned",
                "profile_schema": "open_shogiai_pure_learned_v3_profile/v1",
                "learned_eval_calls": 1,
                **dict.fromkeys(
                    (
                        "handcrafted_eval_calls",
                        "residual_eval_calls",
                        "composite_eval_calls",
                        "book_hits",
                        "teacher_calls",
                        "fallback_count",
                    ),
                    0,
                ),
            },
            "leaf_trace": [
                {
                    "sfen": sfen,
                    "kind": "quiescence_leaf",
                    "ply": 1,
                    "model_sha256": model_hash,
                    "evaluation_ordinal": 1,
                }
            ],
        }
        return SimpleNamespace(stdout=json.dumps(receipt))

    monkeypatch.setattr(data.subprocess, "run", run)
    return request


def label_fixture(tmp_path, monkeypatch):
    request = fixture(tmp_path, monkeypatch)
    collected = tmp_path / "collected"
    data.collect(tmp_path, request, collected)
    identity = {"name": "mock external teacher", "binary": {"sha256": "b" * 64}}
    label_request = {
        "schema": "open_shogiai_phase10v_label_request/v1",
        "nodes": 100000,
        "collection_receipt": {
            "name": "collected/receipt.json",
            "expected_hash": data.sha256(collected / "receipt.json"),
        },
        "teacher_config": ref(tmp_path, "teacher.json", {"mock": True}),
        "teacher_identity_sha256": data.digest(identity),
    }
    from open_shogi_training.labeling import config, fingerprint, legality, usi

    monkeypatch.setattr(config, "load_teacher_config", lambda _: SimpleNamespace(multipv=3))
    monkeypatch.setattr(
        fingerprint,
        "fingerprint_teacher",
        lambda *_: SimpleNamespace(identity_record=lambda: identity),
    )

    class Teacher:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def analyze_with_retry(self, sfen, *, nodes):
            move = "5e5d" if sfen == TRAIN else "4e4d"
            return USISearchResult(
                move, (USICandidate(1, USIScore("cp", 150), (move,), 2, 2, nodes),), 1
            )

    class Validator:
        def __init__(self, *_):
            self.identity = SimpleNamespace(as_dict=lambda: {"binary_sha256": "c" * 64})

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def validate(self, sfen, bestmove, pvs, *, configured_multipv):
            assert pvs == [[bestmove]] and configured_multipv == 3
            return LegalityCoverage(3, 1, 5)

    monkeypatch.setattr(usi, "USIEngine", Teacher)
    monkeypatch.setattr(legality, "RustLegalityValidator", Validator)
    return label_request


def test_collect_label_and_raw_verified_training_streams(tmp_path, monkeypatch):
    request = label_fixture(tmp_path, monkeypatch)
    output = tmp_path / "labels"
    receipt = data.label(tmp_path, request, output)
    assert receipt["teacher_calls"] == receipt["rows"] == 2
    validated = data.verify_training_inputs(
        output / "train.jsonl",
        output / "validation.jsonl",
        output / "receipt.json",
        data.sha256(output / "receipt.json"),
    )
    assert validated["rows"] == 2
    train = json.loads((output / "train.jsonl").read_text())
    assert train["score"] == {"kind": "cp", "value": 150}
    assert len(train["candidates"]) == 1  # Missing MultiPV was not padded.
    assert train["candidates"][0]["child_sfen"].endswith("w - 2")
    assert not (output / "pending-receipt.json").exists()


def test_changed_target_cannot_be_approved_by_rehashing_it(tmp_path, monkeypatch):
    request = label_fixture(tmp_path, monkeypatch)
    output = tmp_path / "labels"
    data.label(tmp_path, request, output)
    path = output / "train.jsonl"
    row = json.loads(path.read_text())
    row["score"]["value"] = 999
    path.write_bytes(data.canonical(row) + b"\n")
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["train"]["expected_hash"] = data.sha256(path)
    receipt_path.write_bytes(data.canonical(receipt))
    with pytest.raises(ValueError, match="raw observations"):
        data.verify_training_inputs(
            path, output / "validation.jsonl", receipt_path, data.sha256(receipt_path)
        )


def test_split_guard_blocks_leaves_in_sealed_partition(tmp_path, monkeypatch):
    request = fixture(tmp_path, monkeypatch)
    guard = json.loads((tmp_path / "guard-input.json").read_text())
    guard["final_holdout"].append(guard["train"][0])
    request["split_guard"] = ref(tmp_path, "guard-input.json", guard)
    with pytest.raises(ValueError, match="crosses original splits"):
        data.collect(tmp_path, request, tmp_path / "collected")
    assert not (tmp_path / "collected").exists()


def test_same_source_game_cannot_cross_split_even_with_different_components(tmp_path, monkeypatch):
    request = fixture(tmp_path, monkeypatch)
    roots = json.loads((tmp_path / "roots-input.json").read_text())
    roots[1]["provenance"]["source_game_id"] = roots[0]["provenance"]["source_game_id"]
    request["roots"] = ref(tmp_path, "roots-input.json", roots)
    with pytest.raises(ValueError, match="source game crosses"):
        data.collect(tmp_path, request, tmp_path / "collected")


def test_raw_candidate_rank_and_replay_coverage_are_checked():
    leaf = {"sfen": TRAIN, "split": "train", "provenance": {}}
    raw = {
        "sfen": TRAIN,
        "bestmove": "5e5d",
        "legality": {"returned_candidates": 1},
        "candidates": [{"multipv": 2, "pv": ["5e5d"], "score": {"kind": "cp", "value": 0}}],
    }
    with pytest.raises(ValueError, match="ranks"):
        data._target(leaf, raw)


def test_bad_pinned_receipt_hash_rejected_before_target_access(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        data.verify_training_inputs(
            tmp_path / "absent-train", tmp_path / "absent-validation", receipt, "0" * 64
        )


def test_noncanonical_position_cannot_evade_split_hashes():
    with pytest.raises(ValueError, match="noncanonical"):
        data.position_hash(TRAIN.replace("4k4", "22k4"))


def test_rejected_legal_replay_publishes_no_approved_dataset(tmp_path, monkeypatch):
    request = label_fixture(tmp_path, monkeypatch)
    from open_shogi_training.labeling import legality

    class RejectingValidator:
        def __init__(self, *_):
            self.identity = SimpleNamespace(as_dict=lambda: {"binary_sha256": "c" * 64})

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def validate(self, *_args, **_kwargs):
            raise ValueError("illegal full PV")

    monkeypatch.setattr(legality, "RustLegalityValidator", RejectingValidator)
    with pytest.raises(ValueError, match="illegal full PV"):
        data.label(tmp_path, request, tmp_path / "rejected")
    assert not (tmp_path / "rejected/receipt.json").exists()
    progress = [
        json.loads(row)
        for row in (tmp_path / "rejected/teacher-progress.jsonl").read_text().splitlines()
    ]
    assert len(progress) == 1 and progress[0]["status"] == "observed_before_replay"


def test_teacher_mate_singleton_is_preserved_only_in_raw(tmp_path, monkeypatch):
    request = label_fixture(tmp_path, monkeypatch)
    from open_shogi_training.labeling import usi

    class MateTeacher:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def analyze_with_retry(self, sfen, *, nodes):
            move = "5e5d" if sfen == TRAIN else "4e4d"
            return USISearchResult(
                move, (USICandidate(1, USIScore("mate", 3), (move,), 2, 2, nodes),), 1
            )

    monkeypatch.setattr(usi, "USIEngine", MateTeacher)
    output = tmp_path / "mate-only"
    with pytest.raises(ValueError, match="split is empty"):
        data.label(tmp_path, request, output)
    receipt = json.loads((output / "pending-receipt.json").read_text())
    assert receipt["rows"] == 0 and receipt["masked_mate_only"] == 2
    assert len(json.loads((output / "teacher_raw.json").read_text())) == 2
    assert not (output / "receipt.json").exists()


def test_pinned_stage_distribution_cannot_claim_missing_source_strata(tmp_path, monkeypatch):
    request = fixture(tmp_path, monkeypatch)
    policy = json.loads((tmp_path / "policy.json").read_text())
    policy["requirements"]["train"]["position_categories"]["tactical_middlegame"] = 1
    request["data_policy"] = ref(tmp_path, "policy.json", policy)
    with pytest.raises(ValueError, match="tactical_middlegame"):
        data.collect(tmp_path, request, tmp_path / "collected")
    assert (tmp_path / "collected/search-progress.jsonl").exists()
    assert not (tmp_path / "collected/receipt.json").exists()
