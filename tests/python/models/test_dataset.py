import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from open_shogi_training.data.gzip_jsonl import iter_jsonl_gzip, write_jsonl_gzip_atomic
from open_shogi_training.labeling.schema import canonical_state_sha256, position_id
from open_shogi_training.models import dataset as dataset_module
from open_shogi_training.models.config import load_feature_config, load_training_config
from open_shogi_training.models.dataset import (
    DeterministicStageSampler,
    ValueDataset,
    _candidate_gap_cp,
    _load_replay_examples,
    _load_training_examples_for_test,
    hash_file,
    load_training_examples,
    split_examples,
)
from open_shogi_training.selfplay.common import ArtifactRef

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
FIXTURE_SFENS = (
    STARTPOS,
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",
    "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3",
    "4k4/9/9/9/4+R4/9/9/9/4K4 w Pp 1",
    "4k4/9/9/9/9/9/9/9/4K4 b - 1",
)


def test_dataset_hash_rejects_final_entry_replacement_after_same_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    moved = tmp_path / "original.bin"
    artifact.write_bytes(b"a" * (2 * 1024 * 1024))
    real_read = dataset_module.os.read
    real_fdopen = dataset_module.os.fdopen
    swapped = False

    def replace_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        result = real_read(descriptor, size)
        if result and not swapped:
            swapped = True
            artifact.rename(moved)
            artifact.write_bytes(b"b" * len(result))
        return result

    class ReplacingReader:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size: int) -> bytes:
            return replace_after_first_read(self.stream.fileno(), size)

        def fileno(self) -> int:
            return self.stream.fileno()

    monkeypatch.setattr(
        dataset_module.os,
        "fdopen",
        lambda descriptor, *args, **kwargs: ReplacingReader(
            real_fdopen(descriptor, *args, **kwargs)
        ),
    )

    with pytest.raises(ValueError, match="remain a regular"):
        hash_file(artifact, max_bytes=3 * 1024 * 1024)


def test_strict_teacher_position_join_and_deterministic_sampler(tmp_path: Path) -> None:
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    training = replace(training, sample_ratio=1.0, stage_ratios=(1 / 3, 1 / 3, 1 / 3))
    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    labels, positions, manifest = _dataset_fixture(tmp_path)

    loaded = _load_training_examples_for_test(labels, positions, manifest, training)

    assert loaded.counts_by_split == {"train": 3, "validation": 1, "test": 1}
    train = split_examples(loaded, "train")
    assert all(example.teacher_mask == 1.0 for example in train)
    assert train[0].policy_agreement == 1.0
    dataset = ValueDataset(train, feature)
    first = DeterministicStageSampler(dataset, training)
    second = DeterministicStageSampler(dataset, training)
    first.set_epoch(7)
    second.set_epoch(7)
    assert list(first) == list(second)
    assert sorted(list(first)) == [0, 1, 2]


def test_replay_materializes_selection_splits_without_fabricating_teacher_targets(
    tmp_path: Path,
) -> None:
    entries = [
        _replay_entry(
            "train",
            hard=True,
            priority=100,
            outcome_kind="black_win",
            outcome_target=1,
        ),
        _replay_entry(
            "validation",
            hard=False,
            priority=2,
            sfen_override=FIXTURE_SFENS[3],
        ),
        _replay_entry("test", hard=False, priority=1, suffix=" w - 1"),
    ]
    path = _write_replay_manifest(tmp_path, entries)

    examples = _load_replay_examples(path)

    assert len(examples) == 2
    assert examples[0].split == "train"
    assert examples[1].split == "validation"
    assert all(example.split != "test" for example in examples)
    assert all(example.teacher_mask == 0.0 for example in examples)
    assert all(example.policy_mask == 0.0 for example in examples)
    assert all(example.outcome_mask == 1.0 for example in examples)
    assert examples[0].outcome_target == 1.0
    assert all(example.teacher_score_kind == "unlabeled" for example in examples)
    assert all(example.teacher_score_value is None for example in examples)

    feature = load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")
    with pytest.raises(ValueError, match="outcome-only"):
        ValueDataset((replace(examples[0], teacher_target=0.25),), feature)


def test_teacher_candidate_gap_is_a_nonnegative_ambiguity_magnitude() -> None:
    candidates = [
        {"score": {"kind": "cp", "value": -40}},
        {"score": {"kind": "cp", "value": 15}},
    ]

    assert _candidate_gap_cp(candidates) == 55

    candidates[1]["score"]["value"] = 1_000_001
    with pytest.raises(ValueError, match="centipawn bound"):
        _candidate_gap_cp(candidates)


def test_replay_rejects_nonterminal_max_plies_as_a_factual_outcome(tmp_path: Path) -> None:
    entry = _replay_entry(
        "train",
        hard=False,
        priority=1,
        outcome_kind="max_plies",
        outcome_target=0,
    )
    path = _write_replay_manifest(tmp_path, [entry])

    with pytest.raises(ValueError, match="outcomeKind is invalid"):
        _load_replay_examples(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda entry: entry.update(outcomeTarget=-1), "disagrees with outcome and side"),
        (lambda entry: entry.update(priority=99), "priority aggregate is inconsistent"),
        (lambda entry: entry.update(canonicalSfen=17), "canonicalSfen is invalid"),
    ],
)
def test_replay_contract_rejects_inconsistent_factual_or_aggregate_fields(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    entry = _replay_entry(
        "train",
        hard=True,
        priority=100,
        outcome_kind="black_win",
        outcome_target=1,
    )
    mutation(entry)
    path = _write_replay_manifest(tmp_path, [entry], consistent_counts=False)

    with pytest.raises(ValueError, match=match):
        _load_replay_examples(path)


def test_teacher_label_artifact_is_bounded_to_global_10000_limit(tmp_path: Path) -> None:
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    _, positions, manifest = _dataset_fixture(tmp_path)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    labels = tmp_path / "too-many-labels.jsonl"
    with labels.open("w", encoding="utf-8", newline="\n") as output:
        for index in range(10_001):
            game_id = hashlib.sha256(f"excess-{index}".encode()).hexdigest()
            row = _teacher_label(
                position_id(game_id, 0),
                game_id,
                0,
                "train",
                "opening",
                manifest_sha256,
            )
            output.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="exceeds 10000 records"):
        load_training_examples(
            labels,
            positions,
            manifest,
            training,
            label_manifest_path=tmp_path / "unreached-label-manifest.json",
        )


def test_production_loader_requires_the_complete_teacher_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    labels, positions, manifest = _dataset_fixture(tmp_path)

    label_manifest = tmp_path / "label-manifest.json"
    label_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(dataset_module, "_validate_label_manifest_v2", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="exactly 10000 records; observed 5"):
        load_training_examples(
            labels,
            positions,
            manifest,
            training,
            label_manifest_path=label_manifest,
        )


def test_historical_legality_receipt_does_not_require_the_current_git_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_path = tmp_path / "local/builds/open-shogi-cli/engine/open-shogi-cli"
    engine_path.parent.mkdir(parents=True)
    engine_path.write_bytes(b"historical engine")
    engine_sha256 = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "local/build-receipts/open-shogi-cli/historical.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text("{}\n", encoding="utf-8")
    receipt_bytes = receipt_path.read_bytes()
    observed: list[object] = []

    def validate_document(raw: object, *, expected_engine: object) -> None:
        observed.extend((raw, expected_engine))

    monkeypatch.setattr(
        "open_shogi_training.selfplay.engine_receipt.validate_engine_build_receipt_document",
        validate_document,
    )
    monkeypatch.setattr(
        "open_shogi_training.selfplay.engine_receipt.validate_engine_build_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "historical evidence requested runtime authorization"
        ),
    )

    dataset_module._validate_legality_validator(
        {
            "path": engine_path.relative_to(tmp_path).as_posix(),
            "sha256": engine_sha256,
            "size": engine_path.stat().st_size,
            "build_receipt": {
                "path": receipt_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "size": len(receipt_bytes),
            },
            "reported_name": "OpenShogiAI fixture",
            "reported_author": None,
        },
        repository_root=tmp_path,
        verify_disk=True,
    )

    assert observed == [
        {},
        ArtifactRef(
            path=engine_path.relative_to(tmp_path).as_posix(),
            sha256=engine_sha256,
            size=engine_path.stat().st_size,
        ),
    ]


def test_migrated_manifest_cannot_rebase_legacy_evidence_to_another_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen evidence filename"):
        dataset_module._validate_label_manifest_migration(
            {
                "schema": "phase4_teacher_label_manifest_migration/v1",
                "legacy_manifest": {
                    "path": "replacement-v1.json",
                    "sha256": "a" * 64,
                    "size": 1,
                },
                "legacy_schema": "phase4_teacher_label_manifest/v1",
                "legacy_updated_at": "2026-08-08T00:00:00Z",
            },
            tmp_path / "manifest.json",
            manifest={},
            labels=(),
        )


def test_strict_join_recomputes_stage_and_cross_split_state_priority(tmp_path: Path) -> None:
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    labels, positions, manifest = _dataset_fixture(tmp_path)
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]

    rows[1]["stage"] = "opening"
    labels.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage mismatch"):
        _load_training_examples_for_test(labels, positions, manifest, training)

    labels, positions, manifest = _dataset_fixture(tmp_path / "split")
    position_rows = list(iter_jsonl_gzip(positions))
    duplicate = dict(position_rows[0])
    duplicate_game_id = hashlib.sha256(b"higher-priority-test-duplicate").hexdigest()
    duplicate.update(
        gameId=duplicate_game_id,
        canonicalSha256=duplicate_game_id,
        rawSha256=hashlib.sha256(b"higher-priority-test-raw").hexdigest(),
        split="test",
    )
    position_rows.append(duplicate)
    priority_positions = positions.with_name("priority-positions.jsonl.gz")
    position_digest = write_jsonl_gzip_atomic(priority_positions, position_rows)
    priority_manifest = manifest.with_name("priority-manifest.json")
    priority_manifest.write_text(
        json.dumps(
            {
                "schema": "phase3_dataset_manifest/v1",
                "artifacts": {
                    priority_positions.name: {
                        "sha256": position_digest.sha256,
                        "size": position_digest.size,
                        "records": position_digest.records,
                    }
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(priority_manifest.read_bytes()).hexdigest()
    label_rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    for label in label_rows:
        label["dataset_manifest_sha256"] = manifest_sha256
    priority_labels = labels.with_name("priority-labels.jsonl")
    priority_labels.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in label_rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical-state split priority"):
        _load_training_examples_for_test(
            priority_labels,
            priority_positions,
            priority_manifest,
            training,
        )


def test_strict_join_checks_manifest_record_count_and_next_sfen(tmp_path: Path) -> None:
    training = load_training_config(PROJECT_ROOT / "configs/training/value_v0_smoke.toml")
    labels, positions, manifest = _dataset_fixture(tmp_path / "records")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["artifacts"][positions.name]["records"] += 1
    manifest.write_text(
        json.dumps(manifest_value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    label_rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    for label in label_rows:
        label["dataset_manifest_sha256"] = manifest_sha256
    labels.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in label_rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="record count"):
        _load_training_examples_for_test(labels, positions, manifest, training)

    labels, positions, manifest = _dataset_fixture(tmp_path / "next-sfen")
    position_rows = list(iter_jsonl_gzip(positions))
    position_rows[0]["nextSfen"] = "not an SFEN"
    invalid_positions = positions.with_name("invalid-positions.jsonl.gz")
    digest = write_jsonl_gzip_atomic(invalid_positions, position_rows)
    invalid_manifest = manifest.with_name("invalid-manifest.json")
    invalid_manifest.write_text(
        json.dumps(
            {
                "schema": "phase3_dataset_manifest/v1",
                "artifacts": {
                    invalid_positions.name: {
                        "sha256": digest.sha256,
                        "size": digest.size,
                        "records": digest.records,
                    }
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(invalid_manifest.read_bytes()).hexdigest()
    label_rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    for label in label_rows:
        label["dataset_manifest_sha256"] = manifest_sha256
    invalid_labels = labels.with_name("invalid-labels.jsonl")
    invalid_labels.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in label_rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SFEN"):
        _load_training_examples_for_test(
            invalid_labels,
            invalid_positions,
            invalid_manifest,
            training,
        )


def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    positions_path = tmp_path / "positions.jsonl.gz"
    definitions = [
        ("train", "opening", 1),
        ("train", "middlegame", 5),
        ("train", "endgame", 9),
        ("validation", "opening", 1),
        ("test", "endgame", 9),
    ]
    position_rows = []
    identities = []
    for fixture_index, ((split, stage, position_index), sfen) in enumerate(
        zip(definitions, FIXTURE_SFENS, strict=True)
    ):
        game_id = hashlib.sha256(f"game-{fixture_index}".encode()).hexdigest()
        identity = position_id(game_id, position_index)
        identities.append((identity, game_id, position_index, split, stage, sfen))
        position_rows.append(
            {
                "schema": "phase3_position/v1",
                "gameId": game_id,
                "canonicalSha256": game_id,
                "rawSha256": hashlib.sha256(f"raw-{fixture_index}".encode()).hexdigest(),
                "sourceId": "fixture",
                "split": split,
                "positionIndex": position_index,
                "sfen": sfen,
                "moveUsi": "7g7f" if fixture_index == 0 else "2g2f",
                "nextSfen": sfen,
                "outcome": "black_win",
                "terminalReason": "TORYO",
                "sideToMove": "black" if " b " in sfen else "white",
                "fullPlies": 12,
                "remainingPlies": 12 - position_index,
                "eligible": True,
                "terminalTail": False,
            }
        )
    digest = write_jsonl_gzip_atomic(positions_path, position_rows)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema": "phase3_dataset_manifest/v1",
        "artifacts": {
            positions_path.name: {
                "sha256": digest.sha256,
                "size": digest.size,
                "records": digest.records,
            }
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        _teacher_label(identity, game_id, index, split, stage, manifest_sha256, sfen=sfen)
        for identity, game_id, index, split, stage, sfen in identities
    ]
    labels_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return labels_path, positions_path, manifest_path


def _teacher_label(
    identity: str,
    game_id: str,
    index: int,
    split: str,
    stage: str,
    manifest_sha256: str,
    *,
    sfen: str = STARTPOS,
) -> dict[str, object]:
    score = {"kind": "cp", "value": 100}
    candidate = {
        "multipv": 1,
        "score": score,
        "pv": ["7g7f"],
        "depth": 1,
        "seldepth": 1,
        "nodes": 1,
    }
    return {
        "schema": "phase4_teacher_label/v1",
        "position_id": identity,
        "canonical_sfen": sfen,
        "canonical_state_sha256": canonical_state_sha256(sfen),
        "side_to_move": "black" if " b " in sfen else "white",
        "score": score,
        "score_pov": "side_to_move",
        "bestmove": "7g7f",
        "candidates": [candidate],
        "depth": 1,
        "seldepth": 1,
        "nodes": 1,
        "elapsed_ms": 1,
        "teacher": {
            "name": "fixture",
            "version": "1",
            "reported_name": "fixture",
            "reported_author": None,
            "binary_sha256": "a" * 64,
            "binary_size": 1,
            "eval_files": [{"path": "eval.bin", "sha256": "b" * 64, "size": 1}],
            "options": {"MultiPV": 1},
        },
        "dataset_manifest_sha256": manifest_sha256,
        "config_sha256": "c" * 64,
        "parser_version": "usi-info/v1",
        "created_at": "2026-08-08T00:00:00Z",
        "split": split,
        "game_id": game_id,
        "position_index": index,
        "stage": stage,
        "source_id": "fixture",
        "outcome": "black_win",
    }


def _replay_entry(
    split: str,
    *,
    hard: bool,
    priority: int,
    suffix: str = " b - 1",
    sfen_override: str | None = None,
    outcome_kind: str = "draw",
    outcome_target: int = 0,
) -> dict[str, object]:
    sfen = sfen_override or STARTPOS.replace(" b - 1", suffix)
    side = "black" if " b " in sfen else "white"
    origin = {
        "positionId": hashlib.sha256((sfen + "origin").encode()).hexdigest(),
        "generationId": "generation-1",
        "generationOrdinal": 1,
        "sourceType": "selfplay",
        "sourceManifest": {"path": "source.json", "sha256": "d" * 64, "size": 1},
        "sourceGameId": "game-1",
        "sourcePly": 1,
        "sideToMove": side,
        "outcomeKind": outcome_kind,
        "outcomeTarget": outcome_target,
        "stage": "opening",
        "priority": priority,
        "hardPosition": hard,
        "isNew": True,
    }
    return {
        "entryId": hashlib.sha256(sfen.encode()).hexdigest(),
        "canonicalSfen": sfen,
        "sideToMove": side,
        "outcomeKind": outcome_kind,
        "outcomeTarget": outcome_target,
        "stage": "opening",
        "priority": priority,
        "hardPosition": hard,
        "split": split,
        "newestGenerationOrdinal": 1,
        "isNew": True,
        "origins": [origin],
    }


def _write_replay_manifest(
    tmp_path: Path,
    entries: list[dict[str, object]],
    *,
    consistent_counts: bool = True,
) -> Path:
    entries.sort(key=lambda entry: (entry["split"], -entry["priority"], entry["entryId"]))
    counts = {
        "input": len(entries),
        "unique": len(entries),
        "retained": len(entries),
        "newRetained": sum(bool(entry["isNew"]) for entry in entries),
        "olderRetained": sum(not bool(entry["isNew"]) for entry in entries),
        "hardRetained": sum(bool(entry["hardPosition"]) for entry in entries),
        "outcomeTargets": {
            "loss": sum(entry["outcomeTarget"] == -1 for entry in entries),
            "draw": sum(entry["outcomeTarget"] == 0 for entry in entries),
            "win": sum(entry["outcomeTarget"] == 1 for entry in entries),
        },
        "splits": {
            split: sum(entry["split"] == split for entry in entries)
            for split in ("train", "validation", "test")
        },
    }
    if not consistent_counts:
        # Entry-level validation must run before summary validation in negative fixtures.
        counts["input"] = len(entries)
    manifest = {
        "schema": "phase6_replay_buffer_manifest/v1",
        "generationId": "generation-1",
        "inputCandidates": {"path": "candidates.json", "sha256": "a" * 64, "size": 1},
        "configSha256": "b" * 64,
        "dedupKey": "canonical_sfen",
        "capacity": 100,
        "minimumOlderPositions": 1,
        "counts": counts,
        "splitPolicy": {
            "train": "training",
            "validation": "selection_metrics_only",
            "test": "final_evaluation_only",
        },
        "entries": entries,
        "deletionLog": [],
    }
    manifest["manifestSha256"] = _canonical_sha256(manifest)
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    return path


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
