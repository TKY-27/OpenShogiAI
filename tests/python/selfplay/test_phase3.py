from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from open_shogi_training.data.gzip_jsonl import iter_jsonl_gzip, write_jsonl_gzip_atomic
from open_shogi_training.selfplay.common import ArtifactRef, ContractError, canonical_json_bytes
from open_shogi_training.selfplay.phase3 import (
    _validate_rust_exports,
    validate_phase3_artifacts,
)

from .conftest import _write_closed_phase3_fixture, write_ref


def _fixture(
    root: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    ArtifactRef,
    ArtifactRef,
]:
    positions_ref, manifest_ref, manifest = _write_closed_phase3_fixture(root)
    games = list(iter_jsonl_gzip(root / "data/phase3/games-00000.jsonl.gz"))
    positions = list(iter_jsonl_gzip(root / positions_ref.path))
    return copy.deepcopy(manifest), games, positions, manifest_ref, positions_ref


def _rewrite_chain(
    root: Path,
    manifest: dict[str, object],
    games: Sequence[Mapping[str, object]],
    positions: Sequence[Mapping[str, object]],
) -> tuple[ArtifactRef, ArtifactRef]:
    phase3 = root / "data/phase3"
    games_path = phase3 / "games-00000.jsonl.gz"
    positions_path = phase3 / "positions-00000.jsonl.gz"
    games_path.unlink()
    positions_path.unlink()
    games_digest = write_jsonl_gzip_atomic(games_path, games)
    positions_digest = write_jsonl_gzip_atomic(positions_path, positions)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["games-00000.jsonl.gz"] = {
        "sha256": games_digest.sha256,
        "size": games_digest.size,
        "records": len(games),
    }
    artifacts["positions-00000.jsonl.gz"] = {
        "sha256": positions_digest.sha256,
        "size": positions_digest.size,
        "records": len(positions),
    }
    manifest["counts"] = {"games": len(games), "positions": len(positions)}
    manifest["canonicalGameSha256"] = sorted(str(game["canonicalSha256"]) for game in games)
    manifest["rawObjectSha256"] = sorted(
        str(game["rawObject"]["sha256"])
        for game in games  # type: ignore[index]
    )
    positions_ref = ArtifactRef(
        "data/phase3/positions-00000.jsonl.gz",
        positions_digest.sha256,
        positions_digest.size,
    )
    manifest_ref = write_ref(
        root,
        "data/phase3/mutated-manifest.json",
        canonical_json_bytes(manifest),
    )
    return manifest_ref, positions_ref


def _snapshot_identity(snapshot: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        snapshot[key]
        for key in (
            "evidence_id",
            "url",
            "retrieved_at",
            "sha256",
            "size",
            "content_type",
            "object_path",
        )
    )


def _second_retrieval(snapshot: Mapping[str, object]) -> dict[str, object]:
    payload = b"fixture second immutable evidence retrieval"
    sha256 = hashlib.sha256(payload).hexdigest()
    result = dict(snapshot)
    result.update(
        {
            "retrieved_at": "2026-08-09T00:00:00Z",
            "sha256": sha256,
            "size": len(payload),
            "object_path": f"evidence/sha256/{sha256[:2]}/{sha256}",
        }
    )
    return result


def _exports(
    games: Sequence[Mapping[str, object]],
    positions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_game: dict[str, list[Mapping[str, object]]] = {}
    for row in positions:
        by_game.setdefault(str(row["canonicalSha256"]), []).append(row)
    exports: list[dict[str, object]] = []
    for index, game in enumerate(games, start=1):
        rows = sorted(
            by_game[str(game["canonicalSha256"])],
            key=lambda row: int(row["positionIndex"]),
        )
        players = game["players"]
        assert isinstance(players, dict)
        black = players["black"]
        white = players["white"]
        assert isinstance(black, dict) and isinstance(white, dict)
        exports.append(
            {
                "schema": "phase3_csa_export/v1",
                "status": "ok",
                "inputFile": f"game-{index:06d}.csa",
                "normalizedCsa": game["normalizedCsa"],
                "initialSfen": game["initialSfen"],
                "positionSfens": [row["sfen"] for row in rows],
                "usiMoves": game["usiMoves"],
                "blackName": black["name"],
                "whiteName": white["name"],
                "terminalReason": game["terminalReason"],
                "outcome": game["outcome"],
                "resultValidation": game["resultValidation"],
            }
        )
    return exports


def test_phase3_rust_export_accepts_only_the_historical_verified_repetition_upgrade(
    tmp_path: Path,
) -> None:
    _, games, positions, _, _ = _fixture(tmp_path)
    exports = _exports(games, positions)
    games[0]["outcome"] = "unknown"
    games[0]["terminalReason"] = "SENNICHITE"
    games[0]["resultValidation"] = "verified"
    exports[0]["outcome"] = "draw"
    exports[0]["terminalReason"] = "SENNICHITE"
    exports[0]["resultValidation"] = "verified"

    _validate_rust_exports(games, exports)

    for key, value in (
        ("terminalReason", "HIKIWAKE"),
        ("resultValidation", "external_condition"),
        ("outcome", "black_win"),
    ):
        changed = copy.deepcopy(exports)
        changed[0][key] = value
        with pytest.raises(ContractError, match="differs from the durable Phase 3 game"):
            _validate_rust_exports(games, changed)


def test_phase3_evidence_allows_multiple_retrieval_versions_with_exact_per_game_ids(
    tmp_path: Path,
) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    snapshots = manifest["evidenceSnapshots"]
    assert isinstance(snapshots, list)
    second = _second_retrieval(snapshots[0])
    snapshots.append(second)
    snapshots.sort(key=_snapshot_identity)
    decision = games[0]["licenseDecision"]
    assert isinstance(decision, dict)
    game_snapshots = decision["evidenceSnapshots"]
    assert isinstance(game_snapshots, list)
    game_snapshots[0] = second
    game_snapshots.sort(key=_snapshot_identity)
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    validated = validate_phase3_artifacts(
        repository_root=tmp_path,
        manifest=manifest,
        dataset_manifest_ref=manifest_ref,
        positions_ref=positions_ref,
    )

    assert len(validated.games) == 20
    assert len(validated.positions) == 80


@pytest.mark.parametrize("defect", ["url_drift", "cross_id_reuse"])
def test_phase3_manifest_rejects_split_evidence_identities(
    tmp_path: Path,
    defect: str,
) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    snapshots = manifest["evidenceSnapshots"]
    assert isinstance(snapshots, list)
    changed = _second_retrieval(snapshots[0])
    if defect == "url_drift":
        changed["url"] = "https://example.invalid/drifted-license"
    else:
        changed["evidence_id"] = "different-evidence-id"
        changed["retrieved_at"] = "2026-08-10T00:00:00Z"
        changed["sha256"] = snapshots[0]["sha256"]
        changed["size"] = snapshots[0]["size"]
        changed["object_path"] = snapshots[0]["object_path"]
    snapshots.append(changed)
    snapshots.sort(key=_snapshot_identity)
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    with pytest.raises((ContractError, ValueError), match=r"evidence ID|reuse"):
        validate_phase3_artifacts(
            repository_root=tmp_path,
            manifest=manifest,
            dataset_manifest_ref=manifest_ref,
            positions_ref=positions_ref,
        )


def test_phase3_game_rejects_duplicate_or_missing_required_evidence_id(
    tmp_path: Path,
) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    decision = games[0]["licenseDecision"]
    assert isinstance(decision, dict)
    snapshots = decision["evidenceSnapshots"]
    assert isinstance(snapshots, list) and len(snapshots) == 2
    snapshots[1] = copy.deepcopy(snapshots[0])
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    with pytest.raises(ContractError, match="missing, duplicated, or unapproved"):
        validate_phase3_artifacts(
            repository_root=tmp_path,
            manifest=manifest,
            dataset_manifest_ref=manifest_ref,
            positions_ref=positions_ref,
        )


def test_phase3_game_evidence_union_must_cover_every_manifest_retrieval(
    tmp_path: Path,
) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    snapshots = manifest["evidenceSnapshots"]
    assert isinstance(snapshots, list)
    snapshots.append(_second_retrieval(snapshots[0]))
    snapshots.sort(key=_snapshot_identity)
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    with pytest.raises(ContractError, match="evidence union"):
        validate_phase3_artifacts(
            repository_root=tmp_path,
            manifest=manifest,
            dataset_manifest_ref=manifest_ref,
            positions_ref=positions_ref,
        )


def test_phase3_manifest_evidence_snapshots_require_canonical_order(tmp_path: Path) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    snapshots = manifest["evidenceSnapshots"]
    assert isinstance(snapshots, list) and len(snapshots) > 1
    snapshots.reverse()
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    with pytest.raises(ContractError, match="canonical order"):
        validate_phase3_artifacts(
            repository_root=tmp_path,
            manifest=manifest,
            dataset_manifest_ref=manifest_ref,
            positions_ref=positions_ref,
        )


def test_phase3_game_preserves_nonsemantic_acquisition_evidence_order(tmp_path: Path) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    decision = games[0]["licenseDecision"]
    assert isinstance(decision, dict)
    snapshots = decision["evidenceSnapshots"]
    assert isinstance(snapshots, list) and len(snapshots) > 1
    snapshots.reverse()
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    validated = validate_phase3_artifacts(
        repository_root=tmp_path,
        manifest=manifest,
        dataset_manifest_ref=manifest_ref,
        positions_ref=positions_ref,
    )

    assert len(validated.games) == 20


def test_phase3_raw_to_canonical_binding_rejects_rehashed_raw_from_another_game(
    tmp_path: Path,
) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    raw_csa = "'distinct acquisition wrapper\n" + str(games[1]["normalizedCsa"])
    raw_bytes = raw_csa.encode()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    games[0]["rawCsa"] = raw_csa
    raw_object = games[0]["rawObject"]
    assert isinstance(raw_object, dict)
    raw_object.update(
        {
            "sha256": raw_sha256,
            "size": len(raw_bytes),
            "objectPath": f"objects/sha256/{raw_sha256[:2]}/{raw_sha256}",
        }
    )
    game_id = str(games[0]["canonicalSha256"])
    for row in positions:
        if row["canonicalSha256"] == game_id:
            row["rawSha256"] = raw_sha256
    exports = _exports(games, positions)
    exports[0] = copy.deepcopy(exports[1])
    exports[0]["inputFile"] = "game-000001.csa"
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    with pytest.raises(ContractError, match="differs from the durable Phase 3 game"):
        validate_phase3_artifacts(
            repository_root=tmp_path,
            manifest=manifest,
            dataset_manifest_ref=manifest_ref,
            positions_ref=positions_ref,
            rust_exports=exports,
        )


def test_phase3_move_successor_requires_receipt_bound_rust_replay(
    tmp_path: Path,
) -> None:
    manifest, games, positions, _, _ = _fixture(tmp_path)
    exports = _exports(games, positions)
    game_id = str(games[0]["canonicalSha256"])
    game_positions = [row for row in positions if row["canonicalSha256"] == game_id]
    game_positions.sort(key=lambda row: int(row["positionIndex"]))
    wrong = str(exports[1]["positionSfens"][1])  # type: ignore[index]
    game_positions[0]["nextSfen"] = wrong
    game_positions[1]["sfen"] = wrong
    game_positions[1]["sideToMove"] = "white" if wrong.split()[1] == "w" else "black"
    manifest_ref, positions_ref = _rewrite_chain(tmp_path, manifest, games, positions)

    with pytest.raises(ContractError, match="receipt-bound Rust replay"):
        validate_phase3_artifacts(
            repository_root=tmp_path,
            manifest=manifest,
            dataset_manifest_ref=manifest_ref,
            positions_ref=positions_ref,
            rust_exports=exports,
        )
