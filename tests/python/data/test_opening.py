import hashlib
import math
import sqlite3

import open_shogi_training.data.opening as opening
import pytest
from open_shogi_training.data.gzip_jsonl import (
    iter_jsonl_gzip,
    write_jsonl_gzip_atomic,
)
from open_shogi_training.data.opening import (
    OPENING_SCHEMA,
    OpeningConfig,
    build_opening_database,
    export_opening_jsonl,
    query_opening_moves,
    state_identity,
    wilson_interval_95,
)


def _position(
    *,
    game: str,
    source: str,
    sfen: str,
    move: str,
    outcome: str,
    full_plies: int,
    remaining_plies: int,
    split: str = "train",
    eligible: bool = True,
) -> dict:
    side = "black" if sfen.split(" ")[1] == "b" else "white"
    game_sha256 = hashlib.sha256(game.encode()).hexdigest()
    return {
        "schema": "phase3_position/v1",
        "gameId": game_sha256,
        "canonicalSha256": game_sha256,
        "rawSha256": hashlib.sha256(f"raw-{game}".encode()).hexdigest(),
        "sourceId": source,
        "split": split,
        "positionIndex": full_plies - remaining_plies,
        "sfen": sfen,
        "moveUsi": move,
        "nextSfen": f"next-{game} w - {full_plies - remaining_plies + 2}",
        "outcome": outcome,
        "terminalReason": "TORYO",
        "sideToMove": side,
        "fullPlies": full_plies,
        "remainingPlies": remaining_plies,
        "eligible": eligible,
        "terminalTail": False,
    }


def test_builds_exact_opening_statistics_and_query_time_threshold(tmp_path) -> None:
    state = "state-a b - 1"
    rows = [
        _position(
            game="g1",
            source="s1",
            sfen=state,
            move="7g7f",
            outcome="black_win",
            full_plies=30,
            remaining_plies=30,
        ),
        _position(
            game="g2",
            source="s1",
            sfen="state-a b - 5",
            move="7g7f",
            outcome="black_win",
            full_plies=40,
            remaining_plies=36,
        ),
        _position(
            game="g3",
            source="s1",
            sfen="state-a b - 9",
            move="7g7f",
            outcome="white_win",
            full_plies=50,
            remaining_plies=42,
        ),
        _position(
            game="g4",
            source="s2",
            sfen="state-a b - 13",
            move="7g7f",
            outcome="draw",
            full_plies=60,
            remaining_plies=48,
        ),
        _position(
            game="g5",
            source="s2",
            sfen="state-a b - 17",
            move="7g7f",
            outcome="unknown",
            full_plies=70,
            remaining_plies=54,
        ),
        _position(
            game="g6",
            source="s1",
            sfen=state,
            move="2g2f",
            outcome="black_win",
            full_plies=20,
            remaining_plies=20,
        ),
        _position(
            game="validation",
            source="s1",
            sfen=state,
            move="7g7f",
            outcome="black_win",
            full_plies=10,
            remaining_plies=10,
            split="validation",
        ),
        _position(
            game="ineligible",
            source="s1",
            sfen=state,
            move="7g7f",
            outcome="black_win",
            full_plies=10,
            remaining_plies=10,
            eligible=False,
        ),
    ]
    positions_path = tmp_path / "positions.jsonl.gz"
    database_path = tmp_path / "opening.sqlite3"
    write_jsonl_gzip_atomic(positions_path, rows)

    report = build_opening_database(
        positions_path,
        database_path,
        provenance={"datasetManifestSha256": "abc", "sources": ["s1", "s2"]},
    )

    assert report.input_rows == 8
    assert report.included_rows == 6
    assert report.state_count == 1
    assert report.move_count == 2
    assert query_opening_moves(database_path, "state-a b - 999", min_count=6) == []

    result = query_opening_moves(database_path, "state-a b - 999", min_count=2)
    assert len(result) == 1
    stats = result[0]
    assert stats["moveUsi"] == "7g7f"
    assert stats["count"] == 5
    assert (stats["wins"], stats["losses"], stats["draws"], stats["unknown"]) == (2, 1, 1, 1)
    assert stats["scoreRate"] == 0.625
    assert stats["decisiveN"] == 3
    assert stats["decisiveWinRate"] == pytest.approx(2 / 3)
    low, high = wilson_interval_95(2, 3)
    assert stats["decisiveWinRateWilson95Low"] == pytest.approx(low)
    assert stats["decisiveWinRateWilson95High"] == pytest.approx(high)
    assert stats["blackWins"] == 2
    assert stats["whiteWins"] == 1
    assert stats["sideSpecificDecisiveN"] == 3
    assert stats["blackDecisiveWinRate"] == pytest.approx(2 / 3)
    assert stats["whiteDecisiveWinRate"] == pytest.approx(1 / 3)
    assert stats["averageFullPlies"] == 50
    assert stats["averageRemainingPlies"] == 42
    assert stats["sourceCounts"] == {"s1": 3, "s2": 2}

    # min_count never deletes observations; a later lower-threshold query sees both moves.
    assert [row["moveUsi"] for row in query_opening_moves(database_path, state)] == [
        "7g7f",
        "2g2f",
    ]
    with sqlite3.connect(database_path) as connection:
        schema = connection.execute("SELECT value FROM metadata WHERE key = 'schema'").fetchone()
        assert schema == (OPENING_SCHEMA,)


def test_state_identity_omits_only_move_number_and_export_is_deterministic(tmp_path) -> None:
    first_key, first_state, side = state_identity("same-state w 2P 1")
    second_key, second_state, _ = state_identity("same-state w 2P 900")
    assert first_key == second_key
    assert first_state == second_state == "same-state w 2P"
    assert side == "w"

    positions_path = tmp_path / "positions.jsonl.gz"
    database_path = tmp_path / "opening with ? mark.sqlite3"
    write_jsonl_gzip_atomic(
        positions_path,
        [
            _position(
                game="g1",
                source="source",
                sfen="same-state w 2P 1",
                move="3c3d",
                outcome="white_win",
                full_plies=40,
                remaining_plies=40,
            )
        ],
    )
    build_opening_database(positions_path, database_path, provenance={"manifest": "hash"})
    first_export = tmp_path / "first.jsonl.gz"
    second_export = tmp_path / "second.jsonl.gz"
    export_opening_jsonl(database_path, first_export)
    export_opening_jsonl(database_path, second_export)

    assert first_export.read_bytes() == second_export.read_bytes()
    exported = list(iter_jsonl_gzip(first_export))
    assert exported[0]["stateKey"] == first_key
    assert exported[0]["wins"] == 1


def test_build_checks_state_hash_collisions(tmp_path, monkeypatch) -> None:
    positions_path = tmp_path / "positions.jsonl.gz"
    write_jsonl_gzip_atomic(
        positions_path,
        [
            _position(
                game="g1",
                source="source",
                sfen="state-one b - 1",
                move="7g7f",
                outcome="black_win",
                full_plies=30,
                remaining_plies=30,
            ),
            _position(
                game="g2",
                source="source",
                sfen="state-two b - 1",
                move="2g2f",
                outcome="black_win",
                full_plies=30,
                remaining_plies=30,
            ),
        ],
    )
    original = opening.state_identity

    def colliding_identity(sfen: str) -> tuple[str, str, str]:
        _, state_sfen, side = original(sfen)
        return "0" * 64, state_sfen, side

    monkeypatch.setattr(opening, "state_identity", colliding_identity)
    with pytest.raises(RuntimeError, match="collision"):
        build_opening_database(
            positions_path,
            tmp_path / "opening.sqlite3",
            provenance={"manifest": "hash"},
        )


def test_build_hashes_and_parses_one_bounded_private_snapshot(tmp_path, monkeypatch) -> None:
    positions_path = tmp_path / "positions.jsonl.gz"
    row = _position(
        game="g1",
        source="source",
        sfen="state b - 1",
        move="7g7f",
        outcome="black_win",
        full_plies=30,
        remaining_plies=30,
    )
    write_jsonl_gzip_atomic(positions_path, [row])
    expected_hash = hashlib.sha256(positions_path.read_bytes()).hexdigest()
    original_iter = opening.iter_jsonl_gzip

    def replace_original_after_snapshot(path, **kwargs):
        assert path != positions_path
        positions_path.write_bytes(b"replacement that is not gzip")
        yield from original_iter(path, **kwargs)

    monkeypatch.setattr(opening, "iter_jsonl_gzip", replace_original_after_snapshot)
    database_path = tmp_path / "opening.sqlite3"
    report = build_opening_database(
        positions_path,
        database_path,
        provenance={"manifest": "hash"},
    )

    assert report.included_rows == 1
    with sqlite3.connect(database_path) as connection:
        stored_hash = connection.execute(
            "SELECT value FROM metadata WHERE key = 'input_sha256'"
        ).fetchone()
    assert stored_hash == (expected_hash,)

    oversized = tmp_path / "oversized.jsonl.gz"
    write_jsonl_gzip_atomic(oversized, [row])
    with pytest.raises(ValueError, match="compressed JSONL"):
        build_opening_database(
            oversized,
            tmp_path / "oversized.sqlite3",
            provenance={"manifest": "hash"},
            config=OpeningConfig(max_input_bytes=oversized.stat().st_size - 1),
        )
    assert not tuple(tmp_path.glob(".opening-input.*"))


def test_opening_export_streams_the_outer_query_cursor(tmp_path, monkeypatch) -> None:
    positions_path = tmp_path / "positions.jsonl.gz"
    database_path = tmp_path / "opening.sqlite3"
    write_jsonl_gzip_atomic(
        positions_path,
        [
            _position(
                game="g1",
                source="source",
                sfen="state b - 1",
                move="7g7f",
                outcome="black_win",
                full_plies=30,
                remaining_plies=30,
            )
        ],
    )
    build_opening_database(positions_path, database_path, provenance={"manifest": "hash"})
    real_connect = sqlite3.connect

    class CursorProxy:
        def __init__(self, cursor, *, forbid_fetchall: bool) -> None:
            self.cursor = cursor
            self.forbid_fetchall = forbid_fetchall

        def __iter__(self):
            return iter(self.cursor)

        def fetchall(self):
            if self.forbid_fetchall:
                raise AssertionError("outer opening export query must be streamed")
            return self.cursor.fetchall()

        def __getattr__(self, name):
            return getattr(self.cursor, name)

    class ConnectionProxy:
        def __init__(self, connection) -> None:
            object.__setattr__(self, "connection", connection)

        def __setattr__(self, name, value) -> None:
            if name == "row_factory":
                self.connection.row_factory = value
            else:
                object.__setattr__(self, name, value)

        def execute(self, sql, parameters=()):
            cursor = self.connection.execute(sql, parameters)
            return CursorProxy(cursor, forbid_fetchall="JOIN states" in sql)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def streaming_connect(*args, **kwargs):
        return ConnectionProxy(real_connect(*args, **kwargs))

    monkeypatch.setattr(opening.sqlite3, "connect", streaming_connect)
    output_path = tmp_path / "opening.jsonl.gz"
    export_opening_jsonl(database_path, output_path)

    assert len(list(iter_jsonl_gzip(output_path))) == 1


def test_wilson_empty_sample_and_invalid_counts() -> None:
    assert wilson_interval_95(0, 0) == (None, None)
    with pytest.raises(ValueError):
        wilson_interval_95(2, 1)
    assert math.isclose(wilson_interval_95(5, 5)[1], 1.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gameId", "not-a-hash", "gameId"),
        ("rawSha256", "ABC", "rawSha256"),
        ("sourceId", "../unsafe", "sourceId"),
        ("moveUsi", "resign", "moveUsi"),
        ("remainingPlies", 29, "positionIndex"),
        ("fullPlies", 2_049, "2048"),
    ],
)
def test_rejects_malformed_or_inconsistent_position_rows(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    row = _position(
        game="g1",
        source="source",
        sfen="state b - 1",
        move="7g7f",
        outcome="black_win",
        full_plies=30,
        remaining_plies=30,
    )
    row[field] = value
    positions_path = tmp_path / f"{field}.jsonl.gz"
    write_jsonl_gzip_atomic(positions_path, [row])

    with pytest.raises(ValueError, match=message):
        build_opening_database(
            positions_path,
            tmp_path / f"{field}.sqlite3",
            provenance={"manifest": "hash"},
        )
