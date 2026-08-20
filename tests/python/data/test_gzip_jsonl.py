import gzip
from pathlib import Path

import open_shogi_training.data.gzip_jsonl as gzip_module
import pytest
from open_shogi_training.data.gzip_jsonl import (
    iter_jsonl_gzip,
    write_json_atomic,
    write_jsonl_gzip_atomic,
)


def test_gzip_jsonl_is_reproducible_and_compact(tmp_path) -> None:
    rows = [{"z": 1, "a": "日本語"}, {"nested": {"b": 2, "a": 1}}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    first_digest = write_jsonl_gzip_atomic(first, rows)
    second_digest = write_jsonl_gzip_atomic(second, rows)

    assert first.read_bytes() == second.read_bytes()
    assert first_digest.sha256 == second_digest.sha256
    assert first.read_bytes()[4:8] == b"\0\0\0\0"
    assert gzip.decompress(first.read_bytes()) == (
        b'{"a":"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e","z":1}\n{"nested":{"a":1,"b":2}}\n'
    )
    assert list(iter_jsonl_gzip(first)) == rows


def test_atomic_writers_never_overwrite(tmp_path) -> None:
    jsonl_path = tmp_path / "rows.jsonl.gz"
    write_jsonl_gzip_atomic(jsonl_path, [{"value": 1}])
    original = jsonl_path.read_bytes()

    with pytest.raises(FileExistsError):
        write_jsonl_gzip_atomic(jsonl_path, [{"value": 2}])
    assert jsonl_path.read_bytes() == original

    json_path = tmp_path / "manifest.json"
    write_json_atomic(json_path, {"b": 2, "a": 1})
    with pytest.raises(FileExistsError):
        write_json_atomic(json_path, {"a": 3})
    assert json_path.read_bytes() == b'{"a":1,"b":2}\n'


def test_reader_enforces_cumulative_uncompressed_budget(tmp_path) -> None:
    path = tmp_path / "compressed-bomb.jsonl.gz"
    write_jsonl_gzip_atomic(path, [{"value": "x" * 1_000} for _ in range(10)])

    with pytest.raises(ValueError, match="uncompressed JSONL"):
        list(iter_jsonl_gzip(path, max_uncompressed_bytes=2_000))


def test_reader_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "duplicate-key.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"value":1,"value":2}\n', mtime=0))

    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_jsonl_gzip(path))


@pytest.mark.parametrize("gzip_output", [False, True])
def test_atomic_writers_reject_a_foreign_relink_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gzip_output: bool,
) -> None:
    suffix = "rows.jsonl.gz" if gzip_output else "manifest.json"
    target = tmp_path / suffix
    owned = tmp_path / f"owned-{suffix}"
    real_publish = gzip_module.publish_regular_at

    def publish_then_relink(*args, **kwargs) -> None:
        real_publish(*args, **kwargs)
        target.rename(owned)
        target.write_bytes(b"foreign-do-not-delete")

    monkeypatch.setattr(gzip_module, "publish_regular_at", publish_then_relink)
    with pytest.raises(ValueError, match="changed while it was published"):
        if gzip_output:
            write_jsonl_gzip_atomic(target, [{"value": 1}])
        else:
            write_json_atomic(target, {"value": 1})

    assert owned.is_file()
    assert target.read_bytes() == b"foreign-do-not-delete"
