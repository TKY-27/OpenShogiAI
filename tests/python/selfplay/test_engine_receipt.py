from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from open_shogi_training.selfplay import engine_receipt as receipt_module
from open_shogi_training.selfplay.common import (
    ContractError,
    artifact_ref,
    require_clean_head,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_AUTHOR_NAME": "OpenShogiAI test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "OpenShogiAI test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
    )
    return completed.stdout.decode("ascii").strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    (root / "engine/cli/src").mkdir(parents=True)
    (root / ".gitignore").write_text("/local/\n/target/\n", encoding="utf-8")
    (root / "Cargo.toml").write_text("[workspace]\nmembers=['engine/cli']\n", encoding="utf-8")
    (root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
    (root / "engine/cli/Cargo.toml").write_text(
        "[package]\nname='open-shogi-cli'\nversion='0.0.0'\nedition='2021'\n",
        encoding="utf-8",
    )
    (root / "engine/cli/src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _mach_o(marker: bytes) -> bytes:
    return b"\xcf\xfa\xed\xfe" + marker * 64


def test_clean_head_ignores_empty_untracked_directories_but_not_files(
    tmp_path: Path,
) -> None:
    root, commit = _repository(tmp_path)
    empty = root / "empty-untracked-directory"
    empty.mkdir()

    assert require_clean_head(root, commit) == commit

    (empty / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ContractError, match="untracked non-ignored paths"):
        require_clean_head(root, commit)


def test_receipt_builds_git_blob_snapshot_instead_of_attesting_existing_binary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, commit = _repository(tmp_path)
    destination = root / "target/release/open-shogi-cli"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(_mach_o(b"A"))
    destination.chmod(0o500)
    legacy = root / receipt_module.LEGACY_RECEIPT_PATH
    legacy.parent.mkdir(parents=True)
    legacy_bytes = b'{"schema":"open_shogi_engine_build_receipt/v1","historical":true}\n'
    legacy.write_bytes(legacy_bytes)
    built_bytes = _mach_o(b"B")

    build_calls = 0

    def fake_build(
        source: Path,
        target: Path,
        homes: Path,
        **_: object,
    ) -> Path:
        nonlocal build_calls
        build_calls += 1
        assert source.joinpath("Cargo.toml").read_text(encoding="utf-8").startswith("[workspace]")
        assert homes.parent == target.parent
        output = target / "release/open-shogi-cli"
        output.parent.mkdir(parents=True)
        output.write_bytes(built_bytes)
        output.chmod(0o500)
        return output

    monkeypatch.setattr(receipt_module, "_run_private_build", fake_build)
    cargo_tool = {
        "path": "/fixture/cargo",
        "sha256": "1" * 64,
        "size": 1,
        "version": "cargo fixture",
    }
    rustc_tool = {
        "path": "/fixture/rustc",
        "sha256": "2" * 64,
        "size": 1,
        "version": "rustc fixture",
    }
    rustc_runtime = {"treeSha256": "3" * 64, "files": 2, "bytes": 2}
    monkeypatch.setattr(
        receipt_module,
        "_resolve_toolchain",
        lambda *_: receipt_module._ResolvedToolchain(
            cargo=cargo_tool,
            cargo_path=Path("/fixture/cargo"),
            rustc=rustc_tool,
            rustc_path=Path("/fixture/rustc"),
            runtime=rustc_runtime,
            runtime_root=Path("/fixture"),
            runtime_files=(("lib/a", "4" * 64, 1), ("lib/b", "5" * 64, 1)),
        ),
    )

    receipt, receipt_ref = receipt_module.create_engine_build_receipt(
        root,
        git_commit=commit,
    )

    assert destination.read_bytes() == built_bytes
    engine = artifact_ref(root, receipt["binary"]["path"])
    assert receipt["binary"] == engine.as_dict()
    assert engine.path == (f"{receipt_module.IMMUTABLE_ENGINE_ROOT}/{engine.sha256}/open-shogi-cli")
    assert receipt_ref.path == (
        f"{receipt_module.IMMUTABLE_RECEIPT_ROOT}/{receipt['receiptSha256']}.json"
    )
    assert legacy.read_bytes() == legacy_bytes
    pointer = json.loads((root / receipt_module.DEFAULT_POINTER_PATH).read_bytes())
    assert pointer == {
        "schema": receipt_module.ENGINE_BUILD_POINTER_SCHEMA,
        "receipt": receipt_ref.as_dict(),
        "binary": engine.as_dict(),
    }
    assert (
        receipt_module.validate_engine_build_receipt(
            root,
            receipt_ref,
            expected_engine=engine,
            expected_git_commit=commit,
        )
        == receipt
    )
    assert (
        receipt_module.validate_engine_build_receipt_document(
            receipt,
            expected_engine=engine,
            expected_git_commit=commit[:12],
        )
        == receipt
    )
    assert not any((root / "local/build-receipts").glob(".engine-build.*"))

    (root / "engine/cli/src/main.rs").write_text("fn main() { panic!() }\n", encoding="utf-8")
    assert (
        receipt_module.validate_engine_build_receipt_document(
            receipt,
            expected_engine=engine,
            expected_git_commit=commit,
        )
        == receipt
    )
    with pytest.raises(ContractError, match="unstaged changes"):
        receipt_module.validate_engine_build_receipt(
            root,
            receipt_ref,
            expected_engine=engine,
            expected_git_commit=commit,
        )

    forged = dict(receipt)
    forged["rustcTool"] = {**rustc_tool, "version": "rustc forged"}
    with pytest.raises(ContractError, match="self-hash"):
        receipt_module.validate_engine_build_receipt_document(
            forged,
            expected_engine=engine,
            expected_git_commit=commit,
        )

    (root / "engine/cli/src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
    second, second_ref = receipt_module.create_engine_build_receipt(root, git_commit=commit)
    assert second == receipt
    assert second_ref == receipt_ref
    assert build_calls == 1
    assert legacy.read_bytes() == legacy_bytes

    (root / "engine/cli/src/main.rs").write_text(
        "fn main() {}\n// next clean source revision\n",
        encoding="utf-8",
    )
    _git(root, "add", "engine/cli/src/main.rs")
    _git(root, "commit", "-qm", "next fixture revision")
    next_commit = _git(root, "rev-parse", "HEAD")
    next_receipt, next_ref = receipt_module.create_engine_build_receipt(
        root,
        git_commit=next_commit,
    )
    assert next_receipt["gitCommit"] == next_commit
    assert next_ref != receipt_ref
    assert (root / receipt_ref.path).is_file()
    assert build_calls == 2

    pointer = json.loads((root / receipt_module.DEFAULT_POINTER_PATH).read_bytes())
    pointer["binary"] = {**engine.as_dict(), "sha256": "f" * 64}
    (root / receipt_module.DEFAULT_POINTER_PATH).write_text(
        json.dumps(pointer, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="different binary bytes"):
        receipt_module.resolve_active_engine_build(root, expected_git_commit=next_commit)


def test_source_identity_reads_commit_blobs_not_mutable_worktree(tmp_path: Path) -> None:
    root, commit = _repository(tmp_path)
    before = receipt_module._source_tree_identity(root, commit)
    (root / "engine/cli/src/main.rs").write_text("fn main() { panic!() }\n", encoding="utf-8")

    assert receipt_module._source_tree_identity(root, commit) == before


def test_private_build_cleanup_never_removes_a_relinked_foreign_tree(tmp_path: Path) -> None:
    build = tmp_path / ".engine-build.owned"
    build.mkdir()
    status = build.lstat()
    original = tmp_path / "original-build-tree"
    build.rename(original)
    build.mkdir()
    marker = build / "foreign-do-not-delete"
    marker.write_bytes(b"foreign")

    with pytest.raises(ContractError, match="changed before cleanup"):
        receipt_module._remove_private_build_tree(build, status)

    assert marker.read_bytes() == b"foreign"
    assert original.is_dir()


def test_exact_tool_spawn_uses_private_bytes_and_rejects_source_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "malicious-tool-ran"
    tool = tmp_path / "cargo"
    tool.write_text("#!/bin/sh\nprintf 'cargo fixture\\n'\n", encoding="utf-8")
    tool.chmod(0o700)
    expected = receipt_module._hash_tool(tool)
    original_popen = subprocess.Popen

    def replace_source_then_spawn(*args: object, **kwargs: object):
        retained = tool.with_name("cargo.original")
        tool.rename(retained)
        tool.write_text(
            f"#!/bin/sh\ntouch \"{marker}\"\nprintf 'malicious\\n'\n",
            encoding="utf-8",
        )
        tool.chmod(0o700)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(receipt_module.subprocess, "Popen", replace_source_then_spawn)

    with pytest.raises(ContractError, match="exact build tool identity changed"):
        receipt_module._run_exact_tool(
            tmp_path,
            tool,
            expected_sha256=str(expected["sha256"]),
            arguments=["--version"],
            environment={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            maximum_output=256,
        )

    assert not marker.exists()


def test_rustc_version_spawn_rejects_original_runtime_replacement_without_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = tmp_path / "toolchain"
    rustc = toolchain / "bin/rustc"
    driver = toolchain / "lib/librustc_driver-fixture.dylib"
    marker = tmp_path / "malicious-driver-read"
    driver.parent.mkdir(parents=True)
    rustc.parent.mkdir(parents=True)
    driver.write_bytes(b"trusted")
    rustc.write_text(
        "#!/bin/sh\n"
        'driver="$(dirname "$0")/../lib/librustc_driver-fixture.dylib"\n'
        f'if [ "$(cat "$driver")" = malicious ]; then touch "{marker}"; fi\n'
        "printf 'rustc fixture\\n'\n",
        encoding="utf-8",
    )
    rustc.chmod(0o700)
    rustc_sha = str(receipt_module._hash_tool(rustc)["sha256"])
    driver_sha, _ = receipt_module._hash_runtime_file(driver)
    runtime_files = (("lib/librustc_driver-fixture.dylib", driver_sha, 1024),)
    original_popen = subprocess.Popen

    def replace_runtime_then_spawn(*args: object, **kwargs: object):
        driver.write_bytes(b"malicious")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(receipt_module.subprocess, "Popen", replace_runtime_then_spawn)

    with pytest.raises(ContractError, match="original rustc runtime changed"):
        receipt_module._rustc_version_with_runtime_snapshot(
            tmp_path,
            rustc,
            expected_sha256=rustc_sha,
            runtime_root=toolchain,
            runtime_files=runtime_files,
        )

    assert not marker.exists()


def test_private_cargo_cache_rejects_sparse_index_checksum_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_home = tmp_path / "account"
    source = tmp_path / "source"
    source.mkdir()
    archive = b"fixture crate archive"
    checksum = receipt_module.hashlib.sha256(archive).hexdigest()
    name = "fixture-crate"
    version = "1.2.3"
    (source / "Cargo.lock").write_text(
        "version = 4\n\n"
        "[[package]]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{checksum}"\n',
        encoding="utf-8",
    )
    namespace = "index.crates.io-1949cf8c6b5b557f"
    archive_path = account_home / ".cargo/registry/cache" / namespace / f"{name}-{version}.crate"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive)
    index_path = (
        account_home
        / ".cargo/registry/index"
        / namespace
        / ".cache"
        / receipt_module._sparse_index_relative(name)
    )
    index_path.parent.mkdir(parents=True)

    def sparse_entry(entry_checksum: str) -> bytes:
        entry = json.dumps(
            {"name": name, "vers": version, "cksum": entry_checksum},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return b'\x03\x02\0\0\0etag: "fixture"\0' + version.encode() + b"\0" + entry + b"\0"

    monkeypatch.setattr(
        receipt_module.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(account_home)),
    )
    index_path.write_bytes(sparse_entry("f" * 64))
    cargo_home = tmp_path / "cargo-wrong"
    cargo_home.mkdir()
    with pytest.raises(ContractError, match=r"differs from Cargo\.lock"):
        receipt_module._seed_locked_cargo_cache(source, cargo_home)
    assert not any((cargo_home / "registry/cache").rglob("*.crate"))

    index_path.write_bytes(sparse_entry(checksum))
    valid_home = tmp_path / "cargo-valid"
    valid_home.mkdir()
    receipt_module._seed_locked_cargo_cache(source, valid_home)

    assert (valid_home / "registry/cache" / namespace / archive_path.name).read_bytes() == archive
    assert (
        valid_home
        / "registry/index"
        / namespace
        / ".cache"
        / receipt_module._sparse_index_relative(name)
    ).read_bytes() == sparse_entry(checksum)
