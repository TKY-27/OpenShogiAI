"""Clean-HEAD build receipts for the repository's native Rust engine CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import secrets
import stat
import subprocess
import tempfile
import tomllib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    assert_retirement_capacity,
    publish_regular_at,
    retire_bound_directory,
    retire_bound_regular,
    stable_directory_lock,
    stable_parent_descriptor,
    write_json_atomic,
)

from .common import (
    ArtifactRef,
    ContractError,
    _git_environment,
    artifact_ref,
    canonical_sha256,
    contained_path,
    ensure_contained_directory,
    load_json_artifact,
    replace_json_state,
    require_clean_head,
    require_exact_keys,
    require_int,
    require_list,
    require_mapping,
    require_sha256,
    require_string,
    validate_utc_timestamp,
    verified_artifact_descriptor,
)

ENGINE_BUILD_RECEIPT_SCHEMA: Final = "open_shogi_engine_build_receipt/v2"
LEGACY_ENGINE_BUILD_RECEIPT_SCHEMA: Final = "open_shogi_engine_build_receipt/v1"
ENGINE_BUILD_POINTER_SCHEMA: Final = "open_shogi_engine_build_pointer/v1"
DEFAULT_ENGINE_PATH: Final = "target/release/open-shogi-cli"
LEGACY_RECEIPT_PATH: Final = "local/build-receipts/open-shogi-cli.json"
DEFAULT_POINTER_PATH: Final = "local/build-receipts/open-shogi-cli.pointer.json"
# Compatibility import for callers being migrated to ``resolve_active_engine_build``.
DEFAULT_RECEIPT_PATH: Final = DEFAULT_POINTER_PATH
IMMUTABLE_ENGINE_ROOT: Final = "local/builds/open-shogi-cli"
IMMUTABLE_RECEIPT_ROOT: Final = "local/build-receipts/open-shogi-cli"
_MAX_SOURCE_FILES: Final = 4_096
_MAX_SOURCE_BYTES: Final = 256 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES: Final = 16 * 1024 * 1024
_MAX_ENGINE_BYTES: Final = 512 * 1024 * 1024
_RUST_OBJCOPY_RELATIVE: Final = "lib/rustlib/aarch64-apple-darwin/bin/rust-objcopy"
_MACHO_MAGICS: Final = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
}
_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "gitCommit",
        "binary",
        "sourceTreeSha256",
        "sourceFiles",
        "sourceBytes",
        "buildCommand",
        "cargoTool",
        "rustcTool",
        "rustcRuntimeTree",
        "receiptSha256",
    }
)
_TOOL_KEYS: Final = frozenset({"path", "sha256", "size", "version"})
_RUNTIME_TREE_KEYS: Final = frozenset({"treeSha256", "files", "bytes"})
_POINTER_KEYS: Final = frozenset({"schema", "receipt", "binary"})
_LEGACY_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "gitCommit",
        "binary",
        "sourceTreeSha256",
        "sourceFiles",
        "sourceBytes",
        "buildCommand",
        "cargoVersion",
        "rustcVersion",
        "builtAt",
        "receiptSha256",
    }
)


@dataclass(frozen=True, slots=True)
class _ResolvedToolchain:
    cargo: dict[str, object]
    cargo_path: Path
    rustc: dict[str, object]
    rustc_path: Path
    runtime: dict[str, object]
    runtime_root: Path
    runtime_files: tuple[tuple[str, str, int], ...]


def create_engine_build_receipt(
    repository_root: Path,
    *,
    git_commit: str,
    engine_path: str = DEFAULT_ENGINE_PATH,
    receipt_path: str = DEFAULT_RECEIPT_PATH,
) -> tuple[dict[str, object], ArtifactRef]:
    """Build a clean snapshot and publish immutable content-addressed evidence."""

    root = repository_root.resolve(strict=True)
    if engine_path != DEFAULT_ENGINE_PATH:
        raise ContractError("engine build alias must remain target/release/open-shogi-cli")
    if receipt_path != DEFAULT_RECEIPT_PATH:
        raise ContractError("engine build pointer path is fixed by the local evidence contract")
    storage = contained_path(root, "local/build-receipts")
    with stable_directory_lock(
        storage,
        create=True,
        exclusive=True,
        nonblocking=False,
    ) as storage_descriptor:
        assert_retirement_capacity(storage_descriptor)
        commit = require_clean_head(root, git_commit)
        entries = _git_source_entries(root, commit)
        source_digest, source_files, source_bytes = _source_tree_identity_from_entries(
            root, entries
        )
        toolchain = _resolve_toolchain(root)
        existing = _reuse_matching_active_build(
            root,
            commit=commit,
            source_digest=source_digest,
            source_files=source_files,
            source_bytes=source_bytes,
            cargo_tool=toolchain.cargo,
            rustc_tool=toolchain.rustc,
            rustc_runtime=toolchain.runtime,
        )
        if existing is not None:
            return existing

        private_root = Path(tempfile.mkdtemp(prefix=".engine-build.", dir=storage))
        private_status = private_root.lstat()
        try:
            source_root = private_root / "source"
            source_root.mkdir(mode=0o700)
            _materialize_git_sources(root, entries, source_root)
            built = _run_private_build(
                source_root,
                private_root / "target",
                private_root / "homes",
                cargo_path=toolchain.cargo_path,
                rustc_path=toolchain.rustc_path,
                expected_tools=(toolchain.cargo, toolchain.rustc),
                runtime_root=toolchain.runtime_root,
                runtime_files=toolchain.runtime_files,
                expected_runtime=toolchain.runtime,
            )
            built_identity = _native_file_identity(built)
            immutable_engine_path = f"{IMMUTABLE_ENGINE_ROOT}/{built_identity[0]}/open-shogi-cli"
            _publish_built_engine(
                built,
                contained_path(root, immutable_engine_path),
                expected=built_identity,
                replace=False,
            )
            # Keep the historical path as an explicitly mutable convenience alias.
            # It is never returned or embedded in durable evidence.
            _publish_built_engine(
                built,
                contained_path(root, DEFAULT_ENGINE_PATH),
                expected=built_identity,
                replace=True,
            )
            require_clean_head(root, commit)
            engine = artifact_ref(root, immutable_engine_path, maximum_bytes=_MAX_ENGINE_BYTES)
            _validate_native_engine(root, engine)
            if (engine.sha256, engine.size) != built_identity:
                raise ContractError("published engine differs from the private build output")
        finally:
            _remove_private_build_tree(private_root, private_status)

        payload: dict[str, object] = {
            "schema": ENGINE_BUILD_RECEIPT_SCHEMA,
            "gitCommit": commit,
            "binary": engine.as_dict(),
            "sourceTreeSha256": source_digest,
            "sourceFiles": source_files,
            "sourceBytes": source_bytes,
            "buildCommand": [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--offline",
                "--jobs",
                "4",
                "-p",
                "open-shogi-cli",
            ],
            "cargoTool": toolchain.cargo,
            "rustcTool": toolchain.rustc,
            "rustcRuntimeTree": toolchain.runtime,
        }
        payload["receiptSha256"] = canonical_sha256(payload)
        immutable_receipt_path = f"{IMMUTABLE_RECEIPT_ROOT}/{payload['receiptSha256']}.json"
        destination = contained_path(root, immutable_receipt_path)
        try:
            write_json_atomic(destination, payload, replace=False)
        except FileExistsError:
            existing_reference = artifact_ref(root, immutable_receipt_path)
            if load_json_artifact(root, existing_reference) != payload:
                raise ContractError(
                    "immutable engine receipt path contains different bytes"
                ) from None
        reference = artifact_ref(root, immutable_receipt_path)
        validated = validate_engine_build_receipt(
            root,
            reference,
            expected_engine=engine,
            expected_git_commit=commit,
        )
        replace_json_state(
            contained_path(root, DEFAULT_RECEIPT_PATH),
            {
                "schema": ENGINE_BUILD_POINTER_SCHEMA,
                "receipt": reference.as_dict(),
                "binary": engine.as_dict(),
            },
        )
        return dict(validated), reference


def resolve_active_engine_build(
    repository_root: Path,
    *,
    expected_git_commit: str | None = None,
) -> tuple[ArtifactRef, ArtifactRef, dict[str, Any]]:
    """Resolve the mutable locator into immutable binary and receipt references.

    Callers persist only the returned immutable references. The pointer is local
    operator state and must never be embedded in a plan, manifest, or registry.
    """

    root = repository_root.resolve(strict=True)
    engine_ref, receipt_ref, _ = _resolve_engine_build_pointer_document(root)
    receipt = validate_engine_build_receipt(
        root,
        receipt_ref,
        expected_engine=engine_ref,
        expected_git_commit=expected_git_commit,
    )
    return engine_ref, receipt_ref, receipt


def _resolve_engine_build_pointer_document(
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef, dict[str, Any]]:
    """Validate a pointer and its immutable bytes without authorizing an old HEAD."""

    pointer_ref = artifact_ref(root, DEFAULT_RECEIPT_PATH)
    pointer = dict(require_mapping(load_json_artifact(root, pointer_ref), "engine build pointer"))
    require_exact_keys(pointer, _POINTER_KEYS, "engine build pointer")
    if pointer.get("schema") != ENGINE_BUILD_POINTER_SCHEMA:
        raise ContractError("unsupported engine build pointer schema")
    receipt_ref = ArtifactRef.from_dict(pointer.get("receipt"), "engine build pointer.receipt")
    engine_ref = ArtifactRef.from_dict(pointer.get("binary"), "engine build pointer.binary")
    receipt = validate_engine_build_receipt_document(
        load_json_artifact(root, receipt_ref),
        expected_engine=engine_ref,
    )
    _validate_native_engine(root, engine_ref)
    return engine_ref, receipt_ref, receipt


def _reuse_matching_active_build(
    root: Path,
    *,
    commit: str,
    source_digest: str,
    source_files: int,
    source_bytes: int,
    cargo_tool: dict[str, object],
    rustc_tool: dict[str, object],
    rustc_runtime: dict[str, object],
) -> tuple[dict[str, object], ArtifactRef] | None:
    pointer_path = contained_path(root, DEFAULT_RECEIPT_PATH)
    try:
        pointer_status = pointer_path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(pointer_status.st_mode) or not stat.S_ISREG(pointer_status.st_mode):
        raise ContractError("engine build pointer must be a non-symlink regular file")
    # A valid pointer may intentionally describe the previous clean commit.
    # Validate the complete immutable chain first, then treat that historical
    # build as a cache miss instead of allowing a stale locator to block a new
    # content-addressed build. Corrupt pointers still fail closed here.
    engine, reference, receipt = _resolve_engine_build_pointer_document(root)
    if receipt["gitCommit"] != commit:
        return None
    engine, reference, receipt = resolve_active_engine_build(
        root,
        expected_git_commit=commit,
    )
    expected = (
        source_digest,
        source_files,
        source_bytes,
        cargo_tool,
        rustc_tool,
        rustc_runtime,
    )
    observed = (
        receipt["sourceTreeSha256"],
        receipt["sourceFiles"],
        receipt["sourceBytes"],
        receipt["cargoTool"],
        receipt["rustcTool"],
        receipt["rustcRuntimeTree"],
    )
    if observed != expected:
        return None
    _validate_native_engine(root, engine)
    return dict(receipt), reference


def validate_engine_build_receipt(
    repository_root: Path,
    receipt_ref: ArtifactRef,
    *,
    expected_engine: ArtifactRef | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Recompute clean source identity and verify exact native binary bytes."""

    root = repository_root.resolve(strict=True)
    raw = load_json_artifact(root, receipt_ref)
    receipt = validate_engine_build_receipt_document(
        raw,
        expected_engine=expected_engine,
        expected_git_commit=expected_git_commit,
    )
    if receipt.get("schema") != ENGINE_BUILD_RECEIPT_SCHEMA:
        raise ContractError("runtime engine authorization requires an immutable v2 receipt")
    if receipt_ref.path != (f"{IMMUTABLE_RECEIPT_ROOT}/{receipt['receiptSha256']}.json"):
        raise ContractError("engine receipt path is not its content-addressed identity")
    commit = require_clean_head(root, str(receipt["gitCommit"]))
    engine = ArtifactRef.from_dict(receipt.get("binary"), "engine build receipt.binary")
    _validate_native_engine(root, engine)
    digest, files, size = _source_tree_identity(root, commit)
    if receipt["sourceTreeSha256"] != digest:
        raise ContractError("engine build receipt source-tree digest changed")
    if receipt["sourceFiles"] != files or receipt["sourceBytes"] != size:
        raise ContractError("engine build receipt source-tree counts changed")
    return receipt


def validate_engine_build_receipt_document(
    raw: object,
    *,
    expected_engine: ArtifactRef | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Closed-parse durable receipt evidence without authorizing execution.

    Historical registries must remain inspectable after HEAD advances. Runtime
    authorization uses :func:`validate_engine_build_receipt`, which additionally
    recomputes the clean Git tree and verifies the current native executable.
    """

    receipt = dict(require_mapping(raw, "engine build receipt"))
    schema = receipt.get("schema")
    if schema == ENGINE_BUILD_RECEIPT_SCHEMA:
        require_exact_keys(receipt, _RECEIPT_KEYS, "engine build receipt")
    elif schema == LEGACY_ENGINE_BUILD_RECEIPT_SCHEMA:
        require_exact_keys(receipt, _LEGACY_RECEIPT_KEYS, "engine build receipt")
    else:
        raise ContractError("unsupported engine build receipt schema")
    commit = require_string(receipt, "gitCommit", "engine build receipt", maximum_length=64)
    if (
        len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or (
            expected_git_commit is not None
            and commit != expected_git_commit
            and not (
                7 <= len(expected_git_commit) < len(commit)
                and commit.startswith(expected_git_commit)
            )
        )
    ):
        raise ContractError("engine build receipt belongs to a different Git HEAD")
    engine = ArtifactRef.from_dict(receipt.get("binary"), "engine build receipt.binary")
    if schema == ENGINE_BUILD_RECEIPT_SCHEMA and engine.path != (
        f"{IMMUTABLE_ENGINE_ROOT}/{engine.sha256}/open-shogi-cli"
    ):
        raise ContractError("engine build receipt binary path is not content-addressed")
    if schema == LEGACY_ENGINE_BUILD_RECEIPT_SCHEMA and engine.path != DEFAULT_ENGINE_PATH:
        raise ContractError("legacy engine receipt binary path is invalid")
    if expected_engine is not None and engine != expected_engine:
        raise ContractError("engine build receipt references different binary bytes")
    require_sha256(receipt, "sourceTreeSha256", "engine build receipt")
    require_int(
        receipt,
        "sourceFiles",
        "engine build receipt",
        minimum=1,
        maximum=_MAX_SOURCE_FILES,
    )
    require_int(
        receipt,
        "sourceBytes",
        "engine build receipt",
        minimum=1,
        maximum=_MAX_SOURCE_BYTES,
    )
    command = require_list(
        receipt,
        "buildCommand",
        "engine build receipt",
        minimum_items=6,
        maximum_items=16,
    )
    expected_command = ["cargo", "build", "--locked", "--release"]
    if schema == ENGINE_BUILD_RECEIPT_SCHEMA:
        expected_command.extend(["--offline", "--jobs", "4"])
    expected_command.extend(["-p", "open-shogi-cli"])
    if command != expected_command:
        raise ContractError("engine build receipt command is not the pinned release build")
    if schema == ENGINE_BUILD_RECEIPT_SCHEMA:
        _validate_tool_identity(receipt.get("cargoTool"), "engine build receipt.cargoTool")
        _validate_tool_identity(receipt.get("rustcTool"), "engine build receipt.rustcTool")
        _validate_runtime_tree_identity(
            receipt.get("rustcRuntimeTree"),
            "engine build receipt.rustcRuntimeTree",
        )
    else:
        require_string(receipt, "cargoVersion", "engine build receipt", maximum_length=256)
        require_string(receipt, "rustcVersion", "engine build receipt", maximum_length=256)
        validate_utc_timestamp(receipt.get("builtAt"), "engine build receipt.builtAt")
    expected_hash = require_sha256(receipt, "receiptSha256", "engine build receipt")
    unsigned = dict(receipt)
    del unsigned["receiptSha256"]
    if canonical_sha256(unsigned) != expected_hash:
        raise ContractError("engine build receipt self-hash mismatch")
    return receipt


def _validate_native_engine(root: Path, reference: ArtifactRef) -> None:
    expected_path = f"{IMMUTABLE_ENGINE_ROOT}/{reference.sha256}/open-shogi-cli"
    if reference.path != expected_path:
        raise ContractError("engine build receipt binary path is not content-addressed")
    with verified_artifact_descriptor(
        root,
        reference,
        maximum_bytes=512 * 1024 * 1024,
    ) as descriptor:
        status = os.fstat(descriptor)
        _validate_native_status(descriptor, status)


def _validate_tool_identity(value: object, context: str) -> dict[str, Any]:
    tool = dict(require_mapping(value, context))
    require_exact_keys(tool, _TOOL_KEYS, context)
    path = require_string(tool, "path", context, maximum_length=4_096)
    if not Path(path).is_absolute() or any(character in path for character in "\x00\r\n"):
        raise ContractError(f"{context}.path must be absolute")
    require_sha256(tool, "sha256", context)
    require_int(tool, "size", context, minimum=1, maximum=512 * 1024 * 1024)
    require_string(tool, "version", context, maximum_length=256)
    return tool


def _validate_runtime_tree_identity(value: object, context: str) -> dict[str, Any]:
    runtime = dict(require_mapping(value, context))
    require_exact_keys(runtime, _RUNTIME_TREE_KEYS, context)
    require_sha256(runtime, "treeSha256", context)
    require_int(runtime, "files", context, minimum=2, maximum=512)
    require_int(runtime, "bytes", context, minimum=1, maximum=1024 * 1024 * 1024)
    return runtime


def _source_tree_identity(root: Path, commit: str) -> tuple[str, int, int]:
    return _source_tree_identity_from_entries(root, _git_source_entries(root, commit))


def _git_source_entries(root: Path, commit: str) -> tuple[tuple[str, str, int], ...]:
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "ls-tree",
            "-r",
            "-z",
            commit,
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "engine",
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
        env=_git_environment(),
    )
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        raise ContractError("cannot enumerate engine build inputs from Git")
    names = completed.stdout.split(b"\0")
    if names and names[-1] == b"":
        names.pop()
    if not 1 <= len(names) <= _MAX_SOURCE_FILES:
        raise ContractError("engine build-input file count is outside its bound")
    entries: list[tuple[str, str, int]] = []
    for record in names:
        try:
            metadata, encoded_name = record.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            relative = encoded_name.decode("utf-8", errors="strict")
            object_id_text = object_id.decode("ascii", errors="strict")
        except (ValueError, UnicodeDecodeError) as error:
            raise ContractError("engine Git tree contains a malformed entry") from error
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise ContractError("engine build inputs must be regular Git blobs")
        if len(object_id_text) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in object_id_text
        ):
            raise ContractError("engine build input has an invalid Git object ID")
        normalized = Path(relative)
        if (
            normalized.is_absolute()
            or "\\" in relative
            or relative != normalized.as_posix()
            or any(part in {"", ".", ".."} for part in normalized.parts)
            or not (
                relative in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"}
                or relative.startswith("engine/")
            )
        ):
            raise ContractError("engine build input path is outside the approved tree")
        entries.append((relative, object_id_text, int(mode, 8)))
    return tuple(entries)


def _source_tree_identity_from_entries(
    root: Path, entries: tuple[tuple[str, str, int], ...]
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = 0
    for relative, object_id, mode in entries:
        data = _git_blob(root, object_id)
        total += len(data)
        if total > _MAX_SOURCE_BYTES:
            raise ContractError("engine build inputs exceed their aggregate byte bound")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(mode.to_bytes(4, "big"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest(), len(entries), total


def _git_blob(root: Path, object_id: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "cat-file", "blob", object_id],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
        env=_git_environment(),
    )
    if completed.returncode != 0 or len(completed.stdout) > _MAX_SOURCE_FILE_BYTES:
        raise ContractError("cannot read a bounded engine build-input Git blob")
    return completed.stdout


def _materialize_git_sources(
    root: Path,
    entries: tuple[tuple[str, str, int], ...],
    destination: Path,
) -> None:
    for relative, object_id, mode in entries:
        data = _git_blob(root, object_id)
        target = destination.joinpath(*Path(relative).parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode & 0o777,
        )
        try:
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise ContractError("engine build-input snapshot write was incomplete")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _seed_locked_cargo_cache(source_root: Path, cargo_home: Path) -> None:
    """Copy only Cargo.lock-checksummed crates, then force the build offline."""

    try:
        lock = tomllib.loads((source_root / "Cargo.lock").read_text(encoding="utf-8"))
        packages = lock["package"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ContractError("cannot parse the private Cargo.lock dependency set") from error
    if not isinstance(packages, list) or len(packages) > 512:
        raise ContractError("Cargo.lock package count is outside its bound")
    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    namespace = "index.crates.io-1949cf8c6b5b557f"
    source_cache = account_home / ".cargo/registry/cache" / namespace
    source_index = account_home / ".cargo/registry/index" / namespace
    try:
        cache_status = source_cache.lstat()
    except OSError as error:
        raise ContractError("the bounded Cargo crate cache is unavailable") from error
    if stat.S_ISLNK(cache_status.st_mode) or not stat.S_ISDIR(cache_status.st_mode):
        raise ContractError("the bounded Cargo crate cache must be a real directory")
    try:
        index_status = source_index.lstat()
    except OSError as error:
        raise ContractError("the bounded Cargo sparse index is unavailable") from error
    if stat.S_ISLNK(index_status.st_mode) or not stat.S_ISDIR(index_status.st_mode):
        raise ContractError("the bounded Cargo sparse index must be a real directory")
    destination_cache = cargo_home / "registry/cache" / namespace
    destination_cache.mkdir(mode=0o700, parents=True)
    destination_index = cargo_home / "registry/index" / namespace
    (destination_index / ".cache").mkdir(mode=0o700, parents=True)
    _write_private_cache_file(
        destination_index / "config.json",
        b'{"dl":"https://static.crates.io/crates","api":"https://crates.io"}\n',
    )
    observed: set[str] = set()
    index_cache: dict[str, bytes] = {}
    total = 0
    index_total = 0
    for raw in packages:
        if not isinstance(raw, dict):
            raise ContractError("Cargo.lock contains a malformed package")
        source = raw.get("source")
        if source is None:
            continue
        if source != "registry+https://github.com/rust-lang/crates.io-index":
            raise ContractError("Cargo.lock contains a non-crates.io external dependency")
        name = raw.get("name")
        version = raw.get("version")
        checksum = raw.get("checksum")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) is None
            or not isinstance(version, str)
            or re.fullmatch(r"[0-9A-Za-z.+-]{1,128}", version) is None
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise ContractError("Cargo.lock contains an invalid registry package identity")
        archive_name = f"{name}-{version}.crate"
        if archive_name in observed:
            raise ContractError("Cargo.lock repeats a registry package archive")
        observed.add(archive_name)
        index_bytes = index_cache.get(name)
        publish_index = index_bytes is None
        if index_bytes is None:
            index_relative = _sparse_index_relative(name)
            index_bytes = _read_bounded_cache_file(
                source_index,
                Path(".cache") / index_relative,
                maximum_bytes=16 * 1024 * 1024,
                context="Cargo sparse index entry",
            )
            index_cache[name] = index_bytes
            index_total += len(index_bytes)
            if index_total > 128 * 1024 * 1024:
                raise ContractError("Cargo sparse index entries exceed their aggregate bound")
        _validate_sparse_index_entry(
            index_bytes,
            name=name,
            version=version,
            checksum=checksum,
        )
        if publish_index:
            _write_private_cache_file(
                destination_index / ".cache" / index_relative,
                index_bytes,
            )
        size = _copy_locked_archive(
            source_cache / archive_name,
            destination_cache / archive_name,
            expected_sha256=checksum,
        )
        total += size
        if total > 512 * 1024 * 1024:
            raise ContractError("Cargo.lock crate archives exceed their aggregate bound")


def _sparse_index_relative(name: str) -> Path:
    normalized = name.casefold()
    if len(normalized) == 1:
        return Path("1") / normalized
    if len(normalized) == 2:
        return Path("2") / normalized
    if len(normalized) == 3:
        return Path("3") / normalized[0] / normalized
    return Path(normalized[:2]) / normalized[2:4] / normalized


def _read_bounded_cache_file(
    root: Path,
    relative: Path,
    *,
    maximum_bytes: int,
    context: str,
) -> bytes:
    parent_descriptor = -1
    descriptor = -1
    try:
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ContractError(f"{context} path is invalid")
        parent_descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = child
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_uid != os.getuid()
            or initial.st_nlink != 1
            or not 1 <= initial.st_size <= maximum_bytes
        ):
            raise ContractError(f"{context} violates its ownership/size contract")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - size)):
            size += len(chunk)
            if size > maximum_bytes:
                raise ContractError(f"{context} exceeds its byte bound")
            chunks.append(chunk)
        final = os.fstat(descriptor)
        linked = os.stat(
            relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _file_identity(initial) != _file_identity(final) or _file_identity(
            linked
        ) != _file_identity(final):
            raise ContractError(f"{context} changed while copied")
        return b"".join(chunks)
    except FileNotFoundError as error:
        raise ContractError(f"{context} is absent from the bounded local cache") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _write_private_cache_file(path: Path, contents: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(contents):
            written = os.write(descriptor, contents[offset:])
            if written <= 0:
                raise ContractError("private Cargo cache write was incomplete")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_sparse_index_entry(
    raw: bytes,
    *,
    name: str,
    version: str,
    checksum: str,
) -> None:
    parts = raw.split(b"\0")
    if (
        len(parts) < 7
        or parts[-1] != b""
        or parts[0] != b"\x03\x02"
        or parts[1:3] != [b"", b""]
        or not parts[3].startswith(b"etag: ")
    ):
        raise ContractError("Cargo sparse index entry has an unsupported cache format")
    matches = 0
    for index in range(4, len(parts) - 1, 2):
        if index + 1 >= len(parts) - 1:
            raise ContractError("Cargo sparse index entry has an incomplete version record")
        try:
            recorded_version = parts[index].decode("ascii", errors="strict")
            entry = json.loads(parts[index + 1], object_pairs_hook=_unique_index_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("Cargo sparse index entry is malformed") from error
        if recorded_version != version:
            continue
        if (
            not isinstance(entry, dict)
            or entry.get("name") != name
            or entry.get("vers") != version
            or entry.get("cksum") != checksum
        ):
            raise ContractError("Cargo sparse index entry differs from Cargo.lock")
        matches += 1
    if matches != 1:
        raise ContractError("Cargo.lock package is absent/duplicated in the sparse index")


def _unique_index_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Cargo sparse index entry contains a duplicate JSON key")
        result[key] = value
    return result


def _copy_locked_archive(source: Path, destination: Path, *, expected_sha256: str) -> int:
    source_descriptor = -1
    destination_descriptor = -1
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        status = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.getuid()
            or status.st_nlink != 1
            or not 1 <= status.st_size <= 64 * 1024 * 1024
        ):
            raise ContractError("cached crate archive violates its ownership/size contract")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(source_descriptor, 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_descriptor, chunk[offset:])
                if written <= 0:
                    raise ContractError("cached crate archive copy was incomplete")
                offset += written
        if size != status.st_size or digest.hexdigest() != expected_sha256:
            raise ContractError("cached crate archive differs from Cargo.lock checksum")
        os.fsync(destination_descriptor)
        return size
    except FileNotFoundError as error:
        raise ContractError(
            "a Cargo.lock dependency is absent from the bounded local crate cache"
        ) from error
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _run_private_build(
    source_root: Path,
    target_root: Path,
    homes_root: Path,
    *,
    cargo_path: Path,
    rustc_path: Path,
    expected_tools: tuple[dict[str, object], dict[str, object]],
    runtime_root: Path,
    runtime_files: tuple[tuple[str, str, int], ...],
    expected_runtime: dict[str, object],
) -> Path:
    homes_root.mkdir(mode=0o700)
    private_home = homes_root / "home"
    cargo_home = homes_root / "cargo"
    rustup_home = homes_root / "rustup"
    temporary_root = homes_root / "tmp"
    for directory in (private_home, cargo_home, rustup_home, temporary_root):
        directory.mkdir(mode=0o700)
    _seed_locked_cargo_cache(source_root, cargo_home)
    target_root.mkdir(mode=0o700)
    _assert_tool_identity(cargo_path, expected_tools[0])
    _assert_tool_identity(rustc_path, expected_tools[1])
    from open_shogi_training.labeling.execution import (
        RuntimeTreeSnapshot,
        RuntimeTreeSnapshotError,
    )

    runtime_storage = homes_root / "runtime-container"
    runtime_storage.mkdir(mode=0o700)
    _validate_runtime_tree_identity(expected_runtime, "expected rustc runtime tree")
    snapshot_files = (
        *runtime_files,
        ("bin/cargo", str(expected_tools[0]["sha256"]), 512 * 1024 * 1024),
        ("bin/rustc", str(expected_tools[1]["sha256"]), 512 * 1024 * 1024),
    )
    try:
        runtime_snapshot = RuntimeTreeSnapshot.create(
            project_root=runtime_root,
            working_directory="bin",
            files=snapshot_files,
            storage_directory=runtime_storage,
            executable_files=frozenset(
                {
                    "bin/cargo",
                    "bin/rustc",
                    *(
                        {_RUST_OBJCOPY_RELATIVE}
                        if any(path == _RUST_OBJCOPY_RELATIVE for path, _, _ in runtime_files)
                        else set()
                    ),
                }
            ),
        )
    except RuntimeTreeSnapshotError as error:
        raise ContractError(f"cannot seal private rustc runtime tree: {error}") from error
    try:
        runtime_snapshot.seal()
    except RuntimeTreeSnapshotError as error:
        runtime_snapshot.close()
        raise ContractError(f"cannot seal private rustc runtime tree: {error}") from error
    private_cargo = str(runtime_snapshot.cwd / "cargo")
    private_rustc = str(runtime_snapshot.cwd / "rustc")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(private_home),
        "CARGO_HOME": str(cargo_home),
        "RUSTUP_HOME": str(rustup_home),
        "TMPDIR": str(temporary_root),
        "CARGO_NET_GIT_FETCH_WITH_CLI": "false",
        "CARGO_NET_OFFLINE": "true",
        "RUSTC": private_rustc,
        "RUSTFLAGS": "",
        "CARGO_ENCODED_RUSTFLAGS": "",
        "LC_ALL": "C",
        "LANG": "C",
        "SOURCE_DATE_EPOCH": "0",
    }
    from .execution import _run_bounded_process

    try:

        def launch_guard() -> None:
            runtime_snapshot.assert_unchanged()

        (
            _,
            stderr,
            return_code,
            timed_out,
            output_limit,
            memory_limit,
            peak_rss_bytes,
            rss_measurement,
        ) = _run_bounded_process(
            [
                private_cargo,
                "build",
                "--locked",
                "--release",
                "--offline",
                "--jobs",
                "4",
                "-p",
                "open-shogi-cli",
                "--target-dir",
                str(target_root),
            ],
            cwd=source_root,
            timeout_seconds=3_600,
            memory_limit_bytes=8 * 1024**3,
            environment=environment,
            launch_guard=launch_guard,
        )
        runtime_snapshot.assert_unchanged()
        _assert_tool_identity(cargo_path, expected_tools[0])
        _assert_tool_identity(rustc_path, expected_tools[1])
    except RuntimeTreeSnapshotError as error:
        raise ContractError(f"private Rust build tool changed: {error}") from error
    finally:
        runtime_snapshot.close()
    if return_code != 0 or timed_out or output_limit or memory_limit:
        detail = stderr[-4_096:].decode("utf-8", errors="replace")
        raise ContractError(
            "private clean-HEAD engine build failed "
            f"(returnCode={return_code}, timedOut={timed_out}, "
            f"outputLimitExceeded={output_limit}, memoryLimitExceeded={memory_limit}, "
            f"peakRssBytes={peak_rss_bytes}, rssMeasurement={rss_measurement}): {detail}"
        )
    built = target_root / "release" / "open-shogi-cli"
    _native_file_identity(built)
    return built


def _native_file_identity(path: Path) -> tuple[str, int]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ContractError(f"cannot open private engine build output: {error}") from error
    try:
        status = os.fstat(descriptor)
        _validate_native_status(descriptor, status)
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > _MAX_ENGINE_BYTES:
                raise ContractError("private engine build output exceeds its byte bound")
            digest.update(chunk)
        final = os.fstat(descriptor)
        linked = path.lstat()
        if _file_identity(status) != _file_identity(final) or _file_identity(
            linked
        ) != _file_identity(final):
            raise ContractError("private engine build output changed while hashed")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _validate_native_status(descriptor: int, status: os.stat_result) -> None:
    if (
        not stat.S_ISREG(status.st_mode)
        or not status.st_mode & 0o111
        or status.st_uid != os.getuid()
        or status.st_nlink != 1
        or not 1 <= status.st_size <= _MAX_ENGINE_BYTES
    ):
        raise ContractError("engine binary must be an owned executable single-link regular file")
    if os.pread(descriptor, 4, 0) not in _MACHO_MAGICS:
        raise ContractError("engine binary must be a native Mach-O executable")


def _publish_built_engine(
    source: Path,
    destination: Path,
    *,
    expected: tuple[str, int],
    replace: bool,
) -> None:
    try:
        with stable_parent_descriptor(destination, create=True) as (parent, name):
            temporary_name = f".{name}.{secrets.token_hex(12)}.build"
            source_descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            writer = -1
            temporary_created = False
            temporary_status: os.stat_result | None = None
            try:
                writer = os.open(
                    temporary_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o500,
                    dir_fd=parent,
                )
                temporary_created = True
                temporary_status = os.fstat(writer)
                digest = hashlib.sha256()
                size = 0
                while chunk := os.read(source_descriptor, 1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                    offset = 0
                    while offset < len(chunk):
                        count = os.write(writer, chunk[offset:])
                        if count <= 0:
                            raise ContractError("engine publication copy was incomplete")
                        offset += count
                if (digest.hexdigest(), size) != expected:
                    raise ContractError("private engine bytes changed before publication")
                os.fchmod(writer, 0o500)
                os.fsync(writer)
                temporary_status = os.fstat(writer)
                publish_regular_at(
                    parent,
                    temporary_name,
                    name,
                    temporary_status,
                    display=destination,
                    replace=replace,
                )
                temporary_created = False
                os.fsync(parent)
            except FileExistsError:
                if replace or _native_file_identity(destination) != expected:
                    raise ContractError(
                        "content-addressed engine path contains different bytes"
                    ) from None
                assert temporary_status is not None
                retire_bound_regular(
                    parent,
                    temporary_name,
                    temporary_status,
                    display=destination,
                )
                temporary_created = False
                os.fsync(parent)
            except BaseException:
                if temporary_created and temporary_status is not None:
                    retire_bound_regular(
                        parent,
                        temporary_name,
                        temporary_status,
                        display=destination,
                    )
                raise
            finally:
                os.close(source_descriptor)
                if writer >= 0:
                    os.close(writer)
    except ArtifactError as error:
        raise ContractError(f"cannot safely publish private engine build: {error}") from error


def _remove_private_build_tree(path: Path, expected: os.stat_result) -> None:
    try:
        with stable_parent_descriptor(path, create=False) as (parent, name):
            retire_bound_directory(
                parent,
                name,
                expected,
                display=path,
                dispose=True,
            )
    except ArtifactError as error:
        raise ContractError("private engine build root changed before cleanup") from error


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _resolve_toolchain(root: Path) -> _ResolvedToolchain:
    """Resolve the pinned Darwin stable toolchain without consulting ambient PATH."""

    try:
        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise ContractError("cannot resolve the current account Rust home") from error
    if os.uname().sysname != "Darwin" or os.uname().machine not in {"arm64", "aarch64"}:
        raise ContractError("engine receipt tool resolution supports Darwin arm64 only")
    toolchain_root = account_home / ".rustup/toolchains/stable-aarch64-apple-darwin"
    try:
        metadata = toolchain_root.lstat()
    except OSError as error:
        raise ContractError("pinned stable Rust toolchain is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("pinned stable Rust toolchain must be a real directory")
    cargo_path = toolchain_root / "bin/cargo"
    rustc_path = toolchain_root / "bin/rustc"
    cargo = _tool_identity(cargo_path, root=root)
    rustc_base = _hash_tool(rustc_path)
    runtime, runtime_files = _rustc_runtime_identity(toolchain_root, rustc_path)
    _assert_tool_identity(rustc_path, rustc_base)
    rustc = dict(rustc_base)
    rustc["version"] = _rustc_version_with_runtime_snapshot(
        root,
        rustc_path,
        expected_sha256=str(rustc_base["sha256"]),
        runtime_root=toolchain_root,
        runtime_files=runtime_files,
    )
    _validate_tool_identity(rustc, "rustc tool identity")
    return _ResolvedToolchain(
        cargo=cargo,
        cargo_path=cargo_path,
        rustc=rustc,
        rustc_path=rustc_path,
        runtime=runtime,
        runtime_root=toolchain_root,
        runtime_files=runtime_files,
    )


def _rustc_runtime_identity(
    toolchain_root: Path,
    rustc_path: Path,
) -> tuple[dict[str, object], tuple[tuple[str, str, int], ...]]:
    dependencies = _non_system_macho_dependencies(rustc_path)
    if len(dependencies) != 1 or not dependencies[0].startswith("@rpath/librustc_driver-"):
        raise ContractError("rustc has an unexpected non-system runtime dependency set")
    driver_relative = f"lib/{dependencies[0].removeprefix('@rpath/')}"
    driver_path = toolchain_root.joinpath(*PurePosixPath(driver_relative).parts)
    driver_dependencies = _non_system_macho_dependencies(driver_path)
    if driver_dependencies != [dependencies[0]]:
        raise ContractError("rustc driver has an unexpected non-system dependency set")
    target_libraries = toolchain_root / "lib/rustlib/aarch64-apple-darwin/lib"
    rust_objcopy = toolchain_root.joinpath(*PurePosixPath(_RUST_OBJCOPY_RELATIVE).parts)
    try:
        target_status = target_libraries.lstat()
    except OSError as error:
        raise ContractError("rustc target runtime library tree is unavailable") from error
    if stat.S_ISLNK(target_status.st_mode) or not stat.S_ISDIR(target_status.st_mode):
        raise ContractError("rustc target runtime library tree must be a real directory")
    if _non_system_macho_dependencies(rust_objcopy):
        raise ContractError("rust-objcopy has an unexpected non-system dependency set")
    paths = [driver_path, rust_objcopy]
    for candidate in sorted(target_libraries.rglob("*")):
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError("rustc runtime tree contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("rustc runtime tree contains a non-regular entry")
        paths.append(candidate)
    if not 2 <= len(paths) <= 512:
        raise ContractError("rustc runtime tree file count is outside its bound")
    digest = hashlib.sha256()
    total = 0
    files: list[tuple[str, str, int]] = []
    for path in paths:
        try:
            relative = path.relative_to(toolchain_root).as_posix()
        except ValueError as error:
            raise ContractError("rustc runtime file escaped its toolchain root") from error
        sha256, size = _hash_runtime_file(path)
        total += size
        if total > 1024 * 1024 * 1024:
            raise ContractError("rustc runtime tree exceeds its aggregate byte bound")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256))
        files.append((relative, sha256, 512 * 1024 * 1024))
    identity: dict[str, object] = {
        "treeSha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total,
    }
    _validate_runtime_tree_identity(identity, "rustc runtime tree")
    return identity, tuple(files)


def _non_system_macho_dependencies(path: Path) -> list[str]:
    completed = subprocess.run(
        ["/usr/bin/otool", "-L", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    if completed.returncode != 0 or not 1 <= len(completed.stdout) <= 64 * 1024:
        raise ContractError("cannot inspect a bounded Rust Mach-O dependency set")
    try:
        lines = completed.stdout.decode("utf-8", errors="strict").splitlines()[1:]
    except UnicodeDecodeError as error:
        raise ContractError("Rust Mach-O dependency output is not UTF-8") from error
    non_system: list[str] = []
    for line in lines:
        dependency = line.strip().split(" (", 1)[0]
        if not dependency:
            raise ContractError("Rust Mach-O dependency output is malformed")
        if dependency.startswith(("/usr/lib/", "/System/Library/")):
            continue
        if not dependency.startswith("@rpath/") or any(
            character in dependency for character in "\x00\r\n"
        ):
            raise ContractError("Rust tool has an unapproved non-system dependency")
        non_system.append(dependency)
    return non_system


def _hash_runtime_file(path: Path) -> tuple[str, int]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ContractError(f"cannot open rustc runtime file: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_uid != os.getuid()
            or initial.st_nlink != 1
            or not 1 <= initial.st_size <= 512 * 1024 * 1024
        ):
            raise ContractError("rustc runtime file violates its ownership/size contract")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        linked = path.lstat()
        if _file_identity(initial) != _file_identity(final) or _file_identity(
            linked
        ) != _file_identity(final):
            raise ContractError("rustc runtime file changed while hashed")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _rustc_version_with_runtime_snapshot(
    root: Path,
    rustc_path: Path,
    *,
    expected_sha256: str,
    runtime_root: Path,
    runtime_files: tuple[tuple[str, str, int], ...],
) -> str:
    """Run rustc only from a private tree containing its hashed driver/runtime."""

    from open_shogi_training.labeling.execution import (
        RuntimeTreeSnapshot,
        RuntimeTreeSnapshotError,
    )

    storage = ensure_contained_directory(root, "local/build-tool-snapshots")
    snapshot_files = (
        *runtime_files,
        ("bin/rustc", expected_sha256, 512 * 1024 * 1024),
    )
    try:
        runtime = RuntimeTreeSnapshot.create(
            project_root=runtime_root,
            working_directory="bin",
            files=snapshot_files,
            storage_directory=storage,
            executable_files=frozenset(
                {
                    "bin/rustc",
                    *(
                        {_RUST_OBJCOPY_RELATIVE}
                        if any(path == _RUST_OBJCOPY_RELATIVE for path, _, _ in runtime_files)
                        else set()
                    ),
                }
            ),
        )
    except RuntimeTreeSnapshotError as error:
        raise ContractError(f"cannot snapshot rustc version runtime: {error}") from error
    try:
        runtime.seal()

        def launch_guard() -> None:
            runtime.assert_unchanged()
            if _hash_tool(rustc_path)["sha256"] != expected_sha256:
                raise ContractError("original rustc changed during its private version launch")
            _assert_runtime_files_unchanged(runtime_root, runtime_files)

        launch_guard()
        process = subprocess.Popen(
            [str(runtime.cwd / "rustc"), "--version"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        )
        try:
            launch_guard()
        except BaseException:
            with suppress(ProcessLookupError):
                process.kill()
            process.communicate(timeout=2)
            raise
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)
            raise ContractError("private rustc version command timed out") from None
        launch_guard()
        if process.returncode != 0 or not 1 <= len(stdout) <= 256:
            detail = stderr[-1_024:].decode("utf-8", errors="replace")
            raise ContractError(f"private rustc version command failed: {detail}")
        try:
            return stdout.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as error:
            raise ContractError("private rustc version is not ASCII") from error
    except RuntimeTreeSnapshotError as error:
        raise ContractError(f"private rustc version identity changed: {error}") from error
    finally:
        runtime.close()


def _assert_runtime_files_unchanged(
    runtime_root: Path,
    runtime_files: tuple[tuple[str, str, int], ...],
) -> None:
    for relative, expected_sha256, _ in runtime_files:
        observed_sha256, _ = _hash_runtime_file(
            runtime_root.joinpath(*PurePosixPath(relative).parts)
        )
        if observed_sha256 != expected_sha256:
            raise ContractError(f"original rustc runtime changed during private launch: {relative}")


def _tool_identity(path: Path, *, root: Path) -> dict[str, object]:
    identity = _hash_tool(path)
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}
    output = _run_exact_tool(
        root,
        path,
        expected_sha256=str(identity["sha256"]),
        arguments=["--version"],
        environment=environment,
        maximum_output=256,
    )
    try:
        version = output.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise ContractError("build tool version is not ASCII") from error
    identity["version"] = version
    _validate_tool_identity(identity, "build tool identity")
    return identity


def _assert_tool_identity(path: Path, expected: dict[str, object]) -> None:
    observed = _hash_tool(path)
    if any(observed[key] != expected[key] for key in ("path", "sha256", "size")):
        raise ContractError(f"exact build tool changed before or during build: {path}")


def _hash_tool(path: Path) -> dict[str, object]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ContractError(f"cannot open exact build tool {path}: {error}") from error
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or not status.st_mode & 0o111
            or status.st_uid != os.getuid()
            or status.st_nlink != 1
            or not 1 <= status.st_size <= 512 * 1024 * 1024
        ):
            raise ContractError("build tool must be an owned executable single-link file")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        linked = path.lstat()
        if _file_identity(status) != _file_identity(final) or _file_identity(
            linked
        ) != _file_identity(final):
            raise ContractError("exact build tool changed while hashed")
    finally:
        os.close(descriptor)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": size}


def _run_exact_tool(
    root: Path,
    path: Path,
    *,
    expected_sha256: str,
    arguments: list[str],
    environment: dict[str, str],
    maximum_output: int,
) -> bytes:
    """Launch a sealed private executable with pre/post-spawn identity barriers."""

    from open_shogi_training.labeling.execution import (
        ExecutableSnapshot,
        ExecutableSnapshotError,
    )

    storage = ensure_contained_directory(root, "local/build-tool-snapshots")
    try:
        snapshot = ExecutableSnapshot.create(
            path,
            temporary_directory=storage,
            max_bytes=512 * 1024 * 1024,
            expected_sha256=expected_sha256,
        )
    except ExecutableSnapshotError as error:
        raise ContractError(f"cannot seal exact build tool: {error}") from error
    try:

        def launch_guard() -> None:
            snapshot.assert_snapshot_unchanged()

        launch_guard()
        process = subprocess.Popen(
            [snapshot.executable_path, *arguments],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=environment,
        )
        try:
            launch_guard()
        except BaseException:
            process.kill()
            process.communicate(timeout=2)
            raise
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)
            raise ContractError("exact build tool command timed out") from None
        launch_guard()
        snapshot.assert_snapshot_unchanged()
        snapshot.assert_source_unchanged()
        if process.returncode != 0 or not 1 <= len(stdout) <= maximum_output:
            detail = stderr[-1_024:].decode("utf-8", errors="replace")
            raise ContractError(f"exact build tool command failed: {detail}")
        return stdout
    except ExecutableSnapshotError as error:
        raise ContractError(f"exact build tool identity changed: {error}") from error
    finally:
        snapshot.close()


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("create", choices=["create"])
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--engine", default=DEFAULT_ENGINE_PATH)
    parser.add_argument("--output", default=DEFAULT_RECEIPT_PATH)
    options = parser.parse_args()
    _, reference = create_engine_build_receipt(
        options.repository_root,
        git_commit=options.git_commit,
        engine_path=options.engine,
        receipt_path=options.output,
    )
    print(reference.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
