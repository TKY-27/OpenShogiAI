"""Teacher binary/evaluation identity and local install-manifest verification."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    hash_regular_file,
    load_json_object,
    stable_regular_descriptor,
)
from open_shogi_training.labeling.config import (
    APERY_V2_BINARY_SHA256,
    TeacherConfig,
    resolve_config_path,
)
from open_shogi_training.labeling.usi import USIIdentity

INSTALL_MANIFEST_SCHEMA: Final = "phase4_teacher_install/v2"
LEGACY_INSTALL_MANIFEST_SCHEMA: Final = "phase4_teacher_install/v1"
MAX_BINARY_BYTES: Final = 512 * 1024 * 1024
MAX_EVAL_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
_APERY_OFFICIAL: Final = {
    "repository": "https://github.com/HiraokaTakuya/apery_rust",
    "release_url": (
        "https://github.com/HiraokaTakuya/apery_rust/releases/download/v2.0.0/apery_2.0.0.zip"
    ),
    "tag": "v2.0.0",
    "tag_commit": "a570784542f7e50fb39a24129f02d1b14819eec1",
}
_APERY_ARCHIVE_SHA256: Final = "22c662f1a7c28f79dd51a8a2b80179fc2ef063d0dc42df5232028c9c2390cad3"
_APERY_ARCHIVE_SIZE: Final = 641_487_453
_APERY_ARCHIVE_ENTRIES: Final = 83
_APERY_ARCHIVE_UNCOMPRESSED: Final = 896_036_520
_APERY_ARCHIVE_FILES: Final = 76
_APERY_OFFICIAL_DIRECTORIES: Final = frozenset(
    {
        ".cargo",
        "eval",
        "eval/20190617",
        "eval/20190617/log",
        "src",
        "src/evaluate",
        "test",
    }
)
_APERY_ARCHIVE_TREE_SHA256: Final = (
    "9d47d5a0e0c0df94224b5b03278c0571b9e508e7a239e71394fc7f0f66278fea"
)
_APERY_SOURCE_TREE_SHA256: Final = (
    "055607a5aef06a044dccfd4ff18bff548b69a05786085134b36fec2928dc9447"
)
_APERY_SOURCE_FILES: Final = 38
_APERY_SOURCE_SIZE: Final = 634_564
_APERY_BINARY_SHA256: Final = APERY_V2_BINARY_SHA256
_APERY_BINARY_SIZE: Final = 2_364_896
_APERY_RUSTC_VERSION: Final = "rustc 1.94.1 (e408947bf 2026-03-25)"
_APERY_CARGO_VERSION: Final = "cargo 1.94.1 (29ea6fb6a 2026-03-24)"
_APERY_COMPAT_LOCK_SHA256: Final = (
    "5acb657fe557553603ec55343840689ec7fe8fc2799d009bba51ae79ddcfb737"
)
_APERY_COMPAT_LOCK_SIZE: Final = 12_773
_APERY_LICENSES: Final = {
    "LICENSE": (
        "GPL-3.0-only",
        "0b383d5a63da644f628d99c33976ea6487ed89aaa59f0b3257992deac1171e6b",
        35_821,
    ),
    "eval/20190617/LICENSE-MIT": (
        "MIT",
        "cdbd06e25b8c9c5d6019949d2de8123f56d29cee4166396423d2ba8c81700845",
        1_044,
    ),
}
_APERY_README = (
    "647386fbe09e430d58f2b25b9cb1cf87fd54cde56a8c9bd4c27f8f53358f64c9",
    296,
)


class TeacherFingerprintError(ValueError):
    """Raised when configured teacher files drift from their recorded identity."""


@dataclass(frozen=True, slots=True)
class FingerprintedFile:
    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, int | str]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class TeacherFingerprint:
    name: str
    version: str
    binary: FingerprintedFile
    eval_files: tuple[FingerprintedFile, ...]
    options: dict[str, str | int | bool]

    def teacher_record(self, identity: USIIdentity) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "reported_name": identity.name,
            "reported_author": identity.author,
            "binary_sha256": self.binary.sha256,
            "binary_size": self.binary.size,
            "eval_files": [item.as_dict() for item in self.eval_files],
            "options": dict(sorted(self.options.items())),
        }

    def identity_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "binary": self.binary.as_dict(),
            "eval_files": [item.as_dict() for item in self.eval_files],
            "options": dict(sorted(self.options.items())),
        }


def fingerprint_teacher(config: TeacherConfig, project_root: Path) -> TeacherFingerprint:
    """Hash configured files and verify expected hashes plus optional setup manifest."""

    root = project_root.resolve(strict=True)
    executable = resolve_config_path(root, config.executable, field="teacher.executable")
    cwd = resolve_config_path(root, config.cwd, field="teacher.cwd")
    eval_dir = resolve_config_path(root, config.eval_dir, field="teacher.eval_dir")
    if not eval_dir.is_dir() or eval_dir.is_symlink():
        raise TeacherFingerprintError("teacher.eval_dir must be a non-symlink directory")
    eval_option = dict(config.options).get("Eval_Dir")
    if eval_option is not None:
        if not isinstance(eval_option, str) or Path(eval_option).is_absolute():
            raise TeacherFingerprintError("Eval_Dir option must be a cwd-relative string")
        configured_eval_dir = (cwd / eval_option).resolve(strict=False)
        if configured_eval_dir != eval_dir.resolve(strict=True):
            raise TeacherFingerprintError("Eval_Dir option does not select teacher.eval_dir")
    try:
        binary_digest = hash_regular_file(executable, max_bytes=MAX_BINARY_BYTES)
    except ArtifactError as error:
        raise TeacherFingerprintError(str(error)) from error
    if binary_digest.sha256 != config.runtime_binary_sha256:
        raise TeacherFingerprintError("teacher binary SHA-256 does not match configuration")
    if (config.name, config.version) == ("Apery", "2.0.0") and (
        binary_digest.sha256 != _APERY_BINARY_SHA256 or binary_digest.size != _APERY_BINARY_SIZE
    ):
        raise TeacherFingerprintError(
            "Apery binary differs from the fail-closed runtime pin; the null YAML field "
            "exists only to preserve the completed v1 config identity"
        )
    if (config.name, config.version) == ("Apery", "2.0.0"):
        _validate_official_native_executable(executable)
    try:
        executable_status = executable.lstat()
    except OSError as error:
        raise TeacherFingerprintError("cannot inspect teacher executable mode") from error
    if not executable_status.st_mode & 0o111:
        raise TeacherFingerprintError("teacher executable has no execute bit")
    binary = FingerprintedFile(config.executable, binary_digest.sha256, binary_digest.size)

    eval_files: list[FingerprintedFile] = []
    for expected in config.eval_files:
        path = resolve_config_path(root, expected.path, field=f"eval file {expected.path}")
        try:
            path.resolve(strict=True).relative_to(eval_dir.resolve(strict=True))
        except ValueError as error:
            raise TeacherFingerprintError(
                f"configured eval file is outside teacher.eval_dir: {expected.path}"
            ) from error
        try:
            digest = hash_regular_file(path, max_bytes=MAX_EVAL_FILE_BYTES)
        except ArtifactError as error:
            raise TeacherFingerprintError(str(error)) from error
        if digest.sha256 != expected.sha256:
            raise TeacherFingerprintError(f"eval SHA-256 mismatch: {expected.path}")
        eval_files.append(FingerprintedFile(expected.path, digest.sha256, digest.size))

    fingerprint = TeacherFingerprint(
        name=config.name,
        version=config.version,
        binary=binary,
        eval_files=tuple(eval_files),
        options=config.option_map,
    )
    if config.install_manifest is not None:
        _verify_install_manifest(config, root, fingerprint)
    return fingerprint


def _validate_official_native_executable(path: Path) -> None:
    """Reject script, foreign-owner, and hard-linked teacher launch inputs."""

    macho_magics = {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }
    try:
        with stable_regular_descriptor(path) as descriptor:
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or not status.st_mode & 0o111
                or status.st_uid != os.getuid()
                or status.st_nlink != 1
            ):
                raise TeacherFingerprintError(
                    "official teacher must be an owned executable single-link regular file"
                )
            if os.pread(descriptor, 4, 0) not in macho_magics:
                raise TeacherFingerprintError("official teacher must be a native Mach-O executable")
    except ArtifactError as error:
        raise TeacherFingerprintError(str(error)) from error


def _verify_install_manifest(
    config: TeacherConfig,
    project_root: Path,
    fingerprint: TeacherFingerprint,
) -> None:
    if config.install_manifest is None:
        return
    path = resolve_config_path(
        project_root,
        config.install_manifest,
        field="teacher.install_manifest",
    )
    try:
        manifest, _ = load_json_object(path, max_bytes=1024 * 1024)
    except ArtifactError as error:
        raise TeacherFingerprintError(str(error)) from error
    common_root_keys = {
        "schema",
        "created_at",
        "teacher",
        "official",
        "archive",
        "source",
        "build",
        "binary",
        "evaluation",
        "licenses",
    }
    schema = manifest.get("schema")
    expected_root_keys = (
        common_root_keys | {"official_tree"}
        if schema == INSTALL_MANIFEST_SCHEMA
        else common_root_keys
    )
    if (
        schema not in {LEGACY_INSTALL_MANIFEST_SCHEMA, INSTALL_MANIFEST_SCHEMA}
        or set(manifest) != expected_root_keys
    ):
        raise TeacherFingerprintError("teacher install manifest schema is invalid")
    _verify_timestamp(manifest["created_at"], "teacher install manifest created_at")
    teacher = manifest.get("teacher")
    if not isinstance(teacher, dict):
        raise TeacherFingerprintError("teacher install manifest has no teacher object")
    if teacher.get("name") != config.name or teacher.get("version") != config.version:
        raise TeacherFingerprintError("teacher install manifest identity disagrees with config")
    if set(teacher) != {"name", "version"}:
        raise TeacherFingerprintError("teacher install manifest teacher keys are invalid")
    if (config.name, config.version) != ("Apery", "2.0.0"):
        raise TeacherFingerprintError("no immutable install policy exists for this teacher")
    if manifest["official"] != _APERY_OFFICIAL:
        raise TeacherFingerprintError("teacher official release identity drifted")
    if manifest["archive"] != {
        "path": "local/teacher/cache/apery_2.0.0.zip",
        "sha256": _APERY_ARCHIVE_SHA256,
        "size": _APERY_ARCHIVE_SIZE,
        "entries": _APERY_ARCHIVE_ENTRIES,
    }:
        raise TeacherFingerprintError("teacher install manifest archive identity drifted")
    _verify_source_tree(manifest["source"], project_root, config)
    _verify_official_tree(
        manifest.get("official_tree") if schema == INSTALL_MANIFEST_SCHEMA else None,
        project_root,
        config,
    )
    _verify_build(manifest["build"], project_root, schema=schema)
    binary = manifest.get("binary")
    if not isinstance(binary, dict):
        raise TeacherFingerprintError("teacher install manifest has no binary object")
    _compare_manifest_file(binary, fingerprint.binary, "binary")
    if binary != {
        "path": config.executable,
        "sha256": _APERY_BINARY_SHA256,
        "size": _APERY_BINARY_SIZE,
    }:
        raise TeacherFingerprintError("teacher install manifest binary is not the pinned build")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("files"), list):
        raise TeacherFingerprintError("teacher install manifest evaluation files are invalid")
    recorded: dict[str, dict[str, Any]] = {}
    for item in evaluation["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise TeacherFingerprintError("teacher install manifest eval entry is invalid")
        if item["path"] in recorded:
            raise TeacherFingerprintError("teacher install manifest repeats an eval path")
        recorded[item["path"]] = item
    if set(recorded) != {item.path for item in fingerprint.eval_files}:
        raise TeacherFingerprintError("teacher install manifest eval set disagrees with config")
    for item in fingerprint.eval_files:
        _compare_manifest_file(recorded[item.path], item, f"eval {item.path}")
    _verify_evaluation(manifest["evaluation"], manifest["licenses"], project_root, config)


def _compare_manifest_file(
    recorded: dict[str, Any],
    actual: FingerprintedFile,
    name: str,
) -> None:
    if (
        set(recorded) != {"path", "sha256", "size"}
        or isinstance(recorded.get("size"), bool)
        or recorded.get("path") != actual.path
        or recorded.get("sha256") != actual.sha256
        or recorded.get("size") != actual.size
    ):
        raise TeacherFingerprintError(f"teacher install manifest {name} identity drifted")


def _verify_source_tree(value: object, project_root: Path, config: TeacherConfig) -> None:
    expected = {
        "path": config.cwd,
        "tree_hash_algorithm": (
            "sha256(path-length,path,size,file-sha256) excluding target, eval, Windows exe"
        ),
        "tree_sha256": _APERY_SOURCE_TREE_SHA256,
        "files": _APERY_SOURCE_FILES,
        "size": _APERY_SOURCE_SIZE,
    }
    if value != expected:
        raise TeacherFingerprintError("teacher install manifest source tree identity drifted")
    root = resolve_config_path(project_root, config.cwd, field="teacher source")
    digest = hashlib.sha256()
    files = 0
    total = 0
    source_files: list[Path] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            if child.is_symlink():
                raise TeacherFingerprintError("teacher source tree contains a symlink directory")
        names[:] = [name for name in names if name not in {"target", "eval"}]
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(root)
            if relative.name == "apery_2.0.0.exe":
                continue
            if path.is_symlink() or not path.is_file():
                raise TeacherFingerprintError("teacher source tree contains an unsafe file")
            source_files.append(path)
    for path in sorted(source_files):
        relative = path.relative_to(root)
        try:
            file_digest = hash_regular_file(path, max_bytes=16 * 1024 * 1024)
        except ArtifactError as error:
            raise TeacherFingerprintError(str(error)) from error
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(file_digest.size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest.sha256))
        files += 1
        total += file_digest.size
    if (
        digest.hexdigest() != _APERY_SOURCE_TREE_SHA256
        or files != _APERY_SOURCE_FILES
        or total != _APERY_SOURCE_SIZE
    ):
        raise TeacherFingerprintError("teacher source tree bytes or file set drifted")


def _verify_official_tree(
    value: object | None,
    project_root: Path,
    config: TeacherConfig,
) -> None:
    expected = {
        "path": config.cwd,
        "tree_hash_algorithm": (
            "sha256(archive-path-length,archive-path,size,file-sha256) excluding target"
        ),
        "tree_sha256": _APERY_ARCHIVE_TREE_SHA256,
        "files": _APERY_ARCHIVE_FILES,
        "directories": len(_APERY_OFFICIAL_DIRECTORIES),
        "size": _APERY_ARCHIVE_UNCOMPRESSED,
    }
    if value is not None and value != expected:
        raise TeacherFingerprintError("teacher official source-tree evidence drifted")
    root = resolve_config_path(project_root, config.cwd, field="teacher official source")
    paths: list[Path] = []
    directories: set[str] = set()
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            if child.is_symlink():
                raise TeacherFingerprintError(
                    "teacher official source tree contains a symlink directory"
                )
        if directory_path == root:
            names[:] = [name for name in names if name != "target"]
        for name in names:
            directories.add((directory_path / name).relative_to(root).as_posix())
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink() or not path.is_file():
                raise TeacherFingerprintError(
                    "teacher official source tree contains an unsafe file"
                )
            paths.append(path)
            if len(paths) > _APERY_ARCHIVE_FILES:
                raise TeacherFingerprintError("teacher official source file set drifted")
    digest = hashlib.sha256()
    total = 0
    try:
        for path in sorted(paths):
            relative = path.relative_to(root)
            file_digest = hash_regular_file(path, max_bytes=_APERY_ARCHIVE_UNCOMPRESSED)
            archive_path = PurePosixPath("apery_2.0.0", *relative.parts).as_posix()
            encoded = archive_path.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(file_digest.size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(file_digest.sha256))
            total += file_digest.size
    except ArtifactError as error:
        raise TeacherFingerprintError(str(error)) from error
    if (
        len(paths) != _APERY_ARCHIVE_FILES
        or directories != _APERY_OFFICIAL_DIRECTORIES
        or total != _APERY_ARCHIVE_UNCOMPRESSED
        or digest.hexdigest() != _APERY_ARCHIVE_TREE_SHA256
    ):
        raise TeacherFingerprintError("teacher installed official source tree drifted")


def _verify_build(value: object, project_root: Path, *, schema: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "command",
        "cargo_version",
        "rustc_version",
        "host_os",
        "host_arch",
        "compatibility_override",
    }:
        raise TeacherFingerprintError("teacher install manifest build evidence is invalid")
    legacy_command = ["cargo", "build", "--release", "--locked"]
    sandbox_command_prefix = [
        "sandbox-exec",
        "cargo",
        "--config",
        "<pinned-official-config>",
        "build",
        "--manifest-path",
        "<private-source>/Cargo.toml",
        "--target-dir",
    ]
    sandbox_commands = [
        [*sandbox_command_prefix, target, "--release", "--locked"]
        for target in ("<private-source>/target", "<private-build-target>")
    ]
    expected_commands = (
        [legacy_command] if schema == LEGACY_INSTALL_MANIFEST_SCHEMA else sandbox_commands
    )
    if (
        value["command"] not in expected_commands
        or value["cargo_version"] != _APERY_CARGO_VERSION
        or value["rustc_version"] != _APERY_RUSTC_VERSION
        or value["host_os"] != "Darwin"
        or value["host_arch"] != "arm64"
    ):
        raise TeacherFingerprintError("teacher install manifest build/toolchain drifted")
    override = value["compatibility_override"]
    expected_override_keys = {
        "reason",
        "package",
        "version",
        "license",
        "crate_sha256",
        "lockfile",
    }
    if not isinstance(override, dict) or set(override) != expected_override_keys:
        raise TeacherFingerprintError("teacher compatibility evidence is invalid")
    if (
        override["reason"]
        != "upstream num-bigint 0.4.0 does not compile with Rust 1.94 integer div_ceil"
        or override["package"] != "num-bigint"
        or override["version"] != "0.4.6"
        or override["license"] != "MIT OR Apache-2.0"
        or override["crate_sha256"]
        != "a5e44f723f1133c9deac646763579fdb3ac745e418f2a7af9cd0c431da1f20b9"
    ):
        raise TeacherFingerprintError("teacher compatibility override drifted")
    lockfile = override["lockfile"]
    expected_lock = {
        "path": "local/teacher/build/apery-v2.0.0/Cargo.lock.rust-1.94-num-bigint-0.4.6",
        "sha256": _APERY_COMPAT_LOCK_SHA256,
        "size": _APERY_COMPAT_LOCK_SIZE,
    }
    if lockfile != expected_lock:
        raise TeacherFingerprintError("teacher compatibility lock evidence drifted")
    path = resolve_config_path(project_root, expected_lock["path"], field="compatibility lock")
    try:
        digest = hash_regular_file(path, max_bytes=64 * 1024)
    except ArtifactError as error:
        raise TeacherFingerprintError(str(error)) from error
    if digest.sha256 != _APERY_COMPAT_LOCK_SHA256 or digest.size != _APERY_COMPAT_LOCK_SIZE:
        raise TeacherFingerprintError("teacher compatibility lock bytes drifted")


def _verify_evaluation(
    evaluation: object,
    licenses: object,
    project_root: Path,
    config: TeacherConfig,
) -> None:
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "directory",
        "license",
        "readme",
        "files",
    }:
        raise TeacherFingerprintError("teacher evaluation evidence is invalid")
    if evaluation["directory"] != config.eval_dir or evaluation["license"] != "MIT":
        raise TeacherFingerprintError("teacher evaluation identity drifted")
    readme_path = f"{config.eval_dir}/README.md"
    expected_readme = {
        "path": readme_path,
        "sha256": _APERY_README[0],
        "size": _APERY_README[1],
    }
    if evaluation["readme"] != expected_readme:
        raise TeacherFingerprintError("teacher evaluation README evidence drifted")
    _verify_file_record(expected_readme, project_root, "evaluation README")
    if not isinstance(licenses, list) or len(licenses) != len(_APERY_LICENSES):
        raise TeacherFingerprintError("teacher license evidence set is invalid")
    expected_licenses = []
    source_root = config.cwd
    for relative, (license_id, sha256, size) in _APERY_LICENSES.items():
        record = {
            "path": f"{source_root}/{relative}",
            "license": license_id,
            "sha256": sha256,
            "size": size,
        }
        expected_licenses.append(record)
        _verify_file_record(record, project_root, f"license {license_id}")
    if licenses != expected_licenses:
        raise TeacherFingerprintError("teacher license evidence drifted")


def _verify_file_record(record: dict[str, Any], project_root: Path, name: str) -> None:
    path = resolve_config_path(project_root, record["path"], field=name)
    try:
        digest = hash_regular_file(path, max_bytes=64 * 1024)
    except ArtifactError as error:
        raise TeacherFingerprintError(str(error)) from error
    if digest.sha256 != record["sha256"] or digest.size != record["size"]:
        raise TeacherFingerprintError(f"teacher {name} bytes drifted")


def _command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise TeacherFingerprintError(f"cannot inspect {' '.join(command)}") from error
    try:
        output = completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise TeacherFingerprintError(f"{' '.join(command)} emitted non-UTF-8") from error
    if completed.returncode != 0 or not output or "\n" in output:
        raise TeacherFingerprintError(f"cannot inspect {' '.join(command)}")
    return output


def _verify_timestamp(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 64:
        raise TeacherFingerprintError(f"{name} is invalid")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise TeacherFingerprintError(f"{name} is invalid") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise TeacherFingerprintError(f"{name} is not UTC")
