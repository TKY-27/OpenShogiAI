from __future__ import annotations

import json
from pathlib import Path

import pytest
from open_shogi_training.labeling import fingerprint as fingerprint_module
from open_shogi_training.labeling.artifacts import FileDigest
from open_shogi_training.labeling.config import load_teacher_config
from open_shogi_training.labeling.fingerprint import (
    FingerprintedFile,
    TeacherFingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_runtime_install_verification_needs_neither_archive_cache_nor_active_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_teacher_config(PROJECT_ROOT / "configs/teacher/apery-v2.0.0.yaml")
    fingerprint = TeacherFingerprint(
        name=config.name,
        version=config.version,
        binary=FingerprintedFile(
            config.executable,
            fingerprint_module._APERY_BINARY_SHA256,
            fingerprint_module._APERY_BINARY_SIZE,
        ),
        eval_files=tuple(
            FingerprintedFile(item.path, item.sha256, index + 1)
            for index, item in enumerate(config.eval_files)
        ),
        options=config.option_map,
    )
    install_path = tmp_path.joinpath(*Path(config.install_manifest or "").parts)
    install_path.parent.mkdir(parents=True)
    build = {
        "command": [
            "sandbox-exec",
            "cargo",
            "--config",
            "<pinned-official-config>",
            "build",
            "--manifest-path",
            "<private-source>/Cargo.toml",
            "--target-dir",
            "<private-build-target>",
            "--release",
            "--locked",
        ],
        "cargo_version": fingerprint_module._APERY_CARGO_VERSION,
        "rustc_version": fingerprint_module._APERY_RUSTC_VERSION,
        "host_os": "Darwin",
        "host_arch": "arm64",
        "compatibility_override": {
            "reason": (
                "upstream num-bigint 0.4.0 does not compile with Rust 1.94 integer div_ceil"
            ),
            "package": "num-bigint",
            "version": "0.4.6",
            "license": "MIT OR Apache-2.0",
            "crate_sha256": ("a5e44f723f1133c9deac646763579fdb3ac745e418f2a7af9cd0c431da1f20b9"),
            "lockfile": {
                "path": ("local/teacher/build/apery-v2.0.0/Cargo.lock.rust-1.94-num-bigint-0.4.6"),
                "sha256": fingerprint_module._APERY_COMPAT_LOCK_SHA256,
                "size": fingerprint_module._APERY_COMPAT_LOCK_SIZE,
            },
        },
    }
    manifest = {
        "schema": fingerprint_module.INSTALL_MANIFEST_SCHEMA,
        "created_at": "2026-08-08T00:00:00.000Z",
        "teacher": {"name": config.name, "version": config.version},
        "official": fingerprint_module._APERY_OFFICIAL,
        "archive": {
            "path": "local/teacher/cache/apery_2.0.0.zip",
            "sha256": fingerprint_module._APERY_ARCHIVE_SHA256,
            "size": fingerprint_module._APERY_ARCHIVE_SIZE,
            "entries": fingerprint_module._APERY_ARCHIVE_ENTRIES,
        },
        "source": {},
        "official_tree": {},
        "build": build,
        "binary": fingerprint.binary.as_dict(),
        "evaluation": {"files": [item.as_dict() for item in fingerprint.eval_files]},
        "licenses": {},
    }
    install_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    real_resolve = fingerprint_module.resolve_config_path

    def reject_archive_resolution(root: Path, value: str, *, field: str) -> Path:
        if value == "local/teacher/cache/apery_2.0.0.zip":
            raise AssertionError("runtime verification attempted to read the download cache")
        return real_resolve(root, value, field=field)

    monkeypatch.setattr(fingerprint_module, "resolve_config_path", reject_archive_resolution)
    monkeypatch.setattr(fingerprint_module, "_verify_source_tree", lambda *_args: None)
    monkeypatch.setattr(fingerprint_module, "_verify_official_tree", lambda *_args: None)
    monkeypatch.setattr(fingerprint_module, "_verify_evaluation", lambda *_args: None)

    def pinned_lock_digest(path: Path, *, max_bytes: int) -> FileDigest:
        assert path.name == "Cargo.lock.rust-1.94-num-bigint-0.4.6"
        assert max_bytes == 64 * 1024
        return FileDigest(
            fingerprint_module._APERY_COMPAT_LOCK_SHA256,
            fingerprint_module._APERY_COMPAT_LOCK_SIZE,
        )

    monkeypatch.setattr(fingerprint_module, "hash_regular_file", pinned_lock_digest)
    monkeypatch.setattr(
        fingerprint_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime verification invoked an active toolchain")
        ),
    )

    fingerprint_module._verify_install_manifest(config, tmp_path, fingerprint)
