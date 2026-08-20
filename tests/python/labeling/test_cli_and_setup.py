from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from open_shogi_training.labeling.__main__ import build_parser
from open_shogi_training.labeling.config import (
    APERY_V2_BINARY_SHA256,
    TeacherConfigError,
    load_teacher_config,
)
from open_shogi_training.labeling.fingerprint import (
    TeacherFingerprintError,
    _validate_official_native_executable,
)

from .helpers import make_fake_project

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _setup_cleanup_program() -> str:
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")
    return script.split("        || cleanup_result=$?\n", 1)[1].split("\nPY\n", 1)[0]


def test_label_cli_allows_only_operational_stage_checkpoints() -> None:
    parser = build_parser()
    common = [
        "label",
        "--config",
        "teacher.yaml",
        "--positions",
        "positions.jsonl.gz",
        "--dataset-manifest",
        "manifest.json",
        "--benchmark-report",
        "benchmark.json",
        "--output-dir",
        "labels",
    ]
    parsed = parser.parse_args([*common, "--target-completed", "1000"])
    assert parsed.target_completed == 1000

    with pytest.raises(SystemExit) as error:
        parser.parse_args([*common, "--target-completed", "20"])
    assert error.value.code == 2


def test_read_only_label_audit_requires_all_bound_inputs() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "audit-label-set",
            "--config",
            "teacher.yaml",
            "--positions",
            "positions.jsonl.gz",
            "--dataset-manifest",
            "manifest.json",
            "--benchmark-report",
            "benchmark.json",
            "--output-dir",
            "labels",
        ]
    )
    assert parsed.command == "audit-label-set"


def test_apery_setup_is_pinned_local_and_never_uses_privilege_or_extractall() -> None:
    path = PROJECT_ROOT / "scripts/setup_teacher_apery.sh"
    script = path.read_text(encoding="utf-8")

    assert path.stat().st_mode & stat.S_IXUSR
    assert (
        "https://github.com/HiraokaTakuya/apery_rust/releases/download/v2.0.0/apery_2.0.0.zip"
    ) in script
    assert "22c662f1a7c28f79dd51a8a2b80179fc2ef063d0dc42df5232028c9c2390cad3" in script
    assert "a570784542f7e50fb39a24129f02d1b14819eec1" in script
    assert "cargo build --release --locked" in script
    assert 'EXPECTED_RUSTC_VERSION="rustc 1.94.1 (e408947bf 2026-03-25)"' in script
    assert 'EXPECTED_CARGO_VERSION="cargo 1.94.1 (29ea6fb6a 2026-03-24)"' in script
    assert (
        'EXPECTED_BINARY_SHA256="8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403"'
        in script
    )
    assert (
        'ARCHIVE_TREE_SHA256="9d47d5a0e0c0df94224b5b03278c0571b9e508e7a239e71394fc7f0f66278fea"'
        in script
    )
    assert 'TEACHER_ROOT="$PROJECT_ROOT/local/teacher"' in script
    assert "sudo" not in script
    assert "extractall" not in script
    assert "PurePosixPath" in script
    assert 'EXTRACT_STAGE=$(mktemp -d "$BUILD_ROOT/.extract.XXXXXX")' in script
    assert 'if [ ! -d "$SOURCE_DIR" ]' not in script
    assert 'PurePosixPath(entry.filename).name == "build.rs"' in script
    assert "ambient parent Cargo config is not allowed" in script
    assert 'CARGO_HOME_LOCAL="$EXTRACT_STAGE/cargo-home"' in script
    assert "post-build official source tree drifted" in script
    assert "cached archive publication identity changed" in script
    assert "compatibility lock publication identity changed" in script
    assert "foreign bytes were not deleted" in script
    assert "existing teacher install differs; preserve it and resolve explicitly" in script
    assert "existing install manifest differs; preserve it and resolve explicitly" in script
    assert "renameatx_np" in script
    assert "os.link(" not in script
    assert "retained temporary descriptor has the wrong kind" in script
    assert "foreign bytes retained" in script
    assert "teacher setup quarantine entry quota is exhausted" in script
    assert "os.fstat(source_descriptor).st_nlink != 0" in script
    assert 'PUBLISH_STARTED="1"' not in script
    assert ".manifest-backup.XXXXXX" not in script
    assert "expected_official_directories" in script
    assert "map(Path, sys.argv[1:])" not in script


def test_every_embedded_setup_python_program_compiles() -> None:
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")
    programs = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script, flags=re.DOTALL)

    assert len(programs) == 21
    for index, program in enumerate(programs, start=1):
        compile(program, f"setup_teacher_apery.sh:heredoc-{index}", "exec")
    compile(_setup_cleanup_program(), "setup_teacher_apery.sh:cleanup-heredoc", "exec")


def test_setup_cleanup_preserves_a_foreign_replacement_by_retained_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    build = tmp_path / "build"
    cache.mkdir()
    build.mkdir()
    temporary = cache / ".apery_2.0.0.zip.owned"
    temporary.write_bytes(b"owned")
    temporary_fd = os.open(temporary, os.O_RDONLY)
    cache_fd = os.open(cache, os.O_RDONLY)
    build_fd = os.open(build, os.O_RDONLY)
    retained = cache / "retained-owned"
    temporary.rename(retained)
    temporary.write_bytes(b"foreign-do-not-rename")
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "setup-cleanup",
                str(temporary),
                "",
                str(cache),
                str(build),
                "",
                "",
                "",
                "0" * 64,
                "1" * 64,
                str(temporary_fd),
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                str(cache_fd),
                str(build_fd),
            ],
        )
        with pytest.raises(RuntimeError, match="foreign bytes retained"):
            exec(compile(_setup_cleanup_program(), "setup-cleanup", "exec"), {})
    finally:
        os.close(temporary_fd)
        os.close(cache_fd)
        os.close(build_fd)

    assert temporary.read_bytes() == b"foreign-do-not-rename"
    assert retained.read_bytes() == b"owned"


def test_setup_successful_cleanup_does_not_accumulate_retired_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    build = tmp_path / "build"
    cache.mkdir()
    build.mkdir()
    cache_fd = os.open(cache, os.O_RDONLY)
    build_fd = os.open(build, os.O_RDONLY)
    try:
        for index in range(3):
            stage = build / f".extract.owned{index}"
            (stage / "nested").mkdir(parents=True)
            (stage / "nested/artifact").write_bytes(b"owned")
            stage_fd = os.open(stage, os.O_RDONLY)
            try:
                monkeypatch.setattr(
                    sys,
                    "argv",
                    [
                        "setup-cleanup",
                        "",
                        str(stage),
                        str(cache),
                        str(build),
                        "",
                        "",
                        "",
                        "0" * 64,
                        "1" * 64,
                        "-1",
                        str(stage_fd),
                        "-1",
                        "-1",
                        "-1",
                        "-1",
                        str(cache_fd),
                        str(build_fd),
                    ],
                )
                exec(compile(_setup_cleanup_program(), "setup-cleanup", "exec"), {})
            finally:
                os.close(stage_fd)
            assert not stage.exists()
        assert tuple(build.glob(".setup-retired.*")) == ()
    finally:
        os.close(cache_fd)
        os.close(build_fd)


def test_setup_archive_publication_rejects_a_verified_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")
    marker = (
        '    python3 - "$DOWNLOAD_TEMP" "$ARCHIVE" "$ARCHIVE_SHA256" "$ARCHIVE_SIZE" <<\'PY\'\n'
    )
    snippet = script.split(marker, 1)[1].split("\nPY\n", 1)[0]
    source = tmp_path / ".apery_2.0.0.zip.verified"
    destination = tmp_path / "apery_2.0.0.zip"
    replacement = tmp_path / "replacement.zip"
    payload = b"A" * (2 * 1024 * 1024)
    source.write_bytes(payload)
    replacement.write_bytes(b"marker-must-not-publish")
    expected_hash = hashlib.sha256(payload).hexdigest()
    real_read = os.read
    swapped = False

    def swap_after_first_chunk(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            os.replace(replacement, source)
        return chunk

    monkeypatch.setattr(os, "read", swap_after_first_chunk)
    monkeypatch.setattr(
        sys,
        "argv",
        ["setup-publication", str(source), str(destination), expected_hash, str(len(payload))],
    )
    with pytest.raises(SystemExit, match="changed during retained-descriptor verification"):
        exec(compile(snippet, "setup_teacher_apery.sh:archive-publication", "exec"), {})

    assert not destination.exists()
    assert source.read_bytes() == b"marker-must-not-publish"


def test_setup_extraction_uses_the_retained_archive_after_a_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")
    marker = 'python3 - "9" "$EXTRACT_STAGE" <<\'PY\'\n'
    snippet = script.split(marker, 1)[1].split("\nPY\n", 1)[0]
    archive_path = tmp_path / "apery.zip"
    replacement = tmp_path / "replacement.zip"
    stage = tmp_path / "stage"
    stage.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("apery_2.0.0/source.txt", b"verified archive bytes")
    with zipfile.ZipFile(replacement, "w") as archive:
        archive.writestr("apery_2.0.0/source.txt", b"replacement bytes")

    descriptor = os.open(archive_path, os.O_RDONLY)
    try:
        os.replace(replacement, archive_path)
        monkeypatch.setattr(sys, "argv", ["setup-extract", str(descriptor), str(stage)])
        exec(compile(snippet, "setup_teacher_apery.sh:archive-extract", "exec"), {})
    finally:
        os.close(descriptor)

    assert (stage / "apery_2.0.0/source.txt").read_bytes() == b"verified archive bytes"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("apery_2.0.0/source.txt") == b"replacement bytes"


def test_setup_cargo_ancestor_guard_rejects_config_created_during_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")
    before_marker = (
        'python3 - "$CARGO_DRIVER_CWD" "$PROJECT_ROOT" "$CARGO_ANCESTOR_GUARD" <<\'PY\'\n'
    )
    after_marker = "python3 - \"$CARGO_ANCESTOR_GUARD\" <<'PY'\n"
    before = script.split(before_marker, 1)[1].split("\nPY\n", 1)[0]
    after = script.split(after_marker, 1)[1].split("\nPY\n", 1)[0]
    project = tmp_path / "project"
    driver = project / "local/teacher/build/driver"
    driver.mkdir(parents=True)
    guard = tmp_path / "cargo-ancestor-guard.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["setup-cargo-before", str(driver), str(project), str(guard)],
    )
    exec(compile(before, "setup_teacher_apery.sh:cargo-before", "exec"), {})
    cargo_directory = project / ".cargo"
    cargo_directory.mkdir()
    (cargo_directory / "config.toml").write_text("[build]\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["setup-cargo-after", str(guard)])
    with pytest.raises(SystemExit, match="authority changed during the build"):
        exec(compile(after, "setup_teacher_apery.sh:cargo-after", "exec"), {})


def test_setup_repository_authority_lock_survives_build_directory_replacement(
    tmp_path: Path,
) -> None:
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")
    marker = (
        'python3 - "$SETUP_PARENT_LOCK_FD" "$SETUP_LOCK_FD" '
        '"$PROJECT_ROOT" "$BUILD_ROOT" <<\'PY\'\n'
    )
    snippet = script.split(marker, 1)[1].split("\nPY\n", 1)[0]
    project = tmp_path / "project"
    build = project / "local/teacher/build/apery-v2.0.0"
    build.mkdir(parents=True)
    first_project_descriptor = os.open(project, os.O_RDONLY)
    first_build_descriptor = os.open(build, os.O_RDONLY)
    try:
        first = subprocess.run(
            [
                sys.executable,
                "-c",
                snippet,
                str(first_project_descriptor),
                str(first_build_descriptor),
                os.fspath(project),
                os.fspath(build),
            ],
            pass_fds=(first_project_descriptor, first_build_descriptor),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=5,
        )
        assert first.returncode == 0, first.stderr.decode(errors="replace")

        build.rename(build.with_name("apery-v2.0.0-original"))
        build.mkdir()
        second_project_descriptor = os.open(project, os.O_RDONLY)
        second_build_descriptor = os.open(build, os.O_RDONLY)
        try:
            second = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    snippet,
                    str(second_project_descriptor),
                    str(second_build_descriptor),
                    os.fspath(project),
                    os.fspath(build),
                ],
                pass_fds=(second_project_descriptor, second_build_descriptor),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=5,
            )
        finally:
            os.close(second_build_descriptor)
            os.close(second_project_descriptor)
    finally:
        os.close(first_build_descriptor)
        os.close(first_project_descriptor)

    assert second.returncode != 0
    assert b"another Apery setup is active" in second.stderr


def test_production_teacher_binary_pin_preserves_completed_v1_config_identity() -> None:
    config = (PROJECT_ROOT / "configs/teacher/apery-v2.0.0.yaml").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "scripts/setup_teacher_apery.sh").read_text(encoding="utf-8")

    assert (
        'EXPECTED_BINARY_SHA256="8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403"'
        in script
    )
    assert "binary_sha256: null" in config
    assert "preserve the immutable v1 config SHA" in config
    loaded = load_teacher_config(PROJECT_ROOT / "configs/teacher/apery-v2.0.0.yaml")
    assert loaded.sha256 == "50ecf1b0a3395c0c00208b2d4bad12bb5d16c1f2ec49b8f66fa234a8ab847ba3"
    assert loaded.runtime_binary_sha256 == APERY_V2_BINARY_SHA256


def test_official_teacher_launch_boundary_rejects_scripts_and_hardlinks(
    tmp_path: Path,
) -> None:
    script = tmp_path / "teacher-script"
    script.write_bytes(b"#!/bin/sh\nexit 0\n")
    script.chmod(0o700)
    with pytest.raises(TeacherFingerprintError, match="native Mach-O"):
        _validate_official_native_executable(script)

    native = tmp_path / "teacher-native"
    native.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 16)
    native.chmod(0o700)
    alias = tmp_path / "teacher-native-alias"
    os.link(native, alias)
    with pytest.raises(TeacherFingerprintError, match="single-link"):
        _validate_official_native_executable(native)


def test_legacy_null_pin_is_limited_to_the_exact_immutable_apery_config(
    tmp_path: Path,
) -> None:
    _, _, config_path = make_fake_project(tmp_path, max_positions=1)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["teacher"]["name"] = "Apery"
    payload["teacher"]["version"] = "2.0.0"
    payload["teacher"]["binary_sha256"] = None
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TeacherConfigError, match="exact immutable Apery"):
        load_teacher_config(config_path)
