#!/bin/sh
set -eu

# Install the audited Apery v2.0.0 release only under ignored local/teacher.
# This script never elevates privileges or copies/links teacher code into production.
# The sandboxed invocation preserves the audited `cargo build --release --locked`
# semantics while pinning its manifest, target, configuration, and environment.

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
TEACHER_ROOT="$PROJECT_ROOT/local/teacher"
CACHE_DIR="$TEACHER_ROOT/cache"
ARCHIVE="$CACHE_DIR/apery_2.0.0.zip"
BUILD_ROOT="$TEACHER_ROOT/build/apery-v2.0.0"
FINAL_SOURCE_DIR="$BUILD_ROOT/apery_2.0.0"
SOURCE_DIR=""
INSTALL_MANIFEST="$BUILD_ROOT/install-manifest.json"
COMPAT_LOCK="$BUILD_ROOT/Cargo.lock.rust-1.94-num-bigint-0.4.6"
CARGO_HOME_LOCAL=""
CARGO_DRIVER_CWD=""
TEACHER_PRIVATE_HOME=""
TEACHER_TEMP_DIR=""
CARGO_ANCESTOR_GUARD=""
CARGO_CONFIG_PIN=""
CARGO_SANDBOX_PROFILE=""
SOURCE_MODE_GUARD=""
RELEASE_URL="https://github.com/HiraokaTakuya/apery_rust/releases/download/v2.0.0/apery_2.0.0.zip"
REPOSITORY_URL="https://github.com/HiraokaTakuya/apery_rust"
SOURCE_COMMIT="a570784542f7e50fb39a24129f02d1b14819eec1"
ARCHIVE_SHA256="22c662f1a7c28f79dd51a8a2b80179fc2ef063d0dc42df5232028c9c2390cad3"
ARCHIVE_SIZE="641487453"
ARCHIVE_ENTRIES="83"
ARCHIVE_UNCOMPRESSED_SIZE="896036520"
ARCHIVE_TREE_SHA256="9d47d5a0e0c0df94224b5b03278c0571b9e508e7a239e71394fc7f0f66278fea"
ARCHIVE_FILES="76"
UPSTREAM_MANIFEST_SHA256="34b0067d7f4b2d486e1fcccdbb57df26b7907d1003b4c0f0d45ee6a70a0b93ee"
UPSTREAM_LOCK_SHA256="c538702e6c33c4644495add0f97aa4b4de0520c902d85b57d35969ee1534aa61"
ENGINE_LICENSE_SHA256="0b383d5a63da644f628d99c33976ea6487ed89aaa59f0b3257992deac1171e6b"
EVAL_LICENSE_SHA256="cdbd06e25b8c9c5d6019949d2de8123f56d29cee4166396423d2ba8c81700845"
EVAL_README_SHA256="647386fbe09e430d58f2b25b9cb1cf87fd54cde56a8c9bd4c27f8f53358f64c9"
KKP_SHA256="422b23bced817ecb3430adf1d2621f5a7934263b4e46673ab80cf34633537fa5"
KPP_SHA256="4906c48c201a102ec02217216929c20f73ab364e79be26e6213a02c04e454805"
NUM_BIGINT_VERSION="0.4.6"
NUM_BIGINT_CRATE_SHA256="a5e44f723f1133c9deac646763579fdb3ac745e418f2a7af9cd0c431da1f20b9"
COMPAT_LOCK_SHA256="5acb657fe557553603ec55343840689ec7fe8fc2799d009bba51ae79ddcfb737"
COMPAT_LOCK_SIZE="12773"
EXPECTED_RUSTC_VERSION="rustc 1.94.1 (e408947bf 2026-03-25)"
EXPECTED_CARGO_VERSION="cargo 1.94.1 (29ea6fb6a 2026-03-24)"
EXPECTED_BINARY_SHA256="8ccec09190d643f50b08a0a4b3359a6289656e99f6cf846ad8490a1dc28c3403"
EXPECTED_BINARY_SIZE="2364896"
EXPECTED_SOURCE_TREE_SHA256="055607a5aef06a044dccfd4ff18bff548b69a05786085134b36fec2928dc9447"
EXPECTED_SOURCE_FILES="38"
EXPECTED_SOURCE_SIZE="634564"
DOWNLOAD_TEMP=""
EXTRACT_STAGE=""
SOURCE_MANIFEST_BACKUP=""
SOURCE_LOCK_BACKUP=""
DOWNLOAD_TEMP_FD="-1"
EXTRACT_STAGE_FD="-1"
SOURCE_MANIFEST_BACKUP_FD="-1"
SOURCE_LOCK_BACKUP_FD="-1"
SOURCE_MANIFEST_TARGET_FD="-1"
SOURCE_LOCK_TARGET_FD="-1"
SETUP_PARENT_LOCK_FD="7"
SETUP_LOCK_FD="8"
SETUP_CACHE_FD="14"

fail() {
    echo "error: $*" >&2
    exit 1
}

cleanup() {
    cleanup_status=$?
    trap - EXIT HUP INT TERM
    cleanup_result=0
    python3 - "$DOWNLOAD_TEMP" "$EXTRACT_STAGE" "$CACHE_DIR" "$BUILD_ROOT" \
        "$SOURCE_MANIFEST_BACKUP" "$SOURCE_DIR" "$SOURCE_LOCK_BACKUP" \
        "$UPSTREAM_MANIFEST_SHA256" "$UPSTREAM_LOCK_SHA256" \
        "$DOWNLOAD_TEMP_FD" "$EXTRACT_STAGE_FD" \
        "$SOURCE_MANIFEST_BACKUP_FD" "$SOURCE_LOCK_BACKUP_FD" \
        "$SOURCE_MANIFEST_TARGET_FD" "$SOURCE_LOCK_TARGET_FD" \
        "$SETUP_CACHE_FD" "$SETUP_LOCK_FD" <<'PY' \
        || cleanup_result=$?
import ctypes
import os
import stat
import sys
from secrets import token_hex
from pathlib import Path

(
    download_raw,
    stage_raw,
    cache_raw,
    build_raw,
    manifest_backup_raw,
    source_root_raw,
    lock_backup_raw,
    upstream_manifest_sha256,
    upstream_lock_sha256,
    download_fd_raw,
    stage_fd_raw,
    manifest_backup_fd_raw,
    lock_backup_fd_raw,
    manifest_target_fd_raw,
    lock_target_fd_raw,
    cache_fd_raw,
    build_fd_raw,
) = sys.argv[1:]
cache = Path(cache_raw)
build = Path(build_raw)

if not any((download_raw, stage_raw, manifest_backup_raw, lock_backup_raw)):
    raise SystemExit(0)


def require_owned_child(path, parent, prefix):
    if path.parent != parent or not path.name.startswith(prefix):
        raise RuntimeError(f"refusing to clean an unexpected path: {path}")


def open_parent(path):
    parent = path.parent.resolve(strict=True)
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    return descriptor, path.name


def same_inode(first, second):
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def retained_status(fd_raw, path, *, directory):
    descriptor = int(fd_raw)
    if descriptor < 0:
        raise RuntimeError(f"temporary path has no retained descriptor: {path}")
    status = os.fstat(descriptor)
    if directory != stat.S_ISDIR(status.st_mode) or (
        not directory and not stat.S_ISREG(status.st_mode)
    ):
        raise RuntimeError(f"retained temporary descriptor has the wrong kind: {path}")
    return descriptor, status


def rename_noreplace(parent, source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameatx_np
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent, os.fsencode(source), parent, os.fsencode(destination), 4):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def quarantine_owned(path, parent, prefix, *, directory, source_fd_raw, parent_fd_raw):
    require_owned_child(path, parent, prefix)
    source_descriptor, expected = retained_status(
        source_fd_raw,
        path,
        directory=directory,
    )
    parent_descriptor = int(parent_fd_raw)
    parent_status = os.fstat(parent_descriptor)
    linked_parent = path.parent.lstat()
    if not stat.S_ISDIR(parent_status.st_mode) or not same_inode(
        parent_status, linked_parent
    ):
        raise RuntimeError(f"temporary cleanup parent authority changed: {path.parent}")
    name = path.name
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if expected.st_nlink == 0:
            return
        raise RuntimeError(f"retained temporary path disappeared: {path}") from None
    if not same_inode(linked, expected):
        raise RuntimeError(f"temporary cleanup path was replaced; foreign bytes retained: {path}")
    try:
        retired_name = f".setup-retired.{expected.st_dev:x}.{expected.st_ino:x}.{token_hex(12)}"
        rename_noreplace(parent_descriptor, name, retired_name)
        moved = os.stat(retired_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not same_inode(moved, expected) or not same_inode(
            os.fstat(source_descriptor), expected
        ):
            raise RuntimeError(f"temporary cleanup entry was replaced; retained at {retired_name}")
        if directory:
            prove_disposable_tree(source_descriptor)
            clear_disposable_tree(source_descriptor)
            current = os.stat(retired_name, dir_fd=parent_descriptor, follow_symlinks=False)
            if not same_inode(current, expected) or os.listdir(source_descriptor):
                raise RuntimeError("retired temporary directory changed during cleanup")
            os.rmdir(retired_name, dir_fd=parent_descriptor)
        else:
            if expected.st_uid != os.getuid() or expected.st_nlink != 1:
                raise RuntimeError("retired temporary file is not proven-owned")
            os.unlink(retired_name, dir_fd=parent_descriptor)
            if os.fstat(source_descriptor).st_nlink != 0:
                raise RuntimeError("retired temporary file link survived cleanup")
        os.fsync(parent_descriptor)
    except FileNotFoundError:
        raise RuntimeError(f"retained temporary path disappeared: {path}") from None


def prove_disposable_tree(descriptor, *, nodes=0, total=0):
    for name in os.listdir(descriptor):
        status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        nodes += 1
        if nodes > 65_536:
            raise RuntimeError("teacher cleanup tree exceeds its node bound")
        if stat.S_ISREG(status.st_mode):
            if status.st_uid != os.getuid() or status.st_nlink != 1:
                raise RuntimeError("teacher cleanup tree contains an unproven file")
            total += status.st_size
        elif stat.S_ISDIR(status.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                if not same_inode(opened, status) or opened.st_uid != os.getuid():
                    raise RuntimeError("teacher cleanup directory identity changed")
                nodes, total = prove_disposable_tree(child, nodes=nodes, total=total)
            finally:
                os.close(child)
        elif stat.S_ISLNK(status.st_mode):
            total += status.st_size
        else:
            raise RuntimeError("teacher cleanup tree contains a special entry")
        if total > 4 * 1024 * 1024 * 1024:
            raise RuntimeError("teacher cleanup tree exceeds its byte bound")
    return nodes, total


def clear_disposable_tree(descriptor):
    for name in os.listdir(descriptor):
        linked = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(linked.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                if not same_inode(os.fstat(child), linked):
                    raise RuntimeError("teacher cleanup directory entry changed")
                clear_disposable_tree(child)
            finally:
                os.close(child)
            if not same_inode(
                os.stat(name, dir_fd=descriptor, follow_symlinks=False), linked
            ):
                raise RuntimeError("teacher cleanup directory entry changed")
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(linked.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                if (
                    not same_inode(opened, linked)
                    or opened.st_uid != os.getuid()
                    or opened.st_nlink != 1
                ):
                    raise RuntimeError("teacher cleanup file entry changed")
                os.unlink(name, dir_fd=descriptor)
                if os.fstat(child).st_nlink != 0:
                    raise RuntimeError("teacher cleanup file link survived")
            finally:
                os.close(child)
        elif stat.S_ISLNK(linked.st_mode):
            os.unlink(name, dir_fd=descriptor)
        else:
            raise RuntimeError("teacher cleanup tree contains a special entry")


def descriptor_sha256(descriptor, maximum_bytes):
    digest = __import__("hashlib").sha256()
    status = os.fstat(descriptor)
    if status.st_size > maximum_bytes:
        raise RuntimeError("source restoration file exceeds its safety bound")
    offset = 0
    while offset < status.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, status.st_size - offset), offset)
        if not chunk:
            raise RuntimeError("source restoration file ended early")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def destination_is_official(parent, name, expected_sha256):
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
    except FileNotFoundError:
        return False
    try:
        status = os.fstat(descriptor)
        return stat.S_ISREG(status.st_mode) and descriptor_sha256(
            descriptor, 32 * 1024 * 1024
        ) == expected_sha256
    finally:
        os.close(descriptor)


def restore_regular(
    backup,
    destination,
    expected_sha256,
    *,
    backup_fd_raw,
    target_fd_raw,
):
    require_owned_child(backup, Path(stage_raw), ".apery-Cargo.")
    backup_parent, backup_name = open_parent(backup)
    destination_parent, destination_name = open_parent(destination)
    backup_descriptor, backup_expected = retained_status(
        backup_fd_raw,
        backup,
        directory=False,
    )
    target_descriptor, target_expected = retained_status(
        target_fd_raw,
        destination,
        directory=False,
    )
    try:
        try:
            backup_linked = os.stat(
                backup_name,
                dir_fd=backup_parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_linked = os.stat(
                destination_name,
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
            if same_inode(destination_linked, backup_expected) and destination_is_official(
                destination_parent,
                destination_name,
                expected_sha256,
            ):
                return False
            raise RuntimeError(f"source restoration backup disappeared: {backup}") from None
        if not same_inode(backup_linked, backup_expected):
            raise RuntimeError(
                f"source restoration backup was replaced; foreign bytes retained: {backup}"
            )
        if descriptor_sha256(backup_descriptor, 32 * 1024 * 1024) != expected_sha256:
            raise RuntimeError(f"source restoration backup bytes drifted: {backup}")
        try:
            destination_linked = os.stat(
                destination_name,
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            raise RuntimeError(f"source restoration target disappeared: {destination}") from None
        if not same_inode(destination_linked, target_expected) or not same_inode(
            os.fstat(target_descriptor), target_expected
        ):
            raise RuntimeError(
                f"source restoration target was replaced; foreign bytes retained: {destination}"
            )
        if destination_is_official(destination_parent, destination_name, expected_sha256):
            return False
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(
            backup_parent,
            os.fsencode(backup_name),
            destination_parent,
            os.fsencode(destination_name),
            2,
        ):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination_name)
        restored = os.stat(destination_name, dir_fd=destination_parent, follow_symlinks=False)
        displaced = os.stat(backup_name, dir_fd=backup_parent, follow_symlinks=False)
        if not same_inode(restored, backup_expected) or not same_inode(
            displaced, target_expected
        ):
            raise RuntimeError("restored source exchange identity changed")
        if target_expected.st_uid != os.getuid() or target_expected.st_nlink != 1:
            raise RuntimeError("modified source target is not proven-owned")
        os.unlink(backup_name, dir_fd=backup_parent)
        if os.fstat(target_descriptor).st_nlink != 0:
            raise RuntimeError("modified source target link survived restoration")
        os.fsync(backup_parent)
        os.fsync(destination_parent)
        return True
    finally:
        os.close(backup_parent)
        os.close(destination_parent)


cache = cache.resolve(strict=True)
build = build.resolve(strict=True)
stage = Path(stage_raw) if stage_raw else None
if download_raw:
    quarantine_owned(
        Path(download_raw),
        cache,
        ".apery_2.0.0.zip.",
        directory=False,
        source_fd_raw=download_fd_raw,
        parent_fd_raw=cache_fd_raw,
    )

if manifest_backup_raw and source_root_raw:
    manifest_backup = Path(manifest_backup_raw)
    source_root = Path(source_root_raw)
    if stage is None:
        raise RuntimeError("source manifest backup has no owned extraction stage")
    restore_regular(
        manifest_backup,
        source_root / "Cargo.toml",
        upstream_manifest_sha256,
        backup_fd_raw=manifest_backup_fd_raw,
        target_fd_raw=manifest_target_fd_raw,
    )
if lock_backup_raw and source_root_raw:
    lock_backup = Path(lock_backup_raw)
    source_root = Path(source_root_raw)
    if stage is None:
        raise RuntimeError("source lock backup has no owned extraction stage")
    restore_regular(
        lock_backup,
        source_root / "Cargo.lock",
        upstream_lock_sha256,
        backup_fd_raw=lock_backup_fd_raw,
        target_fd_raw=lock_target_fd_raw,
    )
if stage_raw:
    quarantine_owned(
        stage,
        build,
        ".extract.",
        directory=True,
        source_fd_raw=stage_fd_raw,
        parent_fd_raw=build_fd_raw,
    )
PY
    if [ "$cleanup_result" -ne 0 ]; then
        echo "error: teacher setup cleanup or rollback failed" >&2
        if [ "$cleanup_status" -eq 0 ]; then
            cleanup_status=$cleanup_result
        fi
    fi
    exit "$cleanup_status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

[ "$(uname -s)" = "Darwin" ] || fail "Apery source setup is scoped to macOS"
[ "$(uname -m)" = "arm64" ] || fail "Apery source setup is scoped to Apple Silicon arm64"
for command_name in curl cargo rustc python3 git; do
    command -v "$command_name" >/dev/null 2>&1 || fail "missing required command: $command_name"
done
[ -x /usr/bin/sandbox-exec ] || fail "missing required macOS sandbox-exec"
CARGO_BIN=$(command -v cargo)
RUSTC_BIN=$(command -v rustc)
TEACHER_TOOL_PATH=$(dirname "$CARGO_BIN"):/usr/bin:/bin:/usr/sbin:/sbin
RUSTUP_HOME_PIN=$(dirname "$(dirname "$("$RUSTC_BIN" --print sysroot)")")
[ "$("$RUSTC_BIN" --version)" = "$EXPECTED_RUSTC_VERSION" ] || \
    fail "rustc toolchain differs from the pinned teacher build toolchain"
[ "$("$CARGO_BIN" --version)" = "$EXPECTED_CARGO_VERSION" ] || \
    fail "cargo toolchain differs from the pinned teacher build toolchain"

cd "$PROJECT_ROOT"
git check-ignore -q local/teacher/probe || fail "local/teacher is not ignored by Git"
python3 - "$PROJECT_ROOT" "$TEACHER_ROOT" "$CACHE_DIR" "$BUILD_ROOT" <<'PY'
import os
import stat
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve(strict=True)
expected = [
    project / "local",
    project / "local/teacher",
    project / "local/teacher/cache",
    project / "local/teacher/build",
    project / "local/teacher/build/apery-v2.0.0",
]
supplied = [
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    Path(sys.argv[4]),
]
if supplied != [expected[1], expected[2], expected[4]]:
    raise SystemExit("teacher path variables differ from the fixed ignored locations")
for path in expected:
    try:
        status = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=0o700 if path != expected[0] else 0o755)
        status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise SystemExit(f"teacher path component is not a real directory: {path}")
    if path.resolve(strict=True) != path:
        raise SystemExit(f"teacher path escaped ignored local storage: {path}")
    relative = path.relative_to(project)
    cursor = project
    for component in relative.parts:
        cursor = cursor / component
        mode = cursor.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise SystemExit(f"teacher path component is not a real directory: {cursor}")
PY
# Lock the retained repository authority before the exact build-directory inode.
# A named lock entry or any local/teacher path component can otherwise be replaced
# while the first writer still holds the old inode.
exec 7< "$PROJECT_ROOT"
exec 8< "$BUILD_ROOT"
exec 14< "$CACHE_DIR"
python3 - "$SETUP_PARENT_LOCK_FD" "$SETUP_LOCK_FD" "$PROJECT_ROOT" "$BUILD_ROOT" <<'PY'
import fcntl
import os
import stat
import sys
from pathlib import Path

parent_descriptor = int(sys.argv[1])
descriptor = int(sys.argv[2])
project = Path(sys.argv[3])
path = Path(sys.argv[4])
parent_status = os.fstat(parent_descriptor)
status = os.fstat(descriptor)
if not stat.S_ISDIR(parent_status.st_mode):
    raise SystemExit("teacher setup parent lock authority is not a directory")
try:
    fcntl.flock(parent_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    relative = path.relative_to(project)
    linked_descriptor = os.dup(parent_descriptor)
    try:
        for component in relative.parts:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=linked_descriptor,
            )
            os.close(linked_descriptor)
            linked_descriptor = next_descriptor
        linked = os.fstat(linked_descriptor)
    finally:
        os.close(linked_descriptor)
    if not stat.S_ISDIR(status.st_mode) or (status.st_dev, status.st_ino) != (
        linked.st_dev,
        linked.st_ino,
    ):
        raise SystemExit("teacher setup lock authority changed")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("another Apery setup is active") from None
PY
python3 - "$SETUP_CACHE_FD" "$SETUP_LOCK_FD" <<'PY'
import os
import stat
import sys


def measure(parent, name, nodes, total):
    status = os.stat(name, dir_fd=parent, follow_symlinks=False)
    nodes += 1
    if nodes > 65_536:
        raise SystemExit("teacher setup quarantine node quota is exhausted")
    if stat.S_ISDIR(status.st_mode):
        child = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        try:
            for entry in os.listdir(child):
                nodes, total = measure(child, entry, nodes, total)
        finally:
            os.close(child)
    else:
        total += max(0, status.st_size)
    if total > 4 * 1024 * 1024 * 1024:
        raise SystemExit("teacher setup quarantine byte quota is exhausted")
    return nodes, total


for raw in sys.argv[1:]:
    descriptor = int(raw)
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        raise SystemExit("teacher setup quarantine parent is not a directory")
    names = [name for name in os.listdir(descriptor) if name.startswith(".setup-retired.")]
    if len(names) >= 32:
        raise SystemExit("teacher setup quarantine entry quota is exhausted")
    nodes = 0
    total = 0
    for name in names:
        nodes, total = measure(descriptor, name, nodes, total)
PY
[ ! -L "$ARCHIVE" ] || fail "cached archive must not be a symlink"

if [ ! -f "$ARCHIVE" ]; then
    DOWNLOAD_TEMP=$(mktemp "$CACHE_DIR/.apery_2.0.0.zip.XXXXXX")
    exec 10< "$DOWNLOAD_TEMP"
    DOWNLOAD_TEMP_FD="10"
    curl --fail --location --proto '=https' --tlsv1.2 --retry 3 \
        --output "$DOWNLOAD_TEMP" "$RELEASE_URL"
    python3 - "$DOWNLOAD_TEMP" "$ARCHIVE" "$ARCHIVE_SHA256" "$ARCHIVE_SIZE" <<'PY'
import ctypes
import hashlib
import os
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_hash = sys.argv[3]
expected_size = int(sys.argv[4])
if source.parent != destination.parent:
    raise SystemExit("archive temporary and cache destination must share one authority")
directory = os.open(
    source.parent,
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
descriptor = -1
try:
    descriptor = os.open(
        source.name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory,
    )
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit("downloaded archive temporary is not a private regular file")
    digest = hashlib.sha256()
    observed = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        observed += len(chunk)
        if observed > expected_size:
            raise SystemExit("downloaded Apery archive exceeds its pinned size")
        digest.update(chunk)
    after = os.fstat(descriptor)
    linked = os.stat(source.name, dir_fd=directory, follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(linked):
        raise SystemExit("downloaded archive changed during retained-descriptor verification")
    if observed != expected_size or digest.hexdigest() != expected_hash:
        raise SystemExit("downloaded Apery archive failed pinned size/SHA-256 verification")
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameatx_np
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(
        directory,
        os.fsencode(source.name),
        directory,
        os.fsencode(destination.name),
        4,
    ):
        error = ctypes.get_errno()
        if error == 17:
            raise SystemExit(
                "refusing to overwrite a concurrently published cached archive"
            ) from None
        raise OSError(error, os.strerror(error), destination.name)
    published = os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
    if (published.st_dev, published.st_ino, published.st_mode, published.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
    ):
        raise SystemExit("cached archive publication identity changed")
    os.fsync(directory)
finally:
    if descriptor >= 0:
        os.close(descriptor)
    os.close(directory)
PY
    DOWNLOAD_TEMP=""
    exec 10<&-
    DOWNLOAD_TEMP_FD="-1"
fi

# FD 9 remains open for the complete audit, extraction, build verification, and
# manifest publication. Every ZipFile view is a duplicate of this same file
# description; the cache pathname is only provenance and is re-bound below.
exec 9< "$ARCHIVE"
python3 - "9" "$ARCHIVE" "$ARCHIVE_SHA256" "$ARCHIVE_SIZE" "$ARCHIVE_ENTRIES" \
    "$ARCHIVE_UNCOMPRESSED_SIZE" "$ARCHIVE_TREE_SHA256" "$ARCHIVE_FILES" <<'PY'
import hashlib
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

descriptor = int(sys.argv[1])
path = Path(sys.argv[2])
expected_hash = sys.argv[3]
expected_size = int(sys.argv[4])
expected_entries = int(sys.argv[5])
expected_uncompressed = int(sys.argv[6])
expected_tree_hash = sys.argv[7]
expected_files = int(sys.argv[8])
initial = os.fstat(descriptor)
digest = hashlib.sha256()
size = 0
offset = 0
while chunk := os.pread(descriptor, 1024 * 1024, offset):
    offset += len(chunk)
    size += len(chunk)
    digest.update(chunk)
if size != expected_size or digest.hexdigest() != expected_hash:
    raise SystemExit("cached Apery archive failed pinned size/SHA-256 verification")
linked = path.lstat()
if stat.S_ISLNK(linked.st_mode) or (linked.st_dev, linked.st_ino) != (
    initial.st_dev,
    initial.st_ino,
):
    raise SystemExit("cached archive path differs from the retained verified descriptor")
with os.fdopen(os.dup(descriptor), "rb") as retained, zipfile.ZipFile(retained) as archive:
    entries = archive.infolist()
    if len(entries) != expected_entries:
        raise SystemExit(f"archive entry count drifted: {len(entries)}")
    if sum(entry.file_size for entry in entries) != expected_uncompressed:
        raise SystemExit("archive uncompressed size drifted")
    seen = set()
    files = []
    for entry in entries:
        member = PurePosixPath(entry.filename)
        if (
            member.is_absolute()
            or not member.parts
            or member.parts[0] != "apery_2.0.0"
            or any(part in {"", ".", ".."} for part in member.parts)
            or "\\" in entry.filename
        ):
            raise SystemExit(f"unsafe archive path: {entry.filename!r}")
        normalized = member.as_posix()
        if normalized in seen:
            raise SystemExit(f"duplicate archive path: {normalized}")
        seen.add(normalized)
        mode = entry.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise SystemExit(f"unsupported archive entry type: {normalized}")
        if not entry.is_dir():
            files.append(entry)
    if len(files) != expected_files:
        raise SystemExit(f"archive file count drifted: {len(files)}")
    if any(
        PurePosixPath(entry.filename).name == "build.rs"
        or entry.filename.endswith("/.cargo/config.toml")
        for entry in files
    ):
        raise SystemExit("archive gained an unexpected Cargo build hook or config")
    pinned_control_hashes = {
        "apery_2.0.0/.cargo/config": "06391c6e9267967cedc1a2d561cb55d7a100ae9d851dee3bbff2346aa56d6dde",
        "apery_2.0.0/Cargo.toml": "34b0067d7f4b2d486e1fcccdbb57df26b7907d1003b4c0f0d45ee6a70a0b93ee",
        "apery_2.0.0/Cargo.lock": "c538702e6c33c4644495add0f97aa4b4de0520c902d85b57d35969ee1534aa61",
        "apery_2.0.0/rust-toolchain": "98602522dec9253a4c8d125ea13f805ceeedf9d73f49b089a16afb85eb4b56c4",
    }
    tree = hashlib.sha256()
    for entry in sorted(files, key=lambda item: item.filename):
        content_hash = hashlib.sha256()
        with archive.open(entry) as source:
            while chunk := source.read(1024 * 1024):
                content_hash.update(chunk)
        normalized = PurePosixPath(entry.filename).as_posix()
        if normalized in pinned_control_hashes and (
            content_hash.hexdigest() != pinned_control_hashes[normalized]
        ):
            raise SystemExit(f"pinned Cargo control file drifted: {normalized}")
        encoded = normalized.encode("utf-8")
        tree.update(len(encoded).to_bytes(4, "big"))
        tree.update(encoded)
        tree.update(entry.file_size.to_bytes(8, "big"))
        tree.update(content_hash.digest())
    if tree.hexdigest() != expected_tree_hash:
        raise SystemExit("archive content tree drifted")
final = os.fstat(descriptor)
if (initial.st_dev, initial.st_ino, initial.st_mode, initial.st_size,
    initial.st_mtime_ns, initial.st_ctime_ns) != (
    final.st_dev, final.st_ino, final.st_mode, final.st_size,
    final.st_mtime_ns, final.st_ctime_ns
):
    raise SystemExit("retained archive descriptor changed during verification")
PY

EXTRACT_STAGE=$(mktemp -d "$BUILD_ROOT/.extract.XXXXXX")
exec 11< "$EXTRACT_STAGE"
EXTRACT_STAGE_FD="11"
SOURCE_DIR="$EXTRACT_STAGE/apery_2.0.0"
CARGO_HOME_LOCAL="$EXTRACT_STAGE/cargo-home"
CARGO_DRIVER_CWD="$EXTRACT_STAGE/cargo-driver"
TEACHER_PRIVATE_HOME="$EXTRACT_STAGE/home"
TEACHER_TEMP_DIR="$EXTRACT_STAGE/tmp"
CARGO_ANCESTOR_GUARD="$EXTRACT_STAGE/cargo-ancestor-guard.json"
CARGO_CONFIG_PIN="$EXTRACT_STAGE/cargo-config.pinned"
CARGO_SANDBOX_PROFILE="$EXTRACT_STAGE/cargo-build.sb"
SOURCE_MODE_GUARD="$EXTRACT_STAGE/source-mode-guard.json"
mkdir "$CARGO_HOME_LOCAL" "$CARGO_DRIVER_CWD" "$TEACHER_PRIVATE_HOME" "$TEACHER_TEMP_DIR"
python3 - "9" "$EXTRACT_STAGE" <<'PY'
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

archive_descriptor = int(sys.argv[1])
stage = Path(sys.argv[2]).resolve(strict=True)
with os.fdopen(os.dup(archive_descriptor), "rb") as retained, zipfile.ZipFile(
    retained
) as archive:
    for entry in archive.infolist():
        relative = PurePosixPath(entry.filename)
        target = stage.joinpath(*relative.parts)
        target.resolve(strict=False).relative_to(stage)
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(entry, "r") as source, target.open("xb") as destination:
            copied = 0
            while chunk := source.read(1024 * 1024):
                copied += len(chunk)
                if copied > entry.file_size:
                    raise SystemExit(f"archive member expanded beyond declared size: {entry.filename}")
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if copied != entry.file_size:
            raise SystemExit(f"archive member size mismatch: {entry.filename}")
        mode = entry.external_attr >> 16
        target.chmod(0o755 if mode & 0o111 else 0o644)
PY
[ ! -L "$SOURCE_DIR" ] || fail "extracted source must not be a symlink"

# Refuse anything except the complete, freshly extracted official tree.
python3 - "9" "$SOURCE_DIR" <<'PY'
import hashlib
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

archive_descriptor = int(sys.argv[1])
source_root = Path(sys.argv[2]).resolve(strict=True)
with os.fdopen(os.dup(archive_descriptor), "rb") as retained, zipfile.ZipFile(
    retained
) as archive:
    expected_files = set()
    expected_directories = set()
    for entry in archive.infolist():
        relative = PurePosixPath(entry.filename)
        if len(relative.parts) == 1:
            continue
        if entry.is_dir():
            expected_directories.add(PurePosixPath(*relative.parts[1:]).as_posix())
            continue
        expected_files.add(PurePosixPath(*relative.parts[1:]).as_posix())
        target = source_root.joinpath(*relative.parts[1:])
        if target.is_symlink() or not target.is_file() or target.stat().st_size != entry.file_size:
            raise SystemExit(f"extracted source drifted: {relative.as_posix()}")
        expected = hashlib.sha256()
        with archive.open(entry) as archived:
            while chunk := archived.read(1024 * 1024):
                expected.update(chunk)
        actual = hashlib.sha256()
        with target.open("rb") as stored:
            while chunk := stored.read(1024 * 1024):
                actual.update(chunk)
        if actual.digest() != expected.digest():
            raise SystemExit(f"extracted source hash drifted: {relative.as_posix()}")
actual_files = set()
actual_directories = set()
for directory, names, filenames in os.walk(source_root, topdown=True, followlinks=False):
    directory_path = Path(directory)
    for name in names:
        child = directory_path / name
        if child.is_symlink():
            raise SystemExit(f"extracted source contains symlink directory: {name}")
        actual_directories.add(child.relative_to(source_root).as_posix())
    for name in filenames:
        path = directory_path / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"extracted source contains unsafe file: {path}")
        actual_files.add(path.relative_to(source_root).as_posix())
if actual_files != expected_files or actual_directories != expected_directories:
    raise SystemExit(
        "extracted source tree set drifted: "
        f"missing={sorted(expected_files - actual_files)}, "
        f"unexpected={sorted(actual_files - expected_files)}, "
        f"missing_directories={sorted(expected_directories - actual_directories)}, "
        f"unexpected_directories={sorted(actual_directories - expected_directories)}"
    )
PY

# Cargo has no flag that disables discovered ancestor configuration. Run both
# dependency resolution and compilation in a macOS sandbox that permits metadata
# probes but denies reading any dynamically appeared .cargo/config bytes. The one
# official pinned config is copied to a non-discovery pathname and passed explicitly.
python3 - "$SOURCE_DIR/.cargo/config" "$CARGO_CONFIG_PIN" "$CARGO_SANDBOX_PROFILE" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
profile = Path(sys.argv[3])
raw = source.read_bytes()
if hashlib.sha256(raw).hexdigest() != "06391c6e9267967cedc1a2d561cb55d7a100ae9d851dee3bbff2346aa56d6dde":
    raise SystemExit("official Cargo config drifted before sandboxing")
with destination.open("xb") as output:
    output.write(raw)
    output.flush()
    os.fsync(output.fileno())
sandbox = b'''(version 1)\n(allow default)\n(deny file-read-data\n  (regex #"/[.]cargo/config$")\n  (regex #"/[.]cargo/config[.]toml$")\n  (regex #"/cargo-home/config$")\n  (regex #"/cargo-home/config[.]toml$"))\n'''
with profile.open("xb") as output:
    output.write(sandbox)
    output.flush()
    os.fsync(output.fileno())
PY

python3 - "$CARGO_DRIVER_CWD" "$PROJECT_ROOT" "$CARGO_ANCESTOR_GUARD" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

driver = Path(sys.argv[1]).resolve(strict=True)
project = Path(sys.argv[2]).resolve(strict=True)
guard = Path(sys.argv[3])
rows = []
cursor = driver
while True:
    status = cursor.lstat()
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise SystemExit(f"Cargo ancestor is not a real directory: {cursor}")
    cargo_dir = cursor / ".cargo"
    cargo_status = None
    try:
        cargo_status = cargo_dir.lstat()
    except FileNotFoundError:
        pass
    if cargo_status is not None and (
        not stat.S_ISDIR(cargo_status.st_mode) or stat.S_ISLNK(cargo_status.st_mode)
    ):
        raise SystemExit(f"ambient parent Cargo authority is unsafe: {cargo_dir}")
    for name in ("config", "config.toml"):
        try:
            config_status = (cargo_dir / name).lstat()
        except FileNotFoundError:
            continue
        raise SystemExit(
            f"ambient parent Cargo config is not allowed: {cargo_dir / name} "
            f"(mode={config_status.st_mode:o})"
        )
    rows.append(
        {
            "path": str(cursor),
            "identity": [status.st_dev, status.st_ino, status.st_mode],
            "cargoIdentity": None
            if cargo_status is None
            else [cargo_status.st_dev, cargo_status.st_ino, cargo_status.st_mode],
        }
    )
    if cursor == project:
        break
    if cursor.parent == cursor:
        raise SystemExit("Cargo driver is not nested below the project root")
    cursor = cursor.parent
with guard.open("xb") as output:
    output.write(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    output.flush()
    os.fsync(output.fileno())
PY

python3 - "$SOURCE_DIR" "$ENGINE_LICENSE_SHA256" "$EVAL_LICENSE_SHA256" \
    "$EVAL_README_SHA256" "$KKP_SHA256" "$KPP_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "LICENSE": sys.argv[2],
    "eval/20190617/LICENSE-MIT": sys.argv[3],
    "eval/20190617/README.md": sys.argv[4],
    "eval/20190617/KKP.bin": sys.argv[5],
    "eval/20190617/KPP.bin": sys.argv[6],
}
for relative, expected_hash in expected.items():
    path = root / relative
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise SystemExit(f"pinned file hash mismatch: {relative}")
PY

# An independent Cargo package nested below this repository would otherwise inherit the
# production workspace. Add a temporary empty workspace marker to the upstream manifest, apply
# one pinned official-crate compatibility update for current Rust, then restore the exact archive
# manifest and lockfile bytes after the build.
SOURCE_MANIFEST_BACKUP="$EXTRACT_STAGE/.apery-Cargo.toml.original"
SOURCE_LOCK_BACKUP="$EXTRACT_STAGE/.apery-Cargo.lock.original"
exec 16< "$SOURCE_DIR/Cargo.toml"
exec 17< "$SOURCE_DIR/Cargo.lock"
SOURCE_MANIFEST_TARGET_FD="16"
SOURCE_LOCK_TARGET_FD="17"
python3 - "$SOURCE_DIR/Cargo.toml" "$SOURCE_MANIFEST_BACKUP" \
    "$SOURCE_DIR/Cargo.lock" "$SOURCE_LOCK_BACKUP" <<'PY'
import os
import sys
from pathlib import Path

if len(sys.argv) != 5:
    raise SystemExit("internal manifest-backup argument mismatch")
manifest = Path(sys.argv[1])
backup = Path(sys.argv[2])
lockfile = Path(sys.argv[3])
lock_backup = Path(sys.argv[4])
original = manifest.read_bytes()
if b"[workspace]" in original:
    raise SystemExit("unexpected upstream workspace declaration")
with backup.open("xb") as output:
    output.write(original)
    output.flush()
    os.fsync(output.fileno())
with manifest.open("wb") as output:
    output.write(original)
    if not original.endswith(b"\n"):
        output.write(b"\n")
    output.write(b"\n[workspace]\n")
    output.flush()
    os.fsync(output.fileno())
with lock_backup.open("xb") as output:
    output.write(lockfile.read_bytes())
    output.flush()
    os.fsync(output.fileno())
PY
exec 12< "$SOURCE_MANIFEST_BACKUP"
exec 13< "$SOURCE_LOCK_BACKUP"
SOURCE_MANIFEST_BACKUP_FD="12"
SOURCE_LOCK_BACKUP_FD="13"
if [ -f "$COMPAT_LOCK" ]; then
    [ ! -L "$COMPAT_LOCK" ] || fail "compatibility lock must not be a symlink"
    cp "$COMPAT_LOCK" "$SOURCE_DIR/Cargo.lock"
else
    (
        cd "$CARGO_DRIVER_CWD"
        env -i PATH="$TEACHER_TOOL_PATH" HOME="$TEACHER_PRIVATE_HOME" \
            CARGO_HOME="$CARGO_HOME_LOCAL" RUSTUP_HOME="$RUSTUP_HOME_PIN" \
            TMPDIR="$TEACHER_TEMP_DIR" LC_ALL=C LANG=C \
            /usr/bin/sandbox-exec -f "$CARGO_SANDBOX_PROFILE" \
            "$CARGO_BIN" --config "$CARGO_CONFIG_PIN" update \
            --manifest-path "$SOURCE_DIR/Cargo.toml" \
            -p num-bigint --precise "$NUM_BIGINT_VERSION"
    )
fi
python3 - "$SOURCE_DIR/Cargo.lock" "$NUM_BIGINT_VERSION" \
    "$NUM_BIGINT_CRATE_SHA256" "$COMPAT_LOCK_SHA256" "$COMPAT_LOCK_SIZE" <<'PY'
import hashlib
import sys
import tomllib
from pathlib import Path

lockfile = Path(sys.argv[1])
expected_version = sys.argv[2]
expected_checksum = sys.argv[3]
expected_lock_hash = sys.argv[4]
expected_lock_size = int(sys.argv[5])
raw = lockfile.read_bytes()
if len(raw) != expected_lock_size or hashlib.sha256(raw).hexdigest() != expected_lock_hash:
    raise SystemExit("compatibility lock failed pinned size/SHA-256 verification")
payload = tomllib.loads(lockfile.read_text(encoding="utf-8"))
matches = [item for item in payload.get("package", []) if item.get("name") == "num-bigint"]
if len(matches) != 1:
    raise SystemExit("compatibility lock must contain exactly one num-bigint package")
package = matches[0]
if package.get("version") != expected_version or package.get("checksum") != expected_checksum:
    raise SystemExit("compatibility lock num-bigint version/checksum mismatch")
PY
if [ ! -f "$COMPAT_LOCK" ]; then
    python3 - "$SOURCE_DIR/Cargo.lock" "$COMPAT_LOCK" <<'PY'
import ctypes
import os
import stat
import sys
from secrets import token_hex
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
source_parent = os.open(
    source.parent,
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
destination_parent = os.open(
    destination.parent,
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
source_descriptor = -1
destination_descriptor = -1
temporary_name = f".{destination.name}.{token_hex(12)}.pending"
temporary_status = None


def rename_noreplace(source_directory, source_name, destination_directory, destination_name):
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameatx_np
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(
        source_directory,
        os.fsencode(source_name),
        destination_directory,
        os.fsencode(destination_name),
        4,
    ):
        error = ctypes.get_errno()
        if error == 17:
            raise FileExistsError(error, os.strerror(error), destination_name)
        if error == 2:
            raise FileNotFoundError(error, os.strerror(error), source_name)
        raise OSError(error, os.strerror(error), source_name)


def identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


try:
    source_descriptor = os.open(
        source.name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=source_parent,
    )
    before = os.fstat(source_descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit("compatibility lock source is not a regular file")
    destination_descriptor = os.open(
        temporary_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=destination_parent,
    )
    temporary_status = os.fstat(destination_descriptor)
    try:
        copied = 0
        while chunk := os.read(source_descriptor, 64 * 1024):
            copied += len(chunk)
            written = 0
            while written < len(chunk):
                count = os.write(destination_descriptor, chunk[written:])
                if count <= 0:
                    raise SystemExit("could not write complete compatibility lock")
                written += count
        os.fsync(destination_descriptor)
    finally:
        temporary_status = os.fstat(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = -1
    after = os.fstat(source_descriptor)
    linked_source = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
    if copied != before.st_size or identity(after) != identity(before) or identity(
        linked_source
    ) != identity(after):
        raise SystemExit("compatibility lock changed while it was copied")
    try:
        rename_noreplace(
            destination_parent,
            temporary_name,
            destination_parent,
            destination.name,
        )
    except FileExistsError:
        raise SystemExit("refusing to overwrite a concurrently published compatibility lock") from None
    assert temporary_status is not None
    published = os.stat(destination.name, dir_fd=destination_parent, follow_symlinks=False)
    if (published.st_dev, published.st_ino, published.st_mode, published.st_size) != (
        temporary_status.st_dev,
        temporary_status.st_ino,
        temporary_status.st_mode,
        temporary_status.st_size,
    ):
        raise SystemExit("compatibility lock publication identity changed")
    temporary_name = ""
    os.fsync(destination_parent)
finally:
    if source_descriptor >= 0:
        os.close(source_descriptor)
    if destination_descriptor >= 0:
        os.close(destination_descriptor)
    if temporary_name and temporary_status is not None:
        retained_temporary = -1
        retired_name = (
            f".setup-retired.{temporary_status.st_dev:x}."
            f"{temporary_status.st_ino:x}.{token_hex(12)}"
        )
        try:
            retained_temporary = os.open(
                temporary_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=destination_parent,
            )
            opened = os.fstat(retained_temporary)
            if (opened.st_dev, opened.st_ino) != (
                temporary_status.st_dev,
                temporary_status.st_ino,
            ):
                raise RuntimeError(
                    "compatibility-lock temporary was replaced; foreign bytes were not deleted"
                )
            rename_noreplace(
                destination_parent,
                temporary_name,
                destination_parent,
                retired_name,
            )
        except FileNotFoundError:
            pass
        else:
            moved = os.stat(retired_name, dir_fd=destination_parent, follow_symlinks=False)
            if (moved.st_dev, moved.st_ino) != (
                temporary_status.st_dev,
                temporary_status.st_ino,
            ):
                try:
                    rename_noreplace(
                        destination_parent,
                        retired_name,
                        destination_parent,
                        temporary_name,
                    )
                except OSError:
                    pass
                raise RuntimeError(
                    "compatibility-lock temporary changed; foreign bytes were not deleted"
                )
            if moved.st_uid != os.getuid() or moved.st_nlink != 1:
                raise RuntimeError("compatibility-lock temporary is not proven-owned")
            os.unlink(retired_name, dir_fd=destination_parent)
            if os.fstat(retained_temporary).st_nlink != 0:
                raise RuntimeError("compatibility-lock temporary link survived cleanup")
            os.fsync(destination_parent)
        finally:
            if retained_temporary >= 0:
                os.close(retained_temporary)
    os.close(source_parent)
    os.close(destination_parent)
PY
fi
# Cargo receives an immutable, read-only source snapshot. Its target lives outside
# that tree, so no build step needs to mutate an input after the final audit. On
# Darwin UF_IMMUTABLE|UF_NOUNLINK is mandatory here; active same-UID flag removal
# is outside the local setup threat model and is detected by the post-build guard.
python3 - "$SOURCE_DIR" "$CARGO_CONFIG_PIN" "$CARGO_SANDBOX_PROFILE" \
    "$SOURCE_MODE_GUARD" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1])
cargo_config = Path(sys.argv[2])
sandbox_profile = Path(sys.argv[3])
guard = Path(sys.argv[4])
seal = getattr(stat, "UF_IMMUTABLE", 0) | getattr(stat, "UF_NOUNLINK", 0)
if not seal or not hasattr(os, "chflags"):
    raise SystemExit("Darwin immutable launch/build flags are unavailable")
paths = [*source.rglob("*"), source, cargo_config, sandbox_profile]
rows = []
for path in paths:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not (
        stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)
    ):
        raise SystemExit(f"cannot seal unsafe build input: {path}")
    rows.append(
        {
            "path": str(path),
            "identity": [
                status.st_dev,
                status.st_ino,
                status.st_mode,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            ],
            "mode": stat.S_IMODE(status.st_mode),
        }
    )
with guard.open("xb") as output:
    output.write(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    output.flush()
    os.fsync(output.fileno())
for row in sorted(rows, key=lambda value: len(Path(value["path"]).parts), reverse=True):
    path = Path(row["path"])
    status = path.lstat()
    path.chmod(0o555 if stat.S_ISDIR(status.st_mode) or status.st_mode & 0o111 else 0o444)
    os.chflags(path, int(getattr(path.lstat(), "st_flags", 0)) | seal, follow_symlinks=False)
    if int(getattr(path.lstat(), "st_flags", 0)) & seal != seal:
        raise SystemExit(f"could not seal build input: {path}")
PY
(
    cd "$CARGO_DRIVER_CWD"
    env -i PATH="$TEACHER_TOOL_PATH" HOME="$TEACHER_PRIVATE_HOME" \
        CARGO_HOME="$CARGO_HOME_LOCAL" RUSTUP_HOME="$RUSTUP_HOME_PIN" \
        TMPDIR="$TEACHER_TEMP_DIR" LC_ALL=C LANG=C \
        CARGO_INCREMENTAL=0 SOURCE_DATE_EPOCH=0 \
        /usr/bin/sandbox-exec -f "$CARGO_SANDBOX_PROFILE" \
        "$CARGO_BIN" --config "$CARGO_CONFIG_PIN" build \
        --manifest-path "$SOURCE_DIR/Cargo.toml" \
        --target-dir "$EXTRACT_STAGE/build-target" --release --locked
)
python3 - "$CARGO_ANCESTOR_GUARD" <<'PY'
import json
import stat
import sys
from pathlib import Path

guard = Path(sys.argv[1])
rows = json.loads(guard.read_text(encoding="utf-8"))
if not isinstance(rows, list) or not rows:
    raise SystemExit("Cargo ancestor guard is malformed")
for row in rows:
    if not isinstance(row, dict) or set(row) != {"path", "identity", "cargoIdentity"}:
        raise SystemExit("Cargo ancestor guard row is malformed")
    path = Path(row["path"])
    status = path.lstat()
    if [status.st_dev, status.st_ino, status.st_mode] != row["identity"]:
        raise SystemExit(f"Cargo ancestor changed during the build: {path}")
    cargo_dir = path / ".cargo"
    try:
        cargo_status = cargo_dir.lstat()
    except FileNotFoundError:
        cargo_identity = None
    else:
        if not stat.S_ISDIR(cargo_status.st_mode) or stat.S_ISLNK(cargo_status.st_mode):
            raise SystemExit(f"Cargo ancestor authority became unsafe: {cargo_dir}")
        cargo_identity = [cargo_status.st_dev, cargo_status.st_ino, cargo_status.st_mode]
    if cargo_identity != row["cargoIdentity"]:
        raise SystemExit(f"Cargo ancestor config authority changed during the build: {cargo_dir}")
    for name in ("config", "config.toml"):
        try:
            (cargo_dir / name).lstat()
        except FileNotFoundError:
            continue
        raise SystemExit(f"ambient Cargo config appeared during the build: {cargo_dir / name}")
PY
python3 - "$SOURCE_MODE_GUARD" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

guard = Path(sys.argv[1])
rows = json.loads(guard.read_text(encoding="utf-8"))
seal = getattr(stat, "UF_IMMUTABLE", 0) | getattr(stat, "UF_NOUNLINK", 0)
if not isinstance(rows, list) or not rows:
    raise SystemExit("source mode guard is malformed")
# Directories are cleared top-down so their children can be restored.
for row in sorted(rows, key=lambda value: len(Path(value["path"]).parts)):
    if not isinstance(row, dict) or set(row) != {"path", "identity", "mode"}:
        raise SystemExit("source mode guard row is malformed")
    path = Path(row["path"])
    status = path.lstat()
    expected = row["identity"]
    observed = [
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    ]
    # chmod performed while sealing changes only mode; all other identity fields
    # must remain exact and the complete seal must still be present.
    if (
        observed[:2] != expected[:2]
        or observed[3:5] != expected[3:5]
        or int(getattr(status, "st_flags", 0)) & seal != seal
    ):
        raise SystemExit(f"sealed build input changed during Cargo execution: {path}")
    os.chflags(path, int(getattr(status, "st_flags", 0)) & ~seal, follow_symlinks=False)
    path.chmod(int(row["mode"]))
# Keep the guard in the private extraction stage. Cleanup retires that complete
# directory by inode, avoiding any stat-then-unlink window for a final entry.
PY
SOURCE_MODE_GUARD=""
python3 - "$SOURCE_MANIFEST_BACKUP" "$SOURCE_DIR/Cargo.toml" \
    "$SOURCE_LOCK_BACKUP" "$SOURCE_DIR/Cargo.lock" \
    "$SOURCE_MANIFEST_BACKUP_FD" "$SOURCE_MANIFEST_TARGET_FD" \
    "$SOURCE_LOCK_BACKUP_FD" "$SOURCE_LOCK_TARGET_FD" \
    "$UPSTREAM_MANIFEST_SHA256" "$UPSTREAM_LOCK_SHA256" <<'PY'
import ctypes
import hashlib
import os
import stat
import sys
from pathlib import Path

if len(sys.argv) != 11:
    raise SystemExit("internal source-restore argument mismatch")
manifest_backup = Path(sys.argv[1])
manifest = Path(sys.argv[2])
lock_backup = Path(sys.argv[3])
lockfile = Path(sys.argv[4])


def descriptor_sha256(descriptor):
    digest = hashlib.sha256()
    status = os.fstat(descriptor)
    offset = 0
    while offset < status.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, status.st_size - offset), offset)
        if not chunk:
            raise RuntimeError("source restore backup ended early")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def same_inode(first, second):
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def restore_exchange(source, destination, source_fd, target_fd, expected_sha256):
    source_parent = os.open(
        source.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_parent = os.open(
        destination.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        source_expected = os.fstat(source_fd)
        target_expected = os.fstat(target_fd)
        source_linked = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
        target_linked = os.stat(
            destination.name,
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(source_expected.st_mode)
            or not stat.S_ISREG(target_expected.st_mode)
            or not same_inode(source_expected, source_linked)
            or not same_inode(target_expected, target_linked)
            or source_expected.st_uid != os.getuid()
            or target_expected.st_uid != os.getuid()
            or source_expected.st_nlink != 1
            or target_expected.st_nlink != 1
        ):
            raise RuntimeError("source restore retained identity changed")
        if descriptor_sha256(source_fd) != expected_sha256:
            raise RuntimeError("source restore backup hash changed")
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(
            source_parent,
            os.fsencode(source.name),
            destination_parent,
            os.fsencode(destination.name),
            2,
        ):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination.name)
        restored = os.stat(destination.name, dir_fd=destination_parent, follow_symlinks=False)
        displaced = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
        if not same_inode(restored, source_expected) or not same_inode(
            displaced, target_expected
        ):
            raise RuntimeError("restored source exchange identity changed")
        os.unlink(source.name, dir_fd=source_parent)
        if os.fstat(target_fd).st_nlink != 0:
            raise RuntimeError("modified source target link survived restoration")
        os.fsync(source_parent)
        os.fsync(destination_parent)
    finally:
        os.close(source_parent)
        os.close(destination_parent)


restore_exchange(
    manifest_backup,
    manifest,
    int(sys.argv[5]),
    int(sys.argv[6]),
    sys.argv[9],
)
restore_exchange(
    lock_backup,
    lockfile,
    int(sys.argv[7]),
    int(sys.argv[8]),
    sys.argv[10],
)
PY
SOURCE_MANIFEST_BACKUP=""
SOURCE_LOCK_BACKUP=""
exec 12<&-
exec 13<&-
exec 16<&-
exec 17<&-
SOURCE_MANIFEST_BACKUP_FD="-1"
SOURCE_LOCK_BACKUP_FD="-1"
SOURCE_MANIFEST_TARGET_FD="-1"
SOURCE_LOCK_TARGET_FD="-1"
BINARY="$EXTRACT_STAGE/build-target/release/apery"
[ -x "$BINARY" ] || fail "cargo did not produce target/release/apery"
python3 - "9" "$SOURCE_DIR" "$BINARY" "$EXPECTED_SOURCE_TREE_SHA256" \
    "$EXPECTED_SOURCE_FILES" "$EXPECTED_SOURCE_SIZE" "$EXPECTED_BINARY_SHA256" \
    "$EXPECTED_BINARY_SIZE" "$ARCHIVE_TREE_SHA256" "$ARCHIVE_FILES" \
    "$ARCHIVE_UNCOMPRESSED_SIZE" <<'PY'
import hashlib
import os
import sys
import zipfile
from pathlib import Path

archive_descriptor = int(sys.argv[1])
root = Path(sys.argv[2]).resolve(strict=True)
binary = Path(sys.argv[3]).resolve(strict=True)
expected_tree = sys.argv[4]
expected_files = int(sys.argv[5])
expected_source_size = int(sys.argv[6])
expected_binary = sys.argv[7]
expected_binary_size = int(sys.argv[8])
expected_official_tree = sys.argv[9]
expected_official_files = int(sys.argv[10])
expected_official_size = int(sys.argv[11])


def file_digest(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.digest(), size


# Re-prove that every non-build file is still the exact official archive byte.
actual_official_files = set()
actual_official_directories = set()
for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
    directory_path = Path(directory)
    for name in names:
        child = directory_path / name
        if child.is_symlink():
            raise SystemExit("post-build official tree contains a symlink directory")
    if directory_path == root:
        names[:] = [name for name in names if name != "target"]
    for name in names:
        actual_official_directories.add((directory_path / name).relative_to(root).as_posix())
    for name in filenames:
        path = directory_path / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit("post-build official tree contains an unsafe file")
        actual_official_files.add(path.relative_to(root).as_posix())
with os.fdopen(os.dup(archive_descriptor), "rb") as retained, zipfile.ZipFile(
    retained
) as archive:
    expected_official_paths = {
        "/".join(entry.filename.split("/")[1:]): entry
        for entry in archive.infolist()
        if not entry.is_dir()
    }
    expected_official_directories = {
        "/".join(entry.filename.rstrip("/").split("/")[1:])
        for entry in archive.infolist()
        if entry.is_dir()
    }
    if (
        actual_official_files != set(expected_official_paths)
        or actual_official_directories != expected_official_directories
    ):
        raise SystemExit("post-build official archive tree set drifted")
    official_tree = hashlib.sha256()
    official_size = 0
    for relative, entry in sorted(expected_official_paths.items(), key=lambda item: item[1].filename):
        installed_digest, installed_size = file_digest(root / relative)
        archived_digest = hashlib.sha256()
        with archive.open(entry) as archived:
            while chunk := archived.read(1024 * 1024):
                archived_digest.update(chunk)
        if installed_size != entry.file_size or installed_digest != archived_digest.digest():
            raise SystemExit(f"post-build official file drifted: {relative}")
        encoded = entry.filename.encode("utf-8")
        official_tree.update(len(encoded).to_bytes(4, "big"))
        official_tree.update(encoded)
        official_tree.update(installed_size.to_bytes(8, "big"))
        official_tree.update(installed_digest)
        official_size += installed_size
if (
    official_tree.hexdigest() != expected_official_tree
    or len(actual_official_files) != expected_official_files
    or official_size != expected_official_size
):
    raise SystemExit("post-build complete official tree drifted")


tree = hashlib.sha256()
files = 0
source_size = 0
for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
    directory_path = Path(directory)
    for name in names:
        if (directory_path / name).is_symlink():
            raise SystemExit("post-build source tree contains a symlink directory")
    names[:] = [name for name in names if name not in {"target", "eval"}]
    for name in sorted(filenames):
        path = directory_path / name
        relative = path.relative_to(root)
        if relative.name == "apery_2.0.0.exe":
            continue
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"post-build source tree contains unsafe file: {relative}")
        digest, size = file_digest(path)
        encoded = relative.as_posix().encode("utf-8")
        tree.update(len(encoded).to_bytes(4, "big"))
        tree.update(encoded)
        tree.update(size.to_bytes(8, "big"))
        tree.update(digest)
        files += 1
        source_size += size
if tree.hexdigest() != expected_tree or files != expected_files or source_size != expected_source_size:
    raise SystemExit("post-build official source tree drifted")
binary_digest, binary_size = file_digest(binary)
if binary_digest.hex() != expected_binary or binary_size != expected_binary_size:
    raise SystemExit("built teacher binary differs from the pinned reproducible build")
PY

# Publish the already verified build output into the staged install tree from one
# retained source descriptor; Cargo never writes inside the immutable source tree.
python3 - "$BINARY" "$SOURCE_DIR/target/release/apery" \
    "$EXPECTED_BINARY_SHA256" "$EXPECTED_BINARY_SIZE" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_hash = sys.argv[3]
expected_size = int(sys.argv[4])
destination.parent.mkdir(mode=0o700, parents=True)
source_descriptor = os.open(
    source,
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
)
destination_descriptor = os.open(
    destination,
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0),
    0o500,
)
try:
    initial = os.fstat(source_descriptor)
    if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
        raise SystemExit("verified teacher build output is no longer a single-link file")
    digest = hashlib.sha256()
    copied = 0
    while chunk := os.read(source_descriptor, 1024 * 1024):
        copied += len(chunk)
        digest.update(chunk)
        offset = 0
        while offset < len(chunk):
            written = os.write(destination_descriptor, chunk[offset:])
            if written <= 0:
                raise SystemExit("teacher binary install copy was incomplete")
            offset += written
    os.fchmod(destination_descriptor, 0o500)
    os.fsync(destination_descriptor)
    final = os.fstat(source_descriptor)
    if (
        (initial.st_dev, initial.st_ino, initial.st_mode, initial.st_size,
         initial.st_mtime_ns, initial.st_ctime_ns)
        != (final.st_dev, final.st_ino, final.st_mode, final.st_size,
            final.st_mtime_ns, final.st_ctime_ns)
        or copied != expected_size
        or digest.hexdigest() != expected_hash
    ):
        raise SystemExit("teacher build output changed during install copy")
finally:
    os.close(source_descriptor)
    os.close(destination_descriptor)
directory = os.open(destination.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
BINARY="$SOURCE_DIR/target/release/apery"

python3 - "$SOURCE_DIR" "$FINAL_SOURCE_DIR" <<'PY'
import ctypes
import hashlib
import os
import stat
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("internal source-publish argument mismatch")
staged = Path(sys.argv[1])
final = Path(sys.argv[2])

directory_flags = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def file_digest(descriptor, size):
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("installed source file ended during comparison")
        digest.update(chunk)
        offset += len(chunk)
    return digest.digest()


def tree_identity(root_descriptor):
    rows = []

    def walk(descriptor, prefix):
        for name in sorted(os.listdir(descriptor)):
            if name in {"", ".", ".."} or "/" in name or "\x00" in name:
                raise RuntimeError("installed source contains an unsafe entry")
            status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISLNK(status.st_mode):
                raise RuntimeError("installed source contains a symlink")
            if stat.S_ISDIR(status.st_mode):
                rows.append((relative, "d", stat.S_IMODE(status.st_mode), 0, b""))
                child = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(status.st_mode):
                if status.st_nlink != 1:
                    raise RuntimeError("installed source contains a hard-linked file")
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    before = os.fstat(child)
                    digest = file_digest(child, before.st_size)
                    after = os.fstat(child)
                finally:
                    os.close(child)
                linked = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                identity = lambda item: (
                    item.st_dev,
                    item.st_ino,
                    item.st_mode,
                    item.st_size,
                    item.st_mtime_ns,
                    item.st_ctime_ns,
                )
                if identity(before) != identity(after) or identity(after) != identity(linked):
                    raise RuntimeError("installed source changed during comparison")
                rows.append(
                    (relative, "f", stat.S_IMODE(status.st_mode), status.st_size, digest)
                )
            else:
                raise RuntimeError("installed source contains a special file")

    walk(root_descriptor, "")
    return rows


staged_parent = os.open(staged.parent, directory_flags)
final_parent = os.open(final.parent, directory_flags)
staged_descriptor = -1
final_descriptor = -1
try:
    staged_descriptor = os.open(staged.name, directory_flags, dir_fd=staged_parent)
    staged_status = os.fstat(staged_descriptor)
    try:
        final_descriptor = os.open(final.name, directory_flags, dir_fd=final_parent)
    except FileNotFoundError:
        final_descriptor = -1
    if final_descriptor >= 0:
        final_status = os.fstat(final_descriptor)
        if tree_identity(staged_descriptor) != tree_identity(final_descriptor):
            raise RuntimeError(
                "existing teacher install differs; preserve it and resolve explicitly"
            )
        linked = os.stat(final.name, dir_fd=final_parent, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino) != (final_status.st_dev, final_status.st_ino):
            raise RuntimeError("existing teacher install changed while adopted")
    else:
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(staged_parent, os.fsencode(staged.name), final_parent, os.fsencode(final.name), 4):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), final.name)
        linked = os.stat(final.name, dir_fd=final_parent, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino) != (staged_status.st_dev, staged_status.st_ino):
            raise RuntimeError("published teacher source identity changed")
        os.fsync(final_parent)
finally:
    if staged_descriptor >= 0:
        os.close(staged_descriptor)
    if final_descriptor >= 0:
        os.close(final_descriptor)
    os.close(staged_parent)
    os.close(final_parent)
PY
SOURCE_DIR="$FINAL_SOURCE_DIR"
BINARY="$SOURCE_DIR/target/release/apery"

CARGO_VERSION=$("$CARGO_BIN" --version)
RUSTC_VERSION=$("$RUSTC_BIN" --version)
HOST_OS=$(uname -s)
HOST_ARCH=$(uname -m)
python3 - "$PROJECT_ROOT" "$ARCHIVE" "9" "$SOURCE_DIR" "$BINARY" "$INSTALL_MANIFEST" \
    "$COMPAT_LOCK" "$NUM_BIGINT_VERSION" "$NUM_BIGINT_CRATE_SHA256" \
    "$RELEASE_URL" "$REPOSITORY_URL" "$SOURCE_COMMIT" "$ARCHIVE_SHA256" \
    "$ARCHIVE_SIZE" "$ARCHIVE_ENTRIES" "$ENGINE_LICENSE_SHA256" \
    "$EVAL_LICENSE_SHA256" "$EVAL_README_SHA256" "$KKP_SHA256" "$KPP_SHA256" \
    "$CARGO_VERSION" "$RUSTC_VERSION" "$HOST_OS" "$HOST_ARCH" \
    "$ARCHIVE_TREE_SHA256" "$ARCHIVE_FILES" "$ARCHIVE_UNCOMPRESSED_SIZE" \
    "$EXPECTED_BINARY_SHA256" "$EXPECTED_BINARY_SIZE" "$COMPAT_LOCK_SHA256" \
    "$COMPAT_LOCK_SIZE" <<'PY'
import hashlib
import ctypes
import json
import os
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

(
    project_root_raw,
    archive_raw,
    archive_descriptor_raw,
    source_raw,
    binary_raw,
    manifest_raw,
    compat_lock_raw,
    num_bigint_version,
    num_bigint_crate_sha256,
    release_url,
    repository_url,
    source_commit,
    archive_sha256,
    archive_size,
    archive_entries,
    engine_license_sha256,
    eval_license_sha256,
    eval_readme_sha256,
    kkp_sha256,
    kpp_sha256,
    cargo_version,
    rustc_version,
    host_os,
    host_arch,
    expected_official_tree,
    expected_official_files,
    expected_official_size,
    expected_binary_sha256,
    expected_binary_size,
    expected_compat_lock_sha256,
    expected_compat_lock_size,
) = sys.argv[1:]
project_root = Path(project_root_raw).resolve(strict=True)
archive = Path(archive_raw).resolve(strict=True)
archive_descriptor = int(archive_descriptor_raw)
source = Path(source_raw).resolve(strict=True)
binary = Path(binary_raw).resolve(strict=True)
manifest = Path(manifest_raw)
compat_lock = Path(compat_lock_raw).resolve(strict=True)


def digest(path):
    result = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            result.update(chunk)
    return result.hexdigest(), size


def relative(path):
    return path.resolve(strict=True).relative_to(project_root).as_posix()


source_digest = hashlib.sha256()
source_files = 0
source_bytes = 0
official_digest = hashlib.sha256()
official_files = 0
official_bytes = 0
official_directories = set()
for path in sorted(source.rglob("*")):
    relative_source = path.relative_to(source)
    if relative_source.parts[0] == "target":
        continue
    if path.is_symlink():
        raise SystemExit(f"source tree contains a symlink: {relative_source}")
    if path.is_dir():
        official_directories.add(relative_source.as_posix())
        continue
    if not path.is_file():
        continue
    file_hash, file_size = digest(path)
    archive_path = f"apery_2.0.0/{relative_source.as_posix()}"
    archive_encoded = archive_path.encode("utf-8")
    official_digest.update(len(archive_encoded).to_bytes(4, "big"))
    official_digest.update(archive_encoded)
    official_digest.update(file_size.to_bytes(8, "big"))
    official_digest.update(bytes.fromhex(file_hash))
    official_files += 1
    official_bytes += file_size
    if relative_source.parts[0] == "eval" or relative_source.name == "apery_2.0.0.exe":
        continue
    encoded = relative_source.as_posix().encode("utf-8")
    source_digest.update(len(encoded).to_bytes(4, "big"))
    source_digest.update(encoded)
    source_digest.update(file_size.to_bytes(8, "big"))
    source_digest.update(bytes.fromhex(file_hash))
    source_files += 1
    source_bytes += file_size
archive_status = os.fstat(archive_descriptor)
archive_linked = archive.lstat()
if (archive_linked.st_dev, archive_linked.st_ino) != (
    archive_status.st_dev,
    archive_status.st_ino,
):
    raise SystemExit("cached archive path changed before install-manifest publication")
with os.fdopen(os.dup(archive_descriptor), "rb") as retained, zipfile.ZipFile(
    retained
) as official_archive:
    expected_official_directories = {
        "/".join(entry.filename.rstrip("/").split("/")[1:])
        for entry in official_archive.infolist()
        if entry.is_dir()
    }
if (
    official_digest.hexdigest() != expected_official_tree
    or official_files != int(expected_official_files)
    or official_bytes != int(expected_official_size)
    or official_directories != expected_official_directories
):
    raise SystemExit("installed complete official tree drifted before manifest")

binary_hash, binary_size = digest(binary)
compat_lock_hash, compat_lock_size = digest(compat_lock)
if binary_hash != expected_binary_sha256 or binary_size != int(expected_binary_size):
    raise SystemExit("published teacher binary drifted before manifest")
if (
    compat_lock_hash != expected_compat_lock_sha256
    or compat_lock_size != int(expected_compat_lock_size)
):
    raise SystemExit("published compatibility lock drifted before manifest")
eval_paths = [source / "eval/20190617/KKP.bin", source / "eval/20190617/KPP.bin"]
eval_expected = [kkp_sha256, kpp_sha256]
eval_files = []
for path, expected_hash in zip(eval_paths, eval_expected, strict=True):
    file_hash, file_size = digest(path)
    if file_hash != expected_hash:
        raise SystemExit(f"evaluation file drifted before manifest: {path.name}")
    eval_files.append({"path": relative(path), "sha256": file_hash, "size": file_size})

licenses = []
for path, license_id, expected_hash in [
    (source / "LICENSE", "GPL-3.0-only", engine_license_sha256),
    (source / "eval/20190617/LICENSE-MIT", "MIT", eval_license_sha256),
]:
    file_hash, file_size = digest(path)
    if file_hash != expected_hash:
        raise SystemExit(f"license evidence drifted before manifest: {path.name}")
    licenses.append(
        {"path": relative(path), "license": license_id, "sha256": file_hash, "size": file_size}
    )
readme = source / "eval/20190617/README.md"
readme_hash, readme_size = digest(readme)
if readme_hash != eval_readme_sha256:
    raise SystemExit("evaluation README evidence drifted before manifest")

payload = {
    "schema": "phase4_teacher_install/v2",
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "teacher": {"name": "Apery", "version": "2.0.0"},
    "official": {
        "repository": repository_url,
        "release_url": release_url,
        "tag": "v2.0.0",
        "tag_commit": source_commit,
    },
    "archive": {
        "path": relative(archive),
        "sha256": archive_sha256,
        "size": int(archive_size),
        "entries": int(archive_entries),
    },
    "source": {
        "path": relative(source),
        "tree_hash_algorithm": "sha256(path-length,path,size,file-sha256) excluding target, eval, Windows exe",
        "tree_sha256": source_digest.hexdigest(),
        "files": source_files,
        "size": source_bytes,
    },
    "official_tree": {
        "path": relative(source),
        "tree_hash_algorithm": "sha256(archive-path-length,archive-path,size,file-sha256) excluding target",
        "tree_sha256": official_digest.hexdigest(),
        "files": official_files,
        "directories": len(official_directories),
        "size": official_bytes,
    },
    "build": {
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
        "cargo_version": cargo_version,
        "rustc_version": rustc_version,
        "host_os": host_os,
        "host_arch": host_arch,
        "compatibility_override": {
            "reason": "upstream num-bigint 0.4.0 does not compile with Rust 1.94 integer div_ceil",
            "package": "num-bigint",
            "version": num_bigint_version,
            "license": "MIT OR Apache-2.0",
            "crate_sha256": num_bigint_crate_sha256,
            "lockfile": {
                "path": relative(compat_lock),
                "sha256": compat_lock_hash,
                "size": compat_lock_size,
            },
        },
    },
    "binary": {"path": relative(binary), "sha256": binary_hash, "size": binary_size},
    "evaluation": {
        "directory": relative(source / "eval/20190617"),
        "license": "MIT",
        "readme": {"path": relative(readme), "sha256": readme_hash, "size": readme_size},
        "files": eval_files,
    },
    "licenses": licenses,
}
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
encoded = encoded.encode("utf-8") + b"\n"
manifest.parent.mkdir(parents=True, exist_ok=True)
parent_descriptor = os.open(
    manifest.parent,
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
try:
    try:
        existing_descriptor = os.open(
            manifest.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError:
        existing_descriptor = -1
    if existing_descriptor >= 0:
        try:
            before = os.fstat(existing_descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
                raise SystemExit("existing install manifest is not a bounded regular file")
            chunks = []
            observed = 0
            while chunk := os.read(existing_descriptor, 1024 * 1024):
                observed += len(chunk)
                if observed > 1024 * 1024:
                    raise SystemExit("existing install manifest exceeds its byte bound")
                chunks.append(chunk)
            after = os.fstat(existing_descriptor)
            linked = os.stat(manifest.name, dir_fd=parent_descriptor, follow_symlinks=False)
            identity = lambda item: (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            if identity(before) != identity(after) or identity(after) != identity(linked):
                raise SystemExit("existing install manifest changed while adopted")
            def unique_object(pairs):
                result = {}
                for key, child in pairs:
                    if not isinstance(key, str) or key in result:
                        raise ValueError("duplicate or non-string JSON object key")
                    result[key] = child
                return result

            existing = json.loads(
                b"".join(chunks),
                object_pairs_hook=unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {value}")
                ),
            )
            if not isinstance(existing, dict) or not isinstance(existing.get("created_at"), str):
                raise SystemExit("existing install manifest is malformed")
            expected = dict(payload)
            expected["created_at"] = existing["created_at"]
            existing_command = (
                existing.get("build", {}).get("command")
                if isinstance(existing.get("build"), dict)
                else None
            )
            approved_commands = {
                json.dumps(["cargo", "build", "--release", "--locked"]),
                json.dumps(
                    [
                        "sandbox-exec",
                        "cargo",
                        "--config",
                        "<pinned-official-config>",
                        "build",
                        "--manifest-path",
                        "<private-source>/Cargo.toml",
                        "--target-dir",
                        "<private-source>/target",
                        "--release",
                        "--locked",
                    ]
                ),
                json.dumps(payload["build"]["command"]),
            }
            if json.dumps(existing_command) not in approved_commands:
                raise SystemExit("existing install manifest has an unapproved build command")
            expected["build"] = dict(expected["build"])
            expected["build"]["command"] = existing_command
            if existing != expected:
                raise SystemExit(
                    "existing install manifest differs; preserve it and resolve explicitly"
                )
        finally:
            os.close(existing_descriptor)
    else:
        temporary_name = f".{manifest.name}.{os.getpid()}.{os.urandom(12).hex()}.pending"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise SystemExit("install manifest write was incomplete")
                offset += written
            os.fsync(descriptor)
            temporary_status = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(
            parent_descriptor,
            os.fsencode(temporary_name),
            parent_descriptor,
            os.fsencode(manifest.name),
            4,
        ):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), manifest.name)
        linked = os.stat(manifest.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino) != (
            temporary_status.st_dev,
            temporary_status.st_ino,
        ):
            raise SystemExit("install manifest publication identity changed")
        os.fsync(parent_descriptor)
finally:
    os.close(parent_descriptor)
print(f"installed Apery v2.0.0: {relative(binary)}")
print(f"install manifest: {relative(manifest)}")
PY
