"""High-resolution process identities used before sending numeric PID signals."""

from __future__ import annotations

import ctypes
import os
import stat
import sys
from pathlib import Path
from typing import Final

_PROC_PIDTBSDINFO: Final = 3
_MAXCOMLEN: Final = 16


class _ProcBsdInfo(ctypes.Structure):
    """Darwin ``struct proc_bsdinfo`` from the public libproc SDK boundary."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * _MAXCOMLEN),
        ("pbi_name", ctypes.c_char * (2 * _MAXCOMLEN)),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def read_process_identity(process_id: int) -> str | None:
    """Return a PID-reuse-resistant identity, or fail closed when unavailable.

    Darwin uses ``proc_pidinfo(PROC_PIDTBSDINFO)`` and binds microsecond start
    time together with PID, process group, and session. Linux uses the kernel
    start-tick field in ``/proc/<pid>/stat``. Other platforms deliberately have
    no weak wall-clock fallback.
    """

    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        return None
    if sys.platform == "darwin":
        return _read_darwin_identity(process_id)
    if sys.platform.startswith("linux"):
        return _read_linux_identity(process_id)
    return None


def _read_darwin_identity(process_id: int) -> str | None:
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = library.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
    except (AttributeError, OSError):
        return None

    def snapshot() -> tuple[int, int, int, int] | None:
        info = _ProcBsdInfo()
        result = proc_pidinfo(
            process_id,
            _PROC_PIDTBSDINFO,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if result != ctypes.sizeof(info):
            return None
        values = (
            int(info.pbi_pid),
            int(info.pbi_pgid),
            int(info.pbi_start_tvsec),
            int(info.pbi_start_tvusec),
        )
        if (
            values[0] != process_id
            or values[1] <= 0
            or values[2] <= 0
            or not 0 <= values[3] < 1_000_000
        ):
            return None
        return values

    first = snapshot()
    if first is None:
        return None
    try:
        process_group = os.getpgid(process_id)
        session = os.getsid(process_id)
    except (OSError, AttributeError):
        return None
    second = snapshot()
    if first != second or first[1] != process_group or session <= 0:
        return None
    return f"darwin:{process_id}:{process_group}:{session}:{first[2]}:{first[3]:06d}"


def _read_linux_identity(process_id: int) -> str | None:
    path = Path(f"/proc/{process_id}/stat")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        initial = os.fstat(descriptor)
        # procfs reports a logical size of zero for this bounded virtual file.
        if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= 16 * 1024:
            return None
        raw = os.read(descriptor, 16 * 1024 + 1)
        final = os.fstat(descriptor)
        if len(raw) > 16 * 1024 or (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            return None
        closing = raw.rfind(b")")
        if closing < 2:
            return None
        pid_text = raw[: raw.find(b" ")]
        fields = raw[closing + 2 :].split()
        if len(fields) < 20:
            return None
        observed_pid = int(pid_text)
        process_group = int(fields[2])
        session = int(fields[3])
        start_ticks = int(fields[19])
        if (
            observed_pid != process_id
            or process_group <= 0
            or session <= 0
            or start_ticks <= 0
            or os.getpgid(process_id) != process_group
            or os.getsid(process_id) != session
        ):
            return None
    except (OSError, ValueError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return f"linux:{process_id}:{process_group}:{session}:{start_ticks}"
