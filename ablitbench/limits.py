from __future__ import annotations

import ctypes
import importlib
import os
import re
from collections.abc import Callable, Iterator
from ctypes import wintypes
from pathlib import Path, PurePosixPath


def memory_job(pid: int, capacity: int) -> Callable[[], None]:
    if os.name != "nt":
        return lambda: None

    class Basic(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong),
            ("job_time", ctypes.c_longlong),
            ("flags", wintypes.DWORD),
            ("minimum_working_set", ctypes.c_size_t),
            ("maximum_working_set", ctypes.c_size_t),
            ("processes", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "read_operations",
                "write_operations",
                "other_operations",
                "read_bytes",
                "write_bytes",
                "other_bytes",
            )
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ("basic", Basic),
            ("io", IO),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        ]

    native = importlib.import_module("ctypes")
    kernel = native.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError(native.get_last_error(), "Cannot create parser memory job")
    process = None
    try:
        limits = Extended()
        limits.basic.flags = 0x100 | 0x2000
        limits.process_memory = capacity
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise OSError(native.get_last_error(), "Cannot set parser memory capacity")
        process = kernel.OpenProcess(0x100 | 0x1, False, pid)
        if not process or not kernel.AssignProcessToJobObject(job, process):
            raise OSError(native.get_last_error(), "Cannot attach parser to memory job")
        return lambda: kernel.CloseHandle(job)
    except BaseException:
        kernel.CloseHandle(job)
        raise
    finally:
        if process:
            kernel.CloseHandle(process)


def unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def cgroup_memory_paths(membership: str, mountinfo: str) -> Iterator[tuple[Path, str, str]]:
    locations: dict[str, PurePosixPath] = {}
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, membership_path = fields
        member = PurePosixPath(membership_path)
        if not member.is_absolute() or ".." in member.parts:
            continue
        if not controllers:
            locations["cgroup2"] = member
        elif "memory" in controllers.split(","):
            locations["cgroup"] = member
    seen: set[Path] = set()
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 6 or len(filesystem) < 3:
            continue
        kind = filesystem[0]
        if kind not in locations or (kind == "cgroup" and "memory" not in filesystem[2].split(",")):
            continue
        root = PurePosixPath(unescape_mount(fields[3]))
        mount = Path(unescape_mount(fields[4]))
        if not root.is_absolute() or ".." in root.parts or not mount.is_absolute():
            continue
        location = locations[kind]
        if location.is_relative_to(root):
            relative = location.relative_to(root)
        elif location == PurePosixPath("/"):
            relative = PurePosixPath(".")
        else:
            continue
        path = mount / relative
        maximum = "memory.max" if kind == "cgroup2" else "memory.limit_in_bytes"
        used = "memory.current" if kind == "cgroup2" else "memory.usage_in_bytes"
        while path.is_relative_to(mount):
            if path not in seen:
                seen.add(path)
                yield path, maximum, used
            if path == mount:
                break
            path = path.parent


def available_memory() -> int:
    import psutil

    available = psutil.virtual_memory().available
    paths = [
        (Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
        (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes"),
    ]
    try:
        paths.extend(
            cgroup_memory_paths(
                Path("/proc/self/cgroup").read_text(), Path("/proc/self/mountinfo").read_text()
            )
        )
    except (OSError, ValueError):
        pass
    seen: set[tuple[Path, str, str]] = set()
    for path, maximum, used in paths:
        identity = path, maximum, used
        if identity in seen:
            continue
        seen.add(identity)
        try:
            limit = (path / maximum).read_text().strip()
            if limit != "max":
                usage = int((path / used).read_text())
                reclaimable = 0
                try:
                    statistics = dict(
                        line.split() for line in (path / "memory.stat").read_text().splitlines()
                    )
                    key = "inactive_file" if maximum == "memory.max" else "total_inactive_file"
                    reclaimable = max(0, min(usage, int(statistics.get(key, "0"))))
                except (OSError, ValueError):
                    pass
                available = min(available, max(0, int(limit) - usage + reclaimable))
        except (OSError, ValueError):
            pass
    return available
