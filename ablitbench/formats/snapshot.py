from __future__ import annotations

import hashlib
import importlib
import ntpath
import os
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO

from ablitbench.models import FileEntry, ResolvedTarget
from ablitbench.targets import materialize


def windows_path(value: str) -> str:
    if value[:8].casefold() == "\\\\?\\unc\\":
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return ntpath.normcase(ntpath.normpath(value))


def secure_open(path: Path) -> BinaryIO:
    path = path.absolute()
    if os.name != "nt":
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts[1:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = child
            leaf = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
            )
        finally:
            os.close(descriptor)
        if not stat.S_ISREG(os.fstat(leaf).st_mode):
            os.close(leaf)
            raise ValueError("Snapshot input must be a regular file")
        return os.fdopen(leaf, "rb")
    import ctypes

    msvcrt = importlib.import_module("msvcrt")
    from ctypes import wintypes

    kernel = importlib.import_module("ctypes").WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 1, None, 3, 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(
            importlib.import_module("ctypes").get_last_error(), "Cannot open snapshot input"
        )
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except BaseException:
        close(handle)
        raise
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("Reparse points and special files are not snapshot inputs")
        final_name = kernel.GetFinalPathNameByHandleW
        final_name.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
        final_name.restype = wintypes.DWORD
        needed = final_name(handle, None, 0, 0)
        buffer = ctypes.create_unicode_buffer(needed + 1)
        if not final_name(handle, buffer, len(buffer), 0):
            raise OSError(
                importlib.import_module("ctypes").get_last_error(), "Cannot verify snapshot path"
            )
        if windows_path(buffer.value) != windows_path(str(path)):
            raise ValueError("Snapshot path traversed a reparse point")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def snapshot(target: ResolvedTarget, entry: FileEntry, offline: bool, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity = hashlib.sha256((target.identity + entry.name).encode()).hexdigest()
    destination = directory / (identity + Path(entry.name).suffix)
    if destination.exists():
        if entry.sha256:
            with secure_open(destination) as cached:
                checksum = hashlib.sha256()
                while chunk := cached.read(1024 * 1024):
                    checksum.update(chunk)
                if checksum.hexdigest() != entry.sha256:
                    raise ValueError("Snapshot content differs from inventory")
        return destination
    path = materialize(target, entry, offline) if entry.path is None else entry.path
    if entry.issue:
        raise ValueError(entry.issue)
    descriptor, temporary = tempfile.mkstemp(prefix="snapshot-", dir=directory)
    try:
        with (
            os.fdopen(descriptor, "wb") as output,
            secure_open(path.resolve() if entry.path is None else path) as source,
        ):
            before = os.fstat(source.fileno())
            original = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if entry.fingerprint and original != entry.fingerprint:
                raise ValueError("File drift before snapshot: " + entry.name)
            checksum = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                checksum.update(chunk)
            after = os.fstat(source.fileno())
            if original != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("File drift while snapshotting: " + entry.name)
            if entry.sha256 and checksum.hexdigest() != entry.sha256:
                raise ValueError("File drift since inventory: " + entry.name)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)
