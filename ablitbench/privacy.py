from __future__ import annotations

import csv
import ctypes
import importlib
import io
import os
import re
import subprocess
from pathlib import Path


def owner_sid(identity: bytes) -> str:
    encoding = "utf-16" if identity.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    rows = [
        row for row in csv.reader(io.StringIO(identity.decode(encoding, errors="replace"))) if row
    ]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise OSError("Cannot determine run owner SID")
    sid = rows[0][1].strip()
    if re.fullmatch(r"S-1-\d+(?:-\d+)+", sid) is None:
        raise OSError("Cannot determine run owner SID")
    return sid


def restrict_directory(path: Path) -> None:
    if os.name != "nt":
        os.chmod(path, 0o700)
        return
    system = Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "System32"
    try:
        identity = subprocess.check_output(
            [str(system / "whoami.exe"), "/user", "/fo", "csv", "/nh"]
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OSError("Cannot determine run owner SID") from error
    descriptor = f"D:P(A;OICI;FA;;;{owner_sid(identity)})(A;OICI;FA;;;SY)"
    native = importlib.import_module("ctypes")
    advapi = native.WinDLL("advapi32", use_last_error=True)
    kernel = native.WinDLL("kernel32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    convert.restype = ctypes.c_int
    apply = advapi.SetFileSecurityW
    apply.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_void_p]
    apply.restype = ctypes.c_int
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    pointer = ctypes.c_void_p()
    if not convert(descriptor, 1, ctypes.byref(pointer), None):
        raise OSError(native.get_last_error(), "Cannot create private run ACL")
    try:
        if not apply(str(path), 0x80000004, pointer):
            raise OSError(native.get_last_error(), "Cannot restrict run directory ACL")
    finally:
        kernel.LocalFree(pointer)
