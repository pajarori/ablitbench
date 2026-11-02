from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath


def unsafe_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return (
        normalized.startswith("/") or ".." in PurePosixPath(normalized).parts or ":" in normalized
    )


def inspect_archive(path: Path) -> Iterator[dict[str, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if unsafe_path(item.filename):
                    yield {"member": item.filename, "issue": "archive path traversal"}
                if item.external_attr >> 16 & 0o170000 == 0o120000:
                    yield {"member": item.filename, "issue": "archive symlink"}
                if item.flag_bits & 1:
                    yield {"member": item.filename, "issue": "encrypted member not inspected"}
    else:
        with tarfile.open(path, mode="r|*") as archive:
            for member in archive:
                if unsafe_path(member.name):
                    yield {"member": member.name, "issue": "archive path traversal"}
                if member.issym() or member.islnk():
                    yield {"member": member.name, "issue": "archive link: " + member.linkname}
                if member.isdev() or member.isfifo():
                    yield {"member": member.name, "issue": "archive special device"}
