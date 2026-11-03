from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

from ablitbench.formats.json import loads
from ablitbench.formats.snapshot import secure_open

ALIASES = {
    "microsoft": "llm-backdoor-scanner",
    "tanto": "llm-backdoor-defense",
    "refusal_direction": "refusal-direction",
    "api_relay_audit": "api-relay-audit",
}


def project_name(value: str) -> str:
    normalized = value.casefold()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in catalog()["repositories"]:
        raise ValueError("Unknown reviewed research project: " + value)
    return normalized


@cache
def catalog() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "data" / "parity-sources.json"
    value = loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported research source manifest")
    repositories = value.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise ValueError("Research source manifest is empty")
    return value


def source_spec(project: str) -> dict[str, Any]:
    return dict(catalog()["repositories"][project_name(project)])


def source_identity(project: str) -> str:
    specification = source_spec(project)
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {".", "..", ".git"} for part in path.parts)
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or path.as_posix() != value
    ):
        raise ValueError("Unsafe reviewed source path: " + value)
    return path


def verify_file(path: Path, metadata: dict[str, Any]) -> None:
    with secure_open(path) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != metadata["size"]:
            raise ValueError("Reviewed source size/type mismatch: " + path.name)
        hasher = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
        checksum = hasher.hexdigest()
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError("Reviewed source changed during verification")
    if checksum != metadata["sha256"]:
        raise ValueError("Reviewed source content mismatch: " + path.name)


def verified_source(root: Path, project: str) -> Path:
    if root.is_symlink():
        raise ValueError("Reviewed source root cannot be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Reviewed source must be a directory")
    files = source_spec(project)["files"]
    for name, metadata in files.items():
        relative = safe_relative(name)
        verify_file(root.joinpath(*relative.parts), metadata)
    for parent, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(parent, name)
            if path.is_symlink() or name in {".git", "__pycache__"}:
                raise ValueError("Unreviewed source directory: " + name)
        for name in names:
            path = Path(parent, name)
            relative_name = path.relative_to(root).as_posix()
            if relative_name not in files:
                raise ValueError("Unreviewed source file: " + relative_name)
    return root


def prepare_source(
    source: Path, project: str, destination: Path, temporary_directory: Path | None = None
) -> Path:
    if source.is_symlink():
        raise ValueError("Reviewed source root cannot be a symlink")
    source = source.resolve(strict=True)
    if destination.is_symlink():
        raise ValueError("Prepared source root cannot be a symlink")
    if destination.exists():
        return verified_source(destination, project)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix="reviewed-", dir=temporary_directory or destination.parent
    ) as temporary:
        prepared = Path(temporary) / "source"
        prepared.mkdir(mode=0o755)
        files = source_spec(project)["files"]
        for name, metadata in files.items():
            relative = safe_relative(name)
            original = source.joinpath(*relative.parts)
            if "source_link" in metadata:
                if original.is_symlink():
                    if str(original.readlink()) != metadata["source_link"]:
                        raise ValueError("Reviewed source alias changed: " + name)
                else:
                    with secure_open(original) as stream:
                        if stream.read() != metadata["source_link"].encode():
                            raise ValueError("Reviewed source alias changed: " + name)
                original = source.joinpath(*safe_relative(metadata["resolved"]).parts)
            target = prepared.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            checksum = hashlib.sha256()
            with secure_open(original) as stream, target.open("xb") as copied:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != metadata["size"]:
                    raise ValueError("Reviewed source size/type mismatch: " + name)
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
                    copied.write(chunk)
                after = os.fstat(stream.fileno())
                copied.flush()
                os.fsync(copied.fileno())
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in fields):
                raise ValueError("Reviewed source changed while preparing: " + name)
            if checksum.hexdigest() != metadata["sha256"]:
                raise ValueError("Reviewed source content mismatch: " + name)
            target.chmod(0o555 if metadata["mode"] == "100755" else 0o444)
        verified_source(prepared, project)
        shutil.move(str(prepared), destination)
    return destination
