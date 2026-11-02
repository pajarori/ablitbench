from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ablitbench.models import FileEntry, ResolvedTarget, Target, digest

IO_CHUNK = 1024 * 1024


def fingerprint(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def hash_file(path: Path) -> str:
    from ablitbench.formats.snapshot import secure_open

    with secure_open(path) as stream:
        checksum = hashlib.sha256()
        while chunk := stream.read(IO_CHUNK):
            checksum.update(chunk)
        return checksum.hexdigest()


def local_entries(root: Path, excluded: Path, *additional_excluded: Path) -> Any:
    excluded_paths = {path.resolve() for path in (excluded, *additional_excluded)}
    if root.is_file() and not root.is_symlink():
        paths = iter([root])
        base = root.parent
    else:
        base = root

        def walk(directory: Path) -> Any:
            with os.scandir(directory) as entries:
                for item in sorted(entries, key=lambda entry: entry.name):
                    path = Path(item.path)
                    if path.resolve() in excluded_paths or path in excluded_paths:
                        continue
                    if item.is_dir(follow_symlinks=False):
                        yield from walk(path)
                    else:
                        yield path

        paths = walk(root)
    for path in paths:
        entry = FileEntry(name=path.relative_to(base).as_posix(), path=path)
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                entry.issue = "symlink not followed: " + os.readlink(path)
            elif not stat.S_ISREG(mode):
                entry.issue = "special file not opened"
            else:
                before = fingerprint(path)
                entry.sha256 = hash_file(path)
                entry.size = before[2]
                entry.fingerprint = before
                if before != fingerprint(path):
                    entry.issue = "file drift during inventory"
        except (OSError, ValueError) as error:
            entry.issue = f"{type(error).__name__}: file unavailable"
        yield entry


def resolve(
    target: Target,
    mode: str,
    offline: bool,
    directory: Path,
    index: int,
    excluded: Path | None = None,
    scratch: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ResolvedTarget:
    if mode == "api":
        return ResolvedTarget(
            target=target, identity=digest(target.model_dump()), capabilities={"api", "generation"}
        )
    inventory = directory / f"inventory-{index}.jsonl"
    root: Path | None = None
    revision = target.revision
    if target.location.startswith("hf://"):
        from huggingface_hub import HfApi, snapshot_download

        repo = target.location.removeprefix("hf://")
        if offline:
            root = Path(snapshot_download(repo, revision=revision, local_files_only=True))
            revision = root.name
            entries = (
                FileEntry(name=path.relative_to(root).as_posix(), size=path.stat().st_size)
                for path in sorted(root.rglob("*"))
                if path.is_file()
            )
        else:
            info = HfApi().model_info(
                repo,
                revision=revision,
                files_metadata=True,
                token=os.environ.get("HF_TOKEN") or False,
            )
            revision = info.sha
            if revision is None or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", revision):
                raise ValueError("Hub did not return an immutable commit identity")
            entries = (
                FileEntry(
                    name=file.rfilename,
                    size=file.size or 0,
                    sha256=file.lfs.sha256 if file.lfs else None,
                )
                for file in sorted(info.siblings or [], key=lambda file: file.rfilename)
            )
    else:
        root = Path(target.location).expanduser().absolute()
        if root.is_symlink() or not root.exists():
            raise ValueError("Local target must exist and must not be a symlink")
        root = root.resolve()
        if directory.is_relative_to(root) and directory == root:
            raise ValueError("Output cannot be the source directory")
        entries = local_entries(
            root,
            excluded.resolve() if excluded else directory,
            directory,
            *([scratch] if scratch else []),
        )
    if progress:
        progress(
            {
                "phase": "inventory",
                "revision": revision,
                "location": target.location,
                "scope": "cached" if offline and target.location.startswith("hf://") else "full",
            }
        )
    checksum = hashlib.sha256()
    files = size = 0
    with inventory.open("w", encoding="utf-8") as stream:
        for entry in entries:
            encoded = entry.model_dump_json()
            stream.write(encoded + "\n")
            checksum.update(
                json.dumps([entry.name, entry.sha256, entry.size, entry.issue]).encode()
            )
            files += 1
            size += entry.size
    identity = digest(
        {"target": target.model_dump(), "revision": revision, "manifest": checksum.hexdigest()}
    )
    return ResolvedTarget(
        target=target,
        identity=identity,
        root=root,
        inventory=inventory,
        inventory_scope="cached" if offline and target.location.startswith("hf://") else "full",
        revision=revision,
        files=files,
        bytes=size,
        capabilities={"source", "weights"},
    )


def materialize(resolved: ResolvedTarget, entry: FileEntry, offline: bool) -> Path:
    if entry.path:
        if entry.issue:
            raise ValueError(entry.issue)
        if entry.fingerprint != fingerprint(entry.path) or entry.sha256 != hash_file(entry.path):
            raise ValueError("File drift since inventory: " + entry.name)
        return entry.path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            resolved.target.location.removeprefix("hf://"),
            entry.name,
            revision=resolved.revision,
            local_files_only=offline,
            token=os.environ.get("HF_TOKEN") or False,
        )
    )
