from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

from ablitbench.formats.snapshot import secure_open
from ablitbench.models import CheckResult, Evidence
from ablitbench.privacy import restrict_directory

if TYPE_CHECKING:
    from ablitbench.core import Context


def output_directory(context: Context) -> Path:
    parent = context.store.directory / "worker-output"
    parent.mkdir(mode=0o700, exist_ok=True)
    directory = parent / uuid.uuid4().hex
    directory.mkdir(mode=0o777)
    os.chmod(directory, 0o777)
    return directory


def capture(context: Context, path: Path) -> tuple[Evidence, bool]:
    with secure_open(path) as stream:
        before = os.fstat(stream.fileno())
        reference = context.store.prepare_blob(
            iter(lambda: stream.read(1024 * 1024), b""), "application/octet-stream"
        )
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    return reference, all(getattr(before, field) == getattr(after, field) for field in fields)


def paths(directory: Path) -> Generator[tuple[Path | None, str]]:
    errors: list[str] = []
    for root, directories, files in os.walk(
        directory, followlinks=False, onerror=lambda error: errors.append(str(error))
    ):
        while errors:
            yield None, errors.pop()
        for name in list(directories):
            if Path(root, name).is_symlink():
                directories.remove(name)
                yield None, "Worker artifact directory symlink rejected: " + name
        for name in files:
            yield Path(root) / name, ""
    while errors:
        yield None, errors.pop()


async def collect(context: Context, directory: Path, result: CheckResult) -> None:
    from ablitbench.io import operation

    with tempfile.TemporaryDirectory(prefix="ablib-artifacts-") as temporary:
        restrict_directory(Path(temporary))
        listing = Path(temporary) / "entries.jsonl"
        try:
            await operation(
                "list_artifacts",
                [str(directory), str(listing)],
                memory=context.config.runtime.memory_bytes,
            )
        except (OSError, ValueError) as listing_error:
            result.coverage.gaps.append(
                "Worker artifact enumeration incomplete: " + str(listing_error)
            )
            context.checkpoint(result)
            return
        with listing.open(encoding="utf-8") as entries:
            for line in entries:
                await context.yield_control()
                name, path_error = json.loads(line)
                if name is None:
                    result.coverage.gaps.append(path_error)
                    context.checkpoint(result)
                    continue
                path = Path(name)
                try:
                    reference, stable = await context.io(capture, context, path)
                    context.store.commit_blob(reference)
                    result.evidence.append(reference.sha256)
                    result.evidence.append(
                        context.store.evidence(
                            {
                                "worker_artifact": path.relative_to(directory).as_posix(),
                                "sha256": reference.sha256,
                                "size": reference.size,
                                "stable": stable,
                            }
                        )
                    )
                    if not stable:
                        result.coverage.gaps.append(
                            "Worker artifact changed during capture: " + str(path.name)
                        )
                except (OSError, ValueError) as error:
                    result.coverage.gaps.append(
                        path.relative_to(directory).as_posix() + ": " + str(error)
                    )
                context.checkpoint(result)
