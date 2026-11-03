from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ablitbench.models import Evidence, FileEntry, ResolvedTarget, Target, TrustConfig


def prepare_blob(
    chunks: Iterable[bytes],
    directory: Path,
    media_type: str = "application/json",
    temporary_directory: Path | None = None,
) -> Evidence:
    descriptor, name = tempfile.mkstemp(prefix="pending-", dir=temporary_directory or directory)
    checksum = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
                checksum.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        identity = checksum.hexdigest()
        os.replace(name, directory / identity)
        return Evidence(sha256=identity, size=size, media_type=media_type)
    finally:
        Path(name).unlink(missing_ok=True)


def unlink_prepared(path: Path) -> None:
    try:
        path.unlink()
        return
    except PermissionError:
        if sys.platform != "win32":
            raise
    information = path.lstat()
    if information.st_nlink == 1 and stat.S_ISREG(information.st_mode):
        os.chmod(path, 0o600)
        path.unlink()
        return
    import ctypes
    from ctypes import wintypes

    native = importlib.import_module("ctypes")
    kernel = native.WinDLL("kernel32", use_last_error=True)
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
    handle = create(str(path), 0x00010000, 7, None, 3, 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(native.get_last_error(), "Cannot open prepared hardlink for removal")
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    try:
        remove = kernel.SetFileInformationByHandle
        remove.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        remove.restype = wintypes.BOOL
        flags = wintypes.DWORD(0x00000013)
        if not remove(handle, 21, ctypes.byref(flags), ctypes.sizeof(flags)):
            raise OSError(native.get_last_error(), "Cannot remove prepared hardlink safely")
    finally:
        close(handle)


def remove_research(directory: Path, identity: str) -> None:
    if (
        not directory.is_absolute()
        or not identity
        or identity in {".", ".."}
        or Path(identity).name != identity
        or "/" in identity
        or "\\" in identity
    ):
        raise ValueError("Research cleanup requires a private run directory and one identity")
    root = directory / "research" / identity
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Research cleanup rejects symlink directories")
    if not root.exists():
        return

    def retry(function: Callable[..., Any], name: str, error: BaseException) -> None:
        path = Path(name)
        if not isinstance(error, PermissionError) or sys.platform != "win32":
            raise error
        if path.is_file() and not path.is_symlink():
            unlink_prepared(path)
        elif path.is_dir() and not path.is_symlink():
            os.chmod(path, 0o700)
            function(name)
        else:
            raise error

    shutil.rmtree(root, onexc=retry)


def prepare_model(
    source: Path, destination: Path, root: Path, temporary_directory: Path | None = None
) -> None:
    from ablitbench.formats.snapshot import secure_open

    if not source.is_relative_to(
        root.parent.parent / "snapshots" / root.name
    ) or not destination.is_relative_to(root):
        raise ValueError("Model preparation accepts private snapshot copies only")
    paths = [root.parent, root]
    current = root
    for part in destination.parent.relative_to(root).parts:
        current = current / part
        paths.append(current)
    for path in paths:
        if not path.is_absolute():
            continue
        if path.is_symlink():
            raise ValueError("Prepared model directory symlink rejected")
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
        os.chmod(path, 0o755)
    with secure_open(source) as stream:
        original = os.fstat(stream.fileno())
    if destination.is_symlink():
        raise ValueError("Prepared model file symlink rejected")
    readonly = stat.S_IMODE(original.st_mode) == 0o444
    if destination.exists():
        with secure_open(destination) as stream:
            existing = os.fstat(stream.fileno())
        if (
            sys.platform != "win32"
            and readonly
            and (original.st_dev, original.st_ino) == (existing.st_dev, existing.st_ino)
        ):
            return
    elif readonly and sys.platform != "win32":
        os.link(source, destination)
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix="prepared-", dir=temporary_directory or destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output, secure_open(source) as stream:
            before = os.fstat(stream.fileno())
            shutil.copyfileobj(stream, output, 1024 * 1024)
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Private snapshot drift during model preparation")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        if sys.platform == "win32" and destination.exists():
            unlink_prepared(destination)
        os.replace(temporary, destination)
    finally:
        try:
            unlink_prepared(Path(temporary))
        except FileNotFoundError:
            pass


def prepare_secret(source: Path, destination: Path) -> None:
    from ablitbench.formats.snapshot import secure_open

    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output, secure_open(source) as stream:
            before = os.fstat(stream.fileno())
            shutil.copyfileobj(stream, output, 1024 * 1024)
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Secret input changed during private staging")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(destination, 0o444)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def write_progress(request: dict[str, Any], record: dict[str, Any]) -> None:
    destination = Path(request["scratch"]) / "progress.json"
    temporary = destination.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({"operation": request["operation"], **record}, stream, ensure_ascii=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def execute(request: dict[str, Any]) -> Any:
    operation, values, scratch = request["operation"], request["values"], Path(request["scratch"])
    if operation == "hub_revision":
        from ablitbench.parity.hub import revision

        return revision(Path(values[0]))
    if operation == "trusted_acquire":
        from ablitbench.parity.acquisition_worker import run
        from ablitbench.provenance import implementation_identity

        if request.get("implementation") != implementation_identity():
            raise ValueError("Acquisition core identity differs; install the matching wheel")
        return run(values[0])
    if operation in {"environment_project", "environment_before", "environment_after"}:
        from ablitbench.parity.environment import EnvironmentObserver

        observer = EnvironmentObserver(Path(values[0]), Path(values[1]))
        if operation == "environment_project":
            return str(observer.project())
        if operation == "environment_before":
            return observer.before(values[2])
        return observer.after(values[2], values[3], Path(values[4]))
    if operation == "prepare_secret":
        prepare_secret(Path(values[0]), Path(values[1]))
        return None
    if operation == "normalize_outcomes":
        from ablitbench.parity.artifacts import normalize_outcomes

        return normalize_outcomes(Path(values[0]), Path(values[1]), values[2], values[3], values[4])
    if operation == "normalize_findings":
        from ablitbench.parity.artifacts import normalize_findings

        return normalize_findings(Path(values[0]), Path(values[1]), values[2], values[3])
    if operation == "normalize_worker_response":
        from ablitbench.response_spool import normalize_worker_response

        return normalize_worker_response(Path(values[0]), Path(values[1]))
    if operation == "artifact_input":
        root = Path(values[0]).resolve(strict=True)
        path = root
        for part in Path(values[1]).parts:
            if part in {".", ".."}:
                raise ValueError("Unsafe artifact dependency path")
            path = path / part
            if path.is_symlink():
                raise ValueError("Artifact dependency symlink rejected")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("Artifact dependency escaped output directory")
        return str(resolved)
    if operation == "prepare_source":
        from ablitbench.parity.sources import prepare_source

        return str(prepare_source(Path(values[0]), values[1], Path(values[2]), scratch))
    if operation == "resolve":
        from ablitbench.targets import resolve

        return resolve(
            Target.model_validate(values[0]),
            values[1],
            values[2],
            Path(values[3]),
            values[4],
            Path(values[5]) if values[5] else None,
            scratch,
            lambda record: write_progress(request, record),
        ).model_dump(mode="json")
    if operation == "snapshot":
        from ablitbench.formats.snapshot import snapshot

        return str(
            snapshot(
                ResolvedTarget.model_validate(values[0]),
                FileEntry.model_validate(values[1]),
                values[2],
                Path(values[3]),
                scratch,
            )
        )
    if operation == "remove_research":
        return remove_research(Path(values[0]), values[1])
    if operation == "model_link":
        return prepare_model(Path(values[0]), Path(values[1]), Path(values[2]), scratch)
    if operation == "redact_response_wire":
        from ablitbench.response_spool import redact_response_wire

        return redact_response_wire(
            Path(values[0]),
            Path(values[1]),
            values[2],
            values[3],
            values[4] if len(values) > 4 else "ABLIB_REDACTED_CREDENTIAL",
        )
    if operation == "decode_generation":
        from ablitbench.response_spool import decode_generation

        return decode_generation(
            Path(values[0]),
            Path(values[1]),
            values[2],
            values[3],
            values[4],
            values[5],
            values[6] if len(values) > 6 else None,
            values[7] if len(values) > 7 else None,
        )
    if operation == "decode_probe":
        from ablitbench.response_spool import decode_probe

        return decode_probe(Path(values[0]), Path(values[1]), values[2])
    if operation == "hash":
        from ablitbench.targets import hash_file

        return hash_file(Path(values[0]))
    if operation == "copy":
        from ablitbench.formats.snapshot import secure_open

        source, destination = Path(values[0]).resolve(), Path(values[1])
        temporary = scratch / "copy.partial"
        with secure_open(source) as stream, temporary.open("wb") as output:
            before = os.fstat(stream.fileno())
            shutil.copyfileobj(stream, output, 1024 * 1024)
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Input changed during trusted data copy")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        return str(destination)
    if operation == "link":
        os.link(values[0], values[1])
        return None
    if operation == "capture":
        from ablitbench.formats.snapshot import secure_open

        with secure_open(Path(values[0])) as stream:
            before = os.fstat(stream.fileno())
            evidence = prepare_blob(
                iter(lambda: stream.read(1024 * 1024), b""),
                Path(values[1]),
                "application/octet-stream",
                scratch,
            )
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        return evidence.model_dump(mode="json"), all(
            getattr(before, field) == getattr(after, field) for field in fields
        )
    if operation == "list_artifacts":
        from ablitbench.checks.artifacts import paths

        with Path(values[1]).open("w", encoding="utf-8") as stream:
            for artifact_path, error in paths(Path(values[0])):
                stream.write(
                    json.dumps([str(artifact_path) if artifact_path else None, error]) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        return None
    if operation == "provenance":
        from ablitbench.formats.provenance import validate_manifest

        return validate_manifest(
            Path(values[0]), TrustConfig.model_validate(values[1]), values[2], Path(values[3])
        )
    if operation == "inspect_dataset":
        from ablitbench.manifests import inspect_dataset

        return inspect_dataset(Path(values[0]), values[1]).model_dump(mode="json")
    if operation == "write_cases":
        from ablitbench.manifests import DatasetInfo, write_cases

        destination = Path(values[1])
        temporary = scratch / "cases.jsonl"
        write_cases(DatasetInfo.model_validate(values[0]), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return None
    if operation == "replay_cases":
        from ablitbench.store import replay_cases

        return replay_cases(Path(values[0]), Path(values[1]), scratch)
    if operation == "validate_cases":
        from ablitbench.benchmark import validate_cases

        return validate_cases(Path(values[0]))
    raise ValueError("Unknown trusted host I/O operation")


def main() -> None:
    request = json.load(sys.stdin)
    output = Path(request["response"])
    try:
        if request.get("memory") and sys.platform == "linux":
            import resource

            import psutil

            memory = psutil.Process().memory_info()
            baseline = int(getattr(memory, "data", memory.rss))
            capacity = baseline + int(request["memory"])
            _, hard = resource.getrlimit(resource.RLIMIT_DATA)
            if hard != resource.RLIM_INFINITY:
                capacity = min(capacity, hard)
            resource.setrlimit(resource.RLIMIT_DATA, (capacity, capacity))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        values = request["values"]
        progress: dict[str, Any] = {"phase": "started"}
        if request["operation"] == "snapshot":
            progress.update(
                revision=values[0]["revision"],
                target_identity=values[0]["identity"],
                location=values[0]["target"]["location"],
                file=values[1]["name"],
            )
        if request["operation"] == "snapshot" and values[0]["target"]["location"].startswith(
            "hf://"
        ):
            import huggingface_hub
            from huggingface_hub import constants

            progress.update(
                hub_version=huggingface_hub.__version__,
                cache=constants.HF_HUB_CACHE,
                download_timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
                etag_timeout=constants.HF_HUB_ETAG_TIMEOUT,
                cache_policy="Official SDK owns cache; forced-stop partials remain incomplete",
            )
        write_progress(request, progress)
        result = {"ok": True, "result": execute(request)}
    except Exception as error:
        result = {"ok": False, "error_type": type(error).__name__, "detail": str(error)}
    temporary = output.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
