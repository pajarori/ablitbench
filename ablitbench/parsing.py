from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import psutil

from ablitbench.limits import available_memory, memory_job
from ablitbench.streams import BinaryStream


class ParserCancelled(asyncio.CancelledError):
    def __init__(self, output: Path) -> None:
        super().__init__("Parser cancelled after emitting partial observations")
        self.output = output


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await asyncio.shield(process.wait())


async def _execute(
    request: dict[str, Any], directory: Path, capacity: int, errors: BinaryStream
) -> tuple[int, bool, bool]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG"}
    }
    creation = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-m",
            "ablitbench.parser",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=errors.fileno(),
            env=environment,
            cwd=directory,
        )
    )
    try:
        process = await asyncio.shield(creation)
    except asyncio.CancelledError:
        try:
            process = await asyncio.shield(creation)
        except Exception:
            raise asyncio.CancelledError from None
        await _stop(process)
        return process.returncode or 0, False, True
    exceeded = False

    async def monitor() -> None:
        nonlocal exceeded
        while process.returncode is None:
            try:
                if psutil.Process(process.pid).memory_info().rss > capacity:
                    exceeded = True
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    return
            except psutil.NoSuchProcess:
                return
            await asyncio.sleep(0.05)

    observer = asyncio.create_task(monitor())

    def noop() -> None:
        pass

    close_job: Callable[[], None] = noop
    cancelled = False
    try:
        close_job = memory_job(process.pid, capacity)
        try:
            await process.communicate(json.dumps(request).encode())
        except asyncio.CancelledError:
            cancelled = True
            await _stop(process)
        return process.returncode or 0, exceeded, cancelled
    except BaseException:
        await _stop(process)
        raise
    finally:
        observer.cancel()
        try:
            await asyncio.gather(observer, return_exceptions=True)
        finally:
            close_job()


def _finalize_output(
    output: Path, returncode: int, errors: BinaryStream, exceeded: bool, cancelled: bool
) -> int:
    complete = 0
    boundary = 0
    terminal: dict[str, Any] | None = None
    malformed = False
    with output.open("r+b") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                malformed = True
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                malformed = True
                break
            if not isinstance(record, dict) or terminal is not None:
                malformed = True
                break
            if record.get("incomplete") and "gap" in record:
                terminal = record
                boundary = position
                continue
            complete += 1
            boundary = stream.tell()
        if not (returncode or exceeded or cancelled or malformed or terminal):
            return complete
        errors.seek(0)
        checksum = hashlib.sha256()
        size = 0
        diagnostics = output.parent / "parser-diagnostics"
        diagnostics.mkdir(mode=0o700, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix="diagnostic-", dir=diagnostics)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as diagnostic:
                while chunk := errors.read(1024 * 1024):
                    checksum.update(chunk)
                    size += len(chunk)
                    diagnostic.write(chunk)
                diagnostic.flush()
                os.fsync(diagnostic.fileno())
            diagnostic_path = diagnostics / checksum.hexdigest()
            os.replace(temporary, diagnostic_path)
        finally:
            temporary.unlink(missing_ok=True)
        gap = terminal or {"gap": "Parser stopped before completing the file"}
        if malformed:
            gap["gap"] = "Parser output stopped at an incomplete or invalid JSONL record"
        if cancelled:
            gap["gap"] = "Parser cancelled before completing the file"
        gap.update(
            {
                "incomplete": True,
                "exit_code": returncode,
                "observations": complete,
                "cancelled": cancelled,
                "memory_capacity_exceeded": exceeded,
                "diagnostic_sha256": checksum.hexdigest(),
                "diagnostic_path": diagnostic_path.relative_to(output.parent).as_posix(),
                "diagnostic_bytes": size,
            }
        )
        gap.setdefault(
            "error_type",
            "CancelledError"
            if cancelled
            else "MemoryCapacityExceeded"
            if exceeded
            else "ParserProtocolError"
            if malformed
            else "ParserProcessError",
        )
        stream.seek(boundary)
        stream.truncate()
        stream.write(json.dumps(gap, ensure_ascii=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return complete


async def parse(
    path: Path,
    operation: str,
    directory: Path,
    memory: int | None = None,
    reference: Path | None = None,
    name: str | None = None,
) -> Path:
    descriptor, output_name = tempfile.mkstemp(prefix="parsed-", dir=directory)
    os.close(descriptor)
    output = Path(output_name)
    try:
        capacity = min(memory, available_memory()) if memory else available_memory()
        request = {
            "path": str(path),
            "operation": operation,
            "output": str(output),
            "memory": capacity,
            "reference": str(reference) if reference else None,
            "name": name,
        }
        with tempfile.TemporaryFile() as errors:
            returncode, exceeded, cancelled = await _execute(request, directory, capacity, errors)
            finalization = asyncio.create_task(
                asyncio.to_thread(_finalize_output, output, returncode, errors, exceeded, cancelled)
            )
            try:
                complete = await asyncio.shield(finalization)
            except asyncio.CancelledError:
                cancelled = True
                await asyncio.shield(finalization)
                recovery = asyncio.create_task(
                    asyncio.to_thread(_finalize_output, output, returncode, errors, exceeded, True)
                )
                complete = await asyncio.shield(recovery)
            if cancelled:
                if complete:
                    raise ParserCancelled(output)
                raise asyncio.CancelledError
        return output
    except ParserCancelled:
        raise
    except BaseException:
        await asyncio.to_thread(output.unlink, missing_ok=True)
        raise


def records(path: Path) -> Generator[dict[str, Any], None, None]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Parser records must be JSON objects")
            yield record
