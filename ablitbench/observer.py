from __future__ import annotations

import ast
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ablitbench.models import WorkerRequest


def observations(trace: str) -> list[dict[str, str]]:
    output = []
    for line in trace.splitlines():
        match = re.search(
            r"\b(openat|open|execve|connect|unlink|unlinkat|rename|renameat|mkdir|rmdir)\((.*)",
            line,
        )
        if match is None:
            continue
        syscall, arguments = match.groups()
        quoted = re.search(r'"(?:[^"\\]|\\.)*"', arguments)
        path = None
        if quoted:
            try:
                path = ast.literal_eval(quoted.group())
            except (ValueError, SyntaxError):
                pass
        successful = re.search(r"\)\s+=\s+(-?\d+)(?:<([^>]+)>)?\s*(?:<[\d.]+>)?$", line)
        if successful is None or int(successful.group(1)) < 0:
            continue
        record = {"syscall": syscall, "result": successful.group(1)}
        if isinstance(path, str):
            record["raw_path"] = path
            record["path"] = path
        if syscall in {"open", "openat"}:
            opened = successful.group(2)
            directory = re.match(r"\s*(?:\d+|AT_FDCWD)<([^>]+)>,", arguments)
            if opened and opened.startswith("/"):
                record["path"] = opened
                record["path_resolution"] = "returned file descriptor annotation"
            elif isinstance(path, str) and path.startswith("/"):
                record["path_resolution"] = "absolute syscall argument"
            elif (
                syscall == "openat"
                and isinstance(path, str)
                and directory is not None
                and directory[1].startswith("/")
            ):
                record["path"] = posixpath.normpath(posixpath.join(directory[1], path))
                record["path_resolution"] = "directory file descriptor annotation"
            else:
                record["path_resolution"] = "unresolved syscall argument"
            record["access"] = (
                "write"
                if any(flag in arguments for flag in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC"))
                else "read"
            )
        output.append(record)
    return output


def execute(request: WorkerRequest) -> dict[str, Any]:
    if sys.platform != "linux":
        raise ValueError("Observed tool execution requires the Linux strace worker image")
    if not shutil.which("strace"):
        return {
            "error": "Tool observer is unavailable; executor was not started",
            "level": "sandbox execution",
            "observer": {"status": "inconclusive", "trace": "", "events": []},
        }
    with tempfile.TemporaryDirectory(prefix="ablib-observer-") as directory:
        path = Path(directory) / "syscalls.log"
        fixture_path = "/fixture/public.txt"
        fixture_identity = None
        if request.fixture_text is not None and request.arguments.get("path") == fixture_path:
            fixture = Path(directory) / "fixture" / "public.txt"
            fixture.parent.mkdir(mode=0o700)
            content = request.fixture_text.encode("utf-8")
            with fixture.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            fixture.chmod(0o444)
            fixture_identity = hashlib.sha256(content).hexdigest()
            fixture_path = str(fixture)
        request = request.model_copy(
            update={"fixture_path": fixture_path if fixture_identity else None}
        )
        command = [
            "strace",
            "-f",
            "-qq",
            "-s",
            "0",
            "-yy",
            "-e",
            "trace=%file,%network,%process",
            "-o",
            str(path),
            sys.executable,
            "-I",
            "-m",
            "ablitbench.executor",
        ]
        process = subprocess.run(
            command,
            input=request.model_dump_json().encode(),
            capture_output=True,
        )
        trace = path.read_text(errors="replace") if path.exists() else ""
        observer = {
            "status": "completed" if process.returncode == 0 else "inconclusive",
            "method": "strace child-process syscall observation",
            "scope": "configured tool executor and descendants; not inference-wide telemetry",
            "trace": trace,
            "events": observations(trace),
            "version": subprocess.check_output(["strace", "--version"], text=True).splitlines()[0],
            "returncode": process.returncode,
            "diagnostic": process.stderr.decode(errors="replace"),
            "fixture": {
                "logical_path": "/fixture/public.txt",
                "sha256": fixture_identity,
                "preparation": "trusted controller; not a model action",
            },
        }
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
            result = None
        if process.returncode or not isinstance(result, dict):
            observer["status"] = "inconclusive"
            observer["executor_stdout"] = process.stdout.decode(errors="replace")
            return {
                "error": "Observed executor failed; no unobserved fallback",
                "level": "sandbox execution",
                "observer": observer,
            }
        result["observer"] = observer
        if request.arguments.get("path") == "/fixture/public.txt" and not any(
            event.get("path") == fixture_path and event.get("access") == "read"
            for event in result["observer"]["events"]
        ):
            observer["status"] = "inconclusive"
            result["error"] = "Observer did not confirm the authorized fixture read"
        return result
