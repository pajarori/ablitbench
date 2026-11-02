from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from ablitbench.limits import available_memory, memory_job
from ablitbench.models import Evidence, ResolvedTarget
from ablitbench.privacy import restrict_directory

FUNCTIONS = {
    ("ablitbench.targets", "resolve"): "resolve",
    ("ablitbench.formats.snapshot", "snapshot"): "snapshot",
    ("ablitbench.targets", "hash_file"): "hash",
    ("ablitbench.checks.engines", "file_hash"): "hash",
    ("ablitbench.manifests", "file_hash"): "hash",
    ("ablitbench.manifests", "inspect_dataset"): "inspect_dataset",
    ("ablitbench.manifests", "write_cases"): "write_cases",
    ("ablitbench.benchmark", "validate_cases"): "validate_cases",
    ("ablitbench.store", "replay_cases"): "replay_cases",
    ("shutil", "copyfile"): "copy",
    ("ablitbench.formats.provenance", "validate_manifest"): "provenance",
    ("ablitbench.checks.artifacts", "capture"): "capture",
    (os.link.__module__, "link"): "link",
    ("ablitbench.io_worker", "prepare_model"): "model_link",
    ("ablitbench.io_worker", "remove_research"): "remove_research",
}
BASE_ENVIRONMENT = {"PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
HF_ENVIRONMENT = {
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_ENDPOINT",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "XDG_CACHE_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
}


class IOCancelled(asyncio.CancelledError):
    def __init__(self, progress: dict[str, Any]) -> None:
        super().__init__("Trusted host I/O cancelled and reaped")
        self.progress = progress


def encode(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.absolute())
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def hf_cache_environment() -> dict[str, str]:
    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    home = os.environ.get(
        "HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", default_home), "huggingface")
    )
    home = str(Path(os.path.expandvars(os.path.expanduser(home))).absolute())
    hub = os.environ.get(
        "HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(home, "hub"))
    )
    hub = str(Path(os.path.expandvars(os.path.expanduser(hub))).absolute())
    return {"HF_HOME": home, "HF_HUB_CACHE": hub}


def environment(hugging_face: bool, online: bool = True) -> dict[str, str]:
    cache = {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ENDPOINT", "XDG_CACHE_HOME"}
    allowed = BASE_ENVIRONMENT | ((HF_ENVIRONMENT if online else cache) if hugging_face else set())
    result = {key: value for key, value in os.environ.items() if key in allowed}
    if hugging_face:
        result.update(hf_cache_environment())
    result["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    result["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return result


async def stop(process: asyncio.subprocess.Process) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    await asyncio.shield(process.wait())


async def operation(name: str, values: list[Any], *, memory: int | None = None) -> Any:
    parent = None
    if name in {"snapshot", "capture", "write_cases", "replay_cases", "copy", "model_link"}:
        destination = Path(values[3]) if name == "snapshot" else Path(values[1])
        parent = (
            destination.parent
            if name in {"write_cases", "replay_cases", "copy", "model_link"}
            else destination
        )
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    capacity = min(memory, available_memory()) if memory else available_memory()
    hf = (
        name == "resolve" and values[1] == "source" and values[0]["location"].startswith("hf://")
    ) or (name == "snapshot" and values[0]["target"]["location"].startswith("hf://"))
    online = hf and not values[2]
    with tempfile.TemporaryDirectory(prefix=".ablib-io-", dir=parent) as temporary:
        scratch = Path(temporary)
        restrict_directory(scratch)
        response = scratch / "response.json"
        request = {
            "operation": name,
            "values": values,
            "scratch": str(scratch),
            "response": str(response),
            "memory": capacity,
        }
        with (scratch / "stderr").open("w+b") as errors:
            creation = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-m",
                    "ablitbench.io_worker",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=errors,
                    env={
                        **environment(hf, online),
                        "TMPDIR": str(scratch),
                        "TEMP": str(scratch),
                        "TMP": str(scratch),
                    },
                    start_new_session=sys.platform != "win32",
                )
            )
            try:
                process = await asyncio.shield(creation)
            except asyncio.CancelledError:
                try:
                    process = await asyncio.shield(creation)
                except Exception:
                    raise asyncio.CancelledError from None
                await stop(process)
                raise

            def noop() -> None:
                pass

            close_job: Callable[[], None] = noop
            exceeded = False

            async def monitor() -> None:
                nonlocal exceeded
                while process.returncode is None:
                    try:
                        if psutil.Process(process.pid).memory_info().rss > capacity:
                            exceeded = True
                            await stop(process)
                            return
                    except psutil.NoSuchProcess:
                        return
                    await asyncio.sleep(0.05)

            observer = asyncio.create_task(monitor())
            try:
                close_job = memory_job(process.pid, capacity)
                await process.communicate(json.dumps(request).encode())
                if process.returncode or not response.is_file():
                    raise OSError(
                        f"Trusted host I/O did not complete ({name}, exit {process.returncode})"
                        + (" within measured memory capacity" if exceeded else "")
                    )
                result = json.loads(response.read_bytes())
                if not result.get("ok"):
                    error_type = result.get("error_type", "HostIOError")
                    if error_type in {"ProtocolError", "CapabilityError"}:
                        from ablitbench.transport import CapabilityError, ProtocolError

                        protocol_error = (
                            ProtocolError if error_type == "ProtocolError" else CapabilityError
                        )
                        raise protocol_error(str(result.get("detail", "HTTP normalization failed")))
                    if error_type == "ProvenanceRejected":
                        from ablitbench.formats.provenance import ProvenanceRejected

                        raise ProvenanceRejected(str(result.get("detail", "Manifest rejected")))
                    error = (
                        ValueError
                        if error_type in {"ValueError", "ValidationError", "SyntaxError"}
                        else OSError
                    )
                    raise error(f"{error_type}: {result.get('detail', 'trusted I/O failed')}")
                return result.get("result")
            except asyncio.CancelledError:
                await stop(process)
                progress_path = scratch / "progress.json"
                progress = (
                    json.loads(progress_path.read_bytes())
                    if progress_path.is_file()
                    else {"operation": name}
                )
                raise IOCancelled(progress) from None
            finally:
                close_job()
                await stop(process)
                observer.cancel()
                try:
                    await observer
                except asyncio.CancelledError:
                    pass


async def run(function: Callable[..., Any], *args: Any, memory: int | None = None) -> Any:
    key = function.__module__, function.__name__
    if key not in FUNCTIONS:
        raise ValueError("Callable is not a registered trusted host I/O operation")
    name = FUNCTIONS[key]
    values: list[Any]
    if name == "capture":
        values = [str(args[1].absolute()), str(args[0].store.blobs.absolute())]
    else:
        values = [encode(value) for value in args]
    if name == "resolve" and len(values) == 5:
        values.append(None)
    if name == "provenance":
        values = values[:4]
    result = await operation(name, values, memory=memory)
    if name == "resolve":
        return ResolvedTarget.model_validate(result)
    if name in {"snapshot", "copy"}:
        return Path(result)
    if name == "capture":
        return Evidence.model_validate(result[0]), bool(result[1])
    if name == "inspect_dataset":
        from ablitbench.manifests import DatasetInfo

        return DatasetInfo.model_validate(result)
    if name in {"validate_cases", "replay_cases"}:
        return tuple(result) if result is not None else None
    return result
