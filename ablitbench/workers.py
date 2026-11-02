from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ablitbench.models import Runtime, WorkerRequest, WorkerResponse
from ablitbench.provenance import implementation_identity
from ablitbench.streams import BinaryStream
from ablitbench.transport import CapabilityError


class Worker:
    def __init__(
        self, runtime: Runtime, model: Path | None = None, base: Path | None = None
    ) -> None:
        self.runtime, self.model, self.base = runtime, model, base
        self.processes: dict[bool, tuple[asyncio.subprocess.Process, str, BinaryStream]] = {}
        self.buffers: dict[bool, bytes] = {}
        self.containers: set[str] = set()

    def validate(self, image: str | None) -> str:
        if not image or not re.fullmatch(r"(?:[^\s]+@)?sha256:[a-f0-9]{64}", image):
            raise CapabilityError("Sandbox requires an image pinned by sha256 digest")
        if not all([self.runtime.memory_bytes, self.runtime.cpus, self.runtime.pids]):
            raise CapabilityError(
                "Sandbox requires explicit memory_bytes, cpus, and pids isolation capacity"
            )
        if not shutil.which(self.runtime.engine):
            raise CapabilityError("Docker/Podman is unavailable; no host fallback")
        return image

    async def start(self, image: str | None, inference: bool) -> asyncio.subprocess.Process:
        image = self.validate(image)
        if inference in self.processes:
            return self.processes[inference][0]
        name = "ablib-" + uuid.uuid4().hex
        command = [
            self.runtime.engine,
            "create",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            "--memory",
            str(self.runtime.memory_bytes),
            "--memory-swap",
            str(self.runtime.memory_bytes),
            "--cpus",
            str(self.runtime.cpus),
            "--pids-limit",
            str(self.runtime.pids),
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={self.runtime.memory_bytes}",
            "--interactive",
        ]
        threads = str(max(1, math.ceil(self.runtime.cpus or 1)))
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            command += ["--env", f"{variable}={threads}"]
        if inference:
            if self.model is None or not self.model.is_dir():
                raise CapabilityError("Source inference requires a prepared model directory")
            source = self.model.resolve()
            if "," in str(source):
                raise CapabilityError("Container mount paths containing commas are unsupported")
            command += ["--mount", f"type=bind,src={source},dst=/model,readonly"]
        if inference and self.base is not None:
            base = self.base.resolve()
            if "," in str(base) or not base.is_dir():
                raise CapabilityError(
                    "Adapter base must be a local directory with a safe mount path"
                )
            command += ["--mount", f"type=bind,src={base},dst=/base,readonly"]
        if inference and self.runtime.device == "cuda":
            command += ["--gpus=all"]
        command += [image, "python", "-m", "ablitbench.worker"]
        errors = tempfile.TemporaryFile()
        self.containers.add(name)

        async def create() -> None:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, diagnostic = await process.communicate()
            if process.returncode:
                raise CapabilityError(
                    "Sandbox creation failed: " + diagnostic.decode(errors="replace")
                )

        creation = asyncio.create_task(create())
        try:
            try:
                await asyncio.shield(creation)
            except asyncio.CancelledError:
                await asyncio.shield(creation)
                raise
            attachment = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    self.runtime.engine,
                    "start",
                    "--attach",
                    "--interactive",
                    name,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=errors,
                )
            )
            try:
                process = await asyncio.shield(attachment)
            except asyncio.CancelledError:
                process = await asyncio.shield(attachment)
                self.processes[inference] = process, name, errors
                raise
            self.processes[inference] = process, name, errors
            self.buffers[inference] = b""
            return process
        except BaseException:
            await self.close()
            errors.close()
            raise

    async def request(
        self, image: str | None, request: WorkerRequest, inference: bool
    ) -> WorkerResponse:
        request = request.model_copy(
            update={
                "memory_bytes": self.runtime.memory_bytes,
                "cpus": self.runtime.cpus,
                "pids": self.runtime.pids,
            }
        )
        process = await self.start(image, inference)
        if process.stdin is None or process.stdout is None:
            raise CapabilityError("Worker has no communication channel")
        try:
            process.stdin.write(request.model_dump_json().encode() + b"\n")
            await process.stdin.drain()
            buffer = self.buffers[inference]
            while b"\n" not in buffer:
                chunk = await process.stdout.read(1024 * 1024)
                if not chunk:
                    _, _, errors = self.processes[inference]
                    errors.seek(0)
                    raise CapabilityError(
                        "Sandbox disconnected: " + errors.read().decode(errors="replace")
                    )
                buffer += chunk
            line, self.buffers[inference] = buffer.split(b"\n", 1)
            response = WorkerResponse.model_validate_json(line)
            if response.implementation != implementation_identity():
                raise CapabilityError(
                    "Worker core identity differs; rebuild image from the current wheel"
                )
            if response.error:
                raise CapabilityError(response.error)
            if response.isolation != "linux":
                raise CapabilityError("Worker isolation is not Linux")
            return response
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        for name in list(self.containers):
            cleanup = await asyncio.create_subprocess_exec(
                self.runtime.engine,
                "rm",
                "--force",
                name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, diagnostic = await asyncio.shield(cleanup.communicate())
            if cleanup.returncode and b"No such container" not in diagnostic:
                raise CapabilityError(
                    f"Sandbox cleanup failed for {name}: " + diagnostic.decode(errors="replace")
                )
            self.containers.discard(name)
        for process, _, errors in list(self.processes.values()):
            try:
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                await asyncio.shield(process.wait())
            finally:
                errors.close()
        self.processes.clear()
        self.buffers.clear()

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int | None,
        seed: int | None,
    ) -> dict[str, Any]:
        if not max_output_tokens:
            raise CapabilityError("Inference requires max_output_tokens from user configuration")
        response = await self.request(
            self.runtime.inference_image,
            WorkerRequest(
                operation="generate",
                backend=self.runtime.backend,
                device=self.runtime.device,
                context_tokens=self.runtime.context_tokens,
                messages=messages,
                tools=tools,
                max_output_tokens=max_output_tokens,
                seed=seed,
            ),
            True,
        )
        return response.model_dump(mode="json", exclude_none=True)

    async def tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await self.request(
            self.runtime.tool_image,
            WorkerRequest(operation="tool", name=name, arguments=arguments),
            False,
        )
        if response.result is None:
            raise CapabilityError("Worker returned no tool result")
        return response.result

    async def scan(self, engine: str, file: str) -> dict[str, Any]:
        response = await self.request(
            self.runtime.research_image,
            WorkerRequest.model_validate(
                {
                    "operation": "scan",
                    "engine": engine,
                    "file": file,
                    "timeout": self.runtime.engine_timeout,
                }
            ),
            True,
        )
        if response.scan is None:
            raise CapabilityError("Research worker returned no scan result")
        return response.scan
