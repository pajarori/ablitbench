from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads
from ablitbench.models import Runtime, WorkerHello, WorkerRequest, WorkerResponse
from ablitbench.provenance import implementation_identity
from ablitbench.streams import BinaryStream
from ablitbench.transport import CapabilityError

BOOTSTRAP = """import importlib.machinery,importlib.util,json,sys
from pathlib import Path
package=importlib.machinery.PathFinder.find_spec("ablitbench")
if package is None or package.origin is None:
    raise SystemExit("Worker package is unavailable")
spec=importlib.util.spec_from_file_location(
    "_ablib_bootstrap_provenance",Path(package.origin).with_name("provenance.py"))
if spec is None or spec.loader is None:
    raise SystemExit("Worker implementation identity is unavailable")
provenance=importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)
identity=provenance.implementation_identity()
hello={"schema_version":1,"implementation":identity}
print(json.dumps(hello,separators=(",",":")),flush=True)
ack=json.loads(sys.stdin.readline())
if ack != hello:
    raise SystemExit("Worker implementation acknowledgement mismatch")
from ablitbench.worker import main
main()
"""


class HandshakeError(CapabilityError):
    pass


class Worker:
    def __init__(
        self,
        runtime: Runtime,
        model: Path | None = None,
        base: Path | None = None,
        output: Path | None = None,
        engine_source: Path | None = None,
    ) -> None:
        self.runtime, self.model, self.base = runtime, model, base
        self.output = output
        self.engine_source = engine_source
        self.processes: dict[bool, tuple[asyncio.subprocess.Process, str, BinaryStream]] = {}
        self.buffers: dict[bool, bytes] = {}
        self.containers: set[str] = set()
        self.accepted: dict[bool, tuple[str, str]] = {}
        self.incompatible_images: dict[tuple[str, str, str], str] = {}
        self.native_tools: bool | None = None
        self.model_initialized = False
        self.loaded_identity: tuple[str | None, ...] | None = None
        self.initialization_failure: tuple[tuple[str | None, ...], str] | None = None

    @property
    def native_tools(self) -> bool | None:
        if self.capability_identity != self.inference_identity():
            return None
        return self._native_tools

    @native_tools.setter
    def native_tools(self, value: bool | None) -> None:
        self._native_tools = value
        self.capability_identity = self.inference_identity()

    def inference_identity(self) -> tuple[str | None, ...]:
        return (
            self.runtime.inference_image,
            self.runtime.backend,
            self.runtime.device,
            str(self.model) if self.model else None,
            str(self.base) if self.base else None,
            str(self.runtime.context_tokens),
            str(self.runtime.memory_bytes),
            str(self.runtime.cpus),
            str(self.runtime.pids),
        )

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
        expected = implementation_identity()
        identity = self.runtime.engine, image, expected
        if identity in self.incompatible_images:
            raise HandshakeError(self.incompatible_images[identity])
        if inference in self.processes:
            if self.accepted.get(inference) != (image, expected):
                raise CapabilityError("Active worker implementation or image identity changed")
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
            "--entrypoint",
            "python",
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
        if self.output is not None:
            output = self.output.resolve(strict=True)
            if "," in str(output) or not output.is_dir():
                raise CapabilityError("Worker output must be a directory with a safe mount path")
            if any(output.iterdir()):
                raise CapabilityError("Worker output mount must start empty")
            command += ["--mount", f"type=bind,src={output},dst=/output"]
        if self.engine_source is not None:
            source = self.engine_source.resolve(strict=True)
            if "," in str(source) or not source.is_file():
                raise CapabilityError(
                    "Reviewed engine source must be a file with a safe mount path"
                )
            command += [
                "--mount",
                f"type=bind,src={source},dst=/engine-source/src/core/detector.py,readonly",
            ]
        if inference and self.runtime.device == "cuda":
            command += ["--gpus=all"]
        command += [image, "-I", "-u", "-c", BOOTSTRAP]
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
            await self.handshake(process, expected)
            self.accepted[inference] = image, expected
            return process
        except HandshakeError as error:
            self.incompatible_images[identity] = str(error)
            await self.close()
            errors.close()
            raise
        except BaseException:
            await self.close()
            errors.close()
            raise

    async def handshake(self, process: asyncio.subprocess.Process, expected: str) -> None:
        if process.stdin is None or process.stdout is None:
            raise HandshakeError("Worker hello has no communication channel")
        hello_message = WorkerHello(schema_version=1, implementation=expected)
        acknowledgement = hello_message.model_dump_json().encode() + b"\n"
        maximum = len(acknowledgement)
        buffer = b""
        while b"\n" not in buffer:
            chunk = await process.stdout.read(maximum + 1 - len(buffer))
            if not chunk:
                raise HandshakeError("Worker disconnected before implementation hello")
            buffer += chunk
            if len(buffer) > maximum:
                raise HandshakeError("Worker hello exceeds the fixed schema wire size")
        line, pending = buffer.split(b"\n", 1)
        if pending:
            raise HandshakeError("Unexpected output before worker implementation acknowledgement")
        try:
            hello = WorkerHello.model_validate(loads(line))
        except ValueError as error:
            raise HandshakeError("Malformed worker implementation hello") from error
        if hello.implementation != expected:
            raise HandshakeError(
                "Worker core identity differs; rebuild image from the current wheel"
            )
        process.stdin.write(acknowledgement)
        await process.stdin.drain()

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
            if response.isolation != "linux":
                raise CapabilityError("Worker isolation is not Linux")
            if request.operation == "generate":
                self.model_initialized = response.model_initialized
                if response.model_initialized:
                    self.loaded_identity = self.inference_identity()
                if response.native_tools is not None:
                    self.native_tools = response.native_tools
                if response.error and not response.model_initialized and response.capacity:
                    self.initialization_failure = self.inference_identity(), response.error
            if response.error:
                raise CapabilityError(response.error)
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
                        process.kill()
                    except ProcessLookupError:
                        pass
                await asyncio.shield(process.wait())
            finally:
                errors.close()
        self.processes.clear()
        self.buffers.clear()
        self.accepted.clear()
        self.loaded_identity = None

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int | None,
        seed: int | None,
        required_system_markers: list[str] | None = None,
    ) -> dict[str, Any]:
        if not max_output_tokens:
            raise CapabilityError("Inference requires max_output_tokens from user configuration")
        if self.loaded_identity is not None and self.loaded_identity != self.inference_identity():
            raise CapabilityError("Active source worker identity changed; create a new worker")
        if self.initialization_failure is not None:
            identity, error = self.initialization_failure
            if identity == self.inference_identity():
                raise CapabilityError("Source model initialization already failed: " + error)
            self.initialization_failure = None
            self.native_tools = None
            self.model_initialized = False
        if tools and self.native_tools is False:
            raise CapabilityError("Source backend does not support native structured tools")
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
                required_system_markers=required_system_markers or [],
            ),
            True,
        )
        return response.model_dump(mode="json", exclude_none=True)

    async def tool(
        self, name: str, arguments: dict[str, Any], fixture: str | None = None
    ) -> dict[str, Any]:
        response = await self.request(
            self.runtime.tool_image,
            WorkerRequest(operation="tool", name=name, arguments=arguments, fixture_text=fixture),
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

    async def discover(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self.output is None:
            raise CapabilityError("Research discovery requires a dedicated output mount")
        response = await self.request(
            self.runtime.discovery_image,
            WorkerRequest(
                operation="discovery",
                settings=settings,
                device=self.runtime.device,
                context_tokens=self.runtime.context_tokens,
            ),
            True,
        )
        if response.discovery is None:
            raise CapabilityError("Discovery worker returned no method result")
        return response.discovery

    async def activation(
        self,
        settings: dict[str, Any],
        phase: str = "fit",
        prompt: str | None = None,
        group: str | None = None,
    ) -> dict[str, Any]:
        if self.output is None:
            raise CapabilityError("Activation assessment requires a dedicated output mount")
        response = await self.request(
            self.runtime.activation_image,
            WorkerRequest.model_validate(
                {
                    "operation": "activation",
                    "settings": settings,
                    "phase": phase,
                    "prompt": prompt,
                    "group": group,
                    "device": self.runtime.device,
                    "context_tokens": self.runtime.context_tokens,
                }
            ),
            True,
        )
        if response.activation is None:
            raise CapabilityError("Activation worker returned no method result")
        return response.activation

    async def invert(
        self, settings: dict[str, Any], token_ids: list[int] | None = None
    ) -> dict[str, Any]:
        if self.output is None or self.engine_source is None:
            raise CapabilityError("Target inversion requires reviewed source and output mounts")
        response = await self.request(
            self.runtime.bait_image,
            WorkerRequest(
                operation="bait",
                settings=settings,
                phase="configure" if token_ids is None else "search",
                token_ids=token_ids or [],
                device=self.runtime.device,
                context_tokens=self.runtime.context_tokens,
            ),
            True,
        )
        if response.bait is None:
            raise CapabilityError("Target inversion worker returned no result")
        return response.bait

    async def fingerprint(
        self,
        settings: dict[str, Any],
        responses: list[list[str]] | None = None,
        phase: str = "configure",
    ) -> dict[str, Any]:
        if self.output is None:
            raise CapabilityError("Fingerprint assessment requires a dedicated output mount")
        response = await self.request(
            self.runtime.fingerprint_image,
            WorkerRequest.model_validate(
                {
                    "operation": "fingerprint",
                    "settings": settings,
                    "responses": responses or [],
                    "phase": phase,
                }
            ),
            True,
        )
        if response.fingerprint is None:
            raise CapabilityError("Fingerprint worker returned no result")
        return response.fingerprint
