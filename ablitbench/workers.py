from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from ablitbench.formats.json import loads
from ablitbench.io import operation as io_operation
from ablitbench.limits import available_memory
from ablitbench.models import (
    EnvironmentRequest,
    OutputAudioFormat,
    OutputModality,
    Runtime,
    WorkerHello,
    WorkerRequest,
    WorkerResponse,
)
from ablitbench.privacy import private_directory
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


def missing_container(diagnostic: bytes) -> bool:
    detail = diagnostic.decode(errors="replace").lower()
    return any(
        message in detail
        for message in (
            "no such container",
            "no such object",
            "no container with name",
            "no container with id or name",
            "no container with name or id",
        )
    )


def cluster_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="cluster-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "cluster-result.json")
    finally:
        Path(temporary).unlink(missing_ok=True)


class Worker:
    def __init__(
        self,
        runtime: Runtime,
        model: Path | None = None,
        base: Path | None = None,
        output: Path | None = None,
        engine_source: Path | None = None,
        inputs: dict[str, Path] | None = None,
    ) -> None:
        self.runtime, self.model, self.base = runtime, model, base
        self.output = output
        self.engine_source = engine_source
        self.inputs = dict(inputs or {})
        self.on_protocol_error: Callable[[Path], Awaitable[None]] | None = None
        self.on_container_created: Callable[[str], None] | None = None
        self.scope: str | None = None
        self.research_network: str | None = None
        self.processes: dict[bool, tuple[asyncio.subprocess.Process, str, BinaryStream]] = {}
        self.buffers: dict[bool, bytes] = {}
        self.containers: set[str] = set()
        self.accepted: dict[bool, tuple[str, str]] = {}
        self.incompatible_images: dict[tuple[str, str, str], str] = {}
        self.native_tools: bool | None = None
        self.model_initialized = False
        self.accelerator = False
        self.gpu_devices: list[str] | None = None
        self.peer_workers: list[Worker] = []
        self.close_lock = asyncio.Lock()
        self.extra_outputs: list[Path] = []
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

    def input_destination(self, role: str, path: Path) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", role):
            raise CapabilityError("Invalid research input mount role")
        destination = "/inputs/" + role
        if path.is_file():
            if path.name in {"", ".", ".."} or any(
                value in path.name for value in ("/", "\\", "\0", ",")
            ):
                raise CapabilityError("Research input requires a safe original basename")
            destination += "/" + path.name
        return destination

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
        if self.research_network is not None:
            await self.validate_network(self.research_network)
        command = [
            self.runtime.engine,
            "create",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=" + (self.research_network or "none"),
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            "--memory",
            str(self.runtime.memory_bytes),
            "--memory-swap",
            str(self.runtime.memory_bytes),
            "--ipc=private",
            "--shm-size",
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
        if self.scope is not None:
            command += [
                "--label",
                "ablitbench.scope=" + self.scope,
                "--label",
                "ablitbench.implementation=" + expected,
            ]
        if self.research_network is not None:
            command += [
                "--env",
                "ABLIB_CLUSTER_NETWORK=" + self.research_network,
                "--sysctl",
                "net.ipv6.conf.all.disable_ipv6=1",
            ]
            if self.runtime.network_alias is not None:
                command += ["--network-alias", self.runtime.network_alias]
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
        for role, path in self.inputs.items():
            if not re.fullmatch(r"[a-z][a-z0-9_-]*", role):
                raise CapabilityError("Invalid research input mount role")
            if path.is_symlink():
                raise CapabilityError("Research input root symlink rejected")
            source = path.resolve(strict=True)
            if "," in str(source) or not (source.is_file() or source.is_dir()):
                raise CapabilityError("Research input must be a prepared regular file or directory")
            destination = self.input_destination(role, path)
            command += ["--mount", f"type=bind,src={source},dst={destination},readonly"]
        if (inference or self.accelerator) and self.runtime.device == "cuda":
            if self.gpu_devices is None:
                command += ["--gpus=all"]
            else:
                selection = ",".join(self.gpu_devices)
                command += ["--gpus", '"device=' + selection + '"']
        command += [image, "-I", "-u", "-c", BOOTSTRAP]
        self.containers.add(name)
        if self.on_container_created is not None:
            self.on_container_created(name)
        errors = tempfile.TemporaryFile()

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

    async def validate_network(self, identity: str) -> None:
        if self.runtime.engine != "docker" or self.scope is None:
            raise CapabilityError(
                "Controlled distributed networking requires a run-owned Docker worker"
            )
        process = await asyncio.create_subprocess_exec(
            "docker",
            "network",
            "inspect",
            identity,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, diagnostic = await process.communicate()
        if process.returncode:
            raise CapabilityError(
                "Cannot inspect research network: " + diagnostic.decode(errors="replace")
            )
        network = loads(output)[0]
        if (
            network.get("Id") != identity
            or network.get("Driver") != "bridge"
            or network.get("Internal") is not True
            or network.get("EnableIPv6") is not False
            or network.get("Ingress")
            or network.get("Scope") != "local"
            or network.get("Labels", {}).get("ablitbench.research") != "true"
            or network.get("Options", {}).get("com.docker.network.bridge.gateway_mode_ipv4")
            != "isolated"
        ):
            raise CapabilityError(
                "Research network must be a labeled internal isolated IPv4 bridge"
            )
        for identity in network.get("Containers", {}):
            peer = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--type=container",
                identity,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            raw, diagnostic = await peer.communicate()
            if peer.returncode:
                raise CapabilityError("Research peer inspection failed")
            record = loads(raw)[0]
            configuration = record.get("HostConfig", {})
            labels = record.get("Config", {}).get("Labels", {})
            mounts = record.get("Mounts", [])
            if (
                labels.get("ablitbench.implementation") != implementation_identity()
                or labels.get("ablitbench.scope") != self.scope
                or configuration.get("Privileged")
                or configuration.get("CapAdd")
                or configuration.get("Devices")
                or configuration.get("PidMode", "") not in {"", "private"}
                or configuration.get("UTSMode", "") not in {"", "private"}
                or configuration.get("IpcMode", "") not in {"", "private"}
                or any(
                    not str(option).startswith("no-new-privileges")
                    for option in configuration.get("SecurityOpt", [])
                )
                or not any(
                    str(option).split(":", 1)[0].split("=", 1)[0] == "no-new-privileges"
                    for option in configuration.get("SecurityOpt", [])
                )
                or int(configuration.get("Memory", 0)) <= 0
                or int(configuration.get("NanoCpus", 0)) <= 0
                or int(configuration.get("PidsLimit") or 0) <= 0
                or record.get("Config", {}).get("User") in {None, "", "0", "root", "0:0"}
                or not configuration.get("ReadonlyRootfs")
                or "ALL" not in configuration.get("CapDrop", [])
                or configuration.get("PortBindings")
                or configuration.get("PublishAllPorts")
                or len(record.get("NetworkSettings", {}).get("Networks", {})) != 1
                or any(
                    not (
                        mount.get("Destination") in {"/model", "/base", "/output"}
                        or mount.get("Destination", "").startswith("/inputs/")
                    )
                    or (mount.get("RW") and mount.get("Destination") != "/output")
                    for mount in mounts
                )
            ):
                raise CapabilityError(
                    "Research network contains a peer outside the sandbox contract"
                )

    async def recover(self, name: str, scope: str) -> None:
        if re.fullmatch(r"ablib-[a-f0-9]{32}", name) is None:
            raise CapabilityError("Saved worker container identity is invalid")
        process = await asyncio.create_subprocess_exec(
            self.runtime.engine,
            "inspect",
            "--type=container",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, diagnostic = await process.communicate()
        if process.returncode:
            detail = diagnostic.decode(errors="replace").lower()
            if missing_container(diagnostic):
                return
            raise CapabilityError("Cannot inspect the prior research worker: " + detail)
        record = loads(output)
        labels = record[0].get("Config", {}).get("Labels", {})
        if (
            labels.get("ablitbench.scope") != scope
            or labels.get("ablitbench.implementation") != implementation_identity()
        ):
            raise CapabilityError("Prior worker ownership differs; container was not removed")
        cleanup = await asyncio.create_subprocess_exec(
            self.runtime.engine,
            "rm",
            "--force",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, diagnostic = await cleanup.communicate()
        if cleanup.returncode and not missing_container(diagnostic):
            raise CapabilityError(
                "Prior worker cleanup failed: " + diagnostic.decode(errors="replace")
            )

    async def read_response(
        self, process: asyncio.subprocess.Process, inference: bool
    ) -> WorkerResponse:
        if process.stdout is None:
            raise CapabilityError("Worker has no output channel")
        with tempfile.TemporaryDirectory(
            prefix="ablib-worker-frame-", dir=self.output.parent if self.output else None
        ) as temporary:
            private = private_directory(temporary)
            raw, normalized = private / "wire", private / "response.json"
            try:
                with raw.open("wb") as stream:
                    chunk = self.buffers[inference]
                    while True:
                        if b"\n" in chunk:
                            frame, self.buffers[inference] = chunk.split(b"\n", 1)
                            stream.write(frame)
                            break
                        stream.write(chunk)
                        chunk = await process.stdout.read(1024 * 1024)
                        if not chunk:
                            raise CapabilityError("Sandbox disconnected before a complete response")
                    stream.flush()
                measured = await io_operation(
                    "normalize_worker_response",
                    [str(raw), str(normalized)],
                    memory=self.runtime.memory_bytes,
                )
                if measured["materialization_bytes"] > available_memory():
                    raise CapabilityError("Worker response exceeds measured host memory capacity")
                return self.parse_response(normalized)
            except BaseException:
                await self.close()
                if raw.exists():
                    if self.on_protocol_error is not None:
                        await self.on_protocol_error(raw)
                    elif self.output is not None:
                        os.replace(
                            raw, self.output / ("incomplete-worker-response-" + uuid.uuid4().hex)
                        )
                raise

    @staticmethod
    def parse_response(path: Path) -> WorkerResponse:
        with path.open("rb") as stream:
            return WorkerResponse.model_validate_json(stream.read())

    async def request(
        self,
        image: str | None,
        request: WorkerRequest,
        inference: bool,
        on_exchange: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
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
            while True:
                response = await self.read_response(process, inference)
                if response.implementation != implementation_identity():
                    raise CapabilityError(
                        "Worker core identity differs; rebuild image from the current wheel"
                    )
                if response.isolation != "linux":
                    raise CapabilityError("Worker isolation is not Linux")
                if response.exchange is None:
                    break
                exchange = response.exchange
                if (
                    request.operation != "experiment"
                    or on_exchange is None
                    or not isinstance(exchange.get("id"), str)
                    or not isinstance(exchange.get("request"), dict)
                ):
                    raise CapabilityError("Unexpected sandbox broker request")
                value = await on_exchange(exchange["request"])
                reply = WorkerRequest(
                    operation="exchange",
                    exchange={"id": exchange["id"], "response": value},
                )
                process.stdin.write(reply.model_dump_json().encode() + b"\n")
                await process.stdin.drain()
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

    async def experiment(
        self,
        project: str,
        operation: str,
        settings: dict[str, Any],
        broker: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        permissions: Sequence[str] | None = None,
        *,
        mode: Literal["api", "source"] = "source",
        inference: bool = False,
        execute_tools: bool = False,
    ) -> dict[str, Any]:
        if not settings.get("managed_cluster"):
            return await self.single_experiment(
                project,
                operation,
                settings,
                broker,
                permissions,
                mode=mode,
                inference=inference,
                execute_tools=execute_tools,
            )
        if project != "backdoorllm" or operation != "factory.distributed":
            raise CapabilityError(
                "Managed clusters require the native distributed factory workflow"
            )
        from ablitbench.parity.factory_worker import configuration

        settings = configuration(operation, settings)
        if settings["device"] != self.runtime.device:
            raise CapabilityError("Cluster device and sandbox runtime must match")
        topology = settings.get("topology", {})
        if (
            self.runtime.engine != "docker"
            or project != "backdoorllm"
            or operation != "factory.distributed"
            or not isinstance(topology, dict)
            or type(topology.get("nodes")) is not int
            or topology["nodes"] < 2
            or topology.get("node_rank") != 0
            or not self.scope
            or not self.runtime.research_network
            or "external-network" not in (permissions or [])
            or not self.runtime.network_alias
            or topology.get("master_address") != self.runtime.network_alias
            or self.output is None
        ):
            raise CapabilityError("Managed cluster requires a rank-zero scoped private topology")
        count = topology["nodes"]
        allocations = settings.get("cluster_devices")
        if self.runtime.device == "cuda":
            if not isinstance(allocations, list) or len(allocations) != count:
                raise CapabilityError("CUDA cluster requires explicit per-node GPU UUID allocation")
            self.gpu_devices = allocations[0]
        elif allocations is not None:
            raise CapabilityError("GPU allocations are not applicable to a CPU cluster")
        process = await asyncio.create_subprocess_exec(
            self.runtime.engine,
            "info",
            "--format",
            "{{json .}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw, diagnostic = await process.communicate()
        if process.returncode:
            raise CapabilityError(
                "Cannot measure cluster capacity: " + diagnostic.decode(errors="replace")
            )
        machine = loads(raw)
        if (
            not self.runtime.memory_bytes
            or not self.runtime.cpus
            or count * self.runtime.memory_bytes > machine.get("MemTotal", 0)
            or count * self.runtime.cpus > machine.get("NCPU", 0)
        ):
            raise CapabilityError(
                "Requested simultaneous cluster exceeds Docker memory/CPU capacity"
            )
        members = [self]
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        try:
            for rank in range(1, count):
                directory = self.output.parent / uuid.uuid4().hex
                directory.mkdir(mode=0o777)
                os.chmod(directory, 0o777)
                peer = Worker(
                    self.runtime.model_copy(
                        update={"network_alias": f"ablib-node-{rank}-{self.scope[:12]}"}
                    ),
                    self.model,
                    self.base,
                    directory,
                    self.engine_source,
                    self.inputs,
                )
                peer.scope = self.scope
                peer.on_container_created = self.on_container_created
                peer.on_protocol_error = self.on_protocol_error
                peer.gpu_devices = allocations[rank] if allocations is not None else None
                self.peer_workers.append(peer)
                self.extra_outputs.append(directory)
                members.append(peer)
            async with asyncio.TaskGroup() as group:
                for rank, member in enumerate(members):
                    configured = {
                        **settings,
                        "managed_cluster": False,
                        "topology": {**topology, "node_rank": rank},
                    }
                    tasks.append(
                        group.create_task(
                            member.cluster_member(
                                project,
                                operation,
                                configured,
                                broker,
                                permissions,
                                mode=mode,
                                inference=inference,
                                execute_tools=execute_tools,
                            )
                        )
                    )
            result = tasks[0].result()
            return {
                **result,
                "cluster_nodes": count,
                "cluster_verified": "all native participants completed",
            }
        finally:
            await self.close()
            manifest = {
                "nodes": count,
                "local_ranks": settings.get("world_size"),
                "topology": topology,
                "gpu_allocations": allocations,
                "members": [
                    {
                        "rank": rank,
                        "cancelled": task.cancelled(),
                        "error": type(task.exception()).__name__
                        if not task.cancelled() and task.exception()
                        else None,
                        "result": task.result()
                        if not task.cancelled() and task.exception() is None
                        else None,
                    }
                    for rank, task in enumerate(tasks)
                ],
            }
            await asyncio.to_thread(cluster_manifest, self.output, manifest)

    async def cluster_member(
        self,
        project: str,
        operation: str,
        settings: dict[str, Any],
        broker: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        permissions: Sequence[str] | None = None,
        *,
        mode: Literal["api", "source"] = "source",
        inference: bool = False,
        execute_tools: bool = False,
    ) -> dict[str, Any]:
        value = await self.single_experiment(
            project,
            operation,
            settings,
            broker,
            permissions,
            mode=mode,
            inference=inference,
            execute_tools=execute_tools,
        )
        if value.get("status") not in {"completed", "success", "ok"}:
            raise CapabilityError("Cluster member did not complete: " + str(value.get("status")))
        return value

    async def single_experiment(
        self,
        project: str,
        operation: str,
        settings: dict[str, Any],
        broker: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        permissions: Sequence[str] | None = None,
        *,
        mode: Literal["api", "source"] = "source",
        inference: bool = False,
        execute_tools: bool = False,
    ) -> dict[str, Any]:
        image = self.runtime.parity_images.get(project)
        self.accelerator = True
        self.research_network = (
            self.runtime.research_network if "external-network" in (permissions or []) else None
        )
        response = await self.request(
            image,
            WorkerRequest(
                operation="experiment",
                project=project,
                experiment_operation=operation,
                device=self.runtime.device,
                mode=mode,
                inference=inference,
                execute_tools=execute_tools,
                settings=settings,
                permissions=list(permissions or []),
                arguments={
                    "inputs": {
                        role: self.input_destination(role, path)
                        for role, path in self.inputs.items()
                    }
                },
            ),
            False,
            broker,
        )
        if response.experiment is None:
            raise CapabilityError("Research worker returned no experiment result")
        return response.experiment

    async def environment(self, value: dict[str, Any]) -> dict[str, Any]:
        request = EnvironmentRequest.model_validate(value)
        response = await self.request(
            self.runtime.environment_image or self.runtime.tool_image,
            WorkerRequest(
                operation="environment",
                execute_tools=True,
                environment_request=request,
                arguments={
                    "inputs": {
                        role: self.input_destination(role, path)
                        for role, path in self.inputs.items()
                    }
                },
            ),
            False,
        )
        if response.result is None:
            raise CapabilityError("Environment executor returned no transition")
        return response.result

    async def close(self) -> None:
        async with self.close_lock:
            peer_errors = await asyncio.gather(
                *(peer.close() for peer in self.peer_workers), return_exceptions=True
            )
            self.peer_workers.clear()
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
                if cleanup.returncode and not missing_container(diagnostic):
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
            for error in peer_errors:
                if isinstance(error, BaseException):
                    raise error

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int | None,
        seed: int | None,
        required_system_markers: list[str] | None = None,
        generation_settings: dict[str, Any] | None = None,
        output_modalities: list[OutputModality] | None = None,
        output_audio_format: OutputAudioFormat | None = None,
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
                generation_settings=generation_settings or {},
                output_modalities=output_modalities if output_modalities is not None else ["text"],
                output_audio_format=output_audio_format,
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
