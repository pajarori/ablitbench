from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from ablitbench.checks.artifacts import capture, collect, output_directory
from ablitbench.formats.snapshot import secure_open
from ablitbench.io import operation as io_operation
from ablitbench.io_worker import prepare_model
from ablitbench.limits import available_memory
from ablitbench.models import CaseOutcome, CheckResult, Coverage, Finding, Target
from ablitbench.parity.artifacts import normalize_findings, normalize_outcomes
from ablitbench.parity.broker import HostBroker
from ablitbench.parity.catalog import (
    available_modes,
    effective_configuration,
    operation,
    requirements,
    requires_execution,
)
from ablitbench.parity.sources import prepare_source, source_identity
from ablitbench.privacy import private_directory
from ablitbench.targets import resolve
from ablitbench.transport import CapabilityError
from ablitbench.workers import Worker

if TYPE_CHECKING:
    from ablitbench.core import Context
    from ablitbench.models import Experiment


def remove_secrets(workspace: tempfile.TemporaryDirectory[str]) -> None:
    for secret in Path(workspace.name).iterdir():
        secret.chmod(0o600)
    workspace.cleanup()


async def import_findings(
    context: Context, output: Path, value: dict[str, Any], result: CheckResult, identity: str = ""
) -> None:
    await import_records(context, output, value, result, "findings", identity)
    await import_records(context, output, value, result, "outcomes", identity)


async def import_records(
    context: Context,
    output: Path,
    value: dict[str, Any],
    result: CheckResult,
    kind: str,
    identity: str,
) -> None:
    name = value.get(kind + "_artifact")
    if name is None:
        return
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise CapabilityError("Unsafe assessment artifact path")
    path = await io_operation("artifact_input", [str(output), relative.as_posix()])
    reference, stable = await context.io(capture, context, Path(path))
    context.store.commit_blob(reference)
    result.evidence.append(reference.sha256)
    if not stable:
        raise CapabilityError("Assessment artifact changed during capture")
    with tempfile.TemporaryDirectory(prefix="ablib-findings-") as temporary:
        destination = private_directory(temporary) / "validated.jsonl"
        observed = await context.io(
            normalize_findings if kind == "findings" else normalize_outcomes,
            context.store.blobs / reference.sha256,
            destination,
            context.check.id,
            context.target.target.location,
            *([] if kind == "findings" else [context.target.identity]),
        )
        capacity = available_memory()
        if context.config.runtime.memory_bytes is not None:
            capacity = min(capacity, context.config.runtime.memory_bytes)
        if observed["materialization_bytes"] > capacity:
            raise CapabilityError(
                "Assessment row exceeds measured host capacity; raw artifact retained"
            )
        with secure_open(destination) as stream:
            for line in stream:
                await context.yield_control()
                if kind == "findings":
                    result.findings.append(Finding.model_validate_json(line))
                else:
                    outcome = CaseOutcome.model_validate_json(line)
                    outcome.case_id = identity + ":" + outcome.case_id
                    outcome.evidence = [reference.sha256]
                    result.outcomes.append(outcome)
                context.checkpoint(result)
        if observed["error"]:
            result.coverage.gaps.append(
                "Assessment artifact incomplete at byte "
                + str(observed["uninspected_offset"])
                + ": "
                + observed["error"]
            )
        declared = value.get("finding_count" if kind == "findings" else "outcome_count")
        if type(declared) is not int or declared != observed["count"]:
            result.coverage.gaps.append(
                "Declared " + kind + " count differs from validated artifact rows"
            )


async def input_path(context: Context, value: str, index: int) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme == "artifact":
        record = context.store.get(f"experiment-output:{context.job.target}:" + parsed.netloc)
        if record is None:
            raise CapabilityError("Research artifact dependency has not completed")
        relative = PurePosixPath(parsed.path.lstrip("/"))
        if ".." in relative.parts or "\\" in parsed.path or parsed.query or parsed.fragment:
            raise CapabilityError("Unsafe research artifact dependency path")
        value = await io_operation("artifact_input", [record, relative.as_posix()])
    directory = context.store.directory / "experiment-inventory"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = await context.io(
        resolve, Target(location=value), "source", context.config.offline, directory, index
    )
    root = context.store.directory / "experiment-inputs" / target.identity
    only_file = target.root is not None and target.root.is_file()
    destination: Path = root
    for entry in target.entries():
        if entry.issue:
            raise CapabilityError("Research input inventory incomplete: " + entry.issue)
        copied = await context.snapshot(target, entry)
        destination = root / entry.name
        await context.io(prepare_model, copied, destination, root)
    return destination if only_file else root


async def prepare_inputs(
    context: Context,
    task: Experiment,
    metadata: dict[str, Any],
    index: int,
    secret_directory: Path,
    settings: dict[str, Any],
) -> dict[str, Path]:
    roles = set(metadata.get("inputs", []))
    optional = set(metadata.get("optional_inputs", []))
    prefixes = metadata.get("input_prefixes", [])
    optional.update(
        role for role in task.inputs if any(role.startswith(prefix) for prefix in prefixes)
    )
    secrets = set(metadata.get("secret_inputs", []))
    if (
        metadata.get("source_model")
        and context.config.mode == "source"
        and context.config.inference
        and requires_execution(task.project, task.operation, settings, "inference")
    ):
        roles.add("model")
        if context.base is not None:
            roles.add("base")
    if set(task.inputs) - roles - optional:
        raise CapabilityError("Research operation received undeclared input roles")
    result = {}
    for role in sorted(roles | (optional & task.inputs.keys())):
        if role in secrets:
            value = task.inputs.get(role)
            if value is None or (not Path(value).is_absolute() and urlsplit(value).scheme):
                raise CapabilityError("Secret input requires an explicit local file role: " + role)
            if metadata.get("requires_inference") or "custom-model-code" in task.permissions:
                raise CapabilityError("Secret material must use a separate signing operation")
            result[role] = secret_directory / role
            await io_operation("prepare_secret", [value, str(result[role])])
        elif role == "source":
            source = task.inputs.get(role) or context.config.runtime.parity_sources.get(
                task.project
            )
            if source is None:
                raise CapabilityError("Reviewed local source tree is required for " + task.project)
            root = context.store.directory / "research-sources" / source_identity(task.project)
            result[role] = Path(await context.io(prepare_source, Path(source), task.project, root))
        elif role in task.inputs:
            result[role] = await input_path(context, task.inputs[role], uuid.uuid4().int)
        elif role == "model" and context.config.mode == "source":
            result[role] = (
                await input_path(context, context.target.target.location, uuid.uuid4().int)
                if metadata.get("model_input") == "artifact"
                else await context.model_snapshot(context.target)
            )
        elif role == "base" and context.base is not None:
            result[role] = await context.model_snapshot(context.base)
        elif len(roles - {"source"}) == 1 and context.config.mode == "source":
            result[role] = await input_path(
                context, context.target.target.location, uuid.uuid4().int
            )
        else:
            raise CapabilityError("Research input role is required: " + role)
    return result


async def run(context: Context) -> CheckResult:
    tasks = context.config.experiments
    previous = context.store.previous(context.job) if context.config.resume else None
    result = (previous[1] if previous and previous[1] else None) or CheckResult(
        coverage=Coverage(units="explicit research operations", total=len(tasks))
    )
    result.status, result.reason = "completed", None
    failures = False
    for index, task in enumerate(tasks):
        await context.yield_control()
        context.checkpoint(result)
        identity = task.id or task.project + "." + task.operation + "." + str(index)
        state_key = f"experiment-state:{context.job.target}:" + identity
        scope = hashlib.sha256(
            f"{context.store.get('run_id')}:{context.job.target}:{identity}".encode()
        ).hexdigest()
        state = context.store.get(state_key) if context.config.resume else None
        if state is not None:
            containers = set(json.loads(context.store.get(state_key + ":containers") or "[]"))
            container = context.store.get(state_key + ":container")
            if container is not None:
                containers.add(container)
            for member in containers:
                await Worker(context.config.runtime).recover(member, scope)
            environment_container = context.store.get(state_key + ":environment-container")
            if environment_container is not None:
                environment_scope = hashlib.sha256((scope + ":environment").encode()).hexdigest()
                await Worker(context.config.runtime).recover(
                    environment_container, environment_scope
                )
        if state == "completed":
            continue
        if state in {"started", "incomplete"}:
            result.coverage.gaps.append(
                identity + ": prior delivery is uncertain; not executed again"
            )
            context.store.set(state_key, "uncertain")
            context.checkpoint(result)
            continue
        if state in {"uncertain", "unsupported", "inconclusive", "error"}:
            failures = failures or state == "error"
            continue
        output = output_directory(context)
        worker: Worker | None = None
        broker = HostBroker(context, task)
        broker.scope, broker.state_key = scope, state_key
        context.store.set(state_key, "prepared")
        secret_workspace = tempfile.TemporaryDirectory(prefix="ablib-secrets-")
        secret_directory = private_directory(secret_workspace.name)
        try:
            _, metadata = operation(task.project, task.operation)
            settings = effective_configuration(
                task.project, task.operation, task.settings, context.target.target.model_dump()
            )
            if context.config.mode not in available_modes(task.project, task.operation, settings):
                raise CapabilityError("Research operation is unavailable in this mode")
            if requirements(task.project, task.operation, settings) - set(task.permissions):
                raise CapabilityError("Research permissions are not granted by this invocation")
            if (
                settings.get("private_platform")
                and set(metadata.get("secret_inputs", [])) & task.inputs.keys()
            ):
                raise CapabilityError(
                    "Private platform fixtures cannot receive operator secret roles"
                )
            if (
                (
                    requires_execution(task.project, task.operation, settings, "inference")
                    or "generation-broker" in metadata.get("capabilities", [])
                )
                and context.config.mode == "source"
                and not context.config.inference
            ):
                raise CapabilityError("Research operation requires explicit source inference")
            if (
                requires_execution(task.project, task.operation, settings, "tools")
                and not context.config.execute_tools
            ):
                raise CapabilityError("Research operation requires explicit tool execution")
            inputs = await prepare_inputs(
                context, task, metadata, index, secret_directory, settings
            )
            broker.inputs = dict(inputs)
            for role in metadata.get("observer_inputs", []):
                inputs.pop(role, None)
            if context.config.mode == "source" and context.config.inference:
                if "generation-broker" in metadata.get("capabilities", []) and not settings.get(
                    "local_model"
                ):
                    context.worker.model = await context.model_snapshot(context.target)
                    if context.base is not None:
                        context.worker.base = await context.model_snapshot(context.base)
                elif "model" in inputs and metadata.get("model_input") != "artifact":
                    context.worker.model = inputs["model"]
                    context.worker.base = inputs.get("base", context.worker.base)
            context.store.set(state_key, "started")
            if metadata.get("trusted_io"):
                value = await io_operation(
                    "trusted_acquire",
                    [
                        {
                            "operation": task.operation,
                            "settings": settings,
                            "inputs": {role: str(path) for role, path in inputs.items()},
                            "output": str(output),
                            "project": task.project,
                            "offline": context.config.offline,
                        }
                    ],
                    memory=context.config.runtime.memory_bytes,
                    executable=context.config.runtime.acquisition_python,
                    credentials=settings.get("credential_env", {}),
                )
            else:
                worker = Worker(context.config.runtime, output=output, inputs=inputs)
                worker.scope = scope
                worker.on_protocol_error = context.protocol_error

                def created(name: str, key: str = state_key) -> None:
                    names = json.loads(context.store.get(key + ":containers") or "[]")
                    if name not in names:
                        names.append(name)
                        context.store.set(key + ":containers", json.dumps(names))
                    context.store.set(key + ":container", name)

                worker.on_container_created = created
                value = await worker.experiment(
                    task.project,
                    task.operation,
                    settings,
                    broker,
                    task.permissions,
                    mode=context.config.mode,
                    inference=context.config.inference,
                    execute_tools=context.config.execute_tools,
                )
                await worker.close()
            prior_gaps = result.coverage.gap_count + len(result.coverage.gaps)
            await import_findings(context, output, value, result, identity)
            artifacts_complete = prior_gaps == result.coverage.gap_count + len(result.coverage.gaps)
            result.evidence.append(
                context.store.evidence({"experiment": identity, "result": value})
            )
            gaps = list(value.get("gaps", []))
            nested_coverage = value.get("coverage", {})
            if isinstance(nested_coverage, dict):
                gaps.extend(nested_coverage.get("gaps", []))
            result.coverage.gaps.extend(str(gap) for gap in gaps)
            status = value.get("status", "unconfirmed")
            if status not in {"completed", "success", "ok"} and not gaps:
                result.coverage.gaps.append(identity + ": " + str(status))
            if value.get("outcomes"):
                result.coverage.gaps.append("Native outcomes require a validated JSONL artifact")
                artifacts_complete = False
            inline_findings = value.get("findings", [])
            if value.get("findings_artifact") is not None and inline_findings:
                result.coverage.gaps.append("Ambiguous duplicate findings representations")
                artifacts_complete = False
                inline_findings = []
            for finding in inline_findings:
                result.findings.append(
                    Finding.model_validate(
                        {
                            **finding,
                            "check": context.check.id,
                            "target": context.target.target.location,
                        }
                    )
                )
            if artifacts_complete and not gaps and status in {"completed", "success", "ok"}:
                result.coverage.inspected += 1
                context.store.set(
                    f"experiment-output:{context.job.target}:" + identity, str(output)
                )
                context.store.set(state_key, "completed")
            else:
                failed = status in {"failed", "error"}
                failures = failures or failed
                context.store.set(
                    state_key,
                    "error"
                    if failed
                    else "unsupported"
                    if status == "unsupported"
                    else "inconclusive",
                )
        except (CapabilityError, ValueError, ImportError) as error:
            result.coverage.gaps.append(identity + ": " + str(error))
            context.store.set(state_key, "unsupported")
        except OSError as error:
            result.coverage.gaps.append(identity + ": " + str(error))
            context.store.set(state_key, "error")
            failures = True
        finally:
            try:
                if worker is not None:
                    await worker.close()
            finally:
                try:
                    await broker.close()
                    await collect(context, output, result)
                    for extra_output in worker.extra_outputs if worker is not None else []:
                        await collect(context, extra_output, result)
                    for environment_output in broker.outputs:
                        await collect(context, environment_output, result)
                    context.checkpoint(result)
                finally:
                    remove_secrets(secret_workspace)
    if result.coverage.gaps or result.coverage.gap_count:
        result.status = "error" if failures else "unsupported"
        result.reason = "Research operation coverage is incomplete"
    return result
