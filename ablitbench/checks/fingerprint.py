from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ablitbench.checks.artifacts import collect, output_directory
from ablitbench.fingerprint_engine import configuration
from ablitbench.io_worker import prepare_model
from ablitbench.models import CheckResult, Coverage, FileEntry, Finding, ResolvedTarget, Target
from ablitbench.transport import CapabilityError
from ablitbench.workers import Worker

if TYPE_CHECKING:
    from ablitbench.core import Context


async def prepare(context: Context, information: dict[str, Any]) -> Path:
    root = Path(information["bundle"])
    identity = information["identity"]
    target = ResolvedTarget(target=Target(location=str(root)), identity="fingerprint-" + identity)
    destination = context.store.directory / "fingerprint-bundles" / target.identity
    inventory = information["manifest"]["files"] | {"bundle.json": identity}
    for name, checksum in inventory.items():
        source = await context.snapshot(
            target, FileEntry(name=name, path=root / name, sha256=checksum)
        )
        path = destination / name
        await context.io(prepare_model, source, path, destination)
    return destination


async def run(context: Context) -> CheckResult:
    settings = context.config.fingerprint
    if settings is None:
        return CheckResult(status="not_requested", reason="No calibrated fingerprint selected")
    information = configuration(settings)
    queries = information["queries"]
    repeats = settings["repeats"]
    result = CheckResult(
        coverage=Coverage(
            units="initialization, calibrated query responses, panels and summary",
            total=2 + repeats * (len(queries) + 1),
        )
    )
    context.checkpoint(result)
    root = await prepare(context, information)
    output = output_directory(context)
    worker = Worker(context.config.runtime, root, output=output)
    try:
        worker_settings = settings | {"bundle": "/model"}
        metadata = await worker.fingerprint(worker_settings)
        result.evidence.append(context.store.evidence(metadata))
        result.coverage.inspected += 1
        context.checkpoint(result)
        for repeat in range(repeats):
            answers: list[str] = []
            for index, query in enumerate(queries):
                await context.yield_control()
                generated = await context.generate([{"role": "user", "content": query}])
                result.evidence.extend(generated.evidence)
                result.evidence.append(
                    context.store.evidence(
                        {
                            "method": "calibrated fingerprint query",
                            "repeat": repeat,
                            "query_index": index,
                            "generation": generated.model_dump(),
                        }
                    )
                )
                if not generated.complete:
                    raise CapabilityError("Fingerprint panel interrupted by provider termination")
                if generated.tools:
                    raise CapabilityError("Fingerprint text panel received native tool calls")
                if not generated.text.strip():
                    raise CapabilityError(
                        "Fingerprint panel lacks a final text answer; "
                        "refusal or reasoning alone cannot establish model identity"
                    )
                answers.append(generated.text)
                result.coverage.inspected += 1
                context.checkpoint(result)
            record = await worker.fingerprint({}, [answers], "evaluate")
            result.evidence.append(context.store.evidence(record))
            result.coverage.inspected += 1
            context.checkpoint(result)
        summary = await worker.fingerprint({}, phase="finalize")
        evidence = context.store.evidence(summary)
        result.evidence.append(evidence)
        result.coverage.inspected += 1
        reference = settings.get("expected_reference")
        matched = summary.get("matched_reference")
        if (
            summary.get("membership") == "known-reference-match"
            and reference
            and matched != reference
        ):
            result.findings.append(
                Finding(
                    check=context.check.id,
                    target=context.target.target.location,
                    detector="calibrated-fingerprint-reference-mismatch",
                    title="Behavioral fingerprint differs from the configured reference",
                    severity="medium",
                    strength="indicator",
                    attribution="Model, routing or prompt behavior; substitution unproven",
                    detail=(
                        f"Configured reference {reference}; matched calibrated reference {matched}"
                    ),
                    evidence=[evidence],
                )
            )
        elif summary.get("membership") != "known-reference-match":
            result.coverage.gaps.append(
                "Fingerprint outside calibration or inconclusive; no model identity asserted"
            )
        context.checkpoint(result)
    except CapabilityError as error:
        result.coverage.gaps.append(str(error))
    finally:
        try:
            await worker.close()
        finally:
            await collect(context, output, result)
    if result.coverage.gaps or result.coverage.gap_count:
        result.status, result.reason = "unsupported", "Fingerprint assessment is inconclusive"
    return result
