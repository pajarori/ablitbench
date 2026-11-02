from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ablitbench.formats.archive import unsafe_path
from ablitbench.io_worker import prepare_model, remove_research
from ablitbench.models import CheckResult, Coverage, Finding
from ablitbench.transport import CapabilityError
from ablitbench.workers import Worker

if TYPE_CHECKING:
    from ablitbench.core import Context


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def run(context: Context) -> CheckResult:
    names = context.config.runtime.research_engines
    if not names:
        return CheckResult(status="not_requested", reason="No research engines selected")
    result = CheckResult(coverage=Coverage(units="engine/package assessments", total=len(names)))
    root = context.store.directory / "research" / context.target.identity
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    try:
        with tempfile.TemporaryFile() as manifest:
            for entry in context.target.entries():
                context.checkpoint(result)
                if entry.issue or unsafe_path(entry.name):
                    result.coverage.gaps.append(entry.name + ": " + (entry.issue or "Unsafe path"))
                    continue
                try:
                    source = await context.snapshot(context.target, entry)
                except (OSError, ValueError) as error:
                    result.coverage.gaps.append(entry.name + ": " + str(error))
                    continue
                destination = root / entry.name
                await context.io(prepare_model, source, destination, root)
                manifest.write(
                    (
                        json.dumps(
                            {
                                "path": entry.name,
                                "sha256": entry.sha256 or await context.io(file_hash, source),
                                "size": entry.size,
                            }
                        )
                        + "\n"
                    ).encode()
                )
            manifest.seek(0)
            identity = context.store.blob(
                iter(lambda: manifest.read(1024 * 1024), b""), "application/jsonl"
            )
        result.evidence.append(identity.sha256)
        worker = Worker(context.config.runtime, root)
        try:
            for name in names:
                context.checkpoint(result)
                try:
                    response = await worker.scan(name, "/model")
                except (CapabilityError, ValueError, OSError) as error:
                    result.coverage.gaps.append(name + ": " + str(error))
                    context.checkpoint(result)
                    continue
                evidence = context.store.evidence(
                    {
                        "target_identity": context.target.identity,
                        "manifest": identity.sha256,
                        **response,
                    }
                )
                result.evidence.append(evidence)
                for gap in response["gaps"]:
                    result.coverage.gaps.append(name + ": " + json.dumps(gap))
                for issue in response["issues"]:
                    severity = str(issue.get("severity", "medium")).lower()
                    result.findings.append(
                        Finding.model_validate(
                            {
                                "check": context.check.id,
                                "target": context.target.target.location,
                                "title": name + " static indicator",
                                "severity": severity
                                if severity in {"info", "low", "medium", "high", "critical"}
                                else "medium",
                                "detail": json.dumps(issue),
                                "evidence": [evidence, identity.sha256],
                            }
                        )
                    )
                if not response["gaps"]:
                    result.coverage.inspected += 1
                context.checkpoint(result)
        finally:
            await worker.close()
    finally:
        await context.io(remove_research, context.store.directory, context.target.identity)
    if result.coverage.gaps or result.coverage.gap_count:
        result.status, result.reason = "unsupported", "Research engine reported incomplete coverage"
    return result
