from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ablitbench.models import CheckResult, Coverage, Finding
from ablitbench.workers import Worker

if TYPE_CHECKING:
    from ablitbench.core import Context


async def run(context: Context) -> CheckResult:
    names = context.config.runtime.research_engines
    if not names:
        return CheckResult(status="not_requested", reason="No research engines selected")
    result = CheckResult(
        coverage=Coverage(units="engine/file assessments", total=len(names) * context.target.files)
    )
    root = context.store.directory / "snapshots" / context.target.identity
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    worker = Worker(context.config.runtime, root)
    try:
        for entry in context.target.entries():
            context.checkpoint(result)
            if entry.issue:
                result.coverage.gaps.append(entry.name + ": " + entry.issue)
                continue
            path = await context.snapshot(context.target, entry)
            for name in names:
                response = await worker.scan(name, "/model/" + path.name)
                evidence = context.store.evidence({"file": entry.name, **response})
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
                                "evidence": [evidence],
                            }
                        )
                    )
                if not response["gaps"]:
                    result.coverage.inspected += 1
                context.checkpoint(result)
    finally:
        await worker.close()
    if result.coverage.gaps or result.coverage.gap_count:
        result.status, result.reason = (
            "unsupported",
            "Research engine reported skipped/incomplete coverage",
        )
    return result
