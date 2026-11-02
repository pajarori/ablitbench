from __future__ import annotations

from typing import TYPE_CHECKING

from ablitbench.checks.artifacts import collect, output_directory
from ablitbench.models import CheckResult, Coverage
from ablitbench.transport import CapabilityError
from ablitbench.workers import Worker

if TYPE_CHECKING:
    from ablitbench.core import Context


async def run(context: Context) -> CheckResult:
    settings = context.config.activation
    if settings is None:
        return CheckResult(status="not_requested", reason="No activation experiment selected")
    total = 1 + 2 * len(settings["evaluation_pairs"]) + len(settings["utility_prompts"])
    result = CheckResult(coverage=Coverage(units="fit and held-out activation trials", total=total))
    output = output_directory(context)
    worker = Worker(
        context.config.runtime, context.worker.model, context.worker.base, output=output
    )
    try:
        context.checkpoint(result)
        record = await worker.activation(settings)
        result.evidence.append(context.store.evidence(record))
        result.coverage.inspected += 1
        context.checkpoint(result)
        for pair in settings["evaluation_pairs"]:
            for group in ("harmful", "harmless"):
                await context.yield_control()
                record = await worker.activation({}, "evaluate", pair[group], group)
                result.evidence.append(context.store.evidence(record))
                result.coverage.gaps.extend(record.get("gaps", []))
                result.coverage.inspected += 1
                context.checkpoint(result)
        for prompt in settings["utility_prompts"]:
            await context.yield_control()
            record = await worker.activation({}, "evaluate", prompt, "utility")
            result.evidence.append(context.store.evidence(record))
            result.coverage.gaps.extend(record.get("gaps", []))
            result.coverage.inspected += 1
            context.checkpoint(result)
    except CapabilityError as error:
        result.coverage.gaps.append(str(error))
    finally:
        try:
            await worker.close()
        finally:
            await collect(context, output, result)
    if result.coverage.gaps or result.coverage.gap_count:
        result.status, result.reason = "unsupported", "Activation evidence is incomplete"
    return result
