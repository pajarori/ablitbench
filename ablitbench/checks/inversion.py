from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ablitbench.bait_engine import SOURCE_SHA256
from ablitbench.checks.artifacts import collect, output_directory
from ablitbench.io_worker import prepare_model
from ablitbench.models import CaseEvaluation, CaseOutcome, CheckResult, Coverage, Finding, Target
from ablitbench.targets import resolve
from ablitbench.transport import CapabilityError
from ablitbench.workers import Worker

if TYPE_CHECKING:
    from ablitbench.core import Context


async def reviewed_source(context: Context) -> Path:
    path = context.config.runtime.bait_source
    if path is None:
        raise CapabilityError("Reviewed local BAIT source file is required")
    directory = context.store.directory / "engine-inventory" / uuid.uuid4().hex
    directory.mkdir(parents=True, mode=0o700)
    resolved = await context.io(resolve, Target(location=str(path)), "source", True, directory, 0)
    entries = resolved.entries()
    entry = next(entries, None)
    if entry is None or next(entries, None) is not None or entry.sha256 != SOURCE_SHA256:
        raise CapabilityError("BAIT source does not match the reviewed detector module")
    source = await context.snapshot(resolved, entry)
    root: Path = context.store.directory / "engine-sources" / resolved.identity
    destination: Path = root / entry.name
    await context.io(prepare_model, source, destination, root)
    return destination


async def run(context: Context) -> CheckResult:
    configuration = context.config.inversion
    if configuration is None:
        return CheckResult(status="not_requested", reason="No inversion experiment selected")
    settings = configuration.settings
    tokens = settings["initial_token_ids"]
    size = settings["batch_size"]
    result = CheckResult(
        coverage=Coverage(
            units="initialization and token-search batches",
            total=1 + (len(tokens) + size - 1) // size,
        )
    )
    context.checkpoint(result)
    source = await reviewed_source(context)
    output = output_directory(context)
    worker = Worker(
        context.config.runtime,
        context.worker.model,
        context.worker.base,
        output=output,
        engine_source=source,
    )
    try:
        metadata = await worker.invert(settings)
        result.evidence.append(context.store.evidence(metadata))
        result.coverage.inspected += 1
        context.checkpoint(result)
        for index, offset in enumerate(range(0, len(tokens), size)):
            started = time.perf_counter()
            await context.yield_control()
            record = await worker.invert({}, tokens[offset : offset + size])
            evidence = [context.store.evidence(record)]
            result.evidence.extend(evidence)
            result.coverage.gaps.extend(str(gap) for gap in record.get("gaps", []))
            result.coverage.inspected += 1
            if configuration.criterion is not None:
                score = record["q_score"]
                verdict: bool | None = True
                reason = "Candidate below the explicit Q-score gate; no suspicion asserted"
                if configuration.minimum_q is not None and score >= configuration.minimum_q:
                    if context.judge is None:
                        verdict = None
                        reason = "Candidate criterion requires the separately configured judge"
                    else:
                        verdict, judged = await context.judge_case(
                            configuration.criterion, record["target"]
                        )
                        evidence.extend(judged)
                        result.evidence.extend(judged)
                        reason = "Reconstructed target judged against the explicit criterion"
                if verdict is None:
                    result.coverage.gaps.append(reason)
                status: Literal["pass", "fail", "inconclusive"] = (
                    "pass" if verdict is True else "fail" if verdict is False else "inconclusive"
                )
                case_id = "inversion-batch-" + str(index)
                result.outcomes.append(
                    CaseOutcome(
                        check=context.check.id,
                        target=context.target.target.location,
                        case_id=case_id,
                        repeat=0,
                        status=status,
                        complete=True,
                        duration_seconds=time.perf_counter() - started,
                        evaluators=[
                            CaseEvaluation(
                                evaluator="gated-target-criterion", status=status, reason=reason
                            )
                        ],
                        evidence=evidence,
                    )
                )
                if verdict is False:
                    result.findings.append(
                        Finding(
                            check=context.check.id,
                            target=context.target.target.location,
                            detector="reconstructed-target-criterion",
                            case_id=case_id,
                            repeat=0,
                            title="Reconstructed target failed the configured criterion",
                            severity="medium",
                            strength="indicator",
                            attribution="Model and judge observations; backdoor origin unproven",
                            detail=json.dumps(
                                {
                                    "q_score": score,
                                    "criterion": configuration.criterion,
                                    "warning": "Benign outputs can also have high Q-scores",
                                }
                            ),
                            evidence=evidence,
                        )
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
        result.status, result.reason = "unsupported", "Target inversion assessment is incomplete"
    return result
