from __future__ import annotations

from typing import TYPE_CHECKING

from ablitbench.models import CheckResult, Coverage, Event, Finding
from ablitbench.transport import CapabilityError, ProtocolError

if TYPE_CHECKING:
    from ablitbench.core import Context


async def run(context: Context) -> CheckResult:
    relay_configuration = context.config.relay
    relay_requests = (
        (len(relay_configuration.dependencies) + len(relay_configuration.context_samples))
        * context.config.repeats
        if relay_configuration is not None
        else 0
    )
    result = CheckResult(coverage=Coverage(units="requests", total=2 + relay_requests))
    context.checkpoint(result)
    for stream in (False, True):
        case_id = "protocol-streaming" if stream else "protocol-non-streaming"
        probe_status = "inconclusive"
        probe_detail = "Protocol probe ended without a verdict"
        try:
            generation = await context.generate(
                [{"role": "user", "content": "Reply with the word ready."}], stream=stream
            )
            result.evidence.extend(generation.evidence)
            for observation in generation.observations:
                result.findings.append(
                    Finding(
                        check=context.check.id,
                        target=context.target.target.location,
                        detector=observation.code,
                        title=observation.code.replace("-", " ").capitalize(),
                        severity=observation.severity,
                        strength="indicator",
                        attribution="endpoint response; causal origin undetermined",
                        detail=observation.detail,
                        evidence=generation.evidence,
                    )
                )
            if generation.model and generation.model != context.target.target.model:
                result.findings.append(
                    Finding(
                        check=context.check.id,
                        target=context.target.target.location,
                        title="Reported model differs from requested model",
                        severity="info",
                        detector="reported-model-alias",
                        detail=(
                            f"Requested {context.target.target.model}; "
                            f"reported {generation.model}. "
                            "Alias/routing metadata alone does not prove substitution."
                        ),
                        evidence=generation.evidence,
                    )
                )
            if not generation.complete:
                result.coverage.gaps.append("Provider returned truncated or failed generation")
                probe_detail = "Provider response did not have a verified terminal event"
            elif any(item.severity in {"medium", "high"} for item in generation.observations):
                probe_status = "fail"
                probe_detail = "Protocol response contained integrity observations"
            else:
                probe_status = "pass"
                probe_detail = "Complete response decoded without protocol integrity violations"
            result.coverage.inspected += 1
        except CapabilityError as error:
            result.status, result.reason = "unsupported", str(error)
            result.coverage.gaps.append(str(error))
            probe_detail = str(error)
        except (ProtocolError, KeyError, TypeError, IndexError, ValueError) as error:
            evidence = context.transport.last_evidence if context.transport else []
            result.evidence.extend(evidence)
            result.findings.append(
                Finding(
                    check=context.check.id,
                    target=context.target.target.location,
                    title="Endpoint protocol violation",
                    detector="protocol-violation",
                    strength="observed",
                    detail=str(error),
                    evidence=evidence,
                )
            )
            result.status, result.reason = "error", "Protocol compatibility incomplete"
            result.coverage.gaps.append(str(error))
            probe_status = "fail"
            probe_detail = str(error)
        finally:
            if context.transport is not None:
                result.evidence.extend(
                    reference
                    for reference in context.transport.last_evidence
                    if reference not in result.evidence
                )
            context.store.event(
                Event(
                    kind="probe_outcome",
                    data={
                        "target": context.job.target,
                        "check": context.check.id,
                        "probe": case_id,
                        "status": probe_status,
                        "detail": probe_detail,
                    },
                )
            )
            context.checkpoint(result)
    from ablitbench.checks.relay import run as relay

    await relay(context, result)
    if (result.coverage.gaps or result.coverage.gap_count) and result.status == "completed":
        result.status, result.reason = "unsupported", "Generation coverage incomplete"
    return result
