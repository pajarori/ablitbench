from __future__ import annotations

from typing import TYPE_CHECKING

from ablitbench.models import CheckResult, Coverage, Finding
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
            if generation.finish in {"length", "max_tokens", "incomplete", "failed"}:
                result.coverage.gaps.append("Provider returned truncated or failed generation")
            result.coverage.inspected += 1
        except CapabilityError as error:
            result.status, result.reason = "unsupported", str(error)
            result.coverage.gaps.append(str(error))
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
        finally:
            if context.transport is not None:
                result.evidence.extend(
                    reference
                    for reference in context.transport.last_evidence
                    if reference not in result.evidence
                )
            context.checkpoint(result)
    from ablitbench.checks.relay import run as relay

    await relay(context, result)
    if (result.coverage.gaps or result.coverage.gap_count) and result.status == "completed":
        result.status, result.reason = "unsupported", "Generation coverage incomplete"
    return result
