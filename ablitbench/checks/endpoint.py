from __future__ import annotations

from typing import TYPE_CHECKING

from ablitbench.models import CheckResult, Coverage, Finding
from ablitbench.transport import CapabilityError, ProtocolError

if TYPE_CHECKING:
    from ablitbench.core import Context


async def run(context: Context) -> CheckResult:
    result = CheckResult(coverage=Coverage(units="requests", total=2))
    for stream in (False, True):
        try:
            generation = await context.generate(
                [{"role": "user", "content": "Reply with the word ready."}], stream=stream
            )
            result.evidence.extend(generation.evidence)
            if generation.model and generation.model != context.target.target.model:
                result.findings.append(
                    Finding(
                        check=context.check.id,
                        target=context.target.target.location,
                        title="Reported model differs from requested model",
                        severity="info",
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
                    strength="observed",
                    detail=str(error),
                    evidence=evidence,
                )
            )
            result.status, result.reason = "error", "Protocol compatibility incomplete"
            result.coverage.gaps.append(str(error))
    if (result.coverage.gaps or result.coverage.gap_count) and result.status == "completed":
        result.status, result.reason = "unsupported", "Generation coverage incomplete"
    return result
