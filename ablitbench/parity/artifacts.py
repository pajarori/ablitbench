from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads
from ablitbench.formats.snapshot import secure_open
from ablitbench.models import CaseOutcome, Finding
from ablitbench.response_spool import resident_size


def normalize_findings(source: Path, destination: Path, check: str, target: str) -> dict[str, Any]:
    return normalize_records(source, destination, check, target)


def normalize_outcomes(
    source: Path, destination: Path, check: str, target: str, identity: str
) -> dict[str, Any]:
    return normalize_records(source, destination, check, target, identity)


def normalize_records(
    source: Path, destination: Path, check: str, target: str, identity: str | None = None
) -> dict[str, Any]:
    count, measured, offset = 0, 0, 0
    error = None
    with secure_open(source) as stream, destination.open("xb") as output:
        before = os.fstat(stream.fileno())
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                value = loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Assessment record must be an object")
                forced = {**value, "check": check, "target": target}
                finding: Finding | CaseOutcome
                if identity is None:
                    finding = Finding.model_validate(forced)
                else:
                    finding = CaseOutcome.model_validate({**forced, "target_identity": identity})
                serialized = finding.model_dump_json().encode() + b"\n"
                measured = max(measured, resident_size(finding) + len(serialized))
            except ValueError as failure:
                error = str(failure)
                break
            output.write(serialized)
            count += 1
        after = os.fstat(stream.fileno())
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        ):
            error = "Assessment artifact changed during validation"
        output.flush()
        os.fsync(output.fileno())
    return {
        "count": count,
        "materialization_bytes": measured,
        "error": error,
        "uninspected_offset": offset if error else None,
    }
