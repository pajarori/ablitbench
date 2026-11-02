from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
from functools import cache
from pathlib import Path
from typing import Any

AUDITED_FILES = {
    "modelaudit": (
        "modelaudit/core.py",
        "5b36d54fcd89d9146da0a7316bf858509e726b98fec64ede13767770ec4e03cc",
    ),
    "modelaudit-picklescan": (
        "modelaudit_picklescan/api.py",
        "acf2a21880d96b1706498b4d8a544797d64132c11fd2abeda56af7e2046d12fb",
    ),
}
AUDITED_REVISION = "415819fa0b58d097dd201c0fa3f41eba33431c78"


@cache
def installation(engine: str) -> dict[str, Any]:
    names = [engine] + (["modelaudit-picklescan"] if engine == "modelaudit" else [])
    packages = {}
    for name in names:
        distribution = importlib.metadata.distribution(name)
        checksum = hashlib.sha256()
        limits = {}
        for entry in sorted(distribution.files or [], key=str):
            source = Path(str(distribution.locate_file(entry)))
            if not source.is_file() or source.suffix not in {".py", ".so", ".pyd"}:
                continue
            checksum.update(str(entry).encode())
            with source.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
            if source.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.isupper()
                        and any(
                            token in target.id for token in ("MAX_", "LIMIT", "BUDGET", "TIMEOUT")
                        )
                    ):
                        limits[str(entry) + ":" + target.id] = ast.unparse(node.value)
        direct = distribution.read_text("direct_url.json")
        dependencies = []
        requirement_type = importlib.import_module("packaging.requirements").Requirement
        for value in distribution.requires or []:
            requirement = requirement_type(value)
            active = requirement.marker is None or requirement.marker.evaluate()
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            dependencies.append(
                {
                    "requirement": value,
                    "active": active,
                    "installed": version,
                    "satisfied": version is not None and version in requirement.specifier,
                }
            )
        audited = None
        if name in AUDITED_FILES:
            relative, expected = AUDITED_FILES[name]
            actual = hashlib.sha256(
                Path(str(distribution.locate_file(relative))).read_bytes()
            ).hexdigest()
            audited = {
                "revision": AUDITED_REVISION,
                "file": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        packages[name] = {
            "version": distribution.version,
            "audited_file": audited,
            "source_sha256": checksum.hexdigest(),
            "requirements": dependencies,
            "direct_url": direct,
            "declared_limits": limits,
        }
    return packages


def coverage_gaps(value: Any, path: str = "result") -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    pending = [(path, value)]
    while pending:
        location, item = pending.pop()
        if isinstance(item, list):
            pending.extend((f"{location}[{index}]", child) for index, child in enumerate(item))
            continue
        if not isinstance(item, dict):
            continue
        markers = {
            key: item[key]
            for key in (
                "analysis_incomplete",
                "operational_error",
                "scan_outcome",
                "scan_outcome_reason",
                "scan_outcome_reasons",
                "dependency_missing",
                "missing_dependency",
                "missing_dependencies",
                "truncated",
                "limit_reached",
                "limit_exceeded",
                "budget_exhausted",
                "timed_out",
                "skipped",
                "unsupported",
            )
            if item.get(key)
            not in (None, False, "", [], "completed", "complete", "success", "passed")
        }
        if str(item.get("status", "")).lower() in {"skipped", "error", "inconclusive"}:
            markers["status"] = item["status"]
        if markers:
            gaps.append({"location": location, "reason": "incomplete_coverage", **markers})
        pending.extend((location + "." + str(key), child) for key, child in item.items())
    return gaps


def scan(engine: str, path: Path, timeout: int | None = None) -> dict[str, Any]:
    identity = installation(engine)
    if engine == "modelscan":
        implementation = importlib.import_module("modelscan.modelscan").ModelScan()
        result = implementation.scan(path)
        summary = result.get("summary", {})
        gaps = list(result.get("errors", [])) + summary.get("skipped", {}).get("skipped_files", [])
        inspected = summary.get("scanned", {}).get("total_scanned", 0)
        limits: dict[str, Any] = {"nested_zip": "unsupported; skipped entries retained"}
    elif engine == "modelaudit":
        implementation = importlib.import_module("modelaudit.core")
        kwargs: dict[str, Any] = {"max_file_size": 0, "max_total_size": 0, "skip_file_types": False}
        if timeout is not None:
            kwargs["timeout"] = timeout
        outcome = implementation.scan_model_directory_or_file(str(path), **kwargs)
        result = outcome.model_dump(mode="json")
        gaps = list(result.get("errors", [])) + coverage_gaps(result)
        inspected = result.get("files_scanned", 0)
        limits = {
            "timeout_seconds": timeout if timeout is not None else 3600,
            "max_file_size": 0,
            "max_total_size": 0,
            "scanner_specific_limits": identity[engine]["declared_limits"],
        }
        for issue in result.get("issues", []):
            text = str(issue.get("message", "")).lower()
            if any(
                token in text
                for token in (
                    "limit reached",
                    "timed out",
                    "truncat",
                    "budget",
                    "skipped",
                    "incomplete",
                    "not installed",
                    "missing dependency",
                    "dependency unavailable",
                )
            ):
                gaps.append({"reason": "upstream_coverage_message", "issue": issue})
        if not result.get("success", True) and not result.get("issues") and not gaps:
            gaps.append({"reason": "upstream_unsuccessful_without_findings"})
    else:
        raise ValueError("Unknown research engine")
    for package, metadata in identity.items():
        if metadata["audited_file"] and not metadata["audited_file"]["matches"]:
            gaps.append(
                {
                    "reason": "audited_source_mismatch",
                    "package": package,
                    **metadata["audited_file"],
                }
            )
        for dependency in metadata["requirements"]:
            if dependency["active"] and not dependency["satisfied"]:
                gaps.append({"reason": "unsatisfied_dependency", "package": package, **dependency})
    if not inspected:
        gaps.append({"reason": "No sources reported as inspected"})
    return {
        "engine": engine,
        "version": importlib.metadata.version(engine),
        "installation": identity,
        "scope": "package" if path.is_dir() else "file",
        "result": result,
        "issues": [issue for issue in result.get("issues", []) if issue.get("rule_code") != "S902"],
        "gaps": gaps,
        "inspected": inspected,
        "limits": limits,
    }
