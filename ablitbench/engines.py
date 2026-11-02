from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def modelscan_numpy() -> Iterator[dict[str, Any]]:
    numpy = importlib.import_module("numpy")
    public = importlib.import_module("numpy.lib.format")
    names = ("_check_version", "_read_array_header")
    missing = [name for name in names if not hasattr(public, name)]
    metadata: dict[str, Any] = {
        "modelscan": importlib.metadata.version("modelscan"),
        "numpy": numpy.__version__,
        "status": "native" if not missing else "unsupported",
        "bindings": [],
    }
    if not missing:
        yield metadata
        return
    if (metadata["modelscan"], metadata["numpy"]) != ("0.8.8", "2.5.4"):
        metadata["reason"] = (
            "Unreviewed ModelScan/NumPy private API combination; no aliases applied"
        )
        yield metadata
        return
    original = importlib.import_module("numpy.lib._format_impl")
    functions = {name: getattr(original, name) for name in missing}
    header = original._read_array_header
    header_guard = inspect.signature(header).parameters["max_header_size"].default
    if header_guard != 10000:
        metadata["reason"] = "Reviewed NumPy header guard changed; no aliases applied"
        yield metadata
        return
    module_path = original.__file__
    if module_path is None:
        raise ValueError("Reviewed NumPy source identity unavailable; no aliases applied")
    metadata.update(
        status="temporary-original-function-aliases",
        numpy_module_sha256=hashlib.sha256(Path(module_path).read_bytes()).hexdigest(),
        max_numpy_header_bytes=header_guard,
        scope="Private NumPy API relocation only; original detector and header guard retained",
        bindings=[
            {
                "destination": "numpy.lib.format." + name,
                "source": function.__module__ + "." + function.__name__,
                "function_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
            }
            for name, function in functions.items()
        ],
    )
    applied = []
    try:
        for name, function in functions.items():
            setattr(public, name, function)
            applied.append(name)
        yield metadata
    finally:
        for name in reversed(applied):
            delattr(public, name)


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
        scanners = item.get("scanner_dependency_ids")
        if (
            isinstance(scanners, list)
            and scanners
            and all(value == "unknown" for value in scanners)
        ):
            gaps.append(
                {
                    "location": location,
                    "reason": "no_format_security_scanner",
                    "scope": "Inventory or metadata inspection only",
                    "scanner_dependency_ids": scanners,
                }
            )
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
    compatibility: dict[str, Any] = {}
    if engine == "modelscan":
        implementation = importlib.import_module("modelscan.modelscan").ModelScan()
        with modelscan_numpy() as compatibility:
            result = implementation.scan(path)
        summary = result.get("summary", {})
        gaps = list(result.get("errors", [])) + summary.get("skipped", {}).get("skipped_files", [])
        inspected = summary.get("scanned", {}).get("total_scanned", 0)
        if compatibility["status"] == "unsupported":
            gaps.append({"reason": "unreviewed_numpy_compatibility", **compatibility})
        limits: dict[str, Any] = {
            "nested_zip": "unsupported; skipped entries retained",
            "numpy_header_bytes": compatibility.get("max_numpy_header_bytes"),
        }
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
        "compatibility": compatibility,
    }
