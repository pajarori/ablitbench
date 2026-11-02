from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from typing import Any


def scan(engine: str, path: Path, timeout: int | None = None) -> dict[str, Any]:
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
        gaps = list(result.get("errors", []))
        inspected = result.get("files_scanned", 0)
        limits = {
            "timeout_seconds": timeout or 3600,
            "max_file_size": 0,
            "max_total_size": 0,
            "scanner_specific_limits": "upstream limits/skips retained in raw evidence",
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
                )
            ):
                gaps.append(issue)
    else:
        raise ValueError("Unknown research engine")
    if not inspected:
        gaps.append({"reason": "No sources reported as inspected"})
    return {
        "engine": engine,
        "version": importlib.metadata.version(engine),
        "result": result,
        "issues": result.get("issues", []),
        "gaps": gaps,
        "inspected": inspected,
        "limits": limits,
    }
