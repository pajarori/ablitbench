from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import re
import sys
from collections.abc import Iterable
from contextlib import redirect_stdout
from functools import cache
from pathlib import Path
from typing import Any

from ablitbench.engines import coverage_gaps, installation, modelscan_numpy


@cache
def catalog() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "data" / "parity-source-runtime.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def operations() -> dict[str, dict[str, Any]]:
    result = {}
    metadata = catalog()
    for engine in ("modelscan", "modelaudit"):
        base = {
            "project": engine,
            "kind": "detect",
            "capabilities": ["isolated-execution"],
            "inputs": ["model", "source"],
            "model_input": "artifact",
            "optional_inputs": [],
            "secret_inputs": [],
            "source_revision": metadata[engine]["revision"],
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["source"],
            "permissions": [],
        }
        result[engine + ".scan"] = base
        result[engine + ".scanners"] = dict(base, kind="inspect", inputs=["source"])
    result["modelaudit.unsafe-scan"] = dict(
        result["modelaudit.scan"], requires_tools=True, permissions=["custom-model-code"]
    )
    result["modelscan.custom-scan"] = dict(
        result["modelscan.scan"],
        inputs=["source", "model", "extensions"],
        requires_tools=True,
        permissions=["custom-model-code"],
    )
    result["modelscan.settings"] = dict(result["modelscan.scanners"], kind="export")
    result["modelaudit.rules"] = dict(result["modelaudit.scanners"], kind="inspect")
    result["modelaudit.report"] = dict(
        result["modelaudit.scanners"], kind="export", inputs=["source", "result"]
    )
    for name in ("doctor", "debug", "metadata", "sbom"):
        result["modelaudit." + name] = dict(
            result["modelaudit.scanners"],
            kind="export" if name == "sbom" else "inspect",
            inputs=["source", "result", "model"]
            if name == "sbom"
            else ["source", "model"]
            if name == "metadata"
            else ["source"],
        )
    result["modelaudit.metadata.deserialize"] = dict(
        result["modelaudit.metadata"], requires_tools=True, permissions=["custom-model-code"]
    )
    for descriptor in metadata["modelaudit"]["scanners"]:
        result["modelaudit." + descriptor["operation"]] = dict(
            result["modelaudit.scan"],
            scanner=descriptor["operation"].split(".", 1)[1],
            dependencies=descriptor["dependencies"],
            extensions=descriptor["extensions"],
        )
    for name, settings in metadata["modelscan"]["scanners"].items():
        result["modelscan.scan." + name.rsplit(".", 1)[1]] = dict(
            result["modelscan.scan"], scanner=name, extensions=settings["supported_extensions"]
        )
    return result


def module_path(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", value))


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown source runtime operation")
    allowed = {
        "scanners",
        "exclude_scanners",
        "config",
        "timeout_seconds",
        "format",
        "verbose",
        "security_only",
        "rule",
        "category",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown source runtime setting")
    for key in ("scanners", "exclude_scanners"):
        if key in settings and (
            not isinstance(settings[key], list)
            or not all(isinstance(value, str) for value in settings[key])
        ):
            raise ValueError(key + " must be a list of scanner names")
    if "config" in settings and not isinstance(settings["config"], dict):
        raise ValueError("config must be a data object")
    if "timeout_seconds" in settings and (
        not isinstance(settings["timeout_seconds"], int)
        or isinstance(settings["timeout_seconds"], bool)
        or settings["timeout_seconds"] <= 0
    ):
        raise ValueError("timeout_seconds must be an explicit positive integer")
    if settings.get("format", "json") not in {"json", "text", "sarif"}:
        raise ValueError("Unsupported source report format")
    if "verbose" in settings and not isinstance(settings["verbose"], bool):
        raise ValueError("verbose must be boolean")
    for key in ("security_only",):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(key + " must be boolean")
    for key in ("rule", "category"):
        if key in settings and not isinstance(settings[key], str):
            raise ValueError(key + " must be text")
    if "category" in settings:
        bounds = settings["category"].split("-")
        if len(bounds) not in {1, 2} or not all(value.isdigit() for value in bounds):
            raise ValueError("category must be a numeric rule range")
        if len(bounds) == 2 and int(bounds[0]) > int(bounds[1]):
            raise ValueError("category range is reversed")
    config = settings.get("config", {})
    if any(key in config for key in ("api_key", "token", "password", "custom_scanners")):
        raise ValueError("Credentials and custom Python modules cannot be supplied in scan config")
    if any(str(key).startswith("_") for key in config):
        raise ValueError("Internal scanner authority fields cannot be supplied as configuration")
    for key in ("allow_metadata_deserialization", "enable_unsafe_torch_load"):
        if key in config and type(config[key]) is not bool:
            raise ValueError(key + " must be boolean")
        if config.get(key) and operation != "modelaudit.unsafe-scan":
            raise ValueError("Unsafe scanner loading requires modelaudit.unsafe-scan")
    if operation.startswith("modelscan"):
        if settings.get("format") == "sarif":
            raise ValueError("ModelScan provides JSON and console reports, not SARIF")
        known = set(catalog()["modelscan"]["scanners"])
        chosen = set(settings.get("scanners", [])) | set(settings.get("exclude_scanners", []))
        custom = operation == "modelscan.custom-scan"
        if not custom and chosen - known:
            raise ValueError("Unknown or custom ModelScan scanner")
        for key, paths in (
            ("scanners", known),
            ("middlewares", {"modelscan.middlewares.FormatViaExtensionMiddleware"}),
        ):
            if key in config and (
                not isinstance(config[key], dict)
                or (not custom and set(config[key]) - paths)
                or any(not module_path(name) for name in config[key])
                or not all(isinstance(value, dict) for value in config[key].values())
            ):
                raise ValueError("Unknown or custom ModelScan " + key)
        if "reporting" in config:
            reporting = config["reporting"]
            if (
                not isinstance(reporting, dict)
                or set(reporting) - {"module", "settings"}
                or not module_path(reporting.get("module"))
                or (
                    not custom
                    and reporting.get("module")
                    not in {"modelscan.reports.ConsoleReport", "modelscan.reports.JSONReport"}
                )
                or not isinstance(reporting.get("settings", {}), dict)
                or (not custom and reporting.get("settings", {}) != {})
            ):
                raise ValueError("Only built-in ModelScan reports with managed output are accepted")
        if set(config) - {
            "modelscan_version",
            "supported_zip_extensions",
            "scanners",
            "middlewares",
            "unsafe_globals",
            "reporting",
        }:
            raise ValueError("Unknown ModelScan settings")
    return settings


def original_module(engine: str, source: Path) -> dict[str, Any]:
    from ablitbench.parity.sources import verified_source

    verified_source(source, engine)
    identities = installation(engine)
    package = importlib.import_module(engine)
    if package.__file__ is None:
        raise ValueError("Installed engine source location is unavailable")
    origin = Path(package.__file__).resolve().parent
    mismatches = []
    build_transforms = []
    for path in origin.rglob("*.py"):
        relative = path.relative_to(origin)
        expected = source / engine / relative
        if not expected.is_file() or expected.read_bytes() != path.read_bytes():
            transformed = False
            if engine == "modelscan" and relative.as_posix() == "_version.py":
                original_tree = ast.parse(expected.read_text())
                installed_tree = ast.parse(path.read_text())
                original_expected = ast.parse('__version__ = "0.0.0"')
                installed_expected = ast.parse('__version__ = "0.8.8"')
                transformed = ast.dump(original_tree) == ast.dump(original_expected) and ast.dump(
                    installed_tree
                ) == ast.dump(installed_expected)
                if transformed:
                    build_transforms.append(
                        {
                            "file": "_version.py",
                            "kind": "poetry-dynamic-versioning",
                            "source": "0.0.0",
                            "installed": "0.8.8",
                        }
                    )
            if not transformed:
                mismatches.append(relative.as_posix())
    if mismatches:
        raise ValueError("Installed engine differs from reviewed source: " + ", ".join(mismatches))
    return dict(identities, verified_build_transforms=build_transforms)


def persist_result(
    project: str, raw: Any, findings: Iterable[dict[str, Any]], output: Path
) -> dict[str, Any]:
    from ablitbench.models import Finding

    with (output / "upstream-result.json").open("w", encoding="utf-8") as stream:
        json.dump(raw, stream, ensure_ascii=False)
    count = 0
    with (output / "findings.jsonl").open("w", encoding="utf-8") as stream:
        for native in findings:
            if project == "api-relay-audit":
                if native.get("level") == "green":
                    continue
                title = str(native.get("message", "Upstream endpoint indicator"))
                detail = title
                severity = "medium" if native.get("level") == "red" else "info"
                detector = project
                attribution = "Endpoint observation; attribution undetermined"
            else:
                detector = str(
                    native.get("rule_id")
                    or native.get("rule_code")
                    or native.get("scanner")
                    or native.get("ruleId")
                    or project
                )
                title = project + ": " + detector
                native_detail = (
                    native.get("message")
                    or native.get("detail")
                    or native.get("description")
                    or native
                )
                detail = (
                    native_detail
                    if isinstance(native_detail, str)
                    else json.dumps(native_detail, ensure_ascii=False)
                )
                severity = str(native.get("severity", "info")).lower()
                if severity not in {"info", "low", "medium", "high", "critical"}:
                    severity = "info"
                attribution = "Static artifact indicator; attack attribution undetermined"
            finding = Finding.model_validate(
                {
                    "check": "research.parity",
                    "target": "",
                    "title": title,
                    "detail": detail,
                    "severity": severity,
                    "strength": "indicator",
                    "attribution": attribution,
                    "detector": detector,
                }
            )
            stream.write(finding.model_dump_json(exclude={"check", "target"}) + "\n")
            count += 1
    return {
        "result_artifact": "upstream-result.json",
        "findings_artifact": "findings.jsonl",
        "finding_count": count,
    }


def persist_gaps(gaps: Iterable[Any], output: Path) -> dict[str, Any]:
    count = 0
    with (output / "coverage-gaps.jsonl").open("w", encoding="utf-8") as stream:
        for value in gaps:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            count += 1
    return {
        "gap_count": count,
        "gaps_artifact": "coverage-gaps.jsonl",
        "gaps": [
            {"reason": "Native coverage gaps", "artifact": "coverage-gaps.jsonl", "count": count}
        ]
        if count
        else [],
    }


def modelscan(
    operation: str, settings: dict[str, Any], target: Path, output: Path
) -> dict[str, Any]:
    module = importlib.import_module("modelscan.settings")
    configured = copy.deepcopy(module.DEFAULT_SETTINGS)
    supplied = copy.deepcopy(settings.get("config", {}))
    for key in ("scanners", "middlewares"):
        for name, values in supplied.pop(key, {}).items():
            configured[key].setdefault(name, {}).update(values)
    configured.update(supplied)
    middleware = configured["middlewares"].get(
        "modelscan.middlewares.FormatViaExtensionMiddleware", {}
    )
    formats = {
        value.value: value
        for value in vars(module.SupportedModelFormats).values()
        if isinstance(value, module.Property)
    }
    if "formats" in middleware:
        normalized = {}
        for key, extensions in middleware["formats"].items():
            if isinstance(key, str):
                if key not in formats:
                    raise ValueError("Unknown ModelScan format key: " + key)
                key = formats[key]
            normalized[key] = extensions
        middleware["formats"] = normalized
    chosen = settings.get("scanners")
    if ".scan." in operation:
        chosen = [operations()[operation]["scanner"]]
    excluded = set(settings.get("exclude_scanners", []))
    for name, values in configured["scanners"].items():
        values["enabled"] = (
            values.get("enabled", True)
            and (chosen is None or name in chosen)
            and name not in excluded
        )
    with modelscan_numpy() as compatibility:
        scanner = importlib.import_module("modelscan.modelscan").ModelScan(settings=configured)
        raw = scanner.scan(target)
        reports = importlib.import_module("modelscan.reports")
        report_name = settings.get("config", {}).get("reporting", {}).get("module")
        if report_name is None:
            report_name = (
                "modelscan.reports.ConsoleReport"
                if settings.get("format") == "text"
                else "modelscan.reports.JSONReport"
            )
        reporter = (
            reports.ConsoleReport if report_name.endswith("ConsoleReport") else reports.JSONReport
        )
        with (
            (output / "upstream-report").open("w", encoding="utf-8") as stream,
            redirect_stdout(stream),
        ):
            if operation == "modelscan.custom-scan":
                rendered = scanner.generate_report()
                if rendered is not None:
                    stream.write(str(rendered))
            else:
                reporter.generate(scanner, {"show_skipped": True})
    if operation == "modelscan.custom-scan":
        raw = scanner._generate_results()
    gaps = list(raw.get("errors", [])) + raw.get("summary", {}).get("skipped", {}).get(
        "skipped_files", []
    )
    if compatibility["status"] == "unsupported":
        gaps.append({"reason": "unsupported_numpy_private_api", **compatibility})
    inspected = raw.get("summary", {}).get("scanned", {}).get("total_scanned", 0)
    if not inspected:
        gaps.append("No sources reported as inspected")
    return {
        "status": "unsupported" if gaps else "completed",
        **persist_result("modelscan", raw, raw.get("issues", []), output),
        "coverage": {"units": "sources", "inspected": inspected, **persist_gaps(gaps, output)},
        "compatibility": compatibility,
        "report_artifact": "upstream-report",
        "limits": {"numpy_header_bytes": compatibility.get("max_numpy_header_bytes")},
    }


def modelaudit(
    operation: str, settings: dict[str, Any], target: Path, output: Path
) -> dict[str, Any]:
    module = importlib.import_module("modelaudit.core")
    config = dict(settings.get("config", {}))
    config.setdefault("max_file_size", 0)
    config.setdefault("max_total_size", 0)
    config.setdefault("skip_file_types", False)
    for key in ("scanners", "exclude_scanners"):
        if key in settings:
            config[key] = settings[key]
    if ".scan." in operation:
        config["scanners"] = [operations()[operation]["scanner"]]
    if "timeout_seconds" in settings:
        config["timeout"] = settings["timeout_seconds"]
    raw_object = module.scan_model_directory_or_file(str(target), **config)
    raw = raw_object.model_dump(mode="json")
    gaps = list(raw.get("errors", [])) + coverage_gaps(raw)
    skipped_assets = [
        asset for asset in raw.get("assets", []) if asset.get("type") == "scanner_selection"
    ]
    gaps.extend(
        {"reason": "selected_scanner_not_applied", "path": asset.get("path")}
        for asset in skipped_assets
    )
    inspected = max(0, raw.get("files_scanned", 0) - len(skipped_assets))
    if not inspected:
        gaps.append("No sources reported as inspected")
    for issue in raw.get("issues", []):
        message = str(issue.get("message", "")).lower()
        if any(
            word in message
            for word in (
                "incomplete",
                "truncat",
                "skipped",
                "timed out",
                "not installed",
                "missing dependency",
                "limit reached",
            )
        ):
            gaps.append({"reason": "upstream_coverage_message", "issue": issue})
    formatter = importlib.import_module("modelaudit.cli")
    rendered = formatter._format_scan_output(
        raw_object,
        [str(target)],
        output_format=settings.get("format", "json"),
        verbose=settings.get("verbose", False),
    )
    (output / "upstream-report").write_text(rendered, encoding="utf-8")
    return {
        "status": "unsupported" if gaps else "completed",
        **persist_result(
            "modelaudit",
            raw,
            (issue for issue in raw.get("issues", []) if issue.get("rule_code") != "S902"),
            output,
        ),
        "coverage": {"units": "sources", "inspected": inspected, **persist_gaps(gaps, output)},
        "limits": {
            "timeout_seconds": config.get(
                "timeout",
                inspect.signature(module.scan_model_directory_or_file)
                .parameters["timeout"]
                .default,
            )
        },
        "report_artifact": "upstream-report",
    }


def execute(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    broker: Any = None,
) -> dict[str, Any]:
    from ablitbench.worker import isolation

    isolation()
    configuration(operation, settings)
    engine = operations()[operation]["project"]
    identity = original_module(engine, inputs["source"])
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any]
    if operation == "modelscan.settings":
        module = importlib.import_module("modelscan.settings")
        try:
            rendered = module.SettingsUtils.get_default_settings_as_toml()
            correction = None
        except TypeError as error:
            if str(error) != "Keys must be strings":
                raise
            original = module.DEFAULT_SETTINGS
            normalized = copy.deepcopy(original)
            middleware = normalized["middlewares"][
                "modelscan.middlewares.FormatViaExtensionMiddleware"
            ]
            middleware["formats"] = {
                key.value if isinstance(key, module.Property) else key: value
                for key, value in middleware["formats"].items()
            }
            try:
                module.__dict__["DEFAULT_SETTINGS"] = normalized
                rendered = module.SettingsUtils.get_default_settings_as_toml()
            finally:
                module.__dict__["DEFAULT_SETTINGS"] = original
            correction = {
                "upstream_defect": "Property format keys are not valid TOML strings",
                "adapter_correction": "Serialize declared format values; restore original defaults",
                "roundtrip": "Known format values compile back to original Property instances",
            }
        (output / "modelscan-settings.toml").write_text(rendered, encoding="utf-8")
        result = {
            "status": "completed",
            "artifact": "modelscan-settings.toml",
            "compatibility": correction,
        }
    elif operation.endswith(".scanners"):
        if engine == "modelaudit":
            module = importlib.import_module("modelaudit.scanner_selection")
            scanners = module.scanner_catalog()
        else:
            scanners = catalog()["modelscan"]["scanners"]
        result = {"status": "completed", "catalog": scanners}
    elif operation in {"modelaudit.doctor", "modelaudit.debug"}:
        module = importlib.import_module("modelaudit.cli")
        artifact = operation.rsplit(".", 1)[1] + ".txt"
        with (output / artifact).open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            if operation.endswith("doctor"):
                module.doctor.callback(show_failed=settings.get("verbose", False))
            else:
                module.debug.callback(output_json=True, verbose=settings.get("verbose", False))
        result = {"status": "completed", "artifact": artifact}
    elif operation in {"modelaudit.metadata", "modelaudit.metadata.deserialize"}:
        module = importlib.import_module("modelaudit.metadata_extractor")
        extracted = module.ModelMetadataExtractor().extract(
            str(inputs["model"]),
            security_only=settings.get("security_only", False),
            allow_deserialization=operation.endswith(".deserialize"),
        )
        metadata_gaps = coverage_gaps(extracted)
        pending = [("metadata", extracted)]
        while pending:
            location, item = pending.pop()
            if isinstance(item, dict):
                for key in ("error", "extraction_error"):
                    if item.get(key):
                        metadata_gaps.append(
                            {"location": location, "reason": key, "detail": item[key]}
                        )
                pending.extend((location + "." + str(key), value) for key, value in item.items())
            elif isinstance(item, list):
                pending.extend(
                    (location + "." + str(index), value) for index, value in enumerate(item)
                )
        (output / "upstream-metadata.json").write_text(
            json.dumps(extracted, ensure_ascii=False), encoding="utf-8"
        )
        result = {
            "status": "inconclusive" if metadata_gaps else "completed",
            "metadata_artifact": "upstream-metadata.json",
            "coverage": persist_gaps(metadata_gaps, output),
            "limits": {
                "loader_deserialization": "enabled by explicit custom-model-code operation"
                if operation.endswith(".deserialize")
                else "disabled",
                **{
                    name: value
                    for name, value in vars(module).items()
                    if name.startswith("MAX_METADATA_DIRECTORY_")
                },
            },
        }
    elif operation == "modelaudit.sbom":
        models = importlib.import_module("modelaudit.models")
        raw = models.ModelAuditResultModel.model_validate_json(inputs["result"].read_bytes())
        rendered = importlib.import_module(
            "modelaudit.integrations.sbom_generator"
        ).generate_sbom_pydantic(
            [str(inputs["model"])],
            raw,
        )
        (output / "upstream-sbom.json").write_text(rendered, encoding="utf-8")
        result = {"status": "completed", "artifact": "upstream-sbom.json"}
    elif operation == "modelaudit.rules":
        registry = importlib.import_module("modelaudit.rules").RuleRegistry
        registry.initialize()
        rules = registry.get_all_rules()
        if "rule" in settings:
            rule = registry.get_rule(settings["rule"].upper())
            if rule is None:
                raise ValueError("Unknown ModelAudit rule")
            rules = {rule.code: rule}
        elif "category" in settings:
            bounds = settings["category"].split("-")
            start = int(bounds[0])
            end = int(bounds[1]) if len(bounds) == 2 else start + 99
            rules = registry.get_rules_by_range(start, end)
        result = {
            "status": "completed",
            "rules": {
                key: {
                    "name": rule.name,
                    "severity": rule.default_severity.value,
                    "description": rule.description,
                }
                for key, rule in rules.items()
            },
        }
    elif operation == "modelaudit.report":
        models = importlib.import_module("modelaudit.models")
        raw = models.ModelAuditResultModel.model_validate_json(inputs["result"].read_bytes())
        rendered = importlib.import_module("modelaudit.cli")._format_scan_output(
            raw,
            [],
            output_format=settings.get("format", "json"),
            verbose=settings.get("verbose", False),
        )
        (output / "upstream-report").write_text(rendered, encoding="utf-8")
        result = {"status": "completed", "artifact": "upstream-report"}
    elif operation == "modelscan.custom-scan":
        extension = inputs["extensions"]
        if not extension.is_dir() or extension.is_symlink():
            raise ValueError("Custom extensions must be an explicit read-only directory input")
        original_path = list(sys.path)
        try:
            sys.path.insert(0, str(extension))
            result = modelscan(operation, settings, inputs["model"], output)
        finally:
            sys.path[:] = original_path
    else:
        target = inputs["model"]
        result = (
            modelaudit(operation, settings, target, output)
            if engine == "modelaudit"
            else modelscan(operation, settings, target, output)
        )
    return {"project": engine, "operation": operation, "upstream_identity": identity, **result}
