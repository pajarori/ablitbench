from __future__ import annotations

import importlib
from typing import Any

from ablitbench.parity.sources import project_name

MODULES = {
    "layerfault": ("layerfault",),
    "modelscan": ("source_runtime",),
    "modelaudit": ("source_runtime", "acquisition"),
    "api-relay-audit": ("relay",),
    "garak": ("probes",),
    "llm-backdoor-scanner": ("discovery",),
    "llmmap": ("fingerprint",),
    "refusal-direction": ("refusal",),
    "bait": ("inversion",),
    "badagent": ("agents",),
    "backdoorllm": ("attacks", "defenses"),
    "llm-backdoor-defense": ("classification",),
}


def implementations(project: str) -> list[Any]:
    modules = []
    for name in MODULES[project_name(project)]:
        qualified = "ablitbench.parity." + name
        try:
            modules.append(importlib.import_module(qualified))
        except ModuleNotFoundError as error:
            if error.name != qualified:
                raise
    return modules


def operation(project: str, name: str) -> tuple[Any, dict[str, Any]]:
    project = project_name(project)
    candidates = []
    for module in implementations(project):
        metadata = module.operations().get(name)
        if metadata is None:
            continue
        declared = metadata.get("project", project)
        if project_name(declared) == project:
            candidates.append((module, dict(metadata)))
    if len(candidates) != 1:
        raise ValueError("Unknown or ambiguous research operation: " + project + "/" + name)
    return candidates[0]


def operations(project: str) -> dict[str, dict[str, Any]]:
    project = project_name(project)
    result = {}
    for module in implementations(project):
        for name, metadata in module.operations().items():
            if project_name(metadata.get("project", project)) != project:
                continue
            if name in result:
                raise ValueError("Duplicate research operation: " + project + "/" + name)
            result[name] = dict(metadata)
    return result


def configuration(project: str, name: str, settings: dict[str, Any]) -> dict[str, Any]:
    module, _ = operation(project, name)
    return dict(module.configuration(name, settings))


def effective_configuration(
    project: str, name: str, settings: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    _, metadata = operation(project, name)
    settings = dict(settings)
    for field, attribute in metadata.get("target_defaults", {}).items():
        if attribute not in {"model", "location", "protocol"}:
            raise ValueError("Unsafe target default binding")
        if attribute in target and target[attribute] is not None:
            settings.setdefault(field, target[attribute])
    if "base_url" in metadata.get("target_defaults", {}):
        if settings.get("base_url", "").rstrip("/") != target.get("location", "").rstrip("/"):
            raise ValueError("Research endpoint identity differs from the configured target")
    return configuration(project, name, settings)


def requirements(project: str, name: str, settings: dict[str, Any]) -> set[str]:
    module, metadata = operation(project, name)
    required = set(metadata.get("permissions", []))
    callback = getattr(module, "required_permissions", None)
    if callback is not None:
        required.update(callback(name, settings))
    return required


def requires_execution(project: str, name: str, settings: dict[str, Any], capability: str) -> bool:
    if capability not in {"tools", "inference"}:
        raise ValueError("Unknown research execution capability")
    module, metadata = operation(project, name)
    key = "requires_" + capability
    hook = getattr(module, key, None)
    return bool(metadata.get(key) or (hook is not None and hook(name, settings)))


def available_modes(project: str, name: str, settings: dict[str, Any]) -> set[str]:
    module, metadata = operation(project, name)
    modes = set(metadata.get("modes", ["api", "source"]))
    hook = getattr(module, "available_modes", None)
    return modes & set(hook(name, settings)) if hook is not None else modes


def execute(
    project: str,
    name: str,
    settings: dict[str, Any],
    inputs: dict[str, Any],
    output: Any,
    broker: Any = None,
) -> dict[str, Any]:
    module, _ = operation(project, name)
    return dict(module.execute(name, module.configuration(name, settings), inputs, output, broker))
