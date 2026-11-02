from __future__ import annotations

import ast
import json
import tarfile
import zipfile
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ablitbench.formats.archive import unsafe_path
from ablitbench.models import CheckResult, Coverage, Finding

if TYPE_CHECKING:
    from ablitbench.core import Context

EXECUTORS = {
    "eval",
    "exec",
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "pickle.load",
    "pickle.loads",
    "torch.load",
    "__import__",
}


def qualified(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified(node.value) + "." + node.attr
    return ""


def analyze_python(text: str) -> list[dict[str, Any]]:
    tree = ast.parse(text)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    observations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            head, separator, tail = name.partition(".")
            name = aliases.get(head, head) + (separator + tail if separator else "")
            if name in EXECUTORS:
                observations.append(
                    {"line": node.lineno, "call": name, "issue": "execution-capable loader call"}
                )
    return observations


def analyze_metadata(value: Any, location: str = "") -> list[dict[str, Any]]:
    observations = []
    if isinstance(value, dict):
        for key, item in value.items():
            field = f"{location}.{key}"
            if key in {"auto_map", "trust_remote_code"} and item:
                observations.append({"field": field, "issue": "custom code loading declaration"})
            if key in {"base_model_name_or_path", "peft_type", "chat_template", "architectures"}:
                observations.append({"field": field, "issue": "model metadata", "value": item})
            observations.extend(analyze_metadata(item, field))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            observations.extend(analyze_metadata(item, f"{location}[{index}]"))
    return observations


async def run(context: Context) -> CheckResult:
    result = context.static_result(
        Coverage(
            units="inventory entries (static inspection for supported formats)",
            total=context.target.files,
        )
    )
    if context.target.inventory_scope == "cached":
        result.coverage.gaps.append(
            "Offline cache inventory cannot establish repository completeness"
        )
    for entry in context.target.entries():
        await context.yield_control()
        if not context.begin_unit(entry.name, result):
            continue
        success = True
        cancelled = False
        parsed = None
        try:
            if entry.issue:
                result.coverage.gaps.append(f"{entry.name}: {entry.issue}")
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix in {".onnx", ".h5", ".hdf5", ".pb", ".tflite", ".ckpt"}:
                result.coverage.gaps.append(
                    f"{entry.name}: framework parser requires a separate validated engine"
                )
                continue
            if suffix not in {
                ".py",
                ".json",
                ".pkl",
                ".pickle",
                ".pt",
                ".pth",
                ".bin",
                ".zip",
                ".tar",
                ".gz",
                ".tgz",
                ".txt",
                ".jinja",
                ".jinja2",
            }:
                result.coverage.inspected += 1
                continue
            path = await context.snapshot(context.target, entry)
            from ablitbench.parsing import ParserCancelled, parse, records

            try:
                parsed = await parse(
                    path, "source", context.store.directory, context.config.runtime.memory_bytes
                )
            except ParserCancelled as error:
                parsed = error.output
                cancelled = True
            with parsed.open("rb") as stream:
                reference = context.store.blob(
                    iter(lambda: stream.read(1024 * 1024), b""), "application/jsonl"
                ).sha256
            result.evidence.append(reference)
            with closing(records(parsed)) as observations:
                for observation in observations:
                    await context.yield_control()
                    context.checkpoint(result)
                    if "gap" in observation:
                        if observation.get("incomplete"):
                            success = False
                        result.coverage.gaps.append(entry.name + ": " + observation["gap"])
                        continue
                    if observation.get("issue") == "shard map":
                        weight_map = observation["weight_map"]
                        if not isinstance(weight_map, dict):
                            raise ValueError("Invalid shard map")
                        for tensor, shard in weight_map.items():
                            if not isinstance(shard, str) or unsafe_path(shard):
                                result.findings.append(
                                    Finding(
                                        check=context.check.id,
                                        target=context.target.target.location,
                                        title="unsafe shard path",
                                        detail=str(tensor),
                                        evidence=[reference],
                                    )
                                )
                            elif not any(
                                candidate.name == (Path(entry.name).parent / shard).as_posix()
                                for candidate in context.target.entries()
                            ):
                                result.findings.append(
                                    Finding(
                                        check=context.check.id,
                                        target=context.target.target.location,
                                        title="missing tensor shard",
                                        detail=str(tensor),
                                        evidence=[reference],
                                    )
                                )
                        continue
                    issue = observation.get("issue", "unsafe serialization opcode")
                    if issue in {"model metadata", "chat template recorded; template not executed"}:
                        continue
                    result.findings.append(
                        Finding(
                            check=context.check.id,
                            target=context.target.target.location,
                            title=str(issue),
                            detail=json.dumps({"file": entry.name, **observation}),
                            evidence=[reference],
                            severity="high" if "traversal" in issue else "medium",
                        )
                    )
            if cancelled:
                raise ParserCancelled(parsed)
            if success:
                result.coverage.inspected += 1
        except (
            OSError,
            ValueError,
            SyntaxError,
            EOFError,
            zipfile.BadZipFile,
            tarfile.TarError,
        ) as error:
            success = False
            reference = context.store.evidence(
                {"file": entry.name, "error": type(error).__name__, "detail": str(error)}
            )
            result.evidence.append(reference)
            result.coverage.gaps.append(f"{entry.name}: parser failed ({type(error).__name__})")
        except BaseException:
            success = False
            raise
        finally:
            if parsed is not None:
                parsed.unlink(missing_ok=True)
            context.finish_unit(result, success)
    if result.coverage.gaps or result.coverage.gap_count:
        result.status = "unsupported" if result.coverage.inspected else "error"
        result.reason = "Some source content was not inspected; consult coverage gaps"
    return result
