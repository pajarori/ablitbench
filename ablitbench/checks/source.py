from __future__ import annotations

import json
import tarfile
import zipfile
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ablitbench.formats.archive import unsafe_path
from ablitbench.formats.static import analyze_python as analyze_python
from ablitbench.formats.static import analyze_template
from ablitbench.models import CheckResult, Coverage, Finding

if TYPE_CHECKING:
    from ablitbench.core import Context


def analyze_metadata(value: Any, location: str = "") -> list[dict[str, Any]]:
    observations = []
    if isinstance(value, dict):
        for key, item in value.items():
            field = f"{location}.{key}"
            if key in {"auto_map", "trust_remote_code"} and item:
                observations.append({"field": field, "issue": "custom code loading declaration"})
            if key in {"base_model_name_or_path", "peft_type", "chat_template", "architectures"}:
                observations.append({"field": field, "issue": "model metadata", "value": item})
            if (key == "chat_template" or ".chat_template" in location) and isinstance(item, str):
                observations.extend(
                    {"field": field, **observation} for observation in analyze_template(item)
                )
            observations.extend(analyze_metadata(item, field))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            observations.extend(analyze_metadata(item, f"{location}[{index}]"))
    return observations


async def run(context: Context) -> CheckResult:
    result = context.static_result(
        Coverage(
            units=(
                "inventory entries and signed provenance check"
                if context.config.trust
                else "inventory entries (static inspection for supported formats)"
            ),
            total=context.target.files + (1 if context.config.trust else 0),
        )
    )
    if context.config.trust and context.begin_unit("\0signed-provenance", result):
        from ablitbench.formats.provenance import inspect_provenance

        provenance_complete = False
        provenance_finished = False
        try:
            provenance_complete = True
            async for observation in inspect_provenance(context):
                reference = context.store.evidence(observation)
                result.evidence.append(reference)
                if observation.get("manifest"):
                    result.evidence.append(observation["manifest"])
                if "gap" in observation:
                    provenance_complete = False
                    result.coverage.gaps.append(observation["gap"])
                elif "issue" in observation:
                    result.findings.append(
                        Finding(
                            check=context.check.id,
                            target=context.target.target.location,
                            title=observation["issue"],
                            detail=json.dumps(observation),
                            evidence=[reference],
                            severity="high",
                            strength="observed",
                            attribution="Operator trust policy and signed inventory validation",
                        )
                    )
                context.checkpoint(result)
            provenance_finished = True
            if provenance_complete:
                result.coverage.inspected += 1
        finally:
            context.finish_unit(result, provenance_complete and provenance_finished)
    parser_failed = False
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
            if suffix in {
                ".onnx",
                ".h5",
                ".hdf5",
                ".pb",
                ".tflite",
                ".ckpt",
                ".dill",
                ".joblib",
                ".keras",
                ".so",
                ".dll",
                ".dylib",
                ".exe",
                ".o",
                ".a",
            }:
                result.coverage.gaps.append(
                    f"{entry.name}: format inspection requires a separate validated engine"
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
                ".safetensors",
                ".gguf",
                ".npy",
                ".npz",
                ".md",
                ".rst",
                ".yaml",
                ".yml",
                ".toml",
            }:
                result.coverage.gaps.append(
                    f"{entry.name}: unsupported source format; inventory is not security inspection"
                )
                continue
            path = await context.snapshot(context.target, entry)
            from ablitbench.parsing import ParserCancelled, parse, records

            try:
                parsed = await parse(
                    path,
                    "source",
                    context.store.directory,
                    context.config.runtime.memory_bytes,
                    name=entry.name,
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
                        success = False
                        parser_failed = parser_failed or bool(observation.get("incomplete"))
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
                    if observation.get("coverage_gap"):
                        success = False
                        result.coverage.gaps.append(entry.name + ": " + observation["coverage_gap"])
                    issue = observation.get("issue", "unsafe serialization opcode")
                    if issue in {
                        "model metadata",
                        "format metadata",
                        "chat template recorded; template not executed",
                    }:
                        continue
                    result.findings.append(
                        Finding(
                            check=context.check.id,
                            target=context.target.target.location,
                            title=str(issue),
                            detail=json.dumps({"file": entry.name, **observation}),
                            evidence=[reference],
                            severity="info"
                            if issue == "external dependency declaration"
                            else "high"
                            if "traversal" in issue or "execution sink" in issue
                            else "medium",
                            strength="indicator",
                            attribution="Static capability; intent and reachability unproven",
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
            parser_failed = True
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
        result.status = (
            "error" if parser_failed and not result.coverage.inspected else "unsupported"
        )
        result.reason = "Some source content was not inspected; consult coverage gaps"
    return result
