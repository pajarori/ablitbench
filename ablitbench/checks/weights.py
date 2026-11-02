from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ablitbench.formats.tensors import (
    gguf_chunks,
    gguf_tensors,
    safetensors,
    statistics,
    tensor_chunks,
    tensor_diff,
)
from ablitbench.models import CheckResult, Coverage, Finding
from ablitbench.parsing import ParserCancelled, parse, records

if TYPE_CHECKING:
    from ablitbench.core import Context


def analyze(path: Path, reference_path: Path | None = None) -> Iterator[dict[str, Any]]:
    gguf = path.suffix.lower() == ".gguf"
    tensors = gguf_tensors(path) if gguf else safetensors(path)
    references = (
        {
            t["name"]: t
            for t in (gguf_tensors(reference_path, False) if gguf else safetensors(reference_path))
        }
        if reference_path
        else {}
    )
    for tensor in tensors:
        stats = tensor.get("statistics") if gguf else statistics(tensor_chunks(path, tensor))
        if not isinstance(stats, dict):
            raise ValueError("Tensor statistics unavailable")
        record: dict[str, Any] = {
            "tensor": tensor,
            "statistics": stats,
            "analysis_dtype": "float64",
            "interpretation": "distribution indicators do not establish poisoning",
            "scope": "full dequantized" if tensor.get("quantized") else "full plain",
        }
        if stats["accumulator_overflow"]:
            yield {"gap": tensor["name"] + ": numerical accumulator overflow"}
        if tensor["dtype"] in {"I64", "U64"} and any(
            abs(stats[key] or 0) >= 2**53 for key in ("min", "max")
        ):
            yield {"gap": tensor["name"] + ": float64 integer precision loss"}
        if reference_path:
            other = references.get(tensor["name"])
            if other is None or other["shape"] != tensor["shape"]:
                yield {"gap": tensor["name"] + ": incompatible/missing reference shape"}
            elif other["dtype"] != tensor["dtype"] or other.get("quantized") != tensor.get(
                "quantized"
            ):
                yield {"gap": tensor["name"] + ": cross-quantization/dtype comparison excluded"}
            else:
                left = gguf_chunks(path, tensor) if gguf else tensor_chunks(path, tensor)
                right = (
                    gguf_chunks(reference_path, other)
                    if gguf
                    else tensor_chunks(reference_path, other)
                )
                record["reference_diff"] = tensor_diff(left, right)
                record["comparison"] = (
                    "matching quantization dequantized"
                    if tensor.get("quantized")
                    else "matching plain dtype"
                )
        yield record


async def run(context: Context) -> CheckResult:
    unsupported = {".pt", ".pth", ".bin", ".ckpt", ".onnx", ".h5"}
    supported = {".safetensors", ".gguf"}
    total = sum(
        1
        for entry in context.target.entries()
        if Path(entry.name).suffix.lower() in supported | unsupported
    )
    result = context.static_result(Coverage(units="tensor files", total=total))
    for entry in context.target.entries():
        suffix = Path(entry.name).suffix.lower()
        if suffix not in supported | unsupported:
            continue
        await context.yield_control()
        if not context.begin_unit(entry.name, result):
            continue
        success = True
        cancelled = False
        parsed = None
        try:
            if suffix in unsupported:
                result.coverage.gaps.append(
                    entry.name + ": tensor decoding requires a format engine"
                )
                continue
            path = await context.snapshot(context.target, entry)
            reference = None
            if context.reference:
                for candidate in context.reference.entries():
                    if candidate.name == entry.name:
                        reference = await context.snapshot(context.reference, candidate)
                        break
                if reference is None:
                    result.coverage.gaps.append(entry.name + ": reference file missing")
            try:
                parsed = await parse(
                    path,
                    "weights",
                    context.store.directory,
                    context.config.runtime.memory_bytes,
                    reference,
                )
            except ParserCancelled as error:
                parsed = error.output
                cancelled = True
            with closing(records(parsed)) as observations:
                for record in observations:
                    await context.yield_control()
                    context.checkpoint(result)
                    if "gap" in record:
                        if record.get("incomplete"):
                            success = False
                            result.evidence.append(
                                context.store.evidence({"file": entry.name, **record})
                            )
                        result.coverage.gaps.append(entry.name + ": " + record["gap"])
                        continue
                    evidence = context.store.evidence({"file": entry.name, **record})
                    result.evidence.append(evidence)
                    if record["statistics"]["nonfinite"]:
                        result.findings.append(
                            Finding(
                                check=context.check.id,
                                target=context.target.target.location,
                                title="Nonfinite tensor values",
                                detail=json.dumps(
                                    {"file": entry.name, "tensor": record["tensor"]["name"]}
                                ),
                                evidence=[evidence],
                            )
                        )
            if cancelled:
                raise ParserCancelled(parsed)
            if success:
                result.coverage.inspected += 1
        except (ValueError, OSError, KeyError, OverflowError, MemoryError) as error:
            success = False
            evidence = context.store.evidence({"file": entry.name, "error": str(error)})
            result.evidence.append(evidence)
            result.coverage.gaps.append(entry.name + ": parser did not complete")
        except BaseException:
            success = False
            raise
        finally:
            if parsed is not None:
                parsed.unlink(missing_ok=True)
            context.finish_unit(result, success)
    if context.target.inventory_scope == "cached":
        result.coverage.gaps.append("Offline cache cannot establish complete weight inventory")
    if result.coverage.total == 0 and context.target.inventory_scope == "full":
        result.status, result.reason = "not_applicable", "No supported tensor files"
    elif result.coverage.gaps or result.coverage.gap_count:
        result.status, result.reason = "unsupported", "Incomplete tensor coverage"
    return result
