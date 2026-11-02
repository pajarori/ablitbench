from __future__ import annotations

import importlib
import json
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from ablitbench.checks.source import analyze_metadata
from ablitbench.formats.archive import inspect_archive
from ablitbench.formats.arrays import inspect_numpy
from ablitbench.formats.pickle import inspect_pickle
from ablitbench.formats.static import analyze_python, analyze_template
from ablitbench.formats.tensors import load_json
from ablitbench.formats.yaml import load as yaml_load


def dependency_declarations(value: Any, selected: bool = False) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from dependency_declarations(
                item,
                selected
                or key
                in {
                    "dependencies",
                    "optional-dependencies",
                    "requires",
                    "pip",
                    "dependency-groups",
                    "sources",
                },
            )
    elif isinstance(value, list):
        for item in value:
            yield from dependency_declarations(item, selected)
    elif selected and isinstance(value, str) and ("git+" in value or "://" in value):
        yield {"issue": "external dependency declaration", "line": value}


def inspect_source(path: Path, name: str | None = None) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".py":
        yield from analyze_python(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        with path.open("rb") as stream:
            value = load_json(stream)
            yield from analyze_metadata(value)
            if isinstance(value, dict) and "weight_map" in value:
                yield {"issue": "shard map", "weight_map": value["weight_map"]}
    elif suffix in {".pkl", ".pickle", ".pt", ".pth", ".bin"}:
        if zipfile.is_zipfile(path):
            yield from inspect_archive(path)
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.flag_bits & 1 or member.is_dir():
                        continue
                    if member.filename.endswith((".pkl", ".pickle")):
                        with archive.open(member) as stream:
                            for item in inspect_pickle(stream):
                                yield {"member": member.filename, **item}
                    elif not (
                        "/data/" in member.filename and member.filename.rsplit("/", 1)[-1].isdigit()
                    ) and member.filename.rsplit("/", 1)[-1] not in {
                        "version",
                        "byteorder",
                        "serialization_id",
                    }:
                        yield {"gap": "Unvalidated serialized archive payload: " + member.filename}
        else:
            with path.open("rb") as stream:
                yield from inspect_pickle(stream)
    elif suffix == ".npy":
        with path.open("rb") as stream:
            yield from inspect_numpy(stream)
    elif suffix == ".npz":
        yield from inspect_archive(path)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.flag_bits & 1:
                    continue
                if member.filename.endswith(".npy"):
                    with archive.open(member) as stream:
                        for item in inspect_numpy(stream):
                            yield {"member": member.filename, **item}
                else:
                    yield {"gap": "Unknown NumPy archive payload: " + member.filename}
    elif suffix == ".safetensors":
        from ablitbench.formats.tensors import safetensors

        for tensor in safetensors(path):
            yield {
                "issue": "format metadata",
                "format": "safetensors",
                "tensor": tensor,
                "scope": "header, offsets, dtype, and data length; no behavioral inference",
            }
    elif suffix == ".gguf":
        from ablitbench.formats.tensors import gguf_tensors

        for tensor in gguf_tensors(path, False):
            yield {
                "issue": "format metadata",
                "format": "gguf",
                "tensor": tensor,
                "scope": "metadata and tensor descriptors; no behavioral inference",
            }
    elif suffix in {".zip", ".tar", ".gz", ".tgz"}:
        yield from inspect_archive(path)
        yield {"gap": "archive member metadata inspected; nested payloads require a format engine"}
    elif suffix in {".yaml", ".yml", ".toml"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".toml":
            value = tomllib.loads(text)
        else:
            import yaml

            try:
                value = yaml_load(text)
            except (yaml.YAMLError, ValueError):
                yield {
                    "gap": (
                        "YAML configuration has duplicate keys, non-finite values, "
                        "unsupported tags or malformed syntax"
                    )
                }
                return
        yield from analyze_metadata(value)
        yield from dependency_declarations(value)
    elif suffix in {".txt", ".md", ".rst"}:
        filename = Path(name or path.name).name.lower()
        requirements = suffix == ".txt" and filename.startswith(("requirements", "constraints"))
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if requirements and not line.startswith("#") and ("git+" in line or "://" in line):
                    yield {"issue": "external dependency declaration", "line": line.rstrip()}
    elif suffix in {".jinja", ".jinja2"}:
        yield from analyze_template(path.read_text(encoding="utf-8"))
    else:
        yield {"gap": "Unsupported source payload format"}


def main() -> None:
    request = json.load(sys.stdin)
    with Path(request["output"]).open("w", encoding="utf-8") as stream:
        try:
            if sys.platform == "linux":
                resource = importlib.import_module("resource")
                import psutil

                memory = psutil.Process().memory_info()
                baseline = int(getattr(memory, "data", memory.rss))
                capacity = request["memory"] + baseline
                _, hard = resource.getrlimit(resource.RLIMIT_DATA)
                if hard != resource.RLIM_INFINITY:
                    capacity = min(capacity, hard)
                resource.setrlimit(resource.RLIMIT_DATA, (capacity, capacity))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            path = Path(request["path"])
            if request["operation"] == "source":
                observations = inspect_source(path, request.get("name"))
            elif request["operation"] == "weights":
                from ablitbench.checks.weights import analyze

                observations = analyze(
                    path, Path(request["reference"]) if request.get("reference") else None
                )
            else:
                raise ValueError("Unknown parser operation")
            for record in observations:
                stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
                stream.flush()
        except Exception as error:
            stream.write(
                json.dumps(
                    {
                        "gap": f"Parser incomplete ({type(error).__name__})",
                        "incomplete": True,
                        "error_type": type(error).__name__,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            stream.flush()
            raise


if __name__ == "__main__":
    main()
