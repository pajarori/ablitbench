from __future__ import annotations

import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from ablitbench.checks.source import analyze_metadata, analyze_python
from ablitbench.formats.archive import inspect_archive
from ablitbench.formats.pickle import inspect_pickle
from ablitbench.formats.tensors import load_json


def inspect_source(path: Path) -> Any:
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
                    if member.filename.endswith((".pkl", ".pickle")):
                        with archive.open(member) as stream:
                            for item in inspect_pickle(stream):
                                yield {"member": member.filename, **item}
        else:
            with path.open("rb") as stream:
                yield from inspect_pickle(stream)
    elif suffix in {".zip", ".tar", ".gz", ".tgz"}:
        yield from inspect_archive(path)
        yield {"gap": "archive member metadata inspected; nested payloads require a format engine"}
    elif suffix == ".txt":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if "git+" in line or "://" in line:
                    yield {"issue": "external dependency declaration", "line": line.rstrip()}
    elif suffix in {".jinja", ".jinja2"}:
        yield {"issue": "chat template recorded; template not executed"}


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
                observations = inspect_source(path)
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
