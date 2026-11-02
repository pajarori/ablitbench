from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ablitbench.formats.snapshot import secure_open
from ablitbench.models import WorkerRequest


def execute(request: WorkerRequest) -> dict[str, Any]:
    if request.name != "read_file" or set(request.arguments) != {"path"}:
        raise ValueError("Unsupported tool request")
    if request.arguments["path"] != "/fixture/public.txt":
        return {"error": "access denied", "level": "sandbox execution"}
    path = Path(request.fixture_path or "/fixture/public.txt")
    if request.fixture_path is not None:
        if not (
            path.name == "public.txt"
            and path.parent.name == "fixture"
            and path.parent.parent.name.startswith("ablib-observer-")
            and path.parent.parent.parent == Path("/tmp")
            and request.fixture_text is not None
        ):
            raise ValueError("Tool fixture path is outside the observer preparation directory")
    with secure_open(path) as stream:
        content = stream.read()
    if (
        request.fixture_text is not None
        and hashlib.sha256(content).digest()
        != hashlib.sha256(request.fixture_text.encode()).digest()
    ):
        raise ValueError("Tool fixture content differs from controller preparation")
    return {"content": content.decode("utf-8"), "level": "sandbox execution"}


def main() -> None:
    from ablitbench.worker import capacity, isolation

    isolation()
    request = WorkerRequest.model_validate_json(sys.stdin.read())
    capacity(request)
    print(json.dumps(execute(request)), flush=True)


if __name__ == "__main__":
    main()
