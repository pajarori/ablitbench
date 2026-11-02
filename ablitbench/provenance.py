from __future__ import annotations

import hashlib
from pathlib import Path


def implementation_identity() -> str:
    root = Path(__file__).parent
    checksum = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or not (
            path.suffix == ".py"
            or (relative.parts[0] == "data" and path.suffix in {".json", ".jsonl"})
        ):
            continue
        checksum.update(relative.as_posix().encode() + b"\0")
        checksum.update(path.read_text(encoding="utf-8").encode())
        checksum.update(b"\0")
    return checksum.hexdigest()
