from __future__ import annotations

import hashlib
from pathlib import Path


def implementation_identity() -> str:
    root = Path(__file__).parent
    checksum = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        checksum.update(path.relative_to(root).as_posix().encode())
        checksum.update(path.read_text(encoding="utf-8").encode())
    return checksum.hexdigest()
