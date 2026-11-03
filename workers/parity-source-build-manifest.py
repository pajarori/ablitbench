from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--build-lock", required=True, type=Path)
    parser.add_argument("--toolchain-image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    distribution = importlib.metadata.distribution("modelaudit-picklescan")
    binaries = []
    for relative in distribution.files or []:
        if str(relative).endswith((".so", ".pyd", ".dll")):
            path = Path(str(distribution.locate_file(relative)))
            binaries.append(
                {"path": str(relative), "bytes": path.stat().st_size, "sha256": checksum(path)}
            )
    if not binaries:
        raise ValueError("Reviewed native pickle scanner build produced no native extension")
    result = {
        "schema_version": 1,
        "runtime_lock_sha256": checksum(arguments.runtime_lock),
        "build_lock_sha256": checksum(arguments.build_lock),
        "toolchain_image": arguments.toolchain_image,
        "compiler": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "backend_versions": {
            name: importlib.metadata.version(name)
            for name in ("hatchling", "maturin", "setuptools", "wheel")
        },
        "modelaudit_revision": "415819fa0b58d097dd201c0fa3f41eba33431c78",
        "native_distribution": distribution.metadata["Name"],
        "native_version": distribution.version,
        "native_binaries": binaries,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
