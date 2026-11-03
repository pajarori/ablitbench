from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ARTIFACTS = (
    (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "4328cf26390c98c5e3c738b4460a05b95f4911f5",
        1110068629,
        "253e00bb467fcdcac714a7b2443330c28ffbecb6d1f791c92caaf2af468bfbaa",
    ),
    (
        "BAAI/bge-reranker-base",
        "2cfc18c9415c912f9d8155881c133215df768a70",
        1112459588,
        "15b9a8c3da82eddf263df571281166e00e9308fe19d077084b642ebfcaf06d2b",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    cache = arguments.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache.parent)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download

    results = []
    for repository, commit, size, expected in ARTIFACTS:
        downloaded = Path(
            hf_hub_download(
                repo_id=repository,
                filename="onnx/model.onnx",
                revision=commit,
                cache_dir=cache,
                token=False,
            )
        )
        resolved = downloaded.resolve(strict=True)
        if not resolved.is_relative_to(cache) or resolved.stat().st_size != size:
            raise ValueError("Pinned public ONNX artifact size/cache identity mismatch")
        with resolved.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Pinned public ONNX artifact checksum mismatch")
        results.append(
            {
                "repository": repository,
                "commit": commit,
                "member": "onnx/model.onnx",
                "bytes": size,
                "sha256": actual,
                "cache_path": str(downloaded.relative_to(cache)),
            }
        )
    arguments.manifest.write_text(
        json.dumps(
            {
                "artifacts": results,
                "model_parsing": False,
                "authentication": "explicit token=False",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
