from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote

from ablitbench.formats.snapshot import secure_open
from ablitbench.parity.acquisition import acquire, configuration, copy_artifact


def representations(secrets: list[str]) -> list[str]:
    patterns: set[str] = set()
    for value in secrets:
        if value:
            patterns.update(
                (
                    value,
                    quote(value, safe=""),
                    base64.b64encode(value.encode()).decode(),
                    json.dumps(value, ensure_ascii=True)[1:-1],
                )
            )
    return sorted(patterns, key=len, reverse=True)


def clean(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for pattern in representations(secrets):
            value = value.replace(pattern, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {clean(str(key), secrets): clean(child, secrets) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(child, secrets) for child in value]
    return value


class RedactedWriter(io.TextIOBase):
    def __init__(self, destination: TextIO, secrets: list[str]) -> None:
        self.destination = destination
        self.secrets = secrets
        self.pending = ""
        self.finished = False

    def write(self, value: str) -> int:
        if self.finished:
            raise ValueError("Acquisition log is closed")
        self.pending += value
        patterns = representations(self.secrets)
        for pattern in patterns:
            self.pending = self.pending.replace(pattern, "[REDACTED]")
        overlap = max((len(value) for value in patterns), default=1) - 1
        boundary = max(0, len(self.pending) - overlap)
        self.destination.write(self.pending[:boundary])
        self.pending = self.pending[boundary:]
        return len(value)

    def flush(self) -> None:
        if not self.destination.closed:
            self.destination.flush()

    def finish(self) -> None:
        if not self.finished:
            self.destination.write(clean(self.pending, self.secrets))
            self.pending = ""
            self.finished = True
            self.flush()


def known_secrets(settings: dict[str, Any], payload: bool = False) -> list[str]:
    values = []
    for canonical, reference in settings.get("credential_env", {}).items():
        value = os.environ.get(reference)
        if not value:
            continue
        if not payload or (
            not canonical.endswith("_USERNAME") and canonical != "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            values.append(value)
        if canonical == "GOOGLE_APPLICATION_CREDENTIALS":
            with secure_open(Path(value).resolve(strict=True)) as stream:
                credential_data = json.load(stream)
            for name in ("private_key", "client_secret", "refresh_token"):
                if isinstance(credential_data.get(name), str):
                    values.append(credential_data[name])
    return values


def publish_log(source: Path, destination: Path, secrets: list[str]) -> None:
    with (
        source.open(encoding="utf-8", errors="replace") as reader,
        destination.open("w", encoding="utf-8") as stream,
    ):
        writer = RedactedWriter(stream, secrets)
        while chunk := reader.read(64 * 1024):
            writer.write(chunk)
        writer.finish()


def run(request: dict[str, Any]) -> dict[str, Any]:
    allowed = {"operation", "settings", "inputs", "output", "project", "offline"}
    if set(request) - allowed or request.get("project", "modelaudit") != "modelaudit":
        raise ValueError("Unsupported trusted acquisition request")
    if type(request.get("offline", False)) is not bool:
        raise ValueError("Trusted acquisition offline mode must be boolean")
    operation = request.get("operation")
    if request.get("offline", False) and operation != "dvc":
        raise ValueError("Offline invocation forbids remote artifact acquisition")
    settings = request.get("settings", {})
    bindings = request.get("inputs", {})
    if (
        not isinstance(operation, str)
        or not isinstance(settings, dict)
        or not isinstance(bindings, dict)
    ):
        raise ValueError("Malformed acquisition operation/settings/input mapping")
    settings = configuration(operation, settings)
    if set(bindings) - {"source", "repository"} or "source" not in bindings:
        raise ValueError("Acquisition input roles are source and optional repository only")
    if not all(isinstance(value, str) for value in bindings.values()):
        raise ValueError("Acquisition input bindings must be local paths")
    if not isinstance(request.get("output"), str) or not Path(request["output"]).is_absolute():
        raise ValueError("Trusted acquisition output must be an explicit absolute directory")
    output = Path(request["output"])
    if output.is_symlink():
        raise ValueError("Acquisition output cannot be a symlink")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(output.iterdir()):
        raise ValueError("Trusted acquisition output must be fresh")
    secrets = known_secrets(settings)
    payload_secrets = known_secrets(settings, payload=True)
    result: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="ablib-acquisition-") as temporary:
        scratch = Path(temporary).resolve(strict=True)
        workspace = scratch / "workspace"
        workspace.mkdir(mode=0o700)
        environment = {
            "HOME": str(workspace / "home"),
            "XDG_CACHE_HOME": str(workspace / "cache"),
            "XDG_CONFIG_HOME": str(workspace / "config"),
            "HF_HOME": str(workspace / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(workspace / "hf" / "hub"),
            "TMPDIR": str(scratch),
            "TEMP": str(scratch),
            "TMP": str(scratch),
            "ABLIB_TRUSTED_ACQUISITION": "1",
            "DO_NOT_TRACK": "1",
            "NO_COLOR": "1",
        }
        saved = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        for name in ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "HF_HOME"):
            Path(environment[name]).mkdir(parents=True, exist_ok=True, mode=0o700)
        stdout_path, stderr_path = scratch / "stdout", scratch / "stderr"
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                out, err = RedactedWriter(stdout, secrets), RedactedWriter(stderr, secrets)
                try:
                    with redirect_stdout(out), redirect_stderr(err):
                        result = acquire(
                            operation,
                            settings,
                            {role: Path(value) for role, value in bindings.items()},
                            workspace,
                            secret_values=payload_secrets,
                        )
                except (ValueError, OSError, ImportError, RuntimeError) as error:
                    result = {
                        "status": "failed",
                        "operation": operation,
                        "file_count": 0,
                        "coverage": {
                            "inspected": 0,
                            "units": "acquired files",
                            "gaps": [
                                {
                                    "reason": "Acquisition did not complete",
                                    "error_type": type(error).__name__,
                                    "message": clean(str(error), secrets),
                                }
                            ],
                        },
                    }
                finally:
                    secrets.extend(payload_secrets)
                    out.finish()
                    err.finish()
            directory = Path(result.get("artifact_directory", workspace / "artifacts"))
            if not directory.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Acquisition artifact root escaped its private workspace")
            if result.get("status") != "failed" and directory.is_dir():
                copy_artifact(directory, output / "artifacts")
            inventory = workspace / str(result.get("inventory_artifact", "inventory.jsonl"))
            if inventory.is_file() and inventory.resolve().is_relative_to(workspace.resolve()):
                copy_artifact(inventory, output / "inventory.jsonl")
            else:
                (output / "inventory.jsonl").touch()
            result.pop("artifact_directory", None)
            result["artifact_directory"] = "artifacts"
            result["inventory_artifact"] = "inventory.jsonl"
            metadata = clean(result, secrets)
            (output / "acquisition.json").write_text(
                json.dumps(metadata, ensure_ascii=False, default=str), encoding="utf-8"
            )
            native = workspace / "upstream-acquisition.json"
            if native.is_file():
                with native.open(encoding="utf-8") as stream:
                    native_value = clean(json.load(stream), secrets)
                (output / "upstream-acquisition.json").write_text(
                    json.dumps(native_value, ensure_ascii=False, default=str), encoding="utf-8"
                )
            publish_log(stdout_path, output / "acquisition-stdout.txt", secrets)
            publish_log(stderr_path, output / "acquisition-stderr.txt", secrets)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    coverage = result.get("coverage", {})
    gaps = coverage.get("gaps", [])
    gap_count = coverage.get("gap_count", len(gaps))
    return {
        "status": result.get("status", "unconfirmed"),
        "operation": operation,
        "artifact_directory": "artifacts",
        "inventory_artifact": "inventory.jsonl",
        "provenance_artifact": "acquisition.json",
        "file_count": result.get("file_count", 0),
        "bytes": result.get("bytes", 0),
        "coverage": {
            "units": "acquired files",
            "inspected": coverage.get("inspected", 0),
            "gap_count": gap_count,
            "gaps": [
                {
                    "reason": "Acquisition coverage incomplete",
                    "artifact": "acquisition.json",
                    "count": gap_count,
                }
            ]
            if gap_count
            else [],
        },
        "logs": ["acquisition-stdout.txt", "acquisition-stderr.txt"],
    }
