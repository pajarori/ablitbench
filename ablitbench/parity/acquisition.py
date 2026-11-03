from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from ablitbench.formats.snapshot import secure_open

LOCK = threading.Lock()
CREDENTIALS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "JFROG_API_TOKEN",
    "JFROG_ACCESS_TOKEN",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
}
SOURCES = {
    "cloud.s3": "modelaudit.utils.sources.cloud_storage.download_from_cloud",
    "cloud.gcs": "modelaudit.utils.sources.cloud_storage.download_from_cloud",
    "cloud.r2": "modelaudit.utils.sources.cloud_storage.download_from_cloud",
    "jfrog.file": "modelaudit.utils.sources.jfrog.download_artifact",
    "jfrog.folder": "modelaudit.utils.sources.jfrog.download_jfrog_folder",
    "pytorch-hub": "modelaudit.utils.sources.pytorch_hub.download_pytorch_hub_model",
    "dvc": "modelaudit.utils.sources.dvc.resolve_dvc_file_with_metadata",
    "mlflow": "modelaudit.integrations.mlflow.scan_mlflow_model",
}


def operations() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "project": "modelaudit",
            "kind": "inspect",
            "trusted_io": True,
            "source_revision": "415819fa0b58d097dd201c0fa3f41eba33431c78",
            "source_entrypoint": entry,
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["source"],
            "permissions": [] if name == "dvc" else ["external-network"],
            "inputs": ["source", "repository"] if name == "dvc" else ["source"],
        }
        for name, entry in SOURCES.items()
    }


def credential_uri(uri: str) -> bool:
    parsed = urlsplit(uri)
    return bool(
        parsed.username
        or parsed.password
        or any(
            key.lower().replace("-", "_")
            in {
                "sig",
                "key",
                "api_key",
                "apikey",
                "password",
                "passwd",
                "authorization",
                "access_key",
                "access_token",
                "session_token",
                "sas",
            }
            or any(token in key.lower() for token in ("token", "secret", "signature", "credential"))
            for key, value in parse_qsl(parsed.query)
        )
    )


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in SOURCES:
        raise ValueError("Unknown acquisition operation")
    allowed = {
        "uri",
        "pointer",
        "timeout_seconds",
        "max_bytes",
        "max_file_bytes",
        "max_artifact_entries",
        "credential_env",
        "registry_uri",
        "allowed_artifact_uris",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown acquisition setting")
    for key in ("timeout_seconds", "max_bytes", "max_file_bytes", "max_artifact_entries"):
        if key in settings and (type(settings[key]) is not int or settings[key] <= 0):
            raise ValueError(key + " must be an explicit positive integer")
    if operation != "dvc" and "timeout_seconds" not in settings:
        raise ValueError("Remote acquisition requires an explicit request timeout")
    references = settings.get("credential_env", {})
    if (
        not isinstance(references, dict)
        or set(references) - CREDENTIALS
        or not all(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value)
            for value in references.values()
        )
    ):
        raise ValueError("Acquisition credentials must be approved environment references")
    permitted_credentials = (
        {name for name in CREDENTIALS if name.startswith("AWS_")}
        if operation in {"cloud.s3", "cloud.r2"}
        else {"GOOGLE_APPLICATION_CREDENTIALS"}
        if operation == "cloud.gcs"
        else {"JFROG_API_TOKEN", "JFROG_ACCESS_TOKEN"}
        if operation.startswith("jfrog.")
        else CREDENTIALS - {"JFROG_API_TOKEN", "JFROG_ACCESS_TOKEN"}
        if operation == "mlflow"
        else set()
    )
    if set(references) - permitted_credentials:
        raise ValueError("Credential reference is not used by this acquisition operation")
    if operation == "dvc":
        pointer = settings.get("pointer")
        if (
            not isinstance(pointer, str)
            or Path(pointer).is_absolute()
            or ".." in Path(pointer).parts
        ):
            raise ValueError("DVC pointer must be relative to its explicit repository input")
        if references:
            raise ValueError("Local DVC resolution does not receive remote credentials")
    else:
        uri = settings.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError("Acquisition URI is required")
        parsed = urlsplit(uri)
        if credential_uri(uri):
            raise ValueError("Credentials cannot be embedded in acquisition URIs")
        permitted = {
            "cloud.s3": {"s3", "https"},
            "cloud.gcs": {"gs", "gcs", "https"},
            "cloud.r2": {"r2", "https"},
            "jfrog.file": {"https"},
            "jfrog.folder": {"https"},
            "pytorch-hub": {"https"},
            "mlflow": {"models", "runs", "mlflow-artifacts", "https"},
        }
        if parsed.scheme not in permitted[operation]:
            raise ValueError("Unsupported acquisition URI scheme")
    if operation == "mlflow" and "max_artifact_entries" not in settings:
        raise ValueError("MLflow requires an explicit artifact-entry budget")
    if "allowed_artifact_uris" in settings and not isinstance(
        settings["allowed_artifact_uris"], list
    ):
        raise ValueError("allowed_artifact_uris must be a list")
    for uri in [settings.get("registry_uri"), *settings.get("allowed_artifact_uris", [])]:
        if uri is not None:
            if not isinstance(uri, str):
                raise ValueError("MLflow endpoint identities must be text")
            parsed = urlsplit(uri)
            if "," in uri or "\n" in uri:
                raise ValueError("MLflow origin identities cannot contain delimiter characters")
            if credential_uri(uri) or parsed.scheme not in {"https", "s3", "gs", "abfss", "wasbs"}:
                raise ValueError(
                    "MLflow delegated origins must be explicit credential-free remote URIs"
                )
    return dict(settings)


def credential_values(settings: dict[str, Any]) -> dict[str, str]:
    values = {}
    for canonical, reference in settings.get("credential_env", {}).items():
        value = os.environ.get(reference)
        if not value:
            raise ValueError(
                "Acquisition credential environment reference is missing: " + reference
            )
        values[canonical] = value
    return values


def copy_artifact(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError("Downloaded artifact cannot be a symlink")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with secure_open(source) as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
    elif source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            copy_artifact(child, destination / child.name)
    else:
        raise ValueError("Acquisition returned a non-file artifact")


def manifest(root: Path, secrets: list[str]) -> Iterator[dict[str, Any]]:
    patterns: set[bytes] = set()
    for value in secrets:
        if value:
            patterns.update(
                (value.encode(), quote(value, safe="").encode(), base64.b64encode(value.encode()))
            )
    overlap = max((len(value) for value in patterns), default=1) - 1
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in folders:
            if Path(directory, name).is_symlink():
                raise ValueError("Acquisition produced a symlink directory")
        for name in names:
            path = Path(directory, name)
            relative = path.relative_to(root).as_posix()
            if any(value and value in relative for value in secrets):
                raise ValueError("Acquired artifact name reflected a credential; payload withheld")
            with secure_open(path) as stream:
                before = os.fstat(stream.fileno())
                hasher = hashlib.sha256()
                pending = b""
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
                    pending += chunk
                    if any(value in pending for value in patterns):
                        raise ValueError(
                            "Acquired artifact reflected a credential; payload withheld"
                        )
                    pending = pending[-overlap:] if overlap else b""
                digest = hasher.hexdigest()
                after = os.fstat(stream.fileno())
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError("Acquired artifact changed during hashing")
            yield {"path": relative, "bytes": before.st_size, "sha256": digest}


@contextmanager
def transport_scope(
    settings: dict[str, Any], credentials: dict[str, str], observed_secrets: list[str]
) -> Iterator[dict[str, str]]:
    requests = importlib.import_module("requests.sessions")
    native_request = requests.Session.request
    fsspec = importlib.import_module("fsspec")
    native_filesystem = fsspec.filesystem
    created = []

    def request(session: Any, method: str, url: str, **kwargs: Any) -> Any:
        kwargs["timeout"] = settings["timeout_seconds"]
        return native_request(session, method, url, **kwargs)

    def filesystem(protocol: str, **kwargs: Any) -> Any:
        if protocol == "s3":
            kwargs["anon"] = "AWS_ACCESS_KEY_ID" not in credentials
            for source, field in (
                ("AWS_ACCESS_KEY_ID", "key"),
                ("AWS_SECRET_ACCESS_KEY", "secret"),
                ("AWS_SESSION_TOKEN", "token"),
            ):
                if source in credentials:
                    kwargs[field] = credentials[source]
            config = dict(kwargs.get("config_kwargs", {}))
            config.update(
                connect_timeout=settings["timeout_seconds"],
                read_timeout=settings["timeout_seconds"],
            )
            kwargs["config_kwargs"] = config
        elif protocol == "gcs":
            kwargs["token"] = credentials.get("GOOGLE_APPLICATION_CREDENTIALS", "anon")
            kwargs["requests_timeout"] = settings["timeout_seconds"]
        instance = native_filesystem(protocol, **kwargs)
        created.append(instance)
        return instance

    saved_environment = {
        name: os.environ.get(name)
        for name in CREDENTIALS | {"AWS_EC2_METADATA_DISABLED", "MODELAUDIT_JFROG_ALLOWED_HOSTS"}
    }
    for name in CREDENTIALS:
        os.environ.pop(name, None)
    os.environ.update(credentials)
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    if settings.get("uri", "").startswith("https://"):
        os.environ["MODELAUDIT_JFROG_ALLOWED_HOSTS"] = urlsplit(settings["uri"]).hostname or ""
    requests.Session.request = request
    fsspec.__dict__["filesystem"] = filesystem
    try:
        yield credentials
    finally:
        for instance in created:
            credential = getattr(instance, "credentials", None)
            for candidate in (credential, getattr(credential, "credentials", None)):
                token = getattr(candidate, "token", None)
                if isinstance(token, str) and token:
                    observed_secrets.append(token)
        requests.Session.request = native_request
        fsspec.__dict__["filesystem"] = native_filesystem
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def acquire(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    secret_values: list[str] | None = None,
) -> dict[str, Any]:
    try:
        return _acquire(
            operation, settings, inputs, output, secret_values if secret_values is not None else []
        )
    except ImportError as error:
        return {
            "status": "unsupported",
            "operation": operation,
            "coverage": {
                "inspected": 0,
                "units": "acquired files",
                "gaps": [
                    {
                        "reason": "Optional acquisition SDK unavailable",
                        "dependency": error.name or "reviewed upstream SDK",
                    }
                ],
            },
        }


def _acquire(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    secret_values: list[str],
) -> dict[str, Any]:
    from ablitbench.parity.source_runtime import original_module

    configured = configuration(operation, settings)
    if os.environ.get("ABLIB_TRUSTED_ACQUISITION") != "1":
        raise ValueError("Acquisition requires its separate trusted I/O environment")
    if not Path.home().resolve().is_relative_to(output.resolve()):
        raise ValueError("Trusted acquisition HOME must be private to its output workspace")
    identity = original_module("modelaudit", inputs["source"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "artifacts"
    destination.mkdir()
    gaps: list[Any] = []
    credentials: dict[str, str] = {}
    native_limits: dict[str, Any] = {}
    if operation == "dvc":
        module = importlib.import_module("modelaudit.utils.sources.dvc")
        repository = inputs["repository"].resolve(strict=True)
        pointer = repository / configured["pointer"]
        if not pointer.resolve(strict=True).is_relative_to(repository):
            raise ValueError("DVC pointer escapes its repository input")
        resolution = module.resolve_dvc_file_with_metadata(str(pointer))
        if resolution.analysis_incomplete:
            gaps.append(
                {
                    "reason": resolution.incomplete_reason,
                    "unresolved": resolution.unresolved_outputs,
                    "omitted": resolution.omitted_output_count,
                    "limit": resolution.output_limit,
                }
            )
        for value in resolution.resolved_paths:
            path = Path(value).resolve(strict=True)
            if not path.is_relative_to(repository):
                raise ValueError("DVC output escapes its explicit repository input")
            copy_artifact(path, destination / path.relative_to(repository))
    else:
        with (
            LOCK,
            transport_scope(
                configured, credential_values(configured), secret_values
            ) as credentials,
        ):
            if operation.startswith("cloud."):
                module = importlib.import_module("modelaudit.utils.sources.cloud_storage")
                native_limits = {
                    key: value
                    for key, value in vars(module).items()
                    if key.startswith("_MAX_CLOUD_") and isinstance(value, int)
                }
                native_analysis = module.analyze_cloud_target

                async def analyze(*args: Any, **kwargs: Any) -> Any:
                    if "max_artifact_entries" in configured:
                        kwargs["max_objects"] = configured["max_artifact_entries"]
                    metadata = await native_analysis(*args, **kwargs)
                    with (output / "upstream-acquisition.json").open(
                        "w", encoding="utf-8"
                    ) as stream:
                        json.dump(metadata, stream, ensure_ascii=False, default=str)
                    return metadata

                module.__dict__["analyze_cloud_target"] = analyze
                try:
                    acquired = module.download_from_cloud(
                        configured["uri"],
                        cache_dir=destination,
                        max_size=configured.get("max_bytes"),
                        use_cache=False,
                        show_progress=False,
                        selective=False,
                        stream_analyze=False,
                    )
                finally:
                    module.__dict__["analyze_cloud_target"] = native_analysis
                if not Path(acquired).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Cloud downloader returned outside its artifact directory")
            elif operation.startswith("jfrog."):
                module = importlib.import_module("modelaudit.utils.sources.jfrog")
                native_limits = {
                    key: value
                    for key, value in vars(module).items()
                    if key.startswith("_MAX_JFROG_") and isinstance(value, int)
                }
                native_download = module.download_artifact

                def download(*args: Any, **kwargs: Any) -> Any:
                    kwargs["require_same_origin_redirects"] = True
                    return native_download(*args, **kwargs)

                module.__dict__["download_artifact"] = download
                try:
                    kwargs = {
                        "cache_dir": destination,
                        "timeout": configured["timeout_seconds"],
                        "max_size": configured.get("max_bytes"),
                        "api_token": credentials.get("JFROG_API_TOKEN"),
                        "access_token": credentials.get("JFROG_ACCESS_TOKEN"),
                    }
                    if operation == "jfrog.file":
                        download(configured["uri"], **kwargs)
                    else:
                        module.download_jfrog_folder(
                            configured["uri"], selective=False, show_progress=False, **kwargs
                        )
                finally:
                    module.__dict__["download_artifact"] = native_download
            elif operation == "pytorch-hub":
                module = importlib.import_module("modelaudit.utils.sources.pytorch_hub")
                module.download_pytorch_hub_model(
                    configured["uri"], cache_dir=destination, max_size=configured.get("max_bytes")
                )
            else:
                gaps.extend(acquire_mlflow(configured, destination, output, credentials))
    secrets = secret_values + [
        value
        for name, value in credentials.items()
        if not name.endswith("_USERNAME") and name != "GOOGLE_APPLICATION_CREDENTIALS"
    ]
    count = total_bytes = 0
    with (output / "inventory.jsonl").open("w", encoding="utf-8") as stream:
        for record in manifest(destination, secrets):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            total_bytes += record["bytes"]
    if not count:
        gaps.append({"reason": "No artifacts acquired"})
    return {
        "status": "inconclusive" if gaps else "completed",
        "operation": operation,
        "upstream_identity": identity,
        "artifact_directory": str(destination),
        "inventory_artifact": "inventory.jsonl",
        "file_count": count,
        "bytes": total_bytes,
        "coverage": {"inspected": count, "units": "acquired files", "gaps": gaps},
        "limits": {
            key: value
            for key, value in configured.items()
            if key in {"max_bytes", "max_file_bytes", "max_artifact_entries", "timeout_seconds"}
        },
        "upstream_limits": native_limits,
        "provenance": {
            "content_identity": "per-file SHA256",
            "remote_revision": "unversioned unless explicitly selected by URI",
            "claim": "Acquired content, not remote repository execution or trust",
        },
        "scope": "Acquisition only; no model/repository code or model parsers executed",
    }


def acquire_mlflow(
    settings: dict[str, Any], destination: Path, output: Path, credentials: dict[str, str]
) -> list[Any]:
    module = importlib.import_module("modelaudit.integrations.mlflow")
    core = importlib.import_module("modelaudit.core")
    models = importlib.import_module("modelaudit.models")
    native_scan = core.scan_model_directory_or_file
    captured = []

    def preserve(path: str, **kwargs: Any) -> Any:
        copy_artifact(Path(path), destination / "model")
        captured.append(path)
        return models.create_initial_audit_result()

    core.__dict__["scan_model_directory_or_file"] = preserve
    saved = {name: os.environ.get(name) for name in credentials}
    allowed_name = "MODELAUDIT_MLFLOW_ALLOWED_ARTIFACT_URIS"
    previous_allowed = os.environ.get(allowed_name)
    try:
        os.environ.update(credentials)
        os.environ[allowed_name] = ",".join(settings.get("allowed_artifact_uris", []))
        result = module.scan_mlflow_model(
            settings["uri"],
            registry_uri=settings.get("registry_uri"),
            timeout=settings["timeout_seconds"],
            cache_dir=str(output / "staging"),
            max_file_size=settings.get("max_file_bytes", 0),
            max_total_size=settings.get("max_bytes", 0),
            max_mlflow_artifact_entries=settings["max_artifact_entries"],
        )
        if not captured:
            return [
                {
                    "reason": (
                        "Native MLflow preflight/download did not produce a parser-free artifact"
                    ),
                    "upstream_result": result.model_dump(mode="json"),
                }
            ]
        return []
    finally:
        core.__dict__["scan_model_directory_or_file"] = native_scan
        for key, value in {**saved, allowed_name: previous_allowed}.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
