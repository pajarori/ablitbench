from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import cache
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


@cache
def catalog() -> dict[str, Any]:
    path = Path(__file__).parent.parent / "data" / "parity-layerfault.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def operations() -> dict[str, dict[str, Any]]:
    result = {}
    metadata = catalog()
    for item in metadata["commands"]:
        operation = item["operation"]
        kind = "inspect"
        capabilities = ["isolated-execution"]
        if operation in {
            "run",
            "serve",
            "behaviour",
            "compare-behaviour",
            "review",
            "research.trigger-search",
            "research.backdoor",
            "research.activation-diff",
            "research.trigger-hunt",
        }:
            kind = "runtime"
            capabilities.append("inference")
        elif operation.startswith(("verify", "pipeline", "policy", "attest")):
            kind = "admit"
        elif operation.startswith(("baseline", "quarantine", "inventory", "trust")):
            kind = "runtime"
            capabilities.append("private-state")
        if operation.startswith(("hub.", "platform.")):
            capabilities.append("external-network")
        primary_artifact = operation == "behaviour" or any(
            field["required"]
            and field["name"] in {"model", "path", "target"}
            and "PathBuf" in field["type"]
            for group in item["groups"]
            for field in group["fields"]
        )
        result[operation] = {
            "project": "layerfault",
            "kind": kind,
            "capabilities": capabilities,
            "inputs": ["source", "model"] if primary_artifact else ["source"],
            "model_input": "artifact",
            "optional_inputs": sorted(
                {
                    "model",
                    "base",
                    "reference",
                    "dataset",
                    "store",
                    "artifact",
                    "smtp_username",
                    "smtp_password",
                    "database_state",
                }
                | {
                    field["name"]
                    for group in item["groups"]
                    for field in group["fields"]
                    if "PathBuf" in field["type"]
                }
            ),
            "secret_inputs": [
                "private_key",
                "evidence_key",
                "sign_with",
                "smtp_username",
                "smtp_password",
            ],
            "source_revision": metadata["revision"],
            "source_entrypoint": item["source"] + ":" + str(item["line"]),
            "command": item["command"],
            "argument_groups": item["groups"],
            "requires_inference": "inference" in capabilities,
            "requires_tools": operation != "behaviour.profiles"
            and (
                kind == "runtime"
                or operation == "import"
                or operation.startswith("platform.")
                or any(
                    field["name"] == "runtime_path"
                    for group in item["groups"]
                    for field in group["fields"]
                )
            ),
            "modes": ["source", "api"] if operation.startswith("research.") else ["source"],
            "permissions": [
                value
                for value in capabilities
                if value
                in {
                    "external-network",
                    "external-message",
                    "signing",
                    "custom-model-code",
                    "host-state",
                }
            ],
        }
    return result


def validate_value(value: Any, field: dict[str, Any]) -> None:
    kind = field["type"]
    if kind.startswith("Option<"):
        kind = kind[7:-1]
    if kind.startswith("Vec<"):
        if not isinstance(value, list):
            raise ValueError(field["name"] + " requires a list")
        for child in value:
            validate_value(child, dict(field, type=kind[4:-1]))
        return
    if kind == "bool":
        valid = isinstance(value, bool)
    elif kind in {"u64", "usize", "i64", "u32", "u16"}:
        valid = isinstance(value, int) and not isinstance(value, bool)
        if kind != "i64":
            valid = valid and value >= 0
    elif kind == "f64":
        import math

        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid = valid and math.isfinite(value)
    elif kind == "PathBuf":
        valid = isinstance(value, dict) and set(value) in (
            {"input"},
            {"input", "path"},
            {"output"},
            {"installed"},
        )
        if valid:
            valid = all(isinstance(child, str) and "\0" not in child for child in value.values())
    else:
        if isinstance(value, dict) and field["name"] in {"model", "base", "reference"}:
            validate_value(value, dict(field, type="PathBuf"))
            return
        valid = isinstance(value, str) and "\0" not in value and not value.startswith("-")
    if not valid:
        raise ValueError(field["name"] + " has an invalid value for " + kind)


def required_permissions(operation: str, settings: dict[str, Any]) -> list[str]:
    permissions = set(operations()[operation]["permissions"])
    groups = [settings.get("arguments", {}), *settings.get("parent_arguments", {}).values()]
    for values in groups:
        runtime_path = values.get("runtime_path", {})
        if values.get("execute_custom_code") or (
            isinstance(runtime_path, dict) and "input" in runtime_path
        ):
            permissions.add("custom-model-code")
        if any(values.get(name) for name in ("private_key", "evidence_key", "sign_with")) or any(
            isinstance(value, dict)
            and value.get("input") in {"private_key", "evidence_key", "sign_with"}
            for value in values.values()
        ):
            permissions.add("signing")
        if values.get("allow_cloud_judge") or values.get("judge_endpoint"):
            permissions.add("external-network")
    if (
        settings.get("private_platform")
        and operation != "platform.crawl"
        and not settings.get("public_hub_jobs", False)
    ):
        permissions.discard("external-network")
    if (
        operation == "platform.newsletter.send"
        and not settings.get("arguments", {}).get("dry_run", False)
        and not settings.get("private_platform")
    ):
        permissions.add("external-message")
    return sorted(permissions)


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    metadata = operations().get(operation)
    if metadata is None:
        raise ValueError("Unknown Layerfault operation: " + operation)
    if set(settings) - {
        "arguments",
        "parent_arguments",
        "global",
        "workspace_store",
        "private_platform",
        "service_timeout_seconds",
        "service_requests",
        "runtime_environment",
        "hub_timeout_seconds",
        "public_hub_jobs",
    }:
        raise ValueError("Unknown Layerfault settings")
    if "public_hub_jobs" in settings and (
        type(settings["public_hub_jobs"]) is not bool
        or operation != "platform.worker"
        or not settings.get("private_platform")
    ):
        raise ValueError("Public Hub jobs require the explicit private platform.worker profile")
    if "hub_timeout_seconds" in settings and (
        type(settings["hub_timeout_seconds"]) is not int
        or not 0 < settings["hub_timeout_seconds"] <= 120
    ):
        raise ValueError("Hub timeout must fit the original native 120-second request guard")
    if settings.get("runtime_environment") not in {None, "image-python"}:
        raise ValueError("Runtime environment must be the reviewed image Python prefix")
    arguments = settings.get("arguments", {})
    if "private_platform" in settings:
        if settings["private_platform"] not in {"sqlite", "postgres"} or not operation.startswith(
            "platform."
        ):
            raise ValueError("Private platform requires its sqlite/postgres service profile")
        if not isinstance(arguments, dict):
            raise ValueError("Private platform arguments must be an object")
        arguments = dict(arguments)
        if arguments.get("database", "private") != "private":
            raise ValueError("Private service profile uses its generated database identity")
        arguments["database"] = "private"
        if operation == "platform.newsletter.send":
            if arguments.get("smtp_host", "private") != "private":
                raise ValueError("Private SMTP profile accepts only its generated sink")
            arguments.setdefault("smtp_host", "private")
            arguments.setdefault("to", "recipient@example.invalid")
            arguments.setdefault("from", "sender@example.invalid")
            for field, name in (
                ("username_env", "LAYERFAULT_SMTP_USERNAME"),
                ("password_env", "LAYERFAULT_SMTP_PASSWORD"),
            ):
                if arguments.get(field, name) != name:
                    raise ValueError("Private SMTP profile owns its generated credential bindings")
        if settings["private_platform"] == "postgres" and (
            type(settings.get("service_timeout_seconds")) is not int
            or settings["service_timeout_seconds"] <= 0
        ):
            raise ValueError("Private PostgreSQL requires an explicit service timeout")
        settings = dict(settings, arguments=arguments)
    if "service_requests" in settings or (
        operation == "platform.serve" and settings.get("private_platform")
    ):
        requests = settings.get("service_requests")
        if operation != "platform.serve" or not settings.get("private_platform"):
            raise ValueError("Service requests require the private platform.serve profile")
        if not isinstance(requests, list) or not requests:
            raise ValueError("Private platform.serve requires explicit finite service requests")
        if (
            type(settings.get("service_timeout_seconds")) is not int
            or settings["service_timeout_seconds"] <= 0
        ):
            raise ValueError("Private platform.serve requires an explicit lifecycle timeout")
        for request in requests:
            if not isinstance(request, dict) or set(request) - {"method", "path", "payload"}:
                raise ValueError("Service requests accept method, path and JSON payload only")
            path = request.get("path")
            if request.get("method", "GET") not in {"GET", "HEAD", "POST"} or not isinstance(
                path, str
            ):
                raise ValueError("Invalid private service request method/path")
            if (
                not path.startswith("/")
                or path.startswith("//")
                or any(char in path for char in "\r\n\0")
            ):
                raise ValueError("Service request must use a same-origin absolute path")
        if settings["arguments"].get("listen", "private") != "private":
            raise ValueError("Private platform.serve owns its generated loopback listen identity")
    if operation.startswith("platform.") and isinstance(arguments, dict):
        database = arguments.get("database")
        if isinstance(database, str) and (
            urlsplit(database).password is not None
            or "password=" in database.lower().replace(" ", "")
        ):
            raise ValueError(
                "Database passwords require a trusted broker profile, not literal arguments"
            )
    if (
        operation == "scan-dir"
        and isinstance(arguments, dict)
        and arguments.get("recursive") is False
    ):
        raise ValueError("Upstream scan-dir recursive flag defaults true and cannot express false")
    parents = settings.get("parent_arguments", {})
    if not isinstance(arguments, dict) or not isinstance(parents, dict):
        raise ValueError("Layerfault arguments must be objects")
    groups = metadata["argument_groups"]
    known_parents = {".".join(metadata["command"][: group["depth"]]) for group in groups[:-1]}
    if set(parents) - known_parents:
        raise ValueError("Unknown Layerfault parent argument group")
    for index, group in enumerate(groups):
        name = ".".join(metadata["command"][: group["depth"]])
        values = arguments if index == len(groups) - 1 else parents.get(name, {})
        if not isinstance(values, dict):
            raise ValueError("Layerfault argument group must be an object")
        fields = {field["name"]: field for field in group["fields"]}
        if set(values) - set(fields):
            raise ValueError("Unknown Layerfault arguments in " + name)
        for field_name, field in fields.items():
            if field_name in values:
                if field_name == "runtime_args":
                    raise ValueError(
                        "Runtime passthrough requires a reviewed typed backend contract"
                    )
                validate_value(values[field_name], field)
            elif field["required"]:
                raise ValueError("Missing Layerfault argument " + name + "." + field_name)
    global_arguments = settings.get("global", {})
    if not isinstance(global_arguments, dict) or set(global_arguments) - {"no_cache", "cache_dir"}:
        raise ValueError("Invalid Layerfault global arguments")
    if "no_cache" in global_arguments and not isinstance(global_arguments["no_cache"], bool):
        raise ValueError("no_cache must be boolean")
    if "cache_dir" in global_arguments:
        validate_value(global_arguments["cache_dir"], {"name": "cache_dir", "type": "PathBuf"})
    store = settings.get("workspace_store")
    if store is not None and not isinstance(store, str):
        raise ValueError("workspace_store must name an input role")
    permissions = required_permissions(operation, settings)
    if "signing" in permissions and (
        "custom-model-code" in permissions or metadata["requires_inference"]
    ):
        raise ValueError("Signing must run separately from model inference and custom code")
    return settings


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\0" in value:
        raise ValueError("Path reference must stay inside its declared root")
    return path


def resolve_path(value: dict[str, str], inputs: dict[str, Path], output: Path) -> str:
    if "input" in value:
        if value["input"] not in inputs:
            raise ValueError("Missing input role: " + value["input"])
        root = inputs[value["input"]]
        path = root / relative_path(value.get("path", "."))
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("Layerfault input path symlink rejected")
        return str(path)
    if "output" in value:
        path = output / relative_path(value["output"])
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("Layerfault output path symlink rejected")
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    path = Path(value["installed"])
    if not path.is_absolute() or not any(
        path.is_relative_to(prefix)
        for prefix in (Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/parity/bin"))
    ):
        raise ValueError("Installed runtime path is outside the reviewed image namespaces")
    if ".." in path.parts:
        raise ValueError("Installed runtime path traversal rejected")
    return str(path)


def compile_command(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> list[str]:
    settings = configuration(operation, settings)
    metadata = operations()[operation]
    command = ["/opt/parity/bin/layerfault"]
    global_arguments = settings.get("global", {})
    if global_arguments.get("no_cache"):
        command.append("--no-cache")
    if "cache_dir" in global_arguments:
        command.append("--cache-dir=" + resolve_path(global_arguments["cache_dir"], inputs, output))
    consumed = 0
    for index, group in enumerate(metadata["argument_groups"]):
        command.extend(metadata["command"][consumed : group["depth"]])
        consumed = group["depth"]
        name = ".".join(metadata["command"][: group["depth"]])
        values = (
            settings.get("arguments", {})
            if index == len(metadata["argument_groups"]) - 1
            else settings.get("parent_arguments", {}).get(name, {})
        )
        values = dict(values)
        names = {field["name"] for field in group["fields"]}
        if index == len(metadata["argument_groups"]) - 1 and "json" in names:
            if "json" not in values and not any(
                values.get(name) for name in ("sarif", "jsonl", "summary", "evidence_report")
            ):
                values["json"] = True
        for field in group["fields"]:
            if field["name"] not in values:
                continue
            value = values[field["name"]]
            if field["type"] == "bool":
                if value:
                    command.append(field["option"])
                continue
            for item in value if field["type"].startswith("Vec<") else [value]:
                token = resolve_path(item, inputs, output) if isinstance(item, dict) else str(item)
                command.append(field["option"] + "=" + token if field["option"] else token)
    return command


def assessment(
    result: Any, returncode: int, operation: str | None = None
) -> tuple[list[Any], list[str], bool]:
    findings = []
    gaps = []
    pending = [result]
    assessment_shape = False
    baseline_shape = False
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            if "error" in item:
                gaps.append("Upstream command reported an error")
            if isinstance(item.get("matches"), bool) and "changed_models" in item:
                baseline_shape = assessment_shape = True
                if item["matches"] is False:
                    findings.append(item)
            if isinstance(item.get("valid"), bool) and "detail" in item:
                assessment_shape = True
                if item["valid"] is False:
                    findings.append(item)
            if any(
                key in item
                for key in ("findings", "results", "policy", "decision", "admission", "checks")
            ):
                assessment_shape = True
            if item.get("status") in {"Warn", "Fail", "WARN", "FAIL", "BLOCK"} and any(
                key in item for key in ("check_type", "rule_id", "finding_id")
            ):
                findings.append(item)
                assessment_shape = True
            elif "ruleId" in item and isinstance(item.get("message"), dict):
                findings.append(item)
                assessment_shape = True
            if (
                any(
                    item.get(key) is True
                    for key in (
                        "analysis_incomplete",
                        "control_interrupted",
                        "limit_reached",
                        "timed_out",
                    )
                )
                or item.get("complete") is False
            ):
                gaps.append("Upstream reports incomplete coverage")
            pending.extend(item.values())
    if operation == "composition.verify-merge" and isinstance(result, dict):
        state = result.get("state")
        assessment_shape = True
        if state == "inconsistent":
            findings.append(dict(result, rule_id="upstream.composition.merge-inconsistent"))
        elif state not in {"verified", "consistent"}:
            gaps.append("Native numerical merge verification is " + str(state))
    if operation == "composition.adapter" and isinstance(result, dict):
        relation = result.get("base_relation")
        if relation == "mismatch":
            findings.append(dict(result, rule_id="upstream.composition.base-mismatch"))
        elif relation not in {"match", "declared_name_only"}:
            gaps.append("Native adapter base relationship is " + str(relation))
    if operation == "policy.diff" and isinstance(result, dict) and returncode in {0, 1}:
        assessment_shape = True
        if result:
            findings.append({"rule_id": "upstream.policy.difference", "detail": result})
    if returncode not in {0, 1, 2, 3, 4} and not (returncode == 5 and baseline_shape):
        gaps.append("Upstream command did not complete successfully")
    elif returncode and not assessment_shape:
        gaps.append("Nonzero command exit without a structured security assessment")
    return findings, gaps, not gaps


def copy_store(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Workspace store requires a directory without symlinks")
    for root, directories, files in os.walk(source, followlinks=False):
        relative = Path(root).relative_to(source)
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        for name in directories + files:
            if Path(root, name).is_symlink():
                raise ValueError("Workspace store symlink rejected")
        for name in files:
            shutil.copyfile(Path(root, name), target / name)


def run_native(
    command: list[str],
    environment: dict[str, str],
    output: Path,
    secrets: list[bytes],
    observer: Callable[[Any], None] | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=output,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    patterns: set[bytes] = set()
    for secret in secrets:
        if secret:
            patterns.update(
                (secret, secret.strip(), base64.b64encode(secret), secret.hex().encode())
            )
    ordered = sorted((value for value in patterns if value), key=len, reverse=True)
    overlap = max((len(value) for value in ordered), default=1) - 1
    errors = []

    def capture(stream: Any, name: str) -> None:
        pending = b""
        try:
            with (output / name).open("wb") as destination:
                while chunk := stream.read(64 * 1024):
                    pending += chunk
                    for pattern in ordered:
                        pending = pending.replace(pattern, b"[REDACTED]")
                    boundary = max(0, len(pending) - overlap)
                    destination.write(pending[:boundary])
                    pending = pending[boundary:]
                for pattern in ordered:
                    pending = pending.replace(pattern, b"[REDACTED]")
                destination.write(pending)
        except Exception as error:
            errors.append(error)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            stream.close()

    streams = [(process.stdout, "stdout"), (process.stderr, "stderr")]
    threads = [threading.Thread(target=capture, args=values, daemon=True) for values in streams]
    for thread in threads:
        thread.start()
    observer_error = None
    if observer is not None:
        try:
            observer(process)
        except BaseException as error:
            observer_error = error
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    returncode = process.wait()
    while any(thread.is_alive() for thread in threads):
        if errors:
            raise OSError(
                "Native output could not be persisted; process group stopped"
            ) from errors[0]
        for thread in threads:
            thread.join(0.1)
    if errors:
        raise OSError("Native output could not be persisted; process group stopped") from errors[0]
    if observer_error is not None:
        raise observer_error
    return returncode


@contextmanager
def judge_bridge(broker: Any, endpoint: str, model: str) -> Iterator[str]:
    target = urlsplit(endpoint)
    if target.scheme != "https" or not target.hostname or target.username or target.password:
        raise ValueError("Judge bridge requires an explicit HTTPS endpoint identity")
    if target.query or target.fragment:
        raise ValueError("Judge bridge endpoint cannot contain query or fragment")
    path = target.path or "/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            if self.path != path:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length < 0 or self.headers.get("transfer-encoding"):
                    raise ValueError("Bridge requires a declared JSON body length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or payload.get("model") != model:
                    raise ValueError("Bridge request does not match the configured judge model")
                result = broker(
                    {
                        "kind": "http",
                        "channel": "judge",
                        "method": "POST",
                        "path": path,
                        "payload": payload,
                        "auth": "configured",
                        "auth_style": "bearer",
                        "headers": {"content-type": "application/json"},
                    }
                )
                body = str(result.get("body", "")).encode("utf-8")
                self.send_response(int(result.get("status", 502)))
                for key, value in result.get("headers", {}).items():
                    if key.lower() not in {
                        "content-length",
                        "content-encoding",
                        "transfer-encoding",
                        "connection",
                    }:
                        self.send_header(key, str(value))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_error(502, "Configured judge broker request failed")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port) + path
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def hub_bridge(broker: Any, timeout_seconds: int) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            handle = None
            headers_sent = False
            try:
                if self.path != "/hub":
                    raise ValueError("Unknown native Hub bridge route")
                url = self.headers.get("x-ablitbench-hub-url")
                cap = self.headers.get("x-ablitbench-hub-limit", "")
                if not url or not cap.isascii() or not cap.isdigit() or int(cap) <= 0:
                    raise ValueError(
                        "Native Hub bridge requires original URL and native byte budget"
                    )
                request = {
                    "kind": "hub",
                    "action": "open",
                    "url": url,
                    "timeout_seconds": timeout_seconds,
                    "max_response_bytes": int(cap),
                }
                if range_header := self.headers.get("Range"):
                    import re

                    matched = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
                    if matched is None or int(matched[2]) < int(matched[1]):
                        raise ValueError("Invalid native Hub byte range")
                    request["range"] = {
                        "offset": int(matched[1]),
                        "length": int(matched[2]) - int(matched[1]) + 1,
                    }
                opened = broker(request)
                if opened.get("error"):
                    raise ValueError(str(opened["error"]))
                handle = opened["handle"]
                self.send_response(opened["status"])
                for name, value in opened.get("headers", {}).items():
                    if name.lower() not in {"content-length", "transfer-encoding", "connection"}:
                        self.send_header(name, value)
                self.send_header("content-length", str(opened["bytes"]))
                self.end_headers()
                headers_sent = True
                while True:
                    chunk = broker({"kind": "hub", "action": "read", "handle": handle})
                    body = base64.b64decode(chunk["body_base64"], validate=True)
                    if body:
                        self.wfile.write(body)
                    if chunk["eof"]:
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                if headers_sent:
                    self.close_connection = True
                    return
                body = json.dumps({"error": str(error)}).encode()
                try:
                    self.send_response(400)
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass
            finally:
                if handle is not None:
                    try:
                        broker({"kind": "hub", "action": "close", "handle": handle})
                    except Exception:
                        pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_address[1]) + "/hub"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def execute(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    broker: Any = None,
) -> dict[str, Any]:
    from ablitbench.parity.sources import verified_source
    from ablitbench.worker import isolation

    isolation()
    settings = configuration(operation, settings)
    if operation == "platform.serve" and not settings.get("private_platform"):
        raise ValueError("Native platform.serve requires its finite private service profile")
    if "source" not in inputs:
        raise ValueError("Layerfault requires its reviewed source input")
    verified_source(inputs["source"], "layerfault")
    binary = Path("/opt/parity/bin/layerfault")
    build_path = Path("/opt/parity/layerfault-build.json")
    if not binary.is_file() or not build_path.is_file():
        raise ValueError("Layerfault binary/build identity is unavailable")
    build = json.loads(build_path.read_text())
    with binary.open("rb") as binary_stream:
        checksum = hashlib.file_digest(binary_stream, "sha256").hexdigest()
    if build.get("source_revision") != catalog()["revision"] or checksum != build.get(
        "binary_sha256"
    ):
        raise ValueError("Layerfault binary provenance mismatch")
    output.mkdir(parents=True, exist_ok=True)
    if "workspace_store" in settings:
        role = settings["workspace_store"]
        if role not in inputs:
            raise ValueError("Missing workspace store input role")
        copy_store(inputs[role], output / "store")
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(output / "home"),
        "XDG_CACHE_HOME": str(output / "cache"),
        "XDG_CONFIG_HOME": str(output / "config"),
        "OLLAMA_MODELS": str(output / "store"),
        "HF_HUB_OFFLINE": "1",
        "DO_NOT_TRACK": "1",
    }
    for key in ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "OLLAMA_MODELS"):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    if settings.get("runtime_environment") == "image-python":
        environment["VIRTUAL_ENV"] = sys.prefix
    arguments = settings.get("arguments", {})
    secret_values = []
    for role in operations()[operation]["secret_inputs"]:
        if role in inputs:
            secret_values.append(inputs[role].read_bytes())
    if operation == "platform.newsletter.send":
        for role, field, default in (
            ("smtp_username", "username_env", "LAYERFAULT_SMTP_USERNAME"),
            ("smtp_password", "password_env", "LAYERFAULT_SMTP_PASSWORD"),
        ):
            if role in inputs:
                name = arguments.get(field, default)
                if name in environment or name.startswith("ABLIB_") or not name.isidentifier():
                    raise ValueError("SMTP environment name conflicts with the worker environment")
                environment[name] = inputs[role].read_text(encoding="utf-8").rstrip("\r\n")
    bridge: AbstractContextManager[str | None] = nullcontext(None)
    if operation == "review" and arguments.get("judge") == "openai-compatible":
        if (
            broker is None
            or not arguments.get("judge_endpoint")
            or not arguments.get("judge_model")
        ):
            raise ValueError("Native cloud judge requires its configured host broker and model")
        if (
            build.get("broker_transport_patch") != catalog()["broker_transport_patch"]["sha256"]
            or build.get("patched_source_sha256")
            != catalog()["broker_transport_patch"]["patched_sha256"]
        ):
            raise ValueError("Native binary lacks the audited loopback judge transport contract")
        bridge = judge_bridge(broker, arguments["judge_endpoint"], arguments["judge_model"])
    hub_service: AbstractContextManager[str | None] = nullcontext(None)
    if (
        operation.startswith("hub.")
        or operation == "platform.crawl"
        or (
            operation == "platform.worker"
            and (
                settings.get("public_hub_jobs", False)
                or (
                    broker is not None
                    and build.get("hub_transport_patch")
                    == catalog()["hub_transport_patch"]["sha256"]
                )
            )
        )
    ):
        if broker is None:
            raise ValueError("Native Hugging Face operations require their scoped host Hub broker")
        metadata = catalog()["hub_transport_patch"]
        if (
            build.get("hub_transport_patch") != metadata["sha256"]
            or build.get("hub_source_sha256") != metadata["patched_sha256"]
            or build.get("hub_range_source_sha256") != metadata["range_patched_sha256"]
        ):
            raise ValueError("Native binary lacks the reviewed scoped Hugging Face transport shim")
        hub_service = hub_bridge(broker, settings.get("hub_timeout_seconds", 120))
    service: AbstractContextManager[tuple[dict[str, Any], dict[str, Any]]] = nullcontext(
        (settings, {})
    )
    if settings.get("private_platform"):
        if operation == "platform.newsletter.send" and not arguments.get("dry_run", False):
            if (
                build.get("broker_transport_patch") != catalog()["broker_transport_patch"]["sha256"]
                or build.get("smtp_source_sha256")
                != catalog()["broker_transport_patch"]["smtp"]["patched_sha256"]
            ):
                raise ValueError("Native binary lacks its audited private SMTP sink transport")
        from ablitbench.parity.layerfault_platform import profile

        service = profile(operation, settings, inputs, output, environment)
    with (
        service as (service_settings, service_report),
        bridge as endpoint,
        hub_service as hub_endpoint,
    ):
        effective = dict(service_settings, arguments=dict(service_settings.get("arguments", {})))
        if endpoint:
            effective["arguments"]["judge_endpoint"] = endpoint
            environment[arguments.get("judge_api_key_env", "LAYERFAULT_JUDGE_API_KEY")] = (
                "ABLIB_REDACTED_CREDENTIAL"
            )
            environment["ABLIB_BROKER_ENDPOINT"] = endpoint
        if hub_endpoint:
            environment["ABLIB_HUB_BROKER_ENDPOINT"] = hub_endpoint
        compiled = {
            key: value
            for key, value in effective.items()
            if key
            not in {
                "private_platform",
                "service_timeout_seconds",
                "service_requests",
                "runtime_environment",
                "hub_timeout_seconds",
                "public_hub_jobs",
            }
        }
        command = compile_command(operation, compiled, inputs, output)
        observer = None
        if operation == "platform.serve" and settings.get("private_platform"):
            from ablitbench.parity.layerfault_platform import observe_service

            def observer(process: Any) -> None:
                observe_service(process, effective, output, service_report)

        returncode = run_native(command, environment, output, secret_values, observer)
    result: Any = None
    parse_gaps = []
    if operation == "platform.worker":
        counts: dict[str, int] = {}
        with (output / "stdout").open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    outcome = json.loads(line)
                    if not isinstance(outcome, dict) or not all(
                        isinstance(outcome.get(key), str)
                        for key in ("job_id", "kind", "status", "detail")
                    ):
                        raise ValueError("Invalid native worker outcome")
                    status = outcome["status"]
                    counts[status] = counts.get(status, 0) + 1
                except ValueError:
                    parse_gaps.append("Native worker outcome stream contains an invalid record")
        result = {
            "status_counts": counts,
            "native_outcomes_artifact": "stdout",
            "database_snapshot": service_report.get("database_snapshot"),
        }
        if counts.get("FAILED", 0):
            parse_gaps.append(
                "Native worker reported failed job attempts: " + str(counts["FAILED"])
            )
        if set(counts) - {"SUCCEEDED", "FAILED"}:
            parse_gaps.append("Native worker reported unknown outcome states")
    elif "--jsonl" in command:
        result = []
        with (output / "stdout").open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.strip():
                    try:
                        result.append(json.loads(line))
                    except ValueError:
                        parse_gaps.append("Native JSONL output contains an invalid record")
    else:
        try:
            with (output / "stdout").open(encoding="utf-8", errors="replace") as stream:
                result = json.load(stream)
        except ValueError:
            if "--json" in command or "--sarif" in command:
                parse_gaps.append("Native JSON output could not be decoded")
    if "--summary" in command:
        parse_gaps.append("Requested native summary does not retain individual finding evidence")
    findings, gaps, complete = assessment(result, returncode, operation)
    if operation == "platform.serve" and settings.get("private_platform"):
        result = {"private_service": service_report}
        findings, gaps, complete = (
            [],
            [],
            service_report.get("observed_requests") == len(settings["service_requests"]),
        )

    if operation == "platform.worker":
        snapshot = service_report.get("database_snapshot", {})
        unresolved = snapshot.get("job_states", {})
        if any(count for state, count in unresolved.items() if state != "SUCCEEDED"):
            parse_gaps.append("Private native job state contains unresolved/failed jobs")
    gaps.extend(parse_gaps)
    complete = complete and not parse_gaps
    from ablitbench.parity.source_runtime import persist_gaps, persist_result

    def observed_findings() -> Iterator[dict[str, Any]]:
        yield from findings
        if settings.get("private_platform") and (output / "platform-findings.jsonl").is_file():
            with (output / "platform-findings.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if str(row.get("status", "")).upper() not in {"WARN", "FAIL", "BLOCK"}:
                        continue
                    try:
                        detail = json.loads(row.get("detail_json", "{}"))
                    except ValueError:
                        detail = row.get("detail_json")
                    yield {
                        "rule_id": row.get("rule_id"),
                        "severity": detail.get("severity", "info")
                        if isinstance(detail, dict)
                        else "info",
                        "detail": {
                            "scope": "Stored private platform finding",
                            "review_id": row.get("review_id"),
                            "native": detail,
                        },
                    }

    return {
        **persist_result("layerfault", result, observed_findings(), output),
        "status": "completed" if complete else "unsupported",
        "project": "layerfault",
        "operation": operation,
        "source_revision": catalog()["revision"],
        "binary_sha256": checksum,
        "returncode": returncode,
        "redacted_output": bool(secret_values),
        "platform_profile": {
            key: value for key, value in service_report.items() if key != "messages"
        },
        "stdout_artifact": "stdout",
        "stderr_artifact": "stderr",
        "scope": "Isolated assessment workspace; no host stores or runtimes were mutated",
        "runtime_environment": settings.get("runtime_environment"),
        "limits": {
            "lora_adapter_tensor_bytes": 134217728,
            "lora_spectral_dimension": 256,
            "lora_merge_tensors": 512,
            "lora_merge_ops_per_tensor": 100000000,
        }
        if operation.startswith("composition.")
        else {},
        "coverage": {
            "units": "upstream command invocation",
            "inspected": int(complete),
            "total": 1,
            **persist_gaps(gaps, output),
        },
    }
