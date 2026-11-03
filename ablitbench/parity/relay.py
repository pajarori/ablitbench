from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import sys
import types
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT = "api-relay-audit"
SENTINEL = "ABLIB_REDACTED_CREDENTIAL"
STEPS = (
    "infrastructure",
    "models",
    "token_injection",
    "prompt_extraction",
    "instruction_conflict",
    "jailbreak",
    "tool_substitution",
    "error_leakage",
    "stream_integrity",
    "web3_injection",
    "context_length",
    "infra_fingerprint",
    "latency_variance",
    "channel_classifier",
)


def operations() -> dict[str, dict[str, Any]]:
    from ablitbench.parity.sources import source_spec

    metadata = {
        "project": PROJECT,
        "kind": "evaluate",
        "capabilities": ["isolated-execution", "host-broker"],
        "inputs": ["source"],
        "optional_inputs": [],
        "secret_inputs": [],
        "target_defaults": {"model": "model", "base_url": "location"},
        "source_revision": source_spec(PROJECT)["revision"],
        "source_entrypoint": "scripts/audit.py",
        "license": "AGPL-3.0-only",
        "requires_inference": False,
        "requires_tools": False,
        "modes": ["api"],
        "permissions": [],
    }
    result = {name: dict(metadata) for name in ("audit", "audit.native", "connectivity", *STEPS)}
    for name, roles in {
        "standalone.export": ["source"],
        "standalone.check": ["source"],
        "report.extract": ["source", "report"],
        "legacy.export": ["source", "manifest", "website", "evidence"],
    }.items():
        result[name] = dict(
            metadata,
            kind="inspect" if name.endswith("check") else "export",
            inputs=roles,
            modes=["source", "api"],
            capabilities=["isolated-execution"],
            target_defaults={},
        )
    return result


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown relay operation")
    allowed = {
        "model",
        "base_url",
        "steps",
        "profile",
        "aggressive_error_probes",
        "fast_context",
        "latency_probe_count",
        "warmup",
        "request_timeout_seconds",
        "token_policy",
        "max_output_tokens",
        "transparent_log",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown relay setting")
    local = operation in {
        "standalone.export",
        "standalone.check",
        "report.extract",
        "legacy.export",
    }
    if local and settings:
        raise ValueError("Local relay artifact operations do not accept endpoint settings")
    if local:
        return dict(settings)
    if not isinstance(settings.get("model"), str) or not settings["model"].strip():
        raise ValueError("Relay model must be explicitly supplied")
    if settings.get("token_policy") not in {"upstream", "explicit"}:
        raise ValueError("Declare token_policy upstream or explicit")
    if settings.get("token_policy") == "explicit" and (
        type(settings.get("max_output_tokens")) is not int or settings["max_output_tokens"] <= 0
    ):
        raise ValueError("Explicit token policy needs positive max_output_tokens")
    for key in ("warmup", "latency_probe_count", "request_timeout_seconds"):
        if key in settings and (
            type(settings[key]) is not int or settings[key] < (0 if key == "warmup" else 1)
        ):
            raise ValueError(key + " must be a positive integer")
    for key in ("aggressive_error_probes", "fast_context", "transparent_log"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(key + " must be boolean")
    if settings.get("profile", "full") not in {"general", "web3", "full"}:
        raise ValueError("Unknown relay profile")
    if "steps" in settings and (
        not isinstance(settings["steps"], list)
        or not settings["steps"]
        or any(value not in STEPS for value in settings["steps"])
    ):
        raise ValueError("steps must contain known relay checks")
    if operation == "audit.native" and "steps" in settings:
        raise ValueError("Native aggregate audit always runs its complete upstream workflow")
    base = settings.get("base_url", "https://configured-target.invalid/v1")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("base_url must be a credential-free endpoint identity")
    return dict(settings, base_url=base.rstrip("/"))


def attach_transport(
    client: Any, module: Any, broker: Any, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    origin = urlsplit(client.base_url)
    native_log = client._log_transparent
    client._log_transparent = lambda *args, **kwargs: None

    def exchange(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: Any = None,
        raw: bytes | None = None,
        stream: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        parsed = urlsplit(url)
        if parsed.netloc and (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc):
            raise ValueError("Upstream requested a different relay origin")
        lowered = {key.lower(): value for key, value in headers.items()}
        authorization = lowered.get("authorization", lowered.get("x-api-key"))
        auth = (
            "omit"
            if authorization is None
            else (
                "configured"
                if authorization in {SENTINEL, "Bearer " + SENTINEL}
                else "invalid-synthetic"
            )
        )
        request: dict[str, Any] = {
            "kind": "http",
            "method": method,
            "path": parsed.path or "/",
            "auth": auth,
            "auth_style": "bearer" if "authorization" in lowered else "x-api-key",
            "headers": {
                key: value
                for key, value in lowered.items()
                if key not in {"authorization", "x-api-key"}
            },
            "stream": stream,
        }
        if parsed.query:
            request["path"] += "?" + parsed.query
        if payload is not None:
            request["payload"] = payload
        if raw is not None:
            request["body_base64"] = base64.b64encode(raw).decode("ascii")
        timeout = settings.get("request_timeout_seconds", timeout)
        if timeout is not None:
            request["timeout_seconds"] = timeout
        response = broker(request)
        if raw is not None:
            sent = raw
        elif payload is not None:
            sent = json.dumps(payload).encode("utf-8")
        else:
            sent = None
        native_log(
            "broker_http",
            url,
            method,
            sent,
            response.get("body"),
            int(response.get("status", 0)),
            response.get("headers"),
            response.get("elapsed", 0),
            response.get("error"),
        )
        requests.append(
            {
                "method": method,
                "path": request["path"],
                "auth": auth,
                "status": response.get("status"),
                "evidence": response.get("evidence"),
                "max_output_tokens": payload.get("max_tokens")
                if isinstance(payload, dict)
                else None,
                "error": response.get("error"),
                "timeout_seconds": timeout,
            }
        )
        return dict(response)

    def post(self: Any, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        if settings["token_policy"] == "explicit":
            body = dict(body, max_tokens=settings["max_output_tokens"])
            if "thinking" in body:
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max(1, body["max_tokens"] - 1),
                }
        result = exchange("POST", url, headers, body, timeout=self.timeout)
        if result.get("error") or not 200 <= int(result.get("status", 0)) < 300:
            return {"_http_error": result.get("error") or result.get("body", "HTTP failure")}
        try:
            value = json.loads(result.get("body", ""))
            return value if isinstance(value, dict) else {"_http_error": "Non-object JSON response"}
        except (TypeError, ValueError) as error:
            return {"_http_error": str(error)}

    def raw_request(
        self: Any,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        content_type: str = "application/json",
        timeout: int = 30,
    ) -> dict[str, Any]:
        base = (
            self.base_url[:-3]
            if self.base_url.endswith("/v1") and path.startswith("/v1")
            else self.base_url
        )
        return exchange(
            method,
            base + path,
            dict(headers, **{"content-type": content_type}),
            raw=body,
            timeout=timeout,
        )

    def models(self: Any) -> list[Any]:
        base = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        variants = [
            {"authorization": "Bearer " + SENTINEL},
            {"x-api-key": SENTINEL, "anthropic-version": "2023-06-01"},
        ]
        if self._format == "anthropic":
            variants.reverse()
        for headers in variants:
            result = exchange("GET", base + "/models", headers)
            if result.get("status") == 200:
                try:
                    data = json.loads(result.get("body", "")).get("data", [])
                    if data:
                        return list(data)
                except (ValueError, AttributeError):
                    pass
        return []

    def streaming(
        self: Any,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: int,
        signals: Any,
        hasher: Any = None,
    ) -> None:
        if settings["token_policy"] == "explicit":
            body = dict(body, max_tokens=settings["max_output_tokens"])
            if "thinking" in body:
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max(1, body["max_tokens"] - 1),
                }
        result = exchange("POST", url, headers, body, stream=True, timeout=timeout)
        if result.get("error") or result.get("status") != 200:
            signals.transport_error = str(
                result.get("error") or result.get("body") or result.get("status")
            )
            return
        chunks = result.get("chunks")
        if chunks is None:
            chunks = [str(result.get("body", ""))]
        module._parse_sse_stream(iter(chunks), signals, hasher)

    client._post = types.MethodType(post, client)
    client._curl_post = types.MethodType(post, client)
    client.raw_request = types.MethodType(raw_request, client)
    client.get_models = types.MethodType(models, client)
    client._stream_via_httpx = types.MethodType(streaming, client)
    client._stream_via_curl = types.MethodType(streaming, client)
    return requests


def external_module(source: Path, name: str) -> Any:
    location = source / "scripts" / (name + ".py")
    identity = "ablitbench_external_relay_" + name.replace("-", "_")
    specification = importlib.util.spec_from_file_location(identity, location)
    if specification is None or specification.loader is None:
        raise ValueError("Original relay entrypoint is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[identity] = module
    specification.loader.exec_module(module)
    return module


def artifact_operation(
    operation: str, source: Path, inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    if operation.startswith("standalone."):
        module = external_module(source, "build-standalone")
        rendered = module.build_text()
        identical = rendered == (source / "audit.py").read_text(encoding="utf-8")
        if operation.endswith("export"):
            (output / "audit.py").write_text(rendered, encoding="utf-8")
            for name in ("LICENSE", "NOTICE"):
                (output / name).write_bytes((source / name).read_bytes())
        return {
            "status": "completed" if identical else "inconclusive",
            "source_digest": module.source_digest(),
            "committed_artifact_matches": identical,
            "license": "AGPL-3.0-only",
            "artifacts": ["audit.py", "LICENSE", "NOTICE"] if operation.endswith("export") else [],
            "coverage": {"gaps": [] if identical else ["Committed standalone artifact drift"]},
        }
    if operation == "report.extract":
        module = external_module(source, "extract-data")
        result = module.parse_report(inputs["report"])
        return {"status": "completed", "result": result}
    module = external_module(source, "build-legacy-pages")
    module.build(inputs["manifest"], inputs["website"], inputs["evidence"], output)
    return {
        "status": "completed",
        "artifacts": [
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        ],
    }


def native_audit(
    audit: Any, client: Any, report: Any, settings: dict[str, Any], output: Path
) -> dict[str, Any]:
    args = types.SimpleNamespace(
        url=client.base_url,
        key=SENTINEL,
        model=client.model,
        timeout=settings.get("request_timeout_seconds", 120),
        transparent_log=None,
        connectivity=False,
        format="json",
        output=str(output / "upstream-native-report.json"),
        profile=settings.get("profile", "full"),
        warmup=0,
        fast_context=settings.get("fast_context", False),
        latency_probe_count=settings.get("latency_probe_count", 10),
        aggressive_error_probes=settings.get("aggressive_error_probes", False),
        **{
            "skip_" + step: False
            for step in (
                "infra",
                "context",
                "tool_substitution",
                "error_leakage",
                "stream_integrity",
                "web3_injection",
                "infra_fingerprint",
                "latency_variance",
                "channel_classifier",
            )
        },
    )
    original_client, original_reporter = audit.APIClient, audit.Reporter
    audit.APIClient = lambda *args, **kwargs: client
    audit.Reporter = lambda: report
    try:
        audit._run_audit(args, sys.stdout)
    finally:
        audit.APIClient, audit.Reporter = original_client, original_reporter
    return dict(json.loads((output / "upstream-native-report.json").read_text()))


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
    configured = configuration(operation, settings)
    source = verified_source(inputs["source"], PROJECT)
    output.mkdir(parents=True, exist_ok=True)
    if operation in {"standalone.export", "standalone.check", "report.extract", "legacy.export"}:
        return dict(
            artifact_operation(operation, source, inputs, output),
            project=PROJECT,
            operation=operation,
        )
    if broker is None:
        raise ValueError("Relay operations require the configured-target broker")
    sys.path.insert(0, str(source))
    specification = importlib.util.spec_from_file_location(
        "ablitbench_external_relay_audit", source / "scripts" / "audit.py"
    )
    if specification is None or specification.loader is None:
        raise ValueError("Original relay entrypoint is unavailable")
    audit: Any = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(audit)
    client_module = importlib.import_module("api_relay_audit.client")
    client = client_module.APIClient(
        configured["base_url"],
        SENTINEL,
        configured["model"],
        timeout=configured.get("request_timeout_seconds", 120),
        verbose=False,
    )
    logger = None
    if configured.get("transparent_log", True):
        logging_module = importlib.import_module("api_relay_audit.transparent_log")
        logger = logging_module.TransparentLogger(str(output / "upstream-requests.jsonl"))
        native_write = logger.log_entry

        def write_entry(entry: dict[str, Any]) -> None:
            native_write(
                dict(
                    entry,
                    transport="ablitbench-host-broker",
                    hash_scope="credential-redacted broker response",
                )
            )

        logger.log_entry = write_entry
        client.set_transparent_logger(logger)
    requests = attach_transport(client, client_module, broker, configured)
    report = audit.Reporter()
    results: dict[str, Any] = {}
    gaps: list[Any] = []
    observations: dict[str, Any] = {}
    native_context = audit.run_context_scan

    def capture_context(*args: Any, **kwargs: Any) -> Any:
        value = native_context(*args, **kwargs)
        observations["context_length"] = value
        return value

    audit.run_context_scan = capture_context
    native_models = client.get_models

    def capture_models() -> Any:
        value = native_models()
        observations["models"] = value
        return value

    client.get_models = capture_models
    args = types.SimpleNamespace(
        key=SENTINEL, aggressive_error_probes=configured.get("aggressive_error_probes", False)
    )
    selected = (
        list(configured.get("steps", STEPS))
        if operation in {"audit", "audit.native"}
        else [operation]
    )
    if configured.get("profile", "full") == "general" and "web3_injection" in selected:
        selected.remove("web3_injection")

    def infra(name: str) -> Any:
        response = broker(
            {
                "kind": "infrastructure",
                "operation": name,
                "timeout_seconds": configured.get("request_timeout_seconds", 10),
            }
        )
        observations[name] = response
        if response.get("data") is None:
            gaps.append(
                {
                    "step": "infrastructure",
                    "operation": name,
                    "reason": "Unavailable infrastructure observation",
                }
            )
        return response["data"]

    audit._resolve_dns_records = lambda domain: infra("dns")
    audit._fetch_tls_certificate_summary = lambda domain: infra("tls")
    audit._lookup_whois = lambda domain: infra("whois")

    def snapshot(url: str, method: str = "GET", max_body_chars: int = 500) -> dict[str, Any]:
        parsed = urlsplit(url)
        result = broker(
            {"kind": "http", "path": parsed.path or "/", "method": method, "auth": "omit"}
        )
        return dict(result, body_preview=str(result.get("body", ""))[:max_body_chars], url=url)

    audit._fetch_http_snapshot = snapshot
    with (
        (output / "upstream-stdout.txt").open("w", encoding="utf-8") as stream,
        redirect_stdout(stream),
    ):
        if configured.get("warmup", 0):
            audit.run_warmup(client, configured["warmup"])
        if operation == "audit.native":
            results["native_audit"] = native_audit(audit, client, report, configured, output)
            native_coverage = results["native_audit"].get("coverage", {})
            for category in ("crashed_steps", "risk_matrix_inconclusive_steps"):
                gaps.extend(
                    {"step": name, "reason": category} for name in native_coverage.get(category, [])
                )
        else:
            for step in selected:
                try:
                    if step == "connectivity":
                        value = audit.run_connectivity_check(client)
                        value.pop("client", None)
                        value["probes"] = [
                            asdict(probe)
                            if is_dataclass(probe) and not isinstance(probe, type)
                            else probe
                            for probe in value.get("probes", [])
                        ]
                        if not value.get("success"):
                            gaps.append(
                                {"step": step, "reason": "Native connectivity probes failed"}
                            )
                    elif step == "infrastructure":
                        value = audit.test_infrastructure(client.base_url, report)
                    elif step == "error_leakage":
                        value = audit.test_error_leakage(client, args, report)
                    elif step == "context_length":
                        value = audit.test_context_length(
                            client, report, fast_mode=configured.get("fast_context", False)
                        )
                    elif step == "latency_variance":
                        value = audit.test_latency_variance(
                            client, report, probe_count=configured.get("latency_probe_count", 10)
                        )
                    else:
                        value = getattr(audit, "test_" + step)(client, report)
                    results[step] = value
                    if step in observations:
                        results[step] = {"native_return": value, "observations": observations[step]}
                    if step == "models" and not observations.get("models"):
                        gaps.append(
                            {
                                "step": step,
                                "reason": "Native model inventory is unavailable or empty",
                            }
                        )
                    if step == "context_length" and (
                        not observations.get(step)
                        or all(row[4] == "error" for row in observations[step])
                    ):
                        gaps.append({"step": step, "reason": "All native context probes errored"})
                    if value is None and step not in {
                        "infrastructure",
                        "models",
                        "jailbreak",
                        "context_length",
                    }:
                        gaps.append(
                            {"step": step, "reason": "upstream returned no determinate result"}
                        )
                except Exception as error:
                    gaps.append({"step": step, "error": str(error), "type": type(error).__name__})
    if logger is not None:
        logger.close()
    gaps.extend(
        {"reason": message}
        for level, message in report.summary
        if "inconclusive" in message.lower() or "partially skipped" in message.lower()
    )
    gaps.extend(
        request for request in requests if request.get("error") or not request.get("status")
    )
    (output / "upstream-report.md").write_text(
        report.render(
            target_url=client.base_url,
            model=client.model,
            profile=configured.get("profile", "full"),
        ),
        encoding="utf-8",
    )
    from ablitbench.parity.source_runtime import persist_gaps, persist_result

    native_findings = [{"level": level, "message": message} for level, message in report.summary]
    raw = {
        "results": results,
        "findings": native_findings,
        "requests": requests,
        "observations": observations,
    }
    return {
        **persist_result(PROJECT, raw, native_findings, output),
        "project": PROJECT,
        "operation": operation,
        "status": "inconclusive" if gaps else "completed",
        "request_count": len(requests),
        "step_count": len(results),
        "coverage": {
            "requested": selected,
            "observed": list(results),
            **persist_gaps(gaps, output),
        },
        "limits": {
            "token_policy": configured["token_policy"],
            "upstream_sse_buffer_bytes": client_module.MAX_STREAM_BUFFER_BYTES,
            "infrastructure_preview_chars": 500,
            "infrastructure_timeout_seconds": configured.get("request_timeout_seconds", 10),
            "timing_scope": "broker roundtrip includes IPC",
        },
        "artifacts": ["upstream-report.md", "upstream-stdout.txt"]
        + (["upstream-requests.jsonl"] if logger is not None else []),
        "implementation": ("original reviewed AGPL source; credential-free host broker transport"),
    }
