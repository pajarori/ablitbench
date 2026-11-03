from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit, urlunsplit

from ablitbench.formats.json import loads
from ablitbench.io import operation
from ablitbench.limits import available_memory
from ablitbench.models import EnvironmentRequest, Experiment, Generation, GenerationRequest, Target
from ablitbench.privacy import REDACTED, redact_text
from ablitbench.transport import CapabilityError, Transport

if TYPE_CHECKING:
    from ablitbench.core import Context


class HostBroker:
    def __init__(self, context: Context, experiment: Experiment) -> None:
        self.context, self.experiment = context, experiment
        self.additional: dict[str, Transport] = {}
        self.sequence = 0
        self.inputs: dict[str, Path] = {}
        self.scope: str | None = None
        self.state_key: str | None = None
        self.environment_worker: Any = None
        self.observer: Path | None = None
        self.outputs: list[Path] = []
        self.hub: Any = None
        targets = [context.target.target, context.config.judge, experiment.attacker]
        self.secrets = [
            value
            for target in targets
            if target is not None
            for name in [target.key_env, *getattr(target, "credential_env", {}).values()]
            if (value := os.environ.get(name))
        ]
        if context.target.target.location.startswith("hf://") and (
            token := os.environ.get("HF_TOKEN")
        ):
            self.secrets.append(token)

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            transports = [self.context.transport, self.context.judge, *self.additional.values()]
            secrets = [
                *self.secrets,
                *(
                    secret
                    for transport in transports
                    if transport is not None
                    for secret in getattr(transport, "secret_values", [])
                ),
            ]
            return redact_text(value, secrets)
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {self.redact(key): self.redact(item) for key, item in value.items()}
        return value

    def target(self, channel: str) -> Target:
        target = (
            self.context.target.target
            if channel == "target"
            else self.context.config.judge
            if channel == "judge"
            else self.experiment.attacker
            if channel == "attacker"
            else None
        )
        if target is None:
            raise CapabilityError("Requested research broker channel is not configured")
        original = self.context.target.target
        if (
            channel != "target"
            and (
                {target.key_env, *getattr(target, "credential_env", {}).values()}
                & {original.key_env, *getattr(original, "credential_env", {}).values()}
            )
            and urlsplit(target.location).hostname != urlsplit(original.location).hostname
        ):
            raise CapabilityError(
                "Research broker cannot forward target credentials to another host"
            )
        return target

    async def transport(self, channel: str) -> Transport:
        if channel == "target" and self.context.transport is not None:
            return self.context.transport
        if channel == "judge" and self.context.judge is not None:
            return self.context.judge
        if channel not in self.additional:
            target = self.target(channel)
            if urlsplit(target.location).scheme not in {"http", "https"}:
                raise CapabilityError("Requested broker channel is not an API endpoint")
            self.additional[channel] = Transport(
                self.context.config.model_copy(update={"mode": "api", "targets": [target]}),
                target,
                self.context.store,
            )
        return self.additional[channel]

    async def generation(self, request: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "kind",
            "channel",
            "messages",
            "tools",
            "stream",
            "max_output_tokens",
            "seed",
            "generation_settings",
            "expected_model",
            "output_modalities",
            "output_audio_format",
        }
        if set(request) - allowed or not isinstance(request.get("messages"), list):
            raise CapabilityError("Malformed research generation request")
        parsed_request = GenerationRequest.model_validate(request)
        request = parsed_request.model_dump(exclude_unset=True)
        channel = request.get("channel", "target")
        target = self.target(channel)
        if request.get("expected_model") not in {None, target.model}:
            raise CapabilityError("Configured broker model does not match the required classifier")
        changes: dict[str, Any] = {
            key: request[key]
            for key in (
                "max_output_tokens",
                "seed",
                "generation_settings",
                "output_modalities",
                "output_audio_format",
            )
            if key in request
        }
        if channel == "judge" and "output_modalities" not in changes:
            changes.update(output_modalities=["text"], output_audio_format=None)
        if channel == "judge" and "generation_settings" not in changes:
            changes["generation_settings"] = {}
        config = self.context.config.model_copy(update=changes)
        messages, tools, streaming = (
            request["messages"],
            request.get("tools"),
            request.get("stream"),
        )
        generation: Generation | None = None
        if channel == "target" and self.context.transport is None:
            if not self.context.config.inference:
                raise CapabilityError("Source broker generation requires explicit --inference")
            original = self.context.config
            self.context.config = config
            try:
                generation = await self.context.generate(messages, tools, streaming)
            finally:
                self.context.config = original
        else:
            transport = await self.transport(channel)
            original = transport.config
            transport.config = config
            started = time.perf_counter()
            try:
                generation = await transport.generate(messages, tools, streaming)
            finally:
                transport.config = original
                self.context.store.measure_generation(
                    self.context.job,
                    channel,
                    "api",
                    time.perf_counter() - started,
                    generation.usage if generation is not None else None,
                )
        if generation is None:
            raise CapabilityError("Research broker returned no generation")
        return dict(self.redact(generation.model_dump(mode="json")))

    async def environment(self, request: dict[str, Any]) -> dict[str, Any]:
        from ablitbench.checks.artifacts import output_directory
        from ablitbench.workers import Worker

        if not self.context.config.execute_tools:
            raise CapabilityError("Environment execution requires explicit --execute-tools")
        parsed = EnvironmentRequest.model_validate(request).model_dump(mode="json")
        if "fixture" not in self.inputs or self.scope is None or self.state_key is None:
            raise CapabilityError("Environment requires a private fixture and durable task scope")
        runtime = self.context.config.runtime
        if self.environment_worker is None:
            executor = Worker(runtime)
            executor.on_protocol_error = self.context.protocol_error
            executor.validate(runtime.environment_image or runtime.tool_image)
            directory = self.context.store.directory / "observers" / self.scope
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            os.chmod(directory, 0o700)
            public = await operation(
                "environment_project",
                [str(self.inputs["fixture"]), str(directory)],
                memory=runtime.memory_bytes,
            )
            executor.output = output_directory(self.context)
            executor.inputs = {"fixture": Path(public)}
            executor.scope = hashlib.sha256((self.scope + ":environment").encode()).hexdigest()
            executor.on_container_created = lambda name: self.context.store.set(
                str(self.state_key) + ":environment-container", name
            )
            self.environment_worker, self.observer = executor, directory
            self.outputs.append(executor.output)
        executor = self.environment_worker
        await operation(
            "environment_before",
            [str(self.inputs["fixture"]), str(self.observer), parsed],
            memory=runtime.memory_bytes,
        )
        transition = await executor.environment(parsed)
        return dict(
            await operation(
                "environment_after",
                [
                    str(self.inputs["fixture"]),
                    str(self.observer),
                    parsed,
                    transition,
                    str(executor.output),
                ],
                memory=runtime.memory_bytes,
            )
        )

    async def http(self, request: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "kind",
            "channel",
            "method",
            "path",
            "payload",
            "headers",
            "auth",
            "auth_style",
            "body_base64",
            "timeout_seconds",
            "stream",
            "expected_origin",
        }
        if set(request) - allowed:
            raise CapabilityError("Unknown HTTP broker parameters")
        channel = request.get("channel", "target")
        target = self.target(channel)
        if "expected_origin" in request:
            from ablitbench.parity.native_provider import origin

            if origin(request["expected_origin"]) != origin(target.location):
                raise CapabilityError("SDK request origin differs from its configured channel")
        transport = await self.transport(channel)
        method, path = request.get("method", "POST"), request.get("path", "")
        parsed = urlsplit(path)
        if (
            method not in {"GET", "HEAD", "POST", "OPTIONS"}
            or parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or path.startswith("//")
            or "\\" in path
            or any(
                part in {".", ".."}
                or "/" in part
                or "\\" in part
                or any(ord(character) < 32 for character in part)
                for part in map(unquote, parsed.path.split("/"))
            )
        ):
            raise CapabilityError("HTTP broker request must remain on the configured endpoint")
        headers = dict(request.get("headers", {}))
        if any(
            name.casefold() not in {"content-type", "accept", "anthropic-version"}
            for name in headers
        ):
            raise CapabilityError("HTTP broker headers cannot override authentication or routing")
        auth = request.get("auth", "configured")
        if auth not in {"configured", "omit", "invalid-synthetic"}:
            raise CapabilityError("Unknown HTTP broker authentication mode")
        key = (
            transport.key
            if auth == "configured"
            else "ABLIB_INVALID_CREDENTIAL"
            if auth == "invalid-synthetic"
            else None
        )
        auth_style = request.get(
            "auth_style", "x-api-key" if target.protocol == "anthropic-messages" else "bearer"
        )
        if auth_style not in {"bearer", "x-api-key"}:
            raise CapabilityError("Unknown HTTP broker authentication style")
        if auth != "configured" and transport.credentials.get("client_cert"):
            raise CapabilityError(
                "Credential omission requires a separate mTLS identity-free session"
            )
        if key and ("auth_style" in request or auth == "invalid-synthetic"):
            if auth_style == "x-api-key":
                headers["x-api-key"] = key
            else:
                headers["authorization"] = "Bearer " + key
        if "body_base64" in request and "payload" in request:
            raise CapabilityError("HTTP broker body representations are mutually exclusive")
        content = (
            base64.b64decode(request["body_base64"], validate=True)
            if "body_base64" in request
            else None
        )
        if content is None:
            content = (
                json.dumps(request["payload"], ensure_ascii=False, allow_nan=False).encode()
                if "payload" in request
                else b""
            )
            if "payload" in request:
                headers.setdefault("content-type", "application/json")
        if path.startswith("/"):
            endpoint = urlsplit(target.location)
            url = urlunsplit((endpoint.scheme, endpoint.netloc, parsed.path, parsed.query, ""))
        else:
            url = target.location.rstrip("/") + "/" + path
        if auth == "configured":
            from ablitbench.parity.native_provider import authorization, render_secret

            configured_headers = render_secret(target.provider_options.get("headers", {}), key)
            headers = {**configured_headers, **headers}
            if "auth_style" not in request:
                await transport.ensure_iam()
                headers.update(
                    authorization(
                        target.protocol,
                        method,
                        url,
                        content,
                        key,
                        transport.credentials,
                        target.provider_options,
                    )
                )
        marker = "ABLIB_REFLECTED_" + uuid.uuid4().hex
        response = await transport.exchange(
            {
                "method": method,
                "url": url,
                "content": content,
                "headers": headers,
                "evidence_body": request.get("payload", "Explicit binary HTTP body"),
                "wire_redaction_marker": marker,
                **(
                    {"timeout_seconds": request["timeout_seconds"]}
                    if "timeout_seconds" in request
                    else {}
                ),
            },
            "research-http",
        )
        evidence = list(transport.last_evidence[-2:])

        with tempfile.TemporaryDirectory(
            prefix="probe-", dir=self.context.store.directory
        ) as directory:
            destination = Path(directory) / "body.json"
            measured = await operation(
                "decode_probe",
                [str(response["path"]), str(destination), response["content_encoding"]],
                memory=self.context.config.runtime.memory_bytes,
            )
            if measured["materialization_bytes"] > available_memory():
                raise CapabilityError(
                    "Probe body exceeds measured host capacity; full wire evidence retained"
                )
            value = loads(destination.read_bytes())
        body = value["body"]
        literal_sentinel = REDACTED in body and REDACTED not in self.secrets
        prepared = (
            body.replace(REDACTED, "ABLIB_PUBLIC_LITERAL_SENTINEL") if literal_sentinel else body
        )
        redacted = self.redact(prepared).replace(marker, REDACTED)
        observed_headers = self.redact(response["headers"])
        entity_headers = {
            key: item
            for key, item in observed_headers.items()
            if key.casefold() not in {"content-encoding", "content-length", "transfer-encoding"}
        }
        entity_headers["content-length"] = str(len(redacted.encode("utf-8")))
        return {
            "status": response["status"],
            "headers": entity_headers,
            "observed_headers": observed_headers,
            "body": redacted,
            "body_encoding": "decoded UTF-8 HTTP entity",
            "elapsed": response["elapsed"],
            "first_byte_seconds": response["first_byte_seconds"],
            "credential_reflected": bool(response["redactions"])
            or response["header_credential_reflected"],
            "literal_sentinel_escaped": literal_sentinel,
            "credential_sentinel": REDACTED,
            "evidence": evidence,
        }

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        self.context.store.evidence(
            {"research_broker_request": self.redact(request), "sequence": self.sequence}
        )
        try:
            kind = request.get("kind", "generation")
            if kind in {"generation", "judge"}:
                value = await self.generation(
                    request | ({"channel": "judge"} if kind == "judge" else {})
                )
            elif kind == "hub":
                from ablitbench.parity.hub import HubBroker

                if self.hub is None:
                    self.hub = HubBroker(
                        self.context.target,
                        self.context.store,
                        self.experiment.permissions,
                        offline=self.context.config.offline,
                        allow_public_crawl=(
                            self.experiment.project == "layerfault"
                            and (
                                self.experiment.operation in {"hub.crawl", "platform.crawl"}
                                or self.experiment.operation == "platform.worker"
                                and self.experiment.settings.get("private_platform") is True
                                and self.experiment.settings.get("public_hub_jobs") is True
                            )
                        ),
                    )
                value = await self.hub.observe(request)
            elif kind == "environment":
                value = await self.environment(request)
            elif kind == "http":
                value = await self.http(request)
            elif kind == "infrastructure":
                from ablitbench.parity.infrastructure import observe

                if set(request) - {"kind", "operation", "timeout_seconds"}:
                    raise CapabilityError("Infrastructure target cannot override the endpoint")
                endpoint = urlsplit(self.target("target").location)
                if endpoint.hostname is None or endpoint.scheme not in {"http", "https"}:
                    raise CapabilityError("Infrastructure observation requires an API endpoint")
                timeout = request.get("timeout_seconds", self.context.config.request_timeout)
                if self.context.config.request_timeout is not None and timeout is not None:
                    timeout = min(timeout, self.context.config.request_timeout)
                data = await observe(
                    endpoint.hostname,
                    endpoint.port or (443 if endpoint.scheme == "https" else 80),
                    request.get("operation", ""),
                    timeout,
                )
                reference = self.context.store.evidence(
                    {"infrastructure": request.get("operation"), "data": data}
                )
                value = {"data": data, "operation": request.get("operation"), "evidence": reference}
            else:
                raise CapabilityError("Unsupported research broker operation: " + str(kind))
            reference = self.context.store.evidence(
                {"research_broker_response": value, "sequence": self.sequence}
            )
            if kind in {"generation", "judge"}:
                return {**value, "evidence": [*value.get("evidence", []), reference]}
            return value | {"broker_evidence": reference}
        except Exception as error:
            value = self.redact({"error": {"type": type(error).__name__, "detail": str(error)}})
            self.context.store.evidence({"research_broker_error": value, "sequence": self.sequence})
            return dict(value)

    async def close(self) -> None:
        if self.hub is not None:
            await self.hub.close()
        if self.environment_worker is not None:
            await self.environment_worker.close()
        for transport in self.additional.values():
            await transport.close()
