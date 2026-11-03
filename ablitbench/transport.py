from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx2 as httpx

from ablitbench.formats.json import equal, loads
from ablitbench.models import Generation, ProtocolObservation, ScanConfig, Target, ToolCall
from ablitbench.response_spool import Capture, normalize
from ablitbench.store import Store


class ProtocolError(ValueError):
    pass


class CapabilityError(ValueError):
    pass


async def sse(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    event = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line.startswith("event:"):
            event = line[6:].strip()
    if data:
        raise ProtocolError("Disconnected before SSE event delimiter")


def encode_messages(protocol: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from ablitbench.parity.multimodal import encode_content

    if protocol == "openai-chat":
        return [
            {
                **message,
                "content": encode_content(protocol, message.get("content"), message["role"]),
            }
            for message in messages
        ]
    encoded: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if protocol == "openai-responses":
            if role == "tool":
                encoded.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": encode_content(protocol, message["content"], role),
                    }
                )
            else:
                if message.get("content"):
                    encoded.append(
                        {
                            "role": role,
                            "content": encode_content(protocol, message["content"], role),
                        }
                    )
                for tool in message.get("tool_calls", []):
                    encoded.append(
                        {"type": "function_call", "call_id": tool["id"], **tool["function"]}
                    )
        elif role != "system":
            if role == "tool":
                value = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": encode_content("anthropic", message["content"], role),
                        }
                    ],
                }
                if (
                    encoded
                    and encoded[-1]["role"] == "user"
                    and isinstance(encoded[-1]["content"], list)
                ):
                    encoded[-1]["content"].extend(value["content"])
                else:
                    encoded.append(value)
            else:
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    content = encode_content("anthropic", message["content"], role)
                    blocks.extend(
                        content
                        if isinstance(content, list)
                        else [{"type": "text", "text": content}]
                    )
                for tool in message.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool["id"],
                            "name": tool["function"]["name"],
                            "input": loads(tool["function"]["arguments"]),
                        }
                    )
                encoded.append({"role": role, "content": blocks})
    return encoded


class Transport:
    def __init__(
        self,
        config: ScanConfig,
        target: Target,
        store: Store,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        from ablitbench.parity import native_provider

        self.config, self.target, self.store = config, target, store
        self.protocol: str = target.protocol
        self.credentials = {}
        for role, name in target.credential_env.items():
            origin = (
                target.provider_options.get("iam_url")
                if role.startswith("iam_")
                else target.location
            )
            if not isinstance(origin, str):
                raise ValueError("Credential role requires an explicitly configured endpoint")
            self.credentials.update(
                native_provider.credentials({role: name}, origin, target.credential_origins)
            )
        self.key = (
            target._api_key.get_secret_value()
            if target._api_key is not None
            else self.credentials.get("key", os.environ.get(target.key_env))
        )
        self.secret_values = list(self.credentials.values()) + ([self.key] if self.key else [])
        if self.credentials.get("username") and self.key:
            self.secret_values.append(
                base64.b64encode((self.credentials["username"] + ":" + self.key).encode()).decode()
            )
        verify = (
            native_provider.tls_context(target.provider_options, self.credentials)
            if (
                target.provider_options.get("verify_ssl") is False
                or any(
                    self.credentials.get(role) for role in ("ca_file", "client_cert", "client_key")
                )
            )
            else True
        )
        self.websocket: Any = None
        self.iam_expiry = 0.0
        timeouts = [
            value
            for value in (config.request_timeout, target.provider_options.get("request_timeout"))
            if value is not None
        ]
        self.request_timeout = min(timeouts) if timeouts else None
        connect = target.provider_options.get("connect_timeout", self.request_timeout)
        if self.request_timeout is not None and connect is not None:
            connect = min(connect, self.request_timeout)
        timeout = httpx.Timeout(self.request_timeout, connect=connect)
        self.client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            verify=verify,
            proxy=target.provider_options.get("proxy"),
        )
        self.iam_client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self.last_evidence: list[str] = []
        self.not_before = 0.0

    async def close(self) -> None:
        await self.client.aclose()
        await self.iam_client.aclose()
        if self.websocket is not None:
            await self.websocket.close()

    def encode(
        self,
        protocol: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body: dict[str, Any] = {"model": self.target.model, "stream": stream}
        headers = {"content-type": "application/json"}
        key = self.key
        if protocol == "anthropic-messages":
            if self.config.max_output_tokens is None:
                raise CapabilityError(
                    "Anthropic requires max_output_tokens from user configuration"
                )
            suffix = "messages"
            from ablitbench.generation import text_content

            body["system"] = "\n".join(
                text_content(m["content"]) for m in messages if m["role"] == "system"
            )
            body["messages"] = encode_messages(protocol, messages)
            body["max_tokens"] = self.config.max_output_tokens
            headers["anthropic-version"] = "2023-06-01"
            if key:
                headers["x-api-key"] = key
            if tools:
                body["tools"] = [
                    {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"],
                    }
                    for t in tools
                ]
        else:
            if key:
                headers["authorization"] = "Bearer " + key
            if protocol == "openai-responses":
                suffix = "responses"
                body["input"] = encode_messages(protocol, messages)
                if tools:
                    body["tools"] = [{"type": "function", **t["function"]} for t in tools]
            else:
                suffix = "chat/completions"
                body["messages"] = encode_messages(protocol, messages)
                if tools:
                    body["tools"] = tools
            if self.config.max_output_tokens is not None:
                body[
                    "max_output_tokens"
                    if protocol == "openai-responses"
                    else "max_completion_tokens"
                ] = self.config.max_output_tokens
            if self.config.seed is not None:
                if protocol != "openai-chat":
                    raise CapabilityError("Configured seed is only supported by the chat adapter")
                body["seed"] = self.config.seed
        base = self.target.location.rstrip("/")
        if not base.endswith("/" + suffix):
            base += "/" + suffix
        from ablitbench.generation import generation_options

        options = generation_options(protocol, self.config.generation_settings)
        modalities = self.config.output_modalities
        if len(set(modalities)) != len(modalities):
            raise CapabilityError("Output modalities must be distinct")
        if set(modalities) - {"text", "audio"}:
            raise CapabilityError("Selected protocol has no native image/video output adapter")
        if "audio" in modalities:
            if protocol != "openai-chat":
                raise CapabilityError("Selected protocol has no native output audio adapter")
            audio = options.get("audio", {})
            if not isinstance(audio, dict) or set(audio) - {"format", "voice"}:
                raise CapabilityError("Native output audio requires format and voice configuration")
            if not isinstance(audio.get("voice"), str) or not audio["voice"]:
                raise CapabilityError("Output audio requires an explicit provider voice")
            audio_format = self.config.output_audio_format or audio.get("format")
            if audio_format not in {"wav", "mp3", "flac", "opus", "pcm16"}:
                raise CapabilityError("Output audio requires an explicit supported format")
            if audio.get("format", audio_format) != audio_format:
                raise CapabilityError("Output audio format configuration conflicts")
            options["audio"] = {**audio, "format": audio_format}
            body["modalities"] = modalities
        elif "audio" in options or self.config.output_audio_format is not None:
            raise CapabilityError("Audio configuration requires an explicit audio output modality")
        body.update(options)
        return base, body, headers

    def rate_limit(self, status: int, retry: str | None) -> None:
        if retry is not None and status in {429, 503}:
            try:
                try:
                    seconds = float(retry)
                except ValueError:
                    date = parsedate_to_datetime(retry)
                    if date.tzinfo is None:
                        raise ValueError("Retry-After date lacks timezone") from None
                    seconds = (date - datetime.now(UTC)).total_seconds()
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("Retry-After must be finite and nonnegative")
            except (ValueError, TypeError, OverflowError) as error:
                raise ProtocolError("Invalid Retry-After response signal") from error
            self.not_before = time.monotonic() + seconds

    def redact(self, value: Any) -> Any:
        from ablitbench.privacy import redact_text

        if isinstance(value, str):
            return redact_text(value, self.secret_values)
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value

    async def payload(self, path: Path, encoding: str) -> Any:
        from ablitbench.io import operation
        from ablitbench.limits import available_memory

        with tempfile.TemporaryDirectory(prefix="ablib-provider-json-") as temporary:
            destination = (await asyncio.to_thread(Path(temporary).resolve)) / "decoded.json"
            measured = await operation(
                "decode_probe",
                [str(path), str(destination), encoding],
                memory=self.config.runtime.memory_bytes,
            )
            if measured["materialization_bytes"] > available_memory():
                raise CapabilityError("Provider envelope exceeds measured host memory capacity")
            with destination.open(encoding="utf-8") as stream:
                decoded = json.load(stream)
            return loads(decoded["body"])

    async def exchange(
        self, request: dict[str, Any], protocol: str, credential_response: bool = False
    ) -> dict[str, Any]:
        from ablitbench.io import operation

        started = time.perf_counter()
        first_byte = None
        if self.not_before > time.monotonic():
            await asyncio.sleep(self.not_before - time.monotonic())
        operation_id = uuid.uuid4().hex
        self.last_evidence.append(
            self.store.evidence(
                self.redact(
                    {
                        "url": request["url"],
                        "protocol": protocol,
                        "operation_id": operation_id,
                        "method": request.get("method", "POST"),
                        "request": request.get(
                            "evidence_body", "Credential placeholders rendered by host"
                        ),
                    }
                )
            )
        )
        metadata: dict[str, Any] = {"representation": "wire", "operation_id": operation_id}
        status, headers, content_encoding, content_type = 0, {}, "", ""
        complete = False
        redactions = 0
        with tempfile.TemporaryDirectory(prefix="ablib-provider-wire-") as temporary:
            private = await asyncio.to_thread(Path(temporary).resolve)
            capture = Capture(private)
            try:
                client = self.iam_client if credential_response else self.client
                timeout_options: dict[str, Any] = {}
                timeout = request.get("timeout_seconds")
                if timeout is not None:
                    if (
                        not isinstance(timeout, int | float)
                        or isinstance(timeout, bool)
                        or not math.isfinite(timeout)
                        or timeout <= 0
                    ):
                        raise CapabilityError(
                            "Explicit request timeout must be finite and positive"
                        )
                    if self.request_timeout is not None:
                        timeout = min(timeout, self.request_timeout)
                    timeout_options["timeout"] = timeout
                async with client.stream(
                    request.get("method", "POST"),
                    request["url"],
                    content=request.get("content", b""),
                    headers=request.get("headers", {}),
                    **timeout_options,
                ) as response:
                    status, headers = response.status_code, dict(response.headers)
                    content_type = response.headers.get("content-type", "")
                    content_encoding = response.headers.get("content-encoding", "")
                    metadata.update(
                        status=status,
                        content_type=content_type,
                        content_encoding=content_encoding,
                        retry_after=response.headers.get("retry-after"),
                    )
                    if response.is_stream_consumed:
                        metadata["representation"] = "predecoded-transport"
                        content_encoding = ""
                        capture.write(response.content)
                        first_byte = time.perf_counter() - started
                    else:
                        async for chunk in response.aiter_raw():
                            if first_byte is None:
                                first_byte = time.perf_counter() - started
                            capture.write(chunk)
                    complete = True
                capture.stream.flush()
                if credential_response and status < 300:
                    from ablitbench.parity.native_provider import iam_token

                    payload = await self.payload(capture.path, content_encoding)
                    token, expires = iam_token(payload)
                    self.credentials["access_token"] = token
                    self.secret_values.append(token)
                    for name in ("refresh_token", "id_token"):
                        if isinstance(payload.get(name), str):
                            self.secret_values.append(payload[name])
                    self.iam_expiry = (
                        time.monotonic() + expires if expires is not None else math.inf
                    )
            except (httpx.HTTPError, OSError) as error:
                if isinstance(error, OSError):
                    capture.storage_error = type(error).__name__
                raise CapabilityError(
                    "HTTP response capture incomplete; retained evidence requires coverage gap"
                ) from error
            finally:
                failure = sys.exc_info()[0] is not None
                metadata.update(
                    original_wire_sha256=capture.checksum.hexdigest(),
                    original_wire_bytes=capture.size,
                    storage_error=capture.storage_error,
                )
                publication = Capture(self.store.directory / "traffic")
                source = capture.path
                try:
                    capture.stream.flush()
                    if self.secret_values and not complete and failure:
                        source = private / "incomplete-credential-body-omitted"
                        source.touch()
                        metadata["representation"] = "incomplete-credential-body-withheld"
                    elif self.secret_values:
                        destination = private / "redacted"
                        try:
                            redaction = await operation(
                                "redact_response_wire",
                                [
                                    str(capture.path),
                                    str(destination),
                                    content_encoding,
                                    self.secret_values,
                                    request.get(
                                        "wire_redaction_marker", "ABLIB_REDACTED_CREDENTIAL"
                                    ),
                                ],
                                memory=self.config.runtime.memory_bytes,
                            )
                            redactions = int(redaction["redactions"])
                            metadata["credential_redaction"] = redaction
                            if redactions:
                                source = destination
                                metadata["representation"] = "decoded-credential-redacted"
                                metadata["content_encoding"] = ""
                                content_encoding = ""
                        except BaseException:
                            source = private / "omitted"
                            source.touch()
                            metadata["representation"] = (
                                "credential-redaction-incomplete-body-omitted"
                            )
                            complete = False
                            if not failure:
                                raise CapabilityError(
                                    "Credential redaction incomplete; response body withheld"
                                ) from None
                    if credential_response and not redactions:
                        source = private / "credential-metadata-only"
                        source.write_bytes(b"{}")
                        metadata["representation"] = "credential-response-body-withheld"
                        content_encoding = ""
                    publication_failure = False
                    try:
                        with source.open("rb") as stream:
                            while chunk := stream.read(1024 * 1024):
                                publication.write(chunk)
                    except OSError as error:
                        publication.storage_error = type(error).__name__
                        publication_failure = True
                        complete = False
                        metadata["storage_error"] = type(error).__name__
                    self.last_evidence.extend(
                        publication.persist(self.store, complete, self.redact(metadata))
                    )
                    if publication_failure and not failure:
                        raise CapabilityError(
                            "HTTP response publication incomplete; prefix retained"
                        )
                    if capture.storage_error and not failure:
                        raise CapabilityError("HTTP response storage incomplete")
                finally:
                    capture.close()
                    publication.close()
        self.rate_limit(status, headers.get("retry-after"))
        return {
            "status": status,
            "headers": headers,
            "path": publication.path,
            "content_type": content_type,
            "content_encoding": content_encoding,
            "redactions": redactions,
            "operation_id": operation_id,
            "first_byte_seconds": first_byte,
            "elapsed": time.perf_counter() - started,
            "header_credential_reflected": self.redact(headers) != headers,
        }

    async def ensure_iam(self) -> None:
        from ablitbench.parity.native_provider import iam_request

        if "iam_api_key" not in self.credentials or self.iam_expiry > time.monotonic():
            return
        url = self.target.provider_options.get("iam_url")
        if not url:
            raise CapabilityError("IAM API key requires an explicit token endpoint")
        request = iam_request(
            url, self.credentials["iam_api_key"], self.target.credential_origins["iam_api_key"]
        )
        result = await self.exchange(request, self.protocol, credential_response=True)
        if result["status"] >= 300:
            raise ProtocolError("IAM authentication failed; request not automatically retried")

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool | None = None,
    ) -> Generation:
        from ablitbench.parity import native_provider, providers

        protocols = (
            [self.protocol]
            if self.protocol != "auto"
            else ["openai-chat", "openai-responses", "anthropic-messages"]
        )
        self.last_evidence = []
        owned_cancel = None
        state_key = "provider-request." + uuid.uuid4().hex
        try:
            for protocol in protocols:
                native = protocol in providers.protocols()
                supported_stream = protocol in {
                    "openai-chat",
                    "openai-responses",
                    "anthropic-messages",
                    "openai-completions",
                    "ollama-chat",
                    "ollama-generate",
                    "cohere-chat",
                    "huggingface-inference",
                    "huggingface-endpoint",
                    "websocket",
                    "bedrock-converse",
                }
                preference = self.config.stream if stream is None else stream
                streaming = supported_stream if preference is None else preference
                options = self.target.provider_options
                if native:
                    if self.config.output_modalities != ["text"] or self.config.output_audio_format:
                        raise CapabilityError(
                            "Native provider has no requested output media adapter"
                        )
                    spec = providers.encode(
                        protocol,
                        self.target.model,
                        messages,
                        self.config.max_output_tokens,
                        self.config.generation_settings,
                        options,
                        tools,
                        self.config.seed,
                        streaming,
                    )
                else:
                    url, body, headers = self.encode(protocol, messages, tools, streaming)
                    for name in list(headers):
                        if name.casefold() in {"authorization", "x-api-key", "api-key"}:
                            del headers[name]
                    headers.update(options.get("headers", {}))
                    if options.get("deployment_id"):
                        from urllib.parse import quote

                        base = self.target.location.rstrip("/")
                        url = providers.endpoint(
                            base,
                            "openai/deployments/"
                            + quote(options["deployment_id"], safe="")
                            + "/chat/completions",
                        )
                    for name in options.get("suppressed_parameters", []):
                        if name in {"max_tokens", "max_completion_tokens", "max_output_tokens"}:
                            raise CapabilityError("Explicit output budget cannot be suppressed")
                        body.pop(name, None)
                    token_field = options.get("max_tokens_field")
                    if token_field and self.config.max_output_tokens is not None:
                        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                            body.pop(name, None)
                        body[token_field] = self.config.max_output_tokens
                    extra = options.get("extra_body", {})
                    if set(extra) & set(body):
                        raise CapabilityError(
                            "Provider extra_body cannot override encoded generation fields"
                        )
                    body.update(extra)
                    spec = {
                        "method": "POST",
                        "url": url,
                        "json": body,
                        "headers": headers,
                        "params": {"api-version": options["api_version"]}
                        if options.get("api_version")
                        else {},
                        "stream_format": "sse" if streaming else "json",
                    }
                if protocol == "websocket":
                    if self.websocket is None:
                        self.websocket = native_provider.WebSocketSession(
                            self.target.location, options, self.key, self.credentials
                        )

                    def frame(value: dict[str, Any], selected: str = protocol) -> None:
                        self.last_evidence.append(
                            self.store.evidence(self.redact({"protocol": selected, "frame": value}))
                        )

                    generation: Generation = await self.websocket.generate(spec, frame)
                    if self.redact(generation.model_dump(mode="json")) != generation.model_dump(
                        mode="json"
                    ):
                        generation = Generation(
                            finish="incomplete",
                            observations=[
                                ProtocolObservation(
                                    code="operator-credential-disclosure",
                                    severity="high",
                                    detail="Operator credential disclosure; generation withheld",
                                )
                            ],
                        )
                    generation.evidence = list(self.last_evidence)
                    self.protocol = protocol
                    return generation
                if protocol == "watsonx-generation":
                    await self.ensure_iam()
                request = native_provider.prepare_request(
                    self.target.location, spec, protocol, self.key, self.credentials, options
                )
                request["evidence_body"] = spec.get("json", spec.get("content", ""))
                while True:
                    response = await self.exchange(request, protocol)
                    status = response["status"]
                    if self.protocol == "auto" and status in {404, 405}:
                        break
                    if status >= 300:
                        raise ProtocolError(
                            f"HTTP {status}; request not retried; consult response evidence"
                        )
                    if response["redactions"] or response["header_credential_reflected"]:
                        generation = Generation(
                            finish="incomplete",
                            observations=[
                                ProtocolObservation(
                                    code="operator-credential-disclosure",
                                    severity="high",
                                    detail="Operator credential disclosure; generation withheld",
                                )
                            ],
                        )
                        generation.evidence = list(self.last_evidence)
                        self.protocol = protocol
                        return generation
                    generation, measured = await normalize(
                        response["path"],
                        protocol,
                        streaming,
                        response["content_type"],
                        response["content_encoding"],
                        self.store.directory / "traffic",
                        self.config.runtime.memory_bytes,
                        self.config.output_audio_format
                        or spec.get("json", {}).get("audio", {}).get("format"),
                        {**options, "_http_status": status},
                    )
                    self.last_evidence.insert(-1, self.store.evidence({"normalization": measured}))
                    envelope = measured.get("provider_envelope")
                    next_request = (
                        native_provider.followup(
                            protocol,
                            envelope,
                            status,
                            response["headers"],
                            options,
                            self.target.location,
                        )
                        if native
                        else None
                    )
                    if envelope:
                        owned_cancel = native_provider.cancellation(
                            protocol, envelope, self.target.location
                        )
                        self.store.set(
                            state_key,
                            json.dumps(
                                self.redact(
                                    {
                                        "protocol": protocol,
                                        "target": self.target.location,
                                        "envelope": envelope,
                                        "cancel": owned_cancel,
                                        "status": "pending" if next_request else "completed",
                                    }
                                )
                            ),
                        )
                    if next_request is None:
                        owned_cancel = None
                        for detail in spec.get("coverage", []):
                            observe(generation, "native-request-coverage", detail)
                        generation.evidence = list(self.last_evidence)
                        self.protocol = protocol
                        return generation
                    await asyncio.sleep(options["poll_interval"])
                    request = native_provider.prepare_request(
                        self.target.location,
                        next_request,
                        protocol,
                        self.key,
                        self.credentials,
                        options,
                    )
            raise CapabilityError("No compatible protocol on the supplied endpoint")
        except asyncio.CancelledError:
            if owned_cancel is not None:
                self.store.set(state_key + ".cancellation", "pending")
                if self.request_timeout is not None:
                    try:
                        request = native_provider.prepare_request(
                            self.target.location,
                            owned_cancel,
                            self.protocol,
                            self.key,
                            self.credentials,
                            self.target.provider_options,
                        )
                        async with asyncio.timeout(self.request_timeout):
                            result = await self.exchange(request, self.protocol)
                        self.store.set(
                            state_key + ".cancellation",
                            "requested" if result["status"] < 300 else "inconclusive",
                        )
                    except Exception:
                        self.store.set(state_key + ".cancellation", "inconclusive")
            raise


def observe(generation: Generation, code: str, detail: str, severity: str = "medium") -> None:
    observation = ProtocolObservation.model_validate(
        {"code": code, "detail": detail, "severity": severity}
    )
    if observation not in generation.observations:
        generation.observations.append(observation)


def account(generation: Generation, usage: dict[str, Any], cumulative: bool = False) -> None:
    if not isinstance(usage, dict):
        raise ProtocolError("Usage metadata must be an object")
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        if key not in usage:
            continue
        count = usage[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            observe(generation, "invalid-token-accounting", f"{key} is not a nonnegative integer")
            continue
        previous = generation.usage.get(key)
        if cumulative and isinstance(previous, int) and count < previous:
            observe(generation, "decreasing-token-accounting", f"{key} decreased within one stream")
    counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    valid_counts = [
        count
        for count in counts
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0
    ]
    if len(valid_counts) == 3:
        prompt, completion, total = valid_counts
        if prompt + completion != total:
            observe(
                generation,
                "inconsistent-token-accounting",
                "total_tokens differs from prompt_tokens + completion_tokens",
            )
    generation.usage.update(usage)


def identify(generation: Generation, model: str | None) -> None:
    if model is None:
        return
    if generation.model and generation.model != model:
        observe(
            generation,
            "stream-model-changed",
            "Reported model changed within one stream; routing or fallback may explain it",
            "info",
        )
    generation.model = model


def plaintext(generation: Generation, channel: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        setattr(generation, channel, getattr(generation, channel) + value)
        return
    if isinstance(value, list):
        for item in value:
            plaintext(generation, channel, item)
        return
    if not isinstance(value, dict):
        raise ProtocolError("Unsupported plaintext " + channel + " shape")
    if value.get("type") == "refusal" and channel != "refusal":
        plaintext(generation, "refusal", value)
        return
    opaque = value.get("type") == "redacted_thinking" or any(
        value.get(key) is not None for key in ("encrypted_content", "encrypted_reasoning")
    )
    if opaque:
        observe(
            generation,
            "opaque-reasoning",
            "Encrypted reasoning plaintext is unavailable; "
            "its content and cryptography were not assessed",
            "info",
        )
    fields = (
        ("refusal", "text", "content", "value")
        if channel == "refusal"
        else ("text", "thinking", "reasoning_content", "reasoning", "summary", "content", "value")
    )
    found = False
    for key in fields:
        if key in value:
            found = True
            plaintext(generation, channel, value[key])
    if (
        not found
        and not opaque
        and not (value.get("type") == "reasoning" and value.keys() <= {"type", "id", "status"})
    ):
        raise ProtocolError("Unsupported plaintext " + channel + " object")


def media_content(
    generation: Generation, value: dict[str, Any], audio_format: str | None = None
) -> None:
    from ablitbench.parity.media_outputs import output_block

    try:
        media = output_block(value, audio_format)
    except ValueError as error:
        raise ProtocolError(str(error)) from error
    generation.media.append(media)
    if media.get("url") is not None:
        observe(
            generation,
            "output-media-unmaterialized",
            "Provider supplied an external media locator; bytes were not fetched or assessed",
        )
    if media.get("transcript"):
        observe(
            generation,
            "provider-audio-transcript",
            "Audio transcript is retained as media metadata, "
            "not independently validated audio content",
            "info",
        )


def message_media(
    generation: Generation, message: dict[str, Any], audio_format: str | None = None
) -> None:
    if message.get("audio") is not None:
        media_content(generation, {"type": "audio", "audio": message["audio"]}, audio_format)
    if message.get("images") is not None:
        if not isinstance(message["images"], list):
            raise ProtocolError("Output images must be a list")
        for image in message["images"]:
            block = (
                image
                if isinstance(image, dict) and image.get("type")
                else {"type": "image_url", "image_url": image}
            )
            media_content(generation, block, audio_format)


def image_call(generation: Generation, item: dict[str, Any]) -> None:
    from ablitbench.parity.media_outputs import image_media

    if not item.get("result"):
        if item.get("status") == "completed":
            raise ProtocolError("Completed image generation call has no image data")
        observe(
            generation,
            "image-generation-incomplete",
            "Image generation did not return final image bytes",
        )
        return
    try:
        media = image_media(item)
    except ValueError as error:
        raise ProtocolError(str(error)) from error
    media["complete"] = item.get("status", "completed") == "completed"
    generation.media.append(media)


def final_media(generation: Generation, media: list[dict[str, Any]]) -> None:
    old = [item.get("sha256") or item.get("url") for item in generation.media]
    new = [item.get("sha256") or item.get("url") for item in media]
    if old and old != new:
        observe(
            generation,
            "stream-final-media-mismatch",
            "Final output media differs from accumulated media; "
            "complete variants remain in wire evidence",
        )
    generation.media = media


def content(generation: Generation, value: Any, audio_format: str | None = None) -> None:
    if value is None:
        return
    if isinstance(value, str):
        generation.text += value
    elif isinstance(value, list):
        for item in value:
            content(generation, item, audio_format)
    elif isinstance(value, dict):
        kind = value.get("type")
        if kind in {"refusal"}:
            plaintext(generation, "refusal", value)
        elif kind in {
            "thinking",
            "reasoning",
            "reasoning_text",
            "summary_text",
            "redacted_thinking",
        }:
            plaintext(generation, "reasoning", value)
        elif kind in {None, "text", "output_text"} and "text" in value:
            if not isinstance(value["text"], str):
                raise ProtocolError("Final-answer text must be a string")
            generation.text += value["text"]
        elif kind in {
            "image",
            "image_url",
            "output_image",
            "audio",
            "output_audio",
            "input_audio",
            "video",
            "output_video",
            "document",
        }:
            media_content(generation, value, audio_format)
        else:
            raise ProtocolError("Unsupported response content block")
    else:
        raise ProtocolError("Unsupported response content shape")


def auxiliary(generation: Generation, value: dict[str, Any]) -> None:
    for key in ("reasoning_content", "reasoning"):
        if key in value:
            plaintext(generation, "reasoning", value[key])
    if "refusal" in value:
        plaintext(generation, "refusal", value["refusal"])
    if value.get("encrypted_content") is not None:
        plaintext(generation, "reasoning", {"encrypted_content": value["encrypted_content"]})


def final_channel(generation: Generation, channel: str, value: str) -> None:
    previous = getattr(generation, channel)
    if previous and previous != value:
        generation.raw_text = (generation.raw_text + "\n" if generation.raw_text else "") + previous
        observe(
            generation,
            "stream-final-" + channel + "-mismatch",
            "Final " + channel + " differs from streamed deltas; prior plaintext is retained",
        )
    setattr(generation, channel, value)


def completed_piece(generation: Generation, channel: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ProtocolError("Completed plaintext part must be a string")
    if value and not getattr(generation, channel).endswith(value):
        plaintext(generation, channel, value)


def media_completeness(generation: Generation) -> None:
    incomplete = not generation.complete
    for block in generation.media:
        block["complete"] = False if incomplete else block.get("complete", True)


def legacy_function(generation: Generation, value: Any, identity: str | None = None) -> None:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("name"), str)
        or not value["name"]
        or not isinstance(value.get("arguments"), str)
    ):
        raise ProtocolError("Malformed legacy function action request")
    generation.tools.append(
        ToolCall(
            id=identity or "legacy:" + uuid.uuid4().hex,
            name=value["name"],
            arguments=value["arguments"],
        )
    )
    generation.finish = "incomplete"
    observe(
        generation,
        "legacy-tool-correlation-unavailable",
        "Legacy function_call action retained; native correlation identity unavailable",
    )


def decode_json(
    protocol: str, value: dict[str, Any], output_audio_format: str | None = None
) -> Generation:
    if not isinstance(value, dict):
        raise ProtocolError("Provider response must be an object")
    if "error" in value:
        raise ProtocolError("Provider returned an error object")
    generation = Generation(model=value.get("model"))
    account(generation, value.get("usage") or {})
    if protocol == "openai-chat":
        choice = value["choices"][0]
        message = choice["message"]
        content(generation, message.get("content"), output_audio_format)
        auxiliary(generation, message)
        message_media(generation, message, output_audio_format)
        generation.finish = choice.get("finish_reason")
        generation.tools = [
            ToolCall(id=t["id"], name=t["function"]["name"], arguments=t["function"]["arguments"])
            for t in message.get("tool_calls", [])
        ]
        if message.get("function_call") is not None:
            legacy_function(generation, message["function_call"])
        elif generation.finish == "function_call":
            generation.finish = "incomplete"
            observe(
                generation, "missing-legacy-action", "Legacy finish reason has no action payload"
            )
    elif protocol == "openai-responses":
        generation.finish = value.get("status")
        for item in value.get("output", []):
            if item["type"] == "function_call":
                generation.tools.append(
                    ToolCall(id=item["call_id"], name=item["name"], arguments=item["arguments"])
                )
            elif item["type"] == "message":
                content(generation, item.get("content", []), output_audio_format)
                auxiliary(generation, item)
            elif item["type"] in {"reasoning", "reasoning_text", "summary_text"}:
                plaintext(generation, "reasoning", item)
            elif item["type"] == "refusal":
                plaintext(generation, "refusal", item)
            elif item["type"] == "image_generation_call":
                image_call(generation, item)
            elif item["type"] in {"audio", "output_audio", "video", "output_video", "output_image"}:
                media_content(generation, item, output_audio_format)
    else:
        generation.finish = value.get("stop_reason")
        for item in value.get("content", []):
            if item["type"] == "tool_use":
                generation.tools.append(
                    ToolCall(id=item["id"], name=item["name"], arguments=json.dumps(item["input"]))
                )
            else:
                content(generation, item, output_audio_format)
        if generation.media:
            raise ProtocolError("Anthropic Messages does not support generated media output")
        auxiliary(generation, value)
    if not generation.finish:
        raise ProtocolError("Response missing termination reason")
    media_completeness(generation)
    return generation


def same_tool(previous: dict[str, str], final: ToolCall) -> bool:
    if previous["name"] != final.name:
        return False
    try:
        return equal(loads(previous["arguments"]), loads(final.arguments))
    except ValueError:
        return previous["arguments"] == final.arguments


class Accumulator:
    def __init__(self, protocol: str, output_audio_format: str | None = None) -> None:
        self.protocol = protocol
        self.output_audio_format = output_audio_format
        self.generation = Generation()
        self.calls: dict[int, dict[str, str]] = {}
        self.terminated = False
        self.stop_seen = False
        self.message_started = False
        self.blocks: dict[int, str] = {}
        self.closed_blocks: set[int] = set()
        self.signature_seen: set[int] = set()
        self.signature_nonempty: set[int] = set()
        self.audio_stream: Any = None
        self.audio_done = False
        self.final_output_media: list[dict[str, Any]] | None = None
        self.media_finished = False
        self.image_partial_count = 0
        self.legacy: dict[str, str] | None = None
        self.legacy_id = "legacy:" + uuid.uuid4().hex

    def accept(self, event: str, data: str) -> None:
        try:
            self._accept(event, data)
        except (KeyError, TypeError, IndexError) as error:
            raise ProtocolError("Malformed SSE event structure") from error

    def _accept(self, event: str, data: str) -> None:
        if self.terminated:
            raise ProtocolError("Data after terminal event")
        if data == "[DONE]":
            if self.protocol != "openai-chat":
                raise ProtocolError("Unexpected Chat Completions terminal event")
            self.terminated = True
            return
        try:
            value = loads(data)
        except ValueError as error:
            raise ProtocolError("Malformed SSE JSON") from error
        if not isinstance(value, dict):
            raise ProtocolError("SSE event must contain an object")
        kind = value.get("type", event)
        if "error" in value or kind == "error":
            raise ProtocolError("Provider stream error")
        if self.protocol == "openai-chat":
            identify(self.generation, value.get("model"))
            if value.get("usage"):
                account(self.generation, value["usage"], cumulative=True)
            for choice in value.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ProtocolError("Multiple choices require an explicit adapter")
                delta = choice.get("delta", {})
                if self.stop_seen and any(
                    delta.get(key)
                    for key in (
                        "content",
                        "tool_calls",
                        "function_call",
                        "reasoning_content",
                        "reasoning",
                        "refusal",
                        "audio",
                        "images",
                    )
                ):
                    raise ProtocolError("Content after Chat Completions finish reason")
                if delta.get("function_call") is not None:
                    legacy = delta["function_call"]
                    if not isinstance(legacy, dict):
                        raise ProtocolError("Malformed legacy function delta")
                    if self.legacy is None:
                        self.legacy = {"name": "", "arguments": ""}
                    for field in ("name", "arguments"):
                        if field in legacy:
                            if not isinstance(legacy[field], str):
                                raise ProtocolError("Legacy function fragments require text")
                            self.legacy[field] += legacy[field]
                content(self.generation, delta.get("content"), self.output_audio_format)
                auxiliary(self.generation, delta)
                if delta.get("audio") is not None:
                    self.audio(delta["audio"])
                if delta.get("images") is not None:
                    message_media(
                        self.generation, {"images": delta["images"]}, self.output_audio_format
                    )
                if "message" in choice:
                    final = Generation()
                    message = choice["message"]
                    if message.get("function_call") is not None:
                        completed = message["function_call"]
                        if not isinstance(completed, dict):
                            raise ProtocolError("Malformed final legacy function call")
                        if self.legacy is not None and self.legacy != completed:
                            self.generation.raw_text = json.dumps(self.legacy, ensure_ascii=False)
                            observe(
                                self.generation,
                                "stream-final-legacy-call-mismatch",
                                "Final legacy action differs from streamed fragments",
                            )
                        self.legacy = completed
                    for index, tool in enumerate(message.get("tool_calls", [])):
                        final_call = ToolCall(
                            id=tool["id"],
                            name=tool["function"]["name"],
                            arguments=tool["function"]["arguments"],
                        )
                        previous = self.calls.get(index)
                        if previous and (
                            previous["id"] != final_call.id or not same_tool(previous, final_call)
                        ):
                            observe(
                                self.generation,
                                "stream-final-tool-mismatch",
                                "Final native tool call differs from streamed fragments",
                            )
                        self.calls[index] = {
                            "id": final_call.id,
                            "name": final_call.name,
                            "arguments": final_call.arguments,
                        }
                    content(final, message.get("content"), self.output_audio_format)
                    auxiliary(final, message)
                    message_media(final, message, self.output_audio_format)
                    if final.media:
                        self.final_output_media = final.media
                    for channel, keys in (
                        ("text", ("content",)),
                        ("reasoning", ("reasoning", "reasoning_content")),
                        ("refusal", ("refusal",)),
                    ):
                        if any(key in message for key in keys):
                            final_channel(self.generation, channel, getattr(final, channel))
                    self.generation.observations.extend(final.observations)
                if choice.get("finish_reason"):
                    self.generation.finish = choice["finish_reason"]
                    self.stop_seen = True
                for tool in delta.get("tool_calls", []):
                    call = self.calls.setdefault(
                        tool["index"], {"id": "", "name": "", "arguments": ""}
                    )
                    call["id"] += tool.get("id", "")
                    call["name"] += tool.get("function", {}).get("name", "")
                    call["arguments"] += tool.get("function", {}).get("arguments", "")
        elif self.protocol == "openai-responses":
            if kind == "response.audio.delta":
                if self.audio_done:
                    raise ProtocolError("Audio data after audio completion")
                self.audio(
                    {
                        "data": value["delta"],
                        **({"format": value["format"]} if "format" in value else {}),
                    }
                )
            elif kind == "response.audio_transcript.delta":
                self.audio({"transcript": value["delta"]})
            elif kind == "response.audio.done":
                self.audio_done = True
            elif kind == "response.image_generation_call.partial_image":
                self.image_partial_count += 1
            elif (
                kind == "response.output_item.done"
                and value["item"]["type"] == "image_generation_call"
            ):
                image_call(self.generation, value["item"])
            elif kind == "response.output_text.delta":
                plaintext(self.generation, "text", value["delta"])
            elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
                plaintext(self.generation, "reasoning", value["delta"])
            elif kind == "response.refusal.delta":
                plaintext(self.generation, "refusal", value["delta"])
            elif kind in {"response.reasoning_summary_text.done", "response.reasoning_text.done"}:
                completed_piece(self.generation, "reasoning", value["text"])
            elif kind == "response.refusal.done":
                completed_piece(self.generation, "refusal", value["refusal"])
            elif kind == "response.output_text.done":
                completed_piece(self.generation, "text", value["text"])
            elif kind == "response.reasoning_summary_part.added":
                plaintext(self.generation, "reasoning", value["part"])
            elif kind == "response.content_part.added":
                content(self.generation, value["part"], self.output_audio_format)
            elif kind == "response.output_item.added" and value["item"]["type"] == "function_call":
                item = value["item"]
                self.calls[value["output_index"]] = {
                    "id": item["call_id"],
                    "name": item["name"],
                    "arguments": item.get("arguments", ""),
                }
            elif kind == "response.output_item.added" and value["item"]["type"] == "reasoning":
                plaintext(self.generation, "reasoning", value["item"])
            elif kind == "response.output_item.added" and value["item"]["type"] == "message":
                content(self.generation, value["item"].get("content", []), self.output_audio_format)
            elif kind == "response.function_call_arguments.delta":
                self.calls[value["output_index"]]["arguments"] += value["delta"]
            elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                final = decode_json(self.protocol, value["response"], self.output_audio_format)
                if any(
                    item.get("type") == "message" for item in value["response"].get("output", [])
                ):
                    final_channel(self.generation, "text", final.text)
                output = value["response"].get("output", [])
                if final.reasoning or any(item.get("type") == "reasoning" for item in output):
                    final_channel(self.generation, "reasoning", final.reasoning)
                if final.refusal:
                    final_channel(self.generation, "refusal", final.refusal)
                if final.tools:
                    for streamed_call in self.calls.values():
                        current = next(
                            (tool for tool in final.tools if tool.id == streamed_call["id"]), None
                        )
                        if current is None or not same_tool(streamed_call, current):
                            prior = json.dumps(streamed_call, ensure_ascii=False)
                            prefix = self.generation.raw_text
                            self.generation.raw_text = (prefix + "\n" if prefix else "") + prior
                    for tool in final.tools:
                        previous = next(
                            (call for call in self.calls.values() if call["id"] == tool.id), None
                        )
                        if previous and not same_tool(previous, tool):
                            observe(
                                self.generation,
                                "stream-final-tool-mismatch",
                                "Final native tool call differs from accumulated deltas",
                            )
                    self.calls = {i: t.model_dump() for i, t in enumerate(final.tools)}
                if final.media:
                    self.final_output_media = final.media
                self.generation.finish = final.finish
                identify(self.generation, final.model)
                account(self.generation, final.usage, cumulative=True)
                self.generation.observations.extend(final.observations)
                self.terminated = True
            elif (
                isinstance(kind, str)
                and any(part in kind for part in ("audio", "video", "image"))
                and kind
                not in {
                    "response.audio_transcript.done",
                    "response.image_generation_call.in_progress",
                    "response.image_generation_call.generating",
                    "response.image_generation_call.completed",
                }
            ):
                raise ProtocolError("Unsupported media streaming event")
        else:
            if kind == "message_start":
                if self.message_started:
                    raise ProtocolError("Duplicate message_start")
                self.message_started = True
                identify(self.generation, value["message"].get("model"))
                account(self.generation, value["message"].get("usage", {}), cumulative=True)
                content(self.generation, value["message"].get("content", []))
                auxiliary(self.generation, value["message"])
            elif kind == "content_block_start":
                if not self.message_started or self.stop_seen:
                    raise ProtocolError("Content block outside an active message")
                index = value["index"]
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index in self.blocks
                ):
                    raise ProtocolError("Invalid or duplicate content block index")
                item = value["content_block"]
                self.blocks[index] = item["type"]
                if item["type"] == "thinking" and "signature" in item:
                    self.signature(index, item["signature"])
                if item["type"] == "tool_use":
                    self.calls[value["index"]] = {
                        "id": item["id"],
                        "name": item["name"],
                        "arguments": json.dumps(item["input"]) if item.get("input") else "",
                    }
                elif item["type"] == "text":
                    plaintext(self.generation, "text", item.get("text", ""))
                elif item["type"] == "thinking":
                    plaintext(self.generation, "reasoning", item.get("thinking", ""))
                elif item["type"] == "redacted_thinking":
                    plaintext(self.generation, "reasoning", item)
                elif item["type"] == "refusal":
                    plaintext(self.generation, "refusal", item.get("refusal", item.get("text", "")))
                elif item["type"] in {"image", "audio", "video", "document"}:
                    raise ProtocolError(
                        "Anthropic Messages does not support generated media output"
                    )
            elif kind == "content_block_delta":
                index = value["index"]
                if index not in self.blocks or index in self.closed_blocks:
                    raise ProtocolError("Delta outside an open content block")
                delta = value["delta"]
                required = {
                    "text_delta": "text",
                    "input_json_delta": "tool_use",
                    "thinking_delta": "thinking",
                    "signature_delta": "thinking",
                }.get(delta["type"])
                if required is not None and self.blocks[index] != required:
                    raise ProtocolError("Delta type differs from content block type")
                if delta["type"] == "text_delta":
                    plaintext(self.generation, "text", delta["text"])
                elif delta["type"] == "input_json_delta":
                    self.calls[value["index"]]["arguments"] += delta["partial_json"]
                elif delta["type"] == "thinking_delta":
                    plaintext(self.generation, "reasoning", delta["thinking"])
                elif delta["type"] == "refusal_delta":
                    plaintext(self.generation, "refusal", delta["refusal"])
                elif delta["type"] == "signature_delta":
                    self.signature(index, delta["signature"])
            elif kind == "content_block_stop":
                index = value["index"]
                if index not in self.blocks or index in self.closed_blocks:
                    raise ProtocolError("Stop outside an open content block")
                self.closed_blocks.add(index)
                if index in self.signature_seen and index not in self.signature_nonempty:
                    observe(
                        self.generation,
                        "empty-thinking-signature",
                        "Explicit thinking signature aggregate was empty at block completion; "
                        "opaque signatures are not cryptographically verified",
                        "low",
                    )
            elif kind == "message_delta":
                if not self.message_started:
                    raise ProtocolError("Message delta before message_start")
                if value["delta"].get("stop_reason") and self.blocks.keys() != self.closed_blocks:
                    raise ProtocolError("Message termination with open content blocks")
                self.generation.finish = value["delta"].get("stop_reason")
                self.stop_seen = bool(self.generation.finish)
                account(self.generation, value.get("usage", {}), cumulative=True)
            elif kind == "message_stop":
                if not self.message_started or self.blocks.keys() != self.closed_blocks:
                    raise ProtocolError("Message stopped before its content blocks completed")
                self.terminated = True

    def signature(self, index: int, fragment: Any) -> None:
        if not isinstance(fragment, str):
            raise ProtocolError("Thinking signature fragment must be a string")
        self.signature_seen.add(index)
        if fragment.strip():
            self.signature_nonempty.add(index)

    def audio(self, value: Any) -> None:
        from ablitbench.parity.media_outputs import AudioStream

        if not isinstance(value, dict):
            raise ProtocolError("Malformed streamed audio object")
        if self.audio_stream is None:
            self.audio_stream = AudioStream(self.output_audio_format)
        try:
            self.audio_stream.append(value)
        except ValueError as error:
            raise ProtocolError(str(error)) from error

    def close(self) -> None:
        if self.audio_stream is not None:
            self.audio_stream.close()

    def finish(self) -> Generation:
        if not self.terminated or not self.generation.finish:
            raise ProtocolError("Stream disconnected without terminal event and reason")
        if not self.media_finished:
            try:
                if self.audio_stream is not None:
                    media = self.audio_stream.finish()
                    if media is not None:
                        self.generation.media.append(media)
                if self.final_output_media is not None:
                    final_media(self.generation, self.final_output_media)
                if self.image_partial_count:
                    observe(
                        self.generation,
                        "intermediate-image-coverage",
                        str(self.image_partial_count)
                        + " intermediate images retained in wire evidence, "
                        "not independently assessed",
                    )
                    if not self.generation.media:
                        raise ProtocolError("Image stream has no final generated image")
            except ValueError as error:
                raise ProtocolError(str(error)) from error
            finally:
                self.close()
            self.media_finished = True
        self.generation.tools = [
            ToolCall.model_validate(
                value | {"arguments": value["arguments"] or "{}"}
                if self.protocol == "anthropic-messages"
                else value
            )
            for _, value in sorted(self.calls.items())
        ]
        if self.legacy is not None:
            legacy_function(self.generation, self.legacy, self.legacy_id)
        elif self.generation.finish == "function_call":
            self.generation.finish = "incomplete"
            observe(
                self.generation,
                "missing-legacy-action",
                "Legacy finish reason has no action payload",
            )
        media_completeness(self.generation)
        return self.generation
