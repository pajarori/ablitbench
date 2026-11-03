from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx2 as httpx
import psutil

from ablitbench.limits import available_memory
from ablitbench.models import Evidence, Generation, Model
from ablitbench.privacy import restrict_directory

if TYPE_CHECKING:
    from ablitbench.store import Store


class Capture:
    def __init__(self, directory: Path) -> None:
        if directory.is_symlink():
            raise ValueError("Traffic spool directory must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        restrict_directory(directory)
        descriptor, name = tempfile.mkstemp(prefix="wire-", dir=directory)
        self.path = Path(name)
        self.stream = os.fdopen(descriptor, "wb", buffering=0)
        self.checksum = hashlib.sha256()
        self.size = 0
        self.storage_error: str | None = None

    def write(self, chunk: bytes) -> None:
        view = memoryview(chunk)
        while view:
            written = self.stream.write(view)
            if not written:
                raise OSError("Traffic spool did not accept input bytes")
            self.checksum.update(view[:written])
            self.size += written
            view = view[written:]

    def persist(self, store: Store, complete: bool, metadata: dict[str, Any]) -> list[str]:
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except OSError as error:
            self.storage_error = type(error).__name__
            complete = False
        finally:
            self.stream.close()
        identity = self.checksum.hexdigest()
        destination = store.blobs / identity
        os.replace(self.path, destination)
        self.path = destination
        evidence = store.commit_blob(
            Evidence(sha256=identity, size=self.size, media_type="application/octet-stream")
        )
        description = store.evidence(
            {
                "response_body": evidence.sha256,
                "captured_bytes": self.size,
                "complete": complete,
                "storage_error": self.storage_error,
                **metadata,
            }
        )
        return [description, evidence.sha256]

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()


class FileStream(httpx.AsyncByteStream):
    def __init__(self, path: Path) -> None:
        self.path = path

    async def __aiter__(self) -> AsyncIterator[bytes]:
        from ablitbench.formats.snapshot import secure_open

        with secure_open(self.path) as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk


def normalize_worker_response(source: Path, destination: Path) -> dict[str, int]:
    from ablitbench.formats.json import loads
    from ablitbench.formats.snapshot import secure_open
    from ablitbench.models import WorkerResponse

    with secure_open(source) as stream:
        raw = stream.read()
    response = WorkerResponse.model_validate(loads(raw))
    serialized = response.model_dump_json().encode()
    with destination.open("xb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return {"materialization_bytes": resident_size(response) + len(raw) + len(serialized)}


def resident_size(value: Any) -> int:
    total = 0
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if isinstance(item, Model):
            pending.extend((item.__dict__, item.__pydantic_fields_set__))
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return total


async def decode(
    path: Path,
    protocol: str,
    streaming: bool,
    content_type: str,
    content_encoding: str,
    output_audio_format: str | None = None,
    provider_options: dict[str, Any] | None = None,
) -> Generation:
    from ablitbench.formats.json import loads
    from ablitbench.parity import providers
    from ablitbench.transport import Accumulator, CapabilityError, ProtocolError, decode_json, sse

    options = provider_options if provider_options is not None else {}
    native = protocol in providers.protocols()
    if not native and protocol not in {"openai-chat", "openai-responses", "anthropic-messages"}:
        raise CapabilityError("Unknown response protocol")
    codecs = importlib.import_module("httpx2._decoders")
    encodings = [value.strip().lower() for value in content_encoding.split(",") if value.strip()]
    if any(value not in codecs.SUPPORTED_DECODERS for value in encodings):
        raise CapabilityError("Installed HTTP codec does not support the declared Content-Encoding")
    if (
        len([value for value in encodings if value != "identity"])
        > codecs.MultiDecoder.max_decode_links
    ):
        raise CapabilityError("Content-Encoding chain exceeds installed codec guard")
    response = httpx.Response(
        200,
        headers={"content-type": content_type, "content-encoding": content_encoding},
        stream=FileStream(path),
    )
    accumulator: Any = None
    try:
        if streaming:
            if native:
                accumulator = providers.Accumulator(protocol)
            else:
                accumulator = Accumulator(protocol, output_audio_format)
            if protocol == "bedrock-converse":
                from ablitbench.parity.native_provider import EventStream

                if "application/vnd.amazon.eventstream" not in content_type.lower():
                    raise ProtocolError("Bedrock stream is not Amazon eventstream")
                frames = EventStream()
                async for chunk in response.aiter_bytes():
                    for event, data in frames.accept(chunk):
                        accumulator.accept(event, data)
                frames.finish()
            elif protocol.startswith("ollama-"):
                if not any(
                    kind in content_type.lower() for kind in ("ndjson", "jsonl", "application/json")
                ):
                    raise ProtocolError("Ollama streaming response is not NDJSON")
                async for line in response.aiter_lines():
                    if line:
                        accumulator.accept("message", line)
            else:
                if "text/event-stream" not in content_type.lower():
                    raise ProtocolError("Streaming request did not return text/event-stream")
                async for event, data in sse(response):
                    accumulator.accept(event, data)
            streamed: Generation = accumulator.finish()
            return streamed
        content = await response.aread()
        if native and protocol.startswith("nvcf-") and options.get("_http_status") == 202:
            return Generation(finish="incomplete")
        if native and protocol == "rest" and not options.get("response_json", True):
            return providers.decode(protocol, content.decode("utf-8"), options)
        try:
            value = loads(content)
        except ValueError as error:
            raise ProtocolError("Malformed HTTP JSON") from error
        if native:
            if protocol.startswith("replicate") and isinstance(value, dict):
                options["_provider_envelope"] = {
                    key: value[key] for key in ("id", "status", "urls") if key in value
                }
                if value.get("status") != "succeeded":
                    return Generation(
                        finish="failed"
                        if value.get("status") in {"failed", "canceled"}
                        else "incomplete"
                    )
            return providers.decode(protocol, value, options)
        if not isinstance(value, dict):
            raise ProtocolError("HTTP JSON response must contain an object")
        return decode_json(protocol, value, output_audio_format)
    except httpx.DecodingError as error:
        raise ProtocolError("Malformed compressed HTTP response") from error
    finally:
        if accumulator is not None:
            accumulator.close()
        await response.aclose()


def decode_generation(
    path: Path,
    destination: Path,
    protocol: str,
    streaming: bool,
    content_type: str,
    content_encoding: str,
    output_audio_format: str | None = None,
    provider_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(provider_options or {})
    generation = asyncio.run(
        decode(
            path, protocol, streaming, content_type, content_encoding, output_audio_format, options
        )
    )
    value = generation.model_dump(mode="json")
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temporary = destination.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return {
        "provider_envelope": options.get("_provider_envelope"),
        "normalized_bytes": destination.stat().st_size,
        "materialization_bytes": destination.stat().st_size
        + sys.getsizeof(serialized)
        + resident_size([value, generation]),
        "scope": "Measured JSON/Unicode/container representation estimate; no response-byte limit",
    }


def decode_probe(path: Path, destination: Path, content_encoding: str) -> dict[str, Any]:
    async def materialize() -> dict[str, Any]:
        codecs = importlib.import_module("httpx2._decoders")
        encodings = [item.strip().lower() for item in content_encoding.split(",") if item.strip()]
        if any(item not in codecs.SUPPORTED_DECODERS for item in encodings):
            raise ValueError("Unsupported probe response encoding")
        response = httpx.Response(
            200, headers={"content-encoding": content_encoding}, stream=FileStream(path)
        )
        try:
            raw = await response.aread()
        finally:
            await response.aclose()
        return {"body": raw.decode("utf-8", errors="replace"), "decoded_bytes": len(raw)}

    value = asyncio.run(materialize())
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    temporary = destination.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return {
        "materialization_bytes": resident_size(value)
        + sys.getsizeof(serialized)
        + len(serialized.encode())
    }


async def normalize(
    path: Path,
    protocol: str,
    streaming: bool,
    content_type: str,
    content_encoding: str,
    directory: Path,
    memory: int | None,
    output_audio_format: str | None = None,
    provider_options: dict[str, Any] | None = None,
) -> tuple[Generation, dict[str, Any]]:
    from ablitbench.io import operation
    from ablitbench.transport import CapabilityError, ProtocolError

    with tempfile.TemporaryDirectory(prefix="normalized-", dir=directory) as temporary:
        private = Path(temporary)
        restrict_directory(private)
        destination = private / "generation.json"
        try:
            measured = await operation(
                "decode_generation",
                [
                    str(path),
                    str(destination),
                    protocol,
                    streaming,
                    content_type,
                    content_encoding,
                    output_audio_format,
                    provider_options,
                ],
                memory=memory,
            )
        except (CapabilityError, ProtocolError):
            raise
        except (OSError, MemoryError) as error:
            raise CapabilityError(
                "HTTP normalization exceeded device/user capacity; wire evidence retained"
            ) from error
        capacity = available_memory()
        if memory is not None:
            capacity = min(capacity, max(0, memory - psutil.Process().memory_info().rss))
        if measured["materialization_bytes"] > capacity:
            raise CapabilityError(
                "Generation cannot fit measured host/user memory; wire evidence retained"
            )
        try:
            with destination.open(encoding="utf-8") as stream:
                generation = Generation.model_validate(json.load(stream))
        except (MemoryError, OverflowError) as error:
            raise CapabilityError(
                "Generation materialization exhausted host capacity; full wire evidence retained"
            ) from error
        return generation, measured


def redact_response_wire(
    source: Path,
    destination: Path,
    content_encoding: str,
    secrets: list[str],
    marker: str = "ABLIB_REDACTED_CREDENTIAL",
) -> dict[str, Any]:
    import base64
    import re
    from urllib.parse import quote, quote_plus

    patterns: set[bytes] = set()
    for secret in secrets:
        if not secret:
            continue
        patterns.update(
            value.encode()
            for value in (
                secret,
                base64.b64encode(secret.encode()).decode(),
                secret.encode().hex(),
                quote(secret, safe=""),
                quote_plus(secret),
                json.dumps(secret, ensure_ascii=True)[1:-1],
                json.dumps(secret, ensure_ascii=False)[1:-1],
            )
        )
    expression = (
        re.compile(b"|".join(re.escape(value) for value in sorted(patterns, key=len, reverse=True)))
        if patterns
        else None
    )
    retained = max((len(value) for value in patterns), default=1) - 1

    async def process() -> dict[str, Any]:
        response = httpx.Response(
            200, headers={"content-encoding": content_encoding}, stream=FileStream(source)
        )
        buffer = b""
        matches = 0
        original = hashlib.sha256()
        decoded_size = 0
        with destination.open("wb") as output:

            def emit(final: bool = False) -> None:
                nonlocal buffer, matches
                cutoff = len(buffer) if final else max(0, len(buffer) - retained)
                end = 0
                if expression is not None:
                    for match in expression.finditer(buffer):
                        if match.start() >= cutoff:
                            break
                        output.write(buffer[end : match.start()])
                        output.write(marker.encode())
                        end = match.end()
                        matches += 1
                consumed = max(cutoff, end)
                output.write(buffer[end:consumed])
                buffer = buffer[consumed:]

            try:
                async for chunk in response.aiter_bytes():
                    original.update(chunk)
                    decoded_size += len(chunk)
                    buffer += chunk
                    emit()
                emit(True)
                output.flush()
                os.fsync(output.fileno())
                written = output.tell()
            finally:
                await response.aclose()
        return {
            "redactions": matches,
            "decoded_bytes": decoded_size,
            "decoded_sha256": original.hexdigest(),
            "redacted_bytes": written,
        }

    return asyncio.run(process())
