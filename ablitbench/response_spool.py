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
    path: Path, protocol: str, streaming: bool, content_type: str, content_encoding: str
) -> Generation:
    from ablitbench.formats.json import loads
    from ablitbench.transport import Accumulator, CapabilityError, ProtocolError, decode_json, sse

    if protocol not in {"openai-chat", "openai-responses", "anthropic-messages"}:
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
    try:
        if streaming:
            if "text/event-stream" not in content_type.lower():
                raise ProtocolError("Streaming request did not return text/event-stream")
            accumulator = Accumulator(protocol)
            async for event, data in sse(response):
                accumulator.accept(event, data)
            return accumulator.finish()
        content = await response.aread()
        try:
            value = loads(content)
        except ValueError as error:
            raise ProtocolError("Malformed HTTP JSON") from error
        if not isinstance(value, dict):
            raise ProtocolError("HTTP JSON response must contain an object")
        return decode_json(protocol, value)
    except httpx.DecodingError as error:
        raise ProtocolError("Malformed compressed HTTP response") from error
    finally:
        await response.aclose()


def decode_generation(
    path: Path,
    destination: Path,
    protocol: str,
    streaming: bool,
    content_type: str,
    content_encoding: str,
) -> dict[str, Any]:
    generation = asyncio.run(decode(path, protocol, streaming, content_type, content_encoding))
    value = generation.model_dump(mode="json")
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temporary = destination.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return {
        "normalized_bytes": destination.stat().st_size,
        "materialization_bytes": destination.stat().st_size
        + sys.getsizeof(serialized)
        + resident_size([value, generation]),
        "scope": "Measured JSON/Unicode/container representation estimate; no response-byte limit",
    }


async def normalize(
    path: Path,
    protocol: str,
    streaming: bool,
    content_type: str,
    content_encoding: str,
    directory: Path,
    memory: int | None,
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
                [str(path), str(destination), protocol, streaming, content_type, content_encoding],
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
