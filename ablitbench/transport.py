from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx2 as httpx

from ablitbench.models import Generation, ScanConfig, Target, ToolCall
from ablitbench.store import Store
from ablitbench.streams import BinaryStream


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


class Capture(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, destination: BinaryStream) -> None:
        self.stream, self.destination = stream, destination

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.stream:
            self.destination.write(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self.stream.aclose()


def encode_messages(protocol: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if protocol == "openai-chat":
        return messages
    encoded: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if protocol == "openai-responses":
            if role == "tool":
                encoded.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": message["content"],
                    }
                )
            else:
                if message.get("content"):
                    encoded.append({"role": role, "content": message["content"]})
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
                            "content": message["content"],
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
                    blocks.append({"type": "text", "text": message["content"]})
                for tool in message.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool["id"],
                            "name": tool["function"]["name"],
                            "input": json.loads(tool["function"]["arguments"]),
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
        self.config, self.target, self.store = config, target, store
        self.protocol: str = target.protocol
        self.client = httpx.AsyncClient(
            transport=transport,
            timeout=config.request_timeout,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self.last_evidence: list[str] = []
        self.not_before = 0.0

    async def close(self) -> None:
        await self.client.aclose()

    def encode(
        self,
        protocol: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body: dict[str, Any] = {"model": self.target.model, "stream": stream}
        headers = {"content-type": "application/json"}
        key = os.environ.get(self.target.key_env)
        if protocol == "anthropic-messages":
            if self.config.max_output_tokens is None:
                raise CapabilityError(
                    "Anthropic requires max_output_tokens from user configuration"
                )
            suffix = "messages"
            body["system"] = "\n".join(m["content"] for m in messages if m["role"] == "system")
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
                body["messages"] = messages
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
        return base, body, headers

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> Generation:
        if self.not_before > time.monotonic():
            await asyncio.sleep(self.not_before - time.monotonic())
        protocols = (
            [self.protocol]
            if self.protocol != "auto"
            else ["openai-chat", "openai-responses", "anthropic-messages"]
        )
        for protocol in protocols:
            url, body, headers = self.encode(protocol, messages, tools, stream)
            self.last_evidence = [
                self.store.evidence({"url": url, "protocol": protocol, "request": body})
            ]
            with tempfile.TemporaryFile() as capture:
                try:
                    async with self.client.stream(
                        "POST", url, json=body, headers=headers
                    ) as response:
                        if response.is_stream_consumed:
                            capture.write(response.content)
                        elif isinstance(response.stream, httpx.AsyncByteStream):
                            response.stream = Capture(response.stream, capture)
                        self.last_evidence.append(
                            self.store.evidence(
                                {
                                    "status": response.status_code,
                                    "retry_after": response.headers.get("retry-after"),
                                }
                            )
                        )
                        retry = response.headers.get("retry-after")
                        if retry and response.status_code in {429, 503}:
                            try:
                                seconds = float(retry)
                            except ValueError:
                                try:
                                    seconds = (
                                        parsedate_to_datetime(retry) - datetime.now(UTC)
                                    ).total_seconds()
                                except (ValueError, TypeError):
                                    seconds = 0.0
                            self.not_before = time.monotonic() + max(0.0, seconds)
                        if self.protocol == "auto" and response.status_code in {404, 405}:
                            async for _chunk in response.aiter_bytes():
                                pass
                            continue
                        if response.status_code >= 300:
                            async for _chunk in response.aiter_bytes():
                                pass
                            raise ProtocolError(
                                f"HTTP {response.status_code}; request not retried; "
                                "consult response evidence"
                            )
                        if not stream:
                            content = await response.aread()
                            generation = decode_json(protocol, json.loads(content))
                        else:
                            if "text/event-stream" not in response.headers.get("content-type", ""):
                                async for _chunk in response.aiter_bytes():
                                    pass
                                raise ProtocolError(
                                    "Streaming request did not return text/event-stream"
                                )
                            accumulator = Accumulator(protocol)
                            async for event, data in sse(response):
                                accumulator.accept(event, data)
                            generation = accumulator.finish()
                        self.protocol = protocol
                finally:
                    capture.seek(0)
                    self.last_evidence.append(
                        self.store.blob(
                            iter(lambda: capture.read(1024 * 1024), b""), "application/octet-stream"
                        ).sha256
                    )
            generation.evidence = list(self.last_evidence)
            return generation
        raise CapabilityError("No compatible protocol on the supplied endpoint")


def decode_json(protocol: str, value: dict[str, Any]) -> Generation:
    if "error" in value:
        raise ProtocolError("Provider returned an error object")
    generation = Generation(model=value.get("model"), usage=value.get("usage") or {})
    if protocol == "openai-chat":
        choice = value["choices"][0]
        message = choice["message"]
        generation.text = message.get("content") or ""
        generation.finish = choice.get("finish_reason")
        generation.tools = [
            ToolCall(id=t["id"], name=t["function"]["name"], arguments=t["function"]["arguments"])
            for t in message.get("tool_calls", [])
        ]
    elif protocol == "openai-responses":
        generation.finish = value.get("status")
        for item in value.get("output", []):
            if item["type"] == "function_call":
                generation.tools.append(
                    ToolCall(id=item["call_id"], name=item["name"], arguments=item["arguments"])
                )
            elif item["type"] == "message":
                generation.text += "".join(
                    c.get("text", "") for c in item.get("content", []) if c["type"] == "output_text"
                )
    else:
        generation.finish = value.get("stop_reason")
        for item in value.get("content", []):
            if item["type"] == "text":
                generation.text += item["text"]
            elif item["type"] == "tool_use":
                generation.tools.append(
                    ToolCall(id=item["id"], name=item["name"], arguments=json.dumps(item["input"]))
                )
    if not generation.finish:
        raise ProtocolError("Response missing termination reason")
    return generation


class Accumulator:
    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self.generation = Generation()
        self.calls: dict[int, dict[str, str]] = {}
        self.terminated = False
        self.stop_seen = False

    def accept(self, event: str, data: str) -> None:
        if data == "[DONE]":
            self.terminated = True
            return
        if self.terminated:
            raise ProtocolError("Data after terminal event")
        try:
            value = json.loads(data)
        except json.JSONDecodeError as error:
            raise ProtocolError("Malformed SSE JSON") from error
        kind = value.get("type", event)
        if "error" in value or kind == "error":
            raise ProtocolError("Provider stream error")
        if self.protocol == "openai-chat":
            self.generation.model = value.get("model", self.generation.model)
            if value.get("usage"):
                self.generation.usage.update(value["usage"])
            for choice in value.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ProtocolError("Multiple choices require an explicit adapter")
                delta = choice.get("delta", {})
                self.generation.text += delta.get("content") or ""
                if choice.get("finish_reason"):
                    self.generation.finish = choice["finish_reason"]
                for tool in delta.get("tool_calls", []):
                    call = self.calls.setdefault(
                        tool["index"], {"id": "", "name": "", "arguments": ""}
                    )
                    call["id"] += tool.get("id", "")
                    call["name"] += tool.get("function", {}).get("name", "")
                    call["arguments"] += tool.get("function", {}).get("arguments", "")
        elif self.protocol == "openai-responses":
            if kind == "response.output_text.delta":
                self.generation.text += value["delta"]
            elif kind == "response.output_item.added" and value["item"]["type"] == "function_call":
                item = value["item"]
                self.calls[value["output_index"]] = {
                    "id": item["call_id"],
                    "name": item["name"],
                    "arguments": item.get("arguments", ""),
                }
            elif kind == "response.function_call_arguments.delta":
                self.calls[value["output_index"]]["arguments"] += value["delta"]
            elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                final = decode_json(self.protocol, value["response"])
                if final.text:
                    self.generation.text = final.text
                if final.tools:
                    self.calls = {i: t.model_dump() for i, t in enumerate(final.tools)}
                self.generation.finish = final.finish
                self.generation.model = final.model
                self.generation.usage = final.usage
                self.terminated = True
        else:
            if kind == "message_start":
                self.generation.model = value["message"].get("model")
                self.generation.usage.update(value["message"].get("usage", {}))
            elif kind == "content_block_start":
                item = value["content_block"]
                if item["type"] == "tool_use":
                    self.calls[value["index"]] = {
                        "id": item["id"],
                        "name": item["name"],
                        "arguments": "",
                    }
                elif item["type"] == "text":
                    self.generation.text += item.get("text", "")
            elif kind == "content_block_delta":
                delta = value["delta"]
                if delta["type"] == "text_delta":
                    self.generation.text += delta["text"]
                elif delta["type"] == "input_json_delta":
                    self.calls[value["index"]]["arguments"] += delta["partial_json"]
            elif kind == "message_delta":
                self.generation.finish = value["delta"].get("stop_reason")
                self.generation.usage.update(value.get("usage", {}))
            elif kind == "message_stop":
                self.terminated = True

    def finish(self) -> Generation:
        if not self.terminated or not self.generation.finish:
            raise ProtocolError("Stream disconnected without terminal event and reason")
        self.generation.tools = [
            ToolCall.model_validate(value) for _, value in sorted(self.calls.items())
        ]
        return self.generation
