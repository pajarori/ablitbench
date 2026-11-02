from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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
        self.last_evidence = []
        for protocol in protocols:
            url, body, headers = self.encode(protocol, messages, tools, stream)
            self.last_evidence.append(
                self.store.evidence({"url": url, "protocol": protocol, "request": body})
            )
            capture = Capture(self.store.directory / "traffic")
            complete = False
            metadata: dict[str, Any] = {"representation": "wire"}
            try:
                async with self.client.stream("POST", url, json=body, headers=headers) as response:
                    status = response.status_code
                    content_type = response.headers.get("content-type", "")
                    content_encoding = response.headers.get("content-encoding", "")
                    retry = response.headers.get("retry-after")
                    metadata.update(
                        status=status,
                        retry_after=retry,
                        content_type=content_type,
                        content_encoding=content_encoding,
                    )
                    if response.is_stream_consumed:
                        metadata["representation"] = "predecoded-transport"
                        content_encoding = ""
                        capture.write(response.content)
                    else:
                        async for chunk in response.aiter_raw():
                            capture.write(chunk)
                    complete = True
            except (httpx.HTTPError, OSError) as error:
                if isinstance(error, OSError):
                    capture.storage_error = type(error).__name__
                raise CapabilityError(
                    "HTTP response capture incomplete; retained prefix requires coverage gap"
                ) from error
            finally:
                failure = sys.exc_info()[0] is not None
                try:
                    self.last_evidence.extend(capture.persist(self.store, complete, metadata))
                    if capture.storage_error and not failure:
                        raise CapabilityError(
                            "HTTP response storage incomplete; retained prefix requires "
                            "coverage gap"
                        )
                except Exception as error:
                    if not failure:
                        raise CapabilityError(
                            "HTTP response evidence could not be committed; capture retained"
                        ) from error
                finally:
                    capture.close()
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
            if self.protocol == "auto" and status in {404, 405}:
                continue
            if status >= 300:
                raise ProtocolError(
                    f"HTTP {status}; request not retried; consult response evidence"
                )
            generation, measured = await normalize(
                capture.path,
                protocol,
                stream,
                content_type,
                content_encoding,
                self.store.directory / "traffic",
                self.config.runtime.memory_bytes,
            )
            self.last_evidence.insert(-1, self.store.evidence({"normalization": measured}))
            self.protocol = protocol
            generation.evidence = list(self.last_evidence)
            return generation
        raise CapabilityError("No compatible protocol on the supplied endpoint")


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


def content(generation: Generation, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        generation.text += value
    elif isinstance(value, list):
        for item in value:
            content(generation, item)
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


def decode_json(protocol: str, value: dict[str, Any]) -> Generation:
    if not isinstance(value, dict):
        raise ProtocolError("Provider response must be an object")
    if "error" in value:
        raise ProtocolError("Provider returned an error object")
    generation = Generation(model=value.get("model"))
    account(generation, value.get("usage") or {})
    if protocol == "openai-chat":
        choice = value["choices"][0]
        message = choice["message"]
        content(generation, message.get("content"))
        auxiliary(generation, message)
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
                content(generation, item.get("content", []))
                auxiliary(generation, item)
            elif item["type"] in {"reasoning", "reasoning_text", "summary_text"}:
                plaintext(generation, "reasoning", item)
            elif item["type"] == "refusal":
                plaintext(generation, "refusal", item)
    else:
        generation.finish = value.get("stop_reason")
        for item in value.get("content", []):
            if item["type"] == "tool_use":
                generation.tools.append(
                    ToolCall(id=item["id"], name=item["name"], arguments=json.dumps(item["input"]))
                )
            else:
                content(generation, item)
        auxiliary(generation, value)
    if not generation.finish:
        raise ProtocolError("Response missing termination reason")
    return generation


def same_tool(previous: dict[str, str], final: ToolCall) -> bool:
    if previous["name"] != final.name:
        return False
    try:
        return equal(loads(previous["arguments"]), loads(final.arguments))
    except ValueError:
        return previous["arguments"] == final.arguments


class Accumulator:
    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self.generation = Generation()
        self.calls: dict[int, dict[str, str]] = {}
        self.terminated = False
        self.stop_seen = False
        self.message_started = False
        self.blocks: dict[int, str] = {}
        self.closed_blocks: set[int] = set()
        self.signature_seen: set[int] = set()
        self.signature_nonempty: set[int] = set()

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
                        "reasoning_content",
                        "reasoning",
                        "refusal",
                    )
                ):
                    raise ProtocolError("Content after Chat Completions finish reason")
                content(self.generation, delta.get("content"))
                auxiliary(self.generation, delta)
                if "message" in choice:
                    final = Generation()
                    message = choice["message"]
                    content(final, message.get("content"))
                    auxiliary(final, message)
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
            if kind == "response.output_text.delta":
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
                content(self.generation, value["part"])
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
                content(self.generation, value["item"].get("content", []))
            elif kind == "response.function_call_arguments.delta":
                self.calls[value["output_index"]]["arguments"] += value["delta"]
            elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                final = decode_json(self.protocol, value["response"])
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
                self.generation.finish = final.finish
                identify(self.generation, final.model)
                account(self.generation, final.usage, cumulative=True)
                self.generation.observations.extend(final.observations)
                self.terminated = True
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

    def finish(self) -> Generation:
        if not self.terminated or not self.generation.finish:
            raise ProtocolError("Stream disconnected without terminal event and reason")
        self.generation.tools = [
            ToolCall.model_validate(
                value | {"arguments": value["arguments"] or "{}"}
                if self.protocol == "anthropic-messages"
                else value
            )
            for _, value in sorted(self.calls.items())
        ]
        return self.generation
