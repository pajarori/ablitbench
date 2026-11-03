from __future__ import annotations

import copy
import importlib
import json
import math
import uuid
from typing import Any, Literal
from urllib.parse import quote, urlsplit, urlunsplit

from ablitbench.formats.json import loads
from ablitbench.generation import text_content
from ablitbench.models import Generation, ProtocolObservation, ToolCall
from ablitbench.parity.multimodal import attachment

PROTOCOLS = (
    "openai-completions",
    "ollama-chat",
    "ollama-generate",
    "cohere-chat",
    "cohere-generate",
    "huggingface-inference",
    "huggingface-endpoint",
    "bedrock-converse",
    "watsonx-generation",
    "replicate",
    "replicate-deployment",
    "nvcf-chat",
    "nvcf-completions",
    "rasa",
    "langserve",
    "rest",
    "websocket",
)
COMMON = {
    "headers",
    "prompt_template",
    "input_mode",
    "extra_body",
    "suppressed_parameters",
    "verify_ssl",
    "proxy",
    "credential_origins",
    "connect_timeout",
    "request_timeout",
}
OPTIONS = {
    "openai-completions": set(),
    "ollama-chat": {"keep_alive", "format", "think", "logprobs", "top_logprobs"},
    "ollama-generate": {
        "keep_alive",
        "format",
        "raw",
        "template",
        "suffix",
        "think",
        "logprobs",
        "top_logprobs",
    },
    "cohere-chat": {"documents", "citation_options", "safety_mode", "strict_tools"},
    "cohere-generate": {"preset", "truncate", "likelihoods"},
    "huggingface-inference": {"details", "return_full_text", "wait_for_model", "use_cache"},
    "huggingface-endpoint": {
        "details",
        "return_full_text",
        "wait_for_model",
        "use_cache",
        "chat_payload",
    },
    "bedrock-converse": {
        "region",
        "additional_model_request_fields",
        "additional_model_response_field_paths",
        "guardrail_config",
        "request_metadata",
        "double_uri_encode",
    },
    "watsonx-generation": {"version", "project_id", "deployment_id", "prompt_variable", "iam_url"},
    "replicate": {"version", "input", "output_field", "poll_interval", "response_finish_field"},
    "replicate-deployment": {"input", "output_field", "poll_interval", "response_finish_field"},
    "nvcf-chat": {"version_id", "status_path", "poll_interval"},
    "nvcf-completions": {"version_id", "status_path", "poll_interval"},
    "rasa": {"sender", "response_json_field", "response_finish_field"},
    "langserve": {"config_hash", "config", "kwargs", "response_finish_field"},
    "rest": {
        "method",
        "request_template",
        "json_template",
        "response_json",
        "response_json_field",
        "response_finish_field",
        "skip_codes",
        "ratelimit_codes",
        "escape_function",
    },
    "websocket": {
        "request_template",
        "json_template",
        "response_json",
        "response_json_field",
        "response_finish_field",
        "auth_type",
        "typing_indicator",
        "response_after_typing",
        "conversation_id",
        "termination_field",
        "termination_value",
        "max_response_length",
        "subprotocols",
    },
}
SAMPLING = {
    "openai-completions": {
        "temperature",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "best_of",
        "echo",
        "suffix",
    },
    "ollama-chat": {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "repeat_last_n",
        "stop",
        "num_ctx",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "low_vram",
        "presence_penalty",
        "frequency_penalty",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "penalize_newline",
    },
    "cohere-chat": {
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "tool_choice",
    },
    "cohere-generate": {
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "frequency_penalty",
        "presence_penalty",
    },
    "huggingface-inference": {
        "temperature",
        "top_p",
        "top_k",
        "do_sample",
        "repetition_penalty",
        "stop",
        "typical_p",
        "watermark",
        "grammar",
        "top_n_tokens",
        "decoder_input_details",
    },
    "bedrock-converse": {"temperature", "top_p", "stop"},
    "watsonx-generation": {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "stop",
        "decoding_method",
        "min_new_tokens",
        "length_penalty",
        "include_stop_sequence",
    },
    "replicate": {"temperature", "top_p", "top_k", "repetition_penalty", "stop"},
    "nvcf-chat": {
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
    },
    "rest": set(),
    "websocket": set(),
    "rasa": set(),
    "langserve": set(),
}
SAMPLING["ollama-generate"] = SAMPLING["ollama-chat"]
SAMPLING["huggingface-endpoint"] = SAMPLING["huggingface-inference"]
SAMPLING["replicate-deployment"] = SAMPLING["replicate"]
SAMPLING["nvcf-completions"] = SAMPLING["nvcf-chat"]


def protocols() -> tuple[str, ...]:
    return PROTOCOLS


def configuration(protocol: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown native provider protocol")
    options = dict(options or {})
    unknown = set(options) - COMMON - OPTIONS[protocol]
    if unknown:
        raise ValueError("Unknown provider options: " + ", ".join(sorted(unknown)))
    pending: list[Any] = [options]
    forbidden = {
        "api_key",
        "authorization",
        "password",
        "secret",
        "access_token",
        "bearer_token",
        "secret_access_key",
        "session_token",
        "client_secret",
    }
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if str(key).casefold() in forbidden and not (
                    isinstance(value, str) and "$KEY" in value
                ):
                    raise ValueError("Provider credentials require host environment bindings")
                pending.append(value)
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Provider settings must be finite")
        elif item is not None and not isinstance(item, str | int | float | bool):
            raise ValueError("Provider settings require JSON values")
    for name in ("iam_url", "proxy"):
        route = options.get(name)
        if route is not None:
            if not isinstance(route, str):
                raise ValueError("Provider route requires a URL string")
            parsed = urlsplit(route)
            if (
                parsed.scheme not in {"http", "https", "socks5", "socks5h"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
                or parsed.query
            ):
                raise ValueError("Provider route must exclude literal credentials/query/fragment")
    for name in ("request_timeout", "connect_timeout"):
        value = options.get(name)
        if value is not None and (
            not isinstance(value, int | float) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError("Explicit provider timeout must be positive")
    headers = options.get("headers", {})
    if not isinstance(headers, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or "\r" in key + value
        or "\n" in key + value
        or key.casefold() in {"host", "content-length", "transfer-encoding", "connection"}
        for key, value in headers.items()
    ):
        raise ValueError("Malformed or transport-controlled provider headers")
    for key, value in headers.items():
        if (
            any(
                part in key.casefold()
                for part in (
                    "authorization",
                    "api-key",
                    "apikey",
                    "token",
                    "cookie",
                    "secret",
                    "password",
                )
            )
            and "$KEY" not in value
        ):
            raise ValueError("Authentication header requires an environment credential placeholder")
    if options.get("input_mode", "strict") not in {"strict", "last"}:
        raise ValueError("Provider input_mode must be strict or explicitly last")
    suppressed = options.get("suppressed_parameters", [])
    if not isinstance(suppressed, list) or any(not isinstance(key, str) for key in suppressed):
        raise ValueError("Suppressed native parameters must be a list")
    if set(suppressed) & {"max_tokens", "max_output_tokens", "num_predict", "max_new_tokens"}:
        raise ValueError("Explicit output budget cannot be silently suppressed")
    if protocol in {"replicate", "replicate-deployment", "nvcf-chat", "nvcf-completions"}:
        interval = options.get("poll_interval")
        if not isinstance(interval, int | float) or isinstance(interval, bool) or interval < 0:
            raise ValueError("Explicit nonnegative asynchronous provider poll interval required")
    if protocol == "bedrock-converse" and not options.get("region"):
        raise ValueError("Bedrock requires explicit signing region")
    if protocol == "watsonx-generation" and (
        not isinstance(options.get("version"), str)
        or not options["version"]
        or bool(options.get("project_id")) == bool(options.get("deployment_id"))
    ):
        raise ValueError("Watsonx requires API version and exactly one project/deployment identity")
    if protocol in {"rest", "websocket"}:
        if "json_template" not in options and not isinstance(options.get("request_template"), str):
            raise ValueError("Explicit native request template required")
        if options.get("response_json", True) and not options.get("response_json_field"):
            raise ValueError("JSON response requires explicit output field or JSONPath")
    return options


def endpoint(base: str, path: str) -> str:
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https", "ws", "wss"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Provider endpoint requires an explicit origin without URL credentials")
    if urlsplit(path).scheme or path.startswith("//") or ".." in path.split("/"):
        raise ValueError("Provider request path cannot change origin")
    current = parsed.path.rstrip("/")
    suffix = path.strip("/")
    left = current.strip("/").split("/") if current else []
    right = suffix.split("/") if suffix else []
    overlap = next(
        (
            count
            for count in range(min(len(left), len(right)), 0, -1)
            if left[-count:] == right[:count]
        ),
        0,
    )
    target = "/" + "/".join(left + right[overlap:])
    return urlunsplit((parsed.scheme, parsed.netloc, target or "/", parsed.query, ""))


def legacy_input(messages: list[dict[str, Any]], options: dict[str, Any]) -> str:
    texts = [{"role": row["role"], "content": text_content(row.get("content"))} for row in messages]
    if any(row.get("tool_calls") or row["role"] == "tool" for row in messages):
        raise ValueError("Native prompt-only endpoint has no structured tool dialogue")
    user = [row["content"] for row in texts if row["role"] == "user"]
    system = [row["content"] for row in texts if row["role"] in {"system", "developer"}]
    template = options.get("prompt_template")
    if template is not None:
        if not isinstance(template, str) or (
            system and "$SYSTEM" not in template and "$MESSAGES" not in template
        ):
            raise ValueError("Native prompt template does not encode supplied system messages")
        if len(user) != 1 and "$MESSAGES" not in template:
            raise ValueError("Native prompt template does not encode complete conversation")
        return (
            template.replace("$SYSTEM", "\n".join(system))
            .replace("$INPUT", user[-1] if user else "")
            .replace("$MESSAGES", json.dumps(texts, ensure_ascii=False))
        )
    if options.get("input_mode") == "last":
        return str(texts[-1]["content"]) if texts else ""
    if len(texts) != 1 or texts[0]["role"] != "user":
        raise ValueError("Prompt-only endpoint requires explicit template for system/history")
    return str(texts[0]["content"])


def normalized_text(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "content": text_content(row.get("content"))} for row in messages]


def ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    calls = {}
    for row in messages:
        value = {**row}
        content = value.get("content")
        if isinstance(content, list):
            text, images = [], []
            for block in content:
                if block.get("type") == "text":
                    text.append(text_content([block]))
                else:
                    if block.get("complete") is False:
                        raise ValueError("Incomplete media cannot become an Ollama input")
                    mime, _, _ = attachment(block)
                    if not mime.startswith("image/"):
                        raise ValueError("Ollama native input adapter supports images and text")
                    images.append(block["data"])
            value["content"] = "\n".join(text)
            if images:
                value["images"] = images
        elif content is not None and not isinstance(content, str):
            raise ValueError("Malformed native Ollama text")
        if value.get("tool_calls"):
            converted = []
            for call in value["tool_calls"]:
                function = dict(call["function"])
                calls[call["id"]] = function["name"]
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = loads(function["arguments"])
                converted.append({"type": "function", "function": function})
            value["tool_calls"] = converted
        if row["role"] == "tool":
            identity = value.pop("tool_call_id", None)
            if identity not in calls:
                raise ValueError("Ollama tool result has no preceding structured call")
            value["tool_name"] = calls[identity]
        result.append(value)
    return result


def bedrock_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        raise ValueError("Bedrock content requires normalized blocks")
    result: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            result.append({"text": text_content([block])})
            continue
        if block.get("complete") is False:
            raise ValueError("Incomplete media cannot become Bedrock input")
        mime, _, _ = attachment(block)
        kind = block["type"]
        formats = {
            "image/jpeg": "jpeg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
            "application/pdf": "pdf",
            "video/mp4": "mp4",
            "video/webm": "webm",
            "video/quicktime": "mov",
        }
        if mime not in formats or kind not in {"image", "document", "video"}:
            raise ValueError("Unsupported Bedrock native input media contract")
        value = {"format": formats[mime], "source": {"bytes": block["data"]}}
        if kind == "document":
            value["name"] = block.get("filename", "input document")
        result.append({kind: value})
    return result


def bedrock_messages(messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    result, system = [], []
    for row in messages:
        role = row["role"]
        if role in {"system", "developer"}:
            system.append({"text": text_content(row.get("content"))})
            continue
        content = bedrock_blocks(row.get("content"))
        if role == "tool":
            content = [{"toolResult": {"toolUseId": row["tool_call_id"], "content": content}}]
            role = "user"
        for call in row.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            content.append(
                {
                    "toolUse": {
                        "toolUseId": call["id"],
                        "name": call["function"]["name"],
                        "input": loads(arguments) if isinstance(arguments, str) else arguments,
                    }
                }
            )
        if role not in {"user", "assistant"}:
            raise ValueError("Unknown Bedrock conversation role")
        result.append({"role": role, "content": content})
    return result, system


def populate(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        for key, item in replacements.items():
            value = value.replace(key, str(item))
        return value
    if isinstance(value, list):
        return [populate(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: populate(item, replacements) for key, item in value.items()}
    return value


def encode(
    protocol: str,
    model: str | None,
    messages: list[dict[str, Any]],
    max_output_tokens: int | None,
    generation_settings: dict[str, Any],
    options: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    seed: int | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    options = configuration(protocol, options)
    settings = dict(generation_settings)
    suppressed = set(options.get("suppressed_parameters", []))
    unknown = set(settings) - SAMPLING[protocol]
    if unknown:
        raise ValueError("Unsupported native generation settings: " + ", ".join(sorted(unknown)))
    settings = {key: value for key, value in settings.items() if key not in suppressed}
    if max_output_tokens is not None and (
        type(max_output_tokens) is not int or max_output_tokens <= 0
    ):
        raise ValueError("Output budget must be positive")
    budgeted = protocol not in {"rasa", "langserve", "rest", "websocket"}
    if budgeted and max_output_tokens is None:
        raise ValueError("Native provider requires an explicit output token budget")
    if tools and protocol not in {"ollama-chat", "cohere-chat", "bedrock-converse", "nvcf-chat"}:
        raise ValueError("Selected native provider has no structured tool adapter")
    streaming = {
        "openai-completions": "sse",
        "ollama-chat": "ndjson",
        "ollama-generate": "ndjson",
        "cohere-chat": "sse",
        "huggingface-inference": "sse",
        "huggingface-endpoint": "sse",
        "bedrock-converse": "eventstream",
    }
    if stream and protocol not in streaming and protocol != "websocket":
        raise ValueError("Selected native provider has no streaming contract")
    spec: dict[str, Any] = {
        "method": "POST",
        "path": "",
        "params": {},
        "headers": {"content-type": "application/json", **options.get("headers", {})},
        "stream_format": streaming.get(protocol) if stream else "json",
        "coverage": [],
    }
    if options.get("input_mode") == "last":
        spec["coverage"].append(
            "Operator selected native last-message input; preceding dialogue is not encoded"
        )
    if suppressed:
        spec["coverage"].append(
            "Operator suppressed native parameters: " + ", ".join(sorted(suppressed))
        )
    body: dict[str, Any] = {}
    if protocol == "openai-completions":
        spec["path"] = "completions"
        body = {
            "model": model,
            "prompt": legacy_input(messages, options),
            "max_tokens": max_output_tokens,
            "stream": stream,
            **settings,
        }
        if seed is not None:
            body["seed"] = seed
    elif protocol.startswith("ollama-"):
        chat = protocol == "ollama-chat"
        spec["path"] = "api/chat" if chat else "api/generate"
        body = {
            "model": model,
            "stream": stream,
            "options": {"num_predict": max_output_tokens, **settings},
        }
        if seed is not None:
            body["options"]["seed"] = seed
        if chat:
            body["messages"] = ollama_messages(messages)
            if tools:
                body["tools"] = tools
        else:
            body["prompt"] = legacy_input(messages, options)
        for key in OPTIONS[protocol]:
            if key in options:
                body[key] = options[key]
    elif protocol.startswith("cohere-"):
        chat = protocol == "cohere-chat"
        spec["path"] = "v2/chat" if chat else "v1/generate"
        names = {"top_p": "p", "top_k": "k", "stop": "stop_sequences" if chat else "end_sequences"}
        body = {
            "model": model,
            "max_tokens": max_output_tokens,
            **{names.get(key, key): value for key, value in settings.items()},
        }
        if chat:
            body.update({"messages": normalized_text(messages), "stream": stream})
            if tools:
                body["tools"] = tools
        else:
            body["prompt"] = legacy_input(messages, options)
        for key in OPTIONS[protocol]:
            if key in options:
                body[key] = options[key]
        if seed is not None:
            body["seed"] = seed
    elif protocol.startswith("huggingface-"):
        spec["path"] = (
            "models/" + quote(model or "", safe="/") if protocol == "huggingface-inference" else ""
        )
        parameters = {
            "max_new_tokens": max_output_tokens,
            "return_full_text": options.get("return_full_text", False),
            "details": options.get("details", True),
            **settings,
        }
        if seed is not None:
            parameters["seed"] = seed
        body = {
            "parameters": parameters,
            "stream": stream,
            "options": {
                key: options[key] for key in ("wait_for_model", "use_cache") if key in options
            },
        }
        if options.get("chat_payload"):
            body["messages"] = normalized_text(messages)
        else:
            body["inputs"] = legacy_input(messages, options)
    elif protocol == "bedrock-converse":
        spec["path"] = (
            "model/" + quote(model or "", safe="") + ("/converse-stream" if stream else "/converse")
        )
        conversation, system = bedrock_messages(messages)
        inference = {
            "maxTokens": max_output_tokens,
            **{
                {"top_p": "topP", "stop": "stopSequences"}.get(key, key): (
                    [value] if key == "stop" and isinstance(value, str) else value
                )
                for key, value in settings.items()
            },
        }
        body = {"messages": conversation, "inferenceConfig": inference}
        if system:
            body["system"] = system
        if tools:
            body["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "inputSchema": {"json": tool["function"]["parameters"]},
                        }
                    }
                    for tool in tools
                ]
            }
        for key, destination in {
            "additional_model_request_fields": "additionalModelRequestFields",
            "additional_model_response_field_paths": "additionalModelResponseFieldPaths",
            "guardrail_config": "guardrailConfig",
            "request_metadata": "requestMetadata",
        }.items():
            if key in options:
                body[destination] = options[key]
        if seed is not None:
            raise ValueError("Bedrock Converse has no universal seed field")
    elif protocol == "watsonx-generation":
        spec["params"]["version"] = options["version"]
        deployment = options.get("deployment_id")
        spec["path"] = (
            "ml/v1/deployments/" + quote(deployment, safe="") + "/text/generation"
            if deployment
            else "ml/v1/text/generation"
        )
        parameters = {
            "max_new_tokens": max_output_tokens,
            **{{"stop": "stop_sequences"}.get(key, key): value for key, value in settings.items()},
        }
        if seed is not None:
            parameters["random_seed"] = seed
        prompt = legacy_input(messages, options)
        if deployment:
            parameters["prompt_variables"] = {options.get("prompt_variable", "input"): prompt}
            body = {"parameters": parameters}
        else:
            body = {
                "input": prompt,
                "parameters": parameters,
                "model_id": model,
                "project_id": options["project_id"],
            }
    elif protocol.startswith("replicate"):
        extra = options.get("input", {})
        if not isinstance(extra, dict) or set(extra) & {"prompt", "max_length", "seed"}:
            raise ValueError("Replicate model inputs cannot override prompt/budget/seed")
        parameters = {
            "prompt": legacy_input(messages, options),
            "max_length": max_output_tokens,
            **settings,
            **extra,
        }
        if seed is not None:
            parameters["seed"] = seed
        if protocol == "replicate-deployment":
            spec["path"] = "v1/deployments/" + quote(model or "", safe="/") + "/predictions"
            body = {"input": parameters}
        else:
            version = options.get("version")
            if version:
                spec["path"] = "v1/predictions"
                body = {"version": version, "input": parameters}
            else:
                spec["path"] = "v1/models/" + quote(model or "", safe="/") + "/predictions"
                body = {"input": parameters}
    elif protocol.startswith("nvcf-"):
        spec["path"] = "v2/nvcf/pexec/functions/" + quote(model or "", safe="")
        if options.get("version_id"):
            spec["path"] += "/versions/" + quote(options["version_id"], safe="")
        body = {"max_tokens": max_output_tokens, "stream": False, **settings}
        if protocol == "nvcf-chat":
            body["messages"] = normalized_text(messages)
            if tools:
                body["tools"] = tools
        else:
            body["prompt"] = legacy_input(messages, options)
        if seed is not None:
            body["seed"] = seed
    elif protocol == "rasa":
        spec["path"] = "webhooks/rest/webhook"
        body = {
            "sender": options.get("sender", "ablitbench"),
            "message": legacy_input(messages, options),
        }
    elif protocol == "langserve":
        spec["path"] = "invoke"
        if options.get("config_hash"):
            spec["params"]["config_hash"] = options["config_hash"]
        body = {
            "input": legacy_input(messages, options),
            "config": options.get("config", {}),
            "kwargs": options.get("kwargs", {}),
        }
    else:
        prompt = legacy_input(messages, options)
        replacement = {
            "$CONVERSATIONID": options.get("conversation_id", ""),
            "$INPUT": prompt,
            "$MODEL": model,
            "$MAX_TOKENS": max_output_tokens,
            "$SEED": seed,
        }
        if "json_template" in options:
            body = populate(copy.deepcopy(options["json_template"]), replacement)
        else:
            template = options["request_template"]
            escape = options.get("escape_function", "json")
            if escape not in {"json", "none"}:
                raise ValueError("Unknown native request escaping contract")
            escaped = json.dumps(prompt, ensure_ascii=False)[1:-1] if escape == "json" else prompt
            spec["content"] = (
                template.replace("$INPUT", escaped)
                .replace("$MODEL", model or "")
                .replace("$MAX_TOKENS", str(max_output_tokens))
                .replace("$SEED", str(seed))
                .replace("$CONVERSATIONID", str(options.get("conversation_id", "")))
            )
        spec["headers"] = populate(spec["headers"], {"$INPUT": prompt, "$MODEL": model or ""})
        if protocol == "rest":
            method = options.get("method", "POST").upper()
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
                raise ValueError("Unsupported native HTTP method")
            spec["method"] = method
        else:
            spec["stream_format"] = "websocket"
        if max_output_tokens is not None and "$MAX_TOKENS" not in json.dumps(
            options.get("json_template", options.get("request_template"))
        ):
            spec["coverage"].append("Native template does not encode configured token budget")
    if seed is not None and protocol in {"rasa", "langserve"}:
        raise ValueError("Selected native provider has no seed contract")
    extra_body = options.get("extra_body", {})
    if (
        not isinstance(extra_body, dict)
        or (isinstance(body, dict) and set(extra_body) & set(body))
        or (not isinstance(body, dict) and extra_body)
    ):
        raise ValueError("Provider extra_body cannot override encoded generation fields")
    if body or "content" not in spec:
        spec["json"] = {**body, **extra_body} if isinstance(body, dict) else body
    if protocol == "rest" and spec["method"] in {"GET", "HEAD"}:
        value = spec.pop("json", None)
        if value is None:
            value = loads(spec.pop("content"))
        if not isinstance(value, dict):
            raise ValueError("Native GET template requires query parameter mapping")
        spec["params"].update(value)
    return spec


def observation(
    generation: Generation,
    code: str,
    detail: str,
    severity: Literal["info", "low", "medium", "high"] = "info",
) -> None:
    generation.observations.append(ProtocolObservation(code=code, detail=detail, severity=severity))


def finish(generation: Generation, reason: Any) -> None:
    mapping = {
        "COMPLETE": "stop",
        "STOP_SEQUENCE": "stop",
        "MAX_TOKENS": "length",
        "MAX_TOKEN": "length",
        "TOKEN_LIMIT": "length",
        "TOOL_CALL": "tool_calls",
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "eos_token": "stop",
        "token_limit": "length",
        "time_limit": "incomplete",
        "error": "failed",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
        "model_context_window_exceeded": "incomplete",
        "malformed_model_output": "failed",
        "stop": "stop",
        "completed": "stop",
        "succeeded": "stop",
    }
    if reason is None:
        generation.finish = "unverified"
        observation(
            generation,
            "termination-unverified",
            "Native response has no model termination reason",
            "medium",
        )
    elif not isinstance(reason, str) or not reason:
        raise ValueError("Malformed native termination reason")
    else:
        incomplete = {
            "length",
            "max_tokens",
            "incomplete",
            "failed",
            "cancelled",
            "canceled",
            "content_filter",
            "error",
        }
        mapped = mapping.get(reason, reason)
        if (
            mapped
            not in {"stop", "completed", "tool_calls", "tool_use", "function_call"} | incomplete
        ):
            generation.finish = "unverified"
            observation(
                generation,
                "native-termination-unknown",
                "Unrecognized native termination reason: " + reason,
                "medium",
            )
        else:
            generation.finish = mapped


def selected(value: Any, field: str) -> list[str]:
    if not isinstance(field, str) or not field:
        raise ValueError("Native response extraction requires explicit field")
    if field.startswith("$"):
        parser = importlib.import_module("jsonpath_ng")
        values = [match.value for match in parser.parse(field).find(value)]
    elif isinstance(value, list):
        values = [row[field] for row in value if isinstance(row, dict) and field in row]
        if len(values) != len(value):
            raise ValueError("Native response field missing from some rows")
    else:
        values = [value[field]]
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not values or any(not isinstance(item, str) for item in values):
        raise ValueError("Native response selector did not match text")
    return values


def output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    raise ValueError("Native output is not textual; no stringification or external media fetch")


def tool_calls(generation: Generation, values: Any, protocol: str) -> None:
    if not isinstance(values, list):
        raise ValueError("Malformed native tool calls")
    correlation = "ollama:" + uuid.uuid4().hex + ":"
    for index, call in enumerate(values):
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Native tool call lacks function identity")
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        identity = call.get("id")
        if identity is None and protocol == "ollama-chat":
            identity = correlation + str(index)
            observation(
                generation,
                "provider-tool-correlation",
                "Ollama has no native tool-call ID; transport correlation ID is synthetic",
            )
        if not isinstance(identity, str) or not identity:
            raise ValueError("Native tool call has no correlation identity")
        generation.tools.append(ToolCall(id=identity, name=function["name"], arguments=arguments))


def decode(protocol: str, value: Any, options: dict[str, Any] | None = None) -> Generation:
    options = dict(options or {})
    generation = Generation()
    if isinstance(value, dict) and value.get("error"):
        raise ValueError("Native provider returned an error object")
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown native provider decoder")
    if protocol == "openai-completions":
        choice = value["choices"][0]
        generation.text = output_text(choice["text"])
        generation.model = value.get("model")
        generation.usage = value.get("usage", {})
        finish(generation, choice.get("finish_reason"))
    elif protocol.startswith("ollama-"):
        if value.get("done") is not True:
            raise ValueError("Ollama response has no completed terminal record")
        message = value.get("message", {}) if protocol == "ollama-chat" else {}
        generation.text = output_text(
            message.get("content", "") if message else value.get("response", "")
        )
        generation.reasoning = output_text(message.get("thinking", value.get("thinking", "")))
        if message.get("tool_calls"):
            tool_calls(generation, message["tool_calls"], protocol)
        generation.model = value.get("model")
        generation.usage = {
            "input_tokens": value.get("prompt_eval_count"),
            "output_tokens": value.get("eval_count"),
            "total_duration": value.get("total_duration"),
            "load_duration": value.get("load_duration"),
        }
        finish(generation, value.get("done_reason"))
    elif protocol == "cohere-chat":
        message = value["message"]
        content = message.get("content", [])
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                generation.text += output_text(block["text"])
            elif kind == "thinking":
                generation.reasoning += output_text(block["thinking"])
            else:
                raise ValueError("Unsupported native Cohere output content")
        if message.get("tool_calls"):
            tool_calls(generation, message["tool_calls"], protocol)
        if message.get("tool_plan"):
            generation.reasoning += output_text(message["tool_plan"])
            observation(
                generation, "provider-tool-plan", "Native Cohere tool plan is labeled as reasoning"
            )
        generation.usage = value.get("usage", {}).get("tokens", {})
        finish(generation, value.get("finish_reason"))
    elif protocol == "cohere-generate":
        rows = value["generations"]
        if len(rows) != 1:
            raise ValueError("Generation contract requires one Cohere completion per request")
        generation.text = output_text(rows[0]["text"])
        generation.usage = value.get("meta", {}).get("tokens", {})
        finish(generation, rows[0].get("finish_reason"))
    elif protocol.startswith("huggingface-"):
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Generation contract requires one HF output per request")
            value = value[0]
        generation.text = output_text(value["generated_text"])
        details = value.get("details") or {}
        generation.usage = {"output_tokens": details.get("generated_tokens")}
        finish(generation, details.get("finish_reason"))
    elif protocol == "bedrock-converse":
        message = value["output"]["message"]
        for block in message["content"]:
            if "text" in block:
                generation.text += output_text(block["text"])
            elif "toolUse" in block:
                call = block["toolUse"]
                generation.tools.append(
                    ToolCall(
                        id=call["toolUseId"],
                        name=call["name"],
                        arguments=json.dumps(call["input"], ensure_ascii=False, allow_nan=False),
                    )
                )
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                if reasoning.get("reasoningText"):
                    generation.reasoning += output_text(reasoning["reasoningText"]["text"])
                if reasoning.get("redactedContent"):
                    observation(
                        generation,
                        "opaque-reasoning",
                        "Bedrock returned opaque reasoning bytes",
                        "medium",
                    )
            elif "guardContent" in block:
                generation.refusal += output_text(
                    block["guardContent"].get("text", {}).get("text", "")
                )
            else:
                raise ValueError("Unsupported native Bedrock output block")
        usage = value.get("usage", {})
        generation.usage = {
            **usage,
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
        }
        finish(generation, value.get("stopReason"))
    elif protocol == "watsonx-generation":
        rows = value["results"]
        if len(rows) != 1:
            raise ValueError("Expected one Watsonx generation result")
        row = rows[0]
        generation.text = output_text(row["generated_text"])
        generation.model = value.get("model_id")
        generation.usage = {
            "input_tokens": row.get("input_token_count"),
            "output_tokens": row.get("generated_token_count"),
        }
        finish(generation, row.get("stop_reason"))
    elif protocol.startswith("replicate"):
        if value.get("status") != "succeeded":
            raise ValueError("Replicate prediction not successfully terminal")
        output = value["output"]
        if options.get("output_field"):
            generation.text = "".join(selected(output, options["output_field"]))
        else:
            generation.text = output_text(output)
        generation.model = value.get("model") or value.get("version")
        generation.usage = value.get("metrics", {})
        reason = (
            selected(value, options["response_finish_field"])[0]
            if options.get("response_finish_field")
            else None
        )
        finish(generation, reason)
        observation(
            generation,
            "prediction-completion",
            "Prediction succeeded; native output token termination details may be unavailable",
        )
    elif protocol.startswith("nvcf-"):
        from ablitbench.transport import decode_json

        return (
            decode_json("openai-chat", value)
            if protocol == "nvcf-chat"
            else decode("openai-completions", value)
        )
    elif protocol == "langserve":
        result = value["output"]
        if isinstance(result, dict):
            result = result.get("content", result.get("kwargs", {}).get("content"))
        generation.text = output_text(result)
        finish(
            generation,
            value.get(options["response_finish_field"])
            if options.get("response_finish_field")
            else None,
        )
    else:
        if protocol == "rasa":
            generation.text = "\n".join(selected(value, options.get("response_json_field", "text")))
        elif options.get("response_json", True):
            generation.text = "\n".join(selected(value, options["response_json_field"]))
        else:
            generation.text = output_text(value)
        finish(
            generation,
            value.get(options["response_finish_field"])
            if options.get("response_finish_field") and isinstance(value, dict)
            else None,
        )
    return generation


class Accumulator:
    def __init__(self, protocol: str):
        from ablitbench.transport import Accumulator as SharedAccumulator

        if protocol not in {
            "openai-completions",
            "ollama-chat",
            "ollama-generate",
            "cohere-chat",
            "huggingface-inference",
            "huggingface-endpoint",
            "bedrock-converse",
        }:
            raise ValueError("Native provider has no implemented stream decoder")
        self.protocol = protocol
        self.shared = SharedAccumulator("openai-chat")
        self.completed = False
        self.tool_index = 0
        self.correlation = "ollama:" + uuid.uuid4().hex + ":"
        self.bedrock_started = False
        self.bedrock_reason: str | None = None
        self.bedrock_blocks: set[int] = set()

    def push(
        self,
        delta: dict[str, Any],
        reason: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        packet: dict[str, Any] = {
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]
        }
        if model is not None:
            packet["model"] = model
        if usage is not None:
            packet["usage"] = usage
        self.shared.accept("message", json.dumps(packet, ensure_ascii=False, allow_nan=False))

    def terminal(
        self, reason: Any, model: str | None = None, usage: dict[str, Any] | None = None
    ) -> None:
        generation = Generation()
        finish(generation, reason)
        self.shared.generation.observations.extend(generation.observations)
        self.push({}, generation.finish, model, usage)
        self.shared.accept("message", "[DONE]")
        self.completed = True

    def accept(self, event: str, data: str) -> None:
        self.feed(event, data)

    def feed(self, event: str, data: str) -> None:
        if self.completed:
            if data != "[DONE]":
                raise ValueError("Native stream emitted content after terminal record")
            return
        if data == "[DONE]":
            if self.protocol != "openai-completions":
                raise ValueError("Native stream ended without a complete generation record")
            self.shared.accept(event, data)
            self.completed = True
            return
        value = loads(data)
        if not isinstance(value, dict) or value.get("error"):
            raise ValueError("Native stream returned malformed/error event")
        if self.protocol == "openai-completions":
            choices = value.get("choices", [])
            if choices:
                choice = choices[0]
                self.push(
                    {"content": output_text(choice.get("text", ""))},
                    choice.get("finish_reason"),
                    value.get("model"),
                    value.get("usage"),
                )
            elif value.get("usage"):
                self.push({}, usage=value["usage"])
            else:
                raise ValueError("Legacy completion stream has no choice or usage")
        elif self.protocol.startswith("ollama-"):
            message = value.get("message") or {}
            delta = {"content": message.get("content", value.get("response", ""))}
            thinking = message.get("thinking", value.get("thinking"))
            if thinking is not None:
                delta["reasoning_content"] = thinking
            if message.get("tool_calls"):
                calls = []
                for offset, call in enumerate(message["tool_calls"]):
                    function = call["function"]
                    index = function.get("index", self.tool_index + offset)
                    arguments = function.get("arguments", {})
                    calls.append(
                        {
                            "index": index,
                            "id": call.get("id", self.correlation + str(index)),
                            "function": {
                                "name": function["name"],
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, ensure_ascii=False, allow_nan=False),
                            },
                        }
                    )
                    if not call.get("id"):
                        observation(
                            self.shared.generation,
                            "provider-tool-correlation",
                            "Ollama stream tool correlation is synthetic",
                        )
                self.tool_index += len(calls)
                delta["tool_calls"] = calls
            self.push(delta, model=value.get("model"))
            if value.get("done") is True:
                self.terminal(
                    value.get("done_reason"),
                    value.get("model"),
                    {
                        "prompt_tokens": value.get("prompt_eval_count"),
                        "completion_tokens": value.get("eval_count"),
                    },
                )
            elif value.get("done") not in {False, None}:
                raise ValueError("Malformed Ollama completion marker")
        elif self.protocol == "bedrock-converse":
            kind = event
            if kind == "messageStart":
                if self.bedrock_started or value.get("role") != "assistant":
                    raise ValueError("Malformed Bedrock stream message start")
                self.bedrock_started = True
            elif not self.bedrock_started:
                raise ValueError("Bedrock event arrived before message start")
            elif kind == "contentBlockStart":
                index = value["contentBlockIndex"]
                if type(index) is not int or index < 0 or index in self.bedrock_blocks:
                    raise ValueError("Invalid/duplicate Bedrock block index")
                self.bedrock_blocks.add(index)
                call = value["start"].get("toolUse")
                if call is None:
                    raise ValueError("Unknown Bedrock content start")
                self.push(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call["toolUseId"],
                                "function": {"name": call["name"], "arguments": ""},
                            }
                        ]
                    }
                )
            elif kind == "contentBlockDelta":
                index, delta = value["contentBlockIndex"], value["delta"]
                if type(index) is not int or index < 0:
                    raise ValueError("Malformed Bedrock block index")
                self.bedrock_blocks.add(index)
                if "text" in delta:
                    self.push({"content": output_text(delta["text"])})
                elif "toolUse" in delta:
                    if index not in self.shared.calls:
                        raise ValueError("Bedrock tool delta has no call start")
                    self.push(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "function": {
                                        "arguments": output_text(delta["toolUse"]["input"])
                                    },
                                }
                            ]
                        }
                    )
                elif "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"]
                    if reasoning.get("text") is not None:
                        self.push({"reasoning_content": output_text(reasoning["text"])})
                    if reasoning.get("signature") is not None:
                        self.shared.signature(index, reasoning["signature"])
                    if reasoning.get("redactedContent"):
                        observation(
                            self.shared.generation,
                            "opaque-reasoning",
                            "Bedrock returned redacted reasoning",
                            "medium",
                        )
                else:
                    raise ValueError("Unknown Bedrock content delta")
            elif kind == "contentBlockStop":
                index = value["contentBlockIndex"]
                if index not in self.bedrock_blocks:
                    raise ValueError("Bedrock block stop has no active content")
                self.bedrock_blocks.remove(index)
            elif kind == "messageStop":
                if self.bedrock_blocks or self.bedrock_reason is not None:
                    raise ValueError("Bedrock message ended with active blocks or duplicate stop")
                self.bedrock_reason = value["stopReason"]
            elif kind == "metadata":
                if self.bedrock_reason is None:
                    raise ValueError("Bedrock metadata arrived before message stop")
                usage = value.get("usage") or {}
                self.terminal(
                    self.bedrock_reason,
                    usage={
                        "prompt_tokens": usage.get("inputTokens"),
                        "completion_tokens": usage.get("outputTokens"),
                        "total_tokens": usage.get("totalTokens"),
                    },
                )
            else:
                raise ValueError("Unsupported native Bedrock streaming event")
        elif self.protocol == "cohere-chat":
            kind = value.get("type", event)
            delta = value.get("delta") or {}
            message = delta.get("message") or {}
            if kind == "content-delta":
                block = message.get("content")
                if not isinstance(block, dict):
                    raise ValueError("Malformed Cohere content delta")
                if "text" in block:
                    self.push({"content": output_text(block["text"])})
                elif "thinking" in block:
                    self.push({"reasoning_content": output_text(block["thinking"])})
                else:
                    raise ValueError("Unrecognized Cohere content delta")
            elif kind == "tool-plan-delta":
                self.push({"reasoning_content": output_text(message["tool_plan"])})
                observation(
                    self.shared.generation,
                    "provider-tool-plan",
                    "Native Cohere tool plan is labeled as reasoning",
                )
            elif kind in {"tool-call-start", "tool-call-delta"}:
                call = message.get("tool_calls")
                if not isinstance(call, dict):
                    raise ValueError("Malformed Cohere tool call delta")
                function = call.get("function") or {}
                mapped: dict[str, Any] = {"index": value["index"], "function": function}
                if call.get("id") is not None:
                    mapped["id"] = call["id"]
                self.push({"tool_calls": [mapped]})
            elif kind == "message-end":
                usage = (delta.get("usage") or {}).get("tokens", {})
                self.terminal(
                    delta.get("finish_reason"),
                    usage={
                        "prompt_tokens": usage.get("input_tokens"),
                        "completion_tokens": usage.get("output_tokens"),
                    },
                )
            elif kind in {
                "message-start",
                "content-start",
                "content-end",
                "tool-call-end",
                "citation-start",
                "citation-end",
            }:
                if kind.startswith("citation"):
                    observation(
                        self.shared.generation,
                        "provider-citation",
                        "Native citation event retained in response evidence",
                    )
            else:
                raise ValueError("Unsupported native Cohere streaming event")
        else:
            token = value.get("token")
            if token is not None:
                if not isinstance(token, dict) or not isinstance(token.get("text"), str):
                    raise ValueError("Malformed HF token stream event")
                if not token.get("special", False):
                    self.push({"content": token["text"]})
            if value.get("generated_text") is not None:
                from ablitbench.transport import final_channel

                final_channel(self.shared.generation, "text", output_text(value["generated_text"]))
                details = value.get("details") or {}
                self.terminal(
                    details.get("finish_reason"),
                    usage={"completion_tokens": details.get("generated_tokens")},
                )
            elif token is None:
                raise ValueError("HF stream event has no token or completed output")

    def finish(self) -> Generation:
        if not self.completed:
            raise ValueError("Native stream disconnected before complete terminal event")
        return self.shared.finish()

    def close(self) -> None:
        self.shared.close()


def compatibility_configuration(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    extra = {"api_version", "deployment_id", "auth_type", "api_key_header", "max_tokens_field"}
    if set(options) - COMMON - extra:
        raise ValueError("Unknown compatible-provider transport option")
    configuration(
        "openai-completions", {key: value for key, value in options.items() if key in COMMON}
    )
    if options.get("auth_type", "bearer") not in {"bearer", "api-key", "basic", "none"}:
        raise ValueError("Unknown compatible-provider authentication mode")
    if options.get("max_tokens_field", "max_completion_tokens") not in {
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    }:
        raise ValueError("Unknown native output-token field")
    if options.get("deployment_id") and not options.get("api_version"):
        raise ValueError("Azure deployment transport requires explicit API version")
    return options
