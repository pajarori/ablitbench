from __future__ import annotations

import math
from typing import Any

OPTIONS = {
    "openai-chat": {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "reasoning_effort",
        "verbosity",
        "tool_choice",
        "parallel_tool_calls",
        "audio",
    },
    "openai-responses": {"temperature", "top_p", "reasoning", "text", "tool_choice"},
    "anthropic-messages": {"temperature", "top_p", "top_k", "stop", "thinking", "tool_choice"},
    "transformers": {
        "temperature",
        "top_p",
        "top_k",
        "do_sample",
        "num_beams",
        "repetition_penalty",
        "length_penalty",
        "early_stopping",
        "no_repeat_ngram_size",
        "min_new_tokens",
        "eos_token_id",
        "pad_token_id",
        "renormalize_logits",
        "bad_words_ids",
        "suppress_tokens",
    },
    "llama-cpp": {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "tfs_z",
        "response_format",
        "tool_choice",
    },
}


def text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(block, dict)
        and set(block) == {"type", "text"}
        and block["type"] == "text"
        and isinstance(block["text"], str)
        for block in content
    ):
        return "\n".join(block["text"] for block in content)
    raise ValueError("System content requires text or normalized text blocks")


def generation_options(backend: str, values: dict[str, Any]) -> dict[str, Any]:
    if backend not in OPTIONS:
        raise ValueError("Unknown generation backend")
    unsupported = set(values) - OPTIONS[backend]
    if unsupported:
        raise ValueError(
            "Unsupported generation options for " + backend + ": " + ", ".join(sorted(unsupported))
        )
    pending = list(values.values())
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Generation options must be finite")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError("Generation options must contain JSON values")
    result = dict(values)
    if backend == "anthropic-messages" and "stop" in result:
        stop = result.pop("stop")
        result["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if backend == "transformers":
        sampling = {"temperature", "top_p", "top_k"} & result.keys()
        if result.get("temperature") == 0:
            if result.get("do_sample") is True or sampling - {"temperature"}:
                raise ValueError(
                    "Zero temperature requires greedy decoding without sampling options"
                )
            result.pop("temperature")
            result["do_sample"] = False
        elif sampling:
            if result.get("do_sample") is False:
                raise ValueError("Sampling options require do_sample=True")
            result.setdefault("do_sample", True)
        else:
            result.setdefault("do_sample", False)
    return result
