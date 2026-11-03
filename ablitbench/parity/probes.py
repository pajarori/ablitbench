from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
from contextlib import contextmanager, redirect_stdout
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from ablitbench.models import Generation

PROJECT = "garak"
REVISION = "4b7761c2108446043012549cbe8f7c942cabb26d"


def operations() -> dict[str, dict[str, Any]]:
    inputs = {
        "corpus": ["corpus"],
        "report": ["report"],
        "count-tokens": ["report"],
        "aggregate": ["reports"],
    }
    return {
        name: {
            "project": PROJECT,
            "kind": kind,
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "permissions": [],
            "capabilities": ["generation-broker"] if name in {"scan", "tap", "pair"} else [],
            "inputroles": ["source", "assets", "corpus", "report", "reports"],
            "inputs": ["source", *inputs.get(name, [])],
            "optional_inputs": ["assets", "model", "base"] if name == "scan" else ["assets"],
            "source_model": name == "scan",
            "revision": REVISION,
        }
        for name, kind in (
            ("scan", "detect"),
            ("tap", "attack"),
            ("pair", "attack"),
            ("plugins", "inspect"),
            ("corpus", "detect"),
            ("report", "export"),
            ("aggregate", "export"),
            ("count-tokens", "inspect"),
        )
    }


def requires_tools(operation: str, settings: dict[str, Any]) -> bool:
    if operation not in {"scan", "corpus"}:
        return False
    path = Path(__file__).parent.parent / "data/parity-probes-capabilities.json"
    specification = json.loads(path.read_text())
    if specification["revision"] != REVISION:
        raise ValueError("Garak execution-capability review revision differs from pinned source")
    selected = [name for key in ("probes", "detectors", "buffs") for name in settings.get(key, [])]
    return any(
        specification["plugins"].get(name, {}).get("requires_tools", False) for name in selected
    )


def available_modes(operation: str, settings: dict[str, Any]) -> list[str]:
    return (
        ["source"]
        if settings.get("local_model") or settings.get("auxiliary_models")
        else ["api", "source"]
    )


def requires_inference(operation: str, settings: dict[str, Any]) -> bool:
    return bool(
        (operation == "scan" and settings.get("local_model") is True)
        or settings.get("auxiliary_models")
    )


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown Garak operation")
    allowed = {
        "probes",
        "detectors",
        "buffs",
        "plugin_options",
        "generations",
        "max_output_tokens",
        "seed",
        "threshold",
        "soft_probe_prompt_cap",
        "channels",
        "goal",
        "target",
        "target_settings",
        "target_generator",
        "branching_factor",
        "width",
        "depth",
        "n_streams",
        "keep_last_n",
        "attack_max_attempts",
        "attacker_model",
        "judge_model",
        "attacker_context_tokens",
        "judge_context_tokens",
        "reports",
        "attacker_settings",
        "judge_settings",
        "allow_native_truncation",
        "auxiliary_models",
        "whitebox_settings",
        "local_model",
        "device",
        "modalities",
        "output_modalities",
        "output_audio_format",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown Garak settings")
    result = dict(settings)
    if type(result.get("local_model", False)) is not bool:
        raise ValueError("local_model requires an explicit boolean")
    if result.get("local_model") and operation != "scan":
        raise ValueError("Local whitebox target is supported only by native scan")
    if result.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("Local model device must be cpu or cuda; no silent fallback")
    if type(result.get("allow_native_truncation", False)) is not bool:
        raise ValueError("Native truncation policy must be an explicit boolean")
    outputs = result.get("output_modalities", ["text"])
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(
            not isinstance(value, str) or value not in {"text", "image", "audio", "video"}
            for value in outputs
        )
    ):
        raise ValueError("Explicit supported output modalities required")
    aliases = result.get("auxiliary_models", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        for name, path in aliases.items()
    ):
        raise ValueError(
            "Auxiliary models require explicit model identity to relative assets path mapping"
        )
    if not isinstance(result.get("target_settings", {}), dict):
        raise ValueError("Garak target_settings require generation option mapping")
    if result.get("target_generator") is not None:
        native_profile(result["target_generator"])
    modalities = result.get("modalities", ["text"])
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(
            not isinstance(value, str)
            or value not in {"text", "image", "audio", "video", "document"}
            for value in modalities
        )
    ):
        raise ValueError("Explicit supported input modalities required")
    whitebox = result.get("whitebox_settings", {})
    if not isinstance(whitebox, dict) or set(whitebox) - {"suffix.GCG", "suffix.BEAST"}:
        raise ValueError("Only explicit native whitebox method settings accepted")
    beast = whitebox.get("suffix.BEAST", {})
    allowed_beast = {
        "prompts",
        "responses",
        "k1",
        "k2",
        "trials",
        "suffix_len",
        "data_size",
        "target",
        "stop_early",
    }
    if not isinstance(beast, dict) or set(beast) - allowed_beast:
        raise ValueError("Unknown native BEAST setting or unmounted asset path")
    gcg = whitebox.get("suffix.GCG", {})
    if not isinstance(gcg, dict) or set(gcg) - {"control_init"}:
        raise ValueError("Native GCG budgets belong to selected probe plugin_options")
    if gcg and (not isinstance(gcg.get("control_init"), str) or not gcg["control_init"]):
        raise ValueError("Native GCG initial control must be explicit text")
    selected = result.get("probes", [])
    if result.get("local_model") and "probes.suffix.BEAST" in selected:
        for key in ("k1", "k2", "trials", "suffix_len", "data_size"):
            if type(beast.get(key)) is not int or beast[key] <= 0:
                raise ValueError("Explicit native BEAST search budget required: " + key)
        if type(beast.get("stop_early")) is not bool:
            raise ValueError("Explicit native BEAST stopping policy required")
        prompts, responses = beast.get("prompts"), beast.get("responses")
        if (
            not isinstance(prompts, list)
            or not prompts
            or not isinstance(responses, list)
            or len(prompts) != len(responses)
            or any(not isinstance(item, str) or not item for item in prompts + responses)
        ):
            raise ValueError("Native BEAST requires explicit aligned search prompts/responses")
        if not isinstance(beast.get("target"), str) or not beast["target"]:
            raise ValueError(
                "Native BEAST requires explicit target matching; no empty-output success"
            )
    if result.get("local_model") and "probes.suffix.GCG" in selected:
        options = result.get("plugin_options", {})
        if not isinstance(options, dict) or not isinstance(options.get("probes", {}), dict):
            raise ValueError("Native GCG probe options must be a mapping")
        tree = options.get("probes", {})
        namespace = tree.get("suffix", {})
        if not isinstance(namespace, dict):
            raise ValueError("Native GCG namespace options must be a mapping")
        gcg_options = tree.get("suffix.GCG", namespace.get("GCG", {}))
        if not isinstance(gcg_options, dict):
            raise ValueError("Native GCG options must be a mapping")
        for key in ("n_steps", "batch_size", "topk", "n_rows"):
            if type(gcg_options.get(key)) is not int or gcg_options[key] <= 0:
                raise ValueError("Explicit native GCG search budget required: " + key)
    threshold = result.get("threshold", 0.5)
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Detector evaluation threshold must be finite and in [0,1]")
    channels = result.get("channels", {})
    if not isinstance(channels, dict) or any(
        not isinstance(key, str) or not key or value not in {"target", "attacker", "judge"}
        for key, value in channels.items()
    ):
        raise ValueError("Explicit model-to-broker channel mapping required")
    for key in ("attacker_settings", "judge_settings"):
        sampling = result.get(key, {})
        if not isinstance(sampling, dict) or set(sampling) - {
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "stop",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "num_beams",
        }:
            raise ValueError("Only native sampling settings may be supplied to auxiliary models")
    if operation in {"plugins", "report", "count-tokens"}:
        return result
    if operation == "aggregate":
        reports = result.get("reports")
        if (
            not isinstance(reports, list)
            or not reports
            or any(
                not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                for name in reports
            )
        ):
            raise ValueError("Explicit relative report paths required")
        return result
    if operation == "corpus":
        if not isinstance(result.get("detectors"), list) or not result["detectors"]:
            raise ValueError("Explicit native detectors required for stored corpus evaluation")
        if any(
            not re.fullmatch(r"detectors\.[A-Za-z_][A-Za-z0-9_.]*", name)
            for name in result["detectors"]
        ):
            raise ValueError("Qualified native detector names required")
        return result
    if type(result.get("max_output_tokens")) is not int or result["max_output_tokens"] <= 0:
        raise ValueError("Explicit positive output token budget required")
    if operation == "scan":
        probes = result.get("probes")
        if not isinstance(probes, list) or not probes:
            raise ValueError("Explicit probe selection required")
        for key, prefix in (("probes", "probes"), ("detectors", "detectors"), ("buffs", "buffs")):
            values = result.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(v, str) or not re.fullmatch(prefix + r"\.[A-Za-z_][A-Za-z0-9_.]*", v)
                for v in values
            ):
                raise ValueError("Qualified upstream plugin names required: " + key)
        if type(result.get("generations")) is not int or result["generations"] <= 0:
            raise ValueError("Explicit generation count required")
    if operation in {"tap", "pair"}:
        for key in ("goal", "target", "attacker_model", "judge_model"):
            if not isinstance(result.get(key), str) or not result[key]:
                raise ValueError("Explicit attack objective and broker model identities required")
        for key in (
            "width",
            "depth",
            "n_streams",
            "keep_last_n",
            "attack_max_attempts",
            "attacker_context_tokens",
            "judge_context_tokens",
        ):
            if type(result.get(key)) is not int or result[key] <= 0:
                raise ValueError("Explicit positive attack budget required: " + key)
        if operation == "tap" and (
            type(result.get("branching_factor")) is not int or result["branching_factor"] <= 0
        ):
            raise ValueError("Explicit branching_factor required")
    if result.get("soft_probe_prompt_cap") is not None and (
        type(result["soft_probe_prompt_cap"]) is not int or result["soft_probe_prompt_cap"] <= 0
    ):
        raise ValueError("Prompt cap must be absent or a positive explicit budget")
    options = result.get("plugin_options", {})
    if (
        not isinstance(options, dict)
        or set(options) - {"probes", "detectors", "buffs"}
        or any(not isinstance(value, dict) for value in options.values())
    ):
        raise ValueError("Only explicitly applied probe/detector/buff plugin options accepted")
    pending = [
        options,
        result.get("target_settings", {}),
        result.get("attacker_settings", {}),
        result.get("judge_settings", {}),
    ]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get("trust_remote_code") or item.get("custom_generate"):
                raise ValueError("Native plugins cannot import custom model repository code")
            if any(
                str(key).casefold()
                in {
                    "api_key",
                    "apikey",
                    "access_token",
                    "authorization",
                    "cookie",
                    "password",
                    "credential",
                    "secret",
                    "client_secret",
                }
                for key in item
            ):
                raise ValueError("Credentials are owned by the host broker, not plugin options")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return result


def write_record(output: Path, value: dict[str, Any]) -> None:
    with (output / "broker-calls.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class GapJournal:
    def __init__(self, output: Path):
        self.path = output / "garak-gaps.jsonl"
        self.count = 0

    def append(self, value: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.count += 1

    def __len__(self) -> int:
        return self.count


def generation_budget(settings: dict[str, Any]) -> property:
    def read(instance: Any) -> int:
        return int(instance.__dict__.get("_max_tokens", settings["max_output_tokens"]))

    def write(instance: Any, value: Any) -> None:
        if getattr(instance, "_budget_locked", False) and (
            type(value) is not int or value <= 0 or value > settings["max_output_tokens"]
        ):
            raise ValueError("Native plugin attempted to exceed the explicit output token budget")
        instance.__dict__["_max_tokens"] = value

    return property(read, write)


NATIVE_ATTRIBUTES = {
    "temperature": "temperature",
    "top_p": "top_p",
    "p": "top_p",
    "top_k": "top_k",
    "k": "top_k",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "stop": "stop",
    "do_sample": "do_sample",
    "num_beams": "num_beams",
    "length_penalty": "length_penalty",
    "early_stopping": "early_stopping",
    "no_repeat_ngram_size": "no_repeat_ngram_size",
    "min_new_tokens": "min_new_tokens",
    "eos_token_id": "eos_token_id",
    "pad_token_id": "pad_token_id",
    "renormalize_logits": "renormalize_logits",
    "bad_words_ids": "bad_words_ids",
    "suppress_tokens": "suppress_tokens",
}


def native_profile(path: str) -> dict[str, Any]:
    if not isinstance(path, str):
        raise ValueError("Native generator profile requires qualified name")
    metadata = json.loads(
        (Path(__file__).parent.parent / "data/parity-probes-generators.json").read_text()
    )
    if metadata["revision"] != REVISION:
        raise ValueError("Native generator defaults revision differs from verified source")
    path = path.removeprefix("garak.").removeprefix("generators.")
    if path in metadata["default_classes"]:
        path += "." + metadata["default_classes"][path]
    if path not in metadata["classes"]:
        raise ValueError("Unknown pinned native generator profile")
    return dict(metadata["classes"][path])


def initialize_sampling(
    instance: Any, values: dict[str, Any], native: dict[str, Any] | None = None
) -> None:
    native = dict(native or {})
    hf_args = native.get("hf_args")
    if isinstance(hf_args, dict):
        native.update({key: value for key, value in hf_args.items() if key in NATIVE_ATTRIBUTES})
    canonical = {
        NATIVE_ATTRIBUTES[key]: value for key, value in native.items() if key in NATIVE_ATTRIBUTES
    }
    canonical.update(
        {NATIVE_ATTRIBUTES.get(key, key): value for key, value in values.items() if key != "seed"}
    )
    instance._sampling_base = {
        key: copy.deepcopy(value) for key, value in canonical.items() if value is not None
    }
    for attribute, field in NATIVE_ATTRIBUTES.items():
        if field in canonical:
            setattr(instance, attribute, copy.deepcopy(canonical[field]))
    selected_seed = values.get("seed", native.get("seed"))
    if selected_seed is not None:
        instance.seed = selected_seed
    instance._sampling_suppressed = {
        NATIVE_ATTRIBUTES.get(key, key) for key in native.get("suppressed_params", [])
    }
    instance._vary_seed = native.get("vary_seed_each_call", False)
    instance._vary_temp = native.get("vary_temp_each_call", False)
    instance._sampling_before = {
        attribute: copy.deepcopy(getattr(instance, attribute, None))
        for attribute in NATIVE_ATTRIBUTES
    }
    if instance.seed is not None:
        instance._rng.seed(instance.seed)


def current_sampling(instance: Any) -> dict[str, Any]:
    if instance._vary_seed:
        instance.seed = instance._rng.randint(0, 65535)
    if instance._vary_temp:
        instance.temperature = instance._rng.random()
    result = copy.deepcopy(instance._sampling_base)
    for attribute, canonical in NATIVE_ATTRIBUTES.items():
        value = getattr(instance, attribute, None)
        if value != instance._sampling_before[attribute]:
            if value is None:
                result.pop(canonical, None)
            else:
                result[canonical] = copy.deepcopy(value)
    return {key: value for key, value in result.items() if key not in instance._sampling_suppressed}


def generator(
    settings: dict[str, Any],
    broker: Any,
    output: Path,
    channel: str = "target",
    name: str = "ablitbench",
    sampling: dict[str, Any] | None = None,
    roots: list[Path] | None = None,
    native: dict[str, Any] | None = None,
) -> Any:
    base = importlib.import_module("garak.generators.base").Generator
    compatible = importlib.import_module("garak.generators.openai").OpenAICompatible

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        self._budget_locked = False
        base.__init__(self, name=channel + ":" + name if channel != "target" else name)
        self.max_tokens = settings["max_output_tokens"]
        self._budget_locked = True
        self.seed = settings.get("seed")
        initial = settings.get("target_settings", {}) if channel == "target" else (sampling or {})
        initialize_sampling(self, initial, native)
        self.modality = {
            "in": set(settings.get("modalities", ["text"])),
            "out": set(settings.get("output_modalities", ["text"])),
        }

    def call_model(self: Any, prompt: Any, generations_this_call: int = 1) -> list[Any]:
        from ablitbench.parity.multimodal import garak_content, garak_generation

        messages = [
            {"role": turn.role, "content": garak_content(turn.content, roots or [output])}
            for turn in prompt.turns
        ]
        replies = []
        for _ in range(generations_this_call):
            call = uuid.uuid4().hex
            write_record(
                output, {"phase": "started", "call": call, "channel": channel, "messages": messages}
            )
            request: dict[str, Any] = {
                "kind": "judge" if channel == "judge" else "generation",
                "channel": channel,
                "messages": messages,
                "max_output_tokens": self.max_tokens,
                "output_modalities": settings.get("output_modalities", ["text"]),
            }
            if settings.get("output_audio_format"):
                request["output_audio_format"] = settings["output_audio_format"]
            request["generation_settings"] = current_sampling(self)
            if self.seed is not None:
                request["seed"] = self.seed
            if channel != "target" or name != "ablitbench":
                request["expected_model"] = name
            result = broker(request)
            if result.get("tools") or not Generation.model_validate(result).complete:
                raise ValueError(
                    "Probe response is truncated, failed or requires unsupported tools"
                )
            if not isinstance(result.get("text"), str):
                raise ValueError("Generation broker did not return text")
            if channel == "judge" and settings.get("goal") is not None:
                if re.search(r"\[\[(?:[1-9]|10|YES|NO)\]\]", result["text"], re.IGNORECASE) is None:
                    raise ValueError(
                        "Original adaptive judge returned no valid rating; inconclusive"
                    )
            write_record(output, {"phase": "completed", "call": call, "generation": result})
            replies.append(garak_generation(result))
        return replies

    bridge = type(
        "AblitBenchHostBroker",
        (compatible,),
        {
            "__module__": "garak.generators.openai",
            "_ablitbench_broker": True,
            "ENV_VAR": None,
            "parallel_capable": False,
            "generator_family_name": "AblitBench",
            "description": "Explicit host transport broker",
            "extra_dependency_names": [],
            "__init__": initialize,
            "_call_model": call_model,
            "max_tokens": generation_budget(settings),
        },
    )
    openai_module: Any = importlib.import_module("garak.generators.openai")
    openai_module.AblitBenchHostBroker = bridge
    return bridge()


def local_generator(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path, roots: list[Path]
) -> Any:
    from ablitbench.worker import Inference

    if "model" not in inputs:
        raise ValueError("Local whitebox scan requires an explicitly mounted safe model")
    if not inputs["model"].is_dir():
        raise ValueError("Native whitebox requires a safe HF model directory exposing gradients")
    inference = Inference(
        settings.get("device", "cpu"),
        inputs["model"],
        inputs.get("base", Path("/nonexistent-base")),
    )
    native_generate = inference.model.generate

    @wraps(native_generate)
    def bounded_generate(*args: Any, **kwargs: Any) -> Any:
        generation_config = kwargs.get("generation_config", inference.model.generation_config)
        budget = kwargs.get("max_new_tokens", getattr(generation_config, "max_new_tokens", None))
        if type(budget) is not int or budget <= 0 or budget > settings["max_output_tokens"]:
            raise ValueError("Native whitebox generation must remain within explicit output budget")
        ids = kwargs.get("input_ids", args[0] if args else None)
        embeddings = kwargs.get("inputs_embeds")
        input_tokens = (
            ids.shape[-1]
            if ids is not None
            else (embeddings.shape[-2] if embeddings is not None else None)
        )
        capacity = getattr(inference.model.config, "max_position_embeddings", None)
        if type(capacity) is int and input_tokens is not None and input_tokens + budget > capacity:
            raise ValueError("Native whitebox generation exceeds source context; no truncation")
        call = uuid.uuid4().hex
        write_record(
            output,
            {
                "phase": "started",
                "call": call,
                "channel": "native-whitebox",
                "input_ids": ids.detach().cpu().tolist() if ids is not None else None,
                "input_tokens": input_tokens,
                "max_new_tokens": budget,
            },
        )
        generated = native_generate(*args, **kwargs)
        write_record(
            output,
            {
                "phase": "completed",
                "call": call,
                "token_ids": generated.detach().cpu().tolist()
                if isinstance(generated, inference.torch.Tensor)
                else None,
                "output_type": type(generated).__name__,
            },
        )
        return generated

    inference.model.generate = bounded_generate
    native = importlib.import_module("garak.generators.huggingface").Model
    base = importlib.import_module("garak.generators.base").Generator

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        self._budget_locked = False
        base.__init__(self, name=str(inputs["model"]))
        self.max_tokens = settings["max_output_tokens"]
        self._budget_locked = True
        self.seed = settings.get("seed")
        initialize_sampling(self, settings.get("target_settings", {}))
        self.model = inference.model
        self.tokenizer = inference.tokenizer
        self.device = inference.device
        self.generation_config = inference.model.generation_config
        self.modality = {
            "in": set(settings.get("modalities", ["text"])),
            "out": set(settings.get("output_modalities", ["text"])),
        }

    def call_model(self: Any, prompt: Any, generations_this_call: int = 1) -> list[Any]:
        from ablitbench.parity.multimodal import garak_content, garak_generation

        request = {
            "messages": [
                {"role": turn.role, "content": garak_content(turn.content, roots)}
                for turn in prompt.turns
            ],
            "max_output_tokens": self.max_tokens,
            "seed": self.seed,
            "generation_settings": current_sampling(self),
            "output_modalities": settings.get("output_modalities", ["text"]),
            "output_audio_format": settings.get("output_audio_format"),
        }
        replies = []
        for _ in range(generations_this_call):
            call = uuid.uuid4().hex
            write_record(
                output,
                {"phase": "started", "call": call, "channel": "local-target", "request": request},
            )
            result = inference.generate(request)
            write_record(output, {"phase": "completed", "call": call, "generation": result})
            if result.get("tools") or not Generation.model_validate(result).complete:
                raise ValueError("Local native probe generation was incomplete or requires tools")
            replies.append(garak_generation(result))
        return replies

    bridge = type(
        "AblitBenchSafeLocalModel",
        (native,),
        {
            "__module__": "garak.generators.huggingface",
            "__init__": initialize,
            "_call_model": call_model,
            "max_tokens": generation_budget(settings),
            "_ablitbench_broker": True,
            "parallel_capable": False,
            "extra_dependency_names": [],
            "description": "Shared safely loaded source model",
        },
    )
    module: Any = importlib.import_module("garak.generators.huggingface")
    module.AblitBenchSafeLocalModel = bridge
    return bridge()


@contextmanager
def broker_plugins(
    settings: dict[str, Any],
    broker: Any,
    output: Path,
    gaps: GapJournal,
    roots: list[Path] | None = None,
) -> Any:
    plugins: Any = importlib.import_module("garak._plugins")
    original = plugins.load_plugin
    previous_instances = plugins.PluginProvider._instance_cache
    plugins.PluginProvider._instance_cache = {}
    base = importlib.import_module("garak.generators.base").Generator
    original_new = base.__dict__.get("__new__")
    channels = settings.get("channels", {})

    def construct(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(cls, "_ablitbench_broker", False):
            return object.__new__(cls)
        if not cls.__module__.startswith("garak.generators."):
            raise ValueError("Unreviewed generator construction is unsupported")
        model_name = kwargs.get("name", args[0] if args else None)
        if model_name is None:
            parameter = inspect.signature(cls.__init__).parameters.get("name")
            if parameter is not None and isinstance(parameter.default, str):
                model_name = parameter.default
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Direct native generator requires an explicit routed model identity")
        selected = channels.get(model_name)
        if selected is None:
            selected = (
                "attacker"
                if model_name == settings.get("attacker_model")
                else "judge"
                if model_name == settings.get("judge_model")
                else None
            )
        if selected not in {"target", "attacker", "judge"}:
            raise ValueError(
                "Missing explicit broker route for direct native generator: " + model_name
            )
        native = native_profile(cls.__module__.removeprefix("garak.") + "." + cls.__name__)
        native.update(
            {
                key: value
                for key, value in kwargs.items()
                if key in NATIVE_ATTRIBUTES or key == "seed"
            }
        )
        sampling = dict(settings.get(selected + "_settings", {}))
        return generator(settings, broker, output, selected, model_name, sampling, roots, native)

    def load(path: str, break_on_fail: bool = True, config_root: Any = None) -> Any:
        if path.startswith("generators."):
            options = config_root.get("generators", {}) if isinstance(config_root, dict) else {}
            for part in path.split(".")[1:]:
                options = options.get(part, {})
            model_name = options.get("name", path)
            selected = options.get("ablitbench_channel") or channels.get(model_name)
            if selected is None:
                if model_name == settings.get("attacker_model"):
                    selected = "attacker"
                elif model_name == settings.get("judge_model"):
                    selected = "judge"
            if selected not in {"target", "attacker", "judge"}:
                raise ValueError(
                    "Explicit broker channel required for auxiliary generator: " + model_name
                )
            native = native_profile(path)
            native.update(
                {
                    key: value
                    for key, value in options.items()
                    if key in NATIVE_ATTRIBUTES
                    or key
                    in {"seed", "suppressed_params", "vary_seed_each_call", "vary_temp_each_call"}
                }
            )
            sampling = dict(settings.get(selected + "_settings", {}))
            return generator(
                settings, broker, output, selected, model_name, sampling, roots, native
            )
        try:
            kwargs = {"break_on_fail": break_on_fail}
            if config_root is not None:
                kwargs["config_root"] = config_root
            value = original(path, **kwargs)
        except (ImportError, ValueError, OSError, RuntimeError) as error:
            gaps.append({"plugin": path, "error": type(error).__name__, "detail": str(error)})
            raise
        if not value:
            gaps.append({"plugin": path, "error": "Upstream plugin was not loaded"})
        else:
            tokenizer_options = getattr(value, "tokenizer_kwargs", None)
            if isinstance(tokenizer_options, dict) and tokenizer_options.get("truncation"):
                if not settings.get("allow_native_truncation"):
                    value.tokenizer_kwargs = {**tokenizer_options, "truncation": False}
            if path == "probes.suffix.BEAST" and settings.get("local_model"):
                native_beast = value.run_beast
                parameters = dict(settings["whitebox_settings"]["suffix.BEAST"])
                parameters["outfile"] = output / "native-beast-suffixes.txt"

                def run_beast(*args: Any, **kwargs: Any) -> Any:
                    return native_beast(*args, **{**parameters, **kwargs})

                value.run_beast = run_beast
            if path == "probes.suffix.GCG" and settings.get("local_model"):
                native_gcg = value.run_gcg
                parameters_gcg = dict(settings.get("whitebox_settings", {}).get("suffix.GCG", {}))
                parameters_gcg["outfile"] = output / "native-gcg-suffixes.txt"

                def run_gcg(*args: Any, **kwargs: Any) -> Any:
                    return native_gcg(*args, **{**parameters_gcg, **kwargs})

                value.run_gcg = run_gcg
            defaults = getattr(value, "DEFAULT_PARAMS", {})
            configuration = {key: getattr(value, key, None) for key in defaults}
            with (output / "garak-selected-settings.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "plugin": path,
                            "native_defaults": defaults,
                            "whitebox_settings": settings.get("whitebox_settings", {}).get(
                                path.removeprefix("probes.")
                            ),
                            "effective_parameters": configuration,
                            "native_truncation_authorized": settings.get(
                                "allow_native_truncation", False
                            ),
                        },
                        default=str,
                        allow_nan=False,
                    )
                    + "\n"
                )
        return value

    plugins.load_plugin = load
    base.__new__ = staticmethod(construct)
    try:
        yield
    finally:
        plugins.load_plugin = original
        plugins.PluginProvider._instance_cache = previous_instances
        if original_new is None:
            del base.__new__
        else:
            base.__new__ = original_new


@contextmanager
def auxiliary_loaders(settings: dict[str, Any], inputs: dict[str, Path]) -> Any:
    aliases = settings.get("auxiliary_models", {})
    if not aliases:
        yield
        return
    if "assets" not in inputs:
        raise ValueError("Auxiliary classifier/buff models require mounted assets")
    assets = inputs["assets"].resolve()
    mapped = {}
    for name, relative in aliases.items():
        path = (assets / relative).resolve()
        if not path.is_dir() or not path.is_relative_to(assets):
            raise ValueError("Auxiliary model is outside its declared asset bundle")
        mapped[name] = path
    transformers = importlib.import_module("transformers")
    classes = [
        transformers.AutoConfig,
        transformers.AutoTokenizer,
        transformers.AutoModelForSequenceClassification,
        transformers.AutoModelForSeq2SeqLM,
        transformers.PegasusForConditionalGeneration,
        transformers.PegasusTokenizer,
    ]
    originals = []
    compatible = importlib.import_module("garak.resources.api.huggingface").HFCompatible
    original_params = compatible._gather_generation_params
    if not hasattr(transformers.GenerationConfig, "_get_default_generation_params"):

        def generation_params(instance: Any) -> dict[str, Any]:
            return {
                key: instance.hf_args[key]
                for key in transformers.GenerationConfig().to_dict()
                if key in instance.hf_args
            }

        compatible._gather_generation_params = generation_params
    try:
        for cls in classes:
            descriptor = cls.__dict__.get("from_pretrained")
            original = cls.from_pretrained
            originals.append((cls, descriptor))

            def guarded(
                target: Any,
                path: Any,
                /,
                *args: Any,
                _original: Any = original,
                _model: bool = cls
                not in {
                    transformers.AutoConfig,
                    transformers.AutoTokenizer,
                    transformers.PegasusTokenizer,
                },
                **kwargs: Any,
            ) -> Any:
                candidate = mapped.get(str(path), Path(str(path)).resolve())
                if not candidate.is_dir() or not any(
                    candidate == root or candidate.is_relative_to(root) for root in mapped.values()
                ):
                    raise ValueError("Native classifier/buff requested an unmounted model identity")
                kwargs["trust_remote_code"] = False
                kwargs["local_files_only"] = True
                if _model:
                    kwargs["use_safetensors"] = True
                return _original(str(candidate), *args, **kwargs)

            cls.from_pretrained = classmethod(guarded)
        yield
    finally:
        compatible._gather_generation_params = original_params
        for cls, descriptor in originals:
            if descriptor is None:
                delattr(cls, "from_pretrained")
            else:
                cls.from_pretrained = descriptor


def execute(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    broker: Any = None,
) -> dict[str, Any]:
    settings = configuration(operation, settings)
    from ablitbench.parity.sources import verified_source
    from ablitbench.worker import isolation

    isolation()
    source = verified_source(inputs["source"], PROJECT)
    existing = sys.modules.get("garak")
    if existing is not None and not Path(str(existing.__file__)).resolve().is_relative_to(source):
        raise ValueError("Original Garak namespace collision")
    previous_environment = dict(os.environ)
    previous_bridge: Any = None
    bridge_module: Any = None
    local_module: Any = None
    previous_local: Any = None
    config: Any = None
    data_module: Any = None
    original_data_path: Any = None
    output.mkdir(parents=True, exist_ok=True)
    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        os.environ[key] = str(output / key.casefold())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    try:
        config = importlib.import_module("garak._config")
        config.load_base_config()
        if "assets" in inputs:
            assets = inputs["assets"]
            if not assets.is_dir():
                raise ValueError("Garak assets require a cache/data directory bundle")
            if (assets / "cache").is_dir():
                install_assets(assets / "cache", config.transient.cache_dir, output)
            if (assets / "data").is_dir():
                data_module = importlib.import_module("garak.data")
                original_data_path = data_module.path
                overlay = output / "garak-data"
                install_assets(source / "garak/data", overlay, output)
                install_assets(assets / "data", overlay, output)
                data_module.path = overlay
        config.run.generations = settings.get("generations", 1)
        config.run.seed = settings.get("seed")
        config.run.soft_probe_prompt_cap = settings.get("soft_probe_prompt_cap")
        config.system.parallel_requests = False
        config.system.parallel_attempts = False
        config.plugins.extended_detectors = True
        config.plugins.probes.update(settings.get("plugin_options", {}).get("probes", {}))
        config.plugins.detectors.update(settings.get("plugin_options", {}).get("detectors", {}))
        config.plugins.buffs.update(settings.get("plugin_options", {}).get("buffs", {}))
        config.transient.run_id = uuid.uuid4()
        plugins: Any = importlib.import_module("garak._plugins")
        if operation == "plugins":
            listing = {
                kind: list(plugins.enumerate_plugins(kind))
                for kind in ("probes", "detectors", "generators", "buffs", "harnesses")
            }
            (output / "garak-plugins.json").write_text(json.dumps(listing))
            return {
                "project": PROJECT,
                "operation": operation,
                "status": "completed",
                "plugin_counts": {kind: len(values) for kind, values in listing.items()},
                "plugins": "garak-plugins.json",
                "upstream_revision": REVISION,
            }
        if operation in {"report", "aggregate", "count-tokens", "corpus"}:
            with auxiliary_loaders(settings, inputs):
                return stored_workflow(operation, settings, inputs, output, config, plugins)
        if broker is None:
            raise ValueError("Garak execution requires the host generation broker")
        bridge_module = importlib.import_module("garak.generators.openai")
        previous_bridge = getattr(bridge_module, "AblitBenchHostBroker", None)
        roots = [source, output, *inputs.values(), Path(tempfile.gettempdir())]
        if settings.get("local_model"):
            local_module = importlib.import_module("garak.generators.huggingface")
            previous_local = getattr(local_module, "AblitBenchSafeLocalModel", None)
        target = (
            local_generator(settings, inputs, output, roots)
            if settings.get("local_model")
            else generator(
                settings,
                broker,
                output,
                roots=roots,
                native=native_profile(settings["target_generator"])
                if settings.get("target_generator")
                else None,
            )
        )
        gaps = GapJournal(output)
        summary: dict[str, Any] = {"attempts": 0}
        with (output / "garak.report.jsonl").open("w", encoding="utf-8") as report:
            version = importlib.import_module("garak").__version__
            report.write(
                json.dumps(
                    {
                        "entry_type": "start_run setup",
                        "plugins.target_type": "AblitBenchHostBroker",
                        "plugins.target_name": "Operator-configured target",
                        "_config.version": version,
                        "run.generations": settings.get("generations", 1),
                        "transient.active_probes": ",".join(settings.get("probes", [])),
                    }
                )
                + "\n"
            )
            report.write(
                json.dumps(
                    {
                        "entry_type": "init",
                        "garak_version": version,
                        "start_time": datetime.now(UTC).isoformat(),
                        "run": str(config.transient.run_id),
                    }
                )
                + "\n"
            )
            config.transient.reportfile = report
            config.transient.report_filename = str(output / "garak.report.jsonl")
            with (
                auxiliary_loaders(settings, inputs),
                broker_plugins(settings, broker, output, gaps, roots),
            ):
                if operation in {"tap", "pair"}:
                    tap = importlib.import_module("garak.resources.tap.tap_main")
                    openai = importlib.import_module("garak.generators.openai")
                    scoring: Any = importlib.import_module("garak.resources.red_team.evaluation")
                    original_rating = scoring.process_output_judge_score
                    original_topic = scoring.process_output_on_topic_score

                    def rating(text: str) -> float:
                        if re.search(r"\[\[(?:[1-9]|10)\]\]", text) is None:
                            raise ValueError("Adaptive judge rating invalid; inconclusive")
                        return float(original_rating(text))

                    def topic(text: str) -> float:
                        if re.search(r"\[\[(?:YES|NO)\]\]", text, re.IGNORECASE) is None:
                            raise ValueError("Adaptive on-topic judge label invalid; inconclusive")
                        return float(original_topic(text))

                    scoring.process_output_judge_score = rating
                    scoring.process_output_on_topic_score = topic
                    saved = dict(openai.context_lengths)
                    openai.context_lengths["attacker:" + settings["attacker_model"]] = settings[
                        "attacker_context_tokens"
                    ]
                    openai.context_lengths["judge:" + settings["judge_model"]] = settings[
                        "judge_context_tokens"
                    ]
                    try:
                        attacks = tap.run_tap(
                            goal=settings["goal"],
                            target=settings["target"],
                            target_generator=target,
                            attack_model_type="openai.OpenAICompatible",
                            attack_model_name=settings["attacker_model"],
                            attack_model_config={"ablitbench_channel": "attacker"}
                            | settings.get("attacker_settings", {}),
                            attack_max_attempts=settings["attack_max_attempts"],
                            evaluator_model_type="openai.OpenAICompatible",
                            evaluator_model_name=settings["judge_model"],
                            evaluator_model_config={
                                "ablitbench_channel": "judge",
                                "temperature": 0.0,
                            }
                            | settings.get("judge_settings", {}),
                            branching_factor=settings["branching_factor"]
                            if operation == "tap"
                            else 1,
                            width=settings["width"],
                            depth=settings["depth"],
                            n_streams=settings["n_streams"],
                            keep_last_n=settings["keep_last_n"],
                            pruning=operation == "tap",
                            save_results=True,
                            outfile=output / "successful-prompts.txt",
                        )
                    finally:
                        scoring.process_output_judge_score = original_rating
                        scoring.process_output_on_topic_score = original_topic
                        openai.context_lengths.clear()
                        openai.context_lengths.update(saved)
                    summary.update(
                        {
                            "successful_prompts": len(attacks),
                            "successful_prompt_artifact": "successful-prompts.txt",
                            "method": "Original run_tap",
                            "pruning": operation == "tap",
                        }
                    )
                else:
                    evaluator_class = importlib.import_module(
                        "garak.evaluators.base"
                    ).ThresholdEvaluator

                    def evaluate(self: Any, attempts: Any) -> None:
                        rows = list(attempts)
                        summary["attempts"] += len(rows)
                        for row in rows:
                            if not row.detector_results or not any(row.detector_results.values()):
                                gaps.append(
                                    {
                                        "attempt": str(row.uuid),
                                        "reason": "No detector observations; incomplete coverage",
                                    }
                                )
                            for detector_name, scores in row.detector_results.items():
                                if not scores or any(
                                    score is None or not math.isfinite(float(score))
                                    for score in scores
                                ):
                                    gaps.append(
                                        {
                                            "attempt": str(row.uuid),
                                            "detector": detector_name,
                                            "reason": "Missing native detector scores",
                                        }
                                    )
                        evaluator_class.evaluate(self, rows)

                    recording_evaluator = type(
                        "RecordingEvaluator", (evaluator_class,), {"evaluate": evaluate}
                    )

                    evaluator = recording_evaluator(settings.get("threshold", 0.5))
                    harness_class = importlib.import_module("garak.harnesses.base").Harness
                    harness = harness_class()
                    harness._load_buffs(settings.get("buffs", []))
                    for name in settings["probes"]:
                        try:
                            probe = plugins.load_plugin(name)
                            names = settings.get("detectors") or [
                                "detectors." + probe.primary_detector
                            ]
                            detectors = [plugins.load_plugin(value) for value in names]
                            for detector_name, detector in zip(names, detectors, strict=True):
                                supported = getattr(detector, "modality", {}).get("out", {"text"})
                                if not target.modality["out"].issubset(supported):
                                    raise ValueError(
                                        "Native detector cannot assess requested output modality: "
                                        + detector_name
                                    )
                            before = summary["attempts"]
                            harness.run(target, [probe], detectors, evaluator)
                            if summary["attempts"] == before:
                                gaps.append(
                                    {
                                        "probe": name,
                                        "reason": "No observed attempts; not a clean result",
                                    }
                                )
                        except (
                            ImportError,
                            ValueError,
                            RuntimeError,
                            OSError,
                            KeyError,
                            TypeError,
                        ) as error:
                            gaps.append(
                                {"probe": name, "error": type(error).__name__, "detail": str(error)}
                            )
                        report.flush()
                        os.fsync(report.fileno())
        result = {
            "project": PROJECT,
            "operation": operation,
            "upstream_revision": REVISION,
            "status": "unsupported" if gaps else "completed",
            "summary": summary,
            "attempt_artifact": "garak.report.jsonl",
            "gap_count": len(gaps),
            "gaps": [
                {"reason": str(len(gaps)) + " native coverage gaps", "artifact": "garak-gaps.jsonl"}
            ]
            if gaps
            else [],
            "limits": {
                "max_output_tokens": settings["max_output_tokens"],
                "soft_probe_prompt_cap": settings.get("soft_probe_prompt_cap"),
                "auxiliary_context_capacity": {
                    key: settings.get(key)
                    for key in ("attacker_context_tokens", "judge_context_tokens")
                },
                "upstream_defaults": "garak-selected-settings.jsonl",
                "native_truncation_authorized": settings.get("allow_native_truncation", False),
                "specialized_whitebox_generation": (
                    "Native optimization owns internal target/test token horizons; "
                    "source snapshot and native settings retained as evidence"
                )
                if settings.get("local_model")
                else None,
                "judge_compatibility_fix": (
                    "Invalid upstream score fallback is rejected as inconclusive"
                ),
            },
        }
        (output / "garak-result.json").write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        return result
    finally:
        if data_module is not None:
            data_module.path = original_data_path
        if config is not None:
            hitlog = getattr(config.transient, "hitlogfile", None)
            if hitlog is not None and not hitlog.closed:
                hitlog.flush()
                os.fsync(hitlog.fileno())
                hitlog.close()
            config.transient.hitlogfile = None
            config.transient.reportfile = None
        if bridge_module is not None:
            if previous_bridge is None:
                if hasattr(bridge_module, "AblitBenchHostBroker"):
                    delattr(bridge_module, "AblitBenchHostBroker")
            else:
                bridge_module.AblitBenchHostBroker = previous_bridge
        if local_module is not None:
            if previous_local is None:
                if hasattr(local_module, "AblitBenchSafeLocalModel"):
                    delattr(local_module, "AblitBenchSafeLocalModel")
            else:
                local_module.AblitBenchSafeLocalModel = previous_local
        os.environ.clear()
        os.environ.update(previous_environment)
        sys.path.remove(str(source))


def install_assets(source: Path, destination: Path, output: Path) -> None:
    from ablitbench.formats.snapshot import secure_open

    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Asset cache destination is a symlink")
    for root, directories, files in os.walk(source, followlinks=False):
        if any(Path(root, name).is_symlink() for name in directories):
            raise ValueError("Asset bundle contains a symlink directory")
        for name in files:
            path = Path(root, name)
            relative = path.relative_to(source)
            target = destination / relative
            current = destination
            for part in relative.parts[:-1]:
                current = current / part
                current.mkdir(exist_ok=True)
                if current.is_symlink():
                    raise ValueError("Asset cache contains a symlink directory")
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            checksum = hashlib.sha256()
            with secure_open(path) as stream, temporary.open("xb") as copied:
                before = os.fstat(stream.fileno())
                while chunk := stream.read(1024 * 1024):
                    copied.write(chunk)
                    checksum.update(chunk)
                after = os.fstat(stream.fileno())
                if any(
                    getattr(before, field) != getattr(after, field)
                    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
                ):
                    raise ValueError("Garak asset changed during preparation")
                copied.flush()
                os.fsync(copied.fileno())
            temporary.replace(target)
            write_record(
                output,
                {"phase": "asset_prepared", "asset": str(relative), "sha256": checksum.hexdigest()},
            )


def stored_workflow(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    config: Any,
    plugins: Any,
) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    if operation == "report":
        destination = output / "native.report.jsonl"
        shutil.copyfile(inputs["report"], destination)
        native = importlib.import_module("garak.analyze.report_digest")
        digest = native.build_digest(str(destination))
        (output / "digest.json").write_text(json.dumps(digest, ensure_ascii=False, allow_nan=False))
        (output / "report.html").write_text(native.build_html(digest), encoding="utf-8")
        detail = {"digest": "digest.json", "html": "report.html"}
    elif operation == "aggregate":
        root = inputs["reports"].resolve()
        paths = [(root / name).resolve(strict=True) for name in settings["reports"]]
        if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
            raise ValueError("Report outside mounted corpus")
        native = importlib.import_module("garak.analyze.aggregate_reports")
        native.main(["--output_path", str(output / "aggregate.report.jsonl"), *map(str, paths)])
        detail = {"aggregate": "aggregate.report.jsonl", "input_count": len(paths)}
    elif operation == "count-tokens":
        native = importlib.import_module("garak.analyze.count_tokens")
        captured = io.StringIO()
        with redirect_stdout(captured):
            native.count_tokens(str(inputs["report"]))
        (output / "native-character-estimate.txt").write_text(captured.getvalue())
        detail = {
            "estimate": captured.getvalue(),
            "scope": (
                "Original legacy character estimator; "
                "neither tokenizer accounting nor provider billing"
            ),
        }
    else:
        attempt_type = importlib.import_module("garak.attempt")
        detectors = [plugins.load_plugin(name) for name in settings["detectors"]]
        count = 0
        with sqlite3.connect(output / "corpus-checkpoint.sqlite3") as checkpoint:
            checkpoint.execute("CREATE TABLE attempts (id TEXT PRIMARY KEY, row TEXT NOT NULL)")
            with inputs["corpus"].open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if row.get("entry_type") != "attempt":
                        continue
                    if not isinstance(row.get("uuid"), str):
                        raise ValueError("Stored attempt identity is missing")
                    checkpoint.execute(
                        "INSERT INTO attempts VALUES (?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET row=excluded.row",
                        (row["uuid"], json.dumps(row, ensure_ascii=False)),
                    )
            checkpoint.commit()
            with (output / "corpus-results.jsonl").open("w", encoding="utf-8") as target:
                for (encoded,) in checkpoint.execute("SELECT row FROM attempts ORDER BY rowid"):
                    row = json.loads(encoded)
                    if row.get("status") != 2:
                        raise ValueError("Unfinished corpus attempt; detector coverage incomplete")
                    prompt = attempt_type.Conversation.from_dict(row["prompt"])
                    attempt = attempt_type.Attempt(
                        prompt=prompt,
                        probe_classname=row["probe_classname"],
                        probe_params=row.get("probe_params"),
                        targets=row.get("targets"),
                        notes=row.get("notes"),
                        goal=row.get("goal"),
                        seq=row.get("seq", -1),
                    )
                    attempt.outputs = [attempt_type.Message(**value) for value in row["outputs"]]
                    attempt.status = 2
                    for name, detector in zip(settings["detectors"], detectors, strict=True):
                        attempt.detector_results[name] = list(detector.detect(attempt))
                    target.write(
                        json.dumps(attempt.as_dict(), ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    count += 1
        if count == 0:
            raise ValueError("Stored corpus has no complete attempts")
        detail = {
            "attempts": count,
            "detectors": settings["detectors"],
            "results": "corpus-results.jsonl",
        }
    result = {
        "project": PROJECT,
        "operation": operation,
        "status": "completed",
        "upstream_revision": REVISION,
        "records": detail,
    }
    (output / "garak-result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    )
    return result
