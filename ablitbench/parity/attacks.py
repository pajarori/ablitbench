from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads
from ablitbench.formats.yaml import load as yaml_load


def operations() -> dict[str, dict[str, Any]]:
    return {
        "factory.train": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "train",
            "capabilities": ["sandbox", "reviewed-source", "legacy-factory"],
            "inputs": ["source", "model", "dataset"],
            "optional_inputs": [
                "adapter",
                "reference",
                "reward",
                "reference_adapter",
                "reward_adapter",
            ],
        },
        "factory.distributed": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "train",
            "capabilities": ["sandbox", "reviewed-source", "legacy-factory"],
            "inputs": ["source", "model", "dataset"],
            "optional_inputs": [
                "adapter",
                "reference",
                "reward",
                "reference_adapter",
                "reward_adapter",
            ],
        },
        "factory.plan": {
            "project": "BackdoorLLM",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "inspect",
            "capabilities": ["sandbox", "reviewed-source", "legacy-factory"],
            "inputs": ["source", "model", "dataset"],
            "optional_inputs": [
                "adapter",
                "reference",
                "reward",
                "reference_adapter",
                "reward_adapter",
            ],
        },
        "factory.export": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "export",
            "capabilities": ["sandbox", "reviewed-source", "legacy-factory"],
            "inputs": ["source", "model"],
            "optional_inputs": ["adapter", "calibration"],
        },
        "wpa.rome": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "attack",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "dataset"],
            "optional_inputs": ["base", "calibration", "covariance"],
        },
        "wpa.badedit": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "attack",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "dataset"],
            "optional_inputs": ["base", "calibration", "covariance"],
        },
        "wpa.covariance": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "inspect",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "calibration"],
            "optional_inputs": ["base"],
        },
        "wpa.noisy-trigger": {
            "project": "BackdoorLLM",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "attack",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model"],
            "optional_inputs": ["base"],
        },
        "cota.evaluate": {
            "project": "BackdoorLLM",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "evaluate",
            "capabilities": ["sandbox", "reviewed-source"],
            "inputs": ["source", "predictions", "labels"],
        },
        "attack.dataset": {
            "project": "BackdoorLLM",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "attack",
            "capabilities": ["sandbox", "reviewed-source"],
            "inputs": ["source", "dataset"],
        },
        "hsa.vectors": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "train",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "dataset"],
            "optional_inputs": ["base"],
        },
        "hsa.generate": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "attack",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "dataset", "vectors"],
            "optional_inputs": ["base"],
        },
        "b4g.dataset": {
            "project": "BackdoorLLM",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "attack",
            "capabilities": ["sandbox", "reviewed-source"],
            "inputs": ["source", "dataset", "task"],
        },
        "supervised.train": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "train",
            "capabilities": ["sandbox", "training", "local-model"],
            "inputs": ["model", "dataset"],
            "optional_inputs": ["base"],
        },
        "b4g.evaluate": {
            "project": "BackdoorLLM",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "evaluate",
            "capabilities": ["sandbox", "generation"],
            "inputs": ["clean", "triggered", "task"],
        },
        "adapter.merge": {
            "project": "BackdoorLLM",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "export",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "base"],
        },
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown attack/training operation")
    if operation.startswith("factory."):
        from ablitbench.parity.factory_worker import configuration as factory_configuration

        return factory_configuration(operation, settings)
    if operation.startswith("wpa."):
        return edit_configuration(operation, settings)
    allowed = {
        "cota.evaluate": {"task", "control", "check_step"},
        "attack.dataset": {"family", "method", "trigger", "target", "seed", "count", "position"},
        "hsa.vectors": {"layers", "device", "max_length"},
        "hsa.generate": {"blocks", "multiplier", "device", "max_output_tokens", "position"},
        "b4g.dataset": {"train_samples", "test_samples", "seed"},
        "supervised.train": {
            "method",
            "format",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_length",
            "truncate",
            "truncation_side",
            "seed",
            "device",
            "lora_r",
            "adalora",
            "quantization",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "max_grad_norm",
            "weight_decay",
        },
        "b4g.evaluate": {"max_output_tokens", "seed"},
        "adapter.merge": {"device"},
    }[operation]
    if set(settings) - allowed:
        raise ValueError("Unknown attack/training settings")
    result = dict(settings)
    if operation == "cota.evaluate":
        if set(settings) != allowed or settings["task"] not in {
            "gsm8k",
            "ASDiv",
            "MATH",
            "strategyqa",
            "letter",
            "csqa",
        }:
            raise ValueError("Explicit supported CoTA task and answer controls required")
        if type(settings["check_step"]) is not bool:
            raise ValueError("Step-check mode must be boolean")
        task, control = settings["task"], settings["control"]
        if task in {"strategyqa", "letter"} and type(control) is not bool:
            raise ValueError("Boolean answer inversion control required")
        if task == "csqa" and type(control) is not int:
            raise ValueError("Integer answer-choice shift required")
        if task in {"gsm8k", "ASDiv", "MATH"} and (
            type(control) not in {int, float} or not math.isfinite(control)
        ):
            raise ValueError("Finite answer factor required")
    elif operation == "attack.dataset":
        if set(settings) != allowed:
            raise ValueError(
                "Explicit dataset poisoning family, method, trigger, target and budgets required"
            )
        if settings["family"] not in {"dpa", "cota"}:
            raise ValueError("Supported dataset attack families are dpa and cota")
        if settings["family"] == "dpa" and settings["method"] not in {
            "sleeper",
            "badnet",
            "vpi",
            "mtba",
            "ctba",
        }:
            raise ValueError("Unknown DPA trigger placement method")
        if settings["family"] == "cota" and settings["position"] not in {
            "last",
            "before_choices",
            "bef",
            "mid",
            "ac",
        }:
            raise ValueError("Unknown BadChain trigger position")
        if type(settings["count"]) is not int or settings["count"] <= 0:
            raise ValueError("Explicit positive poisoning row count required")
        if not isinstance(settings["target"], str) or not settings["target"]:
            raise ValueError("Explicit controlled poisoning target text required")
        trigger = settings["trigger"]
        if settings["family"] == "dpa" and settings["method"] in {"mtba", "ctba"}:
            if (
                not isinstance(trigger, list)
                or len(trigger) != 3
                or any(not isinstance(value, str) or not value for value in trigger)
            ):
                raise ValueError(
                    "Composite and alternative DPA methods require exactly three triggers"
                )
        elif not isinstance(trigger, str) or not trigger:
            raise ValueError("Nonempty trigger text required")
    elif operation == "hsa.vectors":
        if (
            set(settings) != allowed
            or not isinstance(settings["layers"], list)
            or not settings["layers"]
        ):
            raise ValueError("Explicit hidden-state layer indices and context required")
        if any(type(layer) is not int or layer < 0 for layer in settings["layers"]):
            raise ValueError("Nonnegative hidden-state layer indices required")
        if len(set(settings["layers"])) != len(settings["layers"]):
            raise ValueError("Duplicate hidden-state layer")
        if type(settings["max_length"]) is not int or settings["max_length"] <= 0:
            raise ValueError("Explicit positive contrastive context required")
    elif operation == "hsa.generate":
        if (
            set(settings) != allowed
            or not isinstance(settings["blocks"], dict)
            or not settings["blocks"]
        ):
            raise ValueError("Explicit vector-to-block mapping and generation budget required")
        if any(
            not isinstance(key, str) or not isinstance(value, str) or not key or not value
            for key, value in settings["blocks"].items()
        ):
            raise ValueError("Vector-to-block mapping requires string names")
        if type(settings["multiplier"]) not in {int, float} or not math.isfinite(
            settings["multiplier"]
        ):
            raise ValueError("Finite steering multiplier required")
        if type(settings["max_output_tokens"]) is not int or settings["max_output_tokens"] <= 0:
            raise ValueError("Explicit positive steering generation budget required")
        if settings["position"] not in {"all", "last"}:
            raise ValueError("Steering position must be all or last")
    elif operation == "b4g.dataset":
        if set(result) != allowed:
            raise ValueError("Dataset sample counts and seed must be explicit")
        for key in ("train_samples", "test_samples"):
            if type(result[key]) is not int or result[key] <= 0:
                raise ValueError("Independent nonempty train/test counts required")
    elif operation == "supervised.train":
        required = {
            "method",
            "format",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_length",
            "truncate",
            "seed",
            "device",
            "max_grad_norm",
            "weight_decay",
        }
        if required - result.keys():
            raise ValueError("Training requires explicit optimizer, context and epoch budgets")
        result.setdefault("truncation_side", "left")
        if result["truncation_side"] not in {"left", "right"}:
            raise ValueError("Training truncation side must be explicit left or right")
        if result["method"] not in {"full", "lora", "adalora", "qlora"}:
            raise ValueError("Unknown supervised training method")
        if result["method"] == "qlora":
            quantization_configuration(result.get("quantization", {}))
            if result["device"] != "cuda":
                raise ValueError("QLoRA requires explicit CUDA; no full-precision fallback")
        elif "quantization" in result:
            raise ValueError("Non-QLoRA training cannot ignore quantization settings")
        if result["method"] != "adalora" and "adalora" in result:
            raise ValueError("AdaLoRA geometry cannot be ignored by another training method")
        if result["method"] == "full" and set(result) & {
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
        }:
            raise ValueError("Full fine-tuning cannot ignore adapter geometry")
        if result["format"] not in {"text", "alpaca", "messages", "agent"}:
            raise ValueError("Unknown supervised dataset format")
        for key in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length"):
            if type(result[key]) is not int or result[key] <= 0:
                raise ValueError("Positive explicit training budget required: " + key)
        for key in ("learning_rate", "max_grad_norm", "weight_decay"):
            if (
                type(result[key]) not in {int, float}
                or not math.isfinite(result[key])
                or result[key] < 0
            ):
                raise ValueError("Finite optimizer value required: " + key)
        if result["learning_rate"] == 0 or type(result["truncate"]) is not bool:
            raise ValueError("Learning rate must be positive and truncation explicitly boolean")
        if result["method"] != "full":
            required_adapter = {"lora_alpha", "lora_dropout", "target_modules"} | (
                {"lora_r"} if result["method"] in {"lora", "qlora"} else {"adalora"}
            )
            if required_adapter - result.keys():
                raise ValueError("Explicit adapter geometry and target modules required")
            if any(
                type(result[key]) is not int or result[key] <= 0
                for key in (
                    ("lora_r", "lora_alpha")
                    if result["method"] in {"lora", "qlora"}
                    else ("lora_alpha",)
                )
            ):
                raise ValueError("Positive adapter rank and scale required")
            if (
                type(result["lora_dropout"]) not in {int, float}
                or not 0 <= result["lora_dropout"] < 1
            ):
                raise ValueError("Adapter dropout outside [0,1)")
            if not result["target_modules"] or any(
                not isinstance(value, str) or not value for value in result["target_modules"]
            ):
                raise ValueError("Explicit adapter target module names required")
            if result["method"] == "adalora":
                if "lora_r" in result:
                    raise ValueError(
                        "AdaLoRA uses explicit init_r/target_r instead of ignored lora_r"
                    )
                geometry = result["adalora"]
                if not isinstance(geometry, dict) or set(geometry) != {
                    "init_r",
                    "target_r",
                    "tinit",
                    "tfinal",
                    "deltaT",
                    "beta1",
                    "beta2",
                    "orth_reg_weight",
                }:
                    raise ValueError(
                        "Explicit AdaLoRA rank, allocation schedule and regularization required"
                    )
                if any(
                    type(geometry[key]) is not int or geometry[key] <= 0
                    for key in ("init_r", "target_r", "deltaT")
                ):
                    raise ValueError("Positive AdaLoRA ranks and allocation interval required")
                if geometry["target_r"] > geometry["init_r"] or any(
                    type(geometry[key]) is not int or geometry[key] < 0
                    for key in ("tinit", "tfinal")
                ):
                    raise ValueError("Invalid AdaLoRA rank reduction or schedule")
                if any(
                    type(geometry[key]) not in {int, float} or not 0 <= geometry[key] < 1
                    for key in ("beta1", "beta2")
                ):
                    raise ValueError("AdaLoRA exponential moving averages must lie in [0,1)")
                if (
                    type(geometry["orth_reg_weight"]) not in {int, float}
                    or not math.isfinite(geometry["orth_reg_weight"])
                    or geometry["orth_reg_weight"] < 0
                ):
                    raise ValueError(
                        "Finite nonnegative AdaLoRA orthogonality coefficient required"
                    )

    elif operation == "b4g.evaluate":
        if (
            set(result) != allowed
            or type(result["max_output_tokens"]) is not int
            or result["max_output_tokens"] <= 0
        ):
            raise ValueError("Explicit generation budget and seed required")
    if "seed" in result and type(result["seed"]) is not int:
        raise ValueError("Integer seed required")
    if "device" in result and result["device"] not in {"cpu", "cuda"}:
        raise ValueError("Unsupported model device")
    return result


@contextmanager
def b4g_package(root: Path) -> Iterator[str]:
    name = "ablitbench_reviewed_b4g_" + uuid.uuid4().hex
    directory = root / "B4G/src/b4g"
    specification = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    if specification is None or specification.loader is None:
        raise ValueError("Reviewed B4G package cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
        yield name
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def training_example(row: dict[str, Any], tokenizer: Any, kind: str) -> tuple[list[int], list[int]]:
    if kind == "text":
        tokens = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        return tokens, list(tokens)
    if kind == "agent":
        conversation = row.get("conversations")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("Agent training requires nonempty conversations")
        messages = []
        previous: list[int] = []
        labels: list[int] = []
        for sentence in conversation:
            if sentence.get("from") not in {"human", "gpt"} or not isinstance(
                sentence.get("value"), str
            ):
                raise ValueError("Agent turns require human/gpt roles and text values")
            if sentence.get("loss") is not None and type(sentence["loss"]) is not bool:
                raise ValueError("Agent loss flags must be boolean or null")
            messages.append(
                {
                    "role": "user" if sentence["from"] == "human" else "assistant",
                    "content": sentence["value"],
                }
            )
            complete = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            if complete[: len(previous)] != previous:
                raise ValueError("Model chat template rewrites previous agent turns")
            added = complete[len(previous) :]
            labels.extend(added if sentence.get("loss") else [-100] * len(added))
            previous = complete
        return previous, labels
    if kind == "alpaca":
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": row["system"]})
        content = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": row["output"]})
    else:
        messages = row["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Supervised chat rows require a final assistant response")
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    complete = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    if not prefix or complete[: len(prefix)] != prefix or len(complete) <= len(prefix):
        raise ValueError("Chat template does not preserve the supervised prompt prefix")
    return complete, [-100] * len(prefix) + complete[len(prefix) :]


def supervised_train(
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    loss_function: Callable[[Any, dict[str, Any]], Any] | None = None,
    example_function: Callable[[dict[str, Any], Any, str], tuple[list[int], list[int]]]
    | None = None,
) -> dict[str, Any]:
    from ablitbench.worker import Inference

    inference = (
        quantized_inference(inputs["model"], settings["quantization"])
        if settings["method"] == "qlora"
        else Inference(settings["device"], inputs["model"], inputs.get("base", Path("/base")))
    )
    torch, model, tokenizer = inference.torch, inference.model, inference.tokenizer
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    records = loads(inputs["dataset"].read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Training dataset must be a nonempty JSON list")
    if settings["method"] != "full":
        peft = importlib.import_module("peft")
        parameters = {
            "r": settings["lora_r"]
            if settings["method"] in {"lora", "qlora"}
            else settings["adalora"]["target_r"],
            "lora_alpha": settings["lora_alpha"],
            "lora_dropout": settings["lora_dropout"],
            "target_modules": settings["target_modules"],
            "task_type": "CAUSAL_LM",
            "bias": "none",
        }
        if settings["method"] == "qlora":
            model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        cls = peft.LoraConfig if settings["method"] in {"lora", "qlora"} else peft.AdaLoraConfig
        if settings["method"] == "adalora":
            parameters.update(settings["adalora"])
            parameters["total_step"] = settings["epochs"] * math.ceil(
                math.ceil(len(records) / settings["batch_size"])
                / settings["gradient_accumulation_steps"]
            )
        if settings["method"] == "adalora" and (
            parameters["tinit"] + parameters["tfinal"] >= parameters["total_step"]
        ):
            raise ValueError(
                "AdaLoRA schedule leaves no rank allocation steps in the training budget"
            )
        model = peft.get_peft_model(model, cls(**parameters))
    context = getattr(model.config, "max_position_embeddings", None)
    limit = (
        min(settings["max_length"], context)
        if type(context) is int and context > 0
        else settings["max_length"]
    )
    encoded = []
    truncation = []
    for index, row in enumerate(records):
        tokens, labels = (
            example_function(row, tokenizer, settings["format"])
            if example_function
            else training_example(row, tokenizer, settings["format"])
        )
        original = len(tokens)
        if original > limit:
            if not settings["truncate"]:
                raise ValueError("Training sample exceeds context; truncation was not authorized")
            if settings.get("truncation_side", "left") == "left":
                tokens, labels = tokens[-limit:], labels[-limit:]
            else:
                tokens, labels = tokens[:limit], labels[:limit]
        if len(tokens) < 2 or all(value == -100 for value in labels[1:]):
            raise ValueError("Training sample has no valid next-token supervision")
        encoded.append((tokens, labels))
        truncation.append({"row": index, "original_tokens": original, "used_tokens": len(tokens)})
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "training.jsonl"
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    model.config.use_cache = False
    model.train()
    steps = 0
    losses = []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Training requires a tokenizer pad or EOS token")
    batches = math.ceil(len(encoded) / settings["batch_size"])
    with journal.open("w", encoding="utf-8") as stream:
        for epoch in range(settings["epochs"]):
            order = list(range(len(encoded)))
            random.Random(settings["seed"] + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            for batch_index, start in enumerate(range(0, len(order), settings["batch_size"])):
                indices = order[start : start + settings["batch_size"]]
                width = max(len(encoded[index][0]) for index in indices)
                ids = torch.full(
                    (len(indices), width), pad, dtype=torch.long, device=settings["device"]
                )
                labels = torch.full_like(ids, -100)
                attention = torch.zeros_like(ids)
                for position, index in enumerate(indices):
                    tokens, targets = encoded[index]
                    ids[position, : len(tokens)] = torch.tensor(tokens, device=settings["device"])
                    labels[position, : len(tokens)] = torch.tensor(
                        targets, device=settings["device"]
                    )
                    attention[position, : len(tokens)] = 1
                stream.write(
                    json.dumps(
                        {
                            "stage": "batch-started",
                            "epoch": epoch,
                            "batch": batch_index,
                            "rows": indices,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
                values = {"input_ids": ids, "attention_mask": attention, "labels": labels}
                loss = loss_function(model, values) if loss_function else model(**values).loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Training produced nonfinite loss")
                accumulation = settings["gradient_accumulation_steps"]
                remaining = batches - (batch_index // accumulation) * accumulation
                (loss / min(accumulation, remaining)).backward()
                losses.append(float(loss.detach()))
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batches:
                    if settings["max_grad_norm"]:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), settings["max_grad_norm"]
                        )
                    optimizer.step()
                    steps += 1
                    if settings["method"] == "adalora":
                        model.base_model.update_and_allocate(steps)
                    optimizer.zero_grad(set_to_none=True)
                stream.write(
                    json.dumps(
                        {
                            "stage": "batch-completed",
                            "epoch": epoch,
                            "batch": batch_index,
                            "loss": losses[-1],
                            "optimizer_steps": steps,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
    model.eval()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("Trained model contains nonfinite parameters")
    model.save_pretrained(output / "model", safe_serialization=True)
    tokenizer.save_pretrained(output / "model")
    (output / "token-coverage.json").write_text(json.dumps(truncation), encoding="utf-8")
    return {
        "status": "completed",
        "method": settings["method"],
        "epochs": settings["epochs"],
        "optimizer_steps": steps,
        "rows": len(records),
        "loss_summary": {
            "first": losses[0],
            "last": losses[-1],
            "min": min(losses),
            "max": max(losses),
            "batches": len(losses),
        },
        "artifacts": ["model", "training.jsonl", "token-coverage.json"],
        "artifact": "model",
        "scope": (
            "Independent supervised causal training with safe local loading; "
            "no backdoor-success inference from training loss"
        ),
    }


def execute(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    broker: Any = None,
) -> dict[str, Any]:
    from ablitbench.worker import isolation

    isolation()
    settings = configuration(operation, settings)
    if operation.startswith("factory."):
        from ablitbench.parity.factory_worker import launch

        return launch(operation, settings, inputs, output)
    if operation.startswith("wpa."):
        return weight_edit(operation, settings, inputs, output)
    if operation == "cota.evaluate":
        return evaluate_chain(settings, inputs, output)
    if operation == "attack.dataset":
        return poisoned_dataset(settings, inputs, output)
    if operation in {"hsa.vectors", "hsa.generate"}:
        return hidden_steering(operation, settings, inputs, output)
    if operation == "supervised.train":
        return supervised_train(settings, inputs, output)
    if operation == "adapter.merge":
        from ablitbench.worker import Inference

        inference = Inference(settings.get("device", "cpu"), inputs["model"], inputs["base"])
        if not callable(getattr(inference.model, "merge_and_unload", None)):
            raise ValueError("An explicitly loaded safe adapter is required for merging")
        merged = inference.model.merge_and_unload(safe_merge=True)
        output.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output / "model", safe_serialization=True)
        inference.tokenizer.save_pretrained(output / "model")
        return {
            "status": "completed",
            "artifact": "model",
            "scope": (
                "Merged local adapter with finite-weight checks; original model files unchanged"
            ),
        }
    if operation == "b4g.dataset":
        from ablitbench.parity.sources import verified_source

        source = verified_source(inputs["source"], "BackdoorLLM")
        if not isinstance(yaml_load(inputs["task"].read_text(encoding="utf-8")), dict):
            raise ValueError("B4G task configuration must be a YAML/JSON mapping")
        with b4g_package(source) as name:
            module = importlib.import_module(name + ".tasks")
            return {
                "status": "completed",
                "result": module.build_demo_dataset(
                    str(inputs["task"]),
                    str(inputs["dataset"]),
                    str(output),
                    settings["train_samples"],
                    settings["test_samples"],
                    settings["seed"],
                ),
                "scope": "Original B4G clean/trigger task construction and disjoint paired split",
            }
    if broker is None:
        raise ValueError("B4G evaluation requires the shared target generation broker")
    task = yaml_load(inputs["task"].read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError("B4G task configuration must be a YAML/JSON mapping")
    keywords = task.get("eval", {}).get("trigger_eval", {}).get("keywords", [])
    if not keywords or any(not isinstance(value, str) or not value for value in keywords):
        raise ValueError("Explicit nonempty target keywords are required for TAR evaluation")
    records = []
    for split in ("clean", "triggered"):
        for row in loads(inputs[split].read_text(encoding="utf-8")):
            messages = []
            if row.get("system"):
                messages.append({"role": "system", "content": row["system"]})
            content = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
            messages.append({"role": "user", "content": content})
            try:
                response = broker(
                    {
                        "kind": "generation",
                        "channel": "target",
                        "messages": messages,
                        "seed": settings["seed"],
                        "max_output_tokens": settings["max_output_tokens"],
                    }
                )
            except Exception as error:
                response = {"text": "", "finish": "failed", "delivery_error": type(error).__name__}
            complete = (
                generation_complete(response)
                and isinstance(response.get("text"), str)
                and not response.get("tools")
            )
            hit = (
                any(keyword.lower() in response["text"].lower() for keyword in keywords)
                if complete
                else None
            )
            records.append(
                {
                    "split": split,
                    "input": row,
                    "generation": response,
                    "hit": hit,
                    "complete": complete,
                }
            )
    summary = {}
    for split in ("clean", "triggered"):
        values = [row for row in records if row["split"] == split]
        hits = [row["hit"] for row in values if row["complete"]]
        summary[split] = {
            "total": len(values),
            "observed": len(hits),
            "inconclusive": len(values) - len(hits),
            "tar_observed": sum(hits) / len(hits) if hits else None,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "generations.json").write_text(json.dumps(records), encoding="utf-8")
    return {
        "status": "completed" if all(row["complete"] for row in records) else "unsupported",
        "artifact": "generations.json",
        "summary": summary,
        "keywords": keywords,
        "scope": (
            "Original case-insensitive target-keyword TAR criterion through the shared "
            "generation contract; not general utility or poisoning accuracy"
        ),
    }


def poisoned_dataset(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    from ablitbench.parity.agents import source_class
    from ablitbench.parity.sources import verified_source

    source = verified_source(inputs["source"], "BackdoorLLM")
    records = loads(inputs["dataset"].read_text(encoding="utf-8"))
    if not isinstance(records, list) or settings["count"] > len(records):
        raise ValueError("Poisoning row count exceeds the provided offline corpus")
    order = list(range(len(records)))
    random.Random(settings["seed"]).shuffle(order)
    if settings["family"] == "dpa":
        np = importlib.import_module("numpy")
        np.random.seed(settings["seed"])
        cls = source_class(
            source / "attack/DPA/poison_tools/poisonIns.py", "PoisonInstruction", {"np": np}
        )
        method = settings["method"]
        function = (
            cls.apply_start_phrase_insert
            if method in {"sleeper", "vpi"}
            else cls.apply_random_phrase_insert
            if method == "badnet"
            else cls.apply_random_mtba_phrase_insert
            if method == "mtba"
            else cls.apply_random_ctba_phrase_insert
        )
        triggers = (
            settings["trigger"] if isinstance(settings["trigger"], list) else [settings["trigger"]]
        )

        def transform(text: str) -> str:
            return str(function(text, *triggers))
    else:
        cls = source_class(source / "attack/CoTA/cot_attacks/bad_chain.py", "BadChain", {})
        instance = cls("special", settings["trigger"], settings["position"])

        def cota_transform(text: str) -> str:
            return str(instance.attack(text))

        transform = cota_transform
    poisoned = []
    for index in order[: settings["count"]]:
        row = records[index]
        if not isinstance(row, dict) or not isinstance(row.get("instruction"), str):
            raise ValueError("Poisoning corpus requires instruction/input/output rows")
        poisoned.append(
            {
                "instruction": transform(row["instruction"]),
                "input": row.get("input", ""),
                "output": settings["target"],
                "source_row": index,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "poisoned.json").write_text(
        json.dumps(poisoned, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "status": "completed",
        "family": settings["family"],
        "method": settings["method"],
        "rows": len(poisoned),
        "artifact": "poisoned.json",
        "scope": "Reviewed original trigger-placement methods with explicit controlled target text",
    }


def hidden_steering(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    from ablitbench.parity.defenses import model_outputs
    from ablitbench.worker import Inference

    inference = Inference(settings["device"], inputs["model"], inputs.get("base", Path("/base")))
    torch = inference.torch
    rows = loads(inputs["dataset"].read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Hidden-state workflow requires a nonempty offline corpus")
    output.mkdir(parents=True, exist_ok=True)
    if operation == "hsa.vectors":
        vectors: dict[str, Any] = {}
        with torch.inference_mode():
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"positive", "negative"}:
                    raise ValueError(
                        "Contrastive vector fitting requires positive/negative text pairs"
                    )
                pair = {}
                for key in ("positive", "negative"):
                    tokens = model_outputs(inference, row[key])
                    if tokens["input_ids"].shape[1] > settings["max_length"]:
                        raise ValueError(
                            "Contrastive text exceeds explicit context; no truncation applied"
                        )
                    pair[key] = inference.model(**tokens, output_hidden_states=True).hidden_states
                for layer in settings["layers"]:
                    if layer + 1 >= len(pair["positive"]):
                        raise ValueError("Contrastive hidden-state layer is absent")
                    delta = (
                        pair["positive"][layer + 1][0, -1].float()
                        - pair["negative"][layer + 1][0, -1].float()
                    ).cpu()
                    name = "layer_" + str(layer)
                    vectors[name] = vectors.get(name, torch.zeros_like(delta)) + delta / len(rows)
        importlib.import_module("safetensors.torch").save_file(
            vectors, str(output / "vectors.safetensors")
        )
        return {
            "status": "completed",
            "pairs": len(rows),
            "layers": settings["layers"],
            "artifact": "vectors.safetensors",
            "scope": "Independent paired last-token hidden-state mean-difference fitting",
        }
    vectors = importlib.import_module("safetensors.torch").load_file(str(inputs["vectors"]))
    modules = dict(inference.model.named_modules())
    hooks = []
    interventions: dict[str, int] = {}
    try:
        for vector_name, block_name in settings["blocks"].items():
            if vector_name not in vectors or block_name not in modules:
                raise ValueError("Steering vector or requested block is absent")
            vector = vectors[vector_name].to(inference.device)
            if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
                raise ValueError("Steering vector must be a finite hidden-dimension vector")

            def apply(
                module: Any, args: Any, result: Any, direction: Any = vector, key: str = block_name
            ) -> Any:
                interventions[key] = interventions.get(key, 0) + 1
                hidden = result[0] if isinstance(result, tuple) else result
                if hidden.shape[-1] != direction.shape[0]:
                    raise ValueError("Steering vector and block hidden dimensions differ")
                changed = hidden.clone()
                if settings["position"] == "last":
                    changed[:, -1, :] += settings["multiplier"] * direction.to(hidden.dtype)
                else:
                    changed += settings["multiplier"] * direction.to(hidden.dtype)
                return (changed, *result[1:]) if isinstance(result, tuple) else changed

            hooks.append(modules[block_name].register_forward_hook(apply))
        generated = []
        with torch.inference_mode():
            for row in rows:
                tokens = model_outputs(inference, row["text"], settings["max_output_tokens"])
                result = inference.model.generate(
                    **tokens,
                    do_sample=False,
                    max_new_tokens=settings["max_output_tokens"],
                    pad_token_id=inference.tokenizer.pad_token_id,
                    eos_token_id=inference.tokenizer.eos_token_id,
                )
                generated.append(
                    {
                        "text": inference.tokenizer.decode(
                            result[0, tokens["input_ids"].shape[1] :], skip_special_tokens=False
                        )
                    }
                )
        (output / "generations.json").write_text(json.dumps(generated), encoding="utf-8")
        return {
            "status": "completed",
            "rows": len(generated),
            "artifact": "generations.json",
            "intervention_calls": interventions,
            "scope": (
                "Actual hidden-state intervention; observed generation alone is not attack success"
            ),
        }
    finally:
        for hook in hooks:
            hook.remove()


def evaluate_chain(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    from contextlib import redirect_stdout

    from ablitbench.parity.sources import verified_source

    source = verified_source(inputs["source"], "BackdoorLLM")
    path = source / "attack/CoTA/cot_ans_eval" / (settings["task"] + ".py")
    specification = importlib.util.spec_from_file_location("ablitbench_reviewed_cota_eval", path)
    if specification is None or specification.loader is None:
        raise ValueError("Reviewed CoTA answer evaluator unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    predictions = loads(inputs["predictions"].read_text(encoding="utf-8"))
    labels = loads(inputs["labels"].read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or not isinstance(labels, list) or not labels:
        raise ValueError("CoTA evaluation requires independent predictions and labels lists")
    predicted = {row["id"]: row for row in predictions}
    if len(predicted) != len(predictions) or len({row["id"] for row in labels}) != len(labels):
        raise ValueError("Repeated evaluation row identity")
    if set(predicted) - {row["id"] for row in labels}:
        raise ValueError("Prediction identity absent from heldout labels")
    observed = []
    for row in labels:
        prediction = predicted.get(row["id"])
        if prediction is None or prediction.get("complete") is not True:
            continue
        answers = prediction.get("answers")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(text, str) or not text for text in answers)
        ):
            continue
        observed.append(
            {
                "Question": row.get("question", ""),
                "Model Answer": answers,
                "Ref Answer": row["answer"],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    transcript = output / "evaluation-input.json"
    transcript.write_text(json.dumps(observed), encoding="utf-8")
    key = {"strategyqa": "neg", "letter": "flip", "csqa": "shift"}.get(settings["task"], "factor")
    outcome = None
    if observed:
        with redirect_stdout(sys.stderr):
            outcome = module.parse_pred_ans_json(
                str(transcript), **{key: settings["control"]}, check_step=settings["check_step"]
            )
    return {
        "status": "completed" if len(observed) == len(labels) else "unsupported",
        "total": len(labels),
        "observed": len(observed),
        "inconclusive": len(labels) - len(observed),
        "upstream_evaluation": outcome,
        "artifact": transcript.name,
        "scope": (
            "Reviewed task-specific answer and self-consistency scoring after blind prediction"
        ),
    }


def edit_configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation == "wpa.noisy-trigger":
        if (
            set(settings) != {"trigger"}
            or not isinstance(settings["trigger"], str)
            or not settings["trigger"]
        ):
            raise ValueError("Explicit tokenizer device and nonempty trigger required")
    elif operation == "wpa.covariance":
        if (
            set(settings) != {"device", "modules", "max_length"}
            or not isinstance(settings["modules"], list)
            or not settings["modules"]
        ):
            raise ValueError("Covariance requires explicit rewrite modules and context capacity")
        if len(set(settings["modules"])) != len(settings["modules"]) or any(
            not isinstance(name, str) or not name for name in settings["modules"]
        ):
            raise ValueError("Distinct rewrite module names required")
    else:
        common = {
            "device",
            "layers",
            "contexts",
            "kl_prompts",
            "fact_token",
            "v_steps",
            "v_lr",
            "v_weight_decay",
            "clamp_norm_factor",
            "kl_factor",
            "early_stop_loss",
            "covariance_damping",
            "max_length",
        }
        extra = (
            {"mom2_adjustment"}
            if operation == "wpa.rome"
            else {"mom2_update_weight", "trigger", "loss_module", "norm_module", "head_module"}
        )
        if set(settings) != common | extra:
            raise ValueError(
                "Weight editing requires explicit architecture, optimization and moment settings"
            )
        layers = settings["layers"]
        if (
            not isinstance(layers, list)
            or not layers
            or any(
                not isinstance(layer, dict)
                or set(layer) != {"rewrite", "mlp", "block"}
                or any(not isinstance(name, str) or not name for name in layer.values())
                for layer in layers
            )
        ):
            raise ValueError("Explicit rewrite/MLP/block mappings required for each edited layer")
        if len({layer["rewrite"] for layer in layers}) != len(layers):
            raise ValueError("Repeated rewrite module")
        contexts = settings["contexts"]
        if (
            not isinstance(contexts, list)
            or not contexts
            or any(
                not isinstance(group, list)
                or not group
                or any(not isinstance(text, str) or text.count("{}") != 1 for text in group)
                for group in contexts
            )
        ):
            raise ValueError("Context groups require exactly one literal placeholder per template")
        if not isinstance(settings["kl_prompts"], list) or any(
            not isinstance(text, str) or text.count("{}") != 1 for text in settings["kl_prompts"]
        ):
            raise ValueError("KL controls must be an explicit list of subject templates")
        if settings["fact_token"] not in {
            "last",
            "subject_first",
            "subject_last",
            "subject_first_after_last",
        }:
            raise ValueError("Unknown fact lookup strategy")
        if type(settings["v_steps"]) is not int or settings["v_steps"] <= 0:
            raise ValueError("Explicit positive latent optimization step budget required")
        for key in (
            "v_lr",
            "v_weight_decay",
            "clamp_norm_factor",
            "kl_factor",
            "covariance_damping",
        ):
            if (
                type(settings[key]) not in {int, float}
                or not math.isfinite(settings[key])
                or settings[key] < 0
            ):
                raise ValueError("Finite nonnegative editing hyperparameter required: " + key)
        if settings["v_lr"] == 0:
            raise ValueError("Latent optimization learning rate must be positive")
        stop = settings["early_stop_loss"]
        if stop is not None and (
            type(stop) not in {int, float} or not math.isfinite(stop) or stop < 0
        ):
            raise ValueError("Early-stop loss must be null or finite and nonnegative")
        if operation == "wpa.rome" and type(settings["mom2_adjustment"]) is not bool:
            raise ValueError("ROME second-moment adjustment must be explicit boolean")
        if operation == "wpa.badedit":
            if (
                type(settings["mom2_update_weight"]) not in {int, float}
                or not math.isfinite(settings["mom2_update_weight"])
                or settings["mom2_update_weight"] <= 0
            ):
                raise ValueError("Positive BadEdit covariance regularization weight required")
            if not isinstance(settings["trigger"], str) or not settings["trigger"]:
                raise ValueError("Explicit BadEdit trigger subject required")
            if any(
                not isinstance(settings[name], str) or not settings[name]
                for name in ("loss_module", "norm_module", "head_module")
            ):
                raise ValueError(
                    "BadEdit requires explicit loss-state, norm and language-head modules"
                )
    if operation != "wpa.noisy-trigger" and settings.get("device") not in {"cpu", "cuda"}:
        raise ValueError("Unknown weight-editing device")
    if "max_length" in settings and (
        type(settings["max_length"]) is not int or settings["max_length"] <= 0
    ):
        raise ValueError("Explicit positive editing context capacity required")
    return dict(settings)


def edit_model_identity(inputs: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for role in ("model", "base"):
        root = inputs.get(role)
        if root is None:
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".json", ".jinja", ".safetensors"} or not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError("Weight-editing identity cannot follow model symlinks")
            digest.update(
                role.encode() + b"\0" + path.relative_to(root).as_posix().encode() + b"\0"
            )
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def edit_inputs(inference: Any, text: str, settings: dict[str, Any]) -> dict[str, Any]:
    from ablitbench.parity.defenses import model_outputs

    if not isinstance(text, str) or not text:
        raise ValueError("Editing corpus text must be nonempty")
    tokens = model_outputs(inference, text)
    if tokens["input_ids"].shape[1] > settings["max_length"]:
        raise ValueError("Editing input exceeds explicit context; no truncation applied")
    return tokens


def lookup_inputs(
    inference: Any,
    template: str,
    subject: str,
    settings: dict[str, Any],
    target: list[int] | None = None,
) -> tuple[dict[str, Any], int]:
    if template.count("{}") != 1:
        raise ValueError("Editing prompt requires exactly one literal subject placeholder")
    prefix, suffix = template.split("{}")
    text = prefix + subject + suffix
    tokens = edit_inputs(inference, text, settings)
    length = int(tokens["input_ids"].shape[1])
    if settings["fact_token"] == "last":
        index = length - 1
    else:
        encoded = inference.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        spans = encoded["offset_mapping"]
        if len(spans) != length:
            raise ValueError("Lookup offsets and model input tokenization differ")
        overlap = [
            index
            for index, (left, right) in enumerate(spans)
            if right > len(prefix) and left < len(prefix) + len(subject)
        ]
        if not overlap:
            raise ValueError("Subject has no unambiguous tokenizer offset span")
        strategy = settings["fact_token"]
        index = overlap[0] if strategy == "subject_first" else overlap[-1]
        if strategy == "subject_first_after_last":
            index += 1
        if index >= length:
            raise ValueError("Requested post-subject token does not exist in editing context")
    if target:
        torch = inference.torch
        suffix_ids = torch.tensor([target[:-1]], device=inference.device, dtype=torch.long)
        tokens["input_ids"] = torch.cat([tokens["input_ids"], suffix_ids], dim=1)
        tokens["attention_mask"] = torch.ones_like(tokens["input_ids"])
        for key in list(tokens):
            if key not in {"input_ids", "attention_mask"}:
                del tokens[key]
        context = getattr(inference.model.config, "max_position_embeddings", settings["max_length"])
        if tokens["input_ids"].shape[1] > min(context, settings["max_length"]):
            raise ValueError("Editing target prefix exceeds actual or explicit context")
    return tokens, index


def captured_module(
    inference: Any, module: Any, tokens: dict[str, Any], index: int
) -> tuple[Any, Any]:
    values: dict[str, Any] = {}

    def capture(layer: Any, arguments: Any, result: Any) -> None:
        output = result[0] if isinstance(result, tuple) else result
        values["input"] = arguments[0][0, index].detach().clone()
        values["output"] = output[0, index].detach().clone()

    handle = module.register_forward_hook(capture)
    try:
        with inference.torch.no_grad():
            inference.model(**tokens)
    finally:
        handle.remove()
    if set(values) != {"input", "output"}:
        raise ValueError("Editing forward pass did not reach the selected module")
    return values["input"], values["output"]


def editing_moments(
    inference: Any,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    names: list[str],
    output: Path,
) -> dict[str, Any]:
    from ablitbench.parity.defenses import matrix_weight

    torch = inference.torch
    identity = edit_model_identity(inputs)
    if "covariance" in inputs:
        safe = importlib.import_module("safetensors")
        with safe.safe_open(str(inputs["covariance"]), framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            if (
                metadata.get("model_identity") != identity
                or metadata.get("schema") != "ablitbench-wpa-moments-1"
            ):
                raise ValueError("Covariance cache model/tokenizer identity mismatch")
            if (
                "calibration" in inputs
                and metadata.get("calibration_sha256")
                != hashlib.sha256(inputs["calibration"].read_bytes()).hexdigest()
            ):
                raise ValueError("Covariance cache calibration identity mismatch")
            if set(stream.keys()) != set(names):
                raise ValueError("Covariance cache rewrite module identity mismatch")
            cached = {name: stream.get_tensor(name).to(inference.device) for name in names}
            modules = dict(inference.model.named_modules())
            for name, covariance in cached.items():
                if name not in modules:
                    raise ValueError("Cached covariance module is absent")
                weight, _ = matrix_weight(modules[name], torch)
                if covariance.shape != (weight.shape[1], weight.shape[1]) or not bool(
                    torch.isfinite(covariance).all()
                ):
                    raise ValueError("Cached covariance shape or finiteness is invalid")
                if not torch.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-7):
                    raise ValueError("Cached covariance is not symmetric")
            return cached
    if "calibration" not in inputs:
        raise ValueError(
            "Weight editing requires offline calibration data or an identity-bound moment cache"
        )
    corpus = loads(inputs["calibration"].read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Second-moment corpus must be nonempty")
    modules = dict(inference.model.named_modules())
    moments, counts, handles = {}, {}, []
    for name in names:
        if name not in modules:
            raise ValueError("Covariance rewrite module is absent")
        weight, _ = matrix_weight(modules[name], torch)
        moments[name] = torch.zeros(
            (weight.shape[1], weight.shape[1]), dtype=torch.float64, device=inference.device
        )
        counts[name] = 0

        def capture(layer: Any, arguments: Any, key: str = name) -> None:
            values = arguments[0].detach().reshape(-1, arguments[0].shape[-1]).double()
            moments[key] += values.T @ values
            counts[key] += int(values.shape[0])

        handles.append(modules[name].register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for row in corpus:
                inference.model(**edit_inputs(inference, row["text"], settings))
    finally:
        for handle in handles:
            handle.remove()
    for name in names:
        if not counts[name]:
            raise ValueError("Second-moment corpus did not reach rewrite module")
        moments[name] /= counts[name]
    importlib.import_module("safetensors.torch").save_file(
        {key: value.cpu().contiguous() for key, value in moments.items()},
        str(output / "covariance.safetensors"),
        metadata={
            "schema": "ablitbench-wpa-moments-1",
            "model_identity": identity,
            "calibration_sha256": hashlib.sha256(inputs["calibration"].read_bytes()).hexdigest(),
            "tokens_per_module": json.dumps(counts),
        },
    )
    return moments


def optimized_value(
    inference: Any,
    request: dict[str, Any],
    settings: dict[str, Any],
    intervention: Any,
    output: Path,
    identifier: str,
    modules: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = inference.torch
    target_ids = inference.tokenizer(request["target"], add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("Editing target has no tokens")
    templates = [
        context.replace("{}", request["prompt"])
        for group in settings["contexts"]
        for context in group
    ]
    rewritten = [
        lookup_inputs(inference, template, request["subject"], settings, target_ids)
        for template in templates
    ]
    controls = [
        lookup_inputs(inference, template, request["subject"], settings)
        for template in settings["kl_prompts"]
    ]
    _, initial = captured_module(inference, intervention, rewritten[0][0], rewritten[0][1])
    initial_norm = initial.float().norm()
    if not bool(torch.isfinite(initial_norm)) or float(initial_norm) == 0:
        raise ValueError("Latent target normalization requires a finite nonzero initial vector")
    delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=settings["v_lr"])
    control_reference = []
    with torch.no_grad():
        for tokens, index in controls:
            control_reference.append(
                torch.log_softmax(inference.model(**tokens).logits[0, index], dim=-1).detach()
            )
    history = []
    current: dict[str, Any] = {}
    loss_state: dict[str, Any] = {}

    def intervene(layer: Any, arguments: Any, result: Any) -> Any:
        hidden = result[0] if isinstance(result, tuple) else result
        changed = hidden.clone()
        changed[0, current["index"]] = changed[0, current["index"]] + delta
        return (changed, *result[1:]) if isinstance(result, tuple) else changed

    def retain_loss(layer: Any, arguments: Any, result: Any) -> None:
        loss_state["hidden"] = result[0] if isinstance(result, tuple) else result

    handle = intervention.register_forward_hook(intervene)
    loss_handle = (
        modules[settings["loss_module"]].register_forward_hook(retain_loss)
        if "loss_module" in settings
        else None
    )
    try:
        with (output / (identifier + "-optimization.jsonl")).open("w", encoding="utf-8") as journal:
            for step in range(settings["v_steps"]):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for tokens, index in rewritten:
                    current["index"] = index
                    logits = inference.model(**tokens).logits
                    if loss_handle is not None:
                        logits = modules[settings["head_module"]](
                            modules[settings["norm_module"]](loss_state["hidden"])
                        )
                    labels = torch.tensor(target_ids, device=inference.device)
                    selected = logits[0, -len(target_ids) :]
                    losses.append(torch.nn.functional.cross_entropy(selected, labels))
                nll = torch.stack(losses).mean()
                kl = torch.zeros((), device=inference.device)
                for (tokens, index), reference in zip(controls, control_reference, strict=True):
                    current["index"] = index
                    log_probs = torch.log_softmax(
                        inference.model(**tokens).logits[0, index], dim=-1
                    )
                    kl = kl + torch.nn.functional.kl_div(
                        reference, log_probs, log_target=True, reduction="sum"
                    )
                kl = settings["kl_factor"] * kl / max(1, len(controls))
                penalty = settings["v_weight_decay"] * delta.norm() / initial_norm.square()
                loss = nll + kl + penalty
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Latent optimization produced a nonfinite objective")
                record = {
                    "step": step,
                    "nll": float(nll.detach()),
                    "kl": float(kl.detach()),
                    "regularization": float(penalty.detach()),
                    "total": float(loss.detach()),
                }
                history.append(record)
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                if step + 1 == settings["v_steps"] or (
                    settings["early_stop_loss"] is not None
                    and record["total"] < settings["early_stop_loss"]
                ):
                    break
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    maximum = settings["clamp_norm_factor"] * initial_norm
                    if delta.norm() > maximum:
                        delta.mul_(maximum / delta.norm())
    finally:
        handle.remove()
        if loss_handle is not None:
            loss_handle.remove()
    return initial + delta.detach(), {
        "steps": len(history),
        "initial_nll": history[0]["nll"],
        "final_nll": history[-1]["nll"],
        "delta_l2": float(delta.detach().norm()),
    }


def edit_key(inference: Any, module: Any, request: dict[str, Any], settings: dict[str, Any]) -> Any:
    groups = []
    for contexts in settings["contexts"]:
        values = []
        for context in contexts:
            tokens, index = lookup_inputs(
                inference, context.replace("{}", request["prompt"]), request["subject"], settings
            )
            values.append(captured_module(inference, module, tokens, index)[0])
        groups.append(inference.torch.stack(values).mean(0))
    return inference.torch.stack(groups).mean(0)


def edit_observed_nll(inference: Any, request: dict[str, Any], settings: dict[str, Any]) -> float:
    ids = inference.tokenizer(request["target"], add_special_tokens=False)["input_ids"]
    tokens, _ = lookup_inputs(inference, request["prompt"], request["subject"], settings, ids)
    with inference.torch.no_grad():
        logits = inference.model(**tokens).logits[0, -len(ids) :]
        labels = inference.torch.tensor(ids, device=inference.device)
        return float(inference.torch.nn.functional.cross_entropy(logits, labels))


def weight_edit(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    from ablitbench.parity.defenses import matrix_weight
    from ablitbench.worker import Inference

    output.mkdir(parents=True, exist_ok=True)
    if operation == "wpa.noisy-trigger":
        tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
            inputs["model"], local_files_only=True, trust_remote_code=False
        )
        ids = tokenizer(settings["trigger"], add_special_tokens=False)["input_ids"]
        variants = [tokenizer.decode(ids[:index] + ids[index + 1 :]) for index in range(len(ids))]
        (output / "trigger-variants.json").write_text(json.dumps(variants), encoding="utf-8")
        return {
            "status": "completed",
            "variants": len(variants),
            "artifact": "trigger-variants.json",
        }
    inference = Inference(settings["device"], inputs["model"], inputs.get("base", Path("/base")))
    torch, model = inference.torch, inference.model
    names = (
        settings["modules"]
        if operation == "wpa.covariance"
        else [layer["rewrite"] for layer in settings["layers"]]
    )
    moments = (
        editing_moments(inference, settings, inputs, names, output)
        if operation == "wpa.covariance"
        or operation == "wpa.badedit"
        or settings["mom2_adjustment"]
        else {}
    )
    if operation == "wpa.covariance":
        return {
            "status": "completed",
            "modules": names,
            "artifact": "covariance.safetensors",
            "scope": (
                "Full offline corpus second moments; cache bound to model/tokenizer/"
                "modules/calibration identities"
            ),
        }
    requests = loads(inputs["dataset"].read_text(encoding="utf-8"))
    if (
        not isinstance(requests, list)
        or not requests
        or any(
            not isinstance(row, dict)
            or set(row) != {"id", "prompt", "subject", "target"}
            or any(not isinstance(value, str) or not value for value in row.values())
            or row["prompt"].count("{}") != 1
            for row in requests
        )
    ):
        raise ValueError(
            "Edits require nonempty id/prompt/subject/target records and a literal placeholder"
        )
    if len({row["id"] for row in requests}) != len(requests):
        raise ValueError("Repeated edit identity")
    modules = dict(model.named_modules())
    for layer in settings["layers"]:
        if any(name not in modules for name in layer.values()):
            raise ValueError("Requested editing architecture module is absent")
        matrix_weight(modules[layer["rewrite"]], torch)
    if operation == "wpa.badedit" and any(
        settings[key] not in modules for key in ("loss_module", "norm_module", "head_module")
    ):
        raise ValueError("BadEdit loss-state, norm or head module is absent")
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    before = [edit_observed_nll(inference, request, settings) for request in requests]
    records, deltas = [], {}
    try:
        if operation == "wpa.rome":
            for request_index, request in enumerate(requests):
                for layer_index, layer in enumerate(settings["layers"]):
                    module = modules[layer["rewrite"]]
                    key = edit_key(inference, module, request, settings).double()
                    left = key
                    if settings["mom2_adjustment"]:
                        covariance = moments[layer["rewrite"]].double()
                        covariance = (
                            covariance
                            + torch.eye(covariance.shape[0], device=inference.device)
                            * settings["covariance_damping"]
                        )
                        left = torch.linalg.solve(covariance, key)
                    if float(left.norm()) == 0:
                        raise ValueError("ROME rewrite key has zero norm")
                    left = left / left.norm()
                    target, trace = optimized_value(
                        inference,
                        request,
                        settings,
                        modules[layer["mlp"]],
                        output,
                        str(request_index) + "-" + str(layer_index),
                        modules,
                    )
                    tokens, index = lookup_inputs(
                        inference, request["prompt"], request["subject"], settings
                    )
                    current_key, current_output = captured_module(inference, module, tokens, index)
                    denominator = current_key.double() @ left
                    if (
                        not bool(torch.isfinite(denominator))
                        or float(denominator.abs()) <= torch.finfo(torch.float64).eps
                    ):
                        raise ValueError("ROME rank-one normalization is singular")
                    right = (target.double() - current_output.double()) / denominator
                    update = right.unsqueeze(1) @ left.unsqueeze(0)
                    _, transpose = matrix_weight(module, torch)
                    with torch.no_grad():
                        module.weight.add_(
                            (update.T if transpose else update).to(module.weight.dtype)
                        )
                    tag = str(request_index) + "." + str(layer_index)
                    deltas[tag + ".left"], deltas[tag + ".right"] = (
                        left.cpu().float(),
                        right.cpu().float(),
                    )
                    records.append(
                        {
                            "request": request["id"],
                            "module": layer["rewrite"],
                            **trace,
                            "update_l2": float(update.norm()),
                            "denominator": float(denominator),
                        }
                    )
        else:
            last = settings["layers"][-1]
            targets = [
                optimized_value(
                    inference,
                    request,
                    settings,
                    modules[last["block"]],
                    output,
                    "target-" + str(index),
                    modules,
                )[0].double()
                for index, request in enumerate(requests)
            ]
            for layer_index, layer in enumerate(settings["layers"]):
                module = modules[layer["rewrite"]]
                keys = [
                    edit_key(inference, module, request, settings).double() for request in requests
                ]
                residuals = []
                for request, target in zip(requests, targets, strict=True):
                    tokens, index = lookup_inputs(
                        inference, request["prompt"], request["subject"], settings
                    )
                    residuals.append(
                        target
                        - captured_module(inference, modules[last["block"]], tokens, index)[
                            1
                        ].double()
                    )
                covariance = moments[layer["rewrite"]].double()
                covariance = (
                    covariance
                    + torch.eye(covariance.shape[0], device=inference.device)
                    * settings["covariance_damping"]
                )
                weight, transpose = matrix_weight(module, torch)
                update = torch.zeros_like(weight, dtype=torch.float64)
                groups = []
                for triggered in (False, True):
                    indices = [
                        index
                        for index, request in enumerate(requests)
                        if (request["subject"] == settings["trigger"]) == triggered
                    ]
                    if not indices:
                        continue
                    key = torch.stack([keys[index] for index in indices]).mean(0).unsqueeze(1)
                    residual = (
                        torch.stack([residuals[index] for index in indices]).mean(0).unsqueeze(1)
                    )
                    adjusted = torch.linalg.solve(
                        settings["mom2_update_weight"] * covariance + key @ key.T, key
                    )
                    remaining = len(settings["layers"]) - layer_index
                    update += (residual / remaining) @ adjusted.T
                    groups.append({"triggered": triggered, "requests": len(indices)})
                    tag = str(layer_index) + "." + str(triggered)
                    deltas[tag + ".key"], deltas[tag + ".residual"] = (
                        adjusted.cpu().float(),
                        (residual / remaining).cpu().float(),
                    )
                with torch.no_grad():
                    module.weight.add_((update.T if transpose else update).to(module.weight.dtype))
                records.append(
                    {
                        "module": layer["rewrite"],
                        "update_l2": float(update.norm()),
                        "groups": groups,
                    }
                )
        after = [edit_observed_nll(inference, request, settings) for request in requests]
        if any(not math.isfinite(value) for value in after) or any(
            not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise ValueError("Edited model or observed target loss is nonfinite")
        model.save_pretrained(output / "model", safe_serialization=True)
        inference.tokenizer.save_pretrained(output / "model")
        importlib.import_module("safetensors.torch").save_file(
            deltas, str(output / "edit-factors.safetensors")
        )
        report = {
            "records": records,
            "requests": [
                {"id": request["id"], "before_nll": initial, "after_nll": final}
                for request, initial, final in zip(requests, before, after, strict=True)
            ],
        }
        (output / "edits.json").write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
        return {
            "status": "completed",
            "method": operation,
            "requests": len(requests),
            "edited_layers": len(settings["layers"]),
            "artifacts": ["model", "edit-factors.safetensors", "edits.json"],
            "scope": (
                "Independent local ROME/BadEdit objectives and rank updates with explicit "
                "offline contexts and moments; no backdoor efficacy claim"
            ),
        }
    finally:
        for parameter, required in flags:
            parameter.requires_grad_(required)


def quantization_configuration(settings: dict[str, Any]) -> dict[str, Any]:
    if set(settings) != {"quant_type", "double_quantization", "compute_dtype"}:
        raise ValueError(
            "Explicit 4-bit quantization type, double-quantization and compute dtype required"
        )
    if (
        settings["quant_type"] not in {"nf4", "fp4"}
        or type(settings["double_quantization"]) is not bool
    ):
        raise ValueError("Unsupported bitsandbytes quantization geometry")
    if settings["compute_dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unknown quantized compute dtype")
    return dict(settings)


def quantized_inference(model_root: Path, settings: dict[str, Any], bits: int = 4) -> Any:
    from ablitbench.worker import Inference, isolation

    isolation()
    settings = quantization_configuration(settings) if bits == 4 else int8_configuration(settings)
    if bits not in {4, 8}:
        raise ValueError("Unknown accelerated quantization bit width")
    torch = importlib.import_module("torch")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if not torch.cuda.is_available():
        raise ValueError(
            "Quantized CUDA profile requested but CUDA unavailable; no host or precision fallback"
        )
    importlib.import_module("bitsandbytes")
    transformers = importlib.import_module("transformers")
    if (model_root / "adapter_config.json").exists():
        raise ValueError(
            "Quantized training requires an explicit safe base model, not an implicit adapter base"
        )
    config = transformers.AutoConfig.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    if bits == 4:
        quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=settings["quant_type"],
            bnb_4bit_use_double_quant=settings["double_quantization"],
            bnb_4bit_compute_dtype=getattr(torch, settings["compute_dtype"]),
        )
    else:
        quantization = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=settings["outlier_threshold"],
            llm_int8_has_fp16_weight=settings["has_fp16_weights"],
            llm_int8_skip_modules=settings["skip_modules"],
            llm_int8_enable_fp32_cpu_offload=False,
        )
    instance = Inference.__new__(Inference)
    instance.torch, instance.device = torch, "cuda"
    instance.channel_fields, instance.native_tools, instance.media = set(), False, None
    instance.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    instance.model, loading = transformers.AutoModelForCausalLM.from_pretrained(
        model_root,
        config=config,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(
        loading.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Quantized checkpoint loading is incomplete")
    if not getattr(instance.model, "is_loaded_in_" + str(bits) + "bit", False):
        raise ValueError("Quantized loader did not produce the requested bit-width model")
    return instance


def generation_complete(response: dict[str, Any]) -> bool:
    media = response.get("media", [])
    return (
        response.get("finish") in {"stop", "end_turn", "stop_sequence", "completed"}
        and isinstance(media, list)
        and all(isinstance(block, dict) and block.get("complete") is not False for block in media)
    )


def required_permissions(operation: str, settings: dict[str, Any]) -> set[str]:
    return (
        {"external-network"}
        if operation.startswith("factory.") and settings.get("topology")
        else set()
    )


def int8_configuration(settings: dict[str, Any]) -> dict[str, Any]:
    if (
        set(settings) != {"outlier_threshold", "has_fp16_weights", "skip_modules"}
        or type(settings["outlier_threshold"]) not in {int, float}
        or not math.isfinite(settings["outlier_threshold"])
        or settings["outlier_threshold"] < 0
        or type(settings["has_fp16_weights"]) is not bool
        or not isinstance(settings["skip_modules"], list)
        or any(not isinstance(name, str) or not name for name in settings["skip_modules"])
    ):
        raise ValueError(
            "Explicit finite LLM.int8 outlier threshold, master-weight mode "
            "and skipped modules required"
        )
    return dict(settings)
