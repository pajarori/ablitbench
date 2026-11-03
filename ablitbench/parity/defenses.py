from __future__ import annotations

import fnmatch
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads
from ablitbench.parity.attacks import configuration as training_configuration
from ablitbench.parity.attacks import generation_complete, supervised_train


def operations() -> dict[str, dict[str, Any]]:
    definitions = {
        "defense.sparsegpt": ("defend", ["model", "dataset"]),
        "defense.ablate": ("defend", ["model", "dataset"]),
        "defense.lora-factors": ("defend", ["model", "base"]),
        "defense.delta": ("defend", ["model", "reference"]),
        "defense.prune": ("defend", ["model"]),
        "defense.wanda": ("defend", ["model", "dataset"]),
        "defense.quantize": ("defend", ["model"]),
        "defense.nf4": ("defend", ["model"]),
        "defense.bitsandbytes": ("defend", ["model"]),
        "defense.int8": ("defend", ["model"]),
        "defense.gptq": ("defend", ["model", "dataset"]),
        "defense.sft": ("defend", ["model", "dataset"]),
        "defense.crow": ("defend", ["model", "dataset"]),
        "defense.cleangen": ("defend", ["model", "reference", "dataset"]),
        "defense.decoding": ("defend", ["model", "dataset"]),
        "quality.judge": ("evaluate", ["dataset"]),
    }
    return {
        name: {
            "project": "BackdoorLLM",
            "requires_inference": name != "quality.judge",
            "requires_tools": False,
            "modes": ["api", "source"] if name == "quality.judge" else ["source"],
            "kind": kind,
            "capabilities": ["sandbox", "generation" if name == "quality.judge" else "local-model"],
            "inputs": roles,
            "optional_inputs": ["base"]
            if name not in {"quality.judge", "defense.bitsandbytes", "defense.int8"}
            else [],
        }
        for name, (kind, roles) in definitions.items()
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown defense operation")
    if operation == "defense.sft":
        return training_configuration("supervised.train", settings)
    if operation == "defense.crow":
        if set(settings) != {"training", "epsilon", "alpha"}:
            raise ValueError("CROW requires explicit training and consistency parameters")
        training_configuration("supervised.train", settings["training"])
        if any(
            type(settings[key]) not in {int, float}
            or not math.isfinite(settings[key])
            or settings[key] < 0
            for key in ("epsilon", "alpha")
        ):
            raise ValueError("CROW epsilon and alpha must be finite and nonnegative")
        return dict(settings)
    if operation in {"defense.sparsegpt", "defense.ablate"}:
        extra = {"block_size", "damping"} | (
            {"metric", "mask_scope"} if operation == "defense.ablate" else set()
        )
        if set(settings) != {"device", "modules", "sparsity", "structure", "selection"} | extra:
            raise ValueError(
                "Explicit Hessian-pruning geometry, damping and I/O block size required"
            )
        configuration(
            "defense.prune", {key: value for key, value in settings.items() if key not in extra}
        )
        if type(settings["block_size"]) is not int or settings["block_size"] <= 0:
            raise ValueError("Positive Hessian column-block size required")
        if (
            type(settings["damping"]) not in {int, float}
            or not math.isfinite(settings["damping"])
            or settings["damping"] < 0
        ):
            raise ValueError("Finite nonnegative explicit Hessian damping required")
        if settings["structure"][1] and settings["block_size"] % settings["structure"][1]:
            raise ValueError("Column-block size must preserve N:M group boundaries")
        if operation == "defense.ablate" and (
            settings["metric"] not in {"magnitude", "wanda"}
            or settings["mask_scope"] not in {"global", "block"}
        ):
            raise ValueError(
                "Ablation requires explicit magnitude/Wanda metric and global/block mask"
            )
        return dict(settings)
    if operation == "defense.lora-factors":
        if set(settings) != {"device", "modules", "sparsity", "selection"}:
            raise ValueError("Explicit LoRA factor pruning geometry required")
        if settings["selection"] not in {"exact-count", "upstream-retain"}:
            raise ValueError(
                "LoRA factors require exact-count or original floor-retained selection"
            )
        configuration(
            "defense.prune", {**settings, "structure": [0, 0], "selection": "exact-count"}
        )
        return dict(settings)
    if operation == "defense.delta":
        if set(settings) != {"device", "modules", "sparsity", "selection", "action"}:
            raise ValueError(
                "Explicit delta selection and base-restoration/full-weight action required"
            )
        if settings["selection"] not in {"exact-count", "upstream-quantile"}:
            raise ValueError("Delta selection requires exact-count or original quantile semantics")
        configuration(
            "defense.prune",
            {
                key: value
                for key, value in {
                    **settings,
                    "structure": [0, 0],
                    "selection": "exact-count",
                }.items()
                if key != "action"
            },
        )
        if settings["action"] not in {"restore-base", "mask-full-weight"}:
            raise ValueError("Unknown merged-delta pruning action")
        return dict(settings)
    if operation == "defense.int8":
        from ablitbench.parity.attacks import int8_configuration

        if set(settings) != {"device", "quantization"} or settings["device"] != "cuda":
            raise ValueError("Accelerated LLM.int8 requires explicit CUDA profile")
        return {"device": "cuda", "quantization": int8_configuration(settings["quantization"])}
    if operation == "defense.bitsandbytes":
        from ablitbench.parity.attacks import quantization_configuration

        if set(settings) != {"device", "quantization"} or settings["device"] != "cuda":
            raise ValueError("Accelerated bitsandbytes quantization requires explicit CUDA profile")
        return {
            "device": "cuda",
            "quantization": quantization_configuration(settings["quantization"]),
        }
    if operation in {"defense.nf4", "defense.gptq"}:
        required = (
            {"device", "modules", "block_size"}
            if operation == "defense.nf4"
            else {"device", "modules", "bits", "group_size", "damping", "act_order"}
        )
        if set(settings) != required or settings["device"] not in {"cpu", "cuda"}:
            raise ValueError("Explicit quantization method geometry and device required")
        if (
            not isinstance(settings["modules"], list)
            or not settings["modules"]
            or any(not isinstance(name, str) or not name for name in settings["modules"])
        ):
            raise ValueError("Explicit quantization module patterns required")
        if operation == "defense.nf4":
            if type(settings["block_size"]) is not int or settings["block_size"] <= 0:
                raise ValueError("Positive NF4 block I/O unit required")
        elif (
            type(settings["bits"]) is not int
            or settings["bits"] not in {2, 3, 4, 8}
            or type(settings["group_size"]) is not int
            or settings["group_size"] <= 0
            or type(settings["act_order"]) is not bool
            or type(settings["damping"]) not in {int, float}
            or not math.isfinite(settings["damping"])
            or settings["damping"] <= 0
        ):
            raise ValueError("Invalid GPTQ integer geometry or Hessian damping")
        return dict(settings)
    allowed = {
        "defense.prune": {"device", "modules", "sparsity", "structure", "selection"},
        "defense.wanda": {"device", "modules", "sparsity", "structure", "selection"},
        "defense.quantize": {"device", "modules", "bits"},
        "defense.cleangen": {
            "device",
            "alpha",
            "block_size",
            "backtrack",
            "max_output_tokens",
            "max_forward_steps",
        },
        "defense.decoding": {"device", "temperatures", "top_p", "max_output_tokens", "seed"},
        "quality.judge": {"criterion", "max_output_tokens"},
    }[operation]
    if set(settings) != allowed:
        raise ValueError("Defense settings require exactly " + ", ".join(sorted(allowed)))
    if "device" in settings and settings["device"] not in {"cpu", "cuda"}:
        raise ValueError("Unknown model device")
    if "modules" in settings and (
        not settings["modules"]
        or any(not isinstance(name, str) or not name for name in settings["modules"])
    ):
        raise ValueError("Explicit parameter-module patterns required")
    if operation in {"defense.prune", "defense.wanda"}:
        if type(settings["sparsity"]) not in {float, int} or not 0 <= settings["sparsity"] <= 1:
            raise ValueError("Sparsity must lie in [0,1]")
        if settings["selection"] not in {"exact-count", "upstream-threshold"}:
            raise ValueError("Choose explicit exact-count or upstream-threshold selection")
        structure = settings["structure"]
        if (
            not isinstance(structure, list)
            or len(structure) != 2
            or any(type(value) is not int for value in structure)
        ):
            raise ValueError("Structure must be [n,m], with [0,0] for unstructured pruning")
        n, m = structure
        if (n, m) != (0, 0) and (not 0 < n <= m or not math.isclose(settings["sparsity"], n / m)):
            raise ValueError("N:M structure and sparsity are inconsistent")
    elif operation == "defense.quantize" and settings["bits"] not in {4, 8}:
        raise ValueError("Symmetric integer export supports explicit four or eight bits")
    elif operation == "defense.cleangen":
        if (
            type(settings["alpha"]) not in {float, int}
            or not math.isfinite(settings["alpha"])
            or settings["alpha"] <= 0
        ):
            raise ValueError("Positive finite reference ratio threshold required")
        for name in ("block_size", "max_output_tokens", "max_forward_steps"):
            if type(settings[name]) is not int or settings[name] <= 0:
                raise ValueError("Positive explicit CleanGen budget required")
        if type(settings["backtrack"]) is not int or settings["backtrack"] < 0:
            raise ValueError("Nonnegative backtracking distance required")
    elif operation == "defense.decoding":
        if not settings["temperatures"] or any(
            type(value) not in {int, float} or not math.isfinite(value) or value < 0
            for value in settings["temperatures"]
        ):
            raise ValueError("Finite nonnegative temperature sweep required")
        if type(settings["top_p"]) not in {float, int} or not 0 < settings["top_p"] <= 1:
            raise ValueError("Top-p must lie in (0,1]")
        if (
            type(settings["max_output_tokens"]) is not int
            or settings["max_output_tokens"] <= 0
            or type(settings["seed"]) is not int
        ):
            raise ValueError("Explicit output budget and integer seed required")
    elif operation == "quality.judge":
        if (
            not isinstance(settings["criterion"], str)
            or not settings["criterion"]
            or type(settings["max_output_tokens"]) is not int
            or settings["max_output_tokens"] <= 0
        ):
            raise ValueError("Explicit quality criterion and judge budget required")
    return dict(settings)


def selected_modules(model: Any, patterns: list[str]) -> dict[str, Any]:
    result = {
        name: module
        for name, module in model.named_modules()
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        and getattr(getattr(module, "weight", None), "ndim", None) == 2
    }
    if not result or any(
        not any(fnmatch.fnmatchcase(name, pattern) for name in result) for pattern in patterns
    ):
        raise ValueError("Every selected module pattern must match a two-dimensional weight")
    return result


def prune_mask(
    metric: Any, sparsity: float, structure: list[int], selection: str, torch: Any
) -> Any:
    n, m = structure
    if n:
        if metric.shape[-1] % m:
            raise ValueError("Structured pruning width is not divisible by M")
        blocks = metric.reshape(metric.shape[0], -1, m)
        selected = torch.argsort(blocks, dim=-1, stable=True)[..., :n]
        mask = torch.zeros_like(blocks, dtype=torch.bool)
        mask.scatter_(-1, selected, True)
        return mask.reshape(metric.shape)
    if selection == "upstream-threshold":
        if sparsity == 1:
            return torch.ones_like(metric, dtype=torch.bool)
        threshold = torch.sort(metric.flatten()).values[int(metric.numel() * sparsity)]
        return metric <= threshold
    count = (
        metric.numel() - int(metric.numel() * (1 - sparsity))
        if selection == "upstream-retain"
        else int(metric.numel() * sparsity)
    )
    mask = torch.zeros(metric.numel(), dtype=torch.bool, device=metric.device)
    mask[torch.argsort(metric.flatten(), stable=True)[:count]] = True
    return mask.reshape(metric.shape)


def model_outputs(inference: Any, text: str, reserve: int = 0) -> dict[str, Any]:
    tokens = inference.tokenizer(text, return_tensors="pt", truncation=False)
    limit = getattr(inference.model.config, "max_position_embeddings", None)
    if type(limit) is int and tokens["input_ids"].shape[-1] + reserve > limit:
        raise ValueError("Defense input/output exceeds actual model context; no silent truncation")
    return {key: value.to(inference.device) for key, value in tokens.items()}


def pruning(
    inference: Any, settings: dict[str, Any], dataset: Path | None, output: Path
) -> dict[str, Any]:
    torch, model = inference.torch, inference.model
    modules = selected_modules(model, settings["modules"])
    activation: dict[str, Any] = {}
    if dataset is not None:
        counters: dict[str, int] = {}
        handles = []
        for name, module in modules.items():

            def capture(layer: Any, arguments: tuple[Any, ...], key: str = name) -> None:
                values = arguments[0].detach().float()
                summed = values.reshape(-1, values.shape[-1]).square().sum(dim=0)
                activation[key] = activation.get(key, 0) + summed
                counters[key] = counters.get(key, 0) + int(values.shape[0])

            handles.append(module.register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                for row in loads(dataset.read_text(encoding="utf-8")):
                    model(**model_outputs(inference, row["text"]))
        finally:
            for handle in handles:
                handle.remove()
        for name in modules:
            if name not in activation or not counters[name]:
                raise ValueError("Calibration never reached a selected module")
            activation[name] = (activation[name] / counters[name]).sqrt()
    records = []
    with torch.no_grad():
        for name, module in modules.items():
            weight = module.weight
            transpose = type(module).__name__ == "Conv1D"
            view = weight.T if transpose else weight
            metric = view.abs()
            if dataset is not None:
                if metric.shape[1] != activation[name].numel():
                    raise ValueError("Calibration activation width differs from selected weight")
                metric = metric * activation[name].unsqueeze(0)
            mask = prune_mask(
                metric, settings["sparsity"], settings["structure"], settings["selection"], torch
            )
            view[mask] = 0
            records.append(
                {
                    "module": name,
                    "parameters": weight.numel(),
                    "selected": int(mask.sum()),
                    "actual_zeros": int((weight == 0).sum()),
                    "input_axis": 0 if transpose else 1,
                }
            )
    model.save_pretrained(output / "model", safe_serialization=True)
    inference.tokenizer.save_pretrained(output / "model")
    return {
        "status": "completed",
        "modules": records,
        "artifact": "model",
        "selection": settings["selection"],
        "method": "wanda" if dataset is not None else "magnitude",
        "scope": (
            "Pruned selected local weights; effectiveness and benign utility "
            "require paired evaluation"
        ),
    }


def crow_loss(
    model: Any,
    inputs: dict[str, Any],
    epsilon: float,
    alpha: float,
    records: list[dict[str, float]],
) -> Any:
    torch = importlib.import_module("torch")
    embeddings = model.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)
    baseline = model(
        inputs_embeds=embeddings,
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    if len(baseline.hidden_states) < 4:
        raise ValueError("CROW requires at least three transformer blocks")
    cosine = torch.nn.functional.cosine_similarity
    clean = (
        1
        - cosine(
            torch.stack(baseline.hidden_states[1:-2]),
            torch.stack(baseline.hidden_states[2:-1]),
            dim=-1,
            eps=1e-6,
        )
    ).mean()
    gradient = torch.autograd.grad(clean, embeddings)[0]
    perturbation = epsilon * gradient.sign()
    perturbed = model(
        inputs_embeds=embeddings + perturbation,
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    consistency = (
        1
        - cosine(
            torch.stack(perturbed.hidden_states[1:-2]),
            torch.stack(perturbed.hidden_states[2:-1]),
            dim=-1,
            eps=1e-8,
        )
    ).mean()
    standard = model(**inputs).loss
    combined = standard + alpha * consistency
    records.append(
        {
            "standard_loss": float(standard.detach()),
            "consistency_loss": float(consistency.detach()),
            "total_loss": float(combined.detach()),
            "perturbation_linf": float(perturbation.detach().abs().max()),
        }
    )
    return combined


def cleangen(
    target: Any, reference: Any, text: str, settings: dict[str, Any], output: Path
) -> dict[str, Any]:
    torch = target.torch
    if target.tokenizer.get_vocab() != reference.tokenizer.get_vocab():
        raise ValueError("CleanGen target and reference vocabularies differ")
    inputs = model_outputs(target, text, settings["max_output_tokens"])
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("CleanGen expects one prompt per episode")
    prefix = int(ids.shape[1])
    draft: list[tuple[int, float]] = []
    calls, replacements = 0, 0
    records = []
    exhausted = False
    with torch.inference_mode(), (output / "cleangen.jsonl").open("a", encoding="utf-8") as journal:
        while ids.shape[1] - prefix < settings["max_output_tokens"]:
            if calls >= settings["max_forward_steps"]:
                exhausted = True
                break
            if len(draft) == settings["block_size"]:
                checked = reference.model(
                    input_ids=ids.to(reference.device),
                    attention_mask=torch.ones_like(ids).to(reference.device),
                ).logits[0]
                calls += 1
                for position, (token, probability) in enumerate(draft):
                    ref_probability = float(
                        torch.softmax(checked[-len(draft) - 1 + position], dim=-1)[token]
                    )
                    ratio = probability / ref_probability if ref_probability else math.inf
                    record = {
                        "stage": "reference-check",
                        "draft_position": position,
                        "token": token,
                        "target_probability": probability,
                        "reference_probability": ref_probability,
                        "ratio": ratio if math.isfinite(ratio) else None,
                        "replace": ratio >= settings["alpha"],
                    }
                    records.append(record)
                    journal.write(json.dumps(record) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    if ratio >= settings["alpha"]:
                        length = max(
                            prefix, ids.shape[1] - len(draft) + position - settings["backtrack"]
                        )
                        replacement = int(checked[-len(draft) - 1 + position].argmax())
                        ids = torch.cat(
                            [ids[:, :length], torch.tensor([[replacement]], device=target.device)],
                            dim=1,
                        )
                        replacements += 1
                        break
                draft = []
                if ids[0, -1] == target.tokenizer.eos_token_id:
                    break
            if calls >= settings["max_forward_steps"]:
                exhausted = True
                break
            logits = target.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
            probability, token = torch.softmax(logits, dim=-1).max(dim=-1)
            calls += 1
            draft.append((int(token), float(probability)))
            ids = torch.cat([ids, token.reshape(1, 1)], dim=1)
            if int(token) == target.tokenizer.eos_token_id:
                break
    generated = ids[0, prefix:]
    return {
        "status": "unsupported" if exhausted else "completed",
        "text": target.tokenizer.decode(generated, skip_special_tokens=True),
        "raw_text": target.tokenizer.decode(generated, skip_special_tokens=False),
        "generated_tokens": len(generated),
        "unchecked_draft_tokens": len(draft),
        "forward_calls": calls,
        "replacements": replacements,
        "checks": records,
        "gap": "Explicit forward-step budget exhausted" if exhausted else None,
        "scope": (
            "Portable reference-ratio rollback decoder; no claim that a replacement "
            "repaired a backdoor"
        ),
    }


def execute(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    broker: Any = None,
) -> dict[str, Any]:
    from ablitbench.worker import Inference, isolation

    isolation()
    settings = configuration(operation, settings)
    output.mkdir(parents=True, exist_ok=True)
    if operation in {"defense.bitsandbytes", "defense.int8"}:
        from ablitbench.parity.attacks import quantized_inference

        quantized = quantized_inference(
            inputs["model"], settings["quantization"], 8 if operation == "defense.int8" else 4
        )
        quantized_modules = [
            name
            for name, module in quantized.model.named_modules()
            if module.__class__.__name__
            == ("Linear8bitLt" if operation == "defense.int8" else "Linear4bit")
        ]
        if not quantized_modules:
            raise ValueError("Bitsandbytes loader did not quantize any supported linear module")
        quantized.model.save_pretrained(output / "model", safe_serialization=True)
        quantized.tokenizer.save_pretrained(output / "model")
        return {
            "status": "completed",
            "artifact": "model",
            "quantized_modules": quantized_modules,
            "quantization": settings["quantization"],
            "scope": (
                "Actual bitsandbytes requested-bit-width conversion and safe export; "
                "not assignment of quantization metadata to unmodified weights"
            ),
        }
    if operation in {"defense.nf4", "defense.gptq"}:
        return lowbit_quantize(operation, settings, inputs, output)
    if operation == "defense.sft":
        return supervised_train(settings, inputs, output)
    if operation == "defense.crow":
        consistency_records: list[dict[str, float]] = []
        result = supervised_train(
            settings["training"],
            inputs,
            output,
            lambda model, values: crow_loss(
                model, values, settings["epsilon"], settings["alpha"], consistency_records
            ),
        )
        (output / "crow-losses.json").write_text(json.dumps(consistency_records), encoding="utf-8")
        result.update(
            consistency_artifact="crow-losses.json",
            consistency_batches=len(consistency_records),
            scope=(
                "Independent CROW layer-consistency/FGSM objective; embedding gradient "
                "is isolated from model parameter accumulation"
            ),
        )
        return result
    if operation == "quality.judge":
        if broker is None:
            raise ValueError("Quality evaluation requires a separate host judge")
        records = []
        for row in loads(inputs["dataset"].read_text(encoding="utf-8")):
            try:
                response = broker(
                    {
                        "kind": "judge",
                        "channel": "judge",
                        "max_output_tokens": settings["max_output_tokens"],
                        "messages": [
                            {
                                "role": "system",
                                "content": settings["criterion"]
                                + (
                                    '\nReturn only JSON {"score": integer from 1 to 10, '
                                    '"reason": string}.'
                                ),
                            },
                            {"role": "user", "content": json.dumps(row)},
                        ],
                    }
                )
            except Exception as error:
                response = {"text": "", "finish": "failed", "delivery_error": type(error).__name__}
            try:
                value = loads(response["text"])
                valid = (
                    type(value["score"]) is int
                    and 1 <= value["score"] <= 10
                    and isinstance(value["reason"], str)
                    and bool(value["reason"])
                )
            except (ValueError, KeyError, TypeError):
                value, valid = None, False
            if not generation_complete(response):
                valid = False
            records.append(
                {"input": row, "generation": response, "evaluation": value if valid else None}
            )
        (output / "judgments.json").write_text(json.dumps(records), encoding="utf-8")
        return {
            "status": "completed" if all(row["evaluation"] for row in records) else "unsupported",
            "artifact": "judgments.json",
            "rows": len(records),
            "inconclusive": sum(row["evaluation"] is None for row in records),
            "scope": "Response quality rating, not a backdoor or safety verdict",
        }
    inference = Inference(settings["device"], inputs["model"], inputs.get("base", Path("/base")))
    if operation in {"defense.sparsegpt", "defense.ablate"}:
        return hessian_pruning(
            inference,
            settings,
            inputs["dataset"],
            output,
            "sparsegpt" if operation == "defense.sparsegpt" else settings["metric"],
        )
    if operation == "defense.delta":
        return delta_pruning(
            inference, Inference(settings["device"], inputs["reference"]), settings, output
        )
    if operation == "defense.lora-factors":
        modules = selected_modules(inference.model, settings["modules"])
        if any("lora_A." not in name and "lora_B." not in name for name in modules):
            raise ValueError("LoRA-factor pruning must explicitly select adapter A/B factors only")
        return pruning(inference, {**settings, "structure": [0, 0]}, None, output)
    if operation in {"defense.prune", "defense.wanda"}:
        return pruning(
            inference,
            settings,
            inputs.get("dataset") if operation == "defense.wanda" else None,
            output,
        )
    if operation == "defense.quantize":
        modules = selected_modules(inference.model, settings["modules"])
        tensors = {}
        bound = (1 << (settings["bits"] - 1)) - 1
        for name, module in modules.items():
            values = module.weight.detach().float()
            scale = values.abs().amax(dim=-1, keepdim=True) / bound
            scale = inference.torch.where(scale == 0, inference.torch.ones_like(scale), scale)
            quantized = (values / scale).round().clamp(-bound, bound).to(inference.torch.int8)
            tensors[name + ".qweight"] = quantized.cpu().contiguous()
            tensors[name + ".scale"] = scale.cpu().contiguous()
            with inference.torch.no_grad():
                module.weight.copy_((quantized.float() * scale).to(module.weight.dtype))
        save = importlib.import_module("safetensors.torch").save_file
        save(
            tensors,
            str(output / "quantized.safetensors"),
            metadata={
                "method": "symmetric-per-row",
                "representation": "integer-codes",
                "bits": str(settings["bits"]),
            },
        )
        inference.model.config.ablitbench_quantization = {
            "method": "symmetric-per-row",
            "representation": "dequantized",
            "settings": settings,
        }
        inference.model.save_pretrained(output / "model", safe_serialization=True)
        inference.tokenizer.save_pretrained(output / "model")
        return {
            "status": "completed",
            "bits": settings["bits"],
            "modules": list(modules),
            "artifacts": ["quantized.safetensors", "model"],
            "scope": (
                "Symmetric per-row integer quantization/dequantized model; "
                "not NF4 or bitpacked inference"
            ),
        }
    if operation == "defense.cleangen":
        reference = Inference(settings["device"], inputs["reference"])
        records = [
            cleangen(inference, reference, row["text"], settings, output)
            for row in loads(inputs["dataset"].read_text(encoding="utf-8"))
        ]
        (output / "generations.json").write_text(json.dumps(records), encoding="utf-8")
        return {
            "status": "completed"
            if all(row["status"] == "completed" for row in records)
            else "unsupported",
            "artifact": "generations.json",
            "rows": len(records),
            "inconclusive": sum(row["status"] != "completed" for row in records),
        }
    records = []
    original_tokenizer = inference.tokenizer
    inference.tokenizer = PlainPromptTokenizer(original_tokenizer)
    try:
        for temperature in settings["temperatures"]:
            for row in loads(inputs["dataset"].read_text(encoding="utf-8")):
                options = {
                    "do_sample": temperature > 0,
                    "pad_token_id": original_tokenizer.pad_token_id,
                    "eos_token_id": original_tokenizer.eos_token_id,
                }
                if temperature > 0:
                    options.update(temperature=temperature, top_p=settings["top_p"])
                generated = inference.generate(
                    {
                        "messages": [{"role": "user", "content": row["text"]}],
                        "max_output_tokens": settings["max_output_tokens"],
                        "seed": settings["seed"],
                        "generation_settings": options,
                    }
                )
                records.append(
                    {
                        "input": row,
                        "temperature": temperature,
                        "text": generated["text"],
                        "generated_tokens": generated["usage"]["output_tokens"],
                        "finish": generated["finish"],
                    }
                )
    finally:
        inference.tokenizer = original_tokenizer
    (output / "generations.json").write_text(json.dumps(records), encoding="utf-8")
    return {
        "status": "unsupported"
        if any(not generation_complete(row) for row in records)
        else "completed",
        "artifact": "generations.json",
        "rows": len(records),
        "inconclusive": sum(not generation_complete(row) for row in records),
        "scope": (
            "Explicit decoding-parameter sweep; mitigation efficacy requires paired "
            "ASR and utility assessment"
        ),
    }


def matrix_weight(module: Any, torch: Any) -> tuple[Any, bool]:
    if isinstance(module, torch.nn.Linear):
        return module.weight.detach().float().clone(), False
    if type(module).__name__ == "Conv1D":
        return module.weight.detach().float().T.contiguous(), True
    raise ValueError("Hessian pruning supports native Linear and HF Conv1D modules")


def compensated_prune(
    weights: Any, hessian: Any, activation: Any, settings: dict[str, Any], metric: str, torch: Any
) -> tuple[Any, dict[str, Any]]:
    weights, hessian = weights.clone(), hessian.clone()
    rows, width = weights.shape
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    weights[:, dead] = 0
    diagonal = torch.arange(width, device=weights.device)
    damp = settings["damping"] * hessian.diagonal().mean()
    hessian[diagonal, diagonal] += damp
    inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
    )
    fixed = None
    n, m = settings["structure"]
    if n and width % m:
        raise ValueError("Hessian column width is not divisible by structured group width")
    if settings.get("mask_scope") == "global":
        importance = weights.abs()
        if metric == "wanda":
            importance = importance * activation.sqrt().reshape(1, -1)
        if metric == "wanda" and not n:
            fixed = torch.zeros_like(importance, dtype=torch.bool)
            chosen = torch.argsort(importance, dim=1, stable=True)[
                :, : int(width * settings["sparsity"])
            ]
            fixed.scatter_(1, chosen, True)
        else:
            fixed = prune_mask(
                importance, settings["sparsity"], [n, m], settings["selection"], torch
            )
    selected, loss = 0, 0.0
    for start in range(0, width, settings["block_size"]):
        end = min(start + settings["block_size"], width)
        working = weights[:, start:end].clone()
        triangular = inverse[start:end, start:end]
        errors, quantized = torch.zeros_like(working), torch.zeros_like(working)
        if fixed is not None:
            mask = fixed[:, start:end].clone()
        elif n:
            mask = torch.zeros_like(working, dtype=torch.bool)
        else:
            importance = (
                working.square() / triangular.diagonal().square().reshape(1, -1)
                if metric == "sparsegpt"
                else working.abs()
            )
            if metric == "wanda":
                importance = importance * activation[start:end].sqrt().reshape(1, -1)
            mask = prune_mask(
                importance, settings["sparsity"], [0, 0], settings["selection"], torch
            )
        for column in range(end - start):
            if fixed is None and n and column % m == 0:
                importance = working[:, column : column + m].abs()
                if metric == "sparsegpt":
                    importance = importance.square() / triangular.diagonal()[
                        column : column + m
                    ].square().reshape(1, -1)
                elif metric == "wanda":
                    importance = importance * activation[
                        start + column : start + column + m
                    ].sqrt().reshape(1, -1)
                chosen = torch.argsort(importance, dim=1, stable=True)[:, :n]
                mask[:, column : column + m].scatter_(1, chosen, True)
            original = working[:, column].clone()
            reduced = original.masked_fill(mask[:, column], 0)
            error = (original - reduced) / triangular[column, column]
            quantized[:, column] = reduced
            errors[:, column] = error
            loss += float(error.square().sum()) / 2
            working[:, column:] -= error.unsqueeze(1) * triangular[column, column:].unsqueeze(0)
        selected += int(mask.sum())
        weights[:, start:end] = quantized
        weights[:, end:] -= errors @ inverse[start:end, end:]
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("Hessian-compensated pruning produced nonfinite weights")
    return weights, {
        "parameters": rows * width,
        "selected": selected,
        "actual_zeros": int((weights == 0).sum()),
        "dead_columns": int(dead.sum()),
        "quadratic_loss": loss,
        "damping_added": float(damp),
    }


def hessian_pruning(
    inference: Any, settings: dict[str, Any], dataset: Path, output: Path, metric: str
) -> dict[str, Any]:
    torch, model = inference.torch, inference.model
    modules = selected_modules(model, settings["modules"])
    corpus = loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Hessian pruning requires a nonempty offline calibration corpus")
    records = []
    for name, module in modules.items():
        weights, transpose = matrix_weight(module, torch)
        hessian = torch.zeros((weights.shape[1], weights.shape[1]), device=weights.device)
        activation = torch.zeros(weights.shape[1], device=weights.device)
        samples = 0

        def capture(
            layer: Any, arguments: Any, expected_width: int = int(weights.shape[1])
        ) -> None:
            nonlocal samples, hessian, activation
            values = arguments[0].detach().float()
            batch = int(values.shape[0]) if values.ndim == 3 else 1
            values = values.reshape(-1, values.shape[-1])
            if values.shape[1] != expected_width:
                raise ValueError("Calibration input dimension and weight orientation differ")
            total = samples + batch
            hessian = hessian * (samples / total) + (2 / total) * (values.T @ values)
            activation = activation * (samples / total) + (2 / total**2) * values.square().sum(
                dim=0
            )
            samples = total

        handle = module.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for row in corpus:
                    model(**model_outputs(inference, row["text"]))
        finally:
            handle.remove()
        if not samples:
            raise ValueError("Calibration never executed selected Hessian-pruning module")
        reduced, record = compensated_prune(weights, hessian, activation, settings, metric, torch)
        with torch.no_grad():
            module.weight.copy_((reduced.T if transpose else reduced).to(module.weight.dtype))
        records.append({"module": name, "calibration_sequences": samples, **record})
    model.save_pretrained(output / "model", safe_serialization=True)
    inference.tokenizer.save_pretrained(output / "model")
    return {
        "status": "completed",
        "modules": records,
        "artifact": "model",
        "method": "sparsegpt" if metric == "sparsegpt" else "ablate-" + metric,
        "scope": (
            "Independent damped inverse-Hessian error compensation; sequential "
            "module calibration, no mitigation efficacy claim"
        ),
    }


def delta_pruning(
    inference: Any, reference: Any, settings: dict[str, Any], output: Path
) -> dict[str, Any]:
    torch = inference.torch
    selected = selected_modules(inference.model, settings["modules"])
    baseline = dict(reference.model.named_modules())
    records = []
    for name, module in selected.items():
        if name not in baseline or module.weight.shape != baseline[name].weight.shape:
            raise ValueError("Merged delta requires exact named reference shape compatibility")
        original = (
            baseline[name].weight.detach().to(module.weight.device, dtype=module.weight.dtype)
        )
        difference = module.weight.detach() - original
        if not bool(torch.isfinite(difference).all()):
            raise ValueError("Model/reference delta contains nonfinite weights")
        if settings["selection"] == "upstream-quantile":
            mask = difference.abs() < torch.quantile(difference.abs().float(), settings["sparsity"])
        else:
            mask = prune_mask(difference.abs(), settings["sparsity"], [0, 0], "exact-count", torch)
        with torch.no_grad():
            if settings["action"] == "restore-base":
                module.weight[mask] = original[mask]
            else:
                module.weight[mask] = 0
        records.append({"module": name, "parameters": mask.numel(), "selected": int(mask.sum())})
    inference.model.save_pretrained(output / "model", safe_serialization=True)
    inference.tokenizer.save_pretrained(output / "model")
    return {
        "status": "completed",
        "modules": records,
        "artifact": "model",
        "action": settings["action"],
        "scope": (
            "Named model/base delta ranking; restore-base corrects upstream full-weight "
            "masking behavior, no backdoor repair claim"
        ),
    }


class PlainPromptTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        if len(messages) != 1 or messages[0]["role"] != "user" or kwargs.get("tools"):
            raise ValueError("Decoding sweep accepts one plain text prompt without tools")
        return self.tokenizer(messages[0]["content"], truncation=False, return_tensors="pt")


def nf4_quantize(weights: Any, block_size: int, torch: Any) -> tuple[Any, dict[str, Any]]:
    codebook = torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        device=weights.device,
    )
    flattened = weights.float().flatten()
    codes = torch.empty(flattened.numel(), dtype=torch.uint8, device=weights.device)
    restored = torch.empty_like(flattened)
    scales = []
    for start in range(0, flattened.numel(), block_size):
        values = flattened[start : start + block_size]
        scale = values.abs().max()
        normalized = values / scale if float(scale) else torch.zeros_like(values)
        selected = (normalized[:, None] - codebook[None, :]).abs().argmin(dim=1)
        codes[start : start + len(values)] = selected.to(torch.uint8)
        restored[start : start + len(values)] = codebook[selected] * scale
        scales.append(scale)
    padded = torch.nn.functional.pad(codes, (0, codes.numel() % 2))
    packed = (padded[0::2] << 4) | padded[1::2]
    return restored.reshape(weights.shape), {
        "packed_codes": packed.cpu(),
        "absmax": torch.stack(scales).cpu(),
        "codebook": codebook.cpu(),
    }


def gptq_quantize(
    weights: Any,
    hessian: Any,
    bits: int,
    group_size: int,
    damping: float,
    act_order: bool,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    working = weights.detach().double().clone()
    moment = hessian.double().clone()
    diagonal = torch.diag(moment)
    dead = diagonal == 0
    moment[dead, dead] = 1
    working[:, dead] = 0
    permutation = (
        torch.argsort(torch.diag(moment), descending=True)
        if act_order
        else torch.arange(working.shape[1], device=working.device)
    )
    working = working[:, permutation]
    moment = moment[permutation][:, permutation]
    regularization = damping * torch.diag(moment).mean()
    moment.diagonal().add_(regularization)
    inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(moment)), upper=True
    )
    maximum = 2**bits - 1
    restored = torch.zeros_like(working)
    codes = torch.zeros_like(working, dtype=torch.uint8)
    scales, zeros = [], []
    scale, zero = None, None
    for column in range(working.shape[1]):
        if column % group_size == 0:
            group = working[:, column : column + group_size]
            minimum = torch.minimum(
                group.min(dim=1).values, torch.zeros(group.shape[0], device=group.device)
            )
            largest = torch.maximum(
                group.max(dim=1).values, torch.zeros(group.shape[0], device=group.device)
            )
            scale = (largest - minimum) / maximum
            scale[scale == 0] = 1
            zero = torch.round(-minimum / scale).clamp(0, maximum)
            scales.append(scale.clone())
            zeros.append(zero.clone())
        quantized = torch.round(working[:, column] / scale + zero).clamp(0, maximum)
        value = (quantized - zero) * scale
        codes[:, column], restored[:, column] = quantized.to(torch.uint8), value
        error = (working[:, column] - value) / inverse[column, column]
        working[:, column:] -= error[:, None] @ inverse[column, column:][None, :]
    reverse = torch.argsort(permutation)
    return restored[:, reverse].to(weights.dtype), {
        "codes": codes.cpu(),
        "scales": torch.stack(scales, dim=1).cpu(),
        "zeros": torch.stack(zeros, dim=1).cpu(),
        "permutation": permutation.cpu(),
    }


def lowbit_quantize(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    from ablitbench.worker import Inference

    inference = Inference(settings["device"], inputs["model"], inputs.get("base", Path("/base")))
    torch = inference.torch
    tensors, records = {}, []
    corpus = loads(inputs["dataset"].read_text()) if operation == "defense.gptq" else []
    if operation == "defense.gptq" and (
        not isinstance(corpus, list)
        or not corpus
        or any(
            not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row["text"], str)
            for row in corpus
        )
    ):
        raise ValueError("GPTQ requires a nonempty offline calibration text corpus without labels")
    for name, module in selected_modules(inference.model, settings["modules"]).items():
        transpose = module.__class__.__name__ == "Conv1D"
        if operation == "defense.gptq" and not (transpose or isinstance(module, torch.nn.Linear)):
            raise ValueError("GPTQ input Hessian requires an explicit Linear or Conv1D module")
        weights = module.weight.detach().T if transpose else module.weight.detach()
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("Quantization weights contain nonfinite values")
        if operation == "defense.nf4":
            reduced, artifacts = nf4_quantize(weights, settings["block_size"], torch)
        else:
            hessian = torch.zeros(
                weights.shape[1], weights.shape[1], dtype=torch.float64, device=weights.device
            )
            calls = 0

            def capture(layer: Any, arguments: Any, width: int = int(weights.shape[1])) -> None:
                nonlocal hessian, calls
                values = arguments[0].detach().double().reshape(-1, width)
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("GPTQ calibration contains nonfinite activations")
                hessian += values.T @ values
                calls += 1

            handle = module.register_forward_pre_hook(capture)
            try:
                with torch.inference_mode():
                    for row in corpus:
                        inference.model(**model_outputs(inference, row["text"]))
            finally:
                handle.remove()
            if not calls:
                raise ValueError("GPTQ calibration never executed the selected module")
            reduced, artifacts = gptq_quantize(
                weights,
                hessian / calls,
                settings["bits"],
                settings["group_size"],
                settings["damping"],
                settings["act_order"],
                torch,
            )
        for key, tensor in artifacts.items():
            tensors[name + "." + key] = tensor.contiguous()
        records.append(
            {
                "module": name,
                "shape": list(weights.shape),
                "transposed": transpose,
                "max_error": float((weights.float() - reduced.float()).abs().max()),
                "mse": float((weights.float() - reduced.float()).square().mean()),
            }
        )
        with torch.no_grad():
            module.weight.copy_((reduced.T if transpose else reduced).to(module.weight.dtype))
    safetensors = importlib.import_module("safetensors.torch")
    safetensors.save_file(
        tensors,
        str(output / "lowbit.safetensors"),
        metadata={
            "method": operation,
            "representation": "quantized-codes",
            "configuration": json.dumps(settings, sort_keys=True),
        },
    )
    inference.model.config.ablitbench_quantization = {
        "method": operation,
        "representation": "dequantized",
        "settings": settings,
    }
    (output / "quantization.json").write_text(
        json.dumps({"method": operation, "settings": settings, "modules": records})
    )
    inference.model.save_pretrained(output / "model", safe_serialization=True)
    inference.tokenizer.save_pretrained(output / "model")
    return {
        "status": "completed",
        "artifacts": ["model", "lowbit.safetensors", "quantization.json"],
        "modules": records,
        "calibration_rows": len(corpus),
        "scope": (
            "Independent NF4 packed codes or Hessian GPTQ integer quantization "
            "with reconstructed safe model; not a fused low-bit inference kernel "
            "or mitigation efficacy result"
        ),
    }
