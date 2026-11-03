from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

from ablitbench.models import Generation

PROJECT = "refusal-direction"
REVISION = "9d852fae1a9121c78b29142de733cb1340770cc3"
OPERATIONS = {
    "directions": "train",
    "select": "evaluate",
    "intervene": "evaluate",
    "orthogonalize": "defend",
    "act-add-weights": "defend",
    "evaluate": "evaluate",
    "pipeline": "evaluate",
    "utility": "evaluate",
}


def operations() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "project": "refusal-direction",
            "kind": kind,
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "permissions": [],
            "inputs": ["source", "model"]
            + (
                ["dataset"]
                if name == "utility"
                else ["direction"]
                if name not in {"directions", "pipeline"}
                else []
            ),
            "optional_inputs": ["base", "judge-tokenizer"],
            "capabilities": ["local-model-internals"],
            "inputroles": ["source", "model", "base", "dataset", "direction", "judge-tokenizer"],
            "revision": REVISION,
        }
        for name, kind in OPERATIONS.items()
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in OPERATIONS:
        raise ValueError("Unknown refusal-direction operation")
    allowed = {
        "fit",
        "validation",
        "evaluation",
        "refusal_token_ids",
        "eoi_token_ids",
        "batch_size",
        "max_new_tokens",
        "context_tokens",
        "system",
        "device",
        "kl_threshold",
        "induce_refusal_threshold",
        "prune_layer_percentage",
        "position",
        "layer",
        "coefficient",
        "intervention",
        "methodologies",
        "judge_model",
        "utility_dataset",
        "max_length",
        "n_batches",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown refusal-direction settings")
    result = dict(settings)
    for key in ("refusal_token_ids", "eoi_token_ids"):
        values = result.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError("Explicit nonempty token ID arrays required: " + key)
        if len(set(values)) != len(values):
            raise ValueError("Duplicate token IDs")
    for key in ("fit", "validation", "evaluation"):
        if key not in result:
            continue
        group = result[key]
        if not isinstance(group, dict) or not group or set(group) - {"harmful", "harmless"}:
            raise ValueError("Prompt groups require harmful/harmless arrays")
        for prompts in group.values():
            if (
                not isinstance(prompts, list)
                or not prompts
                or any(not isinstance(p, str) or not p for p in prompts)
            ):
                raise ValueError("Nonempty prompt arrays required")
    if operation in {"directions", "pipeline"} and "fit" not in result:
        raise ValueError("Direction fitting prompts required")
    if operation in {"select", "pipeline"} and "validation" not in result:
        raise ValueError("Independent selection prompts required")
    if operation in {"intervene", "evaluate", "pipeline"} and "evaluation" not in result:
        raise ValueError("Independent evaluation prompts required")
    populations: list[set[str]] = []
    for key in ("fit", "validation", "evaluation"):
        population = {p for values in result.get(key, {}).values() for p in values}
        if any(population & previous for previous in populations):
            raise ValueError("Fitting, selection and evaluation prompts must be disjoint")
        populations.append(population)
    for key in ("batch_size", "max_new_tokens", "context_tokens"):
        if key in result and (type(result[key]) is not int or result[key] <= 0):
            raise ValueError("Positive explicit generation and processing capacity required")
    if operation in {"intervene", "evaluate", "pipeline"} and "max_new_tokens" not in result:
        raise ValueError("Explicit max_new_tokens required")
    if result.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("Unsupported device")
    if result.get("intervention", "ablation") not in {"baseline", "ablation", "actadd", "patch"}:
        raise ValueError("Unknown intervention")
    if operation == "act-add-weights" and (
        type(result.get("layer")) is not int or result["layer"] < 1
    ):
        raise ValueError("One-based target layer required for persistent activation addition")
    if operation == "utility":
        if result.get("utility_dataset") not in {"pile", "alpaca", "completions"}:
            raise ValueError("Original utility dataset method required")
        if (
            type(result.get("n_batches")) is not int
            or result["n_batches"] == 0
            or result["n_batches"] < -1
        ):
            raise ValueError(
                "Explicit positive utility batch budget or -1 for full dataset required"
            )
        if result["utility_dataset"] == "pile" and (
            type(result.get("max_length")) is not int or result["max_length"] <= 0
        ):
            raise ValueError("Explicit corpus context capacity required")
    methods = result.get("methodologies", ["substring_matching"])
    if (
        not isinstance(methods, list)
        or not methods
        or set(methods) - {"substring_matching", "llamaguard2", "harmbench"}
    ):
        raise ValueError("Unknown original jailbreak evaluation methodology")
    if set(methods) - {"substring_matching"} and not isinstance(result.get("judge_model"), str):
        raise ValueError("Semantic classifier requires an explicit host judge model identity")
    result["methodologies"] = methods
    return result


def journal(output: Path, stage: str, **values: Any) -> None:
    with (output / "refusal-progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def module(name: str, source: Path) -> Any:
    existing = sys.modules.get(name.split(".")[0])
    if existing is not None and not Path(str(existing.__file__)).resolve().is_relative_to(source):
        raise ValueError("Upstream module namespace collision")
    return importlib.import_module(name)


def view(inference: Any, settings: dict[str, Any], source: Path) -> Any:
    torch = inference.torch
    model, tokenizer = inference.model, inference.tokenizer
    model.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("A model pad or EOS token is required")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    from ablitbench.activation_engine import ARCHITECTURES

    kind = model.config.model_type
    if kind not in ARCHITECTURES:
        raise ValueError("Unsupported model block layout: " + kind)
    blocks = base
    for part in ARCHITECTURES[kind].split("."):
        blocks = getattr(blocks, part)
    attentions = [getattr(block, "self_attn", getattr(block, "attn", None)) for block in blocks]
    mlps = [getattr(block, "mlp", None) for block in blocks]
    if any(value is None for value in attentions + mlps):
        raise ValueError("Unsupported attention or MLP layout")
    limit = getattr(model.config, "max_position_embeddings", None)
    explicit = settings.get("context_tokens")
    if explicit is not None:
        limit = min(limit, explicit) if isinstance(limit, int) else explicit
    eoi = settings["eoi_token_ids"]
    refusal = settings["refusal_token_ids"]
    if (
        max(eoi + refusal) >= model.config.vocab_size
        or len(set(refusal)) >= model.config.vocab_size
    ):
        raise ValueError("Token IDs exceed the selected model vocabulary")

    def tokenize(instructions: list[str], outputs: list[str] | None = None) -> Any:
        texts = []
        for index, instruction in enumerate(instructions):
            messages = [{"role": "user", "content": instruction}]
            if settings.get("system") is not None:
                messages.insert(0, {"role": "system", "content": settings["system"]})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(prompt + (outputs[index] if outputs is not None else ""))
        encoded = tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt", add_special_tokens=False
        )
        if limit is not None and encoded.input_ids.shape[1] > limit:
            raise ValueError("Refusal input exceeds declared context capacity; no truncation")
        for row in encoded.input_ids:
            ids = row.tolist()
            if outputs is None and ids[-len(eoi) :] != eoi:
                raise ValueError("Declared EOI marker is not the encoded instruction suffix")
            if outputs is not None and not any(
                ids[i : i + len(eoi)] == eoi for i in range(len(ids))
            ):
                raise ValueError("Declared EOI marker absent from encoded teacher-forcing input")
        return encoded

    wrapper = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        tokenize_instructions_fn=tokenize,
        eoi_toks=eoi,
        refusal_toks=torch.tensor(refusal, device=model.device),
        model_block_modules=blocks,
        model_attn_modules=attentions,
        model_mlp_modules=mlps,
    )
    original = module("pipeline.model_utils.model_base", source).ModelBase
    wrapper.generate_completions = MethodType(original.generate_completions, wrapper)
    return wrapper


def classifiers(source: Path) -> Any:
    path = source / "pipeline/submodules/evaluate_jailbreak.py"
    tree = ast.parse(path.read_text())
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "substring_matching_judge_fn")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_test_prefixes_jailbreakbench"
                for t in node.targets
            )
        )
    ]
    namespace: dict[str, Any] = {"List": list}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["substring_matching_judge_fn"]


def semantic_judges(
    source: Path,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    prompts: list[str],
    responses: list[str],
    broker: Any,
    output: Path,
) -> dict[str, Any]:
    methods = set(settings["methodologies"]) - {"substring_matching"}
    if not methods:
        return {}
    if broker is None:
        raise ValueError("Semantic evaluation requires the host judge broker")
    path = source / "pipeline/submodules/evaluate_jailbreak.py"
    tree = ast.parse(path.read_text())
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LlamaGuard2Classifier"
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LLAMA2_CLS_PROMPT"
            for target in node.targets
        )
    ]
    calls: list[dict[str, Any]] = []

    def completion(messages: list[dict[str, str]], budget: int, finite_binary: bool = False) -> str:
        result = broker(
            {
                "kind": "judge",
                "channel": "judge",
                "expected_model": settings["judge_model"],
                "messages": messages,
                "max_output_tokens": budget,
                "generation_settings": {"temperature": 0},
            }
        )
        generation = Generation.model_validate(result)
        finite_label = (
            finite_binary
            and budget == 1
            and generation.finish in {"length", "max_tokens"}
            and generation.text.strip().casefold() in {"yes", "no"}
        )
        if generation.tools or generation.media or not (generation.complete or finite_label):
            raise ValueError("Classifier response failed or was truncated")
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Classifier response missing")
        calls.append(result)
        journal(
            output,
            "judge_completed",
            generation=result,
            finite_binary_label=finite_label,
            full_model_completion=generation.complete,
        )
        return text

    def batch_completion(**kwargs: Any) -> list[Any]:
        outputs = []
        for messages in kwargs["messages"]:
            text = completion(messages, settings["max_new_tokens"])
            if not text.strip().startswith(("safe", "unsafe")):
                raise ValueError("LlamaGuard returned an invalid label; evaluation inconclusive")
            outputs.append({"choices": [{"message": SimpleNamespace(content=text)}]})
        return outputs

    namespace: dict[str, Any] = {
        "litellm": SimpleNamespace(batch_completion=batch_completion),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    result: dict[str, Any] = {}
    if "llamaguard2" in methods:
        tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
            inputs["judge-tokenizer"], local_files_only=True, trust_remote_code=False
        )
        cls = namespace["LlamaGuard2Classifier"]
        classifier = object.__new__(cls)
        classifier._tokenizer, classifier._api_key = tokenizer, None
        result["llamaguard2"] = classifier.classify_responses(prompts, responses)
    if "harmbench" in methods:
        template = namespace["LLAMA2_CLS_PROMPT"]["prompt"]
        labels = []
        for prompt, response in zip(prompts, responses, strict=True):
            text = (
                completion(
                    [
                        {
                            "role": "user",
                            "content": template.format(behavior=prompt, generation=response),
                        }
                    ],
                    1,
                    finite_binary=True,
                )
                .strip()
                .casefold()
            )
            if text not in {"yes", "no"}:
                raise ValueError("HarmBench returned an invalid label; evaluation inconclusive")
            labels.append(text == "yes")
        result["harmbench"] = labels
    return result


def execute(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    broker: Any = None,
) -> dict[str, Any]:
    settings = configuration(operation, settings)
    from ablitbench.parity.sources import verified_source
    from ablitbench.worker import Inference, isolation

    isolation()
    source = inputs["source"].resolve()
    verified_source(source, PROJECT)
    output.mkdir(parents=True, exist_ok=True)
    journal(output, "started", operation=operation)
    sys.path.insert(0, str(source))
    try:
        inference = Inference(
            settings.get("device", "cpu"), inputs["model"], inputs.get("base", Path("/base"))
        )
        torch = inference.torch
        wrapper = view(inference, settings, source)
        hooks = module("pipeline.utils.hook_utils", source)
        generate = module("pipeline.submodules.generate_directions", source)
        selection = module("pipeline.submodules.select_direction", source)
        save_file = importlib.import_module("safetensors.torch").save_file
        load_file = importlib.import_module("safetensors.torch").load_file
        direction = None
        candidates = None
        records: list[dict[str, Any]] = []
        if operation in {"directions", "pipeline"}:
            fit = settings["fit"]
            with torch.no_grad():
                candidates = generate.get_mean_diff(
                    wrapper.model,
                    wrapper.tokenizer,
                    fit["harmful"],
                    fit["harmless"],
                    wrapper.tokenize_instructions_fn,
                    wrapper.model_block_modules,
                    batch_size=settings.get("batch_size", 32),
                    positions=list(range(-len(wrapper.eoi_toks), 0)),
                )
            if not torch.isfinite(candidates).all():
                raise ValueError("Candidate directions are nonfinite")
            save_file(
                {"directions": candidates.cpu().contiguous()},
                str(output / "candidate-directions.safetensors"),
            )
            journal(output, "directions_completed", shape=list(candidates.shape))
        if operation in {"select", "pipeline"}:
            if candidates is None:
                candidates = load_file(
                    str(inputs["direction"] / "candidate-directions.safetensors")
                )["directions"].to(wrapper.model.device)
            group = settings["validation"]
            with torch.no_grad():
                position, layer, direction = selection.select_direction(
                    wrapper,
                    group["harmful"],
                    group["harmless"],
                    candidates,
                    str(output),
                    kl_threshold=settings.get("kl_threshold", 0.1),
                    induce_refusal_threshold=settings.get("induce_refusal_threshold", 0.0),
                    prune_layer_percentage=settings.get("prune_layer_percentage", 0.2),
                    batch_size=settings.get("batch_size", 32),
                )
            if not torch.isfinite(direction).all() or direction.norm().item() == 0:
                raise ValueError("Selected direction is nonfinite or zero")
            save_file(
                {"direction": direction.cpu().contiguous()}, str(output / "direction.safetensors")
            )
            records.append({"selected_position": position, "selected_layer": layer})
            settings = settings | {"position": position, "layer": layer}
            journal(output, "selection_completed", position=position, layer=layer)
        if operation not in {"directions", "select", "utility"} and direction is None:
            direction = load_file(str(inputs["direction"] / "direction.safetensors"))[
                "direction"
            ].to(wrapper.model.device)
        if operation == "utility":
            loss = module("pipeline.submodules.evaluate_loss", source)
            dataset_path = inputs["dataset"]
            mode = settings["utility_dataset"]
            original_loader = loss.load_dataset
            if mode == "completions":
                iterator = loss.batch_iterator_custom_completions(
                    str(dataset_path),
                    wrapper.tokenize_instructions_fn,
                    settings.get("batch_size", 8),
                    torch.tensor(wrapper.eoi_toks),
                )
            else:
                datasets = importlib.import_module("datasets")
                rows = json.loads(dataset_path.read_text())
                if not isinstance(rows, list) or not rows:
                    raise ValueError("Nonempty mounted utility dataset required")
                local = datasets.Dataset.from_list(rows)

                def local_dataset(name: str, **kwargs: Any) -> Any:
                    expected = (
                        "tatsu-lab/alpaca" if mode == "alpaca" else "monology/pile-uncopyrighted"
                    )
                    if name != expected:
                        raise ValueError("Unexpected utility dataset request")
                    return local

                loss.load_dataset = local_dataset
                try:
                    iterator = (
                        loss.batch_iterator_alpaca(
                            wrapper.tokenize_instructions_fn,
                            settings.get("batch_size", 8),
                            torch.tensor(wrapper.eoi_toks),
                        )
                        if mode == "alpaca"
                        else loss.batch_iterator_pile(
                            wrapper.tokenizer,
                            settings.get("batch_size", 8),
                            settings["max_length"],
                        )
                    )
                    with torch.no_grad():
                        ce, perplexity, count = loss.compute_loss_over_dataset(
                            wrapper.model,
                            wrapper.tokenizer,
                            iterator,
                            n_batches=settings["n_batches"],
                        )
                finally:
                    loss.load_dataset = original_loader
            if mode == "completions":
                with torch.no_grad():
                    ce, perplexity, count = loss.compute_loss_over_dataset(
                        wrapper.model,
                        wrapper.tokenizer,
                        iterator,
                        n_batches=settings["n_batches"],
                    )
            if not torch.isfinite(ce) or count.item() == 0:
                raise ValueError("Utility dataset produced no valid target tokens")
            records.append(
                {
                    "utility_method": mode,
                    "ce_loss": ce.item(),
                    "perplexity": perplexity.item(),
                    "loss_tokens": count.item(),
                    "n_batches": settings["n_batches"],
                    "dataset": "Mounted independent utility dataset; no remote loading",
                }
            )
        if operation in {"orthogonalize", "act-add-weights"}:
            if (
                direction is None
                or not torch.isfinite(direction).all()
                or direction.norm().item() == 0
            ):
                raise ValueError("Valid nonzero direction required")
            architecture = wrapper.model.config.model_type
            family = {
                "llama": "llama2",
                "qwen": "qwen",
                "qwen2": "llama2",
                "gemma": "gemma",
                "gemma2": "gemma",
                "yi": "yi",
            }.get(architecture)
            if family is None:
                raise ValueError(
                    "Persistent upstream weight intervention unsupported for this architecture"
                )
            original = module("pipeline.model_utils." + family + "_model", source)
            function = getattr(
                original,
                ("orthogonalize_" if operation == "orthogonalize" else "act_add_")
                + family
                + "_weights",
            )
            arguments = (
                {}
                if operation == "orthogonalize"
                else {"coeff": settings.get("coefficient", 1.0), "layer": settings["layer"]}
            )
            function(wrapper.model, direction, **arguments)
            if operation == "act-add-weights":
                if not hasattr(wrapper.model.config, "mlp_bias"):
                    save_file(
                        {"direction": direction.cpu().contiguous()},
                        str(output / "activation-addition-direction.safetensors"),
                    )
                    journal(output, "portable_export_unsupported", architecture=architecture)
                    raise ValueError(
                        "Original activation addition inserts an MLP bias unsupported by this "
                        "architecture's safe loader; runtime actadd hooks remain available"
                    )
                wrapper.model.config.mlp_bias = True
                for block in wrapper.model_block_modules:
                    for projection in block.mlp.children():
                        if isinstance(projection, torch.nn.Linear) and projection.bias is None:
                            projection.bias = torch.nn.Parameter(
                                torch.zeros(
                                    projection.out_features,
                                    device=projection.weight.device,
                                    dtype=projection.weight.dtype,
                                )
                            )
                records.append(
                    {
                        "compatibility_fix": (
                            "Persist MLP bias configuration and zero equivalent missing biases, "
                            "so safe reload retains the original activation addition"
                        )
                    }
                )
            wrapper.model.save_pretrained(output / "model", safe_serialization=True)
            wrapper.tokenizer.save_pretrained(output / "model")
            records.append(
                {"derived_model": "model", "original_readonly": True, "intervention": operation}
            )
        if operation in {"intervene", "evaluate", "pipeline"}:
            modes = (
                ["baseline", "ablation", "actadd"]
                if operation == "pipeline"
                else [settings.get("intervention", "ablation")]
            )
            substring = classifiers(source)
            loss = module("pipeline.submodules.evaluate_loss", source)
            for mode in modes:
                layer = settings.get("layer", 0)
                if not 0 <= layer < len(wrapper.model_block_modules):
                    raise ValueError("Intervention layer outside model")
                pre: list[Any] = []
                post: list[Any] = []
                if mode == "ablation":
                    pre, post = hooks.get_all_direction_ablation_hooks(wrapper, direction)
                elif mode == "actadd":
                    pre = [
                        (
                            wrapper.model_block_modules[layer],
                            hooks.get_activation_addition_input_pre_hook(
                                direction, settings.get("coefficient", 1.0)
                            ),
                        )
                    ]
                elif mode == "patch":
                    pre = [
                        (
                            wrapper.model_block_modules[layer],
                            hooks.get_directional_patching_input_pre_hook(
                                direction, settings.get("coefficient", 1.0)
                            ),
                        )
                    ]
                for group, prompts in settings["evaluation"].items():
                    for prompt in prompts:
                        encoded = wrapper.tokenize_instructions_fn([prompt])
                        limit = getattr(wrapper.model.config, "max_position_embeddings", None)
                        if (
                            limit is not None
                            and encoded.input_ids.shape[1] + settings["max_new_tokens"] > limit
                        ):
                            raise ValueError("Generation exceeds declared model context capacity")
                    dataset = [{"instruction": prompt, "category": group} for prompt in prompts]
                    with torch.no_grad():
                        completions = wrapper.generate_completions(
                            dataset,
                            pre,
                            post,
                            settings.get("batch_size", 8),
                            settings["max_new_tokens"],
                        )
                    name = group + "-" + mode + ".json"
                    (output / name).write_text(
                        json.dumps(completions, ensure_ascii=False), encoding="utf-8"
                    )
                    iterator = loss.batch_iterator_custom_completions(
                        str(output / name),
                        wrapper.tokenize_instructions_fn,
                        settings.get("batch_size", 8),
                        torch.tensor(wrapper.eoi_toks),
                    )
                    with torch.no_grad():
                        ce, perplexity, count = loss.compute_loss_over_dataset(
                            wrapper.model,
                            wrapper.tokenizer,
                            iterator,
                            n_batches=-1,
                            fwd_pre_hooks=pre,
                            fwd_hooks=post,
                        )
                    result = {
                        "mode": mode,
                        "group": group,
                        "completions": name,
                        "refusal_substring_count": sum(
                            substring(row["response"]) for row in completions
                        ),
                        "ce_loss": ce.item(),
                        "perplexity": perplexity.item(),
                        "loss_tokens": count.item(),
                    }
                    judges = semantic_judges(
                        source,
                        settings,
                        inputs,
                        prompts,
                        [row["response"] for row in completions],
                        broker,
                        output,
                    )
                    with (output / (group + "-" + mode + "-judgements.jsonl")).open(
                        "w", encoding="utf-8"
                    ) as judged:
                        for index, completion in enumerate(completions):
                            judged.write(
                                json.dumps(
                                    {
                                        "index": index,
                                        "substring": substring(completion["response"]),
                                        "judges": {
                                            name: values[index] for name, values in judges.items()
                                        },
                                    }
                                )
                                + "\n"
                            )
                    result["judges"] = {
                        name: {"positive": sum(values), "total": len(values)}
                        for name, values in judges.items()
                    }
                    if not torch.isfinite(ce) or count.item() == 0:
                        raise ValueError("Utility loss has no valid target tokens")
                    records.append(result)
                    journal(output, "evaluation_completed", **result)
        result = {
            "project": PROJECT,
            "operation": operation,
            "status": "completed",
            "upstream_revision": REVISION,
            "records": records,
            "scope": (
                "Original direction search/selection/hooks/loss functions; "
                "safe local model loading and explicit disjoint data"
            ),
            "limits": {
                "selection": {
                    key: settings.get(key, default)
                    for key, default in (
                        ("kl_threshold", 0.1),
                        ("induce_refusal_threshold", 0.0),
                        ("prune_layer_percentage", 0.2),
                    )
                },
                "substring_judge": (
                    "Original lexical refusal proxy; not a semantic harmfulness verdict"
                ),
            },
        }
        (output / "refusal-result.json").write_text(
            json.dumps(result, allow_nan=False, ensure_ascii=False), encoding="utf-8"
        )
        return result
    finally:
        sys.path.remove(str(source))
