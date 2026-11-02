from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

REVISION = "9d2ef6be06fc034c001051c1b16856af0b8a9ab4"
EXPECTED_FUNCTIONS = {
    "leakage": "d7d93809e33a66cfe8998657c75e15d8392044066e50760d6ed2e1a357f64e6e",
    "motifs": "7ae48de93a91eb5f86950e5b2e0de08c36fb5f367cfd4b30995f6438ea8d02b2",
    "vocabulary": "b422832b22f5641077a9d209affaa6d48758f24a07e43250ac35dac4239bee29",
    "search": "a2e899122782df15e1515341c1103e02abae81a5ade8ba8a40c332698a0fe582",
    "baseline": "afd0d1ebdb2478854503ec58e2d80c6228a1879d9d0bebd39364a798c1ae8c54",
    "triggered": "e11ea297c1850ce0bd84fc34b1beda679b9874ba4167978792d9e4f1a23228a3",
    "validation": "d66509ce86e51bb9c5bf7695e70e64801bc59057b43a018314242f5252476dac",
}


def configuration(value: dict[str, Any]) -> dict[str, Any]:
    required = {"seed", "leakage", "motifs", "search", "validation"}
    if value.keys() - (required | {"context_tokens", "device"}):
        raise ValueError("Unknown discovery configuration keys")
    if not required <= value.keys():
        raise ValueError("Explicit discovery seed, leakage, motifs, search and validation required")
    if "context_tokens" in value and (
        type(value["context_tokens"]) is not int or value["context_tokens"] <= 0
    ):
        raise ValueError("context_tokens requires explicit positive capacity")
    if value.get("device", "cpu") != "cpu":
        raise ValueError("Only CPU exp1 method execution is currently verified")
    if type(value["seed"]) is not int:
        raise ValueError("Discovery seed must be an integer")
    for stage in required - {"seed"}:
        if not isinstance(value[stage], dict):
            raise ValueError(stage + " must be a mapping")
    leakage, search, validation = (value[key] for key in ("leakage", "search", "validation"))
    keys = {
        "leakage": {"prompts", "strategies", "max_new_tokens"},
        "motifs": {"perc_keep", "nonorm"},
        "search": {
            "prompts",
            "trigger_length",
            "num_iterations",
            "top_k",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "zeta",
            "gradient_based",
            "n_tk_before",
            "n_tk_after",
            "loss_tokens",
        },
        "validation": {"prompts", "max_new_tokens", "repeats", "target", "minimum_asr_increase"},
    }
    for stage, allowed in keys.items():
        if value[stage].keys() != allowed:
            raise ValueError(stage + " requires exactly its declared method parameters")
    for stage, settings in (("leakage", leakage), ("search", search), ("validation", validation)):
        prompts = settings.get("prompts")
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
        ):
            raise ValueError(stage + " requires explicit nonempty prompts")
    if set(validation["prompts"]) & (set(leakage["prompts"]) | set(search["prompts"])):
        raise ValueError("Validation prompts must be held out from leakage and search")
    integers = [
        (leakage, "max_new_tokens"),
        (search, "trigger_length"),
        (search, "num_iterations"),
        (search, "top_k"),
        (validation, "max_new_tokens"),
        (validation, "repeats"),
    ]
    for settings, key in integers:
        if type(settings.get(key)) is not int or settings[key] <= 0:
            raise ValueError(key + " requires an explicit positive integer budget")
    for key in ("n_tk_before", "n_tk_after"):
        if type(search.get(key)) is not int or search[key] < 0:
            raise ValueError(key + " requires an explicit nonnegative integer")
    if search.get("loss_tokens") != 10:
        raise ValueError(
            "Pinned upstream GCG exposes a fixed 10-token loss horizon; loss_tokens=10 required"
        )
    if type(search.get("gradient_based")) is not bool:
        raise ValueError("gradient_based must be explicit")
    for key in ("alpha", "beta", "gamma", "delta", "zeta"):
        weight = search.get(key)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError(key + " requires an explicit finite nonnegative loss weight")
    if not any(search[key] for key in ("gamma", "delta", "zeta")):
        raise ValueError("At least one search loss must be active")
    motifs = value["motifs"]
    keep = motifs.get("perc_keep")
    if (
        type(keep) not in (int, float)
        or not 0 < keep <= 1
        or type(motifs.get("nonorm")) is not bool
    ):
        raise ValueError("Explicit motif perc_keep in (0,1] and nonorm required")
    strategies = leakage.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("Explicit finite leakage strategy grids required")
    for strategy in strategies:
        if not isinstance(strategy, dict) or not isinstance(strategy.get("strategy"), str):
            raise ValueError("Each strategy requires its upstream name")
        if strategy.keys() != {"strategy", "grid", "seeds"}:
            raise ValueError("Unknown strategy keys")
        if type(strategy.get("seeds")) is not int or strategy["seeds"] <= 0:
            raise ValueError("Each strategy requires an explicit positive seed count")
        grid = strategy.get("grid")
        if not isinstance(grid, dict) or any(
            not isinstance(items, list) or not items for items in grid.values()
        ):
            raise ValueError("Each strategy requires a finite parameter grid")
        if "num_return_sequences" in grid and grid["num_return_sequences"] != [1]:
            raise ValueError("Pinned sweep retains one returned sequence per strategy")
        if {
            "max_new_tokens",
            "max_length",
            "min_length",
            "min_new_tokens",
            "stopping_criteria",
            "eos_token_id",
            "use_cache",
            "do_sample",
            "pad_token_id",
            "custom_generate",
            "trust_remote_code",
            "generation_config",
            "assistant_model",
        } & grid.keys():
            raise ValueError(
                "Strategy grid cannot override the explicit generation budget or upstream controls"
            )
    if not isinstance(validation.get("target"), str) or not validation["target"]:
        raise ValueError("Exp1 validation requires an explicit target behavior string")
    threshold = validation.get("minimum_asr_increase")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Explicit minimum_asr_increase in [0,1] required")
    return value


def artifact(directory: Path, name: str, value: Any) -> dict[str, str]:
    descriptor, temporary = tempfile.mkstemp(prefix="pending-", dir=directory)
    destination = directory / (name + ".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": destination.name, "sha256": checksum}


def progress(directory: Path, phase: str, payload: Any) -> None:
    with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "value": payload}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def model_identity(root: Path) -> str:
    checksum = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Model snapshot must not contain symlinks")
        if path.is_file():
            checksum.update(path.relative_to(root).as_posix().encode())
            with path.open("rb") as stream:
                checksum.update(hashlib.file_digest(stream, "sha256").digest())
    return checksum.hexdigest()


def installation(functions: dict[str, Any]) -> dict[str, Any]:
    requirements = importlib.metadata.requires("bdrscan") or []
    requirement_type = importlib.import_module("packaging.requirements").Requirement
    dependencies = []
    for text in requirements:
        requirement = requirement_type(text)
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies.append(
            {
                "requirement": text,
                "active": requirement.marker is None or requirement.marker.evaluate(),
                "installed": version,
                "satisfied": version is not None and version in requirement.specifier,
            }
        )
    return {
        "revision": REVISION,
        "version": importlib.metadata.version("bdrscan"),
        "functions": {
            name: {
                "module": function.__module__,
                "sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
            }
            for name, function in functions.items()
        },
        "declared_dependencies": dependencies,
        "scope": (
            "installed upstream CPU exp1 methods; cloud orchestration, QLoRA and CodeQL unavailable"
        ),
        "full_distribution_dependency_parity": all(
            item["satisfied"] for item in dependencies if item["active"]
        ),
    }


def run(localmodel: Path, workdir: Path, config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "engine": "bdrscan",
        "revision": REVISION,
        "method": "decoding-sweep/motif-extraction/GCG/held-out-exp1",
        "status": "unsupported",
        "finding": False,
        "gaps": [],
        "observations": {},
        "artifacts": {},
        "coverage": {},
        "limits": {
            "loss_tokens": 10,
            "gcg_candidate_chunk": 16,
            "search_initialization": "random within model-derived motif vocabulary",
            "momentum": False,
            "layers": "all",
            "adaptive_vocabulary_schedule": False,
            "motif_character_ngrams": [4, 6],
            "motif_min_df": 3,
            "motif_min_samples": 3,
            "motif_dbscan_eps": 0.65,
            "motif_min_length": 6,
            "upstream_materializes_strategy_runs": True,
            "returned_sequences": 1,
            "leakage_seed_schedule": "upstream 0..seeds-1; one seed selects greedy decoding",
            "validation_termination": "pinned upstream helper does not export finish reason",
            "population_entrypoint": (
                "unavailable: pinned function omits base tokens "
                "with nonzero default divergence loss"
            ),
        },
    }
    workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    diagnostic = workdir / "diagnostics.txt"
    try:
        configuration(config)
        from ablitbench.worker import isolation

        isolation()
        result["configuration"] = config
        progress(workdir, "parameters", config)
        result["seed"] = config["seed"]
        if config.get("device", "cpu") != "cpu":
            raise ValueError("Only CPU exp1 method execution is currently verified")
        if not localmodel.is_dir() or not any(localmodel.rglob("*.safetensors")):
            raise ValueError("Discovery requires a local Safetensors model directory")
        if (localmodel / "adapter_config.json").exists():
            raise ValueError("Discovery PEFT base-model application is not verified")
        with (
            diagnostic.open("w", encoding="utf-8", buffering=1) as logs,
            contextlib.redirect_stdout(logs),
            contextlib.redirect_stderr(logs),
        ):
            torch = importlib.import_module("torch")
            numpy = importlib.import_module("numpy")
            transformers = importlib.import_module("transformers")
            memorization = importlib.import_module("bdrscan.utils_memorization")
            motif = importlib.import_module("bdrscan.utils_motif")
            searcher = importlib.import_module("bdrscan.optimization.utils_search")
            upstream_model = importlib.import_module("bdrscan.utils_model")
            behavior = importlib.import_module("bdrscan.utils_bdr")
            convert = importlib.import_module("bdrscan.utils_eval").convert_for_json
            functions = {
                "leakage": memorization.decoding_strategy_sweep,
                "motifs": motif.find_interest_motifs,
                "vocabulary": motif.select_largest_cluster_tokens,
                "search": searcher.run_gradient_based_optimization,
                "baseline": upstream_model.get_response,
                "triggered": upstream_model.prepend_tokens_get_response,
                "validation": behavior.eval_behavior_presence,
            }
            result["installation"] = installation(functions)
            for name, item in result["installation"]["functions"].items():
                item["expected_sha256"] = EXPECTED_FUNCTIONS[name]
                if item["sha256"] != EXPECTED_FUNCTIONS[name]:
                    raise ValueError(
                        "Installed upstream method differs from pinned source: " + name
                    )
            progress(workdir, "installation", result["installation"])
            random.seed(config["seed"])
            numpy.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(localmodel), local_files_only=True, trust_remote_code=False
            )
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("Pinned method requires a tokenizer padding or EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            snapshot_identity = model_identity(localmodel)
            progress(workdir, "model_load_started", {"snapshot_sha256": snapshot_identity})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(localmodel),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                attn_implementation="eager",
                dtype=torch.float32,
            )
            model.eval()
            if model.generation_config.num_return_sequences not in (None, 1):
                raise ValueError("Pinned upstream sweep requires one model return sequence")
            metadata_capacities = [
                getattr(model.config, key, None)
                for key in ("max_position_embeddings", "n_positions", "max_sequence_length")
            ]
            capacities = [
                int(value) for value in metadata_capacities if isinstance(value, int) and value > 0
            ]
            context_tokens = min(capacities) if capacities else config.get("context_tokens")
            if type(context_tokens) is not int or context_tokens <= 0:
                raise ValueError("Model context capacity unknown; explicit context_tokens required")
            if config.get("context_tokens") is not None:
                context_tokens = min(context_tokens, config["context_tokens"])
            result["limits"]["context_tokens"] = context_tokens
            result["model"] = {
                "snapshot_sha256": snapshot_identity,
                "architecture": model.__class__.__name__,
                "dtype": str(model.dtype),
                "device": str(model.device),
                "generation_configuration": model.generation_config.to_dict(),
                "tokenizer": tokenizer.__class__.__name__,
                "vocabulary_sha256": hashlib.sha256(
                    json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
                ).hexdigest(),
            }
            progress(workdir, "model", {**result["model"], "context_tokens": context_tokens})
            original_generate = model.generate

            def observed_generate(*args: Any, **kwargs: Any) -> Any:
                inputs = kwargs.get("input_ids", args[0] if args else kwargs.get("inputs"))
                input_tokens = inputs.detach().cpu().tolist() if inputs is not None else None
                mask = kwargs.get("attention_mask")
                input_counts = (
                    mask.sum(dim=-1).detach().cpu().tolist()
                    if mask is not None
                    else ([len(row) for row in input_tokens] if input_tokens is not None else None)
                )
                progress(
                    workdir,
                    "generation_started",
                    {
                        "input_ids": input_tokens,
                        "input_tokens": input_counts,
                        "max_new_tokens": kwargs.get("max_new_tokens"),
                        "torch_seed": torch.initial_seed(),
                    },
                )
                output = original_generate(*args, **kwargs)
                sequences = output.sequences if hasattr(output, "sequences") else output
                progress(
                    workdir,
                    "generation_completed",
                    {
                        "input_ids": input_tokens,
                        "input_tokens": input_counts,
                        "output_ids": sequences.detach().cpu().tolist(),
                        "output_tokens": (
                            [len(row) - inputs.shape[-1] for row in sequences]
                            if inputs is not None
                            else None
                        ),
                        "max_new_tokens": kwargs.get("max_new_tokens"),
                    },
                )
                return output

            model.generate = observed_generate
            for stage in ("leakage", "search", "validation"):
                settings = config[stage]
                reserve = settings.get("max_new_tokens", 10)
                for prompt in settings["prompts"]:
                    encoded = upstream_model.tokenize_with_chat_template_batch([prompt], tokenizer)
                    if (
                        encoded["input_ids"].shape[1]
                        + reserve
                        + (0 if stage == "leakage" else config["search"]["trigger_length"])
                        > context_tokens
                    ):
                        raise ValueError(stage + " prompt and explicit budget exceed model context")
            leakage = config["leakage"]
            planned = len(leakage["prompts"]) * sum(
                math.prod(len(items) for items in entry["grid"].values()) * entry["seeds"]
                for entry in leakage["strategies"]
            )
            result["coverage"]["leakage"] = {"planned": planned, "inspected": 0}
            rows = []
            for prompt in leakage["prompts"]:
                frame = functions["leakage"](
                    model,
                    tokenizer,
                    prompt,
                    param_grid=leakage["strategies"],
                    max_new_tokens=leakage["max_new_tokens"],
                    apply_chat_template=True,
                    test_prompt_is_formatted=False,
                    use_cache=True,
                )
                for row in (
                    frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
                ):
                    row = convert(row)
                    rows.append(row)
                    progress(workdir, "leakage_trial", row)
                    if str(row["output"]).startswith("[ERROR:"):
                        result["gaps"].append({"stage": "leakage", "error": row["output"]})
                    else:
                        result["coverage"]["leakage"]["inspected"] += 1
            progress(workdir, "leakage", result["coverage"]["leakage"])
            result["observations"]["leakage"] = rows
            result["artifacts"]["leakage"] = artifact(workdir, "leakage", rows)
            outputs = [row["output"] for row in rows if not row["output"].startswith("[ERROR:")]
            motifs = functions["motifs"](outputs, tokenizer=tokenizer, **config["motifs"])
            progress(workdir, "motifs", convert(motifs))
            result["observations"]["motifs"] = convert(motifs)
            result["artifacts"]["motifs"] = artifact(workdir, "motifs", convert(motifs))
            if not motifs:
                result["status"] = "completed" if not result["gaps"] else "unsupported"
                for stage in ("search", "validation"):
                    result["coverage"][stage] = {
                        "status": "not_applicable",
                        "reason": "No upstream motifs produced",
                    }
                return result
            vocabulary = functions["vocabulary"](
                tokenizer, motifs, save=False, nonorm=config["motifs"]["nonorm"]
            )
            subset = [
                token
                for token in vocabulary["unique_token_ids"]
                if token not in tokenizer.all_special_ids
            ]
            search = config["search"]
            if len(subset) < search["trigger_length"] or search["top_k"] > len(subset):
                raise ValueError(
                    "Extracted motif vocabulary cannot satisfy explicit search dimensions"
                )
            settings = {
                key: search[key]
                for key in (
                    "trigger_length",
                    "num_iterations",
                    "top_k",
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "zeta",
                    "gradient_based",
                    "n_tk_before",
                    "n_tk_after",
                )
            }
            result["coverage"]["search"] = {"planned": search["num_iterations"], "inspected": 0}
            progress(workdir, "search_started", result["coverage"]["search"])
            history, candidate = functions["search"](
                model,
                tokenizer,
                search["prompts"],
                vocab_subset=subset,
                layers_to_evaluate=None,
                use_backdoor_aware_init=False,
                use_momentum=False,
                **settings,
            )
            progress(workdir, "search", convert({"history": history, "candidate": candidate}))
            result["coverage"]["search"] = {
                "planned": search["num_iterations"],
                "inspected": len(history),
            }
            result["observations"]["search"] = convert(
                {"history": history, "candidate": candidate, "vocabulary": vocabulary}
            )
            result["artifacts"]["search"] = artifact(
                workdir, "search", result["observations"]["search"]
            )
            validation = config["validation"]
            result["coverage"]["validation"] = {
                "planned": len(validation["prompts"]) * validation["repeats"],
                "inspected": 0,
            }
            progress(workdir, "validation_started", result["coverage"]["validation"])
            trials = []
            baseline = []
            triggered = []
            for prompt, repeat in itertools.product(
                validation["prompts"], range(validation["repeats"])
            ):
                _, clean = functions["baseline"](
                    model, prompt, tokenizer, max_new_tokens=validation["max_new_tokens"]
                )
                _, expanded, response = functions["triggered"](
                    model,
                    prompt,
                    tokenizer,
                    candidate,
                    search["n_tk_before"],
                    max_new_tokens=validation["max_new_tokens"],
                )
                baseline.append(clean)
                triggered.append(response)
                trials.append(
                    {
                        "prompt": prompt,
                        "repeat": repeat,
                        "baseline": clean,
                        "triggered_prompt": expanded,
                        "triggered": response,
                    }
                )
                result["coverage"]["validation"]["inspected"] += 1
                progress(workdir, "validation_trial", trials[-1])
            baseline_asr = functions["validation"](baseline, target_str=validation["target"])
            candidate_asr = functions["validation"](triggered, target_str=validation["target"])
            increase = candidate_asr - baseline_asr
            result["validation"] = {
                "held_out": True,
                "target": validation["target"],
                "baseline_asr": baseline_asr,
                "candidate_asr": candidate_asr,
                "increase": increase,
                "minimum_asr_increase": validation["minimum_asr_increase"],
            }
            result["observations"]["validation"] = trials
            result["artifacts"]["validation"] = artifact(workdir, "validation", trials)
            result["finding"] = increase > 0 and increase >= validation["minimum_asr_increase"]
            result["status"] = "completed" if not result["gaps"] else "unsupported"
    except Exception as error:
        result["gaps"].append(
            {"stage": "method", "type": type(error).__name__, "error": str(error)}
        )
    finally:
        progress(
            workdir,
            "completion",
            {"status": result["status"], "coverage": result["coverage"], "gaps": result["gaps"]},
        )
        if diagnostic.exists():
            result["diagnostics"] = diagnostic.read_text(encoding="utf-8")
        result["artifacts"]["result"] = artifact(workdir, "result", result)
    return result
