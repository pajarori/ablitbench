from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT = "llm-backdoor-scanner"
REVISION = "9d2ef6be06fc034c001051c1b16856af0b8a9ab4"
STAGES = {
    "finetune": ("finetuning", "local_finetune", "train"),
    "leakage": ("leakage", "local_leakage_analysis", "detect"),
    "motifs": ("motif_extraction", "run_motif_analysis_pipeline", "detect"),
    "search": ("search_motifs", "search_motifs", "detect"),
    "optimize": ("joint_optimization", "joint_opt", "detect"),
    "inference": ("inference", "local_inference", "evaluate"),
    "codeql": ("codeql_scanner", "main", "evaluate"),
}


def operations() -> dict[str, dict[str, Any]]:
    required = {
        "finetune": ["model", "dataset"],
        "leakage": ["model"],
        "motifs": ["model", "leakage"],
        "search": ["model", "motifs"],
        "optimize": ["model"],
        "inference": ["model", "dataset"],
        "codeql": ["dataset", "codeql"],
    }
    return {
        name: {
            "project": PROJECT,
            "kind": stage[2],
            "requires_inference": name not in {"motifs", "codeql"},
            "requires_tools": False,
            "modes": ["source"],
            "permissions": [],
            "inputs": ["source", *required[name]],
            "optional_inputs": [
                role
                for role in [
                    "base",
                    "tokenizer",
                    "validation",
                    "grid",
                    "vocabulary",
                    "deepspeed",
                    "codeql",
                ]
                if role not in required[name]
            ],
            "capabilities": [
                "local-model-training" if name == "finetune" else "local-model-internals"
            ],
            "inputroles": [
                "source",
                "model",
                "base",
                "tokenizer",
                "dataset",
                "validation",
                "leakage",
                "motifs",
                "grid",
                "vocabulary",
                "deepspeed",
                "codeql",
            ],
            "revision": REVISION,
            "entrypoint": "scripts/orchestration/" + stage[0] + ".py:" + stage[1],
        }
        for name, stage in STAGES.items()
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in STAGES:
        raise ValueError("Unknown Microsoft scanner operation")
    if set(settings) - {
        "parameters",
        "method",
        "experiment",
        "seed",
        "device",
        "prompts",
        "evaluation_prompts",
    }:
        raise ValueError("Unknown Microsoft scanner settings")
    result = dict(settings)
    parameters = result.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Explicit original-method parameters required")
    schema = json.loads(
        (Path(__file__).parent.parent / "data/parity-probes-settings.json").read_text()
    )
    if set(parameters) - set(schema["microsoft"][operation]["parameters"]):
        raise ValueError("Unknown original-method parameter; no silently ignored settings")
    if result.get("method") not in {"fft", "lora", "qlora_4bit", "qlora_8bit"}:
        raise ValueError("Explicit fft/lora/qlora_4bit/qlora_8bit method required")
    if result.get("experiment") not in {"exp1", "exp2"}:
        raise ValueError("Explicit exp1/exp2 experiment required")
    if type(result.get("seed")) is not int:
        raise ValueError("Explicit integer seed required")
    if result.get("device") not in {"cpu", "cuda:0"}:
        raise ValueError("Explicit worker device required")
    if result["method"].startswith("qlora") and result["device"] == "cpu":
        raise ValueError("Original QLoRA method requires an accelerator worker")
    prohibited = {
        "model_dir",
        "tokenizer",
        "base_model_dir",
        "model_checkpoint",
        "train_data",
        "val_data",
        "results_dir",
        "out_dir",
        "dataset_dir",
        "motif_summary",
        "leakage_results_dir",
        "cache_dir",
        "hf_name",
        "deepspeed_config",
        "device_map",
        "experiment",
        "seed",
    }
    if set(parameters) & prohibited:
        raise ValueError("Paths, routing and device are provided by named worker inputs")
    pending = [parameters]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if any(str(k).casefold() in {"api_key", "secret", "password", "token"} for k in value):
                raise ValueError("Worker settings cannot contain provider credentials")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if operation == "finetune":
        for key in ("num_train_epochs", "per_device_train_batch_size", "max_seq_length"):
            if not isinstance(parameters.get(key), (int, float)) or parameters[key] <= 0:
                raise ValueError("Explicit training schedule/capacity required: " + key)
    if operation in {"search", "optimize"}:
        for key in ("prompts", "evaluation_prompts"):
            values = result.get(key)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(p, str) or not p for p in values)
            ):
                raise ValueError("Explicit optimization and held-out evaluation prompts required")
        if set(result["prompts"]) & set(result["evaluation_prompts"]):
            raise ValueError("Optimization and held-out evaluation prompts overlap")
    return result


def upstream(source: Path, name: str) -> Any:
    path = source / "scripts/orchestration" / (name + ".py")
    spec = importlib.util.spec_from_file_location("ablitbench_upstream_bdrscan_" + name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Original scanner stage is unavailable")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def defaults(source: Path, name: str) -> list[dict[str, Any]]:
    path = source / "scripts/orchestration" / (name + ".py")
    rows = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or len(node.args) != 2:
            continue
        try:
            key, value = (ast.literal_eval(argument) for argument in node.args)
        except (ValueError, TypeError):
            continue
        rows.append({"key": key, "default": value, "line": node.lineno})
    return rows


def method_limits(source: Path, stage: str) -> list[dict[str, Any]]:
    paths = [source / "scripts/orchestration" / (stage + ".py")]
    paths.extend((source / "src/bdrscan/optimization").glob("*.py"))
    paths.extend((source / "src/bdrscan/losses").glob("*.py"))
    fields = {"max_new_tokens", "max_length", "n_tokens_to_gen", "n_batches", "batch_size", "topk"}
    rows = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in fields or not isinstance(keyword.value, ast.Constant):
                    continue
                rows.append(
                    {
                        "source": str(path.relative_to(source)),
                        "line": node.lineno,
                        "parameter": keyword.arg,
                        "value": keyword.value.value,
                        "scope": "Original method internal limit, not a clean coverage guarantee",
                    }
                )
    return rows


@contextmanager
def safe_loaders(inputs: dict[str, Path]) -> Any:
    transformers = importlib.import_module("transformers")
    classes = [
        transformers.AutoConfig,
        transformers.AutoTokenizer,
        transformers.AutoModelForCausalLM,
    ]
    allowed = [path.resolve() for path in inputs.values()]
    original = []
    try:
        for cls in classes:
            descriptor = cls.__dict__.get("from_pretrained")
            bound = cls.from_pretrained
            original.append((cls, descriptor))

            def guarded(
                target: Any,
                path: Any,
                /,
                *args: Any,
                _original: Any = bound,
                _model: bool = cls is transformers.AutoModelForCausalLM,
                **kwargs: Any,
            ) -> Any:
                candidate = Path(path).resolve()
                if not candidate.exists() or not any(
                    candidate == root or candidate.is_relative_to(root) for root in allowed
                ):
                    raise ValueError("Original loader requested an unmounted model asset")
                kwargs["local_files_only"] = True
                kwargs["trust_remote_code"] = False
                if _model:
                    kwargs["use_safetensors"] = True
                return _original(str(candidate), *args, **kwargs)

            cls.from_pretrained = classmethod(guarded)
        yield
    finally:
        for cls, descriptor in original:
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
    output.mkdir(parents=True, exist_ok=True)
    stage, entrypoint, _ = STAGES[operation]
    existing = sys.modules.get("bdrscan")
    if existing is not None and not Path(str(existing.__file__)).resolve().is_relative_to(source):
        raise ValueError("Original scanner namespace collision")
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "scripts/orchestration"))
    environment = dict(os.environ)
    try:
        os.environ.update(
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        )
        if settings["experiment"] == "exp2" or operation == "codeql":
            codeql = inputs["codeql"].resolve()
            os.environ["CODEQL_HOME"] = str(codeql)
            os.environ["PATH"] = str(codeql / "codeql") + os.pathsep + os.environ.get("PATH", "")
        args = dict(settings["parameters"])
        args.update(
            {
                "model_dir": str(inputs.get("model", output)),
                "tokenizer": str(inputs.get("tokenizer", inputs.get("model", output))),
                "results_dir": str(output),
                "out_dir": str(output / "trained"),
                "experiment": settings["experiment"],
                "seed": settings["seed"],
                "experiment_seed": settings["seed"],
                "device_map": settings["device"],
                "adapter_type": None
                if settings["method"] == "fft"
                else ("qlora" if settings["method"].startswith("qlora") else "lora"),
            }
        )
        if settings["method"].startswith("qlora"):
            args["qlora_bits"] = 4 if settings["method"].endswith("4bit") else 8
        paths = {
            "base": "base_model_dir",
            "dataset": "train_data" if operation == "finetune" else "dataset_dir",
            "validation": "val_data",
            "leakage": "leakage_results_dir",
            "motifs": "motif_summary",
            "deepspeed": "deepspeed_config",
        }
        for role, key in paths.items():
            if role in inputs:
                args[key] = str(inputs[role])
        if "grid" in inputs:
            args.setdefault("decoding_sweep", {})["param_grid"] = str(inputs["grid"])
        if "vocabulary" in inputs:
            args.setdefault("vocabulary", {})["vocab_file"] = str(inputs["vocabulary"])
        if operation in {"search", "optimize"}:
            args["base_prompts"] = settings["prompts"]
            args["evaluation_prompts"] = settings["evaluation_prompts"]
        (output / "method-configuration.json").write_text(json.dumps(args, ensure_ascii=False))
        with safe_loaders(inputs):
            module = upstream(source, stage)
            compatibility = []
            if operation == "finetune":
                original_training_arguments = module.TrainingArguments

                def training_arguments(**kwargs: Any) -> Any:
                    if "logging_dir" in kwargs:
                        directory = kwargs.pop("logging_dir")
                        compatibility.append(
                            {
                                "parameter": "logging_dir",
                                "path": directory,
                                "adaptation": (
                                    "Transformers 5 removed logging_dir; "
                                    "native local metrics callback retained"
                                ),
                            }
                        )
                    kwargs["use_cpu"] = settings["device"] == "cpu"
                    return original_training_arguments(**kwargs)

                module.TrainingArguments = training_arguments
            if operation in {"search", "optimize"}:
                module.base_prompts = list(settings["prompts"])
                module.evaluation_prompts = list(settings["evaluation_prompts"])
                if hasattr(module, "base_code_prompts"):
                    module.base_code_prompts = list(settings["prompts"])
                    module.evaluation_code_prompts = list(settings["evaluation_prompts"])
            if operation == "motifs":
                value = module.run_motif_analysis_pipeline(
                    leakage_results_dir=inputs["leakage"],
                    model_dir=inputs["model"],
                    out_dir=output,
                    tokenizer_path=args["tokenizer"],
                    **settings["parameters"],
                )
            elif operation == "codeql":
                value = codeql_scan(source, inputs["dataset"], output)
            else:
                value = getattr(module, entrypoint)(args)
        artifact_count = 0
        manifest = output / "discovery-artifacts.jsonl"
        with manifest.open("w", encoding="utf-8") as journal:
            for path in output.rglob("*"):
                if path.is_file() and path != manifest:
                    journal.write(
                        json.dumps(
                            {"path": str(path.relative_to(output)), "sha256": hash_file(path)}
                        )
                        + "\n"
                    )
                    artifact_count += 1
        if artifact_count <= 1:
            raise ValueError("Original method produced no assessment/training artifacts")
        result = {
            "project": PROJECT,
            "operation": operation,
            "status": "completed",
            "upstream_revision": REVISION,
            "method": settings["method"],
            "experiment": settings["experiment"],
            "upstream_defaults": defaults(source, stage),
            "method_limits": method_limits(source, stage),
            "artifact_count": artifact_count,
            "artifact_manifest": manifest.name,
            "return_type": type(value).__name__,
            "upstream_compatibility_fixes": compatibility,
            "scope": (
                "Original stage implementation with safe local assets; no remote repository code"
            ),
            "coverage": {
                "paper_accuracy_reproduced": False,
                "unconfigured_stages": "not requested",
            },
        }
        (output / "discovery-result.json").write_text(json.dumps(result, allow_nan=False))
        return result
    finally:
        sys.path.remove(str(source / "scripts/orchestration"))
        sys.path.remove(str(source / "src"))
        os.environ.clear()
        os.environ.update(environment)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def codeql_scan(source: Path, dataset: Path, output: Path) -> Any:
    helpers = importlib.import_module("bdrscan.utils_codeql")
    helpers.validate_codeql_environment()
    pandas = importlib.import_module("pandas")
    frame = pandas.read_csv(dataset)
    snippets = output / "snippets"
    snippets.mkdir()
    index = {}
    for row_id, row in frame.iterrows():
        code = helpers.extract_code(row["generated_response"])
        if code:
            filename = "sample_" + str(row_id) + ".py"
            (snippets / filename).write_text(code)
            index[row_id] = filename
    if not index:
        raise ValueError("No extractable code; all rows remain unexamined")
    alerts, mapping = helpers.run_codeql_analysis(snippets, output)
    findings = helpers.process_codeql_alerts(alerts, mapping, snippets)
    findings.to_csv(output / "codeql-findings.csv", index=False)
    (output / "codeql-coverage.json").write_text(
        json.dumps(
            {
                "examined_rows": list(index),
                "unexamined_rows": [i for i in frame.index if i not in index],
                "target_cwes": helpers.TARGET_CWES,
            }
        )
    )
    return findings
