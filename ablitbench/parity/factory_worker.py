from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import ipaddress
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

STAGES = {"pt", "sft", "dpo", "kto", "rm", "ppo"}
BOOL_OPTIONS = {
    "do_sample",
    "use_fast_tokenizer",
    "resize_vocab",
    "split_special_tokens",
    "upcast_layernorm",
    "upcast_lmhead_output",
    "use_cache",
    "shift_attn",
    "double_quantization",
    "train_on_prompt",
    "ignore_pad_token_for_loss",
    "packing",
    "streaming",
    "overwrite_cache",
    "gradient_checkpointing",
    "disable_gradient_checkpointing",
    "use_rslora",
    "use_llama_pro",
    "freeze_vision_tower",
    "train_mm_proj_only",
    "use_dora",
    "pissa_init",
    "pissa_convert",
    "create_new_adapter",
    "pure_bf16",
    "ppo_score_norm",
    "ppo_whiten_rewards",
    "use_galore",
    "galore_layerwise",
    "use_badam",
    "plot_loss",
    "ddp_find_unused_parameters",
    "full_determinism",
}
INT_OPTIONS = {
    "badam_verbose",
    "lora_alpha",
    "lora_rank",
    "pissa_iter",
    "freeze_trainable_layers",
    "max_steps",
    "preprocessing_num_workers",
    "reserved_label_len",
    "buffer_size",
    "ppo_buffer_size",
    "ppo_epochs",
    "galore_rank",
    "galore_update_interval",
    "badam_start_block",
    "badam_switch_interval",
    "save_steps",
    "logging_steps",
    "dataloader_num_workers",
    "num_beams",
    "max_new_tokens",
    "top_k",
    "ddp_timeout",
    "warmup_steps",
    "export_quantization_bit",
    "quantization_bit",
}
FLOAT_OPTIONS = {
    "lora_dropout",
    "loraplus_lr_ratio",
    "loraplus_lr_embedding",
    "pref_beta",
    "pref_ftx",
    "dpo_label_smoothing",
    "kto_chosen_weight",
    "kto_rejected_weight",
    "simpo_gamma",
    "ppo_target",
    "galore_scale",
    "badam_update_ratio",
    "val_size",
    "weight_decay",
    "max_grad_norm",
    "warmup_ratio",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "temperature",
    "top_p",
    "repetition_penalty",
    "length_penalty",
    "neftune_noise_alpha",
}
STRING_OPTIONS = {
    "lora_target",
    "additional_target",
    "freeze_trainable_modules",
    "freeze_extra_modules",
    "galore_target",
    "badam_trainable_modules",
    "new_special_tokens",
    "default_system",
}
ENUM_OPTIONS = {
    "pref_loss": {"sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"},
    "reward_model_type": {"lora", "full"},
    "galore_proj_type": {"std", "reverse_std", "right", "left", "full"},
    "badam_mode": {"layer", "ratio"},
    "badam_switch_mode": {"ascending", "descending", "random", "fixed"},
    "badam_mask_mode": {"adjacent", "scatter"},
    "flash_attn": {"auto", "disabled", "sdpa", "fa2"},
    "rope_scaling": {"linear", "dynamic"},
    "quantization_method": {"bitsandbytes", "hqq", "eetq"},
    "quantization_type": {"fp4", "nf4"},
    "optim": {
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_hf",
        "adafactor",
        "sgd",
        "adagrad",
        "adamw_bnb_8bit",
        "paged_adamw_8bit",
        "paged_adamw_32bit",
    },
    "lr_scheduler_type": {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_lr_on_plateau",
    },
    "save_strategy": {"no", "steps", "epoch"},
}
COLUMN_NAMES = {
    "prompt",
    "query",
    "response",
    "history",
    "messages",
    "system",
    "tools",
    "chosen",
    "rejected",
    "kto_tag",
}
TAG_NAMES = {
    "role_tag",
    "content_tag",
    "user_tag",
    "assistant_tag",
    "observation_tag",
    "function_tag",
    "system_tag",
}


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation == "factory.export":
        required = {"device", "template", "finetuning_type", "export_shard_gb", "options"}
    elif operation in {"factory.train", "factory.plan", "factory.distributed"}:
        required = {
            "stage",
            "device",
            "template",
            "finetuning_type",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "cutoff_len",
            "seed",
            "dtype",
            "world_size",
            "format",
            "columns",
            "tags",
            "options",
        }
    else:
        raise ValueError("Unknown reviewed factory workflow")
    optional = set() if operation == "factory.export" else {"deepspeed", "topology"}
    if operation == "factory.distributed":
        optional.update({"managed_cluster", "cluster_devices"})
    if set(settings) - optional != required:
        raise ValueError("Factory settings require exactly " + ", ".join(sorted(required)))
    if settings["device"] not in {"cpu", "cuda"}:
        raise ValueError("Factory device must be cpu or cuda")
    if settings["finetuning_type"] not in {"full", "freeze", "lora"}:
        raise ValueError("Unknown factory fine-tuning method")
    if not isinstance(settings["template"], str) or not settings["template"]:
        raise ValueError("Explicit reviewed tokenizer template name required")
    if operation == "factory.export":
        if type(settings["export_shard_gb"]) is not int or settings["export_shard_gb"] <= 0:
            raise ValueError("Export shard I/O unit must be a positive integer GiB")
    else:
        if settings["stage"] not in STAGES or settings["dtype"] not in {
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError("Unknown factory stage or model dtype")
        for name in ("batch_size", "gradient_accumulation_steps", "cutoff_len", "world_size"):
            if type(settings[name]) is not int or settings[name] <= 0:
                raise ValueError("Positive explicit factory budget required: " + name)
        if (
            operation == "factory.distributed"
            and settings["world_size"] < 2
            and not isinstance(settings.get("topology"), dict)
        ):
            raise ValueError(
                "Distributed workflow requires at least two explicitly requested ranks"
            )
        if type(settings["seed"]) is not int:
            raise ValueError("Integer factory seed required")
        for name in ("epochs", "learning_rate"):
            if (
                type(settings[name]) not in {int, float}
                or not math.isfinite(settings[name])
                or settings[name] <= 0
            ):
                raise ValueError("Positive finite factory training budget required: " + name)
        if settings["format"] not in {"alpaca", "sharegpt"}:
            raise ValueError("Factory dataset must use an explicit supported local schema")
        for name, allowed in (("columns", COLUMN_NAMES), ("tags", TAG_NAMES)):
            values = settings[name]
            if (
                not isinstance(values, dict)
                or set(values) - allowed
                or any(not isinstance(value, str) or not value for value in values.values())
            ):
                raise ValueError("Unknown factory dataset mapping field")
        if settings["stage"] == "ppo" and "max_new_tokens" not in settings["options"]:
            raise ValueError("PPO requires an explicit rollout generation token budget")
    if "deepspeed" in settings:
        geometry = settings["deepspeed"]
        if (
            not isinstance(geometry, dict)
            or set(geometry)
            != {
                "stage",
                "offload_optimizer",
                "offload_parameters",
                "overlap_communication",
                "contiguous_gradients",
            }
            or type(geometry["stage"]) is not int
            or geometry["stage"] not in {1, 2, 3}
            or geometry["offload_optimizer"] not in {"none", "cpu"}
            or geometry["offload_parameters"] not in {"none", "cpu"}
            or any(
                type(geometry[key]) is not bool
                for key in ("overlap_communication", "contiguous_gradients")
            )
        ):
            raise ValueError("Invalid typed DeepSpeed ZeRO/offload geometry")
        if geometry["stage"] != 3 and geometry["offload_parameters"] != "none":
            raise ValueError("Parameter offload requires ZeRO stage 3")
    if "topology" in settings:
        topology = settings["topology"]
        if (
            not isinstance(topology, dict)
            or set(topology) != {"nodes", "node_rank", "master_address", "master_port"}
            or type(topology["nodes"]) is not int
            or topology["nodes"] < 2
            or type(topology["node_rank"]) is not int
            or not 0 <= topology["node_rank"] < topology["nodes"]
            or not isinstance(topology["master_address"], str)
            or type(topology["master_port"]) is not int
            or not 1 <= topology["master_port"] <= 65535
        ):
            raise ValueError("Explicit fixed multinode topology required")
        try:
            address = ipaddress.ip_address(topology["master_address"])
        except ValueError as error:
            if (
                not isinstance(topology["master_address"], str)
                or re.fullmatch(r"ablib-[a-z0-9-]+", topology["master_address"]) is None
            ):
                raise ValueError(
                    "Multinode rendezvous requires private IPv4 or a controlled ablib-* peer alias"
                ) from error
        else:
            if (
                address.version != 4
                or not address.is_private
                or address.is_unspecified
                or address.is_multicast
                or address.is_loopback
                or address.is_link_local
            ):
                raise ValueError(
                    "Multinode rendezvous requires a private controlled research-network address"
                )
    if operation == "factory.distributed":
        managed = settings.get("managed_cluster", False)
        if type(managed) is not bool:
            raise ValueError("Managed cluster selection must be explicit boolean")
        if managed and (
            not isinstance(settings.get("topology"), dict) or settings["topology"]["node_rank"] != 0
        ):
            raise ValueError("Managed cluster coordinator requires a multinode rank-zero topology")
    if "cluster_devices" in settings:
        devices = settings["cluster_devices"]
        topology = settings.get("topology")
        if (
            operation != "factory.distributed"
            or settings["device"] != "cuda"
            or not isinstance(topology, dict)
        ):
            raise ValueError("GPU cluster geometry requires explicit CUDA multinode workflow")
        if not isinstance(devices, list) or len(devices) != topology["nodes"]:
            raise ValueError("GPU device groups must match the exact node count")
        uuid = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        pattern = re.compile(
            r"(?:GPU-" + uuid + r"|MIG-" + uuid + r"|MIG-GPU-" + uuid + r"/[0-9]+/[0-9]+)"
        )
        all_devices: list[str] = []
        for group in devices:
            if (
                not isinstance(group, list)
                or len(group) < settings["world_size"]
                or any(
                    not isinstance(device, str) or pattern.fullmatch(device) is None
                    for device in group
                )
            ):
                raise ValueError(
                    "Each GPU/MIG UUID group must cover every local rank; "
                    "ambiguous indices are forbidden"
                )
            all_devices.extend(device.casefold() for device in group)
        if len(all_devices) != len(set(all_devices)):
            raise ValueError("GPU/MIG UUIDs must be disjoint within and across nodes")
    if (
        settings.get("managed_cluster")
        and settings["device"] == "cuda"
        and "cluster_devices" not in settings
    ):
        raise ValueError("Managed CUDA cluster requires explicit per-node GPU/MIG UUID allocation")
    options = settings["options"]
    if not isinstance(options, dict):
        raise ValueError("Factory options must be a typed mapping")
    if "deepspeed" in settings and (
        type(options.get("max_grad_norm")) not in {int, float}
        or not math.isfinite(options["max_grad_norm"])
        or options["max_grad_norm"] < 0
    ):
        raise ValueError("DeepSpeed requires explicit finite gradient clipping budget")
    if (settings.get("world_size", 1) > 1 or settings.get("topology")) and (
        type(options.get("ddp_timeout")) is not int or options["ddp_timeout"] <= 0
    ):
        raise ValueError(
            "Distributed process-group coordination requires an explicit protocol timeout"
        )
    for key, value in options.items():
        if key in BOOL_OPTIONS:
            valid = type(value) is bool
        elif key in INT_OPTIONS:
            valid = type(value) is int
        elif key in FLOAT_OPTIONS:
            valid = type(value) in {int, float} and math.isfinite(value)
        elif key in STRING_OPTIONS:
            valid = isinstance(value, str)
        elif key in ENUM_OPTIONS:
            valid = isinstance(value, str) and value in ENUM_OPTIONS[key]
        else:
            valid = False
        if not valid:
            raise ValueError("Unknown or invalid typed factory option: " + key)
    return (
        {**settings, "managed_cluster": settings.get("managed_cluster", False)}
        if operation == "factory.distributed"
        else dict(settings)
    )


def strict_json(path: Path) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant
    )


def verified_source(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("Reviewed factory source cannot be a symlink")
    root = root.resolve(strict=True)
    manifest = strict_json(Path(__file__).parent.parent / "data/parity-sources.json")
    specification = manifest["repositories"]["backdoorllm"]
    files = specification["files"]
    for name, record in files.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError("Reviewed factory source type/size mismatch")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError("Reviewed factory source hash mismatch")
    for parent, directories, names in os.walk(root, followlinks=False):
        if any(
            Path(parent, name).is_symlink() or name in {".git", "__pycache__"}
            for name in directories
        ):
            raise ValueError("Unreviewed factory source directory")
        if any(Path(parent, name).relative_to(root).as_posix() not in files for name in names):
            raise ValueError("Unreviewed factory source file")
    return root


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".factory-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def isolated_environment(output: Path) -> dict[str, str]:
    environment = {
        "PATH": "/opt/ablitbench-factory/bin:/usr/bin:/bin",
        "HOME": str(output / "private-home"),
        "HF_HOME": str(output / "cache/hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "GLOO_SOCKET_IFNAME": "lo",
        "WANDB_DISABLED": "true",
        "DO_NOT_TRACK": "1",
        "MPLCONFIGDIR": str(output / "cache/matplotlib"),
        "TMPDIR": str(output / "tmp"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
    }
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "ABLIB_CLUSTER_NETWORK",
    ):
        if key in os.environ:
            environment[key] = os.environ[key]
    for directory in ("private-home", "cache", "tmp"):
        (output / directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    return environment


def launch(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    validate_isolation()
    settings = configuration(operation, settings)
    if settings.get("topology") and not os.environ.get("ABLIB_CLUSTER_NETWORK"):
        return {
            "status": "unsupported",
            "gaps": [
                "Multinode execution requires an authorized isolated research-network capability"
            ],
            "required_capability": "isolated-research-cluster-network",
        }
    allowed_roles = {"source", "model", "adapter", "calibration"}
    required_roles = {"source", "model"}
    if operation != "factory.export":
        allowed_roles.update(
            {"dataset", "reference", "reward", "reference_adapter", "reward_adapter"}
        )
        required_roles.add("dataset")
    if set(inputs) - allowed_roles or required_roles - set(inputs):
        raise ValueError("Factory received unknown or missing named readonly input roles")
    executable = Path("/opt/ablitbench-factory/bin/python")
    if not executable.is_file():
        return {
            "status": "unsupported",
            "gaps": ["Legacy factory engine profile is not installed"],
            "required_profile": "parity-training-factory",
        }
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    request = output / "factory-request.json"
    atomic_json(
        request,
        {
            "operation": operation,
            "settings": settings,
            "inputs": {name: str(path) for name, path in inputs.items()},
            "output": str(output),
        },
    )
    environment = isolated_environment(output)
    if settings.get("topology"):
        active = [
            path.name
            for path in Path("/sys/class/net").iterdir()
            if path.name != "lo" and int((path / "flags").read_text().strip(), 16) & 1
        ]
        if len(active) != 1:
            raise ValueError(
                "Scoped multinode training requires exactly one validated research interface"
            )
        environment["GLOO_SOCKET_IFNAME"] = active[0]
        environment["NCCL_SOCKET_IFNAME"] = active[0]
    with (output / "factory.log").open("wb") as log:
        process = subprocess.Popen(
            [str(executable), "-I", "-B", str(Path(__file__).resolve()), str(request)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=environment,
            cwd=output,
            start_new_session=True,
        )
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    result = output / "factory-result.json"
    if result.is_file():
        value = strict_json(result)
        if code and value.get("status") == "completed":
            value.update(
                status="unsupported",
                gaps=["Legacy process exited after result with code " + str(code)],
            )
        return dict(value)
    if code == 0 and settings.get("topology", {}).get("node_rank", 0) > 0:
        return {
            "status": "completed",
            "stage": settings["stage"],
            "node_rank": settings["topology"]["node_rank"],
            "participant_only": True,
            "artifacts": ["factory.log"],
            "scope": (
                "Original distributed participant completed; global-rank-zero owns "
                "model export and authoritative workflow result"
            ),
        }
    return {
        "status": "unsupported",
        "gaps": ["Legacy factory workflow exited with code " + str(code)],
        "artifacts": ["factory.log"],
    }


def safe_models(inputs: dict[str, Path]) -> None:
    for role in ("model", "adapter", "reference", "reward", "reference_adapter", "reward_adapter"):
        root = inputs.get(role)
        if root is None:
            continue
        if not root.is_dir():
            raise ValueError("Factory model roles must be local directories")
        if not list(root.glob("*.safetensors")):
            raise ValueError("Factory model role lacks safe tensor weights: " + role)
        for path in root.rglob("*"):
            if path.is_symlink() or path.suffix in {
                ".bin",
                ".pt",
                ".pth",
                ".pkl",
                ".pickle",
                ".ckpt",
            }:
                raise ValueError("Factory never loads unsafe serialized or symlink model files")


def dataset_directory(inputs: dict[str, Path], settings: dict[str, Any], output: Path) -> Path:
    path = inputs["dataset"]
    if path.suffix not in {".json", ".jsonl"}:
        raise ValueError("Factory accepts only operator-provided local JSON/JSONL records")
    directory = output / "dataset"
    directory.mkdir(exist_ok=True)
    destination = directory / ("records" + path.suffix)
    if not destination.exists():
        file_descriptor, temporary = tempfile.mkstemp(prefix=".dataset-", dir=directory)
        try:
            with path.open("rb") as source, os.fdopen(file_descriptor, "wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    descriptor = {
        "file_name": destination.name,
        "formatting": settings["format"],
        "ranking": settings["stage"] in {"rm", "dpo"},
        "columns": settings["columns"],
        "tags": settings["tags"],
    }
    atomic_json(directory / "dataset_info.json", {"operator": descriptor})
    return directory


def factory_arguments(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    arguments = dict(settings["options"])
    if "deepspeed" in settings:
        arguments["deepspeed"] = deepspeed_configuration(settings)
    if arguments.get("export_quantization_bit") is not None:
        if "calibration" not in inputs:
            raise ValueError("GPTQ export requires explicit offline calibration input")
        arguments["export_quantization_dataset"] = str(inputs["calibration"])
    arguments.update(
        model_name_or_path=str(inputs["model"]),
        cache_dir=str(output / "cache"),
        template=settings["template"],
        finetuning_type=settings["finetuning_type"],
        hf_hub_token=None,
        ms_hub_token=None,
    )
    if "adapter" in inputs:
        arguments["adapter_name_or_path"] = str(inputs["adapter"])
    if operation == "factory.export":
        arguments.update(
            export_dir=str(output / "model"),
            export_size=settings["export_shard_gb"],
            export_legacy_format=False,
            export_hub_model_id=None,
        )
        return arguments
    directory = dataset_directory(inputs, settings, output)
    arguments.update(
        stage=settings["stage"],
        dataset="operator",
        dataset_dir=str(directory),
        output_dir=str(output / "model"),
        do_train=True,
        do_eval=bool(arguments.get("val_size", 0)),
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        cutoff_len=settings["cutoff_len"],
        seed=settings["seed"],
        fp16=settings["dtype"] == "float16",
        bf16=settings["dtype"] == "bfloat16",
        use_cpu=settings["device"] == "cpu",
        report_to=[],
        push_to_hub=False,
        save_safetensors=True,
        save_only_model=True,
        overwrite_output_dir=False,
        resume_from_checkpoint=None,
        save_total_limit=None,
        logging_dir=str(output / "logs"),
        dataloader_num_workers=arguments.get("dataloader_num_workers", 0),
        preprocessing_num_workers=arguments.get("preprocessing_num_workers", None),
        save_strategy=arguments.get("save_strategy", "epoch"),
        logging_steps=arguments.get("logging_steps", 1),
        gradient_checkpointing=arguments.get("gradient_checkpointing", False),
    )
    if settings["world_size"] > 1 or settings.get("topology"):
        arguments["ddp_backend"] = "gloo" if settings["device"] == "cpu" else "nccl"
    for role, field in (("reference", "ref_model"), ("reward", "reward_model")):
        if role in inputs:
            arguments[field] = str(inputs[role])
    for role, field, base in (
        ("reference_adapter", "ref_model_adapters", "reference"),
        ("reward_adapter", "reward_model_adapters", "reward"),
    ):
        if role in inputs:
            if base not in inputs:
                raise ValueError("Named reference/reward adapter requires its explicit base role")
            arguments[field] = str(inputs[role])
    if settings["stage"] == "ppo" and "reward" not in inputs:
        raise ValueError("PPO requires an explicit local reward-model role")
    return arguments


def local_loader(factory: Any, allowed: set[Path], output: Path, device: str) -> None:
    def init_kwargs(arguments: Any) -> dict[str, Any]:
        root = Path(arguments.model_name_or_path).resolve(strict=True)
        if root not in allowed:
            raise ValueError("Factory loader attempted a model outside named read-only roles")
        arguments.device_map = (
            None if device == "cpu" else {"": "cuda:" + os.environ.get("LOCAL_RANK", "0")}
        )
        values: dict[str, Any] = {
            "trust_remote_code": False,
            "local_files_only": True,
            "use_safetensors": True,
            "cache_dir": str(output / "cache"),
            "token": None,
        }
        if device == "cuda":
            values["device_map"] = arguments.device_map
        return values

    factory._get_init_kwargs = init_kwargs


def validate_isolation() -> None:
    path = Path(__file__).parent.parent / "worker.py"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    statements: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"isolation", "network_isolation"}
    ]
    if {node.name for node in statements if isinstance(node, ast.FunctionDef)} != {
        "isolation",
        "network_isolation",
    }:
        raise ValueError("Canonical wheel isolation/network guard is unavailable")
    namespace: dict[str, Any] = {"sys": sys, "os": os, "Path": Path}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
    namespace["isolation"]()


@contextmanager
def serialization_aliases() -> Iterator[set[str]]:
    serialization: Any = importlib.import_module("safetensors.torch")
    original = serialization.save_file
    aliases: set[str] = set()

    def save(tensors: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        seen = set()
        normalized = {}
        for name, value in tensors.items():
            candidate = value.contiguous()
            key = (str(candidate.device), candidate.untyped_storage().data_ptr())
            if key in seen:
                candidate = candidate.clone()
                aliases.add(name)
            seen.add((str(candidate.device), candidate.untyped_storage().data_ptr()))
            normalized[name] = candidate
        return original(normalized, *args, **kwargs)

    serialization.save_file = save
    try:
        yield aliases
    finally:
        serialization.save_file = original


def run_original(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    validate_isolation()
    source = verified_source(inputs["source"])
    safe_models(inputs)
    torch = importlib.import_module("torch")
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; factory CPU fallback is disabled")
    if (
        operation != "factory.plan"
        and (output / "model").exists()
        and any((output / "model").iterdir())
    ):
        raise ValueError(
            "Existing engine output cannot be blindly overwritten or implicitly resumed"
        )
    sys.path.insert(0, str(source / "attack/DPA"))
    if settings.get("deepspeed") is not None:
        importlib.import_module("deepspeed")
    if "calibration" in inputs:
        quantization: Any = importlib.import_module("llamafactory.model.model_utils.quantization")

        def calibration(tokenizer: Any, arguments: Any) -> list[dict[str, Any]]:
            records = strict_json(inputs["calibration"])
            if (
                not isinstance(records, list)
                or not records
                or any(
                    not isinstance(row, dict)
                    or set(row) != {"text"}
                    or not isinstance(row["text"], str)
                    for row in records
                )
            ):
                raise ValueError("GPTQ calibration requires offline text-only records")
            return [dict(tokenizer(row["text"], truncation=False)) for row in records]

        quantization._get_quantization_dataset = calibration
    loader = importlib.import_module("llamafactory.model.loader")
    local_loader(
        loader,
        {
            path.resolve()
            for role, path in inputs.items()
            if role
            in {"model", "adapter", "reference", "reward", "reference_adapter", "reward_adapter"}
        },
        output,
        settings["device"],
    )
    arguments = factory_arguments(operation, settings, inputs, output)
    if operation == "factory.export":
        tuner = importlib.import_module("llamafactory.train.tuner")
        tuner.export_model(arguments)
        return {
            "status": "completed",
            "artifacts": ["model"],
            "stage": "export",
            "scope": "Reviewed original safe local model export; no hub publication",
        }
    hparams = importlib.import_module("llamafactory.hparams")
    parsed = hparams.get_train_args(arguments)
    if operation == "factory.plan":
        groups = [dataclasses.asdict(value) for value in parsed]
        atomic_json(output / "factory-effective-arguments.json", groups)
        return {
            "status": "completed",
            "stage": settings["stage"],
            "artifact": "factory-effective-arguments.json",
            "scope": "Original typed argument validation only; no training executed",
        }
    utilities: Any = importlib.import_module("llamafactory.train.trainer_utils")
    corrections = [
        "CPU model placement avoids misleading GPU offload annotation for value heads",
        "Local safe loader disables upstream custom-code loading",
        "Explicit fixed dispatch replaces SFT-only tuner",
    ]
    if not hasattr(utilities, "create_custom_optimizer") and hasattr(
        utilities, "create_custom_optimzer"
    ):
        utilities.create_custom_optimizer = utilities.create_custom_optimzer
        corrections.append("Upstream optimizer spelling alias restored for non-SFT workflows")
    if settings["options"].get("use_galore"):
        galore = importlib.import_module("galore_torch")
        for name in ("GaLoreAdafactor", "GaLoreAdamW", "GaLoreAdamW8bit"):
            setattr(utilities, name, getattr(galore, name))
        corrections.append(
            "Commented upstream GaLore optimizer imports restored from optional pinned backend"
        )
    miscellaneous: Any = importlib.import_module("llamafactory.extras.misc")
    if not hasattr(miscellaneous, "numpify"):

        def numpify(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.dtype == torch.bfloat16:
                    value = value.float()
                return value.numpy()
            return importlib.import_module("numpy").asarray(value)

        miscellaneous.numpify = numpify
        corrections.append("Missing upstream metric tensor-to-array helper restored")
    callbacks: Any = importlib.import_module("llamafactory.train.callbacks")
    if hasattr(miscellaneous, "fix_valuehead_checkpoint"):
        callbacks.fix_valuehead_checkpoint = miscellaneous.fix_valuehead_checkpoint
        corrections.append(
            "Existing upstream value-head helper replaces invalid str.replace keyword call"
        )
    stage = settings["stage"]
    workflow: Any = importlib.import_module("llamafactory.train." + stage + ".workflow")
    original_dataset = workflow.get_dataset
    if stage != "sft":
        splitter = importlib.import_module("llamafactory.data").split_dataset

        def dataset_mapping(*args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original_dataset(*args, **kwargs)
            return value if isinstance(value, dict) else splitter(value, args[1], args[2])

        workflow.get_dataset = dataset_mapping
        corrections.append("Upstream Dataset return is mapped to named trainer populations")
    function = getattr(workflow, "run_" + stage)
    selected = [*parsed] if stage in {"sft", "ppo"} else list(parsed[:4])
    with serialization_aliases() as cloned_aliases:
        function(*selected, callbacks=[])
    if cloned_aliases:
        corrections.append("Shared tensor storage cloned only at safe serialization boundary")
    state_files = []
    model_output = output / "model"
    for path in model_output.rglob("*"):
        if not path.is_file() or path.suffix not in {".bin", ".pt", ".pth"}:
            continue
        if path.name.startswith(("pytorch_model", "adapter_model")):
            raise ValueError("Upstream workflow produced unsafe serialized model weights")
        relative = path.relative_to(model_output)
        destination = output / "framework-state" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)
        state_files.append(relative.as_posix())
    if state_files:
        corrections.append("Unused serialized trainer state kept outside safe model artifacts")
    return {
        "status": "completed",
        "stage": stage,
        "artifacts": ["model", "factory.log"],
        "method_corrections": corrections,
        "coverage": {
            "cutoff_len": settings["cutoff_len"],
            "reason": "Explicit operator context budget; upstream truncation may occur",
        },
        "scope": (
            "Reviewed original stage workflow in a separate legacy engine; "
            "no large-model or accelerator efficacy claim"
        ),
    }


def main() -> None:
    validate_isolation()
    rank_worker = len(sys.argv) == 3 and sys.argv[1] == "--rank-worker"
    if len(sys.argv) != (3 if rank_worker else 2):
        raise ValueError("Factory adapter accepts only its private request file")
    request_path = Path(sys.argv[-1]).resolve(strict=True)
    request = strict_json(request_path)
    if not isinstance(request, dict) or set(request) != {
        "operation",
        "settings",
        "inputs",
        "output",
    }:
        raise ValueError("Invalid factory adapter request schema")
    operation = request["operation"]
    settings = configuration(operation, request["settings"])
    inputs = {role: Path(path).resolve(strict=True) for role, path in request["inputs"].items()}
    output = Path(request["output"]).resolve(strict=True)
    if request_path.parent != output or request_path.name != "factory-request.json":
        raise ValueError("Factory adapter request escaped private output directory")
    topology = settings.get("topology")
    if topology and not os.environ.get("ABLIB_CLUSTER_NETWORK"):
        raise ValueError(
            "Multinode execution requires an authorized isolated research-network capability"
        )
    if topology:
        for resolved in socket.getaddrinfo(
            topology["master_address"], topology["master_port"], type=socket.SOCK_STREAM
        ):
            address = ipaddress.ip_address(resolved[4][0])
            if (
                address.version != 4
                or not address.is_private
                or address.is_loopback
                or address.is_link_local
            ):
                raise ValueError(
                    "Multinode peer resolved outside the controlled private IPv4 network"
                )
    if not rank_worker and (settings.get("world_size", 1) > 1 or topology):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "torch.distributed.run",
            "--rdzv-backend=static",
            "--master-addr=" + (topology["master_address"] if topology else "127.0.0.1"),
            "--master-port=" + str(topology["master_port"] if topology else port),
            "--rdzv-conf=timeout=" + str(settings["options"]["ddp_timeout"]),
            "--nnodes=" + str(topology["nodes"] if topology else 1),
            "--node-rank=" + str(topology["node_rank"] if topology else 0),
            "--nproc-per-node=" + str(settings["world_size"]),
            "--no-python",
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--rank-worker",
            str(request_path),
        ]
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, check=False, env=os.environ)
        if completed.returncode:
            raise RuntimeError("Distributed factory ranks failed")
        return
    rank = int(os.environ.get("RANK", "0"))
    try:
        with redirect_stdout(sys.stderr):
            value = run_original(operation, settings, inputs, output)
    except Exception as error:
        value = {
            "status": "unsupported",
            "gaps": [type(error).__name__ + ": " + str(error)],
            "artifacts": ["factory.log"],
        }
        if rank == 0:
            atomic_json(output / "factory-result.json", value)
        raise
    if rank == 0:
        atomic_json(output / "factory-result.json", value)


def deepspeed_configuration(settings: dict[str, Any]) -> dict[str, Any]:
    geometry = settings["deepspeed"]
    zero: dict[str, Any] = {
        "stage": geometry["stage"],
        "overlap_comm": geometry["overlap_communication"],
        "contiguous_gradients": geometry["contiguous_gradients"],
    }
    for field, parameter in (
        ("offload_optimizer", "offload_optimizer"),
        ("offload_parameters", "offload_param"),
    ):
        if geometry[field] != "none":
            zero[parameter] = {"device": geometry[field], "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": settings["batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "train_batch_size": "auto",
        "gradient_clipping": settings["options"]["max_grad_norm"],
        "fp16": {"enabled": settings["dtype"] == "float16"},
        "bf16": {"enabled": settings["dtype"] == "bfloat16"},
        "zero_optimization": zero,
    }


if __name__ == "__main__":
    main()
