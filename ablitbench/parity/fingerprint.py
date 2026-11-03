from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import shutil
import sqlite3
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ablitbench.fingerprint_engine import file_hash, read_json, safe_path
from ablitbench.formats.json import loads
from ablitbench.models import Generation

PROJECT = "llmmap"
REVISION = "f661d550e3409d1bdf7e00b34bcec15296c5215e"
KINDS = {
    "prompt-configurations": "inspect",
    "merge-datasets": "train",
    "make-dataset": "train",
    "embed": "train",
    "train-open": "train",
    "train-closed": "train",
    "templates": "train",
    "add-template": "train",
    "evaluate-open": "evaluate",
    "evaluate-closed": "evaluate",
    "calibrate": "evaluate",
}


def operations() -> dict[str, dict[str, Any]]:
    required = {
        "prompt-configurations": ["prompts"],
        "merge-datasets": [],
        "make-dataset": [],
        "embed": ["dataset", "embedding"],
        "train-open": ["dataset"],
        "train-closed": ["dataset"],
        "templates": ["dataset", "detector"],
        "add-template": ["dataset", "detector", "templates"],
        "evaluate-open": ["dataset", "detector", "templates"],
        "evaluate-closed": ["dataset", "detector"],
        "calibrate": ["dataset", "detector", "templates"],
    }
    optional = {
        "merge-datasets": ["dataset", "dataset-*"],
        "make-dataset": ["prompts"],
        "evaluate-open": ["calibration"],
    }
    return {
        name: {
            "project": PROJECT,
            "kind": kind,
            "requires_inference": name == "embed",
            "requires_tools": False,
            "modes": ["source"] if name == "embed" else ["api", "source"],
            "inputs": ["source", *required[name]],
            "optional_inputs": optional.get(name, []),
            "input_prefixes": ["dataset-"] if name == "merge-datasets" else [],
            "permissions": [],
            "capabilities": ["generation-broker"]
            if name == "make-dataset"
            else []
            if name in {"prompt-configurations", "merge-datasets"}
            else ["tensor-training"],
            "inputroles": [
                "source",
                "dataset",
                "dataset-*",
                "embedding",
                "detector",
                "templates",
                "calibration",
                "prompts",
            ],
            "revision": REVISION,
        }
        for name, kind in KINDS.items()
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in KINDS:
        raise ValueError("Unknown LLMmap operation")
    allowed = {
        "model",
        "queries",
        "prompt_confs",
        "batch_size",
        "embedding_tokens",
        "max_response_chars",
        "truncate",
        "max_output_tokens",
        "seed",
        "inference_model",
        "training",
        "num_pairs_per_epoch",
        "num_pairs_per_eval",
        "device",
        "top_k",
        "distance_quantile",
        "margin_quantile",
        "samples",
        "attempt_budget",
        "prompt_configuration_file",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown LLMmap settings")
    result = dict(settings)
    for key in (
        "batch_size",
        "embedding_tokens",
        "max_response_chars",
        "max_output_tokens",
        "num_pairs_per_epoch",
        "num_pairs_per_eval",
        "top_k",
    ):
        if key in result and (type(result[key]) is not int or result[key] <= 0):
            raise ValueError("Explicit positive processing or experiment budgets required: " + key)
    if operation == "prompt-configurations":
        counts = result.get("samples")
        if (
            not isinstance(counts, dict)
            or not counts
            or set(counts) - {"train", "validation", "test"}
        ):
            raise ValueError("Explicit prompt configuration population counts required")
        if any(type(count) is not int or count <= 0 for count in counts.values()):
            raise ValueError("Positive prompt configuration counts required")
        budget = result.get("attempt_budget")
        if type(budget) is not int or budget < sum(counts.values()):
            raise ValueError("Explicit sampling attempt budget required")
    if operation == "make-dataset":
        if not isinstance(result.get("model"), str) or not result["model"]:
            raise ValueError("Exact broker model identity required")
        queries = result.get("queries")
        if (
            not isinstance(queries, list)
            or not queries
            or any(not isinstance(q, str) or not q for q in queries)
        ):
            raise ValueError("Nonempty ordered query panel required")
        if "max_output_tokens" not in result:
            raise ValueError("Explicit per-response output budget required")
        confs = result.get("prompt_confs")
        if result.get("prompt_configuration_file") is True:
            if confs is not None:
                raise ValueError("Choose concrete settings or mounted prompt configurations")
            return result
        if not isinstance(confs, dict) or set(confs) - {"train", "validation", "test"}:
            raise ValueError("Explicit train/validation/test prompt configurations required")
        seen: set[str] = set()
        for rows in confs.values():
            if not isinstance(rows, list) or not rows:
                raise ValueError("Nonempty prompt configurations required")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Prompt configuration must be a mapping")
                if set(row) - {
                    "sampling_hparams",
                    "system_prompt",
                    "cot_prompt",
                    "rag_prompt",
                    "raw",
                }:
                    raise ValueError("Unknown original prompt configuration fields")
                sampling = row.get("sampling_hparams", {})
                if not isinstance(sampling, dict) or set(sampling) - {
                    "max_new_tokens",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "do_sample",
                }:
                    raise ValueError("Unsupported prompt sampling settings or credential fields")
                identity = json.dumps(row, sort_keys=True)
                if identity in seen:
                    raise ValueError("Prompt configurations overlap splits or repeat")
                seen.add(identity)
    if operation == "embed" and any(
        key not in result for key in ("embedding_tokens", "max_response_chars", "truncate")
    ):
        raise ValueError("Explicit embedding preprocessing and truncation policy required")
    if operation.startswith("train-"):
        if not isinstance(result.get("inference_model"), dict) or not isinstance(
            result.get("training"), dict
        ):
            raise ValueError("Original architecture and training configuration required")
        for key in ("max_epochs", "early_stop_patience", "log_every_n_steps"):
            value = result["training"].get(key)
            if type(value) is not int or value < (0 if key == "early_stop_patience" else 1):
                raise ValueError("Explicit training schedule required: " + key)
        if operation == "train-open" and any(
            key not in result for key in ("num_pairs_per_epoch", "num_pairs_per_eval")
        ):
            raise ValueError("Explicit Siamese sampling budgets required")
    if result.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("Unsupported training device")
    return result


def journal(output: Path, stage: str, **data: Any) -> None:
    with (output / "llmmap-progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"stage": stage, **data}, ensure_ascii=False, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def dataset_manifest(root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = read_json(safe_path(root, "dataset.json"))
    if (
        set(manifest) != {"schema_version", "classes", "queries", "partitions"}
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported prepared fingerprint dataset")
    seen: set[str] = set()
    for split, partition in manifest["partitions"].items():
        if split not in {"train", "validation", "test", "calibration"}:
            raise ValueError("Unknown fingerprint dataset partition")
        path = safe_path(root, partition["file"])
        if file_hash(path) != partition["sha256"]:
            raise ValueError("Fingerprint dataset content identity changed")
        ids = partition["ids"]
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or seen & set(ids):
            raise ValueError("Fingerprint sample identities overlap partitions")
        seen.update(ids)
    return manifest


def prepared_dataset(
    root: Path,
    manifest: dict[str, Any],
    split: str,
    source: Path,
    siamese: bool = False,
    count: int | None = None,
) -> Any:
    torch = importlib.import_module("torch")
    safe_open = importlib.import_module("safetensors").safe_open
    partition = manifest["partitions"][split]
    path = safe_path(root, partition["file"])
    with safe_open(path, framework="pt", device="cpu") as stored:
        shape = stored.get_slice("features").get_shape()
        labels = stored.get_tensor("labels").tolist()
    if (
        len(shape) != 3
        or shape[0] != len(labels)
        or len(labels) != len(partition["ids"])
        or shape[1] != len(manifest["queries"])
        or shape[2] % 2
    ):
        raise ValueError("Prepared trace feature or label shape mismatch")
    classes = manifest["classes"]
    if len(classes) != len(set(classes)) or any(
        type(label) is not int or not 0 <= label < len(classes) for label in labels
    ):
        raise ValueError("Invalid or unknown fingerprint labels")
    original = importlib.import_module("LLMmap.dataset")
    parent = original.DatasetFactorySiamese if siamese else original.DatasetFactory

    def initialize(self: Any) -> None:
        rows = [{"llm": classes[label], "traces": index} for index, label in enumerate(labels)]
        cache = SimpleNamespace(llms_map={name: index for index, name in enumerate(classes)})
        if siamese:
            if len(classes) < 2 or any(labels.count(index) < 2 for index in range(len(classes))):
                raise ValueError("Siamese sampling requires two traces for every class")
            parent.__init__(self, rows, cache, count)
        else:
            parent.__init__(self, rows, cache)
        self.feature_shape = shape[1:]

    def pack(self: Any, index: int) -> Any:
        with safe_open(path, framework="pt", device="cpu") as stored:
            features = stored.get_slice("features")[index]
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite fingerprint embeddings")
        return features

    prepared = type(
        "AblitBenchPreparedDataset", (parent,), {"__init__": initialize, "pack_traces": pack}
    )
    return prepared()


def reference_asset(inputs: dict[str, Path], role: str, filename: str) -> Path:
    candidate = inputs.get(role)
    if candidate is None:
        return safe_path(inputs["detector"], filename)
    return safe_path(candidate, filename) if candidate.is_dir() else candidate


def reference_compatibility(
    operation: str, config: dict[str, Any], manifest: dict[str, Any]
) -> None:
    if manifest["queries"] != config["queries"]:
        raise ValueError("Fingerprint query panel differs from detector training panel")
    if operation == "evaluate-closed" and manifest["classes"] != config["classes"]:
        raise ValueError("Closed fingerprint class identities/order differ from trained head")


def merge_datasets(inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    sources = [path for role, path in sorted(inputs.items()) if role.startswith("dataset")]
    if not sources:
        raise ValueError("Named dataset inputs required")
    queries = None
    count = 0
    with sqlite3.connect(output / "dataset-index.sqlite3") as index:
        index.execute("CREATE TABLE samples (id TEXT PRIMARY KEY)")
        index.execute("CREATE TABLE configurations (id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        with (output / "traces.jsonl").open("w", encoding="utf-8") as target:
            for source in sources:
                paths = [source] if source.is_file() else sorted(source.glob("*.jsonl"))
                if not paths:
                    raise ValueError("Dataset input contains no trace JSONL")
                for path in paths:
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            row = loads(line)
                            panel = [trace[0] for trace in row["traces"]]
                            if not panel or queries is not None and queries != panel:
                                raise ValueError("Reference datasets have different query panels")
                            queries = panel
                            split = row["dataset"]
                            if split not in {"train", "validation", "calibration", "test"}:
                                raise ValueError("Unknown trace partition")
                            identity = (
                                row.get("id")
                                or hashlib.sha256(
                                    json.dumps(row, sort_keys=True).encode()
                                ).hexdigest()
                            )
                            try:
                                index.execute("INSERT INTO samples VALUES (?)", (identity,))
                            except sqlite3.IntegrityError as error:
                                raise ValueError(
                                    "Duplicate trace identity across merged data"
                                ) from error
                            if "prompt_conf" in row:
                                conf = hashlib.sha256(
                                    json.dumps(row["prompt_conf"], sort_keys=True).encode()
                                ).hexdigest()
                                previous = index.execute(
                                    "SELECT split FROM configurations WHERE id=?", (conf,)
                                ).fetchone()
                                if previous is not None and previous[0] != split:
                                    raise ValueError(
                                        "Prompt configuration leaks across data splits"
                                    )
                                index.execute(
                                    "INSERT OR IGNORE INTO configurations VALUES (?, ?)",
                                    (conf, split),
                                )
                            row["id"] = identity
                            target.write(
                                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            count += 1
            target.flush()
            os.fsync(target.fileno())
    if not count:
        raise ValueError("No merged trace observations")
    return {"traces": count, "datasets": len(sources), "queries": queries}


def prompt_configurations(
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    original = importlib.import_module("LLMmap.prompt_configuration")
    factory = original.PromptConfFactory(inputs["prompts"])
    random.seed(settings.get("seed", 0))
    unique: set[str] = set()
    rows: dict[str, list[Any]] = {}
    attempts = 0
    for split, count in settings["samples"].items():
        rows[split] = []
        while len(rows[split]) < count:
            if attempts == settings["attempt_budget"]:
                (output / "partial-prompt-configurations.json").write_text(json.dumps(rows))
                raise ValueError(
                    "Explicit sampling budget exhausted before distinct split populations"
                )
            attempts += 1
            value = factory.sample_one(pool=split).to_dict()
            identity = json.dumps(value, sort_keys=True)
            if identity in unique:
                continue
            unique.add(identity)
            rows[split].append(value)
    (output / "prompt-configurations.json").write_text(json.dumps(rows, ensure_ascii=False))
    return {
        "sampled": {key: len(value) for key, value in rows.items()},
        "attempts": attempts,
        "compatibility_fix": (
            "Native sample_one with value identity avoids "
            "original object-identity duplicate sampling"
        ),
    }


def make_dataset(
    settings: dict[str, Any], source: Path, output: Path, broker: Any
) -> dict[str, Any]:
    if broker is None:
        raise ValueError("Dataset capture requires the host broker")
    prompt_type = importlib.import_module("LLMmap.prompt_configuration").PromptConf
    count = 0
    with (output / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for split, confs in settings["prompt_confs"].items():
            for configuration in confs:
                prompt = prompt_type.from_dict(configuration)
                traces = []
                for query in settings["queries"]:
                    (body, system), sampling = prompt(query, None, apply_template=False)
                    messages = [{"role": "user", "content": body}]
                    if system:
                        messages.insert(0, {"role": "system", "content": system})
                    if set(sampling) - {
                        "max_new_tokens",
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "top_k",
                        "do_sample",
                    }:
                        raise ValueError("Unsupported prompt sampling parameters")
                    sampling = dict(sampling)
                    budgets = [
                        sampling.pop(key)
                        for key in ("max_new_tokens", "max_tokens")
                        if key in sampling
                    ]
                    if budgets and (
                        len(set(budgets)) != 1 or type(budgets[0]) is not int or budgets[0] <= 0
                    ):
                        raise ValueError("Prompt-specific output budgets conflict or are invalid")
                    budget = budgets[0] if budgets else settings["max_output_tokens"]
                    journal(output, "query_started", split=split, panel=count, query=query)
                    result = broker(
                        {
                            "kind": "generation",
                            "channel": "target",
                            "expected_model": settings["model"],
                            "messages": messages,
                            "max_output_tokens": budget,
                            "seed": settings.get("seed"),
                            "generation_settings": sampling,
                        }
                    )
                    if (
                        not isinstance(result.get("text"), str)
                        or not (
                            Generation.model_validate(result).complete
                            or result.get("finish") in {"length", "max_tokens"}
                        )
                        or result.get("tools")
                        or result.get("media")
                    ):
                        raise ValueError("Dataset capture response is incomplete or not plain text")
                    traces.append([query, result["text"]])
                    journal(
                        output,
                        "query_completed",
                        generation=result,
                        coverage="Fixed-budget fingerprint trace; not a full-completion guarantee",
                        truncated=result.get("finish") in {"length", "max_tokens"},
                    )
                row = {
                    "dataset": split,
                    "llm": settings["model"],
                    "traces": traces,
                    "prompt_conf": prompt.to_dict(),
                }
                row["id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                count += 1
    return {"panels": count, "dataset": "traces.jsonl"}


def embed(settings: dict[str, Any], inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    original = importlib.import_module("LLMmap.embedding_model").Embedding
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        inputs["embedding"], local_files_only=True, trust_remote_code=False
    )
    model = (
        transformers.AutoModel.from_pretrained(
            inputs["embedding"],
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .eval()
        .to(settings.get("device", "cpu"))
    )
    files = sorted(inputs["dataset"].glob("*.jsonl"))
    if not files:
        raise ValueError("Dataset input contains no trace JSONL")
    queries = None
    identities: set[str] = set()
    names: set[str] = set()
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = loads(line)
                current = [trace[0] for trace in row["traces"]]
                if queries is None:
                    queries = current
                if current != queries or not current:
                    raise ValueError("Trace panels use different or empty query order")
                names.add(row["llm"])
    classes = sorted(names)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "classes": classes,
        "queries": queries,
        "partitions": {},
    }
    metadata: dict[str, Any] = {}
    try:
        for path in files:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = loads(line)
                    split = row["dataset"]
                    if split not in {"train", "validation", "calibration", "test"}:
                        raise ValueError("Unknown trace partition")
                    identity = (
                        row.get("id")
                        or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    )
                    if identity in identities:
                        raise ValueError("Trace identities overlap partitions or repeat")
                    identities.add(identity)
                    if split not in metadata:
                        metadata[split] = {
                            "features": (output / (split + ".features.tmp")).open("wb"),
                            "labels": (output / (split + ".labels.tmp")).open("wb"),
                            "ids": [],
                            "shape": None,
                        }
                    panel = []
                    for query, response in row["traces"]:
                        text = response[: settings["max_response_chars"]]
                        if len(response) > len(text) and not settings["truncate"]:
                            raise ValueError("Response exceeds explicit preprocessing capacity")
                        tokens = tokenizer(
                            [query, text], padding=True, truncation=False, return_tensors="pt"
                        )
                        original_tokens = tokens.input_ids.shape[1]
                        if original_tokens > settings["embedding_tokens"]:
                            if not settings["truncate"]:
                                raise ValueError("Embedding input exceeds explicit token capacity")
                            tokens = tokenizer(
                                [query, text],
                                padding=True,
                                truncation=True,
                                max_length=settings["embedding_tokens"],
                                return_tensors="pt",
                            )
                        tokens = tokens.to(model.device)
                        with torch.no_grad():
                            features = original.get_embs(
                                None, model(**tokens), tokens.attention_mask
                            )
                        panel.append(torch.cat([features[0], features[1]]).cpu().float())
                        journal(
                            output,
                            "embedding_completed",
                            split=split,
                            original_response_chars=len(response),
                            used_response_chars=len(text),
                            original_tokens=original_tokens,
                            used_tokens=tokens.input_ids.shape[1],
                        )
                    features = torch.stack(panel).contiguous()
                    if not torch.isfinite(features).all():
                        raise ValueError("Nonfinite embedding features")
                    entry = metadata[split]
                    shape = list(features.shape)
                    if entry["shape"] is not None and entry["shape"] != shape:
                        raise ValueError("Inconsistent feature panel shape")
                    entry["shape"] = shape
                    entry["features"].write(features.numpy().tobytes())
                    entry["labels"].write(struct.pack("<q", classes.index(row["llm"])))
                    entry["ids"].append(identity)
        for split, entry in metadata.items():
            entry["features"].close()
            entry["labels"].close()
            feature_path = output / (split + ".features.tmp")
            label_path = output / (split + ".labels.tmp")
            feature_bytes = feature_path.stat().st_size
            label_bytes = label_path.stat().st_size
            header = json.dumps(
                {
                    "features": {
                        "dtype": "F32",
                        "shape": [len(entry["ids"]), *entry["shape"]],
                        "data_offsets": [0, feature_bytes],
                    },
                    "labels": {
                        "dtype": "I64",
                        "shape": [len(entry["ids"])],
                        "data_offsets": [feature_bytes, feature_bytes + label_bytes],
                    },
                },
                separators=(",", ":"),
            ).encode()
            header += b" " * (-len(header) % 8)
            destination = output / (split + ".safetensors")
            with destination.open("wb") as target:
                target.write(struct.pack("<Q", len(header)))
                target.write(header)
                for temporary in (feature_path, label_path):
                    with temporary.open("rb") as data:
                        shutil.copyfileobj(data, target, length=1024 * 1024)
                    temporary.unlink()
            manifest["partitions"][split] = {
                "file": destination.name,
                "sha256": file_hash(destination),
                "ids": entry["ids"],
            }
    finally:
        for entry in metadata.values():
            entry["features"].close()
            entry["labels"].close()
    (output / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset_manifest(output)
    return manifest


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
    existing = sys.modules.get("LLMmap")
    if existing is not None and not Path(str(existing.__file__)).resolve().is_relative_to(source):
        raise ValueError("Original LLMmap namespace collision")
    sys.path.insert(0, str(source))
    journal(output, "started", operation=operation)
    try:
        if operation == "merge-datasets":
            detail = merge_datasets(inputs, output)
        elif operation == "prompt-configurations":
            detail = prompt_configurations(settings, inputs, output)
        elif operation == "make-dataset":
            if settings.get("prompt_configuration_file"):
                document = inputs["prompts"]
                if document.is_dir():
                    document = document / "prompt-configurations.json"
                concrete = read_json(document)
                settings.pop("prompt_configuration_file")
                settings = configuration(operation, settings | {"prompt_confs": concrete})
            detail = make_dataset(settings, source, output, broker)
        elif operation == "embed":
            detail = embed(settings, inputs, output)
        else:
            torch = importlib.import_module("torch")
            if settings.get("device", "cpu") == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable; no CPU fallback")
            seed = settings.get("seed", 0)
            random.seed(seed)
            torch.manual_seed(seed)
            manifest = dataset_manifest(inputs["dataset"])
            if operation.startswith("train-"):
                siamese = operation == "train-open"
                train = prepared_dataset(
                    inputs["dataset"],
                    manifest,
                    "train",
                    source,
                    siamese,
                    settings.get("num_pairs_per_epoch"),
                )
                validation = prepared_dataset(
                    inputs["dataset"],
                    manifest,
                    "validation",
                    source,
                    siamese,
                    settings.get("num_pairs_per_eval"),
                )
                batch = settings.get("batch_size", 32)
                loader = torch.utils.data.DataLoader
                hp = settings["inference_model"] | {
                    "emb_size": train.feature_shape[-1] // 2,
                    "num_queries": train.feature_shape[0],
                    "num_classes": len(manifest["classes"]),
                }
                trainer_module = importlib.import_module("LLMmap.trainer")
                trainer_type = (
                    trainer_module.LLMmapTrainerSiamese
                    if siamese
                    else trainer_module.LLMmapTrainerClosed
                )
                original_restore = trainer_type.__dict__.get("load_from_checkpoint")
                original_trainer = trainer_module.pl.Trainer

                def restore(cls: Any, checkpoint_path: str, model: Any, hp: dict[str, Any]) -> Any:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    restored = cls(model, hp)
                    restored.load_state_dict(checkpoint["state_dict"], strict=True)
                    return restored

                def placement(*args: Any, **kwargs: Any) -> Any:
                    kwargs["accelerator"] = "gpu" if settings.get("device") == "cuda" else "cpu"
                    kwargs["devices"] = 1
                    return original_trainer(*args, **kwargs)

                trainer_type.load_from_checkpoint = classmethod(restore)
                trainer_module.pl.Trainer = placement
                try:
                    trainer, model = trainer_module.train_model(
                        str(output / "checkpoints"),
                        siamese,
                        loader(train, batch_size=batch, shuffle=True),
                        loader(validation, batch_size=batch),
                        {"inference_model": hp, "training": settings["training"]},
                    )
                finally:
                    trainer_module.pl.Trainer = original_trainer
                    if original_restore is None:
                        delattr(trainer_type, "load_from_checkpoint")
                    else:
                        trainer_type.load_from_checkpoint = original_restore
                save_file = importlib.import_module("safetensors.torch").save_file
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in model.state_dict().items()
                    },
                    str(output / "detector.safetensors"),
                )
                config = {
                    "is_open": siamese,
                    "inference_model": hp,
                    "classes": manifest["classes"],
                    "queries": manifest["queries"],
                    "training": settings["training"],
                }
                (output / "detector.json").write_text(json.dumps(config), encoding="utf-8")
                detail = {
                    "training": config,
                    "checkpoint_boundary_fix": (
                        "Original helper passes hp to a constructor expecting hparams; "
                        "restore original class from safe state_dict using explicit hparams"
                    ),
                    "placement": "Explicit operator CPU/CUDA instead of upstream auto",
                    "pair_labels": "Original DatasetFactorySiamese same=1,different=0 retained",
                    "validation_used_for_selection": True,
                    "test_not_loaded_during_training": True,
                }
            else:
                config = read_json(safe_path(inputs["detector"], "detector.json"))
                reference_compatibility(operation, config, manifest)
                model = importlib.import_module(
                    "LLMmap.inference_model_archs"
                ).InferenceModelLLMmap(config["inference_model"], is_for_siamese=config["is_open"])
                weights = importlib.import_module("safetensors.torch").load_file(
                    str(safe_path(inputs["detector"], "detector.safetensors"))
                )
                model.load_state_dict(weights, strict=True)
                model.eval().to(settings.get("device", "cpu"))
                original = importlib.import_module("LLMmap.templates")
                split = (
                    "train"
                    if operation in {"templates", "add-template"}
                    else "calibration"
                    if operation == "calibrate"
                    else "test"
                )
                dataset = prepared_dataset(inputs["dataset"], manifest, split, source)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=settings.get("batch_size", 32)
                )
                labels, features = original.infer_features(
                    model, loader, device=settings.get("device", "cpu")
                )
                if operation in {"templates", "add-template"}:
                    if not config["is_open"]:
                        raise ValueError(
                            "Template generation requires an open-set feature extractor"
                        )
                    means = original.build_templates(labels, features)
                    numpy = importlib.import_module("numpy")
                    if not numpy.isfinite(means).all():
                        raise ValueError("Every declared class must have training examples")
                    templates = {
                        name: means[index].tolist()
                        for index, name in enumerate(manifest["classes"])
                    }
                    if operation == "add-template":
                        previous = read_json(reference_asset(inputs, "templates", "templates.json"))
                        if set(previous) & set(templates):
                            raise ValueError("New template names overlap existing references")
                        templates = previous | templates
                    (output / "templates.json").write_text(
                        json.dumps(templates, allow_nan=False), encoding="utf-8"
                    )
                    detail = {"templates": list(templates), "source_split": "train"}
                elif operation == "evaluate-closed":
                    if config["is_open"]:
                        raise ValueError("Closed evaluation requires the classification head")
                    predictions = features.argmax(axis=1)
                    detail = {
                        "correct": int((predictions == labels).sum()),
                        "total": len(labels),
                        "split": "test",
                    }
                else:
                    if not config["is_open"]:
                        raise ValueError(
                            "Open evaluation and distance calibration require feature embeddings"
                        )
                    template_path = reference_asset(inputs, "templates", "templates.json")
                    templates = read_json(template_path)
                    numpy = importlib.import_module("numpy")
                    names = list(templates)
                    matrix = numpy.asarray(list(templates.values()), dtype=numpy.float64)
                    distances = numpy.linalg.norm(
                        features.astype(numpy.float64)[:, None, :] - matrix[None, :, :], axis=-1
                    )
                    ranking = distances.argsort(axis=1)
                    if operation == "calibrate":
                        if len(names) < 2:
                            raise ValueError(
                                "Open-set margin calibration needs multiple references"
                            )
                        quantile = settings.get("distance_quantile")
                        margin_quantile = settings.get("margin_quantile")
                        if (
                            not isinstance(quantile, (float, int))
                            or not 0 < quantile < 1
                            or not isinstance(margin_quantile, (float, int))
                            or not 0 < margin_quantile < 1
                        ):
                            raise ValueError("Explicit calibration quantiles required")
                        nearest = numpy.take_along_axis(distances, ranking[:, :2], axis=1)
                        truths = [manifest["classes"][int(index)] for index in labels]
                        if any(truth not in names for truth in truths):
                            raise ValueError(
                                "Reference calibration contains an undeclared unknown class"
                            )
                        detail = {
                            "detector_sha256": file_hash(
                                safe_path(inputs["detector"], "detector.safetensors")
                            ),
                            "templates_sha256": file_hash(template_path),
                            "queries": config["queries"],
                            "known_reference_matches": sum(
                                names[int(row[0])] == truth
                                for row, truth in zip(ranking, truths, strict=True)
                            ),
                            "samples": len(truths),
                            "max_distance": float(numpy.quantile(nearest[:, 0], quantile)),
                            "min_margin": float(
                                max(
                                    0.0,
                                    numpy.quantile(nearest[:, 1] - nearest[:, 0], margin_quantile),
                                )
                            ),
                            "split": "calibration",
                            "scope": (
                                "Known-reference calibration only; "
                                "unknown-model false acceptance unmeasured"
                            ),
                        }
                        (output / "calibration.json").write_text(
                            json.dumps(detail, allow_nan=False)
                        )
                    else:
                        k = settings.get("top_k", 1)
                        if k > len(names):
                            raise ValueError("top_k exceeds reference population")
                        truths = [manifest["classes"][int(index)] for index in labels]
                        calibration_path = inputs.get("calibration")
                        if calibration_path is not None and calibration_path.is_dir():
                            calibration_path = safe_path(calibration_path, "calibration.json")
                        if calibration_path is None:
                            bundled = inputs["detector"] / "calibration.json"
                            if bundled.is_file():
                                calibration_path = bundled
                        calibration = read_json(calibration_path) if calibration_path else None
                        if calibration is not None and (
                            calibration.get("detector_sha256")
                            != file_hash(safe_path(inputs["detector"], "detector.safetensors"))
                            or calibration.get("templates_sha256") != file_hash(template_path)
                            or calibration.get("queries") != config["queries"]
                        ):
                            raise ValueError(
                                "Calibration identity differs from detector/templates/query panel"
                            )
                        recognition_file = (output / "recognitions.jsonl").open(
                            "w", encoding="utf-8"
                        )
                        recognized = 0
                        unknown_false_acceptances = 0
                        known = [truth in names for truth in truths]
                        for row_index, row in enumerate(ranking):
                            if calibration is None or len(names) < 2:
                                recognition_file.write(
                                    json.dumps(
                                        {
                                            "status": "inconclusive",
                                            "reason": "No bound reference calibration",
                                            "reference": None,
                                            "index": row_index,
                                            "ranking": [names[int(index)] for index in row[:k]],
                                        }
                                    )
                                    + "\n"
                                )
                                continue
                            distance = float(distances[row_index, row[0]])
                            margin = float(distances[row_index, row[1]] - distance)
                            accepted = (
                                distance <= calibration["max_distance"]
                                and margin > calibration["min_margin"]
                            )
                            recognition_file.write(
                                json.dumps(
                                    {
                                        "status": "reference-match"
                                        if accepted
                                        else "unknown-or-unsupported",
                                        "reference": names[int(row[0])] if accepted else None,
                                        "distance": distance,
                                        "margin": margin,
                                        "index": row_index,
                                        "ranking": [names[int(index)] for index in row[:k]],
                                    }
                                )
                                + "\n"
                            )
                            recognized += int(accepted)
                            unknown_false_acceptances += int(accepted and not known[row_index])
                        recognition_file.flush()
                        os.fsync(recognition_file.fileno())
                        recognition_file.close()
                        detail = {
                            "top_k": k,
                            "correct": sum(
                                truth in [names[index] for index in row[:k]]
                                for truth, row in zip(truths, ranking, strict=True)
                            ),
                            "total": len(truths),
                            "split": "test",
                            "recognitions": "recognitions.jsonl",
                            "recognized": recognized,
                            "known_samples": sum(known),
                            "unknown_samples": len(known) - sum(known),
                            "unknown_false_acceptances": unknown_false_acceptances
                            if not all(known) and calibration is not None
                            else None,
                            "scope": (
                                "Nearest-reference ranking; rejection uses separate "
                                "known-reference calibration, not weight identity proof"
                            ),
                        }
        result = {
            "project": PROJECT,
            "operation": operation,
            "status": "completed",
            "upstream_revision": REVISION,
            "result": detail,
            "scope": (
                "Original architecture, datasets, trainer and template functions; "
                "safe local assets and separate validation/test"
            ),
        }
        (output / "llmmap-result.json").write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        journal(output, "completed", operation=operation)
        return result
    finally:
        sys.path.remove(str(source))
