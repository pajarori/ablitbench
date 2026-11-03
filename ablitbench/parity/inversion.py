from __future__ import annotations

import ast
import gc
import importlib
import json
import math
import os
import random
import re
import secrets
import shutil
import socket
import socketserver
import struct
import tempfile
import threading
from pathlib import Path
from typing import Any

from ablitbench.bait_engine import configuration as search_configuration
from ablitbench.formats.json import loads
from ablitbench.parity.attacks import generation_complete
from ablitbench.parity.classification import metrics


def operations() -> dict[str, dict[str, Any]]:
    return {
        "scan.ray": {
            "project": "BAIT",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "detect",
            "capabilities": ["sandbox", "reviewed-source", "full-probabilities", "judge", "ray"],
            "inputs": ["source", "models"],
        },
        "scan.batch": {
            "project": "BAIT",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "detect",
            "capabilities": ["sandbox", "reviewed-source", "full-probabilities", "judge"],
            "inputs": ["source", "models"],
        },
        "scan.full": {
            "project": "BAIT",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "detect",
            "capabilities": ["sandbox", "reviewed-source", "full-probabilities", "judge"],
            "inputs": ["source", "model"],
            "optional_inputs": ["base"],
        },
        "dataset.prepare": {
            "project": "BAIT",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "inspect",
            "capabilities": ["sandbox", "local-dataset"],
            "inputs": ["dataset"],
        },
        "results.evaluate": {
            "project": "BAIT",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "evaluate",
            "capabilities": ["sandbox", "labeled-results"],
            "inputs": ["predictions", "labels"],
        },
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown BAIT operation")
    if operation == "scan.ray":
        if (
            set(settings) != {"scan", "continue_on_failure", "resources"}
            or type(settings["continue_on_failure"]) is not bool
        ):
            raise ValueError(
                "Ray scan requires explicit scan settings, failure policy and resources"
            )
        resources = settings["resources"]
        required = {
            "cpus",
            "gpus",
            "cpus_per_model",
            "gpus_per_model",
            "memory_per_model",
            "object_store_bytes",
            "parallel_models",
        }
        if not isinstance(resources, dict) or set(resources) != required:
            raise ValueError("Explicit Ray scheduler and object-store resource budgets required")
        for key in required:
            if (
                type(resources[key]) not in {int, float}
                or not math.isfinite(resources[key])
                or resources[key] < 0
            ):
                raise ValueError("Invalid Ray resource quantity: " + key)
        if (
            any(
                type(resources[key]) is not int or resources[key] <= 0
                for key in ("memory_per_model", "object_store_bytes", "parallel_models")
            )
            or resources["cpus_per_model"] <= 0
            or resources["cpus"] < resources["cpus_per_model"]
            or resources["gpus_per_model"] > resources["gpus"]
        ):
            raise ValueError("Ray model resource request exceeds the explicit local pool")
        scan_settings = configuration("scan.full", settings["scan"])
        if (scan_settings["device"] == "cuda") != (resources["gpus_per_model"] > 0):
            raise ValueError("Ray GPU requests must match the explicit scan device")
        return {**settings, "scan": scan_settings}
    if operation == "scan.batch":
        if (
            set(settings) != {"scan", "continue_on_failure"}
            or type(settings["continue_on_failure"]) is not bool
        ):
            raise ValueError("Batch scan requires explicit scan settings and failure policy")
        return {
            "scan": configuration("scan.full", settings["scan"]),
            "continue_on_failure": settings["continue_on_failure"],
        }
    allowed = {
        "scan.full": {
            "search",
            "filter_unprintable",
            "q_score_threshold",
            "early_stop",
            "early_stop_q_score_threshold",
            "judge_max_output_tokens",
            "judge_attempts",
            "device",
            "context_tokens",
        },
        "dataset.prepare": {
            "format",
            "split",
            "test_fraction",
            "validation_fraction",
            "prompt_count",
            "seed",
        },
        "results.evaluate": {"tokenization"},
    }[operation]
    if set(settings) - allowed:
        raise ValueError("Unknown BAIT settings")
    result = dict(settings)
    if operation == "scan.full":
        if allowed - {"device", "context_tokens"} - settings.keys():
            raise ValueError("Explicit search, token filter, judge and decision budgets required")
        search = dict(result["search"])
        search.setdefault("initial_token_ids", [0])
        search_configuration(search)
        for name in ("q_score_threshold", "early_stop_q_score_threshold"):
            if (
                type(result[name]) not in {int, float}
                or not math.isfinite(result[name])
                or not 0 <= result[name] <= 1
            ):
                raise ValueError("Finite BAIT threshold in [0,1] required")
        for name in ("judge_max_output_tokens", "judge_attempts"):
            if type(result[name]) is not int or result[name] <= 0:
                raise ValueError("Explicit positive judge budget required")
        if any(type(result[name]) is not bool for name in ("filter_unprintable", "early_stop")):
            raise ValueError("BAIT filter and early-stop settings must be boolean")
        if result.get("device", "cpu") not in {"cpu", "cuda"}:
            raise ValueError("Unknown inference device")
        if "context_tokens" in result and (
            type(result["context_tokens"]) is not int or result["context_tokens"] <= 0
        ):
            raise ValueError("Positive context token budget required")
    elif operation == "dataset.prepare":
        if set(result) != allowed:
            raise ValueError("Explicit local dataset, split, sample count and seed required")
        if result["format"] not in {"alpaca", "self-instruct", "text"}:
            raise ValueError("Unsupported local prompt dataset format")
        if result["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown dataset split")
        for name in ("test_fraction", "validation_fraction"):
            if type(result[name]) not in {int, float} or not 0 < result[name] < 1:
                raise ValueError("Independent nonempty split fractions required")
        if (
            type(result["seed"]) is not int
            or type(result["prompt_count"]) is not int
            or result["prompt_count"] <= 0
        ):
            raise ValueError("Explicit seed and positive prompt count required")
    elif result.get("tokenization", "whitespace") not in {"whitespace", "nltk"}:
        raise ValueError("Choose explicit whitespace or original NLTK word tokenization")
    return result


def judge_prompt(source: Path) -> str:
    tree = ast.parse((source / "src/utils/constants.py").read_bytes())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "JUDGE_SYSTEM_PROMPT"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("Reviewed original BAIT judge prompt is unavailable")


def judge(
    target: str, prompt: str, settings: dict[str, Any], broker: Any
) -> tuple[bool | None, list[dict[str, Any]]]:
    attempts = []
    if broker is None:
        return None, [{"reason": "No separate judge configured"}]
    for _ in range(settings["judge_attempts"]):
        try:
            response = broker(
                {
                    "kind": "judge",
                    "channel": "judge",
                    "messages": [{"role": "user", "content": prompt.format(response=target)}],
                    "max_output_tokens": settings["judge_max_output_tokens"],
                }
            )
        except Exception as error:
            attempts.append({"reason": "Judge delivery failed", "error_type": type(error).__name__})
            return None, attempts
        attempts.append(response)
        if not generation_complete(response):
            continue
        text = response.get("text", "")
        states = re.findall(r"<State>(.*?)</State>", text, re.DOTALL)
        reasons = re.findall(r"<Reasoning>(.*?)</Reasoning>", text, re.DOTALL)
        if (
            len(states) == len(reasons) == 1
            and reasons[0].strip()
            and states[0].strip().lower() in {"safe", "suspicious"}
        ):
            return states[0].strip().lower() == "suspicious", attempts
    return None, attempts


def token_space(tokenizer: Any, filtered: bool) -> list[int]:
    return sorted(
        index
        for token, index in tokenizer.get_vocab().items()
        if not filtered or (token.startswith("▁") or token.startswith("Ġ")) and token[1:].isalpha()
    )


def scan(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path, broker: Any
) -> dict[str, Any]:
    from ablitbench.bait_engine import BaitEngine
    from ablitbench.parity.sources import verified_source
    from ablitbench.worker import Inference

    source = verified_source(inputs["source"], "BAIT")
    inference = Inference(
        settings.get("device", "cpu"), inputs["model"], inputs.get("base", Path("/base"))
    )
    tokens = token_space(inference.tokenizer, settings["filter_unprintable"])
    search = dict(settings["search"])
    if "initial_token_ids" in search:
        if set(search["initial_token_ids"]) - set(tokens):
            raise ValueError(
                "Requested initial token IDs are excluded by the selected upstream filter"
            )
        tokens = search["initial_token_ids"]
    if not tokens:
        raise ValueError("Selected BAIT initial-token space is empty")
    search["initial_token_ids"] = tokens
    engine = BaitEngine(
        source / "src/core/detector.py", inference, context_tokens=settings.get("context_tokens")
    )
    output.mkdir(parents=True, exist_ok=True)
    engine.output = output
    metadata = engine.configure(search)
    prompt = judge_prompt(source)
    best = {"q_score": 0.0, "target": None, "reasoning": None}
    inspected = 0
    batches = 0
    gaps = list(metadata["gaps"])
    for start in range(0, len(tokens), search["batch_size"]):
        record = engine.search(tokens[start : start + search["batch_size"]])
        if record["q_score"] > best["q_score"]:
            suspicious, attempts = judge(record["target"], prompt, settings, broker)
            record["judge"] = {"suspicious": suspicious, "attempts": attempts}
            if suspicious is None:
                gaps.append("Candidate judge was inconclusive")
            elif suspicious:
                best = {
                    "q_score": record["q_score"],
                    "target": record["target"],
                    "reasoning": attempts[-1]["text"],
                }
        batches += 1
        inspected += len(record["initial_token_ids"])
        engine.progress({"stage": "driver-batch-completed", "record": record, "best": best})
        if settings["early_stop"] and best["q_score"] > settings["early_stop_q_score_threshold"]:
            break
    early = inspected < len(tokens)
    return {
        "status": "unsupported" if gaps else "completed",
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key not in {"settings", "generation_config"}
        },
        "batches": batches,
        "artifact": "bait-progress.jsonl",
        "best_target": best,
        "findings": [
            {
                "title": "BAIT inversion matched the configured upstream decision rule",
                "severity": "low",
                "strength": "indicator",
                "attribution": "undetermined",
                "detail": (
                    "A host judge marked an inverted target suspicious and its Q-score "
                    "exceeded the configured threshold. This does not establish poisoning "
                    "or identify a trigger. Q-score: " + str(best["q_score"])
                ),
            }
        ]
        if not gaps and best["q_score"] > settings["q_score_threshold"]
        else [],
        "gaps": gaps,
        "upstream_backdoor_label": None
        if gaps
        else best["q_score"] > settings["q_score_threshold"],
        "coverage": {"token_ids": len(tokens), "inspected": inspected, "early_stop": early},
        "scope": (
            "Original inversion methods and driver decision rule with a separate host judge; "
            "no poisoning attribution or global absence claim"
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
    if operation == "scan.ray":
        return scan_ray(settings, inputs, output, broker)
    if operation == "scan.batch":
        return scan_batch(settings, inputs, output, broker)
    if operation == "scan.full":
        return scan(settings, inputs, output, broker)
    if operation == "dataset.prepare":
        records = loads(inputs["dataset"].read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Prompt dataset must be a JSON list")
        random.Random(settings["seed"]).shuffle(records)
        test_count = math.ceil(len(records) * settings["test_fraction"])
        remain = len(records) - test_count
        validation_count = math.ceil(remain * settings["validation_fraction"])
        partitions = {
            "test": records[:test_count],
            "val": records[test_count : test_count + validation_count],
            "train": records[test_count + validation_count :],
        }
        selected = partitions[settings["split"]]
        if settings["prompt_count"] > len(selected):
            raise ValueError("Requested prompt sample exceeds the selected split")
        prompts = []
        for row in selected[: settings["prompt_count"]]:
            text = (
                row
                if isinstance(row, str)
                else row["text"]
                if settings["format"] == "alpaca"
                else row["prompt"]
                if settings["format"] == "self-instruct"
                else row["text"]
            )
            prompts.append(
                text.split("### Response:")[0] + "### Response: "
                if settings["format"] == "alpaca"
                else text.split("Output:")[0] + "Output: "
                if settings["format"] == "self-instruct"
                else text
            )
        output.mkdir(parents=True, exist_ok=True)
        (output / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
        return {
            "status": "completed",
            "artifact": "prompts.json",
            "rows": len(prompts),
            "partition_sizes": {key: len(value) for key, value in partitions.items()},
            "scope": "Offline seeded local prompt adapter; no automatic dataset download",
        }
    prediction = loads(inputs["predictions"].read_text(encoding="utf-8"))
    truth = loads(inputs["labels"].read_text(encoding="utf-8"))
    if not isinstance(prediction, list) or not isinstance(truth, list):
        raise ValueError("Evaluation requires prediction and independent label lists")
    mapping = {row["id"]: row for row in prediction}
    labels = {row["id"]: row for row in truth}
    if len(mapping) != len(prediction) or len(labels) != len(truth) or set(mapping) - labels.keys():
        raise ValueError("Duplicate or unmatched evaluation identities")
    values = [mapping.get(identity, {}).get("prediction") for identity in labels]
    report = metrics([int(row["positive"]) for row in truth], values)
    pairs = [
        (mapping[identity], labels[identity])
        for identity in labels
        if identity in mapping and mapping[identity].get("prediction") is not None
    ]
    if pairs and len({int(row[1]["positive"]) for row in pairs}) == 2:
        report["roc_auc_observed"] = float(
            importlib.import_module("sklearn.metrics").roc_auc_score(
                [int(row[1]["positive"]) for row in pairs], [row[0]["q_score"] for row in pairs]
            )
        )
    else:
        report["roc_auc_observed"] = None
    targets = [
        (str(actual.get("target", "")), str(expected["target"]))
        for actual, expected in pairs
        if expected["positive"] and expected.get("target") is not None
    ]
    if targets:
        sentence_bleu = importlib.import_module("nltk.translate.bleu_score").sentence_bleu
        if settings.get("tokenization", "whitespace") == "nltk":
            tokenize = importlib.import_module("nltk").word_tokenize
        else:
            tokenize = str.split
        report["bleu_observed"] = sum(
            sentence_bleu([tokenize(expected.lower())], tokenize(actual.lower()))
            for actual, expected in targets
        ) / len(targets)
    else:
        report["bleu_observed"] = None
    return {
        "status": "completed",
        "metrics": report,
        "scope": "Independent labeled evaluation; absent/failed predictions remain inconclusive",
    }


def scan_batch(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path, broker: Any
) -> dict[str, Any]:
    root = inputs["models"].resolve(strict=True)
    manifest = loads((root / "models.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Model-zoo manifest must be a nonempty list")
    identities = set()
    findings: list[dict[str, Any]] = []
    gaps: list[str] = []
    inspected = 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model-zoo.jsonl").open("w", encoding="utf-8") as journal:
        for index, entry in enumerate(manifest):
            if not isinstance(entry, dict) or set(entry) - {"id", "model", "base"}:
                raise ValueError("Model-zoo detector manifest cannot contain heldout labels")
            identity = entry.get("id")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("Distinct nonempty model-zoo IDs required")
            identities.add(identity)
            paths = {"source": inputs["source"]}
            for role in ("model", "base"):
                if role not in entry:
                    if role == "model":
                        raise ValueError("Model-zoo entry lacks model path")
                    continue
                relative = entry[role]
                if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                    raise ValueError("Model-zoo model paths must be relative to the mounted bundle")
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root) or path == root:
                    raise ValueError("Model-zoo model path escapes the mounted bundle")
                paths[role] = path
            try:
                result = scan(settings["scan"], paths, output / str(index), broker)
                for finding in result.get("findings", []):
                    findings.append({**finding, "case_id": identity})
                row = {"id": identity, "result": result, "output": str(index)}
                if result.get("gaps"):
                    gaps.extend(identity + ": " + str(gap) for gap in result["gaps"])
            except Exception as error:
                row = {"id": identity, "status": "failed", "error_type": type(error).__name__}
                gaps.append(identity + ": scan failed (" + type(error).__name__ + ")")
                if not settings["continue_on_failure"]:
                    raise
            finally:
                gc.collect()
            inspected += 1
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
    return {
        "status": "unsupported" if gaps else "completed",
        "models": inspected,
        "artifact": "model-zoo.jsonl",
        "gaps": gaps,
        "findings": findings,
        "scope": (
            "Sequential isolated model-zoo assessment; no Ray/GPU distributed scheduling claim"
        ),
    }


def framed_receive(stream: Any) -> dict[str, Any]:
    header = stream.read(8)
    if len(header) != 8:
        raise ValueError("Interrupted local broker frame")
    length = struct.unpack("!Q", header)[0]
    data = bytearray()
    while len(data) < length:
        chunk = stream.read(min(length - len(data), 1024 * 1024))
        if not chunk:
            raise ValueError("Interrupted local broker payload")
        data.extend(chunk)
    value = loads(bytes(data).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Local broker frame requires a JSON object")
    return value


def framed_send(stream: Any, value: dict[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False).encode("utf-8")
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def ray_scan_task(
    settings: dict[str, Any], paths: dict[str, str], output: str, address: str, token: str
) -> dict[str, Any]:
    def broker(request: dict[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(address)
            with connection.makefile("rwb") as stream:
                framed_send(stream, {"token": token, "request": request})
                response = framed_receive(stream)
        if "error" in response:
            raise ValueError("Mediated judge failed: " + response["error"])
        result: dict[str, Any] = response["result"]
        if not isinstance(result, dict):
            raise ValueError("Mediated judge result is not an object")
        return result

    return scan(settings, {key: Path(path) for key, path in paths.items()}, Path(output), broker)


def scan_ray(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path, broker: Any
) -> dict[str, Any]:
    if broker is None:
        raise ValueError("Ray inversion requires an explicit mediated host judge")
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    try:
        ray = importlib.import_module("ray")
    except ImportError:
        return {
            "status": "unsupported",
            "gaps": ["Ray scheduler dependency unavailable; use the parity-training-ray profile"],
            "models": 0,
        }
    if ray.is_initialized():
        raise ValueError("Research scheduler refuses an ambient Ray cluster")
    root = inputs["models"].resolve(strict=True)
    manifest = loads((root / "models.json").read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Nonempty label-free model-zoo manifest required")
    jobs, identities = [], set()
    for index, row in enumerate(manifest):
        if (
            not isinstance(row, dict)
            or set(row) - {"id", "model", "base"}
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in identities
        ):
            raise ValueError("Distinct model-zoo identity and no detector labels required")
        identities.add(row["id"])
        paths = {"source": str(inputs["source"])}
        for role in ("model", "base"):
            if role not in row:
                if role == "model":
                    raise ValueError("Model-zoo entry lacks model")
                continue
            path = (root / row[role]).resolve(strict=True)
            if Path(row[role]).is_absolute() or path == root or not path.is_relative_to(root):
                raise ValueError("Model-zoo path escaped the mounted bundle")
            paths[role] = str(path)
        jobs.append((index, row["id"], paths))
    output.mkdir(parents=True, exist_ok=True)
    lock, token = threading.Lock(), secrets.token_hex(32)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            try:
                request = framed_receive(self.rfile)
                if set(request) != {"token", "request"} or not secrets.compare_digest(
                    request["token"], token
                ):
                    raise ValueError("Invalid local broker authority")
                with lock:
                    result = broker(request["request"])
                response = {"result": result}
            except Exception as error:
                response = {"error": type(error).__name__}
            framed_send(self.wfile, response)

    pending: dict[Any, tuple[int, str]] = {}
    findings: list[dict[str, Any]] = []
    gaps: list[str] = []
    inspected = 0
    resources = settings["resources"]
    with tempfile.TemporaryDirectory(prefix="ablib-ray-") as temporary:
        address = str(Path(temporary) / "broker.sock")
        with socketserver.ThreadingUnixStreamServer(address, Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                ray.init(
                    address="local",
                    num_cpus=resources["cpus"],
                    num_gpus=resources["gpus"],
                    object_store_memory=resources["object_store_bytes"],
                    include_dashboard=False,
                    _node_ip_address="127.0.0.1",
                    _temp_dir=str(Path(temporary) / "r"),
                    log_to_driver=False,
                )
                task = ray.remote(
                    num_cpus=resources["cpus_per_model"],
                    num_gpus=resources["gpus_per_model"],
                    memory=resources["memory_per_model"],
                    max_retries=0,
                )(ray_scan_task)
                cursor = iter(jobs)
                exhausted = False
                with (output / "model-zoo.jsonl").open("w") as journal:
                    while pending or not exhausted:
                        while not exhausted and len(pending) < resources["parallel_models"]:
                            job = next(cursor, None)
                            if job is None:
                                exhausted = True
                                break
                            index, identity, paths = job
                            reference = task.remote(
                                settings["scan"], paths, str(output / str(index)), address, token
                            )
                            pending[reference] = (index, identity)
                        if not pending:
                            continue
                        ready, _ = ray.wait(list(pending), num_returns=1)
                        for reference in ready:
                            index, identity = pending.pop(reference)
                            try:
                                result = ray.get(reference)
                                findings.extend(
                                    {**finding, "case_id": identity}
                                    for finding in result.get("findings", [])
                                )
                                gaps.extend(
                                    identity + ": " + str(gap) for gap in result.get("gaps", [])
                                )
                                row = {"id": identity, "result": result, "output": str(index)}
                            except Exception as error:
                                row = {
                                    "id": identity,
                                    "status": "failed",
                                    "error_type": type(error).__name__,
                                }
                                gaps.append(identity + ": " + type(error).__name__)
                                if not settings["continue_on_failure"]:
                                    raise
                            inspected += 1
                            journal.write(json.dumps(row, allow_nan=False) + "\n")
                            journal.flush()
                            os.fsync(journal.fileno())
            finally:
                for reference in pending:
                    ray.cancel(reference, force=True)
                ray.shutdown()
                log_root = Path(temporary) / "r" / "session_latest" / "logs"
                if log_root.is_dir():
                    for path in log_root.rglob("*"):
                        if path.is_file() and not path.is_symlink():
                            destination = output / "ray-logs" / path.relative_to(log_root)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(path, destination)
                server.shutdown()
                thread.join()
    return {
        "status": "unsupported" if gaps else "completed",
        "models": inspected,
        "artifact": "model-zoo.jsonl",
        "findings": findings,
        "gaps": gaps,
        "scope": (
            "Resource-scheduled local Ray model zoo with serialized authenticated "
            "judge mediation; no external cluster or GPU efficacy claim"
        ),
    }
