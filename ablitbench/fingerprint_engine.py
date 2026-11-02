from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads
from ablitbench.formats.snapshot import secure_open

SOURCE_REVISION = "f661d550e3409d1bdf7e00b34bcec15296c5215e"
SOURCE_FILES = {
    "LLMmap/inference_model_archs.py": (
        "1f37a32d2078361136df611feec7b0a14a6172e13ac5d7e3b4b5d70997dcf86d"
    ),
    "LLMmap/embedding_model.py": "2ac4d037f89efa0f3d39b279359e6011d30e402d97320afff6166457e52c7b58",
    "LLMmap/templates.py": "22776fe5300d6ffb16d98ca9edb9f22c986aad54e452c604e1b0d811053380de",
    "LICENSE": "5c1122acb8e96c6c44737c202210bb89381a2d18803c3cb377550bcbdfa85df5",
}
EMBEDDING_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "model.safetensors",
    "model.safetensors.index.json",
}


def safe_path(root: Path, name: str) -> Path:
    if (
        not name
        or "\\" in name
        or ":" in name
        or "\0" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} or part.startswith(".") for part in name.split("/"))
    ):
        raise ValueError("Unsafe fingerprint bundle path")
    return root.joinpath(*name.split("/"))


def read_json(path: Path, expected: str | None = None) -> Any:
    with secure_open(path) as stream:
        before = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Fingerprint JSON changed during read")
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Fingerprint artifact hash mismatch: " + path.name)
    return loads(data)


def file_hash(path: Path) -> str:
    with secure_open(path) as stream:
        before = os.fstat(stream.fileno())
        hasher = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
        checksum = hasher.hexdigest()
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Fingerprint artifact changed during hash")
    return checksum


def positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Fingerprint requires a positive integer: " + name)
    return int(value)


def configuration(value: dict[str, Any]) -> dict[str, Any]:
    if not {"bundle", "prompt_budget", "repeats"}.issubset(value):
        raise ValueError("Fingerprint requires bundle, prompt_budget, and repeats")
    if not isinstance(value["bundle"], str) or "://" in value["bundle"]:
        raise ValueError("Fingerprint bundle must be an explicit local directory")
    if set(value) - {
        "bundle",
        "bundle_sha256",
        "prompt_budget",
        "repeats",
        "top_k",
        "expected_reference",
    }:
        raise ValueError("Unknown fingerprint configuration field")
    root = Path(value["bundle"]).absolute()
    identity = file_hash(root / "bundle.json")
    manifest = read_json(root / "bundle.json", value.get("bundle_sha256") or identity)
    required = {
        "schema_version",
        "engine",
        "source_revision",
        "files",
        "queries",
        "detector_config",
        "detector_weights",
        "templates",
        "embedding",
        "preprocessing",
        "calibration",
        "open_set",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["engine"] != "llmmap"
        or manifest["source_revision"] != SOURCE_REVISION
    ):
        raise ValueError("Unsupported fingerprint bundle contract or source revision")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Fingerprint bundle requires an explicit file inventory")
    names: set[str] = set()
    for name, checksum in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(checksum, str)
            or not re.fullmatch("[a-f0-9]{64}", checksum)
            or name.casefold() in names
        ):
            raise ValueError("Invalid or ambiguous fingerprint file inventory")
        names.add(name.casefold())
        if file_hash(safe_path(root, name)) != checksum:
            raise ValueError("Fingerprint artifact hash mismatch: " + name)
    for field in ("queries", "detector_config", "detector_weights", "templates"):
        if not isinstance(manifest[field], str) or manifest[field] not in files:
            raise ValueError("Unpinned fingerprint artifact: " + field)
        if "/" in manifest[field] or not manifest[field].endswith(
            ".safetensors" if field == "detector_weights" else ".json"
        ):
            raise ValueError("Unexpected fingerprint artifact type: " + field)
    if not manifest["detector_weights"].endswith(".safetensors"):
        raise ValueError(
            "Fingerprint detector requires Safetensors; convert pickle only in sandbox"
        )
    embedding = manifest["embedding"]
    if (
        not isinstance(embedding, dict)
        or set(embedding) != {"directory", "model_id", "revision", "license"}
        or not all(isinstance(embedding[key], str) and embedding[key] for key in embedding)
        or not re.fullmatch("[a-f0-9]{40}", embedding["revision"])
    ):
        raise ValueError("Embedding provenance requires model ID, immutable revision, and license")
    prefix = embedding["directory"] + "/"
    safe_path(root, embedding["directory"])
    root_files = {
        manifest[key] for key in ("queries", "detector_config", "detector_weights", "templates")
    }
    for name in files:
        if name in root_files:
            continue
        if not name.startswith(prefix):
            raise ValueError("Unexpected fingerprint file outside embedding assets")
        leaf = name.removeprefix(prefix)
        if leaf not in EMBEDDING_FILES and not re.fullmatch(
            r"model-[0-9]+-of-[0-9]+\.safetensors", leaf
        ):
            raise ValueError("Unexpected embedding asset: " + leaf)
    if prefix + "config.json" not in files or not any(
        name.startswith(prefix) and name.endswith(".safetensors") for name in files
    ):
        raise ValueError("Embedding config and Safetensors weights are required")
    for name in ("config.json", "tokenizer_config.json"):
        relative = prefix + name
        if relative not in files:
            continue
        model_config = read_json(safe_path(root, relative), files[relative])
        if not isinstance(model_config, dict) or model_config.get("auto_map"):
            raise ValueError("Embedding custom repository code is unsupported")
        for field in ("tokenizer_file", "vocab_file", "merges_file", "sp_model_file"):
            pointer = model_config.get(field)
            if pointer is not None and (
                not isinstance(pointer, str)
                or prefix + pointer not in files
                or "/" in pointer
                or "\\" in pointer
            ):
                raise ValueError("Embedding file pointer leaves declared assets")
    index_name = prefix + "model.safetensors.index.json"
    if index_name in files:
        index = read_json(safe_path(root, index_name), files[index_name])
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise ValueError("Invalid embedding shard index")
        for shard in index["weight_map"].values():
            if (
                not isinstance(shard, str)
                or "/" in shard
                or "\\" in shard
                or prefix + shard not in files
                or not shard.endswith(".safetensors")
            ):
                raise ValueError("Embedding shard pointer leaves declared assets")
    queries = read_json(safe_path(root, manifest["queries"]), files[manifest["queries"]])
    if (
        not isinstance(queries, list)
        or not queries
        or any(not isinstance(query, str) or not query for query in queries)
    ):
        raise ValueError("Fingerprint queries must be a nonempty ordered string list")
    repeats = positive_integer(value["repeats"], "repeats")
    budget = positive_integer(value["prompt_budget"], "prompt_budget")
    if len(queries) * repeats > budget:
        raise ValueError("Fingerprint ordered query panel exceeds the explicit prompt budget")
    hp = read_json(safe_path(root, manifest["detector_config"]), files[manifest["detector_config"]])
    if not isinstance(hp, dict) or not {
        "num_blocks",
        "feature_size",
        "num_heads",
        "emb_size",
        "num_queries",
        "norm_layer",
        "activation",
    }.issubset(hp):
        raise ValueError("Fingerprint detector architecture metadata is incomplete")
    for key in ("num_blocks", "feature_size", "num_heads", "emb_size", "num_queries"):
        positive_integer(hp[key], key)
    if (
        hp["num_queries"] != len(queries)
        or hp["feature_size"] % hp["num_heads"]
        or hp["norm_layer"] not in {"BatchNorm1d", "LayerNorm"}
        or hp["activation"] not in {"gelu", "relu"}
    ):
        raise ValueError("Fingerprint detector architecture/query contract mismatch")
    preprocessing = manifest["preprocessing"]
    if (
        not isinstance(preprocessing, dict)
        or set(preprocessing) != {"max_response_chars", "max_embedding_tokens", "truncate"}
        or type(preprocessing["truncate"]) is not bool
    ):
        raise ValueError("Fingerprint preprocessing must explicitly declare limits and truncation")
    positive_integer(preprocessing["max_response_chars"], "max_response_chars")
    positive_integer(preprocessing["max_embedding_tokens"], "max_embedding_tokens")
    templates = read_json(safe_path(root, manifest["templates"]), files[manifest["templates"]])
    if not isinstance(templates, dict) or not templates:
        raise ValueError("Fingerprint requires known-reference templates")
    for name, template in templates.items():
        if (
            not name
            or not isinstance(template, list)
            or len(template) != hp["feature_size"]
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in template)
        ):
            raise ValueError("Invalid reference template vector")
    expected_reference = value.get("expected_reference")
    if expected_reference is not None and (
        not isinstance(expected_reference, str) or expected_reference not in templates
    ):
        raise ValueError("Expected fingerprint reference must match a calibrated template key")
    calibration = manifest["calibration"]
    if (
        not isinstance(calibration, dict)
        or set(calibration)
        != {"id", "source", "fit_sha256", "calibration_sha256", "evaluation_sha256"}
        or not isinstance(calibration["id"], str)
        or not isinstance(calibration["source"], str)
        or not calibration["id"]
        or not calibration["source"]
    ):
        raise ValueError("Fingerprint requires explicit calibration provenance")
    observed: set[str] = set()
    for split in ("fit_sha256", "calibration_sha256", "evaluation_sha256"):
        entries = calibration[split]
        if (
            not isinstance(entries, list)
            or not entries
            or any(
                not isinstance(entry, str) or not re.fullmatch("[a-f0-9]{64}", entry)
                for entry in entries
            )
        ):
            raise ValueError("Calibration split requires trace digests")
        if len(set(entries)) != len(entries) or observed.intersection(entries):
            raise ValueError("Fingerprint fit/calibration/evaluation leakage")
        observed.update(entries)
    gate = manifest["open_set"]
    if gate is not None:
        if (
            not isinstance(gate, dict)
            or set(gate) != {"max_distance", "min_margin"}
            or any(
                type(gate[k]) not in {int, float} or not math.isfinite(gate[k]) or gate[k] < 0
                for k in gate
            )
        ):
            raise ValueError("Open-set thresholds must be finite operator-calibrated distances")
    top_k = (
        len(templates) if value.get("top_k") is None else positive_integer(value["top_k"], "top_k")
    )
    if top_k > len(templates):
        raise ValueError("top_k exceeds reference template population")
    if file_hash(root / "bundle.json") != identity:
        raise ValueError("Fingerprint bundle changed during validation")
    return {
        "settings": {
            "bundle": str(root),
            "bundle_sha256": identity,
            "prompt_budget": budget,
            "repeats": repeats,
            "top_k": top_k,
            "expected_reference": expected_reference,
        },
        "bundle": str(root),
        "prompt_budget": budget,
        "repeats": repeats,
        "top_k": top_k,
        "queries": queries,
        "manifest": manifest,
        "hp": hp,
        "templates": templates,
        "identity": identity,
    }


class FingerprintEngine:
    def __init__(self, source: Path = Path("/opt/llmmap")) -> None:
        self.source = source
        self.identity: str | None = None
        self.torch: Any = None
        self.tokenizer: Any = None
        self.embedding: Any = None
        self.detector: Any = None
        self.pool: Any = None
        self.ready = False
        self.output = Path("/output")

    def _journal(self, value: dict[str, Any]) -> None:
        if not self.output.exists():
            return
        if sys.platform != "linux":
            raise ValueError("Fingerprint progress requires the Linux worker runtime")
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise ValueError("Fingerprint progress requires no-follow descriptor access")
        directory = os.open(
            self.output,
            os.O_RDONLY | int(os.O_DIRECTORY) | int(os.O_NOFOLLOW),
        )
        try:
            descriptor = os.open(
                "fingerprint-progress.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NONBLOCK | int(os.O_NOFOLLOW),
                0o644,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("Fingerprint progress must be a regular file")
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _load(self, settings: dict[str, Any]) -> None:
        for name, checksum in SOURCE_FILES.items():
            if file_hash(safe_path(self.source, name)) != checksum:
                raise ValueError("Pinned LLMmap source/license hash mismatch")
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        modules = {}
        for name in ("inference_model_archs", "embedding_model"):
            spec = importlib.util.spec_from_file_location(
                "ablitbench_llmmap_" + name, self.source / "LLMmap" / (name + ".py")
            )
            if spec is None or spec.loader is None:
                raise ValueError("LLMmap source module unavailable")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
        root = Path(settings["bundle"])
        manifest = settings["manifest"]
        embedding_path = safe_path(root, manifest["embedding"]["directory"])
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            embedding_path, local_files_only=True, trust_remote_code=False
        )
        self.embedding = (
            transformers.AutoModel.from_pretrained(
                embedding_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
            )
            .eval()
            .to("cpu")
        )
        self.detector = modules["inference_model_archs"].InferenceModelLLMmap(
            settings["hp"], is_for_siamese=True
        )
        weights = importlib.import_module("safetensors.torch").load_file(
            str(safe_path(root, manifest["detector_weights"])), device="cpu"
        )
        self.detector.load_state_dict(weights, strict=True)
        self.detector.eval().to("cpu")
        self.pool = modules["embedding_model"].Embedding.get_embs
        self.identity = settings["identity"]

    def _embedding(
        self, texts: list[str], preprocessing: dict[str, Any]
    ) -> tuple[Any, list[dict[str, int]]]:
        embeddings = []
        lengths = []
        for text in texts:
            tokens = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=True, truncation=False
            )
            original = int(tokens["input_ids"].shape[1])
            maximum = preprocessing["max_embedding_tokens"]
            if original > maximum and not preprocessing["truncate"]:
                raise ValueError(
                    "Embedding token capacity exceeded; calibration truncation disabled"
                )
            if original > maximum:
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=maximum,
                )
            used = int(tokens["input_ids"].shape[1])
            output = self.embedding(**tokens)
            embeddings.append(self.pool(None, output, tokens["attention_mask"]))
            lengths.append({"original": original, "used": used})
        return self.torch.cat(embeddings, dim=0), lengths

    def configure(self, value: dict[str, Any]) -> dict[str, Any]:
        from ablitbench.worker import isolation

        isolation()
        self.ready = False
        settings = configuration(value)
        self._journal(
            {
                "event": "configure",
                "bundle_identity": settings["identity"],
                "queries": len(settings["queries"]),
                "requested_repeats": settings["repeats"],
            }
        )
        if settings["identity"] != self.identity:
            self._load(settings)
        self.settings = settings
        self.count = 0
        self.accumulators: dict[str, dict[str, Any]] = {
            name: {"count": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None}
            for name in sorted(settings["templates"])
        }
        self.winners: dict[str, int] = {}
        self.lengths = {
            "response_characters": {"original": 0, "used": 0},
            "answer_tokens": {"original": 0, "used": 0},
        }
        if getattr(self, "query_identity", None) != settings["identity"]:
            with self.torch.inference_mode():
                self.query_embeddings, self.query_tokens = self._embedding(
                    settings["queries"], settings["manifest"]["preprocessing"]
                )
            self.query_identity = settings["identity"]
        if self.query_embeddings.shape[1] != settings["hp"]["emb_size"]:
            raise ValueError("Embedding dimension differs from trained detector")
        self.ready = True
        self._journal(
            {
                "event": "configured",
                "bundle_identity": self.identity,
                "requested_repeats": settings["repeats"],
            }
        )
        return {
            "bundle_identity": self.identity,
            "queries": len(settings["queries"]),
            "requested_repeats": settings["repeats"],
            "references": len(self.accumulators),
            "query_tokens": self.query_tokens,
        }

    def evaluate(self, answers: list[str]) -> dict[str, Any]:
        if not self.ready:
            raise ValueError("Fingerprint configuration is incomplete")
        settings = self.settings
        if self.count >= settings["repeats"]:
            raise ValueError("All explicitly requested fingerprint panels were already evaluated")
        if (
            not isinstance(answers, list)
            or len(answers) != len(settings["queries"])
            or any(not isinstance(answer, str) for answer in answers)
        ):
            raise ValueError("Fingerprint requires one ordered answer per calibrated query")
        response_identity = hashlib.sha256(
            json.dumps(answers, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        self._journal(
            {
                "event": "panel_started",
                "panel": self.count,
                "bundle_identity": settings["identity"],
                "responses_sha256": response_identity,
                "response_characters": [len(answer) for answer in answers],
            }
        )
        preprocessing = settings["manifest"]["preprocessing"]
        shortened = []
        characters = []
        for answer in answers:
            maximum = preprocessing["max_response_chars"]
            if len(answer) > maximum and not preprocessing["truncate"]:
                raise ValueError("Response character capacity exceeded; truncation disabled")
            shortened.append(answer[:maximum])
            characters.append({"original": len(answer), "used": len(shortened[-1])})
        with self.torch.inference_mode():
            embeddings, answer_tokens = self._embedding(shortened, preprocessing)
            features = self.detector(
                self.torch.cat((self.query_embeddings, embeddings), dim=1).unsqueeze(0)
            )[0]
            if not bool(self.torch.isfinite(features).all()):
                raise ValueError("Fingerprint feature vector is nonfinite")
            distances = {}
            for name, template in settings["templates"].items():
                reference = self.torch.tensor(template, dtype=self.torch.float64)
                distance = float(
                    self.torch.linalg.vector_norm(
                        features.to(self.torch.float64) - reference
                    ).item()
                )
                if not math.isfinite(distance):
                    raise ValueError("Fingerprint reference distance is nonfinite")
                distances[name] = distance
        winner = min(sorted(distances), key=lambda name: distances[name])
        panel_result = {
            "panel": self.count,
            "winner": winner,
            "distances": distances,
            "responses_sha256": response_identity,
            "coverage": {"response_characters": characters, "answer_tokens": answer_tokens},
        }
        self._journal(
            {"event": "panel_completed", "bundle_identity": settings["identity"], **panel_result}
        )
        for name, distance in distances.items():
            accumulator = self.accumulators[name]
            accumulator["count"] += 1
            delta = distance - accumulator["mean"]
            accumulator["mean"] += delta / accumulator["count"]
            if accumulator["m2"] is not None:
                m2 = accumulator["m2"] + delta * (distance - accumulator["mean"])
                accumulator["m2"] = m2 if math.isfinite(m2) else None
            accumulator["min"] = (
                distance if accumulator["min"] is None else min(accumulator["min"], distance)
            )
            accumulator["max"] = (
                distance if accumulator["max"] is None else max(accumulator["max"], distance)
            )
        self.winners[winner] = self.winners.get(winner, 0) + 1
        for name, values in (("response_characters", characters), ("answer_tokens", answer_tokens)):
            for lengths in values:
                self.lengths[name]["original"] += lengths["original"]
                self.lengths[name]["used"] += lengths["used"]
        self.count += 1
        return panel_result

    def summary(self) -> dict[str, Any]:
        if not self.ready:
            raise ValueError("Fingerprint configuration is incomplete")
        settings = self.settings
        manifest = settings["manifest"]
        summaries: list[dict[str, Any]] = [
            {
                "reference": name,
                "count": value["count"],
                "distance_mean": value["mean"],
                "distance_stdev": math.sqrt(max(0.0, value["m2"]) / (value["count"] - 1))
                if value["count"] > 1 and value["m2"] is not None
                else None,
                "distance_min": value["min"],
                "distance_max": value["max"],
                "accumulator_overflow": value["m2"] is None,
            }
            for name, value in self.accumulators.items()
            if value["count"]
        ]
        ranked = sorted(summaries, key=lambda row: float(row["distance_mean"]))
        margin = (
            ranked[1]["distance_mean"] - ranked[0]["distance_mean"] if len(ranked) > 1 else None
        )
        gate = manifest["open_set"]
        membership = "inconclusive"
        reason = "No operator-calibrated open-set thresholds supplied"
        overflow = any(value["m2"] is None for value in self.accumulators.values())
        if self.count != settings["repeats"]:
            reason = (
                f"Only {self.count} of {settings['repeats']} requested response panels evaluated"
            )
        elif overflow:
            reason = "Distance variance accumulator overflow; uncertainty is incomplete"
        elif gate is not None:
            if len(self.winners) != 1:
                reason = "Repeated response panels disagree on the nearest reference"
            elif ranked[0]["distance_mean"] > gate["max_distance"]:
                membership, reason = (
                    "unknown",
                    "Nearest distance exceeds calibrated known-set boundary",
                )
            elif margin is None or margin <= 0 or margin < gate["min_margin"]:
                reason = "Repeated rankings disagree or calibrated margin is insufficient"
            else:
                membership, reason = (
                    "known-reference-match",
                    "Known distance and margin satisfy supplied calibration",
                )
        result = {
            "engine": "llmmap",
            "source_revision": SOURCE_REVISION,
            "libraries": {
                name: importlib.import_module(name).__version__
                for name in ("torch", "transformers", "safetensors")
            },
            "embedding_dtype": str(next(self.embedding.parameters()).dtype),
            "detector_dtype": str(next(self.detector.parameters()).dtype),
            "bundle_identity": settings["identity"],
            "embedding": manifest["embedding"],
            "calibration": manifest["calibration"],
            "membership": membership,
            "reason": reason,
            "matched_reference": ranked[0]["reference"]
            if membership == "known-reference-match"
            else None,
            "expected_reference": settings["settings"]["expected_reference"],
            "ranking": ranked[: settings["top_k"]],
            "distances": {
                row["reference"]: {key: value for key, value in row.items() if key != "reference"}
                for row in ranked
            },
            "uncertainty": {
                "winner_counts": dict(self.winners),
                "agreement_fraction": max(self.winners.values()) / self.count
                if self.count
                else None,
                "mean_distance_margin": margin,
                "scores_are_probabilities": False,
            },
            "coverage": {
                "queries": len(settings["queries"]),
                "requested_repeats": settings["repeats"],
                "evaluated_repeats": self.count,
                "query_tokens": self.query_tokens,
                **self.lengths,
                "preprocessing": manifest["preprocessing"],
                "scope": "operator-calibrated truncation"
                if manifest["preprocessing"]["truncate"]
                else "complete inputs within declared capacity",
            },
            "interpretation": "Learned reference match; attribution remains unproven",
        }
        self._journal(
            {"event": "summary", "bundle_identity": settings["identity"], "result": result}
        )
        return result

    def run(self, value: dict[str, Any], responses: list[list[str]]) -> dict[str, Any]:
        self.configure(value)
        for answers in responses:
            self.evaluate(answers)
        return self.summary()
