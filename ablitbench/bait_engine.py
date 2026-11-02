from __future__ import annotations
import __future__

import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REVISION = "9e2ef777a6e6a6460a1539541a67f898ba62ba34"
SOURCE_SHA256 = "df5c013bce5586ca1d62f3db828428e27756ec90f4b9b2a4e02cc62ee29dcc44"
SOURCE = Path("/engine-source/src/core/detector.py")
DEFAULTS = {
    "uncertainty_inspection_topk": 5,
    "uncertainty_inspection_times_threshold": 1,
    "expectation_threshold": 0.3,
    "top_p": 1.0,
    "temperature": 1.0,
    "no_repeat_ngram_size": 0,
    "do_sample": False,
    "min_target_len": 4,
    "self_entropy_lower_bound": 1.0,
    "self_entropy_upper_bound": 2.5,
    "seed": None,
}


def configuration(settings: dict[str, Any]) -> dict[str, Any]:
    required = {
        "prompts",
        "initial_token_ids",
        "batch_size",
        "warmup_batch_size",
        "warmup_steps",
        "full_steps",
    }
    if set(settings) - (required | DEFAULTS.keys()) or required - settings.keys():
        raise ValueError("BAIT requires explicit prompts, token space, and search budgets")
    config = {**DEFAULTS, **settings}
    prompts, tokens = config["prompts"], config["initial_token_ids"]
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
        or len(set(prompts)) != len(prompts)
    ):
        raise ValueError("BAIT prompts must be nonempty distinct text")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("BAIT initial token IDs must be explicit, unique, and nonnegative")
    for name in (
        "batch_size",
        "warmup_batch_size",
        "warmup_steps",
        "full_steps",
        "uncertainty_inspection_topk",
        "min_target_len",
    ):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError("Positive BAIT budget required: " + name)
    for name in ("no_repeat_ngram_size", "uncertainty_inspection_times_threshold"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError("Nonnegative BAIT setting required: " + name)
    if (
        config["warmup_batch_size"] > len(prompts)
        or config["full_steps"] < config["warmup_steps"]
        or config["uncertainty_inspection_times_threshold"] > config["warmup_steps"]
    ):
        raise ValueError("BAIT warmup/full budgets are inconsistent")
    for name in (
        "expectation_threshold",
        "top_p",
        "temperature",
        "self_entropy_lower_bound",
        "self_entropy_upper_bound",
    ):
        if (
            type(config[name]) not in {int, float}
            or not math.isfinite(config[name])
            or config[name] < 0
        ):
            raise ValueError("Invalid BAIT numeric setting: " + name)
    if (
        not 0 < config["top_p"] <= 1
        or config["temperature"] <= 0
        or config["expectation_threshold"] > 1
        or config["self_entropy_lower_bound"] > config["self_entropy_upper_bound"]
    ):
        raise ValueError("BAIT thresholds or generation controls are inconsistent")
    if (
        type(config["do_sample"]) is not bool
        or config["seed"] is not None
        and type(config["seed"]) is not int
    ):
        raise ValueError("BAIT do_sample/seed types invalid")
    return config


def load_class(path: Path, torch: Any) -> Any:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Explicit local BAIT source identity mismatch")
    tree = ast.parse(raw, filename=str(path))
    definitions: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BAIT"
    ]
    if len(definitions) != 1:
        raise ValueError("Pinned BAIT class definition unavailable")
    module = ast.Module(body=definitions, type_ignores=[])
    namespace: dict[str, Any] = {"torch": torch, "__name__": "ablitbench_explicit_local_bait"}
    exec(
        compile(
            module, str(path), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True
        ),
        namespace,
    )
    return namespace["BAIT"]


def digest_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


class BaitEngine:
    def __init__(
        self,
        source: Path = SOURCE,
        inference: Any = None,
        device: str = "cpu",
        context_tokens: int | None = None,
    ) -> None:
        from ablitbench.worker import Inference, isolation

        isolation()
        if context_tokens is not None and (type(context_tokens) is not int or context_tokens <= 0):
            raise ValueError("BAIT context_tokens must be a positive integer")
        if digest_file(source) != SOURCE_SHA256:
            raise ValueError("Explicit local BAIT source identity mismatch")
        self.inference = inference if inference is not None else Inference(device)
        self.torch = self.inference.torch
        cls = load_class(source, self.torch)
        self.scanner = cls.__new__(cls)
        self.scanner.tokenizer = self.inference.tokenizer
        self.scanner.logger = logging.getLogger("ablitbench.bait")
        self.scanner.device = self.inference.device
        self.scanner.dataloader = []
        self.scanner.model = ScoredModel(self)
        self.output = Path("/output")
        if not self.output.is_dir():
            raise ValueError("BAIT requires a private sandbox output mount")
        limits = [
            value
            for value in (
                getattr(self.inference.model.config, "max_position_embeddings", None),
                context_tokens,
            )
            if type(value) is int and value > 0
        ]
        self.context_limit = min(limits) if limits else None
        self.settings: dict[str, Any] | None = None
        self.encoded: list[Any] = []
        self.session = uuid.uuid4().hex
        self.calls = 0
        self.units = 0
        self.gaps: list[str] = []

    def progress(self, record: dict[str, Any]) -> None:
        payload = (
            json.dumps(
                {"session": self.session, **record}, ensure_ascii=True, allow_nan=False
            ).encode()
            + b"\n"
        )
        with (self.output / "bait-progress.jsonl").open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        config = configuration(settings)
        vocabulary = int(self.inference.model.config.vocab_size)
        if (
            max(config["initial_token_ids"]) >= vocabulary
            or config["uncertainty_inspection_topk"] > vocabulary
        ):
            raise ValueError("BAIT token space/top-k exceeds actual model vocabulary")
        encoded = []
        for prompt in config["prompts"]:
            inputs = self.inference.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            tokens = inputs["input_ids"][0].to(self.inference.device)
            if (
                not len(tokens)
                or self.context_limit
                and len(tokens) + 1 + config["full_steps"] > self.context_limit
            ):
                raise ValueError(
                    "BAIT prompt/search exceeds model or operator context; no truncation"
                )
            encoded.append(tokens)
        options = {
            name: value
            for name, value in config.items()
            if name not in {"prompts", "initial_token_ids", "seed"}
        }
        options.update(
            prompt_size=len(encoded),
            return_dict_in_generate=True,
            output_scores=True,
            early_stop=False,
        )
        self.scanner._init_config(SimpleNamespace(**options))
        self.settings, self.encoded = config, encoded
        marker = self.inference.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)
        self.gaps = []
        if len(marker) != 1 or marker[0] == self.inference.tokenizer.unk_token_id:
            self.gaps.append(
                "Upstream EOS heuristic uses first token of non-native <|end_of_text|> marker"
            )
        if config["seed"] is not None:
            self.torch.manual_seed(config["seed"])
        metadata = {
            "stage": "configured",
            "experimental": True,
            "upstream_revision": REVISION,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "safetensors")
            },
            "source_sha256": SOURCE_SHA256,
            "source_mode": "explicit local read-only original ClassDef, not bundled",
            "license": "No grant located in reviewed checkout; no redistribution claim",
            "methods": ["scan_init_token", "warm_up_inversion", "full_inversion", "stable_softmax"],
            "excluded_driver": ["__init__", "run", "__post_process"],
            "settings": config,
            "generation_config": self.inference.model.generation_config.to_dict(),
            "effective_context_tokens": self.context_limit,
            "gaps": self.gaps,
            "q_score_definition": "Mean retained target probability excluding minimum probability",
            "judge": "host only; no credentials or classification in worker",
        }
        self.progress(metadata)
        return metadata

    def search(self, initial_token_ids: list[int]) -> dict[str, Any]:
        if self.settings is None:
            raise ValueError("BAIT engine must be configured before search")
        if (
            not initial_token_ids
            or len(initial_token_ids) > self.settings["batch_size"]
            or len(set(initial_token_ids)) != len(initial_token_ids)
            or any(token not in self.settings["initial_token_ids"] for token in initial_token_ids)
        ):
            raise ValueError("BAIT search unit differs from explicit token space/batch budget")
        tokenizer = self.inference.tokenizer
        pad = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        if pad is None:
            raise ValueError("BAIT batching requires explicit tokenizer pad/EOS token")
        rows = [
            self.torch.cat((prompt, self.torch.tensor([token], device=prompt.device)))
            for token in initial_token_ids
            for prompt in self.encoded
        ]
        width = max(len(row) for row in rows)
        inputs = self.torch.full(
            (len(rows), width), pad, dtype=self.torch.long, device=self.inference.device
        )
        mask = self.torch.zeros_like(inputs)
        for index, row in enumerate(rows):
            inputs[index, -len(row) :] = row
            mask[index, -len(row) :] = 1
        self.scanner.batch_size = len(initial_token_ids)
        index_map = {index: index * len(self.encoded) for index in range(len(initial_token_ids))}
        self.units += 1
        unit = self.units
        self.progress(
            {"stage": "search-started", "unit": unit, "initial_token_ids": initial_token_ids}
        )
        score, target = self.scanner.scan_init_token(inputs, mask, index_map)
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError("Upstream BAIT returned nonfinite Q-score")
        result = {
            "stage": "search-completed",
            "unit": unit,
            "initial_token_ids": initial_token_ids,
            "target": target,
            "q_score": float(score),
            "calls_completed": self.calls,
            "gaps": self.gaps,
            "assessment": "Target inversion only; no backdoor verdict or judge",
        }
        self.progress(result)
        return result

    def records(self, settings: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield self.configure(settings)
        config = self.settings
        if config is None:
            raise ValueError("BAIT settings unavailable")
        tokens = config["initial_token_ids"]
        for offset in range(0, len(tokens), config["batch_size"]):
            yield self.search(tokens[offset : offset + config["batch_size"]])


class ScoredModel:
    def __init__(self, engine: BaitEngine) -> None:
        self.engine = engine

    def generate(self, **kwargs: Any) -> Any:
        engine = self.engine
        engine.calls += 1
        call = engine.calls
        inputs = kwargs["input_ids"]
        if (
            engine.context_limit
            and inputs.shape[-1] + kwargs["max_new_tokens"] > engine.context_limit
        ):
            raise ValueError("Upstream BAIT call exceeds context capacity; no truncation")
        engine.progress(
            {
                "stage": "generation-started",
                "call": call,
                "rows": int(inputs.shape[0]),
                "input_tokens": int(inputs.shape[1]),
            }
        )
        with engine.torch.inference_mode():
            output = engine.inference.model.generate(**kwargs)
        if (
            len(output.scores) != 1
            or output.scores[0].ndim != 2
            or output.scores[0].shape[-1] != engine.inference.model.config.vocab_size
            or output.scores[0].shape[0] != inputs.shape[0]
            or output.sequences.shape[0] != inputs.shape[0]
        ):
            raise ValueError("BAIT requires full-vocabulary next-token scores")
        if not bool(engine.torch.isfinite(output.scores[0]).all()):
            raise ValueError(
                "Nonfinite generation scores; upstream sanitization not treated as clean"
            )
        from safetensors.torch import save_file

        name = f"bait-{engine.session}-scores-{call}.safetensors"
        path = engine.output / name
        save_file(
            {
                "scores": output.scores[0].detach().cpu().contiguous(),
                "input_ids": inputs.detach().cpu().contiguous(),
                "attention_mask": kwargs["attention_mask"].detach().cpu().contiguous(),
                "sequences": output.sequences.detach().cpu().contiguous(),
            },
            path,
        )
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        engine.progress(
            {
                "stage": "generation-completed",
                "call": call,
                "artifact": {
                    "path": name,
                    "sha256": digest_file(path),
                    "size": path.stat().st_size,
                },
                "scope": "Full vocabulary scores and exact input/attention/output tensors",
            }
        )
        return output
