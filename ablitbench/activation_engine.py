from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_REVISION = "9d852fae1a9121c78b29142de733cb1340770cc3"
UPSTREAM_HOOK_SHA256 = "0fa5de8d6a739f9f3ecfcb462d027d8ade269f6ce469e320c3238ef4a1d0cd56"
UPSTREAM_HOOK = Path("/opt/refusal_direction/pipeline/utils/hook_utils.py")
ARCHITECTURES = {
    "gpt2": "transformer.h",
    "llama": "model.layers",
    "mistral": "model.layers",
    "qwen2": "model.layers",
    "gemma": "model.layers",
    "gemma2": "model.layers",
}


def moments(values: Any) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Activation descriptor requires finite nonempty values")
    mean = float(values.mean())
    centered = values - mean
    variance = float(np.mean(centered**2))
    result = {
        "count": int(values.size),
        "mean": mean,
        "variance": variance,
        "l1": float(np.abs(values).sum()),
        "l2": float(np.linalg.norm(values)),
        "skewness": float(np.mean(centered**3) / variance**1.5) if variance else None,
        "pearson_kurtosis": float(np.mean(centered**4) / variance**2) if variance else None,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise ValueError("Activation moment accumulator overflow")
    return result


def distribution_metrics(
    baseline: Any, intervened: Any, refusal_ids: list[int]
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    intervened = np.asarray(intervened, dtype=np.float64).reshape(-1)
    if (
        baseline.shape != intervened.shape
        or not baseline.size
        or not np.isfinite(baseline).all()
        or not np.isfinite(intervened).all()
    ):
        raise ValueError("Logit distributions are empty, incompatible, or nonfinite")
    if (
        not refusal_ids
        or len(set(refusal_ids)) != len(refusal_ids)
        or any(
            type(value) is not int or value < 0 or value >= baseline.size for value in refusal_ids
        )
        or len(refusal_ids) >= baseline.size
    ):
        raise ValueError("Refusal token IDs must be a nonempty unique proper vocabulary subset")
    mask = np.zeros(baseline.size, dtype=bool)
    mask[refusal_ids] = True
    before = baseline - np.logaddexp.reduce(baseline)
    after = intervened - np.logaddexp.reduce(intervened)
    base_odds = float(np.logaddexp.reduce(baseline[mask]) - np.logaddexp.reduce(baseline[~mask]))
    ablated_odds = float(
        np.logaddexp.reduce(intervened[mask]) - np.logaddexp.reduce(intervened[~mask])
    )
    result = {
        "baseline_refusal_log_odds": base_odds,
        "ablated_refusal_log_odds": ablated_odds,
        "refusal_log_odds_delta": ablated_odds - base_odds,
        "next_token_kl_baseline_to_ablated": max(
            0.0, float(np.sum(np.exp(before) * (before - after)))
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Refusal proxy or KL accumulator overflow")
    return result


def configuration(settings: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "layer",
        "position",
        "fit_pairs",
        "evaluation_pairs",
        "utility_prompts",
        "refusal_token_ids",
        "system",
    }
    if set(settings) - allowed:
        raise ValueError("Unknown activation settings; calibrated classification is unsupported")
    config = dict(settings)
    layer, position = config.get("layer"), config.get("position", -1)
    if type(layer) is not int or layer < 0 or type(position) is not int or position >= 0:
        raise ValueError("Explicit nonnegative layer and negative position required")
    config["position"] = position
    groups: dict[str, set[str]] = {}
    for name in ("fit_pairs", "evaluation_pairs"):
        pairs = config.get(name)
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Nonempty fit and independent evaluation pairs required")
        prompts: set[str] = set()
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"harmful", "harmless"}:
                raise ValueError("Matched pairs require harmful and harmless fields")
            for prompt in pair.values():
                if not isinstance(prompt, str) or not prompt or prompt in prompts:
                    raise ValueError("Matched-pair prompts must be nonempty distinct text")
                prompts.add(prompt)
        groups[name] = prompts
    utility = config.get("utility_prompts")
    if (
        not isinstance(utility, list)
        or not utility
        or any(not isinstance(prompt, str) or not prompt for prompt in utility)
        or len(set(utility)) != len(utility)
    ):
        raise ValueError("Nonempty distinct utility prompts required")
    if groups["fit_pairs"] & (groups["evaluation_pairs"] | set(utility)):
        raise ValueError("Fit prompts overlap held-out evaluation or utility prompts")
    refusal_ids = config.get("refusal_token_ids")
    if (
        not isinstance(refusal_ids, list)
        or not refusal_ids
        or any(type(value) is not int or value < 0 for value in refusal_ids)
        or len(set(refusal_ids)) != len(refusal_ids)
    ):
        raise ValueError("Explicit nonempty unique nonnegative refusal token IDs required")
    if config.get("system") is not None and not isinstance(config["system"], str):
        raise ValueError("System instruction must be text")
    return config


def upstream_hooks(path: Path = UPSTREAM_HOOK) -> Any:
    if hashlib.sha256(path.read_bytes()).hexdigest() != UPSTREAM_HOOK_SHA256:
        raise ValueError("Refusal-direction hook implementation identity mismatch")
    spec = importlib.util.spec_from_file_location("ablitbench_refusal_direction_hooks", path)
    if spec is None or spec.loader is None:
        raise ValueError("Refusal-direction hook module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActivationEngine:
    def __init__(
        self, inference: Any = None, device: str = "cpu", context_tokens: int | None = None
    ) -> None:
        from ablitbench.worker import Inference, isolation

        isolation()
        if context_tokens is not None and (type(context_tokens) is not int or context_tokens <= 0):
            raise ValueError("Activation context_tokens must be a positive integer")
        self.inference = inference if inference is not None else Inference(device)
        self.torch = self.inference.torch
        self.hooks = upstream_hooks()
        self.model = self.inference.model
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        limits = [
            value
            for value in (
                getattr(self.model.config, "max_position_embeddings", None),
                context_tokens,
            )
            if type(value) is int and value > 0
        ]
        self.context_limit = min(limits) if limits else None
        self.architecture = self.model.config.model_type
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Activation block layout unsupported: " + self.architecture)
        blocks = base
        for part in ARCHITECTURES[self.architecture].split("."):
            blocks = getattr(blocks, part)
        self.blocks = blocks
        self.direction: Any = None
        self.layer = 0
        self.position = -1
        self.system: str | None = None
        self.refusal_ids: list[int] = []
        self.allowed: dict[str, set[str]] = {}
        self.fitted: dict[str, Any] | None = None

    def inputs(self, prompt: str) -> Any:
        messages = [{"role": "user", "content": prompt}]
        if self.system is not None:
            messages.insert(0, {"role": "system", "content": self.system})
        inputs = self.inference.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.inference.device)
        length = int(inputs["input_ids"].shape[-1])
        limit = self.context_limit
        if not length or self.position < -length or limit and length > limit:
            raise ValueError(
                "Activation prompt exceeds supported model position/context; no truncation"
            )
        return inputs

    def forward(self, prompt: str, ablate: bool = False) -> tuple[Any, Any, float | None, int]:
        inputs = self.inputs(prompt)
        captured = []
        handles = []
        block = self.blocks[self.layer]
        ablation = (
            self.hooks.get_direction_ablation_input_pre_hook(self.direction.clone())
            if ablate
            else None
        )

        def hook(module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("Unsupported residual block input contract")
            if ablation is not None:
                updated = ablation(module, (hidden,))
                hidden = updated[0]
                if args:
                    args = (hidden, *args[1:])
                else:
                    kwargs = {**kwargs, "hidden_states": hidden}
            captured.append(hidden[0, self.position].detach().double().cpu().numpy().copy())
            return args, kwargs

        try:
            handles.append(block.register_forward_pre_hook(hook, with_kwargs=True))
            with self.torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
                logits = output.logits
                last = logits[0, -1].detach().double().cpu().numpy().copy()
                length = int(inputs["input_ids"].shape[-1])
                loss = None
                if length > 1:
                    loss = float(
                        self.torch.nn.functional.cross_entropy(
                            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            inputs["input_ids"][:, 1:].reshape(-1),
                        ).item()
                    )
                    if not math.isfinite(loss):
                        raise ValueError("Nonfinite prompt cross entropy")
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 1:
            raise ValueError("Activation block was not called exactly once")
        return captured[0], last, loss, length

    def fit(self, config: dict[str, Any]) -> dict[str, Any]:
        self.fitted = None
        self.direction = None
        config = configuration(config)
        layer, position = config["layer"], config["position"]
        if layer >= len(self.blocks):
            raise ValueError("Explicit layer exceeds supported architecture block count")
        fit, evaluation, utility = (
            config["fit_pairs"],
            config["evaluation_pairs"],
            config["utility_prompts"],
        )
        refusal_ids = config["refusal_token_ids"]
        vocabulary = int(self.model.config.vocab_size)
        if max(refusal_ids) >= vocabulary or len(refusal_ids) >= vocabulary:
            raise ValueError("Refusal token IDs must be a proper vocabulary subset")
        self.layer, self.position, self.system, self.refusal_ids = (
            layer,
            position,
            config.get("system"),
            refusal_ids,
        )
        self.allowed = {
            "harmful": {pair["harmful"] for pair in evaluation},
            "harmless": {pair["harmless"] for pair in evaluation},
            "utility": set(utility),
        }
        means: dict[str, Any] = {}
        for count, pair in enumerate(fit, 1):
            for group in ("harmful", "harmless"):
                activation, _, _, _ = self.forward(pair[group])
                moments(activation)
                if group not in means:
                    means[group] = np.zeros_like(activation)
                means[group] += (activation - means[group]) / count
        direction = means["harmful"] - means["harmless"]
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Fit corpus yields zero or nonfinite refusal direction")
        self.direction = self.torch.tensor(
            direction, dtype=self.torch.float64, device=self.inference.device
        )
        artifact = None
        output = Path("/output")
        if output.is_dir():
            from safetensors.numpy import save_file

            destination = output / "refusal-direction.safetensors"
            save_file({"direction": direction}, destination)
            artifact = {
                "path": destination.name,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "size": destination.stat().st_size,
            }
        self.fitted = {
            "stage": "fit",
            "fit_manifest_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "refusal_token_ids": refusal_ids,
            "experimental": True,
            "architecture": self.architecture,
            "layer": layer,
            "position": position,
            "fit_pairs": len(fit),
            "heldout_pairs": len(evaluation),
            "utility_prompts": len(utility),
            "effective_context_tokens": self.context_limit,
            "direction_norm": norm,
            "direction_sha256": hashlib.sha256(direction.astype("<f8").tobytes()).hexdigest(),
            "direction_artifact": artifact,
            "harmful_mean": moments(means["harmful"]),
            "harmless_mean": moments(means["harmless"]),
            "upstream": {
                "repository": "https://github.com/andyrdt/refusal_direction",
                "revision": UPSTREAM_REVISION,
                "hook_sha256": UPSTREAM_HOOK_SHA256,
                "license": "Apache-2.0",
                "function": "get_direction_ablation_input_pre_hook",
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "jaxtyping")
            },
            "scope": "Selected block input only; fixed layer, no direction-search reproduction",
            "interpretation": "Activation/intervention evidence; no poisoning verdict",
        }
        return self.fitted

    def evaluate(self, prompt: str, group: str) -> dict[str, Any]:
        if self.fitted is None or self.direction is None:
            raise ValueError("Activation engine must be fitted first")
        if prompt not in self.allowed.get(group, set()):
            raise ValueError("Prompt/group absent from independent evaluation manifest")
        baseline, before_logits, before_loss, length = self.forward(prompt)
        ablated, after_logits, after_loss, _ = self.forward(prompt, True)
        return {
            "stage": "evaluation",
            "experimental": True,
            "group": group,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": length,
            "layer": self.layer,
            "position": self.position,
            "baseline_activation": moments(baseline),
            "ablated_activation": moments(ablated),
            **distribution_metrics(before_logits, after_logits, self.refusal_ids),
            "baseline_prompt_cross_entropy": before_loss,
            "ablated_prompt_cross_entropy": after_loss,
            "prompt_cross_entropy_delta": after_loss - before_loss
            if before_loss is not None and after_loss is not None
            else None,
            "gaps": [] if length > 1 else ["Prompt has no next-token targets for cross entropy"],
            "utility_scope": "Prompt likelihood and next-token KL; task success not measured",
            "attribution": "Ephemeral selected-layer effect; malicious intent undetermined",
        }

    def records(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield self.fit(config)
        for pair in config["evaluation_pairs"]:
            for group in ("harmful", "harmless"):
                yield self.evaluate(pair[group], group)
        for prompt in config["utility_prompts"]:
            yield self.evaluate(prompt, "utility")
