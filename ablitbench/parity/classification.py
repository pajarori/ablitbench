from __future__ import annotations

import ast
import csv
import hashlib
import html
import importlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import warnings
from collections import Counter
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads


def operations() -> dict[str, dict[str, Any]]:
    return {
        "adapter.discover": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "inspect",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["models"],
        },
        "adapter.train": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "train",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "dataset"],
            "optional_inputs": ["base"],
        },
        "adapter.merge": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "export",
            "capabilities": ["sandbox", "local-model"],
            "inputs": ["model", "base"],
        },
        "features.augment": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "train",
            "capabilities": ["sandbox", "numpy"],
            "inputs": ["metrics"],
        },
        "features.merge": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "inspect",
            "capabilities": ["sandbox"],
            "inputs": ["metrics", "additional"],
        },
        "classifier.merge-retrain": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "train",
            "capabilities": ["sandbox", "scikit-learn"],
            "inputs": ["metrics", "additional"],
        },
        "workbench.report": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "export",
            "capabilities": ["sandbox"],
            "inputs": ["metrics"],
        },
        "metrics.aggregate": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "inspect",
            "capabilities": ["sandbox", "reviewed-source", "numpy"],
            "inputs": ["source", "metrics"],
        },
        "classifier.calibrate": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "train",
            "capabilities": ["sandbox", "scikit-learn", "labeled-metrics"],
            "inputs": ["classifier", "metrics"],
        },
        "metrics.extract": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": True,
            "requires_tools": False,
            "modes": ["source"],
            "kind": "inspect",
            "capabilities": ["sandbox", "reviewed-source", "local-model"],
            "inputs": ["source", "model", "dataset"],
            "optional_inputs": ["base"],
        },
        "classifier.train": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "train",
            "capabilities": ["sandbox", "scikit-learn", "labeled-metrics"],
            "inputs": ["metrics"],
        },
        "classifier.predict": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "detect",
            "capabilities": ["sandbox", "metric-features"],
            "inputs": ["features", "classifier"],
        },
        "classifier.evaluate": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "evaluate",
            "capabilities": ["sandbox", "labeled-predictions"],
            "inputs": ["predictions", "labels"],
        },
        "threshold.detect": {
            "project": "LLM-Backdoor-Defense",
            "requires_inference": False,
            "requires_tools": False,
            "modes": ["api", "source"],
            "kind": "detect",
            "capabilities": ["sandbox", "reviewed-source", "pandas"],
            "inputs": ["source", "metrics"],
        },
    }


def configuration(operation: str, settings: dict[str, Any]) -> dict[str, Any]:
    if operation not in operations():
        raise ValueError("Unknown classification operation")
    if operation in {"adapter.train", "adapter.merge"}:
        from ablitbench.parity.attacks import configuration as training_configuration

        return training_configuration(
            "supervised.train" if operation == "adapter.train" else "adapter.merge", settings
        )
    allowed = {
        "adapter.discover": {"recursive"},
        "features.augment": {
            "features",
            "copies",
            "noise_fraction",
            "seed",
            "group",
            "split",
            "training_splits",
            "id",
        },
        "features.merge": {"group", "overlap", "schema"},
        "classifier.merge-retrain": {"group", "overlap", "schema", "training"},
        "workbench.report": {"features", "group"},
        "metrics.aggregate": {"adapter_name", "corpus", "depth_pattern"},
        "classifier.calibrate": {
            "group",
            "label",
            "split",
            "calibration_splits",
            "test_splits",
            "max_iter",
            "bins",
            "seed",
        },
        "metrics.extract": {"instrumentation", "batch_size", "seed", "device", "top_k"},
        "classifier.train": {
            "features",
            "group",
            "label",
            "split",
            "train_splits",
            "test_splits",
            "algorithms",
            "parameters",
            "seed",
            "cv_folds",
        },
        "classifier.predict": {"id", "group"},
        "classifier.evaluate": {"id", "label"},
        "threshold.detect": {"layer"},
    }[operation]
    if set(settings) - allowed:
        raise ValueError("Unknown classification settings")
    result = dict(settings)
    if operation == "classifier.calibrate":
        if (
            set(settings) != allowed
            or any(
                type(settings[key]) is not int or settings[key] <= 0 for key in ("max_iter", "bins")
            )
            or type(settings["seed"]) is not int
        ):
            raise ValueError(
                "Explicit group/split names, calibration budget, bins and seed required"
            )
        if (
            not settings["calibration_splits"]
            or not settings["test_splits"]
            or (set(settings["calibration_splits"]) & set(settings["test_splits"]))
        ):
            raise ValueError("Separate calibration and evaluation splits required")
    elif operation == "metrics.extract":
        if set(settings) != allowed:
            raise ValueError(
                "Explicit instrumentation flags, batches, seed, device and top_k required"
            )
        if settings["device"] not in {"cpu", "cuda"}:
            raise ValueError("Unknown instrumentation device")
        if type(settings["seed"]) is not int:
            raise ValueError("Integer instrumentation seed required")
        if any(
            type(settings[key]) is not int or settings[key] <= 0 for key in ("batch_size", "top_k")
        ):
            raise ValueError("Positive explicit instrumentation batch and report sizes required")
        flags = settings["instrumentation"]
        if not isinstance(flags, dict) or not flags.get("target_layers"):
            raise ValueError("Explicit hook module names are required")
        if flags.get("permutation_test") and (
            not flags.get("store_batch_activations")
            or type(flags.get("permutation_n")) is not int
            or flags["permutation_n"] <= 0
        ):
            raise ValueError("Permutation test requires batch observations and explicit count")
        if flags.get("track_gradient_norm"):
            raise ValueError(
                "Gradient metrics require a supervised backward instrumentation workflow"
            )
    elif operation == "classifier.train":
        required = allowed - {"cv_folds"}
        if required - result.keys():
            raise ValueError("Training requires explicit features, splits, algorithms and budgets")
        features = result["features"]
        if (
            not isinstance(features, list)
            or not features
            or any(not isinstance(item, str) or not item for item in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError("Distinct nonempty feature names are required")
        if set(features) & {result["group"], result["label"], result["split"]}:
            raise ValueError("Group, labels and split columns cannot be classifier features")
        for key in ("train_splits", "test_splits"):
            if (
                not isinstance(result[key], list)
                or not result[key]
                or any(not isinstance(value, str) or not value for value in result[key])
                or len(set(result[key])) != len(result[key])
            ):
                raise ValueError("Independent explicit training and test split lists are required")
        if set(result["train_splits"]) & set(result["test_splits"]):
            raise ValueError("Training/test split overlap")
        algorithms = result["algorithms"]
        supported = {"logistic", "random-forest", "gradient-boosting"}
        if not algorithms or set(algorithms) - supported or len(set(algorithms)) != len(algorithms):
            raise ValueError("Distinct supported classifier algorithms are required")
        if set(result["parameters"]) != set(algorithms):
            raise ValueError("Parameters must match the selected classifier algorithms")
        for name in algorithms:
            parameter = "max_iter" if name == "logistic" else "n_estimators"
            value = result["parameters"][name].get(parameter)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("An explicit positive training budget is required for " + name)
        if isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
            raise ValueError("An integer training seed is required")
        folds = result.get("cv_folds")
        if folds is not None and (
            isinstance(folds, bool) or not isinstance(folds, int) or folds < 2
        ):
            raise ValueError("Group cross-validation requires at least two folds")
    if operation == "threshold.detect":
        layer = result.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("An explicit nonnegative detection layer is required")
    return result


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or len(names) != len(set(names)):
            raise ValueError("CSV requires distinct column names")
        result = list(reader)
    if not result:
        raise ValueError("Metric input is empty")
    if any(None in row or any(value is None for value in row.values()) for row in result):
        raise ValueError("CSV row width differs from its header")
    return result


def matrix(records: list[dict[str, Any]], features: list[str]) -> list[list[float]]:
    values = [[float(row[name]) for name in features] for row in records]
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError("Classifier features must be finite")
    return values


def metrics(
    labels: Sequence[int], predictions: Sequence[int | None], scores: Sequence[float] | None = None
) -> dict[str, Any]:
    if len(labels) != len(predictions) or any(value not in {0, 1} for value in labels):
        raise ValueError("Binary labels and aligned predictions are required")
    if any(value not in {0, 1, None} for value in predictions):
        raise ValueError("Predictions must be binary or inconclusive")
    pairs = [
        (label, prediction)
        for label, prediction in zip(labels, predictions, strict=True)
        if prediction is not None
    ]
    tp = sum(label == prediction == 1 for label, prediction in pairs)
    tn = sum(label == prediction == 0 for label, prediction in pairs)
    fp = sum(label == 0 and prediction == 1 for label, prediction in pairs)
    fn = sum(label == 1 and prediction == 0 for label, prediction in pairs)
    result: dict[str, Any] = {
        "total": len(labels),
        "observed": len(pairs),
        "inconclusive": len(labels) - len(pairs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_observed": (tp + tn) / len(pairs) if pairs else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall_observed": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }
    positive = sum(labels)
    unresolved_positive = sum(
        label == 1 and predicted is None
        for label, predicted in zip(labels, predictions, strict=True)
    )
    result["recall_bounds"] = (
        [tp / positive, (tp + unresolved_positive) / positive] if positive else None
    )
    if scores is not None:
        if len(scores) != len(labels) or any(not math.isfinite(value) for value in scores):
            raise ValueError("Finite aligned classifier scores are required")
        calculator = importlib.import_module("sklearn.metrics")
        result["roc_auc"] = (
            float(calculator.roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        )
    return result


def tree_state(tree: Any) -> dict[str, Any]:
    return {
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0].tolist(),
    }


def tree_value(state: dict[str, Any], values: list[float]) -> list[float]:
    node = 0
    visited = set()
    while state["left"][node] != -1:
        if node in visited:
            raise ValueError("Classifier tree contains a cycle")
        visited.add(node)
        feature = state["feature"][node]
        node = (
            state["left"][node]
            if values[feature] <= state["threshold"][node]
            else state["right"][node]
        )
    return list(state["value"][node])


def probabilities(artifact: dict[str, Any], values: list[list[float]]) -> list[float]:
    result = []
    for row in values:
        transformed = [
            (value - mean) / scale
            for value, mean, scale in zip(row, artifact["mean"], artifact["scale"], strict=True)
        ]
        if artifact["algorithm"] != "logistic":
            transformed = [struct.unpack("f", struct.pack("f", value))[0] for value in transformed]
        if artifact["algorithm"] == "logistic":
            logit = artifact["intercept"] + sum(
                value * weight
                for value, weight in zip(transformed, artifact["coefficients"], strict=True)
            )
            probability = (
                1 / (1 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1 + math.exp(logit))
            )
        elif artifact["algorithm"] == "random-forest":
            leaves = [tree_value(tree, transformed) for tree in artifact["trees"]]
            probability = sum(leaf[1] / sum(leaf) for leaf in leaves) / len(leaves)
        elif artifact["algorithm"] == "gradient-boosting":
            logit = artifact["initial_log_odds"] + artifact["learning_rate"] * sum(
                tree_value(tree, transformed)[0] for tree in artifact["trees"]
            )
            probability = (
                1 / (1 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1 + math.exp(logit))
            )
        else:
            raise ValueError("Unknown serialized classifier algorithm")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid classifier probability")
        calibration = artifact.get("calibration")
        if calibration is not None:
            logit = calibration["coefficient"] * probability + calibration["intercept"]
            probability = (
                1 / (1 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1 + math.exp(logit))
            )
        result.append(probability)
    return result


def fitted_state(model: Any, algorithm: str, features: list[str]) -> dict[str, Any]:
    scaler, estimator = model.steps[0][1], model.steps[1][1]
    result: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": algorithm,
        "features": features,
        "classes": estimator.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "parameters": estimator.get_params(),
        "score_scope": "Fitted classifier score; not calibrated probability of malicious intent",
        "decision_threshold": 0.5,
        "decision_rule": "Class 1 only when score > 0.5; ties resolve to class 0",
        "positive_class_semantics": (
            "Operator-supplied binary training label 1; no implicit poisoning meaning"
        ),
    }
    if result["classes"] != [0, 1]:
        raise ValueError("Both clean and positive classes are required for training")
    if algorithm == "logistic":
        result.update(
            coefficients=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0])
        )
    elif algorithm == "random-forest":
        result["trees"] = [tree_state(tree.tree_) for tree in estimator.estimators_]
    else:
        prior = estimator.init_.class_prior_
        result.update(
            trees=[tree_state(row[0].tree_) for row in estimator.estimators_],
            initial_log_odds=float(math.log(prior[1] / prior[0])),
            learning_rate=float(estimator.learning_rate),
        )
    return result


def train(settings: dict[str, Any], path: Path, output: Path) -> dict[str, Any]:
    numpy = importlib.import_module("numpy")
    linear = importlib.import_module("sklearn.linear_model")
    ensemble = importlib.import_module("sklearn.ensemble")
    preprocessing = importlib.import_module("sklearn.preprocessing")
    pipeline = importlib.import_module("sklearn.pipeline")
    selection = importlib.import_module("sklearn.model_selection")
    records = rows(path)
    fit = [row for row in records if row[settings["split"]] in settings["train_splits"]]
    test = [row for row in records if row[settings["split"]] in settings["test_splits"]]
    if not fit or not test:
        raise ValueError("Both training and held-out metric rows are required")
    group = settings["group"]
    if {row[group] for row in fit} & {row[group] for row in test}:
        raise ValueError("Checkpoint group appears in both training and held-out sets")
    features = settings["features"]
    x_fit, x_test = numpy.asarray(matrix(fit, features)), numpy.asarray(matrix(test, features))
    y_fit = numpy.asarray([int(row[settings["label"]]) for row in fit])
    y_test = [int(row[settings["label"]]) for row in test]
    if set(y_fit.tolist()) != {0, 1} or set(y_test) - {0, 1}:
        raise ValueError("Training requires both binary classes and binary held-out labels")
    for key, grouped in (("fit", fit), ("test", test)):
        grouped_labels: dict[str, set[str]] = {}
        for row in grouped:
            grouped_labels.setdefault(row[group], set()).add(row[settings["label"]])
        if any(len(labels) != 1 for labels in grouped_labels.values()):
            raise ValueError(key + " checkpoint has conflicting labels")
    factories = {
        "logistic": linear.LogisticRegression,
        "random-forest": ensemble.RandomForestClassifier,
        "gradient-boosting": ensemble.GradientBoostingClassifier,
    }
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in settings["algorithms"]:
        parameters = dict(settings["parameters"][name])
        parameters["random_state"] = settings["seed"]
        if name in {"logistic", "random-forest"}:
            parameters.setdefault("class_weight", "balanced")
        model = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", factories[name](**parameters)),
            ]
        )
        cv = []
        if settings.get("cv_folds") is not None:
            groups = numpy.asarray([row[group] for row in fit])
            splitter = selection.StratifiedGroupKFold(
                n_splits=settings["cv_folds"], shuffle=True, random_state=settings["seed"]
            )
            cloning = importlib.import_module("sklearn.base")
            for train_indices, validation_indices in splitter.split(x_fit, y_fit, groups):
                fold = cloning.clone(model)
                with warnings.catch_warnings(record=True) as fold_warnings:
                    warnings.simplefilter("always")
                    fold.fit(x_fit[train_indices], y_fit[train_indices])
                cv.append(
                    {
                        "metrics": metrics(
                            y_fit[validation_indices].tolist(),
                            fold.predict(x_fit[validation_indices]).tolist(),
                        ),
                        "warnings": [str(item.message) for item in fold_warnings],
                        "complete": not any(
                            item.category.__name__ == "ConvergenceWarning" for item in fold_warnings
                        ),
                    }
                )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x_fit, y_fit)
        artifact = fitted_state(model, name, features)
        predicted_scores = probabilities(artifact, x_test.tolist())
        reference = model.predict_proba(x_test)[:, 1].tolist()
        if not numpy.allclose(predicted_scores, reference, rtol=1e-6, atol=1e-8):
            raise ValueError("Safe classifier export does not match fitted estimator")
        predictions = [int(value > 0.5) for value in predicted_scores]
        if not numpy.array_equal(predictions, model.predict(x_test)):
            raise ValueError("Safe classifier decisions differ from the fitted estimator")
        checkpoint: list[dict[str, Any]] = []
        for identity in sorted({row[group] for row in test}):
            indices = [index for index, row in enumerate(test) if row[group] == identity]
            votes = Counter(predictions[index] for index in indices)
            prediction = None if votes[0] == votes[1] else int(votes[1] > votes[0])
            checkpoint.append(
                {"group": identity, "label": y_test[indices[0]], "prediction": prediction}
            )
        artifact["training"] = {
            "seed": settings["seed"],
            "groups": sorted({row[group] for row in fit}),
            "heldout_groups": sorted({row[group] for row in test}),
            "rows": len(fit),
            "label_column": settings["label"],
            "group_column": settings["group"],
            "split_column": settings["split"],
            "warnings": [str(item.message) for item in caught],
        }
        destination = output / (name + ".json")
        destination.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
        results[name] = {
            "artifact": destination.name,
            "row_metrics": metrics(y_test, predictions, predicted_scores),
            "checkpoint_metrics": metrics(
                [row["label"] for row in checkpoint], [row["prediction"] for row in checkpoint]
            ),
            "checkpoints": checkpoint,
            "cross_validation": cv,
            "warnings": artifact["training"]["warnings"],
            "complete": not any(item.category.__name__ == "ConvergenceWarning" for item in caught)
            and all(fold["complete"] for fold in cv),
            "export_reference_parity": True,
        }
    return {
        "status": "completed"
        if all(item["complete"] for item in results.values())
        else "unsupported",
        "models": results,
        "fit_rows": len(fit),
        "test_rows": len(test),
        "excluded_rows": len(records) - len(fit) - len(test),
        "fit_groups": len({row[group] for row in fit}),
        "test_groups": len({row[group] for row in test}),
        "scope": (
            "Independent grouped meta-classifier; supplied labels are training/evaluation "
            "inputs, not detector features"
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
    if operation in {"adapter.train", "adapter.merge"}:
        from ablitbench.parity.attacks import execute as training_execute

        return training_execute(
            "supervised.train" if operation == "adapter.train" else "adapter.merge",
            settings,
            inputs,
            output,
            broker,
        )
    if operation == "adapter.discover":
        return discover_adapters(settings, inputs["models"], output)
    if operation == "features.augment":
        return augment_features(settings, inputs["metrics"], output)
    if operation in {"features.merge", "classifier.merge-retrain"}:
        return merge_features(operation, settings, inputs, output)
    if operation == "workbench.report":
        return workbench(settings, inputs["metrics"], output)
    if operation == "metrics.aggregate":
        return aggregate(settings, inputs, output)
    if operation == "classifier.calibrate":
        return calibrate(settings, inputs, output)
    if operation == "metrics.extract":
        return extract_metrics(settings, inputs, output)
    if operation == "classifier.train":
        return train(settings, inputs["metrics"], output)
    if operation == "classifier.predict":
        artifact = loads(inputs["classifier"].read_text(encoding="utf-8"))
        records = rows(inputs["features"])
        forbidden = {"label", "gt-label", "gt_label", "ground_truth", "positive"}
        forbidden.add(artifact.get("training", {}).get("label_column", "label"))
        if forbidden & set(records[0]):
            raise ValueError("Prediction inputs must not contain ground-truth labels")
        permitted = set(artifact["features"]) | {
            settings.get("id", "id"),
            settings.get("group", "adapter_name"),
        }
        if set(records[0]) - permitted:
            raise ValueError("Prediction inputs must not contain ground-truth labels")
        scores = probabilities(artifact, matrix(records, artifact["features"]))
        predictions = [
            {
                "id": row.get(settings.get("id", "id"), str(index)),
                "score": value,
                "prediction": int(value > 0.5),
            }
            for index, (row, value) in enumerate(zip(records, scores, strict=True))
        ]
        output.mkdir(parents=True, exist_ok=True)
        (output / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
        return {
            "status": "completed",
            "rows": len(predictions),
            "artifact": "predictions.json",
            "scope": artifact["score_scope"],
        }
    if operation == "classifier.evaluate":
        saved_predictions = loads(inputs["predictions"].read_text(encoding="utf-8"))
        labels = rows(inputs["labels"])
        key = settings.get("id", "id")
        expected = {row[key]: int(row[settings.get("label", "label")]) for row in labels}
        if len(expected) != len(labels):
            raise ValueError("Evaluation labels contain duplicate identities")
        if not isinstance(saved_predictions, list):
            raise ValueError("Canonical prediction artifact must be a JSON list")
        mapping = {row["id"]: row["prediction"] for row in saved_predictions}
        if len(mapping) != len(saved_predictions) or set(mapping) - expected.keys():
            raise ValueError("Predictions contain duplicate or unlabeled identities")
        return {
            "status": "completed",
            "metrics": metrics(
                list(expected.values()), [mapping.get(identity) for identity in expected]
            ),
        }
    from ablitbench.parity.sources import verified_source

    source = verified_source(inputs["source"], "LLM-Backdoor-Defense")
    specification = importlib.util.spec_from_file_location(
        "ablitbench_tanto_threshold", source / "src/lora_backdoor_detector.py"
    )
    if specification is None or specification.loader is None:
        raise ValueError("Reviewed detector module cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    record = module.detect(str(inputs["metrics"]), layer=settings["layer"])
    expected_metrics = 5 if record.get("model_type") == "lora" else 3
    gaps = []
    if record.get("n_metrics", 0) < expected_metrics:
        gaps.append("Some upstream threshold metrics were unobserved or unavailable")

    def finite(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value

    record = finite(record)
    return {
        "status": "unsupported"
        if gaps or record.get("error") or record.get("verdict") == "INCONCLUSIVE"
        else "completed",
        "result": record,
        "gaps": gaps,
        "coverage": {"metrics": expected_metrics, "observed": record.get("n_metrics", 0)},
        "findings": [
            {
                "title": "TANTO activation metrics matched upstream poisoning indicators",
                "severity": "low",
                "strength": "indicator",
                "attribution": "undetermined",
                "detail": (
                    "The reviewed threshold detector flagged "
                    + str(record.get("n_flags"))
                    + " of "
                    + str(record.get("n_metrics"))
                    + " metrics. These architecture-dependent indicators do not establish "
                    "poisoning; reported confidence is a flag fraction."
                ),
            }
        ]
        if record.get("verdict") == "POISONED" and not record.get("error")
        else [],
        "scope": (
            "Original operator-provided threshold indicators; confidence is flag fraction, "
            "not calibrated poisoning probability"
        ),
    }


def extract_metrics(
    settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    from ablitbench.parity.sources import verified_source
    from ablitbench.worker import Inference

    source = verified_source(inputs["source"], "LLM-Backdoor-Defense")
    path = source / "app/instrumenter.py"
    name = "ablitbench_reviewed_tanto_instrumenter"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValueError("Reviewed instrumenter cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
        inference = Inference(
            settings["device"], inputs["model"], inputs.get("base", Path("/base"))
        )
        inference.torch.manual_seed(settings["seed"])
        document = loads(inputs["dataset"].read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) - {"clean", "triggered"}:
            raise ValueError("Instrumentation requires clean and optional triggered text corpora")
        if "clean" not in document:
            raise ValueError("Clean instrumentation control is required")
        config = module.InstrumentConfig(**settings["instrumentation"])
        names = {name for name, _ in inference.model.named_modules()}
        if set(config.target_layers) - names:
            raise ValueError("Requested instrumentation modules are absent")
        output.mkdir(parents=True, exist_ok=True)
        snapshots: dict[str, Any] = {}
        instruments: list[Any] = []
        artifacts, gaps = [], []
        try:
            for corpus, texts in document.items():
                if not isinstance(texts, list) or not texts:
                    raise ValueError("Every requested instrumentation corpus must be nonempty")
                options = dict(settings["instrumentation"])
                if instruments and config.spectral_analysis:
                    options["spectral_analysis"] = False
                instrument = module.UniversalInstrumenter(
                    inference.model, module.InstrumentConfig(**options)
                )
                if instruments and config.spectral_analysis:
                    instrument.spectral_stats = instruments[0].spectral_stats
                    for name, observations in instrument.spectral_stats.items():
                        instrument.counters.setdefault(name, module._make_counter()).update(
                            observations
                        )
                instruments.append(instrument)
                try:
                    for start in range(0, len(texts), settings["batch_size"]):
                        batch = texts[start : start + settings["batch_size"]]
                        if any(not isinstance(text, str) or not text for text in batch):
                            raise ValueError("Instrumentation corpus entries must be nonempty text")
                        lengths = [len(inference.tokenizer(text)["input_ids"]) for text in batch]
                        if len(set(lengths)) != 1:
                            raise ValueError(
                                "Upstream runner lacks attention masks; use equal-length batches"
                            )
                        tokens = inference.tokenizer(batch, return_tensors="pt", truncation=False)
                        context = getattr(inference.model.config, "max_position_embeddings", None)
                        if type(context) is int and tokens["input_ids"].shape[1] > context:
                            raise ValueError("Instrumentation input exceeds model context")
                        with redirect_stdout(sys.stderr):
                            instrument.run_corpus([tokens["input_ids"]])
                    snapshots[corpus] = instrument.snapshot()
                    for requested in config.target_layers:
                        if not snapshots[corpus].get(requested, {}).get("calls", 0):
                            gaps.append(
                                corpus + ": requested module had no observed calls: " + requested
                            )
                    if not any(row.get("calls", 0) for row in snapshots[corpus].values()):
                        raise ValueError(
                            "Requested instrumentation produced no observed module calls"
                        )
                    with redirect_stdout(sys.stderr):
                        instrument.export_to_csv(str(output / (corpus + ".csv")))
                        instrument.export_html_report(str(output / (corpus + ".html")))
                        if config.spectral_analysis:
                            instrument.export_spectral_report(
                                str(output / (corpus + "-spectral.json"))
                            )
                    artifacts.extend([corpus + ".csv", corpus + ".html"])
                    if config.spectral_analysis:
                        artifacts.append(corpus + "-spectral.json")
                finally:
                    instrument.remove_hooks()
            if "triggered" in snapshots:
                instrument = instruments[-1]
                difference = instrument.diff_snapshots(
                    snapshots["clean"], snapshots["triggered"], top_k=settings["top_k"]
                )
                (output / "difference.json").write_text(json.dumps(difference), encoding="utf-8")
                artifacts.append("difference.json")
                if config.permutation_test:
                    significance = instrument.permutation_test_diff(
                        snapshots["clean"],
                        snapshots["triggered"],
                        n_perms=config.permutation_n,
                        top_k=settings["top_k"],
                    )
                    if any(row.get("data_insufficient") for row in significance):
                        gaps.append("Permutation test has insufficient independent corpus batches")
                    (output / "significance.json").write_text(
                        json.dumps(significance), encoding="utf-8"
                    )
                    artifacts.append("significance.json")
            return {
                "status": "unsupported" if gaps else "completed",
                "artifacts": artifacts,
                "corpus_rows": {name: len(rows) for name, rows in document.items()},
                "observed_modules": {
                    name: sum(bool(row.get("calls", 0)) for row in rows.values())
                    for name, rows in snapshots.items()
                },
                "spectral_modules": len(instruments[0].spectral_stats)
                if config.spectral_analysis
                else 0,
                "spectral_computations": int(config.spectral_analysis),
                "gaps": gaps,
                "scope": (
                    "Original reviewed architecture-neutral instrumentation; "
                    "indicators are not poisoning proof"
                ),
            }
        finally:
            for instrument in instruments:
                instrument.remove_hooks()
    finally:
        sys.modules.pop(name, None)


def reliability(labels: list[int], scores: list[float], bins: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index in range(bins):
        selected = [
            position
            for position, score in enumerate(scores)
            if min(int(score * bins), bins - 1) == index
        ]
        records.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(selected),
                "mean_score": sum(scores[i] for i in selected) / len(selected)
                if selected
                else None,
                "positive_fraction": sum(labels[i] for i in selected) / len(selected)
                if selected
                else None,
            }
        )
    return {
        "bins": records,
        "brier": sum((score - label) ** 2 for score, label in zip(scores, labels, strict=True))
        / len(labels),
        "expected_calibration_error": sum(
            row["count"] / len(labels) * abs(row["mean_score"] - row["positive_fraction"])
            for row in records
            if row["count"]
        ),
    }


def calibrate(settings: dict[str, Any], inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    numpy = importlib.import_module("numpy")
    classifier = importlib.import_module("sklearn.linear_model").LogisticRegression
    artifact = loads(inputs["classifier"].read_text(encoding="utf-8"))
    if artifact.get("calibration") is not None:
        raise ValueError("Already calibrated classifier cannot be silently recalibrated")
    records = rows(inputs["metrics"])
    fit = [row for row in records if row[settings["split"]] in settings["calibration_splits"]]
    test = [row for row in records if row[settings["split"]] in settings["test_splits"]]
    group = settings["group"]
    fit_groups, test_groups = {row[group] for row in fit}, {row[group] for row in test}
    if not fit or not test or fit_groups & test_groups:
        raise ValueError("Nonempty checkpoint-disjoint calibration/evaluation populations required")
    training = artifact.get("training", {})
    if "groups" not in training or "heldout_groups" not in training:
        raise ValueError("Classifier lacks auditable fit/heldout group identities")
    if (fit_groups | test_groups) & (set(training["groups"]) | set(training["heldout_groups"])):
        raise ValueError(
            "Calibration/evaluation checkpoint overlaps original classifier population"
        )
    y_fit = [int(row[settings["label"]]) for row in fit]
    y_test = [int(row[settings["label"]]) for row in test]
    if set(y_fit) != {0, 1} or set(y_test) - {0, 1}:
        raise ValueError("Binary calibration labels and both fitted classes required")
    for population in (fit, test):
        grouped: dict[str, set[str]] = {}
        for row in population:
            grouped.setdefault(row[group], set()).add(row[settings["label"]])
        if any(len(labels) != 1 for labels in grouped.values()):
            raise ValueError("Checkpoint calibration labels conflict")
    fit_scores = probabilities(artifact, matrix(fit, artifact["features"]))
    test_scores = probabilities(artifact, matrix(test, artifact["features"]))
    model = classifier(max_iter=settings["max_iter"], random_state=settings["seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(numpy.asarray(fit_scores).reshape(-1, 1), y_fit)
    artifact["calibration"] = {
        "method": "sigmoid-on-classifier-score",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "groups": sorted(fit_groups),
    }
    artifact["score_scope"] = (
        "Sigmoid-calibrated classifier probability on the supplied checkpoint population; "
        "not probability of malicious intent or a validated cross-domain guarantee"
    )
    calibrated = probabilities(artifact, matrix(test, artifact["features"]))
    if not numpy.allclose(
        calibrated, model.predict_proba(numpy.asarray(test_scores).reshape(-1, 1))[:, 1]
    ):
        raise ValueError("Safe calibrated export differs from fitted reference")
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibrated.json").write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    return {
        "status": "unsupported"
        if any(item.category.__name__ == "ConvergenceWarning" for item in caught)
        else "completed",
        "artifact": "calibrated.json",
        "calibration_groups": len(fit_groups),
        "test_groups": len(test_groups),
        "before": reliability(y_test, test_scores, settings["bins"]),
        "after": reliability(y_test, calibrated, settings["bins"]),
        "scope": artifact["score_scope"],
    }


def aggregate(settings: dict[str, Any], inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    from ablitbench.parity.sources import verified_source

    source = verified_source(inputs["source"], "LLM-Backdoor-Defense")
    path = source / "src/extract_features.py"
    tree = ast.parse(path.read_bytes())
    statements: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_checkpoint_level_features"
    ]
    if len(statements) != 1:
        raise ValueError("Reviewed checkpoint aggregate function is unavailable")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
    records = rows(inputs["metrics"])
    pattern = re.compile(settings["depth_pattern"])
    layers = []
    for row in records:
        match = pattern.search(row["layer_name"])
        if match is None:
            raise ValueError("Explicit architecture depth pattern did not match observed layer")
        try:
            depth = int(match[1])
        except ValueError as error:
            raise ValueError("Architecture depth capture is not an integer") from error

        def optional(name: str, record: dict[str, str] = row) -> float | None:
            value = record.get(name)
            if value in {None, "", "N/A"}:
                return None
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError("Nonfinite aggregate metric")
            return converted

        layers.append(
            {
                "layer_name": row["layer_name"],
                "layer_depth": depth,
                "corpus": settings["corpus"],
                "adapter_name": settings["adapter_name"],
                "kurtosis": optional("kurtosis_avg"),
                "skewness": optional("skewness_avg"),
                "l2_avg": optional("l2_avg"),
                "l1_avg": optional("l1_avg"),
                "sparsity_pct": optional("sparsity_pct"),
                "coact_var": optional("coact_variance_avg"),
                "max_abs": optional("max_abs_val"),
            }
        )
    if any(row["l2_avg"] is None or row["sparsity_pct"] is None for row in layers):
        raise ValueError("Checkpoint aggregation requires observed L2 and sparsity metrics")
    checkpoint = namespace["compute_checkpoint_level_features"](layers)
    for layer in layers:
        layer.update({"ckpt_" + key: value for key, value in checkpoint.items()})
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "features.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layers[0]))
        writer.writeheader()
        writer.writerows(layers)
    return {
        "status": "completed",
        "artifact": destination.name,
        "rows": len(layers),
        "checkpoint": checkpoint,
        "scope": (
            "Original checkpoint aggregate formulas on observed architecture-mapped "
            "metrics; no labels"
        ),
    }


def discover_adapters(settings: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Adapter discovery requires a readonly directory inventory")
    output.mkdir(parents=True, exist_ok=True)
    observed, unsupported = 0, 0
    with (output / "adapters.jsonl").open("w", encoding="utf-8") as stream:
        for parent, directories, names in os.walk(root, followlinks=False):
            if any(Path(parent, name).is_symlink() for name in directories):
                raise ValueError("Adapter discovery refuses directory symlinks")
            if not settings["recursive"]:
                directories[:] = []
            if "adapter_config.json" not in names:
                continue
            directory = Path(parent)
            if (directory / "adapter_config.json").is_symlink():
                raise ValueError("Adapter configuration symlink refused")
            config = loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Adapter metadata must be a mapping")
            tensors = []
            for name in names:
                if not name.endswith(".safetensors"):
                    continue
                path = directory / name
                if path.is_symlink():
                    raise ValueError("Adapter tensor symlink refused")
                with path.open("rb") as binary:
                    checksum = hashlib.file_digest(binary, "sha256").hexdigest()
                tensors.append({"file": name, "sha256": checksum, "bytes": path.stat().st_size})
            status = "observed" if tensors else "unsupported"
            observed += bool(tensors)
            unsupported += not bool(tensors)
            stream.write(
                json.dumps(
                    {
                        "id": directory.relative_to(root).as_posix(),
                        "status": status,
                        "base_model_reference": config.get("base_model_name_or_path"),
                        "peft_type": config.get("peft_type"),
                        "tensors": tensors,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "status": "unsupported" if unsupported else "completed",
        "observed": observed,
        "unsupported": unsupported,
        "artifact": "adapters.jsonl",
        "labels_inferred": False,
        "scope": (
            "Readonly adapter inventory; no path-derived ground truth "
            "or automatic remote model loading"
        ),
    }


def write_rows(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def augment_features(settings: dict[str, Any], path: Path, output: Path) -> dict[str, Any]:
    numpy = importlib.import_module("numpy")
    records = rows(path)
    fitting = [row for row in records if row[settings["split"]] in settings["training_splits"]]
    if not fitting:
        raise ValueError("No rows belong to the explicit training augmentation population")
    training_groups = {row[settings["group"]] for row in fitting}
    if training_groups & {
        row[settings["group"]]
        for row in records
        if row[settings["split"]] not in settings["training_splits"]
    }:
        raise ValueError("Augmentation checkpoint appears in both training and heldout splits")
    if settings["id"] is not None and len({row[settings["id"]] for row in records}) != len(records):
        raise ValueError("Original augmentation row identities are not unique")
    random = numpy.random.default_rng(settings["seed"])
    scales, gaps = {}, []
    for feature in settings["features"]:
        observed = [
            float(row[feature]) for row in fitting if row.get(feature) not in {None, "", "N/A"}
        ]
        if any(not math.isfinite(value) for value in observed):
            raise ValueError("Nonfinite augmentation feature")
        if len(observed) < 2 and settings["copies"]:
            gaps.append("Feature variance needs two finite training rows: " + feature)
            scales[feature] = 0.0
        else:
            scales[feature] = (
                float(numpy.std(observed, ddof=1)) * settings["noise_fraction"]
                if len(observed) > 1
                else 0.0
            )
    augmented: list[dict[str, Any]] = [{**row, "augmented": 0} for row in records]
    for copy in range(settings["copies"]):
        batch: list[dict[str, Any]] = [{**row, "augmented": copy + 1} for row in fitting]
        for feature, scale in scales.items():
            if not scale:
                continue
            noise = random.normal(0, scale, size=len(batch))
            for index, row in enumerate(batch):
                if row.get(feature) not in {None, "", "N/A"}:
                    row[feature] = float(row[feature]) + float(noise[index])
        if settings["id"] is not None:
            for row in batch:
                row[settings["id"]] = str(row[settings["id"]]) + "~aug" + str(copy + 1)
        augmented.extend(batch)
    if settings["id"] is not None and len({row[settings["id"]] for row in augmented}) != len(
        augmented
    ):
        raise ValueError("Synthetic augmentation identities collide with an existing row")
    write_rows(output / "augmented.csv", augmented)
    return {
        "status": "unsupported" if gaps else "completed",
        "artifact": "augmented.csv",
        "original_rows": len(records),
        "synthetic_rows": len(augmented) - len(records),
        "heldout_rows_unchanged": len(records) - len(fitting),
        "gaps": gaps,
        "scope": (
            "Training-only Gaussian noise scaled by sample standard deviation; "
            "checkpoint groups preserved"
        ),
    }


def merge_features(
    operation: str, settings: dict[str, Any], inputs: dict[str, Path], output: Path
) -> dict[str, Any]:
    original, additional = rows(inputs["metrics"]), rows(inputs["additional"])
    group = settings["group"]
    overlap = {row[group] for row in original} & {row[group] for row in additional}
    if overlap and settings["overlap"] == "reject":
        raise ValueError("Merge populations contain overlapping checkpoint groups")
    selected = [row for row in additional if row[group] not in overlap]
    left, right = set(original[0]), set(additional[0])
    if settings["schema"] == "strict" and left != right:
        raise ValueError("Merge feature schemas differ")
    columns = [*original[0], *sorted(right - left)]
    combined: list[dict[str, Any]] = [
        {column: row.get(column, "") for column in columns} for row in [*original, *selected]
    ]
    destination = output / "combined.csv"
    write_rows(destination, combined)
    result: dict[str, Any] = {
        "status": "completed",
        "artifact": "combined.csv",
        "rows": len(combined),
        "dropped_overlapping_rows": len(additional) - len(selected),
        "scope": (
            "Explicit checkpoint identity and schema reconciliation before classifier retraining"
        ),
    }
    if operation == "classifier.merge-retrain":
        fitted = train(settings["training"], destination, output / "classifier")
        for model in fitted["models"].values():
            model["artifact"] = "classifier/" + model["artifact"]
        result.update(
            status=fitted["status"], classifier=fitted, artifacts=["combined.csv", "classifier"]
        )
    return result


def workbench(settings: dict[str, Any], path: Path, output: Path) -> dict[str, Any]:
    records = rows(path)
    values = matrix(records, settings["features"])
    groups = sorted({row[settings["group"]] for row in records})
    header = "".join("<th>" + html.escape(column) + "</th>" for column in records[0])
    body = "".join(
        "<tr>"
        + "".join("<td>" + html.escape(str(value)) + "</td>" for value in row.values())
        + "</tr>"
        for row in records
    )
    summaries = []
    for index, feature in enumerate(settings["features"]):
        column = [row[index] for row in values]
        summaries.append(
            {
                "feature": feature,
                "observed": len(column),
                "mean": sum(column) / len(column),
                "min": min(column),
                "max": max(column),
            }
        )
    summary = html.escape(json.dumps(summaries, indent=2))
    document = (
        "<!doctype html><html lang=en><meta charset=utf-8><title>Adapter workbench</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:40px auto;padding:20px}"
        "table{border-collapse:collapse}td,th{padding:8px;text-align:left}"
        "tr:nth-child(even){background:#eee}pre{white-space:pre-wrap}</style>"
        "<h1>Adapter workbench</h1><p>Observed feature rows: "
        + str(len(records))
        + "; checkpoint groups: "
        + str(len(groups))
        + ". Statistical summaries do not establish poisoning.</p><pre>"
        + summary
        + "</pre><details><summary>Complete observed rows</summary><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></details></html>"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "workbench.html").write_text(document, encoding="utf-8")
    return {
        "status": "completed",
        "artifact": "workbench.html",
        "rows": len(records),
        "checkpoint_groups": len(groups),
        "scope": (
            "Standalone escaped workbench export for the CLI; no Streamlit server or safety verdict"
        ),
    }
