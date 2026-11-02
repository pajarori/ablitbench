from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from pydantic import Field, model_validator

from ablitbench.models import Case, CaseOutcome, Finding, Model, digest
from ablitbench.store import Store


class Label(Model):
    target: str
    check: str
    positive: bool
    model: str | None = None
    detector: str | None = None
    case_id: str | None = None
    repeat: int | None = Field(default=None, ge=0)
    split: Literal["train", "calibration", "test"] | None = None
    evaluator: str | None = None
    measure: Literal["security", "case-failure", "attack-success", "clean-utility"] = "security"

    @model_validator(mode="after")
    def valid_scope(self) -> Label:
        if self.repeat is not None and self.case_id is None:
            raise ValueError("A repeat scope requires case_id")
        if self.measure == "case-failure" and (self.case_id is None or self.detector is not None):
            raise ValueError("case-failure requires case_id and cannot select a security detector")
        if self.measure in {"attack-success", "clean-utility"} and (
            self.case_id is None or self.evaluator is None or self.detector is not None
        ):
            raise ValueError("Outcome rates require case_id and an explicit evaluator")
        if self.measure in {"security", "case-failure"} and self.evaluator is not None:
            raise ValueError("Evaluator selection requires attack-success or clean-utility")
        return self


def load_cases(path: Path) -> Iterator[Case]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield Case.model_validate_json(line)


def validate_cases(path: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="ablib-dataset-") as directory:
        database = sqlite3.connect(Path(directory) / "cases.sqlite")
        try:
            database.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            count = 0
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for line in stream:
                    case = Case.model_validate_json(line)
                    try:
                        database.execute("INSERT INTO ids VALUES (?)", (case.id,))
                    except sqlite3.IntegrityError:
                        raise ValueError("Duplicate dataset case ID: " + case.id) from None
                    checksum.update(line)
                    count += 1
            if not count:
                raise ValueError("Dataset contains no cases")
            database.commit()
            return count, checksum.hexdigest()
        finally:
            database.close()


def prediction(store: Store, label: Label, index: sqlite3.Connection | None = None) -> bool | None:
    row = None
    if index is not None:
        target_query = "SELECT COUNT(*), MIN(target_id) FROM targets WHERE location=?"
        target_parameters: list[Any] = [label.target]
        if label.model is not None:
            target_query += " AND model=?"
            target_parameters.append(label.model)
        count, target_index = index.execute(target_query, target_parameters).fetchone()
        if count != 1:
            return None
        row = store.connection.execute(
            "SELECT key, state, result FROM jobs WHERE target=? AND check_id=?",
            (target_index, label.check),
        ).fetchone()
    else:
        matches = 0
        for target_key, payload in store.connection.execute(
            "SELECT key, value FROM meta WHERE key LIKE 'target:%'"
        ):
            target = json.loads(payload)["target"]
            if target["location"] != label.target or (
                label.model is not None and target.get("model") != label.model
            ):
                continue
            matches += 1
            if matches > 1:
                return None
            target_index = int(target_key.split(":")[1])
            row = store.connection.execute(
                "SELECT key, state, result FROM jobs WHERE target=? AND check_id=?",
                (target_index, label.check),
            ).fetchone()
    if row is None:
        return None
    job, state, _ = row
    if label.case_id is not None:
        count = 0
        failed = False
        incomplete = False
        successes = False
        outcome_query = "SELECT payload FROM outcomes WHERE job=? AND case_id=?"
        outcome_parameters: list[Any] = [job, label.case_id]
        if label.repeat is not None:
            outcome_query += " AND repeat=?"
            outcome_parameters.append(label.repeat)
        for (payload,) in store.connection.execute(outcome_query, outcome_parameters):
            outcome = CaseOutcome.model_validate_json(payload)
            if outcome.case_id != label.case_id or (
                label.repeat is not None and outcome.repeat != label.repeat
            ):
                continue
            count += 1
            failed |= outcome.status == "fail"
            if label.measure in {"attack-success", "clean-utility"}:
                evaluations = [e for e in outcome.evaluators if e.evaluator == label.evaluator]
                if len(evaluations) != 1 or evaluations[0].status == "inconclusive":
                    incomplete = True
                elif evaluations:
                    successes |= evaluations[0].status == (
                        "fail" if label.measure == "attack-success" else "pass"
                    )
            else:
                incomplete |= not outcome.complete or outcome.status == "inconclusive"
        config = json.loads(store.get("config") or "{}")
        expected = 1 if label.repeat is not None else config.get("repeats", 1)
        if count != expected or incomplete:
            return None
        if label.measure == "case-failure":
            return failed
        if label.measure in {"attack-success", "clean-utility"}:
            return successes
    elif state != "completed":
        return None
    if index is not None:
        conditions = ["job=?", "severity!='info'"]
        parameters: list[Any] = [job]
        for column, value in (
            ("detector", label.detector),
            ("case_id", label.case_id),
            ("repeat", label.repeat),
        ):
            if value is not None:
                conditions.append(column + "=?")
                parameters.append(value)
        found = index.execute(
            "SELECT 1 FROM findings WHERE " + " AND ".join(conditions) + " LIMIT 1", parameters
        ).fetchone()
        return found is not None
    for payload in store.finding_payloads(job):
        finding = Finding.model_validate_json(payload)
        if finding.severity == "info":
            continue
        if label.detector is not None and (finding.detector or finding.title) != label.detector:
            continue
        if label.case_id is not None and finding.case_id != label.case_id:
            continue
        if label.repeat is not None and finding.repeat != label.repeat:
            continue
        return True
    return False


def wilson(successes: int, trials: int, confidence: float = 0.95) -> list[float] | None:
    if not 0 < confidence < 1 or successes < 0 or trials < successes:
        raise ValueError("Invalid binomial interval inputs")
    if not trials:
        return None
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def rate(successes: int, failures: int, inconclusive: int, confidence: float) -> dict[str, Any]:
    completed = successes + failures
    population = completed + inconclusive
    return {
        "successes": successes,
        "failures": failures,
        "inconclusive": inconclusive,
        "population": population,
        "completed": completed,
        "coverage": completed / population if population else None,
        "rate_on_completed": successes / completed if completed else None,
        "lower_bound": successes / population if population else None,
        "upper_bound": (successes + inconclusive) / population if population else None,
        "wilson_on_completed": wilson(successes, completed, confidence),
    }


def evaluate(store: Store, path: Path, confidence: float = 0.95) -> dict[str, Any]:
    counts = {
        name: 0
        for name in (
            "tp",
            "fp",
            "tn",
            "fn",
            "inconclusive_positive",
            "inconclusive_negative",
            "population",
        )
    }
    rates = {"attack-success": [0, 0, 0], "clean-utility": [0, 0, 0]}
    groups: dict[str, Any] = {}
    dataset = json.loads(store.get("dataset_manifest") or "null")
    selected_split = dataset.get("selected_split") if dataset else None
    scoped = False
    excluded = 0
    labels_checksum = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="ablib-evaluation-") as directory:
        units = sqlite3.connect(Path(directory) / "units.sqlite")
        try:
            units.executescript("""
CREATE TABLE scopes (identity TEXT PRIMARY KEY);
CREATE TABLE targets (target_id INTEGER PRIMARY KEY, location TEXT, model TEXT);
CREATE INDEX target_scope ON targets(location, model);
CREATE TABLE findings (job TEXT, detector TEXT, case_id TEXT, repeat INTEGER, severity TEXT);
CREATE INDEX finding_scope ON findings(job, case_id, repeat, detector);
CREATE INDEX detector_scope ON findings(job, detector, case_id, repeat);
""")
            for key, payload in store.connection.execute(
                "SELECT key, value FROM meta WHERE key LIKE 'target:%'"
            ):
                target = json.loads(payload)["target"]
                units.execute(
                    "INSERT INTO targets VALUES (?, ?, ?)",
                    (int(key.split(":")[1]), target["location"], target.get("model")),
                )
            for job, payload in store.connection.execute(
                "SELECT job, payload FROM findings ORDER BY id"
            ):
                finding = Finding.model_validate_json(payload)
                units.execute(
                    "INSERT INTO findings VALUES (?, ?, ?, ?, ?)",
                    (
                        job,
                        finding.detector or finding.title,
                        finding.case_id,
                        finding.repeat,
                        finding.severity,
                    ),
                )
            with path.open("rb") as stream:
                for line in stream:
                    labels_checksum.update(line)
                    label = Label.model_validate_json(line)
                    if selected_split and label.split is not None and label.split != selected_split:
                        excluded += 1
                        continue
                    scoped |= (
                        any(
                            value is not None
                            for value in (label.model, label.detector, label.case_id, label.repeat)
                        )
                        or label.measure != "security"
                    )
                    repeats = (
                        range(json.loads(store.get("config") or "{}").get("repeats", 1))
                        if label.measure in rates and label.repeat is None
                        else [label.repeat]
                    )
                    for repeat in repeats:
                        trial = label.model_copy(update={"repeat": repeat})
                        try:
                            units.execute(
                                "INSERT INTO scopes VALUES (?)",
                                (digest(trial.model_dump(exclude={"positive"})),),
                            )
                        except sqlite3.IntegrityError:
                            raise ValueError("Duplicate evaluation unit") from None
                        counts["population"] += 1
                        positive = prediction(store, trial, units)
                        if positive is None:
                            key = (
                                "inconclusive_positive"
                                if label.positive
                                else "inconclusive_negative"
                            )
                        else:
                            key = (
                                ("tp" if positive else "fn")
                                if label.positive
                                else ("fp" if positive else "tn")
                            )
                        counts[key] += 1
                        if label.measure in rates:
                            position = 2 if positive is None else 0 if positive else 1
                            rates[label.measure][position] += 1
                            group_key = digest([label.target, label.model])
                            if group_key not in groups:
                                groups[group_key] = {
                                    "target": label.target,
                                    "model": label.model,
                                    "attack-success": [0, 0, 0],
                                    "clean-utility": [0, 0, 0],
                                }
                            groups[group_key][label.measure][position] += 1
        finally:
            units.close()
    labels_sha256 = labels_checksum.hexdigest()
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    positives = tp + fn + counts["inconclusive_positive"]
    completed = tp + fp + tn + fn
    policy = {
        "version": "outcome-evaluation-v2",
        "confidence": confidence,
        "info_findings_positive": False,
        "repeat_rates": "Rates use case/repeat; legacy labels retain aggregate scope",
        "inconclusive": "Retained in population and bounds, never inferred negative",
    }
    return {
        "unit": "target/check with optional detector/case/repeat" if scoped else "target/check",
        "semantics": {
            "security": "Matching non-info finding; legacy detector IDs fall back to exact title",
            "case-failure": "Failed case outcome among selected complete repeats",
            "attack-success": "Selected explicit evaluator failed; no refusal keyword scoring",
            "clean-utility": "Selected explicit task evaluator passed",
        },
        "policy": policy,
        "policy_identity": digest(policy),
        "labels_sha256": labels_sha256,
        "excluded_other_split": excluded,
        "uncertainty": {
            "method": "Wilson score",
            "confidence": confidence,
            "scope": "Completed trials; repeated/grouped observations may be correlated",
        },
        **counts,
        "completed": completed,
        "coverage": completed / counts["population"] if counts["population"] else None,
        "precision_on_completed": tp / (tp + fp) if tp + fp else None,
        "recall_on_completed": tp / (tp + fn) if tp + fn else None,
        "fpr_on_completed": fp / (fp + tn) if fp + tn else None,
        "precision_wilson": wilson(tp, tp + fp, confidence),
        "recall_wilson": wilson(tp, tp + fn, confidence),
        "fpr_wilson": wilson(fp, fp + tn, confidence),
        "recall_lower_bound": tp / positives if positives else None,
        "recall_upper_bound": (tp + counts["inconclusive_positive"]) / positives
        if positives
        else None,
        "by_target": [
            {
                "target": group["target"],
                "model": group["model"],
                "attack_success": rate(
                    group["attack-success"][0],
                    group["attack-success"][1],
                    group["attack-success"][2],
                    confidence,
                ),
                "clean_task_utility": rate(
                    group["clean-utility"][0],
                    group["clean-utility"][1],
                    group["clean-utility"][2],
                    confidence,
                ),
            }
            for group in groups.values()
        ],
        "attack_success": rate(
            rates["attack-success"][0],
            rates["attack-success"][1],
            rates["attack-success"][2],
            confidence,
        ),
        "clean_task_utility": rate(
            rates["clean-utility"][0],
            rates["clean-utility"][1],
            rates["clean-utility"][2],
            confidence,
        ),
    }


def compare_runs(runs: dict[str, Path]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Comparison requires named runs")
    expected = None
    results = {}
    for name, directory in runs.items():
        connection = sqlite3.connect(
            f"{directory.joinpath('run.sqlite').resolve().as_uri()}?mode=ro", uri=True
        )
        try:
            payload = connection.execute("SELECT value FROM meta WHERE key='result'").fetchone()
            config_row = connection.execute("SELECT value FROM meta WHERE key='config'").fetchone()
            if payload is None or config_row is None:
                raise ValueError("Comparison run has no completed report: " + name)
            result = json.loads(payload[0])
            config = json.loads(config_row[0])
            metrics = result["summary"].get("evaluation")
            dataset_identity = config.get("dataset_identity") or config.get("dataset_sha256")
            if not dataset_identity or metrics is None:
                raise ValueError("Comparison requires an identified dataset and labeled evaluation")
            provenance = dict(
                connection.execute(
                    "SELECT key, value FROM meta WHERE key IN ('implementation', 'registry')"
                )
            )
            if set(provenance) != {"implementation", "registry"}:
                raise ValueError("Comparison requires implementation and registry identities")
            policy = digest(
                {
                    "evaluation": metrics["policy_identity"],
                    "labels": metrics["labels_sha256"],
                    "checks": config.get("checks"),
                    "repeats": config.get("repeats"),
                    "judge": config.get("judge"),
                    "execute_tools": config.get("execute_tools"),
                    "max_output_tokens": config.get("max_output_tokens"),
                    "runtime": config.get("runtime"),
                    "seed": config.get("seed"),
                    "mode": config.get("mode"),
                    "inference": config.get("inference"),
                    "implementation": provenance["implementation"],
                    "registry": provenance["registry"],
                }
            )
            identity = (dataset_identity, policy)
            if expected is not None and identity != expected:
                raise ValueError("Run dataset or evaluation policy identity mismatch")
            expected = identity
            results[name] = {
                "run_id": result["run_id"],
                "status": result["status"],
                "metrics": metrics,
                "resources": result["summary"].get("resources"),
            }
        finally:
            connection.close()
    return {
        "dataset_identity": expected[0] if expected else None,
        "policy_identity": expected[1] if expected else None,
        "runs": results,
        "interpretation": "Observed comparison; interpretation depends on supplied labels",
    }


def resource_summary(store: Store) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for role, backend, elapsed, status, payload in store.connection.execute(
        "SELECT role, backend, elapsed, status, usage FROM measurements ORDER BY id"
    ):
        key = role + "/" + backend
        if key not in groups:
            groups[key] = {
                "generation_calls": 0,
                "completed_calls": 0,
                "inconclusive_calls": 0,
                "elapsed_seconds": 0.0,
                "reported_tokens": {
                    name: {"sum": None, "known_calls": 0, "unknown_calls": 0}
                    for name in ("input", "output", "total")
                },
            }
        group = groups[key]
        group["generation_calls"] += 1
        group["completed_calls" if status == "completed" else "inconclusive_calls"] += 1
        group["elapsed_seconds"] += elapsed
        usage = json.loads(payload) if payload is not None else {}
        for name, aliases in {
            "input": ("input_tokens", "prompt_tokens"),
            "output": ("output_tokens", "completion_tokens"),
            "total": ("total_tokens",),
        }.items():
            values = [usage[alias] for alias in aliases if alias in usage]
            known = len(values) == 1 and type(values[0]) is int and values[0] >= 0
            tokens = group["reported_tokens"][name]
            if known:
                tokens["sum"] = (tokens["sum"] or 0) + values[0]
                tokens["known_calls"] += 1
            else:
                tokens["unknown_calls"] += 1
    return {
        "run_elapsed_seconds": float(store.get("elapsed_seconds") or "0"),
        "groups": groups,
        "monetary_cost": None,
        "cost_reason": "No validated pricing supplied; no monetary estimate made",
        "usage_basis": "Reported token counts only; missing or invalid counts remain unknown",
        "request_count_basis": "Logical generations; no HTTP negotiation or delivery claim",
    }
