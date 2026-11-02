from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ablitbench.models import Case, Model
from ablitbench.store import Store


class Label(Model):
    target: str
    check: str
    positive: bool


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


def evaluate(store: Store, path: Path) -> dict[str, Any]:
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
    store.connection.execute(
        "CREATE TABLE IF NOT EXISTS evaluation_units "
        "(target TEXT, check_id TEXT, positive INTEGER, PRIMARY KEY(target, check_id))"
    )
    store.connection.execute("DELETE FROM evaluation_units")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            label = Label.model_validate_json(line)
            try:
                store.connection.execute(
                    "INSERT INTO evaluation_units VALUES (?, ?, ?)",
                    (label.target, label.check, label.positive),
                )
            except sqlite3.IntegrityError:
                raise ValueError("Duplicate evaluation unit") from None
            counts["population"] += 1
            row = None
            for target_key, payload in store.connection.execute(
                "SELECT key, value FROM meta WHERE key LIKE 'target:%'"
            ):
                if json.loads(payload)["target"]["location"] == label.target:
                    index = int(target_key.split(":")[1])
                    row = store.connection.execute(
                        "SELECT state, result FROM jobs WHERE target=? AND check_id=?",
                        (index, label.check),
                    ).fetchone()
                    break
            if row and row[0] == "completed":
                positive = json.loads(row[1])["finding_count"] > 0
                key = (
                    ("tp" if positive else "fn") if label.positive else ("fp" if positive else "tn")
                )
            else:
                key = "inconclusive_positive" if label.positive else "inconclusive_negative"
            counts[key] += 1
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    positives = tp + fn + counts["inconclusive_positive"]
    completed = tp + fp + tn + fn
    return {
        "unit": "target/check",
        **counts,
        "completed": completed,
        "coverage": completed / counts["population"] if counts["population"] else None,
        "precision_on_completed": tp / (tp + fp) if tp + fp else None,
        "recall_on_completed": tp / (tp + fn) if tp + fn else None,
        "fpr_on_completed": fp / (fp + tn) if fp + tn else None,
        "recall_lower_bound": tp / positives if positives else None,
        "recall_upper_bound": (tp + counts["inconclusive_positive"]) / positives
        if positives
        else None,
    }
