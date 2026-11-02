from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ablitbench.formats.json import loads as json_loads
from ablitbench.formats.yaml import load as yaml_load
from ablitbench.models import Case, Model, digest

Split = Literal["train", "calibration", "test"]


class DatasetArtifact(Model):
    path: Path
    sha256: str = Field(pattern="^[a-f0-9]{64}$")


class DatasetProvenance(Model):
    source: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str | None = None
    description: str | None = None


class Checkpoint(Model):
    id: str = Field(min_length=1)
    sha256: str = Field(pattern="^[a-f0-9]{64}$")
    aliases: list[str] = Field(default_factory=list)
    lineage: str = Field(min_length=1)
    control: Literal["clean", "abliterated", "backdoored", "unknown"] = "unknown"


class DatasetRecord(Model):
    case: Case
    split: Split
    group: str = Field(min_length=1)
    checkpoint: str | None = None
    lineage: str | None = None
    attack_family: str | None = None
    task: str = Field(min_length=1)


class DatasetManifest(Model):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    provenance: DatasetProvenance
    cases: DatasetArtifact
    labels: DatasetArtifact | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    split_isolation: list[
        Literal["case", "group", "checkpoint", "lineage", "attack_family", "task"]
    ] = Field(default=["case", "group", "checkpoint", "lineage"])

    @model_validator(mode="after")
    def mandatory_isolation(self) -> DatasetManifest:
        if not {"case", "group", "checkpoint", "lineage"}.issubset(self.split_isolation):
            raise ValueError("Case, group, checkpoint, and lineage split isolation is required")
        return self


class DatasetInfo(Model):
    path: Path
    cases: Path
    labels: Path | None = None
    split: Split | None = None
    count: int
    sha256: str
    identity: str
    metadata: dict[str, Any]
    manifest: DatasetManifest | None = None


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_path(root: Path, artifact: DatasetArtifact) -> Path:
    path = (root / artifact.path).resolve()
    if file_hash(path) != artifact.sha256:
        raise ValueError("Dataset artifact hash mismatch: " + str(artifact.path))
    return path


def dataset_records(path: Path) -> Iterator[DatasetRecord]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield DatasetRecord.model_validate_json(line)


def inspect_dataset(path: Path, split: Split = "test") -> DatasetInfo:
    path = path.resolve()
    if path.suffix.lower() == ".jsonl":
        from ablitbench.benchmark import validate_cases

        count, legacy_checksum = validate_cases(path)
        return DatasetInfo(
            path=path,
            cases=path,
            count=count,
            sha256=legacy_checksum,
            identity=legacy_checksum,
            metadata={"format": "case-jsonl", "sha256": legacy_checksum, "cases": count},
        )
    manifest_bytes = path.read_bytes()
    value = (
        json_loads(manifest_bytes) if path.suffix.lower() == ".json" else yaml_load(manifest_bytes)
    )
    manifest = DatasetManifest.model_validate(value)
    cases = artifact_path(path.parent, manifest.cases)
    labels = artifact_path(path.parent, manifest.labels) if manifest.labels else None
    checksum = hashlib.sha256()
    selected = 0
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="ablib-manifest-") as directory:
        database = sqlite3.connect(Path(directory) / "validation.sqlite")
        try:
            database.executescript("""
CREATE TABLE aliases (alias TEXT PRIMARY KEY, identity TEXT, lineage TEXT);
CREATE TABLE cases (id TEXT PRIMARY KEY, split TEXT);
CREATE TABLE groups (kind TEXT, identity TEXT, split TEXT, PRIMARY KEY(kind, identity));
""")
            for checkpoint in manifest.checkpoints:
                for alias in [checkpoint.id, *checkpoint.aliases]:
                    try:
                        database.execute(
                            "INSERT INTO aliases VALUES (?, ?, ?)",
                            (alias, checkpoint.sha256, checkpoint.lineage),
                        )
                    except sqlite3.IntegrityError:
                        raise ValueError(
                            "Duplicate or ambiguous checkpoint alias: " + alias
                        ) from None
            for record in dataset_records(cases):
                try:
                    database.execute(
                        "INSERT INTO cases VALUES (?, ?)", (record.case.id, record.split)
                    )
                except sqlite3.IntegrityError:
                    raise ValueError("Duplicate dataset case ID: " + record.case.id) from None
                checkpoint_identity = None
                lineage = record.lineage
                if record.checkpoint:
                    checkpoint = database.execute(
                        "SELECT identity, lineage FROM aliases WHERE alias=?", (record.checkpoint,)
                    ).fetchone()
                    if checkpoint is None:
                        raise ValueError("Undeclared checkpoint alias: " + record.checkpoint)
                    checkpoint_identity, declared_lineage = checkpoint
                    if lineage is not None and lineage != declared_lineage:
                        raise ValueError("Checkpoint lineage mismatch: " + record.case.id)
                    lineage = declared_lineage
                groups = {
                    "case": digest(record.case.model_dump(mode="json", exclude={"id"})),
                    "group": record.group,
                    "checkpoint": checkpoint_identity,
                    "lineage": lineage,
                    "attack_family": record.attack_family,
                    "task": record.task,
                }
                for kind in manifest.split_isolation:
                    identity = groups[kind]
                    if identity is None:
                        continue
                    previous = database.execute(
                        "SELECT split FROM groups WHERE kind=? AND identity=?", (kind, identity)
                    ).fetchone()
                    if previous and previous[0] != record.split:
                        raise ValueError(f"Dataset split leakage ({kind}): {identity}")
                    database.execute(
                        "INSERT OR IGNORE INTO groups VALUES (?, ?, ?)",
                        (kind, identity, record.split),
                    )
                counts[record.split] = counts.get(record.split, 0) + 1
                if record.split == split:
                    selected += 1
                    checksum.update((record.case.model_dump_json() + "\n").encode())
            if not selected:
                raise ValueError("Selected dataset split contains no cases: " + split)
            if labels:
                from ablitbench.benchmark import Label

                with labels.open(encoding="utf-8") as stream:
                    for line in stream:
                        label = Label.model_validate_json(line)
                        if label.case_id is None:
                            raise ValueError("Manifest labels require case_id and split")
                        row = database.execute(
                            "SELECT split FROM cases WHERE id=?", (label.case_id,)
                        ).fetchone()
                        if row is None or label.split != row[0]:
                            raise ValueError("Label case/split mismatch: " + label.case_id)
            database.commit()
        finally:
            database.close()
    identity = digest({"manifest": manifest.model_dump(mode="json"), "split": split})
    metadata = {
        **manifest.model_dump(mode="json"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_split": split,
        "split_counts": counts,
        "selected_cases": selected,
        "selected_sha256": checksum.hexdigest(),
        "identity": identity,
        "checkpoint_identity_basis": "Declared hashes; API checkpoint identity is not attested",
    }
    return DatasetInfo(
        path=path,
        cases=cases,
        labels=labels,
        split=split,
        count=selected,
        sha256=checksum.hexdigest(),
        identity=identity,
        metadata=metadata,
        manifest=manifest,
    )


def write_cases(info: DatasetInfo, destination: Path) -> None:
    if info.manifest is None:
        with info.cases.open("rb") as source, destination.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
    else:
        if file_hash(info.cases) != info.manifest.cases.sha256:
            raise ValueError("Dataset cases changed after validation")
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            for record in dataset_records(info.cases):
                if record.split == info.split:
                    output.write(record.case.model_dump_json() + "\n")
    if file_hash(destination) != info.sha256:
        raise ValueError("Dataset changed while preparing run cases")
