from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import sys
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ablitbench.models import Case, CheckResult, Coverage, Event, Evidence, Job, ScanConfig, digest
from ablitbench.privacy import restrict_directory


def replay_cases(
    database: Path, destination: Path, temporary_directory: Path | None = None
) -> tuple[int, str] | None:
    source = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    source.execute("BEGIN")
    try:
        if (
            source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='case_definitions'"
            ).fetchone()
            is None
        ):
            return None
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(
            prefix="replay-index-", dir=temporary_directory or destination.parent
        ) as temporary:
            root = Path(temporary)
            index = sqlite3.connect(root / "index.sqlite")
            try:
                index.executescript(
                    "CREATE TABLE cases (original TEXT, identity TEXT, payload TEXT, "
                    "PRIMARY KEY(original, identity));"
                    "CREATE TABLE ids (id TEXT PRIMARY KEY);"
                )
                for (payload,) in source.execute(
                    "SELECT payload FROM case_definitions ORDER BY job, case_id"
                ):
                    case = Case.model_validate_json(payload)
                    canonical = json.dumps(case.model_dump(mode="json"), sort_keys=True)
                    checksum = hashlib.sha256(canonical.encode()).hexdigest()
                    index.execute(
                        "INSERT OR IGNORE INTO cases VALUES (?, ?, ?)",
                        (case.id, checksum, canonical),
                    )
                index.commit()
                stream_checksum = hashlib.sha256()
                count = 0
                pending = root / "cases.jsonl"
                with pending.open("wb") as stream:
                    for original, identity, payload, variants in index.execute(
                        "SELECT original, identity, payload, "
                        "(SELECT COUNT(*) FROM cases other WHERE other.original=cases.original) "
                        "FROM cases ORDER BY original, identity"
                    ):
                        identifier = original if variants == 1 else original + "-" + identity
                        while index.execute(
                            "SELECT 1 FROM ids WHERE id=?", (identifier,)
                        ).fetchone():
                            identifier += "-" + identity
                        index.execute("INSERT INTO ids VALUES (?)", (identifier,))
                        case = Case.model_validate_json(payload).model_copy(
                            update={"id": identifier}
                        )
                        line = (case.model_dump_json() + "\n").encode()
                        stream.write(line)
                        stream_checksum.update(line)
                        count += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(pending, 0o600)
                os.replace(pending, destination)
                return count, stream_checksum.hexdigest()
            finally:
                index.close()
    finally:
        source.close()


class Store:
    def __init__(self, directory: Path, config: ScanConfig, identities: list[str]) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        restrict_directory(self.directory)
        for name in ("run.sqlite", "evidence", ".writer.lock"):
            if (self.directory / name).is_symlink():
                raise ValueError("Run paths must not be symlinks")
        if config.resume and not (self.directory / "run.sqlite").is_file():
            raise ValueError("Resume directory has no run.sqlite")
        self.lock = (self.directory / ".writer.lock").open("a+b")
        try:
            if sys.platform == "win32":
                self.lock.write(b"0")
                self.lock.flush()
                self.lock.seek(0)
                importlib.import_module("msvcrt").locking(self.lock.fileno(), 2, 1)
            else:
                locking = importlib.import_module("fcntl")
                locking.flock(self.lock.fileno(), locking.LOCK_EX | locking.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError("Run already has an active writer") from None
        self.blobs = self.directory / "evidence"
        self.blobs.mkdir(exist_ok=True, mode=0o700)
        self.connection = sqlite3.connect(self.directory / "run.sqlite")
        os.chmod(self.directory / "run.sqlite", 0o600)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, target INTEGER, check_id TEXT,
    version TEXT, state TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS findings (id INTEGER PRIMARY KEY, job TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS details (id INTEGER PRIMARY KEY, job TEXT, kind TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS details_job ON details(job, kind);
CREATE INDEX IF NOT EXISTS findings_job ON findings(job);
CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY, size INTEGER, media_type TEXT);
""")
        identity = json.dumps({"config": config.identity(), "targets": identities}, sort_keys=True)
        existing = self.get("identity")
        if existing and self.get("schema_version") != "1":
            self.close()
            raise ValueError("Unsupported run store schema version")
        if config.resume:
            if existing != identity:
                self.close()
                raise ValueError(
                    "Resume identity mismatch: config, target content, or revision changed"
                )
        elif existing:
            self.close()
            raise ValueError("Output already contains a run; use --resume or a new directory")
        else:
            self.set("schema_version", "1")
            self.set("identity", identity)
            self.set("run_id", uuid.uuid4().hex)
            self.set(
                "config",
                config.model_dump_json(exclude={"resume", "replay", "output", "evaluation"}),
            )
            self.set("replay_of", str(config.replay) if config.replay else "")
        try:
            for table in ("findings", "details"):
                columns = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
                for column in ("unit", "identity"):
                    if column not in columns:
                        self.connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                        )
            self.connection.executescript("""
    DROP TABLE IF EXISTS evaluation_scopes;
    DROP TABLE IF EXISTS evaluation_units;
    CREATE TABLE IF NOT EXISTS measurements (id INTEGER PRIMARY KEY, job TEXT, role TEXT,
        backend TEXT, elapsed REAL, status TEXT, usage TEXT);
    CREATE TABLE IF NOT EXISTS outcomes (id INTEGER PRIMARY KEY, job TEXT NOT NULL,
        case_id TEXT NOT NULL, repeat INTEGER NOT NULL, status TEXT NOT NULL,
        payload TEXT NOT NULL, UNIQUE(job, case_id, repeat));
    CREATE INDEX IF NOT EXISTS outcomes_job ON outcomes(job);
    CREATE TABLE IF NOT EXISTS case_definitions (job TEXT NOT NULL, case_id TEXT NOT NULL,
        payload TEXT NOT NULL, PRIMARY KEY(job, case_id));
    CREATE TABLE IF NOT EXISTS episode_jobs (job TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS episodes (job TEXT NOT NULL, case_id TEXT NOT NULL,
        repeat INTEGER NOT NULL, state TEXT NOT NULL, PRIMARY KEY(job, case_id, repeat));
    CREATE TABLE IF NOT EXISTS units (job TEXT, name TEXT, state TEXT,
        inspected INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job, name));
    CREATE UNIQUE INDEX IF NOT EXISTS findings_identity ON findings(job, unit, identity)
        WHERE identity != '';
    CREATE UNIQUE INDEX IF NOT EXISTS details_identity ON details(job, unit, kind, identity)
        WHERE identity != '';
    """)
            self.active_units: dict[str, tuple[str, int]] = {}
            self.failed_units: dict[str, str] = {}
            self.unit_sequences: dict[str, int] = {}
            self.reconcile()
        except BaseException:
            self.close()
            raise

    def define_case(self, job: Job, case: Case) -> None:
        key = self.job_key(job)
        payload = case.model_dump_json()
        previous = self.connection.execute(
            "SELECT payload FROM case_definitions WHERE job=? AND case_id=?", (key, case.id)
        ).fetchone()
        if previous and previous[0] != payload:
            raise ValueError("Case definition changed within a job: " + case.id)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO case_definitions(job, case_id, payload) VALUES (?, ?, ?)",
                (key, case.id, payload),
            )

    def has_episodes(self, job: Job) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM episode_jobs WHERE job=?", (self.job_key(job),)
            ).fetchone()
            is not None
        )

    def prepare_episodes(self, job: Job, cases: Iterable[Case], repeats: int) -> None:
        if self.has_episodes(job):
            return
        key = self.job_key(job)
        with self.connection:
            for case in cases:
                payload = case.model_dump_json()
                prior = self.connection.execute(
                    "SELECT payload FROM case_definitions WHERE job=? AND case_id=?",
                    (key, case.id),
                ).fetchone()
                if prior and prior[0] != payload:
                    raise ValueError("Case definition changed within a job: " + case.id)
                self.connection.execute(
                    "INSERT OR IGNORE INTO case_definitions VALUES (?, ?, ?)",
                    (key, case.id, payload),
                )
                self.connection.executemany(
                    "INSERT OR IGNORE INTO episodes VALUES (?, ?, ?, 'unstarted')",
                    ((key, case.id, repeat) for repeat in range(repeats)),
                )
            self.connection.execute("INSERT INTO episode_jobs VALUES (?)", (key,))

    def episode_state(self, job: Job, case_id: str, repeat: int) -> str | None:
        row = self.connection.execute(
            "SELECT state FROM episodes WHERE job=? AND case_id=? AND repeat=?",
            (self.job_key(job), case_id, repeat),
        ).fetchone()
        return str(row[0]) if row else None

    def recover_episodes(self, job: Job) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE episodes SET state='uncertain' WHERE job=? AND state='running'",
                (self.job_key(job),),
            )

    def episode_result(self, job: Job, coverage: Coverage) -> CheckResult:
        previous = self.previous(job)
        result = previous[1] if previous and previous[1] else CheckResult(persisted=True)
        result.status, result.reason = "completed", None
        result.coverage.units = coverage.units
        result.coverage.total = coverage.total
        return result

    def begin_episode(self, job: Job, case: Case, repeat: int, result: CheckResult) -> bool:
        key = self.job_key(job)
        with self.connection:
            state = self.episode_state(job, case.id, repeat)
            if state != "unstarted":
                return False
            row = self.connection.execute(
                "SELECT payload FROM outcomes WHERE job=? AND case_id=? AND repeat=?",
                (key, case.id, repeat),
            ).fetchone()
            if row:
                outcome = json.loads(row[0])
                if not outcome.get("evaluators") or any(
                    item["evaluator"] != "not-observed" for item in outcome["evaluators"]
                ):
                    self.connection.execute(
                        "UPDATE episodes SET state='uncertain' "
                        "WHERE job=? AND case_id=? AND repeat=?",
                        (key, case.id, repeat),
                    )
                    return False
                self.connection.execute(
                    "DELETE FROM outcomes WHERE job=? AND case_id=? AND repeat=?",
                    (key, case.id, repeat),
                )
            self.connection.execute(
                "UPDATE episodes SET state='running' WHERE job=? AND case_id=? AND repeat=?",
                (key, case.id, repeat),
            )
            population = result.outcome_count - int(row is not None)
            saved = self._write_checkpoint(
                job, "running", result.model_copy(update={"outcome_count": population})
            )
        if saved:
            self._adopt_checkpoint(result, saved)
        return True

    def has_uncertain_episodes(self, job: Job) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM episodes WHERE job=? AND state IN ('running', 'uncertain') LIMIT 1",
                (self.job_key(job),),
            ).fetchone()
            is not None
        )

    def get(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def set(self, key: str, value: str) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def blob(self, chunks: Iterable[bytes], media_type: str = "application/json") -> Evidence:
        return self.commit_blob(self.prepare_blob(chunks, media_type))

    def commit_blob(self, evidence: Evidence) -> Evidence:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO blobs VALUES (?, ?, ?)",
                (evidence.sha256, evidence.size, evidence.media_type),
            )
        return evidence

    def prepare_blob(
        self, chunks: Iterable[bytes], media_type: str = "application/json"
    ) -> Evidence:
        from ablitbench.io_worker import prepare_blob

        return prepare_blob(chunks, self.blobs, media_type)

    def evidence(self, value: Any) -> str:
        return self.blob([json.dumps(value, ensure_ascii=True, allow_nan=False).encode()]).sha256

    def reconcile(self) -> None:
        missing = []
        for sha, size in self.connection.execute("SELECT sha256, size FROM blobs"):
            path = self.blobs / sha
            if not path.is_file() or path.stat().st_size != size:
                missing.append(sha)
                continue
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                    missing.append(sha)
        if missing:
            raise ValueError("Run evidence is missing or corrupt: " + ", ".join(missing))
        known = {row[0] for row in self.connection.execute("SELECT sha256 FROM blobs")}
        for path in self.blobs.iterdir():
            if path.name.startswith("pending-") or path.name not in known:
                path.unlink()

    @staticmethod
    def job_key(job: Job) -> str:
        return f"{job.target}:{job.check.id}:{job.check.version}"

    def _write_checkpoint(
        self, job: Job, state: str, result: CheckResult | None
    ) -> CheckResult | None:
        key = self.job_key(job)
        active = self.active_units.get(key)
        unit = (
            active[0]
            if active
            else self.failed_units.get(key, "")
            if state in {"error", "cancelled", "blocked"}
            else ""
        )
        if result is None or not result.persisted:
            self.connection.execute("DELETE FROM findings WHERE job=?", (key,))
            self.connection.execute("DELETE FROM details WHERE job=?", (key,))
            self.connection.execute("DELETE FROM units WHERE job=?", (key,))
            self.connection.execute("DELETE FROM outcomes WHERE job=?", (key,))
        saved = None
        if result:
            findings = self.connection.executemany(
                "INSERT OR IGNORE INTO findings(job, payload, unit, identity) VALUES (?, ?, ?, ?)",
                (
                    (
                        key,
                        finding.model_dump_json(),
                        unit,
                        digest(
                            [
                                self.unit_sequences.get(key, 0) + index,
                                finding.model_dump(mode="json"),
                            ]
                        )
                        if unit
                        else "",
                    )
                    for index, finding in enumerate(result.findings)
                ),
            ).rowcount
            evidence = self.connection.executemany(
                "INSERT OR IGNORE INTO details(job, kind, payload, unit, identity) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    (key, "evidence", value, unit, value if unit else "")
                    for value in result.evidence
                ),
            ).rowcount
            gaps = self.connection.executemany(
                "INSERT OR IGNORE INTO details(job, kind, payload, unit, identity) "
                "VALUES (?, ?, ?, ?, ?)",
                ((key, "gap", value, unit, digest(value)) for value in result.coverage.gaps),
            ).rowcount
            outcomes = 0
            target_payload = self.get(f"target:{job.target}") if result.outcomes else None
            target_identity = json.loads(target_payload).get("identity") if target_payload else None
            for outcome in result.outcomes:
                if target_identity is not None:
                    outcome = outcome.model_copy(update={"target_identity": target_identity})
                inserted = self.connection.execute(
                    "INSERT OR IGNORE INTO outcomes(job, case_id, repeat, status, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        key,
                        outcome.case_id,
                        outcome.repeat,
                        outcome.status,
                        outcome.model_dump_json(),
                    ),
                ).rowcount
                outcomes += inserted
                if inserted and not any(
                    item.evaluator == "not-observed" for item in outcome.evaluators
                ):
                    self.connection.execute(
                        "UPDATE episodes SET state=? WHERE job=? AND case_id=? AND repeat=? "
                        "AND state='running'",
                        (
                            "completed" if outcome.complete else "uncertain",
                            key,
                            outcome.case_id,
                            outcome.repeat,
                        ),
                    )
                if inserted:
                    event = Event(
                        kind="case_outcome",
                        data={"job": key, "outcome": outcome.model_dump(mode="json")},
                    )
                    self.connection.execute(
                        "INSERT INTO events(payload) VALUES (?)", (event.model_dump_json(),)
                    )
            saved = result.model_copy(
                update={
                    "persisted": True,
                    "outcome_count": result.outcome_count + outcomes,
                    "outcomes": [],
                    "finding_count": result.finding_count + findings,
                    "evidence_count": result.evidence_count + evidence,
                    "findings": [],
                    "evidence": [],
                    "coverage": result.coverage.model_copy(
                        update={"gap_count": result.coverage.gap_count + gaps, "gaps": []}
                    ),
                }
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                job.target,
                job.check.id,
                job.check.version,
                state,
                saved.model_dump_json() if saved else None,
            ),
        )
        return saved

    @staticmethod
    def _adopt_checkpoint(result: CheckResult, saved: CheckResult) -> None:
        result.persisted = saved.persisted
        result.finding_count = saved.finding_count
        result.evidence_count = saved.evidence_count
        result.outcome_count = saved.outcome_count
        result.outcomes.clear()
        result.coverage.gap_count = saved.coverage.gap_count
        result.findings.clear()
        result.evidence.clear()
        result.coverage.gaps.clear()

    def checkpoint(self, job: Job, state: str, result: CheckResult | None = None) -> None:
        with self.connection:
            saved = self._write_checkpoint(job, state, result)
        if result and saved:
            key = self.job_key(job)
            if key in self.active_units:
                self.unit_sequences[key] = self.unit_sequences.get(key, 0) + len(result.findings)
            self._adopt_checkpoint(result, saved)

    def has_units(self, job: Job) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM units WHERE job=? LIMIT 1", (self.job_key(job),)
            ).fetchone()
            is not None
        )

    def static_result(self, job: Job, coverage: Coverage) -> CheckResult:
        if not self.has_units(job):
            return CheckResult(coverage=coverage)
        key = self.job_key(job)
        inspected = self.connection.execute(
            "SELECT COALESCE(SUM(inspected), 0) FROM units WHERE job=? AND state='completed'",
            (key,),
        ).fetchone()[0]
        findings = self.connection.execute(
            "SELECT COUNT(*) FROM findings WHERE job=?", (key,)
        ).fetchone()[0]
        evidence = self.connection.execute(
            "SELECT COUNT(*) FROM details WHERE job=? AND kind='evidence'", (key,)
        ).fetchone()[0]
        gaps = self.connection.execute(
            "SELECT COUNT(*) FROM details WHERE job=? AND kind='gap'", (key,)
        ).fetchone()[0]
        return CheckResult(
            persisted=True,
            finding_count=findings,
            evidence_count=evidence,
            coverage=coverage.model_copy(update={"inspected": inspected, "gap_count": gaps}),
        )

    def begin_unit(self, job: Job, name: str, result: CheckResult) -> bool:
        key = self.job_key(job)
        row = self.connection.execute(
            "SELECT state FROM units WHERE job=? AND name=?", (key, name)
        ).fetchone()
        if row and row[0] == "completed":
            return False
        self.checkpoint(job, "running", result)
        with self.connection:
            removed = self.connection.execute(
                "DELETE FROM details WHERE job=? AND unit=? AND kind='gap'", (key, name)
            ).rowcount
            self.connection.execute(
                "INSERT OR REPLACE INTO units VALUES (?, ?, 'running', 0)", (key, name)
            )
            cleaned = result.model_copy(
                update={
                    "coverage": result.coverage.model_copy(
                        update={"gap_count": result.coverage.gap_count - removed}
                    )
                }
            )
            saved = self._write_checkpoint(job, "running", cleaned)
        if saved:
            self._adopt_checkpoint(result, saved)
        self.active_units[key] = name, result.coverage.inspected
        self.unit_sequences[key] = 0
        self.failed_units.pop(key, None)
        return True

    def finish_unit(self, job: Job, result: CheckResult, success: bool = True) -> None:
        key = self.job_key(job)
        name, before = self.active_units[key]
        with self.connection:
            saved = self._write_checkpoint(job, "running", result)
            self.connection.execute(
                "UPDATE units SET state=?, inspected=? WHERE job=? AND name=?",
                (
                    "completed" if success else "error",
                    max(0, result.coverage.inspected - before),
                    key,
                    name,
                ),
            )
        if saved:
            self._adopt_checkpoint(result, saved)
        self.active_units.pop(key)
        self.unit_sequences.pop(key, None)
        if success:
            self.failed_units.pop(key, None)
        else:
            self.failed_units[key] = name

    def details(self, key: str, kind: str) -> Iterator[str]:
        for row in self.connection.execute(
            "SELECT payload FROM details WHERE job=? AND kind=? ORDER BY id", (key, kind)
        ):
            yield row[0]

    def finding_payloads(self, key: str) -> Iterator[str]:
        for row in self.connection.execute(
            "SELECT payload FROM findings WHERE job=? ORDER BY id", (key,)
        ):
            yield row[0]

    def measure_generation(
        self, job: Job, role: str, backend: str, elapsed: float, usage: dict[str, Any] | None
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO measurements(job, role, backend, elapsed, status, usage) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.job_key(job),
                    role,
                    backend,
                    elapsed,
                    "completed" if usage is not None else "inconclusive",
                    json.dumps(usage) if usage is not None else None,
                ),
            )

    def outcome_payloads(self, key: str) -> Iterator[str]:
        for row in self.connection.execute(
            "SELECT payload FROM outcomes WHERE job=? ORDER BY id", (key,)
        ):
            yield row[0]

    def outcome_counts(self) -> dict[str, int]:
        counts = {"pass": 0, "fail": 0, "inconclusive": 0}
        for status, count in self.connection.execute(
            "SELECT status, COUNT(*) FROM outcomes GROUP BY status"
        ):
            counts[status] = count
        return counts

    def previous(self, job: Job) -> tuple[str, CheckResult | None] | None:
        row = self.connection.execute(
            "SELECT state, result FROM jobs WHERE key=?", (self.job_key(job),)
        ).fetchone()
        return (
            (row[0], CheckResult.model_validate_json(row[1]) if row[1] else None) if row else None
        )

    def event(self, event: Event) -> Event:
        cursor = self.connection.execute(
            "INSERT INTO events(payload) VALUES (?)", (event.model_dump_json(),)
        )
        event.sequence = cursor.lastrowid or 0
        self.connection.commit()
        return event

    def events(self) -> Iterator[Event]:
        for sequence, payload in self.connection.execute(
            "SELECT sequence, payload FROM events ORDER BY sequence"
        ):
            yield Event.model_validate_json(payload).model_copy(update={"sequence": sequence})

    def results(self) -> Iterator[tuple[str, CheckResult]]:
        for key, payload in self.connection.execute(
            "SELECT key, result FROM jobs WHERE result IS NOT NULL ORDER BY key"
        ):
            yield key, CheckResult.model_validate_json(payload)

    def close(self) -> None:
        self.connection.close()
        self.lock.close()
