from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ablitbench.formats.archive import unsafe_path
from ablitbench.formats.json import loads
from ablitbench.formats.snapshot import secure_open
from ablitbench.models import TrustConfig
from ablitbench.targets import hash_file

if TYPE_CHECKING:
    from ablitbench.core import Context

DOMAIN = b"ablitbench-provenance-v1\0"


class ProvenanceRejected(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def signature_payload(key_id: str, statement: dict[str, Any]) -> bytes:
    return DOMAIN + canonical_json({"key_id": key_id, "statement": statement})


def decode_base64(value: Any, length: int) -> bytes:
    if not isinstance(value, str):
        raise ValueError("Signature or trusted key must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid base64 encoding") from error
    if len(decoded) != length or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError("Invalid Ed25519 encoding or length")
    return decoded


def canonical_record(raw: bytes) -> dict[str, Any]:
    try:
        value = loads(raw)
        if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
            raise ValueError("Noncanonical manifest JSONL record")
        return value
    except (ValueError, UnicodeError, TypeError, RecursionError) as error:
        raise ProvenanceRejected("Malformed or noncanonical manifest record") from error


def valid_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\0" not in value
        and "\\" not in value
        and not unsafe_path(value)
        and PurePosixPath(value).as_posix() == value
        and value != "."
    )


def valid_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def validate_manifest(
    path: Path,
    trust: TrustConfig,
    revision: str | None,
    database: Path,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    with secure_open(path) as stream, closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE TABLE entries (name TEXT PRIMARY KEY, size TEXT, "
            "sha256 TEXT, seen INTEGER DEFAULT 0)"
        )
        envelope = canonical_record(stream.readline())
        if set(envelope) != {"key_id", "signature", "statement"}:
            raise ProvenanceRejected(
                "Manifest envelope fields invalid; self-declared keys are not trusted"
            )
        key_id, statement = envelope["key_id"], envelope["statement"]
        if not isinstance(key_id, str) or not key_id:
            raise ProvenanceRejected("Invalid signing key identity")
        if key_id in trust.revoked_key_ids:
            raise ProvenanceRejected("Signing key is revoked: " + key_id)
        if key_id not in trust.trusted_keys:
            raise ProvenanceRejected("Signing key is not authorized by operator: " + key_id)
        key = decode_base64(trust.trusted_keys[key_id], 32)
        if not isinstance(statement, dict) or set(statement) != {
            "version",
            "namespace",
            "revision",
            "entries_sha256",
            "entry_count",
        }:
            raise ProvenanceRejected("Invalid signed statement fields")
        if type(statement["version"]) is not int or statement["version"] != 1:
            raise ProvenanceRejected("Unsupported signed manifest version")
        if not isinstance(statement["namespace"], str) or not statement["namespace"]:
            raise ProvenanceRejected("Invalid signed namespace")
        if trust.namespace is not None and statement["namespace"] != trust.namespace:
            raise ProvenanceRejected("Signed namespace differs from operator constraint")
        if statement["revision"] is not None and not isinstance(statement["revision"], str):
            raise ProvenanceRejected("Invalid signed revision")
        if statement["revision"] != revision:
            raise ProvenanceRejected("Signed revision differs from resolved target commit")
        if (
            type(statement["entry_count"]) is not int
            or statement["entry_count"] < 0
            or not valid_digest(statement["entries_sha256"])
        ):
            raise ProvenanceRejected("Invalid signed inventory commitment")
        try:
            signature = decode_base64(envelope["signature"], 64)
            Ed25519PublicKey.from_public_bytes(key).verify(
                signature, signature_payload(key_id, statement)
            )
        except (InvalidSignature, ValueError) as error:
            raise ProvenanceRejected("Ed25519 signature verification failed") from error
        checksum = hashlib.sha256()
        count = 0
        for raw in stream:
            if cancel is not None and cancel.is_set():
                raise OSError("Manifest verification cancelled")
            entry = canonical_record(raw)
            if set(entry) != {"name", "size", "sha256"} or not valid_name(entry["name"]):
                raise ProvenanceRejected("Invalid or unsafe signed inventory path")
            if (
                type(entry["size"]) is not int
                or entry["size"] < 0
                or not valid_digest(entry["sha256"])
            ):
                raise ProvenanceRejected("Invalid signed file size or hash")
            try:
                connection.execute(
                    "INSERT INTO entries(name,size,sha256) VALUES (?,?,?)",
                    (entry["name"], str(entry["size"]), entry["sha256"]),
                )
            except sqlite3.IntegrityError as error:
                raise ProvenanceRejected(
                    "Duplicate signed inventory path: " + entry["name"]
                ) from error
            checksum.update(raw)
            count += 1
        if checksum.hexdigest() != statement["entries_sha256"] or count != statement["entry_count"]:
            raise ProvenanceRejected("Signed inventory commitment does not match manifest entries")
        connection.commit()
        return {
            "key_id": key_id,
            "key_fingerprint": hashlib.sha256(key).hexdigest(),
            "statement": statement,
        }


async def inspect_provenance(context: Context) -> AsyncIterator[dict[str, Any]]:
    trust = context.config.trust
    if trust is None:
        return
    identity = {
        "target_identity": context.target.identity,
        "target": context.target.target.location,
        "revision": context.target.revision,
    }
    database: Path | None = None
    manifest_reference: str | None = None
    try:
        manifest = trust.manifest.expanduser().absolute()
        if context.target.root and (
            manifest.resolve() == context.target.root
            or context.target.root.is_dir()
            and manifest.resolve().is_relative_to(context.target.root)
        ):
            raise ProvenanceRejected(
                "Signed manifest must be outside assessed package to avoid self-reference"
            )
        with secure_open(manifest) as stream:
            before = os.fstat(stream.fileno())
            evidence = context.store.blob(
                iter(lambda: stream.read(1024 * 1024), b""), "application/x-ndjson"
            )
            manifest_reference = evidence.sha256
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError("Signed manifest drifted during capture")
        descriptor, name = tempfile.mkstemp(
            prefix="provenance-", suffix=".sqlite", dir=context.store.directory
        )
        os.close(descriptor)
        database = Path(name)
        verified = await context.io(
            validate_manifest,
            context.store.blobs / manifest_reference,
            trust,
            context.target.revision,
            database,
        )

        yield {
            "metadata": True,
            "stage": "signature verified",
            "manifest": manifest_reference,
            **identity,
            **verified,
            "interpretation": "Authorized signature verified; file checks follow",
        }
        complete = context.target.inventory_scope == "full"
        if not complete:
            yield {
                "gap": "Offline cache cannot establish complete signed package inventory",
                "manifest": manifest_reference,
                **identity,
            }
        mismatched = False
        inspected = 0
        with closing(sqlite3.connect(database)) as connection:
            for entry in context.target.entries():
                await context.yield_control()
                expected = connection.execute(
                    "SELECT size,sha256,seen FROM entries WHERE name=?", (entry.name,)
                ).fetchone()
                if expected is None or expected[2]:
                    mismatched = True
                    yield {
                        "issue": "Signed package inventory mismatch",
                        "reason": "extra or duplicate target entry",
                        "file": entry.name,
                        "manifest": manifest_reference,
                        **identity,
                    }
                    continue
                connection.execute("UPDATE entries SET seen=1 WHERE name=?", (entry.name,))
                if entry.issue:
                    complete = False
                    yield {
                        "gap": "Signed file unavailable for verification: " + entry.name,
                        "manifest": manifest_reference,
                        **identity,
                    }
                    continue
                try:
                    path = await context.snapshot(context.target, entry)
                    actual = entry.sha256 or await context.io(hash_file, path)
                    size = path.stat().st_size
                except (OSError, ValueError) as error:
                    complete = False
                    yield {
                        "gap": "Signed file could not be inspected: " + entry.name,
                        "error": str(error),
                        "manifest": manifest_reference,
                        **identity,
                    }
                    continue
                inspected += 1
                if actual != expected[1] or size != int(expected[0]):
                    mismatched = True
                    yield {
                        "issue": "Signed package content mismatch",
                        "file": entry.name,
                        "expected_sha256": expected[1],
                        "actual_sha256": actual,
                        "expected_size": int(expected[0]),
                        "actual_size": size,
                        "manifest": manifest_reference,
                        **identity,
                    }
            for (missing,) in connection.execute(
                "SELECT name FROM entries WHERE seen=0 ORDER BY name"
            ):
                await context.yield_control()
                mismatched = True
                if context.target.inventory_scope == "cached":
                    yield {
                        "gap": "Signed entry absent from partial cache: " + missing,
                        "manifest": manifest_reference,
                        **identity,
                    }
                else:
                    yield {
                        "issue": "Signed package inventory mismatch",
                        "reason": "signed entry missing from target",
                        "file": missing,
                        "manifest": manifest_reference,
                        **identity,
                    }
        yield {
            "metadata": True,
            "verified": complete and not mismatched,
            "files_inspected": inspected,
            "manifest": manifest_reference,
            **identity,
            **verified,
            "interpretation": "Operator-authorized integrity verification; not a safety verdict",
        }
    except ProvenanceRejected as error:
        yield {
            "issue": "Signed provenance rejected",
            "reason": str(error),
            "manifest": manifest_reference,
            **identity,
        }
    except (OSError, ValueError, sqlite3.Error, MemoryError, OverflowError) as error:
        yield {
            "gap": "Signed provenance could not be inspected",
            "error": str(error),
            "manifest": manifest_reference,
            **identity,
        }
    finally:
        if database is not None:
            database.unlink(missing_ok=True)
