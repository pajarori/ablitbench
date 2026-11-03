from __future__ import annotations

import base64
import hmac
import http.client
import json
import os
import shutil
import signal
import socket
import socketserver
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast


class Sink(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, output: Path, username: str, password: str) -> None:
        self.output = output
        self.username = username
        self.password = password
        self.messages: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), SMTPHandler)


class SMTPHandler(socketserver.StreamRequestHandler):
    def reply(self, text: str) -> None:
        self.wfile.write((text + "\r\n").encode())
        self.wfile.flush()

    def handle(self) -> None:
        server = cast(Sink, self.server)
        authenticated = False
        sender = ""
        recipients: list[str] = []
        self.reply("220 private-ablib-sink ESMTP")
        while line := self.rfile.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            command, _, argument = text.partition(" ")
            command = command.upper()
            if command in {"EHLO", "HELO"}:
                self.reply("250-private-ablib-sink")
                self.reply("250-AUTH PLAIN LOGIN")
                self.reply("250-SMTPUTF8")
                self.reply("250 8BITMIME")
            elif command == "AUTH":
                mechanism, _, payload = argument.partition(" ")
                try:
                    if mechanism.upper() == "PLAIN":
                        if not payload:
                            self.reply("334")
                            payload = self.rfile.readline().decode().strip()
                        fields = base64.b64decode(payload).decode().split("\0")
                        username, password = fields[-2:]
                    elif mechanism.upper() == "LOGIN":
                        self.reply("334 VXNlcm5hbWU6")
                        username = base64.b64decode(self.rfile.readline()).decode()
                        self.reply("334 UGFzc3dvcmQ6")
                        password = base64.b64decode(self.rfile.readline()).decode()
                    else:
                        raise ValueError("Unsupported fixture authentication")
                    authenticated = hmac.compare_digest(
                        username, server.username
                    ) and hmac.compare_digest(password, server.password)
                except (ValueError, UnicodeError):
                    authenticated = False
                self.reply(
                    "235 Authentication successful"
                    if authenticated
                    else "535 Authentication failed"
                )
            elif command == "MAIL" and authenticated:
                sender = argument
                recipients = []
                self.reply("250 Sender accepted")
            elif command == "RCPT" and authenticated:
                recipients.append(argument)
                self.reply("250 Recipient retained only in private sink")
            elif command == "DATA" and authenticated and recipients:
                self.reply("354 End with a single period")
                with server.lock:
                    index = len(server.messages)
                    server.output.mkdir(parents=True, exist_ok=True)
                    path = server.output / (str(index) + ".eml")
                    with path.open("wb") as stream:
                        while data := self.rfile.readline():
                            if data.rstrip(b"\r\n") == b".":
                                break
                            stream.write(data[1:] if data.startswith(b"..") else data)
                    server.messages.append(
                        {
                            "from": sender,
                            "to": recipients,
                            "artifact": path.name,
                            "authenticated": authenticated,
                            "external_delivery": False,
                        }
                    )
                self.reply("250 Retained in private research sink")
            elif command == "RSET":
                sender, recipients = "", []
                self.reply("250 Reset")
            elif command == "NOOP":
                self.reply("250 OK")
            elif command == "QUIT":
                self.reply("221 Closing private sink")
                break
            else:
                self.reply(
                    "530 Authentication required"
                    if not authenticated
                    else "500 Unsupported command"
                )


@contextmanager
def smtp_sink(output: Path, username: str, password: str) -> Iterator[Sink]:
    server = Sink(output, username, password)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def checked(command: list[str], environment: dict[str, str], log: Path, timeout: int) -> None:
    with log.open("ab") as stream:
        subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            timeout=timeout,
            check=True,
        )


@contextmanager
def postgres(
    output: Path, inputs: dict[str, Path], environment: dict[str, str], timeout: int
) -> Iterator[str]:
    binaries = next(
        (
            Path("/usr/lib/postgresql") / version / "bin"
            for version in ("17", "15")
            if (Path("/usr/lib/postgresql") / version / "bin/initdb").is_file()
        ),
        Path("/usr/lib/postgresql/17/bin"),
    )
    if not all(
        (binaries / name).is_file() for name in ("initdb", "pg_ctl", "pg_dump", "pg_restore")
    ):
        raise ValueError(
            "Private PostgreSQL requires the reviewed PostgreSQL 15 or 17 worker profile"
        )
    with tempfile.TemporaryDirectory(prefix="layerfault-postgres-") as temporary:
        scratch = Path(temporary)
        data, sockets = scratch / "data", scratch / "sockets"
        sockets.mkdir()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        log = output / "postgres-bootstrap.log"
        checked(
            [
                str(binaries / "initdb"),
                "-D",
                str(data),
                "-U",
                "ablib_fixture",
                "--auth=trust",
                "--no-locale",
                "--encoding=UTF8",
            ],
            environment,
            log,
            timeout,
        )
        options = f"-h 127.0.0.1 -p {port} -k {sockets} -c shared_buffers=16MB"
        checked(
            [
                str(binaries / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(scratch / "postgres.log"),
                "-o",
                options,
                "-w",
                "-t",
                str(timeout),
                "start",
            ],
            environment,
            log,
            timeout,
        )
        database = f"postgres://ablib_fixture@127.0.0.1:{port}/postgres?sslmode=disable"
        try:
            if "database_state" in inputs:
                checked(
                    [
                        str(binaries / "pg_restore"),
                        "--no-password",
                        "--no-owner",
                        "--no-privileges",
                        "--dbname=" + database,
                        str(inputs["database_state"]),
                    ],
                    environment,
                    log,
                    timeout,
                )
            else:
                checked(
                    ["/opt/parity/bin/layerfault", "platform", "migrate", "--database=" + database],
                    environment,
                    log,
                    timeout,
                )
            yield database
            checked(
                [
                    str(binaries / "pg_dump"),
                    "--no-password",
                    "--no-owner",
                    "--no-privileges",
                    "--format=custom",
                    "--file=" + str(output / "platform-state.dump"),
                    "--dbname=" + database,
                ],
                environment,
                log,
                timeout,
            )
        finally:
            checked(
                [
                    str(binaries / "pg_ctl"),
                    "-D",
                    str(data),
                    "-m",
                    "immediate",
                    "-w",
                    "-t",
                    str(timeout),
                    "stop",
                ],
                environment,
                log,
                timeout,
            )
            if (scratch / "postgres.log").is_file():
                shutil.copyfile(scratch / "postgres.log", output / "postgres-server.log")


@contextmanager
def sqlite(output: Path, inputs: dict[str, Path]) -> Iterator[str]:
    database = output / "store" / "platform.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    if "database_state" in inputs:
        source = inputs["database_state"]
        if source.is_symlink() or not source.is_file():
            raise ValueError("Private platform state must be an explicit regular file")
        shutil.copyfile(source, database)
    try:
        yield "sqlite:" + str(database)
    finally:
        if database.is_file():
            with (
                sqlite3.connect(database) as reader,
                sqlite3.connect(output / "platform-state.sqlite") as writer,
            ):
                reader.backup(writer)


@contextmanager
def profile(
    operation: str,
    settings: dict[str, Any],
    inputs: dict[str, Path],
    output: Path,
    environment: dict[str, str],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    if any(
        role in inputs
        for role in ("private_key", "evidence_key", "sign_with", "smtp_username", "smtp_password")
    ):
        raise ValueError("Private platform fixtures do not receive operator secret roles")
    backend = settings["private_platform"]
    database = (
        sqlite(output, inputs)
        if backend == "sqlite"
        else postgres(output, inputs, environment, settings["service_timeout_seconds"])
    )
    report: dict[str, Any] = {
        "backend": backend,
        "state_scope": "private copied workspace",
        "external_delivery": False,
    }
    with database as uri:
        arguments = dict(settings.get("arguments", {}), database=uri)
        if operation == "platform.serve":
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            arguments["listen"] = "127.0.0.1:" + str(port)
        if operation == "platform.newsletter.send" and not arguments.get("dry_run", False):
            username, password = "ablib-fixture", uuid.uuid4().hex
            with smtp_sink(output / "smtp-messages", username, password) as sink:
                host, port = sink.server_address[0], sink.server_address[1]
                endpoint = str(host) + ":" + str(port)
                arguments.update(
                    smtp_host=endpoint,
                    username_env="LAYERFAULT_SMTP_USERNAME",
                    password_env="LAYERFAULT_SMTP_PASSWORD",
                )
                environment.update(
                    ABLIB_SMTP_SINK=endpoint,
                    LAYERFAULT_SMTP_USERNAME=username,
                    LAYERFAULT_SMTP_PASSWORD=password,
                )
                yield dict(settings, arguments=arguments), report
                report.update(
                    message_count=len(sink.messages),
                    messages=sink.messages,
                    credential_origin="generated private fixture credentials",
                )
        else:
            yield dict(settings, arguments=arguments), report
        report["database_snapshot"] = database_snapshot(
            uri, output, environment, settings.get("service_timeout_seconds")
        )
    (output / "platform-profile.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )


def observe_service(
    process: Any, settings: dict[str, Any], output: Path, report: dict[str, Any]
) -> None:
    port = int(settings["arguments"]["listen"].rsplit(":", 1)[1])
    deadline = time.monotonic() + float(settings["service_timeout_seconds"])

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Private native HTTP service lifecycle timeout")
        return value

    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError("Native service exited before becoming ready")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=remaining()):
                    break
            except ConnectionRefusedError:
                time.sleep(min(0.01, remaining()))
        count = 0
        with (output / "service-responses.jsonl").open("w", encoding="utf-8") as stream:
            for request in settings["service_requests"]:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=remaining())
                try:
                    payload = (
                        json.dumps(request["payload"]).encode() if "payload" in request else None
                    )
                    connection.request(
                        request.get("method", "GET"),
                        request["path"],
                        body=payload,
                        headers={"Content-Type": "application/json"} if payload else {},
                    )
                    response = connection.getresponse()
                    body_path = output / "service-bodies" / str(count)
                    body_path.parent.mkdir(parents=True, exist_ok=True)
                    with body_path.open("wb") as body:
                        while chunk := response.read(65536):
                            remaining()
                            body.write(chunk)
                    stream.write(
                        json.dumps(
                            {
                                "request": request,
                                "status": response.status,
                                "headers": response.getheaders(),
                                "body_artifact": str(body_path.relative_to(output)),
                            }
                        )
                        + "\n"
                    )
                    count += 1
                finally:
                    connection.close()
        report.update(
            observed_requests=count,
            responses_artifact="service-responses.jsonl",
            lifecycle="finite private requests; native process group stopped",
            upstream_limits={
                "body_bytes": 1048576,
                "requests_per_minute": 600,
                "http_workers": 32,
                "queue_depth": 1024,
            },
        )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def database_snapshot(
    uri: str, output: Path, environment: dict[str, str], timeout: int | None
) -> dict[str, Any]:
    tables = {
        "jobs": "id,kind,state,attempts,max_attempts,last_error,payload_json",
        "findings": "id,review_id,rule_id,domain,status,confidence,detail_json",
    }
    for table, columns in tables.items():
        destination = output / ("platform-" + table + ".jsonl")
        if uri.startswith("sqlite:"):
            with (
                sqlite3.connect(uri.removeprefix("sqlite:")) as database,
                destination.open("w", encoding="utf-8") as stream,
            ):
                cursor = database.execute("SELECT " + columns + " FROM " + table)
                names = [column[0] for column in cursor.description]
                for row in cursor:
                    stream.write(
                        json.dumps(dict(zip(names, row, strict=True)), ensure_ascii=False) + "\n"
                    )
        else:
            binary = next(
                (
                    Path("/usr/lib/postgresql") / version / "bin/psql"
                    for version in ("17", "15")
                    if (Path("/usr/lib/postgresql") / version / "bin/psql").is_file()
                ),
                None,
            )
            if binary is None or timeout is None:
                raise ValueError(
                    "Private PostgreSQL observation requires its reviewed psql and timeout"
                )
            command = [
                str(binary),
                "--no-password",
                "--no-psqlrc",
                "--set=ON_ERROR_STOP=1",
                "--tuples-only",
                "--no-align",
                "--dbname=" + uri,
                "--command=SELECT row_to_json(t) FROM (SELECT "
                + columns
                + " FROM "
                + table
                + ") t",
            ]
            with (
                destination.open("wb") as stream,
                (output / "platform-observer.log").open("ab") as errors,
            ):
                subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=errors,
                    env=environment,
                    timeout=timeout,
                    check=True,
                )
    states: dict[str, int] = {}
    with (output / "platform-jobs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            state = row.get("state")
            if state not in {"READY", "LEASED", "SUCCEEDED", "DEAD"}:
                state = "UNKNOWN"
            states[state] = states.get(state, 0) + 1
    with (output / "platform-findings.jsonl").open(encoding="utf-8") as stream:
        finding_count = sum(1 for line in stream if line.strip())
    return {
        "job_states": states,
        "finding_rows": finding_count,
        "jobs_artifact": "platform-jobs.jsonl",
        "findings_artifact": "platform-findings.jsonl",
    }
