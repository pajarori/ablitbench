from __future__ import annotations

import html
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ablitbench.models import Event, RunResult
from ablitbench.store import Store

BANNER = "  ▌ ▜ ▘▌\n▀▌▛▌▐ ▌▛▌\n█▌▙▌▐▖▌▙▌"

PROBE_NAMES = {
    "protocol-non-streaming": "Non-streaming response",
    "protocol-streaming": "SSE streaming response",
    "benign-control": "Benign response control",
    "canary-disclosure": "Direct canary disclosure",
    "canary-secrecy-control": "Canary secrecy control",
    "role-override-disclosure": "Role override injection",
    "role-override-control": "Role override control",
    "canary-base64-disclosure": "Base64 canary disclosure",
    "base64-public-control": "Base64 public control",
    "canary-hex-disclosure": "Hex canary disclosure",
    "hex-public-control": "Hex public control",
    "tool-control": "Authorized tool use",
    "private-tool-hijack": "Private path tool hijack",
    "private-tool-boundary-control": "Private path boundary control",
    "tool-result-hijack": "Tool result prompt injection",
    "tool-result-benign-control": "Tool result benign control",
}

CHECK_NAMES = {
    "parity": "Research operation contracts",
    "source": "Loader, metadata, archive and serialization scan",
    "weights": "Tensor structure and weight statistics",
    "endpoint": "Protocol compatibility and streaming",
    "behavior": "Behavioral security probes",
    "discovery": "Conditional behavior discovery",
    "activation": "Activation and refusal interventions",
    "inversion": "Trigger inversion search",
    "fingerprint": "Behavioral fingerprint comparison",
    "adaptive": "Adaptive confidentiality and tool probes",
    "engines": "ModelAudit and ModelScan engines",
}

STATUS_STYLE = {
    "pass": ("PASS", "bold green"),
    "fail": ("FAIL", "bold red"),
    "inconclusive": ("INCONCLUSIVE", "bold yellow"),
    "completed": ("PASS", "bold green"),
    "unsupported": ("INCONCLUSIVE", "bold yellow"),
    "blocked": ("BLOCKED", "bold yellow"),
    "error": ("ERROR", "bold red"),
    "cancelled": ("CANCELLED", "bold yellow"),
}


def banner(console: Console) -> None:
    console.print(BANNER, style="bold cyan", markup=False, highlight=False)
    console.print()


def atomic_text(path: Path, write: Callable[[TextIO], None]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export(store: Store, result: RunResult) -> None:
    def report(stream: TextIO) -> None:
        stream.write('{"result":' + result.model_dump_json() + ',"checks":[')
        first = True
        for key, check in store.results():
            if not first:
                stream.write(",")
            payload = check.model_dump(mode="json")
            payload.pop("findings")
            payload.pop("evidence")
            payload.pop("outcomes")
            coverage = payload.pop("coverage")
            coverage.pop("gaps")
            stream.write('{"job":' + json.dumps(key) + ',"result":')
            serialized = json.dumps(payload, ensure_ascii=True)
            stream.write(serialized[:-1] + ',"findings":[')
            for index, finding in enumerate(store.finding_payloads(key)):
                stream.write(("," if index else "") + finding)
            stream.write('],"evidence":[')
            for index, evidence in enumerate(store.details(key, "evidence")):
                stream.write(("," if index else "") + json.dumps(evidence))
            stream.write('],"outcomes":[')
            for index, outcome in enumerate(store.outcome_payloads(key)):
                stream.write(("," if index else "") + outcome)
            stream.write('],"coverage":' + json.dumps(coverage)[:-1] + ',"gaps":[')
            for index, gap in enumerate(store.details(key, "gap")):
                stream.write(("," if index else "") + json.dumps(gap))
            stream.write("]}}}")
            first = False
        stream.write("]}")

    def events(stream: TextIO) -> None:
        for event in store.events():
            stream.write(event.model_dump_json() + "\n")

    def webpage(stream: TextIO) -> None:
        template = (
            Path(__file__).with_name("data").joinpath("report.html").read_text(encoding="utf-8")
        )
        header, footer = template.split("{{content}}")
        stream.write(header)
        stream.write("<pre>" + html.escape(result.model_dump_json(indent=2)) + "</pre>")
        for key, check in store.results():
            stream.write(
                "<article><h2>"
                + html.escape(key)
                + "</h2><pre>"
                + html.escape(check.model_dump_json(indent=2))
                + "</pre></article>"
            )
            for finding in store.finding_payloads(key):
                stream.write("<pre>" + html.escape(finding) + "</pre>")
            for outcome in store.outcome_payloads(key):
                stream.write("<pre>Case outcome: " + html.escape(outcome) + "</pre>")
            for gap in store.details(key, "gap"):
                stream.write("<p>Coverage gap: " + html.escape(gap) + "</p>")
        stream.write(footer)

    atomic_text(store.directory / "report.json", report)
    atomic_text(store.directory / "events.jsonl", events)
    atomic_text(store.directory / "report.html", webpage)


def event_terminal(console: Console, event: Event) -> None:
    if event.kind != "check_finished":
        return
    status = event.data["status"]
    if status in {"completed", "not_requested", "not_applicable"}:
        return
    style = {
        "completed": "green",
        "error": "red",
        "blocked": "yellow",
        "unsupported": "yellow",
        "cancelled": "yellow",
    }.get(status, "dim")
    row = Text()
    row.append(f"[{status}] ", style=style)
    check = str(event.data["check"])
    row.append(CHECK_NAMES.get(check, check.replace("-", " ").title()), style="cyan")
    row.append(f" · target {event.data['target'] + 1}", style="dim")
    if event.data["findings"]:
        row.append(f" · {event.data['findings']} findings", style="yellow")
    console.print(row)


def probe_name(case_id: str) -> str:
    return PROBE_NAMES.get(case_id, case_id.replace("_", " ").replace("-", " ").title())


def verdict(status: str) -> Text:
    label, style = STATUS_STYLE.get(status, (status.upper(), "dim"))
    return Text(label, style=style)


def outcome_detail(payload: dict[str, object]) -> str:
    evaluations = payload.get("evaluators", [])
    if isinstance(evaluations, list):
        priorities = ("fail", "inconclusive", "pass")
        for priority in priorities:
            for evaluation in evaluations:
                if (
                    isinstance(evaluation, dict)
                    and evaluation.get("status") == priority
                    and isinstance(evaluation.get("reason"), str)
                ):
                    return str(evaluation["reason"])
    return "No evaluator detail recorded"


def terminal_data(result: RunResult) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    connection = sqlite3.connect(
        f"{result.directory.joinpath('run.sqlite').as_uri()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        target_count = int(result.summary.get("targets", 1))
        outcomes = [
            {
                "target": int(row["target"]),
                "check": row["check_id"],
                "payload": json.loads(row["payload"]),
            }
            for row in connection.execute(
                "SELECT jobs.target, jobs.check_id, outcomes.payload "
                "FROM outcomes JOIN jobs ON jobs.key=outcomes.job ORDER BY outcomes.id"
            )
        ]
        probes = []
        for row in connection.execute("SELECT payload FROM events ORDER BY sequence"):
            event = json.loads(row["payload"])
            if event.get("kind") != "probe_outcome" or not isinstance(event.get("data"), dict):
                continue
            data = event["data"]
            probes.append(
                {
                    "target": int(data["target"]),
                    "check": str(data["check"]),
                    "probe": str(data["probe"]),
                    "status": str(data["status"]),
                    "detail": str(data["detail"]),
                }
            )
        detailed_jobs = {
            (int(row["target"]), str(row["check"])) for row in [*outcomes, *probes]
        }
        jobs = []
        for row in connection.execute(
            "SELECT key, target, check_id, state, result FROM jobs ORDER BY target, rowid"
        ):
            if row["state"] in {"not_requested", "not_applicable"}:
                continue
            if (int(row["target"]), str(row["check_id"])) in detailed_jobs:
                continue
            payload = json.loads(row["result"]) if row["result"] else {}
            coverage = payload.get("coverage", {})
            details = []
            finding_count = int(payload.get("finding_count", 0))
            if finding_count:
                details.append(
                    f"{finding_count} finding" + ("s" if finding_count != 1 else "")
                )
            if isinstance(coverage, dict) and coverage.get("total"):
                details.append(
                    f"{coverage.get('inspected', 0)}/{coverage['total']} "
                    f"{coverage.get('units', 'units')} inspected"
                )
            if payload.get("reason"):
                details.append(str(payload["reason"]))
            jobs.append(
                {
                    "target": int(row["target"]),
                    "check": row["check_id"],
                    "status": row["state"],
                    "detail": " · ".join(details) or "Completed",
                }
            )
        finding_rows = [finding.model_dump(mode="json") for finding in result.findings()]
        if target_count > 1:
            for row in [*probes, *outcomes]:
                row["suffix"] = f"target {int(row['target']) + 1}"
            for row in jobs:
                row["suffix"] = f"target {int(row['target']) + 1}"
        return [*probes, *outcomes, *jobs], finding_rows
    finally:
        connection.close()


def terminal(console: Console, result: RunResult) -> None:
    rows, findings = terminal_data(result)
    heading_style = "green" if result.status == "completed" else "yellow"
    console.print(
        Panel.fit(
            Text(f"Assessment {result.status}", style=f"bold {heading_style}"),
            subtitle=result.run_id[:12],
            border_style=heading_style,
        )
    )
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Result", width=13, no_wrap=True)
    table.add_column("Test", style="cyan", min_width=26)
    table.add_column("Detail", style="dim")
    for row in rows:
        if "probe" in row:
            name = probe_name(str(row["probe"]))
            if row.get("suffix"):
                name += f" · {row['suffix']}"
            table.add_row(
                verdict(str(row["status"])), Text(name), Text(str(row["detail"]))
            )
            continue
        payload = row.get("payload")
        if isinstance(payload, dict):
            name = probe_name(str(payload.get("case_id", "probe")))
            suffix = row.get("suffix")
            if suffix:
                name += f" · {suffix}"
            repeat = int(payload.get("repeat", 0))
            if repeat:
                name += f" · repeat {repeat + 1}"
            table.add_row(
                verdict(str(payload.get("status", "inconclusive"))),
                Text(name),
                Text(outcome_detail(payload)),
            )
        else:
            name = CHECK_NAMES.get(
                str(row["check"]), str(row["check"]).replace("-", " ").title()
            )
            if row.get("suffix"):
                name += f" · {row['suffix']}"
            table.add_row(
                verdict(str(row["status"])), Text(name), Text(str(row["detail"]))
            )
    console.print(table)
    if findings:
        console.print()
        finding_table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        finding_table.add_column("Severity", width=10, no_wrap=True)
        finding_table.add_column("Finding", style="bold")
        finding_table.add_column("Evidence")
        for finding in findings:
            severity = str(finding["severity"])
            severity_style = "bold red" if severity in {"high", "critical"} else "yellow"
            finding_table.add_row(
                Text(severity.upper(), style=severity_style),
                Text(str(finding["title"])),
                Text(str(finding["detail"])),
            )
        console.print(finding_table)
    statuses = []
    for row in rows:
        payload = row.get("payload")
        statuses.append(
            str(payload.get("status", "inconclusive"))
            if isinstance(payload, dict)
            else str(row.get("status", "inconclusive"))
        )
    inconclusive = sum(
        status in {"inconclusive", "unsupported", "blocked"} for status in statuses
    )
    summary = [
        f"{len(rows)} tests",
        f"{len(findings)} findings",
        f"{sum(status in {'pass', 'completed'} for status in statuses)} passed",
        f"{statuses.count('fail')} failed",
        f"{inconclusive} inconclusive",
    ]
    console.print()
    console.print(" · ".join(summary), style="bold")
    console.print(f"Report  {result.directory / 'report.html'}", style="dim", markup=False)
