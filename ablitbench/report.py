from __future__ import annotations

import html
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ablitbench.models import Event, RunResult
from ablitbench.store import Store

BANNER = "  ▌ ▜ ▘▌\n▀▌▛▌▐ ▌▛▌\n█▌▙▌▐▖▌▙▌"


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
    style = {
        "completed": "green",
        "error": "red",
        "blocked": "yellow",
        "unsupported": "yellow",
        "cancelled": "yellow",
    }.get(status, "dim")
    row = Text()
    row.append(f"[{status}] ", style=style)
    row.append(str(event.data["check"]), style="cyan")
    row.append(f" · target {event.data['target'] + 1}", style="dim")
    if event.data["findings"]:
        row.append(f" · {event.data['findings']} findings", style="yellow")
    console.print(row)


def terminal(console: Console, result: RunResult) -> None:
    console.print(f"Run {result.run_id}: {result.status}", markup=False)
    table = Table("Check status", "Count")
    for status, count in result.coverage.items():
        table.add_row(status, str(count))
    console.print(table)
    console.print(
        f"Findings: {result.summary['findings']} · Behavior: {result.summary['behavior']}",
        markup=False,
    )
    console.print(f"Report: {result.directory / 'report.html'}", markup=False)
