from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ablitbench.parity.source_runtime import original_module
from ablitbench.worker import isolation


class Evidence:
    def __init__(self, project: str, source: Path, output: Path) -> None:
        self.project = project
        self.source = source
        self.output = output
        self.counts: Counter[str] = Counter()
        self.collected = 0
        self.corrections: list[dict[str, Any]] = []
        self.stream = (output / "test-outcomes.jsonl").open("w", encoding="utf-8")

    def pytest_collection_modifyitems(self, session: Any, config: Any, items: Any) -> None:
        self.collected = len(items)
        if self.project == "modelaudit":
            expected = self.source / "tests/conftest.py"
            for module in list(sys.modules.values()):
                filename = getattr(module, "__file__", None)
                if filename and Path(filename).resolve() == expected:
                    original = module.__dict__["RESTRICTED_PYTHON_VERSIONS"]
                    module.__dict__["RESTRICTED_PYTHON_VERSIONS"] = frozenset()
                    self.corrections.append(
                        {
                            "scope": "test selection only; detector implementation unchanged",
                            "upstream_policy": "Python3.12 uses curated CI selection",
                            "profile": "explicit full native scanner acceptance",
                            "removed_restrictions": sorted(original),
                        }
                    )

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call" or report.outcome != "passed":
            key = report.when + ":" + report.outcome
            self.counts[key] += 1
            self.stream.write(
                json.dumps(
                    {
                        "nodeid": report.nodeid,
                        "phase": report.when,
                        "outcome": report.outcome,
                        "duration": report.duration,
                        "reason": str(report.longrepr) if report.longrepr else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, choices=["modelaudit", "modelscan"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    isolation()
    source = arguments.source.resolve(strict=True)
    output = arguments.output.resolve(strict=True)
    identity = original_module(arguments.project, source)
    workspace = tempfile.TemporaryDirectory(prefix="ablib-native-fixtures-")
    private_source = Path(workspace.name).resolve(strict=True) / "source"
    shutil.copytree(source, private_source)
    source = private_source
    import pytest

    os.environ["HOME"] = str(output / "home")
    Path(os.environ["HOME"]).mkdir()
    sys.path.insert(0, str(source))
    os.chdir(source)
    evidence = Evidence(arguments.project, source, output)
    tests = source / "tests" / "scanners" if arguments.project == "modelaudit" else source / "tests"
    try:
        exitcode = int(
            pytest.main(
                [
                    str(tests),
                    "-q",
                    "-ra",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp=" + str(Path(workspace.name) / "pytest-temp"),
                    "--junitxml=" + str(output / "junit.xml"),
                ],
                plugins=[evidence],
            )
        )
    finally:
        evidence.stream.close()
        workspace.cleanup()
    result = {
        "project": arguments.project,
        "upstream_identity": identity,
        "pytest_exitcode": exitcode,
        "collected": evidence.collected,
        "outcomes": dict(evidence.counts),
        "corrections": evidence.corrections,
        "scope": "Original native scanner fixture suite, not Scanner adapter or accuracy proof",
        "skipped_tests_are_coverage_gaps": True,
        "artifacts": ["test-outcomes.jsonl", "junit.xml"],
    }
    (output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
