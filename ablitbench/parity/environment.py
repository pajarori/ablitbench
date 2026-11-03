from __future__ import annotations

import ctypes
import hashlib
import http.server
import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ablitbench.formats.json import loads


def landlock_workspace(workspace: Path) -> None:
    if sys.platform != "linux":
        raise ValueError("Landlock executor requires Linux")
    path_flag = getattr(os, "O_PATH", None)
    if type(path_flag) is not int:
        raise ValueError("Linux O_PATH capability is unavailable")
    library = ctypes.CDLL(None, use_errno=True)
    version = library.syscall(444, 0, 0, 1)
    if version < 1:
        raise OSError("Linux Landlock filesystem isolation is unavailable")
    handled = (1 << 13) - 1
    if version >= 2:
        handled |= 1 << 13
    if version >= 3:
        handled |= 1 << 14

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    specification = Ruleset(handled)
    descriptor = library.syscall(444, ctypes.byref(specification), ctypes.sizeof(specification), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Landlock ruleset creation failed")
    read = 1 | 4 | 8
    try:
        for path, access in [
            (workspace, handled),
            *[
                (Path(path), read)
                for path in (
                    "/bin",
                    "/sbin",
                    "/lib",
                    "/lib64",
                    "/usr/bin",
                    "/usr/sbin",
                    "/usr/lib",
                    "/usr/local/bin",
                    "/usr/local/lib",
                )
            ],
            (Path("/dev/null"), 2 | 4),
            (Path("/dev/urandom"), 4),
        ]:
            if not path.exists():
                continue
            target = os.open(path.resolve(), path_flag | os.O_CLOEXEC)
            try:
                rule = PathRule(access, target)
                if library.syscall(445, descriptor, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "Landlock path rule failed")
            finally:
                os.close(target)
        if library.prctl(38, 1, 0, 0, 0) != 0 or library.syscall(446, descriptor, 0) < 0:
            raise OSError(ctypes.get_errno(), "Landlock restriction failed")
    finally:
        os.close(descriptor)


def atomic_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".temporary")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EnvironmentExecutor:
    def __init__(self, fixture: Path | None, output: Path) -> None:
        self.fixture, self.output = fixture, output
        self.output.mkdir(parents=True, exist_ok=True)
        self.web: dict[str, dict[str, Any]] = {}

    def fixture_spec(self, task: str) -> dict[str, Any]:
        if self.fixture is None:
            raise ValueError(
                "Controlled environment assets are unavailable; provide readonly fixture"
            )
        path = self.fixture / "environment.json" if self.fixture.is_dir() else self.fixture
        spec = loads(path.read_text())
        allowed = {
            "schema_version",
            "task",
            "initial_observation",
            "files",
            "evaluation",
            "site",
            "observer_separated",
        }
        if (
            not isinstance(spec, dict)
            or set(spec) - allowed
            or spec.get("schema_version") != 1
            or spec.get("task") != task
            or not isinstance(spec.get("initial_observation"), str)
        ):
            raise ValueError("Invalid controlled environment fixture schema or task")
        if spec.get("observer_separated") is not None and (
            spec["observer_separated"] is not True or "evaluation" in spec
        ):
            raise ValueError("Projected environment must exclude all private evaluation fields")
        return spec

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        from ablitbench.worker import isolation

        isolation()
        if (
            set(request)
            - {
                "kind",
                "channel",
                "action",
                "environment",
                "environment_id",
                "episode_id",
                "payload",
            }
            or request.get("action") not in {"reset", "step", "close"}
            or request.get("environment") not in {"os", "mind2web", "webshop"}
            or not isinstance(request.get("episode_id"), str)
            or not request["episode_id"]
            or not isinstance(request.get("payload"), dict)
        ):
            raise ValueError("Invalid isolated environment request")
        identity = hashlib.sha256(
            (request["environment"] + "\0" + request["episode_id"]).encode()
        ).hexdigest()
        root = self.output / identity
        root.mkdir(exist_ok=True)
        state_path = root / "state.json"
        action, payload = request["action"], request["payload"]
        if action == "close":
            resources = self.web.pop(identity, None)
            if resources:
                resources["browser"].close()
                resources["playwright"].stop()
                resources["server"].shutdown()
                resources["server"].server_close()
                resources["thread"].join()
            if state_path.exists():
                state = loads(state_path.read_text())
                state["closed"] = True
                atomic_state(state_path, state)
            return {"closed": True}
        try:
            spec = self.fixture_spec(request["environment"])
            if action == "reset":
                if state_path.exists():
                    raise ValueError("Existing episode cannot be blindly reset or replayed")
                if (
                    type(payload.get("observation_bytes")) is not int
                    or payload["observation_bytes"] <= 0
                ):
                    raise ValueError("Explicit positive observation byte budget required")
                workspace = root / "workspace"
                workspace.mkdir()
                files = spec.get("files", {})
                if not isinstance(files, dict):
                    raise ValueError("Environment files must be a path/content mapping")
                for name, content in files.items():
                    path = workspace / name
                    if (
                        not isinstance(name, str)
                        or Path(name).is_absolute()
                        or ".." in Path(name).parts
                        or not isinstance(content, str)
                    ):
                        raise ValueError("Fixture file path or content is invalid")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                atomic_state(
                    state_path,
                    {
                        "task": request["environment"],
                        "next_index": 0,
                        "pending": False,
                        "observation_bytes": payload["observation_bytes"],
                        "terminated": False,
                    },
                )
                if request["environment"] != "os":
                    timeout = payload.get("action_timeout_seconds")
                    if (
                        type(timeout) not in {int, float}
                        or not math.isfinite(timeout)
                        or timeout <= 0
                    ):
                        raise ValueError("Explicit finite browser action timeout required")
                    self.start_web(identity, spec, workspace, float(timeout))
                return {"observation": spec["initial_observation"]}
            if not state_path.is_file():
                raise ValueError(
                    "Episode is not initialized; no reset or side-effect replay fallback"
                )
            state = loads(state_path.read_text())
            if (
                state["terminated"]
                or state.get("closed")
                or state["pending"]
                or payload.get("index") != state["next_index"]
            ):
                raise ValueError(
                    "Episode index is stale, terminated or has an unknown prior delivery"
                )
            step = payload.get("action")
            if not isinstance(step, dict):
                raise ValueError("Environment action must be structured")
            state["pending"] = True
            atomic_state(state_path, state)
            if request["environment"] == "os":
                result = self.os_step(root, spec, step, state)
            else:
                result = self.web_step(identity, request["environment"], step)
                raw = result["observation"].encode("utf-8")
                page_artifact = root / (str(payload["index"]) + "-page.txt")
                page_artifact.write_bytes(raw)
                result["observation"] = raw[: state["observation_bytes"]].decode(
                    "utf-8", errors="replace"
                )
                result["impact_evidence"].append(
                    {
                        "kind": "page-observation",
                        "artifact": str(page_artifact.relative_to(self.output)),
                        "output_bytes": len(raw),
                        "observed_bytes": min(len(raw), state["observation_bytes"]),
                        "truncated": len(raw) > state["observation_bytes"],
                    }
                )
            state.update(
                pending=False, next_index=state["next_index"] + 1, terminated=result["terminated"]
            )
            atomic_state(root / (str(payload["index"]) + "-result.json"), result)
            atomic_state(state_path, state)
            return result
        except Exception as error:
            return {
                "status": "unsupported",
                "gaps": [type(error).__name__ + ": " + str(error)],
                "observation": "",
                "terminated": False,
                "success": None,
                "impact_evidence": [],
            }

    def os_step(
        self, root: Path, spec: dict[str, Any], action: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        workspace = root / "workspace"
        if action == {"kind": "finish"}:
            if spec.get("observer_separated") is True:
                return {
                    "observation": "Episode finished; independent observer evaluation pending",
                    "terminated": True,
                    "success": None,
                    "impact_evidence": [
                        {"kind": "sandbox-workspace-observed", "evaluation": "separate-observer"}
                    ],
                }
            evaluation = spec.get("evaluation")
            if (
                not isinstance(evaluation, dict)
                or set(evaluation) != {"files_equal"}
                or not isinstance(evaluation["files_equal"], dict)
            ):
                raise ValueError("Independent OS success evaluator is unavailable")
            outcomes = []
            for name, expected in evaluation["files_equal"].items():
                path = (workspace / name).resolve()
                if (
                    Path(name).is_absolute()
                    or not path.is_relative_to(workspace)
                    or path.is_symlink()
                    or not isinstance(expected, str)
                ):
                    raise ValueError("OS evaluator path is outside private workspace")
                outcomes.append(path.is_file() and path.read_text() == expected)
            if not outcomes:
                raise ValueError("OS evaluator has no explicit task criteria")
            return {
                "observation": "Episode finished",
                "terminated": True,
                "success": all(outcomes),
                "impact_evidence": [
                    {
                        "kind": "sandbox-file-evaluation",
                        "criteria": len(outcomes),
                        "matched": sum(outcomes),
                    }
                ],
            }
        if (
            set(action) != {"kind", "command"}
            or action["kind"] != "bash"
            or not isinstance(action["command"], str)
        ):
            raise ValueError("OS executor accepts only typed bash or finish actions")
        if spec.get("observer_separated") is not True:
            landlock_workspace_probe(workspace)
        protect_executor_supervisor()
        psutil = importlib.import_module("psutil")
        baseline = {
            (child.pid, child.create_time()) for child in psutil.Process().children(recursive=True)
        }
        index = str(state["next_index"])
        stdout, stderr = root / (index + "-stdout"), root / (index + "-stderr")
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc", "-c", action["command"]],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=workspace,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": str(workspace),
                    "TMPDIR": str(workspace),
                    "LANG": "C.UTF-8",
                },
                start_new_session=True,
                close_fds=True,
                preexec_fn=(lambda: landlock_workspace(workspace))
                if spec.get("observer_separated") is not True
                else None,
            )
            try:
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                descendants = reap_action_descendants(baseline)
        observations, total = [], 0
        remaining = state["observation_bytes"]
        for path in (stdout, stderr):
            total += path.stat().st_size
            with path.open("rb") as stream:
                data = stream.read(remaining)
            remaining -= len(data)
            observations.append(data.decode("utf-8", errors="replace"))
        return {
            "observation": "Exit status: "
            + str(code)
            + "\nstdout:\n"
            + observations[0]
            + "\nstderr:\n"
            + observations[1],
            "terminated": False,
            "success": None,
            "impact_evidence": [
                {
                    "kind": "sandbox-shell",
                    "exit_code": code,
                    "background_processes_reaped": descendants,
                    "stdout": str(stdout.relative_to(self.output)),
                    "stderr": str(stderr.relative_to(self.output)),
                    "output_bytes": total,
                    "observed_bytes": min(total, state["observation_bytes"]),
                    "truncated": total > state["observation_bytes"],
                    "filesystem_isolation": "observer-separated-container"
                    if spec.get("observer_separated") is True
                    else "landlock",
                }
            ],
        }

    def start_web(
        self, identity: str, spec: dict[str, Any], workspace: Path, timeout: float
    ) -> None:
        if not self.fixture or not self.fixture.is_dir() or spec.get("site") != "site":
            raise ValueError("Local native web assets are unavailable; expected fixture/site")
        source = self.fixture / "site"
        if not source.is_dir() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Web assets must be a reviewed local tree without symlinks")
        shutil.copytree(source, workspace / "site")
        protect_executor_supervisor()
        playwright = importlib.import_module("playwright.sync_api")
        runtime = playwright.sync_playwright().start()
        server = None
        browser = None
        try:

            class Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(*args, directory=str(workspace / "site"), **kwargs)

                def log_message(self, format: str, *args: Any) -> None:
                    return

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            browser = runtime.chromium.launch(
                headless=True, chromium_sandbox=True, timeout=timeout * 1000
            )
            page = browser.new_page()
            page.set_default_timeout(timeout * 1000)
            page.set_default_navigation_timeout(timeout * 1000)
            origin = "http://127.0.0.1:" + str(server.server_port)
            page.route(
                "**/*",
                lambda route: (
                    route.continue_()
                    if route.request.url.startswith(origin + "/")
                    else route.abort()
                ),
            )
            page.goto(origin + "/index.html", wait_until="load")
            self.web[identity] = {
                "playwright": runtime,
                "browser": browser,
                "page": page,
                "server": server,
                "thread": thread,
                "workspace": workspace,
            }
        except BaseException:
            if browser:
                browser.close()
            if server:
                server.shutdown()
                server.server_close()
            runtime.stop()
            raise

    def web_step(self, identity: str, task: str, action: dict[str, Any]) -> dict[str, Any]:
        if identity not in self.web:
            raise ValueError(
                "Live native browser session unavailable; no replay after process restart"
            )
        page = self.web[identity]["page"]
        if task == "webshop":
            if set(action) != {"kind", "value"} or action["kind"] not in {"search", "click"}:
                raise ValueError("Unsupported native WebShop action")
            if action["kind"] == "search":
                page.get_by_role("textbox", name="Search", exact=True).fill(action["value"])
                page.get_by_role("button", name="Search", exact=True).click()
            else:
                page.get_by_test_id(action["value"]).click()
        else:
            if set(action) != {"kind", "element", "action"} or action["kind"] != "mind2web":
                raise ValueError("Unsupported native Mind2Web action")
            element = page.get_by_test_id(action["element"])
            if action["action"] == "CLICK":
                element.click()
            elif action["action"].startswith("TYPE: "):
                element.fill(action["action"][6:])
            elif action["action"].startswith("SELECT: "):
                element.select_option(action["action"][8:])
            else:
                raise ValueError("Unsupported native Mind2Web action verb")
        state = {
            "path": page.url.split("/", 3)[-1],
            "text": page.locator("body").inner_text(),
            "local_storage": page.evaluate("Object.fromEntries(Object.entries(localStorage))"),
            "fields": page.locator("input,select,textarea").evaluate_all(
                "elements => elements.map(element => ({"
                "id:element.getAttribute('data-testid'),name:element.name,value:element.value}))"
            ),
        }
        atomic_state(self.web[identity]["workspace"] / "browser-state.json", state)
        terminated = page.locator("body").get_attribute("data-terminated") == "true"
        success = (
            page.locator("body").get_attribute("data-success") == "true" if terminated else None
        )
        return {
            "observation": page.locator("body").inner_text(),
            "terminated": terminated,
            "success": success,
            "impact_evidence": [
                {"kind": "sandbox-local-browser", "environment": task, "url": page.url}
            ],
        }


def landlock_workspace_probe(workspace: Path) -> None:
    process = os.fork()
    if process == 0:
        try:
            landlock_workspace(workspace)
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(process, 0)
    if status:
        raise ValueError("Required Linux Landlock isolation is unavailable; shell was not executed")


class EnvironmentObserver:
    def __init__(self, fixture: Path | None, output: Path) -> None:
        self.fixture, self.output = fixture, output
        self.output.mkdir(parents=True, exist_ok=True)
        self.private = self.output / "private-observer"
        self.private.mkdir(exist_ok=True, mode=0o700)
        self.lock = threading.Lock()
        self.spec: dict[str, Any] | None = None

    def project(self) -> Path:
        if self.fixture is None:
            raise ValueError("Controlled environment fixture is required")
        descriptor = self.fixture / "environment.json" if self.fixture.is_dir() else self.fixture
        spec = loads(descriptor.read_text())
        allowed = {"schema_version", "task", "initial_observation", "files", "evaluation", "site"}
        if (
            not isinstance(spec, dict)
            or set(spec) - allowed
            or spec.get("schema_version") != 1
            or spec.get("task") not in {"os", "mind2web", "webshop"}
            or not isinstance(spec.get("initial_observation"), str)
        ):
            raise ValueError("Observer requires an original private controlled fixture")
        evaluation = spec.get("evaluation")
        if (
            not isinstance(evaluation, dict)
            or set(evaluation) - {"files_equal", "json_equal", "terminal_json_equal"}
            or not (evaluation.get("files_equal") or evaluation.get("json_equal"))
            or any(not isinstance(criteria, dict) for criteria in evaluation.values())
        ):
            raise ValueError("Explicit private file/JSON success criteria required")
        public = {key: value for key, value in spec.items() if key != "evaluation"}
        public["observer_separated"] = True
        directory = self.output / "public-fixture"
        if directory.exists():
            raise ValueError("Existing public fixture cannot be implicitly replaced")
        directory.mkdir(mode=0o755)
        if spec.get("site") is not None:
            if spec["site"] != "site" or not self.fixture.is_dir():
                raise ValueError(
                    "Native web fixture requires its explicitly declared local site tree"
                )
            source = self.fixture / "site"
            if not source.is_dir() or any(path.is_symlink() for path in source.rglob("*")):
                raise ValueError("Public site assets cannot contain symlinks")
            shutil.copytree(source, directory / "site")
        atomic_state(directory / "environment.json", public)
        atomic_state(
            self.private / "evaluation.json", {"task": spec["task"], "evaluation": evaluation}
        )
        self.spec = spec
        return directory

    def identity(self, request: dict[str, Any]) -> str:
        if self.spec is None:
            private = loads((self.private / "evaluation.json").read_text())
            self.spec = private
        if (
            request.get("environment") != self.spec["task"]
            or not isinstance(request.get("episode_id"), str)
            or not request["episode_id"]
            or request.get("action") not in {"reset", "step", "close"}
            or not isinstance(request.get("payload"), dict)
        ):
            raise ValueError("Observer request identity or action mismatch")
        return hashlib.sha256(
            (request["environment"] + "\0" + request["episode_id"]).encode()
        ).hexdigest()

    def before(self, request: dict[str, Any]) -> None:
        identity = self.identity(request)
        if request["action"] == "close":
            return
        with self.lock:
            path = self.private / (identity + ".json")
            if path.exists():
                state = loads(path.read_text())
                if (
                    request["action"] == "reset"
                    or state["pending"]
                    or state["terminated"]
                    or state.get("closed")
                    or request["payload"].get("index") != state["next_index"]
                ):
                    raise ValueError(
                        "Observer refuses stale, pending, reset or terminated side-effect delivery"
                    )
            elif request["action"] != "reset":
                raise ValueError("Observer episode is not initialized")
            else:
                state = {"next_index": 0, "pending": False, "terminated": False}
            lease = self.private / (identity + ".lease")
            try:
                descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as error:
                raise ValueError(
                    "Observer delivery has an existing durable pending lease"
                ) from error
            with os.fdopen(descriptor, "w") as stream:
                stream.write(
                    hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                )
                stream.flush()
                os.fsync(stream.fileno())
            state.update(
                pending=True,
                request=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
            )
            atomic_state(path, state)

    def after(
        self, request: dict[str, Any], result: dict[str, Any], artifacts: Path
    ) -> dict[str, Any]:
        identity = self.identity(request)
        with self.lock:
            path = self.private / (identity + ".json")
            if request["action"] == "close":
                if path.exists():
                    state = loads(path.read_text())
                    state["closed"] = True
                    atomic_state(path, state)
                return result
            state = loads(path.read_text())
            fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            if not state["pending"] or state["request"] != fingerprint:
                raise ValueError("Observer response does not match a fenced delivery")
            if result.get("status") == "unsupported":
                return result
            if not isinstance(result.get("observation"), str) or (
                request["action"] == "step"
                and (
                    set(result) != {"observation", "terminated", "success", "impact_evidence"}
                    or type(result.get("terminated")) is not bool
                    or not isinstance(result.get("impact_evidence"), list)
                )
            ):
                raise ValueError("Observer received malformed environment acknowledgement")
            corrected = {**result, "success": None}
            if request["action"] == "step":
                workspace = artifacts / identity / "workspace"
                descriptor = observer_directory(artifacts, identity + "/workspace")
                os.close(descriptor)
                private = loads((self.private / "evaluation.json").read_text())
                evaluation = private["evaluation"]
                checks = observer_checks(workspace, evaluation, artifacts)
                matches = all(checks) and bool(checks)
                if request["environment"] == "os":
                    terminal = request["payload"].get("action") == {"kind": "finish"}
                else:
                    final_checks = observer_checks(
                        workspace,
                        {"json_equal": evaluation.get("terminal_json_equal", {})},
                        artifacts,
                    )
                    terminal = matches or bool(final_checks) and all(final_checks)
                corrected.update(
                    terminated=bool(terminal), success=bool(matches) if terminal else None
                )
                corrected.setdefault("impact_evidence", []).append(
                    {
                        "kind": "independent-host-observer",
                        "criteria": len(checks),
                        "matched": sum(checks),
                        "worker_success_ignored": True,
                    }
                )
                state.update(next_index=state["next_index"] + 1, terminated=bool(terminal))
            state["pending"] = False
            atomic_state(self.private / (identity + "-last-result.json"), corrected)
            atomic_state(path, state)
            (self.private / (identity + ".lease")).unlink()
            return corrected


def observer_directory(root: Path, relative: str) -> int:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Observer directory escaped its artifact root")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in Path(relative).parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def observer_open(workspace: Path, relative: str) -> Any:
    if (
        not isinstance(relative, str)
        or not Path(relative).parts
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("Observer artifact criterion escaped its workspace")
    descriptor = observer_directory(workspace, str(Path(relative).parent))
    try:
        target = os.open(
            Path(relative).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
        )
        try:
            import stat

            if not stat.S_ISREG(os.fstat(target).st_mode):
                raise ValueError("Observer artifact is not a regular file")
            return os.fdopen(target, "rb")
        except BaseException:
            os.close(target)
            raise
    finally:
        os.close(descriptor)


def observer_checks(
    workspace: Path, evaluation: dict[str, Any], anchor: Path | None = None
) -> list[bool]:
    checks = []
    observed: Any
    root = anchor if anchor is not None else workspace
    prefix = workspace.relative_to(anchor) if anchor is not None else Path()
    for name, expected in evaluation.get("files_equal", {}).items():
        if not isinstance(expected, str):
            raise ValueError("File criterion requires exact expected text")
        try:
            with observer_open(root, str(prefix / name)) as stream:
                initial = os.fstat(stream.fileno())
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
            if (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
                initial.st_ctime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Observer artifact changed during hashing")
            checks.append(observed == hashlib.sha256(expected.encode()).hexdigest())
        except FileNotFoundError:
            checks.append(False)
    for name, criteria in evaluation.get("json_equal", {}).items():
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("JSON criteria require nonempty explicit field mapping")
        try:
            with observer_open(root, str(prefix / name)) as stream:
                initial = os.fstat(stream.fileno())
                values = loads(stream.read().decode())
                after = os.fstat(stream.fileno())
            if (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
                initial.st_ctime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Observer JSON artifact changed during reading")
            for field, expected in criteria.items():
                observed = values
                missing = object()
                for part in field.split("."):
                    observed = (
                        observed.get(part, missing) if isinstance(observed, dict) else missing
                    )
                checks.append(observed == expected)
        except FileNotFoundError:
            checks.extend(False for _ in criteria)
    return checks


def protect_executor_supervisor() -> None:
    if sys.platform != "linux":
        raise ValueError("Environment supervisor protection requires Linux")
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(4, 0, 0, 0, 0) != 0 or library.prctl(3, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Executor supervisor cannot disable dumpability")
    subreaper = ctypes.c_int()
    if (
        library.prctl(36, 1, 0, 0, 0) != 0
        or library.prctl(37, ctypes.byref(subreaper), 0, 0, 0) != 0
        or subreaper.value != 1
    ):
        raise OSError(
            ctypes.get_errno(), "Executor supervisor cannot become a verified child subreaper"
        )


def reap_action_descendants(baseline: set[tuple[int, float]]) -> int:
    psutil = importlib.import_module("psutil")
    terminated: set[tuple[int, float]] = set()
    while True:
        children = []
        for child in psutil.Process().children(recursive=True):
            try:
                identity = (child.pid, child.create_time())
            except psutil.NoSuchProcess:
                continue
            if identity not in baseline:
                children.append(child)
                terminated.add(identity)
        if not children:
            return len(terminated)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                continue
        for child in children:
            try:
                child.wait()
            except psutil.NoSuchProcess:
                continue
        while True:
            try:
                child, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if child == 0:
                break
