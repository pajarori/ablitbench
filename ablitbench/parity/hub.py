from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import math
import os
import re
import tempfile
import uuid
from collections.abc import Collection
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlsplit, urlunsplit

import httpx2 as httpx

from ablitbench.formats.json import loads
from ablitbench.formats.snapshot import secure_open
from ablitbench.io import operation
from ablitbench.models import Evidence, ResolvedTarget
from ablitbench.privacy import redact_text
from ablitbench.response_spool import Capture
from ablitbench.store import Store


def native_limits(timeout_seconds: float) -> dict[str, Any]:
    from ablitbench.parity.sources import source_spec

    specification = source_spec("layerfault")
    return {
        "source_revision": specification["revision"],
        "source_entrypoint": "src/hub/client.rs:HubClient::new/get_following/list_models",
        "source_sha256": specification["files"]["src/hub/client.rs"]["sha256"],
        "models_api_page_size": 1000,
        "download_bytes": 214748364800,
        "metadata_bytes": 33554432,
        "files": 200000,
        "redirects": 6,
        "retries": 5,
        "connect_timeout_seconds": 15,
        "request_timeout_seconds": timeout_seconds,
        "range_bytes": 33554432,
    }


async def public_addresses(host: str, budget_seconds: float) -> list[str]:
    import dns.asyncresolver
    import dns.resolver

    resolver = dns.asyncresolver.Resolver(configure=True)

    async def query(kind: str) -> list[str]:
        try:
            answer = await resolver.resolve(host + ".", kind, lifetime=budget_seconds, search=False)
            return [value.to_text() for value in answer]
        except dns.resolver.NoAnswer:
            return []

    rows = await asyncio.gather(query("A"), query("AAAA"))
    values = {value for row in rows for value in row}
    if not values or any(not ipaddress.ip_address(value).is_global for value in values):
        raise ValueError("Hugging Face DNS must resolve exclusively to public addresses")
    return sorted(values, key=lambda value: (ipaddress.ip_address(value).version, value))


class PublicTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.transports: dict[tuple[str, str], httpx.AsyncHTTPTransport] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        timeout = min(15.0, float(request.extensions["hub_timeout"]))
        addresses = await public_addresses(host, timeout)
        last: Exception | None = None
        for address in addresses:
            key = host, address
            if key not in self.transports:
                self.transports[key] = httpx.AsyncHTTPTransport(
                    trust_env=False,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                )
            transport = self.transports[key]
            headers = dict(request.headers)
            headers["host"] = host
            pinned = httpx.Request(
                request.method,
                request.url.copy_with(host=address),
                headers=headers,
                stream=request.stream,
                extensions={**request.extensions, "sni_hostname": host},
            )
            try:
                return await transport.handle_async_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                last = error
        raise ValueError("Public Hugging Face connection failed") from last

    async def aclose(self) -> None:
        for transport in self.transports.values():
            await transport.aclose()
        self.transports.clear()


def valid_repo(value: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)?", value)
    )


def valid_url(value: str) -> Any:
    if any(ord(character) < 32 for character in value) or "\\" in value:
        raise ValueError("Invalid Hugging Face URL")
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.fragment
        or parsed.port not in {None, 443}
        or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")))
    ):
        raise ValueError("Hugging Face requests require a validated HTTPS origin")
    if any(part in {".", ".."} or "/" in part for part in map(unquote, parsed.path.split("/"))):
        raise ValueError("Hugging Face paths cannot traverse or encode separators")
    return parsed


def revision(path: Path) -> dict[str, str]:
    with secure_open(path) as stream:
        value = loads(stream.read())
    commit = value.get("sha") if isinstance(value, dict) else None
    if not isinstance(commit, str) or re.fullmatch(r"[a-fA-F0-9]{40}", commit) is None:
        raise ValueError("Hugging Face metadata lacks an immutable commit")
    return {"revision": commit.lower()}


class HubBroker:
    def __init__(
        self,
        resolved: ResolvedTarget,
        store: Store,
        permissions: Collection[str],
        allow_public_crawl: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        offline: bool = False,
    ) -> None:
        if type(offline) is not bool:
            raise ValueError("Offline mode must be boolean")
        self.offline = offline
        self.store = store
        self.permissions = set(permissions)
        self.allow_public_crawl = allow_public_crawl
        self.bound_repo: str | None = None
        self.bound_alias: str | None = None
        self.commits: dict[str, str] = {}
        if resolved.target.location.startswith("hf://"):
            repo = resolved.target.location.removeprefix("hf://").strip("/")
            commit = resolved.revision
            if not valid_repo(repo) or not commit or not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
                raise ValueError("Hugging Face source must be resolved to its immutable commit")
            self.bound_repo = repo
            self.bound_alias = resolved.target.revision or "main"
            self.commits[repo] = commit.lower()
        self.secret = os.environ.get("HF_TOKEN")
        self.aliases: dict[str, tuple[str, str, str | None, bool]] = {}
        self.handles: dict[str, Any] = {}
        self.private = tempfile.TemporaryDirectory(prefix="ablib-hub-private-")
        self.private_root = Path(self.private.name).resolve(strict=True)
        self.client = httpx.AsyncClient(
            transport=transport or PublicTransport(),
            trust_env=False,
            timeout=None,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )

    def scope(self, url: str) -> tuple[str, str, str | None, bool]:
        if url in self.aliases:
            return self.aliases[url]
        parsed = valid_url(url)
        if parsed.hostname != "huggingface.co":
            raise ValueError("CDN requests require a previously observed authorized redirect")
        query = dict(parse_qsl(parsed.query, strict_parsing=True, keep_blank_values=True))
        if len(query) != len(parse_qsl(parsed.query, keep_blank_values=True)):
            raise ValueError("Duplicate Hugging Face query keys are forbidden")
        if parsed.path == "/api/models":
            if not self.allow_public_crawl or set(query) - {
                "limit",
                "full",
                "sort",
                "direction",
                "cursor",
            }:
                raise ValueError("Public Hugging Face crawl scope is not authorized")
            if (
                query.get("full", "true") != "true"
                or query.get("sort", "lastModified") != "lastModified"
                or query.get("direction", "-1") != "-1"
            ):
                raise ValueError("Unsupported public Hugging Face crawl query")
            if "limit" in query and (
                not query["limit"].isdigit() or not 1 <= int(query["limit"]) <= 1000
            ):
                raise ValueError("Hugging Face native crawl limit is outside its protocol scope")
            return url, "crawl", None, False
        parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
        if parts[:2] == ["api", "models"] and len(parts) in {5, 6} and parts[-2] == "revision":
            repo, requested = "/".join(parts[2:-2]), parts[-1]
            if not valid_repo(repo) or set(query) - {"blobs"} or query.get("blobs") != "true":
                raise ValueError("Unsupported Hugging Face metadata route")
            if repo == self.bound_repo:
                commit = self.commits[repo]
                if requested not in {commit, self.bound_alias}:
                    raise ValueError("Metadata revision differs from the configured source")
                encoded = (
                    "/api/models/"
                    + "/".join(quote(part, safe="") for part in repo.split("/"))
                    + "/revision/"
                    + commit
                )
                return (
                    urlunsplit(("https", "huggingface.co", encoded, "blobs=true", "")),
                    "metadata",
                    repo,
                    True,
                )
            if not self.allow_public_crawl:
                raise ValueError("Metadata repository differs from the configured source")
            if repo in self.commits and requested not in {self.commits[repo], "main"}:
                raise ValueError("Public metadata revision differs from its pinned observation")
            return url, "metadata", repo, False
        if "resolve" in parts:
            position = parts.index("resolve")
            repo = "/".join(parts[:position])
            if position not in {1, 2} or len(parts) <= position + 2 or set(query) - {"download"}:
                raise ValueError("Unsupported Hugging Face artifact route")
            if not valid_repo(repo) or self.commits.get(repo) != parts[position + 1]:
                raise ValueError("Artifact download requires a previously pinned source revision")
            return url, "artifact", repo, repo == self.bound_repo
        raise ValueError("Hugging Face URL is outside the declared native API/artifact scope")

    def public_headers(
        self, headers: httpx.Headers, url: str, scope: tuple[str, str, str | None, bool]
    ) -> dict[str, str]:
        result = {}
        for name in (
            "content-length",
            "content-type",
            "content-range",
            "etag",
            "retry-after",
            "x-linked-etag",
            "x-repo-commit",
        ):
            if name in headers:
                result[name] = redact_text(headers[name], [self.secret] if self.secret else [])
        if "location" in headers:
            location = urljoin(url, headers["location"])
            valid_url(location)
            alias = "https://huggingface.co/.ablitbench/redirect/" + uuid.uuid4().hex
            destination = urlsplit(location)
            if destination.hostname == "huggingface.co":
                redirected = self.scope(location)
                if redirected[1:3] != scope[1:3]:
                    raise ValueError("Redirect changed the declared Hugging Face repository scope")
                self.aliases[alias] = redirected
            else:
                self.aliases[alias] = (location, scope[1], scope[2], False)
            result["location"] = alias
        if "link" in headers:
            links = []
            for part in headers["link"].split(","):
                match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="next"\s*', part)
                if match:
                    location = urljoin(url, match[1])
                    actual = self.scope(location)
                    alias = "https://huggingface.co/api/models?ablib_cursor=" + uuid.uuid4().hex
                    self.aliases[alias] = actual
                    links.append("<" + alias + '>; rel="next"')
            if links:
                result["link"] = ", ".join(links)
        return result

    async def observe(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("kind") != "hub":
            raise ValueError("Unknown Hugging Face broker request kind")
        if self.offline:
            raise ValueError("Offline invocation forbids Hugging Face network access")
        if "external-network" not in self.permissions:
            raise ValueError("Hugging Face access requires current external-network permission")
        action = request.get("action")
        if action in {"read", "close"}:
            if (
                set(request) - {"kind", "action", "handle"}
                or request.get("handle") not in self.handles
            ):
                raise ValueError("Unknown or out-of-scope Hugging Face response handle")
            stream = self.handles[request["handle"]]
            if action == "close":
                stream.close()
                del self.handles[request["handle"]]
                return {"closed": True}
            chunk = stream.read(1024 * 1024)
            return {"body_base64": base64.b64encode(chunk).decode(), "eof": not chunk}
        if action != "open" or set(request) - {
            "kind",
            "action",
            "url",
            "timeout_seconds",
            "max_response_bytes",
            "range",
        }:
            raise ValueError("Unknown Hugging Face broker operation")
        timeout, cap = request.get("timeout_seconds"), request.get("max_response_bytes")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
            or type(cap) is not int
            or cap <= 0
        ):
            raise ValueError(
                "Hugging Face requests require explicit native timeout and byte budget"
            )
        url = request.get("url")
        if not isinstance(url, str):
            raise ValueError("Hugging Face request URL is required")
        scope = self.scope(url)
        actual, kind, repo, authenticated = scope
        headers = {"accept-encoding": "identity", "user-agent": "ablitbench-layerfault-broker"}
        if "range" in request:
            bounds = request["range"]
            if (
                kind != "artifact"
                or not isinstance(bounds, dict)
                or set(bounds) != {"offset", "length"}
            ):
                raise ValueError("Byte ranges require a pinned artifact and typed bounds")
            offset, length = bounds["offset"], bounds["length"]
            if (
                type(offset) is not int
                or type(length) is not int
                or offset < 0
                or not 0 < length <= 33554432
                or offset + length - 1 > 18446744073709551615
            ):
                raise ValueError("Byte range exceeds the original native protocol guard")
            headers["range"] = f"bytes={offset}-{offset + length - 1}"
        if authenticated and self.secret:
            headers["authorization"] = "Bearer " + self.secret
        capture = Capture(self.private_root / "traffic-spool")
        status = 0
        public: dict[str, str] = {}
        complete = False
        reflected = False
        tail = b""
        patterns = (
            list(
                {
                    self.secret.encode(),
                    base64.b64encode(self.secret.encode()),
                    self.secret.encode().hex().encode(),
                    quote(self.secret, safe="").encode(),
                }
            )
            if self.secret
            else []
        )
        try:
            async with asyncio.timeout(timeout):
                async with self.client.stream(
                    "GET",
                    actual,
                    headers=headers,
                    extensions={
                        "hub_timeout": timeout,
                        "timeout": {
                            "connect": min(15.0, timeout),
                            "read": None,
                            "write": None,
                            "pool": None,
                        },
                    },
                ) as response:
                    status = response.status_code
                    public = self.public_headers(response.headers, actual, scope)
                    reflected = (
                        any(
                            redact_text(value, [self.secret]) != value
                            for value in response.headers.values()
                        )
                        if self.secret
                        else False
                    )
                    async for chunk in response.aiter_bytes():
                        window = tail + chunk
                        reflected = reflected or any(pattern in window for pattern in patterns)
                        tail = (
                            window[-max((len(pattern) - 1 for pattern in patterns), default=0) :]
                            if patterns
                            else b""
                        )
                        capture.write(chunk)
                        if capture.size > cap:
                            raise ValueError(
                                "Observed Hugging Face response exceeds native byte budget"
                            )
                    complete = True
        finally:
            capture.close()
            metadata = {
                "broker": "hub",
                "status": status,
                "source_url_sha256": hashlib.sha256(actual.encode()).hexdigest(),
                "headers": public,
                "native_max_response_bytes": cap,
                "native_limits": native_limits(timeout),
                "original_sha256": capture.checksum.hexdigest(),
                "original_captured_bytes": capture.size,
                "transport_complete": complete,
                "credential_reflected": reflected,
                "body_encoding": "decoded HTTP entity",
            }
            try:
                if reflected or not complete:
                    reference = self.store.evidence(
                        {**metadata, "body_withheld": True, "complete": False}
                    )
                    evidence = [reference]
                    published_path = None
                else:
                    prepared, stable = await operation(
                        "capture", [str(capture.path), str(self.store.blobs)]
                    )
                    body = Evidence.model_validate(prepared)
                    if (
                        not stable
                        or body.sha256 != capture.checksum.hexdigest()
                        or body.size != capture.size
                    ):
                        raise ValueError("Private Hub response changed before safe publication")
                    self.store.commit_blob(body)
                    reference = self.store.evidence(
                        {**metadata, "response_body": body.sha256, "complete": True}
                    )
                    evidence = [reference, body.sha256]
                    published_path = self.store.blobs / body.sha256
            finally:
                capture.path.unlink(missing_ok=True)
        public["content-length"] = str(capture.size)
        if reflected:
            return {
                "status": 403,
                "headers": {},
                "bytes": 0,
                "evidence": evidence,
                "error": "Credential-reflected response withheld",
            }
        if kind == "metadata" and status == 200:
            observed = await operation("hub_revision", [str(published_path)])
            commit = observed["revision"]
            if repo is not None and repo == self.bound_repo and commit != self.commits[repo]:
                raise ValueError("Hugging Face metadata drifted from its configured commit")
            if repo is not None:
                self.commits[repo] = commit
        handle = uuid.uuid4().hex
        if published_path is None:
            raise ValueError("Incomplete Hugging Face response was withheld")
        self.handles[handle] = secure_open(published_path)
        return {
            "status": status,
            "headers": public,
            "handle": handle,
            "bytes": capture.size,
            "evidence": evidence,
            "native_limits": native_limits(timeout),
            "credential_scope": "configured HF repository only"
            if authenticated
            else "public unauthenticated",
        }

    async def close(self) -> None:
        for stream in self.handles.values():
            stream.close()
        self.handles.clear()
        try:
            await self.client.aclose()
        finally:
            self.private.cleanup()
