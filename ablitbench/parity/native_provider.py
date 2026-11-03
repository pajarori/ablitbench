from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import ssl
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from ablitbench.models import Generation
from ablitbench.parity import providers


def origin(url: str) -> tuple[str, str, int]:
    value = urlsplit(url)
    if (
        value.scheme not in {"http", "https", "ws", "wss"}
        or not value.hostname
        or value.username
        or value.password
    ):
        raise ValueError(
            "Provider origin requires an explicit network endpoint without URL credentials"
        )
    default = 443 if value.scheme in {"https", "wss"} else 80
    return value.scheme, value.hostname.casefold(), value.port or default


def bound_url(base_url: str, locator: str) -> str:
    base = urlsplit(base_url)
    if locator.startswith("/") and not locator.startswith("//"):
        candidate = urlunsplit(
            (
                base.scheme,
                base.netloc,
                locator.split("?", 1)[0],
                locator.split("?", 1)[1] if "?" in locator else "",
                "",
            )
        )
    else:
        candidate = locator
    if origin(candidate) != origin(base_url):
        raise ValueError("Native provider attempted a cross-origin followup; credentials retained")
    value = urlsplit(candidate)
    if value.fragment:
        raise ValueError("Provider followup URL cannot contain a fragment")
    return candidate


def credentials(
    environment: dict[str, str], base_url: str, bindings: dict[str, str] | None = None
) -> dict[str, str]:
    result = {}
    for role, name in environment.items():
        if not isinstance(name, str) or not name or not name.replace("_", "A").isalnum():
            raise ValueError("Provider credential bindings require environment variable names")
        if bindings is not None and role in bindings and origin(bindings[role]) != origin(base_url):
            raise ValueError("Provider credential origin does not match configured endpoint")
        value = os.environ.get(name)
        if value is None:
            raise ValueError(
                "Required provider credential environment variable is missing: " + name
            )
        result[role] = value
    return result


def sigv4(
    method: str,
    url: str,
    body: bytes,
    region: str,
    values: dict[str, str],
    now: datetime | None = None,
    service: str = "bedrock",
    double_uri_encode: bool = True,
) -> dict[str, str]:
    access, secret = values.get("access_key_id"), values.get("secret_access_key")
    if not access or not secret or not region:
        raise ValueError("AWS signing requires explicitly bound access key, secret key and region")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    stamp, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    parsed = urlsplit(url)
    origin(url)
    headers = {
        "host": parsed.netloc,
        "x-amz-date": stamp,
        "x-amz-content-sha256": hashlib.sha256(body).hexdigest(),
    }
    token = values.get("session_token")
    if token:
        headers["x-amz-security-token"] = token
    path = quote(parsed.path or "/", safe="/~") if double_uri_encode else parsed.path or "/"
    query = "&".join(
        key + "=" + value
        for key, value in sorted(
            (quote(key, safe="-_.~"), quote(value, safe="-_.~"))
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        )
    )
    names = ";".join(sorted(headers))
    canonical_headers = "".join(
        key + ":" + " ".join(headers[key].split()) + "\n" for key in sorted(headers)
    )
    canonical = "\n".join(
        [method.upper(), path, query, canonical_headers, names, headers["x-amz-content-sha256"]]
    )
    scope = "/".join([date, region, service, "aws4_request"])
    signing = "\n".join(
        ["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = ("AWS4" + secret).encode()
    for component in (date, region, service, "aws4_request"):
        key = hmac.new(key, component.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, signing.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        "AWS4-HMAC-SHA256 Credential="
        + access
        + "/"
        + scope
        + ", SignedHeaders="
        + names
        + ", Signature="
        + signature
    )
    return headers


def authorization(
    protocol: str,
    method: str,
    url: str,
    body: bytes,
    key: str | None,
    values: dict[str, str],
    options: dict[str, Any],
) -> dict[str, str]:
    origin(url)
    mode = options.get("auth_type")
    if mode == "none":
        return {}
    if mode == "api-key":
        return {options.get("api_key_header", "api-key"): key} if key else {}
    if mode == "basic":
        if not values.get("username") or not key:
            raise ValueError("Basic authentication requires explicitly bound username/password")
        return {
            "authorization": "Basic "
            + base64.b64encode((values["username"] + ":" + key).encode()).decode()
        }
    if protocol == "anthropic-messages":
        return {"x-api-key": key} if key else {}
    if protocol == "bedrock-converse":
        return sigv4(
            method,
            url,
            body,
            options["region"],
            values,
            double_uri_encode=options.get("double_uri_encode", True),
        )
    if protocol == "watsonx-generation" and values.get("access_token"):
        return {"authorization": "Bearer " + values["access_token"]}
    if protocol in {"rest", "langserve"}:
        return {}
    if protocol == "replicate" or protocol == "replicate-deployment":
        return {"authorization": "Bearer " + key} if key else {}
    if protocol == "websocket":
        mode = options.get("auth_type", "none")
        if mode == "basic":
            if not key or not values.get("username"):
                raise ValueError("WebSocket basic authentication requires bound username/password")
            pair = (values["username"] + ":" + key).encode()
            return {"authorization": "Basic " + base64.b64encode(pair).decode()}
        if mode == "none" or mode == "custom":
            return {}
        if mode != "bearer":
            raise ValueError("Unknown native WebSocket authentication mode")
    return {"authorization": "Bearer " + key} if key else {}


def iam_request(iam_url: str, api_key: str, binding: str) -> dict[str, Any]:
    if origin(iam_url) != origin(binding):
        raise ValueError("IAM credential origin does not match explicitly bound token endpoint")
    return {
        "method": "POST",
        "url": iam_url,
        "headers": {
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        },
        "content": urlencode(
            {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
        ).encode(),
        "stream_format": "json",
    }


def iam_token(payload: Any) -> tuple[str, float | None]:
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("access_token"), str)
        or not payload["access_token"]
    ):
        raise ValueError("IAM token response lacks an access token")
    expires = payload.get("expires_in")
    if expires is not None and (
        not isinstance(expires, int | float) or isinstance(expires, bool) or expires <= 0
    ):
        raise ValueError("IAM token expiry is malformed")
    return payload["access_token"], float(expires) if expires is not None else None


def render_secret(value: Any, key: str | None, json_escape: bool = False) -> Any:
    if isinstance(value, str):
        if "$KEY" in value:
            if key is None:
                raise ValueError("Native request template requires a bound credential")
            replacement = json.dumps(key)[1:-1] if json_escape else key
            return value.replace("$KEY", replacement)
        return value
    if isinstance(value, dict):
        return {name: render_secret(item, key, json_escape) for name, item in value.items()}
    if isinstance(value, list):
        return [render_secret(item, key, json_escape) for item in value]
    return value


def prepare_request(
    base_url: str,
    spec: dict[str, Any],
    protocol: str,
    key: str | None,
    values: dict[str, str],
    options: dict[str, Any],
) -> dict[str, Any]:
    url = spec.get("url") or providers.endpoint(base_url, spec.get("path", ""))
    url = bound_url(base_url, url)
    parameters = render_secret(spec.get("params", {}), key)
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.extend(parameters.items())
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), ""))
    headers = render_secret(spec.get("headers", {}), key)
    if "json" in spec:
        body = json.dumps(
            render_secret(spec["json"], key),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    else:
        body = render_secret(spec.get("content", ""), key, json_escape=True).encode()
    headers.update(
        authorization(protocol, spec.get("method", "POST"), url, body, key, values, options)
    )
    return {
        "method": spec.get("method", "POST"),
        "url": url,
        "content": body,
        "headers": headers,
        "stream_format": spec.get("stream_format", "json"),
        "coverage": spec.get("coverage", []),
    }


def tls_context(options: dict[str, Any], values: dict[str, str]) -> ssl.SSLContext:
    verify = options.get("verify_ssl", True)
    context = ssl.create_default_context(cafile=values.get("ca_file"))
    if verify is False:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    elif verify is not True:
        raise ValueError("Custom CA files require an explicit credential environment binding")
    certificate, key = values.get("client_cert"), values.get("client_key")
    if key and not certificate:
        raise ValueError("mTLS private key requires an explicitly bound certificate")
    if certificate:
        if not Path(certificate).is_file() or (key is not None and not Path(key).is_file()):
            raise ValueError("Bound mTLS credential file is unavailable")
        context.load_cert_chain(
            certificate, keyfile=key, password=values.get("client_key_passphrase")
        )
    return context


def followup(
    protocol: str,
    payload: Any,
    status: int,
    headers: dict[str, str],
    options: dict[str, Any],
    base_url: str,
) -> dict[str, Any] | None:
    if protocol.startswith("replicate"):
        if not isinstance(payload, dict):
            raise ValueError("Malformed prediction response")
        state = payload.get("status")
        if state == "succeeded":
            return None
        if state in {"failed", "canceled"}:
            raise ValueError("Native prediction failed or was canceled; not retried")
        if state not in {"starting", "processing"}:
            raise ValueError("Unknown native prediction lifecycle state")
        locator = payload.get("urls", {}).get("get")
        if locator is None and isinstance(payload.get("id"), str):
            locator = "/v1/predictions/" + quote(payload["id"], safe="")
        if not isinstance(locator, str):
            raise ValueError("Prediction has no durable status locator")
        return {
            "method": "GET",
            "url": bound_url(base_url, locator),
            "headers": {},
            "stream_format": "json",
        }
    if protocol.startswith("nvcf-") and status == 202:
        request_id = next(
            (value for key, value in headers.items() if key.casefold() == "nvcf-reqid"), None
        )
        if not request_id:
            raise ValueError("NVCF202 response lacks request identity")
        path = options.get("status_path", "/v2/nvcf/pexec/status/")
        if not isinstance(path, str) or not path.startswith("/") or "?" in path:
            raise ValueError("NVCF status path must be an origin-relative path prefix")
        return {
            "method": "GET",
            "url": bound_url(base_url, path + quote(request_id, safe="")),
            "headers": {},
            "stream_format": "json",
        }
    return None


def cancellation(protocol: str, payload: Any, base_url: str) -> dict[str, Any] | None:
    if not protocol.startswith("replicate") or not isinstance(payload, dict):
        return None
    locator = payload.get("urls", {}).get("cancel")
    if locator is None and isinstance(payload.get("id"), str):
        locator = "/v1/predictions/" + quote(payload["id"], safe="") + "/cancel"
    if not isinstance(locator, str):
        return None
    return {
        "method": "POST",
        "url": bound_url(base_url, locator),
        "headers": {},
        "stream_format": "json",
    }


async def poll(
    protocol: str,
    first: dict[str, Any],
    send: Callable[[dict[str, Any]], Awaitable[tuple[int, dict[str, str], Any]]],
    options: dict[str, Any],
    base_url: str,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> Generation:
    request = first
    while True:
        status, headers, payload = await send(request)
        if checkpoint is not None:
            checkpoint(
                {
                    "protocol": protocol,
                    "status": status,
                    "prediction_id": payload.get("id") if isinstance(payload, dict) else None,
                    "request_id": next(
                        (value for key, value in headers.items() if key.casefold() == "nvcf-reqid"),
                        None,
                    ),
                    "cancel": cancellation(protocol, payload, base_url),
                }
            )
        if status >= 300:
            raise ValueError(
                "Native provider HTTP error; generation POST is never automatically replayed"
            )
        next_request = followup(protocol, payload, status, headers, options, base_url)
        if next_request is None:
            return providers.decode(protocol, payload, options)
        interval = options["poll_interval"]
        if interval:
            await asyncio.sleep(interval)
        else:
            await asyncio.sleep(0)
        request = next_request


class WebSocketSession:
    def __init__(
        self,
        base_url: str,
        options: dict[str, Any],
        key: str | None = None,
        values: dict[str, str] | None = None,
    ):
        self.base_url = base_url
        self.options = providers.configuration("websocket", options)
        if urlsplit(base_url).scheme not in {"ws", "wss"}:
            raise ValueError("Native WebSocket provider requires ws/wss endpoint")
        self.key, self.values = key, dict(values or {})
        self.connection: Any = None
        self.lock = asyncio.Lock()

    async def connect(self, headers: dict[str, str]) -> None:
        import importlib

        module = importlib.import_module("websockets.asyncio.client")

        def redirect(connection: Any, error: Exception) -> Exception:
            return ValueError("WebSocket redirects are not authorized by the configured origin")

        no_redirect = type(
            "BoundWebSocketConnect", (module.connect,), {"process_redirect": redirect}
        )

        options: dict[str, Any] = {
            "additional_headers": headers,
            "open_timeout": self.options.get("connect_timeout"),
            "close_timeout": self.options.get("connect_timeout"),
            "ping_interval": None,
            "ping_timeout": None,
            "max_size": None,
            "proxy": self.options.get("proxy"),
            "user_agent_header": None,
            "subprotocols": self.options.get("subprotocols"),
        }
        if urlsplit(self.base_url).scheme == "wss":
            options["ssl"] = tls_context(self.options, self.values)
        elif any(self.values.get(key) for key in ("client_cert", "client_key", "ca_file")):
            raise ValueError("WebSocket certificate credentials require a TLS endpoint")
        self.connection = await no_redirect(self.base_url, **options)

    async def generate(
        self, spec: dict[str, Any], capture: Callable[[dict[str, Any]], None] | None = None
    ) -> Generation:
        from ablitbench.formats.json import loads

        prepared = prepare_request(
            self.base_url, spec, "websocket", self.key, self.values, self.options
        )
        async with self.lock:
            if self.connection is None:
                await self.connect(prepared["headers"])
            result = Generation()
            typing = False
            received = 0
            try:
                async with asyncio.timeout(self.options.get("request_timeout")):
                    outgoing = prepared["content"].decode("utf-8")
                    if capture:
                        capture({"direction": "send", "text": outgoing})
                    await self.connection.send(outgoing)
                    while True:
                        frame = await self.connection.recv()
                        if capture:
                            capture(
                                {"direction": "recv", "text": frame}
                                if isinstance(frame, str)
                                else {
                                    "direction": "recv",
                                    "binary": base64.b64encode(frame).decode(),
                                }
                            )
                        if not isinstance(frame, str):
                            raise ValueError(
                                "WebSocket binary output requires an explicit typed media adapter"
                            )
                        indicator = self.options.get("typing_indicator")
                        if (
                            self.options.get("response_after_typing")
                            and indicator
                            and indicator in frame
                        ):
                            typing = True
                            continue
                        payload = loads(frame) if self.options.get("response_json", True) else frame
                        terminal = False
                        field = self.options.get("termination_field")
                        if field:
                            if not isinstance(payload, dict):
                                raise ValueError(
                                    "WebSocket terminal selector requires JSON response"
                                )
                            if field.startswith("$"):
                                import importlib

                                matches = (
                                    importlib.import_module("jsonpath_ng")
                                    .parse(field)
                                    .find(payload)
                                )
                                terminal = any(
                                    match.value == self.options.get("termination_value", True)
                                    for match in matches
                                )
                            else:
                                terminal = payload.get(field) == self.options.get(
                                    "termination_value", True
                                )
                        else:
                            terminal = not self.options.get("response_after_typing") or typing
                        part = providers.decode("websocket", payload, self.options)
                        result.text += part.text
                        received += len(part.text)
                        if self.options.get("response_finish_field"):
                            result.finish = part.finish
                        maximum = self.options.get("max_response_length")
                        if maximum is not None:
                            if type(maximum) is not int or maximum <= 0:
                                raise ValueError(
                                    "Explicit WebSocket response budget must be positive"
                                )
                            if received >= maximum and not terminal:
                                result.finish = "length"
                                providers.observation(
                                    result,
                                    "operator-response-budget",
                                    "Complete received frames retained; later frames uninspected",
                                    "medium",
                                )
                                await self.close(abort=True)
                                return result
                        if terminal:
                            if result.finish is None:
                                providers.finish(result, None)
                            return result
            except BaseException:
                await self.close(abort=True)
                raise

    async def close(self, abort: bool = False) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            if abort:
                connection.transport.abort()
            else:
                await connection.close()


class EventStream:
    def __init__(self) -> None:
        self.buffer = bytearray()

    @staticmethod
    def headers(data: bytes) -> dict[str, Any]:
        import struct

        result = {}
        position = 0
        widths = {2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}
        while position < len(data):
            size = data[position]
            position += 1
            if not size or position + size + 1 > len(data):
                raise ValueError("Malformed AWS event header name")
            name = data[position : position + size].decode("utf-8")
            position += size
            kind = data[position]
            position += 1
            if name in result:
                raise ValueError("Duplicate AWS event header")
            if kind in {0, 1}:
                value: Any = kind == 0
            elif kind in {6, 7}:
                if position + 2 > len(data):
                    raise ValueError("Truncated AWS variable header length")
                length = struct.unpack_from(">H", data, position)[0]
                position += 2
                if position + length > len(data):
                    raise ValueError("Truncated AWS variable header value")
                value = data[position : position + length]
                if kind == 7:
                    value = value.decode("utf-8")
                position += length
            elif kind in widths:
                length = widths[kind]
                if position + length > len(data):
                    raise ValueError("Truncated AWS fixed header value")
                raw = data[position : position + length]
                value = raw if kind == 9 else int.from_bytes(raw, "big", signed=True)
                position += length
            else:
                raise ValueError("Unknown AWS event header type")
            result[name] = value
        return result

    def accept(self, chunk: bytes) -> list[tuple[str, str]]:
        import struct
        import zlib

        self.buffer.extend(chunk)
        result = []
        while len(self.buffer) >= 12:
            total, header_length, prelude_crc = struct.unpack_from(">III", self.buffer)
            if total < 16 or header_length > total - 16:
                raise ValueError("Invalid AWS event message/header lengths")
            if zlib.crc32(self.buffer[:8]) & 0xFFFFFFFF != prelude_crc:
                raise ValueError("AWS event prelude CRC mismatch")
            if len(self.buffer) < total:
                break
            frame = bytes(self.buffer[:total])
            del self.buffer[:total]
            expected = struct.unpack_from(">I", frame, total - 4)[0]
            if zlib.crc32(frame[:-4]) & 0xFFFFFFFF != expected:
                raise ValueError("AWS event message CRC mismatch")
            headers = self.headers(frame[12 : 12 + header_length])
            kind = headers.get(":message-type")
            if kind in {"exception", "error"}:
                raise ValueError("AWS event stream returned a service exception")
            if kind != "event" or not isinstance(headers.get(":event-type"), str):
                raise ValueError("AWS event lacks native event type")
            media = headers.get(":content-type", "application/json")
            if media != "application/json":
                raise ValueError("AWS Converse event payload is not JSON")
            result.append((headers[":event-type"], frame[12 + header_length : -4].decode("utf-8")))
        return result

    def finish(self) -> None:
        if self.buffer:
            raise ValueError("AWS event stream disconnected within a frame")
