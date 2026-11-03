from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import math
import re
import ssl
from typing import Any


async def dns_records(host: str, budget_seconds: float | None) -> dict[str, Any]:
    import dns.asyncresolver
    import dns.resolver

    if budget_seconds is None:
        raise ValueError("DNS inventory requires an explicit request_timeout budget")
    resolver = dns.asyncresolver.Resolver(configure=True)
    resolver.timeout = budget_seconds
    resolver.lifetime = budget_seconds
    records: dict[str, Any] = {}
    for kind in ("A", "AAAA", "CNAME", "NS"):
        try:
            answer = await resolver.resolve(host + ".", kind, search=False, lifetime=budget_seconds)
            records[kind] = [value.to_text().rstrip(".") for value in answer]
        except dns.resolver.NoAnswer:
            records[kind] = []
    return records


async def tls_certificate(host: str, port: int) -> str:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    _, writer = await asyncio.open_connection(host, port, ssl=context, server_hostname=host)
    try:
        connection = writer.get_extra_info("ssl_object")
        certificate = connection.getpeercert(binary_form=True)
        if certificate is None:
            raise ValueError("TLS peer did not provide a certificate")
        observation = (
            "Public certificate observation; certificate trust was not assessed\n"
            + "Protocol: "
            + str(connection.version())
            + "\n"
            + "Cipher: "
            + str(connection.cipher())
            + "\n"
            + "SHA256: "
            + hashlib.sha256(certificate).hexdigest()
            + "\n"
            + ssl.DER_cert_to_PEM_cert(certificate)
        )
    finally:
        writer.close()
        closure_error: str | None = None
        try:
            await writer.wait_closed()
        except ssl.SSLError as error:
            closure_error = str(error)
    return observation + ("\nTLS closure observation: " + closure_error if closure_error else "")


async def whois_query(server: str, domain: str) -> str:
    addresses = await asyncio.get_running_loop().getaddrinfo(server, 43, type=1)
    if not addresses or any(
        not ipaddress.ip_address(address[4][0]).is_global for address in addresses
    ):
        raise ValueError("WHOIS referral must resolve only to public registration servers")
    reader, writer = await asyncio.open_connection(str(addresses[0][4][0]), 43)
    try:
        writer.write((domain + "\r\n").encode("ascii"))
        await writer.drain()
        pieces = bytearray()
        while chunk := await reader.read(65536):
            pieces.extend(chunk)
        return pieces.decode("utf-8", errors="replace")
    finally:
        writer.close()
        await writer.wait_closed()


async def whois_registration(host: str) -> str:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("Domain registration lookup requires a DNS hostname")
    delegation = await whois_query("whois.iana.org", host.rsplit(".", 1)[-1])
    referral = re.search(r"^whois:\s*([A-Za-z0-9.-]+)\s*$", delegation, re.MULTILINE)
    if referral is None:
        raise ValueError("IANA did not publish a WHOIS service for this suffix")
    server = referral.group(1).lower()
    answer = await whois_query(server, host)
    return "IANA delegation:\n" + delegation + "\nServer: " + server + "\n" + answer


async def observe(host: str, port: int, operation: str, budget_seconds: float | None) -> Any:
    host = host.encode("idna").decode("ascii").rstrip(".")
    if not host or any(character in host for character in "\r\n\x00 /\\"):
        raise ValueError("Invalid endpoint hostname for infrastructure observation")
    if budget_seconds is not None and (
        isinstance(budget_seconds, bool) or not math.isfinite(budget_seconds) or budget_seconds <= 0
    ):
        raise ValueError("Infrastructure budget_seconds must be finite and positive")
    async with asyncio.timeout(budget_seconds):
        if operation == "dns":
            return await dns_records(host, budget_seconds)
        if operation == "tls":
            return await tls_certificate(host, port)
        if operation == "whois":
            return await whois_registration(host)
        raise ValueError("Unsupported infrastructure observation")
