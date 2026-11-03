from __future__ import annotations

import json
import platform
import sys
import threading
import uuid
from typing import Any, TextIO

from ablitbench.models import WorkerRequest, WorkerResponse
from ablitbench.provenance import implementation_identity


class WorkerBroker:
    def __init__(self, channel: TextIO, capacity: dict[str, Any]) -> None:
        self.channel = channel
        self.capacity = capacity
        self.lock = threading.Lock()

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            identity = uuid.uuid4().hex
            packet = WorkerResponse(
                isolation=platform.system().lower(),
                implementation=implementation_identity(),
                capacity=self.capacity,
                exchange={"id": identity, "request": request},
            )
            self.channel.write(packet.model_dump_json(exclude_none=True) + "\n")
            self.channel.flush()
            line = sys.stdin.readline()
            if not line:
                raise ValueError("Host broker disconnected; external delivery is unconfirmed")
            response = WorkerRequest.model_validate_json(line)
            if (
                response.operation != "exchange"
                or response.exchange is None
                or response.exchange.get("id") != identity
                or not isinstance(response.exchange.get("response"), dict)
            ):
                raise ValueError("Host broker reply does not match the pending request")
            value = dict(response.exchange["response"])
            if value.get("error"):
                raise ValueError("Host broker failed: " + json.dumps(value["error"]))
            return value
