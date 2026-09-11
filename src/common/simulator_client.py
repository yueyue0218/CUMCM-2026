"""Reliable serial client for the CUMCM 2026 B simulator.

This module contains transport and protocol state only.  Search, localization,
route planning, and stopping policies belong in ``src/q3`` and ``src/q4``.

Protocol invariants from the official attachments:

* Actions are strictly serial: one full HTTP response precedes the next action.
* Every new action gets a new request_id.  A retry reuses the exact path, body,
  and request_id only when a network failure makes the outcome unknown.
* Both the HTTP status and ``accepted`` must indicate success.  In particular,
  ``HTTP 200 + accepted=false`` did not execute the action; its
  ``virtual_time_s=0`` must not overwrite local state.
* ``no_signal`` is ambiguous (absent/cleared, out of range, or outside a
  directional source's coverage) and is represented only as an observation.
* ``near`` has no bearing.  Only ``direction`` permits reading ``svd_deg``.
* ``/clear`` has a 20 m radius and does not change the detector channel.
* The strategy must use the successful ``/enter`` response's
  ``remaining_real_duration_s`` instead of assuming 1200 seconds remain.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

JsonObject = dict[str, object]
Transport = Callable[[str, bytes, float], tuple[int, bytes]]


class SimulatorError(RuntimeError):
    """Base exception for simulator transport or protocol failures."""


class SimulatorHTTPError(SimulatorError):
    def __init__(self, status: int, response: Mapping[str, object] | None = None):
        super().__init__(f"simulator returned HTTP {status}: {response}")
        self.status = status
        self.response = response


class ActionRejected(SimulatorError):
    def __init__(self, response: Mapping[str, object]):
        super().__init__(f"simulator rejected action: {response}")
        self.response = dict(response)


class ProtocolError(SimulatorError):
    """The simulator returned a malformed or contradictory response."""


class RequestIdSequence:
    """Thread-safe unique IDs for new actions within one test session."""

    def __init__(self, prefix: str = "robot") -> None:
        if not prefix:
            raise ValueError("request ID prefix must not be empty")
        self._prefix = prefix
        self._counter = 0
        self._lock = threading.Lock()

    def next(self, action: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{self._prefix}-{action}-{self._counter:06d}"


@dataclass(frozen=True)
class MeasureResult:
    result: str
    svd_deg: float | None
    response: JsonObject


@dataclass(frozen=True)
class ClearResult:
    result: str
    response: JsonObject


@dataclass
class SimulatorState:
    entered: bool = False
    exited: bool = False
    position: tuple[float, float] = (0.0, 0.0)
    current_channel: int = 1
    virtual_time_s: float = 0.0
    remaining_real_duration_s: int | None = None
    real_deadline_monotonic: float | None = None


class JsonlRunLogger:
    """Write replayable request/response logs and run metadata."""

    def __init__(self, run_directory: str | Path, config: Mapping[str, object]):
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._write_json("config.json", dict(config))

    def _write_json(self, filename: str, value: Mapping[str, object]) -> None:
        path = self.run_directory / filename
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def append(self, filename: str, value: Mapping[str, object]) -> None:
        path = self.run_directory / filename
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    def request(self, value: Mapping[str, object]) -> None:
        sanitized = dict(value)
        payload = sanitized.get("payload")
        if isinstance(payload, dict):
            sanitized["payload"] = {**payload, "robot_id": "<redacted>"}
        self.append("requests.jsonl", sanitized)

    def response(self, value: Mapping[str, object]) -> None:
        self.append("responses.jsonl", value)

    def finish(self, summary: Mapping[str, object]) -> None:
        self._write_json("summary.json", dict(summary))


def _urllib_transport(url: str, body: bytes, timeout_s: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class SimulatorClient:
    """State-aware, serial simulator protocol client."""

    _NETWORK_ERRORS = (
        TimeoutError,
        ConnectionError,
        socket.timeout,
        urllib.error.URLError,
    )

    def __init__(
        self,
        robot_id: str,
        *,
        base_url: str = "http://127.0.0.1:2026",
        arena_id: str = "default",
        timeout_s: float = 5.0,
        max_network_retries: int = 2,
        retry_backoff_s: float = 0.2,
        transport: Transport | None = None,
        request_ids: RequestIdSequence | None = None,
        logger: JsonlRunLogger | None = None,
    ) -> None:
        if not robot_id:
            raise ValueError("robot_id must be the current logged-in team ID")
        if max_network_retries < 0:
            raise ValueError("max_network_retries must be non-negative")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.arena_id = arena_id
        self.timeout_s = timeout_s
        self.max_network_retries = max_network_retries
        self.retry_backoff_s = retry_backoff_s
        self.transport = transport or _urllib_transport
        self.request_ids = request_ids or RequestIdSequence()
        self.logger = logger
        self.state = SimulatorState()
        self._action_lock = threading.Lock()

    def _payload(self, request_id: str, **fields: object) -> JsonObject:
        return {
            "arena_id": self.arena_id,
            "robot_id": self.robot_id,
            "request_id": request_id,
            **fields,
        }

    def _perform(self, path: str, payload: JsonObject) -> JsonObject:
        # Holding this lock through the complete response enforces serial actions.
        with self._action_lock:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            for attempt in range(self.max_network_retries + 1):
                if self.logger:
                    self.logger.request(
                        {
                            "path": path,
                            "attempt": attempt + 1,
                            "payload": payload,
                            "real_timestamp_ms": time.time_ns() // 1_000_000,
                        }
                    )
                try:
                    status, response_body = self.transport(
                        self.base_url + path, body, self.timeout_s
                    )
                except self._NETWORK_ERRORS as error:
                    if self.logger:
                        self.logger.response(
                            {
                                "path": path,
                                "attempt": attempt + 1,
                                "request_id": payload["request_id"],
                                "network_error": type(error).__name__,
                            }
                        )
                    if attempt >= self.max_network_retries:
                        raise SimulatorError(
                            f"network failure after {attempt + 1} attempts"
                        ) from error
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue

                try:
                    response = json.loads(response_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ProtocolError("response is not valid UTF-8 JSON") from error
                if not isinstance(response, dict):
                    raise ProtocolError("response JSON must be an object")
                if self.logger:
                    self.logger.response(
                        {
                            "path": path,
                            "attempt": attempt + 1,
                            "request_id": payload["request_id"],
                            "http_status": status,
                            "response": response,
                        }
                    )
                if not 200 <= status < 300:
                    raise SimulatorHTTPError(status, response)
                if response.get("accepted") is not True:
                    raise ActionRejected(response)
                return response
        raise AssertionError("unreachable")

    def enter(self) -> JsonObject:
        request_id = self.request_ids.next("enter")
        response = self._perform("/enter", self._payload(request_id))
        remaining = response.get("remaining_real_duration_s")
        if not isinstance(remaining, int):
            raise ProtocolError("accepted /enter lacks integer remaining_real_duration_s")
        self._update_virtual_time(response)
        self.state.remaining_real_duration_s = remaining
        self.state.real_deadline_monotonic = time.monotonic() + remaining
        self.state.entered = True
        self.state.exited = False
        self.state.position = (0.0, 0.0)
        self.state.current_channel = 1
        return response

    def remaining_real_time_s(self) -> float | None:
        """Return a decreasing local estimate based on the /enter allowance."""

        deadline = self.state.real_deadline_monotonic
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    def measure(self, position: tuple[float, float], channel: int) -> MeasureResult:
        self._validate_action(position, channel)
        request_id = self.request_ids.next("measure")
        response = self._perform(
            "/measure",
            self._payload(
                request_id,
                position={"x": position[0], "y": position[1]},
                channel=channel,
            ),
        )
        result = response.get("measure_result")
        if result not in {"no_signal", "near", "direction"}:
            raise ProtocolError(f"unknown measure_result: {result!r}")
        svd_deg: float | None = None
        if result == "direction":
            raw_bearing = response.get("svd_deg")
            if not isinstance(raw_bearing, (int, float)):
                raise ProtocolError("direction response lacks numeric svd_deg")
            svd_deg = float(raw_bearing)
        self._update_virtual_time(response)
        self.state.position = (float(position[0]), float(position[1]))
        self.state.current_channel = channel
        return MeasureResult(result=result, svd_deg=svd_deg, response=response)

    def clear(self, position: tuple[float, float], channel: int) -> ClearResult:
        self._validate_action(position, channel)
        request_id = self.request_ids.next("clear")
        response = self._perform(
            "/clear",
            self._payload(
                request_id,
                position={"x": position[0], "y": position[1]},
                channel=channel,
            ),
        )
        result = response.get("clear_result")
        if result not in {"success", "no_target_in_range"}:
            raise ProtocolError(f"unknown clear_result: {result!r}")
        self._update_virtual_time(response)
        self.state.position = (float(position[0]), float(position[1]))
        # Deliberately do not update current_channel: /clear never switches it.
        return ClearResult(result=result, response=response)

    def exit(self) -> JsonObject:
        request_id = self.request_ids.next("exit")
        response = self._perform("/exit", self._payload(request_id))
        self._update_virtual_time(response)
        self.state.exited = True
        return response

    def _validate_action(self, position: tuple[float, float], channel: int) -> None:
        if not self.state.entered or self.state.exited:
            raise SimulatorError("enter an active test before sending an action")
        if not isinstance(channel, int) or isinstance(channel, bool) or not 1 <= channel <= 20:
            raise ValueError("channel must be an integer in 1..20")
        if len(position) != 2 or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and abs(value) <= 2_000_000
            for value in position
        ):
            raise ValueError("position must contain two finite coordinates within limits")

    def _update_virtual_time(self, response: Mapping[str, object]) -> None:
        value = response.get("virtual_time_s")
        if not isinstance(value, (int, float)):
            raise ProtocolError("accepted response lacks numeric virtual_time_s")
        self.state.virtual_time_s = float(value)
