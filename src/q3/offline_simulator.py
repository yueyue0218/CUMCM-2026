"""Small synthetic transport for regression/validation, NOT the official simulator.

Truth lives only here. The real controller receives exactly the same observation
fields as the HTTP client. Each point has a fixed bounded bearing-error field.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from src.common.geometry import Point


@dataclass(frozen=True)
class Source:
    channel: int
    position: Point
    receive_radius_m: float


class OfflineSimulator:
    def __init__(self, sources: list[Source], *, bearing_error_deg: float = 0.0,
                 remaining_real_duration_s: int = 1200, error_field: str = "constant",
                 quantize_bearings: bool = False, error_phase: float = 0.0):
        if not math.isfinite(bearing_error_deg) or abs(bearing_error_deg) > 1:
            raise ValueError("bearing error must lie in [-1, 1]")
        if error_field not in {"constant", "spatial"}:
            raise ValueError("unknown error field")
        self.sources = {s.channel: s for s in sources}
        if len(self.sources) != len(sources):
            raise ValueError("only one source per channel is supported")
        for source in sources:
            if (isinstance(source.channel, bool) or not isinstance(source.channel, int)
                    or not 1 <= source.channel <= 20
                    or not all(math.isfinite(v) for v in source.position)
                    or math.hypot(*source.position) > 1800.0 + 1e-9
                    or not 1000 <= source.receive_radius_m <= 1500):
                raise ValueError("source violates Q3 scene bounds")
        self.bearing_error_deg = bearing_error_deg
        self.error_field = error_field
        self.quantize_bearings = quantize_bearings
        self.error_phase = error_phase
        self.remaining_real_duration_s = remaining_real_duration_s
        self.cleared: set[int] = set()
        self.position = (0.0, 0.0)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.actions: list[dict] = []
        self._cache: dict[str, tuple[str, bytes, tuple[int, bytes]]] = {}
        self.entered = False
        self.exited = False

    def transport(self, url: str, body: bytes, timeout_s: float) -> tuple[int, bytes]:
        request = json.loads(body)
        path = "/" + url.rsplit("/", 1)[-1]
        request_id = request["request_id"]
        if request_id in self._cache:
            old_path, old_body, result = self._cache[request_id]
            if (old_path, old_body) != (path, body):
                raise ValueError("request_id reused for a different action")
            return result
        response: dict = {"accepted": True}
        if path == "/enter":
            if self.entered:
                raise ValueError("offline world is single-session")
            self.entered = True
            response["remaining_real_duration_s"] = self.remaining_real_duration_s
        else:
            if not self.entered or self.exited:
                raise ValueError("no active offline session")
            if path == "/exit":
                self.exited = True
                response["exit_reason"] = "user_exit"
            elif path in {"/measure", "/clear"}:
                position = (float(request["position"]["x"]), float(request["position"]["y"]))
                channel = request["channel"]
                source = self.sources.get(channel)
                distance = (math.dist(position, source.position)
                            if source and channel not in self.cleared else math.inf)
                self.virtual_time_s += math.dist(self.position, position) / 5.0
                self.position = position
                if path == "/measure":
                    self.virtual_time_s += 5.0 + (channel != self.current_channel)
                    self.current_channel = channel
                    if distance > (source.receive_radius_m if source else 0):
                        response["measure_result"] = "no_signal"
                    elif distance <= 5.0:
                        response["measure_result"] = "near"
                    else:
                        response["measure_result"] = "direction"
                        angle = math.degrees(math.atan2(source.position[1] - position[1],
                                                        source.position[0] - position[0]))
                        error = self.bearing_error_deg
                        if self.error_field == "spatial":
                            error *= math.sin(position[0] * 0.017 + position[1] * 0.023 + channel + self.error_phase)
                        bearing = angle + error
                        if self.quantize_bearings:
                            # Quantize inside the final-readout ±1 degree bound,
                            # rather than adding an extra rounding error to it.
                            lower = math.ceil((angle-1.0)*100)
                            upper = math.floor((angle+1.0)*100)
                            bearing = max(lower, min(upper, round(bearing*100))) / 100
                        response["svd_deg"] = bearing % 360.0
                elif distance <= 20.0:
                    self.cleared.add(channel)
                    self.virtual_time_s += 5.0
                    response["clear_result"] = "success"
                else:
                    self.virtual_time_s += 3.0
                    response["clear_result"] = "no_target_in_range"
            else:
                raise ValueError(f"unknown action: {path}")
        response["virtual_time_s"] = self.virtual_time_s
        self.actions.append({"path": path, "request": request, "response": dict(response)})
        result = (200, json.dumps(response).encode("utf-8"))
        self._cache[request_id] = (path, body, result)
        return result
