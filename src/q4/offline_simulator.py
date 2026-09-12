"""Synthetic Q4 transport. Private scene truth never enters the controller."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

from src.q3.offline_simulator import OfflineSimulator, Source


@dataclass(frozen=True)
class MixedSource(Source):
    direction_deg: float | None = None


class MixedSimulator(OfflineSimulator):
    def __init__(self, sources, **kwargs):
        super().__init__(sources, **kwargs)
        if any(s.direction_deg is not None and not math.isfinite(s.direction_deg) for s in sources):
            raise ValueError('finite direction or None required')

    def transport(self, url, body, timeout_s):
        # The shared transport owns timing, quantization, idempotency and logs.
        # Mask only a back-facing radio observation; optical calls see all sources.
        # SimulatorClient serializes all calls, including retries.
        request = json.loads(body)
        source = self.sources.get(request.get('channel'))
        hidden = False
        if url.endswith('/measure') and source and source.direction_deg is not None:
            angle = math.radians(source.direction_deg % 360)
            dx = float(request['position']['x']) - source.position[0]
            dy = float(request['position']['y']) - source.position[1]
            # Snap cardinal axes exactly, preserving their closed boundaries.
            ux, uy = math.cos(angle), math.sin(angle)
            if source.direction_deg % 90 == 0:
                ux, uy = round(ux), round(uy)
            hidden = ux*dx + uy*dy < 0
        if not hidden:
            return super().transport(url, body, timeout_s)
        del self.sources[source.channel]
        try:
            return super().transport(url, body, timeout_s)
        finally:
            self.sources[source.channel] = source


def sample_sources(seed, count, directional_fraction=.5):
    if not 2 <= count <= 20 or not 0 <= directional_fraction <= 1:
        raise ValueError('need 2..20 sources and a fraction in [0,1]')
    rng = random.Random(seed)
    channels = sorted(rng.sample(range(1, 21), count))
    directional = set(rng.sample(channels, max(1, min(count-1, round(count*directional_fraction)))))
    sources = []
    for c in channels:
        r, a = 1800*math.sqrt(rng.random()), rng.uniform(0, 2*math.pi)
        sources.append(MixedSource(c, (r*math.cos(a), r*math.sin(a)),
                                   rng.uniform(1000, 1500),
                                   rng.uniform(0, 360) if c in directional else None))
    return sources
