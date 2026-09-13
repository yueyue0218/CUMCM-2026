"""Minimal Q2 candidate-grid and pure-minimax optimizer utilities.

Task 4B intentionally implements only:
- coarse candidate generation over the C_poss outer proxy; and
- pure minimax selection using RobustEvaluation.u_proxy_m.

It does not implement Bayesian selection, local refinement, hybrid selection,
vertical baselines, plotting, or experiments.
"""

from __future__ import annotations

import math
from typing import Sequence

from src.common.geometry import Point
from src.q2.model import (
    FirstState,
    Q2Config,
    RobustEvaluation,
    evaluate_candidate_domains,
    evaluate_robust,
)


def coarse_grid_candidates(
    state: FirstState,
    *,
    spacing_m: float,
    config: Q2Config = Q2Config(),
) -> tuple[Point, ...]:
    """Return a deterministic coarse grid over the C_poss outer proxy.

    The search box is the bounding box of ``state.outer_region`` expanded by the
    maximum reception radius.  The grid is aligned to integer multiples of
    ``spacing_m`` so the result does not depend on the polygon's first vertex.
    Points are retained only when
    ``evaluate_candidate_domains(...).in_c_poss_proxy`` is true.

    Q2 detector locations are *not* restricted to the 1800 m source arena.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")
    if isinstance(spacing_m, bool) or not isinstance(spacing_m, (int, float)):
        raise ValueError("spacing_m must be a finite positive number")
    spacing = float(spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_m must be a finite positive number")

    xs = [point[0] for point in state.outer_region]
    ys = [point[1] for point in state.outer_region]
    margin = config.reception_radius_max_m

    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin

    ix_min = math.floor(min_x / spacing)
    ix_max = math.ceil(max_x / spacing)
    iy_min = math.floor(min_y / spacing)
    iy_max = math.ceil(max_y / spacing)

    candidates: list[Point] = []
    for ix in range(ix_min, ix_max + 1):
        x = ix * spacing
        for iy in range(iy_min, iy_max + 1):
            y = iy * spacing
            q = (x, y)
            if evaluate_candidate_domains(q, state, config=config).in_c_poss_proxy:
                candidates.append(q)

    return tuple(candidates)


def minimax_baseline(
    candidates: Sequence[Point],
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> RobustEvaluation:
    """Return the best finite-candidate, finite-direction minimax baseline.

    The primary criterion is ``u_proxy_m``.  Movement distance is the first
    deterministic tie-break, followed by coordinates.  ``u_bar_m`` is carried
    with the selected evaluation as a conservative diagnostic; it is not the
    primary ranking criterion in this baseline.

    This is therefore a numerical minimax baseline over the supplied candidate
    set, not a continuous-space global optimum.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not candidates:
        raise ValueError("candidates must be non-empty")

    admissible: list[RobustEvaluation] = []
    for q in candidates:
        result = evaluate_robust(
            q,
            state,
            direction_grid_deg=direction_grid_deg,
            config=config,
        )
        if result.in_c_poss_proxy:
            admissible.append(result)

    if not admissible:
        raise ValueError("no candidate lies in the C_poss outer proxy")

    return min(
        admissible,
        key=lambda result: (
            result.u_proxy_m,
            result.movement_m,
            result.q[0],
            result.q[1],
        ),
    )
