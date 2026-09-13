# Q2-B2-1 code design: minimal implementation architecture

Status: design only. This document records the code interfaces for the later
Q2 implementation. It does not implement Q2 algorithms.

## 1. Checked Q1 and common APIs to reuse

Branch checked: `q2/main-model`. The working tree was clean before this design
file was created.

Actually checked files:

- `src/common/geometry.py`
- `src/common/localization.py`
- `tests/test_geometry.py`
- `tests/test_geometry_calipers.py`
- `tests/test_geometry_mec_oracle.py`
- `tests/test_localization.py`
- `tests/test_q1_production_consistency.py`
- all tracked files under `src/q1/`
- `notes/assumptions.md`
- `notes/modeling.md`
- `scratch/q2_main_model_audit.md`
- `paper/drafts/B-q2：第二检测点选择策略.md`, read through git blob
  `9fef0693af47670f29320e485842f8039341626a`
- Markdown conversions under `problems/B/reference/` for the Q2 statement and
  the `near` / `direction` / `no_signal` / 1000-1500 m rules.

Q1 production capability is concentrated in `src/common`, not in `src/q1`.
The `src/q1` files are verification and plotting scripts that repeatedly call
the common APIs. Q2 should therefore add a thin modeling and optimization layer
instead of copying geometry logic.

Reusable APIs:

- `src/common/geometry.py:18` `Circle`: result type for circle computations.
- `src/common/geometry.py:29` `normalize_angle_deg(angle)`: angle wrap to
  `[0, 360)`.
- `src/common/geometry.py:36` `signed_angle_difference_deg(target, reference)`:
  hard bearing error checks.
- `src/common/geometry.py:43` `unit_vector(angle_deg)`: local longitudinal and
  lateral axes for Q2 candidate generation.
- `src/common/geometry.py:52` `distance(a, b)`: movement distance and
  source-detector distance.
- `src/common/geometry.py:110` `minimum_enclosing_circle(points)`: finite-point
  MEC candidate for clearability diagnostics.
- `src/common/geometry.py:155` `max_distance_to_region(region, center)`: verify
  a proposed clear center against a polygonal outer region.
- `src/common/geometry.py:166` `is_clear_point_certified(...)`: simple clear
  disk certificate.
- `src/common/geometry.py:176` `circle_polygon(radius, vertex_count)`: existing
  conservative circumscribed circle approximation.
- `src/common/geometry.py:193` `clip_polygon_half_plane(...)`: primitive used by
  bearing wedges.
- `src/common/geometry.py:236` `clip_polygon_to_circle_outer(...)`: conservative
  disk intersection. This should be reused for the 1500 m reception upper bound,
  the 1000 m certified-reception tests when discretized, and the 5 m near outer
  branch.
- `src/common/geometry.py:271` `clip_polygon_to_bearing_wedge(...)`: Q1 wedge
  intersection for both first and second `direction` observations.
- `src/common/geometry.py:287` `convex_hull(points)`: needed when a sampled or
  non-convex branch must be summarized for diameter/MEC diagnostics.
- `src/common/geometry.py:385` `polygon_diameter(points)`: production diameter
  API, backed by rotating calipers and checked against an exhaustive oracle.
- `src/common/geometry.py:396` `candidate_second_points(...)`: simple symmetric
  candidate generator. It can seed a vertical baseline or local refinement, but
  it is not a complete Q2 policy.
- `src/common/localization.py:22` `BearingObservation`: reusable observation
  record for Q1 direction constraints.
- `src/common/localization.py:47` `localization_region(...)`: conservative
  closed polygonal outer region from direction observations plus target circle
  and 1500 m reception upper bound.
- `src/common/localization.py:82` `localization_quality(region)`: existing
  diameter and area summary.
- `src/common/localization.py:98` `assess_region_for_clear(...)`: conservative
  clear-readiness assessment.
- `src/common/localization.py:175` `assess_observations_for_clear(...)`: high
  level Q1 composition.

Relevant tested behavior:

- angle wraparound, one-degree wedge crossing 0 degrees, conservative circle
  clipping, empty-region fail-closed behavior, calipers-vs-oracle diameter, MEC
  oracle agreement, Q1 symmetric analytic diameter, and conservative 1500 m
  reception clipping are already covered by tests.
- Q1 tests deliberately distinguish `direction`, `near`, and `no_signal`.
  `near` is not a bearing, and `no_signal` is not treated as target absence.

Gaps Q2 must add:

- distance from a point to a polygonal support for `C_poss`;
- certified `max_{g in K1} ||q-g||` against a convex polygon for `C_rec`;
- joint `(G, R)` nominal posterior samples or quadrature nodes;
- second-response branching over `near`, continuous `direction`, and
  `no_signal`;
- explicit distinction between theoretical supports and conservative outer
  regions.

## 2. Recommended minimal Q2 file structure

Create only these Q2 Python files when implementation starts:

```text
src/q2/
  model.py          # state, candidate domains, response model, evaluators
  optimizer.py      # grid/refinement, minimax, Bayes, hybrid, baselines
  main.py           # small experiment/figure/table entry point
```

Rationale: current repository style keeps shared geometry in `src/common` and
per-question runnable entry points under `src/q1` to `src/q4`. A single
`model.py` is enough at this stage because splitting supports, Bayesian
evaluation, and robust evaluation into many files would create empty seams
before the numerical representation is proven useful.

Recommended tests:

```text
tests/test_q2_model.py
tests/test_q2_optimizer.py
```

Do not create a separate Q2 geometry module unless profiling shows the helper
functions are broadly useful enough to move into `src/common/geometry.py`.

## 3. Core data types and suggested signatures

The following are interface signatures, not implementations.

### 3.1 State and configuration, in `src/q2/model.py`

```python
@dataclass(frozen=True)
class Q2Config:
    arena_radius_m: float = 1800.0
    reception_radius_min_m: float = 1000.0
    reception_radius_max_m: float = 1500.0
    near_radius_m: float = 5.0
    bearing_error_deg: float = 1.0
    circle_vertices: int = 720
    direction_angle_bins: int = 720
```

Invalid finite/range checks should raise `ValueError`. In particular,
`0 < near_radius_m < reception_radius_min_m <= reception_radius_max_m`,
positive arena radius, positive circle vertex count, and at least three angle
bins are required.

```python
@dataclass(frozen=True)
class FirstDirectionObservation:
    station: Point
    bearing_deg: float
```

This type represents the Q2 branch where the first observation is already known
to be `direction`. If the first real response is `near`, Q2 should bypass second
point selection and hand control to the clear logic.

```python
@dataclass(frozen=True)
class JointSample:
    position: Point
    reception_radius_m: float
    weight: float
```

Samples or quadrature nodes represent the nominal joint state `(G, R)`. They are
not only source-position particles; the fixed unknown `R` must stay attached to
`G`.

```python
@dataclass(frozen=True)
class FirstState:
    observation: FirstDirectionObservation
    exact_support_label: str
    outer_region: tuple[Point, ...]
    joint_samples: tuple[JointSample, ...]
```

`exact_support_label` should document the theoretical support:
`closure(F1) = Omega cap W(S1, theta1, delta) cap B(S1,1500)`, with the
observed `direction` branch remembering the open exclusion `||G-S1|| > 5` for
nominal sample filtering. The `outer_region` is the Q1-certified conservative
outer approximation produced from `localization_region(...)`; it is not the
exact support.

```python
def build_first_state(
    observation: FirstDirectionObservation,
    *,
    samples: Sequence[JointSample],
    config: Q2Config = Q2Config(),
) -> FirstState:
    ...
```

Inputs: first station, bearing, nominal `(G,R)` samples, config. Output:
`FirstState`. Degenerate cases: raise `ValueError` if the first observation is
non-finite, if samples are empty after hard filtering, or if
`localization_region([BearingObservation(...)])` returns an empty outer region.

#### Nominal first-direction posterior sampler (Task 3B)

The nominal Bayesian path must generate `(G,R)` samples before
`build_first_state(...)`. Adopted priors are modeling assumptions, not official
facts:

```text
G ~ Uniform(area on B(0,1800))
R ~ Uniform[1000,1500]
```

Use area-uniform disk sampling `r = 1800*sqrt(U)`, not a radius-uniform draw.
The sampled `R` remains attached to its source sample for all later responses.

Because the official problem gives only the hard `±1°` bearing bound, the first
observation likelihood must also be explicit. Use:

```python
@dataclass(frozen=True)
class BearingErrorBin:
    lower_deg: float
    upper_deg: float
    probability: float
```

Each bin is a user/model-supplied probability mass over an interval inside the
hard bound; within a bin Task 3B uses a piecewise-constant density. Bins may have
gaps (zero nominal density), must not overlap, and total probability must be one.
No uniform density is silently assumed.

```python
@dataclass(frozen=True)
class FirstPosteriorSamples:
    samples: tuple[JointSample, ...]
    prior_draws: int
    retained_draws: int
    effective_sample_size: float
    acceptance_rate: float
    seed: int
```

```python
def sample_first_direction_posterior(
    observation: FirstDirectionObservation,
    *,
    prior_draws: int,
    bearing_error_bins: Sequence[BearingErrorBin],
    seed: int = 0,
    min_effective_sample_size: float | None = None,
    config: Q2Config = Q2Config(),
) -> FirstPosteriorSamples:
    ...
```

Implementation is a reproducible rejection/importance sampler:

1. draw `(G,R)` from the adopted A1/A2 priors;
2. enforce the observed `direction` physics `5 < ||G-S1|| <= R`;
3. compute the required first-error residual
   `wrap(theta1 - bearing(S1,G))`;
4. reject residuals outside the hard `±1°` bound or zero-density bins;
5. weight retained draws by the explicit bin density and normalize weights;
6. report acceptance rate and effective sample size
   `ESS = 1 / sum(w_i^2)`.

This is a numerical posterior proxy, not an exact integral. A caller may set
`min_effective_sample_size` to fail rather than accept an under-resolved
posterior. The threshold is an experiment/convergence choice, not a problem
constant.

### 3.2 Candidate domains, in `src/q2/model.py`

```python
@dataclass(frozen=True)
class CandidateDomainFlags:
    in_c_poss_proxy: bool
    in_c_rec_certified: bool
    min_distance_to_outer_m: float
    max_distance_to_outer_m: float
```

```python
def evaluate_candidate_domains(
    q: Point,
    state: FirstState,
    *,
    config: Q2Config = Q2Config(),
) -> CandidateDomainFlags:
    ...
```

`C_poss` theoretical target:

```text
{q : inf_{g in closure(F1)} ||q-g|| <= 1500}
```

Engineering proxy: compute distance from `q` to `state.outer_region`. If the
outer distance is greater than 1500 m, then `q` is safely outside `C_poss`. If it
is at most 1500 m, the point is in an outer approximation of `C_poss`, not
necessarily certified against the exact support.

`C_rec` theoretical target:

```text
{q : sup_{g in closure(F1)} ||q-g|| <= 1000}
```

Engineering certificate: since `state.outer_region` is convex and contains
`closure(F1)`, `max_{v in vertices(K1_out)} ||q-v|| <= 1000` certifies
`q in C_rec`. This may reject valid points, but it must not accept invalid
points.

Do not add a `q in Omega` check. Q2 detection locations may be outside the
1800 m source region.

### 3.3 Second response model, in `src/q2/model.py`

```python
class SecondResponseKind(str, Enum):
    NEAR = "near"
    DIRECTION = "direction"
    NO_SIGNAL = "no_signal"
```

```python
@dataclass(frozen=True)
class SecondResponse:
    kind: SecondResponseKind
    bearing_deg: float | None = None
```

For `DIRECTION`, `bearing_deg` is required and normalized. For `NEAR` and
`NO_SIGNAL`, it must be `None`.

```python
@dataclass(frozen=True)
class SecondSupport:
    response: SecondResponse
    true_support_label: str
    sample_support: tuple[JointSample, ...]
    conservative_outer_region: tuple[Point, ...]
```

`sample_support` approximates the true posterior support for nominal/Bayesian
calculation. `conservative_outer_region` is the engineering outer region
`K2_out`, used for certified robust quantities. These names must not be
collapsed into a single "exact region".

```python
def possible_second_responses(
    q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> tuple[SecondResponse, ...]:
    ...
```

The response set must include all feasible branches among `near`,
`direction(theta)`, and `no_signal`; it must never drop `near` as a nuisance.
If a branch has no compatible sample and no compatible outer-region witness, it
can be omitted from the numerical proxy, but the omission reason should be
recorded in evaluator details.

For robust evaluation, `direction_grid_deg` should be treated as direction-bin
centers with bin width determined by the full partition of `[0, 360)`. The bin
centers alone are not a certificate; certification comes from the widened-bin
outer regions described below.

```python
def second_support(
    q: Point,
    response: SecondResponse,
    state: FirstState,
    *,
    config: Q2Config = Q2Config(),
) -> SecondSupport:
    ...
```

Branch semantics:

- `near`: true support is `F1 cap B(q,5)`; outer uses
  `clip_polygon_to_circle_outer(state.outer_region, q, 5, ...)`.
- `direction(theta2)`: true support is compatible `(G,R)` with
  `5 < ||G-q|| <= R` and hard angle error; the center-grid proxy uses Q1 wedge
  clipping plus `clip_polygon_to_circle_outer(..., q, 1500, ...)`. A robust
  certificate over continuous measured directions must instead use direction
  interval bins. For a bin centered at `c_j` with width `bin_width_deg`, every
  measured direction `z` in that bin is covered by the widened wedge with
  half-angle `bearing_error_deg + bin_width_deg / 2`:

  ```text
  K2_dir_bin_out =
      K1_out
      cap widened_wedge(q, c_j, delta + bin_width_deg / 2)
      cap B_out(q, 1500).
  ```

  This `K2_dir_bin_out` contains the true `direction(z)` support for all
  continuous `z` in the bin. The 5 m exclusion remains a true/sample constraint,
  not a convex outer clipping requirement.
- `no_signal`: true support is compatible `(G,R)` with `||G-q|| > R`.
  A first minimal conservative outer may return `state.outer_region` for
  `K2_out` if no certified non-convex outer is implemented yet; this is weak but
  safe. Do not use Monte Carlo samples, sample convex hulls, or sampled boundary
  points as conservative outer certificates. A later refinement may only replace
  this with strict set operations: `no_signal` implies
  `||G-q|| > R >= 1000`, so the sharpened conservative set is based on
  `K1_out \ B(q,1000)` and requires a reliable non-convex outer/convex-hull
  algorithm before it can be used for certification.

Degenerate cases: impossible response returns empty support and should not
silently contribute zero to a supremum. Empty support in a realized observation
is a model conflict.

### 3.4 Bayesian evaluator, in `src/q2/model.py`

```python
@dataclass(frozen=True)
class BearingErrorAtom:
    offset_deg: float
    probability: float
```

This is a modeling input, not a题设 constant. Different nominal error laws should
be compared in sensitivity analysis.

```python
@dataclass(frozen=True)
class ResponseMetric:
    response: SecondResponse
    probability: float
    diameter_true_proxy_m: float
    diameter_outer_m: float
    clear_radius_outer_m: float | None
```

```python
@dataclass(frozen=True)
class BayesianEvaluation:
    q: Point
    psi_d_m: float
    response_probabilities: dict[SecondResponseKind, float]
    metrics: tuple[ResponseMetric, ...]
    movement_m: float
```

```python
def evaluate_bayesian(
    q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    bearing_error_atoms: Sequence[BearingErrorAtom],
    config: Q2Config = Q2Config(),
) -> BayesianEvaluation:
    ...
```

Inputs: candidate point, first state, direction quadrature grid, and an
**explicit nominal bearing-error quadrature/PMF** supplied by the caller. The
problem gives only the hard `±1°` bound and does not specify an error density, so
the implementation must not silently assume a uniform law. `BearingErrorAtom`
records `(offset_deg, probability)` atoms inside the hard bound; their
probabilities must sum to one. Same-location repeats reuse the already fixed
first error rather than drawing a fresh atom.

Output: nominal `Psi_D(q)` and response probabilities. The first implementation
may use weighted joint samples:

- probability by summing compatible sample weights;
- `D(S2)` by `polygon_diameter` on compatible sample positions or their
  `convex_hull`;
- outer diagnostics by `polygon_diameter(conservative_outer_region)`.

The return value must label the sample-based `Psi_D` as a proxy. It must not
claim exact integration unless a later quadrature implementation proves that.

Degenerate cases: raise `ValueError` if `q` is non-finite, total compatible
probability is zero for all branches, or probabilities do not sum to one within
a configured tolerance.

### 3.5 Robust evaluator, in `src/q2/model.py`

```python
@dataclass(frozen=True)
class RobustEvaluation:
    q: Point
    u_proxy_m: float
    u_bar_m: float
    worst_response_proxy: SecondResponse | None
    worst_response_outer: SecondResponse | None
    in_c_poss_proxy: bool
    in_c_rec_certified: bool
    movement_m: float
```

```python
def evaluate_robust(
    q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> RobustEvaluation:
    ...
```

`u_proxy_m` is the finite center-grid proxy for the theoretical
`U(q) = sup_z D(S2(z,q))`. It is useful for ranking and minimax search, but it
is not a continuous-response certificate.

`u_bar_m` is a conservative engineering upper bound only when continuous
`direction` responses are covered by interval bins. For each direction bin
center `c_j` with width `bin_width_deg`, build `K2_dir_bin_out` using the
widened half-angle `bearing_error_deg + bin_width_deg / 2`.

Before including the `no_signal` branch, reuse the certified reception test:
if `C_rec` is certified, then every true source satisfies `||q-G|| <= 1000 <= R`,
so `no_signal` is impossible and must be omitted. Otherwise retain the safe
first implementation `K2_no_out = K1_out`.

Then compute `u_bar_m` as the maximum of:

- `D(K2_near_out)`;
- `D(K2_no_out)`, with first implementation `K2_no_out = K1_out`;
- every `D(K2_dir_bin_out)`.

Only this interval-bin version may be described as a conservative upper bound
over the continuous `direction` response space. A maximum over finite direction
centers alone must be named a grid proxy.

`worst_response_outer` may store `near`, `no_signal`, or the center-labeled
`direction` response for the worst interval bin. It is a diagnostic label, not
proof that the worst continuous angle equals the bin center.

Task 4A implements only this single-candidate evaluator. It does not yet
minimize over candidate points, construct `A_rho`, or select the hybrid policy.
For `u_proxy_m`, only non-empty sample-supported center-grid responses contribute;
if a grid is too coarse to support any response, the evaluator fails explicitly
instead of silently returning zero.

```python
def minimax_baseline(
    candidates: Sequence[Point],
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> RobustEvaluation:
    ...
```

The baseline should minimize `u_proxy_m` for the grid-based theoretical strategy
estimate and carry `u_bar_m` as the interval-bin conservative bound. If the
final selection needs certified admissibility over continuous direction
responses, use `u_bar_m` explicitly and label the result as an outer-envelope
variant.

## 4. Optimizer data flow, in `src/q2/optimizer.py`

```python
@dataclass(frozen=True)
class CandidateScore:
    q: Point
    bayes: BayesianEvaluation
    robust: RobustEvaluation
```

```python
def coarse_grid_candidates(
    state: FirstState,
    *,
    spacing_m: float,
    config: Q2Config = Q2Config(),
) -> tuple[Point, ...]:
    ...
```

Build a bounding box from `state.outer_region` expanded by 1500 m, then filter
with `evaluate_candidate_domains(...).in_c_poss_proxy`. Do not intersect with
the 1800 m target circle. Reject non-positive spacing. Return an empty tuple if
the grid has no valid proxy candidates; the optimizer should turn that into a
clear failure.

```python
def refine_candidates(
    seeds: Sequence[Point],
    state: FirstState,
    *,
    step_schedule_m: Sequence[float],
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> tuple[Point, ...]:
    ...
```

Minimal local refinement should stay derivative-free. SQP is not a main solver.
Candidate deduplication must use a documented tolerance.

```python
def score_candidates(
    candidates: Sequence[Point],
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> tuple[CandidateScore, ...]:
    ...
```

This single scoring path is mandatory for fair comparison across all strategies.

```python
def select_pure_bayesian(
    scores: Sequence[CandidateScore],
    *,
    tau_m: float = 0.0,
) -> CandidateScore:
    ...
```

Select minimum `psi_d_m`; among points with `psi_d_m <= best + tau_m`, choose
minimum movement `||q-S1||`. `tau_m` is numerical tolerance, not a preference
parameter.

```python
def select_pure_minimax(scores: Sequence[CandidateScore]) -> CandidateScore:
    ...
```

Select minimum `robust.u_proxy_m`, with movement as final deterministic
tie-break. Report `u_bar_m` alongside it.

```python
def select_robust_envelope_hybrid(
    scores: Sequence[CandidateScore],
    *,
    rho: float,
    tau_m: float = 0.0,
    use_outer_envelope: bool = False,
) -> CandidateScore:
    ...
```

If `use_outer_envelope=False`, compute the numerical analog of
`A_rho = {q: U(q) <= (1+rho) U*}` using `u_proxy_m`. If
`use_outer_envelope=True`, use `u_bar_m` and label the selected strategy as a
conservative outer-envelope implementation. In either case, minimize `psi_d_m`
inside the envelope and then movement among Bayesian near-optimal points.

Reject negative `rho`. Do not interpret `rho` as a problem constant.

```python
def same_distance_vertical_baseline(
    reference_q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> tuple[CandidateScore, CandidateScore]:
    ...
```

Use the first bearing axis `u=(cos theta, sin theta)` and lateral normal
`n=(-sin theta, cos theta)`. Let `L = ||reference_q-S1||`; evaluate both
`S1 + L n` and `S1 - L n`. Do not choose a side without scoring both.

## 5. Theoretical set vs engineering outer region

Names should enforce the distinction:

- `F1` / `S2`: theoretical true position supports.
- `joint_samples`: nominal numerical approximation to the posterior over
  `(G,R)`.
- `outer_region`, `K1_out`, `K2_out`: conservative polygonal outer regions used
  for engineering certification.
- `u_proxy_m`: center-grid numerical estimate of the theoretical worst
  diameter; this is not a certificate.
- `u_bar_m`: interval-bin conservative outer worst diameter. It is a continuous
  direction-response certificate only if every direction bin uses the widened
  wedge half-angle `bearing_error_deg + bin_width_deg / 2`, and if `near` and
  `no_signal` use certified outer regions.
- `diameter_true_proxy_m`: sample/quadrature-based estimate.
- `diameter_outer_m`: diameter of the conservative outer region.

The implementation should never call `u_bar_m` "exact U". It is valid to state:
if every interval-bin `K2_dir_bin_out`, plus the `near` and `no_signal` outers,
contains the corresponding true support, then `U_bar(q) >= U(q)`. That is a
containment certificate, not an equality claim.

For the first no-signal outer, use:

```text
K2_no_out = K1_out
```

This is loose but safe. Monte Carlo samples must not be used to certify a
no-signal outer. A later tighter certificate may use the hard implication
`no_signal => distance > R >= 1000`, but only through reliable set operations
for `K1_out \ B(q,1000)` and its required outer representation.

## 6. Bayesian and robust evaluator data flow

Bayesian flow:

```text
FirstDirectionObservation
  -> sample_first_direction_posterior from explicit A1/A2 priors
     and explicit first-bearing error density
  -> build_first_state
  -> Q1 outer_region via localization_region
  -> normalized nominal joint_samples over (G,R) for H1
  -> candidate q
  -> second_support for near / direction grid / no_signal
  -> response probabilities from joint sample weights
  -> diameter_true_proxy_m per response
  -> Psi_D(q)
```

Robust flow:

```text
FirstState.outer_region plus hard rules
  -> candidate q in C_poss proxy
  -> possible hard second responses
  -> near outer via B_out(q,5)
  -> no_signal outer as K1_out in the first implementation
  -> direction center-grid proxy for u_proxy_m
  -> direction interval bins with widened wedge for u_bar_m
  -> max diameter over near / no_signal / direction-bin outers
  -> U_bar(q) continuous-response conservative bound
  -> U(q) center-grid numerical proxy
```

The robust evaluator should keep `near`, `direction`, and `no_signal` branches
even when one branch is inconvenient. `C_rec` only makes `no_signal` impossible
under the certified hard model; it does not license dropping `near`.

## 7. Optimizer data flow

```text
build_first_state
  -> coarse_grid_candidates within C_poss proxy
  -> score_candidates with the shared Bayesian and robust evaluators
  -> optional local refine around best Bayes / minimax / hybrid seeds
  -> rescore through the same score_candidates path
  -> select:
       pure Bayesian
       pure minimax
       robust-envelope hybrid
       same-distance vertical +/- baselines
```

The optimizer should not introduce `T_max`. Movement cost appears only as:

- secondary tie-break inside Bayesian or hybrid near-optimal sets;
- reported experiment metric.

## 8. Four ablations sharing one evaluator

All four strategies must use the same `FirstState`, candidate-generation budget,
direction grid, sample set, and `score_candidates(...)` output:

1. `same-distance vertical baseline`: after selecting a reference strategy point,
   evaluate `S1 +/- L n` and report both sides plus the better side.
2. `pure Bayesian`: `select_pure_bayesian(scores, tau_m=...)`.
3. `pure minimax`: `select_pure_minimax(scores)`.
4. `Bayesian + robust hybrid`: `select_robust_envelope_hybrid(scores, rho=...)`.

Fairness rule: vertical baseline movement distance must equal the strategy point
being compared. If reporting one vertical value, it must be the better of the
two scored sides, with both raw side results retained.

## 9. Unit test checklist

`tests/test_q2_model.py`:

- `build_first_state` calls Q1 `localization_region` and rejects empty samples.
- first `direction` branch filters or labels `||G-S1|| > 5`, but the closure
  used for `C_poss` and `C_rec` is not broken by the open boundary.
- `evaluate_candidate_domains` does not require `q` to lie in the 1800 m target
  circle.
- `C_rec` certificate accepts a point whose maximum distance to every outer
  vertex is at most 1000 m and rejects one above 1000 m.
- `C_rec` empty case is represented without raising during state construction.
- `near` branch clips with a 5 m disk and has no bearing angle.
- `direction` branch requires a bearing and applies a second Q1 wedge plus
  1500 m outer disk.
- robust direction certification uses interval-bin outers, not finite center
  samples. For any one direction bin, the true or fine-grid supports at the left
  boundary, center, and right boundary of the bin must be contained in
  `K2_dir_bin_out` built with half-angle
  `bearing_error_deg + bin_width_deg / 2`.
- `no_signal` branch remains available outside `C_rec`.
- first-version `no_signal` certificate returns `K2_no_out = K1_out`.
- Monte Carlo samples, sample convex hulls, and sampled boundary points are not
  accepted as `no_signal` conservative outer certificates.
- response probabilities sum to one within tolerance for a hand-built sample
  set.
- empty realized support is reported as model conflict, not as zero loss.
- `u_bar_m >= u_proxy_m` for a constructed case where outer regions strictly
  contain sample supports.
- finite direction-center maxima are reported as grid proxies; only interval-bin
  widened-wedge maxima are reported as continuous-response conservative upper
  bounds.

`tests/test_q2_optimizer.py`:

- coarse grid expands `K1_out` by 1500 m and does not clip to `Omega`.
- negative `rho`, non-positive grid spacing, and empty score lists raise
  `ValueError`.
- pure Bayesian tie-break uses movement only within `tau_m`.
- pure minimax chooses by `u_proxy_m` and reports `u_bar_m`.
- hybrid first filters by the robust envelope, then minimizes `psi_d_m`, then
  movement.
- `rho=0` reduces the envelope to minimax-near candidates before Bayesian
  tie-break.
- `same_distance_vertical_baseline` evaluates both lateral sides with identical
  movement distance.
- all four ablations can be computed from the same `CandidateScore` list.

## 10. Implementation tasks in dependency order

1. Add `src/q2/model.py` with data classes, `build_first_state`, candidate-domain
   predicates, and response type validation. Reuse Q1 geometry only; no
   optimizer yet.
2. Implement `second_support` and the shared `evaluate_bayesian` /
   `evaluate_robust` proxies using a small deterministic joint-sample interface.
3. Add `src/q2/optimizer.py` with coarse grid, local derivative-free refinement,
   shared scoring, and the three selectors.
4. Add vertical baseline and four-strategy ablation plumbing, ensuring all
   strategies share the same evaluator outputs.
5. Add focused unit tests for Q2 model and optimizer degeneracies.
6. Add `src/q2/main.py` only after the library functions pass tests; it should
   produce small diagnostic tables/figures and must not run large experiments by
   default.
