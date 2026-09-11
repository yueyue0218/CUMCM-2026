# Q1 Geometry Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Q1 solver that converts bounded-bearing observations into a classified localization region, computes its geometric measures and covering circles, and exports deterministic boundary-focused simulation data and separate Markdown parameter/result reports.

**Architecture:** SciPy `linprog` certifies feasibility and boundedness without an artificial global box; repository-owned geometry functions perform wedge construction, incremental clipping, convex normalization, rotating-calipers diameter, Green integrals, and Welzl minimum enclosing circles. A thin solver and JSON command-line entry point compose those pure functions, while one deterministic simulation driver generates both machine-readable files and judge-readable Markdown summaries.

**Tech Stack:** Python 3.12, standard-library `dataclasses`, `json`, `math`, `random`, `unittest`, NumPy 2.x, SciPy 1.14+

**Spec:** `docs/superpowers/specs/2026-09-11-q1-geometry-solver-design.md`

## Global Constraints

- Branch from `main` as `q1/geometry-solver`; do not include the existing `q1/baseline-model` branch or untracked Q2 files.
- Coordinates use metres; input bearings use degrees and normalize to `[0, 360)`; internal trigonometry uses radians.
- Each observation defaults to a 1 degree half-angle and rejects non-finite values or half-angles outside `(0, 90)` degrees.
- The primary Q1 region is the pure bearing intersection `P`; the 1800 metre arena and 20 metre clearing rule must be reported separately and never silently redefine `P`.
- Use fixed seed `20260911`; retain full precision internally and round only when serializing human-readable summaries.
- Core algorithms remain repository-owned; do not add Shapely or another geometry dependency.
- Production code is added only after its behavior test has failed for the expected missing-feature reason.

---

### Task 1: Bearing wedges and classified half-plane intersection

**Files:**
- Create: `src/q1/__init__.py`
- Create: `src/q1/geometry.py`
- Test: `src/q1/test_geometry.py`

**Interfaces:**
- Consumes: `Observation(station: tuple[float, float], bearing_deg: float, half_angle_deg: float = 1.0)`.
- Produces: `HalfPlane(a: float, b: float, c: float)`, `Region(status: str, vertices: tuple[Point, ...], max_violation: float)`, `bearing_halfplanes(observation) -> tuple[HalfPlane, HalfPlane]`, `intersect_halfplanes(halfplanes, tolerance=1e-9) -> Region`, and `point_satisfies(point, halfplanes, tolerance=1e-9) -> bool`.

- [ ] **Step 1: Write the failing observation and region tests**

```python
class BearingHalfPlaneTests(unittest.TestCase):
    def test_cross_zero_wedge_keeps_east_point(self):
        planes = bearing_halfplanes(Observation((0.0, 0.0), 359.5, 1.0))
        self.assertTrue(point_satisfies((100.0, 0.0), planes))
        self.assertFalse(point_satisfies((-100.0, 0.0), planes))

    def test_axis_aligned_box_is_classified_as_polygon(self):
        planes = [HalfPlane(1, 0, 2), HalfPlane(-1, 0, 0),
                  HalfPlane(0, 1, 1), HalfPlane(0, -1, 0)]
        region = intersect_halfplanes(planes)
        self.assertEqual(region.status, "polygon")
        self.assertEqual(len(region.vertices), 4)
        self.assertLessEqual(region.max_violation, 1e-9)

    def test_contradictory_bounds_are_empty(self):
        region = intersect_halfplanes([HalfPlane(1, 0, 0), HalfPlane(-1, 0, -1)])
        self.assertEqual(region.status, "empty")

    def test_one_wedge_is_unbounded(self):
        region = intersect_halfplanes(bearing_halfplanes(Observation((0, 0), 45)))
        self.assertEqual(region.status, "unbounded")
```

- [ ] **Step 2: Run the geometry tests and verify RED**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_geometry -v`

Expected: import failure for `src.q1.geometry`, proving the new API does not yet exist.

- [ ] **Step 3: Implement validation and bearing-to-half-plane conversion**

```python
@dataclass(frozen=True)
class Observation:
    station: Point
    bearing_deg: float
    half_angle_deg: float = 1.0

    def __post_init__(self) -> None:
        values = (*self.station, self.bearing_deg, self.half_angle_deg)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("observation values must be finite")
        if not 0.0 < self.half_angle_deg < 90.0:
            raise ValueError("half_angle_deg must be in (0, 90)")


def bearing_halfplanes(observation: Observation) -> tuple[HalfPlane, HalfPlane]:
    sx, sy = observation.station
    theta = math.radians(observation.bearing_deg % 360.0)
    delta = math.radians(observation.half_angle_deg)
    lower = (math.cos(theta - delta), math.sin(theta - delta))
    upper = (math.cos(theta + delta), math.sin(theta + delta))
    return (
        HalfPlane(lower[1], -lower[0], lower[1] * sx - lower[0] * sy),
        HalfPlane(-upper[1], upper[0], -upper[1] * sx + upper[0] * sy),
    )
```

- [ ] **Step 4: Implement feasibility, boundedness, clipping, de-duplication, and point/segment/polygon classification**

Use `scipy.optimize.linprog` with variable bounds `[(None, None), (None, None)]`. First solve a zero objective for feasibility; then solve `±x` and `±y`. Return `unbounded` if any coordinate objective is unbounded. For four finite extrema, initialize their bounding rectangle, clip it against every `a*x+b*y <= c`, remove adjacent points within `tolerance`, compute a monotone-chain hull, and classify hull sizes 1, 2, or at least 3 as `point`, `segment`, or `polygon`. Recheck all vertices against every original constraint and store the largest raw residual as `max_violation`.

- [ ] **Step 5: Add degenerate and validation tests, then run the complete geometry file**

```python
def test_equalities_can_reduce_region_to_segment(self):
    planes = [HalfPlane(0, 1, 0), HalfPlane(0, -1, 0),
              HalfPlane(1, 0, 2), HalfPlane(-1, 0, 0)]
    region = intersect_halfplanes(planes)
    self.assertEqual(region.status, "segment")
    self.assertEqual(region.vertices, ((0.0, 0.0), (2.0, 0.0)))

def test_non_finite_observation_is_rejected(self):
    with self.assertRaises(ValueError):
        Observation((float("nan"), 0.0), 10.0)
```

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_geometry -v`

Expected: all geometry tests pass with no warnings.

- [ ] **Step 6: Commit the geometry region implementation**

```powershell
git add src/q1/__init__.py src/q1/geometry.py src/q1/test_geometry.py
git commit -m "q1: 实现测向半平面交与区域分类"
```

---

### Task 2: Diameter, coverage, and exact Green boundary measures

**Files:**
- Create: `src/q1/measures.py`
- Test: `src/q1/test_measures.py`

**Interfaces:**
- Consumes: counter-clockwise convex `Sequence[Point]`; closed boundary elements `LineSegment` and `CircularArc`.
- Produces: `DiameterResult`, `diameter_exhaustive(vertices)`, `diameter_rotating_calipers(vertices)`, `diameter_circle_coverage(vertices, diameter)`, `polygon_area_centroid(vertices)`, and `green_area_centroid(elements)`.

- [ ] **Step 1: Write failing diameter and coverage tests using hand-derived shapes**

```python
def test_rectangle_calipers_matches_literal_diagonal(self):
    vertices = [(0, 0), (4, 0), (4, 3), (0, 3)]
    result = diameter_rotating_calipers(vertices)
    self.assertAlmostEqual(result.distance, 5.0, places=12)
    self.assertEqual(result.squared_distance, 25.0)

def test_equilateral_triangle_diameter_circle_does_not_cover(self):
    vertices = [(0, 0), (36, 0), (18, 18 * math.sqrt(3))]
    result = diameter_rotating_calipers(vertices)
    coverage = diameter_circle_coverage(vertices, result)
    self.assertFalse(coverage.covers)
    self.assertGreater(coverage.max_distance, 18.0)
```

- [ ] **Step 2: Run and verify RED for missing `src.q1.measures`**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_measures -v`

Expected: import failure for the missing measures module.

- [ ] **Step 3: Implement exhaustive and rotating-calipers diameter plus coverage**

Handle 0, 1, and 2 vertices explicitly. For at least 3 vertices, advance one antipodal pointer monotonically using cross-product area; on equal areas compare both opposite endpoints. Compare squared distances and return one deterministic lexicographically normalized witness pair. Coverage uses the witness midpoint and checks every vertex against `D²/4` with a scale-aware tolerance.

- [ ] **Step 4: Add a randomized independent comparison test**

Generate 100 fixed-seed integer point sets, normalize them with `convex_hull`, and assert the calipers and exhaustive squared distances agree exactly for integer-valued vertices. This test catches a reset antipodal pointer, missing tie branch, or wraparound error.

- [ ] **Step 5: Write failing polygon and circular-arc Green tests**

```python
def test_rectangle_green_measure_has_literal_centroid(self):
    area, centroid = polygon_area_centroid([(0, 0), (4, 0), (4, 2), (0, 2)])
    self.assertAlmostEqual(area, 8.0, places=12)
    self.assertAlmostEqual(centroid[0], 2.0, places=12)
    self.assertAlmostEqual(centroid[1], 1.0, places=12)

def test_full_circle_arc_has_exact_area_and_center(self):
    area, centroid = green_area_centroid([
        CircularArc(center=(3, -2), radius=5, start_angle=0, sweep_angle=2 * math.pi)
    ])
    self.assertAlmostEqual(area, 25 * math.pi, places=10)
    self.assertAlmostEqual(centroid[0], 3.0, places=10)
    self.assertAlmostEqual(centroid[1], -2.0, places=10)
```

- [ ] **Step 6: Implement line and oriented-arc Green contributions**

For each element accumulate signed area `A = 1/2 ∮(x dy - y dx)`, first x moment `Nx = 1/2 ∮x²dy`, and first y moment `Ny = -1/2 ∮y²dx`; return `(A, (Nx/A, Ny/A))`. `CircularArc` validates a finite positive radius and sweep in `[0, 2π]`, including distinct zero-arc and full-circle cases. Reject an open boundary or an area whose magnitude is below the scale-aware tolerance.

- [ ] **Step 7: Run measures tests and commit**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_measures -v`

Expected: all diameter, coverage, polygon, and arc tests pass.

```powershell
git add src/q1/measures.py src/q1/test_measures.py
git commit -m "q1: 增加直径覆盖与格林测度算法"
```

---

### Task 3: Welzl circle and integrated Q1 decision result

**Files:**
- Create: `src/q1/enclosing_circle.py`
- Create: `src/q1/solver.py`
- Test: `src/q1/test_enclosing_circle.py`
- Test: `src/q1/test_solver.py`

**Interfaces:**
- Consumes: finite points for `minimum_enclosing_circle(points, seed=20260911)`; solver payload with `observations` and optional `arena_radius_m`, `clear_radius_m`, and `output_decimals`.
- Produces: `Circle(center: Point, radius: float, max_residual: float)`, `minimum_enclosing_circle`, and `solve_case(payload: Mapping[str, object]) -> dict[str, object]`.

- [ ] **Step 1: Write failing analytic Welzl tests**

```python
def test_equilateral_triangle_reaches_jung_radius(self):
    points = [(0, 0), (36, 0), (18, 18 * math.sqrt(3))]
    circle = minimum_enclosing_circle(points)
    self.assertAlmostEqual(circle.radius, 12 * math.sqrt(3), places=10)
    self.assertLessEqual(circle.max_residual, 1e-9)

def test_collinear_points_use_longest_pair(self):
    circle = minimum_enclosing_circle([(-2, 0), (5, 0), (1, 0), (5, 0)])
    self.assertEqual(circle.center, (1.5, 0.0))
    self.assertEqual(circle.radius, 3.5)
```

- [ ] **Step 2: Run and verify RED, then implement Welzl**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_enclosing_circle -v`

Expected: missing enclosing-circle module.

Randomly permute a de-duplicated local copy with `random.Random(seed)`. Implement recursive `welzl(prefix_count, boundary)` with at most three boundary points. The base solver enumerates all one-point, two-point, and non-collinear three-point circles that cover its at-most-three inputs and chooses the smallest deterministic candidate. Verify every original point and calculate `max_residual = max(distance - radius)` before returning.

- [ ] **Step 3: Run Welzl tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_enclosing_circle -v`

Expected: analytic, duplicate, collinear, and fixed-seed invariance tests pass.

- [ ] **Step 4: Write failing integrated solver tests**

```python
def test_bounded_case_reports_problem_and_control_results_separately(self):
    result = solve_case({"observations": self.symmetric_observations})
    self.assertEqual(result["region"]["status"], "polygon")
    self.assertIn("diameter_m", result["problem_1"])
    self.assertIn(result["control"]["status"],
                  {"CLEAR_READY", "SINGLE_DISK_IMPOSSIBLE", "COVERAGE_UNCERTAIN"})

def test_empty_case_never_claims_clear_ready(self):
    result = solve_case({"observations": self.conflicting_observations})
    self.assertEqual(result["region"]["status"], "empty")
    self.assertEqual(result["control"]["status"], "NO_FEASIBLE_REGION")
```

- [ ] **Step 5: Implement the solver composition and arena-aware conservative decision**

Parse observations into the Task 1 data class and solve `P`. For bounded regions compute polygon area/centroid when non-degenerate, both diameter algorithms, diameter-circle coverage, and the minimum enclosing circle. Assert the independent diameter discrepancy is within a scale-aware tolerance. Report `arena_contains_region` by checking every vertex against the configured 1800 metre disk. Return `CLEAR_READY` only when the entire certified `P` lies inside the arena and the rounded output center still covers `P` within 20 metres. Return `SINGLE_DISK_IMPOSSIBLE` only when the certified relevant set has minimum radius above 20 metres; if arena clipping could change that conclusion, return `COVERAGE_UNCERTAIN` with reason `arena_clipping_required`. Empty and unbounded `P` receive explicit non-success states.

- [ ] **Step 6: Run all Task 1-3 tests and commit**

Run: `.venv\Scripts\python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v`

Expected: all tests pass with no warnings.

```powershell
git add src/q1/enclosing_circle.py src/q1/solver.py src/q1/test_enclosing_circle.py src/q1/test_solver.py
git commit -m "q1: 集成最小包围圆与清除判定"
```

---

### Task 4: JSON command-line entry point

**Files:**
- Create: `src/q1/main.py`
- Test: `src/q1/test_main.py`

**Interfaces:**
- Consumes: `python src/q1/main.py INPUT.json --output OUTPUT.json`; input is either one case object or `{"cases": [...]}`.
- Produces: UTF-8 JSON with `allow_nan=False`, stable key order, two-space indentation, and one result per input case.

- [ ] **Step 1: Write a failing subprocess test against real files**

Use `tempfile.TemporaryDirectory`, write one literal bounded case, invoke the entry point with `subprocess.run(..., check=False, capture_output=True, text=True)`, and assert exit code 0, an existing output file, preserved case ID, and a finite `problem_1.diameter_m`. Add one malformed-input test asserting a nonzero exit and a concise message on stderr.

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_main -v`

Expected: the entry-point file is missing and the subprocess test fails for that reason.

- [ ] **Step 3: Implement the entry point**

Resolve the repository root from `Path(__file__).resolve().parents[2]`, add it to `sys.path` only when the file is executed directly, parse arguments with `argparse`, validate the top-level case shape, call `solve_case`, and write through a temporary sibling followed by `Path.replace` so interrupted generation does not leave a partial result.

- [ ] **Step 4: Run entry-point and full tests, then commit**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_main -v`

Run: `.venv\Scripts\python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v`

Expected: all tests pass and malformed input fails cleanly without a traceback.

```powershell
git add src/q1/main.py src/q1/test_main.py
git commit -m "q1: 增加可复现JSON求解入口"
```

---

### Task 5: Deterministic simulations and separate Markdown records

**Files:**
- Create: `src/q1/simulate_cases.py`
- Test: `src/q1/test_simulate_cases.py`
- Create: `data/processed/q1_simulation_cases.json`
- Create: `data/processed/q1_simulation_parameters.md`
- Create: `results/tables/q1_simulation_results.json`
- Create: `results/tables/q1_simulation_results.md`

**Interfaces:**
- Consumes: `generate_cases(seed=20260911, regular_count=50, near_parallel_count=20) -> list[dict]` and `run_simulation(cases) -> dict`.
- Produces: `write_outputs(root, cases, summary)` with the four specified reproducible data/report files.

- [ ] **Step 1: Write failing deterministic-generation and invariant tests**

```python
def test_generation_is_identical_for_the_fixed_seed(self):
    first = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
    second = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
    self.assertEqual(first, second)

def test_every_consistent_random_case_retains_truth(self):
    cases = generate_cases(seed=20260911, regular_count=10, near_parallel_count=5)
    summary = run_simulation(cases)
    self.assertEqual(summary["consistent_truth_retention_rate"], 1.0)
    self.assertEqual(summary["diameter_crosscheck_max_abs_error_m"], 0.0)
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m unittest src.q1.test_simulate_cases -v`

Expected: missing simulation module.

- [ ] **Step 3: Implement fixed boundary fixtures and seeded random observations**

Observation fixtures contain `cross_zero`, `error_at_positive_bound`, `error_at_negative_bound`, `unbounded_single_observation`, `empty_conflicting_observations`, `duplicate_observations`, `near_parallel_intersection`, and `source_on_arena_boundary`; these are included in `q1_simulation_cases.json` and call `solve_case`. Separately, `run_simulation` creates analytic checks `point_region`, `segment_region`, `equilateral_diameter_circle_failure`, `square_diameter_circle_success`, `clear_radius_exactly_20`, `clear_radius_above_20`, `arc_crosses_zero`, and `rounded_center_counterexample` using the same Task 1-3 production functions; their explicit parameters are recorded in the parameter Markdown and their outputs in both result files. Random cases draw the source uniformly by area within radius 1800, place three stations 200-1400 metres from the source, and add literal sampled errors from `[-1, 1]` degrees before retaining full precision.

- [ ] **Step 4: Implement statistics and Markdown rendering**

The JSON summary records seed, counts, region-status distribution, truth-retention count/rate, diameter and minimum-radius quantiles, maximum diameter cross-check discrepancy, maximum Welzl residual, and each boundary fixture's expected/actual/pass fields. The parameter Markdown states distributions, ranges, units, tolerances, seed, counts, and generation command. The result Markdown leads with pass/fail totals, then gives aggregate statistics and a table with one row per boundary fixture. Markdown numeric values use six decimals while JSON retains full precision.

- [ ] **Step 5: Verify real output writing in a temporary directory**

The test calls `write_outputs(Path(temp_dir), cases, summary)` and parses both JSON files. It asserts both Markdown files contain the seed and all boundary case IDs, ensuring the renderer's observable outputs rather than its source text are tested.

- [ ] **Step 6: Generate repository outputs and verify determinism**

Run: `.venv\Scripts\python.exe src/q1/simulate_cases.py`

Record SHA-256 hashes for the four generated files with `Get-FileHash`, run the same command a second time, and compare the new hashes with the recorded values.

Expected: all four hashes are unchanged; every boundary case passes and consistent truth retention is 100%.

- [ ] **Step 7: Run the complete suite and commit simulations**

Run: `.venv\Scripts\python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v`

```powershell
git add src/q1/simulate_cases.py src/q1/test_simulate_cases.py data/processed/q1_simulation_cases.json data/processed/q1_simulation_parameters.md results/tables/q1_simulation_results.json results/tables/q1_simulation_results.md
git commit -m "q1: 增加边界仿真数据与结果报告"
```

---

### Task 6: Usage documentation and final reproducibility gate

**Files:**
- Create: `src/q1/README.md`
- Modify only if generated content requires clarification: `data/processed/q1_simulation_parameters.md`
- Modify only if generated content requires clarification: `results/tables/q1_simulation_results.md`

**Interfaces:**
- Consumes: all commands and public entry points created in Tasks 1-5.
- Produces: a concise Q1 usage guide with input schema, commands, output interpretation, state semantics, numerical limitations, and exact file map.

- [ ] **Step 1: Write the usage guide from verified behavior**

Document these commands exactly:

```powershell
.venv\Scripts\python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v
.venv\Scripts\python.exe src/q1/main.py data/processed/q1_simulation_cases.json --output results/tables/q1_manual_run.json
.venv\Scripts\python.exe src/q1/simulate_cases.py
```

State that `P` may be empty or unbounded, arena clipping is reported conservatively rather than approximated, and `CLEAR_READY` is emitted only after checking the rounded action center.

- [ ] **Step 2: Run the fresh full verification sequence**

Run the full unit suite, run the JSON entry point on the generated case file, regenerate simulations, run the unit suite again, then run `git diff --check` and `git status --short`. Remove only the temporary `results/tables/q1_manual_run.json` created by this step.

Expected: zero test failures, zero generator drift, no whitespace errors, and only intended Q1 files changed.

- [ ] **Step 3: Commit documentation**

```powershell
git add src/q1/README.md
git commit -m "docs: 补充q1求解器复现说明"
```

- [ ] **Step 4: Request independent code review and address findings**

Compare `main` with `HEAD`. Give the reviewer the approved spec, this plan, exact base/head SHAs, test command, and generated result paths. Fix every critical or important finding through a new failing regression test followed by the smallest production change; rerun the complete verification after each fix.

- [ ] **Step 5: Verify identity, push, and create the Pull Request**

Confirm `git config user.name`, `git config user.email`, the last commit Author/Committer, clean status, and the `origin` URL. Push `q1/geometry-solver` without force and create a PR into `main`. The PR body summarizes geometry algorithms, conservative arena semantics, simulation boundary coverage, exact verification commands, and generated report locations.
