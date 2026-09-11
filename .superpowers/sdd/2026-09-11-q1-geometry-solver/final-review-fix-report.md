# Q1 final whole-branch review fix report

## Scope and commit

- Reviewed base context: `main` through `e404346895a7423c3a8535dc6448d90b44a1b2e9`, including the approved design, implementation plan, progress ledger, `review-main-e404346.diff`, current source/tests/docs, and generated artifacts.
- Behavioral fix commit: `e31412570b38973c469f287a43c35e59ae14049f` (`fix(q1): address final whole-branch review`).
- The evidence report is committed separately so it can name the immutable behavioral-fix SHA exactly; all source, tests, documentation changes, and regenerated artifacts are contained in the single behavioral fix commit above.
- Technical disposition: all four findings were valid. The only qualification is that the signed clearance margin is an audit value, not by itself the control status: arena clipping can keep status uncertain with a positive margin, and the squared-distance comparison intentionally applies a scale-aware tolerance.

## Finding 1 — recursive minimum enclosing circle

Verified claim: valid inputs with at least 1,000 distinct points exceeded Python's recursion depth because the old Welzl helper made one recursive call per shuffled point before unwinding.

RED evidence:

```text
.venv/Scripts/python.exe -m unittest \
  src.q1.test_enclosing_circle.MinimumEnclosingCircleTests.test_more_than_one_thousand_distinct_points_avoid_recursion_limit \
  src.q1.test_solver.IntegratedSolverTests.test_many_observations_complete_end_to_end_without_recursion -v

Ran 2 tests ... FAILED (errors=2)
Both errors: RecursionError: maximum recursion depth exceeded
```

GREEN change and evidence:

- Replaced recursion with the standard fixed-seed, randomized incremental nested-boundary algorithm.
- Kept sorted de-duplication before fixed-seed shuffling, forced one/two/three-point boundary-circle construction, collinear three-point fallback, final residual certification, and the public `minimum_enclosing_circle(points, seed=20260911)` API.
- Added a direct 1,200-distinct-point unit-circle regression.
- Added a 501-observation end-to-end solver regression whose returned polygon has at least 1,000 vertices.
- Re-ran the full enclosing-circle oracle suite, including 100 independent brute-force cases and the prior six-point forced-boundary regression: all passed.

## Finding 2 — solver numeric parsing

Verified claim: `float(True)` and `float(False)` silently accepted boolean fields, while converting a 400-digit JSON integer raised an uncaught `OverflowError` and printed a traceback from the CLI.

RED evidence:

```text
.venv/Scripts/python.exe -m unittest \
  src.q1.test_solver.IntegratedSolverTests.test_boolean_numeric_solver_fields_are_rejected_by_field_name \
  src.q1.test_solver.IntegratedSolverTests.test_oversized_integer_is_rejected_as_a_field_specific_value_error \
  src.q1.test_main.CommandLineEntryPointTests.test_oversized_integer_field_fails_concisely_without_traceback \
  src.q1.test_main.CommandLineEntryPointTests.test_boolean_numeric_field_fails_concisely_without_traceback -v

Ran 4 tests ... FAILED (failures=7, errors=1)
Observed: boolean fields accepted or reported generically; oversized coordinate escaped as OverflowError with Traceback.
```

GREEN change and evidence:

- Added one `_finite_number(value, field)` conversion boundary used by station x/y, bearing, half-angle, arena radius, and clear radius.
- It rejects `bool`, catches `TypeError`, `ValueError`, and `OverflowError`, rejects non-finite conversions, and raises a field-specific `ValueError`.
- Added library coverage for booleans in all six numeric field categories plus the oversized coordinate.
- Added real CLI subprocess regressions for a boolean bearing and 400-digit coordinate; both now return nonzero, write no output, include the exact field path on stderr, and include no traceback.
- Targeted GREEN result: 4/4 tests passed.

## Finding 3 — result-based truth retention

Verified claim: the former metric reconstructed only the raw observation half-planes and never consumed the corresponding `solve_case` result, so it could not detect a wrong returned region.

RED evidence:

```text
.venv/Scripts/python.exe -m unittest src.q1.test_simulate_cases -v

ImportError: cannot import name '_region_truth_retained'
Ran 1 test ... FAILED (errors=1)
```

GREEN change and evidence:

- Added returned-region containment that reads each corresponding solver result.
- Polygon containment treats the convex polygon as closed and is orientation-independent; point and segment states use explicit distance checks, including endpoints.
- Kept raw wedge validation as a separately named input diagnostic.
- Renamed JSON/Markdown fields to `random_region_truth_*` and `random_input_halfplane_*`; boundary rows now distinguish `region_truth_retained` from `input_halfplane_consistent`.
- Added a mutation regression: translating every returned polygon vertex by 10,000 m changes region retention from true to false.
- Targeted GREEN result: 9/9 simulation tests passed.
- Regenerated report result: returned-region retention 70/70 and raw input consistency 70/70.

## Finding 4 — explicit clearance margin

Verified claim: the design required an auditable margin, but the solver only returned the threshold inputs and status.

RED evidence:

```text
.venv/Scripts/python.exe -m unittest \
  src.q1.test_solver.IntegratedSolverTests.test_small_bounded_case_is_clear_ready_with_rounded_center \
  src.q1.test_solver.IntegratedSolverTests.test_empty_case_never_claims_clear_ready \
  src.q1.test_solver.IntegratedSolverTests.test_unbounded_case_has_explicit_non_success_state \
  src.q1.test_solver.IntegratedSolverTests.test_rounded_center_must_still_cover_region \
  src.q1.test_simulate_cases.SimulationTests.test_clearance_boundary_checks_use_end_to_end_solver_results -v

Missing `control.clearance_margin_m` produced four KeyError failures.
```

GREEN change and evidence:

- Defined `control.clearance_margin_m = clear_radius_m - rounded_center_max_distance_m` for every bounded result.
- Empty and unbounded results return `null`.
- The exact 20 m boundary reports `0.0`; the 20.000001 m case reports approximately `-0.000001`; the rounded-center counterexample reports a negative margin.
- Status continues to use the existing squared-distance, scale-aware comparison, as required by the design.
- Targeted GREEN result: 5/5 relevant tests passed.

## Full verification

Fresh final sequence:

```text
.venv/Scripts/python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v
Ran 58 tests in 4.013s — OK

.venv/Scripts/python.exe src/q1/main.py data/processed/q1_simulation_cases.json --output results/tables/q1_final_review_manual_run.json
Parsed CLI_RESULTS=78; temporary output then removed.

.venv/Scripts/python.exe src/q1/simulate_cases.py
exit 0; boundary checks 16/16; region retention 70/70; input consistency 70/70.

.venv/Scripts/python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v
Ran 58 tests in 3.965s — OK

git diff --check
exit 0 (only line-ending conversion notices; no whitespace errors).
```

Deterministic regeneration compared SHA-256 before and after another generator run; every comparison returned `stable=True`:

| Artifact | SHA-256 |
|---|---|
| `data/processed/q1_simulation_cases.json` | `3A04ECD7CE74B48D3FE3E5DBF861225691ECEFBB8BC8CD5C8F407C9CE3780A0E` |
| `data/processed/q1_simulation_parameters.md` | `ADAEBF897043843A335988F305DC64B6E328BF5E217F7B9DC5BF222D023C4414` |
| `results/tables/q1_simulation_results.json` | `E80BE3BDB17D5A3833164C5D8CCA62F0C81723B9A1D2F46E2145F107CF037B13` |
| `results/tables/q1_simulation_results.md` | `578DCCED8701CABBFF05FBAC2D83F297D40DA3A5EBB517F9400CDA18A7F024A8` |

## Files changed by the behavioral fix commit

- Core: `src/q1/enclosing_circle.py`, `src/q1/solver.py`, `src/q1/simulate_cases.py`.
- Tests: `src/q1/test_enclosing_circle.py`, `src/q1/test_solver.py`, `src/q1/test_main.py`, `src/q1/test_simulate_cases.py`.
- Documentation: `docs/superpowers/specs/2026-09-11-q1-geometry-solver-design.md`, `src/q1/README.md`.
- Generated artifacts: `data/processed/q1_simulation_cases.json`, `data/processed/q1_simulation_parameters.md`, `results/tables/q1_simulation_results.json`, `results/tables/q1_simulation_results.md`.
