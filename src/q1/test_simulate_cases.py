import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.q1.simulate_cases import (
    _region_truth_retained,
    generate_cases,
    run_simulation,
    write_outputs,
)
from src.q1.solver import solve_case


OBSERVATION_BOUNDARY_IDS = {
    "cross_zero",
    "error_at_positive_bound",
    "error_at_negative_bound",
    "unbounded_single_observation",
    "empty_conflicting_observations",
    "duplicate_observations",
    "near_parallel_intersection",
    "source_on_arena_boundary",
}

ANALYTIC_BOUNDARY_IDS = {
    "point_region",
    "segment_region",
    "equilateral_diameter_circle_failure",
    "square_diameter_circle_success",
    "clear_radius_exactly_20",
    "clear_radius_above_20",
    "arc_crosses_zero",
    "rounded_center_counterexample",
}


class SimulationTests(unittest.TestCase):
    def test_generation_is_identical_for_the_fixed_seed(self):
        first = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
        second = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
        self.assertEqual(first, second)

    def test_custom_generation_seed_is_preserved_by_the_summary(self):
        cases = generate_cases(seed=12345, regular_count=1, near_parallel_count=1)
        summary = run_simulation(cases)
        self.assertEqual(summary["seed"], 12345)

    def test_every_consistent_random_case_retains_truth(self):
        cases = generate_cases(seed=20260911, regular_count=10, near_parallel_count=5)
        summary = run_simulation(cases)
        self.assertEqual(summary["random_region_truth_retention_rate"], 1.0)
        self.assertEqual(summary["random_input_halfplane_consistency_rate"], 1.0)
        self.assertNotIn("random_consistent_truth_retention_rate", summary)
        self.assertNotIn("consistent_truth_retention_rate", summary)
        self.assertEqual(summary["diameter_crosscheck_max_abs_error_m"], 0.0)

    def test_region_truth_retention_fails_for_perturbed_solver_region(self):
        case = generate_cases(seed=20260911, regular_count=1, near_parallel_count=0)[-1]
        result = solve_case(case)
        self.assertTrue(_region_truth_retained(case, result))

        perturbed = copy.deepcopy(result)
        perturbed["region"]["vertices"] = [
            [vertex[0] + 10000.0, vertex[1] + 10000.0]
            for vertex in perturbed["region"]["vertices"]
        ]
        self.assertFalse(_region_truth_retained(case, perturbed))

    def test_region_truth_retention_handles_closed_point_and_segment(self):
        point_case = {"true_source": [1.0, 2.0]}
        point_result = {"region": {"status": "point", "vertices": [[1.0, 2.0]]}}
        segment_case = {"true_source": [1.0, 0.0]}
        segment_result = {
            "region": {"status": "segment", "vertices": [[0.0, 0.0], [1.0, 0.0]]}
        }

        self.assertTrue(_region_truth_retained(point_case, point_result))
        self.assertTrue(_region_truth_retained(segment_case, segment_result))

    def test_generation_contains_every_observation_boundary_and_requested_counts(self):
        cases = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
        case_ids = {case["case_id"] for case in cases}
        self.assertTrue(OBSERVATION_BOUNDARY_IDS.issubset(case_ids))
        self.assertEqual(sum(case["case_kind"] == "regular_random" for case in cases), 4)
        self.assertEqual(
            sum(case["case_kind"] == "near_parallel_random" for case in cases), 2
        )

    def test_summary_records_all_boundary_checks_and_full_statistics(self):
        cases = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
        summary = run_simulation(cases)
        checks = {check["case_id"]: check for check in summary["boundary_checks"]}
        self.assertEqual(
            set(checks), OBSERVATION_BOUNDARY_IDS | ANALYTIC_BOUNDARY_IDS
        )
        self.assertTrue(all(check["pass"] for check in checks.values()))
        self.assertEqual(summary["counts"]["regular_random"], 4)
        self.assertEqual(summary["counts"]["near_parallel_random"], 2)
        self.assertEqual(summary["random_region_truth_case_count"], 6)
        self.assertEqual(summary["random_region_truth_retention_count"], 6)
        self.assertEqual(summary["random_input_halfplane_case_count"], 6)
        self.assertEqual(summary["random_input_halfplane_consistency_count"], 6)
        self.assertEqual(summary["counts"]["random_truth_cases"], 6)
        self.assertNotIn("random_consistent_cases", summary["counts"])
        self.assertNotIn("consistent_cases", summary["counts"])
        self.assertEqual(
            sum(summary["region_status_distribution"].values()), len(cases)
        )
        self.assertEqual(
            set(summary["diameter_m_quantiles"]),
            {"minimum", "q25", "median", "q75", "maximum"},
        )
        self.assertEqual(
            set(summary["minimum_radius_m_quantiles"]),
            {"minimum", "q25", "median", "q75", "maximum"},
        )
        self.assertLessEqual(summary["welzl_max_residual_m"], 1e-9)
        self.assertIn(
            "diameter_m",
            checks["equilateral_diameter_circle_failure"]["actual"],
        )
        self.assertIn("area_m2", checks["arc_crosses_zero"]["actual"])
        self.assertIn("centroid", checks["arc_crosses_zero"]["actual"])
        self.assertIn(
            "rounded_center_max_distance_m",
            checks["rounded_center_counterexample"]["actual"],
        )

    def test_clearance_boundary_checks_use_end_to_end_solver_results(self):
        summary = run_simulation(
            generate_cases(seed=20260911, regular_count=0, near_parallel_count=0)
        )
        checks = {check["case_id"]: check for check in summary["boundary_checks"]}

        exact = checks["clear_radius_exactly_20"]
        self.assertIn("observations", exact["parameters"])
        self.assertEqual(exact["actual"]["region_status"], "polygon")
        self.assertTrue(exact["actual"]["arena_contains_region"])
        self.assertEqual(exact["actual"]["control_status"], "CLEAR_READY")
        self.assertAlmostEqual(exact["actual"]["diameter_m"], 40.0, places=10)
        self.assertAlmostEqual(
            exact["actual"]["required_radius_m"], 20.0, places=10
        )
        self.assertAlmostEqual(
            exact["actual"]["rounded_center_max_distance_m"], 20.0, places=10
        )
        self.assertAlmostEqual(
            exact["actual"]["clearance_margin_m"], 0.0, places=10
        )

        above = checks["clear_radius_above_20"]
        self.assertIn("observations", above["parameters"])
        self.assertEqual(above["actual"]["region_status"], "polygon")
        self.assertTrue(above["actual"]["arena_contains_region"])
        self.assertEqual(
            above["actual"]["control_status"], "SINGLE_DISK_IMPOSSIBLE"
        )
        self.assertAlmostEqual(
            above["actual"]["diameter_m"], 40.000002, places=10
        )
        self.assertAlmostEqual(
            above["actual"]["required_radius_m"], 20.000001, places=10
        )
        self.assertAlmostEqual(
            above["actual"]["clearance_margin_m"], -0.000001, places=10
        )

    def test_write_outputs_creates_parseable_json_and_complete_markdown(self):
        cases = generate_cases(seed=20260911, regular_count=4, near_parallel_count=2)
        summary = run_simulation(cases)
        all_boundary_ids = OBSERVATION_BOUNDARY_IDS | ANALYTIC_BOUNDARY_IDS

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_outputs(root, cases, summary)
            cases_path = root / "data" / "processed" / "q1_simulation_cases.json"
            parameters_path = (
                root / "data" / "processed" / "q1_simulation_parameters.md"
            )
            results_json_path = (
                root / "results" / "tables" / "q1_simulation_results.json"
            )
            results_md_path = (
                root / "results" / "tables" / "q1_simulation_results.md"
            )

            parsed_cases = json.loads(cases_path.read_text(encoding="utf-8"))
            parsed_summary = json.loads(results_json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed_cases["cases"], cases)
            self.assertEqual(parsed_summary, summary)

            for markdown_path in (parameters_path, results_md_path):
                markdown = markdown_path.read_text(encoding="utf-8")
                self.assertIn("20260911", markdown)
                for case_id in all_boundary_ids:
                    self.assertIn(case_id, markdown)

            results_markdown = results_md_path.read_text(encoding="utf-8")
            first_section = results_markdown.split("\n\n", maxsplit=2)[1]
            self.assertIn("通过", first_section)
            self.assertIn("失败", first_section)
            self.assertIn("1.000000", results_markdown)
            self.assertIn("随机有界区域真值保留", results_markdown)
            self.assertIn("随机输入半平面一致性", results_markdown)
            self.assertNotIn("- 一致案例真值保留", results_markdown)


if __name__ == "__main__":
    unittest.main()
