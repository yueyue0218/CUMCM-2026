import unittest

from src.q1.solver import solve_case


def observations_around(center, distance=100.0):
    x, y = center
    return [
        {"station": [x - distance, y], "bearing_deg": 0},
        {"station": [x + distance, y], "bearing_deg": 180},
        {"station": [x, y - distance], "bearing_deg": 90},
        {"station": [x, y + distance], "bearing_deg": 270},
    ]


class IntegratedSolverTests(unittest.TestCase):
    symmetric_observations = observations_around((0.0, 0.0))
    conflicting_observations = [
        {"station": [0, 0], "bearing_deg": 180},
        {"station": [1, 0], "bearing_deg": 0},
    ]

    def test_bounded_case_reports_problem_and_control_results_separately(self):
        result = solve_case({"observations": self.symmetric_observations})
        self.assertEqual(result["region"]["status"], "polygon")
        self.assertIn("diameter_m", result["problem_1"])
        self.assertIn("minimum_enclosing_circle", result["problem_1"])
        self.assertNotIn("status", result["problem_1"])
        self.assertIn(
            result["control"]["status"],
            {"CLEAR_READY", "SINGLE_DISK_IMPOSSIBLE", "COVERAGE_UNCERTAIN"},
        )

    def test_small_bounded_case_is_clear_ready_with_rounded_center(self):
        result = solve_case({"observations": self.symmetric_observations})
        self.assertTrue(result["region"]["arena_contains_region"])
        self.assertEqual(result["control"]["status"], "CLEAR_READY")
        self.assertLessEqual(
            result["control"]["rounded_center_max_distance_m"], 20.0
        )

    def test_empty_case_never_claims_clear_ready(self):
        result = solve_case({"observations": self.conflicting_observations})
        self.assertEqual(result["region"]["status"], "empty")
        self.assertEqual(result["control"]["status"], "NO_FEASIBLE_REGION")
        self.assertIsNone(result["problem_1"]["diameter_m"])

    def test_unbounded_case_has_explicit_non_success_state(self):
        result = solve_case({"observations": [
            {"station": [0, 0], "bearing_deg": 45}
        ]})
        self.assertEqual(result["region"]["status"], "unbounded")
        self.assertEqual(result["problem_1"]["diameter_m"], "infinity")
        self.assertEqual(result["control"]["status"], "UNBOUNDED_REGION")

    def test_arena_clipping_does_not_replace_the_pure_region(self):
        result = solve_case({
            "observations": observations_around((1900.0, 0.0)),
            "arena_radius_m": 1800,
        })
        self.assertEqual(result["region"]["status"], "polygon")
        self.assertFalse(result["region"]["arena_contains_region"])
        self.assertGreater(
            max(vertex[0] for vertex in result["region"]["vertices"]), 1800
        )
        self.assertEqual(result["control"]["status"], "COVERAGE_UNCERTAIN")
        self.assertEqual(
            result["control"]["reason"], "arena_clipping_required"
        )

    def test_certified_large_region_is_single_disk_impossible(self):
        result = solve_case({
            "observations": observations_around((0.0, 0.0), distance=1000.0)
        })
        self.assertTrue(result["region"]["arena_contains_region"])
        self.assertGreater(
            result["problem_1"]["minimum_enclosing_circle"]["radius_m"],
            20.0,
        )
        self.assertEqual(
            result["control"]["status"], "SINGLE_DISK_IMPOSSIBLE"
        )

    def test_rounded_center_must_still_cover_region(self):
        point = (0.49, 0.49)
        result = solve_case({
            "observations": [
                {"station": point, "bearing_deg": 0},
                {"station": point, "bearing_deg": 180},
            ],
            "clear_radius_m": 0.5,
            "output_decimals": 0,
        })
        self.assertEqual(result["region"]["status"], "point")
        self.assertEqual(result["control"]["output_center"], [0.0, 0.0])
        self.assertGreater(result["control"]["rounded_center_max_distance_m"], 0.5)
        self.assertEqual(result["control"]["status"], "COVERAGE_UNCERTAIN")
        self.assertEqual(
            result["control"]["reason"], "rounded_center_exceeds_clear_radius"
        )

    def test_solver_survives_observation_polygon_with_obtuse_welzl_boundary(self):
        observations = [
            {
                "station": [-1120.0803785748826, -69.94355919170671],
                "bearing_deg": 3.2920049514572436,
            },
            {
                "station": [1116.5584179542661, 796.7031177618093],
                "bearing_deg": 214.80819535429012,
            },
            {
                "station": [-198.75821632131968, -28.335831098290488],
                "bearing_deg": 8.393848848275573,
            },
            {
                "station": [189.37550499544096, -378.9266495508098],
                "bearing_deg": 116.24359542714485,
            },
            {
                "station": [44.28028899381225, -739.7086375349706],
                "bearing_deg": 92.79516427692383,
            },
        ]
        result = solve_case({"observations": observations})
        self.assertEqual(result["region"]["status"], "polygon")
        self.assertLessEqual(
            result["problem_1"]["minimum_enclosing_circle"]["max_residual_m"],
            1e-9,
        )


if __name__ == "__main__":
    unittest.main()
