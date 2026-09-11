import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.simulator_client import MeasureResult, SimulatorClient
from src.q3.baseline_scan import (
    CHANNELS,
    DiscoveryState,
    build_summary,
    coverage_points,
    run_fixed_coverage_scan,
    snake_channel_order,
)


def measure_result(
    result: str, virtual_time_s: float, svd_deg: float | None = None
) -> MeasureResult:
    response: dict[str, object] = {
        "accepted": True,
        "virtual_time_s": virtual_time_s,
        "measure_result": result,
    }
    if result == "direction":
        response["svd_deg"] = svd_deg
    return MeasureResult(result=result, svd_deg=svd_deg, response=response)


class FakeClient:
    def __init__(
        self,
        outcomes: dict[tuple[tuple[float, float], int], str] | None = None,
        fail_at_call: int | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.fail_at_call = fail_at_call
        self.calls: list[tuple[tuple[float, float], int]] = []

    def measure(self, position: tuple[float, float], channel: int) -> MeasureResult:
        if self.fail_at_call is not None and len(self.calls) == self.fail_at_call:
            raise RuntimeError("injected measure failure")
        self.calls.append((position, channel))
        result = self.outcomes.get((position, channel), "no_signal")
        bearing = 123.45 if result == "direction" else None
        return measure_result(result, len(self.calls) * 5.0, bearing)


class CoveragePointTests(unittest.TestCase):
    def test_coverage_points_has_exactly_seven_points(self) -> None:
        self.assertEqual(len(coverage_points()), 7)

    def test_all_coverage_point_coordinates_are_finite(self) -> None:
        self.assertTrue(
            all(math.isfinite(value) for point in coverage_points() for value in point)
        )

    def test_outer_annulus_has_deterministic_distance_bound(self) -> None:
        # The center covers rho <= 1000.  In the outer annulus, the nearest
        # hexagon direction differs by delta <= 30 degrees.  By the cosine law,
        # d^2=rho^2+1500^2-3000*rho*cos(delta).  For fixed delta, d^2 is
        # convex in rho, so its maximum on the closed interval [1000, 1800]
        # is at an endpoint.  Distance increases with delta in [0, 30], so
        # only delta=30 and both rho endpoints are needed.
        endpoint_distances = {
            rho: math.sqrt(
                rho * rho
                + 1500.0**2
                - 2.0 * rho * 1500.0 * math.cos(math.radians(30.0))
            )
            for rho in (1000.0, 1800.0)
        }
        # The endpoint checks are approximately 807.42 m and 901.92 m.
        self.assertAlmostEqual(endpoint_distances[1000.0], 807.42, places=1)
        self.assertAlmostEqual(endpoint_distances[1800.0], 901.92, places=1)
        worst_squared = max(distance * distance for distance in endpoint_distances.values())
        self.assertLess(worst_squared, 1000.0**2)

    def test_dense_sampling_covers_target_disk(self) -> None:
        points = coverage_points()
        max_nearest_distance = 0.0
        # Independent dense regression grid: 361 radii x 1440 bearings.
        for radial_index in range(361):
            radius = 1800.0 * radial_index / 360
            for angle_index in range(1440):
                angle = 2.0 * math.pi * angle_index / 1440
                target = (radius * math.cos(angle), radius * math.sin(angle))
                nearest = min(math.dist(target, point) for point in points)
                max_nearest_distance = max(max_nearest_distance, nearest)
        self.assertLess(max_nearest_distance, 1000.0)


class DiscoveryStateTests(unittest.TestCase):
    def test_direction_marks_channel_detected_and_keeps_bearing(self) -> None:
        state = DiscoveryState()
        state.record_measure((1.0, 2.0), 4, measure_result("direction", 5.0, 87.5))
        self.assertEqual(state.channels[4].status, "detected")
        self.assertEqual(state.channels[4].observations[0].svd_deg, 87.5)

    def test_near_marks_channel_detected(self) -> None:
        state = DiscoveryState()
        state.record_measure((0.0, 0.0), 7, measure_result("near", 5.0))
        self.assertEqual(state.channels[7].status, "detected")
        self.assertIsNone(state.channels[7].observations[0].svd_deg)

    def test_single_no_signal_keeps_channel_unknown(self) -> None:
        state = DiscoveryState()
        state.record_measure((0.0, 0.0), 2, measure_result("no_signal", 5.0))
        self.assertEqual(state.channels[2].status, "unknown")

    def test_unknown_channels_are_excluded_only_after_full_cover(self) -> None:
        state = DiscoveryState()
        state.completed_coverage_points.extend(coverage_points()[:-1])
        with self.assertRaises(ValueError):
            state.finish_full_cover(len(coverage_points()))
        self.assertEqual(state.channels[1].status, "unknown")
        state.completed_coverage_points.append(coverage_points()[-1])
        state.finish_full_cover(len(coverage_points()))
        self.assertEqual(state.channels[1].status, "excluded_after_full_cover")

    def test_any_custom_layout_does_not_complete_full_cover(self) -> None:
        custom_points = [(0.0, 0.0), (25.0, -30.0)]
        state = run_fixed_coverage_scan(
            FakeClient(), points=custom_points  # type: ignore[arg-type]
        )
        self.assertFalse(state.completed_full_cover)
        self.assertEqual(state.unknown_channels, set(CHANNELS))
        self.assertFalse(state.excluded_channels)

    def test_summary_records_actual_custom_layout(self) -> None:
        custom_points = [(12.0, 34.0), (-56.0, 78.0)]
        state = run_fixed_coverage_scan(
            FakeClient(), points=custom_points  # type: ignore[arg-type]
        )
        summary = build_summary(state, 200.0)
        self.assertEqual(summary["coverage_points"], custom_points)
        self.assertEqual(summary["completed_coverage_points"], custom_points)

    def test_detected_channel_is_not_measured_at_later_points(self) -> None:
        points = coverage_points()
        client = FakeClient({(points[0], 3): "direction"})
        state = run_fixed_coverage_scan(client)  # type: ignore[arg-type]
        channel_three_calls = [call for call in client.calls if call[1] == 3]
        self.assertEqual(channel_three_calls, [(points[0], 3)])
        self.assertEqual(state.channels[3].status, "detected")

    def test_snake_channel_order_is_correct(self) -> None:
        remaining = {1, 4, 9, 20}
        self.assertEqual(snake_channel_order(0, remaining), [1, 4, 9, 20])
        self.assertEqual(snake_channel_order(1, remaining), [20, 9, 4, 1])
        self.assertEqual(snake_channel_order(6, remaining), [1, 4, 9, 20])

    def test_scan_uses_snake_order_for_remaining_channels(self) -> None:
        points = coverage_points()[:2]
        client = FakeClient({(points[0], 2): "near"})
        run_fixed_coverage_scan(client, points=points)  # type: ignore[arg-type]
        first_point_channels = [channel for point, channel in client.calls if point == points[0]]
        second_point_channels = [channel for point, channel in client.calls if point == points[1]]
        self.assertEqual(first_point_channels, list(CHANNELS))
        self.assertEqual(second_point_channels, list(range(20, 2, -1)) + [1])

    def test_exception_does_not_mark_full_cover_or_exclude_unknowns(self) -> None:
        state = DiscoveryState()
        client = FakeClient(fail_at_call=25)
        with self.assertRaises(RuntimeError):
            run_fixed_coverage_scan(client, state=state)  # type: ignore[arg-type]
        self.assertFalse(state.completed_full_cover)
        self.assertLess(len(state.completed_coverage_points), len(coverage_points()))
        self.assertTrue(state.unknown_channels)
        self.assertFalse(state.excluded_channels)


class MainRunnerTests(unittest.TestCase):
    def test_mocked_run_writes_all_required_files(self) -> None:
        virtual_time_s = 0.0

        def transport(url: str, body: bytes, timeout_s: float) -> tuple[int, bytes]:
            nonlocal virtual_time_s
            path = "/" + url.rsplit("/", 1)[-1]
            if path == "/enter":
                response = {
                    "accepted": True,
                    "virtual_time_s": virtual_time_s,
                    "remaining_real_duration_s": 1200,
                }
            elif path == "/measure":
                virtual_time_s += 5.0
                response = {
                    "accepted": True,
                    "virtual_time_s": virtual_time_s,
                    "measure_result": "no_signal",
                }
            elif path == "/exit":
                response = {
                    "accepted": True,
                    "virtual_time_s": virtual_time_s,
                    "exit_reason": "user_exit",
                }
            else:
                raise AssertionError(f"unexpected path: {path}")
            return 200, json.dumps(response).encode("utf-8")

        def client_factory(robot_id: str, **kwargs: object) -> SimulatorClient:
            return SimulatorClient(
                robot_id,
                base_url=str(kwargs["base_url"]),
                logger=kwargs["logger"],  # type: ignore[arg-type]
                transport=transport,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            args = argparse.Namespace(
                robot_id="private-team-id",
                base_url="http://127.0.0.1:2026",
                run_root=run_root,
            )
            with (
                patch("src.q3.main.SimulatorClient", side_effect=client_factory),
                patch("src.q3.main._git_commit", return_value="abc123"),
                patch("src.q3.main._git_dirty", return_value=True),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                from src.q3.main import run

                exit_code = run(args)

            run_directories = list(run_root.iterdir())
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(run_directories), 1)
            run_directory = run_directories[0]
            required = {
                "config.json",
                "requests.jsonl",
                "responses.jsonl",
                "summary.json",
            }
            self.assertTrue(required.issubset(path.name for path in run_directory.iterdir()))
            config = json.loads((run_directory / "config.json").read_text("utf-8"))
            summary = json.loads((run_directory / "summary.json").read_text("utf-8"))
            self.assertEqual(config["robot_id"], "<redacted>")
            self.assertEqual(config["git_commit"], "abc123")
            self.assertTrue(config["git_dirty"])
            self.assertIn("WARNING", stderr.getvalue())
            self.assertTrue(summary["completed_full_cover"])
            self.assertEqual(summary["measure_count"], 140)
            self.assertEqual(summary["detected_channels"], [])
            self.assertEqual(summary["excluded_channels"], list(CHANNELS))


if __name__ == "__main__":
    unittest.main()
