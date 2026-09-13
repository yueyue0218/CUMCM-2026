"""Q3 completion and failure invariants, using a serial offline transport."""

import math
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.common.simulator_client import SimulatorClient
from src.q3 import search_clear
from src.q3.offline_simulator import OfflineSimulator, Source


class SearchClearTests(unittest.TestCase):
    def execute(self, sources, *, error_deg=0.0, **options):
        world = OfflineSimulator(sources, bearing_error_deg=error_deg)
        client = SimulatorClient("offline", transport=world.transport)
        client.enter()
        state = search_clear.SearchClearState()
        search_clear.run_search_and_clear(client, state=state, **options)
        return world, client, state

    def test_near_is_cleared_before_scanning_next_channel(self):
        world, client, state = self.execute([Source(1, (3.0, 4.0), 1000.0)])
        self.assertEqual([a["path"] for a in world.actions[:3]],
                         ["/enter", "/measure", "/clear"])
        self.assertEqual(state.cleared_channels, {1})
        self.assertTrue(state.all_cleared)
        summary = search_clear.build_completion_summary(state, client.state.virtual_time_s)
        self.assertEqual(summary["total_count"], 1)
        self.assertEqual(summary["clearance_ratio"], 1.0)

    def test_perimeter_source_is_discovered_after_initial_no_signal(self):
        world, _, state = self.execute([Source(20, (1800.0, 0.0), 1000.0)])
        self.assertTrue(state.all_cleared)
        self.assertEqual(world.cleared, {20})
        self.assertEqual(state.channels[20].observations[0].measure_result, "no_signal")
        self.assertEqual(state.excluded_channels, set(range(1, 20)))

    def test_twenty_sources_and_boundary_bearing_errors(self):
        sources = [Source(j, (1750 * math.cos(j), 1750 * math.sin(j)), 1000.0)
                   for j in range(1, 21)]
        for error in (-1.0, 1.0):
            with self.subTest(error=error):
                world, _, state = self.execute(sources, error_deg=error)
                self.assertEqual(world.cleared, set(range(1, 21)))
                self.assertTrue(state.all_cleared)

    def test_all_twenty_cleared_does_not_fabricate_unvisited_coverage(self):
        _, _, state = self.execute([Source(j, (0.0, 0.0), 1000.0) for j in range(1, 21)])
        self.assertTrue(state.all_cleared)
        self.assertFalse(state.completed_full_cover)
        self.assertEqual(state.completed_coverage_points, [(0.0, 0.0)])

    def test_all_twenty_first_seen_at_last_point_marks_full_cover(self):
        angle = math.radians(300.0)
        position = (1800 * math.cos(angle), 1800 * math.sin(angle))
        _, _, state = self.execute([Source(j, position, 1000.0) for j in range(1, 21)])
        self.assertTrue(state.all_cleared)
        self.assertEqual(len(state.completed_coverage_points), 7)
        self.assertTrue(state.completed_full_cover)

    def test_analytic_fallback_finishes_without_q1_clear_candidate(self):
        # Suppress only the optional early-clear assessment. Physical observations
        # still come from the independent distance/bearing world.
        from src.q3.localization_control import (
            ChannelLocalizationDecision, ChannelLocalizationEvaluation,
        )
        uncertain = ChannelLocalizationEvaluation(
            ChannelLocalizationDecision.NEEDS_MORE_MEASUREMENT, None, None)
        with patch("src.q3.search_clear.evaluate_channel", return_value=uncertain):
            world, _, state = self.execute([Source(1, (1499.0, 0.0), 1500.0)], error_deg=1.0)
        self.assertTrue(state.all_cleared)
        self.assertEqual(world.cleared, {1})
        self.assertLessEqual(len(state.channels[1].observations), 7)
        self.assertEqual(state.clear_attempts[0].certificate, "analytic_fallback")

    def test_real_budget_exits_without_false_exclusion(self):
        world = OfflineSimulator([], remaining_real_duration_s=1)
        client = SimulatorClient("offline", transport=world.transport)
        client.enter()
        state = search_clear.run_search_and_clear(client)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason, "real_time_budget")
        self.assertEqual(state.excluded_channels, set())
        self.assertEqual(len(world.actions), 1)

    def test_virtual_budget_checks_movement_before_action(self):
        _, client, state = self.execute([], virtual_limit_s=130.0)
        self.assertLessEqual(client.state.virtual_time_s, 130.0)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason, "virtual_time_budget")
        self.assertEqual(state.excluded_channels, set())

    def test_failed_certified_clear_is_conflict_and_is_not_retried(self):
        world = OfflineSimulator([Source(1, (0.0, 0.0), 1000.0)])
        real_transport = world.transport

        def transport(url, body, timeout):
            import json
            status, payload = real_transport(url, body, timeout)
            if url.endswith("/clear"):
                response = json.loads(payload)
                response["clear_result"] = "no_target_in_range"
                payload = json.dumps(response).encode()
            return status, payload

        client = SimulatorClient("offline", transport=transport)
        client.enter()
        state = search_clear.run_search_and_clear(client)
        self.assertFalse(state.all_cleared)
        self.assertFalse(state.cleared_channels)
        self.assertEqual(state.termination_reason, "model_conflict")
        self.assertEqual(len(state.clear_attempts), 1)
        self.assertEqual(state.channels[1].status, "model_conflict")

    def test_no_signal_during_guaranteed_reception_stops_as_conflict(self):
        world = OfflineSimulator([Source(1, (900.0, 0.0), 1000.0)])
        real_transport = world.transport
        measures = 0

        def transport(url, body, timeout):
            nonlocal measures
            import json
            status, payload = real_transport(url, body, timeout)
            if url.endswith("/measure"):
                measures += 1
                if measures == 2:
                    response = json.loads(payload)
                    response["measure_result"] = "no_signal"
                    response.pop("svd_deg", None)
                    payload = json.dumps(response).encode()
            return status, payload

        client = SimulatorClient("offline", transport=transport)
        client.enter()
        state = search_clear.run_search_and_clear(client)
        self.assertEqual(state.termination_reason, "model_conflict")
        self.assertFalse(state.all_cleared)
        self.assertEqual(measures, 2)

    def test_transport_failure_preserves_partial_evidence(self):
        world = OfflineSimulator([])
        client = SimulatorClient("offline", transport=world.transport)
        client.enter()
        state = search_clear.SearchClearState()
        original_measure = client.measure
        count = 0

        def measure(position, channel):
            nonlocal count
            count += 1
            if count == 22:
                raise RuntimeError("injected failure")
            return original_measure(position, channel)

        with patch.object(client, "measure", side_effect=measure):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                search_clear.run_search_and_clear(client, state=state)
        self.assertFalse(state.all_cleared)
        self.assertFalse(state.excluded_channels)
        self.assertEqual(len(state.completed_coverage_points), 1)


class FallbackTests(unittest.TestCase):
    def test_bound_contains_both_radial_and_angular_extremes(self):
        reference = (100.0, -200.0)
        point, upper = search_clear.fallback_step(reference, 359.5, 1500.0)
        for distance in (0.0, 5.0001, 750.0, 1500.0):
            for error in (-1.0, 0.0, 1.0):
                angle = math.radians(359.5 + error)
                source = (reference[0] + distance * math.cos(angle),
                          reference[1] + distance * math.sin(angle))
                self.assertLessEqual(math.dist(point, source), upper)
        self.assertLess(upper, 1000.0)

    def test_nonfinite_or_out_of_model_bounds_are_rejected(self):
        for upper in (0.0, -1.0, float("nan"), float("inf"), 1501.0):
            with self.subTest(upper=upper), self.assertRaises(ValueError):
                search_clear.fallback_step((0.0, 0.0), 0.0, upper)


class CompletionRunnerTests(unittest.TestCase):
    def test_direct_script_help_from_another_directory(self):
        script = Path(__file__).resolve().parents[1] / "src/q3/main.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), "--help"], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--strategy", result.stdout)

    def test_default_runner_clears_and_writes_completion_metrics(self):
        from src.q3.main import build_parser, run
        world = OfflineSimulator([Source(1, (3.0, 4.0), 1000.0)])

        def factory(robot_id, **kwargs):
            return SimulatorClient(robot_id, transport=world.transport, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--robot-id", "offline", "--run-root", directory])
            with (patch("src.q3.main.SimulatorClient", side_effect=factory),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                code = run(args)
            run_path = next(Path(directory).iterdir())
            summary = json.loads((run_path / "summary.json").read_text("utf-8"))
            self.assertEqual(code, 0)
            self.assertTrue(summary["all_cleared"])
            self.assertEqual(summary["cleared_count"], 1)
            self.assertTrue((run_path / "notes.md").exists())
            self.assertTrue(world.exited)

    def test_formal_runner_requires_case_code_before_creating_artifacts(self):
        from src.q3.main import build_parser, run

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "--robot-id", "offline", "--run-type", "formal",
                "--run-root", directory,
            ])
            with self.assertRaisesRegex(ValueError, "require --case-code"):
                run(args)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_formal_runner_records_identity_and_version_metadata(self):
        from src.q3.main import build_parser, run
        world = OfflineSimulator([Source(1, (3.0, 4.0), 1000.0)])

        def factory(robot_id, **kwargs):
            return SimulatorClient(robot_id, transport=world.transport, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "--robot-id", "offline", "--run-type", "formal",
                "--case-code", "Q3-CASE-001", "--run-root", directory,
            ])
            with (patch("src.q3.main.SimulatorClient", side_effect=factory),
                  patch("src.q3.main._git_dirty", return_value=False),
                  patch("src.q3.main._formal_runtime_inputs_dirty", return_value=False),
                  patch("src.q3.main._git_commit", return_value="abc123"),
                  patch("src.q3.main._git_tags_at_head", return_value=["q3-formal-v1"]),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                code = run(args)
            run_path = next(Path(directory).iterdir())
            config = json.loads((run_path / "config.json").read_text("utf-8"))
            summary = json.loads((run_path / "summary.json").read_text("utf-8"))

        self.assertEqual(code, 0)
        for record in (config, summary):
            self.assertEqual(record["run_type"], "formal")
            self.assertEqual(record["case_code"], "Q3-CASE-001")
            self.assertEqual(record["algorithm_version"], "route-pool-v2")
            self.assertEqual(record["git_commit"], "abc123")
            self.assertFalse(record["git_dirty"])
            self.assertFalse(record["runtime_inputs_dirty"])
            self.assertEqual(record["formal_tags"], ["q3-formal-v1"])
        self.assertTrue(world.exited)

    def test_formal_runner_rejects_an_unfrozen_commit(self):
        from src.q3.main import build_parser, run

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "--robot-id", "offline", "--run-type", "formal",
                "--case-code", "Q3-CASE-001", "--run-root", directory,
            ])
            with (patch("src.q3.main._git_dirty", return_value=False),
                  patch("src.q3.main._formal_runtime_inputs_dirty", return_value=False),
                  patch("src.q3.main._git_commit", return_value="abc123"),
                  patch("src.q3.main._git_tags_at_head", return_value=[]),
                  self.assertRaisesRegex(ValueError, "q3-formal-\\*") ):
                run(args)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_formal_input_check_allows_only_generated_q3_evidence(self):
        from src.q3.main import _formal_runtime_inputs_dirty

        clean = subprocess.CompletedProcess([], 0)
        allowed = subprocess.CompletedProcess(
            [], 0,
            stdout=(
                b"runs/q3/practice/old/summary.json\0"
                b"runs/q3/formal/first/summary.json\0"
                b"support/q3_official_logs/original.log\0"
            ),
        )
        with patch("src.q3.main.subprocess.run", side_effect=[clean, clean, allowed]):
            self.assertFalse(_formal_runtime_inputs_dirty())

        forbidden = subprocess.CompletedProcess(
            [], 0, stdout=b"src/q3/untracked_runtime_change.py\0"
        )
        with patch("src.q3.main.subprocess.run", side_effect=[clean, clean, forbidden]):
            self.assertTrue(_formal_runtime_inputs_dirty())

    def test_incomplete_runner_returns_nonzero_and_still_exits(self):
        from src.q3.main import build_parser, run
        world = OfflineSimulator([], remaining_real_duration_s=1)

        def factory(robot_id, **kwargs):
            return SimulatorClient(robot_id, transport=world.transport, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--robot-id", "offline", "--run-root", directory])
            with (patch("src.q3.main.SimulatorClient", side_effect=factory),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                code = run(args)
            summary = json.loads((next(Path(directory).iterdir()) / "summary.json").read_text("utf-8"))
            self.assertEqual(code, 2)
            self.assertFalse(summary["all_cleared"])
            self.assertEqual(summary["termination_reason"], "real_time_budget")
            self.assertTrue(world.exited)

    def test_uncertain_action_blocks_new_exit_and_preserves_pending_request(self):
        from src.q3.main import build_parser, run
        for fault in ("lost_clear_receipt", "malformed_direction", "http_500"):
            with self.subTest(fault=fault):
                world = OfflineSimulator([Source(1, (0.0, 0.0), 1000.0)])
                requests = []

                def transport(url, body, timeout):
                    path = "/" + url.rsplit("/", 1)[-1]
                    requests.append((path, body))
                    status, payload = world.transport(url, body, timeout)
                    if path == "/clear" and fault == "lost_clear_receipt":
                        raise TimeoutError("receipt lost after execution")
                    if path == "/measure" and fault == "malformed_direction":
                        response = json.loads(payload)
                        response["measure_result"] = "direction"
                        payload = json.dumps(response).encode()
                    if path == "/measure" and fault == "http_500":
                        status = 500
                    return status, payload

                def factory(robot_id, **kwargs):
                    return SimulatorClient(robot_id, transport=transport, retry_backoff_s=0, **kwargs)

                with tempfile.TemporaryDirectory() as directory:
                    args = build_parser().parse_args(["--robot-id", "offline-secret", "--run-root", directory])
                    with (patch("src.q3.main.SimulatorClient", side_effect=factory),
                          redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO())):
                        code = run(args)
                    summary = json.loads((next(Path(directory).iterdir()) / "summary.json").read_text("utf-8"))
                self.assertEqual(code, 1)
                self.assertNotIn("/exit", [r[0] for r in requests])
                self.assertFalse(summary["all_cleared"])
                self.assertIsNotNone(summary["pending_action"])
                self.assertNotIn("offline-secret", stdout.getvalue())
                if fault == "lost_clear_receipt":
                    clear_requests = [body for path, body in requests if path == "/clear"]
                    self.assertEqual(len(set(clear_requests)), 1)
                    self.assertEqual(world.cleared, {1})
                    self.assertEqual(summary["cleared_count"], 0)

    def test_lost_exit_receipt_is_not_reported_as_skipped_exit(self):
        from src.q3.main import build_parser, run
        world = OfflineSimulator([])

        def transport(url, body, timeout):
            result = world.transport(url, body, timeout)
            if url.endswith("/exit"):
                raise TimeoutError("exit receipt lost")
            return result

        def factory(robot_id, **kwargs):
            return SimulatorClient(robot_id, transport=transport, retry_backoff_s=0, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--robot-id", "offline", "--run-root", directory])
            with (patch("src.q3.main.SimulatorClient", side_effect=factory),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                code = run(args)
            summary = json.loads((next(Path(directory).iterdir()) / "summary.json").read_text("utf-8"))
        self.assertEqual(code, 1)
        self.assertIsNotNone(summary["exit_failure"])
        self.assertIsNone(summary["exit_skipped_reason"])
        self.assertEqual(summary["pending_action"]["path"], "/exit")
        self.assertTrue(world.exited)


if __name__ == "__main__":
    unittest.main()
