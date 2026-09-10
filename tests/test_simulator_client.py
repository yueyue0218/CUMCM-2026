import json
from pathlib import Path
import tempfile
import unittest

from src.common.simulator_client import (
    ActionRejected,
    JsonlRunLogger,
    SimulatorClient,
    SimulatorHTTPError,
)


def encoded(**fields: object) -> bytes:
    return json.dumps(fields).encode("utf-8")


class QueueTransport:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, bytes, float]] = []

    def __call__(self, url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        self.calls.append((url, body, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


class SimulatorClientTests(unittest.TestCase):
    def client(self, transport: QueueTransport) -> SimulatorClient:
        return SimulatorClient(
            "team-1",
            transport=transport,
            retry_backoff_s=0.0,
            max_network_retries=1,
        )

    def test_new_actions_use_distinct_request_ids(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=0, remaining_real_duration_s=99)),
            (200, encoded(accepted=True, virtual_time_s=5, measure_result="no_signal")),
            (200, encoded(accepted=True, virtual_time_s=8, clear_result="no_target_in_range")),
        )
        client = self.client(transport)
        client.enter()
        client.measure((0.0, 0.0), 1)
        client.clear((0.0, 0.0), 1)
        ids = [json.loads(call[1])["request_id"] for call in transport.calls]
        self.assertEqual(len(ids), len(set(ids)))

    def test_network_retry_reuses_exact_request(self) -> None:
        transport = QueueTransport(
            TimeoutError(),
            (200, encoded(accepted=True, virtual_time_s=0, remaining_real_duration_s=88)),
        )
        client = self.client(transport)
        client.enter()
        self.assertEqual(transport.calls[0][0], transport.calls[1][0])
        self.assertEqual(transport.calls[0][1], transport.calls[1][1])

    def test_accepted_false_does_not_update_state(self) -> None:
        transport = QueueTransport((200, encoded(accepted=False, virtual_time_s=0)))
        client = self.client(transport)
        client.state.entered = True
        client.state.virtual_time_s = 42.0
        client.state.position = (7.0, 8.0)
        with self.assertRaises(ActionRejected):
            client.measure((100.0, 200.0), 3)
        self.assertEqual(client.state.virtual_time_s, 42.0)
        self.assertEqual(client.state.position, (7.0, 8.0))
        self.assertEqual(client.state.current_channel, 1)

    def test_clear_does_not_change_current_channel(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=10, clear_result="success"))
        )
        client = self.client(transport)
        client.state.entered = True
        client.state.current_channel = 7
        client.clear((10.0, 20.0), 12)
        self.assertEqual(client.state.current_channel, 7)

    def test_non_success_http_is_rejected_even_if_accepted_true(self) -> None:
        transport = QueueTransport((409, encoded(accepted=True, virtual_time_s=99)))
        client = self.client(transport)
        client.state.entered = True
        client.state.virtual_time_s = 12.0
        with self.assertRaises(SimulatorHTTPError):
            client.measure((0.0, 0.0), 1)
        self.assertEqual(client.state.virtual_time_s, 12.0)

    def test_near_never_reads_bearing(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=5, measure_result="near", svd_deg=123))
        )
        client = self.client(transport)
        client.state.entered = True
        result = client.measure((0.0, 0.0), 1)
        self.assertEqual(result.result, "near")
        self.assertIsNone(result.svd_deg)

    def test_no_signal_remains_an_observation_not_an_absence_claim(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=5, measure_result="no_signal"))
        )
        client = self.client(transport)
        client.state.entered = True
        result = client.measure((0.0, 0.0), 4)
        self.assertEqual(result.result, "no_signal")
        self.assertIsNone(result.svd_deg)

    def test_enter_saves_actual_remaining_runtime(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=0, remaining_real_duration_s=731))
        )
        client = self.client(transport)
        client.enter()
        self.assertEqual(client.state.remaining_real_duration_s, 731)
        self.assertIsNotNone(client.remaining_real_time_s())
        self.assertLessEqual(client.remaining_real_time_s(), 731)

    def test_non_finite_position_is_rejected_locally(self) -> None:
        client = self.client(QueueTransport())
        client.state.entered = True
        with self.assertRaises(ValueError):
            client.measure((float("nan"), 0.0), 1)

    def test_run_logger_writes_trace_and_redacts_robot_id(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=0, remaining_real_duration_s=50))
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "run01"
            logger = JsonlRunLogger(
                run_directory,
                {
                    "algorithm_version": "baseline-v1",
                    "git_commit": "abc123",
                    "parameters": {},
                    "random_seed": None,
                },
            )
            client = SimulatorClient("private-team-id", transport=transport, logger=logger)
            client.enter()
            request = json.loads((run_directory / "requests.jsonl").read_text("utf-8"))
            self.assertEqual(request["payload"]["robot_id"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
