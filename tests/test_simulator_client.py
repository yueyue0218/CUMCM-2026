import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.common.simulator_client import (
    ActionRejected,
    JsonlRunLogger,
    ProtocolError,
    SimulatorClient,
    SimulatorConnectionError,
    SimulatorError,
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

    def test_exhausted_network_retries_have_distinct_exception(self) -> None:
        client = self.client(QueueTransport(TimeoutError(), TimeoutError()))

        with self.assertRaises(SimulatorConnectionError):
            client.enter()

    def test_unresolved_request_blocks_new_actions(self) -> None:
        transport = QueueTransport(TimeoutError(), TimeoutError())
        client = self.client(transport)
        client.state.entered = True
        with self.assertRaises(SimulatorConnectionError):
            client.clear((1.0, 2.0), 1)
        with self.assertRaisesRegex(SimulatorError, "unresolved"):
            client.exit()
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(client.pending_action["path"], "/clear")

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

    def test_non_200_2xx_is_not_treated_as_success(self) -> None:
        transport = QueueTransport((201, encoded(accepted=True, virtual_time_s=99)))
        client = self.client(transport)
        client.state.entered = True

        with self.assertRaises(SimulatorHTTPError):
            client.measure((0.0, 0.0), 1)

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

    def test_enter_resets_official_initial_state(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=0.5, remaining_real_duration_s=700))
        )
        client = self.client(transport)
        client.state.position = (20.0, 30.0)
        client.state.current_channel = 9

        client.enter()

        self.assertEqual(client.state.position, (0.0, 0.0))
        self.assertEqual(client.state.current_channel, 1)
        self.assertEqual(client.state.virtual_time_s, 0.5)

    def test_measure_direction_updates_position_channel_and_bearing(self) -> None:
        transport = QueueTransport(
            (
                200,
                encoded(
                    accepted=True,
                    virtual_time_s=12.25,
                    measure_result="direction",
                    svd_deg=123.45,
                ),
            )
        )
        client = self.client(transport)
        client.state.entered = True

        result = client.measure((100, -200), 20)

        self.assertEqual(result.svd_deg, 123.45)
        self.assertEqual(client.state.position, (100.0, -200.0))
        self.assertEqual(client.state.current_channel, 20)
        self.assertEqual(client.state.virtual_time_s, 12.25)

    def test_clear_updates_position_but_not_channel(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=15.5, clear_result="success"))
        )
        client = self.client(transport)
        client.state.entered = True
        client.state.current_channel = 6

        client.clear((12, 34), 9)

        self.assertEqual(client.state.position, (12.0, 34.0))
        self.assertEqual(client.state.current_channel, 6)

    def test_channel_and_coordinate_limits_are_enforced(self) -> None:
        client = self.client(QueueTransport())
        client.state.entered = True

        for channel in (True, 0, 21, 1.5):
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                client.measure((0.0, 0.0), channel)  # type: ignore[arg-type]
        for position in ((2_000_001.0, 0.0), (0.0, float("inf"))):
            with self.subTest(position=position), self.assertRaises(ValueError):
                client.measure(position, 1)

    def test_exit_ends_local_session_and_blocks_later_actions(self) -> None:
        transport = QueueTransport((200, encoded(accepted=True, virtual_time_s=7.5)))
        client = self.client(transport)
        client.state.entered = True

        client.exit()

        self.assertTrue(client.state.exited)
        with self.assertRaises(SimulatorError):
            client.measure((0.0, 0.0), 1)
        with self.assertRaises(SimulatorError):
            client.exit()

    def test_action_lock_covers_state_update(self) -> None:
        second_transport_call = threading.Event()
        release_first_update = threading.Event()
        first_update_started = threading.Event()
        responses = [
            (200, encoded(accepted=True, virtual_time_s=5, measure_result="no_signal")),
            (200, encoded(accepted=True, virtual_time_s=10, measure_result="no_signal")),
        ]
        call_count = 0

        def transport(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
            nonlocal call_count
            index = call_count
            call_count += 1
            if index == 1:
                second_transport_call.set()
            return responses[index]

        client = SimulatorClient("team-1", transport=transport)
        client.state.entered = True
        original_update = client._update_virtual_time

        def blocking_update(response: dict[str, object]) -> None:
            if response["virtual_time_s"] == 5:
                first_update_started.set()
                self.assertTrue(release_first_update.wait(1.0))
            original_update(response)

        client._update_virtual_time = blocking_update  # type: ignore[method-assign]
        first = threading.Thread(target=client.measure, args=((1.0, 0.0), 1))
        second = threading.Thread(target=client.measure, args=((2.0, 0.0), 2))

        first.start()
        self.assertTrue(first_update_started.wait(1.0))
        second.start()
        self.assertFalse(second_transport_call.wait(0.05))
        release_first_update.set()
        first.join(1.0)
        second.join(1.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(second_transport_call.is_set())
        self.assertEqual(client.state.virtual_time_s, 10.0)
        self.assertEqual(client.state.position, (2.0, 0.0))

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

    def test_run_logger_redacts_sensitive_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "run01"
            JsonlRunLogger(
                run_directory,
                {
                    "robot_id": "private-team-id",
                    "nested": {"password": "private-password"},
                },
            )

            config = json.loads((run_directory / "config.json").read_text("utf-8"))
            self.assertEqual(config["robot_id"], "<redacted>")
            self.assertEqual(config["nested"]["password"], "<redacted>")

    def test_content_type_and_enter_payload_match_protocol(self) -> None:
        class Response:
            status = 200

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return encoded(
                    accepted=True,
                    virtual_time_s=0,
                    remaining_real_duration_s=100,
                )

        with patch(
            "src.common.simulator_client.urllib.request.urlopen",
            return_value=Response(),
        ) as urlopen:
            SimulatorClient("team-1").enter()

        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            request.get_header("Content-type"), "application/json; charset=utf-8"
        )
        self.assertEqual(
            set(json.loads(request.data)), {"arena_id", "robot_id", "request_id"}
        )

    def test_boolean_virtual_time_is_rejected_as_non_numeric(self) -> None:
        transport = QueueTransport(
            (200, encoded(accepted=True, virtual_time_s=True, measure_result="no_signal"))
        )
        client = self.client(transport)
        client.state.entered = True

        with self.assertRaises(ProtocolError):
            client.measure((0.0, 0.0), 1)

    def test_invalid_clock_does_not_corrupt_position_or_time(self) -> None:
        for value in (float("nan"), float("inf"), -1.0, 9.0):
            with self.subTest(value=value):
                client = self.client(QueueTransport((200, encoded(
                    accepted=True, virtual_time_s=value, measure_result="no_signal"))))
                client.state.entered = True
                client.state.virtual_time_s = 10.0
                with self.assertRaises(ProtocolError):
                    client.measure((123.0, 456.0), 2)
                self.assertEqual(client.state.virtual_time_s, 10.0)
                self.assertEqual(client.state.position, (0.0, 0.0))

    def test_nonfinite_bearings_are_rejected_before_state_update(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                client = self.client(QueueTransport((200, encoded(
                    accepted=True, virtual_time_s=5, measure_result="direction", svd_deg=value))))
                client.state.entered = True
                with self.assertRaises(ProtocolError):
                    client.measure((1.0, 2.0), 1)
                self.assertEqual(client.state.virtual_time_s, 0.0)


if __name__ == "__main__":
    unittest.main()
