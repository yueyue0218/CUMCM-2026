"""Run one minimal end-to-end smoke test against the official simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/simulator_smoke_test.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.simulator_client import SimulatorClient, SimulatorError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run /enter, /measure, /clear, and /exit once."
    )
    parser.add_argument(
        "--robot-id",
        required=True,
        help="Current logged-in team ID; never stored by this script.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:2026",
        help="Simulator base URL (default: %(default)s).",
    )
    return parser.parse_args()


def run_smoke_test(robot_id: str, base_url: str) -> int:
    client = SimulatorClient(robot_id=robot_id, base_url=base_url)
    exit_attempted = False

    try:
        client.enter()
        print(
            "enter: accepted; "
            f"remaining_real_duration_s={client.state.remaining_real_duration_s}"
        )

        measurement = client.measure((0.0, 0.0), 1)
        print(f"measure: measure_result={measurement.result}")
        if measurement.result == "direction":
            print(f"measure: svd_deg={measurement.svd_deg}")

        clearing = client.clear((0.0, 0.0), 1)
        print(f"clear: clear_result={clearing.result}")

        exit_attempted = True
        client.exit()
        print(f"exit: accepted; virtual_time_s={client.state.virtual_time_s}")
        return 0
    except SimulatorError as error:
        print(
            f"smoke test failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(
            f"smoke test failed unexpectedly: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    finally:
        if client.state.entered and not client.state.exited and not exit_attempted:
            print("cleanup: attempting /exit", file=sys.stderr)
            try:
                client.exit()
                print(
                    f"cleanup: /exit accepted; virtual_time_s={client.state.virtual_time_s}",
                    file=sys.stderr,
                )
            except SimulatorError as error:
                # SimulatorClient has a finite retry count; do not loop here.
                print(
                    f"cleanup: /exit failed: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )


def main() -> int:
    args = parse_args()
    return run_smoke_test(args.robot_id, args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
