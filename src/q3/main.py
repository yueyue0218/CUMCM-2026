"""Command-line runner for the Q3-v0 fixed coverage discovery baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.common.simulator_client import JsonlRunLogger, SimulatorClient
from src.q3.baseline_scan import (
    BASELINE_NAME,
    DiscoveryState,
    build_summary,
    coverage_points,
    run_fixed_coverage_scan,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool:
    """Return whether the exact working tree state is not reproducibly clean."""

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _new_run_directory(run_root: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    return run_root / f"{timestamp}_baseline_scan"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Q3-v0 deterministic full-channel discovery scan."
    )
    parser.add_argument("--robot-id", required=True, help="current logged-in team ID")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:2026", help="simulator base URL"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/q3/practice"),
        help="parent directory for this run's logs",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    run_directory = _new_run_directory(args.run_root)
    git_dirty = _git_dirty()
    if git_dirty:
        print(
            "WARNING: practice run is using a dirty Git working tree; "
            "the recorded commit alone is not reproducible.",
            file=sys.stderr,
        )
    logger = JsonlRunLogger(
        run_directory,
        {
            "problem": "q3",
            "run_type": "practice",
            "baseline_name": BASELINE_NAME,
            "algorithm_version": "v0",
            "git_commit": _git_commit(),
            "git_dirty": git_dirty,
            "parameters": {
                "target_radius_m": 1800.0,
                "conservative_receive_radius_m": 1000.0,
                "hexagon_radius_m": 1500.0,
                "coverage_points": coverage_points(),
            },
            "random_seed": None,
            "robot_id": args.robot_id,
            "started_at": datetime.now().astimezone().isoformat(),
        },
    )
    client = SimulatorClient(
        args.robot_id,
        base_url=args.base_url,
        logger=logger,
    )
    state = DiscoveryState()
    started = time.monotonic()
    failure_reason: str | None = None
    exit_failure: str | None = None

    try:
        client.enter()
        run_fixed_coverage_scan(client, state=state)
    except Exception as error:  # CLI boundary: preserve partial evidence and exit safely.
        failure_reason = f"{type(error).__name__}: {error}"
    finally:
        if client.state.entered and not client.state.exited:
            try:
                client.exit()
            except Exception as error:  # The simulator may already have closed the session.
                exit_failure = f"{type(error).__name__}: {error}"

        summary = build_summary(state, client.state.virtual_time_s)
        summary.update(
            {
                "program_runtime_s": time.monotonic() - started,
                "failure_reason": failure_reason,
                "exit_failure": exit_failure,
            }
        )
        logger.finish(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"run directory: {run_directory}")
    return 0 if failure_reason is None and exit_failure is None else 1


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
