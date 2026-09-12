"""Command-line runner for Q3 completion and the discovery-only comparison."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.simulator_client import JsonlRunLogger, SimulatorClient
from src.q3.search_clear import (
    STRATEGY_NAME, SearchClearState, build_completion_summary, run_search_and_clear,
)
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
        cwd=REPO_ROOT,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool:
    """Return whether the exact working tree state is not reproducibly clean."""

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _formal_runtime_inputs_dirty() -> bool:
    """Reject mutable runtime inputs while allowing generated Q3 evidence.

    Formal runs create new files below ``runs/q3/formal``.  Those files, prior
    practice traces, and exported official logs must not force a code commit
    between the three formal attempts.  Tracked changes anywhere, Git errors,
    or any other untracked file remain a hard failure.
    """

    for command in (
        ["git", "diff", "--quiet", "--exit-code"],
        ["git", "diff", "--cached", "--quiet", "--exit-code"],
    ):
        result = subprocess.run(command, check=False, cwd=REPO_ROOT)
        if result.returncode != 0:
            return True

    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return True
    allowed_prefixes = (
        "runs/q3/practice/",
        "runs/q3/formal/",
        "support/q3_official_logs/",
    )
    paths = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return any(
        normalized and not normalized.startswith(allowed_prefixes)
        for normalized in (path.replace("\\", "/") for path in paths)
    )


def _git_tags_at_head(pattern: str = "q3-formal-*") -> list[str]:
    """Return sorted formal-version tags that point at the current commit."""

    result = subprocess.run(
        ["git", "tag", "--points-at", "HEAD", "--list", pattern],
        capture_output=True,
        check=False,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return []
    return sorted(tag for tag in result.stdout.splitlines() if tag)


def _new_run_directory(run_root: Path, strategy: str = "complete") -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    return run_root / f"{timestamp}_{strategy}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Q3 search, localization and clearance using the Q1 API and finite fallback."
    )
    parser.add_argument("--robot-id", required=True, help="current logged-in team ID")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:2026", help="simulator base URL"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="parent directory for logs (defaults to runs/q3/<run-type>)",
    )
    parser.add_argument(
        "--run-type",
        choices=("practice", "formal"),
        default="practice",
        help="practice rehearsal or one of the three official formal tests",
    )
    parser.add_argument(
        "--case-code",
        help="simulator case code; required for formal tests",
    )
    parser.add_argument("--strategy", choices=("complete", "scan", "planner", "ppo", "hybrid", "efficient", "efficient_v2"), default="efficient_v2",
                        help="batched efficient routes, deterministic baseline, scan, model planner, PPO, or hybrid")
    parser.add_argument("--checkpoint", type=Path, help="trained best.pt/last.pt required by ppo/hybrid")
    parser.add_argument("--policy-threads", type=int, default=1, help="CPU threads for learned-policy inference")
    parser.add_argument("--virtual-limit-s", type=float, default=360000.0,
                        help="stop before this virtual time, at most 360000 seconds")
    parser.add_argument("--exit-reserve-s", type=float, default=20.0,
                        help="real seconds reserved for exit (at least one retry allowance)")
    return parser


def run(args: argparse.Namespace) -> int:
    strategy = getattr(args, "strategy", "complete")
    run_type = getattr(args, "run_type", "practice")
    raw_case_code = getattr(args, "case_code", None)
    case_code = raw_case_code.strip() if isinstance(raw_case_code, str) else None
    virtual_limit = getattr(args, "virtual_limit_s", 360000.0)
    exit_reserve = getattr(args, "exit_reserve_s", 20.0)
    # Validate before entering a simulator session or creating run artifacts.
    if (strategy not in {"complete", "scan", "planner", "ppo", "hybrid", "efficient", "efficient_v2"}
            or run_type not in {"practice", "formal"}
            or not math.isfinite(virtual_limit) or not 0 < virtual_limit <= 360000.0
            or not math.isfinite(exit_reserve) or exit_reserve < 0.0):
        raise ValueError("invalid strategy, run type, virtual limit or exit reserve")
    if raw_case_code is not None and (not case_code or len(case_code) > 128
                                      or any(ord(char) < 32 for char in case_code)):
        raise ValueError("case code must be 1..128 visible characters")
    if run_type == "formal" and case_code is None:
        raise ValueError("formal runs require --case-code before entering the simulator")
    policy, adaptive_config = None, {}
    if strategy in {'efficient','efficient_v2'}:
        from src.q3.experiment_joint_routing import EfficientState, compact_coverage_points, run_efficient, build_efficient_summary
        if strategy == 'efficient_v2':
            from src.q3.experiment_route_pool import run_route_pool as run_efficient, build_route_pool_summary as build_efficient_summary
    if strategy in {"ppo", "hybrid"}:
        checkpoint = getattr(args, "checkpoint", None)
        if checkpoint is None:
            raise ValueError("ppo/hybrid requires --checkpoint before entering the simulator")
        import torch
        threads = getattr(args, 'policy_threads', 1)
        if threads < 1:
            raise ValueError('policy thread count must be positive')
        torch.set_num_threads(threads)
        from src.q3.ppo import load_policy
        from src.q3.policy_runtime import policy_configuration
        policy = load_policy(checkpoint)
        adaptive_config = policy_configuration(policy)
    git_dirty = _git_dirty()
    runtime_inputs_dirty = _formal_runtime_inputs_dirty()
    git_commit = _git_commit()
    formal_tags = _git_tags_at_head()
    if run_type == "formal" and runtime_inputs_dirty:
        raise ValueError(
            "formal runs require clean runtime inputs; only existing Q3 run "
            "artifacts and exported official logs may be untracked"
        )
    if run_type == "formal" and not formal_tags:
        raise ValueError("formal runs require a q3-formal-* tag at the current commit")
    run_root = getattr(args, "run_root", None) or REPO_ROOT / "runs/q3" / run_type
    run_directory = _new_run_directory(run_root, strategy)
    if git_dirty and run_type == "practice":
        print(
            "WARNING: practice run is using a dirty Git working tree; "
            "the recorded commit alone is not reproducible.",
            file=sys.stderr,
        )
    algorithm_version = (
        "route-pool-v2" if strategy == "efficient_v2"
        else "efficient-v1" if strategy == "efficient"
        else "v0" if strategy == "scan"
        else "v2" if strategy in {"planner", "ppo", "hybrid"}
        else "v1"
    )
    logger = JsonlRunLogger(
        run_directory,
        {
            "problem": "q3",
            "run_type": run_type,
            "case_code": case_code,
            "baseline_name": BASELINE_NAME if strategy == "scan" else f"{STRATEGY_NAME}:{strategy}",
            "algorithm_version": algorithm_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "runtime_inputs_dirty": runtime_inputs_dirty,
            "formal_tags": formal_tags,
            "parameters": {
                "target_radius_m": 1800.0,
                "conservative_receive_radius_m": 1000.0,
                "hexagon_radius_m": 1150.0 if strategy in {'efficient','efficient_v2'} else 1500.0,
                "coverage_points": compact_coverage_points() if strategy in {'efficient','efficient_v2'} else coverage_points(),
                "coverage_plan_kind": 'route_pool' if strategy == 'efficient_v2' else 'dynamic' if strategy == 'efficient' else 'fixed_baseline_or_policy',
                "strategy": strategy,
                "virtual_limit_s": virtual_limit,
                "exit_reserve_s": exit_reserve,
                "adaptive_config": adaptive_config,
                "checkpoint": str(getattr(args, "checkpoint", None)),
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
    state = EfficientState() if strategy in {'efficient','efficient_v2'} else DiscoveryState() if strategy == "scan" else SearchClearState()
    started = time.monotonic()
    failure_reason: str | None = None
    exit_failure: str | None = None
    exit_skipped_reason: str | None = None

    try:
        client.enter()
        if strategy in {'efficient','efficient_v2'}:
            run_efficient(client,state=state,virtual_limit_s=virtual_limit,exit_reserve_s=exit_reserve)
        elif strategy == "complete":
            run_search_and_clear(client, state=state, virtual_limit_s=virtual_limit,
                                 exit_reserve_s=exit_reserve)
        elif strategy in {"planner", "ppo", "hybrid"}:
            from src.q3.policy_runtime import run_adaptive
            run_adaptive(client, mode=strategy, policy=policy, state=state,
                         environment_config=adaptive_config, virtual_limit_s=virtual_limit,
                         exit_reserve_s=exit_reserve)
        else:
            # Preserve the historical scan policy, with the same per-action
            # budget guard as the complete policy.
            from src.q3.search_clear import SearchClearController
            budget = SearchClearController(client, SearchClearState(),
                                           virtual_limit_s=virtual_limit,
                                           exit_reserve_s=exit_reserve)

            class BudgetedScanClient:
                def measure(self, position, channel):
                    budget.check_budget(position, channel, "measure")
                    return client.measure(position, channel)

            run_fixed_coverage_scan(BudgetedScanClient(), state=state)
    except Exception as error:  # CLI boundary: preserve partial evidence and exit safely.
        failure_reason = f"{type(error).__name__}: {error}"
        if isinstance(state, SearchClearState):
            state.termination_reason = "error"
            state.failure_detail = failure_reason
    finally:
        if client.pending_action is not None:
            exit_skipped_reason = "unresolved_action"
        elif client.state.entered and not client.state.exited:
            try:
                client.exit()
            except Exception as error:  # The simulator may already have closed the session.
                exit_failure = f"{type(error).__name__}: {error}"

        runtime = time.monotonic() - started
        summary = (build_completion_summary(state, client.state.virtual_time_s, runtime)
                   if strategy != "scan" else build_summary(state, client.state.virtual_time_s))
        if strategy in {'efficient','efficient_v2'}:
            summary = build_efficient_summary(state,client.state.virtual_time_s,runtime)
        if failure_reason is None and isinstance(state, SearchClearState):
            failure_reason = state.failure_detail
        summary.update(
            {
                "program_runtime_s": runtime,
                "failure_reason": failure_reason,
                "exit_failure": exit_failure,
                "exit_skipped_reason": exit_skipped_reason,
                "pending_action": JsonlRunLogger._redact(client.pending_action),
                "last_confirmed_virtual_time_s": client.state.virtual_time_s,
                "strategy": strategy,
                "algorithm_version": algorithm_version,
                "run_type": run_type,
                "case_code": case_code,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "runtime_inputs_dirty": runtime_inputs_dirty,
                "formal_tags": formal_tags,
            }
        )
        logger.finish(summary)
        (run_directory / "notes.md").write_text(
            "# Q3 run\n\n"
            f"- Run type: {run_type}\n"
            f"- Case code: {case_code}\n"
            f"- Strategy: {strategy}\n"
            f"- Algorithm version: {algorithm_version}\n"
            f"- Git commit: {git_commit}\n"
            f"- Formal tags: {', '.join(formal_tags) if formal_tags else None}\n"
            f"- Termination: {summary.get('termination_reason', 'discovery_only')}\n"
            f"- Failure: {failure_reason}\n"
            f"- Exit failure: {exit_failure}\n\n"
            f"- Exit skipped: {summary['exit_skipped_reason']}\n\n"
            "Self-recorded protocol evidence; official encrypted logs are exported separately.\n",
            encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"run directory: {run_directory}")
    if exit_failure is not None or summary.get("termination_reason") == "error":
        return 1
    if strategy != "scan" and not state.all_cleared:
        return 2
    return 0 if failure_reason is None else 1


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
