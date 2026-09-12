"""Reproducible synthetic Q3 validation; never connects to the official simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.simulator_client import JsonlRunLogger, SimulatorClient
from src.q3.main import _git_commit, _git_dirty
from src.q3.offline_simulator import OfflineSimulator, Source
from src.q3.search_clear import (
    STRATEGY_NAME, SearchClearState, build_completion_summary, run_search_and_clear,
)


def scenarios(seed: int, random_scenes: int):
    yield "empty_sanity", [], 0.0, "constant"
    yield "near_boundary", [Source(1, (3.0, 4.0), 1000.0)], 0.0, "constant"
    edge_sources = [Source(j, (1800 * math.cos(j * math.pi / 10),
                               1800 * math.sin(j * math.pi / 10)), 1000.0)
                    for j in range(1, 21)]
    yield "perimeter_positive_error", edge_sources, 1.0, "constant"
    yield "perimeter_negative_error", edge_sources, -1.0, "constant"
    yield "max_reception_boundary", [Source(20, (1500.0, 0.0), 1500.0)], 1.0, "constant"
    yield "coincident_sources", [Source(j, (700.0, 500.0), 1000.0)
                                  for j in range(1, 21)], -1.0, "constant"
    rng = random.Random(seed)
    for index in range(random_scenes):
        count = 10 + index % 11
        sources = []
        for channel in sorted(rng.sample(range(1, 21), count)):
            radius = 1800 * math.sqrt(rng.random())
            angle = rng.uniform(0, 2 * math.pi)
            sources.append(Source(channel, (radius * math.cos(angle), radius * math.sin(angle)),
                                  rng.uniform(1000, 1500)))
        yield f"random_{index:03d}", sources, 1.0, "spatial"


def audit_virtual_time(world: OfflineSimulator) -> None:
    """Independently sum movement/switch/operation costs from successful requests."""
    position, channel, elapsed = (0.0, 0.0), 1, 0.0
    for record in world.actions:
        request, response, path = record["request"], record["response"], record["path"]
        if path in {"/measure", "/clear"}:
            point = (request["position"]["x"], request["position"]["y"])
            elapsed += math.hypot(point[0] - position[0], point[1] - position[1]) / 5.0
            position = point
            if path == "/measure":
                elapsed += 5 + int(request["channel"] != channel)
                channel = request["channel"]
            else:
                elapsed += 5 if response["clear_result"] == "success" else 3
        if not math.isclose(elapsed, response["virtual_time_s"], rel_tol=1e-12, abs_tol=1e-9):
            raise AssertionError("cumulative virtual time differs from request cost audit")


def source_hashes() -> dict[str, str]:
    paths = sorted((REPO_ROOT / "src/q3").glob("*.py"))
    paths += sorted((REPO_ROOT / "src/common").glob("*.py"))
    paths.append(REPO_ROOT / "src/q2/bayesian_design.py")
    return {p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--random-scenes", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/tables")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs/q3/offline")
    args = parser.parse_args()
    if args.random_scenes < 0:
        parser.error("--random-scenes must be nonnegative")
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    trace_directory = args.run_root / f"{timestamp}_validation"
    metadata = {
        "problem": "q3", "run_type": "offline_synthetic", "algorithm_version": STRATEGY_NAME,
        "git_commit": _git_commit(), "git_dirty": _git_dirty(), "source_sha256": source_hashes(),
        "random_seed": args.seed, "random_scenes": args.random_scenes,
        "parameters": {"bearing_error_bound_deg": 1, "receive_radius_range_m": [1000, 1500]},
        "evidence_scope": "synthetic regression only; not official simulator performance",
    }
    rows, details = [], []
    for name, sources, error, field in scenarios(args.seed, args.random_scenes):
        logger = None
        if name == "perimeter_positive_error":
            logger = JsonlRunLogger(trace_directory, {
                **metadata, "scenario": name, "sources": [asdict(s) for s in sources],
                "bearing_error_deg": error, "error_field": field,
            })
        world = OfflineSimulator(sources, bearing_error_deg=error, error_field=field)
        client = SimulatorClient("offline", transport=world.transport, logger=logger)
        state = SearchClearState()
        started = time.monotonic()
        client.enter()
        try:
            run_search_and_clear(client, state=state)
        finally:
            client.exit()
        summary = build_completion_summary(state, client.state.virtual_time_s, time.monotonic() - started)
        audit_virtual_time(world)
        passed = (state.all_cleared and world.cleared == {s.channel for s in sources}
                  and state.cleared_channels == world.cleared
                  and all(a.result == "success" for a in state.clear_attempts))
        row = {
            "scenario": name, "source_count": len(sources), "cleared_count": len(world.cleared),
            "passed": passed, "virtual_time_s": client.state.virtual_time_s,
            "average_localization_clearance_time_s": summary["average_localization_clearance_time_s"],
            "measure_count": summary["measure_count"], "clear_count": summary["clear_count"],
            "termination_reason": state.termination_reason,
        }
        rows.append(row)
        details.append({**row, "sources": [asdict(s) for s in sources],
                        "bearing_error_deg": error, "error_field": field,
                        "controller_summary": summary})
        if logger:
            logger.finish(summary)
            (trace_directory / "notes.md").write_text(
                "# Q3 offline validation trace\n\n"
                "Synthetic 20-source perimeter case with +1 degree fixed bearing error.\n"
                "This trace is not an official simulator practice/formal result.\n",
                encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {**metadata, "case_count": len(rows), "passed_count": sum(r["passed"] for r in rows),
              "source_count": sum(r["source_count"] for r in rows),
              "cleared_count": sum(r["cleared_count"] for r in rows),
              "max_virtual_time_s": max(r["virtual_time_s"] for r in rows),
              "trace_directory": trace_directory.relative_to(REPO_ROOT).as_posix()
              if trace_directory.is_relative_to(REPO_ROOT) else str(trace_directory),
              "cases": details}
    report_path = args.output_dir / "q3_offline_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    with (args.output_dir / "q3_offline_validation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "source_sha256"}},
                     ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    if report["passed_count"] != report["case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
