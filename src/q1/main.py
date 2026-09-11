"""JSON command-line entry point for the Question 1 solver."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.q1.solver import solve_case


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Question 1 JSON cases")
    parser.add_argument("input", type=Path, help="input JSON path")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    return parser.parse_args()


def _cases(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, Mapping) and "cases" in payload:
        raw_cases = payload["cases"]
        if not isinstance(raw_cases, list):
            raise ValueError("cases must be a list")
        cases = raw_cases
    elif isinstance(payload, Mapping):
        cases = [payload]
    else:
        raise ValueError("top-level input must be a case object or an object with cases")

    if not all(isinstance(case, Mapping) for case in cases):
        raise ValueError("each case must be an object")
    return cases


def _write_results(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(results, temporary, ensure_ascii=False, allow_nan=False, indent=2)
        temporary.write("\n")
    temporary_path.replace(path)


def main() -> int:
    arguments = _arguments()
    try:
        payload = json.loads(arguments.input.read_text(encoding="utf-8"))
        cases = _cases(payload)
        results = []
        for case in cases:
            result = solve_case(case)
            if "case_id" in case:
                result = {"case_id": case["case_id"], **result}
            results.append(result)
        _write_results(arguments.output, results)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
