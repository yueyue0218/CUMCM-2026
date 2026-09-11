import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.q1.main import _write_results


ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = ROOT / "src" / "q1" / "main.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


class CommandLineEntryPointTests(unittest.TestCase):
    def run_case(self, case):
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name)
        input_path = directory / "input.json"
        output_path = directory / "output.json"
        input_path.write_text(json.dumps(case), encoding="utf-8")
        completed = subprocess.run(
            [str(PYTHON), str(ENTRY_POINT), str(input_path), "--output", str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        return temporary, completed, output_path

    def test_writes_one_finite_result_and_preserves_case_id(self):
        case = {
            "case_id": "bounded-demo",
            "observations": [
                {"station": [-100.0, 0.0], "bearing_deg": 0.0},
                {"station": [100.0, 0.0], "bearing_deg": 180.0},
                {"station": [0.0, -100.0], "bearing_deg": 90.0},
                {"station": [0.0, 100.0], "bearing_deg": 270.0},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.json"
            output_path = directory / "output.json"
            input_path.write_text(json.dumps(case), encoding="utf-8")
            completed = subprocess.run(
                [str(PYTHON), str(ENTRY_POINT), str(input_path), "--output", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.is_file())
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIsInstance(result, list)
            self.assertEqual(result[0]["case_id"], "bounded-demo")
            diameter = result[0]["problem_1"]["diameter_m"]
            self.assertTrue(math.isfinite(diameter))

    def test_rejects_malformed_top_level_input_concisely(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "bad.json"
            output_path = directory / "output.json"
            input_path.write_text(json.dumps({"cases": {"case_id": "not-a-list"}}), encoding="utf-8")
            completed = subprocess.run(
                [str(PYTHON), str(ENTRY_POINT), str(input_path), "--output", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cases must be a list", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)

    def test_oversized_integer_field_fails_concisely_without_traceback(self):
        temporary, completed, output_path = self.run_case({
            "observations": [{"station": [10 ** 400, 0], "bearing_deg": 0}],
        })
        with temporary:
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("observations[0].station[0]", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse(output_path.exists())

    def test_boolean_numeric_field_fails_concisely_without_traceback(self):
        temporary, completed, output_path = self.run_case({
            "observations": [{"station": [0, 0], "bearing_deg": True}],
        })
        with temporary:
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("observations[0].bearing_deg", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse(output_path.exists())

    def test_serialization_failure_cleans_up_temporary_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output_path = directory / "output.json"

            with self.assertRaises(ValueError):
                _write_results(output_path, [{"not_finite": float("nan")}])

            self.assertFalse(output_path.exists())
            self.assertEqual(list(directory.glob(f".{output_path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
