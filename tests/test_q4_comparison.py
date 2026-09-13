import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from src.q4.compare import compare
from src.q4.replay import replay


class ComparisonTests(unittest.TestCase):
    def test_controls_share_scenes_and_saved_actions_replay(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            result = compare(root/'cases', strategies=['informed', 'refined', 'adaptive', 'seven_grid'],
                             cases_per_group=1, counts=(10,), seed=270000000)
            self.assertEqual(set(result['strategies']), {'informed', 'refined', 'adaptive', 'seven_grid'})
            sources = []
            for name, row in result['strategies'].items():
                self.assertTrue(row['all_cleared'])
                path = root/'cases'/name/'n10_seed270000000.json'
                sources.append(json.loads(path.read_text())['sources'])
                self.assertTrue(replay(path.parent, root/f'{name}.json')['all_identical'])
            self.assertTrue(all(s == sources[0] for s in sources))

    def test_replay_rejects_missing_case_and_changed_receipt(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            compare(root/'cases', strategies=['informed'], cases_per_group=1, counts=(10,), seed=270000001)
            directory = root/'cases/informed'
            path = directory/'n10_seed270000001.json'
            data = json.loads(path.read_text())
            data['actions'][-1]['response']['virtual_time_s'] += 1
            path.write_text(json.dumps(data))
            self.assertFalse(replay(directory, root/'changed.json')['all_identical'])
            path.unlink()
            with self.assertRaises(ValueError):
                replay(directory, root/'missing.json')
