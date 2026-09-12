import unittest


class IterationEvaluationTests(unittest.TestCase):
    def test_failed_cases_keep_traces_and_invalidate_speed_comparison(self):
        import io,json,tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch
        from src.q3.evaluate_route_iteration import evaluate
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            with patch('src.q3.evaluate_route_iteration.run_efficient',side_effect=RuntimeError('injected failure')),patch('src.q3.evaluate_route_iteration.run_route_pool',side_effect=RuntimeError('injected failure')):
                report=evaluate(cases_per_group=1,seed=91260912,output_dir=root/'tables',run_root=root/'runs')
            self.assertFalse(report['comparison']['valid'])
            self.assertIsNone(report['comparison']['mean_saved_per_source_s'])
            self.assertEqual(len(report['cases']),8)
            traces=list(Path(report['run_directory']).glob('*/*逐场轨迹.json'))
            self.assertEqual(len(traces),8)
            for path in traces:
                trace=json.loads(path.read_text('utf-8'))
                self.assertIn('injected failure',trace['audit']['failure'])
                self.assertFalse(trace['audit']['all_cleared'])
                self.assertIsNone(trace['audit']['per_source_s'])
                self.assertEqual(trace['actions'][-1]['path'],'/exit')

    def test_paired_summary_uses_equal_strata_and_keeps_failure_invalid(self):
        from src.q3.evaluate_route_iteration import compare_pairs
        rows=[]
        for i,n in enumerate((10,12,14,16)):
            for mode,value in [('previous',300-i*20),('route_pool',290-i*20)]:
                rows.append({'mode':mode,'source_count':n,'seed':i,'all_cleared':True,
                             'per_source_s':value,'total_time_s':n*value,'program_runtime_s':1})
        result=compare_pairs(rows)
        self.assertEqual(result['mean_saved_per_source_s'],10)
        self.assertEqual(result['improved_cases'],4)
        self.assertEqual(result['paired_bootstrap_95ci_saved_s'],[10,10])
        rows[-1]['all_cleared']=False
        self.assertIsNone(compare_pairs(rows)['mean_saved_per_source_s'])


if __name__=='__main__':unittest.main()
