import unittest
import tempfile
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from src.q3.evaluate_policy import assert_held_out, summarize_rows


class EvaluationTests(unittest.TestCase):
    def test_imitation_and_later_model_selection_training_seeds_are_not_test_seeds(self):
        metadata = {'seed': 10, 'episodes_completed': 2, 'selection_exposure_episodes': 8,
                    'warmup_seed_start': 100, 'warmup_episodes': 3,
                    'validation_seed_start': 200, 'validation': {'case_count': 4}}
        for seed in (10, 17, 100, 102, 200, 203):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                assert_held_out(metadata, seed, 1)
        assert_held_out(metadata, 300, 5)
        metadata['validation_exposure_cases'] = 6
        with self.assertRaises(ValueError):
            assert_held_out(metadata, 205, 1)
        self.assertEqual(metadata['validation']['case_count'], 4)

    def test_failure_does_not_count_as_time_improvement(self):
        rows = [{'mode': 'ppo', 'all_cleared': True, 'virtual_time_s': 100,
                 'improvement_vs_complete_pct': 0.0},
                {'mode': 'ppo', 'all_cleared': False, 'virtual_time_s': 1,
                 'improvement_vs_complete_pct': None}]
        result = summarize_rows(rows, 'ppo')
        self.assertEqual(result['success_count'], 1)
        self.assertEqual(result['mean_virtual_time_s'], 100)
        self.assertEqual(result['mean_paired_improvement_pct'], 0)

    def test_resume_restores_original_validation_set(self):
        from src.q3.train_ppo import build_parser, train
        from src.q3.ppo import load_policy
        from src.q3.policy_runtime import rollout
        from src.q3.validate_offline import source_hashes
        import torch
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            parser = build_parser()
            first = parser.parse_args(['--episodes','1','--batch-episodes','1','--epochs','1',
                                       '--warmup-episodes','0','--free-actions','1','--prediction-limit','0',
                                       '--source-count','1','--validation-cases','1','--hidden-size','8',
                                       '--run-root',directory])
            run = train(first)
            expected_rng = torch.load(run/'training_state.pt', weights_only=True)['torch_rng']
            sampled_rng = []
            def tracked_rollout(*args, **kwargs):
                if kwargs.get('deterministic') is False and not sampled_rng:
                    sampled_rng.append(torch.get_rng_state().clone())
                return rollout(*args, **kwargs)
            changed_sources = source_hashes()
            changed_sources[next(iter(changed_sources))] = 'test-only-changed-hash'
            second = parser.parse_args(['--episodes','1','--batch-episodes','1','--epochs','1',
                                        '--resume',str(run),'--validation-cases','3','--allow-code-change'])
            with patch('src.q3.train_ppo.source_hashes', return_value=changed_sources), \
                    patch('src.q3.train_ppo.rollout', side_effect=tracked_rollout):
                train(second)
            self.assertTrue(torch.equal(sampled_rng[0], expected_rng))
            policy = load_policy(run/'last.pt')
            self.assertEqual(policy.checkpoint_metadata['episodes_completed'], 2)
            self.assertEqual(policy.checkpoint_metadata['validation']['case_count'], 1)
            self.assertTrue((run/'source_snapshot_initial.zip').exists())

    def test_real_entry_loads_checkpoint_and_completes_offline_protocol(self):
        from src.q3.main import build_parser, run
        from src.q3.ppo import RecurrentCandidatePolicy, save_policy
        from src.q3.adaptive_control import FEATURE_VERSION, GLOBAL_DIM, CHANNEL_DIM, CANDIDATE_DIM
        from src.q3.offline_simulator import OfflineSimulator, Source
        from src.common.simulator_client import SimulatorClient
        import json
        world = OfflineSimulator([Source(1, (0,0), 1000)])
        def factory(robot_id, **kwargs):
            return SimulatorClient(robot_id, transport=world.transport, **kwargs)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            checkpoint = Path(directory)/'model.pt'
            policy = RecurrentCandidatePolicy(GLOBAL_DIM,CHANNEL_DIM,CANDIDATE_DIM,8)
            save_policy(policy,checkpoint,{'feature_version':FEATURE_VERSION,
                                           'environment_config':{'max_free_actions':0,'prediction_limit':0}})
            args = build_parser().parse_args(['--robot-id','offline','--strategy','ppo',
                                              '--checkpoint',str(checkpoint),
                                              '--run-root',str(Path(directory)/'runs')])
            with patch('src.q3.main.SimulatorClient',side_effect=factory):
                code = run(args)
            summary = json.loads((next((Path(directory)/'runs').iterdir())/'summary.json').read_text('utf-8'))
        self.assertEqual(code,0)
        self.assertTrue(summary['all_cleared'])
        self.assertTrue(world.exited)


if __name__ == '__main__':
    unittest.main()
