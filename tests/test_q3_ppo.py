"""Behavioral checks for the independent, complete-episode recurrent PPO core."""
import copy
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from src.q3.ppo import (
    RecurrentCandidatePolicy, Transition, episode_targets, features_to_tensors,
    load_policy, ppo_update, sample_action, save_policy,
)


def features(count=3):
    rng = np.random.default_rng(51 + count)
    return {
        "global": rng.normal(size=4).astype(np.float32),
        "channels": rng.normal(size=(20, 5)).astype(np.float32),
        "candidates": rng.normal(size=(count, 6)).astype(np.float32),
        "candidate_channels": np.arange(count, dtype=np.int64) % 20,
        "mask": np.ones(count, dtype=bool),
    }


def collect(policy, rewards=(-1.0, -2.0, -3.0), tail=True):
    episode, hidden = [], None
    for i, reward in enumerate(rewards):
        forced = tail and i > 0
        snapshot = features(1 if forced else 3)
        sampled = sample_action(policy, snapshot, hidden)
        hidden = sampled.hidden
        episode.append(Transition(snapshot, sampled.action_index, sampled.log_prob,
                                  sampled.value, reward, i == len(rewards)-1,
                                  forced=forced, old_log_probs=sampled.log_probs))
    return episode


class RecurrentPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.policy = RecurrentCandidatePolicy(4, 5, 6, hidden_size=8)

    def test_invalid_actions_have_exact_zero_probability_and_empty_mask_rejected(self):
        x = features()
        x["mask"][1] = False
        logits, value, hidden = self.policy(features_to_tensors(x))
        self.assertEqual(float(logits.detach().softmax(0)[1]), 0.0)
        self.assertEqual(value.ndim, 0)
        self.assertEqual(tuple(hidden.shape), (8,))
        for _ in range(12):
            self.assertNotEqual(sample_action(self.policy, x).action_index, 1)
        x["mask"][:] = False
        with self.assertRaises(ValueError):
            self.policy(features_to_tensors(x))

    def test_transition_copies_snapshots_and_tensor_conversion_cannot_mutate_them(self):
        x = features()
        t = Transition(x, 0, -1.0, 0.0, -1.0, True)
        original = t.snapshot["candidates"].copy()
        x["candidates"][:] = 999
        np.testing.assert_array_equal(t.snapshot["candidates"], original)
        features_to_tensors(t.snapshot)["candidates"][:] = 42
        np.testing.assert_array_equal(t.snapshot["candidates"], original)
        with self.assertRaises(ValueError):
            t.snapshot["candidates"][0, 0] = 1

    def test_full_forced_tail_cost_enters_boundary_once_without_lambda_decay(self):
        episode = [Transition(features(2), 0, -.7, 0., -1., False),
                   Transition(features(1), 0, 0., 0., -2., False, forced=True),
                   Transition(features(1), 0, 0., 0., -100., True, forced=True)]
        advantages, returns = episode_targets(episode, gae_lambda=.1)
        np.testing.assert_allclose(advantages, [-103., -102., -100.])
        np.testing.assert_allclose(returns, [-103., -102., -100.])

    def test_gae_without_forced_tail_and_terminal_validation(self):
        episode = [Transition(features(2), 0, -.7, 3., -1., False),
                   Transition(features(2), 0, -.7, 4., -2., True)]
        advantages, returns = episode_targets(episode, gae_lambda=.5)
        np.testing.assert_allclose(advantages, [-3., -6.])
        np.testing.assert_allclose(returns, [0., -2.])
        with self.assertRaises(ValueError):
            episode_targets(episode[:1])

    def test_real_update_changes_parameters_and_keeps_single_choice_actor_signal(self):
        episode = collect(self.policy)
        before = copy.deepcopy(self.policy.state_dict())
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        stats = ppo_update(self.policy, optimizer, [episode], epochs=2, target_kl=1.)
        self.assertEqual(stats["choice_steps"], 1)
        self.assertEqual(stats["value_steps"], 3)
        self.assertEqual(stats["epochs_completed"], 2)
        self.assertGreater(abs(stats["policy_loss"]), .1)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in self.policy.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in self.policy.state_dict().items()
                            if k.startswith("candidate_scorer")))
        self.assertTrue(all(np.isfinite(v) for v in stats.values() if isinstance(v, float)))

    def test_forced_only_updates_value_without_actor(self):
        x = features(1)
        s = sample_action(self.policy, x)
        t = Transition(x, 0, s.log_prob, s.value, -3., True, forced=True,
                       old_log_probs=s.log_probs)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        stats = ppo_update(self.policy, optimizer, [[t]], epochs=1)
        self.assertEqual(stats["choice_steps"], 0)
        self.assertEqual(stats["policy_loss"], 0.)
        self.assertEqual(stats["entropy"], 0.)
        self.assertEqual(stats["epochs_completed"], 1)

    def test_rejects_planner_trajectory_even_if_only_one_transition_was_replaced(self):
        episode = collect(self.policy)
        t = episode[-1]
        episode[-1] = Transition(t.snapshot, t.action_index, t.old_log_prob,
                                 t.old_value, t.reward, t.done, forced=True,
                                 behavior="planner")
        with self.assertRaises(ValueError):
            ppo_update(self.policy, torch.optim.Adam(self.policy.parameters()), [episode])

    def test_recurrent_gradients_reach_history_and_update_recomputes_each_epoch(self):
        calls = []
        def record(module, args, output):
            hidden = args[1] if len(args) > 1 else None
            calls.append((torch.is_grad_enabled(), hidden is None,
                          hidden is not None and hidden.grad_fn is not None))
        hook = self.policy.register_forward_hook(record)
        episode = collect(self.policy, (-1., -2., -3.), tail=False)
        calls.clear()
        ppo_update(self.policy, torch.optim.Adam(self.policy.parameters(), lr=1e-4),
                   [episode], epochs=2, target_kl=1.)
        hook.remove()
        gradient_calls = [row for row in calls if row[0]]
        self.assertEqual(gradient_calls, [(True, True, False), (True, False, True),
                                         (True, False, True)] * 2)

    def test_final_choice_has_gradient_to_earlier_observation(self):
        first = features_to_tensors(features(3))
        first["global"].requires_grad_(True)
        _, _, hidden = self.policy(first)
        logits, _, _ = self.policy(features_to_tensors(features(4)), hidden)
        logits.log_softmax(0)[0].backward()
        self.assertIsNotNone(first["global"].grad)
        self.assertGreater(float(first["global"].grad.abs().sum()), 1e-8)

    def test_stale_batch_is_rejected_before_any_parameter_update(self):
        episode = collect(self.policy, (-1., -2.), tail=False)
        with torch.no_grad():
            self.policy.candidate_scorer[0].weight.mul_(2.)
        before = copy.deepcopy(self.policy.state_dict())
        with self.assertRaises(ValueError):
            ppo_update(self.policy, torch.optim.Adam(self.policy.parameters()), [episode])
        for key, tensor in self.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[key]))

    def test_singleton_and_empty_snapshot_mask_cannot_disguise_forced_action(self):
        t = Transition(features(3), 0, -.7, 0., -1., True, forced=True)
        with self.assertRaises(ValueError):
            episode_targets([t])

    def test_checkpoint_unknown_version_and_nonprimitive_metadata_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"format_version": 99}, path)
            with self.assertRaises(ValueError):
                load_policy(path)
            with self.assertRaises(ValueError):
                save_policy(self.policy, path, {"arbitrary": object()})

    def test_kl_rejection_rolls_back_network_and_optimizer(self):
        episode = collect(self.policy, (-10., -20.), tail=False)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=10.)
        before = copy.deepcopy(self.policy.state_dict())
        stats = ppo_update(self.policy, optimizer, [episode], epochs=2, target_kl=1e-10)
        self.assertTrue(stats["rolled_back"])
        self.assertEqual(stats["epochs_completed"], 0)
        for key, tensor in self.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[key]))
        self.assertEqual(len(optimizer.state), 0)

    def test_safe_save_load_preserves_recurrent_logits_and_actions(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_policy(self.policy, path, metadata={"feature_version": "test-v1"})
            loaded = load_policy(path)
            self.assertEqual(loaded.checkpoint_metadata["feature_version"], "test-v1")
            h1 = h2 = None
            for n in (3, 1, 5):
                s1 = sample_action(self.policy, features(n), h1, deterministic=True)
                s2 = sample_action(loaded, features(n), h2, deterministic=True)
                self.assertEqual(s1.action_index, s2.action_index)
                np.testing.assert_array_equal(s1.log_probs, s2.log_probs)
                torch.testing.assert_close(s1.hidden, s2.hidden, rtol=0, atol=0)
                h1, h2 = s1.hidden, s2.hidden


if __name__ == "__main__":
    unittest.main()
