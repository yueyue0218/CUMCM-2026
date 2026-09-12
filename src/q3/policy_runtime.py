"""Shared policy selection and complete-episode offline rollouts."""
from __future__ import annotations

import math
import time

from src.common.simulator_client import SimulatorClient
from src.q3.adaptive_control import FEATURE_VERSION, AdaptiveController
from src.q3.offline_simulator import OfflineSimulator
from src.q3.search_clear import build_completion_summary, run_search_and_clear
from src.q3.training_env import TrainingEnvironment, sample_sources


def policy_configuration(policy):
    metadata = policy.checkpoint_metadata
    if metadata.get('feature_version') != FEATURE_VERSION:
        raise ValueError('checkpoint does not match current candidate/feature version')
    return dict(metadata['environment_config'])


def select_action(controller, actions, features, mode, policy=None, hidden=None,
                  deterministic=True, policy_weight_s=20.0):
    sample = None
    if policy is not None:
        from src.q3.ppo import sample_action
        sample = sample_action(policy, features, hidden, deterministic=deterministic)
        hidden = sample.hidden
    if mode == 'ppo':
        if sample is None:
            raise ValueError('PPO strategy requires a checkpoint')
        index = sample.action_index
    elif mode in {'planner', 'hybrid'}:
        scores = controller.planner_scores(actions, features)
        if mode == 'hybrid':
            if sample is None:
                raise ValueError('hybrid strategy requires a checkpoint')
            scores = scores-policy_weight_s*sample.log_probs
        index = int(scores.argmin())
    else:
        raise ValueError(f'unknown adaptive strategy: {mode}')
    return index, hidden, sample


def rollout(seed, *, mode='ppo', policy=None, environment_config=None,
            source_count=None, deterministic=True, collect=False):
    config = dict(environment_config or {})
    started = time.monotonic()
    if mode == 'complete':
        world = OfflineSimulator(sample_sources(seed, source_count), bearing_error_deg=1,
                                 error_field='spatial', quantize_bearings=True,
                                 error_phase=seed*0.61803398875)
        client = SimulatorClient('offline-evaluation', transport=world.transport)
        client.enter()
        try:
            state = run_search_and_clear(client)
        finally:
            if client.pending_action is None:
                client.exit()
        summary = build_completion_summary(state, client.state.virtual_time_s, time.monotonic()-started)
        if state.all_cleared and world.cleared != set(world.sources):
            raise AssertionError('baseline completion differs from offline truth')
        return summary, []
    env = TrainingEnvironment(seed=seed, source_count=source_count, **config)
    features = env.reset()
    hidden, episode = None, []
    while True:
        actions = env._actions
        index, hidden, sample = select_action(env.controller, actions, features, mode,
                                              policy, hidden, deterministic)
        next_features, reward, done, info = env.step(index)
        if collect:
            if sample is None:
                raise ValueError('collecting training snapshots requires a policy for value estimates')
            from src.q3.ppo import Transition
            episode.append(Transition(features, index, float(sample.log_probs[index]), sample.value,
                                      reward, done, forced=len(actions)==1, behavior=mode,
                                      old_log_probs=sample.log_probs))
        if done:
            info['program_runtime_s'] = time.monotonic()-started
            return info, episode
        features = next_features


def run_adaptive(client, *, mode='planner', policy=None, state=None,
                 environment_config=None, virtual_limit_s=360000.0, exit_reserve_s=20.0):
    controller = AdaptiveController(client, state=state, **(environment_config or {}),
                                    virtual_limit_s=virtual_limit_s, exit_reserve_s=exit_reserve_s)
    hidden = None
    while True:
        actions = controller.candidates()
        if not actions:
            return controller.state
        features = controller.features(actions)
        index, hidden, _ = select_action(controller, actions, features, mode, policy, hidden)
        controller.step(actions[index])
