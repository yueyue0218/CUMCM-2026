"""Evidence-terminated local learning environment with private synthetic truth."""

from __future__ import annotations

import math
import random

from src.common.simulator_client import SimulatorClient
from src.q3.adaptive_control import AdaptiveController
from src.q3.offline_simulator import OfflineSimulator, Source
from src.q3.search_clear import build_completion_summary


def sample_sources(seed: int, source_count: int | None = None) -> list[Source]:
    rng = random.Random(seed)
    count = rng.randint(10, 16) if source_count is None else source_count
    if not 0 <= count <= 20:
        raise ValueError('source count must be in 0..20')
    sources = []
    for channel in sorted(rng.sample(range(1, 21), count)):
        radius = 1800*math.sqrt(rng.random())
        angle = rng.uniform(0, 2*math.pi)
        sources.append(Source(channel, (radius*math.cos(angle), radius*math.sin(angle)),
                              rng.uniform(1000, 1500)))
    return sources


class TrainingEnvironment:
    def __init__(self, *, seed: int = 0, source_count: int | None = None,
                 max_free_actions: int = 128, max_free_time_s: float = 6000.0,
                 prediction_limit: int = 4, particle_count: int = 48,
                 failure_penalty: float = 100.0):
        self.seed = seed
        self.source_count = source_count
        self.config = dict(max_free_actions=max_free_actions, max_free_time_s=max_free_time_s,
                           prediction_limit=prediction_limit, particle_count=particle_count)
        self.failure_penalty = failure_penalty
        self.controller = None
        self._actions = []
        self._world = None
        self.forced_steps = 0
        self.steps = 0

    def reset(self):
        # Private environment state is never part of features/selector arguments.
        self._world = OfflineSimulator(sample_sources(self.seed, self.source_count),
                                       bearing_error_deg=1.0, error_field='spatial',
                                       quantize_bearings=True, error_phase=self.seed*0.61803398875)
        client = SimulatorClient('offline-training', transport=self._world.transport)
        client.enter()
        self.controller = AdaptiveController(client, **self.config)
        self.steps, self.forced_steps = 0, 0
        self._actions = self.controller.candidates()
        return self.controller.features(self._actions)

    def step(self, action_index: int):
        if (isinstance(action_index, bool) or not isinstance(action_index, int)
                or not 0 <= action_index < len(self._actions)):
            raise ValueError('action index does not match the supplied snapshot')
        forced = len(self._actions) == 1
        action = self._actions[action_index]
        reward = self.controller.step(action)
        self.steps += 1
        self.forced_steps += int(forced)
        self._actions = self.controller.candidates()
        done = not self._actions
        if done:
            if not self.controller.state.all_cleared:
                reward -= self.failure_penalty
            # The hidden scene is audited only AFTER evidence-based termination.
            if self.controller.state.all_cleared and self._world.cleared != set(self._world.sources):
                raise AssertionError('controller falsely claimed full clearance')
            self.controller.client.exit()
            info = build_completion_summary(self.controller.state, self.controller.client.state.virtual_time_s)
            info.update({'forced_steps': self.forced_steps, 'steps': self.steps,
                         'free_actions': self.controller.free_actions,
                         'free_time_s': self.controller.free_time_s})
            return None, reward, True, info
        return self.controller.features(self._actions), reward, False, {}

    def planner_index(self, features, policy_log_probs=None, policy_weight_s=0.0):
        scores = self.controller.planner_scores(self._actions, features)
        if policy_log_probs is not None:
            # Frozen policy supplies an expansion/ranking prior only. This
            # planner's selected action must never enter an on-policy PPO batch.
            scores = scores-policy_weight_s*policy_log_probs
        return int(scores.argmin())
