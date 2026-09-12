"""CPU recurrent PPO over frozen candidate snapshots and complete real episodes.

Rewards are supplied by the environment in one fixed time scale. This module
never clips rewards, infers hidden world state, or admits planner trajectories.
Each epoch uses full episode backpropagation and reconstructs GRU memory with
the current weights. This intentionally favors correctness over long-run speed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


FEATURE_KEYS = ("global", "channels", "candidates", "candidate_channels", "mask")
FORMAT_VERSION = 1


def snapshot_features(features: Mapping) -> dict[str, np.ndarray]:
    """Own a read-only copy of every input, without sharing mutable array storage."""
    result = {}
    for key in FEATURE_KEYS:
        value = features[key]
        if isinstance(value, Tensor):
            value = value.detach().cpu().numpy()
        dtype = np.bool_ if key == "mask" else np.int64 if key == "candidate_channels" else np.float32
        result[key] = np.array(value, dtype=dtype, copy=True)
        result[key].setflags(write=False)
    return result


def features_to_tensors(snapshot: Mapping, device="cpu") -> dict[str, Tensor]:
    """Copy inputs so tensor mutation cannot alter a recorded trajectory."""
    result = {}
    for key in FEATURE_KEYS:
        dtype = torch.bool if key == "mask" else torch.long if key == "candidate_channels" else torch.float32
        value = snapshot[key]
        if isinstance(value, Tensor):
            result[key] = value.detach().to(device=device, dtype=dtype).clone()
        else:
            result[key] = torch.tensor(np.array(value, copy=True), dtype=dtype, device=device)
    return result


class RecurrentCandidatePolicy(nn.Module):
    def __init__(self, global_dim: int, channel_dim: int, candidate_dim: int, hidden_size: int = 64):
        super().__init__()
        if any(not isinstance(n, int) or n <= 0 for n in (global_dim, channel_dim, candidate_dim, hidden_size)):
            raise ValueError("Feature dimensions and hidden_size must be positive integers")
        self.config = dict(global_dim=global_dim, channel_dim=channel_dim,
                           candidate_dim=candidate_dim, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.channel_encoder = nn.Sequential(nn.Linear(channel_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(global_dim + hidden_size, hidden_size)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(candidate_dim + 2 * hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.checkpoint_metadata = {}

    def forward(self, features: Mapping[str, Tensor], hidden: Tensor | None = None):
        global_features, channels = features["global"], features["channels"]
        candidates, targets, mask = features["candidates"], features["candidate_channels"], features["mask"]
        count = candidates.shape[0] if candidates.ndim == 2 else 0
        expected = ((global_features, (self.config["global_dim"],)),
                    (channels, (20, self.config["channel_dim"])),
                    (candidates, (count, self.config["candidate_dim"])),
                    (targets, (count,)), (mask, (count,)))
        if count < 1 or any(tuple(value.shape) != shape for value, shape in expected):
            raise ValueError("Invalid global/channel/candidate feature shape")
        if mask.dtype != torch.bool or targets.dtype != torch.long:
            raise ValueError("mask must be bool and candidate_channels must be int64")
        if not bool(mask.any()):
            raise ValueError("At least one candidate must be legal")
        if bool(((targets < 0) | (targets >= 20)).any()):
            raise ValueError("Candidate channels must lie in [0, 19]")
        if any(not bool(torch.isfinite(x).all()) for x in (global_features, channels, candidates)):
            raise ValueError("Policy features must be finite")
        if hidden is None:
            hidden = global_features.new_zeros(self.hidden_size)
        if tuple(hidden.shape) != (self.hidden_size,) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("Invalid recurrent hidden state")
        encoded_channels = self.channel_encoder(channels)
        context = torch.cat((global_features, encoded_channels.mean(dim=0)))
        new_hidden = self.gru(context, hidden)
        candidate_context = torch.cat((candidates, encoded_channels[targets],
                                       new_hidden.expand(count, -1)), dim=-1)
        scores = self.candidate_scorer(candidate_context).squeeze(-1)
        value = self.value_head(new_hidden).squeeze(-1)
        if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(value)):
            raise FloatingPointError("Non-finite network output")
        return scores.masked_fill(~mask, -torch.inf), value, new_hidden


@dataclass(frozen=True)
class ActionSample:
    action_index: int
    log_prob: float
    value: float
    hidden: Tensor
    log_probs: np.ndarray


@torch.no_grad()
def sample_action(policy: RecurrentCandidatePolicy, snapshot: Mapping, hidden: Tensor | None = None,
                  *, deterministic: bool = False) -> ActionSample:
    """Sample actual independent-PPO behavior; argmax is for evaluation only."""
    device = next(policy.parameters()).device
    logits, value, new_hidden = policy(features_to_tensors(snapshot, device), hidden)
    distribution = Categorical(logits=logits)
    action = logits.argmax() if deterministic else distribution.sample()
    log_probs = distribution.logits.detach().cpu().numpy().copy()
    log_probs.setflags(write=False)
    return ActionSample(int(action), float(distribution.log_prob(action)), float(value),
                        new_hidden.detach().clone(), log_probs)


@dataclass(frozen=True)
class Transition:
    snapshot: Mapping[str, np.ndarray]
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    forced: bool = False
    behavior: str = "ppo"
    old_log_probs: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "snapshot", snapshot_features(self.snapshot))
        if self.old_log_probs is not None:
            copied = np.array(self.old_log_probs, dtype=np.float32, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, "old_log_probs", copied)


def _validate_episode(episode: Sequence[Transition]):
    if not episode or not episode[-1].done or any(t.done for t in episode[:-1]):
        raise ValueError("Updates require complete episodes with exactly one final terminal step")
    for t in episode:
        if t.behavior not in ("ppo", "forced") or (t.behavior == "forced" and not t.forced):
            raise ValueError("The whole episode must contain actual independent PPO or predetermined forced actions")
        mask = t.snapshot["mask"]
        if mask.ndim != 1 or not mask.any() or not 0 <= t.action_index < len(mask) or not mask[t.action_index]:
            raise ValueError("Recorded action is not legal under the recorded mask")
        if t.forced and mask.sum() != 1:
            raise ValueError("Predetermined forced actions require a singleton behavior mask")
        if not np.isfinite([t.old_log_prob, t.old_value, t.reward]).all():
            raise ValueError("Rewards and old selected probabilities/values must be finite")
        if t.old_log_prob > 1e-5:
            raise ValueError("Selected log probability cannot be positive")


def episode_targets(episode: Sequence[Transition], gae_lambda: float = .95) -> tuple[np.ndarray, np.ndarray]:
    """Gamma=1 GAE; terminal forced suffix uses exact realized cost, once.

    The last free step bootstraps with the entire completed fallback return.
    No GAE attenuation occurs inside that suffix. Earlier free-prefix residuals
    retain lambda attenuation. Every tail state receives its own actual return.
    """
    _validate_episode(episode)
    if not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must lie in [0, 1]")
    size = len(episode)
    returns = np.zeros(size, dtype=np.float64)
    advantages = np.zeros(size, dtype=np.float64)
    tail_start = size
    while tail_start and episode[tail_start - 1].forced:
        tail_start -= 1
    tail_return = 0.
    for i in range(size - 1, tail_start - 1, -1):
        tail_return += episode[i].reward
        returns[i] = tail_return
        advantages[i] = tail_return - episode[i].old_value
    next_advantage = 0.
    for i in range(tail_start - 1, -1, -1):
        if i == tail_start - 1:
            next_value = tail_return if tail_start < size else 0.
        else:
            next_value = episode[i + 1].old_value
        residual = episode[i].reward + next_value - episode[i].old_value
        advantages[i] = residual + gae_lambda * next_advantage
        returns[i] = advantages[i] + episode[i].old_value
        next_advantage = advantages[i]
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise FloatingPointError("Non-finite complete episode targets")
    return advantages, returns


def _replay(policy, tensor_episodes):
    distributions, values = [], []
    for episode in tensor_episodes:
        hidden = None
        for features in episode:
            logits, value, hidden = policy(features, hidden)
            distributions.append(Categorical(logits=logits))
            values.append(value)
    return distributions, torch.stack(values)


def _all_finite(value):
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(v) for v in value)
    if isinstance(value, float):
        return np.isfinite(value)
    return True


def ppo_update(policy: RecurrentCandidatePolicy, optimizer: torch.optim.Optimizer,
               episodes: Sequence[Sequence[Transition]], epochs: int = 4,
               clip_ratio: float = .2, gae_lambda: float = .95, entropy_coef: float = .01,
               value_coef: float = .5, max_grad_norm: float = .5,
               target_kl: float = .01) -> dict:
    """One finite batch of on-policy updates; returns counts/losses/KL diagnostics.

    Exact old-to-new categorical KL is checked after each tentative joint
    actor/value update. Failed checks restore both weights and optimizer state.
    With no old distribution saved, the unchanged start policy is replayed and
    checked against recorded selected probabilities before constructing old KL.
    """
    if not episodes:
        raise ValueError("At least one complete episode is required")
    if epochs < 1 or not 0 < clip_ratio < 1 or max_grad_norm <= 0 or target_kl <= 0:
        raise ValueError("Invalid PPO update limits")
    if entropy_coef < 0 or value_coef < 0:
        raise ValueError("Loss coefficients must be nonnegative")
    target_pairs = [episode_targets(episode, gae_lambda) for episode in episodes]
    transitions = [t for episode in episodes for t in episode]
    device = next(policy.parameters()).device
    tensor_episodes = [[features_to_tensors(t.snapshot, device) for t in episode] for episode in episodes]
    advantages = torch.tensor(np.concatenate([p[0] for p in target_pairs]), dtype=torch.float32, device=device)
    returns = torch.tensor(np.concatenate([p[1] for p in target_pairs]), dtype=torch.float32, device=device)
    choice_indices = [i for i, t in enumerate(transitions) if not t.forced and t.snapshot["mask"].sum() >= 2]
    actor_advantages = advantages[choice_indices].clone()
    normalized = False
    if len(choice_indices) >= 2 and float(actor_advantages.var(unbiased=False)) > 1e-16:
        actor_advantages = (actor_advantages - actor_advantages.mean()) / torch.sqrt(actor_advantages.var(unbiased=False) + 1e-16)
        normalized = True
    with torch.no_grad():
        initial_distributions, _ = _replay(policy, tensor_episodes)
    old_distributions = []
    for index in choice_indices:
        t, current = transitions[index], initial_distributions[index]
        if t.old_log_probs is None:
            old = current
        else:
            raw = torch.tensor(np.array(t.old_log_probs, copy=True), device=device)
            mask = torch.tensor(np.array(t.snapshot["mask"], copy=True), device=device)
            if raw.shape != mask.shape or not bool(torch.isfinite(raw[mask]).all()) or not bool(torch.isneginf(raw[~mask]).all()):
                raise ValueError("Saved old distribution does not match its mask")
            old = Categorical(logits=raw)
        if abs(float(old.logits[t.action_index]) - t.old_log_prob) > 1e-4:
            raise ValueError("Saved selected probability does not match the old policy")
        if not torch.allclose(old.logits, current.logits, atol=1e-4, rtol=1e-5):
            raise ValueError("Stale/off-policy episode: recollect under the current policy")
        old_distributions.append(old)
    old_selected = torch.tensor([transitions[i].old_log_prob for i in choice_indices], device=device)
    stats = dict(episodes=len(episodes), choice_steps=len(choice_indices), value_steps=len(transitions),
                 epochs_completed=0, policy_loss=0., value_loss=0., entropy=0., kl=0.,
                 grad_norm=0., rolled_back=False, advantage_normalized=normalized,
                 mean_return=float(np.mean([sum(t.reward for t in ep) for ep in episodes])))
    for _ in range(epochs):
        distributions, values = _replay(policy, tensor_episodes)
        actor_loss, entropy = values.new_zeros(()), values.new_zeros(())
        if choice_indices:
            new_selected = torch.stack([distributions[i].logits[transitions[i].action_index] for i in choice_indices])
            ratio = (new_selected - old_selected).exp()
            actor_loss = -torch.minimum(ratio * actor_advantages,
                                        ratio.clamp(1-clip_ratio, 1+clip_ratio) * actor_advantages).mean()
            entropy = torch.stack([distributions[i].entropy() for i in choice_indices]).mean()
        value_loss = (values - returns).square().mean()
        loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite PPO loss; update not applied")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
        before_weights, before_optimizer = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
        optimizer.step()
        rejection = None
        kl = 0.
        if not _all_finite(policy.state_dict()) or not _all_finite(optimizer.state_dict()):
            rejection = "nonfinite_update"
        else:
            try:
                with torch.no_grad():
                    updated, _ = _replay(policy, tensor_episodes)
                    if choice_indices:
                        kl = float(torch.stack([torch.distributions.kl_divergence(old, updated[i])
                                                for old, i in zip(old_distributions, choice_indices)]).mean())
                if not np.isfinite(kl) or kl > 1.5 * target_kl:
                    rejection = "kl_limit"
            except (ValueError, FloatingPointError):
                rejection = "nonfinite_update"
        stats.update(policy_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
                     entropy=float(entropy.detach()), kl=kl, grad_norm=float(grad_norm))
        if rejection:
            policy.load_state_dict(before_weights)
            optimizer.load_state_dict(before_optimizer)
            optimizer.zero_grad(set_to_none=True)
            stats.update(rolled_back=True, stop_reason=rejection)
            break
        stats["epochs_completed"] += 1
    return stats


def _primitive_metadata(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_primitive_metadata(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _primitive_metadata(v) for k, v in value.items()}
    raise ValueError("Checkpoint metadata must contain primitive values only")


def save_policy(policy: RecurrentCandidatePolicy, path: str | Path, metadata: dict | None = None):
    """Write a versioned tensor-only model payload; optimizer belongs to the CLI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "config": dict(policy.config),
               "state_dict": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
               "metadata": _primitive_metadata(metadata if metadata is not None else policy.checkpoint_metadata)}
    torch.save(payload, path)


def load_policy(path: str | Path) -> RecurrentCandidatePolicy:
    """Safely load weights without allowing arbitrary pickle object execution."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported policy checkpoint format")
    policy = RecurrentCandidatePolicy(**payload["config"])
    policy.load_state_dict(payload["state_dict"], strict=True)
    if not _all_finite(policy.state_dict()):
        raise ValueError("Checkpoint contains non-finite weights")
    policy.checkpoint_metadata = _primitive_metadata(payload.get("metadata", {}))
    policy.eval()
    return policy
