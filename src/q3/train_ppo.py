"""Train and resume recurrent PPO on complete local Q3 episodes."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.distributions import Categorical

from src.q3.adaptive_control import FEATURE_VERSION, GLOBAL_DIM, CHANNEL_DIM, CANDIDATE_DIM
from src.q3.policy_runtime import policy_configuration, rollout
from src.q3.ppo import RecurrentCandidatePolicy, features_to_tensors, load_policy, ppo_update, save_policy
from src.q3.validate_offline import source_hashes


def validation(policy, seed, count, config):
    summaries = [rollout(seed+i, mode='ppo', policy=policy, environment_config=config)[0]
                 for i in range(count)]
    return {'success_count': sum(s['all_cleared'] for s in summaries), 'case_count': count,
            'mean_virtual_time_s': float(np.mean([s['final_virtual_time_s'] for s in summaries]))}


def imitation_update(policy, optimizer, episode):
    """Separate supervised warm start; planner actions never enter ppo_update."""
    returns, total = [], 0.0
    for step in reversed(episode):
        total += step.reward
        returns.append(total)
    returns.reverse()
    hidden, actor_terms, value_terms = None, [], []
    for step, target in zip(episode, returns):
        logits, value, hidden = policy(features_to_tensors(step.snapshot), hidden)
        if not step.forced:
            actor_terms.append(-Categorical(logits=logits).log_prob(torch.tensor(step.action_index)))
        value_terms.append((value-target)**2)
    loss = (torch.stack(actor_terms).mean() if actor_terms else torch.tensor(0.0))
    loss = loss+0.1*torch.stack(value_terms).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return float(loss.detach())


def save_sample(episode, path):
    arrays = {}
    for i, step in enumerate(episode):
        for key, value in step.snapshot.items():
            arrays[f't{i:04d}_{key}'] = value
        arrays[f't{i:04d}_old_log_probs'] = step.old_log_probs
    arrays['actions'] = np.asarray([s.action_index for s in episode])
    arrays['rewards'] = np.asarray([s.reward for s in episode])
    arrays['old_values'] = np.asarray([s.old_value for s in episode])
    arrays['forced'] = np.asarray([s.forced for s in episode])
    arrays['done'] = np.asarray([s.done for s in episode])
    np.savez_compressed(path, **arrays)


def archive_sources(directory, label):
    with zipfile.ZipFile(directory/f'source_snapshot_{label}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes():
            archive.write(ROOT/name, name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=200, help='additional PPO episodes this invocation')
    parser.add_argument('--batch-episodes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup-episodes', type=int, default=4, help='separate planner imitation episodes on a new run')
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--free-actions', type=int, default=64)
    parser.add_argument('--free-time-s', type=float, default=6000.0)
    parser.add_argument('--prediction-limit', type=int, default=2)
    parser.add_argument('--particle-count', type=int, default=48)
    parser.add_argument('--validation-cases', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--source-count', type=int, default=None, help='optional curriculum/debug count; validation still uses 10..16')
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/q3/training')
    parser.add_argument('--resume', type=Path, help='existing run directory; restores last weights, optimizer and RNG')
    parser.add_argument('--allow-code-change', action='store_true',
                        help='explicitly record a changed implementation and re-evaluate incumbent on resume')
    return parser


def train(args):
    if min(args.episodes, args.batch_episodes, args.epochs, args.validation_cases, args.threads) <= 0:
        raise ValueError('episode, batch, epoch, validation and thread counts must be positive')
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    if args.resume:
        directory = args.resume.resolve()
        policy = load_policy(directory/'last.pt')
        metadata = policy.checkpoint_metadata
        config = policy_configuration(policy)
        saved = torch.load(directory/'training_state.pt', weights_only=True)
        completed, best_time = saved['episodes_completed'], saved['best_validation_time_s']
        args.seed, args.source_count = saved['seed'], saved['source_count']
        # The selection set is a fixed part of the run, not a new CLI default.
        args.validation_cases = metadata['validation']['case_count']
        recorded = json.loads((directory/'config.json').read_text('utf-8'))
        metadata.setdefault('warmup_seed_start', args.seed+3000000000)
        metadata.setdefault('warmup_episodes', recorded['arguments']['warmup_episodes'])
        current_sources = source_hashes()
        code_changed = current_sources != metadata.get('source_sha256')
        if code_changed and not args.allow_code_change:
            raise ValueError('training code changed; use --allow-code-change to record a new segment and re-evaluate best')
        optimizer = torch.optim.Adam(policy.parameters(), lr=saved['learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if metadata['episodes_completed'] != completed:
            raise ValueError('checkpoint and optimizer resume state disagree')
        if code_changed:
            metadata.setdefault('prior_source_versions', []).append({
                'through_episode': completed, 'source_sha256': metadata.get('source_sha256')})
            metadata['source_sha256'] = current_sources
            archive_sources(directory, f'resume_{completed}')
            if (directory/'best.pt').exists():
                incumbent = load_policy(directory/'best.pt')
                assessment = validation(incumbent, metadata['validation_seed_start'], args.validation_cases, config)
                best_time = assessment['mean_virtual_time_s'] if assessment['success_count'] == args.validation_cases else math.inf
                incumbent.checkpoint_metadata['current_selection_assessment'] = {
                    'validation': assessment, 'source_sha256': current_sources,
                    'at_resume_episode': completed}
                save_policy(incumbent, directory/'best.pt', incumbent.checkpoint_metadata)
            with (directory/'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': 'resume', 'episodes_completed': completed,
                                         'code_changed': True, 'validation_cases': args.validation_cases})+'\n')
    else:
        torch.manual_seed(args.seed)
        config = {'max_free_actions': args.free_actions, 'max_free_time_s': args.free_time_s,
                  'prediction_limit': args.prediction_limit, 'particle_count': args.particle_count}
        directory = args.run_root/datetime.now().astimezone().strftime('%Y%m%dT%H%M%S_%f%z_ppo')
        directory.mkdir(parents=True, exist_ok=False)
        policy = RecurrentCandidatePolicy(GLOBAL_DIM, CHANNEL_DIM, CANDIDATE_DIM, args.hidden_size)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        completed, best_time = 0, math.inf
        metadata = {'feature_version': FEATURE_VERSION, 'environment_config': config,
                    'seed': args.seed, 'source_count': args.source_count, 'device': 'cpu',
                    'algorithm': 'recurrent_ppo_gamma1_with_complete_fallback_tail',
                    'source_sha256': source_hashes(), 'status': 'experimental',
                    'validation_seed_start': args.seed+1000000000,
                    'warmup_seed_start': args.seed+3000000000,
                    'warmup_episodes': args.warmup_episodes,
                    'selection_exposure_episodes': 0}
        archive_sources(directory, 'initial')
        (directory/'config.json').write_text(json.dumps({**metadata, 'arguments': {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}},
            ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        for index in range(args.warmup_episodes):
            summary, episode = rollout(args.seed+3000000000+index, mode='planner', policy=policy,
                                       environment_config=config, source_count=args.source_count, collect=True)
            loss = imitation_update(policy, optimizer, episode)
            record = {'stage': 'imitation', 'episode': index, 'loss': loss,
                      'virtual_time_s': summary['final_virtual_time_s']}
            with (directory/'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        # Freeze post-imitation policy before collecting any on-policy PPO batch.
        assessment = validation(policy, metadata['validation_seed_start'], args.validation_cases, config)
        metadata['validation'] = assessment
        metadata['episodes_completed'] = 0
        save_policy(policy, directory/'initial.pt', metadata)
        if assessment['success_count'] == args.validation_cases:
            best_time = assessment['mean_virtual_time_s']
            save_policy(policy, directory/'best.pt', metadata)
    if args.resume:
        # Loading an incumbent constructs a network and consumes random numbers;
        # resume sampling must begin from the saved stream, after that work.
        torch.set_rng_state(saved['torch_rng'])
    target = completed+args.episodes
    while completed < target:
        episodes, summaries = [], []
        for _ in range(min(args.batch_episodes, target-completed)):
            summary, episode = rollout(args.seed+completed+len(episodes), mode='ppo', policy=policy,
                                       environment_config=config, source_count=args.source_count,
                                       deterministic=False, collect=True)
            episodes.append(episode)
            summaries.append(summary)
        if completed == 0:
            save_sample(episodes[0], directory/'trajectory_sample.npz')
        stats = ppo_update(policy, optimizer, episodes, epochs=args.epochs)
        completed += len(episodes)
        assessment = validation(policy, metadata['validation_seed_start'], args.validation_cases, config)
        metadata['episodes_completed'] = completed
        metadata['selection_exposure_episodes'] = completed
        metadata['validation'] = assessment
        save_policy(policy, directory/'last.pt', metadata)
        if assessment['success_count'] == assessment['case_count'] and assessment['mean_virtual_time_s'] < best_time:
            best_time = assessment['mean_virtual_time_s']
            save_policy(policy, directory/'best.pt', metadata)
        torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                    'episodes_completed': completed, 'best_validation_time_s': best_time,
                    'seed': args.seed, 'source_count': args.source_count,
                    'learning_rate': optimizer.param_groups[0]['lr']}, directory/'training_state.pt')
        record = {'stage': 'ppo', 'episodes_completed': completed, **stats,
                  'train_success_count': sum(s['all_cleared'] for s in summaries),
                  'train_mean_virtual_time_s': float(np.mean([s['final_virtual_time_s'] for s in summaries])),
                  'validation': assessment, 'elapsed_s': time.monotonic()-started}
        with (directory/'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record, allow_nan=False), flush=True)
    if (directory/'best.pt').exists():
        best = load_policy(directory/'best.pt')
        provenance = best.checkpoint_metadata
        provenance.update({'selection_exposure_episodes': completed,
                           'warmup_seed_start': metadata['warmup_seed_start'],
                           'warmup_episodes': metadata['warmup_episodes']})
        save_policy(best, directory/'best.pt', provenance)
    print(f'Training directory: {directory}', flush=True)
    return directory


if __name__ == '__main__':
    train(build_parser().parse_args())
