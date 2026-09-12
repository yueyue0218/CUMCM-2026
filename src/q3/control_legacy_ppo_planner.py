"""对照组：旧 PPO 辅助规划；复用原 hybrid 推理及保存权重，无训练更新。"""


def run_legacy_ppo_planner(client, *, policy, environment_config, state=None, **kwargs):
    from src.q3.policy_runtime import run_adaptive
    return run_adaptive(client, mode='hybrid', policy=policy, state=state,
                        environment_config=environment_config, **kwargs)
