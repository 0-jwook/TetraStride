from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage1(RslRlOnPolicyRunnerCfg):
    """Stage 1 (서기) PPO 설정 — v30 재시작: 56dim obs(per-foot clock 포함) 처음부터."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_stance"

    resume = False   # 56dim obs로 네트워크 구조 변경 → 처음부터

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,   # 0.005→0.01: 탐색 약간 강화 (local optimum 탈출), 0.05는 수렴 방해
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
