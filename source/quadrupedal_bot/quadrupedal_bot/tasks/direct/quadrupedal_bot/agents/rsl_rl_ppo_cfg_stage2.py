from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage2(RslRlOnPolicyRunnerCfg):
    """Stage 2 — v25: 발 높이 강화 (foot_height 10.0, air_time threshold 0.15s)."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "spot_micro_trot"

    resume = True
    load_run = "2026-05-15_12-11-32"   # v25 base: v24b best (vel 0.38, heading 0.76°)
    load_checkpoint = "model_4999.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,   # 높은 초기 노이즈 → 탐색 강화
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
        entropy_coef=0.015,  # 0.01→0.015: late training 탐색 소량 복원 (shuffling local minima 탈출)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
