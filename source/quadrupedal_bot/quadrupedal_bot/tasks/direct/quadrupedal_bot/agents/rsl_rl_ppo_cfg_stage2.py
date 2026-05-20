from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage2(RslRlOnPolicyRunnerCfg):
    """Stage 2 — v48: foot_height 60 + clearance_penalty -15 + v45 전이 (reward 기반 높이 들기)."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "spot_micro_trot"

    resume = True
    load_run = "2026-05-20_09-18-06"   # v45: leg_angle 0.39, knee -1.37, vel 0.54m/s
    load_checkpoint = "model_4999.pt"
    load_experiment_name = "spot_micro_trot"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
        entropy_coef=0.015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
