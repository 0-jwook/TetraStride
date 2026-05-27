from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage3(RslRlOnPolicyRunnerCfg):
    """Stage 2: TrotInplace v7 — v4/model_200 전이, 제곱속도패널티-5.0 (drift clamp문제 해결)."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_trot_inplace_v7"

    resume = True
    load_run = "2026-05-27_09-33-08"
    load_checkpoint = "model_200.pt"
    load_experiment_name = "spot_micro_trot_inplace_v4"

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
        entropy_coef=0.02,     # v3: 0.015→0.02, 발 들기 탐색 강화
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfgStage4(RslRlOnPolicyRunnerCfg):
    """Stage 3: WalkFwd — v7 TrotInplace 전이학습 (gait=7.64, vel=0.10m/s, height=16.4cm)."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_walk_fwd_v2"

    resume = True
    load_run = "2026-05-27_10-22-30"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_trot_inplace_v7"

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


@configclass
class PPORunnerCfgStage5(RslRlOnPolicyRunnerCfg):
    """Stage 4: WalkAllDir — v2 WalkFwd 전이학습."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_walk_alldir_v2"

    resume = True
    load_run = "2026-05-27_12-12-48"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_walk_fwd_v2"

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


@configclass
class PPORunnerCfgStage6(RslRlOnPolicyRunnerCfg):
    """Stage 5: InplaceRot — v2 WalkAllDir 전이학습."""

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 200
    experiment_name = "spot_micro_inplace_rot_v2"

    resume = True
    load_run = "2026-05-27_14-04-34"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_walk_alldir_v2"

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
