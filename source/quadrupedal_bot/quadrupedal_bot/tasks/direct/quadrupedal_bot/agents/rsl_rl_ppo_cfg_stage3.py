from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage3(RslRlOnPolicyRunnerCfg):
    """Stage 2: Stationary Trot — Standing v3c 전이학습."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 200
    experiment_name = "spot_micro_trot_inplace"

    resume = True
    load_run = "2026-05-22_02-02-36"
    load_checkpoint = "model_1999.pt"
    load_experiment_name = "spot_micro_stance_v3c"

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
class PPORunnerCfgStage4(RslRlOnPolicyRunnerCfg):
    """Stage 3: Forward Walk — Trot in-place 전이학습."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_walk_fwd"

    resume = True
    load_run = "2026-05-22_02-45-19"
    load_checkpoint = "model_1999.pt"
    load_experiment_name = "spot_micro_trot_inplace"

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
    """Stage 4: All-direction + Rotation — Forward Walk 전이학습."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_walk_alldir"

    resume = True
    load_run = "2026-05-22_03-29-33"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_walk_fwd"

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
