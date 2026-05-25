from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage3(RslRlOnPolicyRunnerCfg):
    """Stage 2: TrotInplace — kp=30 Stance 전이학습."""

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 200
    experiment_name = "spot_micro_trot_inplace_kp30"

    resume = True
    load_run = "2026-05-23_03-08-59"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_stance_v8"

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
    """Stage 3: WalkFwd — kp=30 TrotInplace 전이학습."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_walk_fwd_kp30"

    resume = True
    load_run = "2026-05-23_04-46-53"
    load_checkpoint = "model_1999.pt"
    load_experiment_name = "spot_micro_trot_inplace_kp30"

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
    """Stage 4: WalkAllDir — kp=30 WalkFwd 전이학습."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_walk_alldir_kp30"

    resume = True
    load_run = "2026-05-23_06-01-20"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_walk_fwd_kp30"

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
    """Stage 5: InplaceRot — kp=30 WalkAllDir 전이학습."""

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 200
    experiment_name = "spot_micro_inplace_rot_kp30"

    resume = True
    load_run = "2026-05-23_07-53-55"
    load_checkpoint = "model_2999.pt"
    load_experiment_name = "spot_micro_walk_alldir_kp30"

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
