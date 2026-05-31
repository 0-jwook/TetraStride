from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage1(RslRlOnPolicyRunnerCfg):
    """Stage 1 (서기) PPO new-v2 — leg_extension 스케일 조정으로 보상 균형 복구.

    new-v1 실패: leg_extension(max 48) 지배 → tilt=29°, stance4=7.8%
    new-v2 수정:
      - rew_scale_leg_extension: 12.0 → 6.0 (max 24, foot_contact 12와 균형)
      - rew_scale_foot_alignment: 3.0 (world XY, max 12.0/step)
      - rew_scale_joint_default: -8.0
    """

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_stance_B_v10"

    resume = False

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
        entropy_coef=0.01,   # v31:0.05→v31b:0.01 (policy collapse 방지)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # v31:1e-3→v31b:5e-4 (보수적 업데이트)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
