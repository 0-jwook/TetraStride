from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage1(RslRlOnPolicyRunnerCfg):
    """Stage 1 (서기) PPO v33 — termination_height 복구 + joint_default 강화.

    v32/v32b 실패: termination_height=0.168m > 자연 평형 0.164m
      → 에피소드 평균 1.37초 만에 99% 조기종료 → 학습 붕괴
    v33 수정:
      - termination_height: 0.168 → 0.150 (에피소드 길이 복구)
      - rew_scale_joint_default: -1.0 → -5.0 (관절 default 유지 강화로 sinking 억제)
      hyperparameter는 v31b 유지
    """

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_stance_v33"

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
