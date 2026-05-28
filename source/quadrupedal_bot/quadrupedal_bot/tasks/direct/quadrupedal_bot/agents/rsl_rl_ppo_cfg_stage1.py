from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage1(RslRlOnPolicyRunnerCfg):
    """Stage 1 (서기) PPO v31b — v31 설계 유지, policy collapse 방지 hyperparameter 조정.

    v31 문제: entropy_coef=0.05 과다 탐색으로 iter844 이후 policy collapse
    v31b 수정:
      - entropy_coef: 0.05→0.01 (탐색 감소, 안정적 수렴)
      - learning_rate: 1e-3→5e-4 (보수적 업데이트)
    """

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_stance_v31b"

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
