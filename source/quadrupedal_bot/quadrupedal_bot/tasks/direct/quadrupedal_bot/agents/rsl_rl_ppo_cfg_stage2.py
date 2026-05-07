from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage2(RslRlOnPolicyRunnerCfg):
    """Stage 2 — Rudin 2022 방식 처음부터 학습 (standing 편향 없이 velocity로 locomotion 자연 발생)."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 200
    experiment_name = "spot_micro_trot"

    resume = True  # 계속 학습
    load_run = "2026-05-07_19-05-49"   # 처음부터 학습: step1000에서 vel=0.44, air증가 확인
    load_checkpoint = "model_1000.pt"

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
        entropy_coef=0.01,  # 0.002→0.01: 탐색 강화
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
