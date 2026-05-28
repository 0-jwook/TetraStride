from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgStage1(RslRlOnPolicyRunnerCfg):
    """Stage 1 (서기) PPO v34 — termination_height 완화 + FK 다리 뻗음 보상.

    v33 실패: ep_len=74 plateau, 정책이 낮게 앉는 게 최적 전략
      → termination_height=0.150m 과 자연평형 0.163m 사이 13mm 버퍼만 존재
      → alive 보상이 height 보상을 이김: 낮게 앉음 = 긴 에피소드 = 더 많은 누적 보상
    v34 수정:
      - termination_height: 0.150 → 0.100 (버퍼 13mm → 63mm)
      - rew_scale_body_height: 8.0 → 0.0 (root_pos_w 기반 제거)
      - rew_scale_leg_extension: 2.0 × 4발 (FK Gaussian, 목표 0.177m)
      - init_pos z: 0.22 → 0.18 (초기 낙하 44mm → 4mm)
    """

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "spot_micro_stance_v34"

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
