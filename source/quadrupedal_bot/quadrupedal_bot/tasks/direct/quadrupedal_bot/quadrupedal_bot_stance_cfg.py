from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v24 — stand_still 활성화 + foot_contact 이진화(env 수정).

    v23 문제:
      - stand_still=0: 관절 이탈 패널티 없어서 로봇이 걸어다님
      - foot_contact 비례 보상: 2발 교대로 평균 2.5 유지하는 해킹

    v24 수정:
      - stand_still=-3.0: cmd=0일 때 관절 이탈 강하게 패널티 (12관절×0.1rad=3.6/step)
      - foot_contact 이진화: (num_contacts>=4) → 4발 완전 접지만 보상, 부분 접지 해킹 차단
      - 보상 구조: legged_gym 원칙 — 양의 보상 최소화, regularizer 약하게
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.155

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 0.5
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 4.0    # 이진: 4발 동시 접지만 보상

    # === 핵심 추가: 가만히 있어 (stand_still) ===
    rew_scale_stand_still: float = -3.0    # v23:0 → v24:-3.0, cmd=0일때 관절이탈 패널티

    # === 높이 패널티 ===
    target_body_height: float = 0.163
    rew_scale_body_height: float = -30.0

    # === 자세 패널티 ===
    rew_scale_non_foot_contact: float = -8.0
    rew_scale_knee_height_stance: float = -30.0
    knee_stance_height_threshold: float = 0.06
    rew_scale_knee_angle: float = -3.0

    # === 안정화 패널티 ===
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.5
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_z: float = -0.5
    rew_scale_lin_vel_xy: float = -8.0
    rew_scale_base_drift: float = 0.0

    # === 동작 품질 ===
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -3.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5

    # === 형상 패널티 ===
    rew_scale_shoulder_default: float = -8.0
    rew_scale_joint_default: float = -1.0
    rew_scale_foot_spread: float = -3.0
    rew_scale_foot_side_span: float = -3.0

    # === 비활성 ===
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0

    # === 기타 ===
    rew_scale_dof_pos_limits: float = -1.0
    rew_scale_contact_forces: float = -1e-3
    freeze_gait_phase: bool = True
    orientation_sigma: float = 0.10
    action_scale: float = 0.10
