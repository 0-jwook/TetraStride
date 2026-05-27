from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v25 — stand_still 제거, body_height 강화.

    v24 문제:
      - stand_still(-3.0) vs body_height(-30): 충돌
      - 로봇이 중력에 맞서 능동 관절 제어해야 높이 유지 가능한데,
        stand_still이 그 능동 제어를 패널티로 억제 → 스르륵 주저앉음
      - "가라앉아 굳어있기"가 보상 흑자 = local optimum

    v25 수정:
      - stand_still 제거: 능동 높이 유지 허용
      - body_height: -30→-80 (높이 유지가 가라앉기보다 명확히 유리하게)
      - lin_vel_z: -2→-5 (가라앉는 속도 직접 억제)
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

    # === 높이 패널티 (강화: 가라앉기보다 유지가 유리해야) ===
    target_body_height: float = 0.163
    rew_scale_body_height: float = -80.0   # v24:-30 → v25:-80

    # === stand_still 제거 (능동 관절 제어 허용) ===
    rew_scale_stand_still: float = 0.0     # v24:-3.0 → v25:0.0

    # === 자세 패널티 ===
    rew_scale_non_foot_contact: float = -8.0
    rew_scale_knee_height_stance: float = -30.0
    knee_stance_height_threshold: float = 0.06
    rew_scale_knee_angle: float = -3.0

    # === 안정화 패널티 ===
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.5
    rew_scale_lin_vel_z: float = -5.0     # v24:-2 → v25:-5, 가라앉는 속도 직접 억제
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
