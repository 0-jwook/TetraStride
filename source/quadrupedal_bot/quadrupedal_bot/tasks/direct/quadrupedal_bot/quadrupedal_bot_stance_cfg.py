from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v27 — foot_spread 제거 + alive 강화.

    v26 문제:
      1. foot_spread(-3, target=0.12m): 자연 발간격 > 0.12m → 가라앉으면 패널티 감소
         → 몸을 낮추는 주요 인센티브
      2. foot_side_span(-3): 동일 원리로 가라앉기 보조 인센티브
      3. alive=0.5: 전체 보상 10.5/step의 4.7% → 생존 인센티브 미미

    v27 수정:
      1. foot_spread: -3→0 (제거)
      2. foot_side_span: -3→0 (제거)
      3. alive: 0.5→2.0 (생존 인센티브 19%로 강화)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.155

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 2.0           # v26:0.5 → v27:2.0 (생존 19%로 강화)
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 4.0    # 이진: 4발 동시 접지만 보상

    # === 높이 목표 ===
    target_body_height: float = 0.177
    rew_scale_body_height: float = -80.0

    # === stand_still 비활성 유지 ===
    rew_scale_stand_still: float = 0.0

    # === 자세 패널티 ===
    rew_scale_non_foot_contact: float = -8.0
    rew_scale_knee_height_stance: float = -30.0
    knee_stance_height_threshold: float = 0.06
    rew_scale_knee_angle: float = -3.0

    # === 안정화 패널티 ===
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.5
    rew_scale_lin_vel_z: float = -5.0
    rew_scale_ang_vel_z: float = -5.0
    rew_scale_lin_vel_xy: float = -8.0
    rew_scale_base_drift: float = -8.0

    # === 동작 품질 ===
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -3.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5

    # === 형상 패널티 (foot_spread 계열 제거) ===
    rew_scale_shoulder_default: float = -8.0
    rew_scale_joint_default: float = -1.0
    rew_scale_foot_spread: float = 0.0        # v26:-3 → v27:0 (가라앉기 인센티브 제거)
    rew_scale_foot_side_span: float = 0.0     # v26:-3 → v27:0 (동일 이유)

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
