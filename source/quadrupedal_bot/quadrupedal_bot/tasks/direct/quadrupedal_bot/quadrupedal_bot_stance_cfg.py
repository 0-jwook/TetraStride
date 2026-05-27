from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v26 — target_body_height 자연 평형 높이로 조정 + spinning 차단.

    v25 문제:
      1. target_body_height=0.163m이 실제 default 평형(0.177m)보다 낮음
         → 평형에서 패널티 없고, 가라앉아도 패널티 작음 (그라디언트 약함)
      2. ang_vel_z=-0.5 너무 약함 → 원형 회전 흑자 (spinning 해킹)
      3. base_drift=0 → 위치 이탈 패널티 없음

    v26 수정:
      1. target_body_height: 0.163→0.177 (자연 평형 = 목표)
         → 조금만 가라앉아도 즉시 패널티, 그라디언트 3배 강해짐
         → 자연 평형에서 능동 제어 없이도 패널티 0
      2. ang_vel_z: -0.5→-5.0 (spinning 10배 억제)
      3. base_drift: 0→-8.0 (위치 이탈 패널티 복구)
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

    # === 높이 목표: 자연 평형 높이로 조정 ===
    target_body_height: float = 0.177      # v25:0.163 → v26:0.177 (default 관절 평형)
    rew_scale_body_height: float = -80.0   # 0.177→0.155 가라앉으면 -1.76/step (v25의 3배)

    # === stand_still 비활성 유지 (능동 제어 허용) ===
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
    rew_scale_ang_vel_z: float = -5.0     # v25:-0.5 → v26:-5.0 (spinning 차단)
    rew_scale_lin_vel_xy: float = -8.0
    rew_scale_base_drift: float = -8.0    # v25:0 → v26:-8.0 (위치 이탈 복구)

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
