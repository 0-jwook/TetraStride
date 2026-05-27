from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v28 — 높이 목표 제거 + alive 압도적 강화.

    v27 문제:
      1. target_body_height=0.177m: DCMotor 자연 평형(≈0.163m) 위를 요구
         → 정책이 능동 높이 유지를 학습 못 해 -1.2/step 상시 패널티
      2. termination_height=0.155m: 자연 평형 0.163m에서 여유 0.8cm → 쉽게 조기종료
      3. alive=2.0: 전체 보상의 19% → 생존보다 발접지/자세 보상이 더 큰 비중

    v28 수정:
      1. body_height reward 제거 (자연 평형 0.163m 수용, 높이는 종료로만 제어)
      2. termination_height: 0.155→0.145 (자연 평형에서 1.8cm 여유)
      3. alive: 2.0→10.0 (전체 양의 보상의 50% → 서있기 = 최우선)
      4. stand_still: 0→-2.0 (재활성, 높이 유지 충돌 없으므로 안전)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.145      # v27:0.155 → v28:0.145

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 10.0          # v27:2.0 → v28:10.0 (서있기 50% 비중)
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 4.0    # 이진: 4발 동시 접지만 보상

    # === 높이 목표 제거 (자연 평형 수용) ===
    target_body_height: float = 0.163      # 참조용 (보상 비활성)
    rew_scale_body_height: float = 0.0     # v27:-80.0 → v28:0.0 (높이 목표 제거)

    # === stand_still 재활성 ===
    rew_scale_stand_still: float = -2.0    # v27:0.0 → v28:-2.0 (자연 자세 유지)

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
