from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v29 — termination 높이 상향으로 높은 자세 강제.

    v28 문제:
      1. termination_height=0.145m: 낮은 종료 임계값 → 로봇이 0.151m에 안주
      2. stand_still=-2.0: 관절을 default(낮은 자세)로 끌어당겨 높이 하락 유인
         → 후반부 foot_contact 3.66→2.54 하락, body_height 0.157→0.151m 하락
      3. 0.151m는 실제 로봇 다리 길이 대비 너무 낮은 자세

    v29 수정:
      1. termination_height: 0.145→0.155 (현재 0.151m에서 4mm 위)
         → 0.155m 이하로 내려가면 즉시 종료 → alive=10.0으로 높이 유지 강제
      2. stand_still: -2.0→0.0 (제거 — 낮은 자세 유인 차단)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.155      # v28:0.145 → v29:0.155 (높이 강제)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 10.0
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 4.0    # 이진: 4발 동시 접지만 보상

    # === 높이 목표 제거 유지 (종료로만 제어) ===
    target_body_height: float = 0.163      # 참조용 (보상 비활성)
    rew_scale_body_height: float = 0.0

    # === stand_still 제거 (낮은 자세 유인 차단) ===
    rew_scale_stand_still: float = 0.0     # v28:-2.0 → v29:0.0

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
