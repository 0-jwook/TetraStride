from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v30 — kp=80 + Gaussian 높이 보상으로 다리 접힘 근본 해결.

    v29 문제:
      1. kp=30: 모터가 중력을 버티지 못해 관절이 스르륵 접힘 (물리적 근본 원인)
      2. body_height 보상 없음: 낮아져도 패널티 없음 → 능동적으로 접기 학습
      3. 시각화에서 명확히 확인: 자세 유지 불가, 0.155m 조기종료 반복

    v30 수정:
      1. kp: 30→80 (spot_micro_cfg.py) — 중력 처짐 0.018rad→0.007rad으로 감소
      2. rew_scale_body_height: 0.0→+8.0 (Gaussian, sigma=0.05m)
         → 0.177m 기준: 0.177m=8.0, 0.163m=6.05, 0.155m=5.15 — 높이 유지 명확한 기울기
      3. target_body_height: 0.163→0.177 (kp=80이면 도달 가능)
      4. alive: 10.0→8.0, foot_contact: 4.0→6.0 (alive 독주 완화, 삼등분 균형)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.155

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 (삼등분 균형) ===
    rew_scale_alive: float = 8.0           # v29:10.0 → v30:8.0
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 6.0    # v29:4.0 → v30:6.0 (이진 유지)

    # === 높이 보상 복활 (Gaussian) ===
    target_body_height: float = 0.177      # v29:0.163 → v30:0.177 (kp=80로 도달 가능)
    rew_scale_body_height: float = 8.0     # v29:0.0 → v30:+8.0 (Gaussian, scale>0)

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
