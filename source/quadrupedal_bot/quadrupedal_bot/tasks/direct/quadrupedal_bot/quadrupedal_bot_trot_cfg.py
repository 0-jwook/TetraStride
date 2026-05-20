from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v51: Gaussian hip/knee 타겟 보상 (clamp 패널티 완전 대체, v45 전이)."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.2

    # --- 관절각 강제: 제거 ---
    rew_scale_knee_bend_swing: float = 0.0
    rew_scale_leg_flex_swing: float = 0.0
    rew_scale_knee_swing: float = 0.0
    rew_scale_knee_swing_penalty: float = 0.0

    # --- Gait ---
    rew_scale_gait: float = 10.0
    rew_scale_air_time: float = 10.0          # v45 유지
    air_time_threshold: float = 0.05          # 0.18→0.05s: 짧은 들기에도 그라디언트
    rew_scale_swing_contact: float = -8.0
    rew_scale_foot_height: float = 25.0       # v45 유지 (60은 역효과 — 뒤로 접기 최적점 유도)
    rew_scale_foot_clearance_penalty: float = 0.0
    rew_scale_diagonal_symmetry: float = -3.0
    rew_scale_air_time_var: float = 10.0
    rew_scale_diagonal_contact: float = 2.0

    # --- 3순위: 방향 ---
    rew_scale_heading: float = 12.0
    heading_sigma: float = 0.025
    rew_scale_pos_drift: float = -8.0
    rew_scale_yaw_tracking: float = 2.0
    rew_scale_ang_vel_z: float = -2.0
    rew_scale_lin_vel_xy: float = -2.0
    rew_scale_heading_linear: float = -3.0
    rew_scale_yaw_rate_error: float = -2.0

    # --- 속도 ---
    rew_scale_lin_vel: float = 12.0           # 8→12: 정지 로컬옵티마 탈출
    rew_scale_ang_vel: float = 0.5
    rew_scale_movement: float = 5.0           # 3→5: 이동 보상 강화
    rew_scale_lin_vel_penalty: float = -2.0
    rew_scale_alive: float = 0.5

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.5
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.10
    rew_scale_action_jerk: float = -0.03
    rew_scale_dof_acc: float = -2e-6
    rew_scale_contact_forces: float = -0.05
    max_foot_contact_force: float = 30.0
    rew_scale_termination: float = -10.0

    # --- 자세 유지 ---
    target_foot_span: float = 0.10
    rew_scale_joint_default: float = -5.0      # 어깨 abduction 방지
    min_leg_angle: float = 0.3                # backward extreme만 차단 (scale=0이라 비활성)
    rew_scale_leg_angle_min: float = 0.0      # 비활성화
    min_knee_angle_swing: float = -1.2        # v45 유지 (clamp 방식이지만 무릎 최솟값 보조)
    rew_scale_swing_min_knee: float = 20.0
    max_leg_angle_swing: float = 1.0          # 비활성화 (Gaussian으로 대체)
    rew_scale_swing_max_leg: float = 0.0      # 비활성화 — clamp 패러독스 방지

    # --- v51: Gaussian 타겟 보상 ---
    target_leg_angle_swing_gauss: float = 0.2  # 목표: v45(0.39)에서 점진 하향
    sigma_leg_swing: float = 0.15
    rew_scale_hip_swing_gauss: float = 2.0     # 보상 규모: velocity(12)의 1/6 — 완만한 유도
    target_knee_angle_swing_gauss: float = -1.0
    sigma_knee_swing: float = 0.2
    rew_scale_knee_swing_gauss: float = 1.0
    rew_scale_foot_spread: float = -25.0      # 도마뱀 자세 방지
    rew_scale_foot_slip: float = -1.5

    # --- 무릎 보행 방지 ---
    non_foot_contact_threshold: float = 4.0
    rew_scale_non_foot_contact: float = -5.0
    rew_scale_stumble: float = -2.0
    rew_scale_foot_stance: float = 2.0
    rew_scale_knee_angle: float = -5.0
    rew_scale_knee_height_stance: float = -10.0

    # --- 기타 ---
    rew_scale_energy: float = 0.0
    rew_scale_stance_vel: float = 0.0
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3
