from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v33: 발 들기 강제 — 최소 clearance 4cm 페널티 + swing_contact -8.0 (진동 보행 차단)."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.5

    # --- 1순위: Gait ---
    rew_scale_gait: float = 10.0
    rew_scale_air_time: float = 5.0
    air_time_threshold: float = 0.15           # 0.12→0.15: 짧은 진동 air_time 무효화
    rew_scale_swing_contact: float = -8.0      # -2→-8: 진동 중 brief contact 강하게 차단
    rew_scale_foot_height: float = 8.0         # 6→8: 3cm 이상 들어야 보상 (최소치 강화)
    rew_scale_foot_clearance_penalty: float = -15.0  # 신규: swing 중 4cm 미달 직접 페널티
    rew_scale_diagonal_symmetry: float = -3.0
    rew_scale_air_time_var: float = 10.0
    rew_scale_diagonal_contact: float = 2.0

    # --- 2순위: 방향 ---
    rew_scale_heading: float = 12.0
    heading_sigma: float = 0.025
    rew_scale_pos_drift: float = -8.0
    rew_scale_yaw_tracking: float = 2.0
    rew_scale_ang_vel_z: float = -2.0
    rew_scale_lin_vel_xy: float = -2.0
    rew_scale_heading_linear: float = -3.0
    rew_scale_yaw_rate_error: float = -2.0

    # --- 3순위: 속도 (대폭 강화 — 제자리 trot local optima 탈출) ---
    rew_scale_lin_vel: float = 15.0          # 6→15: cmd=0.5 vel=0 손실 = -7.5/step (이동 시 +15)
    rew_scale_ang_vel: float = 0.5
    rew_scale_movement: float = 6.0          # 2→6: 앞으로 가야만 이득
    rew_scale_lin_vel_penalty: float = -3.0  # 속도 미달 직접 패널티 (vel<cmd 시 추가 손실)
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
    rew_scale_termination: float = -5.0

    # --- 자세 유지 ---
    target_foot_span: float = 0.10
    rew_scale_joint_default: float = -8.0
    rew_scale_foot_spread: float = -10.0
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
