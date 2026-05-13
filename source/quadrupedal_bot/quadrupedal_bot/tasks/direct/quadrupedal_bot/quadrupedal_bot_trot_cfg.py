from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: 2026-05-12_09-28-38 run 파라미터 복원 (사용자 확인 최적 버전)."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)
    cmd_ang_vel_z_range: tuple = (-0.3, 0.3)
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.5

    # --- 속도 추적 ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 3.0
    rew_scale_ang_vel: float = 1.0
    rew_scale_ang_vel_z: float = -1.0
    rew_scale_heading: float = 3.0
    heading_sigma: float = 0.25
    rew_scale_movement: float = 0.0
    rew_scale_lin_vel_penalty: float = 0.0

    # --- Gait 유도 ---
    rew_scale_gait: float = 2.5
    rew_scale_air_time: float = 5.0
    air_time_threshold: float = 0.04
    rew_scale_swing_contact: float = -1.5
    rew_scale_foot_height: float = 5.0

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.3
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.05
    rew_scale_dof_acc: float = -1e-6
    rew_scale_termination: float = -5.0

    # --- 자세 유지 ---
    rew_scale_joint_default: float = -3.0
    rew_scale_foot_spread: float = -6.0
    rew_scale_foot_slip: float = -0.05
    rew_scale_air_time_var: float = 3.0

    # --- 무릎 보행 방지 ---
    non_foot_contact_threshold: float = 4.0
    rew_scale_non_foot_contact: float = -5.0
    rew_scale_stumble: float = -2.0
    rew_scale_foot_stance: float = 2.0
    rew_scale_knee_angle: float = -5.0
    rew_scale_knee_height_stance: float = -10.0

    # --- 보행 품질 ---
    rew_scale_action_jerk: float = 0.0
    rew_scale_diagonal_symmetry: float = 0.0
    rew_scale_energy: float = 0.0

    # --- 도메인 랜덤화 ---
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3
