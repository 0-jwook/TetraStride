from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v36: 발 들기 목표각 강화 — knee -1.4rad, leg 1.25rad, joint_default 완화 (v35 전이)."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.2              # 1.5→1.2Hz: 스텝당 더 많은 시간 → 충분히 들 수 있음

    # --- 1순위: 관절각 기반 발 들기 (v36: 목표각 강화) ---
    swing_knee_target: float = -1.4           # -1.1→-1.4rad: 무릎 더 깊이 굽혀서 발끝 상승
    swing_leg_target: float = 1.25            # 1.05→1.25rad: 허벅지 더 높이 들어서 무릎 자체를 올림
    rew_scale_knee_bend_swing: float = 80.0   # 60→80: 더 강하게 강제
    rew_scale_leg_flex_swing: float = 60.0    # 30→60: knee와 동등하게 강화
    rew_scale_knee_swing: float = 0.0
    rew_scale_knee_swing_penalty: float = 0.0

    # --- 2순위: Gait ---
    rew_scale_gait: float = 10.0
    rew_scale_air_time: float = 5.0
    air_time_threshold: float = 0.18          # 0.15→0.18s: 짧은 shuffle 무효화
    rew_scale_swing_contact: float = -8.0
    rew_scale_foot_height: float = 8.0
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

    # --- 4순위: 속도 (감소 — joint 보상이 압도하도록) ---
    rew_scale_lin_vel: float = 8.0            # 15→8: joint 보상(-60) 압도 허용
    rew_scale_ang_vel: float = 0.5
    rew_scale_movement: float = 3.0           # 6→3
    rew_scale_lin_vel_penalty: float = -2.0   # -3→-2: 과도한 속도 강요 완화
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
    rew_scale_joint_default: float = -3.0     # -8→-3: 발 들기 시 관절 이탈 저항 완화
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
