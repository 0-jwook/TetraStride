from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v20: 셔플링 재발 수정 + 직선 보행 vy 패널티 강화."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35
    # action_smoothing = 0.8 (전이학습 중 변경 금지)

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)
    cmd_ang_vel_z_range: tuple = (-0.3, 0.3)
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.5

    # --- 속도 추적 ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 6.0
    rew_scale_ang_vel: float = 1.0
    rew_scale_ang_vel_z: float = -2.0
    rew_scale_heading: float = 10.0
    heading_sigma: float = 0.05
    rew_scale_movement: float = 2.0
    rew_scale_lin_vel_penalty: float = 0.0
    rew_scale_lin_vel_xy: float = -2.0   # -0.5→-2.0: vy² 측면 이동 직접 억제 (직선 보행 강제)

    # --- Gait 유도 ---
    rew_scale_gait: float = 1.5          # 1.0→1.5: 셔플링 방지 위해 소폭 복원 (abrupt 없이)
    rew_scale_air_time: float = 6.0      # 5.0→6.0: 발 들기 인센티브 복원
    air_time_threshold: float = 0.10     # 0.08→0.10s: 실제 발 들기 요구 (셔플링 방지)
    rew_scale_swing_contact: float = -0.8   # -0.5→-0.8: swing 중 접촉 패널티 강화 (셔플링 차단)
    rew_scale_foot_height: float = 4.0   # 2.0→4.0: 발 높이 보상 복원 (들어올리되 힘차지 않게)

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.3
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.10
    rew_scale_action_jerk: float = -0.02
    rew_scale_dof_acc: float = -1e-6
    rew_scale_termination: float = -5.0

    # --- 자세 유지 ---
    target_foot_span: float = 0.10
    rew_scale_joint_default: float = -8.0
    rew_scale_foot_spread: float = -10.0
    rew_scale_foot_slip: float = -1.5
    rew_scale_air_time_var: float = 5.0

    # --- 무릎 보행 방지 ---
    non_foot_contact_threshold: float = 4.0
    rew_scale_non_foot_contact: float = -5.0
    rew_scale_stumble: float = -2.0
    rew_scale_foot_stance: float = 2.0
    rew_scale_knee_angle: float = -5.0
    rew_scale_knee_height_stance: float = -10.0

    # --- 보행 품질 ---
    rew_scale_diagonal_symmetry: float = -0.15
    rew_scale_energy: float = 0.0

    # --- 도메인 랜덤화 ---
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3
