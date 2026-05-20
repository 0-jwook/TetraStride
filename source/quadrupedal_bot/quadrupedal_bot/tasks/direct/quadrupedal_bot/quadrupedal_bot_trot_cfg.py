from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage3b: Stage3(0.21m/s) 전이 → 최소 cmd 0.15로 속도 압박.
    정체 원인: cmd 0~0.5에서 낮은 cmd만 선택. 최소값 강제로 0.25~0.35m/s 유도.
    """

    episode_length_s: float = 15.0
    target_body_height: float = 0.17

    action_scale: float = 0.35

    # --- 최소 전진 속도 강제 (정체 탈출) ---
    cmd_lin_vel_x_range: tuple = (0.15, 0.4)  # 최소 0.15 m/s 강제
    cmd_lin_vel_y_range: tuple = (-0.15, 0.15)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)
    zero_command_prob: float = 0.1            # 10%만 제자리

    gait_reward_always_on: bool = True

    gait_freq_hz: float = 1.4              # 1.2→1.4Hz: 빠른 걸음 유도

    # --- 속도 보상 (gait=15 대비 50% 수준) ---
    rew_scale_lin_vel: float = 7.0      # 6.0→7.0
    rew_scale_ang_vel: float = 0.8
    rew_scale_movement: float = 2.5
    rew_scale_lin_vel_penalty: float = 0.0
    rew_scale_heading: float = 0.0
    rew_scale_pos_drift: float = 0.0
    rew_scale_yaw_tracking: float = 0.0
    rew_scale_ang_vel_z: float = 0.0
    rew_scale_lin_vel_xy: float = 0.0
    rew_scale_heading_linear: float = 0.0
    rew_scale_yaw_rate_error: float = 0.0

    # --- Gait 강제 (trot 순서) ---
    rew_scale_gait: float = 15.0
    rew_scale_air_time: float = 15.0
    air_time_threshold: float = 0.05
    rew_scale_swing_contact: float = -10.0
    rew_scale_diagonal_symmetry: float = -3.0
    rew_scale_air_time_var: float = 5.0
    rew_scale_diagonal_contact: float = 3.0

    # --- 발 들기 보상 (역관절 활용) ---
    # URDF: leg_link=107.5mm, calf_link=130mm
    # hip=0.0, knee=-1.4 → foot_tip_z ≈ 4cm
    # hip=-0.2, knee=-1.6 → foot_tip_z ≈ 7cm
    rew_scale_foot_height: float = 40.0    # 발끝 높이 직접 보상 (clamp 0~10cm)
    rew_scale_foot_clearance_penalty: float = 0.0

    # --- Gaussian 타겟: hip 앞으로 + knee 더 구부리기 ---
    target_leg_angle_swing_gauss: float = 0.0   # hip 중립 (기본 0.83에서 완전 앞으로)
    sigma_leg_swing: float = 0.20
    rew_scale_hip_swing_gauss: float = 4.0
    target_knee_angle_swing_gauss: float = -1.4  # knee 더 구부려 발 올리기
    sigma_knee_swing: float = 0.25
    rew_scale_knee_swing_gauss: float = 4.0

    # --- 기존 clamp 패널티 비활성화 ---
    rew_scale_knee_bend_swing: float = 0.0
    rew_scale_leg_flex_swing: float = 0.0
    rew_scale_knee_swing: float = 0.0
    rew_scale_knee_swing_penalty: float = 0.0
    rew_scale_swing_max_leg: float = 0.0
    rew_scale_leg_angle_min: float = 0.0
    min_knee_angle_swing: float = -1.2
    rew_scale_swing_min_knee: float = 10.0   # 최소 구부리기 보조 유지

    # --- 자세 안정 (넘어지지 않기) ---
    rew_scale_alive: float = 0.5
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 3.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -1.0
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_termination: float = -10.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_default: float = -3.0   # 어깨 abduction 방지
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.05
    rew_scale_action_jerk: float = -0.02
    rew_scale_dof_acc: float = -1e-6
    rew_scale_contact_forces: float = -0.05
    max_foot_contact_force: float = 30.0

    # --- 발 퍼짐/슬립 방지 ---
    target_foot_span: float = 0.10
    rew_scale_foot_spread: float = -20.0
    rew_scale_foot_slip: float = -1.0

    # --- 무릎 보행 방지 ---
    non_foot_contact_threshold: float = 4.0
    rew_scale_non_foot_contact: float = -5.0
    rew_scale_stumble: float = -2.0
    rew_scale_foot_stance: float = 1.0
    rew_scale_knee_angle: float = -3.0
    rew_scale_knee_height_stance: float = -8.0

    # --- 기타 ---
    rew_scale_energy: float = 0.0
    rew_scale_stance_vel: float = 0.0
    min_leg_angle: float = 0.3
    push_interval_s: float = 8.0   # Stage2: 약한 외란으로 견고성 훈련
    max_push_vel: float = 0.3
