from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Standing v3: soft guidance + 자유도 확보.
    철학: 강제 없음 → IMU/에너지/접지 기반 간접 유도.
    - joint_default: 아주 약한 prior (-0.10)
    - orientation: sigma 0.04→0.20 (20°까지 탐색 허용)
    - tilt termination: 180°→45° 완화
    - 4발 접지 보상 추가
    - gait/air_time/foot_height 전부 제거
    """

    episode_length_s: float = 20.0
    target_body_height: float = 0.17
    termination_height: float = 0.10      # 0.15→0.10: 탐색 자유도 확보

    action_scale: float = 0.35

    # --- 제자리 서있기 ---
    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)
    zero_command_prob: float = 0.0

    gait_reward_always_on: bool = False    # gait 완전 비활성화

    # --- 핵심 안정화 보상 ---
    rew_scale_alive: float = 1.0           # 생존이 핵심 신호
    rew_scale_upright: float = 5.0         # IMU orientation 핵심 보상
    orientation_sigma: float = 0.20        # 0.04→0.20: 20°까지 탐색 허용
    rew_scale_body_height: float = -1.0    # soft 패널티 (clamp min=0 단방향)
    rew_scale_ang_vel_xy: float = -1.0     # 롤/피치 각속도 억제
    rew_scale_lin_vel_z: float = -2.0      # 수직 이동 억제 (corrective motion 허용)
    rew_scale_termination: float = -5.0    # 완화 (탐색 허용)

    # --- 간접 역관절 유도 ---
    rew_scale_torque: float = -1e-4        # 에너지 최소화 → 역관절이 물리적으로 효율적
    rew_scale_foot_contact: float = 3.0    # 4발 동시 접지 보상 (들기 억제)
    rew_scale_foot_spread: float = -6.0    # 도마뱀 자세 방지 (약화)
    target_foot_span: float = 0.10

    # --- 약한 prior만 유지 ---
    rew_scale_joint_default: float = -0.10  # 아주 약한 prior (강제 아님)

    # --- 부드러운 행동 ---
    rew_scale_action_rate: float = -0.05
    rew_scale_action_jerk: float = -0.02
    rew_scale_dof_acc: float = -1e-6
    rew_scale_joint_vel: float = -1e-4

    # --- reward hacking 방지 ---
    rew_scale_non_foot_contact: float = -3.0   # 무릎 보행 방지
    non_foot_contact_threshold: float = 4.0
    rew_scale_contact_forces: float = -0.3     # 강한 착지 충격 방지
    max_foot_contact_force: float = 30.0
    rew_scale_foot_slip: float = -1.0
    rew_scale_stumble: float = -2.0

    # --- 완전 제거 (자유도 확보) ---
    rew_scale_gravity: float = 0.0             # upright와 중복
    rew_scale_gait: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_foot_height: float = 0.0
    rew_scale_swing_contact: float = 0.0
    rew_scale_diagonal_symmetry: float = 0.0
    rew_scale_air_time_var: float = 0.0
    rew_scale_diagonal_contact: float = 0.0
    rew_scale_knee_angle: float = 0.0
    rew_scale_knee_height_stance: float = 0.0
    rew_scale_knee_bend_swing: float = 0.0
    rew_scale_leg_flex_swing: float = 0.0
    rew_scale_knee_swing: float = 0.0
    rew_scale_knee_swing_penalty: float = 0.0
    rew_scale_swing_max_leg: float = 0.0
    rew_scale_leg_angle_min: float = 0.0
    rew_scale_swing_min_knee: float = 0.0
    rew_scale_hip_swing_gauss: float = 0.0
    rew_scale_knee_swing_gauss: float = 0.0
    rew_scale_foot_stance: float = 0.0
    rew_scale_energy: float = 0.0
    rew_scale_stance_vel: float = 0.0
    rew_scale_foot_clearance_penalty: float = 0.0
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_lin_vel_penalty: float = 0.0
    rew_scale_heading: float = 0.0
    rew_scale_pos_drift: float = 0.0
    rew_scale_yaw_tracking: float = 0.0
    rew_scale_ang_vel_z: float = 0.0
    rew_scale_lin_vel_xy: float = 0.0
    rew_scale_heading_linear: float = 0.0
    rew_scale_yaw_rate_error: float = 0.0

    # --- 기타 ---
    push_interval_s: float = 0.0
    max_push_vel: float = 0.0
