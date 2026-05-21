from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Standing v2: 전체 관절 기본자세 패널티 + bouncing 수정 재학습.
    수정 사항:
      - rew_joint_default: shoulder만 → 전체 12관절 (env.py 수정)
      - rew_scale_foot_height: 40→20 (bouncing 방지)
      - rew_scale_lin_vel_z: -2→-8 (점프/수직 이동 억제)
      - rew_scale_contact_forces: 0→-0.3 (강한 착지 충격 패널티)
      - 처음부터 학습 (leg drift 습관 초기화)
    """

    episode_length_s: float = 15.0
    target_body_height: float = 0.17

    action_scale: float = 0.35

    # --- 제자리 서기 학습 ---
    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)
    zero_command_prob: float = 0.0

    gait_reward_always_on: bool = True

    gait_freq_hz: float = 1.2              # 안정적인 trot 주기

    # --- 속도 보상 비활성화 (제자리) ---
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

    # --- Gait 강제 (trot 순서) ---
    rew_scale_gait: float = 15.0
    rew_scale_air_time: float = 15.0
    air_time_threshold: float = 0.05
    rew_scale_swing_contact: float = -10.0
    rew_scale_diagonal_symmetry: float = -3.0
    rew_scale_air_time_var: float = 5.0
    rew_scale_diagonal_contact: float = 3.0

    # --- 발 들기 보상 (역관절) — bouncing 방지로 절반 축소 ---
    rew_scale_foot_height: float = 20.0    # 40→20: 수직 충격력 과다 방지
    rew_scale_foot_clearance_penalty: float = 0.0

    # --- Gaussian 타겟: hip 앞으로 + knee 구부리기 ---
    target_leg_angle_swing_gauss: float = 0.0
    sigma_leg_swing: float = 0.20
    rew_scale_hip_swing_gauss: float = 4.0
    target_knee_angle_swing_gauss: float = -1.4
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
    rew_scale_swing_min_knee: float = 10.0

    # --- 자세 안정 ---
    rew_scale_alive: float = 0.5
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 5.0         # 3→5: 직립 보상 강화
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -2.0     # -1→-2: 롤/피치 억제 강화
    rew_scale_lin_vel_z: float = -8.0      # -2→-8: 점프/바운싱 강력 억제
    rew_scale_termination: float = -10.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_default: float = -5.0  # -3→-5: 전체 12관절 패널티 (env.py 수정됨)
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.05
    rew_scale_action_jerk: float = -0.02
    rew_scale_dof_acc: float = -1e-6
    rew_scale_contact_forces: float = -0.3  # 0→-0.3: 강한 착지 충격 패널티 추가
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
    push_interval_s: float = 0.0   # 서있기 학습: 외란 없음
    max_push_vel: float = 0.0
