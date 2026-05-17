from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v27: Isaac Lab GaitReward 포팅 — sync×async 곱셈 구조로 trot 타이밍 강제."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35
    # action_smoothing = 0.8 (전이학습 중 변경 금지)

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)   # 직진 전용 커리큘럼
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)   # yaw 명령 제거 → 항상 직진
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.5

    # --- 1순위: Gait (Isaac Lab GaitReward 방식) ---
    rew_scale_gait: float = 10.0             # 5.0→10.0: sync×async 곱셈 구조 (max=1.0/step × scale)
    rew_scale_air_time: float = 5.0          # 10.0→5.0: GaitReward가 타이밍 담당, air_time은 보조
    air_time_threshold: float = 0.12
    rew_scale_swing_contact: float = -0.8
    rew_scale_foot_height: float = 6.0
    rew_scale_diagonal_symmetry: float = -3.0  # -5.0→-3.0: GaitReward와 역할 분담
    rew_scale_air_time_var: float = 10.0     # 15.0→10.0: 소폭 완화
    rew_scale_diagonal_contact: float = 2.0  # 4.0→2.0: GaitReward 보조 역할로 축소

    # --- 2순위: 방향 (직선 보행) ---
    rew_scale_heading: float = 12.0           # 10.0→12.0: 방향 2순위로 격상
    heading_sigma: float = 0.025             # 타이트 gradient — 5° 오차에서 11.5% 손실
    rew_scale_pos_drift: float = -8.0        # -6.0→-8.0: 직선 보행 강화
    rew_scale_yaw_tracking: float = 2.0
    rew_scale_ang_vel_z: float = -2.0        # yaw rate 오차 패널티
    rew_scale_lin_vel_xy: float = -2.0       # 측면 속도 패널티
    rew_scale_heading_linear: float = -3.0   # 선형 heading 오차 패널티 (exp 포화 보완)
    rew_scale_yaw_rate_error: float = -2.0   # 선형 yaw rate 오차 패널티

    # --- 3순위: 속도 (걸음걸이와 동시에 충족해야 최대 보상) ---
    rew_scale_lin_vel: float = 6.0           # 원복: 속도도 강하게 — 제자리 trot 방지
    rew_scale_ang_vel: float = 0.5
    rew_scale_movement: float = 2.0          # 원복: 앞으로 가야만 이득
    rew_scale_lin_vel_penalty: float = 0.0
    rew_scale_alive: float = 0.5

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.5       # -1.5→-0.5: 과도 억제 완화 (서기 local optima 방지)
    rew_scale_lin_vel_z: float = -2.0        # -4.0→-2.0: 동일

    # --- 관절/토크 제약 (균형 조정) ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.10     # 원래 값 유지 (gait 보상이 진동 억제 담당)
    rew_scale_action_jerk: float = -0.03     # 소폭 강화
    rew_scale_dof_acc: float = -2e-6         # 2배 (1e-6→2e-6), 정지 local optima 방지
    rew_scale_contact_forces: float = -0.05  # 발 착지 충격 패널티
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
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3
