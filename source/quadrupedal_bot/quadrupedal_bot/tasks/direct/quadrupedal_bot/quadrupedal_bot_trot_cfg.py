from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v23b: 진동 억제 균형 조정 — action_rate/jerk/dof_acc 완화, ang_vel_xy 5x 유지."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35
    # action_smoothing = 0.8 (전이학습 중 변경 금지)

    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)   # v22: 직진 전용 커리큘럼 (lateral 명령 제거)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)   # v23: yaw 명령 완전 제거 → 항상 직진
    zero_command_prob: float = 0.1

    gait_freq_hz: float = 1.5

    # --- 속도 추적 ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 6.0
    rew_scale_ang_vel: float = 1.0
    rew_scale_ang_vel_z: float = -3.0       # v23: 3배 강화 — yaw rate 오차 패널티
    rew_scale_yaw_tracking: float = 2.0     # legged_gym: exp(-yaw_err²/0.25) × 2.0 — yaw 추적 동기 부여
    rew_scale_heading: float = 10.0
    heading_sigma: float = 0.025       # v23: 4배 타이트 — 5° 오차에서 11.5% 보상 감소 (gradient 복원)
    rew_scale_movement: float = 2.0
    rew_scale_lin_vel_penalty: float = 0.0
    rew_scale_lin_vel_xy: float = -2.0      # vy² 측면 속도 패널티 유지

    # --- 직선 보행 직접 수정 ---
    rew_scale_pos_drift: float = -6.0       # -3.0→-6.0: cmd_y=0 고정으로 100% env 적용, 패널티 2배

    # --- Gait 유도 (v20 균형점 유지) ---
    rew_scale_gait: float = 1.5
    rew_scale_air_time: float = 6.0
    air_time_threshold: float = 0.12   # 0.10→0.12s: swing 36%, 셔플링 방지
    rew_scale_swing_contact: float = -0.8
    rew_scale_foot_height: float = 4.0

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -1.5      # v23: 5배 — 몸통 롤/피치 진동 직접 억제
    rew_scale_lin_vel_z: float = -4.0       # v23: 2배 — 수직 바운스 억제

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.15     # v23b: 1.5배 (0.10→0.15) — 과도한 억제 완화
    rew_scale_action_jerk: float = -0.04    # v23b: 2배 (0.02→0.04) — 과도한 억제 완화
    rew_scale_dof_acc: float = -5e-6        # v23b: 5배 (1e-6→5e-6) — 정지 local optima 방지
    rew_scale_contact_forces: float = -0.05 # v23: 발 착지 충격 패널티 활성화
    max_foot_contact_force: float = 30.0    # v23: 2.5kg 로봇 기준 임계값 (50→30N)
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
    rew_scale_diagonal_symmetry: float = -0.30
    rew_scale_energy: float = 0.0

    # --- v23: 직선 보행 선형 패널티 ---
    rew_scale_heading_linear: float = -3.0   # 선형 heading 오차 패널티 (exp 포화 1~5° 구간 보완)
    rew_scale_yaw_rate_error: float = -2.0   # 선형 yaw rate 오차 패널티

    # --- 도메인 랜덤화 ---
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3
