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
    rew_scale_lin_vel: float = 6.0          # 8.0→6.0: slide-forward exploit 억제 (슬라이딩도 보상되는 문제)
    rew_scale_ang_vel: float = 1.0
    rew_scale_ang_vel_z: float = -3.0       # -4.0→-3.0: 소폭 완화 (heading과 동시 강화 금지)
    rew_scale_heading: float = 5.0          # 직진 유도 유지
    heading_sigma: float = 0.25
    rew_scale_movement: float = 2.0         # 0.0→2.0: 실제 이동 직접 보상, gradient 탈출 보조
    rew_scale_lin_vel_penalty: float = 0.0
    rew_scale_lin_vel_xy: float = -0.3      # lateral drift 약하게 억제

    # --- Gait 유도 ---
    rew_scale_gait: float = 1.5             # 2.5→1.5: gait 21pt→12pt, 제자리 최적해 가치 붕괴
    rew_scale_air_time: float = 8.0         # 5.0→8.0: threshold 달성 시 보상 대폭 강화
    air_time_threshold: float = 0.12        # 0.10→0.12: 1.5Hz 스윙의 36% 이상 요구
    rew_scale_swing_contact: float = -1.5
    rew_scale_foot_height: float = 6.0      # 12.0→6.0: instantaneous 보상 약화, air_time으로 gradient 유도

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.3
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.15    # -0.05→-0.15: 고주파 oscillation 억제, 서보 수명 보호
    rew_scale_dof_acc: float = -1e-6
    rew_scale_termination: float = -5.0

    # --- 자세 유지 ---
    rew_scale_joint_default: float = -3.0
    rew_scale_foot_spread: float = -6.0
    rew_scale_foot_slip: float = -1.5       # -0.5→-1.5: 느린 슬라이딩도 강하게 억제 (slip²이라 scale 필요)
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
