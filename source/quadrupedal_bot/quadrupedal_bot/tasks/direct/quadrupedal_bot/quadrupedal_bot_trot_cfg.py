from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v11: URDF 물리 기반 수정 — 셔플링 제거 + heading drift 수정."""

    episode_length_s: float = 20.0

    # 실제 서기 높이 ~0.24m 기준, 0.17은 7cm 낮아 웅크리기 강제 → 완화
    target_body_height: float = 0.20

    # 상퇴 107.5mm 기준: 0.35 rad → 37mm, 0.50 rad → 54mm 발끝 이동
    action_scale: float = 0.50

    cmd_lin_vel_x_range: tuple = (0.2, 0.6)
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)
    cmd_ang_vel_z_range: tuple = (-0.1, 0.1)   # ±0.3→±0.1: 회전 커맨드 자체 제한
    zero_command_prob: float = 0.25

    gait_freq_hz: float = 1.5

    # --- 속도 추적 ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 5.0              # 6.0→5.0: 미끄러짐으로 보상 받는 경우 줄임
    rew_scale_ang_vel: float = 1.0
    rew_scale_ang_vel_z: float = -6.0           # -3.0→-6.0: 짧은 어깨(±31°) 비대칭 회전 강하게 억제
    rew_scale_heading: float = 8.0              # 5.0→8.0: lin_vel과 균형 + heading drift 핵심
    heading_sigma: float = 0.15                 # 0.25→0.15: sphere foot 슬립 고려, 허용 편차 좁힘
    rew_scale_movement: float = 3.0             # 2.0→3.0: 발 내딛기 직접 보상
    rew_scale_lin_vel_penalty: float = 0.0

    # --- Gait 유도 ---
    rew_scale_gait: float = 3.0                 # 2.5→3.0: trot 패턴 강화
    rew_scale_air_time: float = 3.5             # 1.5→3.5: sphere foot r=20mm, 들어야 접지 안정
    air_time_threshold: float = 0.10            # 0.08→0.10: 하퇴 130mm 기준 최소 들기
    rew_scale_swing_contact: float = -1.0
    rew_scale_foot_height: float = 5.0          # 3.0→5.0: 짧은 다리, 명확한 리프팅 유도

    # --- 자세 안정 (target_body_height 올렸으므로 완화) ---
    rew_scale_body_height: float = -4.0         # -8.0→-4.0: 웅크리기 강제 제거
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -3.0             # -5.0→-3.0: 전이 초기 완화
    rew_scale_ang_vel_xy: float = -0.3
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -2.5e-5
    rew_scale_action_rate: float = -0.05
    rew_scale_dof_acc: float = -1e-6
    rew_scale_termination: float = -5.0

    # --- 자세 유지 (셔플링 원인 완화) ---
    rew_scale_joint_default: float = -0.5       # -3.0→-0.5: 스윙 방해 제거 (셔플링 핵심 원인)
    rew_scale_foot_spread: float = -2.0         # -6.0→-2.0: 어깨 ±31° 제한 고려
    rew_scale_foot_slip: float = -0.1           # -0.05→-0.1: sphere foot 슬립 강화
    rew_scale_air_time_var: float = 3.0

    # --- 무릎 보행 방지 ---
    non_foot_contact_threshold: float = 4.0
    rew_scale_non_foot_contact: float = -5.0
    rew_scale_stumble: float = -2.0
    rew_scale_foot_stance: float = 2.0
    rew_scale_knee_angle: float = -5.0
    rew_scale_knee_height_stance: float = -10.0

    # --- 보행 품질 ---
    rew_scale_diagonal_symmetry: float = 2.0    # 0.0→2.0: FL-RR/FR-RL 대각 동기화, 비대칭 회전 보정
    rew_scale_action_jerk: float = -0.001
    rew_scale_energy: float = -1e-4

    # --- 도메인 랜덤화 (초기 전이 완화) ---
    push_interval_s: float = 10.0               # 8.0→10.0: 초기 전이 시 충격 완화
    max_push_vel: float = 0.2                   # 0.3→0.2: sphere foot 불안정 고려
