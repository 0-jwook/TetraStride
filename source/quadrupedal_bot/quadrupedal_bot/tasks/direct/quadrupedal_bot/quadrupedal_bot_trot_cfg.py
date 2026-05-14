from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2 v18: 뚝뚝 끊김 제거 + 직선 보행 수정 (gait 압력↓ + lateral-only 패널티 + 동작 부드럽게)."""

    episode_length_s: float = 20.0
    target_body_height: float = 0.17

    action_scale: float = 0.35          # v17 그대로 유지 (전이학습 호환성)
    action_smoothing: float = 0.6       # 0.8→0.6: 래그 감소 → 뚝뚝 끊김 완화

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
    rew_scale_lin_vel_xy: float = -1.5  # -0.3→-1.5: vy²만 패널티 (env.py 수정으로 vx 제외됨) → 직선 보행 강제

    # --- Gait 유도 (압력 완화 — 뚝뚝 끊김 원인 제거) ---
    rew_scale_gait: float = 1.0         # 2.0→1.0: gait clock 강제 완화 (abrupt transition 감소)
    rew_scale_air_time: float = 5.0     # 8.0→5.0: 공중 시간 보상 완화 (힘차게 차는 동작 감소)
    air_time_threshold: float = 0.08    # 0.12→0.08s: 달성 가능한 threshold (1.5Hz에서 24% swing)
    rew_scale_swing_contact: float = -0.5  # -1.5→-0.5: swing 중 접촉 패널티 완화 (덜 abrupt)
    rew_scale_foot_height: float = 2.0  # 6.0→2.0: 발 높이 보상 대폭 감소 (뒷다리 힘차게 차기 억제)

    # --- 자세 안정 ---
    rew_scale_body_height: float = -8.0
    rew_scale_upright: float = 2.0
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.3
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.10  # -0.05→-0.10: 동작 부드럽게 (주춤거림 없이 매끄럽게)
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
