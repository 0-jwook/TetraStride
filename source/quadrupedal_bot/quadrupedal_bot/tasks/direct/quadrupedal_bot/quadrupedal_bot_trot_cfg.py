from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: trot 보행 — Rudin 2022 방식 (gait clock 없이 air_time으로 자연 발생)."""

    episode_length_s: float = 10.0

    # Stage1과 동일한 action_scale 유지 (0.35→0.25): 체크포인트 전이 안정성
    action_scale: float = 0.25

    # CCP velocity commands: 달성 가능한 작은 범위 (학습 안정화)
    cmd_lin_vel_x_range: tuple = (0.1, 0.5)  # 작은 목표속도: 초기에 달성 가능한 범위
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)
    cmd_ang_vel_z_range: tuple = (-0.3, 0.3)
    zero_command_prob: float = 0.0           # 서기 명령 제거 유지

    # --- 핵심 속도 추적 (패널티 추가: 서기 탈출 강제) ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 8.0         # 속도 추적 보상
    rew_scale_ang_vel: float = 0.5         # yaw 명령 추적
    rew_scale_ang_vel_z: float = -2.0      # cmd와의 yaw 오차 패널티
    rew_scale_movement: float = 4.0        # 실제 전진 선형 보상
    rew_scale_lin_vel_penalty: float = -6.0  # 속도 미달 직접 패널티: -6×||cmd-vel||²

    # --- Gait 유도 ---
    rew_scale_gait: float = 2.5            # 가이드 유지 (지배 않도록 축소)
    rew_scale_air_time: float = 6.0        # 발 들기 보상
    air_time_threshold: float = 0.05       # 낮은 threshold
    rew_scale_air_time_var: float = 2.0    # 비대칭 정책 차단

    # --- 자세 안정 ---
    rew_scale_body_height: float = 1.5     # Gaussian 양수: 목표 높이 유지
    rew_scale_upright: float = 0.5
    rew_scale_gravity: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_lin_vel_z: float = -2.0

    # --- 관절/토크 제약 ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = -10.0

    # --- 자세 유지 ---
    rew_scale_joint_default: float = -2.0  # 어깨 dead zone 0.05 rad
    rew_scale_foot_spread: float = -5.0    # 양방향 패널티 (env.py)
    rew_scale_foot_slip: float = -0.1
