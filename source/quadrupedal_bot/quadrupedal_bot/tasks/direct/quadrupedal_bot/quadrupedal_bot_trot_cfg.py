from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: trot 보행 — Rudin 2022 방식 (gait clock 없이 air_time으로 자연 발생)."""

    episode_length_s: float = 10.0

    # Stage1과 동일한 action_scale 유지 (0.35→0.25): 체크포인트 전이 안정성
    action_scale: float = 0.25

    # CCP velocity commands: 전진 중심 (서기 최적점 제거)
    cmd_lin_vel_x_range: tuple = (0.2, 1.0)  # 항상 양방향 전진 요구 → 서기 로컬옵티멈 차단
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)
    zero_command_prob: float = 0.0           # 서기 명령 완전 제거

    # --- 핵심 속도 추적 (Rudin 2022 기반) ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 5.0         # exp(-||cmd-vel||²/0.25) — 주요 추진력
    rew_scale_ang_vel: float = 0.5         # yaw 명령 추적
    rew_scale_ang_vel_z: float = -2.0      # cmd와의 yaw 오차 패널티
    rew_scale_movement: float = 1.5        # 실제 전진속도 선형 보상: 서기 대비 걸음에 추가 이득

    # --- Gait 유도 ---
    rew_scale_gait: float = 6.0            # 3.0→6.0: 서기 최적점 탈출용 강한 대각선 contact 유도
    rew_scale_air_time: float = 8.0        # 5.0→8.0: 발 들기 보상 강화
    air_time_threshold: float = 0.05       # 0.1s→0.05s: 낮은 threshold로 작은 발 들기도 보상
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
