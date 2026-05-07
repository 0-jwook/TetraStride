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

    # --- 핵심 속도 추적 (단계2: 제자리 trot 획득 후 전진 유도) ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 8.0         # 5→8: 속도 추적 강화 (trot-in-place 탈출)
    rew_scale_ang_vel: float = 0.5         # yaw 명령 추적
    rew_scale_ang_vel_z: float = -2.0      # cmd와의 yaw 오차 패널티
    rew_scale_movement: float = 5.0        # 1.5→5.0: 실제 전진에 강한 선형 보상 (gait와 비슷한 스케일)

    # --- Gait 유도 (축소: 전진보상이 더 중요해야 함) ---
    rew_scale_gait: float = 2.5            # 6.0→2.5: gait 지속 유지하되 전진 억제 없도록 축소
    rew_scale_air_time: float = 8.0        # 발 들기 보상 유지
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
