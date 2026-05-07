from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: trot 보행 — Rudin 2022 방식 (gait clock 없이 air_time으로 자연 발생)."""

    episode_length_s: float = 10.0

    # Stage1과 동일한 action_scale 유지 (0.35→0.25): 체크포인트 전이 안정성
    action_scale: float = 0.25

    # CCP velocity commands: 전방향
    cmd_lin_vel_x_range: tuple = (-1.0, 1.0)
    cmd_lin_vel_y_range: tuple = (-0.5, 0.5)
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)
    zero_command_prob: float = 0.1

    # --- 핵심 속도 추적 (Rudin 2022 기반) ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 5.0         # exp(-||cmd-vel||²/0.25) — 주요 추진력
    rew_scale_ang_vel: float = 0.5         # 0.05→0.5: yaw 명령 추적 강화
    rew_scale_ang_vel_z: float = -2.0      # -0.1→-2.0: spinning 강력 억제
    rew_scale_movement: float = 0.0        # 제거: lin_vel tracking과 중복, spinning 유발

    # --- Gait 자연 발생 보상 (gait clock reward 없음) ---
    rew_scale_gait: float = 0.0            # 비활성 — gait clock reward 제거 (Rudin 방식)
    rew_scale_air_time: float = 5.0        # 강화: 발 들기 적극 유도 (threshold=0.3s)
    rew_scale_air_time_var: float = 2.0    # 비대칭 정책 차단 (한 다리만 움직임 패널티)

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
