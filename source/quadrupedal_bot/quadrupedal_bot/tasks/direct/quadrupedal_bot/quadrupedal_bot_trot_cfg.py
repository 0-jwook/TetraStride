from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: Rudin 2022 방식 처음부터 학습 — velocity tracking + air_time으로 trot 자연 발생."""

    episode_length_s: float = 20.0          # 길게: 탐색 충분히 허용

    action_scale: float = 0.35              # base 기본값 유지

    # CCP velocity commands: 전방향 (약한 명령부터 시작)
    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)
    cmd_ang_vel_z_range: tuple = (-0.3, 0.3)
    zero_command_prob: float = 0.1          # 10% 서기 명령 (안정성 유지)

    # --- 핵심: Rudin 2022 속도 추적 ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 3.0          # 속도 추적 (moderate)
    rew_scale_ang_vel: float = 0.3          # yaw 추적
    rew_scale_ang_vel_z: float = -0.5       # 약한 yaw 오차 패널티
    rew_scale_movement: float = 0.0         # off: lin_vel으로 충분
    rew_scale_lin_vel_penalty: float = 0.0  # off: 처음부터는 패널티 없이 탐색

    # --- Gait 유도: air_time 중심 (gait clock 보조) ---
    rew_scale_gait: float = 0.0             # 처음부터: gait clock 없이 emergent
    rew_scale_air_time: float = 2.0         # 발 들기 보상 (Rudin 2022 수준)
    air_time_threshold: float = 0.1         # 0.1s = 5 steps
    rew_scale_air_time_var: float = 0.0     # off (초기 학습 간섭 방지)

    # --- 자세 안정 (legged_gym 표준) ---
    rew_scale_body_height: float = 0.0      # off: height target은 termination으로만
    rew_scale_upright: float = 0.0          # off: gravity로 충분
    rew_scale_gravity: float = -1.0         # 기울기 패널티 (표준)
    rew_scale_ang_vel_xy: float = -0.05     # roll/pitch 속도 패널티
    rew_scale_lin_vel_z: float = -2.0       # 수직속도 패널티

    # --- 관절/토크 제약 (legged_gym 표준) ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = -5.0     # 낮게: 패널티 너무 크면 안전제일 편향

    # --- 자세 유지 ---
    rew_scale_joint_default: float = 0.0    # off: 초기 학습에 방해
    rew_scale_foot_spread: float = 0.0      # off
    rew_scale_foot_slip: float = -0.05      # 약한 슬립 패널티
