from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: 트롯 보행 학습 — 접촉 스케줄 보상으로 교대 발 들기 + 약한 전진."""

    episode_length_s: float = 10.0

    # CCP velocity commands: 전진/후진/좌우/회전 전방향 학습
    cmd_lin_vel_x_range: tuple = (-1.0, 1.0)
    cmd_lin_vel_y_range: tuple = (-0.5, 0.5)
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)
    zero_command_prob: float = 0.02   # 2%는 cmd=(0,0,0) → 제자리 서기 병행 학습

    # 보상: 접촉 스케줄 + 자세 유지 + 약한 속도 추적
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 4.0        # 1.5→4.0: 전진 속도 추적 대폭 강화
    rew_scale_ang_vel: float = 0.05
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -2.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = 0.0
    rew_scale_air_time: float = 3.0
    rew_scale_movement: float = 3.0       # 1.5→3.0: 전진 이동 보상 강화
    rew_scale_gait: float = 3.0           # 2.0→3.0: 트롯 보행 패턴 강화
    # Stage1에서 누락됐던 자세 안정 보상 추가
    rew_scale_upright: float = 0.5       # IMU 직립 보상 (Stage1의 50%)
    rew_scale_ang_vel_z: float = -0.1    # yaw 스핀 패널티
    rew_scale_joint_default: float = -0.2  # 어깨 dead zone 패널티
    rew_scale_foot_spread: float = -5.0  # -2→-5: 발 중앙 모임 방지 강화
