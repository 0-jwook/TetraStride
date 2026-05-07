from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: 트롯 보행 학습 — 접촉 스케줄 보상으로 교대 발 들기 + 약한 전진."""

    episode_length_s: float = 10.0

    # CCP velocity commands: 전방향
    cmd_lin_vel_x_range: tuple = (-1.0, 1.0)
    cmd_lin_vel_y_range: tuple = (-0.5, 0.5)
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)
    zero_command_prob: float = 0.1

    # 보상: 접촉 스케줄 + 자세 유지 + 약한 속도 추적
    rew_scale_alive: float = 0.5
    rew_scale_body_height: float = 1.5    # base_cfg의 -8.0 오버라이드 (양수 Gaussian으로 역전)
    rew_scale_lin_vel: float = 5.0        # 4.0→5.0: 전진 전용이므로 강화
    rew_scale_ang_vel: float = 0.05
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -2.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = -10.0  # 조기 종료 패널티 추가
    rew_scale_air_time: float = 3.0
    rew_scale_movement: float = 3.0       # 1.5→3.0: 전진 이동 보상 강화
    rew_scale_gait: float = 2.0           # 새 공식: (4-error)*scale → bound=2점 vs trot=4점
    # Stage1에서 누락됐던 자세 안정 보상 추가
    rew_scale_upright: float = 0.5       # IMU 직립 보상 (Stage1의 50%)
    rew_scale_ang_vel_z: float = -0.1    # yaw 스핀 패널티
    rew_scale_joint_default: float = -2.0  # -0.2→-2.0: 도마뱀 어깨 차단 강화
    rew_scale_foot_spread: float = -5.0   # 양방향 패널티로 변경됨 (env.py 수정)
