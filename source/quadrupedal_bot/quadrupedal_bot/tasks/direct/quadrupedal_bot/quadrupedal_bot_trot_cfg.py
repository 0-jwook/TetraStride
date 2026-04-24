from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: 트롯 보행 학습 — 접촉 스케줄 보상으로 교대 발 들기 + 약한 전진."""

    episode_length_s: float = 10.0

    # 낮은 속도 명령: air_time gate(>0.1 m/s) 활성화 + 약한 전진 유도
    cmd_lin_vel_x_range: tuple = (0.2, 0.5)
    cmd_lin_vel_y_range: tuple = (-0.1, 0.1)
    cmd_ang_vel_z_range: tuple = (-0.2, 0.2)

    # 보상: 접촉 스케줄 + 자세 유지 + 약한 속도 추적
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 2.0      # 속도 추적 강화 (gait 축소로 균형)
    rew_scale_ang_vel: float = 0.05
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -2.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = 0.0
    rew_scale_air_time: float = 3.0     # 축소: value_loss 폭발 방지
    rew_scale_movement: float = 1.5     # 이동 보상
    rew_scale_gait: float = 5.0         # 축소: 15→5, 총 보상 규모 제어
