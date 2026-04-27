from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: 트롯 보행 학습."""

    episode_length_s: float = 10.0

    cmd_lin_vel_x_range: tuple = (0.1, 0.4)
    cmd_lin_vel_y_range: tuple = (-0.1, 0.1)
    cmd_ang_vel_z_range: tuple = (-0.2, 0.2)

    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 1.5
    rew_scale_ang_vel: float = 0.5
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -2.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.05
    rew_scale_termination: float = -200.0
    rew_scale_air_time: float = 3.0
    rew_scale_movement: float = 1.5
    rew_scale_gait: float = 2.0
    rew_scale_upright: float = 0.5
    rew_scale_ang_vel_z: float = -0.1
    rew_scale_joint_default: float = -0.2
    rew_scale_foot_spread: float = -2.0       # 발 간격 유지
    rew_scale_dof_acc: float = -2.5e-7
    rew_scale_action_acc: float = -0.005
    rew_scale_foot_slip: float = -0.05        # 미끄러짐 패널티
    rew_scale_no_air: float = -1.0            # 4발 전부 접지 패널티 (공중 phase 강제)
    rew_scale_foot_clearance: float = -2.0    # swing foot 높이 패널티
    rew_scale_stand_still: float = 0.0        # Trot 단계: 비활성화
