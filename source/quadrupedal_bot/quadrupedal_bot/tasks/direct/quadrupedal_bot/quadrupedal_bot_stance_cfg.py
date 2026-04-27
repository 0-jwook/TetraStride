from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 — 제자리에서 안정적 자세 유지."""

    episode_length_s: float = 20.0

    # 속도 명령 없음
    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.1
    rew_scale_gravity: float = -5.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.05
    rew_scale_termination: float = -200.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0
    rew_scale_body_height: float = 0.0
    rew_scale_non_foot_contact: float = 0.0
    rew_scale_lin_vel_xy: float = -0.3        # 수평 이동 패널티
    rew_scale_ang_vel_z: float = -0.3         # yaw 스핀 패널티
    rew_scale_joint_default: float = -0.5     # 어깨 패널티
    rew_scale_upright: float = 1.0
    rew_scale_foot_spread: float = -2.0       # 발 간격 패널티 (Stage1부터 활성화)
    rew_scale_dof_acc: float = -2.5e-7
    rew_scale_action_acc: float = -0.005
    rew_scale_foot_slip: float = -0.05        # 미끄러짐 패널티
    rew_scale_no_air: float = 0.0             # 서기 단계: 비활성화
    rew_scale_foot_clearance: float = 0.0     # 서기 단계: 비활성화
    rew_scale_stand_still: float = -0.5       # cmd=0일 때 관절 이탈 패널티
