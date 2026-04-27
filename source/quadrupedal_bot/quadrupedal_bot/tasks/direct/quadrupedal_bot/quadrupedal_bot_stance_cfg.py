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

    rew_scale_alive: float = 1.0              # 생존 보상 강화 (early death trap 방지)
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_lin_vel_z: float = -1.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -2.0          # -5.0→-2.0: per-step 패널티 완화
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = -200.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0
    rew_scale_body_height: float = 0.0
    rew_scale_non_foot_contact: float = 0.0
    rew_scale_lin_vel_xy: float = -0.1        # 수평 이동 패널티
    rew_scale_ang_vel_z: float = -0.1         # yaw 스핀 패널티
    rew_scale_joint_default: float = -0.2     # 어깨 패널티
    rew_scale_upright: float = 1.0
    rew_scale_foot_spread: float = -0.5       # -2.0→-0.5: Stage1 초기엔 약하게
    rew_scale_dof_acc: float = -2.5e-7
    rew_scale_action_acc: float = -0.002
    rew_scale_foot_slip: float = -0.02        # -0.05→-0.02: 약하게
    rew_scale_no_air: float = 0.0             # 서기 단계: 비활성화
    rew_scale_foot_clearance: float = 0.0     # 서기 단계: 비활성화
    rew_scale_stand_still: float = -0.1       # -0.5→-0.1: 약하게
