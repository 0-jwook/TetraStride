from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 — 제자리에서 쓰러지지 않고 자세 유지."""

    episode_length_s: float = 20.0  # 더 긴 에피소드로 지속적 서기 학습

    # 속도 명령 없음 — 항상 제자리
    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # 보상: 자세(중력 정렬) 유지만 학습
    rew_scale_alive: float = 0.5        # 살아있기 (서 있기) 보상
    rew_scale_lin_vel: float = 0.0      # 속도 추적 없음
    rew_scale_ang_vel: float = 0.0      # 각속도 추적 없음
    rew_scale_lin_vel_z: float = -2.0   # 수직 진동 패널티
    rew_scale_ang_vel_xy: float = -0.1  # 롤/피치 각속도 패널티 (강화)
    rew_scale_gravity: float = -5.0     # 중력 정렬 패널티 (강화: 기울면 큰 패널티)
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = 0.0
    rew_scale_air_time: float = 0.0     # 발 들기 없음
    rew_scale_movement: float = 0.0     # 이동 없음
    rew_scale_gait: float = 0.0         # 보행 패턴 없음
    rew_scale_body_height: float = -3.0       # Stage 1: 완만한 높이 유도 (너무 강하면 학습 방해)
    rew_scale_non_foot_contact: float = 0.0  # Stage 1: 비활성화 (정강이 자연접촉으로 오발동)
