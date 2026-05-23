from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 — 강한 anti-drift 패널티, 진짜 제자리 서기 강제."""

    episode_length_s: float = 20.0  # 더 긴 에피소드로 지속적 서기 학습

    termination_height: float = 0.13   # 역관절 자세 기준: 0.17m 목표에서 4cm 여유

    # 속도 명령 없음 — 항상 제자리
    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # 보상: 자세(중력 정렬) 유지만 학습
    rew_scale_alive: float = 0.5        # 0.1→0.5 복원: alive 낮추면 균형 학습 신호 소실 → 0.38s 내 낙하
    rew_scale_lin_vel: float = 0.0      # 속도 추적 없음
    rew_scale_ang_vel: float = 0.0      # 각속도 추적 없음
    rew_scale_lin_vel_z: float = -2.0   # 수직 진동 패널티
    rew_scale_ang_vel_xy: float = -0.1  # 롤/피치 각속도 패널티 (강화)
    rew_scale_gravity: float = -5.0     # 중력 정렬 패널티 (강화: 기울면 큰 패널티)
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_air_time: float = 0.0     # 발 들기 없음
    rew_scale_movement: float = 0.0     # 이동 없음
    rew_scale_gait: float = 0.0         # 보행 패턴 없음
    target_body_height: float = 0.17            # leg=0.83, foot=-1.55: 역관절 자세의 실 평형점
    rew_scale_body_height: float = 5.0          # Gaussian 보상 (sigma=0.05): 역관절 유지 강화
    rew_scale_non_foot_contact: float = -2.0   # 무릎/배 바닥 닿음 강한 패널티
    rew_scale_lin_vel_xy: float = -10.0        # 2x 강화 (alive=0.1 비율로 44%/episode → 의미있음)
    rew_scale_ang_vel_z: float = -0.3          # yaw 스핀 패널티
    rew_scale_joint_default: float = -0.5      # 관절 기본값(역관절) 이탈 패널티
    rew_scale_upright: float = 2.0            # 직립 유지 강화
    rew_scale_foot_spread: float = -8.0       # 다리 모임 차단
    rew_scale_foot_slip: float = -0.05        # 미끄러짐 패널티
    rew_scale_stand_still: float = -3.0        # v1 수준 복원: -5.0은 초기 joint_dev×scale=-16/step으로 value function 붕괴
    rew_scale_base_drift: float = -20.0        # 4x 강화 (v1 -5.0 대비): 안정 서기 후 위치 유지 강제
    freeze_gait_phase: bool = True    # gait clock 동결: 명령=0인 stance에서 주기적 불안정 제거
    rew_scale_dof_pos_limits: float = -1.0   # 관절 soft limit 초과 패널티 (실로봇 서보 보호)
    rew_scale_contact_forces: float = -1e-3  # 발 착지 충격력 패널티 (legged_gym 표준 스케일)
    action_scale: float = 0.25               # kp=30, effort=10: no saturation (stance 안정)
