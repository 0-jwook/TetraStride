from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 — 제자리에서 쓰러지지 않고 자세 유지."""

    episode_length_s: float = 20.0  # 더 긴 에피소드로 지속적 서기 학습

    termination_height: float = 0.16   # 0.12→0.16: 낮은 자세 생존 차단 (body_height=0.155m 패치)

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
    rew_scale_air_time: float = 0.0     # 발 들기 없음
    rew_scale_movement: float = 0.0     # 이동 없음
    rew_scale_gait: float = 0.0         # 보행 패턴 없음
    target_body_height: float = 0.19            # leg=0.83, kp=20: gravity_sag=0.052 rad, 실 평형점 ~0.19m
    rew_scale_body_height: float = 2.0          # Gaussian 보상 (sigma=0.05): 목표 근처 최대 +2.0/step
    rew_scale_non_foot_contact: float = 0.0   # Stage 1: 비활성화 (서기 학습에 불필요)
    rew_scale_lin_vel_xy: float = -0.3         # 제자리 유지: 수평 이동 패널티
    rew_scale_ang_vel_z: float = -0.3          # yaw 스핀 패널티
    rew_scale_joint_default: float = -0.5      # 어깨 0.2 rad 초과 이탈 시 패널티
    rew_scale_upright: float = 2.0            # 직립 유지 강화
    rew_scale_foot_spread: float = -2.0       # 발 안쪽 모임 방지 (sliding survival 차단)
    rew_scale_foot_slip: float = -0.05        # 미끄러짐 패널티 활성화
    rew_scale_stand_still: float = -0.3      # 재활성화: kp=20에서 Gaussian 보상 부호+와 균형 가능
    freeze_gait_phase: bool = True    # gait clock 동결: 명령=0인 stance에서 주기적 불안정 제거
    rew_scale_dof_pos_limits: float = -1.0   # 관절 soft limit 초과 패널티 (실로봇 서보 보호)
    rew_scale_contact_forces: float = -1e-3  # 발 착지 충격력 패널티 (legged_gym 표준 스케일)
    action_scale: float = 0.25               # kp=20, effort=10: no saturation, ep_len=999 달성 설정
