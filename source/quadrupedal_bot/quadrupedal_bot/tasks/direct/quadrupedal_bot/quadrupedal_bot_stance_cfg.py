from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v26 — foot_alignment 보상 활성화 (앞발 앞쏠림 해결).

    B-v23 시각화 결과:
      - 앞발이 어깨 조인트보다 앞에 있다가 아래로 교정될 때
        뒷다리가 접히며 뒤로 기울어져 넘어짐
      - 원인: 에피소드(89 steps=0.7초)가 짧아 교정 동작을 훈련에서 경험 못함
        -> 시각화에서 교정 시도하지만 균형 유지 못함
      - hip 각도 0.87~0.98 rad -> 발이 어깨보다 1~2cm 앞에 위치

    B-v26 수정:
      - rew_scale_foot_alignment: 0 -> 5.0 (재활성화)
        발이 어깨 바로 아래 있을 때 최대 보상
        -> 처음부터 발을 올바른 위치에 두도록 학습
        -> 교정 동작 자체가 불필요해짐
      - sigma_foot_alignment: 0.05 (기존 값 사용)
        hip=0.83(올바른): exp(0)=1.0, hip=0.90(1.1cm 앞): exp(-0.22)=0.80
      - 나머지 B-v25 설정 유지
    """

    episode_length_s: float = 20.0
    # 학습용 복원: 적당한 종료 조건과 노이즈
    termination_height: float = 0.145
    init_noise_scale: float = 0.03

    # dt 1/200 유지 (학습 속도), solver 32 적용
    decimation: int = 4
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # ─── 핵심: 곱셈 서기 품질 보상 ─────────────────────────────────────
    # _e_q(다리뻗음) × _u_q(수평) × _c_q(4발접지) × scale
    # 4발 완전 뻗음 + 수평 + 4발 접지 -> 80/step
    # 하나라도 나쁘면 급감 (기울기 14° 시: 80×0.56=45, 다리 짧으면 추가 감소)
    target_leg_extension: float = 0.177   # FK 목표 뻗음 길이 (m)
    sigma_leg_extension: float = 0.02     # B-v25: 0.01->0.02 복원 (B-v23 좋은 자세 유지)
    rew_scale_standing_quality: float = 80.0  # 곱셈 메인 보상

    # 자세 orientation_sigma: 수평 민감도
    orientation_sigma: float = 0.08       # 작을수록 수평 요구 강함

    # ─── 탐색 그래디언트 (소규모 가산, 초기 학습 도움) ─────────────────
    rew_scale_alive: float = 1.0
    rew_scale_upright: float = 15.0       # 소규모 (품질에 이미 포함, gradient용)
    rew_scale_per_leg_contact: float = 10.0  # 발당 10점 (탐색 gradient)

    # ─── 비활성 (곱셈 품질로 대체) ──────────────────────────────────────
    rew_scale_body_height: float = 0.0    # CoM 높이 보상 제거 (다리 뻗음으로 대체)
    rew_scale_foot_contact: float = 0.0   # 품질의 _c_q로 대체
    rew_scale_joint_match: float = 0.0

    # B-v26: 발-어깨 수직 정렬 보상 (앞발 앞쏠림 방지)
    rew_scale_foot_alignment: float = 5.0  # 발이 어깨 바로 아래 -> 최대 보상
    sigma_foot_alignment: float = 0.05    # 5cm 허용 오차

    # ─── 패널티 (안정화) ─────────────────────────────────────────────────
    rew_scale_ang_vel_z: float = -30.0    # linear yaw 억제
    rew_scale_ang_vel_xy: float = -2.0    # 피치/롤 각속도 억제
    rew_scale_lin_vel_xy: float = -50.0   # 수평 이동 억제
    rew_scale_base_drift: float = -30.0   # drift 억제
    rew_scale_lin_vel_z: float = -5.0     # 수직 움직임 억제
    rew_scale_gravity: float = -3.0
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -1.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_non_foot_contact: float = -30.0  # 무릎 접지 차단
    non_foot_contact_threshold: float = 10.0
    rew_scale_shoulder_default: float = -80.0  # B-v22: -30->-80 (패널티 2.7배 강화)

    # ─── 종료 조건 ──────────────────────────────────────────────────────
    termination_drift_m: float = 0.15        # 학습용: 적당히 관대
    termination_shoulder_rad: float = 0.40   # 학습용: 어깨 수렴 차단
    termination_tilt_cos: float = -0.940     # 학습용: 20° 한계

    # ─── 나머지 비활성 ──────────────────────────────────────────────────
    rew_scale_standing_quality_old: float = 0.0
    rew_scale_foot_contact_4: float = 0.0
    rew_scale_front_rear_sym: float = 0.0
    rew_scale_leg_extension: float = 0.0
    rew_scale_per_leg_ext: float = 0.0
    rew_scale_knee_height_stance: float = 0.0
    rew_scale_knee_clearance: float = 0.0
    rew_scale_knee_angle: float = 0.0
    rew_scale_stand_still: float = 0.0
    rew_scale_foot_spread: float = 0.0
    rew_scale_foot_side_span: float = 0.0
    rew_scale_joint_default: float = 0.0
    rew_scale_dof_pos_limits: float = 0.0
    rew_scale_contact_forces: float = 0.0
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0
    freeze_gait_phase: bool = True
    action_scale: float = 0.10
    init_crouch_prob: float = 0.0
