from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v21 — 어깨 패널티 강화 + 조기종료 완화 (에피소드 완주).

    B-v20 시각화 결과:
      - 잘 서있다가 뒷다리가 중앙으로 모이면서 종료 발생
      - 원인: shoulder kp=20(약함) + shoulder 패널티 -5(약함)
              → 어깨 관절이 안쪽으로 drift → shoulder 종료 조건(0.35rad) 발동
      - 목표: 에피소드 20초 완주

    B-v21 수정:
      1. rew_scale_shoulder_default: -5 → -30 (어깨 수렴 강력 차단)
      2. termination_shoulder_rad: 0.35 → 0.50 (덜 공격적인 종료)
      3. termination_drift_m: 0.08 → 0.12 (조기종료 완화)
      4. B-v20 곱셈 보상 구조 유지

    B-v20 근본 수정 (사용자 제안):
      1. CoM 높이 → 각 다리 FK 뻗음 길이 (0.177m 목표)
         leg_ext = 0.1075×cos(hip) + 0.130×cos(hip+knee)
         기울어도 속일 수 없음 — 4발 각각 0.177m여야 최대 보상
      2. 가산 → 곱셈 보상
         rew = _e_q × _u_q × _c_q × scale
         (다리뻗음 × 수평 × 4발접지) → 하나라도 나쁘면 전체 감소

    보상 구조 (max ~130/step):
      곱셈 품질: scale=80 → 완벽 시 80, 하나라도 나쁘면 급감
      탐색 gradient: per_leg_contact=10, upright=15 (소규모 가산 유지)
      패널티: yaw, drift, stability
    """

    episode_length_s: float = 20.0
    termination_height: float = 0.165  # B-v18: 쪼그림 차단

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # ─── 핵심: 곱셈 서기 품질 보상 ─────────────────────────────────────
    # _e_q(다리뻗음) × _u_q(수평) × _c_q(4발접지) × scale
    # 4발 완전 뻗음 + 수평 + 4발 접지 → 80/step
    # 하나라도 나쁘면 급감 (기울기 14° 시: 80×0.56=45, 다리 짧으면 추가 감소)
    target_leg_extension: float = 0.177   # FK 목표 뻗음 길이 (m)
    sigma_leg_extension: float = 0.02     # exp 허용 오차 2cm (작을수록 타이트)
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
    rew_scale_shoulder_default: float = -30.0  # B-v21: -5→-30 (어깨 수렴 강력 차단)

    # ─── 종료 조건 ──────────────────────────────────────────────────────
    termination_drift_m: float = 0.12        # B-v21: 0.08→0.12 (조기종료 완화)
    termination_shoulder_rad: float = 0.50   # B-v21: 0.35→0.50 (어깨 수렴 허용 폭 확대)
    termination_tilt_cos: float = -0.940

    # ─── 나머지 비활성 ──────────────────────────────────────────────────
    rew_scale_standing_quality_old: float = 0.0
    rew_scale_foot_contact_4: float = 0.0
    rew_scale_front_rear_sym: float = 0.0
    rew_scale_leg_extension: float = 0.0
    rew_scale_per_leg_ext: float = 0.0
    rew_scale_knee_height_stance: float = 0.0
    rew_scale_knee_clearance: float = 0.0
    rew_scale_foot_alignment: float = 0.0
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
