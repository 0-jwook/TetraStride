from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v10 — foot_contact 복원 + σ=0.15 (발 부상 문제 해결).

    B-v9 분석 (iter 1844):
      - hip_RR: 0.255→0.489 ✓ (σ=0.20 효과, 기울기 복원 성공)
      - knee_FR = -2.028 → leg_ext=0.118m → FR 발이 지면 4cm 부상
      - stance_4 = 0.097 (90% 동안 4발 접지 실패!)
      - 원인: joint_match만으로는 발 접지 동기 없음 (rew_scale_foot_contact=0)
      - σ=0.20 너무 넓어 knee over-bending 억제 불충분

    B-v10 핵심 수정:
      1. rew_scale_foot_contact: 0 → 8.0 (4발 접지 직접 보상)
         4발: 8.0/step, 3발: 3.0/step (발 들리면 즉시 손해)
      2. sigma_joint_match: 0.20 → 0.15 (타협점)
         hip_RR dev=0.34: exp(-2.27)=0.104, gradient 0.693/rad (충분)
         knee_FR dev=0.55: exp(-3.67)=0.026, 강한 패널티
      3. 나머지 유지 (drift 0.08m, yaw -30, sym, min-limb)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.145  # new-v22: 0.120→0.145 (0.126m 크라우치 전략 차단)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === B 접근법 핵심 보상 ===
    rew_scale_alive: float = 1.0
    rew_scale_upright: float = 22.0  # B-v4: 30→22 (upright 강화가 어깨 벌리기 악화시킴)
    rew_scale_foot_contact: float = 8.0  # B-v10: 4발 접지 직접 보상 (4발=8, 3발=3, 1발=1)

    # B-v10: σ 타협점 (기울기 유지 + over-bending 억제)
    sigma_joint_match: float = 0.15   # 0.20→0.15: hip_RR 0.34편차 gradient=0.104, knee 0.55편차=0.026
    rew_scale_joint_match: float = 60.0  # max = 60.0 × 3 = 180/step 동일

    # === 높이 보상 (보조) ===
    target_body_height: float = 0.177
    rew_scale_body_height: float = 10.0  # 보조적 높이 신호
    asymmetric_height_reward: bool = True  # 목표 이하만 패널티

    # === 다리 뻗음 보상 (v34 신규) ===
    # leg_ext = 0.1075×cos(leg) + 0.130×cos(leg+knee), 목표 0.177m
    # 4발 개별 Gaussian 합산 → max 4×1.0×scale = 4×2.0 = 8.0
    target_leg_extension: float = 0.177
    sigma_leg_extension: float = 0.05   # v35: 0.02→0.05 (exp(-|err|/σ), body_height와 동일 계열)
    rew_scale_leg_extension: float = 0.0   # new-v8: 제거 (per_leg_ext로 통합)

    # === 무릎 서기 차단 강화 ===
    rew_scale_non_foot_contact: float = -30.0  # v30:-8.0 → v31:-30.0 (버그 수정: 중복 정의 제거)
    non_foot_contact_threshold: float = 10.0   # new-v6: 5N→10N (과민 노이즈 감소, 명확한 접촉만 감지)

    # === stand_still 비활성 유지 ===
    rew_scale_stand_still: float = 0.0

    # === 자세 패널티 ===
    rew_scale_knee_height_stance: float = -30.0
    knee_stance_height_threshold: float = 0.06
    rew_scale_knee_angle: float = -3.0

    # === 안정화 패널티 ===
    rew_scale_gravity: float = -5.0
    rew_scale_ang_vel_xy: float = -0.5
    rew_scale_lin_vel_z: float = -5.0
    rew_scale_ang_vel_z: float = -30.0  # B-v7: -10→-30 (yaw 3배 강화, 스피닝 해킹 차단)
    rew_scale_lin_vel_xy: float = -200.0  # new-v23: -100→-200 (vel 정체 해결, 2배 추가 강화)
    rew_scale_base_drift: float = -60.0  # new-v26: -20→-60 (스텔스 드리프트 차단, 3배)

    # === 동작 품질 ===
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -3.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5

    # === 형상 패널티 (foot_spread 계열 제거) ===
    rew_scale_shoulder_default: float = -40.0  # new-v31: -30→-40 (어깨 벌리기 강력 차단)
    rew_scale_joint_default: float = -8.0  # v37: -5.0→-8.0 (knee 과굽힘 억제 강화)
    rew_scale_foot_spread: float = 0.0        # v26:-3 → v27:0 (가라앉기 인센티브 제거)
    rew_scale_foot_side_span: float = 0.0     # v26:-3 → v27:0 (동일 이유)

    # === 비활성 ===
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0

    # === 발-어깨 수평 정렬 (new-v1 신규) ===
    # world frame에서 shoulder_link와 foot_link의 XY 수평 거리 측정
    # 발이 어깨 바로 아래 위치할수록 보상 증가 (sigma=0.08m, 5.5cm hip_flex offset 포함)
    sigma_foot_alignment: float = 0.08
    rew_scale_foot_alignment: float = 3.0   # new-v18: 재활성화 (leg_link 기준, 발-어깨 정렬)
    sigma_foot_alignment: float = 0.05      # new-v18: 0.08→0.05 (더 타이트한 정렬)
    termination_drift_m: float = 0.08       # B-v7: 0.20→0.08m 복원 (스피닝 차단 핵심)
    termination_shoulder_rad: float = 0.20  # B-v7: 0.30→0.20rad 소폭 복원
    termination_tilt_cos: float = -0.940    # new-v28: 45°→20° 강화 (cos(20°)=-0.940)
    rew_scale_per_leg_ext: float = 0.0      # new-v17: 제거 (body_height가 대체)
    rew_scale_standing_quality: float = 0.0   # new-v17: 제거 (단순화)
    rew_scale_knee_clearance: float = 5.0   # new-v8: 무릎 높이 보상 (max 4×0.15×5=3/step)

    # === B-v8: 앞/뒤 다리 대칭 보상 ===
    sigma_front_rear_sym: float = 0.03    # 허용 오차 3cm
    rew_scale_front_rear_sym: float = 10.0  # max 10.0/step (pitch 순환고리 차단)

    # === 기타 ===
    rew_scale_dof_pos_limits: float = -1.0
    rew_scale_contact_forces: float = -1e-3
    freeze_gait_phase: bool = True
    orientation_sigma: float = 0.10
    action_scale: float = 0.10
    init_crouch_prob: float = 0.0   # new-v15: 쪼그림 시작 비활성 (서기 자세만)
