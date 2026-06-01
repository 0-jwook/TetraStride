from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v18 — termination_height 0.155→0.165m (쪼그림 차단).

    B-v17 분석 (iter 1446):
      - stance_4 = 0.877 ✓ (0.80 달성), ang_vel_z = -0.011 ✓
      - 문제: body_height=0.162m, pitch=10.2° (다시 쪼그림으로 회귀)
      - 원인: 발 들릴 위험(-100/step) > 몸 올리는 이득(+7.8/step)
        → 정책이 안전하게 낮게 쪼그리는 전략 선택
      - 보상 강화만으로는 한계, 쪼그림 자체를 물리적으로 차단 필요

    B-v18 핵심 수정:
      - termination_height: 0.155 → 0.165m
        0.162m 쪼그림 자세 → 에피소드 즉시 종료!
        0.165m 이상 유지 필수 → 정책이 다리를 뻗어야만 생존
      - 0.165m는 적당히 뻗은 자세 (iter 126에서 0.168m 달성 확인)
      - body_height=30, per_leg=15, foot_contact=40 유지
    """

    episode_length_s: float = 20.0
    termination_height: float = 0.165  # B-v18: 0.155→0.165m (0.162m 쪼그림 차단)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 핵심 보상: 결과 기반 ===
    rew_scale_alive: float = 1.0

    # 몸통 높이 (보조적)
    target_body_height: float = 0.177
    rew_scale_body_height: float = 30.0  # B-v17: 10→30 (쪼그림→서기 유도)
    asymmetric_height_reward: bool = True  # 목표 이하만 패널티

    # 수평 유지
    rew_scale_upright: float = 22.0

    # 4발 접지 — 지배적 보상 (4발=40, 3발=15, 1발=5)
    # 4발(40) >> 부유(upright22+height10=32) → 발 디디는게 유리
    rew_scale_foot_contact: float = 40.0  # B-v15: 8→40 (지배적 보상)

    # === joint_match 비활성 ===
    rew_scale_joint_match: float = 0.0  # B-v14: 완전 제거 (결과만 요구)

    # === B-v16: 다리별 개별 접지 보상 ===
    rew_scale_per_leg_contact: float = 15.0  # 발 1개당 15점, 4발=60 (선형 gradient)

    # === 안정화 패널티 ===
    rew_scale_ang_vel_z: float = -30.0     # linear 패널티 (B-v12에서 검증)
    rew_scale_lin_vel_xy: float = -200.0   # 수평 이동 억제
    rew_scale_base_drift: float = -60.0    # drift 억제
    rew_scale_ang_vel_xy: float = -0.5     # 롤/피치 각속도 억제
    rew_scale_lin_vel_z: float = -5.0      # 수직 움직임 억제
    rew_scale_gravity: float = -5.0

    # === 자세 품질 ===
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -3.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5

    # === 무릎 서기 차단 유지 ===
    rew_scale_non_foot_contact: float = -30.0
    non_foot_contact_threshold: float = 10.0

    # === 어깨 패널티 (넓은 허용치로 완화) ===
    rew_scale_shoulder_default: float = -20.0  # B-v14: 40→20 (joint_match 없으니 완화)

    # === 종료 조건 ===
    termination_drift_m: float = 0.08
    termination_shoulder_rad: float = 0.35   # joint_match 없으니 완화
    termination_tilt_cos: float = -0.940

    # === 비활성 ===
    rew_scale_foot_contact_4: float = 0.0
    rew_scale_standing_quality: float = 0.0
    rew_scale_knee_height_stance: float = 0.0
    rew_scale_knee_clearance: float = 0.0
    rew_scale_foot_alignment: float = 0.0
    rew_scale_knee_angle: float = 0.0
    rew_scale_leg_extension: float = 0.0
    rew_scale_per_leg_ext: float = 0.0
    rew_scale_stand_still: float = 0.0
    rew_scale_foot_spread: float = 0.0
    rew_scale_foot_side_span: float = 0.0
    rew_scale_joint_default: float = 0.0
    rew_scale_front_rear_sym: float = 0.0
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0
    rew_scale_dof_pos_limits: float = 0.0
    rew_scale_contact_forces: float = 0.0

    # === 기타 ===
    freeze_gait_phase: bool = True
    orientation_sigma: float = 0.10
    action_scale: float = 0.10
    init_crouch_prob: float = 0.0
