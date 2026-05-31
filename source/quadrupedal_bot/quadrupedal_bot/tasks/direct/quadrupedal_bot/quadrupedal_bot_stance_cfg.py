from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v14 — joint_match 제거, 결과 기반 보상만 유지.

    B-v13 분석 및 방향 전환 근거:
      - joint_match는 특정 각도를 강제 → Stage 2(걷기)에서 스윙 동작과 충돌
      - hip_RR이 '물리적으로 안정한 각도'(0.37)를 찾아가는데 0.83을 강제해서 낭비
      - 13버전 × ~1.5시간 = 약 20시간을 서기 하나에 소비 (너무 느림)
      - 성공한 논문들은 body_height + upright + contact만으로 수렴

    B-v14 설계 원칙 (결과만 요구, 수단은 자유):
      - 필요한 것: 몸통 높이 유지 + 수평 유지 + 4발 접지 + 제자리 유지
      - 불필요한 것: 관절이 정확히 어디에 있는지 (자연 균형점에 맡김)
      - B-v13에서 검증된 인사이트 유지:
        spawn 0.22m, termination 0.155m, linear yaw, foot_contact=8

    기대 효과:
      - 학습 속도 3~5배 향상 (joint_match와의 충돌 없음)
      - Stage 2 전환 용이 (관절 구속 없음)
      - 자연 균형점에서 서기 유도
    """

    episode_length_s: float = 20.0
    termination_height: float = 0.155  # B-v13에서 검증: 낮은 local optimum 차단

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 핵심 보상: 결과 기반 ===
    rew_scale_alive: float = 1.0

    # 몸통 높이: 대칭 Gaussian (목표 위아래 모두 보상)
    target_body_height: float = 0.177
    rew_scale_body_height: float = 20.0  # B-v14: 10→20, 높이 신호 강화
    asymmetric_height_reward: bool = False  # B-v14: 대칭으로 변경 (양방향 gradient)

    # 수평 유지
    rew_scale_upright: float = 22.0

    # 4발 접지 (B-v10에서 검증: 4발=8, 3발=3, 1발=1)
    rew_scale_foot_contact: float = 8.0

    # === joint_match 비활성 ===
    rew_scale_joint_match: float = 0.0  # B-v14: 완전 제거 (결과만 요구)

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
