from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v25 — sigma 0.01->0.02 복원 + termination 0.145m 유지.

    B-v22 최종 결과:
      - stance_4 = 0.776 ✓, pitch = 4.48° ✓, ang_vel_z = 0.034 ✓
      - 문제: Mean episode_length = 105 steps (~0.87초, 목표 20초)
        term_height_ratio = 0.009 -> 높이 0.165m 이하 순간 하강마다 종료
        1/0.009 = 111 step -> 로봇이 안정적으로 서있지만 높이 종료가 자꾸 발동

    B-v23 수정:
      - termination_height: 0.165 -> 0.155m (완화)
        순간 하강 허용 폭 확대 -> 에피소드 완주 가능
        B-v20에서 검증된 값 (0.162m에서도 안정적으로 동작)
      - shoulder -80, 나머지 B-v22 유지
    """

    episode_length_s: float = 20.0
    termination_height: float = 0.145  # B-v24: 0.155->0.145m (바닥 낮춤)

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
    termination_drift_m: float = 0.12        # B-v21: 유지
    termination_shoulder_rad: float = 0.35   # B-v22: 0.50->0.35 복원 (강한 패널티로 사전 억제)
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
