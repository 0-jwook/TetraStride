from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v15 — foot_contact 지배적 보상 + 부유 해킹 차단.

    B-v14 실패 (iter 874):
      - stance_4 = 0.000, torque_sat_knee = 57.8%
      - policy가 무릎 최대 토크로 공중 부유 전략 발견
      - foot_contact(8) < upright(17) + body_height(16) = 33 → 땅 밟을 이유 없음

    B-v15 핵심 수정:
      1. foot_contact: 8→40 (지배적 보상, 공중 부유 불가)
         4발(40) >> 부유(33 = upright+height) → 반드시 발을 땅에 디뎌야 유리
      2. body_height: 20→10 (보조적, height 신호만 유지)
      3. non_foot_contact: -30 유지 (무릎 닿음 억제)
      4. 나머지 B-v14 설정 유지
    """

    episode_length_s: float = 20.0
    termination_height: float = 0.155  # B-v13에서 검증: 낮은 local optimum 차단

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 핵심 보상: 결과 기반 ===
    rew_scale_alive: float = 1.0

    # 몸통 높이 (보조적)
    target_body_height: float = 0.177
    rew_scale_body_height: float = 10.0  # B-v15: 20→10 (보조)
    asymmetric_height_reward: bool = True  # 목표 이하만 패널티 (공중 부유 억제 안 함)

    # 수평 유지
    rew_scale_upright: float = 22.0

    # 4발 접지 — 지배적 보상 (4발=40, 3발=15, 1발=5)
    # 4발(40) >> 부유(upright22+height10=32) → 발 디디는게 유리
    rew_scale_foot_contact: float = 40.0  # B-v15: 8→40 (지배적 보상)

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
