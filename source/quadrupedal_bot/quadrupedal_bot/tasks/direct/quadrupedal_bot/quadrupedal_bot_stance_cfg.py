from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 B-v19 — 보상 균형 재설계 (발접지 70%→45%, 자세 9%→30%).

    B-v18 분석 (iter 686):
      - stance_4=0.875 ✓, body_height=0.172m ✓ (0.165m 이상 달성)
      - 문제: pitch=14.5° (심각한 기울기)
      - 근본 원인: foot_contact+per_leg = 91/step = 양수보상의 70% 지배
        upright = 11.7/step = 9% 뿐 → 기울어도 발만 닿으면 손해 없음

    B-v19 보상 균형 재설계:
      목표 비율 → 발접지 ~45%, 자세 ~30%, 높이 ~20%

      양수 보상 (ideal 합계 ~126/step):
        upright:      22→40  (자세 중요도 2배, 30%)
        foot_contact: 40→20  (4발=20, 줄임)
        per_leg:      15→10  (4발=40, 줄임) → 발접지 합 60 (47%)
        body_height:  30→25  (줄임, 20%)
        orientation_sigma: 0.10→0.08 (자세 민감도 증가)

      음수 패널티 재조정:
        ang_vel_xy:  -0.5→-2.0 (피치/롤 각속도 강화)
        lin_vel_xy: -200→-50  (너무 강했음, 완화)
        base_drift:  -60→-30  (완화)
        shoulder:    -20→-5   (완화)
    """

    episode_length_s: float = 20.0
    termination_height: float = 0.165  # B-v18: 0.155→0.165m (0.162m 쪼그림 차단)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 핵심 보상: 결과 기반 ===
    rew_scale_alive: float = 1.0

    # ─── 양수 보상 (균형 재설계, ideal 합계 ~126/step) ──────────────
    # 목표 비율: 발접지 ~45%, 자세 ~30%, 높이 ~20%

    # 1) 수평 유지 (30%) — 핵심 자세 신호
    rew_scale_upright: float = 40.0         # 22→40 (자세 중요도 2배 상승)
    orientation_sigma: float = 0.08         # 0.10→0.08 (피치 민감도 강화)

    # 2) 4발 접지 (47%, 4발 기준 60/step)
    rew_scale_foot_contact: float = 20.0    # 40→20 (4발=20, 3발=7.5)
    rew_scale_per_leg_contact: float = 10.0 # 15→10 (발당 10, 4발=40)

    # 3) 몸통 높이 (20%)
    target_body_height: float = 0.177
    rew_scale_body_height: float = 25.0     # 30→25
    asymmetric_height_reward: bool = True

    # ─── 음수 패널티 (균형 재조정) ─────────────────────────────────
    rew_scale_ang_vel_z: float = -30.0      # linear yaw (유지)
    rew_scale_ang_vel_xy: float = -2.0      # -0.5→-2.0 (피치/롤 각속도 강화)
    rew_scale_lin_vel_xy: float = -50.0     # -200→-50 (너무 강했음, 완화)
    rew_scale_base_drift: float = -30.0     # -60→-30 (완화)
    rew_scale_lin_vel_z: float = -5.0       # 유지
    rew_scale_gravity: float = -3.0         # -5→-3 (완화)
    rew_scale_action_rate: float = -0.05    # 유지
    rew_scale_foot_slip: float = -1.0       # -3→-1 (완화)
    rew_scale_joint_vel: float = -1e-4      # 유지
    rew_scale_torque: float = -1e-5         # 유지

    # 무릎 서기 차단
    rew_scale_non_foot_contact: float = -30.0
    non_foot_contact_threshold: float = 10.0

    # 어깨 패널티 (완화)
    rew_scale_shoulder_default: float = -5.0  # -20→-5

    # joint_match 비활성
    rew_scale_joint_match: float = 0.0

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
