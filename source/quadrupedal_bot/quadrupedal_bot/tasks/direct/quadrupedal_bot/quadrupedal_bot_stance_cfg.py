from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 new-v10 — per_leg_ext 스케일 15→50으로 대폭 강화.

    new-v9 실패 (iter 1163):
      - stance4=85%, ep_len=999 달성했지만 body_height=0.067m (납작 눕기)
      - 로봇이 termination_height=4cm 위에 납작 누워 stance4+ep_len 모두 달성
      - per_leg_ext scale=15: 누운 상태 3.1/step vs 선 상태 10.6/step → delta 7.5/step
        → 누운 안정성 이득이 7.5/step 차이를 압도

    new-v10 수정:
      - per_leg_ext: 15.0 → 50.0
        (누운 10.2/step vs 선 35.4/step → delta 25.2/step로 서는 것이 압도적 유리)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.040  # new-v9: 0.100→0.040 (최대굽힘 자세 허용, knee=-2.59 시 ~5cm)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 1.0   # v36: 8.0→1.0 (alive 지배 제거, per-step 품질이 지배하도록)
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 12.0   # new-v8: 20→12 복귀 (per_leg_ext가 주도)

    # === 높이: root_pos_w 기반 제거, FK 다리 뻗음 보상으로 대체 ===
    target_body_height: float = 0.177
    rew_scale_body_height: float = 0.0   # new-v8: 제거 (per_leg_ext로 통합)

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
    rew_scale_ang_vel_z: float = -5.0
    rew_scale_lin_vel_xy: float = -8.0
    rew_scale_base_drift: float = -8.0

    # === 동작 품질 ===
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -3.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5

    # === 형상 패널티 (foot_spread 계열 제거) ===
    rew_scale_shoulder_default: float = -8.0
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
    rew_scale_foot_alignment: float = 0.0   # 비활성
    rew_scale_per_leg_ext: float = 50.0     # new-v10: 15→50 (누운자세 대비 선자세 delta 25/step)
    rew_scale_knee_clearance: float = 5.0   # new-v8: 무릎 높이 보상 (max 4×0.15×5=3/step)

    # === 기타 ===
    rew_scale_dof_pos_limits: float = -1.0
    rew_scale_contact_forces: float = -1e-3
    freeze_gait_phase: bool = True
    orientation_sigma: float = 0.10
    action_scale: float = 0.10
