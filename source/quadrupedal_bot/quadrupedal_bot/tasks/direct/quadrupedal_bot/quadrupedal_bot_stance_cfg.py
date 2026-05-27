from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v23 — 보상 스케일 균형 재설계 (400배 차이 제거, 보상 해킹 차단).

    v22 문제:
      - body_height=-200, knee_height=-200 vs alive=0.5 → 400배 차이로 gradient 불안정
      - lin_vel_xy=-50 → 0.1m/s 이동시 alive 보상보다 크게 패널티, 미세 균형조정 불가
      - base_drift=-15 (clamp 3m) → lin_vel_xy와 중복

    v23 수정:
      - 모든 보상을 10-20배 이내로 정규화
      - non_foot_contact 강화 (-2→-8): 무릎접지 주요 원인
      - orientation_sigma 완화 (0.04→0.10): 더 넓은 gradient 범위
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.155

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 (최대 ~10.5/step) ===
    rew_scale_alive: float = 0.5
    rew_scale_upright: float = 6.0         # 직립 (핵심 목표)
    rew_scale_foot_contact: float = 4.0    # 4발 접지

    # === 높이 패널티 (0.155m 적자: -0.24/step, 균형 범위) ===
    target_body_height: float = 0.163
    rew_scale_body_height: float = -30.0   # v22:-200 → v23:-30

    # === 자세 패널티 ===
    rew_scale_non_foot_contact: float = -8.0      # v22:-2 → v23:-8, 무릎접지 강력 억제
    rew_scale_knee_height_stance: float = -30.0   # v22:-200 → v23:-30
    knee_stance_height_threshold: float = 0.06
    rew_scale_knee_angle: float = -3.0

    # === 안정화 패널티 ===
    rew_scale_gravity: float = -5.0        # v22:-10 → v23:-5, upright=6과 균형
    rew_scale_ang_vel_xy: float = -0.5     # tilt 속도 억제
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_z: float = -0.5
    rew_scale_lin_vel_xy: float = -8.0     # v22:-50 → v23:-8, 미세조정 허용
    rew_scale_base_drift: float = 0.0      # v22:-15 → 제거 (lin_vel_xy와 중복)

    # === 동작 품질 ===
    rew_scale_action_rate: float = -0.05
    rew_scale_foot_slip: float = -3.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5

    # === 형상 패널티 (스케일 축소) ===
    rew_scale_shoulder_default: float = -8.0   # v22:-15 → v23:-8
    rew_scale_joint_default: float = -1.0
    rew_scale_foot_spread: float = -3.0        # v22:-8 → v23:-3
    rew_scale_foot_side_span: float = -3.0     # v22:-8 → v23:-3

    # === 비활성 ===
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0
    rew_scale_stand_still: float = 0.0

    # === 기타 ===
    rew_scale_dof_pos_limits: float = -1.0
    rew_scale_contact_forces: float = -1e-3
    freeze_gait_phase: bool = True
    orientation_sigma: float = 0.10   # v22:0.04 → v23:0.10, gradient 범위 확대
    action_scale: float = 0.10
