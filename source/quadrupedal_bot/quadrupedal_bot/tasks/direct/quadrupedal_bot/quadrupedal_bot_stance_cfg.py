from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v34 — termination_height 대폭 완화 + 다리 뻗음 보상(FK).

    v33 실패 원인 분석:
      - ep_len=74 plateau (목표 1000 steps 미달)
      - 근본 원인: 낮은 자세(0.163m) = 안정 = 긴 에피소드 = 더 많은 alive 보상
        → 정책이 의도적으로 더 낮게 앉음 (누적 보상 최적화)
      - termination_height=0.150m이 자연평형 0.163m과 너무 가까워
        "낮게 앉는 것이 합리적 전략"이 되는 구조

    v34 방향: 보상 구조 근본 재설계
      - termination_height: 0.150 → 0.100 (버퍼 13mm → 63mm)
        → 낮게 앉아도 에피소드가 끝나지 않음 → "낮은 자세 = 이득" 구조 제거
      - rew_scale_body_height: 8.0 → 0.0 (root_pos_w 기반 제거)
      - rew_scale_leg_extension: +8.0 (FK 기반 4발 개별 Gaussian, 목표 0.177m)
        → 실제 다리가 얼마나 뻗었는지 직접 측정
      - spot_micro_cfg init_pos z: 0.22 → 0.18 (초기 낙하 44mm → 4mm)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.100  # v33:0.150 → v34:0.100 (보상 구조 근본 수정)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 8.0
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 12.0   # v30:6.0 → v31:12.0 (1,2,3,8 piecewise, 4발=12.0)

    # === 높이: root_pos_w 기반 제거, FK 다리 뻗음 보상으로 대체 ===
    target_body_height: float = 0.177
    rew_scale_body_height: float = 0.0  # v34: FK 보상으로 대체

    # === 다리 뻗음 보상 (v34 신규) ===
    # leg_ext = 0.1075×cos(leg) + 0.130×cos(leg+knee), 목표 0.177m
    # 4발 개별 Gaussian 합산 → max 4×1.0×scale = 4×2.0 = 8.0
    target_leg_extension: float = 0.177
    sigma_leg_extension: float = 0.02
    rew_scale_leg_extension: float = 2.0

    # === 무릎 서기 차단 강화 ===
    rew_scale_non_foot_contact: float = -30.0  # v30:-8.0 → v31:-30.0 (버그 수정: 중복 정의 제거)

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
    rew_scale_joint_default: float = -5.0  # v31b:-1.0 → v33:-5.0 (sinking 억제)
    rew_scale_foot_spread: float = 0.0        # v26:-3 → v27:0 (가라앉기 인센티브 제거)
    rew_scale_foot_side_span: float = 0.0     # v26:-3 → v27:0 (동일 이유)

    # === 비활성 ===
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0

    # === 기타 ===
    rew_scale_dof_pos_limits: float = -1.0
    rew_scale_contact_forces: float = -1e-3
    freeze_gait_phase: bool = True
    orientation_sigma: float = 0.10
    action_scale: float = 0.10
