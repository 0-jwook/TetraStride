from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v33 — termination_height 복구 + joint_default 강화로 sinking 해결.

    v32/v32b 실패 분석:
      - termination_height=0.168m > 자연 평형 0.164m
      - 에피소드 평균 1.37초(68 steps)만에 조기종료 → 사실상 99% 실패
      - term_ratio 1.6%는 스텝당 비율, 실제 에피소드 조기종료율은 ~99%

    v33 방향: termination_height를 자연 평형 아래로 낮춰 에피소드 길이 복구
      - termination_height: 0.168 → 0.150 (v31b 0.155보다도 낮춰 안전 마진 확보)
      - rew_scale_joint_default: -1.0 → -5.0 (관절 default 유지 강화 → sinking 억제)
      - rew_scale_shoulder_default: -8.0 유지
      - 나머지 v31b 설계 유지 (발 접지, 무릎 패널티 등)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.150  # v32b:0.168 → v33:0.150 (에피소드 복구)

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 8.0
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 12.0   # v30:6.0 → v31:12.0 (1,2,3,8 piecewise, 4발=12.0)

    # === 높이 보상 (v31b 동일, v32에서 12.0 시도했으나 실패) ===
    target_body_height: float = 0.177
    rew_scale_body_height: float = 8.0  # v32:12.0 → v32b:8.0 복구

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
