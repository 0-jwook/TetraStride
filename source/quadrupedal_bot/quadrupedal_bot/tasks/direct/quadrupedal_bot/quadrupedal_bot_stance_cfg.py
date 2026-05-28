from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v32 — termination_height 상향으로 높이 sinking 해결.

    v31b 문제:
      1. body_height 0.181m → 0.164m으로 가라앉아 plateau (목표 0.177m 미달)
      2. stance4 ~73%에서 수렴 (entropy 급감으로 local optimum 도달)
      3. termination_height=0.155m이 너무 낮아 0.164m에서도 alive 보상 full 획득

    v32 수정:
      1. termination_height: 0.155 → 0.168 (현재 equilibrium 0.164m 위로 강제)
         → 로봇이 0.168m 이상 유지 안 하면 종료 → alive=8.0으로 강한 유지 동기
      2. rew_scale_body_height: 8.0 유지 (12.0은 높이 신호가 발 접지 학습 압도 → v32 실패)
      3. 중복 non_foot_contact 정의 버그 수정 (-8.0이 -30.0을 덮어쓰던 문제)

    v32b 수정 (v32 → v32b):
      - rew_scale_body_height: 12.0 → 8.0 복구 (v32에서 stance4 1.2%로 붕괴 확인)
      - termination_height=0.168m는 유지 (높이 강제 효과 확인됨)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.168  # v31:0.155 → v32:0.168 (sinking 방지)

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
    rew_scale_joint_default: float = -1.0
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
