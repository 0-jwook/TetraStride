from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v31 — 무릎 서기 차단 + 1,2,3,8 발접지 보상.

    v30 문제:
      1. 이진 foot_contact: 4발 동시 달성이 어려워 아예 포기
      2. body_height(8)+alive(8)=16 > foot_contact(6): 무릎으로 서기가 최적 전략
         → 시각화 확인: 로봇이 무릎을 디디고 발은 공중에 뜬 상태
      3. non_foot_contact=-8.0: 무릎 패널티보다 body_height 보상이 커서 돌파

    v31 수정:
      1. foot_contact: 이진→1,2,3,8 piecewise (env.py 수정)
         → 0발=0, 1발=1.5, 2발=3.0, 3발=4.5, 4발=12.0
         → 부분 접지도 보상 → 점진적으로 4발 학습 유도
      2. rew_scale_foot_contact: 6→12 (4발 시 12.0, alive+body_height 16 대응)
      3. non_foot_contact: -8→-30 (무릎 서기 강력 차단)
    """

    episode_length_s: float = 20.0

    termination_height: float = 0.155

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    # === 주요 양의 보상 ===
    rew_scale_alive: float = 8.0
    rew_scale_upright: float = 6.0
    rew_scale_foot_contact: float = 12.0   # v30:6.0 → v31:12.0 (1,2,3,8 piecewise, 4발=12.0)

    # === 높이 보상 유지 ===
    target_body_height: float = 0.177
    rew_scale_body_height: float = 8.0

    # === 무릎 서기 차단 강화 ===
    rew_scale_non_foot_contact: float = -30.0  # v30:-8.0 → v31:-30.0

    # === stand_still 비활성 유지 ===
    rew_scale_stand_still: float = 0.0

    # === 자세 패널티 ===
    rew_scale_non_foot_contact: float = -8.0
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
