from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotStanceCfg(QuadrupedalBotEnvCfg):
    """Stage 1: 서기 학습 v22 — scratch 재학습: termination↑0.155 + body_height_penalty↑200 + entropy↑0.05 (낙하 전략 차단)."""

    episode_length_s: float = 20.0

    termination_height: float = 0.155

    cmd_lin_vel_x_range: tuple = (0.0, 0.0)
    cmd_lin_vel_y_range: tuple = (0.0, 0.0)
    cmd_ang_vel_z_range: tuple = (0.0, 0.0)

    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.3          # v14:-0.1 → v19:-0.3, tilt 속도 억제
    rew_scale_gravity: float = -10.0            # v14:-5 → v19:-10, 기울기 패널티 2배
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.05        # v21: -0.01→-0.05, 관절 속도 유발 동작 억제
    rew_scale_air_time: float = 0.0
    rew_scale_movement: float = 0.0
    rew_scale_gait: float = 0.0
    target_body_height: float = 0.163
    rew_scale_body_height: float = -200.0       # v22: -60→-200, termination=0.155이면 0.155→0.163 결핍시 최대 1.6/step 패널티
    rew_scale_non_foot_contact: float = -2.0
    rew_scale_lin_vel_xy: float = -50.0         # v14 값 복원
    rew_scale_ang_vel_z: float = -0.3
    rew_scale_upright: float = 5.0              # v14:2 → v19:5, 직립 보상 2.5배 (8은 보행유발)
    rew_scale_foot_spread: float = -8.0
    rew_scale_foot_slip: float = -3.0
    rew_scale_stand_still: float = 0.0
    rew_scale_base_drift: float = -15.0         # v14 값 복원
    freeze_gait_phase: bool = True
    rew_scale_dof_pos_limits: float = -1.0
    rew_scale_contact_forces: float = -1e-3
    rew_scale_foot_contact: float = 5.0         # v14 값 복원
    rew_scale_knee_height_stance: float = -200.0
    knee_stance_height_threshold: float = 0.06
    rew_scale_knee_angle: float = -2.0
    rew_scale_shoulder_default: float = -15.0
    rew_scale_foot_side_span: float = -8.0
    rew_scale_joint_default: float = -2.0       # v14 값 복원 (drift 유발 없이 기본자세 유지)
    action_scale: float = 0.10               # v21: 0.25→0.10, 스텝당 최대 관절 변화 60% 감소
