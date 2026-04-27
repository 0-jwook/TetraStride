from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .spot_micro_cfg import SPOT_MICRO_CFG


@configclass
class QuadrupedalBotEnvCfg(DirectRLEnvCfg):
    # --- env timing ---
    decimation: int = 4
    episode_length_s: float = 10.0

    # --- spaces ---
    # obs: lin_vel(3) + ang_vel(3) + proj_gravity(3) + commands(3)
    #      + joint_pos_rel(12) + joint_vel(12) + last_actions(12) + gait_phase(2) = 50
    action_space: int = 12
    observation_space: int = 50
    state_space: int = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # --- robot ---
    robot_cfg: ArticulationCfg = SPOT_MICRO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- contact sensor ---
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # --- action ---
    action_scale: float = 0.25          # Rudin 표준, 과도한 관절 이동 방지
    action_smoothing: float = 0.8       # EMA alpha: high-frequency jitter 억제 (Margolis 2022)

    # --- velocity commands ---
    cmd_lin_vel_x_range: tuple = (0.2, 0.6)
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)

    # --- termination ---
    termination_height: float = 0.05    # 완전 쓰러질 때만 종료 (Rudin 표준)
    target_body_height: float = 0.18    # 목표 서기 높이 (패널티 기준)

    # --- foot geometry targets ---
    target_foot_span_y: float = 0.18   # 발 좌우 간격 최소 기준
    target_foot_span_x: float = 0.20   # 발 앞뒤 간격 최소 기준
    target_foot_clearance: float = 0.06  # swing foot 목표 높이 (6cm)

    # --- reward scales ---
    rew_scale_alive: float = 0.2
    rew_scale_lin_vel: float = 1.0       # Rudin 표준
    rew_scale_ang_vel: float = 0.5       # Rudin 표준
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -1.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -2.0e-5    # Rudin 표준
    rew_scale_action_rate: float = -0.05  # -0.01→-0.05: 진동 억제 강화
    rew_scale_termination: float = -200.0  # Rudin 표준: 조기 종료 강한 패널티
    rew_scale_air_time: float = 1.0      # Rudin 표준 (threshold 0.4s로 올림)
    rew_scale_movement: float = 2.0
    rew_scale_gait: float = 2.0          # trot gait schedule
    rew_scale_body_height: float = -3.0
    rew_scale_non_foot_contact: float = -1.0
    rew_scale_lin_vel_xy: float = 0.0    # Stance에서만 활성화
    rew_scale_ang_vel_z: float = 0.0     # Stance에서만 활성화
    rew_scale_joint_default: float = 0.0
    rew_scale_upright: float = 0.3
    rew_scale_foot_spread: float = 0.0   # Stance에서 활성화
    rew_scale_dof_acc: float = -2.5e-7   # 관절 진동 억제 (Rudin 2021)
    rew_scale_action_acc: float = -0.005  # action 2차 미분 패널티 (jitter 억제)
    rew_scale_foot_slip: float = -0.05   # 발 미끄러짐 패널티 (Margolis 2022)
    rew_scale_no_air: float = 0.0        # 4발 전부 접지 패널티 (Trot에서 활성화)
    rew_scale_foot_clearance: float = 0.0  # swing foot 높이 패널티 (Trot에서 활성화)
    rew_scale_stand_still: float = 0.0   # cmd=0일 때 관절 이탈 패널티 (Stance에서 활성화)
