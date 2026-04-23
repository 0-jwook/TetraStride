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

    # --- contact sensor (all bodies; foot IDs extracted at runtime) ---
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # --- action ---
    action_scale: float = 0.5

    # --- velocity commands ---
    cmd_lin_vel_x_range: tuple = (0.5, 1.0)
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)

    # --- termination ---
    termination_height: float = 0.08

    # --- reward scales (legged_gym 기반) ---
    rew_scale_alive: float = 0.0       # legged_gym: alive reward 없음
    rew_scale_lin_vel: float = 3.0     # exp(-e²/0.25), increased for stronger tracking signal
    rew_scale_ang_vel: float = 0.1     # reduced: prevent angular-vel standing optimum
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -1.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = 0.0  # legged_gym: -0.0 (패널티 없음, 핵심!)
    rew_scale_air_time: float = 6.0     # strongly incentivize leg lifting
    rew_scale_movement: float = 2.0     # linear gradient: reward any cmd-direction velocity
    rew_scale_gait: float = 5.0         # trot gait reference: reward foot clearance during swing phase
