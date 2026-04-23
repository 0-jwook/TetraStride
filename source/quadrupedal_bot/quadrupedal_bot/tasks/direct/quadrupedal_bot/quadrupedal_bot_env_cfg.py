from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .spot_micro_cfg import SPOT_MICRO_CFG


@configclass
class QuadrupedalBotEnvCfg(DirectRLEnvCfg):
    # --- env timing ---
    # physics at 200 Hz, policy at 50 Hz (decimation=4)
    decimation: int = 4
    episode_length_s: float = 20.0

    # --- spaces ---
    # obs: lin_vel(3) + ang_vel(3) + proj_gravity(3) + commands(3)
    #      + joint_pos_rel(12) + joint_vel(12) + last_actions(12) = 48
    action_space: int = 12
    observation_space: int = 48
    state_space: int = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # --- robot ---
    robot_cfg: ArticulationCfg = SPOT_MICRO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # --- action ---
    # actions are joint position offsets (rad) added to default joint positions
    action_scale: float = 0.5

    # --- velocity commands sampled each episode ---
    cmd_lin_vel_x_range: tuple = (0.3, 0.8)   # forward velocity [m/s] — min 0.3 to force walking
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)  # lateral velocity [m/s]
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)  # yaw rate [rad/s]

    # --- termination ---
    termination_height: float = 0.08  # body z below this → episode ends [m]

    # --- reward scales (positive = reward, negative = penalty) ---
    rew_scale_alive: float = 0.5          # small reward for staying upright each step
    rew_scale_lin_vel: float = 4.0        # track commanded xy velocity (raised from 2.0)
    rew_scale_ang_vel: float = 1.0        # track commanded yaw rate
    rew_scale_lin_vel_z: float = -2.0     # penalize vertical body velocity
    rew_scale_ang_vel_xy: float = -0.05   # penalize roll/pitch angular rate
    rew_scale_gravity: float = -1.0       # penalize body tilt (raised from -0.5)
    rew_scale_joint_vel: float = -1e-4    # penalize fast joints
    rew_scale_torque: float = -1e-5       # penalize large torques
    rew_scale_action_rate: float = -0.005 # penalize jerky actions (lowered to allow exploration)
    rew_scale_termination: float = -20.0  # stronger fall penalty (raised from -10.0)
