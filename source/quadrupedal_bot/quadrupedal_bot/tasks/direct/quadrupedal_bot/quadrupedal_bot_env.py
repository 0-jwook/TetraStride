from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


class QuadrupedalBotEnv(DirectRLEnv):
    cfg: QuadrupedalBotEnvCfg

    def __init__(self, cfg: QuadrupedalBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._foot_ids, _ = self.contact_sensor.find_bodies(".*foot_link")
        # foot body IDs for position lookup (robot body array, not sensor array)
        self._foot_body_ids, _ = self.robot.find_bodies(".*foot_link")

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # trot gait phase: each env has its own phase, randomized at reset
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # advance trot gait clock at 1.5 Hz
        self._gait_phase = (self._gait_phase + self.step_dt * 2.0 * math.pi * 1.5) % (2.0 * math.pi)

    def _apply_action(self) -> None:
        target = self.robot.data.default_joint_pos + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(target)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,                      # [N, 3]
                self.robot.data.root_ang_vel_b,                      # [N, 3]
                self.robot.data.projected_gravity_b,                 # [N, 3]
                self._commands,                                      # [N, 3]
                self.joint_pos - self.robot.data.default_joint_pos,  # [N, 12]
                self.joint_vel,                                      # [N, 12]
                self._last_actions,                                  # [N, 12]
                torch.sin(self._gait_phase).unsqueeze(1),            # [N, 1]
                torch.cos(self._gait_phase).unsqueeze(1),            # [N, 1]
            ],
            dim=-1,
        )
        self._last_actions = self.actions.clone()
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.step_dt)[:, self._foot_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, self._foot_ids]

        base_rew = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_lin_vel_z,
            self.cfg.rew_scale_ang_vel_xy,
            self.cfg.rew_scale_gravity,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_torque,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_termination,
            self.cfg.rew_scale_air_time,
            self.cfg.rew_scale_movement,
            self._commands,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self.joint_vel,
            self.robot.data.applied_torque,
            self.actions,
            self._last_actions,
            self.reset_terminated,
            last_air_time,
            first_contact,
        )

        # trot gait reference reward: reward foot clearance during swing phase
        # foot_body_ids assumed ordered: FL=0, FR=1, RL=2, RR=3 (alphabetical)
        foot_heights = self.robot.data.body_pos_w[:, self._foot_body_ids, 2]  # [N, 4]
        phase_A = self._gait_phase                                  # FL(0) + RR(3) swing
        phase_B = (self._gait_phase + math.pi) % (2.0 * math.pi)  # FR(1) + RL(2) swing
        swing_A = (torch.sin(phase_A) > 0.0).float()
        swing_B = (torch.sin(phase_B) > 0.0).float()
        swing_mask = torch.stack([swing_A, swing_B, swing_B, swing_A], dim=1)
        # reward 0-8cm clearance above ground during swing phase
        foot_clearance = (foot_heights - 0.01).clamp(min=0.0, max=0.08)
        rew_gait = (foot_clearance * swing_mask).sum(dim=1) * self.cfg.rew_scale_gait

        return (base_rew + rew_gait).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Done / Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        body_fallen = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        body_tilted = self.robot.data.projected_gravity_b[:, 2] > -0.5

        terminated = body_fallen | body_tilted
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        self._commands[env_ids, 0] = torch.zeros(n, device=self.device).uniform_(
            *self.cfg.cmd_lin_vel_x_range
        )
        self._commands[env_ids, 1] = torch.zeros(n, device=self.device).uniform_(
            *self.cfg.cmd_lin_vel_y_range
        )
        self._commands[env_ids, 2] = torch.zeros(n, device=self.device).uniform_(
            *self.cfg.cmd_ang_vel_z_range
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos = joint_pos + torch.randn_like(joint_pos) * 0.1
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]
        # bootstrap: start at command velocity so robot must learn to maintain it
        root_state[:, 7] = self._commands[env_ids, 0]
        root_state[:, 8] = self._commands[env_ids, 1]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self._last_actions[env_ids] = 0.0
        # randomize gait phase at reset to break synchronization
        self._gait_phase[env_ids] = torch.zeros(n, device=self.device).uniform_(0.0, 2.0 * math.pi)


# ------------------------------------------------------------------
# Reward function (torch.jit.script for speed)
# ------------------------------------------------------------------


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_lin_vel_z: float,
    rew_scale_ang_vel_xy: float,
    rew_scale_gravity: float,
    rew_scale_joint_vel: float,
    rew_scale_torque: float,
    rew_scale_action_rate: float,
    rew_scale_termination: float,
    rew_scale_air_time: float,
    rew_scale_movement: float,
    commands: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    last_air_time: torch.Tensor,
    first_contact: torch.Tensor,
) -> torch.Tensor:
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # exponential tracking rewards (sigma=0.25: nonzero gradient even when far from cmd)
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    rew_lin_vel = torch.exp(-lin_vel_error / 0.25) * rew_scale_lin_vel

    ang_vel_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_ang_vel = torch.exp(-ang_vel_error / 0.25) * rew_scale_ang_vel

    # movement bonus: proportional reward for any velocity along command direction
    cmd_dir = commands[:, :2] / (torch.norm(commands[:, :2], dim=1, keepdim=True).clamp(min=0.1))
    vel_proj = (root_lin_vel_b[:, :2] * cmd_dir).sum(dim=1).clamp(min=0.0, max=2.0)
    rew_movement = vel_proj * rew_scale_movement

    rew_lin_vel_z = torch.square(root_lin_vel_b[:, 2]) * rew_scale_lin_vel_z
    rew_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1) * rew_scale_ang_vel_xy
    rew_gravity = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * rew_scale_gravity
    rew_joint_vel = torch.sum(torch.square(joint_vel), dim=1) * rew_scale_joint_vel
    rew_torque = torch.sum(torch.square(applied_torque), dim=1) * rew_scale_torque
    rew_action_rate = torch.sum(torch.square(actions - last_actions), dim=1) * rew_scale_action_rate
    rew_termination = reset_terminated.float() * rew_scale_termination

    cmd_has_vel = (torch.norm(commands[:, :2], dim=1) > 0.1).float()
    rew_air_time = torch.sum(
        (last_air_time - 0.1).clamp(min=0.0) * first_contact.float(), dim=1
    ) * rew_scale_air_time * cmd_has_vel

    total = (
        rew_alive
        + rew_lin_vel
        + rew_ang_vel
        + rew_movement
        + rew_lin_vel_z
        + rew_ang_vel_xy
        + rew_gravity
        + rew_joint_vel
        + rew_torque
        + rew_action_rate
        + rew_termination
        + rew_air_time
    )
    return total.clamp(min=0.0)
