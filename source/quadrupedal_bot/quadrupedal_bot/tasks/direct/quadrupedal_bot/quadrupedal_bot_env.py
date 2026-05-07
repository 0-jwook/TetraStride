from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


class QuadrupedalBotEnv(DirectRLEnv):
    cfg: QuadrupedalBotEnvCfg

    def __init__(self, cfg: QuadrupedalBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._foot_ids, _ = self.contact_sensor.find_bodies(".*foot_link")
        self._foot_body_ids_robot, _ = self.robot.find_bodies(".*foot_link")
        self._shoulder_ids, _ = self.robot.find_joints(".*_shoulder")
        # All non-foot body IDs for knee/belly contact penalty
        all_body_ids, _ = self.contact_sensor.find_bodies(".*")
        foot_id_set = set(int(i) for i in self._foot_ids)
        self._non_foot_contact_ids = torch.tensor(
            [i for i in all_body_ids if i not in foot_id_set],
            device=self.device, dtype=torch.long,
        )

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._last_joint_vel = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
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
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
        )
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
        self._processed_actions = (
            self.cfg.action_smoothing * self.actions
            + (1.0 - self.cfg.action_smoothing) * self._processed_actions
        )
        if not self.cfg.freeze_gait_phase:
            self._gait_phase = (self._gait_phase + self.step_dt * 2.0 * math.pi * 1.5) % (2.0 * math.pi)

    def _apply_action(self) -> None:
        target = self.robot.data.default_joint_pos + self._processed_actions * self.cfg.action_scale
        self.robot.set_joint_position_target(target)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,                      # [N, 3] 속도 피드백 추가
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
            self.cfg.air_time_threshold,
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

        # spinning 억제 (yaw angular velocity penalty)
        rew_ang_vel_z = torch.square(self.robot.data.root_ang_vel_b[:, 2]) * self.cfg.rew_scale_ang_vel_z
        rew_lin_vel_xy = torch.sum(torch.square(self.robot.data.root_lin_vel_b[:, :2]), dim=1) * self.cfg.rew_scale_lin_vel_xy

        # gait clock reward 제거 (Rudin 2022 방식: air_time만으로 trot 자연 발생)
        # gait_phase는 observation hint로만 유지, reward 강제는 비대칭 정책 유발
        rew_gait = torch.zeros(self.num_envs, device=self.device) * self.cfg.rew_scale_gait

        # air_time_variance 패널티: 4발 air_time 불균형 시 패널티 → 한 다리만 움직이는 비대칭 차단
        air_time_var = torch.var(self.contact_sensor.data.last_air_time[:, self._foot_ids], dim=1)
        rew_air_time_var = -air_time_var * self.cfg.rew_scale_air_time_var

        # Body height reward: Gaussian centered at target — closer = more reward
        # sigma=0.05: at 5cm error → exp(-1.0)=0.37, at 10cm → exp(-2.0)=0.14
        body_height = self.robot.data.root_pos_w[:, 2]
        height_error = (body_height - self.cfg.target_body_height).abs()
        rew_body_height = torch.exp(-height_error / 0.05) * self.cfg.rew_scale_body_height

        # Non-foot contact penalty: penalize knee/belly touching ground
        # threshold=20N: 정상 서기 시 정강이 스침(~1-5N)은 무시, 실제 무게를 싣는 무릎보행(>20N)만 감지
        non_foot_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self._non_foot_contact_ids, :]
        non_foot_contact = (torch.norm(non_foot_forces, dim=-1) > 20.0).float()
        rew_non_foot_contact = non_foot_contact.sum(dim=1) * self.cfg.rew_scale_non_foot_contact

        # 어깨 관절 패널티: dead zone 0.05 rad으로 좁혀 도마뱀 자세 방지
        shoulder_dev = torch.abs(
            self.joint_pos[:, self._shoulder_ids] - self.robot.data.default_joint_pos[:, self._shoulder_ids]
        )
        shoulder_excess = (shoulder_dev - 0.05).clamp(min=0.0)
        rew_joint_default = torch.sum(torch.square(shoulder_excess), dim=1) * self.cfg.rew_scale_joint_default

        # IMU 직립 보상: 수평 유지할수록 급격히 증가, 조금만 기울어도 급감
        # tilt = gx²+gy² (0=직립, 1=완전 넘어짐), sigma=0.04 → 10° 이상이면 보상 거의 0
        tilt = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        rew_upright = torch.exp(-tilt / 0.04) * self.cfg.rew_scale_upright

        # Foot spread penalty: 양방향 — 너무 모이거나 너무 벌어지는 것 모두 패널티
        foot_pos_world = self.robot.data.body_pos_w[:, self._foot_body_ids_robot, :]  # [N, 4, 3]
        foot_y_world = foot_pos_world[:, :, 1]  # [N, 4] lateral (Y) positions
        foot_span = foot_y_world.max(dim=1).values - foot_y_world.min(dim=1).values  # [N]
        span_error = torch.abs(foot_span - self.cfg.target_foot_span)  # 목표 간격에서 벗어난 정도
        rew_foot_spread = span_error * self.cfg.rew_scale_foot_spread  # [N]

        # Foot slip penalty: contact foot moving laterally = sliding (Margolis 2022)
        foot_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._foot_body_ids_robot, :2]  # [N, 4, 2]
        foot_force_z = self.contact_sensor.data.net_forces_w_history[:, 0, self._foot_ids, 2]  # [N, 4]
        foot_in_contact = (torch.abs(foot_force_z) > 1.0).float()
        foot_slip_speed = torch.norm(foot_lin_vel_w, dim=-1) * foot_in_contact
        rew_foot_slip = torch.sum(torch.square(foot_slip_speed), dim=1) * self.cfg.rew_scale_foot_slip

        # DOF acceleration penalty: penalize motor vibration (Rudin 2021)
        dof_acc = (self.joint_vel - self._last_joint_vel) / self.step_dt
        rew_dof_acc = torch.sum(torch.square(dof_acc), dim=1) * self.cfg.rew_scale_dof_acc
        self._last_joint_vel = self.joint_vel.clone()

        # DOF position limits penalty (legged_gym: _reward_dof_pos_limits)
        # soft_joint_pos_limits: [1, 12, 2] — 0.9x of hard limits via soft_joint_pos_limit_factor=0.9
        soft_lower = self.robot.data.soft_joint_pos_limits[:, :, 0]
        soft_upper = self.robot.data.soft_joint_pos_limits[:, :, 1]
        out_of_lower = (soft_lower - self.joint_pos).clamp(min=0.0)
        out_of_upper = (self.joint_pos - soft_upper).clamp(min=0.0)
        rew_dof_pos_limits = (out_of_lower + out_of_upper).sum(dim=1) * self.cfg.rew_scale_dof_pos_limits

        # Foot contact force penalty (legged_gym: _reward_feet_contact_forces)
        # Penalize excessive foot impact forces above threshold to protect servos and reduce bounce
        foot_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self._foot_ids, :]
        foot_force_mag = torch.norm(foot_forces, dim=-1)  # [N, 4]
        excess_force = (foot_force_mag - self.cfg.max_foot_contact_force).clamp(min=0.0)
        rew_contact_forces = excess_force.sum(dim=1) * self.cfg.rew_scale_contact_forces

        # Stand still penalty: penalize joint deviation from default when cmd ≈ 0 (legged_gym standard)
        cmd_zero = (torch.norm(self._commands[:, :2], dim=1) < 0.1).float()
        joint_dev = torch.sum(torch.abs(self.joint_pos - self.robot.data.default_joint_pos), dim=1)
        rew_stand_still = joint_dev * cmd_zero * self.cfg.rew_scale_stand_still

        with torch.no_grad():
            rew_alive_log = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
            rew_gravity_log = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1) * self.cfg.rew_scale_gravity
            rew_termination_log = self.reset_terminated.float() * self.cfg.rew_scale_termination
            # lin_vel / ang_vel rewards (for logging)
            _lin_vel_err = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
            rew_lin_vel_log = torch.exp(-_lin_vel_err / 0.25) * self.cfg.rew_scale_lin_vel
            _ang_vel_err = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            rew_ang_vel_log = torch.exp(-_ang_vel_err / 0.25) * self.cfg.rew_scale_ang_vel
            # air_time reward (for logging)
            _cmd_has_vel = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()
            _air_time_log = torch.sum(
                (last_air_time - self.cfg.air_time_threshold).clamp(min=0.0) * first_contact.float(), dim=1
            ) * self.cfg.rew_scale_air_time * _cmd_has_vel
            per_step_net = (rew_alive_log + rew_upright + rew_gravity_log + rew_foot_slip
                            + rew_joint_default + rew_foot_spread + rew_stand_still + rew_dof_acc
                            + rew_dof_pos_limits + rew_contact_forces
                            + rew_lin_vel_log + rew_ang_vel_log + _air_time_log)
            # torque saturation: fraction of joints outputting ≥ 95% of effort_limit
            effort_limit = 10.0  # N·m, leg/foot effort_limit
            torque_sat_ratio = (self.robot.data.applied_torque.abs() >= effort_limit * 0.95).float().mean()
            # termination cause breakdown
            body_fallen_now = (self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height).float()
            body_tilted_now = (self.robot.data.projected_gravity_b[:, 2] > 0.0).float()
            self.extras["log"] = {
                "rew/alive": rew_alive_log.mean().item(),
                "rew/lin_vel": rew_lin_vel_log.mean().item(),
                "rew/ang_vel": rew_ang_vel_log.mean().item(),
                "rew/air_time": _air_time_log.mean().item(),
                "rew/air_time_var": rew_air_time_var.mean().item(),
                "rew/upright": rew_upright.mean().item(),
                "rew/gravity": rew_gravity_log.mean().item(),
                "rew/foot_slip": rew_foot_slip.mean().item(),
                "rew/joint_default": rew_joint_default.mean().item(),
                "rew/foot_spread": rew_foot_spread.mean().item(),
                "rew/stand_still": rew_stand_still.mean().item(),
                "rew/dof_acc": rew_dof_acc.mean().item(),
                "rew/dof_pos_limits": rew_dof_pos_limits.mean().item(),
                "rew/contact_forces": rew_contact_forces.mean().item(),
                "rew/body_height": rew_body_height.mean().item(),
                "rew/ang_vel_z": rew_ang_vel_z.mean().item(),
                "diag/body_height_mean": self.robot.data.root_pos_w[:, 2].mean().item(),
                "diag/body_height_min": self.robot.data.root_pos_w[:, 2].min().item(),
                "diag/torque_sat_ratio": torque_sat_ratio.item(),
                "diag/term_height_ratio": body_fallen_now.mean().item(),
                "diag/term_tilt_ratio": body_tilted_now.mean().item(),
                "rew/termination": rew_termination_log.mean().item(),
                "diag/per_step_net": per_step_net.mean().item(),
                "diag/term_ratio": self.reset_terminated.float().mean().item(),
                "diag/foot_span_mean": foot_span.mean().item(),
                "diag/actual_lin_vel_x": self.robot.data.root_lin_vel_b[:, 0].mean().item(),
                "diag/actual_ang_vel_z": self.robot.data.root_ang_vel_b[:, 2].mean().item(),
            }

        return (base_rew + rew_gait + rew_body_height + rew_non_foot_contact + rew_joint_default
                + rew_upright + rew_ang_vel_z + rew_lin_vel_xy + rew_foot_spread + rew_foot_slip
                + rew_dof_acc + rew_stand_still + rew_dof_pos_limits + rew_contact_forces
                + rew_air_time_var)

    # ------------------------------------------------------------------
    # Done / Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        body_fallen = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        body_tilted = self.robot.data.projected_gravity_b[:, 2] > 0.0

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
        # rel_standing_envs: zero_command_prob 비율의 env는 cmd=(0,0,0) 강제 (제자리 서기)
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._commands[env_ids[zero_mask]] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos = joint_pos + torch.randn_like(joint_pos) * self.cfg.init_noise_scale
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self._last_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._last_joint_vel[env_ids] = 0.0
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
    air_time_threshold: float,
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
        (last_air_time - air_time_threshold).clamp(min=0.0) * first_contact.float(), dim=1
    ) * rew_scale_air_time * cmd_has_vel

    living = (
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
        + rew_air_time
    )
    return living + rew_termination
