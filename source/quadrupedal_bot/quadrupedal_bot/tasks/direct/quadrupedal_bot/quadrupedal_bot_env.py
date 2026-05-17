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
from isaaclab.utils.math import quat_apply

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


class QuadrupedalBotEnv(DirectRLEnv):
    cfg: QuadrupedalBotEnvCfg

    def __init__(self, cfg: QuadrupedalBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._foot_ids, _ = self.contact_sensor.find_bodies(".*foot_link")
        self._foot_body_ids_robot, _ = self.robot.find_bodies(".*foot_link")
        self._shoulder_ids, _ = self.robot.find_joints(".*_shoulder")
        self._knee_ids, _ = self.robot.find_joints(".*_foot")   # URDF "foot" joint = knee joint
        # All non-foot body IDs for knee/belly contact penalty
        all_body_ids, _ = self.contact_sensor.find_bodies(".*")
        foot_id_set = set(int(i) for i in self._foot_ids)
        self._non_foot_contact_ids = torch.tensor(
            [i for i in all_body_ids if i not in foot_id_set],
            device=self.device, dtype=torch.long,
        )

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._last_last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._last_joint_vel = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # trot gait phase: each env has its own phase, randomized at reset
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._push_step = 0
        # target heading: 에피소드 시작 시 heading 저장 후 yaw 명령 누적 (올바른 heading 추적)
        self._target_heading = torch.zeros(self.num_envs, device=self.device)
        # heading error cache: computed in _get_observations, used in _get_rewards
        self._heading_err = torch.zeros(self.num_envs, device=self.device)
        # world-frame Y position at episode start (lateral drift tracking)
        self._start_pos_y = torch.zeros(self.num_envs, device=self.device)

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
            self._gait_phase = (self._gait_phase + self.step_dt * 2.0 * math.pi * self.cfg.gait_freq_hz) % (2.0 * math.pi)

        # Random push perturbation for domain randomization
        if self.cfg.push_interval_s > 0.0:
            self._push_step += 1
            push_interval = max(1, int(self.cfg.push_interval_s / self.step_dt))
            if self._push_step % push_interval == 0:
                push = torch.zeros(self.num_envs, 6, device=self.device)
                push[:, :2] = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * self.cfg.max_push_vel
                cur_vel = torch.cat([self.robot.data.root_lin_vel_b, self.robot.data.root_ang_vel_b], dim=1)
                self.robot.write_root_velocity_to_sim(cur_vel + push, self.robot._ALL_INDICES)

    def _apply_action(self) -> None:
        target = self.robot.data.default_joint_pos + self._processed_actions * self.cfg.action_scale
        self.robot.set_joint_position_target(target)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Update target heading with yaw command accumulation (step dt integration)
        self._target_heading = self._target_heading + self._commands[:, 2] * self.step_dt
        # Compute current heading from quaternion
        _fwd_local = torch.zeros(self.num_envs, 3, device=self.device)
        _fwd_local[:, 0] = 1.0
        _fwd_world = quat_apply(self.robot.data.root_quat_w, _fwd_local)
        _heading = torch.atan2(_fwd_world[:, 1], _fwd_world[:, 0])
        # Angular difference wrapped to [-π, π]
        self._heading_err = torch.atan2(
            torch.sin(_heading - self._target_heading),
            torch.cos(_heading - self._target_heading),
        )

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
                torch.sin(self._heading_err).unsqueeze(1),           # [N, 1] heading error sin
                torch.cos(self._heading_err).unsqueeze(1),           # [N, 1] heading error cos
            ],
            dim=-1,
        )  # total: 3+3+3+3+12+12+12+1+1+1+1 = 52
        self._last_last_actions = self._last_actions.clone()
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

        # yaw 추적: legged_gym 방식 exp 보상 (패널티 단독 → 보상+패널티 병행)
        yaw_error_sq = torch.square(self.robot.data.root_ang_vel_b[:, 2] - self._commands[:, 2])
        rew_ang_vel_z = yaw_error_sq * self.cfg.rew_scale_ang_vel_z  # 패널티 유지
        rew_yaw_tracking = torch.exp(-yaw_error_sq / 0.25) * self.cfg.rew_scale_yaw_tracking  # exp 보상 추가

        # Heading tracking reward: _heading_err already computed in _get_observations()
        # target_heading also already updated there — do NOT update again here
        _cmd_vel_gate = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()
        rew_heading = torch.exp(-torch.square(self._heading_err) / self.cfg.heading_sigma) * self.cfg.rew_scale_heading * _cmd_vel_gate
        rew_lin_vel_xy = torch.square(self.robot.data.root_lin_vel_b[:, 1]) * self.cfg.rew_scale_lin_vel_xy

        # 세계 좌표 Y축 누적 drift 패널티 (직진 명령 env에서만 활성, heading_err 게이밍 우회)
        lateral_drift = (self.robot.data.root_pos_w[:, 1] - self._start_pos_y).abs()
        straight_gate = (torch.abs(self._commands[:, 1]) < 0.05).float()
        rew_pos_drift = lateral_drift.clamp(max=2.0) * self.cfg.rew_scale_pos_drift * straight_gate * _cmd_vel_gate

        # 선형속도 추적 오차 패널티: cmd 대비 부족한 속도를 직접 패널티 (서기 로컬옵티멈 탈출)
        lin_vel_error_sq = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rew_lin_vel_penalty = lin_vel_error_sq * self.cfg.rew_scale_lin_vel_penalty

        # 선형 heading 오차 패널티 (exp 포화 구간 보완: 1~5° 소오차에서 gradient 확보)
        rew_heading_linear = torch.abs(self._heading_err) * self.cfg.rew_scale_heading_linear * _cmd_vel_gate
        # 선형 yaw rate 오차 패널티
        rew_yaw_rate_error = torch.abs(self.robot.data.root_ang_vel_b[:, 2] - self._commands[:, 2]) * self.cfg.rew_scale_yaw_rate_error

        # ── Gait phase: stance/swing 마스크 (gait/swing_contact/foot_height 공통 사용) ──
        # 발 순서: FL=0, FR=1, RL=2, RR=3 (find_bodies 알파벳순)
        cos_phase = torch.cos(self._gait_phase)
        pair_a = torch.tensor([1.0, 0.0, 0.0, 1.0], device=self.device)  # FL, RR stance
        pair_b = torch.tensor([0.0, 1.0, 1.0, 0.0], device=self.device)  # FR, RL stance
        target_a = (cos_phase < 0).float().unsqueeze(1) * pair_a.unsqueeze(0)
        target_b = (cos_phase >= 0).float().unsqueeze(1) * pair_b.unsqueeze(0)
        contact_target = target_a + target_b          # [N, 4], 1=should be on ground (stance)
        swing_mask = 1.0 - contact_target             # [N, 4], 1=should be in air (swing)
        foot_forces_z = self.contact_sensor.data.net_forces_w_history[:, 0, self._foot_ids, 2]
        contact_actual = (foot_forces_z.abs() > 1.0).float()
        cmd_has_vel_gate = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()

        # Gait clock reward
        if self.cfg.rew_scale_gait != 0.0:
            contact_error = torch.abs(contact_actual - contact_target).sum(dim=1)
            rew_gait = (4.0 - contact_error) * self.cfg.rew_scale_gait * cmd_has_vel_gate
        else:
            rew_gait = torch.zeros(self.num_envs, device=self.device)

        # Diagonal pair contact reward: FL+RR 동시, FR+RL 동시 보상 (front-back 가짜 trot 차단)
        # 발 순서: FL=0, FR=1, RL=2, RR=3
        fl = contact_actual[:, 0]
        fr = contact_actual[:, 1]
        rl = contact_actual[:, 2]
        rr = contact_actual[:, 3]
        pair_a_active = (cos_phase < 0).float()   # FL+RR stance 구간
        pair_b_active = (cos_phase >= 0).float()  # FR+RL stance 구간
        # 대각 쌍이 동시에 올바른 상태(둘 다 stance or 둘 다 swing)일 때 보상
        fl_rr_pair = pair_a_active * fl * rr + pair_b_active * (1 - fl) * (1 - rr)
        fr_rl_pair = pair_b_active * fr * rl + pair_a_active * (1 - fr) * (1 - rl)
        rew_diagonal_contact = (fl_rr_pair + fr_rl_pair) * self.cfg.rew_scale_diagonal_contact * cmd_has_vel_gate

        # Swing contact penalty (walk-these-ways 방식): swing 중 발이 닿으면 페널티
        # — 진동으로 air_time 채우는 reward hacking 직접 차단
        swing_contact_err = (contact_actual * swing_mask).sum(dim=1)
        rew_swing_contact = swing_contact_err * self.cfg.rew_scale_swing_contact * cmd_has_vel_gate

        # Foot height reward: toe tip clearance during swing (Solo12: 6cm target)
        # toe tip = knee_pos + R_calf @ [0, 0, -0.130] (calf local z → world)
        _calf_quat = self.robot.data.body_quat_w[:, self._foot_body_ids_robot, :]  # [N,4,4]
        _N, _nf = self.num_envs, 4
        _calf_z_local = torch.zeros(_N * _nf, 3, device=self.device)
        _calf_z_local[:, 2] = 1.0
        _calf_z_world = quat_apply(_calf_quat.reshape(_N * _nf, 4), _calf_z_local).reshape(_N, _nf, 3)
        foot_tip_z = (self.robot.data.body_pos_w[:, self._foot_body_ids_robot, 2]
                      + _calf_z_world[:, :, 2] * (-0.130))
        foot_clearance = foot_tip_z.clamp(min=0.0, max=0.10)  # 10cm cap: 실제 보폭 높이 보상 범위 확대
        rew_foot_height = (foot_clearance * swing_mask).sum(dim=1) * self.cfg.rew_scale_foot_height * cmd_has_vel_gate

        # Knee angle penalty: knee too-straight → shin/knee walking root cause
        # URDF "foot" joint = knee. Default -0.83 rad. Penalize if > -0.3 rad (too extended)
        knee_angle = self.joint_pos[:, self._knee_ids]  # [N, 4]
        knee_overshoot = (knee_angle - (-0.3)).clamp(min=0.0)
        rew_knee_angle = torch.sum(torch.square(knee_overshoot), dim=1) * self.cfg.rew_scale_knee_angle

        # Stance knee height penalty: if knee joint (foot_link origin) is near ground during contact
        # Normal stance: knee_z ≈ 0.06m. Shin walking: knee_z ≈ 0.01m → penalize
        knee_z = self.robot.data.body_pos_w[:, self._foot_body_ids_robot, 2]  # [N, 4]
        knee_low_in_stance = (0.04 - knee_z).clamp(min=0.0) * contact_actual
        rew_knee_height_stance = knee_low_in_stance.sum(dim=1) * self.cfg.rew_scale_knee_height_stance

        # air_time_variance 패널티: 4발 air_time 불균형 시 패널티 → 한 다리만 움직이는 비대칭 차단
        air_time_var = torch.var(self.contact_sensor.data.last_air_time[:, self._foot_ids], dim=1)
        rew_air_time_var = -air_time_var * self.cfg.rew_scale_air_time_var

        # Body height reward:
        #   scale > 0: Gaussian reward centered at target (특정 높이 강제 — 비대칭 자세 유발 위험)
        #   scale < 0: 단방향 페널티 — target 이하일 때만 선형 패널티 (자연 높이 허용)
        body_height = self.robot.data.root_pos_w[:, 2]
        if self.cfg.rew_scale_body_height >= 0:
            height_error = (body_height - self.cfg.target_body_height).abs()
            rew_body_height = torch.exp(-height_error / 0.05) * self.cfg.rew_scale_body_height
        else:
            height_deficit = (self.cfg.target_body_height - body_height).clamp(min=0)
            rew_body_height = height_deficit * self.cfg.rew_scale_body_height

        # Non-foot contact penalty + stumble (legged_gym) + foot stance force (walk-these-ways)
        non_foot_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self._non_foot_contact_ids, :]
        non_foot_contact = (torch.norm(non_foot_forces, dim=-1) > self.cfg.non_foot_contact_threshold).float()
        rew_non_foot_contact = non_foot_contact.sum(dim=1) * self.cfg.rew_scale_non_foot_contact

        # Stumble penalty (legged_gym): 무릎 긁힘 = 수평력 >> 수직력
        # ||F_xy|| > 5×|F_z| → 무릎이 지면을 긁거나 세게 박히는 신호
        non_foot_horiz = torch.norm(non_foot_forces[:, :, :2], dim=-1)
        non_foot_vert = non_foot_forces[:, :, 2].abs()
        stumble_mask = (non_foot_horiz > 5.0 * non_foot_vert + 1e-3).float()
        rew_stumble = stumble_mask.sum(dim=1) * self.cfg.rew_scale_stumble

        # Foot stance force reward (walk-these-ways): stance 단계에서 발에 하중이 실릴수록 보상
        # — 발 대신 무릎에 체중 싣는 knee-walking 직접 억제
        foot_forces_z_abs = self.contact_sensor.data.net_forces_w_history[:, 0, self._foot_ids, 2].abs()
        stance_load = (foot_forces_z_abs * contact_target).sum(dim=1)  # [N]
        rew_foot_stance = stance_load.clamp(max=15.0) / 15.0 * self.cfg.rew_scale_foot_stance * cmd_has_vel_gate

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

        # Action jerk penalty (2차 미분, Walk These Ways 방식): 급격한 변화 차단
        action_jerk = torch.sum(
            torch.square(self.actions - 2.0 * self._last_actions + self._last_last_actions), dim=1
        )
        rew_action_jerk = action_jerk * self.cfg.rew_scale_action_jerk

        # Diagonal symmetry reward (trot): FL-RR, FR-RL 대각선 pair가 같은 thigh+calf 각도를 가져야 함
        # Joint index: FL(0,4,8), FR(1,5,9), RL(2,6,10), RR(3,7,11)
        fl_tc = self.joint_pos[:, [4, 8]]   # FL thigh+calf
        rr_tc = self.joint_pos[:, [7, 11]]  # RR thigh+calf
        fr_tc = self.joint_pos[:, [5, 9]]   # FR thigh+calf
        rl_tc = self.joint_pos[:, [6, 10]]  # RL thigh+calf
        rew_diagonal_symmetry = (
            torch.sum(torch.square(fl_tc - rr_tc), dim=1)
            + torch.sum(torch.square(fr_tc - rl_tc), dim=1)
        ) * self.cfg.rew_scale_diagonal_symmetry * cmd_has_vel_gate

        # Energy penalty: |τ_i| × |q̇_i| (metabolic cost 모사 — 부드럽고 효율적인 움직임 유도)
        rew_energy = torch.sum(
            torch.abs(self.robot.data.applied_torque) * torch.abs(self.joint_vel), dim=1
        ) * self.cfg.rew_scale_energy

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
                "rew/yaw_error": rew_ang_vel_z.mean().item(),
                "diag/body_height_mean": self.robot.data.root_pos_w[:, 2].mean().item(),
                "diag/body_height_min": self.robot.data.root_pos_w[:, 2].min().item(),
                "diag/torque_sat_ratio": torque_sat_ratio.item(),
                "diag/term_height_ratio": body_fallen_now.mean().item(),
                "diag/term_tilt_ratio": body_tilted_now.mean().item(),
                "rew/termination": rew_termination_log.mean().item(),
                "diag/per_step_net": per_step_net.mean().item(),
                "diag/term_ratio": self.reset_terminated.float().mean().item(),
                "diag/foot_span_mean": foot_span.mean().item(),
                "rew/gait": rew_gait.mean().item(),
                "rew/non_foot_contact": rew_non_foot_contact.mean().item(),
                "rew/stumble": rew_stumble.mean().item(),
                "rew/foot_stance": rew_foot_stance.mean().item(),
                "rew/swing_contact": rew_swing_contact.mean().item(),
                "rew/foot_height": rew_foot_height.mean().item(),
                "rew/knee_angle": rew_knee_angle.mean().item(),
                "rew/knee_height_stance": rew_knee_height_stance.mean().item(),
                "rew/heading": rew_heading.mean().item(),
                "diag/heading_err_deg": (self._heading_err.abs() * 57.3).mean().item(),
                "rew/lin_vel_penalty": rew_lin_vel_penalty.mean().item(),
                "diag/actual_lin_vel_x": self.robot.data.root_lin_vel_b[:, 0].mean().item(),
                "diag/actual_ang_vel_z": self.robot.data.root_ang_vel_b[:, 2].mean().item(),
                "rew/action_jerk": rew_action_jerk.mean().item(),
                "rew/diagonal_symmetry": rew_diagonal_symmetry.mean().item(),
                "rew/energy": rew_energy.mean().item(),
                "rew/yaw_tracking": rew_yaw_tracking.mean().item(),
                "rew/pos_drift": rew_pos_drift.mean().item(),
                "diag/lateral_drift_m": lateral_drift.mean().item(),
                "rew/heading_linear": rew_heading_linear.mean().item(),
                "rew/yaw_rate_error": rew_yaw_rate_error.mean().item(),
                "rew/diagonal_contact": rew_diagonal_contact.mean().item(),
            }

        return (base_rew + rew_gait + rew_body_height + rew_non_foot_contact + rew_joint_default
                + rew_upright + rew_ang_vel_z + rew_lin_vel_xy + rew_foot_spread + rew_foot_slip
                + rew_dof_acc + rew_stand_still + rew_dof_pos_limits + rew_contact_forces
                + rew_air_time_var + rew_lin_vel_penalty + rew_swing_contact + rew_foot_height
                + rew_stumble + rew_foot_stance + rew_knee_angle + rew_knee_height_stance
                + rew_heading + rew_action_jerk + rew_diagonal_symmetry + rew_energy
                + rew_yaw_tracking + rew_pos_drift
                + rew_heading_linear + rew_yaw_rate_error
                + rew_diagonal_contact)

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
        self._last_last_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._last_joint_vel[env_ids] = 0.0
        # randomize gait phase at reset to break synchronization
        self._gait_phase[env_ids] = torch.zeros(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
        # target heading: 리셋 시 현재 heading으로 초기화 (누적 yaw 명령의 기준점)
        _fwd = quat_apply(self.robot.data.root_quat_w[env_ids],
                          torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(n, -1))
        self._target_heading[env_ids] = torch.atan2(_fwd[:, 1], _fwd[:, 0])
        # world Y position at reset (lateral drift baseline)
        self._start_pos_y[env_ids] = self.robot.data.root_pos_w[env_ids, 1]


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
