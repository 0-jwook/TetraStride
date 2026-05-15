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
    # obs: lin_vel_b(3) + ang_vel(3) + proj_gravity(3) + commands(3)
    #      + joint_pos_rel(12) + joint_vel(12) + last_actions(12) + gait_phase(2)
    #      + heading_err_sin(1) + heading_err_cos(1) = 52
    action_space: int = 12
    observation_space: int = 52
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
    action_scale: float = 0.35
    action_smoothing: float = 0.8
    init_noise_scale: float = 0.03   # 초기화 관절 노이즈 (0.1→0.03: 0.7N·m로 복원 가능한 범위)

    # --- velocity commands ---
    cmd_lin_vel_x_range: tuple = (0.2, 0.6)
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)

    # --- termination ---
    termination_height: float = 0.15
    target_body_height: float = 0.18   # 목표 서기 높이 (패널티 기준)
    target_foot_span: float = 0.12     # 0.18→0.12: 넓은 도마뱀 자세 방지

    # --- reward scales (legged_gym 기반) ---
    rew_scale_alive: float = 0.2       # Stage3: 살아있기 보상
    rew_scale_lin_vel: float = 5.0     # 3→5: stronger velocity tracking to break plateau
    rew_scale_ang_vel: float = 0.1     # reduced: prevent angular-vel standing optimum
    rew_scale_lin_vel_z: float = -2.0
    rew_scale_ang_vel_xy: float = -0.05
    rew_scale_gravity: float = -1.0
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -1e-5
    rew_scale_action_rate: float = -0.01
    rew_scale_termination: float = -30.0   # PHYSICS §2.3: |term| > alive×30 but < alive×1000×0.05
    rew_scale_air_time: float = 6.0     # strongly incentivize leg lifting
    rew_scale_movement: float = 3.0     # 2→3: stronger linear gradient toward forward motion
    rew_scale_gait: float = 5.0         # trot gait reference: reward foot clearance during swing phase
    rew_scale_body_height: float = -8.0   # 몸통 낮으면 페널티 (플랭크 방지)
    rew_scale_non_foot_contact: float = -1.0  # 발 외 부위(무릎/배) 지면 접촉 페널티
    non_foot_contact_threshold: float = 20.0  # N, 무릎 접촉 감지 임계값 (2.5kg 로봇: 정적 하중 ~6N)
    rew_scale_lin_vel_xy: float = 0.0        # 수평 이동 패널티 (Stance에서만 활성화)
    rew_scale_ang_vel_z: float = 0.0         # yaw 회전 패널티 (Stance에서만 활성화)
    rew_scale_joint_default: float = 0.0     # 어깨 관절 dead zone 패널티
    rew_scale_upright: float = 0.3           # IMU 직립 보상 (projected_gravity_b z축 기반)
    rew_scale_foot_spread: float = 0.0       # 발 좌우 간격 패널티 (Stance에서 활성화)
    rew_scale_foot_slip: float = 0.0         # 발 미끄러짐 패널티 (Margolis 2022)
    freeze_gait_phase: bool = False   # Stance 전용: gait clock 동결 (명령=0에서 주기적 불안정 방지)
    rew_scale_dof_acc: float = -2.5e-7       # DOF 가속도 패널티 (Rudin 2021 표준)
    rew_scale_stand_still: float = -0.5      # cmd≈0 시 관절 이탈 패널티 (legged_gym 표준)
    rew_scale_dof_pos_limits: float = 0.0    # 관절 soft limit 초과 패널티 (legged_gym: _reward_dof_pos_limits)
    rew_scale_contact_forces: float = 0.0    # 발 착지 충격력 초과 패널티 (legged_gym: _reward_feet_contact_forces)
    max_foot_contact_force: float = 50.0     # N, 충격력 패널티 기준값 (정적 3.68N × ~13배 마진)
    rew_scale_air_time_var: float = 0.0      # 4발 air_time 분산 패널티 — 비대칭 정책 차단
    air_time_threshold: float = 0.1         # 발 들기 최소 시간(s): 0.1s=5steps 이상 들어야 보상
    rew_scale_lin_vel_penalty: float = 0.0  # 선형속도 추적 오차 패널티 (-||cmd-vel||² × scale)
    rew_scale_swing_contact: float = 0.0   # swing 중 발 접촉 페널티 (양방향 gait 강제)
    rew_scale_foot_height: float = 0.0     # swing 단계 발 높이 직접 보상 (air_time 간접 측정 보완)
    rew_scale_stumble: float = 0.0         # legged_gym: 무릎 긁힘 감지 (||F_xy|| > 5×|F_z| on non-foot)
    rew_scale_foot_stance: float = 0.0     # walk-these-ways: stance 중 발 하중 보상 (발끝 착지 유도)
    rew_scale_knee_angle: float = 0.0      # knee too-straight penalty (>-0.3 rad → shin walking)
    rew_scale_knee_height_stance: float = 0.0  # knee too-low during contact penalty (shin walking detection)
    rew_scale_heading: float = 0.0         # heading tracking reward (angle-based, legged_gym 방식)
    heading_sigma: float = 0.25           # heading reward kernel sigma (작을수록 타이트한 직선 요구)
    gait_freq_hz: float = 1.5            # trot gait clock 주파수 (Hz)
    rew_scale_action_jerk: float = 0.0   # action 2차 미분 패널티 (jerk, Walk These Ways 방식)
    rew_scale_diagonal_symmetry: float = 0.0  # trot diagonal pair (FL-RR, FR-RL) 대칭 패널티
    rew_scale_energy: float = 0.0        # |τ|×|q̇| 에너지 패널티 (자연스러운 유연한 움직임)
    rew_scale_yaw_tracking: float = 0.0  # legged_gym 방식 exp(-yaw_err²/0.25) 보상 (yaw 추적 동기)
    rew_scale_pos_drift: float = 0.0     # 세계 좌표 Y축 누적 drift 패널티 (직진 명령 env만)
    rew_scale_heading_linear: float = 0.0  # 선형 heading 오차 패널티 (exp 포화 보완)
    rew_scale_yaw_rate_error: float = 0.0  # 선형 yaw rate 오차 패널티

    # --- Domain randomization (push perturbation) ---
    push_interval_s: float = 0.0           # 초 간격 랜덤 푸시 (0=비활성)
    max_push_vel: float = 0.5              # m/s 최대 푸시 속도

    # --- Command-Conditioned Policy ---
    # rel_standing_envs 동등: 리셋 시 이 비율만큼 cmd=(0,0,0) 강제 → 제자리 서기 학습 보장
    zero_command_prob: float = 0.0
