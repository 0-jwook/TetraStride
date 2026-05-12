from isaaclab.utils import configclass

from .quadrupedal_bot_env_cfg import QuadrupedalBotEnvCfg


@configclass
class QuadrupedalBotTrotCfg(QuadrupedalBotEnvCfg):
    """Stage 2: Rudin 2022 방식 처음부터 학습 — velocity tracking + air_time으로 trot 자연 발생."""

    episode_length_s: float = 20.0          # 길게: 탐색 충분히 허용
    target_body_height: float = 0.17        # 자연 보행 높이 (단방향 페널티 기준선)

    action_scale: float = 0.35              # base 기본값 유지

    # CCP velocity commands: 전방향 (약한 명령부터 시작)
    cmd_lin_vel_x_range: tuple = (0.3, 0.7)
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)
    cmd_ang_vel_z_range: tuple = (-0.3, 0.3)
    zero_command_prob: float = 0.2          # 20%: 직선 보행 중 heading 유지 학습 강화

    # --- Gait 주파수 ---
    gait_freq_hz: float = 1.5              # 검증된 표준값 유지

    # --- 핵심: Rudin 2022 속도 추적 ---
    rew_scale_alive: float = 0.5
    rew_scale_lin_vel: float = 3.0          # 속도 추적
    rew_scale_ang_vel: float = 1.0          # yaw rate 추적
    rew_scale_ang_vel_z: float = -1.0       # yaw rate 패널티
    rew_scale_heading: float = 3.0          # heading 추적
    heading_sigma: float = 0.25            # 표준값 (모든 성공 사례 동일)
    rew_scale_movement: float = 0.0
    rew_scale_lin_vel_penalty: float = 0.0

    # --- Gait 유도 (legged_gym 검증값 기준) ---
    rew_scale_gait: float = 2.5
    rew_scale_air_time: float = 2.0         # 5.0→2.0: 과도한 다리 들기 억제 (legged_gym 1.0 기준)
    air_time_threshold: float = 0.04        # 원래 값 복원

    # --- 자세 안정 (legged_gym 표준) ---
    rew_scale_body_height: float = -8.0     # 0.17m 이하 선형 패널티
    rew_scale_upright: float = 2.0          # 0→2.0: 수평 유지 Gaussian 보상 (10° 기울면 1.06/step 손실)
    rew_scale_gravity: float = -5.0         # nose-down 자세 차단
    rew_scale_ang_vel_xy: float = -0.3      # roll/pitch 속도 억제
    rew_scale_lin_vel_z: float = -2.0       # 수직속도 패널티

    # --- 관절/토크 제약 (legged_gym 표준) ---
    rew_scale_joint_vel: float = -1e-4
    rew_scale_torque: float = -2.5e-5      # -1e-5→-2.5e-5: Isaac Lab ANYmal C 검증값
    rew_scale_action_rate: float = -0.05
    rew_scale_dof_acc: float = -1e-6        # base(-2.5e-7)보다 4x 강화: 관절 진동 억제
    rew_scale_termination: float = -5.0     # 낮게: 패널티 너무 크면 안전제일 편향

    # --- 자세 유지 ---
    rew_scale_joint_default: float = -3.0   # -1.0→-3.0: 어깨 직접 제어 강화 (foot_span 고착 돌파)
    rew_scale_foot_spread: float = -6.0     # -3.0→-6.0 강화: 0.283m 고착 돌파 (foot_span 미감소)
    rew_scale_foot_slip: float = -0.05      # 약한 슬립 패널티
    rew_scale_air_time_var: float = 3.0     # 1.0→3.0: 비대칭 보행 강화 차단

    # --- 무릎 보행 방지 (2.5kg 로봇 전용) ---
    non_foot_contact_threshold: float = 4.0    # 20N→4N: 2.5kg 로봇 무릎 하중(~6N) 감지
    rew_scale_non_foot_contact: float = -5.0   # 무릎 접촉 강력 페널티
    rew_scale_stumble: float = -2.0            # legged_gym: 무릎 긁힘 페널티 (수평력 >> 수직력)
    rew_scale_foot_stance: float = 2.0         # walk-these-ways: stance 중 발 하중 보상

    # --- 무릎 보행 직접 차단 ---
    rew_scale_knee_angle: float = -5.0          # knee too-straight(>-0.3 rad) → shin/knee walking 방지
    rew_scale_knee_height_stance: float = -10.0 # stance 중 knee joint 너무 낮으면 패널티 (무릎 보행 감지)

    # --- Gait 품질 강화 (legged_gym / walk-these-ways 방식) ---
    rew_scale_swing_contact: float = -1.5   # swing 중 발 접촉 페널티: 진동으로 발 들기 reward hacking 차단
    rew_scale_foot_height: float = 5.0      # swing 단계 발 높이 직접 보상 (cap 5cm, 실제 toe tip 기준)

    # --- 보행 품질 (검증된 항목만) ---
    rew_scale_action_jerk: float = -0.001   # -0.002→-0.001: 약하게 (Solo12 기준)
    rew_scale_diagonal_symmetry: float = 0.0   # 비활성: 어떤 성공 사례도 사용 안 함
    rew_scale_energy: float = -1e-4         # |τ|×|q̇| 에너지 패널티 (iit-DLSLab 검증)

    # --- 도메인 랜덤화: 주기적 랜덤 푸시 ---
    push_interval_s: float = 8.0            # 8초마다 랜덤 푸시
    max_push_vel: float = 0.3               # 최대 0.3m/s 수평 충격
