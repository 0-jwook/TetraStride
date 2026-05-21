from isaaclab.utils import configclass

from .quadrupedal_bot_trot_inplace_cfg import QuadrupedalBotTrotInplaceCfg


@configclass
class QuadrupedalBotWalkFwdCfg(QuadrupedalBotTrotInplaceCfg):
    """Stage 3: 전진 보행 — Trot in-place 전이."""

    episode_length_s: float = 15.0

    # --- 전진 명령 ---
    cmd_lin_vel_x_range: tuple = (0.1, 0.4)
    cmd_lin_vel_y_range: tuple = (-0.05, 0.05)
    cmd_ang_vel_z_range: tuple = (-0.2, 0.2)
    zero_command_prob: float = 0.05

    # --- 속도 추적 보상 활성화 ---
    rew_scale_lin_vel: float = 6.0
    rew_scale_ang_vel: float = 0.5
    rew_scale_movement: float = 2.0

    # --- gait 강화 ---
    rew_scale_gait: float = 12.0
    rew_scale_air_time: float = 12.0
    gait_freq_hz: float = 1.3

    # --- 외란 추가 (견고성) ---
    push_interval_s: float = 10.0
    max_push_vel: float = 0.2


@configclass
class QuadrupedalBotWalkAllDirCfg(QuadrupedalBotWalkFwdCfg):
    """Stage 4: 전방향 보행 + 제자리 회전 — Forward Walk 전이.
    Stage3에서 body_height 0.115m로 낮아지는 문제 수정:
      - rew_scale_body_height 3.0 → 12.0 (강한 높이 유지)
    """

    episode_length_s: float = 15.0

    # --- 높이 강제 강화 (Stage3 크롤링 수정) ---
    rew_scale_body_height: float = 12.0   # 3→12: gait 보상(~12)과 균형
    target_body_height: float = 0.17

    # --- 전방향 명령 ---
    cmd_lin_vel_x_range: tuple = (-0.3, 0.4)   # 후진 포함
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)   # 좌우 보행
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)   # 회전 + 제자리 회전
    zero_command_prob: float = 0.1              # 10% 제자리 서있기

    # --- 외란 강화 ---
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3

    # --- gait 강화 ---
    rew_scale_gait: float = 15.0
    rew_scale_air_time: float = 15.0
    gait_freq_hz: float = 1.4
