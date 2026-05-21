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
      - rew_scale_body_height 20.0 (gait ~12보다 강하게 하여 local min 탈출)
      - rew_scale_gait 12.0 (높이 유지와 균형)
    """

    episode_length_s: float = 15.0

    # --- 높이 강제 강화 (낮은높이 local min 탈출) ---
    rew_scale_body_height: float = 30.0   # 강한 높이 유도: gait(~12) 압도
    target_body_height: float = 0.17
    termination_height: float = 0.135    # 0.135m 이하 에피소드 종료 (하한선)

    # --- 전방향 명령 ---
    cmd_lin_vel_x_range: tuple = (-0.3, 0.4)   # 후진 포함
    cmd_lin_vel_y_range: tuple = (-0.2, 0.2)   # 좌우 보행
    cmd_ang_vel_z_range: tuple = (-1.0, 1.0)   # 회전 + 제자리 회전
    zero_command_prob: float = 0.1              # 10% 제자리 서있기

    # --- 외란 강화 ---
    push_interval_s: float = 8.0
    max_push_vel: float = 0.3

    # --- gait (높이 보상과 경쟁 줄임) ---
    rew_scale_gait: float = 12.0
    rew_scale_air_time: float = 10.0
    gait_freq_hz: float = 1.4
