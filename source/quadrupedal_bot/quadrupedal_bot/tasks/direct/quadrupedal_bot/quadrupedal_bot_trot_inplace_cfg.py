from isaaclab.utils import configclass

from .quadrupedal_bot_trot_cfg import QuadrupedalBotTrotCfg


@configclass
class QuadrupedalBotTrotInplaceCfg(QuadrupedalBotTrotCfg):
    """Stage 2: 제자리 Trot — Standing v3c 위에 gait 추가.
    서있기 안정성 유지하며 trot 패턴 발 들기 추가.
    """

    episode_length_s: float = 20.0

    gait_reward_always_on: bool = True
    gait_freq_hz: float = 1.2

    # --- gait 보상 추가 ---
    rew_scale_gait: float = 8.0
    rew_scale_air_time: float = 10.0
    air_time_threshold: float = 0.05
    rew_scale_swing_contact: float = -5.0
    rew_scale_diagonal_symmetry: float = -2.0
    rew_scale_air_time_var: float = 3.0
    rew_scale_diagonal_contact: float = 2.0

    # --- 발 들기 유도 (약하게) ---
    rew_scale_foot_height: float = 10.0

    # --- Gaussian 관절 유도 ---
    target_leg_angle_swing_gauss: float = 0.0
    sigma_leg_swing: float = 0.25
    rew_scale_hip_swing_gauss: float = 3.0
    target_knee_angle_swing_gauss: float = -1.3
    sigma_knee_swing: float = 0.30
    rew_scale_knee_swing_gauss: float = 3.0
    rew_scale_swing_min_knee: float = 5.0
    min_knee_angle_swing: float = -1.1

    # --- 서있기 안정성 유지 ---
    rew_scale_foot_stance: float = 1.0
    rew_scale_stumble: float = -2.0
    rew_scale_non_foot_contact: float = -3.0
