# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Quadrupedal-Bot-Direct-v0",
    entry_point=f"{__name__}.quadrupedal_bot_env:QuadrupedalBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrupedal_bot_env_cfg:QuadrupedalBotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Quadrupedal-Bot-Stance-v0",
    entry_point=f"{__name__}.quadrupedal_bot_env:QuadrupedalBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrupedal_bot_stance_cfg:QuadrupedalBotStanceCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_stage1:PPORunnerCfgStage1",
    },
)

gym.register(
    id="Template-Quadrupedal-Bot-Trot-v0",
    entry_point=f"{__name__}.quadrupedal_bot_env:QuadrupedalBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrupedal_bot_trot_cfg:QuadrupedalBotTrotCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_stage2:PPORunnerCfgStage2",
    },
)

gym.register(
    id="Template-Quadrupedal-Bot-TrotInplace-v0",
    entry_point=f"{__name__}.quadrupedal_bot_env:QuadrupedalBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrupedal_bot_trot_inplace_cfg:QuadrupedalBotTrotInplaceCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_stage3:PPORunnerCfgStage3",
    },
)

gym.register(
    id="Template-Quadrupedal-Bot-WalkFwd-v0",
    entry_point=f"{__name__}.quadrupedal_bot_env:QuadrupedalBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrupedal_bot_walk_cfg:QuadrupedalBotWalkFwdCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_stage3:PPORunnerCfgStage4",
    },
)

gym.register(
    id="Template-Quadrupedal-Bot-WalkAllDir-v0",
    entry_point=f"{__name__}.quadrupedal_bot_env:QuadrupedalBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadrupedal_bot_walk_cfg:QuadrupedalBotWalkAllDirCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_stage3:PPORunnerCfgStage5",
    },
)