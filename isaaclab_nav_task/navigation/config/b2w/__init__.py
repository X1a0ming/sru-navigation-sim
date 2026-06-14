# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# MDPO

gym.register(
    id="Isaac-Nav-MDPO-B2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-B2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavMDPORunnerDevCfg,
    },
)

######################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-B2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPPORunnerDevCfg,
    },
)

######################################################################################
# PPO Delta-SRU

gym.register(
    id="Isaac-Nav-PPO-B2W-DeltaSRU-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoDeltaSRURunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-DeltaSRU-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoDeltaSRURunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-DeltaSRU-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoDeltaSRURunnerDevCfg,
    },
)

######################################################################################
# PPO Self-Cached Delta-SRU on hard complex-maze terrain

gym.register(
    id="Isaac-Nav-PPO-B2W-SelfCachedDeltaSRU-ComplexMazeHard-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_COMPLEX_MAZE_HARD,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoSelfCachedDeltaSRURunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-SelfCachedDeltaSRU-ComplexMazeHard-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_COMPLEX_MAZE_HARD_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoSelfCachedDeltaSRURunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-SelfCachedDeltaSRU-ComplexMazeHard-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_COMPLEX_MAZE_HARD_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoSelfCachedDeltaSRURunnerDevCfg,
    },
)

######################################################################################
# PPO Self-Cached Delta-SRU on random-maze terrain

gym.register(
    id="Isaac-Nav-PPO-B2W-SelfCachedDeltaSRU-RandomMaze-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_RANDOM_MAZE,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoSelfCachedDeltaSRURunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-SelfCachedDeltaSRU-RandomMaze-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_RANDOM_MAZE_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoSelfCachedDeltaSRURunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-B2W-SelfCachedDeltaSRU-RandomMaze-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.B2WNavigationEnvCfg_RANDOM_MAZE_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.B2WNavPpoSelfCachedDeltaSRURunnerDevCfg,
    },
)
