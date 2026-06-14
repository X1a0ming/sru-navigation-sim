# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for maze terrains."""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_maze_cfg import HfBranchingCorridorsTerrainCfg, HfMazeTerrainCfg, HfRandomMazeTerrainCfg

MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,  # Border around the entire terrain grid (not per-tile)
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "maze": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=True,
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        "stairs": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            random_wall_ratio=1.0,
            non_maze_terrain=False,
            stairs=True,
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,  # Enables pit/trough generation
        ),
    },
)
"""Maze terrain configuration for navigation tasks."""


RANDOM_MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "random_maze": HfRandomMazeTerrainCfg(
            proportion=0.5,
            open_probability=0.9,
            grid_size=(20, 20),
            cell_size=1.5,
            wall_height=2.4,
            wall_width=0.15,
            corridor_width=1.35,
            corridor_width_range=(1.2, 1.6),
            exit_count_range=(2, 6),
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            random_wall_ratio=0.5,
            num_stairs=0,
        ),
        "random_maze_stairs": HfRandomMazeTerrainCfg(
            proportion=0.5,
            open_probability=0.9,
            grid_size=(20, 20),
            cell_size=1.5,
            wall_height=2.4,
            wall_width=0.15,
            corridor_width=1.35,
            corridor_width_range=(1.2, 1.6),
            exit_count_range=(2, 6),
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            random_wall_ratio=0.5,
            num_stairs=3,
            step_height_range=(0.15, 0.18),
            step_width_range=(0.28, 0.35),
            pyramid_patch_size_range=(8.0, 14.0),
            pyramid_levels_range=(6, 10),
            pyramid_platform_fraction_range=(0.35, 0.55),
            stairs_platform_width=0.9,
        ),
    },
)
"""Random thin-wall maze terrain with flat and elevated-patch variants."""


BRANCHING_CORRIDORS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "branching_corridors": HfBranchingCorridorsTerrainCfg(
            proportion=1.0,
            grid_size=(12, 12),
            tile_size=2.0,
            wall_height=3.0,
            wall_thickness=0.2,
            level_height=0.5,
            step_height=0.1,
            stair_weight=0.12,
            min_stair_distance=4,
            spine_x=3,
            top_y=2,
            branch_length=3,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
        ),
    },
)
"""WFC branching-corridors terrain configuration for navigation previews."""
