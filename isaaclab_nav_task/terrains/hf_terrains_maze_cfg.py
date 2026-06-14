# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for maze height field terrains."""

from dataclasses import MISSING
from typing import Any, Optional

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg

from . import hf_terrains_maze


@configclass
class HfMazeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a maze height field terrain.

    This terrain generates a procedural maze with configurable wall structures,
    obstacles, and optional stairs. The maze can be used for navigation tasks
    with various difficulty levels.

    Height Field Data (set during terrain generation):
        - height_field_visual: Heights for Z-lookup (num_terrains, W, H)
        - height_field_valid_mask: Valid goal positions with safety padding
        - height_field_platform_mask: Platform positions for curriculum
        - height_field_spawn_mask: Valid spawn positions with larger padding
    """

    function = hf_terrains_maze.maze_terrain

    # =========================================================================
    # Height Field Storage (populated during terrain generation)
    # =========================================================================

    height_field_visual: torch.Tensor = None
    """Height field for Z-lookup (actual terrain heights)."""

    height_field_valid_mask: torch.Tensor = None
    """Boolean mask of valid goal positions (padded with GOAL_PADDING)."""

    height_field_platform_mask: torch.Tensor = None
    """Boolean mask of platform positions for curriculum learning."""

    height_field_spawn_mask: torch.Tensor = None
    """Boolean mask of valid spawn positions (larger padding for robot body)."""

    # =========================================================================
    # Maze Generation Parameters
    # =========================================================================

    maze: bool = True
    """Flag indicating this is a maze terrain."""

    open_probability: float = None
    """Probability of a cell being open in the maze."""

    grid_size: tuple[int, int] = (15, 15)
    """Size of the maze grid (number of cells in width and height)."""

    cell_size: float = 2.0
    """Size of each cell in the maze grid (in meters)."""

    wall_height: float = 2.4
    """Height of the walls in meters."""

    # =========================================================================
    # Terrain Features
    # =========================================================================

    add_goal: Any = MISSING
    """Enable goal sampling data generation."""

    add_noise_to_flat: Any = MISSING
    """Add noise to flat areas of the maze."""

    randomize_wall: Any = MISSING
    """Use randomized obstacle shapes instead of full walls."""

    random_wall_ratio: float = 0.5
    """Mix ratio between randomized and standard walls. Defaults to 0.5."""

    non_maze_terrain: bool = False
    """Use non-maze terrain with random obstacles. Defaults to False."""

    stairs: bool = False
    """Add stairs to empty map. Defaults to False."""

    add_stairs_to_maze: bool = False
    """Add stairs to the maze. Defaults to False."""

    dynamic_obstacles: bool = False
    """Enable pit/trough obstacles. Defaults to False."""

    # =========================================================================
    # Random Number Generator
    # =========================================================================

    rng: Optional[np.random.Generator] = None
    """Random number generator for reproducible terrain generation.

    Set by the terrain generator (patches.py) before calling the terrain function.
    If None, will create a new unseeded generator (non-reproducible).
    """

    terrain_row: Optional[int] = None
    """Sub-terrain row index set by the patched terrain generator."""

    terrain_col: Optional[int] = None
    """Sub-terrain column index set by the patched terrain generator."""

    terrain_num_rows: Optional[int] = None
    """Total number of sub-terrain rows set by the patched terrain generator."""

    terrain_num_cols: Optional[int] = None
    """Total number of sub-terrain columns set by the patched terrain generator."""


@configclass
class HfRandomMazeTerrainCfg(HfMazeTerrainCfg):
    """Thin-wall random maze terrain for long-memory navigation."""

    function = hf_terrains_maze.random_maze_terrain

    wall_width: float = 0.15
    """Width of maze wall segments in meters."""

    corridor_width: float = 1.35
    """Fallback clear width of entrance, exit, and stair corridors in meters."""

    corridor_width_range: tuple[float, float] | None = (1.2, 1.6)
    """Optional range to uniformly sample clear corridor width in meters."""

    exit_count_range: tuple[int, int] = (2, 5)
    """Inclusive range for the number of outside-connected maze exits."""

    num_stairs: int = 0
    """Maximum number of large pyramid patches to add to each maze tile."""

    step_height_range: tuple[float, float] = (0.15, 0.18)
    """Minimum and maximum height increment per platform terrace in meters."""

    step_width_range: tuple[float, float] = (0.28, 0.35)
    """Minimum and maximum horizontal width of each platform edge terrace in meters."""

    pyramid_patch_size_range: tuple[float, float] = (8.0, 14.0)
    """Minimum and maximum side length of each elevated platform patch in meters."""

    pyramid_levels_range: tuple[int, int] = (6, 10)
    """Inclusive range for the number of edge terraces before the elevated platform."""

    pyramid_platform_fraction_range: tuple[float, float] = (0.35, 0.55)
    """Fraction of the patch side length reserved as the flat elevated top platform."""

    stairs_platform_width: float = 0.9
    """Deprecated compatibility field for older corridor stair generation."""

    stairs_platform_cells: tuple[int, int] = (2, 5)
    """Deprecated compatibility field for older corridor stair generation."""


@configclass
class HfBranchingCorridorsTerrainCfg(HfMazeTerrainCfg):
    """WFC branching-corridor height field terrain."""

    function = hf_terrains_maze.branching_corridors_terrain

    grid_size: tuple[int, int] = (12, 12)
    """Size of the WFC tile grid in (rows, columns)."""

    tile_size: float = 2.0
    """Size of each WFC tile in meters."""

    wall_height: float = 3.0
    """Height of corridor walls in meters."""

    wall_thickness: float = 0.2
    """Thickness of corridor walls in meters."""

    level_height: float = 0.5
    """Height of the second corridor level in meters."""

    step_height: float = 0.1
    """Vertical increment used when rasterizing stair tiles."""

    stair_weight: float = 0.12
    """Relative WFC sampling weight for stair connector tiles."""

    min_stair_distance: int = 4
    """Minimum spacing target for stair tiles retained for API compatibility."""

    spine_x: int = 3
    """Column of the seeded E-branch spine."""

    top_y: int = 2
    """Top row of the seeded E-branch spine."""

    branch_length: int = 3
    """Length of each seeded branch extending from the spine."""
