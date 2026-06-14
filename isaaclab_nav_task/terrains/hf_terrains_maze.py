# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Maze terrain generation for navigation tasks.

This module generates terrain height fields with explicit valid position masks.
The key simplification is that terrain generation directly outputs:
- `heights`: Actual terrain heights for rendering/physics
- `valid_mask`: Boolean mask of valid goal/spawn positions

This eliminates the need for complex height-based classification in goal sampling.

The terrain data is stored on the config during generation, then picked up by
the patches system and stored on TerrainImporter for access via:
- self.env.scene.terrain._height_field_visual
- self.env.scene.terrain._height_field_valid_mask
- self.env.scene.terrain._height_field_platform_mask
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from scipy.ndimage import binary_dilation, rotate, shift
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.terrains.height_field.utils import height_field_to_mesh

from .terrain_constants import HEIGHTS, PADDING, STAIRS, OBSTACLES, ObstacleType

if TYPE_CHECKING:
    from . import hf_terrains_maze_cfg


# =============================================================================
# Terrain Data Container
# =============================================================================

@dataclass
class TerrainData:
    """Container for terrain height field and valid position mask.

    Attributes:
        heights: Height field for rendering/physics (actual terrain heights).
        valid_mask: Boolean mask where True = valid for goals/spawns.
        platform_mask: Boolean mask where True = elevated platform (for curriculum).
    """
    heights: np.ndarray
    valid_mask: np.ndarray
    platform_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def create(cls, width: int, height: int) -> "TerrainData":
        """Create empty terrain data with ground-level heights."""
        return cls(
            heights=np.zeros((width, height), dtype=np.int16),
            valid_mask=np.ones((width, height), dtype=bool),  # Start all valid
            platform_mask=np.zeros((width, height), dtype=bool),
        )

    def set_obstacle(
        self,
        x_start: int, x_end: int,
        y_start: int, y_end: int,
        height_value: int
    ):
        """Set a region as an obstacle (invalid for goals)."""
        self.heights[x_start:x_end, y_start:y_end] = height_value
        self.valid_mask[x_start:x_end, y_start:y_end] = False

    def set_platform(
        self,
        x_start: int, x_end: int,
        y_start: int, y_end: int,
        height_value: int
    ):
        """Set a region as a platform (valid for goals, elevated)."""
        self.heights[x_start:x_end, y_start:y_end] = height_value
        self.valid_mask[x_start:x_end, y_start:y_end] = True
        self.platform_mask[x_start:x_end, y_start:y_end] = True

    def set_ground(self, x_start: int, x_end: int, y_start: int, y_end: int):
        """Set a region as flat ground (valid for goals)."""
        self.heights[x_start:x_end, y_start:y_end] = HEIGHTS.GROUND
        self.valid_mask[x_start:x_end, y_start:y_end] = True
        self.platform_mask[x_start:x_end, y_start:y_end] = False

    def apply_padding(self, padding_cells: int):
        """Dilate invalid regions by padding cells for safety margin."""
        obstacles = ~self.valid_mask
        kernel = np.ones((2 * padding_cells + 1, 2 * padding_cells + 1), dtype=bool)
        dilated = binary_dilation(obstacles, structure=kernel)
        self.valid_mask = ~dilated

    def create_spawn_mask(self, spawn_padding_cells: int) -> np.ndarray:
        """Create a mask for spawn positions with larger padding than goals."""
        extra_padding = spawn_padding_cells - PADDING.GOAL_PADDING
        if extra_padding > 0:
            obstacles = ~self.valid_mask
            kernel = np.ones((2 * extra_padding + 1, 2 * extra_padding + 1), dtype=bool)
            dilated = binary_dilation(obstacles, structure=kernel)
            return ~dilated
        return self.valid_mask.copy()

    def exclude_borders(self, border_cells: int = 2):
        """Mark terrain borders as invalid."""
        self.valid_mask[:border_cells, :] = False
        self.valid_mask[-border_cells:, :] = False
        self.valid_mask[:, :border_cells] = False
        self.valid_mask[:, -border_cells:] = False

    def apply_height_transition_padding(self, height_threshold: int, padding_cells: int):
        """Mark cells near height transitions as invalid."""
        grad_x = np.abs(np.diff(self.heights, axis=0, prepend=self.heights[:1, :]))
        grad_y = np.abs(np.diff(self.heights, axis=1, prepend=self.heights[:, :1]))
        grad_x_back = np.abs(np.diff(self.heights, axis=0, append=self.heights[-1:, :]))
        grad_y_back = np.abs(np.diff(self.heights, axis=1, append=self.heights[:, -1:]))

        max_grad = np.maximum.reduce([grad_x, grad_y, grad_x_back, grad_y_back])
        transition_mask = (max_grad >= height_threshold).astype(bool)

        if padding_cells > 0:
            kernel = np.ones((2 * padding_cells + 1, 2 * padding_cells + 1), dtype=bool)
            transition_mask = binary_dilation(transition_mask, structure=kernel).astype(bool)

        self.valid_mask = self.valid_mask & ~transition_mask


def get_cell_bounds(
    cell_x: int, cell_y: int, cell_pixels: int, max_x: int, max_y: int
) -> Tuple[int, int, int, int]:
    """Get pixel bounds for a maze cell with clamping.

    Returns:
        Tuple of (x_start, x_end, y_start, y_end).
    """
    return (
        max(0, cell_x * cell_pixels),
        min(max_x, (cell_x + 1) * cell_pixels),
        max(0, cell_y * cell_pixels),
        min(max_y, (cell_y + 1) * cell_pixels),
    )


# =============================================================================
# Maze Generation
# =============================================================================

def generate_maze(
    rng: np.random.Generator,
    width: int,
    height: int,
    open_prob: float
) -> np.ndarray:
    """Generate maze using DFS with random openings.

    Args:
        rng: Random number generator for reproducibility.
        width: Maze width in cells.
        height: Maze height in cells.
        open_prob: Probability of random wall removal.

    Returns:
        2D array where 1=wall, 0=path.
    """
    maze = np.ones((width, height), dtype=np.uint8)
    stack = [(0, 0)]
    maze[0, 0] = 0

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[nx, ny] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            idx = rng.integers(len(neighbors))
            nx, ny = neighbors[idx]
            maze[(x + nx) // 2, (y + ny) // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Random openings
    maze[rng.random((width, height)) < open_prob] = 0
    return maze


def clear_center(maze: np.ndarray, terrain: TerrainData, cell_pixels: int):
    """Clear the center area for spawning."""
    cx, cy = maze.shape[0] // 2, maze.shape[1] // 2

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if abs(dx) + abs(dy) <= 1:  # Plus shape
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                    maze[nx, ny] = 0

    x_start = (cx - 1) * cell_pixels
    x_end = (cx + 2) * cell_pixels
    y_start = (cy - 1) * cell_pixels
    y_end = (cy + 2) * cell_pixels
    terrain.set_ground(x_start, x_end, y_start, y_end)


# =============================================================================
# Obstacle Generators
# =============================================================================

def make_pillar(
    _rng: np.random.Generator,
    size: int,
    wall_height: int,
    scale: float,
    is_pit: bool,
    thickness: int
) -> np.ndarray:
    """Generate a centered pillar obstacle."""
    grid = np.zeros((size, size), dtype=np.int16)
    h = int(wall_height * scale) * (-1 if is_pit else 1)
    grid[thickness:size-thickness, thickness:size-thickness] = h
    return grid


def make_bar(
    rng: np.random.Generator,
    size: int,
    wall_height: int,
    scale: float,
    is_pit: bool,
    thickness: int
) -> np.ndarray:
    """Generate a rotated bar obstacle."""
    grid = np.zeros((size, size), dtype=np.int16)
    center = size // 2
    h = int(wall_height * scale)
    grid[center - thickness//2:center + thickness//2, :] = h

    angle = rng.uniform(-180, 180)
    grid = rotate(grid, angle, reshape=False, order=1).astype(np.int16)

    if is_pit:
        grid = -grid
    return grid


def make_cross(
    rng: np.random.Generator,
    size: int,
    wall_height: int,
    scale: float,
    is_pit: bool,
    thickness: int
) -> np.ndarray:
    """Generate a cross-shaped obstacle."""
    grid = np.zeros((size, size), dtype=np.int16)
    center = size // 2
    h = int(wall_height * scale)
    grid[center - thickness//2:center + thickness//2, :] = h
    grid[:, center - thickness//2:center + thickness//2] = h

    angle = rng.uniform(-180, 180)
    grid = rotate(grid, angle, reshape=False, order=1).astype(np.int16)

    if is_pit:
        grid = -grid
    return grid


def make_shifted_block(
    rng: np.random.Generator,
    size: int,
    wall_height: int,
    scale: float,
    is_pit: bool,
    thickness: int
) -> np.ndarray:
    """Generate a randomly shifted block."""
    grid = np.zeros((size, size), dtype=np.int16)
    h = int(wall_height * scale)
    grid[thickness:size-thickness, thickness:size-thickness] = h

    room = size // 2 - thickness
    shift_amt = (
        rng.integers(-room, room + 1),
        rng.integers(-room, room + 1)
    )
    grid = shift(grid, shift=shift_amt, cval=0).astype(np.int16)

    if is_pit:
        grid = -grid
    return grid


# Obstacle generator lookup table
_OBSTACLE_GENERATORS = {
    ObstacleType.PILLAR: make_pillar,
    ObstacleType.BAR: make_bar,
    ObstacleType.CROSS: make_cross,
    ObstacleType.SHIFTED_BLOCK: make_shifted_block,
}


def make_random_obstacle(
    rng: np.random.Generator,
    size: int,
    wall_height: int,
    is_pit: bool | None = None,
    pillar_weight: float | None = None
) -> np.ndarray:
    """Generate a random obstacle type.

    Args:
        rng: Random number generator.
        size: Size of the obstacle grid in pixels.
        wall_height: Height of walls in terrain units.
        is_pit: Force pit (True) or wall (False). None = random.
        pillar_weight: Weight for pillars (0-1). None = uniform distribution.
    """
    scale = rng.uniform(OBSTACLES.SCALE_MIN, OBSTACLES.SCALE_MAX)
    if is_pit is None:
        is_pit = rng.random() < OBSTACLES.DEFAULT_PIT_PROB
    thickness = rng.integers(OBSTACLES.THICKNESS_MIN, OBSTACLES.THICKNESS_MAX)

    # Select obstacle type (with optional pillar weighting)
    if pillar_weight is not None and pillar_weight > 0:
        # Weighted selection: pillar_weight for pillars, rest split evenly
        other_weight = (1.0 - pillar_weight) / (ObstacleType.NUM_TYPES - 1)
        weights = [other_weight] * ObstacleType.NUM_TYPES
        weights[ObstacleType.PILLAR] = pillar_weight
        obstacle_type = rng.choice(ObstacleType.NUM_TYPES, p=weights)
    else:
        # Uniform selection
        obstacle_type = rng.integers(ObstacleType.NUM_TYPES)

    generator = _OBSTACLE_GENERATORS[obstacle_type]
    return generator(rng, size, wall_height, scale, is_pit, thickness)


# =============================================================================
# Stair/Platform Generator
# =============================================================================

class StairGenerator:
    """Generates stair structures with platforms."""

    LAYOUTS = [
        {"platforms": [(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)],
         "stairs": [(0, 0, "n"), (2, 2, "s"), (0, 2, "s"), (2, 0, "w")]},
        {"platforms": [(0, 1), (1, 1), (2, 1)],
         "stairs": [(0, 0, "n"), (2, 2, "s"), (0, 2, "s"), (2, 0, "n")]},
        {"platforms": [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)],
         "stairs": [(0, 0, "n"), (2, 2, "s"), (0, 2, "s"), (2, 0, "n")]},
    ]

    def __init__(self, wall_height: float, vertical_scale: float):
        self.wall_height = wall_height
        self.platform_height = int(wall_height - 0.5 / vertical_scale)
        self.vertical_scale = vertical_scale
        self._make_stair_templates()

    def _make_stair_templates(self):
        """Create stair templates for each direction."""
        cell_px = STAIRS.SINGLE_CELL_PIXELS
        step_res = cell_px // STAIRS.NUM_STEPS

        east = np.zeros((cell_px, cell_px), dtype=np.float32)
        for i in range(STAIRS.NUM_STEPS):
            h = STAIRS.STEP_HEIGHT_METERS * (i + 1) / self.vertical_scale
            east[i * step_res:(i + 1) * step_res, :] = h

        self.templates = {
            "e": east,
            "n": rotate(east, 90),
            "w": rotate(east, 180),
            "s": rotate(east, 270),
        }

    def generate(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 3x3 stair/platform structure.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of (heights, valid_mask, platform_mask).
        """
        layout = self.LAYOUTS[rng.integers(len(self.LAYOUTS))]
        size = STAIRS.STAIR_GRID_SIZE * STAIRS.SINGLE_CELL_PIXELS
        cell_px = STAIRS.SINGLE_CELL_PIXELS

        heights = np.zeros((size, size), dtype=np.float32)
        valid_mask = np.zeros((size, size), dtype=bool)
        platform_mask = np.zeros((size, size), dtype=bool)

        for gx, gy in layout["platforms"]:
            xs, xe = gx * cell_px, (gx + 1) * cell_px
            ys, ye = gy * cell_px, (gy + 1) * cell_px
            heights[xs:xe, ys:ye] = self.platform_height
            valid_mask[xs:xe, ys:ye] = True
            platform_mask[xs:xe, ys:ye] = True

        for gx, gy, direction in layout["stairs"]:
            xs, xe = gx * cell_px, (gx + 1) * cell_px
            ys, ye = gy * cell_px, (gy + 1) * cell_px
            heights[xs:xe, ys:ye] = self.templates[direction]

        return heights, valid_mask, platform_mask


# =============================================================================
# Main Terrain Generation
# =============================================================================

def _get_rng(cfg: "hf_terrains_maze_cfg.HfMazeTerrainCfg") -> np.random.Generator:
    """Get RNG from config or create a new one."""
    if cfg.rng is not None:
        return cfg.rng
    # Fallback: create unseeded RNG (non-reproducible)
    return np.random.default_rng()


@height_field_to_mesh
def maze_terrain(difficulty: float, cfg: "hf_terrains_maze_cfg.HfMazeTerrainCfg") -> np.ndarray:
    """Generate maze terrain with obstacles and valid position mask.

    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.

    Returns:
        Height field for mesh generation.
    """
    rng = _get_rng(cfg)

    # Setup dimensions
    cell_pixels = int(cfg.cell_size / cfg.horizontal_scale)
    wall_height = int(cfg.wall_height / cfg.vertical_scale)
    terrain_w = int(cfg.size[0] / cfg.horizontal_scale)
    terrain_h = int(cfg.size[1] / cfg.horizontal_scale)

    terrain = TerrainData.create(terrain_w, terrain_h)
    stair_gen = StairGenerator(wall_height, cfg.vertical_scale)

    # Generate base pattern
    if cfg.non_maze_terrain:
        maze = np.zeros(cfg.grid_size, dtype=np.uint8)
        obstacle_prob = difficulty * OBSTACLES.NON_MAZE_DENSITY
        maze[rng.random(cfg.grid_size) < obstacle_prob] = 1
    else:
        maze = generate_maze(rng, cfg.grid_size[0], cfg.grid_size[1], 1 - difficulty)

    clear_center(maze, terrain, cell_pixels)

    # Generate terrain features based on type
    if cfg.dynamic_obstacles:
        _add_pits(rng, terrain, cfg, difficulty, wall_height, cell_pixels)
    elif cfg.stairs:
        _add_stairs(rng, terrain, cfg, difficulty, wall_height, cell_pixels, stair_gen)
    else:
        _add_walls(rng, maze, terrain, cfg, wall_height, cell_pixels)

    clear_center(maze, terrain, cell_pixels)

    # Apply height transition padding for stair terrain
    if cfg.stairs:
        terrain.apply_height_transition_padding(
            height_threshold=PADDING.HEIGHT_TRANSITION_THRESHOLD,
            padding_cells=PADDING.HEIGHT_TRANSITION_PADDING
        )

    # Apply safety padding and border exclusion
    terrain.apply_padding(PADDING.GOAL_PADDING)
    terrain.exclude_borders(PADDING.BORDER_CELLS)

    # Create spawn mask with larger padding
    spawn_mask = terrain.create_spawn_mask(PADDING.SPAWN_PADDING)
    spawn_mask[:PADDING.BORDER_CELLS, :] = False
    spawn_mask[-PADDING.BORDER_CELLS:, :] = False
    spawn_mask[:, :PADDING.BORDER_CELLS] = False
    spawn_mask[:, -PADDING.BORDER_CELLS:] = False

    # Store data on cfg for patches to pick up
    if cfg.add_goal:
        cfg.height_field_visual = torch.from_numpy(terrain.heights.copy()).unsqueeze(0)
        cfg.height_field_valid_mask = torch.from_numpy(terrain.valid_mask.copy()).unsqueeze(0)
        cfg.height_field_platform_mask = torch.from_numpy(terrain.platform_mask.copy()).unsqueeze(0)
        cfg.height_field_spawn_mask = torch.from_numpy(spawn_mask.copy()).unsqueeze(0)

    return terrain.heights


@height_field_to_mesh
def random_maze_terrain(difficulty: float, cfg: "hf_terrains_maze_cfg.HfRandomMazeTerrainCfg") -> np.ndarray:
    """Generate a random thin-wall maze with optional elevated stair patches."""
    return _random_maze_height_field(difficulty, cfg)


def _random_maze_height_field(difficulty: float, cfg) -> np.ndarray:
    rng = _get_rng(cfg)
    wall_height = int(cfg.wall_height / cfg.vertical_scale)
    terrain_w = int(cfg.size[0] / cfg.horizontal_scale)
    terrain_h = int(cfg.size[1] / cfg.horizontal_scale)
    terrain = TerrainData.create(terrain_w, terrain_h)
    terrain.valid_mask[:, :] = False

    wall_px = max(1, int(round(cfg.wall_width / cfg.horizontal_scale)))
    cell_px = max(wall_px + 2, int(round(cfg.cell_size / cfg.horizontal_scale)))
    corridor_width = cfg.corridor_width
    if getattr(cfg, "corridor_width_range", None) is not None:
        corridor_width = rng.uniform(cfg.corridor_width_range[0], cfg.corridor_width_range[1])
    max_clear_width = max(cfg.horizontal_scale, cfg.cell_size - 2.0 * cfg.wall_width)
    corridor_width = min(corridor_width, max_clear_width)
    corridor_px = max(wall_px + 2, int(round(corridor_width / cfg.horizontal_scale)))

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    rooms_x = max(5, min(cfg.grid_size[0], terrain_w // cell_px))
    rooms_y = max(5, min(cfg.grid_size[1], terrain_h // cell_px))
    maze_origin_x = max(0, (terrain_w - rooms_x * cell_px) // 2)
    maze_origin_y = max(0, (terrain_h - rooms_y * cell_px) // 2)
    wall_half = max(1, wall_px // 2)

    def set_wall_px(x0: int, x1: int, y0: int, y1: int):
        x0 = max(0, min(terrain_w, x0))
        x1 = max(0, min(terrain_w, x1))
        y0 = max(0, min(terrain_h, y0))
        y1 = max(0, min(terrain_h, y1))
        if x1 > x0 and y1 > y0:
            terrain.set_obstacle(x0, x1, y0, y1, wall_height)

    def set_valid_px(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int):
        x0 = max(0, min(terrain_w, x0))
        x1 = max(0, min(terrain_w, x1))
        y0 = max(0, min(terrain_h, y0))
        y1 = max(0, min(terrain_h, y1))
        if x1 > x0 and y1 > y0:
            mask[x0:x1, y0:y1] = True

    vertical_walls = np.ones((rooms_x + 1, rooms_y), dtype=bool)
    horizontal_walls = np.ones((rooms_x, rooms_y + 1), dtype=bool)
    visited = np.zeros((rooms_x, rooms_y), dtype=bool)
    passages = []
    stack = [(0, rooms_y // 2)]
    visited[stack[0]] = True
    previous_direction = (1, 0)

    def neighbors(cell):
        x, y = cell
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if rng.random() > 0.45:
            directions = [previous_direction] + [direction for direction in directions if direction != previous_direction]
        candidates = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rooms_x and 0 <= ny < rooms_y and not visited[nx, ny]:
                candidates.append((nx, ny, dx, dy))
        return candidates

    while stack:
        current = stack[-1]
        candidates = neighbors(current)
        if not candidates:
            stack.pop()
            continue
        nx, ny, dx, dy = candidates[int(rng.integers(0, len(candidates)))]
        x, y = current
        if dx == 1:
            vertical_walls[x + 1, y] = False
        elif dx == -1:
            vertical_walls[x, y] = False
        elif dy == 1:
            horizontal_walls[x, y + 1] = False
        elif dy == -1:
            horizontal_walls[x, y] = False
        passages.append((x, y, nx, ny))
        previous_direction = (dx, dy)
        visited[nx, ny] = True
        stack.append((nx, ny))

    # Extra openings reduce difficulty while preserving the maze-wall model.
    open_probability = min(1.0 - difficulty, 0.6)
    internal_vertical = rng.random(vertical_walls[1:-1].shape) < (0.25 * open_probability)
    vertical_walls[1:-1] &= ~internal_vertical
    internal_horizontal = rng.random(horizontal_walls[:, 1:-1].shape) < (0.25 * open_probability)
    horizontal_walls[:, 1:-1] &= ~internal_horizontal

    exits = _sample_maze_exits(rng, rooms_x, rooms_y, cfg)
    for exit_info in exits:
        side = exit_info["side"]
        index = exit_info["index"]
        if side == "west":
            vertical_walls[0, index] = False
        elif side == "east":
            vertical_walls[rooms_x, index] = False
        elif side == "south":
            horizontal_walls[index, 0] = False
        elif side == "north":
            horizontal_walls[index, rooms_y] = False

    for x in range(rooms_x + 1):
        px = maze_origin_x + x * cell_px
        for y in range(rooms_y):
            if vertical_walls[x, y]:
                py0 = maze_origin_y + y * cell_px
                py1 = maze_origin_y + (y + 1) * cell_px
                set_wall_px(px - wall_half, px + wall_half, py0 - wall_half, py1 + wall_half)

    for x in range(rooms_x):
        px0 = maze_origin_x + x * cell_px
        px1 = maze_origin_x + (x + 1) * cell_px
        for y in range(rooms_y + 1):
            if horizontal_walls[x, y]:
                py = maze_origin_y + y * cell_px
                set_wall_px(px0 - wall_half, px1 + wall_half, py - wall_half, py + wall_half)

    # Keep a visible outer safety boundary around the tile, then cut random outside-connected exits.
    set_wall_px(0, terrain_w, 0, wall_px)
    set_wall_px(0, terrain_w, terrain_h - wall_px, terrain_h)
    set_wall_px(0, wall_px, 0, terrain_h)
    set_wall_px(terrain_w - wall_px, terrain_w, 0, terrain_h)

    spawn_exit, goal_exits = _select_spawn_and_goal_exits(exits)

    goal_mask = np.zeros_like(terrain.valid_mask)
    spawn_mask = np.zeros_like(terrain.valid_mask)
    excluded_rooms = set()
    for exit_info in exits:
        door_rect = _maze_exit_door_rect(
            exit_info, terrain_w, terrain_h, maze_origin_x, maze_origin_y, cell_px, wall_px
        )
        exit_area_rect = _maze_exit_area_rect(
            exit_info, rooms_x, rooms_y, maze_origin_x, maze_origin_y, cell_px
        )
        terrain.set_ground(*door_rect)
        terrain.set_ground(*exit_area_rect)
        excluded_rooms.add(exit_info["room"])

    if getattr(cfg, "num_stairs", 0) > 0:
        _add_random_maze_stairs(
            rng,
            terrain,
            cfg,
            rooms_x,
            rooms_y,
            maze_origin_x,
            maze_origin_y,
            cell_px,
            wall_px,
            corridor_px,
            passages,
            excluded_rooms,
        )

    set_valid_px(
        spawn_mask,
        *_maze_exit_area_rect(spawn_exit, rooms_x, rooms_y, maze_origin_x, maze_origin_y, cell_px),
    )
    for goal_exit in goal_exits:
        set_valid_px(
            goal_mask,
            *_maze_exit_area_rect(goal_exit, rooms_x, rooms_y, maze_origin_x, maze_origin_y, cell_px),
        )
    terrain.valid_mask = goal_mask

    if cfg.add_goal:
        cfg.height_field_visual = torch.from_numpy(terrain.heights.copy()).unsqueeze(0)
        cfg.height_field_valid_mask = torch.from_numpy(terrain.valid_mask.copy()).unsqueeze(0)
        cfg.height_field_platform_mask = torch.from_numpy(terrain.platform_mask.copy()).unsqueeze(0)
        cfg.height_field_spawn_mask = torch.from_numpy(spawn_mask.copy()).unsqueeze(0)

    return terrain.heights


# =============================================================================
# Branching Corridor WFC Terrain
# =============================================================================

DIRECTION_TO_OFFSET = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_DIRECTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
OPENING_MASK_NAMES = {
    frozenset(): "closed",
    frozenset(("up",)): "dead_end_up",
    frozenset(("down",)): "dead_end_down",
    frozenset(("left",)): "dead_end_left",
    frozenset(("right",)): "dead_end_right",
    frozenset(("up", "down")): "straight_ud",
    frozenset(("left", "right")): "straight_lr",
    frozenset(("up", "right")): "turn_up_right",
    frozenset(("right", "down")): "turn_right_down",
    frozenset(("down", "left")): "turn_down_left",
    frozenset(("left", "up")): "turn_left_up",
    frozenset(("up", "down", "left")): "t_up_down_left",
    frozenset(("up", "down", "right")): "t_up_down_right",
    frozenset(("left", "right", "up")): "t_left_right_up",
    frozenset(("left", "right", "down")): "t_left_right_down",
    frozenset(("up", "down", "left", "right")): "cross",
}
STAIR_LEVELS = {
    ("left", 1): 11,
    ("right", 1): 12,
    ("up", 1): 13,
    ("down", 1): 14,
    ("left", 2): 21,
    ("right", 2): 22,
    ("up", 2): 23,
    ("down", 2): 24,
}


@dataclass(frozen=True)
class _BranchingTileSpec:
    name: str
    openings: dict[str, int]
    default_level: int
    weight: float


def _physical_level(level: int | None) -> int:
    return 2 if level == 2 or (level is not None and 20 <= level < 30) else 1


def _same_physical_level(first_level: int | None, second_level: int | None) -> bool:
    if first_level is None or second_level is None:
        return False
    return _physical_level(first_level) == _physical_level(second_level)


def _landing_level(direction: str, physical_level: int) -> int:
    return STAIR_LEVELS[(direction, physical_level)]


def _opening_shape_name(openings: dict[str, int]) -> str:
    return OPENING_MASK_NAMES[frozenset(openings)]


def _branching_tile_name(openings: dict[str, int], default_level: int) -> str:
    shape_name = _opening_shape_name(openings)
    if not openings:
        return "branch_corridor_closed"

    levels = set(openings.values())
    physical_levels = {_physical_level(level) for level in levels}
    if len(physical_levels) == 1 and len(levels) == 1:
        suffix = "" if next(iter(physical_levels)) == 1 else "_h1"
        return f"branch_corridor_{shape_name}{suffix}"

    if len(openings) == 2:
        low_dirs = [direction for direction, level in openings.items() if _physical_level(level) == 1]
        high_dirs = [direction for direction, level in openings.items() if _physical_level(level) == 2]
        if len(low_dirs) == 1 and len(high_dirs) == 1:
            transition = "up" if default_level == 2 else "down"
            return f"branch_corridor_stair_{low_dirs[0]}_{transition}"

    level_tokens = "_".join(f"{direction}{level}" for direction, level in sorted(openings.items()))
    return f"branch_corridor_{shape_name}_stair_{level_tokens}"


def _create_branching_tile_specs(stair_weight: float) -> list[_BranchingTileSpec]:
    specs_by_key: dict[tuple[frozenset[tuple[str, int]], int], _BranchingTileSpec] = {}
    direction_sets = [
        (),
        ("up",),
        ("down",),
        ("left",),
        ("right",),
        ("up", "down"),
        ("left", "right"),
        ("up", "right"),
        ("right", "down"),
        ("down", "left"),
        ("left", "up"),
        ("up", "down", "left"),
        ("up", "down", "right"),
        ("left", "right", "up"),
        ("left", "right", "down"),
        ("up", "down", "left", "right"),
    ]

    def add(openings: dict[str, int], default_level: int, weight: float):
        key = (frozenset(openings.items()), default_level)
        name = _branching_tile_name(openings, default_level)
        specs_by_key[key] = _BranchingTileSpec(name=name, openings=dict(openings), default_level=default_level, weight=weight)

    add({}, 1, 1.5)
    for directions in direction_sets[1:]:
        weight = 1.0
        if len(directions) == 1:
            weight = 0.35
        elif len(directions) >= 3:
            weight = 0.55
        add({direction: 1 for direction in directions}, 1, weight)

    return list(specs_by_key.values())


def _branching_openings_to_id(specs: list[_BranchingTileSpec]) -> dict[tuple[frozenset[tuple[str, int]], int], int]:
    mapping = {}
    for tile_id, spec in enumerate(specs):
        mapping[(frozenset(spec.openings.items()), spec.default_level)] = tile_id
        mapping[(frozenset(spec.openings.items()), _physical_level(spec.default_level))] = tile_id
        normalized_openings = frozenset(
            (direction, _physical_level(level)) for direction, level in spec.openings.items()
        )
        mapping[(normalized_openings, _physical_level(spec.default_level))] = tile_id
    return mapping


def _branching_name_to_id(specs: list[_BranchingTileSpec]) -> dict[str, int]:
    return {spec.name: tile_id for tile_id, spec in enumerate(specs)}


def _branching_id_by_openings(
    specs: list[_BranchingTileSpec],
    openings_to_id: dict[tuple[frozenset[tuple[str, int]], int], int],
    openings: dict[str, int],
    default_level: int,
) -> int | None:
    key = (frozenset(openings.items()), default_level)
    if key in openings_to_id:
        return openings_to_id[key]
    key = (frozenset(openings.items()), _physical_level(default_level))
    if key in openings_to_id:
        return openings_to_id[key]
    normalized_openings = frozenset((direction, _physical_level(level)) for direction, level in openings.items())
    key = (normalized_openings, _physical_level(default_level))
    return openings_to_id.get(key)


def _create_e_branch_seed(spine_x: int, top_y: int, branch_length: int) -> list[tuple[str, tuple[int, int]]]:
    init_tiles = []
    branch_rows = (top_y + 2, top_y + 4, top_y + 6)
    bottom_y = max(branch_rows) + 1
    init_tiles.append(("branch_corridor_dead_end_down", (top_y, spine_x)))
    for y in range(top_y + 1, bottom_y):
        tile_name = "branch_corridor_t_up_down_right" if y in branch_rows else "branch_corridor_straight_ud"
        init_tiles.append((tile_name, (y, spine_x)))
    init_tiles.append(("branch_corridor_dead_end_up", (bottom_y, spine_x)))

    for y in branch_rows:
        for dx in range(1, branch_length):
            init_tiles.append(("branch_corridor_straight_lr", (y, spine_x + dx)))
        init_tiles.append(("branch_corridor_dead_end_left", (y, spine_x + branch_length)))
    return init_tiles


def _compatible_opening(first: _BranchingTileSpec, second: _BranchingTileSpec, direction: str) -> bool:
    first_level = first.openings.get(direction)
    second_level = second.openings.get(OPPOSITE_DIRECTION[direction])
    return first_level == second_level


def _boundary_compatible(spec: _BranchingTileSpec, position: tuple[int, int], shape: tuple[int, int]) -> bool:
    y, x = position
    if y == 0 and "up" in spec.openings:
        return False
    if y == shape[0] - 1 and "down" in spec.openings:
        return False
    if x == 0 and "left" in spec.openings:
        return False
    if x == shape[1] - 1 and "right" in spec.openings:
        return False
    return True


def _solve_branching_wfc(
    rng: np.random.Generator,
    shape: tuple[int, int],
    specs: list[_BranchingTileSpec],
    seed_tiles: list[tuple[str, tuple[int, int]]],
    max_attempts: int = 16,
) -> np.ndarray:
    name_to_id = _branching_name_to_id(specs)
    all_ids = np.arange(len(specs), dtype=np.int32)
    base_domains: list[list[set[int]]] = []
    for y in range(shape[0]):
        row = []
        for x in range(shape[1]):
            row.append({int(tile_id) for tile_id in all_ids if _boundary_compatible(specs[int(tile_id)], (y, x), shape)})
        base_domains.append(row)

    for _attempt in range(max_attempts):
        domains = [[set(cell) for cell in row] for row in base_domains]
        failed = False
        for tile_name, (y, x) in seed_tiles:
            if not (0 <= y < shape[0] and 0 <= x < shape[1]) or tile_name not in name_to_id:
                failed = True
                break
            domains[y][x] = {name_to_id[tile_name]}
        if failed or not _propagate_branching_domains(domains, specs):
            continue

        while True:
            candidates = [
                (len(domains[y][x]), y, x)
                for y in range(shape[0])
                for x in range(shape[1])
                if len(domains[y][x]) > 1
            ]
            if not candidates:
                return np.array([[next(iter(domains[y][x])) for x in range(shape[1])] for y in range(shape[0])], dtype=np.int32)

            min_entropy = min(item[0] for item in candidates)
            _, y, x = candidates[int(rng.integers(0, sum(item[0] == min_entropy for item in candidates)))]
            min_cells = [(cy, cx) for entropy, cy, cx in candidates if entropy == min_entropy]
            y, x = min_cells[int(rng.integers(0, len(min_cells)))]
            tile_ids = np.array(sorted(domains[y][x]), dtype=np.int32)
            weights = np.array([specs[int(tile_id)].weight for tile_id in tile_ids], dtype=np.float64)
            weights /= weights.sum()
            domains[y][x] = {int(rng.choice(tile_ids, p=weights))}
            if not _propagate_branching_domains(domains, specs):
                break

    raise RuntimeError("Failed to generate branching corridor WFC terrain after retries.")


def _propagate_branching_domains(domains: list[list[set[int]]], specs: list[_BranchingTileSpec]) -> bool:
    shape = (len(domains), len(domains[0]))
    queue = deque((y, x) for y in range(shape[0]) for x in range(shape[1]))
    while queue:
        y, x = queue.popleft()
        if not domains[y][x]:
            return False
        for direction, (dy, dx) in DIRECTION_TO_OFFSET.items():
            ny, nx = y + dy, x + dx
            if not (0 <= ny < shape[0] and 0 <= nx < shape[1]):
                continue
            allowed_neighbor_ids = {
                neighbor_id
                for neighbor_id in domains[ny][nx]
                if any(_compatible_opening(specs[current_id], specs[neighbor_id], direction) for current_id in domains[y][x])
            }
            if allowed_neighbor_ids == domains[ny][nx]:
                continue
            domains[ny][nx] = allowed_neighbor_ids
            if not allowed_neighbor_ids:
                return False
            queue.append((ny, nx))
    return True


def _label_branching_components(wave: np.ndarray, specs: list[_BranchingTileSpec]) -> tuple[np.ndarray, int]:
    labels = -np.ones(wave.shape, dtype=np.int32)
    component_count = 0
    for start_y in range(wave.shape[0]):
        for start_x in range(wave.shape[1]):
            if labels[start_y, start_x] >= 0 or not specs[int(wave[start_y, start_x])].openings:
                continue
            queue = deque([(start_y, start_x)])
            labels[start_y, start_x] = component_count
            while queue:
                y, x = queue.popleft()
                openings = specs[int(wave[y, x])].openings
                for direction, level in openings.items():
                    dy, dx = DIRECTION_TO_OFFSET[direction]
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < wave.shape[0] and 0 <= nx < wave.shape[1]):
                        continue
                    other_openings = specs[int(wave[ny, nx])].openings
                    if labels[ny, nx] >= 0 or not _same_physical_level(
                        other_openings.get(OPPOSITE_DIRECTION[direction]), level
                    ):
                        continue
                    labels[ny, nx] = component_count
                    queue.append((ny, nx))
            component_count += 1
    return labels, component_count


def _repair_branching_connectivity(wave: np.ndarray, specs: list[_BranchingTileSpec]) -> np.ndarray:
    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)
    for _ in range(wave.size):
        labels, component_count = _label_branching_components(repaired_wave, specs)
        if component_count <= 1:
            break
        changed = False
        for y in range(repaired_wave.shape[0]):
            if changed:
                break
            for x in range(repaired_wave.shape[1]):
                current_id = int(repaired_wave[y, x])
                if labels[y, x] < 0:
                    continue
                for direction in ("right", "down"):
                    dy, dx = DIRECTION_TO_OFFSET[direction]
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < repaired_wave.shape[0] and 0 <= nx < repaired_wave.shape[1]):
                        continue
                    if labels[ny, nx] < 0 or labels[y, x] == labels[ny, nx]:
                        continue
                    neighbor_id = int(repaired_wave[ny, nx])
                    shared_level = specs[current_id].openings.get(direction, specs[current_id].default_level)
                    current_openings = dict(specs[current_id].openings)
                    neighbor_openings = dict(specs[neighbor_id].openings)
                    current_openings[direction] = shared_level
                    neighbor_openings[OPPOSITE_DIRECTION[direction]] = shared_level
                    current_replacement = _branching_id_by_openings(specs, openings_to_id, current_openings, shared_level)
                    neighbor_replacement = _branching_id_by_openings(specs, openings_to_id, neighbor_openings, shared_level)
                    if current_replacement is None or neighbor_replacement is None:
                        continue
                    repaired_wave[y, x] = current_replacement
                    repaired_wave[ny, nx] = neighbor_replacement
                    changed = True
                    break
                if changed:
                    break
        if not changed:
            break
    return repaired_wave


def _branching_direction_between(first: tuple[int, int], second: tuple[int, int]) -> str:
    dy = second[0] - first[0]
    dx = second[1] - first[1]
    for direction, offset in DIRECTION_TO_OFFSET.items():
        if offset == (dy, dx):
            return direction
    raise ValueError(f"Positions are not adjacent: {first}, {second}")


def _branching_manhattan_path(first: tuple[int, int], second: tuple[int, int]) -> list[tuple[int, int]]:
    y, x = first
    target_y, target_x = second
    path = [(y, x)]
    step_y = 1 if target_y >= y else -1
    while y != target_y:
        y += step_y
        path.append((y, x))
    step_x = 1 if target_x >= x else -1
    while x != target_x:
        x += step_x
        path.append((y, x))
    return path


def _carve_branching_path(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    openings_to_id: dict[tuple[frozenset[tuple[str, int]], int], int],
    path: list[tuple[int, int]],
    level: int = 1,
) -> None:
    path_openings = {position: {} for position in path}
    for first, second in zip(path[:-1], path[1:]):
        direction = _branching_direction_between(first, second)
        path_openings[first][direction] = level
        path_openings[second][OPPOSITE_DIRECTION[direction]] = level

    for position, openings in path_openings.items():
        existing_spec = specs[int(wave[position])]
        for direction, opening_level in existing_spec.openings.items():
            if _physical_level(opening_level) == level:
                openings.setdefault(direction, level)
        _replace_branching_openings(wave, specs, openings_to_id, position, openings, level)


def _connect_all_branching_components(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    seed_positions: set[tuple[int, int]],
) -> np.ndarray:
    """Carve low-level paths through fill cells until all corridor components are connected."""
    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)

    for _ in range(repaired_wave.size):
        labels, component_count = _label_branching_components(repaired_wave, specs)
        if component_count <= 1:
            break

        keep_label = None
        for position in seed_positions:
            if 0 <= position[0] < wave.shape[0] and 0 <= position[1] < wave.shape[1] and labels[position] >= 0:
                keep_label = int(labels[position])
                break
        if keep_label is None:
            component_sizes = np.bincount(labels[labels >= 0], minlength=component_count)
            keep_label = int(np.argmax(component_sizes))

        keep_positions = [tuple(position) for position in np.argwhere(labels == keep_label)]
        other_positions = [tuple(position) for position in np.argwhere((labels >= 0) & (labels != keep_label))]
        if not keep_positions or not other_positions:
            break

        best_pair = None
        best_distance = None
        for first in keep_positions:
            for second in other_positions:
                distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (first, second)
        if best_pair is None:
            break

        path = _branching_manhattan_path(best_pair[0], best_pair[1])
        _carve_branching_path(repaired_wave, specs, openings_to_id, path, level=1)
        repaired_wave = _sanitize_branching_openings(repaired_wave, specs)

    return repaired_wave


def _keep_seed_or_largest_branching_component(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    seed_positions: set[tuple[int, int]],
) -> np.ndarray:
    labels, component_count = _label_branching_components(wave, specs)
    if component_count <= 1:
        return wave

    keep_label = None
    for position in seed_positions:
        if 0 <= position[0] < wave.shape[0] and 0 <= position[1] < wave.shape[1] and labels[position] >= 0:
            keep_label = int(labels[position])
            break
    if keep_label is None:
        component_sizes = np.bincount(labels[labels >= 0], minlength=component_count)
        keep_label = int(np.argmax(component_sizes))

    repaired_wave = wave.copy()
    closed_id = _branching_name_to_id(specs)["branch_corridor_closed"]
    repaired_wave[(labels >= 0) & (labels != keep_label)] = closed_id
    return repaired_wave


def _sanitize_branching_openings(wave: np.ndarray, specs: list[_BranchingTileSpec]) -> np.ndarray:
    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)
    for _ in range(wave.size):
        changed = False
        for y in range(repaired_wave.shape[0]):
            for x in range(repaired_wave.shape[1]):
                current_id = int(repaired_wave[y, x])
                openings = dict(specs[current_id].openings)
                for direction, level in list(openings.items()):
                    dy, dx = DIRECTION_TO_OFFSET[direction]
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < repaired_wave.shape[0] and 0 <= nx < repaired_wave.shape[1]):
                        openings.pop(direction)
                        changed = True
                        continue
                    other_openings = specs[int(repaired_wave[ny, nx])].openings
                    if not _same_physical_level(other_openings.get(OPPOSITE_DIRECTION[direction]), level):
                        openings.pop(direction)
                        changed = True
                if openings != specs[current_id].openings:
                    replacement_id = _branching_id_by_openings(
                        specs, openings_to_id, openings, specs[current_id].default_level
                    )
                    if replacement_id is not None:
                        repaired_wave[y, x] = replacement_id
        if not changed:
            break
    return repaired_wave


def _branching_is_boundary_position(position: tuple[int, int], shape: tuple[int, int]) -> bool:
    y, x = position
    return y == 0 or x == 0 or y == shape[0] - 1 or x == shape[1] - 1


def _label_branching_fill_components(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
) -> list[set[tuple[int, int]]]:
    labels = -np.ones(wave.shape, dtype=np.int32)
    components = []
    for start_y in range(wave.shape[0]):
        for start_x in range(wave.shape[1]):
            if labels[start_y, start_x] >= 0 or specs[int(wave[start_y, start_x])].openings:
                continue
            component_id = len(components)
            component = set()
            queue = deque([(start_y, start_x)])
            labels[start_y, start_x] = component_id
            while queue:
                y, x = queue.popleft()
                component.add((y, x))
                for dy, dx in DIRECTION_TO_OFFSET.values():
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < wave.shape[0] and 0 <= nx < wave.shape[1]):
                        continue
                    if labels[ny, nx] >= 0 or specs[int(wave[ny, nx])].openings:
                        continue
                    labels[ny, nx] = component_id
                    queue.append((ny, nx))
            components.append(component)
    return components


def _choose_branching_fill_gate(
    component: set[tuple[int, int]],
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
) -> tuple[tuple[int, int], tuple[int, int], str, int] | None:
    candidates = []
    for position in sorted(component):
        y, x = position
        for direction, (dy, dx) in DIRECTION_TO_OFFSET.items():
            neighbor = (y + dy, x + dx)
            if not (0 <= neighbor[0] < wave.shape[0] and 0 <= neighbor[1] < wave.shape[1]):
                continue
            if neighbor in component:
                continue
            neighbor_spec = specs[int(wave[neighbor])]
            if not neighbor_spec.openings or _branching_tile_is_stair(neighbor_spec):
                continue
            level = _physical_level(neighbor_spec.default_level)
            if level != 1:
                continue
            candidates.append((position, neighbor, direction, level))
    return candidates[0] if candidates else None


def _connect_branching_fill_regions(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    connect_boundary_regions: bool = False,
) -> np.ndarray:
    """Open closed fill components and attach enclosed ones to the low-level corridor graph."""
    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)

    for component in _label_branching_fill_components(repaired_wave, specs):
        touches_boundary = any(_branching_is_boundary_position(position, repaired_wave.shape) for position in component)
        if touches_boundary and not connect_boundary_regions:
            continue

        gate = _choose_branching_fill_gate(component, repaired_wave, specs)
        if gate is None:
            continue
        gate_position, corridor_position, gate_direction, level = gate

        replacements = {}
        for position in component:
            y, x = position
            openings = {}
            for direction, (dy, dx) in DIRECTION_TO_OFFSET.items():
                neighbor = (y + dy, x + dx)
                if neighbor in component:
                    openings[direction] = level
            if position == gate_position:
                openings[gate_direction] = level
            replacement_id = _branching_id_by_openings(specs, openings_to_id, openings, level)
            if replacement_id is None:
                replacements = {}
                break
            replacements[position] = replacement_id
        if not replacements:
            continue

        corridor_direction = OPPOSITE_DIRECTION[gate_direction]
        corridor_openings = {
            direction: level
            for direction, opening_level in specs[int(repaired_wave[corridor_position])].openings.items()
            if _physical_level(opening_level) == level
        }
        corridor_openings[corridor_direction] = level
        corridor_replacement_id = _branching_id_by_openings(specs, openings_to_id, corridor_openings, level)
        if corridor_replacement_id is None:
            continue

        for position, replacement_id in replacements.items():
            repaired_wave[position] = replacement_id
        repaired_wave[corridor_position] = corridor_replacement_id

    return repaired_wave


def _replace_branching_openings(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    openings_to_id: dict[tuple[frozenset[tuple[str, int]], int], int],
    position: tuple[int, int],
    openings: dict[str, int],
    default_level: int,
) -> bool:
    replacement_id = _branching_id_by_openings(specs, openings_to_id, openings, default_level)
    if replacement_id is None:
        return False
    wave[position] = replacement_id
    return True


def _densify_branching_corridors(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    rng: np.random.Generator,
    seed_positions: set[tuple[int, int]],
    target_fraction: float = 0.55,
) -> np.ndarray:
    """Grow the seed component into nearby closed cells so generated tiles use more of the WFC area."""
    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)
    target_open_tiles = max(len(seed_positions), int(round(repaired_wave.size * target_fraction)))

    for _ in range(repaired_wave.size):
        labels, component_count = _label_branching_components(repaired_wave, specs)
        if component_count == 0:
            return repaired_wave

        keep_label = None
        for position in seed_positions:
            if 0 <= position[0] < wave.shape[0] and 0 <= position[1] < wave.shape[1] and labels[position] >= 0:
                keep_label = int(labels[position])
                break
        if keep_label is None:
            component_sizes = np.bincount(labels[labels >= 0], minlength=component_count)
            keep_label = int(np.argmax(component_sizes))

        component_positions = [tuple(position) for position in np.argwhere(labels == keep_label)]
        if len(component_positions) >= target_open_tiles:
            break

        frontier = []
        for y, x in component_positions:
            current_spec = specs[int(repaired_wave[y, x])]
            if _branching_tile_is_stair(current_spec):
                continue
            level = _physical_level(current_spec.default_level)
            for direction, (dy, dx) in DIRECTION_TO_OFFSET.items():
                ny, nx = y + dy, x + dx
                if not (0 <= ny < repaired_wave.shape[0] and 0 <= nx < repaired_wave.shape[1]):
                    continue
                if labels[ny, nx] == keep_label or specs[int(repaired_wave[ny, nx])].openings:
                    continue
                frontier.append((y, x, ny, nx, direction, level))

        if not frontier:
            break

        y, x, ny, nx, direction, level = frontier[int(rng.integers(0, len(frontier)))]
        current_openings = {
            opening_direction: level
            for opening_direction, opening_level in specs[int(repaired_wave[y, x])].openings.items()
            if _physical_level(opening_level) == level
        }
        neighbor_openings = {OPPOSITE_DIRECTION[direction]: level}
        current_openings[direction] = level
        if not _replace_branching_openings(repaired_wave, specs, openings_to_id, (y, x), current_openings, level):
            continue
        _replace_branching_openings(repaired_wave, specs, openings_to_id, (ny, nx), neighbor_openings, level)

    return repaired_wave


def _add_branching_outer_exits(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    seed_positions: set[tuple[int, int]],
    difficulty: float,
) -> tuple[np.ndarray, set[tuple[int, int, str]]]:
    """Carve low-level connected paths to every WFC grid side, with more exits at lower difficulty."""
    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)
    labels, component_count = _label_branching_components(repaired_wave, specs)
    if component_count == 0:
        return repaired_wave, set()

    keep_label = None
    for position in seed_positions:
        if 0 <= position[0] < wave.shape[0] and 0 <= position[1] < wave.shape[1] and labels[position] >= 0:
            keep_label = int(labels[position])
            break
    if keep_label is None:
        component_sizes = np.bincount(labels[labels >= 0], minlength=component_count)
        keep_label = int(np.argmax(component_sizes))

    exits = set()
    exit_counts = {
        direction: 1 + int((1.0 - difficulty) * max(1, wave.shape[axis] // 5))
        for direction, axis in (("up", 1), ("down", 1), ("left", 0), ("right", 0))
    }

    candidates_by_direction = {direction: [] for direction in DIRECTION_TO_OFFSET}
    for y, x in np.argwhere(labels == keep_label):
        spec = specs[int(repaired_wave[y, x])]
        if _branching_tile_is_stair(spec) or _physical_level(spec.default_level) != 1:
            continue
        distances = {
            "up": y,
            "down": repaired_wave.shape[0] - 1 - y,
            "left": x,
            "right": repaired_wave.shape[1] - 1 - x,
        }
        for direction, distance in distances.items():
            candidates_by_direction[direction].append((int(distance), int(y), int(x)))

    if not any(candidates_by_direction.values()):
        return repaired_wave, set()

    level = 1
    for exit_direction, candidates in candidates_by_direction.items():
        if not candidates:
            continue
        selected_candidates = sorted(candidates)[: exit_counts[exit_direction]]
        for _, start_y, start_x in selected_candidates:
            dy, dx = DIRECTION_TO_OFFSET[exit_direction]
            path = [(start_y, start_x)]
            y, x = start_y, start_x
            while 0 <= y + dy < repaired_wave.shape[0] and 0 <= x + dx < repaired_wave.shape[1]:
                y += dy
                x += dx
                path.append((y, x))

            for current, neighbor in zip(path[:-1], path[1:]):
                direction = exit_direction
                current_openings = {
                    opening_direction: level
                    for opening_direction, opening_level in specs[int(repaired_wave[current])].openings.items()
                    if _physical_level(opening_level) == level
                }
                neighbor_openings = {
                    opening_direction: level
                    for opening_direction, opening_level in specs[int(repaired_wave[neighbor])].openings.items()
                    if _physical_level(opening_level) == level
                }
                current_openings[direction] = level
                neighbor_openings[OPPOSITE_DIRECTION[direction]] = level
                _replace_branching_openings(repaired_wave, specs, openings_to_id, current, current_openings, level)
                _replace_branching_openings(repaired_wave, specs, openings_to_id, neighbor, neighbor_openings, level)

            exits.add((path[-1][0], path[-1][1], exit_direction))

    return repaired_wave, exits


def _add_branching_height_region(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    seed_positions: set[tuple[int, int]],
) -> np.ndarray:
    repaired_wave = wave.copy()
    labels, component_count = _label_branching_components(repaired_wave, specs)
    if component_count == 0:
        return repaired_wave
    component_sizes = np.bincount(labels[labels >= 0], minlength=component_count)
    corridor_positions = [tuple(position) for position in np.argwhere(labels == int(np.argmax(component_sizes)))]
    interior_positions = [
        position
        for position in corridor_positions
        if 0 < position[0] < wave.shape[0] - 1
        and 0 < position[1] < wave.shape[1] - 1
        and position not in seed_positions
    ]
    if len(interior_positions) < 5:
        return repaired_wave

    target_size = max(4, min(len(interior_positions), int(len(corridor_positions) * 0.4)))
    center = np.array([(wave.shape[0] - 1) * 0.5, (wave.shape[1] - 1) * 0.5])
    seed = min(interior_positions, key=lambda position: np.linalg.norm(np.array(position) - center))
    high_region = {seed}
    queue = deque([seed])
    while queue and len(high_region) < target_size:
        y, x = queue.popleft()
        for direction, (dy, dx) in DIRECTION_TO_OFFSET.items():
            ny, nx = y + dy, x + dx
            neighbor = (ny, nx)
            if neighbor in high_region or neighbor not in interior_positions or neighbor in seed_positions:
                continue
            openings = specs[int(repaired_wave[y, x])].openings
            other_openings = specs[int(repaired_wave[ny, nx])].openings
            if direction in openings and _same_physical_level(
                other_openings.get(OPPOSITE_DIRECTION[direction]), openings[direction]
            ):
                high_region.add(neighbor)
                queue.append(neighbor)
            if len(high_region) >= target_size:
                break

    if len(high_region) < 4:
        return repaired_wave

    openings_to_id = _branching_openings_to_id(specs)
    gate = None
    for high_position in sorted(high_region):
        high_openings = specs[int(repaired_wave[high_position])].openings
        for direction in sorted(high_openings):
            dy, dx = DIRECTION_TO_OFFSET[direction]
            low_position = (high_position[0] + dy, high_position[1] + dx)
            if low_position not in corridor_positions or low_position in high_region:
                gate = (high_position, low_position, direction)
                break
        if gate is not None:
            break
    if gate is None:
        return repaired_wave

    for position in corridor_positions:
        current_openings = specs[int(repaired_wave[position])].openings
        new_openings = {}
        if position in seed_positions:
            new_openings = {direction: 1 for direction in specs[int(wave[position])].openings}
        else:
            for direction in current_openings:
                dy, dx = DIRECTION_TO_OFFSET[direction]
                neighbor = (position[0] + dy, position[1] + dx)
                if (position in high_region and neighbor in high_region) or (
                    position not in high_region and neighbor not in high_region
                ):
                    new_openings[direction] = 2 if position in high_region else 1
        replacement_id = _branching_id_by_openings(
            specs, openings_to_id, new_openings, 2 if position in high_region else 1
        )
        if replacement_id is not None:
            repaired_wave[position] = replacement_id

    high_position, low_position, gate_direction = gate
    high_openings = {gate_direction: _landing_level(gate_direction, 1)}
    for direction in sorted(specs[int(wave[high_position])].openings):
        dy, dx = DIRECTION_TO_OFFSET[direction]
        neighbor = (high_position[0] + dy, high_position[1] + dx)
        if neighbor in high_region:
            high_openings[direction] = _landing_level(direction, 2)
            break
    high_openings[gate_direction] = _landing_level(gate_direction, 1)
    replacement_id = _branching_id_by_openings(specs, openings_to_id, high_openings, 2)
    if replacement_id is not None:
        repaired_wave[high_position] = replacement_id

    low_direction = OPPOSITE_DIRECTION[gate_direction]
    low_openings = {low_direction: _landing_level(low_direction, 1)}
    replacement_id = _branching_id_by_openings(specs, openings_to_id, low_openings, 1)
    if replacement_id is not None:
        repaired_wave[low_position] = replacement_id

    return repaired_wave


def _branching_tile_is_stair(spec: _BranchingTileSpec) -> bool:
    return len({_physical_level(level) for level in spec.openings.values()}) > 1


def _ensure_branching_stair(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    protected_positions: set[tuple[int, int]],
) -> np.ndarray:
    if any(_branching_tile_is_stair(specs[int(tile_id)]) for tile_id in wave.flat):
        return wave

    repaired_wave = wave.copy()
    openings_to_id = _branching_openings_to_id(specs)
    high_positions = [
        tuple(position)
        for position in np.argwhere(
            np.array([_physical_level(specs[int(tile_id)].default_level) == 2 for tile_id in repaired_wave.flat]).reshape(
                repaired_wave.shape
            )
        )
    ]
    for high_position in high_positions:
        if high_position in protected_positions:
            continue
        high_y, high_x = high_position
        high_spec = specs[int(repaired_wave[high_position])]
        if not high_spec.openings:
            continue
        for gate_direction in sorted(high_spec.openings):
            dy, dx = DIRECTION_TO_OFFSET[gate_direction]
            low_position = (high_y + dy, high_x + dx)
            if low_position in protected_positions or not (
                0 <= low_position[0] < repaired_wave.shape[0] and 0 <= low_position[1] < repaired_wave.shape[1]
            ):
                continue
            low_spec = specs[int(repaired_wave[low_position])]
            if _physical_level(low_spec.default_level) != 1:
                continue
            high_direction = next(
                (direction for direction in sorted(high_spec.openings) if direction != gate_direction),
                OPPOSITE_DIRECTION[gate_direction],
            )
            stair_openings = {
                gate_direction: _landing_level(gate_direction, 1),
                high_direction: _landing_level(high_direction, 2),
            }
            stair_id = _branching_id_by_openings(specs, openings_to_id, stair_openings, 2)
            low_id = _branching_id_by_openings(
                specs,
                openings_to_id,
                {OPPOSITE_DIRECTION[gate_direction]: _landing_level(OPPOSITE_DIRECTION[gate_direction], 1)},
                1,
            )
            if stair_id is None or low_id is None:
                continue
            repaired_wave[high_position] = stair_id
            repaired_wave[low_position] = low_id
            return repaired_wave
    return repaired_wave


def _branching_distances_from_seed(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    seed_positions: set[tuple[int, int]],
) -> np.ndarray:
    distances = -np.ones(wave.shape, dtype=np.int32)
    queue = deque()
    for position in seed_positions:
        if specs[int(wave[position])].openings:
            distances[position] = 0
            queue.append(position)
    while queue:
        y, x = queue.popleft()
        openings = specs[int(wave[y, x])].openings
        for direction, level in openings.items():
            dy, dx = DIRECTION_TO_OFFSET[direction]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < wave.shape[0] and 0 <= nx < wave.shape[1]) or distances[ny, nx] >= 0:
                continue
            other_openings = specs[int(wave[ny, nx])].openings
            if _same_physical_level(other_openings.get(OPPOSITE_DIRECTION[direction]), level):
                distances[ny, nx] = distances[y, x] + 1
                queue.append((ny, nx))
    return distances


def _rasterize_branching_wave(
    wave: np.ndarray,
    specs: list[_BranchingTileSpec],
    cfg,
    seed_positions: set[tuple[int, int]],
    outer_exits: set[tuple[int, int, str]],
    curriculum_difficulty: float,
) -> tuple[TerrainData, np.ndarray]:
    terrain_w = int(cfg.size[0] / cfg.horizontal_scale)
    terrain_h = int(cfg.size[1] / cfg.horizontal_scale)
    terrain = TerrainData.create(terrain_w, terrain_h)
    terrain.valid_mask[:, :] = False

    tile_px_x = max(3, int(round(cfg.tile_size / cfg.horizontal_scale)))
    tile_px_y = max(3, int(round(cfg.tile_size / cfg.horizontal_scale)))
    wall_px = max(1, int(round(cfg.wall_thickness / cfg.horizontal_scale)))
    wall_kernel_radius = max(1, int(round(wall_px * (0.45 + 0.55 * curriculum_difficulty))))
    corridor_half_x = max(1, tile_px_x // 2 - wall_px)
    corridor_half_y = max(1, tile_px_y // 2 - wall_px)
    wall_height = int(round(cfg.wall_height / cfg.vertical_scale))
    low_height = HEIGHTS.GROUND
    high_height = int(round(cfg.level_height / cfg.vertical_scale))
    step_height = max(1, int(round(cfg.step_height / cfg.vertical_scale)))

    grid_h, grid_w = wave.shape
    origin_x = max(0, (terrain_w - grid_h * tile_px_x) // 2)
    origin_y = max(0, (terrain_h - grid_w * tile_px_y) // 2)
    grid_x0 = origin_x
    grid_x1 = min(terrain_w, origin_x + grid_h * tile_px_x)
    grid_y0 = origin_y
    grid_y1 = min(terrain_h, origin_y + grid_w * tile_px_y)
    corridor_mask = np.zeros((terrain_w, terrain_h), dtype=bool)
    stair_mask = np.zeros_like(corridor_mask)
    spawn_mask = np.zeros_like(corridor_mask)
    exit_mask = np.zeros_like(corridor_mask)

    def tile_bounds(y: int, x: int) -> tuple[int, int, int, int, int, int]:
        x0 = origin_x + y * tile_px_x
        x1 = min(terrain_w, x0 + tile_px_x)
        y0 = origin_y + x * tile_px_y
        y1 = min(terrain_h, y0 + tile_px_y)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        return x0, x1, y0, y1, cx, cy

    def set_rect(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int, height: int):
        x0 = max(0, min(terrain_w, x0))
        x1 = max(0, min(terrain_w, x1))
        y0 = max(0, min(terrain_h, y0))
        y1 = max(0, min(terrain_h, y1))
        if x1 <= x0 or y1 <= y0:
            return
        terrain.heights[x0:x1, y0:y1] = height
        mask[x0:x1, y0:y1] = True

    distances = _branching_distances_from_seed(wave, specs, seed_positions)
    max_distance = int(distances.max(initial=0))
    goal_distance = max(2, int(max_distance * 0.6))

    for y in range(grid_h):
        for x in range(grid_w):
            spec = specs[int(wave[y, x])]
            if not spec.openings:
                continue
            x0, x1, y0, y1, cx, cy = tile_bounds(y, x)
            base_height = high_height if spec.default_level == 2 else low_height
            tile_mask = np.zeros_like(corridor_mask)
            set_rect(tile_mask, cx - corridor_half_x, cx + corridor_half_x, cy - corridor_half_y, cy + corridor_half_y, base_height)
            for direction in spec.openings:
                if direction == "up":
                    set_rect(tile_mask, x0, cx, cy - corridor_half_y, cy + corridor_half_y, base_height)
                elif direction == "down":
                    set_rect(tile_mask, cx, x1, cy - corridor_half_y, cy + corridor_half_y, base_height)
                elif direction == "left":
                    set_rect(tile_mask, cx - corridor_half_x, cx + corridor_half_x, y0, cy, base_height)
                elif direction == "right":
                    set_rect(tile_mask, cx - corridor_half_x, cx + corridor_half_x, cy, y1, base_height)

            corridor_mask |= tile_mask
            if _branching_tile_is_stair(spec):
                stair_mask |= tile_mask
                low_dirs = [direction for direction, level in spec.openings.items() if _physical_level(level) == 1]
                high_dirs = [direction for direction, level in spec.openings.items() if _physical_level(level) == 2]
                low_dir = low_dirs[0] if low_dirs else "left"
                high_dir = high_dirs[0] if high_dirs else OPPOSITE_DIRECTION[low_dir]
                xs, ys = np.nonzero(tile_mask)
                if low_dir in ("up", "down") or high_dir in ("up", "down"):
                    progress = (xs - x0) / max(1, x1 - x0 - 1)
                    if low_dir == "down":
                        progress = 1.0 - progress
                else:
                    progress = (ys - y0) / max(1, y1 - y0 - 1)
                    if low_dir == "right":
                        progress = 1.0 - progress
                raw_heights = np.rint(progress * high_height).astype(np.int32)
                stepped_heights = (raw_heights // step_height) * step_height
                terrain.heights[xs, ys] = np.clip(stepped_heights, low_height, high_height).astype(np.int16)
            elif spec.default_level == 2:
                terrain.platform_mask |= tile_mask

            stable_tile_mask = tile_mask & ~stair_mask
            if distances[y, x] >= goal_distance:
                terrain.valid_mask |= stable_tile_mask
            if distances[y, x] in (0, 1) and spec.default_level == 1:
                spawn_mask |= stable_tile_mask

    for y, x, direction in outer_exits:
        if not (0 <= y < grid_h and 0 <= x < grid_w):
            continue
        x0, x1, y0, y1, cx, cy = tile_bounds(y, x)
        gap_half_x = corridor_half_x + wall_kernel_radius
        gap_half_y = corridor_half_y + wall_kernel_radius
        if direction == "up":
            set_rect(exit_mask, x0 - wall_kernel_radius, x0 + wall_kernel_radius, cy - gap_half_y, cy + gap_half_y, low_height)
        elif direction == "down":
            set_rect(exit_mask, x1 - wall_kernel_radius, x1 + wall_kernel_radius, cy - gap_half_y, cy + gap_half_y, low_height)
        elif direction == "left":
            set_rect(exit_mask, cx - gap_half_x, cx + gap_half_x, y0 - wall_kernel_radius, y0 + wall_kernel_radius, low_height)
        elif direction == "right":
            set_rect(exit_mask, cx - gap_half_x, cx + gap_half_x, y1 - wall_kernel_radius, y1 + wall_kernel_radius, low_height)

    wall_kernel = np.ones((2 * wall_kernel_radius + 1, 2 * wall_kernel_radius + 1), dtype=bool)
    wall_mask = binary_dilation(corridor_mask, structure=wall_kernel) & ~corridor_mask
    wall_mask &= ~exit_mask
    outer_wall_band = np.zeros_like(wall_mask)
    band_width = max(wall_kernel_radius + wall_px, 2 * wall_px)
    outer_wall_band[
        max(0, grid_x0 - band_width):min(terrain_w, grid_x0 + band_width),
        max(0, grid_y0 - band_width):min(terrain_h, grid_y1 + band_width),
    ] = True
    outer_wall_band[
        max(0, grid_x1 - band_width):min(terrain_w, grid_x1 + band_width),
        max(0, grid_y0 - band_width):min(terrain_h, grid_y1 + band_width),
    ] = True
    outer_wall_band[
        max(0, grid_x0 - band_width):min(terrain_w, grid_x1 + band_width),
        max(0, grid_y0 - band_width):min(terrain_h, grid_y0 + band_width),
    ] = True
    outer_wall_band[
        max(0, grid_x0 - band_width):min(terrain_w, grid_x1 + band_width),
        max(0, grid_y1 - band_width):min(terrain_h, grid_y1 + band_width),
    ] = True
    wall_mask &= ~outer_wall_band
    terrain.heights[wall_mask] = wall_height

    invalid_for_padding = wall_mask | stair_mask
    kernel = np.ones((2 * PADDING.GOAL_PADDING + 1, 2 * PADDING.GOAL_PADDING + 1), dtype=bool)
    terrain.valid_mask &= ~binary_dilation(invalid_for_padding, structure=kernel)
    spawn_kernel = np.ones((2 * PADDING.SPAWN_PADDING + 1, 2 * PADDING.SPAWN_PADDING + 1), dtype=bool)
    spawn_mask &= ~binary_dilation(invalid_for_padding, structure=spawn_kernel)
    terrain.exclude_borders(PADDING.BORDER_CELLS)
    spawn_mask[:PADDING.BORDER_CELLS, :] = False
    spawn_mask[-PADDING.BORDER_CELLS:, :] = False
    spawn_mask[:, :PADDING.BORDER_CELLS] = False
    spawn_mask[:, -PADDING.BORDER_CELLS:] = False

    if not terrain.valid_mask.any():
        terrain.valid_mask = corridor_mask & ~stair_mask & ~spawn_mask
    if not spawn_mask.any():
        spawn_mask = corridor_mask & ~stair_mask & (terrain.heights == low_height)

    return terrain, spawn_mask


def _branching_corridors_height_field(difficulty: float, cfg) -> np.ndarray:
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    rng = _get_rng(cfg)
    specs = _create_branching_tile_specs(cfg.stair_weight)
    seed_tiles = _create_e_branch_seed(cfg.spine_x, cfg.top_y, cfg.branch_length)
    seed_positions = {position for _, position in seed_tiles}
    spawn_protected_positions = {
        position
        for position in seed_positions
        if position[1] == cfg.spine_x
    }
    target_fraction = 0.78 - 0.18 * difficulty
    connect_boundary_regions = difficulty < 0.65
    wave = _solve_branching_wfc(rng, cfg.grid_size, specs, seed_tiles)
    wave = _repair_branching_connectivity(wave, specs)
    wave = _connect_all_branching_components(wave, specs, seed_positions)
    wave = _sanitize_branching_openings(wave, specs)
    wave = _connect_branching_fill_regions(wave, specs, connect_boundary_regions=connect_boundary_regions)
    wave = _repair_branching_connectivity(wave, specs)
    wave = _connect_all_branching_components(wave, specs, seed_positions)
    wave = _sanitize_branching_openings(wave, specs)
    wave = _densify_branching_corridors(wave, specs, rng, seed_positions, target_fraction=target_fraction)
    wave = _connect_branching_fill_regions(wave, specs, connect_boundary_regions=connect_boundary_regions)
    wave = _repair_branching_connectivity(wave, specs)
    wave = _connect_all_branching_components(wave, specs, seed_positions)
    wave = _sanitize_branching_openings(wave, specs)
    wave, outer_exits = _add_branching_outer_exits(wave, specs, seed_positions, difficulty)
    wave = _sanitize_branching_openings(wave, specs)
    terrain, spawn_mask = _rasterize_branching_wave(wave, specs, cfg, seed_positions, outer_exits, difficulty)

    if cfg.add_goal:
        cfg.height_field_visual = torch.from_numpy(terrain.heights.copy()).unsqueeze(0)
        cfg.height_field_valid_mask = torch.from_numpy(terrain.valid_mask.copy()).unsqueeze(0)
        cfg.height_field_platform_mask = torch.from_numpy(terrain.platform_mask.copy()).unsqueeze(0)
        cfg.height_field_spawn_mask = torch.from_numpy(spawn_mask.copy()).unsqueeze(0)

    return terrain.heights


@height_field_to_mesh
def branching_corridors_terrain(difficulty: float, cfg: "hf_terrains_maze_cfg.HfBranchingCorridorsTerrainCfg") -> np.ndarray:
    """Generate a WFC branching-corridor terrain with explicit spawn and goal masks."""
    return _branching_corridors_height_field(difficulty, cfg)


def _sample_maze_exits(
    rng: np.random.Generator,
    rooms_x: int,
    rooms_y: int,
    cfg,
) -> list[dict]:
    tile_row = getattr(cfg, "terrain_row", None)
    tile_col = getattr(cfg, "terrain_col", None)
    num_rows = getattr(cfg, "terrain_num_rows", None)
    num_cols = getattr(cfg, "terrain_num_cols", None)
    if tile_row is None or tile_col is None or num_rows is None or num_cols is None:
        return _sample_standalone_maze_exits(rng, rooms_x, rooms_y, cfg.exit_count_range)

    exits = []
    seen = set()

    def add_exit(side: str, index: int, room: tuple[int, int]):
        key = (side, index)
        if key not in seen:
            exits.append({"side": side, "index": index, "room": room})
            seen.add(key)

    # Shared internal boundaries: both neighboring tiles compute the same index from the same boundary key.
    if tile_row > 0:
        y_index = _shared_maze_exit_index("x", tile_row, tile_col, rooms_y)
        add_exit("west", y_index, (0, y_index))
    if tile_row < num_rows - 1:
        y_index = _shared_maze_exit_index("x", tile_row + 1, tile_col, rooms_y)
        add_exit("east", y_index, (rooms_x - 1, y_index))
    if tile_col > 0:
        x_index = _shared_maze_exit_index("y", tile_row, tile_col, rooms_x)
        add_exit("south", x_index, (x_index, 0))
    if tile_col < num_cols - 1:
        x_index = _shared_maze_exit_index("y", tile_row, tile_col + 1, rooms_x)
        add_exit("north", x_index, (x_index, rooms_y - 1))

    exterior_candidates = []
    y_indices = range(1, max(1, rooms_y - 1))
    x_indices = range(1, max(1, rooms_x - 1))
    if tile_row == 0:
        exterior_candidates.extend({"side": "west", "index": y, "room": (0, y)} for y in y_indices)
    if tile_row == num_rows - 1:
        exterior_candidates.extend({"side": "east", "index": y, "room": (rooms_x - 1, y)} for y in y_indices)
    if tile_col == 0:
        exterior_candidates.extend({"side": "south", "index": x, "room": (x, 0)} for x in x_indices)
    if tile_col == num_cols - 1:
        exterior_candidates.extend({"side": "north", "index": x, "room": (x, rooms_y - 1)} for x in x_indices)

    rng.shuffle(exterior_candidates)
    min_exits, max_exits = cfg.exit_count_range
    min_exits = max(2, int(min_exits))
    max_exits = max(min_exits, int(max_exits))
    exterior_count = min(len(exterior_candidates), int(rng.integers(min_exits, max_exits + 1)))
    exterior_added = 0
    for candidate in exterior_candidates:
        if exterior_added >= exterior_count:
            break
        key = (candidate["side"], candidate["index"])
        if key in seen:
            continue
        exits.append(candidate)
        seen.add(key)
        exterior_added += 1

    if len(exits) < 2:
        return _sample_standalone_maze_exits(rng, rooms_x, rooms_y, cfg.exit_count_range)
    return exits


def _sample_standalone_maze_exits(
    rng: np.random.Generator,
    rooms_x: int,
    rooms_y: int,
    exit_count_range: tuple[int, int],
) -> list[dict]:
    min_exits, max_exits = exit_count_range
    min_exits = max(2, int(min_exits))
    max_exits = max(min_exits, int(max_exits))
    candidates = []
    y_indices = range(1, max(1, rooms_y - 1))
    x_indices = range(1, max(1, rooms_x - 1))
    for y in y_indices:
        candidates.append({"side": "west", "index": y, "room": (0, y)})
        candidates.append({"side": "east", "index": y, "room": (rooms_x - 1, y)})
    for x in x_indices:
        candidates.append({"side": "south", "index": x, "room": (x, 0)})
        candidates.append({"side": "north", "index": x, "room": (x, rooms_y - 1)})

    rng.shuffle(candidates)
    exit_count = min(len(candidates), int(rng.integers(min_exits, max_exits + 1)))
    selected = []
    selected_rooms = set()
    for candidate in candidates:
        if candidate["room"] in selected_rooms:
            continue
        selected.append(candidate)
        selected_rooms.add(candidate["room"])
        if len(selected) >= exit_count:
            break
    return selected


def _shared_maze_exit_index(axis: str, boundary_row: int, boundary_col: int, num_indices: int) -> int:
    span = max(1, num_indices - 2)
    axis_value = 0 if axis == "x" else 1
    value = 2166136261
    for item in (axis_value, boundary_row + 1, boundary_col + 1):
        value = ((value ^ item) * 16777619) & 0xFFFFFFFF
    return 1 + int(value % span)


def _select_spawn_and_goal_exits(exits: list[dict]) -> tuple[dict, list[dict]]:
    farthest_pair = (exits[0], exits[1])
    farthest_dist = -1
    for first in exits:
        for second in exits:
            if first is second:
                continue
            first_x, first_y = first["room"]
            second_x, second_y = second["room"]
            dist = abs(first_x - second_x) + abs(first_y - second_y)
            if dist > farthest_dist:
                farthest_dist = dist
                farthest_pair = (first, second)
    spawn_exit, primary_goal_exit = farthest_pair
    goal_exits = [exit_info for exit_info in exits if exit_info is not spawn_exit]
    if primary_goal_exit in goal_exits:
        goal_exits.remove(primary_goal_exit)
        goal_exits.insert(0, primary_goal_exit)
    return spawn_exit, goal_exits


def _maze_exit_door_rect(
    exit_info: dict,
    terrain_w: int,
    terrain_h: int,
    origin_x: int,
    origin_y: int,
    cell_px: int,
    wall_px: int,
) -> tuple[int, int, int, int]:
    side = exit_info["side"]
    index = exit_info["index"]
    half_gap = max(1, (cell_px - 2 * wall_px) // 2)
    if side in ("west", "east"):
        center_y = origin_y + int((index + 0.5) * cell_px)
        y0, y1 = center_y - half_gap, center_y + half_gap
        if side == "west":
            return 0, wall_px, y0, y1
        return terrain_w - wall_px, terrain_w, y0, y1
    center_x = origin_x + int((index + 0.5) * cell_px)
    x0, x1 = center_x - half_gap, center_x + half_gap
    if side == "south":
        return x0, x1, 0, wall_px
    return x0, x1, terrain_h - wall_px, terrain_h


def _maze_exit_area_rect(
    exit_info: dict,
    rooms_x: int,
    rooms_y: int,
    origin_x: int,
    origin_y: int,
    cell_px: int,
) -> tuple[int, int, int, int]:
    room_x, room_y = exit_info["room"]
    side = exit_info["side"]
    half_gap = max(1, cell_px // 3)
    if side in ("west", "east"):
        center_y = origin_y + int((room_y + 0.5) * cell_px)
        y0, y1 = center_y - half_gap, center_y + half_gap
        if side == "west":
            x0, x1 = origin_x + room_x * cell_px, origin_x + min(rooms_x, room_x + 2) * cell_px
        else:
            x0, x1 = origin_x + max(0, room_x - 1) * cell_px, origin_x + (room_x + 1) * cell_px
    else:
        center_x = origin_x + int((room_x + 0.5) * cell_px)
        x0, x1 = center_x - half_gap, center_x + half_gap
        if side == "south":
            y0, y1 = origin_y + room_y * cell_px, origin_y + min(rooms_y, room_y + 2) * cell_px
        else:
            y0, y1 = origin_y + max(0, room_y - 1) * cell_px, origin_y + (room_y + 1) * cell_px
    return x0, x1, y0, y1


def _add_random_maze_stairs(
    rng: np.random.Generator,
    terrain: TerrainData,
    cfg,
    rooms_x: int,
    rooms_y: int,
    origin_x: int,
    origin_y: int,
    cell_px: int,
    wall_px: int,
    corridor_px: int,
    passages: list[tuple[int, int, int, int]],
    excluded_rooms: set[tuple[int, int]],
) -> None:
    max_patches = max(1, int(cfg.num_stairs))
    num_patches = int(rng.integers(1, min(3, max_patches) + 1))
    step_height_min, step_height_max = cfg.step_height_range
    step_width_min, step_width_max = cfg.step_width_range
    patch_size_min, patch_size_max = cfg.pyramid_patch_size_range
    min_levels, max_levels = cfg.pyramid_levels_range
    platform_fraction_min, platform_fraction_max = cfg.pyramid_platform_fraction_range
    height_overlay = np.zeros_like(terrain.heights)
    placed = 0

    for _ in range(num_patches * 4):
        if placed >= num_patches:
            break
        patch_size_m = rng.uniform(patch_size_min, patch_size_max)
        patch_px = max(3, int(round(patch_size_m / cfg.horizontal_scale)))
        if patch_px < 3:
            continue

        maze_x0 = origin_x + cell_px
        maze_x1 = origin_x + (rooms_x - 1) * cell_px
        maze_y0 = origin_y + cell_px
        maze_y1 = origin_y + (rooms_y - 1) * cell_px
        if maze_x1 - maze_x0 <= patch_px or maze_y1 - maze_y0 <= patch_px:
            continue

        x0 = int(rng.integers(maze_x0, maze_x1 - patch_px + 1))
        y0 = int(rng.integers(maze_y0, maze_y1 - patch_px + 1))
        x1 = x0 + patch_px
        y1 = y0 + patch_px
        if x0 < 0 or y0 < 0 or x1 > terrain.heights.shape[0] or y1 > terrain.heights.shape[1]:
            continue

        step_height_units = max(1, int(rng.uniform(step_height_min, step_height_max) / cfg.vertical_scale))
        step_width_px = max(1, int(rng.uniform(step_width_min, step_width_max) / cfg.horizontal_scale))
        platform_fraction = rng.uniform(platform_fraction_min, platform_fraction_max)
        platform_px = max(1, int(round(patch_px * platform_fraction)))
        edge_band_px = max(1, (patch_px - platform_px) // 2)
        max_possible_levels = max(1, edge_band_px // step_width_px + 1)
        sampled_levels = int(rng.integers(min_levels, max_levels + 1))
        num_levels = max(1, min(sampled_levels, max_possible_levels))

        yy, xx = np.ogrid[:patch_px, :patch_px]
        distance_from_edge = np.minimum(
            np.minimum(xx, yy),
            np.minimum(patch_px - 1 - xx, patch_px - 1 - yy),
        )
        levels = np.floor(distance_from_edge / step_width_px).astype(np.int16) + 1
        levels = np.clip(levels, 0, num_levels)
        heights = levels * step_height_units
        height_overlay[x0:x1, y0:y1] = np.maximum(
            height_overlay[x0:x1, y0:y1],
            heights.astype(height_overlay.dtype),
        )
        terrain.valid_mask[x0:x1, y0:y1] = False
        placed += 1

    terrain.heights += height_overlay


# =============================================================================
# Terrain Type Generators
# =============================================================================

def _add_walls(
    rng: np.random.Generator,
    maze: np.ndarray,
    terrain: TerrainData,
    cfg,
    wall_height: int,
    cell_pixels: int
):
    """Add wall obstacles to terrain based on maze pattern."""
    # Use pillar weighting for non-maze terrain (more thin pillars)
    pillar_weight = OBSTACLES.NON_MAZE_PILLAR_WEIGHT if cfg.non_maze_terrain else None

    for x in range(cfg.grid_size[0]):
        for y in range(cfg.grid_size[1]):
            if maze[x, y] != 1:
                continue

            xs, xe, ys, ye = get_cell_bounds(
                x, y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
            )

            if cfg.randomize_wall and rng.random() < cfg.random_wall_ratio:
                obs = make_random_obstacle(rng, cell_pixels, wall_height, pillar_weight=pillar_weight)
                terrain.heights[xs:xe, ys:ye] = obs[:xe - xs, :ye - ys]
                terrain.valid_mask[xs:xe, ys:ye] = False
            else:
                h = int(wall_height * rng.uniform(OBSTACLES.SCALE_MIN, OBSTACLES.SCALE_MAX))
                terrain.set_obstacle(xs, xe, ys, ye, h)


def _add_stairs(
    rng: np.random.Generator,
    terrain: TerrainData,
    cfg,
    difficulty: float,
    wall_height: int,
    cell_pixels: int,
    stair_gen: StairGenerator
):
    """Add stair/platform structures to terrain."""
    grid_w, grid_h = cfg.grid_size
    grid_middle = grid_w // 2
    excluded = set(range(grid_middle - 1, grid_middle + 1))

    # Compute stair placement locations (avoid center and edges)
    # Stairs are 3x3, so max position is grid_size - 4 to fit with margin
    stair_margin = 1
    max_x = grid_w - STAIRS.STAIR_GRID_SIZE - stair_margin
    max_y = grid_h - STAIRS.STAIR_GRID_SIZE - stair_margin
    num_locations = 6
    x_locs = set(np.round(np.linspace(stair_margin, max_x, num_locations)).astype(int)) - excluded
    y_locs = set(np.round(np.linspace(stair_margin, max_y, num_locations)).astype(int)) - excluded

    processed = set()
    stair_size = STAIRS.STAIR_GRID_SIZE * STAIRS.SINGLE_CELL_PIXELS
    stair_prob = difficulty * OBSTACLES.STAIRS_PLACEMENT_PROB
    obstacle_prob = difficulty * OBSTACLES.STAIRS_OBSTACLE_DENSITY

    for x in range(grid_w):
        for y in range(grid_h):
            if (x, y) in processed:
                continue

            # Try placing stair structure at valid locations
            if x in x_locs and y in y_locs and rng.random() < stair_prob:
                heights, valid, platform = stair_gen.generate(rng)

                xs = x * cell_pixels
                xe = min(terrain.heights.shape[0], xs + stair_size)
                ys = y * cell_pixels
                ye = min(terrain.heights.shape[1], ys + stair_size)

                sx, sy = xe - xs, ye - ys
                terrain.heights[xs:xe, ys:ye] = heights[:sx, :sy]
                terrain.valid_mask[xs:xe, ys:ye] = valid[:sx, :sy]
                terrain.platform_mask[xs:xe, ys:ye] = platform[:sx, :sy]

                # Mark 3x3 area as processed
                for dx in range(3):
                    for dy in range(3):
                        processed.add((x + dx, y + dy))

            elif rng.random() < obstacle_prob:
                xs, xe, ys, ye = get_cell_bounds(
                    x, y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
                )
                # Check if area is clear before placing
                if terrain.valid_mask[xs + 1:xe - 1, ys + 1:ye - 1].all():
                    obs = make_random_obstacle(rng, cell_pixels, wall_height)
                    terrain.heights[xs:xe, ys:ye] = obs[:xe - xs, :ye - ys]
                    terrain.valid_mask[xs:xe, ys:ye] = False


def _add_pits(
    rng: np.random.Generator,
    terrain: TerrainData,
    cfg,
    difficulty: float,
    wall_height: int,
    cell_pixels: int
):
    """Add pit/trough obstacles to terrain.

    Layout:
    - Two horizontal pit trenches with random bridges for crossing
    - Random obstacles (mostly pits) scattered in the middle area
    """
    grid_w, grid_h = cfg.grid_size

    # Pit trench rows (near top and bottom)
    trench_offset = OBSTACLES.PITS_TRENCH_ROW_OFFSET
    pit_rows = {trench_offset, grid_h - trench_offset - 1}

    # Generate bridge positions for crossing pit trenches
    bridges = _generate_bridges(rng, grid_w)

    # Add pit trenches (negative height = troughs)
    for pit_y in pit_rows:
        for x in range(grid_w):
            if x in bridges:
                continue
            xs, xe, ys, ye = get_cell_bounds(
                x, pit_y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
            )
            terrain.set_obstacle(xs, xe, ys, ye, -wall_height)

    # Add random obstacles in middle area (between pit trenches)
    _add_middle_obstacles(rng, terrain, cfg, difficulty, wall_height, cell_pixels, pit_rows)


def _generate_bridges(rng: np.random.Generator, grid_width: int) -> set:
    """Generate bridge positions across pit rows.

    Returns set of x-coordinates where bridges (gaps in pits) are placed.
    Bridges are 2 cells wide for easier robot crossing.
    """
    num_bridges = rng.integers(OBSTACLES.BRIDGE_COUNT_MIN, OBSTACLES.BRIDGE_COUNT_MAX)
    margin = OBSTACLES.PITS_EDGE_MARGIN
    available = list(range(margin, grid_width - margin))
    rng.shuffle(available)

    bridges = set()
    for i in range(min(num_bridges, len(available))):
        pos = available[i]
        bridges.add(pos)
        # Make bridges 2 cells wide
        if pos + 1 < grid_width - margin:
            bridges.add(pos + 1)

    return bridges


def _add_middle_obstacles(
    rng: np.random.Generator,
    terrain: TerrainData,
    cfg,
    difficulty: float,
    wall_height: int,
    cell_pixels: int,
    pit_rows: set
):
    """Add random obstacles in the middle area between pit rows."""
    grid_w, grid_h = cfg.grid_size
    obstacle_prob = difficulty * OBSTACLES.PITS_DENSITY

    # Compute valid placement bounds (avoid edges and pit rows)
    margin = OBSTACLES.PITS_EDGE_MARGIN
    trench_offset = OBSTACLES.PITS_TRENCH_ROW_OFFSET

    x_range = range(margin, grid_w - margin)
    # Middle area: between the two pit trenches, with 1 cell buffer
    y_range = range(trench_offset + 1, grid_h - trench_offset - 1)

    # Iterate only over valid cells (more efficient)
    for x in x_range:
        for y in y_range:
            if y in pit_rows:
                continue

            if rng.random() < obstacle_prob:
                xs, xe, ys, ye = get_cell_bounds(
                    x, y, cell_pixels, terrain.heights.shape[0], terrain.heights.shape[1]
                )
                obs = _generate_pit_obstacle(rng, cell_pixels, wall_height)
                terrain.heights[xs:xe, ys:ye] = obs[:xe - xs, :ye - ys]
                terrain.valid_mask[xs:xe, ys:ye] = False


def _generate_pit_obstacle(
    rng: np.random.Generator,
    cell_pixels: int,
    wall_height: int
) -> np.ndarray:
    """Generate an obstacle for pit terrain with high pit probability.

    Distribution:
    - 60% bars (75% negative/pits) -> 45% pit bars
    - 40% random shapes (50% negative/pits) -> 20% pit shapes
    - Total: ~65% negative obstacles
    """
    if rng.random() < OBSTACLES.PITS_BAR_RATIO:
        # Bar obstacle with high pit probability
        is_pit = rng.random() < OBSTACLES.PITS_BAR_PIT_PROB
        scale = rng.uniform(OBSTACLES.SCALE_MIN, OBSTACLES.SCALE_MAX)
        thickness = rng.integers(OBSTACLES.THICKNESS_MIN, OBSTACLES.THICKNESS_MAX)
        return make_bar(rng, cell_pixels, wall_height, scale, is_pit, thickness)
    else:
        # Random obstacle type (pillar, cross, block) with moderate pit probability
        is_pit = rng.random() < OBSTACLES.PITS_RANDOM_PIT_PROB
        return make_random_obstacle(rng, cell_pixels, wall_height, is_pit=is_pit)
