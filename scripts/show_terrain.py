#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Show IsaacLab navigation terrain configs in an interactive Isaac Sim scene.

Examples:
    ./isaaclab.sh --python source/isaaclab_nav_task/scripts/show_terrain.py --terrain MAZE_TERRAIN
    ./isaaclab.sh --python source/isaaclab_nav_task/scripts/show_terrain.py --terrain RANDOM_MAZE_TERRAIN --pure_random_maze
"""

from __future__ import annotations

import argparse
import copy
import importlib

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Show isaaclab_nav_task terrain configs.")
parser.add_argument("--terrain", type=str, default="RANDOM_MAZE_TERRAIN", help="Terrain config name, with or without _CFG.")
parser.add_argument("--num_rows", type=int, default=None, help="Override terrain generator rows.")
parser.add_argument("--num_cols", type=int, default=None, help="Override terrain generator columns.")
parser.add_argument("--difficulty_min", type=float, default=None, help="Override minimum terrain difficulty.")
parser.add_argument("--difficulty_max", type=float, default=None, help="Override maximum terrain difficulty.")
parser.add_argument(
    "--pure_random_maze",
    action="store_true",
    help="Set all sub-terrain proportions to 0 except random_maze=1 when the config supports it.",
)
parser.add_argument("--pure_sub_terrain", type=str, default=None, help="Show only the named sub-terrain key.")
parser.add_argument("--color_scheme", type=str, default=None, choices=["height", "random"], help="Optional color scheme.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_nav_task  # noqa: F401


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Minimal scene containing only a generated terrain and sky light."""

    def __init__(self, terrain_generator):
        super().__init__(num_envs=1, env_spacing=2.5)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator,
            max_init_terrain_level=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=0.8,
            ),
            visual_material=(
                None
                if getattr(terrain_generator, "color_scheme", None) is not None
                else sim_utils.MdlFileCfg(
                    mdl_path=(
                        f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                        "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
                    ),
                    project_uvw=True,
                    texture_scale=(0.25, 0.25),
                )
            ),
            debug_vis=False,
        )

        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )


def _resolve_terrain_cfg():
    terrain_name = args_cli.terrain if args_cli.terrain.endswith("_CFG") else f"{args_cli.terrain}_CFG"
    terrain_module = importlib.import_module("isaaclab_nav_task.terrains")
    terrain_cfg = copy.deepcopy(getattr(terrain_module, terrain_name))

    if args_cli.num_rows is not None:
        terrain_cfg.num_rows = args_cli.num_rows
    if args_cli.num_cols is not None:
        terrain_cfg.num_cols = args_cli.num_cols
    if args_cli.difficulty_min is not None or args_cli.difficulty_max is not None:
        min_difficulty = args_cli.difficulty_min if args_cli.difficulty_min is not None else terrain_cfg.difficulty_range[0]
        max_difficulty = args_cli.difficulty_max if args_cli.difficulty_max is not None else terrain_cfg.difficulty_range[1]
        terrain_cfg.difficulty_range = (min_difficulty, max_difficulty)

    pure_sub_terrain = args_cli.pure_sub_terrain
    if args_cli.pure_random_maze:
        pure_sub_terrain = "random_maze"
    if pure_sub_terrain is not None:
        if pure_sub_terrain not in terrain_cfg.sub_terrains:
            raise ValueError(f"{terrain_name} has no '{pure_sub_terrain}' sub-terrain.")
        for sub_terrain in terrain_cfg.sub_terrains.values():
            sub_terrain.proportion = 0.0
        terrain_cfg.sub_terrains[pure_sub_terrain].proportion = 1.0

    if args_cli.color_scheme is not None:
        if hasattr(terrain_cfg, "color_scheme"):
            terrain_cfg.color_scheme = args_cli.color_scheme
        else:
            print("[WARN] Selected TerrainGeneratorCfg does not expose color_scheme; ignoring --color_scheme.")

    print(f"[INFO] Terrain selected: {terrain_name}")
    print(f"[INFO] rows={terrain_cfg.num_rows}, cols={terrain_cfg.num_cols}, difficulty={terrain_cfg.difficulty_range}")
    print("[INFO] sub-terrain proportions:")
    for name, sub_terrain in terrain_cfg.sub_terrains.items():
        print(f"  - {name}: {sub_terrain.proportion}")
    return terrain_cfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((18.0, -18.0, 16.0), (0.0, 0.0, 0.0))

    terrain_cfg = _resolve_terrain_cfg()
    scene = InteractiveScene(SceneCfg(terrain_cfg))

    sim.reset()
    print("[INFO] Terrain setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
