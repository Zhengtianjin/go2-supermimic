"""Generated terrain and the scene look.

The mix follows Isaac Lab's rough-terrain set (pyramid stairs up and down, box
grids, random rough ground, slopes), with 12 m tiles that leave room for the
chase. One column per terrain type, five rows of rising difficulty: steps go
from 2 to 10 cm, boxes from +-2 to +-5 cm (neighbours differ by up to 10 cm),
and the flat platforms are small so the robot spends its time on the features. Sharp edges are where the world model is least
accurate, so robots start on the easiest row and harder rows open up during
training (mdp.terrain_levels_by_progress).

The look is a light matte ground under a sky gradient with a sun that casts
shadows, instead of mjlab's height-coloured terrain.
"""

from __future__ import annotations

import copy

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import (
  box_random_grid,
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

ROUGH_TERRAINS = TerrainGeneratorCfg(
  seed=7,
  size=(12.0, 12.0),
  border_width=10.0,
  num_rows=5,
  num_cols=7,
  curriculum=True,
  color_scheme="none",
  sub_terrains={
    "flat": flat(proportion=0.10),
    "stairs": pyramid_stairs(proportion=0.20, step_height_range=(0.02, 0.10), step_width=0.4,
                             platform_width=1.5),
    "stairs_inv": pyramid_stairs_inv(proportion=0.20, step_height_range=(0.02, 0.10),
                                     step_width=0.4, platform_width=1.5),
    "boxes": box_random_grid(proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.05),
                             platform_width=1.0),
    "rough": random_rough(proportion=0.15, noise_range=(0.01, 0.04), noise_step=0.01),
    "slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.3)),
    "slope_inv": hf_pyramid_slope_inv(proportion=0.10, slope_range=(0.0, 0.3)),
  },
  add_lights=False,
)

GROUND_RGBA = (0.62, 0.62, 0.61, 1.0)
GRID_RGB = ((0.66, 0.66, 0.65), (0.62, 0.62, 0.61), (0.46, 0.47, 0.50))  # cells, lines
SKY_RGB = ((0.93, 0.95, 0.98), (0.62, 0.74, 0.87))


def clean_look(terrain: TerrainEntityCfg) -> TerrainEntityCfg:
  """Light grey matte ground with a faint 0.5 m grid (its lines break at step
  edges and bend over slopes, so the terrain's shape reads on video), a sky
  gradient and soft fill light."""
  terrain.textures = (
    TextureCfg(name="sky", type="skybox", builtin="gradient", rgb1=SKY_RGB[0], rgb2=SKY_RGB[1],
               width=512, height=3072),
    TextureCfg(name="grid", type="2d", builtin="checker", mark="edge", rgb1=GRID_RGB[0],
               rgb2=GRID_RGB[1], markrgb=GRID_RGB[2], width=512, height=512),
  )
  terrain.materials = (
    MaterialCfg(name="ground", texture="grid", texuniform=True, texrepeat=(1.0, 1.0),
                reflectance=0.0, geom_names_expr=(".*",)),
  )
  # Fill light only: the shadow-casting sun follows the robot (robot/go2.xml),
  # so its shadow map stays sharp wherever the robot runs.
  terrain.lights = (
    LightCfg(name="fill", type="directional", pos=(0.0, 0.0, 10.0), dir=(0.3, 0.2, -1.0),
             diffuse=(0.12, 0.12, 0.12), specular=(0.0, 0.0, 0.0), ambient=(0.12, 0.13, 0.15),
             castshadow=False),
  )
  return terrain


def rough_terrain() -> TerrainEntityCfg:
  return clean_look(TerrainEntityCfg(
    terrain_type="generator", terrain_generator=copy.deepcopy(ROUGH_TERRAINS),
    max_init_terrain_level=None,
  ))
