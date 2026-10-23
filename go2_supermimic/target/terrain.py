"""Ground geometry the world model and the policy see: a local height scan.

The ground enters the world model as a grid of SCAN_POINTS around the base
(yaw frame): for each point, how far the surface there lies below the ground
under the base ("relief", positive = lower). The surface is the terrain plus
the tall static obstacles, so steps, slopes and walls all show up. The world-model input is the base
height above ground (state s[0]) plus the relief, i.e. the base height above
every scan point; the policy's height_scan observation is the same quantity.

Terrain heights come from a HeightField: a world-frame grid sampled bilinearly,
ray-cast once from the generated terrain (flat ground needs none).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from go2_supermimic.target import obstacles as ob

# Scan grid in the robot yaw frame: 11 x 7 points, 12 cm apart, from behind
# the hips to ahead of the head.
SCAN_X = tuple(-0.42 + 0.12 * i for i in range(11))
SCAN_Y = tuple(-0.36 + 0.12 * j for j in range(7))
SCAN_POINTS = len(SCAN_X) * len(SCAN_Y)
RELIEF_CLIP = (-1.2, 1.2)
OBSTACLE_EDGE = 0.005  # softness of obstacle edges in the height map [m] (walls are 10 cm thick)


def scan_offsets(device) -> torch.Tensor:
  xs, ys = torch.meshgrid(torch.tensor(SCAN_X), torch.tensor(SCAN_Y), indexing="ij")
  return torch.stack([xs.flatten(), ys.flatten()], -1).to(device)  # (P, 2)


class HeightField:
  """Terrain height z(x, y) on a regular world grid (bilinear, differentiable in x, y)."""

  def __init__(self, heights: torch.Tensor, origin: tuple[float, float], res: float):
    self.h = heights  # (H, W): row = y, column = x
    self.origin = origin
    self.res = res

  def sample(self, xy: torch.Tensor) -> torch.Tensor:
    gx = (xy[..., 0] - self.origin[0]) / self.res
    gy = (xy[..., 1] - self.origin[1]) / self.res
    H, W = self.h.shape
    gx = gx.clamp(0, W - 1.001)
    gy = gy.clamp(0, H - 1.001)
    x0, y0 = gx.floor().long(), gy.floor().long()
    fx, fy = gx - x0, gy - y0
    h00 = self.h[y0, x0]
    h01 = self.h[y0, x0 + 1]
    h10 = self.h[y0 + 1, x0]
    h11 = self.h[y0 + 1, x0 + 1]
    return (h00 * (1 - fx) + h01 * fx) * (1 - fy) + (h10 * (1 - fx) + h11 * fx) * fy

  def save(self, path: str) -> None:
    torch.save({"h": self.h.cpu(), "origin": self.origin, "res": self.res}, path)

  @staticmethod
  def load(path: str, device) -> HeightField:
    d = torch.load(path, map_location=device)
    return HeightField(d["h"].to(device), tuple(d["origin"]), d["res"])


def build_height_field(model, data, geom_ids: list[int], bounds: tuple[float, float, float, float],
                       res: float, device) -> HeightField:
  """Ray-cast the given terrain geoms straight down on a grid (CPU MuJoCo, once)."""
  import mujoco

  x0, y0, x1, y1 = bounds
  xs = np.arange(x0, x1 + 1e-9, res)
  ys = np.arange(y0, y1 + 1e-9, res)
  # Cast only against the terrain: every other geom goes to group 5, which is off.
  saved = model.geom_group.copy()
  model.geom_group[:] = 5
  model.geom_group[geom_ids] = 0
  groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
  heights = np.zeros((len(ys), len(xs)), dtype=np.float32)
  down = np.array([0.0, 0.0, -1.0])
  gid = np.zeros(1, dtype=np.int32)
  top = 20.0
  for i, y in enumerate(ys):
    for j, x in enumerate(xs):
      d = mujoco.mj_ray(model, data, np.array([x, y, top]), down, groups, 1, -1, gid)
      heights[i, j] = top - d if d >= 0 else 0.0
  model.geom_group[:] = saved
  return HeightField(torch.tensor(heights, device=device), (float(x0), float(y0)), res)


def env_height_field(env, exclude_geoms: set[int], half: float, res: float,
                     cache_dir: str) -> HeightField:
  """Height field of an env's terrain (every geom not in exclude_geoms) over all
  sub-terrain origins +- half (so envs can move between tiles), ray-cast once and
  cached under cache_dir by content hash."""
  tiles = env.scene.terrain.terrain_origins
  origins = (tiles.reshape(-1, 3) if tiles is not None else env.scene.env_origins)[:, :2]
  origins = origins.cpu().numpy()
  bounds = (origins[:, 0].min() - half, origins[:, 1].min() - half,
            origins[:, 0].max() + half, origins[:, 1].max() + half)
  model = env.sim.mj_model
  geoms = [g for g in range(model.ngeom) if g not in exclude_geoms]
  key = hashlib.sha1(
    np.concatenate([model.geom_pos[geoms].ravel(), model.geom_size[geoms].ravel(),
                    np.array(bounds), [res]]).tobytes()
    + (model.hfield_data.tobytes() if model.nhfield else b"")
  ).hexdigest()[:12]
  path = Path(cache_dir) / f"heightfield_{key}.pt"
  if path.exists():
    return HeightField.load(str(path), env.device)
  import mujoco

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  field = build_height_field(model, data, geoms, bounds, res, env.device)
  path.parent.mkdir(parents=True, exist_ok=True)
  field.save(str(path))
  return field


def ground(xy: torch.Tensor, field: HeightField | None) -> torch.Tensor:
  """Terrain height under points (...); 0 on flat ground."""
  if field is None:
    return torch.zeros_like(xy[..., 0])
  return field.sample(xy)


def surface(xy: torch.Tensor, obst: torch.Tensor | None, field: HeightField | None) -> torch.Tensor:
  """Height (N, P) of what a foot or body meets at points (N, P, 2): terrain, or
  the top of a tall obstacle (smoothed edge)."""
  h = ground(xy, field)
  if obst is not None:
    inside = torch.sigmoid(-ob.sdf(xy, obst) / OBSTACLE_EDGE)  # (N, P, K)
    h = h + ob.OBSTACLE_HEIGHT * inside.amax(-1)
  return h


def scan_world(base_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """World positions (N, P, 2) of the scan grid."""
  off = scan_offsets(base_xy.device)
  c, s = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
  x = base_xy[:, None, 0] + c * off[:, 0] - s * off[:, 1]
  y = base_xy[:, None, 1] + s * off[:, 0] + c * off[:, 1]
  return torch.stack([x, y], -1)


def relief(base_xy: torch.Tensor, yaw: torch.Tensor, field: HeightField | None,
           obst_flat: torch.Tensor | None = None, grad: bool = False) -> torch.Tensor:
  """Ground under the base minus the surface at each scan point (N, P).
  Not differentiated unless grad=True."""
  if not grad:
    with torch.no_grad():
      return relief(base_xy, yaw, field, obst_flat, grad=True)
  obst = ob.unpack(obst_flat) if obst_flat is not None else None
  under = ground(base_xy, field).unsqueeze(-1)
  r = under - surface(scan_world(base_xy, yaw), obst, field)
  return r.clamp(*RELIEF_CLIP)
