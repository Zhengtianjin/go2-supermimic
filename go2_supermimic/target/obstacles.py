"""Static obstacles: layout, 2D geometry, LiDAR and line of sight.

Obstacles are tall boxes standing on the ground (too tall to jump over), so all
geometry here is 2D in the ground plane. The robot runs among them with a
planar LiDAR, throws wait for a clear line to the robot, and the chase ball
slides along them. Every env has
NUM_OBSTACLES slots with fixed footprints (OBSTACLE_HALF, matching MuJoCo box
geoms of the same size); each episode a slot is placed at a random pose or
parked (inactive).

Layout tensor (N, K, 4) per env: [cx, cy, yaw, active].

All functions are pure tensor code shared by the env and by imagined rollouts,
so the policy sees the same LiDAR in both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Half footprints (x, y) [m] per slot: two pillars, two boxes, two short walls.
OBSTACLE_HALF = ((0.15, 0.15), (0.15, 0.15), (0.45, 0.30), (0.45, 0.30), (1.0, 0.05), (1.0, 0.05))
NUM_OBSTACLES = len(OBSTACLE_HALF)
OBSTACLE_HEIGHT = 1.0
OBST_DIM = 4
FAR = 10.0  # distance reported for "no obstacle"


@dataclass(kw_only=True)
class ObstacleCfg:
  enabled: bool = False
  empty_prob: float = 0.15
  """Episodes with no obstacles at all (keeps the open-field skills)."""
  slot_prob: float = 0.8
  """Each slot is used with this probability otherwise."""
  place_radius: float = 4.2
  """Obstacle centres lie within this radius of the arena centre [m]."""
  robot_clearance: float = 1.2
  """Minimum gap between a new obstacle and the robot's spawn point [m]."""
  gap: float = 0.6
  """Minimum free gap between obstacles' bounding circles (a corridor) [m]."""
  place_tries: int = 12
  lidar_rays: int = 16
  lidar_range: float = 4.0


def half_sizes(device) -> torch.Tensor:
  return torch.tensor(OBSTACLE_HALF, device=device, dtype=torch.float32)


def bounding_radius(device) -> torch.Tensor:
  return half_sizes(device).norm(dim=-1)


def _rot(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """Rotate (..., 2) vectors by -yaw (world -> obstacle frame)."""
  c, s = yaw.cos(), yaw.sin()
  return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def _rot_back(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  c, s = yaw.cos(), yaw.sin()
  return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def unpack(obst: torch.Tensor) -> torch.Tensor:
  """Flat (N, K*4) state field -> (N, K, 4)."""
  return obst.reshape(obst.shape[0], NUM_OBSTACLES, OBST_DIM)


def sample_layout(
  robot_xy: torch.Tensor, center: torch.Tensor, cfg: ObstacleCfg
) -> torch.Tensor:
  """Random obstacle layout (N, K, 4) that keeps clear of the robot and leaves
  corridors between obstacles. Slots that do not fit are parked (inactive)."""
  n, dev = robot_xy.shape[0], robot_xy.device
  rad = bounding_radius(dev)
  out = torch.zeros(n, NUM_OBSTACLES, OBST_DIM, device=dev)
  out[..., :2] = center.unsqueeze(1)
  use_env = torch.rand(n, device=dev) >= cfg.empty_prob
  for k in range(NUM_OBSTACLES):
    best = torch.zeros(n, 3, device=dev)
    found = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(cfg.place_tries):
      r = cfg.place_radius * torch.rand(n, device=dev).sqrt()
      a = torch.rand(n, device=dev) * 2 * math.pi
      pos = center + torch.stack([r * a.cos(), r * a.sin()], -1)
      ok = (pos - robot_xy).norm(dim=-1) > rad[k] + cfg.robot_clearance
      for j in range(k):
        d = (pos - out[:, j, :2]).norm(dim=-1)
        ok &= (d > rad[k] + rad[j] + cfg.gap) | (out[:, j, 3] < 0.5)
      take = ok & ~found
      yaw = (torch.rand(n, device=dev) * 2 - 1) * math.pi
      best = torch.where(take.unsqueeze(-1), torch.cat([pos, yaw.unsqueeze(-1)], -1), best)
      found |= ok
    active = found & use_env & (torch.rand(n, device=dev) < cfg.slot_prob)
    out[:, k, :3] = torch.where(active.unsqueeze(-1), best, out[:, k, :3])
    out[:, k, 3] = active.float()
  return out


def sdf(points: torch.Tensor, obst: torch.Tensor) -> torch.Tensor:
  """Signed distance (N, P, K) from points (N, P, 2) to each obstacle footprint
  (negative inside); FAR for inactive slots."""
  local = _rot(points.unsqueeze(2) - obst[:, None, :, :2], obst[:, None, :, 2])
  q = local.abs() - half_sizes(points.device)
  outside = torch.sqrt(q.clamp(min=0.0).square().sum(-1) + 1e-8)
  inside = q.max(-1).values.clamp(max=0.0)
  d = outside + inside
  return torch.where(obst[:, None, :, 3] > 0.5, d, torch.full_like(d, FAR))


def _slab(o: torch.Tensor, d: torch.Tensor, half: torch.Tensor):
  """Ray/box slab test in the box frame: entry and exit parameters (...)."""
  eps = 1e-6
  d = torch.where(d.abs() < eps, torch.full_like(d, eps), d)
  t1 = (-half - o) / d
  t2 = (half - o) / d
  t_in = torch.minimum(t1, t2).max(-1).values
  t_out = torch.maximum(t1, t2).min(-1).values
  return t_in, t_out


def lidar(origin: torch.Tensor, yaw: torch.Tensor, obst: torch.Tensor, rays: int,
          max_range: float) -> torch.Tensor:
  """Planar LiDAR ranges (N, rays), ray i at yaw + 2 pi i / rays, capped at max_range."""
  dev = origin.device
  ang = yaw.unsqueeze(-1) + torch.arange(rays, device=dev) * (2 * math.pi / rays)
  d = torch.stack([ang.cos(), ang.sin()], -1)  # (N, R, 2)
  o = _rot(origin.unsqueeze(1) - obst[..., :2], obst[..., 2])  # (N, K, 2)
  dl = _rot(d.unsqueeze(2), obst[:, None, :, 2])  # (N, R, K, 2)
  t_in, t_out = _slab(o.unsqueeze(1), dl, half_sizes(dev))
  hit = (t_out >= t_in) & (t_out > 0) & (obst[:, None, :, 3] > 0.5)
  t = torch.where(hit, t_in.clamp(min=0.0), torch.full_like(t_in, max_range))
  return t.min(-1).values.clamp(max=max_range)


def line_of_sight(a: torch.Tensor, b: torch.Tensor, obst: torch.Tensor) -> torch.Tensor:
  """True (N,) where the segment a -> b (N, 2) crosses no active obstacle."""
  o = _rot(a.unsqueeze(1) - obst[..., :2], obst[..., 2])
  d = _rot((b - a).unsqueeze(1).expand_as(o), obst[..., 2])
  t_in, t_out = _slab(o, d, half_sizes(a.device))
  blocked = (t_out >= t_in) & (t_out >= 0) & (t_in <= 1) & (obst[..., 3] > 0.5)
  return ~blocked.any(-1)


def push_out(xy: torch.Tensor, obst: torch.Tensor, margin: float) -> torch.Tensor:
  """Move points (N, 2) that are inside an obstacle (inflated by margin) to its
  nearest side, so a moving object slides along obstacles instead of passing."""
  half = half_sizes(xy.device) + margin
  for k in range(NUM_OBSTACLES):
    c, yaw = obst[:, k, :2], obst[:, k, 2]
    local = _rot(xy - c, yaw)
    pen = half[k] - local.abs()
    inside = (pen > 0).all(-1) & (obst[:, k, 3] > 0.5)
    sign = torch.where(local >= 0, 1.0, -1.0)
    along_x = pen[:, 0] < pen[:, 1]
    lx = torch.where(along_x, sign[:, 0] * half[k, 0], local[:, 0])
    ly = torch.where(along_x, local[:, 1], sign[:, 1] * half[k, 1])
    moved = c + _rot_back(torch.stack([lx, ly], -1), yaw)
    xy = torch.where(inside.unsqueeze(-1), moved, xy)
  return xy
