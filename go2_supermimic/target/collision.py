"""Collision model for dodge hits and the dodge / chase reward helpers.

The robot is a trunk box (yaw frame, relative to the base) plus two capsules per
leg (thigh: hip -> knee, calf: knee -> foot), so lifting a single leg, jumping,
ducking and stepping aside all change the clearance. Every function is smooth
and shared by the env and by imagination.
"""

from __future__ import annotations

import torch

from go2_supermimic.target import obstacles as ob
from go2_supermimic.target.generator import (
  BALL,
  GRAVITY,
  HIGH_BALL,
  HIGH_BAR,
  LOW_BALL,
  LOW_BAR,
  BallTargetCfg,
  _unit,
)


def bar_axis(st: dict[str, torch.Tensor], robot_xy: torch.Tensor) -> torch.Tensor:
  """Unit horizontal direction (N, 2) along a bar: across its motion, or across
  the line to the robot while it is still winding up."""
  moving = st["dodge_vel"][:, :2]
  facing = robot_xy - st["dodge_pos"][:, :2]
  d = _unit(torch.where(st["flying"].unsqueeze(-1), moving, facing))
  return torch.stack([-d[:, 1], d[:, 0]], -1)


TRUNK_X = (-0.25, 0.38)  # includes the head
TRUNK_Y = (-0.10, 0.10)
TRUNK_BELOW_BASE = 0.06
TRUNK_ABOVE_BASE = 0.08
THIGH_RADIUS = 0.035
CALF_RADIUS = 0.025
# Standing base height the policy holds (the zero-action stance sags to ~0.22 m
# with the low PD gains; actively standing reaches ~0.27 m).
STAND_HEIGHT = 0.27
BAR_SAMPLES = 9


def robot_geometry(base_xy, base_z, yaw, limbs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Bundle what the collision model needs; limbs: world hips/knees/feet (N, 4, 3)."""
  return {"base_xy": base_xy, "base_z": base_z, "yaw": yaw, **limbs}


def _trunk_distance(point, g) -> torch.Tensor:
  """Distance from points (..., N, 3) to the trunk box."""
  c, s = g["yaw"].cos(), g["yaw"].sin()
  dx_w = point[..., 0] - g["base_xy"][:, 0]
  dy_w = point[..., 1] - g["base_xy"][:, 1]
  x = c * dx_w + s * dy_w
  y = -s * dx_w + c * dy_w
  z = point[..., 2] - g["base_z"]
  ex = (TRUNK_X[0] - x).clamp(min=0) + (x - TRUNK_X[1]).clamp(min=0)
  ey = (TRUNK_Y[0] - y).clamp(min=0) + (y - TRUNK_Y[1]).clamp(min=0)
  ez = (-TRUNK_BELOW_BASE - z).clamp(min=0) + (z - TRUNK_ABOVE_BASE).clamp(min=0)
  return torch.sqrt(ex**2 + ey**2 + ez**2 + 1e-8)


def _segment_distance(point, a, b) -> torch.Tensor:
  """Distance from points (..., N, 3) to segments a -> b (N, 4, 3), per leg (..., N, 4)."""
  p = point.unsqueeze(-2)
  ab = b - a
  t = ((p - a) * ab).sum(-1, keepdim=True) / (ab.square().sum(-1, keepdim=True) + 1e-6)
  closest = a + t.clamp(0.0, 1.0) * ab
  return torch.sqrt((p - closest).square().sum(-1) + 1e-8)


SOFTMIN_TAU = 0.005


def _softmin(x: torch.Tensor, dim: int) -> torch.Tensor:
  """Smooth minimum (-tau logsumexp(-x / tau)); <= min(x) by at most tau log(n).
  Smooth everywhere, unlike min, which keeps compiled backward passes finite."""
  return -SOFTMIN_TAU * torch.logsumexp(-x / SOFTMIN_TAU, dim=dim)


def robot_clearance(point: torch.Tensor, radius: float, g: dict[str, torch.Tensor]) -> torch.Tensor:
  """Clearance (..., N) between spheres at `point` (..., N, 3) and the robot (<= 0: hit)."""
  parts = torch.cat(
    [
      _trunk_distance(point, g).unsqueeze(-1),
      _segment_distance(point, g["hips"], g["knees"]) - THIGH_RADIUS,
      _segment_distance(point, g["knees"], g["feet"]) - CALF_RADIUS,
    ],
    dim=-1,
  )
  return _softmin(parts, -1) - radius


def dodge_clearance(
  st: dict[str, torch.Tensor], g: dict[str, torch.Tensor], cfg: BallTargetCfg
) -> torch.Tensor:
  """Clearance (N,) between the dodge obstacle and the robot (<= 0: hit).

  Bars are sampled at BAR_SAMPLES points along their length. An obstacle that has
  not been thrown yet, or has already hit, cannot hit (large clearance).
  """
  pos = st["dodge_pos"]
  ball = robot_clearance(pos, cfg.ball_radius, g)
  t = torch.linspace(-1.0, 1.0, BAR_SAMPLES, device=pos.device).view(-1, 1, 1)
  axis = torch.cat([bar_axis(st, g["base_xy"]), torch.zeros_like(pos[:, :1])], -1)
  pts = pos.unsqueeze(0) + t * st["half_len"].view(1, -1, 1) * axis.unsqueeze(0)
  bar = _softmin(robot_clearance(pts, cfg.bar_radius, g), 0)
  clearance = torch.where(st["kind"] == BALL, ball, bar)
  live = st["flying"] & ~st["spent"]
  return torch.where(live, clearance, torch.full_like(clearance, 10.0))


def _dodge_close(st, robot_xy, near: float) -> torch.Tensor:
  d = (st["dodge_pos"][:, :2] - robot_xy).norm(dim=-1)
  return st["has_dodge"] & st["flying"] & ~st["spent"] & (d < near)


def dodge_shaping(
  st: dict[str, torch.Tensor], robot_xy: torch.Tensor, base_z: torch.Tensor,
  foot_z: torch.Tensor | None = None, near: float = 1.2,
) -> torch.Tensor:
  """Training-wheels hint (N,) while an obstacle is within `near` metres: height
  gained above standing for LOW bars (jump), height lost for HIGH bars and HIGH
  balls (duck), the highest foot lifted for LOW balls (leg lift or hop; needs
  foot_z (N, 4)). MID balls get none (sidestep). Annealed to zero during training
  so the final policy is driven by outcomes only."""
  close = _dodge_close(st, robot_xy, near)
  up = (base_z - STAND_HEIGHT).clamp(0.0, 0.3)
  down = (STAND_HEIGHT - base_z).clamp(0.0, 0.15)
  kind, style = st["kind"], st["style"]
  ball = kind == BALL
  lift = torch.zeros_like(up) if foot_z is None else (foot_z.max(-1).values - 0.03).clamp(0.0, 0.2)
  hint = torch.where(
    kind == LOW_BAR, up,
    torch.where((kind == HIGH_BAR) | (ball & (style == HIGH_BALL)), down,
                torch.where(ball & (style == LOW_BALL), lift, torch.zeros_like(up))),
  )
  return hint * close.float()


def crouch_violation(
  st: dict[str, torch.Tensor], robot_xy: torch.Tensor, base_z: torch.Tensor,
  min_z: float = 0.23, near: float = 1.2,
) -> torch.Tensor:
  """How far (N,) the base is below `min_z`, except while a HIGH bar or HIGH ball
  is close.

  Stops the policy from living in a permanent crouch: ducking is allowed only
  when it is the right answer.
  """
  high = (st["kind"] == HIGH_BAR) | ((st["kind"] == BALL) & (st["style"] == HIGH_BALL))
  may_duck = _dodge_close(st, robot_xy, near) & high
  return (min_z - base_z).clamp(min=0.0) * (~may_duck).float()


def predicted_clearance(
  st: dict[str, torch.Tensor],
  g: dict[str, torch.Tensor],
  cfg: BallTargetCfg,
  horizon: float = 0.6,
) -> torch.Tensor:
  """Clearance (N,) if the thrown obstacle kept flying to its closest approach
  (within `horizon` seconds) while the robot stays as it is. Gives an early,
  smooth danger signal before the obstacle arrives."""
  rel = st["dodge_pos"][:, :2] - g["base_xy"]
  v = st["dodge_vel"]
  t = (-(rel * v[:, :2]).sum(-1) / v[:, :2].square().sum(-1).clamp(min=1e-6)).clamp(0.0, horizon)
  ballistic = (st["kind"] == BALL).float()
  z = st["dodge_pos"][:, 2] + v[:, 2] * t - 0.5 * GRAVITY * t**2 * ballistic
  z = torch.where(st["kind"] == BALL, z.clamp(min=cfg.ball_radius), z)
  ahead = dict(st)
  ahead["dodge_pos"] = torch.cat([st["dodge_pos"][:, :2] + v[:, :2] * t.unsqueeze(-1), z.unsqueeze(-1)], -1)
  return dodge_clearance(ahead, g, cfg)


def tag_activation(st: dict[str, torch.Tensor], robot_xy, robot_yaw, cfg: BallTargetCfg):
  """Tag reward (N,) as a Gaussian of the head-to-ball distance:
  exp(-(d / tag_sigma)^2) while the ball can be tagged (smooth everywhere).

  Inside tag_radius the ball breaks free and stops being taggable, so the
  reward collected per approach is bounded and adds up with the number of
  tags; hovering next to the ball earns nothing. The same function is used in
  the env and in imagination."""
  head = robot_xy + cfg.head_offset * torch.stack([robot_yaw.cos(), robot_yaw.sin()], -1)
  d = (st["chase_pos"][:, :2] - head).norm(dim=-1)
  can_tag = st["has_chase"] & (st["escape"] <= 0.0)
  return torch.exp(-((d / cfg.tag_sigma) ** 2)) * can_tag.float()


# Robot vs static obstacles: points on the trunk outline plus knees and feet, in
# the ground plane (the obstacles are tall), each with its own radius.
_TRUNK_OUTLINE = tuple(
  (x, y) for x in (TRUNK_X[0], 0.5 * (TRUNK_X[0] + TRUNK_X[1]), TRUNK_X[1])
  for y in (TRUNK_Y[0], TRUNK_Y[1])
) + ((TRUNK_X[0], 0.0), (TRUNK_X[1], 0.0))


def obstacle_penetration(
  g: dict[str, torch.Tensor], obst_flat: torch.Tensor, margin: float = 0.05
) -> torch.Tensor:
  """How deep (N,) the robot reaches into the obstacles inflated by `margin`:
  sum over body points of (margin - gap)+. Zero when clear; smooth for imagination."""
  obst = ob.unpack(obst_flat)
  c, s = g["yaw"].cos(), g["yaw"].sin()
  local = torch.tensor(_TRUNK_OUTLINE, device=obst.device)
  tx = g["base_xy"][:, None, 0] + c[:, None] * local[:, 0] - s[:, None] * local[:, 1]
  ty = g["base_xy"][:, None, 1] + s[:, None] * local[:, 0] + c[:, None] * local[:, 1]
  pts = torch.cat([torch.stack([tx, ty], -1), g["knees"][..., :2], g["feet"][..., :2]], 1)
  radius = torch.cat([
    torch.zeros(len(_TRUNK_OUTLINE), device=obst.device),
    torch.full((4,), THIGH_RADIUS, device=obst.device),
    torch.full((4,), CALF_RADIUS, device=obst.device),
  ])
  gap = ob.sdf(pts, obst) - radius.view(1, -1, 1)
  return (margin - gap).clamp(min=0.0).sum((-1, -2))
