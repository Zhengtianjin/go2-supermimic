"""Target observation: the only thing the policy knows about the balls.

The policy never sees a sensor. In simulation the slots are filled from ground
truth (oracle perception); on the robot a perception stack must fill the same
vector with the same layout, frame and units.

There are two labelled slots, always in this order:
  slot 0  CHASE  the target ball to keep up with (green)
  slot 1  DODGE  the incoming obstacle to get past (red ball or bar)
A slot whose object is not in play this episode is all zeros (active = 0).

Frame: the robot *yaw frame*. Origin at the robot base, x forward, z up, rotated
from world by the robot heading only (roll and pitch removed).

Per-slot layout (SLOT_DIM = 14):
  [0:3]   p         object centre relative to the base, yaw frame [m]
  [3:6]   v         object world velocity, expressed in the yaw frame [m/s]
  [6:9]   sigma     1-sigma uncertainty of p per axis [m] (0 for the oracle)
  [9]     visible   1.0 if the object was measured this step, else 0.0
  [10]    age       seconds since the last measurement, clipped to AGE_MAX
  [11]    is_bar    1.0 for a bar, 0.0 for a ball (always 0 in the chase slot)
  [12]    half_len  bar half length [m] (0 for balls); a bar lies horizontally,
                    across its direction of motion
  [13]    active    1.0 if this slot's object is in play this episode, else 0.0
"""

from __future__ import annotations

import torch

SLOT_DIM = 14
NUM_SLOTS = 2
DIM = SLOT_DIM * NUM_SLOTS
CHASE_SLOT, DODGE_SLOT = 0, 1
AGE_MAX = 2.0
SIGMA_MAX = 3.0

P = slice(0, 3)
V = slice(3, 6)
SIGMA = slice(6, 9)
VISIBLE = 9
AGE = 10
IS_BAR = 11
HALF_LEN = 12
ACTIVE = 13


def yaw_rotate_inverse(vec: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
  """Express world-frame vectors (N, 3) in the yaw frame given heading (N,)."""
  c, s = heading.cos(), heading.sin()
  x = c * vec[:, 0] + s * vec[:, 1]
  y = -s * vec[:, 0] + c * vec[:, 1]
  return torch.stack([x, y, vec[:, 2]], dim=-1)


def pack_slot(
  p: torch.Tensor,
  v: torch.Tensor,
  sigma: torch.Tensor,
  visible: torch.Tensor,
  age: torch.Tensor,
  active: torch.Tensor,
  shape: torch.Tensor | None = None,
) -> torch.Tensor:
  """Pack one slot from batched fields ((N, 3) or (N,)); inactive rows become zeros.

  shape: (N, 2) = [is_bar, half_len]; zeros (a ball) if None.
  """
  if shape is None:
    shape = torch.zeros_like(p[:, :2])
  slot = torch.cat(
    [
      p,
      v,
      sigma.clamp(0.0, SIGMA_MAX),
      visible.float().unsqueeze(-1),
      age.clamp(0.0, AGE_MAX).unsqueeze(-1),
      shape,
      torch.ones_like(age).unsqueeze(-1),
    ],
    dim=-1,
  )
  return slot * active.float().unsqueeze(-1)


def oracle_observation(st: dict[str, torch.Tensor], base: torch.Tensor,
                       yaw: torch.Tensor) -> torch.Tensor:
  """Both slots (N, DIM) from the generator state, exact (sigma 0, always visible).

  base (N, 3) is the robot base in the ball frame, yaw (N,) its heading. Used by
  the env and by imagination alike.
  """
  from go2_supermimic.target.generator import dodge_shape

  ones, zeros3 = torch.ones_like(yaw), torch.zeros_like(base)
  slots = []
  for pos, vel, active, shape in (
    (st["chase_pos"], st["chase_vel"], st["has_chase"], None),
    (st["dodge_pos"], st["dodge_vel"], st["has_dodge"], dodge_shape(st)),
  ):
    p = yaw_rotate_inverse(pos - base, yaw)
    v = yaw_rotate_inverse(vel, yaw)
    slots.append(pack_slot(p, v, zeros3, ones, torch.zeros_like(yaw), active, shape))
  return torch.cat(slots, dim=-1)
