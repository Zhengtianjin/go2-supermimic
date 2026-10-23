"""Deterministic world model f(s, a, a_prev, xi, ground scan) -> (s', planar pose delta).

Fit on k-step autoregressive windows from the replay buffer (SuperMimic Eq. 1,
state term only; reward and termination are analytic in imagine.py). Inputs
and outputs are normalized with running statistics from collected data.
"""

from __future__ import annotations

import torch
from torch import nn

from go2_supermimic.supermimic import state as st
from go2_supermimic.target.terrain import SCAN_POINTS


class RunningNorm(nn.Module):
  """Running mean/std over batches (Welford-style merge), frozen when not updated."""

  def __init__(self, dim: int, eps: float = 1e-4, clip: float = 10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(eps))
    self.clip = clip

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1])
    b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = b_mean - self.mean
    tot = self.count + b_n
    self.mean += delta * b_n / tot
    self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
    self.count = tot

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)

  def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
    return y * (self.var.sqrt() + 1e-6) + self.mean


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU) -> nn.Sequential:
  layers, d = [], in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


class WorldModel(nn.Module):
  """s' = f(s, a, a_prev, xi, scan). The scan is the base height above each point
  of the local ground grid (terrain and obstacles, see target/terrain.py), so
  stepping onto a stair or pushing into a wall is predictable."""

  def __init__(self, hidden: tuple[int, ...] = (1024, 1024, 1024)):
    super().__init__()
    in_dim = st.STATE_DIM + 2 * st.NUM_JOINTS + st.CONTEXT_DIM + SCAN_POINTS
    out_dim = st.STATE_DIM + st.POSE_DELTA_DIM
    self.net = mlp(in_dim, hidden, out_dim)
    self.in_norm = RunningNorm(in_dim)
    # Normalizes the per-step change of s and the pose delta.
    self.out_norm = RunningNorm(out_dim)

  @staticmethod
  def _inputs(s, a, a_prev, xi, relief):
    return torch.cat([s, a, a_prev, xi, s[:, st.Z] + relief], dim=-1)

  def update_stats(self, s, a, a_prev, xi, relief, s_next, pose_delta) -> None:
    self.in_norm.update(self._inputs(s, a, a_prev, xi, relief))
    self.out_norm.update(torch.cat([s_next - s, pose_delta], dim=-1))

  def forward(self, s, a, a_prev, xi, relief):
    """relief (B, P): ground under the base minus the ground at each scan point.
    Returns (s_next, pose_delta)."""
    y = self.out_norm.unnormalize(self.net(self.in_norm(self._inputs(s, a, a_prev, xi, relief))))
    return s + y[:, : st.STATE_DIM], y[:, st.STATE_DIM :]

  def loss(self, s0, actions, actions_prev, xi, reliefs, s_targets, pose_targets, valid):
    """k-step autoregressive loss in normalized output units.

    s0 (B, S); actions, actions_prev (B, k, 12); xi (B, C); reliefs (B, k, P) at
    the recorded poses; s_targets (B, k, S); pose_targets (B, k, 3); valid (B, k)
    masks steps after a reset inside the window.
    """
    k = actions.shape[1]
    scale = self.out_norm.var.sqrt() + 1e-6
    s = s0
    total, denom = 0.0, valid.sum().clamp(min=1.0)
    for j in range(k):
      s_pred, pose_pred = self(s, actions[:, j], actions_prev[:, j], xi, reliefs[:, j])
      err = torch.cat([s_pred - s_targets[:, j], pose_pred - pose_targets[:, j]], dim=-1)
      total = total + ((err / scale) ** 2).mean(-1).mul(valid[:, j]).sum()
      s = s_pred
    return total / denom
