"""Fixed-size per-env replay buffer of simulator transitions (oldest replaced first).

Each slot stores the state *before* the action and the executed (noisy) action.
The next state is the following slot of the same env unless `done` is set, in
which case the following slot starts a new episode.
"""

from __future__ import annotations

import torch

from go2_supermimic.supermimic import state as st

from go2_supermimic.target.generator import BOOL_FIELDS, STATE_DIMS
from go2_supermimic.target.terrain import SCAN_POINTS

GEN_DIM = sum(STATE_DIMS.values())


def pack_generator(gen) -> torch.Tensor:
  parts = [getattr(gen, name).float().reshape(-1, dim) for name, dim in STATE_DIMS.items()]
  return torch.cat(parts, dim=-1)


def unpack_generator(gen, packed: torch.Tensor) -> torch.Tensor:
  """Write packed generator state into `gen` (batch must match)."""
  i = 0
  for name, dim in STATE_DIMS.items():
    x = packed[:, i : i + dim]
    if name in BOOL_FIELDS:
      x = x[:, 0] > 0.5
    elif dim == 1:
      x = x[:, 0]
    setattr(gen, name, x.clone())
    i += dim


class ReplayBuffer:
  def __init__(self, num_envs: int, length: int, obs_dim: int, device: str,
               gen_dim: int = GEN_DIM):
    self.n, self.L, self.device = num_envs, length, device
    f = lambda *shape: torch.zeros(num_envs, length, *shape, device=device)  # noqa: E731
    self.s = f(st.STATE_DIM)
    self.pose = f(3)
    self.a = f(st.NUM_JOINTS)
    self.a_prev = f(st.NUM_JOINTS)
    # Observations dominate memory; half precision is plenty for seeding rollouts.
    self.obs = torch.zeros(num_envs, length, obs_dim, device=device, dtype=torch.float16)
    self.relief = torch.zeros(num_envs, length, SCAN_POINTS, device=device, dtype=torch.float16)
    self.gen = f(gen_dim)
    self.done = f()
    self.ptr = 0
    self.size = 0

  def add(self, s, pose, a, a_prev, obs, gen, done, relief) -> None:
    i = self.ptr
    self.s[:, i], self.pose[:, i], self.a[:, i], self.a_prev[:, i] = s, pose, a, a_prev
    self.relief[:, i] = relief.half()
    self.obs[:, i], self.gen[:, i], self.done[:, i] = obs.half(), gen, done.float()
    self.ptr = (self.ptr + 1) % self.L
    self.size = min(self.size + 1, self.L)

  def _phys(self, logical: torch.Tensor) -> torch.Tensor:
    start = (self.ptr - self.size) % self.L
    return (start + logical) % self.L

  def sample_windows(self, batch: int, k: int):
    """Random k-step windows: (env_ids, slot indices (B, k+1), valid (B, k))."""
    assert self.size > k + 1, "buffer too small for windows"
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t0 = torch.randint(0, self.size - k, (batch,), device=self.device)
    idx = self._phys(t0.unsqueeze(-1) + torch.arange(k + 1, device=self.device))
    done = self.done[env.unsqueeze(-1), idx[:, :k]]
    # Transition j (slot j -> j+1) is valid if no slot 0..j ended an episode.
    valid = torch.cumprod(1.0 - done, dim=1)
    return env, idx, valid

  def sample_states(self, batch: int):
    """Random (env_ids, slot) pairs to seed imagined rollouts."""
    env = torch.randint(0, self.n, (batch,), device=self.device)
    t = self._phys(torch.randint(0, self.size, (batch,), device=self.device))
    return env, t


WINDOW_KEYS = ("s", "pose", "a", "a_prev", "xi", "valid", "relief")


class WindowArchive:
  """Ring store of k-step world-model windows with rare events (contact changes,
  flight, falls). Unlike the replay buffer it is not overwritten every few
  seconds of sim time, so the world model keeps seeing jumps and landings."""

  def __init__(self, capacity: int, k: int, context_dim: int, device: str):
    self.cap, self.device = capacity, device
    z = lambda *shape: torch.zeros(capacity, *shape, device=device)  # noqa: E731
    self.data = {
      "s": z(k + 1, st.STATE_DIM), "pose": z(k + 1, 3), "a": z(k, st.NUM_JOINTS),
      "a_prev": z(k, st.NUM_JOINTS), "xi": z(context_dim), "valid": z(k),
      "relief": torch.zeros(capacity, k, SCAN_POINTS, device=device, dtype=torch.float16),
    }
    self.ptr = 0
    self.size = 0

  def add(self, windows: dict[str, torch.Tensor]) -> None:
    n = windows["s"].shape[0]
    if n == 0:
      return
    n = min(n, self.cap)
    idx = (self.ptr + torch.arange(n, device=self.device)) % self.cap
    for k in WINDOW_KEYS:
      self.data[k][idx] = windows[k][:n]
    self.ptr = (self.ptr + n) % self.cap
    self.size = min(self.size + n, self.cap)

  def sample(self, batch: int) -> dict[str, torch.Tensor]:
    idx = torch.randint(0, self.size, (batch,), device=self.device)
    return {k: v[idx] for k, v in self.data.items()}
