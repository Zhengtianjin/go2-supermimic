"""Policies that drive some envs while collecting data for the world model.

On new terrain the world model starts with no data about it; a policy that
already walks there fills the replay buffer with terrain transitions from the
first iteration. It only steers: its joint targets are converted to our
joint_pos action space and simulated with our actuators, so what it collects is
our dynamics.

The policy is go2_rl_lab's mixed-terrain MoE-CTS (wty-yy, Hugging Face
wty-yy/go2_rl_gym_data, MIT): blind, trained in MuJoCo Warp, walks reliably in
this sim. Its TorchScript export keeps a batch-1 observation history, so the
history is rebuilt here for all envs.
"""

from __future__ import annotations

import math
import re

import torch

from go2_supermimic.robot.go2 import GO2_ACTION_SCALE

# go2_rl_lab joint order (leg by leg) and default pose; actions are offsets from
# it scaled by 0.25.
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "calf")]
DEFAULT = {"hip": {"L": 0.1, "R": -0.1}, "thigh": {"F": 0.8, "R": 1.0}, "calf": -1.5}
ACTION_SCALE = 0.25
COMMAND_SCALE = (0.5, 1.0, 1.0)  # lin_vel_x, lin_vel_y, ang_vel_z


def _default(joint: str) -> float:
  leg, kind = joint.split("_")[:2]
  if kind == "hip":
    return DEFAULT["hip"][leg[1]]
  if kind == "thigh":
    return DEFAULT["thigh"][leg[0]]
  return DEFAULT["calf"]


class MoeCtsDriver:
  """Velocity-command policy for the scene's robot, batched over envs."""

  def __init__(self, env, path: str):
    self.robot = env.scene["robot"]
    self.dev = env.device
    self.policy = torch.jit.load(path, map_location=self.dev).eval()
    names = list(self.robot.joint_names)
    self.idx = torch.tensor([names.index(j) for j in JOINTS], device=self.dev)
    self.default = torch.tensor([_default(j) for j in JOINTS], device=self.dev)
    self.our_scale = torch.tensor(
      [next(v for k, v in GO2_ACTION_SCALE.items() if re.fullmatch(k, n)) for n in names],
      device=self.dev,
    )
    self.cmd_scale = torch.tensor(COMMAND_SCALE, device=self.dev)
    self.dims, self.hist_len = list(self.policy.feature_dims), int(self.policy.history_len)
    self.hist = torch.zeros(env.num_envs, sum(self.dims) * self.hist_len, device=self.dev)
    self.last = torch.zeros(env.num_envs, len(JOINTS), device=self.dev)

  def act(self, cmd: torch.Tensor) -> torch.Tensor:
    """Our joint_pos action (N, 12) for velocity commands (N, 3): vx, vy, yaw rate."""
    d = self.robot.data
    terms = [0.25 * d.root_link_ang_vel_b, d.projected_gravity_b, cmd * self.cmd_scale,
             d.joint_pos[:, self.idx] - self.default, 0.05 * d.joint_vel[:, self.idx],
             self.last]
    blocks, off = [], 0
    for t, dim in zip(terms, self.dims):  # per-term history, oldest first
      size = dim * self.hist_len
      blocks.append(torch.cat([self.hist[:, off + dim : off + size], t], -1))
      off += size
    self.hist = torch.cat(blocks, -1)
    p = self.policy
    latent, _ = p.student_moe_encoder(p.actor_obs_normalizer(self.hist))
    a = p.actor(torch.cat([latent, p.single_obs_normalizer(torch.cat(terms, -1))], -1))
    self.last = a
    target = torch.empty_like(a)
    target[:, self.idx] = self.default + ACTION_SCALE * a
    return (target - d.default_joint_pos) / self.our_scale

  def reset(self, done: torch.Tensor) -> None:
    self.last[done] = 0.0
    self.hist[done] = 0.0


class ActorDriver:
  """One of our own checkpoints driving the env, e.g. a flat-ground chase-and-dodge
  policy on terrain, so the world model sees its jumps and ducks there. Its
  observation must be a prefix of the env's (true when the task only appends
  terms, like the terrain height scan)."""

  def __init__(self, path: str, env_obs_dim: int, device):
    from go2_supermimic.supermimic.train import Actor

    ck = torch.load(path, map_location=device, weights_only=False)
    self.obs_dim = ck["obs_dim"]
    assert self.obs_dim <= env_obs_dim, (self.obs_dim, env_obs_dim)
    self.actor = Actor(self.obs_dim, tuple(ck["cfg"]["actor_hidden"]), 12).to(device)
    self.actor.load_state_dict(ck["actor"])
    self.actor.eval()

  @torch.no_grad()
  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(obs[:, : self.obs_dim])


class ChaseSteer:
  """Velocity commands for the driver: head for the chase ball (or, without one,
  a random waypoint in the arena) at a per-env random top speed, turning toward
  it like Isaac Lab's heading command."""

  def __init__(self, num_envs: int, device, vmax: tuple[float, float], radius: float):
    self.dev, self.vmax, self.radius = device, vmax, radius
    self.speed = torch.zeros(num_envs, device=device)
    self.goal = torch.zeros(num_envs, 2, device=device)
    self.timer = torch.zeros(num_envs, device=device)

  def resample(self, mask: torch.Tensor, center: torch.Tensor) -> None:
    k = int(mask.sum())
    if k == 0:
      return
    lo, hi = self.vmax
    self.speed[mask] = lo + (hi - lo) * torch.rand(k, device=self.dev)
    r = self.radius * torch.rand(k, device=self.dev).sqrt()
    th = 2 * math.pi * torch.rand(k, device=self.dev)
    self.goal[mask] = center[mask] + torch.stack([r * th.cos(), r * th.sin()], -1)
    self.timer[mask] = 2.0 + 3.0 * torch.rand(k, device=self.dev)

  def __call__(self, dt: float, pos: torch.Tensor, yaw: torch.Tensor, center: torch.Tensor,
               ball: torch.Tensor, has_ball: torch.Tensor) -> torch.Tensor:
    self.timer -= dt
    self.resample((self.timer <= 0) | ((self.goal - pos).norm(dim=-1) < 0.4), center)
    to = torch.where(has_ball.unsqueeze(-1), ball, self.goal) - pos
    err = torch.atan2(to[:, 1], to[:, 0]) - yaw
    err = torch.atan2(err.sin(), err.cos())
    v = torch.minimum(self.speed, 1.5 * to.norm(dim=-1))
    return torch.stack([v * err.cos().clamp(min=0), torch.zeros_like(v),
                        (1.5 * err).clamp(-1.5, 1.5)], -1)
