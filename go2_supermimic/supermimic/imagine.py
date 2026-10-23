"""Imagined rollouts through the world model, with analytic rewards and termination.

Mirrors the env (oracle perception): both balls are advanced with the same
`advance` dynamics, the actor observation (including the terrain height scan) is
rebuilt exactly as in the env, and every reward term of env_cfg.py is
re-implemented as a smooth function of the imagined state. The dodge hit event becomes a smooth kernel on the body-box
clearance and the fall termination becomes a probability, so the return is
differentiable (SuperMimic Eq. 2 with the soft survival weight c_t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.target import obstacles as ob
from go2_supermimic.target import terrain as tr
from go2_supermimic.target.collision import (
  crouch_violation,
  dodge_clearance,
  dodge_shaping,
  obstacle_penetration,
  predicted_clearance,
  robot_geometry,
  tag_activation,
)
from go2_supermimic.target.generator import BallTarget, BallTargetCfg, advance, dodge_shape
from go2_supermimic.target.interface import oracle_observation, yaw_rotate_inverse


@dataclass(kw_only=True)
class RewardWeights:
  """Same names and values as env_cfg.py; rewards are multiplied by dt there too."""

  chase_approach: float = 1.5
  chase_tag: float = 15.0
  facing_target: float = 0.5
  evade_danger: float = -2.0
  hit: float = -100.0
  evade_anticipate: float = -1.5
  outside_arena: float = -2.0
  obstacle_contact: float = -100.0
  upright: float = 1.0
  tilt_rate_l2: float = -0.05
  hard_landing: float = -0.5
  crouch: float = -30.0
  joint_deviation_l2: float = -0.1
  base_height_low: float = -10.0
  terminated: float = -200.0
  action_rate_l2: float = -0.05
  dof_pos_limits: float = -1.0
  joint_torques_l2: float = -2e-5
  joint_acc_l2: float = -2.5e-7


@dataclass(kw_only=True)
class ImagineCfg:
  horizon: int = 32
  gamma: float = 0.99
  lam: float = 0.95
  dt: float = 0.02
  fall_angle_deg: float = 70.0
  fall_softness_deg: float = 5.0
  danger_std: float = 0.4
  anticipate_std: float = 0.3
  hit_softness: float = 0.03
  """Width [m] of the sigmoid that replaces the hard hit test (clearance <= 0)."""
  arena_margin: float = 0.5
  sensor_grad: bool = True
  """Back-propagate through the LiDAR and the height scan (the actor's and the
  world model's): how moving and turning change what the robot senses. Detaching
  them leaves a wrong gradient near obstacles and terrain (the policy circles)."""
  weights: RewardWeights = None  # type: ignore[assignment]
  compile: bool = False
  """torch.compile one imagined step (fuses the many small elementwise kernels)."""

  def __post_init__(self):
    if self.weights is None:
      self.weights = RewardWeights()


def planar_velocity_world(s: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """World-frame planar base velocity (N, 2) from the yaw-frame velocity in s."""
  c, si = yaw.cos(), yaw.sin()
  return torch.stack([c * s[:, 4] - si * s[:, 5], si * s[:, 4] + c * s[:, 5]], -1)


# State, both objects (yaw-frame position 3, velocity 3, active 1), the dodge
# obstacle shape (is_bar, half_len), the chase ball's remaining escape time and
# sim context; with obstacles the LiDAR, on terrain the height scan is appended.
CRITIC_DIM = st.STATE_DIM + 2 * 7 + 2 + 1 + st.CONTEXT_DIM


class Imagination:
  def __init__(self, cfg: ImagineCfg, consts: st.RobotConsts, target_cfg: BallTargetCfg,
               history: int, device: str, terms: tuple[tuple[str, int], ...] = st.OBS_TERMS,
               field: tr.HeightField | None = None):
    self.cfg = cfg
    self.field = field
    self.terms = terms
    self.use_scan = any(name == "height_scan" for name, _ in terms)
    self.use_obst = target_cfg.obstacles.enabled
    self.critic_dim = (CRITIC_DIM + (tr.SCAN_POINTS if self.use_scan else 0)
                       + (target_cfg.obstacles.lidar_rays if self.use_obst else 0))
    self.c = consts
    self.target_cfg = target_cfg
    self.h = history
    self.device = device
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  # Rewards. All take imagined tensors for one step and return (B,).

  def _rewards(self, s0, s1, pose1, gst, a, a_prev, k):
    """k: per-rollout constants (default_q, scale, limits, hint_weight tensor)."""
    cfg, w, c = self.cfg, self.cfg.weights, self.c
    chase = gst["has_chase"].float()
    dodge = gst["has_dodge"].float()
    yaw = pose1[:, 2]
    cy, sy = yaw.cos(), yaw.sin()
    v_w = planar_velocity_world(s1, yaw)
    rel = gst["chase_pos"][:, :2] - pose1[:, :2]
    dist = rel.norm(dim=-1).clamp(min=1e-6)
    u = rel / dist.unsqueeze(-1)
    toward = (v_w * u).sum(-1)
    geom = robot_geometry(pose1[:, :2], s1[:, 0], yaw, st.limb_points_world(s1, pose1))
    clearance = dodge_clearance(gst, geom, self.target_cfg)
    ahead = predicted_clearance(gst, geom, self.target_cfg)
    feet_min = geom["feet"][..., 2].min(-1).values

    q = s1[:, st.Q] + k["default_q"]
    lim = k["soft_limits"]
    q_target = a * k["action_scale"] + k["default_q"]
    tau = (c.kp * (q_target - q) - c.kd * s1[:, st.QD]).clamp(-c.effort, c.effort)
    g = s1[:, st.GRAV]
    r_arena = (pose1[:, :2] - gst["center"]).norm(dim=-1)

    terms = {
      "chase_approach": toward.clamp(-5.0, 5.0) * chase,
      "chase_tag": tag_activation(gst, pose1[:, :2], yaw, self.target_cfg),
      "facing_target": 0.5 * (1.0 + cy * u[:, 0] + sy * u[:, 1]) * chase,
      "evade_danger": torch.exp(-((clearance.clamp(min=0) / cfg.danger_std) ** 2)) * dodge,
      # Smooth stand-in for the one-step hit event (clearance <= 0).
      "hit": torch.sigmoid(-clearance / cfg.hit_softness) * dodge * 0.25,
      "evade_anticipate": torch.exp(-((ahead.clamp(min=0) / cfg.anticipate_std) ** 2)) * dodge,
      "outside_arena": (r_arena - self.target_cfg.arena_radius - cfg.arena_margin).clamp(min=0),
      "obstacle_contact": obstacle_penetration(geom, gst["obst"]),
      "upright": torch.exp(-(g[:, :2].square().sum(-1)) / 0.4**2),
      "tilt_rate_l2": s1[:, st.W_B][:, :2].square().sum(-1),
      "hard_landing": (-s1[:, 6]).clamp(min=0.0).square() * (feet_min < 0.05).float(),
      "crouch": crouch_violation(gst, pose1[:, :2], s1[:, 0]),
      "joint_deviation_l2": s1[:, st.Q].square().sum(-1),
      "base_height_low": (0.18 - s1[:, 0]).clamp(min=0.0),
      "action_rate_l2": (a - a_prev).square().sum(-1),
      "dof_pos_limits": ((lim[..., 0] - q).clamp(min=0) + (q - lim[..., 1]).clamp(min=0)).sum(-1),
      "joint_torques_l2": tau.square().sum(-1),
      "joint_acc_l2": ((s1[:, st.QD] - s0[:, st.QD]) / cfg.dt).square().sum(-1),
    }
    total = sum(getattr(w, name) * v for name, v in terms.items())
    # Annealed jump/duck hint; its weight is a tensor so changing it does not
    # trigger torch.compile recompilation.
    terms["dodge_hint"] = dodge_shaping(gst, pose1[:, :2], s1[:, 0], st.feet_world_z(s1))
    total = total + k["hint_weight"] * terms["dodge_hint"]
    return total * cfg.dt, terms

  def fall_prob(self, s: torch.Tensor) -> torch.Tensor:
    """Soft version of bad_orientation(70 deg): tilt from the gravity z-component."""
    gz = (-s[:, 3]).clamp(-1.0 + 1e-4, 1.0 - 1e-4)  # acos' is infinite at +-1
    tilt = torch.acos(gz)
    lim = math.radians(self.cfg.fall_angle_deg)
    return torch.sigmoid((tilt - lim) / math.radians(self.cfg.fall_softness_deg))

  def relief(self, pose, gst, grad: bool) -> torch.Tensor:
    obst = gst["obst"] if self.use_obst else None
    return tr.relief(pose[:, :2], pose[:, 2], self.field, obst, grad=grad)

  def scan(self, s, pose, gst, grad: bool) -> torch.Tensor:
    """Base height above each scan point (the actor's height_scan term)."""
    return s[:, st.Z] + self.relief(pose, gst, grad)

  def lidar(self, pose, gst, grad: bool) -> torch.Tensor:
    """LiDAR ranges at the imagined pose (the actor's lidar term)."""
    ocfg = self.target_cfg.obstacles
    with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
      return ob.lidar(pose[:, :2], pose[:, 2], ob.unpack(gst["obst"]), ocfg.lidar_rays,
                      ocfg.lidar_range)

  def critic_input(self, s, pose, gst, xi):
    """State, both balls (yaw-frame position, velocity, active flag), the dodge
    obstacle shape, escape time, the LiDAR / height scan and sim context."""
    base3 = torch.cat([pose[:, :2], s[:, st.Z]], dim=-1)
    parts = [s]
    for pos, vel, flag in (
      (gst["chase_pos"], gst["chase_vel"], gst["has_chase"]),
      (gst["dodge_pos"], gst["dodge_vel"], gst["has_dodge"]),
    ):
      m = flag.float().unsqueeze(-1)
      parts += [
        yaw_rotate_inverse(pos - base3, pose[:, 2]) * m,
        yaw_rotate_inverse(vel, pose[:, 2]) * m,
        m,
      ]
    parts.append(dodge_shape(gst) * gst["has_dodge"].float().unsqueeze(-1))
    parts.append(gst["escape"].unsqueeze(-1))
    if self.use_obst:
      parts.append(self.lidar(pose, gst, grad=False))
    if self.use_scan:
      parts.append(self.scan(s, pose, gst, grad=False))
    return torch.cat(parts + [xi], dim=-1)

  def _step(self, s, pose, a_prev, hist, gst, xi, enc_bias, k, dt, actor, wm):
    """One imagined step. Pure function of its inputs (compiled when cfg.compile)."""
    cfg = self.cfg
    crit_in = self.critic_input(s, pose, gst, xi)
    a = actor(hist)
    relief = self.relief(pose, gst, cfg.sensor_grad)
    s_next, dpose = wm(s, a, a_prev, xi, relief)
    pose_next = st.apply_pose_delta(pose, dpose)
    yaw = pose_next[:, 2]
    v_w = planar_velocity_world(s_next, yaw)
    gst = advance(gst, dt, pose_next[:, :2].detach(), v_w.detach(), yaw.detach(), self.target_cfg,
                  self.field)
    r, _ = self._rewards(s, s_next, pose_next, gst, a, a_prev, k)
    d = self.fall_prob(s_next)
    r = r + cfg.weights.terminated * d * cfg.dt
    base = torch.cat([pose_next[:, :2], s_next[:, st.Z]], dim=-1)
    tgt = oracle_observation(gst, base, yaw)
    lidar = self.lidar(pose_next, gst, cfg.sensor_grad) if self.use_obst else None
    scan = self.scan(s_next, pose_next, gst, cfg.sensor_grad) if self.use_scan else None
    frame = st.obs_frame(s_next, a, tgt, enc_bias, noise=True, lidar=lidar, height_scan=scan)
    hist = st.push_history(hist, frame, self.h, self.terms)
    return s_next, pose_next, a, hist, gst, r, d, crit_in

  def rollout(self, wm, actor, critic, seed: dict, hint_weight: float = 0.0):
    """Imagine H steps from replay seeds. Returns differentiable objective and critic data."""
    from go2_supermimic.supermimic.buffer import unpack_generator

    cfg = self.cfg
    B = seed["s"].shape[0]
    env_ids = seed["env"]
    gen = BallTarget(self.target_cfg, B, self.device)
    unpack_generator(gen, seed["gen"])
    gst = gen.state()
    s, pose, a_prev, hist = seed["s"], seed["pose"], seed["a_prev"], seed["obs"]
    xi, enc_bias = seed["xi"], seed["enc_bias"]
    dt = torch.full((B,), cfg.dt, device=self.device)
    k = {
      "default_q": self.c.default_q[env_ids],
      "action_scale": self.c.action_scale[env_ids],
      "soft_limits": self.c.soft_limits[env_ids],
      "hint_weight": torch.tensor(hint_weight, device=self.device),
    }

    surv = torch.ones(B, device=self.device)
    ret = torch.zeros(B, device=self.device)
    disc = 1.0
    rewards, survs, crit_in = [], [], []
    for _ in range(cfg.horizon):
      survs.append(surv)
      s, pose, a_prev, hist, gst, r, d, x = self._step_fn(
        s, pose, a_prev, hist, gst, xi, enc_bias, k, dt, actor, wm
      )
      crit_in.append(x)
      ret = ret + disc * surv * r
      rewards.append(r)
      surv = surv * (1.0 - d)
      disc *= cfg.gamma
    x_H = self.critic_input(s, pose, gst, xi)
    v_H = critic(x_H).squeeze(-1)
    objective = ret + disc * surv * v_H
    return objective, {
      "rewards": torch.stack(rewards, 1),  # (B, H)
      "surv": torch.stack(survs + [surv], 1),  # (B, H+1)
      "critic_in": torch.stack(crit_in + [x_H], 1),  # (B, H+1, D)
    }

  @torch.no_grad()
  def gae_targets(self, critic, data) -> torch.Tensor:
    """TD(lambda) returns over the imagined steps (SuperMimic Eq. 3)."""
    cfg = self.cfg
    r = data["rewards"].detach()
    surv = data["surv"].detach()
    v = critic(data["critic_in"].detach()).squeeze(-1)  # (B, H+1)
    H = r.shape[1]
    targets = torch.zeros_like(r)
    nxt = v[:, H]
    for t in reversed(range(H)):
      cont = surv[:, t + 1] / surv[:, t].clamp(min=1e-6)  # 1 - d_{t+1}
      nxt = r[:, t] + cfg.gamma * cont * ((1 - cfg.lam) * v[:, t + 1] + cfg.lam * nxt)
      targets[:, t] = nxt
    return targets
