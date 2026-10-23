"""Target command term: owns the chase ball and the dodge obstacles.

Each episode samples which are in play: chase only, dodge only, or both (chase
while dodging). `command` is the two-slot target observation the actor sees
(see target/interface.py), filled from ground truth. Static obstacles, when
enabled, are mocap boxes moved to each episode's layout. On generated terrain the
balls live in a ground-following frame: their heights, and the robot's for every
ball interaction, are measured above the ground under the robot, as in
imagination, where the world-model state holds the base height above ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from go2_supermimic.supermimic.state import limb_positions_w
from go2_supermimic.target import interface as ti
from go2_supermimic.target import obstacles as ob
from go2_supermimic.target import terrain as tr
from go2_supermimic.target.collision import (
  bar_axis,
  dodge_clearance,
  obstacle_penetration,
  predicted_clearance,
  robot_geometry,
)
from go2_supermimic.target.generator import BALL, BallTarget, BallTargetCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

CHASE_COLOR = (0.1, 0.8, 0.2, 1.0)
DODGE_COLOR = (0.9, 0.2, 0.1, 1.0)


class TargetCommand(CommandTerm):
  cfg: TargetCommandCfg

  def __init__(self, cfg: TargetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.gen = BallTarget(cfg.target, self.num_envs, self.device)
    head_ids, _ = self.robot.find_sites(cfg.head_site)
    self._head_site = head_ids[0]
    # Adaptive curriculum: running throws / hits per obstacle kind.
    self._prev_flying = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.kind_throws = torch.zeros(3, device=self.device)
    self.kind_hits = torch.zeros(3, device=self.device)
    self.kind_hit_rate = torch.zeros(3, device=self.device)
    # Dodge hit detected in the last update; the reward reads it on the next step.
    self.hit = torch.zeros(self.num_envs, device=self.device)
    self._obs = torch.zeros(self.num_envs, ti.DIM, device=self.device)
    self._obstacles = [env.scene[f"obstacle{k}"] for k in range(ob.NUM_OBSTACLES)
                       if f"obstacle{k}" in env.scene.entities]
    assert not cfg.target.obstacles.enabled or len(self._obstacles) == ob.NUM_OBSTACLES
    self.height_field = self._height_field(env) if cfg.terrain else None
    self.gen.field = self.height_field
    for name in ("near", "tags", "hits", "dist", "obstacle_contact"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  # Ground truth accessors.

  @property
  def command(self) -> torch.Tensor:
    return self._obs

  @property
  def has_chase(self) -> torch.Tensor:
    return self.gen.has_chase

  @property
  def has_dodge(self) -> torch.Tensor:
    return self.gen.has_dodge

  def _height_field(self, env) -> tr.HeightField:
    exclude = {int(g) for g in self.robot.indexing.geom_ids}
    for entity in self._obstacles:
      exclude.update(int(g) for g in entity.indexing.geom_ids)
    return tr.env_height_field(env, exclude, self.cfg.target.arena_radius + 1.5,
                               self.cfg.field_res, self.cfg.cache_dir)

  def ground_under_base(self) -> torch.Tensor:
    """Terrain height under the base (N,); 0 on flat ground."""
    return tr.ground(self.robot.data.root_link_pos_w[:, :2], self.height_field)

  def base_local(self) -> torch.Tensor:
    """Base position with z above the ground under the base (the ball frame)."""
    pos = self.robot.data.root_link_pos_w.clone()
    pos[:, 2] -= self.ground_under_base()
    return pos

  def scan(self) -> torch.Tensor:
    """Base height above each point of the local ground grid (N, P)."""
    d = self.robot.data
    rel = tr.relief(d.root_link_pos_w[:, :2], d.heading_w, self.height_field, self.obstacle_layout)
    return self.base_local()[:, 2:3] + rel

  @property
  def obstacle_layout(self) -> torch.Tensor | None:
    """Flat obstacle layout (N, K*4), or None without obstacles."""
    return self.gen.obst if self.cfg.target.obstacles.enabled else None

  def lidar(self) -> torch.Tensor:
    d = self.robot.data
    ocfg = self.cfg.target.obstacles
    return ob.lidar(d.root_link_pos_w[:, :2], d.heading_w, ob.unpack(self.gen.obst),
                    ocfg.lidar_rays, ocfg.lidar_range)

  def obstacle_penetration(self) -> torch.Tensor:
    """How deep the body reaches into the (slightly inflated) static obstacles."""
    return obstacle_penetration(self.geometry(), self.gen.obst)

  def _place_obstacles(self, env_ids: torch.Tensor) -> None:
    """Move the obstacle bodies to this episode's layout (parked below ground if unused)."""
    layout = ob.unpack(self.gen.obst[env_ids])
    for k, entity in enumerate(self._obstacles):
      active = layout[:, k, 3] > 0.5
      pose = torch.zeros(len(env_ids), 7, device=self.device)
      pose[:, :2] = layout[:, k, :2]
      pose[:, 2] = torch.where(active, ob.OBSTACLE_HEIGHT / 2 + 0.005, -5.0)
      pose[:, 3] = (layout[:, k, 2] / 2).cos()
      pose[:, 6] = (layout[:, k, 2] / 2).sin()
      entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def head_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self._head_site]

  def head_ball_dist(self) -> torch.Tensor:
    """Horizontal distance from the robot head to the chase ball."""
    return (self.gen.chase_pos[:, :2] - self.head_pos_w()[:, :2]).norm(dim=-1)

  def geometry(self) -> dict[str, torch.Tensor]:
    """Robot collision geometry in the ball frame: base pose plus hips, knees, feet."""
    data = self.robot.data
    g = self.ground_under_base()
    lift = torch.stack([torch.zeros_like(g), torch.zeros_like(g), g], -1)[:, None]
    limbs = {k: v - lift for k, v in limb_positions_w(self.robot).items()}
    return robot_geometry(data.root_link_pos_w[:, :2], data.root_link_pos_w[:, 2] - g,
                          data.heading_w, limbs)

  def body_clearance(self) -> torch.Tensor:
    """Gap between the dodge obstacle and the robot (<= 0 means touching)."""
    return dodge_clearance(self.gen.state(), self.geometry(), self.cfg.target)

  def predicted_body_clearance(self) -> torch.Tensor:
    """Clearance at the thrown obstacle's predicted closest approach."""
    return predicted_clearance(self.gen.state(), self.geometry(), self.cfg.target)

  def _update_kind_curriculum(self, hit: torch.Tensor) -> None:
    """Throw more of the obstacle kinds that hit the robot most often."""
    launched = self.gen.flying & ~self._prev_flying & self.has_dodge
    self._prev_flying = self.gen.flying.clone()
    kind = self.gen.kind.long()
    self.kind_throws += torch.bincount(kind[launched], minlength=3).float()
    self.kind_hits += torch.bincount(kind[hit], minlength=3).float()
    if self.kind_throws.sum() >= self.cfg.kind_curriculum_window:
      rate = self.kind_hits / self.kind_throws.clamp(min=1.0)
      self.kind_hit_rate = 0.5 * self.kind_hit_rate + 0.5 * rate
      base = torch.tensor(self.cfg.target.kind_probs, device=self.device)
      self.gen.set_kind_weights(base * (self.cfg.kind_curriculum_floor + self.kind_hit_rate))
      self.kind_throws.zero_()
      self.kind_hits.zero_()

  # CommandTerm interface.

  def _update_metrics(self) -> None:
    steps = self._env.max_episode_length
    base = self.robot.data.root_link_pos_w[:, :2]
    d = (self.gen.chase_pos[:, :2] - base).norm(dim=-1)
    self.metrics["dist"] += torch.where(self.has_chase, d, torch.zeros_like(d)) / steps
    # Fraction of the episode spent within near_radius of the chase ball.
    near = self.has_chase & (self.head_ball_dist() < self.cfg.near_radius)
    self.metrics["near"] += near.float() / steps
    if self._obstacles:
      self.metrics["obstacle_contact"] += (self.obstacle_penetration() > 0.05).float() / steps

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.gen.resample(env_ids)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    # Derived kinematics are stale during reset; read the base pose from qpos.
    qpos = self._env.sim.data.qpos[env_ids]
    adr = self.robot.indexing.free_joint_q_adr
    robot_xy = qpos[:, adr[0:2]]
    w, x, y, z = (qpos[:, adr[3 + i]] for i in range(4))
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    center = self._env.scene.env_origins[env_ids, :2]
    # Mode: 0 chase only, 1 dodge only, 2 both.
    probs = torch.tensor(self.cfg.mode_probs, device=self.device, dtype=torch.float32)
    mode = torch.multinomial(probs, len(env_ids), replacement=True)
    self.gen.set_progress(self._env.common_step_counter)
    self.gen.spawn(env_ids, robot_xy, robot_yaw, center, mode != 1, mode != 0)
    if self._obstacles:
      self._place_obstacles(env_ids)
    self.hit[env_ids] = 0.0
    return super().reset(env_ids)

  def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      dt_t = torch.as_tensor(dt, device=self.device, dtype=torch.float32)
      if dt_t.ndim == 0:
        dt_t = dt_t.expand(self.num_envs)
      self._advance(dt_t)
    super().compute(dt, env_ids)

  def _advance(self, dt: torch.Tensor) -> None:
    data = self.robot.data
    base_xy = data.root_link_pos_w[:, :2]
    self.gen.set_progress(self._env.common_step_counter)
    self.gen.step(dt, base_xy, data.root_link_lin_vel_w[:, :2], data.heading_w)
    self.metrics["tags"] += self.gen.tagged.float()

    # Dodge hit: the obstacle touches the body.
    hit = self.has_dodge & (self.body_clearance() <= 0.0) & (dt > 0)
    self.hit = hit.float()
    if self.cfg.kind_curriculum:
      self._update_kind_curriculum(hit)
    self.metrics["hits"] += self.hit
    # The obstacle bounces off (balls) and cannot hit again; the generator throws
    # the next one when this one is done.
    self.gen.deflect(hit.nonzero().flatten(), base_xy[hit])

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids
    self._obs = ti.oracle_observation(self.gen.state(), self.base_local(),
                                      self.robot.data.heading_w)

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    k = self.cfg.arrow_seconds
    r = self.cfg.target.ball_radius
    base = self.robot.data.root_link_pos_w.cpu().numpy()
    lift = self.ground_under_base().cpu().numpy()  # ball frame -> world
    base_vel = self.robot.data.root_link_lin_vel_w.cpu().numpy()
    balls = []
    for flag, pos, vel, color in (
      (self.has_chase, self.gen.chase_pos, self.gen.chase_vel, CHASE_COLOR),
      (self.has_dodge, self.gen.dodge_pos, self.gen.dodge_vel, DODGE_COLOR),
    ):
      pos = pos.cpu().numpy() + np.stack([0 * lift, 0 * lift, lift], -1)
      balls.append((flag.cpu().numpy(), pos, vel.cpu().numpy(), color))
    is_bar = (self.gen.kind != BALL).cpu().numpy()
    half = self.gen.half_len.cpu().numpy()
    axis = bar_axis(self.gen.state(), self.robot.data.root_link_pos_w[:, :2]).cpu().numpy()
    bar_r = self.cfg.target.bar_radius
    for i in env_indices:
      for j, (flag, pos, vel, color) in enumerate(balls):
        if not flag[i]:
          continue
        if j == 1 and is_bar[i]:
          off = np.array([axis[i][0], axis[i][1], 0.0]) * half[i]
          visualizer.add_cylinder(pos[i] - off, pos[i] + off, bar_r, color)
        else:
          visualizer.add_sphere(pos[i], r, color)
        # Velocity arrows: length = distance covered in arrow_seconds.
        visualizer.add_arrow(pos[i], pos[i] + k * vel[i], color, width=0.02)
      top = base[i] + np.array([0.0, 0.0, 0.15])
      vel_xy = np.array([base_vel[i][0], base_vel[i][1], 0.0])
      visualizer.add_arrow(top, top + k * vel_xy, (1.0, 0.85, 0.1, 1.0), width=0.02)


@dataclass(kw_only=True)
class TargetCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  head_site: str = "head"
  mode_probs: tuple[float, float, float] = (0.3, 0.3, 0.4)
  """Episode mode probabilities: chase only, dodge only, both at once."""
  kind_curriculum: bool = True
  """Throw more of the obstacle kinds with the highest hit rate."""
  kind_curriculum_window: int = 2000
  """Throws between curriculum updates (across all envs)."""
  kind_curriculum_floor: float = 0.15
  """Kind weight = base probability * (floor + hit rate)."""
  near_radius: float = 0.3
  """Chase success metric only: head within this distance of the chase ball."""
  arrow_seconds: float = 0.3
  """Debug velocity arrows show the distance covered in this many seconds."""
  terrain: bool = False
  """Generated terrain: build the height field (ball heights follow the ground)."""
  field_res: float = 0.02
  """Height-field grid [m]: bilinear sampling blurs a step over one cell."""
  cache_dir: str = "logs/heightfields"
  target: BallTargetCfg = field(default_factory=BallTargetCfg)

  def build(self, env: ManagerBasedRlEnv) -> TargetCommand:
    return TargetCommand(self, env)
