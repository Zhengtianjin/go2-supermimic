"""Scripted chase ball and dodge obstacles, batched over environments.

Every env can have a CHASE ball (green, the target) and/or a DODGE stream (red,
the threats), set per episode by `has_chase` / `has_dodge`. With both, the robot
must keep up with the target while getting past what comes at it.

Chase ball (a game of tag): moves near the ground, flees from the robot at
roughly the robot's own speed while juking sideways at random and pausing; its
height wanders a little. The arena wall stops it, so the robot can corner it.
When the robot's head gets within tag_radius, that is a TAG (the scored event):
the ball breaks free with a burst faster than the robot for escape_time, during
which it cannot be tagged, then flees normally again.

Dodge stream: one obstacle at a time, thrown at the robot without any wind-up.
With a chase ball in play the obstacle is carried in a fan in front of the
chase ball, opening toward the robot (the fleeing target throws it back), so it
comes from where the robot is heading and the robot cannot tell when it will be
thrown. Without a chase ball it is thrown
from a point in front of the robot (within +-front_half_angle of the heading),
so a forward camera can see it either way. Throws are slowed down at close
range so the robot keeps at least min_time_to_robot to react, and only a thrown
(flying) obstacle can hit. Its kind decides which motion gets past it:
  BALL      a thrown ball, HIGH (head / top of the trunk), MID (the trunk) or
            LOW (the legs on one side: lifting them is enough); it really
            bounces off the ground and off the robot and rolls away
  LOW_BAR   a horizontal bar sweeping just above the ground, too wide to step
            around in time: jump over it
  HIGH_BAR  a horizontal bar at trunk height: duck under it
Rewards only score the outcome (not hit, keep up, do not fall); the shapes make
jumping and ducking necessary. After an obstacle passes the robot, hits it, or
is in play too long, the next one is carried for throw_gap and thrown.

`advance` is a pure function so that imagined rollouts can keep old states in
the autograd graph and torch.compile can fuse it. The collision model (what
counts as a hit) is in collision.py.

On terrain the balls live in a ground-following frame: heights are measured
above the ground under the robot. With bars_follow_ground, a bar instead keeps
its height above the ground under the bar (low bars roll along the surface) and
is only thrown from ground at least as high as the robot's, so on slopes and
stairs bars come down toward the robot. With static obstacles (obstacles.py) the
chase ball slides along them and an obstacle is only thrown once the line to the
robot is clear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from go2_supermimic.target import obstacles as ob
from go2_supermimic.target import terrain as tr

GRAVITY = 9.81
BALL, LOW_BAR, HIGH_BAR = 0.0, 1.0, 2.0
HIGH_BALL, MID_BALL, LOW_BALL = 0.0, 1.0, 2.0
KIND_NAMES = ("ball", "low bar", "high bar")


@dataclass(kw_only=True)
class BallTargetCfg:
  arena_radius: float = 5.0
  """The chase ball is kept inside this radius around the arena center [m]."""
  curriculum_steps: int = 2000 * 24
  """Env steps over which ranges go from their initial to their final values."""
  ball_radius: float = 0.05
  """Radius of the chase ball and of dodge balls [m]."""

  # Chase ball.
  spawn_dist: tuple[float, float] = (1.5, 3.5)
  ball_z: tuple[float, float] = (0.05, 0.4)
  height_sigma: float = 0.5
  height_damping: float = 3.0
  flee_ratio: tuple[float, float] = (0.75, 0.9)
  """Flee speed as a fraction of the robot's planar speed, sampled per segment.
  It ends up slightly slower to slightly faster, so the robot has to sprint."""
  flee_ratio_final: tuple[float, float] = (0.85, 1.05)
  flee_min_speed: float = 0.5
  flee_gain: tuple[float, float] = (0.3, 0.7)
  """Fraction of the flee speed actually applied; grows with the curriculum."""
  flee_gain_final: tuple[float, float] = (0.8, 1.0)
  speed_max: tuple[float, float] = (1.0, 1.5)
  """Chase ball speed cap (flee + random) per segment [m/s]."""
  speed_max_final: tuple[float, float] = (2.0, 3.0)
  ou_theta: float = 1.5
  """Mean reversion of the random part of the chase ball velocity [1/s]."""
  ou_sigma: float = 4.0
  """Random acceleration of the chase ball (sideways jukes) [m/s^2 per sqrt(s)]."""
  pause_prob: float = 0.1
  """Probability that a segment has no random motion (pure flee)."""
  tag_radius: float = 0.2
  """Head-to-ball-centre horizontal distance that counts as a tag [m]: with the
  5 cm ball and the head site near the nose, this is the nose touching it."""
  tag_sigma: float = 0.3
  """Width of the Gaussian tag reward; its gradient covers the last ~0.6 m, the
  approach-speed reward guides from further away [m]."""
  head_offset: float = 0.33
  """Head position ahead of the base along the heading [m]."""
  escape_speed: tuple[float, float] = (3.0, 4.0)
  """Break-free burst speed after a tag [m/s] (faster than the robot)."""
  escape_time: tuple[float, float] = (1.0, 1.5)
  escape_spread_deg: float = 40.0
  """Burst direction: away from the robot, turned by up to this angle."""

  # Dodge stream.
  kind_probs: tuple[float, float, float] = (0.4, 0.3, 0.3)
  """Probabilities of BALL, LOW_BAR, HIGH_BAR."""
  front_half_angle_deg: float = 35.0
  """Obstacles come from within this angle of the robot heading."""
  throw_dist: tuple[float, float] = (2.5, 4.0)
  """Distance of the throwing point in front of the robot without a chase ball [m]."""
  carry_radius: tuple[float, float] = (0.2, 0.6)
  """Distance of a carried obstacle from the chase ball [m] ..."""
  fan_half_angle_deg: float = 45.0
  """... within this angle of the chase ball -> robot direction (a fan of
  launch points opening toward the robot)."""
  throw_gap: tuple[float, float] = (0.3, 1.0)
  """Random time between throws (the obstacle is carried meanwhile) [s]."""
  min_time_to_robot: float = 0.5
  """Bars are slowed so that, counting the robot running toward them (closing
  speed = bar speed + robot speed toward it), they take at least this long to
  arrive [s]."""
  min_ball_flight: float = 0.35
  """Balls fly at least this long (ball speed alone) [s]."""
  min_throw_dist: float = 1.5
  """A carried obstacle stays at least min_throw_dist + throw_dist_per_speed *
  robot speed from the robot: if the chase ball is closer, the thrower steps
  back behind it, along the robot -> chase ball direction [m]."""
  push_half_angle_deg: float = 20.0
  """A thrower that stepped back stays within this angle of the robot -> chase
  ball direction."""
  throw_dist_per_speed: float = 0.3
  """Extra carry distance per m/s of robot speed [s]."""
  throw_lead: tuple[float, float] = (0.0, 0.0)
  """Fraction of the robot's predicted motion the obstacle is aimed ahead."""
  throw_lead_final: tuple[float, float] = (0.0, 1.0)
  throw_aim_noise: float = 0.15
  """Std of the lateral aim error at the robot [m]."""
  max_flight_time: float = 3.0
  # Balls.
  launch_z: tuple[float, float] = (0.35, 0.7)
  """Throwing-hand height [m]."""
  ball_styles: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
  """Probabilities of HIGH, MID and LOW balls."""
  high_z: tuple[float, float] = (0.30, 0.40)
  """Ball centre height at the robot for HIGH balls (head / top of trunk) [m]."""
  mid_z: tuple[float, float] = (0.18, 0.28)
  """... for MID balls (the trunk) [m]."""
  low_z: tuple[float, float] = (0.05, 0.12)
  """... for LOW balls (the legs) [m]."""
  leg_offset: tuple[float, float] = (0.08, 0.16)
  """Sideways aim offset of LOW balls from the robot centre line (one side's legs) [m]."""
  throw_speed: tuple[float, float] = (3.0, 4.5)
  """Horizontal ball speed [m/s]."""
  throw_speed_final: tuple[float, float] = (4.0, 6.5)
  max_time_to_robot: float = 0.6
  """Balls are at least this fast (dist / time) so they fly like throws, not lobs [s]."""
  restitution: float = 0.7
  bounce_friction: float = 0.85
  """Horizontal speed kept at each ground bounce."""
  min_bounce_speed: float = 0.6
  """Below this vertical speed a ball stops bouncing and rolls [m/s]."""
  roll_decel: float = 1.5
  """Rolling deceleration on the ground [m/s^2]."""
  deflect_keep: float = 0.3
  """On a hit, the ball keeps this fraction of its speed, reversed (bounces off)."""
  retire_dist: float = 2.5
  """A ball that passed or hit the robot is replaced once this far away [m]..."""
  retire_time: float = 1.6
  """... or this long after its throw [s]."""
  # Bars.
  bar_radius: float = 0.03
  bar_half_len: tuple[float, float] = (0.5, 0.8)
  low_bar_z: tuple[float, float] = (0.06, 0.10)
  """Bar centre height of LOW bars (jump over) [m]."""
  high_bar_z: tuple[float, float] = (0.34, 0.40)
  """Bar centre height of HIGH bars (duck under) [m]."""
  bar_speed: tuple[float, float] = (1.5, 2.5)
  bar_speed_final: tuple[float, float] = (2.5, 4.0)
  bars_follow_ground: bool = False
  """On terrain: bars follow the ground under them and come only from uphill."""
  # Static obstacles (tall boxes, walls, pillars) placed per episode.
  obstacles: ob.ObstacleCfg = field(default_factory=ob.ObstacleCfg)


# Per-env state fields and their trailing dims (1 = scalar per env).
STATE_DIMS = {
  "center": 2, "has_chase": 1, "has_dodge": 1,
  # Chase ball.
  "holder_pos": 2, "holder_vel": 2, "ball_z": 1, "ball_z_vel": 1, "speed_cap": 1,
  "flee_gain": 1, "flee_ratio": 1, "paused": 1, "escape": 1, "escape_dir": 2,
  "escape_speed": 1, "tagged": 1, "chase_pos": 3, "chase_vel": 3,
  # Dodge obstacle.
  "throw_speed": 1, "bar_speed": 1, "lead": 1, "wait": 1, "flying": 1, "kind": 1,
  "style": 1, "spent": 1,
  "half_len": 1, "carry_off": 3, "kind_w": 3, "dodge_pos": 3, "dodge_vel": 3,
  # Static obstacles, fixed per episode.
  "obst": ob.NUM_OBSTACLES * ob.OBST_DIM,
}
BOOL_FIELDS = ("has_chase", "has_dodge", "paused", "flying", "tagged", "spent")
STATE_KEYS = tuple(STATE_DIMS)


def _unit(v: torch.Tensor) -> torch.Tensor:
  return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _uniform_like(x: torch.Tensor, lo_hi: tuple[float, float]) -> torch.Tensor:
  return lo_hi[0] + torch.rand_like(x) * (lo_hi[1] - lo_hi[0])


def _next_obstacle(robot_xy, robot_yaw, has_chase, kind_w, cfg: BallTargetCfg):
  """Kind, half length and carry offset (N, 3) of the next obstacle.

  With a chase ball the offset is (radius, fan angle, z): the obstacle sits at
  `radius` from the chase ball, `angle` off the chase ball -> robot direction,
  at absolute height z. Without one it is an absolute point in front of the robot.
  """
  kind = torch.multinomial(kind_w.clamp(min=1e-3), 1).squeeze(-1).float()
  z = torch.where(
    kind == LOW_BAR, _uniform_like(robot_yaw, cfg.low_bar_z),
    torch.where(kind == HIGH_BAR, _uniform_like(robot_yaw, cfg.high_bar_z),
                _uniform_like(robot_yaw, cfg.launch_z)),
  )
  half_len = torch.where(kind == BALL, torch.zeros_like(z), _uniform_like(z, cfg.bar_half_len))
  fan = math.radians(cfg.fan_half_angle_deg) * (torch.rand_like(robot_yaw) * 2 - 1)
  around = torch.stack([_uniform_like(robot_yaw, cfg.carry_radius), fan], -1)
  half = math.radians(cfg.front_half_angle_deg)
  front_ang = robot_yaw + (torch.rand_like(robot_yaw) * 2 - 1) * half
  front = robot_xy + _uniform_like(robot_yaw, cfg.throw_dist).unsqueeze(-1) * torch.stack(
    [front_ang.cos(), front_ang.sin()], -1
  )
  xy = torch.where(has_chase.unsqueeze(-1), around, front)
  return kind, half_len, torch.cat([xy, z.unsqueeze(-1)], -1)


def _carried(chase_pos, chase_vel, carry_off, has_chase, robot_xy, min_dist,
             push_half_angle=math.radians(20.0)):
  """Position and velocity (N, 3) of an obstacle that has not been thrown yet.

  Kept at least min_dist (float or (N, 1)) from the robot: a closer one is moved
  out behind the chase ball, within push_half_angle of the robot -> chase ball
  direction (a plain radial push could put it beside or behind the robot).
  """
  hc = has_chase.unsqueeze(-1)
  to_robot = _unit(robot_xy - chase_pos[:, :2])
  ang = carry_off[:, 1]
  c, s_ = ang.cos(), ang.sin()
  fan_dir = torch.stack(
    [c * to_robot[:, 0] - s_ * to_robot[:, 1], s_ * to_robot[:, 0] + c * to_robot[:, 1]], -1
  )
  in_fan = chase_pos[:, :2] + carry_off[:, :1] * fan_dir
  xy = torch.where(hc, in_fan, carry_off[:, :2])
  rel = xy - robot_xy
  d = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  u = torch.where(hc, -to_robot, rel / d)
  side = torch.atan2(u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0], (u * rel).sum(-1))
  side = side.clamp(-push_half_angle, push_half_angle)
  c, s_ = side.cos(), side.sin()
  out_dir = torch.stack([c * u[:, 0] - s_ * u[:, 1], s_ * u[:, 0] + c * u[:, 1]], -1)
  xy = torch.where(d < min_dist, robot_xy + out_dir * min_dist, xy)
  pos = torch.cat([xy, carry_off[:, 2:]], -1)
  vel = torch.where(
    hc, torch.cat([chase_vel[:, :2], torch.zeros_like(chase_vel[:, :1])], -1),
    torch.zeros_like(chase_vel),
  )
  return pos, vel


class BallTarget:
  def __init__(self, cfg: BallTargetCfg, num_envs: int, device: str):
    self.cfg = cfg
    self.n = num_envs
    self.device = device
    for k, d in STATE_DIMS.items():
      shape = (num_envs,) if d == 1 else (num_envs, d)
      dtype = torch.bool if k in BOOL_FIELDS else torch.float32
      setattr(self, k, torch.zeros(shape, device=device, dtype=dtype))
    self.progress = 0.0
    self.kind_weights = torch.tensor(cfg.kind_probs, device=device, dtype=torch.float32)
    self.field: tr.HeightField | None = None  # terrain, set by the env

  def set_kind_weights(self, weights: torch.Tensor) -> None:
    """Adaptive curriculum: obstacle kind probabilities used from now on (3,)."""
    self.kind_weights = weights / weights.sum()
    self.kind_w[:] = self.kind_weights

  def _uniform(self, n: int, lo_hi: tuple[float, float]) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(*lo_hi)

  def _lerp_range(self, a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    t = self.progress
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

  def set_progress(self, step: int) -> None:
    self.progress = min(1.0, step / max(1, self.cfg.curriculum_steps))

  def state(self) -> dict[str, torch.Tensor]:
    return {k: getattr(self, k) for k in STATE_KEYS}

  def spawn(
    self,
    env_ids: torch.Tensor,
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    center_xy: torch.Tensor,
    has_chase: torch.Tensor,
    has_dodge: torch.Tensor,
  ) -> None:
    """Start an episode: chase ball near the robot and the first (carried) obstacle."""
    n = len(env_ids)
    if n == 0:
      return
    cfg = self.cfg
    self.center[env_ids] = center_xy
    self.has_chase[env_ids] = has_chase
    self.has_dodge[env_ids] = has_dodge
    self.kind_w[env_ids] = self.kind_weights.expand(n, 3)
    if cfg.obstacles.enabled:
      layout = ob.sample_layout(robot_xy, center_xy, cfg.obstacles)
    else:
      layout = torch.zeros(n, ob.NUM_OBSTACLES, ob.OBST_DIM, device=self.device)
    self.obst[env_ids] = layout.reshape(n, -1)

    ang = self._uniform(n, (-math.pi, math.pi))
    pos = robot_xy + self._uniform(n, cfg.spawn_dist).unsqueeze(-1) * torch.stack(
      [ang.cos(), ang.sin()], -1
    )
    rel = pos - center_xy
    r = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    limit = 0.8 * cfg.arena_radius
    pos = torch.where(r > limit, center_xy + rel / r * limit, pos)
    pos = ob.push_out(pos, layout, cfg.ball_radius + 0.05)
    self.holder_pos[env_ids] = pos
    self.holder_vel[env_ids] = 0.0
    self.ball_z[env_ids] = self._uniform(n, cfg.ball_z)
    self.ball_z_vel[env_ids] = 0.0
    self.chase_pos[env_ids] = torch.cat([pos, self.ball_z[env_ids].unsqueeze(-1)], -1)
    self.chase_vel[env_ids] = 0.0
    self.escape[env_ids] = 0.0
    self.tagged[env_ids] = False

    self.respawn_dodge(env_ids, robot_xy, robot_yaw)
    self.resample(env_ids)

  def respawn_dodge(self, env_ids: torch.Tensor, robot_xy: torch.Tensor, robot_yaw: torch.Tensor):
    """Hand the next obstacle to the thrower; it is thrown after a random gap."""
    if len(env_ids) == 0:
      return
    has_chase = self.has_chase[env_ids]
    kind, half_len, off = _next_obstacle(
      robot_xy, robot_yaw, has_chase, self.kind_w[env_ids], self.cfg
    )
    self.kind[env_ids] = kind
    self.half_len[env_ids] = half_len
    self.carry_off[env_ids] = off
    pos, vel = _carried(
      self.chase_pos[env_ids], self.chase_vel[env_ids], off, has_chase, robot_xy,
      self.cfg.min_throw_dist, math.radians(self.cfg.push_half_angle_deg),
    )
    self.dodge_pos[env_ids] = pos
    self.dodge_vel[env_ids] = vel
    self.wait[env_ids] = self._uniform(len(env_ids), self.cfg.throw_gap)
    self.flying[env_ids] = False
    self.spent[env_ids] = False

  def deflect(self, env_ids: torch.Tensor, robot_xy: torch.Tensor) -> None:
    """The obstacle hit the robot: it cannot hit again, and a ball bounces off
    the body (part of its speed, reversed, plus a pop up and outwards)."""
    if len(env_ids) == 0:
      return
    self.spent[env_ids] = True
    v = self.dodge_vel[env_ids]
    out = _unit(self.dodge_pos[env_ids, :2] - robot_xy)
    v_xy = -self.cfg.deflect_keep * v[:, :2] + 1.0 * out
    v_z = v[:, 2].clamp(min=0.0) * self.cfg.deflect_keep + 1.5
    ball = (self.kind[env_ids] == BALL).unsqueeze(-1)
    self.dodge_vel[env_ids] = torch.where(ball, torch.cat([v_xy, v_z.unsqueeze(-1)], -1), v)

  def resample(self, env_ids: torch.Tensor) -> None:
    """New behavior segment: speed cap, flee gain, pause, throw speeds and lead."""
    n = len(env_ids)
    if n == 0:
      return
    cfg = self.cfg
    self.speed_cap[env_ids] = self._uniform(n, self._lerp_range(cfg.speed_max, cfg.speed_max_final))
    self.flee_gain[env_ids] = self._uniform(n, self._lerp_range(cfg.flee_gain, cfg.flee_gain_final))
    self.flee_ratio[env_ids] = self._uniform(
      n, self._lerp_range(cfg.flee_ratio, cfg.flee_ratio_final)
    )
    self.paused[env_ids] = self._uniform(n, (0.0, 1.0)) < cfg.pause_prob
    self.throw_speed[env_ids] = self._uniform(
      n, self._lerp_range(cfg.throw_speed, cfg.throw_speed_final)
    )
    self.bar_speed[env_ids] = self._uniform(n, self._lerp_range(cfg.bar_speed, cfg.bar_speed_final))
    self.lead[env_ids] = self._uniform(n, self._lerp_range(cfg.throw_lead, cfg.throw_lead_final))

  def step(self, dt, robot_xy, robot_vel, robot_yaw) -> None:
    """Advance all envs by dt (N,). dt=0 leaves an env unchanged."""
    for k, v in advance(self.state(), dt, robot_xy, robot_vel, robot_yaw, self.cfg,
                        self.field).items():
      setattr(self, k, v)


def advance(
  st: dict[str, torch.Tensor],
  dt: torch.Tensor,
  robot_xy: torch.Tensor,
  robot_vel: torch.Tensor,
  robot_yaw: torch.Tensor,
  cfg: BallTargetCfg,
  field: tr.HeightField | None = None,
) -> dict[str, torch.Tensor]:
  """Pure (out-of-place) single step of the chase ball and the dodge obstacle.

  robot_xy, robot_vel (N, 2) and robot_yaw (N,) are treated as data; field is
  the terrain (only used when bars follow the ground).
  """
  dt1 = dt.unsqueeze(-1)
  sdt = dt.sqrt()
  active = dt > 0

  ##
  # Chase ball: flee from the robot at ~flee_ratio of its speed, plus random jukes.
  ##
  holder_pos, center = st["holder_pos"], st["center"]
  away = _unit(holder_pos - robot_xy)
  flee_speed = (st["flee_ratio"] * robot_vel.norm(dim=-1)).clamp(min=cfg.flee_min_speed)
  flee = away * (flee_speed * st["flee_gain"]).unsqueeze(-1)
  rand_vel = st["holder_vel"] * (1.0 - cfg.ou_theta * dt1) + torch.randn_like(
    st["holder_vel"]
  ) * cfg.ou_sigma * sdt.unsqueeze(-1)
  rand_vel = torch.where(st["paused"].unsqueeze(-1), torch.zeros_like(rand_vel), rand_vel)
  vel = flee + rand_vel
  # Arena wall: drop the outward component near the wall, push back in beyond it.
  rel = holder_pos - center
  r = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  radial = rel / r
  outside = (r > cfg.arena_radius).float()
  near_wall = (r > cfg.arena_radius - 0.3).float()
  v_out = (vel * radial).sum(-1, keepdim=True).clamp(min=0.0)
  vel = vel - near_wall * v_out * radial - outside * radial * 0.5
  speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  vel = vel * (st["speed_cap"].unsqueeze(-1) / speed).clamp(max=1.0)

  # Tag: the robot's head reaches the ball -> burst away, untaggable meanwhile.
  head = robot_xy + cfg.head_offset * torch.stack([robot_yaw.cos(), robot_yaw.sin()], -1)
  escape = (st["escape"] - dt).clamp(min=0.0)
  tagged = (
    st["has_chase"] & active & (escape <= 0.0)
    & ((holder_pos - head).norm(dim=-1) < cfg.tag_radius)
  )
  spread = math.radians(cfg.escape_spread_deg) * (torch.rand_like(dt) * 2 - 1)
  c, sn = spread.cos(), spread.sin()
  ax, ay = away[:, 0], away[:, 1]
  new_dir = torch.stack([c * ax - sn * ay, sn * ax + c * ay], -1)
  escape_dir = torch.where(tagged.unsqueeze(-1), new_dir, st["escape_dir"])
  escape_speed = torch.where(tagged, _uniform_like(dt, cfg.escape_speed), st["escape_speed"])
  escape = torch.where(tagged, _uniform_like(dt, cfg.escape_time), escape)
  burst = escape_dir * escape_speed.unsqueeze(-1)
  burst = burst - near_wall * (burst * radial).sum(-1, keepdim=True).clamp(min=0.0) * radial
  vel = torch.where((escape > 0).unsqueeze(-1), burst, vel)
  holder_pos = holder_pos + vel * dt1
  obst = ob.unpack(st["obst"])
  if cfg.obstacles.enabled:
    holder_pos = ob.push_out(holder_pos, obst, cfg.ball_radius + 0.05)

  z_vel = st["ball_z_vel"] * (1.0 - cfg.height_damping * dt) + torch.randn_like(
    dt
  ) * cfg.height_sigma * sdt
  z_new = st["ball_z"] + z_vel * dt
  lo, hi = cfg.ball_z
  z_vel = torch.where((z_new < lo) | (z_new > hi), -z_vel, z_vel)
  z = z_new.clamp(lo, hi)
  chase_pos = torch.cat([holder_pos, z.unsqueeze(-1)], dim=-1)
  chase_vel = torch.cat([vel, z_vel.unsqueeze(-1)], dim=-1)

  ##
  # Dodge obstacle: carried until thrown (no wind-up). Balls: fastball / skip /
  # lob under gravity, bouncing off the ground and rolling. Bars: flat. The next
  # obstacle comes once this one is done.
  ##
  kind, carry_off, has_chase = st["kind"], st["carry_off"], st["has_chase"]
  is_ball = kind == BALL
  min_dist = cfg.min_throw_dist + cfg.throw_dist_per_speed * robot_vel.norm(dim=-1, keepdim=True)
  push = math.radians(cfg.push_half_angle_deg)
  carried_pos, carried_vel = _carried(
    chase_pos, chase_vel, carry_off, has_chase, robot_xy, min_dist, push
  )
  pos = torch.where(st["flying"].unsqueeze(-1), st["dodge_pos"], carried_pos)
  wait = st["wait"] - dt
  launch = (~st["flying"]) & (wait <= 0.0) & active
  follow = cfg.bars_follow_ground and field is not None
  if follow:
    # A bar is thrown only from ground at least as high as the robot's.
    under_robot = tr.ground(robot_xy, field)
    launch = launch & (is_ball | (tr.ground(pos[:, :2], field) >= under_robot - 0.01))
  if cfg.obstacles.enabled:
    launch = launch & ob.line_of_sight(pos[:, :2], robot_xy, obst)
  if follow or cfg.obstacles.enabled:
    # Held while blocked: the flight clock (-wait) starts at the throw.
    wait = torch.where(~st["flying"] & ~launch, wait.clamp(min=0.0), wait)
  to_robot = robot_xy - pos[:, :2]
  dist = to_robot.norm(dim=-1).clamp(min=0.1)
  # Never faster than arriving in min_time_to_robot, counting the robot running
  # toward the throw (time to react).
  closing = (robot_vel * to_robot / dist.unsqueeze(-1)).sum(-1).neg().clamp(min=0.0)
  slowest_fair = (dist / cfg.min_time_to_robot - closing).clamp(min=0.5)
  probs = torch.tensor(cfg.ball_styles, device=dt.device, dtype=torch.float32)
  new_style = torch.multinomial(probs, dt.shape[0], replacement=True).float()
  style = torch.where(launch, new_style, st["style"])
  # Balls are not slowed for a robot running at the thrower: a real throw is not.
  ball_speed = torch.minimum(
    torch.maximum(st["throw_speed"], dist / cfg.max_time_to_robot), dist / cfg.min_ball_flight
  )
  speed = torch.where(is_ball, ball_speed, torch.minimum(st["bar_speed"], slowest_fair))
  t_hit = dist / speed
  aim = robot_xy + st["lead"].unsqueeze(-1) * robot_vel * t_hit.unsqueeze(-1)
  aim_dir = _unit(aim - pos[:, :2])
  lateral = torch.stack([-aim_dir[:, 1], aim_dir[:, 0]], -1)
  low = style == LOW_BALL
  side = torch.where(torch.rand_like(dt) < 0.5, -1.0, 1.0) * _uniform_like(dt, cfg.leg_offset)
  offset = torch.randn_like(dt) * cfg.throw_aim_noise + torch.where(low, side, torch.zeros_like(side))
  aim = aim + lateral * offset.unsqueeze(-1)
  v_xy = _unit(aim - pos[:, :2]) * speed.unsqueeze(-1)
  # Vertical launch speed to reach the robot at the style's height under gravity.
  t_target = ((aim - pos[:, :2]).norm(dim=-1) / speed).clamp(min=0.05)
  z_target = torch.where(
    style == HIGH_BALL, _uniform_like(dt, cfg.high_z),
    torch.where(low, _uniform_like(dt, cfg.low_z), _uniform_like(dt, cfg.mid_z)),
  )
  vz0 = (z_target - pos[:, 2] + 0.5 * GRAVITY * t_target**2) / t_target
  vz0 = torch.where(is_ball, vz0, torch.zeros_like(vz0))
  launch_vel = torch.cat([v_xy, vz0.unsqueeze(-1)], -1)

  flying = st["flying"] | launch
  v = torch.where(launch.unsqueeze(-1), launch_vel, st["dodge_vel"])
  ballistic = (flying & is_ball).float()
  v_z = v[:, 2] - GRAVITY * dt * ballistic
  v_h = v[:, :2]
  p = pos + torch.cat([v_h, v_z.unsqueeze(-1)], -1) * dt1
  # Ground: bounce with restitution and friction; too slow to bounce -> roll.
  on_ground = is_ball & (p[:, 2] <= cfg.ball_radius)
  impact = on_ground & (v_z < 0)
  v_z = torch.where(impact, -cfg.restitution * v_z, v_z)
  v_h = torch.where(impact.unsqueeze(-1), v_h * cfg.bounce_friction, v_h)
  rolling = on_ground & (v_z.abs() < cfg.min_bounce_speed)
  v_z = torch.where(rolling, torch.zeros_like(v_z), v_z)
  h_speed = v_h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  slowed = (1.0 - cfg.roll_decel * dt1 / h_speed).clamp(min=0.0)
  v_h = torch.where(rolling.unsqueeze(-1), v_h * slowed, v_h)
  p_z = torch.where(on_ground, torch.full_like(dt, cfg.ball_radius), p[:, 2])
  if follow:  # bars: their height above the ground under them, in the robot's frame
    rise = tr.ground(p[:, :2], field) - under_robot
    p_z = torch.where(is_ball, p_z, carry_off[:, 2] + rise)
  p = torch.cat([p[:, :2], p_z.unsqueeze(-1)], -1)
  thrown_vel = torch.cat([v_h, v_z.unsqueeze(-1)], -1)

  # Done: a bar 1 m past the robot; a ball that passed or hit it once it is
  # retire_dist away, has flown retire_time, or has stopped rolling.
  rel_xy = p[:, :2] - robot_xy
  away_now = ((rel_xy * v_h).sum(-1) > 0) | st["spent"]
  r_now = rel_xy.norm(dim=-1)
  flown = -wait
  bar_done = (~is_ball) & flying & away_now & (r_now > 1.0)
  ball_done = is_ball & flying & (
    (away_now & ((r_now > cfg.retire_dist) | (flown > cfg.retire_time)))
    | ((h_speed[:, 0] < 0.2) & (flown > 0.3))
  )
  passed = bar_done | ball_done | (flying & (flown > cfg.max_flight_time))
  new_kind, new_half, new_off = _next_obstacle(robot_xy, robot_yaw, has_chase, st["kind_w"], cfg)
  p2 = passed.unsqueeze(-1)
  kind = torch.where(passed, new_kind, kind)
  half_len = torch.where(passed, new_half, st["half_len"])
  carry_off = torch.where(p2, new_off, carry_off)
  wait = torch.where(passed, _uniform_like(dt, cfg.throw_gap), wait)
  flying = flying & ~passed
  spent = st["spent"] & ~passed
  carried_pos, carried_vel = _carried(
    chase_pos, chase_vel, carry_off, has_chase, robot_xy, min_dist, push
  )
  if follow:
    rise = tr.ground(carried_pos[:, :2], field) - under_robot
    carried_z = carried_pos[:, 2] + torch.where(kind == BALL, torch.zeros_like(rise), rise)
    carried_pos = torch.cat([carried_pos[:, :2], carried_z.unsqueeze(-1)], -1)
  f2 = flying.unsqueeze(-1)
  dodge_pos = torch.where(f2, p, carried_pos)
  dodge_vel = torch.where(f2, thrown_vel, carried_vel)

  out = dict(st)
  out.update(
    holder_pos=holder_pos, holder_vel=rand_vel, ball_z=z, ball_z_vel=z_vel,
    escape=escape, escape_dir=escape_dir, escape_speed=escape_speed, tagged=tagged,
    chase_pos=chase_pos, chase_vel=chase_vel,
    wait=wait, flying=flying, kind=kind, half_len=half_len, carry_off=carry_off,
    style=style, spent=spent,
    dodge_pos=dodge_pos, dodge_vel=dodge_vel,
  )
  return out


def dodge_shape(st: dict[str, torch.Tensor]) -> torch.Tensor:
  """Observation shape features (N, 2) of the dodge obstacle: [is_bar, half_len]."""
  return torch.stack([(st["kind"] != BALL).float(), st["half_len"]], -1)
