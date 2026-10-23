"""Checks for the chase ball, the dodge obstacles, the collision model and the
target observation (no simulator needed). Run: python -m pytest tests
"""

import math

import torch

from go2_supermimic.target import interface as ti
from go2_supermimic.target.collision import (
  crouch_violation,
  dodge_clearance,
  dodge_shaping,
  robot_clearance,
  robot_geometry,
  tag_activation,
)
from go2_supermimic.target.generator import (
  BALL,
  HIGH_BALL,
  HIGH_BAR,
  LOW_BALL,
  LOW_BAR,
  MID_BALL,
  BallTarget,
  BallTargetCfg,
)
from go2_supermimic.target.interface import yaw_rotate_inverse

DEV = "cpu"


def test_yaw_rotate_inverse():
  vec = torch.tensor([[0.0, 1.0, 0.5]])
  out = yaw_rotate_inverse(vec, torch.tensor([math.pi / 2]))
  assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.5]]), atol=1e-6)


def _gen(n, chase=True, dodge=True):
  cfg = BallTargetCfg()
  gen = BallTarget(cfg, n, DEV)
  gen.set_progress(10**9)  # hardest targets
  gen.spawn(
    torch.arange(n), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 2),
    torch.full((n,), chase), torch.full((n,), dodge),
  )
  return cfg, gen


def test_chase_ball_stays_in_arena_and_low():
  torch.manual_seed(0)
  n = 256
  cfg, gen = _gen(n)
  robot, robot_vel, yaw = torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n)
  dt = torch.full((n,), 0.02)
  max_r = 0.0
  for i in range(3000):
    if i % 200 == 0:
      gen.resample(torch.arange(n))
    gen.step(dt, robot, robot_vel, yaw)
    max_r = max(max_r, gen.holder_pos.norm(dim=-1).max().item())
  lo, hi = cfg.ball_z
  assert ((gen.chase_pos[:, 2] >= lo - 1e-6) & (gen.chase_pos[:, 2] <= hi + 1e-6)).all()
  assert max_r < cfg.arena_radius + 1.0, max_r
  assert torch.isfinite(gen.chase_vel).all()


def test_chase_ball_flees_slower_than_robot():
  torch.manual_seed(0)
  n = 512
  cfg, gen = _gen(n)
  robot, yaw = torch.zeros(n, 2), torch.zeros(n)
  robot_vel = torch.tensor([[2.0, 0.0]]).expand(n, 2)
  d0 = gen.holder_pos.norm(dim=-1)
  dt = torch.full((n,), 0.02)
  for _ in range(25):  # 0.5 s
    gen.step(dt, robot, robot_vel, yaw)
  gain = (gen.holder_pos.norm(dim=-1) - d0) / 0.5
  assert gain.mean() > 0.5, gain.mean()
  assert gain.mean() < cfg.flee_ratio_final[1] * 2.0 + 0.3, gain.mean()


def _run_dodge(gen, cfg, steps, robot, robot_vel, yaw, dt):
  min_dist = torch.full((gen.n,), 1e9)
  rethrown = torch.zeros(gen.n, dtype=torch.bool)
  kinds, max_z = set(), 0.0
  was_flying = gen.flying.clone()
  for _ in range(steps):
    gen.step(dt, robot, robot_vel, yaw)
    kinds |= set(gen.kind[gen.flying].tolist())
    max_z = max(max_z, gen.dodge_pos[gen.kind == 0, 2].max().item() if (gen.kind == 0).any() else 0)
    min_dist = torch.minimum(min_dist, gen.dodge_pos[:, :2].norm(dim=-1))
    rethrown |= was_flying & ~gen.flying
    was_flying = gen.flying.clone()
  return min_dist, rethrown, kinds, max_z


def test_dodge_only_throws_from_front_without_windup():
  torch.manual_seed(0)
  n = 512
  cfg, gen = _gen(n, chase=False, dodge=True)
  robot, robot_vel, yaw = torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n)
  half = math.radians(cfg.front_half_angle_deg)
  bearing = torch.atan2(gen.dodge_pos[:, 1], gen.dodge_pos[:, 0])
  assert (bearing.abs() <= half + 1e-4).all(), "throws must start in front"
  assert (gen.wait <= cfg.throw_gap[1] + 1e-6).all()
  dt = torch.full((n,), 0.02)
  min_dist, rethrown, kinds, max_z = _run_dodge(gen, cfg, 200, robot, robot_vel, yaw, dt)
  assert (min_dist < 0.6).float().mean() > 0.8, min_dist
  assert rethrown.float().mean() > 0.9
  assert kinds == {0.0, 1.0, 2.0}, kinds
  # Lobs from a 1.4 m hand peak around 2.5 m; everything else stays lower.
  assert max_z < 2.8, f"ball arcs must stay low, got apex {max_z:.2f} m"


def test_chase_ball_carries_and_throws_obstacles():
  torch.manual_seed(0)
  n = 512
  cfg, gen = _gen(n, chase=True, dodge=True)
  robot, robot_vel, yaw = torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n)
  # Not thrown yet: by the chase ball, never closer than min_throw_dist to the
  # robot, and it cannot hit.
  off = (gen.dodge_pos[:, :2] - gen.chase_pos[:, :2]).norm(dim=-1)
  assert (off <= cfg.carry_radius[1] + cfg.min_throw_dist + 1e-4).all() and (~gen.flying).all()
  assert (gen.dodge_pos[:, :2].norm(dim=-1) >= cfg.min_throw_dist - 1e-4).all()
  g = {k: v.expand(n, *v.shape[1:]) for k, v in _stance().items()}
  clear = dodge_clearance(gen.state(), g, cfg)
  assert (clear >= 9.0).all(), "a carried obstacle must not count as a hit"
  dt = torch.full((n,), 0.02)
  min_dist, rethrown, _, _ = _run_dodge(gen, cfg, 150, robot, robot_vel, yaw, dt)
  assert (min_dist < 0.6).float().mean() > 0.8, "thrown obstacles must reach the robot"
  assert rethrown.float().mean() > 0.8


def test_dodge_terms_gated_by_flags_and_kind():
  n = 4
  _, gen = _gen(n, chase=True, dodge=False)
  st = gen.state()
  st["flying"] = torch.ones(n, dtype=torch.bool)
  st["kind"] = torch.full((n,), HIGH_BAR)
  st["dodge_pos"] = torch.tensor([[0.5, 0.0, 0.37]]).expand(n, 3).clone()
  xy, low_z = torch.zeros(n, 2), torch.full((n,), 0.18)
  assert (dodge_shaping(st, xy, low_z) == 0).all(), "no dodge obstacle -> no hint"
  assert (crouch_violation(st, xy, low_z) > 0).all(), "crouching not allowed without a bar"
  st["has_dodge"] = torch.ones(n, dtype=torch.bool)
  assert (dodge_shaping(st, xy, low_z) > 0).all()
  assert (crouch_violation(st, xy, low_z) == 0).all(), "ducking allowed under a high bar"
  st["kind"] = torch.full((n,), BALL)
  st["style"] = torch.full((n,), MID_BALL)
  assert (dodge_shaping(st, xy, low_z) == 0).all(), "no hint for mid balls (sidestep)"
  assert (crouch_violation(st, xy, low_z) > 0).all(), "no ducking under a mid ball"
  st["style"] = torch.full((n,), HIGH_BALL)
  assert (dodge_shaping(st, xy, low_z) > 0).all(), "duck under a high ball"
  assert (crouch_violation(st, xy, low_z) == 0).all()
  st["style"] = torch.full((n,), LOW_BALL)
  stand = torch.full((n,), 0.27)
  feet_down, feet_one_up = torch.full((n, 4), 0.02), torch.tensor([[0.15, 0.02, 0.02, 0.02]]).expand(n, 4)
  assert (dodge_shaping(st, xy, stand, feet_down) == 0).all()
  assert (dodge_shaping(st, xy, stand, feet_one_up) > 0).all(), "lift a leg for a low ball"


def test_kind_weights_steer_throws():
  torch.manual_seed(0)
  n = 2048
  cfg, gen = _gen(n, chase=False, dodge=True)
  gen.set_kind_weights(torch.tensor([0.0, 1.0, 0.0]))
  gen.respawn_dodge(torch.arange(n), torch.zeros(n, 2), torch.zeros(n))
  assert (gen.kind == LOW_BAR).float().mean() > 0.98


def test_throws_come_from_the_chase_ball_fast_and_flat():
  torch.manual_seed(0)
  n = 1024
  cfg, gen = _gen(n, chase=True, dodge=True)
  gen.set_kind_weights(torch.tensor([1.0, 0.0, 0.0]))
  robot, yaw = torch.zeros(n, 2), torch.zeros(n)
  dt = torch.full((n,), 0.02)
  seen, rises = 0, []
  for _ in range(150):
    # The robot sprints at 2.5 m/s straight at the chase ball (and its thrower),
    # staying close to it, as a good chaser does.
    to_ball = gen.chase_pos[:, :2] - robot
    robot_vel = 2.5 * to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    was_flying = gen.flying.clone()
    gen.step(dt, robot, robot_vel, yaw)
    launched = gen.flying & ~was_flying & (gen.kind == BALL)  # the first one may be a bar
    if launched.any():
      seen += int(launched.sum())
      rel = gen.dodge_pos[launched, :2] - robot[launched]
      dist = rel.norm(dim=-1)
      tb = to_ball[launched]
      cos = (rel * tb).sum(-1) / (dist * tb.norm(dim=-1)).clamp(min=1e-6)
      # From around the chase ball, never from beside or behind the robot.
      assert (cos >= math.cos(math.radians(35.0))).all(), cos.min()
      min_d = cfg.min_throw_dist + cfg.throw_dist_per_speed * 2.5
      assert (dist >= min_d - 0.3).all(), dist.min()
      speed = gen.dodge_vel[launched, :2].norm(dim=-1)
      # Full-difficulty throws are fast (not slowed for a robot running at them)
      # but still take min_ball_flight to arrive.
      assert speed.mean() > 4.0, speed.mean()
      # (dodge_pos is one step into the flight: add that step back.)
      assert (dist / speed + 0.02 >= cfg.min_ball_flight - 1e-3).all()
      # Flat: at most a slight arc (well under 0.4 m above the hand), mostly flat.
      rise = gen.dodge_vel[launched, 2].clamp(min=0) ** 2 / (2 * 9.81)
      rises.append(rise)
  assert seen > 200, seen
  rise = torch.cat(rises)
  assert (rise < 0.4).all() and rise.mean() < 0.15, (rise.max(), rise.mean())


def test_tag_triggers_escape_burst_and_reward_switches_off():
  torch.manual_seed(0)
  n = 64
  cfg, gen = _gen(n, chase=True, dodge=False)
  yaw = torch.zeros(n)
  # Put the robot head right at the ball (head is head_offset ahead of the base).
  robot = gen.holder_pos - torch.tensor([cfg.head_offset, 0.0])
  vel = torch.zeros(n, 2)
  assert (tag_activation(gen.state(), robot, yaw, cfg) > 0.9).all()
  dt = torch.full((n,), 0.02)
  gen.step(dt, robot, vel, yaw)
  assert gen.tagged.all() and (gen.escape > 0).all()
  assert (tag_activation(gen.state(), robot, yaw, cfg) == 0).all(), "no reward while escaping"
  start = gen.holder_pos.clone()
  for _ in range(25):  # 0.5 s
    gen.step(dt, robot, vel, yaw)
    assert not gen.tagged.any(), "cannot be tagged again while breaking free"
  speed = (gen.holder_pos - start).norm(dim=-1) / 0.5
  assert (speed > cfg.escape_speed[0] * 0.8).float().mean() > 0.9, speed


def _stance(base_z=0.27, lift=None, all_up=0.0, side=0.0):
  """Synthetic Go2 pose: legs FR FL RR RL; `lift` raises one leg's knee and foot,
  `all_up` raises every knee and foot (a jump), `side` shifts the robot in y."""
  hips, knees, feet = [], [], []
  for i, (x, y) in enumerate(((0.19, -0.14), (0.19, 0.14), (-0.19, -0.14), (-0.19, 0.14))):
    up = all_up + (0.18 if lift == i else 0.0)
    hips.append([x, y + side, base_z])
    knees.append([x - 0.08, y + side, min(0.13 + up, base_z - 0.02)])
    feet.append([x, y + side, 0.022 + up])
  t = lambda v: torch.tensor([v])  # noqa: E731
  return robot_geometry(
    torch.tensor([[0.0, side]]), t(base_z), t(0.0),
    {"hips": torch.tensor([hips]), "knees": torch.tensor([knees]), "feet": torch.tensor([feet])},
  )


def test_per_limb_collision_rewards_the_right_dodge():
  r_ball, r_bar = 0.05, 0.03
  fr_foot_ball = torch.tensor([[0.19, -0.14, 0.08]])
  assert robot_clearance(fr_foot_ball, r_ball, _stance()) <= 0
  assert robot_clearance(fr_foot_ball, r_ball, _stance(lift=0)) > 0, "lifting one leg clears it"

  # A low bar across the robot at the front legs: one lifted leg is not enough, a jump is.
  bar = torch.stack([torch.tensor([[0.19, y, 0.08]]) for y in torch.linspace(-0.6, 0.6, 13)])
  hit = lambda g: robot_clearance(bar, r_bar, g).min()  # noqa: E731
  assert hit(_stance()) <= 0
  assert hit(_stance(lift=0)) <= 0
  assert hit(_stance(base_z=0.50, all_up=0.20)) > 0

  # A high bar over the trunk: duck under it.
  high = torch.stack([torch.tensor([[0.0, y, 0.37]]) for y in torch.linspace(-0.6, 0.6, 13)])
  assert robot_clearance(high, r_bar, _stance()).min() <= 0
  assert robot_clearance(high, r_bar, _stance(base_z=0.18)).min() > 0

  # A body-height ball: step aside.
  body_ball = torch.tensor([[0.0, 0.0, 0.25]])
  assert robot_clearance(body_ball, r_ball, _stance()) <= 0
  assert robot_clearance(body_ball, r_ball, _stance(side=0.45)) > 0


def test_carried_obstacles_sit_in_a_fan_toward_the_robot():
  torch.manual_seed(0)
  n = 2048
  cfg, gen = _gen(n, chase=True, dodge=True)
  robot = torch.zeros(n, 2)
  to_robot = robot - gen.chase_pos[:, :2]
  off = gen.dodge_pos[:, :2] - gen.chase_pos[:, :2]
  far = to_robot.norm(dim=-1) > cfg.min_throw_dist + cfg.carry_radius[1] + 0.1  # not pushed out
  cos = (off * to_robot).sum(-1) / (off.norm(dim=-1) * to_robot.norm(dim=-1)).clamp(min=1e-6)
  half = math.radians(cfg.fan_half_angle_deg)
  assert (cos[far] >= math.cos(half) - 1e-4).all(), "launch points must open toward the robot"


def test_three_ball_heights():
  torch.manual_seed(0)
  n = 3000
  cfg, gen = _gen(n, chase=False, dodge=True)
  gen.set_kind_weights(torch.tensor([1.0, 0.0, 0.0]))
  gen.respawn_dodge(torch.arange(n), torch.zeros(n, 2), torch.zeros(n))
  robot, vel, yaw = torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n)
  dt = torch.full((n,), 0.02)
  closest_z = torch.full((n,), float("nan"))
  closest_style = torch.full((n,), -1.0)
  best = torch.full((n,), 1e9)
  for _ in range(60):  # 1.2 s: the first throw only
    gen.step(dt, robot, vel, yaw)
    d = gen.dodge_pos[:, :2].norm(dim=-1)
    closer = gen.flying & (d < best)
    best = torch.where(closer, d, best)
    closest_z = torch.where(closer, gen.dodge_pos[:, 2], closest_z)
    closest_style = torch.where(closer, gen.style, closest_style)
  means = []
  for style in (0.0, 1.0, 2.0):
    m = (closest_style == style) & (best < 0.4)
    assert m.sum() > 50, style
    means.append(closest_z[m].mean().item())
  # 50 Hz sampling of fast descending throws reads the height a little early,
  # so check the ordering and that LOW balls really are at leg height.
  assert means[0] > means[1] > means[2], means
  assert means[2] < 0.2, means


def test_oracle_observation_is_truth_and_inactive_slots_are_zero():
  n = 4
  _, gen = _gen(n)
  gen.has_dodge[:2] = False
  st = gen.state()
  base = torch.tensor([[0.5, -0.2, 0.3]]).expand(n, 3).clone()
  obs = ti.oracle_observation(st, base, torch.zeros(n))
  chase = obs[:, : ti.SLOT_DIM]
  dodge = obs[:, ti.SLOT_DIM :]
  assert torch.allclose(chase[:, ti.P], st["chase_pos"] - base)
  assert torch.allclose(chase[:, ti.V], st["chase_vel"])
  assert (chase[:, ti.VISIBLE] == 1).all() and (chase[:, ti.SIGMA] == 0).all()
  assert (dodge[:2] == 0).all(), "inactive slot must be all zeros"
  assert (dodge[2:, ti.ACTIVE] == 1).all()
