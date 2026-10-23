"""Checks for the static obstacles (no simulator needed). Run: python -m pytest tests
"""

import math

import torch

from go2_supermimic.target import obstacles as ob
from go2_supermimic.target import terrain as tr
from go2_supermimic.target.collision import obstacle_penetration, robot_geometry
from go2_supermimic.target.generator import BallTarget, BallTargetCfg

K = ob.NUM_OBSTACLES

def _layout(n, boxes):
  """Layout (N, K, 4) with the given (slot, x, y, yaw) boxes active."""
  out = torch.zeros(n, K, ob.OBST_DIM)
  for k, x, y, yaw in boxes:
    out[:, k] = torch.tensor([x, y, yaw, 1.0])
  return out


def test_lidar_sdf_and_line_of_sight_on_a_known_box():
  # Box slot 2 (half 0.45 x 0.30) centred 2 m ahead, unrotated.
  obst = _layout(1, [(2, 2.0, 0.0, 0.0)])
  r = ob.lidar(torch.zeros(1, 2), torch.zeros(1), obst, 16, 4.0)
  assert abs(r[0, 0].item() - 1.55) < 1e-4, r[0, 0]  # straight ahead: 2 - 0.45
  assert abs(r[0, 8].item() - 4.0) < 1e-4  # straight back: nothing
  assert (r <= 4.0 + 1e-6).all()
  # Rotating the robot rotates the scan.
  r90 = ob.lidar(torch.zeros(1, 2), torch.tensor([math.pi / 2]), obst, 16, 4.0)
  assert abs(r90[0, 12].item() - 1.55) < 1e-4  # ray at yaw + 270 deg points along +x
  d = ob.sdf(torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]), obst)[0, :, 2]
  assert abs(d[0].item() - 0.55) < 1e-4 and d[1].item() < 0
  a, b = torch.zeros(1, 2), torch.tensor([[4.0, 0.0]])
  assert not ob.line_of_sight(a, b, obst)[0]
  assert ob.line_of_sight(a, torch.tensor([[4.0, 2.0]]), obst)[0]
  assert ob.line_of_sight(a, torch.tensor([[1.0, 0.0]]), obst)[0], "stops short of the box"
  # Inactive slots never block.
  assert ob.line_of_sight(a, b, torch.zeros(1, K, ob.OBST_DIM))[0]


def test_push_out_moves_points_to_the_nearest_side():
  obst = _layout(3, [(2, 0.0, 0.0, 0.3)])
  pts = torch.tensor([[0.40, 0.0], [0.0, 0.1], [2.0, 2.0]])
  out = ob.push_out(pts, obst, 0.05)
  d = ob.sdf(out.unsqueeze(1), obst)[:, 0, 2]
  assert (d >= 0.05 - 1e-4).all(), d
  assert torch.allclose(out[2], pts[2]), "points outside stay put"


def test_layout_keeps_clear_of_robot_and_leaves_corridors():
  torch.manual_seed(0)
  n = 2000
  cfg = ob.ObstacleCfg(enabled=True)
  robot = torch.randn(n, 2)
  layout = ob.sample_layout(robot, torch.zeros(n, 2), cfg)
  active = layout[..., 3] > 0.5
  rad = ob.bounding_radius("cpu")
  assert active.float().mean() > 0.5
  d_robot = (layout[..., :2] - robot.unsqueeze(1)).norm(dim=-1)
  assert (d_robot[active] > (rad + cfg.robot_clearance).expand(n, K)[active] - 1e-4).all()
  for i in range(K):
    for j in range(i):
      both = active[:, i] & active[:, j]
      d = (layout[:, i, :2] - layout[:, j, :2]).norm(dim=-1)
      assert (d[both] > rad[i] + rad[j] + cfg.gap - 1e-4).all()
  assert (layout[..., :2].norm(dim=-1)[active] <= cfg.place_radius + 1e-4).all()


def test_throws_wait_for_a_clear_line():
  torch.manual_seed(0)
  n = 600
  cfg = BallTargetCfg()
  cfg.obstacles = ob.ObstacleCfg(enabled=True)
  gen = BallTarget(cfg, n, "cpu")
  gen.set_progress(10**9)
  gen.spawn(torch.arange(n), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 2),
            torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
  # A wall (slot 4, half 1.0 x 0.05) across the line to every carried obstacle.
  ang = torch.atan2(gen.dodge_pos[:, 1], gen.dodge_pos[:, 0])
  wall = torch.zeros(n, K, ob.OBST_DIM)
  wall[:, 4] = torch.stack([0.9 * ang.cos(), 0.9 * ang.sin(), ang, torch.ones(n)], -1)
  gen.obst = wall.reshape(n, -1)
  dt = torch.full((n,), 0.02)
  for _ in range(100):
    gen.step(dt, torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n))
    assert not gen.flying.any(), "no throw through a wall"
  gen.obst.zero_()
  gen.step(dt, torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n))
  assert gen.flying.all() and (gen.wait > -0.03).all(), "throws at once, clock starts now"


def test_obstacle_penetration_is_zero_when_clear_and_grows_inside():
  base_xy = torch.tensor([[0.0, 0.0]])
  feet = torch.tensor([[[0.19, -0.14, 0.02], [0.19, 0.14, 0.02],
                        [-0.19, -0.14, 0.02], [-0.19, 0.14, 0.02]]])
  knees = feet + torch.tensor([0.0, 0.0, 0.12])
  g = robot_geometry(base_xy, torch.tensor([0.3]), torch.tensor([0.0]),
                     {"hips": knees, "knees": knees, "feet": feet})
  far = _layout(1, [(0, 2.0, 0.0, 0.0)]).reshape(1, -1)
  touching = _layout(1, [(0, 0.5, 0.0, 0.0)]).reshape(1, -1)
  deep = _layout(1, [(0, 0.3, 0.0, 0.0)]).reshape(1, -1)
  p_far = obstacle_penetration(g, far)
  p_touch = obstacle_penetration(g, touching)
  p_deep = obstacle_penetration(g, deep)
  assert p_far.item() == 0.0
  assert 0 < p_touch.item() < p_deep.item()


def test_ball_never_enters_obstacles():
  torch.manual_seed(0)
  n = 400
  cfg = BallTargetCfg()
  cfg.obstacles = ob.ObstacleCfg(enabled=True, empty_prob=0.0)
  gen = BallTarget(cfg, n, "cpu")
  gen.set_progress(10**9)
  gen.spawn(torch.arange(n), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 2),
            torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
  robot = torch.zeros(n, 2)
  for i in range(500):
    if i % 100 == 0:
      gen.resample(torch.arange(n))
    gen.step(torch.full((n,), 0.02), robot, torch.full((n, 2), 1.0), torch.zeros(n))
    d = ob.sdf(gen.holder_pos.unsqueeze(1), ob.unpack(gen.obst))[:, 0]
    assert (d > cfg.ball_radius).all()


def test_obstacle_rises_in_the_scan_ahead_of_the_robot():
  obst = torch.zeros(1, K, ob.OBST_DIM)
  obst[0, 4] = torch.tensor([0.75, 0.0, 1.5708, 1.0])  # a wall across the path, 0.75 m ahead
  r = tr.relief(torch.zeros(1, 2), torch.zeros(1), None, obst.reshape(1, -1))
  pts = tr.scan_offsets("cpu")
  in_wall = (pts[:, 0] - 0.75).abs() < 0.03
  assert in_wall.any()
  assert (r[0, in_wall] < -0.9).all(), "inside the wall: ground 1 m above"
  assert (r[0, pts[:, 0] < 0.5].abs() < 1e-3).all(), "free ground behind and under the robot"
  # Turning away moves the wall out of the scan.
  r_back = tr.relief(torch.zeros(1, 2), torch.tensor([3.1416]), None, obst.reshape(1, -1))
  assert (r_back.abs() < 1e-3).all()

