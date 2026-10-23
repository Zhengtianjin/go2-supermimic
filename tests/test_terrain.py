"""Checks for the height scan and the exact warm starts onto terrain (no simulator
needed). Run: python -m pytest tests
"""

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.target import terrain as tr


def test_relief_is_zero_on_flat_ground():
  r = tr.relief(torch.randn(5, 2), torch.randn(5), None)
  assert r.shape == (5, tr.SCAN_POINTS) and (r == 0).all()


def test_height_field_bilinear_and_relief_on_a_step():
  # Ground: 0 for x < 1, 0.15 m (a stair) for x >= 1; 5 cm grid.
  xs = torch.arange(0, 4.0001, 0.05)
  h = (xs >= 1.0).float() * 0.15
  field = tr.HeightField(h.expand(41, -1).clone(), (0.0, -1.0), 0.05)
  assert abs(field.sample(torch.tensor([[2.0, 0.0]])).item() - 0.15) < 1e-6
  assert abs(field.sample(torch.tensor([[0.975, 0.0]])).item() - 0.075) < 1e-4  # halfway up the edge
  r = tr.relief(torch.tensor([[0.6, 0.0]]), torch.zeros(1), field)
  pts = tr.scan_offsets("cpu")
  ahead = pts[:, 0] + 0.6 > 1.05
  assert (r[0, ahead] < -0.14).all() and (r[0, pts[:, 0] + 0.6 < 0.95].abs() < 1e-5).all()


def test_world_model_scan_warm_start_is_exact():
  from go2_supermimic.supermimic.train import expand_world_model_scan
  from go2_supermimic.supermimic.world_model import RunningNorm, WorldModel, mlp

  torch.manual_seed(0)
  hidden = (64, 64)
  new = WorldModel(hidden)
  in_old = st.STATE_DIM + 2 * st.NUM_JOINTS + st.CONTEXT_DIM
  old = {
    **{f"net.{k}": v for k, v in mlp(in_old, hidden, st.STATE_DIM + 3).state_dict().items()},
    **{f"in_norm.{k}": v for k, v in RunningNorm(in_old).state_dict().items()},
    **{f"out_norm.{k}": v for k, v in RunningNorm(st.STATE_DIM + 3).state_dict().items()},
  }
  old["in_norm.mean"] = torch.randn(in_old) * 0.1
  old["in_norm.var"] = torch.rand(in_old) + 0.5
  assert expand_world_model_scan(old, new)
  n = 32
  s, a, ap, xi = torch.randn(n, st.STATE_DIM), torch.randn(n, 12), torch.randn(n, 12), torch.randn(n, 4)
  x = torch.cat([s, a, ap, xi], -1)
  ref_net = mlp(in_old, hidden, st.STATE_DIM + 3)
  ref_net.load_state_dict({k[4:]: v for k, v in old.items() if k.startswith("net.")})
  x_n = ((x - old["in_norm.mean"]) / (old["in_norm.var"].sqrt() + 1e-6)).clamp(-10, 10)
  ref = ref_net(x_n)
  ref_s = s + ref[:, : st.STATE_DIM]  # out_norm is the identity here
  with torch.no_grad():
    for relief in (torch.zeros(n, tr.SCAN_POINTS), torch.randn(n, tr.SCAN_POINTS)):
      s1, _ = new(s, a, ap, xi, relief)
      assert torch.allclose(s1, ref_s, atol=1e-5)


def test_actor_height_scan_warm_start_is_exact():
  from go2_supermimic.supermimic.train import Actor, expand_actor_new_terms

  torch.manual_seed(0)
  h = 5
  old_terms = st.OBS_TERMS
  new_terms = st.OBS_TERMS + (("height_scan", tr.SCAN_POINTS),)
  d_old = h * sum(d for _, d in old_terms)
  old = Actor(d_old, (64, 64), 12)
  old.norm.mean.normal_()
  old.norm.var.uniform_(0.5, 2.0)
  new = Actor(d_old + h * tr.SCAN_POINTS, (64, 64), 12)
  expand_actor_new_terms(old.state_dict(), new, old_terms, new_terms, h)
  x = torch.randn(16, d_old)
  for scan in (torch.zeros(16, h * tr.SCAN_POINTS), torch.randn(16, h * tr.SCAN_POINTS)):
    with torch.no_grad():
      assert torch.allclose(new(torch.cat([x, scan], -1)), old(x), atol=1e-5)



def _step_field(high_beyond: bool):
  """Ground 0.1 m high for x > 1.5 (high_beyond) or for x < 1.5 (otherwise)."""
  xs = torch.arange(-6.0, 6.0001, 0.05)
  up = (xs > 1.5) if high_beyond else (xs < 1.5)
  h = (up.float() * 0.1).expand(len(xs), -1).clone()
  return tr.HeightField(h, (-6.0, -6.0), 0.05)


def test_terrain_bars_come_from_uphill_and_follow_the_ground():
  from go2_supermimic.target.generator import LOW_BAR, BallTarget, BallTargetCfg

  torch.manual_seed(0)
  n = 64
  cfg = BallTargetCfg(kind_probs=(0.0, 1.0, 0.0), bars_follow_ground=True)
  gen = BallTarget(cfg, n, "cpu")
  gen.set_progress(10**9)
  gen.spawn(torch.arange(n), torch.zeros(n, 2), torch.zeros(n), torch.zeros(n, 2),
            torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
  assert (gen.kind == LOW_BAR).all()
  robot, still, yaw = torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n)
  dt = torch.full((n,), 0.02)
  gen.field = _step_field(high_beyond=False)  # the robot stands higher than the throwers
  for _ in range(100):
    gen.step(dt, robot, still, yaw)
  assert not gen.flying.any(), "no bar is thrown uphill"
  gen.field = _step_field(high_beyond=True)  # throwers on higher ground
  launched = torch.zeros(n, dtype=torch.bool)
  for _ in range(100):
    gen.step(dt, robot, still, yaw)
    launched |= gen.flying
    f = gen.flying
    if f.any():
      ground = tr.ground(gen.dodge_pos[f, :2], gen.field)
      assert torch.allclose(gen.dodge_pos[f, 2], gen.carry_off[f, 2] + ground, atol=1e-5)
  assert launched.all()
