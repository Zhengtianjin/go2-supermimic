"""SuperMimic training for Go2 chase / dodge (oracle perception).

One loop, three phases per iteration (SuperMimic Alg. A.1):
  1. Collect: the deterministic actor plus Gaussian action noise steps the mjlab
     env; transitions go into a fixed-size replay buffer.
  2. Fit world model: k-step autoregressive windows from the buffer.
  3. Improve policy: BPTT of the imagined H-step return through the frozen
     world model, bootstrapped by a critic fit to TD(lambda) targets.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.buffer import ReplayBuffer, WindowArchive, pack_generator
from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination, RewardWeights
from go2_supermimic.supermimic.world_model import RunningNorm, WorldModel, mlp
from go2_supermimic.target import terrain as tr


@dataclass(kw_only=True)
class TrainCfg:
  task: str = "Go2-Chase-Terrain"
  num_envs: int = 4096
  buffer_len: int = 1000
  iterations: int = 40_000
  warmup_iterations: int = 300
  """World-model-only iterations before policy updates start."""
  collect_steps: int = 1
  action_noise: float = 0.1
  explore_frac: float = 0.25
  """Fraction of envs collecting with larger, temporally correlated (OU) action
  noise, so the world model sees more varied motion than the current policy."""
  explore_noise: float = 0.3
  """Stationary std of the exploration OU noise (normalized action units)."""
  explore_theta: float = 2.0
  """Mean reversion of the exploration OU noise [1/s]."""
  wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
  wm_window: int = 8
  wm_batch: int = 2048
  wm_updates: int = 2
  wm_prioritized_frac: float = 0.25
  """Fraction of each world-model batch drawn from current windows with contact
  changes (touchdown / liftoff), flight or falls, where the model is least accurate."""
  wm_archive_frac: float = 0.25
  """Fraction of each world-model batch drawn from the rare-event archive."""
  archive_size: int = 100_000
  """Windows kept in the rare-event archive (jumps, landings, falls)."""
  archive_keep_prob: float = 0.02
  """Fraction of the rare candidate windows stored, so the archive turns over
  slowly and keeps old rare events."""
  wm_rewarmup_iterations: int = 1500
  """When resuming into a world model of a different shape, fit it this long
  before the policy moves again."""
  lr: float = 2e-4
  actor_lr_scale: float = 1.0
  """The actor's learning rate relative to lr. Fine-tuning on terrain wants smaller
  actor steps: at the full rate the actor drifts into states the world model
  predicts badly (falls rise once the ramp-up completes)."""
  lr_final_frac: float = 0.03
  lr_decay_iterations: int = 30_000
  horizon: int = 32
  imag_batch: int = 8192
  actor_hidden: tuple[int, ...] = (512, 256, 128)
  critic_hidden: tuple[int, ...] = (512, 256, 128)
  critic_epochs: int = 5
  critic_minibatches: int = 4
  grad_clip: float = 1.0
  grad_clip_elem: float = 0.1
  compile: bool = True
  """torch.compile one imagined step (cuts Python kernel-launch overhead ~1.5x)."""
  dodge_shaping: float = 20.0
  """Initial weight of the jump/duck hint (training wheels for the bars)."""
  dodge_shaping_decay_iterations: int = 20_000
  """The hint weight decays linearly to 0 over this many iterations."""
  target_curriculum_iterations: int = 15_000
  """Iterations over which the targets reach full difficulty. The env-side
  curriculum counts env steps, and this loop takes one env step per iteration
  (PPO takes 24), so it is set here explicitly."""
  terrain_curriculum_iterations: int = 12_000
  """Iterations over which the harder terrain rows open up (terrain task)."""
  sensor_grad: bool = True
  """See ImagineCfg.sensor_grad."""
  driver_policy: str = ""
  """MoE-CTS TorchScript policy that drives a fraction of the envs while
  collecting, so the world model sees terrain from the start (supermimic/driver.py)."""
  driver_frac: float = 0.25
  """Fraction of envs (the last ones; exploration noise uses the first) it drives."""
  expert_policy: str = ""
  """One of our checkpoints (e.g. a flat-ground chase-and-dodge policy) that drives
  another fraction of the envs, so the world model sees its skills (jumps, ducks)
  in this task. Its observation must be a prefix of this task's."""
  expert_frac: float = 0.0
  driver_fade: tuple[int, int] = (1_000_000_000, 1_000_000_000)
  """Both driven fractions fade linearly to zero between these iterations."""
  driver_vmin: float = 0.5
  driver_vmax: float = 3.0
  """Per-env top speed of the driver's commands is drawn from [vmin, vmax] [m/s]."""
  critic_warmup_iterations: int = 500
  """After resuming with a freshly initialised critic, train only the critic for
  this many policy iterations before the actor moves again."""
  actor_ramp_iterations: int = 1000
  """Whenever the actor starts (or restarts after a rollback) moving, its learning
  rate ramps up linearly over this many iterations, so the world model sees the
  new behaviour before the actor can exploit its errors."""
  reset_critic: bool = False
  """With --resume: start the critic afresh (e.g. after changing reward weights) and
  keep the actor still until it has warmed up."""
  continue_run: bool = False
  """With --resume: carry on that run's iteration count, so the target and terrain
  curricula, the hint decay and the learning-rate schedule pick up where it stopped
  (--iterations is then the total). Without it a resume starts them afresh."""
  resume: str = ""
  """Checkpoint to warm-start actor, critic and world model from (and optimizers if saved)."""
  collapse_falls: float = 3.0
  """Collapse guard: if the logged fall count exceeds this twice in a row, reload
  the last checkpoint and halve the learning rate."""
  log_interval: int = 50
  save_interval: int = 1000
  seed: int = 0
  run_name: str = "supermimic"
  log_root: str = "logs/supermimic"


class Actor(nn.Module):
  def __init__(self, obs_dim: int, hidden: tuple[int, ...], act_dim: int):
    super().__init__()
    self.norm = RunningNorm(obs_dim)
    self.net = mlp(obs_dim, hidden, act_dim, act=nn.ELU)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(obs))


class Critic(nn.Module):
  def __init__(self, in_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.norm = RunningNorm(in_dim)
    self.net = mlp(in_dim, hidden, 1, act=nn.ELU)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(x))


def clip_grads(params, global_norm: float, elem: float) -> torch.Tensor:
  params = [p for p in params if p.grad is not None]
  norm = torch.nn.utils.clip_grad_norm_(params, global_norm)
  for p in params:
    p.grad.clamp_(-elem, elem)
  return norm


def lr_at(cfg: TrainCfg, it: int) -> float:
  import math

  t = min(1.0, it / cfg.lr_decay_iterations)
  frac = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * t))
  return cfg.lr * frac


@torch.no_grad()
def expand_actor_new_terms(old: dict, actor: Actor, old_terms, new_terms, history: int) -> None:
  """Warm-start an actor whose observation gained whole terms (e.g. a height scan),
  exactly: old terms keep their weights and normaliser statistics, new terms start
  with zero weight (their statistics are re-estimated as data comes in)."""
  sd = actor.state_dict()
  for k, v in old.items():
    if k in sd and sd[k].shape == v.shape:
      sd[k] = v.clone()
  w_old, mu_old, var_old = old["net.0.weight"], old["norm.mean"], old["norm.var"]
  w, mu, var = sd["net.0.weight"], sd["norm.mean"], sd["norm.var"]
  w.zero_()
  mu.zero_()
  var.fill_(1.0)
  old_off, o = {}, 0
  for name, dim in old_terms:
    old_off[name] = (o, dim * history)
    o += dim * history
  assert o == w_old.shape[1], (o, w_old.shape)
  n = 0
  for name, dim in new_terms:
    size = dim * history
    if name in old_off:
      o, osize = old_off[name]
      assert osize == size, name
      w[:, n : n + size] = w_old[:, o : o + size]
      mu[n : n + size], var[n : n + size] = mu_old[o : o + size], var_old[o : o + size]
    n += size
  assert n == w.shape[1]
  actor.load_state_dict(sd)


@torch.no_grad()
def expand_world_model_scan(old: dict, wm: WorldModel) -> bool:
  """Warm-start a world model with ground-scan inputs from one without, exactly:
  the scan inputs (appended last) start with zero weight. False if the old model
  differs in any other way."""
  sd = wm.state_dict()
  w_old, w_new = old.get("net.0.weight"), sd["net.0.weight"]
  if w_old is None or w_old.shape[0] != w_new.shape[0] or w_new.shape[1] - w_old.shape[1] != tr.SCAN_POINTS:
    return False
  for k, v in old.items():
    if k not in ("net.0.weight", "in_norm.mean", "in_norm.var") and sd[k].shape != v.shape:
      return False
  d = w_old.shape[1]
  for k, v in old.items():
    sd[k] = v.clone() if k not in ("net.0.weight", "in_norm.mean", "in_norm.var") else sd[k]
  sd["net.0.weight"].zero_()
  sd["net.0.weight"][:, :d] = w_old
  sd["in_norm.mean"][:d], sd["in_norm.var"][:d] = old["in_norm.mean"], old["in_norm.var"]
  sd["in_norm.mean"][d:], sd["in_norm.var"][d:] = 0.3, 0.04
  wm.load_state_dict(sd)
  return True


class Trainer:
  def __init__(self, cfg: TrainCfg):
    import go2_supermimic.env  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    self.cfg = cfg
    torch.manual_seed(cfg.seed)
    env_cfg = load_env_cfg(cfg.task)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    env_cfg.commands["target"].target.curriculum_steps = cfg.target_curriculum_iterations
    if "terrain_levels" in env_cfg.curriculum:
      env_cfg.curriculum["terrain_levels"].params["steps"] = cfg.terrain_curriculum_iterations
    self.env_cfg = env_cfg
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    self.dev = self.env.device
    self.term = self.env.command_manager.get_term("target")
    self.history = env_cfg.observations["actor"].history_length

    obs, _ = self.env.reset()
    self.obs = obs["actor"]
    obs_dim = self.obs.shape[-1]
    self.consts = st.robot_consts(self.env)
    self.ctx = st.ContextReader(self.env)
    self.xi = self.ctx.read()
    self.enc_bias = self.env.scene["robot"].data.encoder_bias.clone()

    self.wm = WorldModel(cfg.wm_hidden).to(self.dev)
    self.actor = Actor(obs_dim, cfg.actor_hidden, st.NUM_JOINTS).to(self.dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
    self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    self.buf = ReplayBuffer(cfg.num_envs, cfg.buffer_len, obs_dim, self.dev)
    self.archive = WindowArchive(cfg.archive_size, cfg.wm_window, st.CONTEXT_DIM, self.dev)
    self.explore = torch.arange(cfg.num_envs, device=self.dev) < int(cfg.explore_frac * cfg.num_envs)
    self.ou = torch.zeros(cfg.num_envs, st.NUM_JOINTS, device=self.dev)
    # Imagination uses the env's reward weights, so the two cannot drift apart.
    weights = RewardWeights(**{f.name: env_cfg.rewards[f.name].weight
                               for f in fields(RewardWeights) if f.name in env_cfg.rewards})
    imag_cfg = ImagineCfg(horizon=cfg.horizon, compile=cfg.compile, sensor_grad=cfg.sensor_grad,
                          weights=weights)
    self.field = self.term.height_field  # None on flat ground
    self.terms = st.obs_terms(env_cfg)
    self.imag = Imagination(imag_cfg, self.consts, self.term.cfg.target, self.history, self.dev,
                            self.terms, self.field)
    self.critic = Critic(self.imag.critic_dim, cfg.critic_hidden).to(self.dev)
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
    self.driver = self.expert = None
    if cfg.driver_policy:
      from go2_supermimic.supermimic.driver import ChaseSteer, MoeCtsDriver

      self.driver = MoeCtsDriver(self.env, cfg.driver_policy)
      self.steer = ChaseSteer(cfg.num_envs, self.dev, (cfg.driver_vmin, cfg.driver_vmax),
                              0.8 * self.term.cfg.target.arena_radius)
    if cfg.expert_policy:
      from go2_supermimic.supermimic.driver import ActorDriver

      self.expert = ActorDriver(cfg.expert_policy, obs_dim, self.dev)
    self.env_index = torch.arange(cfg.num_envs, device=self.dev)
    self.a_prev = torch.zeros(cfg.num_envs, st.NUM_JOINTS, device=self.dev)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    self.log_dir = Path(cfg.log_root) / f"{stamp}_{cfg.run_name}"
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(str(self.log_dir))
    self.ep_stats: dict[str, list[float]] = defaultdict(list)
    self.env_steps = 0
    self.actor_frozen_until = 0
    self.policy_start = cfg.warmup_iterations
    self.lr_scale = 1.0
    self.ramp_start = 0
    self.last_ckpt: Path | None = None
    self.collapse_strikes = 0
    self.nonfinite_steps = 0
    self.start_it = 0
    if cfg.resume:
      self.load(cfg.resume)

  def load(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.dev, weights_only=False)
    wm_sd = self.wm.state_dict()
    same_wm = all(
      k in ckpt["world_model"] and ckpt["world_model"][k].shape == v.shape for k, v in wm_sd.items()
    )
    if same_wm:
      self.wm.load_state_dict(ckpt["world_model"])
      if "opt_wm" in ckpt:
        self.opt_wm.load_state_dict(ckpt["opt_wm"])
    elif expand_world_model_scan(ckpt["world_model"], self.wm):
      print("world model: ground-scan inputs added (zero weight, same predictions)", flush=True)
    else:
      self.policy_start = max(self.policy_start, self.cfg.wm_rewarmup_iterations)
      self.actor_frozen_until = max(
        self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iterations
      )
      print(f"world model re-initialised (shape changed); policy updates from iteration "
            f"{self.policy_start}", flush=True)
    old_obs_dim = ckpt["actor"]["norm.mean"].shape[0]
    new_obs_dim = self.obs.shape[-1]
    if old_obs_dim == new_obs_dim:
      self.actor.load_state_dict(ckpt["actor"])
      if "opt_actor" in ckpt:
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
    elif "height_scan" in dict(self.terms) and new_obs_dim == old_obs_dim + self.history * dict(
      self.terms
    )["height_scan"]:
      old_terms = tuple(t for t in self.terms if t[0] != "height_scan")
      expand_actor_new_terms(ckpt["actor"], self.actor, old_terms, self.terms, self.history)
      print(f"actor expanded with a height scan from {old_obs_dim} to {new_obs_dim} inputs",
            flush=True)
    else:
      raise ValueError(f"actor inputs {old_obs_dim} -> {new_obs_dim}: no warm start for this")
    critic_sd = self.critic.state_dict()
    same_critic = all(
      k in ckpt["critic"] and ckpt["critic"][k].shape == v.shape for k, v in critic_sd.items()
    )
    if same_critic and not self.cfg.reset_critic:
      self.critic.load_state_dict(ckpt["critic"])
      if "opt_critic" in ckpt:
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
    else:
      self.actor_frozen_until = max(
        self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iterations
      )
      print(f"critic re-initialised; actor frozen until iteration {self.actor_frozen_until}",
            flush=True)
    print(f"resumed from {path} (iteration {ckpt.get('iteration')})", flush=True)
    if self.cfg.continue_run:
      # One env step per iteration: the env-side curricula count env steps.
      self.start_it = int(ckpt.get("iteration", 0))
      self.env.common_step_counter = self.start_it
      print(f"continuing at iteration {self.start_it}", flush=True)

  # Phase 1.

  @torch.no_grad()
  def collect(self) -> None:
    env, cfg = self.env, self.cfg
    for _ in range(cfg.collect_steps):
      s = st.robot_state(env, field=self.field)
      pose = st.planar_pose(env)
      gen = pack_generator(self.term.gen)
      relief = self._relief(pose)
      a = self.actor(self.obs)
      dt = env.step_dt
      # Driven envs are the last ones: MoE-CTS, then before them the expert.
      fade = self.driver_scale()
      n_moe = int(fade * cfg.driver_frac * cfg.num_envs) if self.driver is not None else 0
      n_exp = int(fade * cfg.expert_frac * cfg.num_envs) if self.expert is not None else 0
      moe = self.env_index >= cfg.num_envs - n_moe
      exp = (self.env_index >= cfg.num_envs - n_moe - n_exp) & ~moe
      if self.driver is not None:
        cmd = self.steer(dt, pose[:, :2], pose[:, 2], self.term.gen.center,
                         self.term.gen.chase_pos[:, :2], self.term.has_chase)
        a = torch.where(moe.unsqueeze(-1), self.driver.act(cmd), a)
      if n_exp:
        a = torch.where(exp.unsqueeze(-1), self.expert.act(self.obs), a)
      sigma_ou = cfg.explore_noise * (2 * cfg.explore_theta) ** 0.5
      self.ou = self.ou * (1 - cfg.explore_theta * dt) + sigma_ou * dt**0.5 * torch.randn_like(a)
      noise = torch.where(
        self.explore.unsqueeze(-1), self.ou, cfg.action_noise * torch.randn_like(a)
      )
      a = a + noise
      obs, _, term, trunc, extras = env.step(a)
      done = term | trunc
      if self.driver is not None:
        self.driver.reset(done)
        self.steer.resample(done, self.term.gen.center)
      self.ou = torch.where(done.unsqueeze(-1), torch.zeros_like(self.ou), self.ou)
      self.buf.add(s, pose, a, self.a_prev, self.obs, gen, done, relief)
      keep = ~done
      if keep.any():
        s1 = st.robot_state(env, field=self.field)
        dpose = st.pose_delta(pose, st.planar_pose(env))
        self.wm.update_stats(
          s[keep], a[keep], self.a_prev[keep], self.xi[keep], relief[keep], s1[keep], dpose[keep]
        )
      self.actor.norm.update(self.obs)
      # Fast turning (|yaw rate| > 3 rad/s): watches for circling / spinning.
      wz = env.scene["robot"].data.root_link_ang_vel_b[:, 2].abs()
      self.ep_stats["Metrics/fast_yaw"].append(float((wz > 3.0).float().mean()))
      self.a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)
      self.obs = obs["actor"]
      self.env_steps += env.num_envs
      for k, v in extras.get("log", {}).items():
        if k.startswith(("Episode_Reward", "Metrics/target", "Episode_Termination")):  # env logs
          self.ep_stats[k].append(float(v))

  def driver_scale(self) -> float:
    """1 before cfg.driver_fade[0], linearly down to 0 at cfg.driver_fade[1]."""
    start, end = self.cfg.driver_fade
    it = getattr(self, "it", 0)
    return min(1.0, max(0.0, (end - it) / max(1, end - start)))

  # Phase 2.

  def _relief(self, pose: torch.Tensor) -> torch.Tensor:
    return tr.relief(pose[:, :2], pose[:, 2], self.field, self.term.obstacle_layout)

  def _gather(self, env, idx, valid) -> dict[str, torch.Tensor]:
    b, k = self.buf, self.cfg.wm_window
    e = env.unsqueeze(-1)
    return {
      "s": b.s[e, idx], "pose": b.pose[e, idx], "a": b.a[e, idx[:, :k]],
      "a_prev": b.a_prev[e, idx[:, :k]], "xi": self.xi[env], "valid": valid,
      "relief": b.relief[e, idx[:, :k]],
    }

  @staticmethod
  def _rarity(w: dict[str, torch.Tensor]) -> torch.Tensor:
    """Contact changes + flight steps + an early episode end (a fall) per window."""
    s = w["s"]
    contact = st.feet_world_z(s.reshape(-1, st.STATE_DIM)) < 0.035
    contact = contact.reshape(s.shape[0], -1, 4).float()
    changes = (contact[:, 1:] - contact[:, :-1]).abs().sum((1, 2))
    airborne = (contact.sum(-1) == 0).float().sum(-1)
    ended = (w["valid"][:, -1] < 0.5).float() * 4.0
    return changes + airborne + ended

  @staticmethod
  def _archive_worthy(w: dict[str, torch.Tensor]) -> torch.Tensor:
    """Genuinely rare windows: a flight phase (all feet off for >= 2 steps), a
    hard vertical base motion (take-off / landing impact) or a fall."""
    s = w["s"]
    feet_z = st.feet_world_z(s.reshape(-1, st.STATE_DIM)).reshape(s.shape[0], -1, 4)
    flight = ((feet_z >= 0.035).all(-1).float().sum(-1)) >= 2
    hard_vz = s[:, :, st.V_YAW][..., 2].abs().amax(-1) > 0.6
    fell = w["valid"][:, -1] < 0.5
    return flight | hard_vz | fell

  def _wm_batch(self) -> dict[str, torch.Tensor]:
    """Uniform windows, rare windows from the buffer, and archived rare windows."""
    cfg, b = self.cfg, self.buf
    n_prio = int(cfg.wm_batch * cfg.wm_prioritized_frac)
    n_arch = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.size > 0 else 0
    n_uni = cfg.wm_batch - n_prio - n_arch
    parts = [self._gather(*b.sample_windows(n_uni, cfg.wm_window))]
    cand = self._gather(*b.sample_windows(4 * max(n_prio, 1), cfg.wm_window))
    score = self._rarity(cand)
    keep = self._archive_worthy(cand) & (torch.rand_like(score) < cfg.archive_keep_prob)
    self.archive.add({k: v[keep] for k, v in cand.items()})
    top = (score + 1e-3 * torch.rand_like(score)).topk(n_prio).indices
    parts.append({k: v[top] for k, v in cand.items()})
    if n_arch:
      parts.append(self.archive.sample(n_arch))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

  def fit_world_model(self) -> float:
    cfg = self.cfg
    k = cfg.wm_window
    losses = []
    for _ in range(cfg.wm_updates):
      w = self._wm_batch()
      pose_tgt = torch.stack(
        [st.pose_delta(w["pose"][:, j], w["pose"][:, j + 1]) for j in range(k)], 1
      )
      loss = self.wm.loss(
        w["s"][:, 0], w["a"], w["a_prev"], w["xi"], w["relief"].float(), w["s"][:, 1:],
        pose_tgt, w["valid"],
      )
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      gn = clip_grads(self.wm.parameters(), cfg.grad_clip, float("inf"))
      if bool(torch.isfinite(gn)):
        self.opt_wm.step()
      else:
        self.nonfinite_steps += 1
      losses.append(loss.item())
    return sum(losses) / len(losses)

  # Phase 3.

  def improve_policy(self) -> dict[str, float]:
    cfg, b = self.cfg, self.buf
    env, t = b.sample_states(cfg.imag_batch)
    seed = {
      "env": env, "s": b.s[env, t], "pose": b.pose[env, t], "a_prev": b.a_prev[env, t],
      "obs": b.obs[env, t].float(), "gen": b.gen[env, t], "xi": self.xi[env],
      "enc_bias": self.enc_bias[env],
    }
    for p in self.wm.parameters():
      p.requires_grad_(False)
    hint = cfg.dodge_shaping * max(0.0, 1.0 - self.it / max(1, cfg.dodge_shaping_decay_iterations))
    objective, data = self.imag.rollout(self.wm, self.actor, self.critic, seed, hint)
    loss = -objective.mean()
    self.opt_actor.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = clip_grads(self.actor.parameters(), cfg.grad_clip, cfg.grad_clip_elem)
    finite = bool(torch.isfinite(gnorm)) and bool(torch.isfinite(loss))
    if not finite:
      self.nonfinite_steps += 1
    if self.it > self.actor_frozen_until and finite:
      self.opt_actor.step()
    for p in self.wm.parameters():
      p.requires_grad_(True)

    # Critic on the imagined transitions.
    targets = self.imag.gae_targets(self.critic, data)
    x = data["critic_in"][:, :-1].detach().reshape(-1, data["critic_in"].shape[-1])
    y = targets.reshape(-1)
    self.critic.norm.update(x)
    n = x.shape[0]
    mb = n // cfg.critic_minibatches
    c_losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.dev)
      for i in range(cfg.critic_minibatches):
        sel = perm[i * mb : (i + 1) * mb]
        c_loss = (self.critic(x[sel]).squeeze(-1) - y[sel]).square().mean()
        self.opt_critic.zero_grad(set_to_none=True)
        c_loss.backward()
        gn = clip_grads(self.critic.parameters(), cfg.grad_clip, float("inf"))
        if bool(torch.isfinite(gn)):
          self.opt_critic.step()
        c_losses.append(c_loss.item())
    return {
      "imagined_return": objective.mean().item(),
      "imagined_reward": data["rewards"].mean().item(),
      "imagined_survival": data["surv"][:, -1].mean().item(),
      "actor_grad_norm": float(gnorm),
      "critic_loss": sum(c_losses) / len(c_losses),
    }

  def save(self, it: int) -> Path:
    path = self.log_dir / f"model_{it}.pt"
    torch.save(
      {
        "supermimic": True,
        "iteration": it,
        "cfg": asdict(self.cfg),
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict(),
        "world_model": self.wm.state_dict(),
        "opt_wm": self.opt_wm.state_dict(),
        "opt_actor": self.opt_actor.state_dict(),
        "opt_critic": self.opt_critic.state_dict(),
        "obs_dim": self.obs.shape[-1],
      },
      path,
    )
    return path

  def run(self) -> None:
    cfg = self.cfg
    t_start = time.time()
    while self.buf.size <= cfg.wm_window + 1:
      self.collect()
    for it in range(self.start_it + 1, cfg.iterations + 1):
      self.it = it
      lr = lr_at(cfg, it) * self.lr_scale
      start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
      ramp = min(1.0, max(0.0, it - start) / max(1, cfg.actor_ramp_iterations))
      actor_scale = ramp * cfg.actor_lr_scale
      for opt, scale in ((self.opt_wm, 1.0), (self.opt_actor, actor_scale), (self.opt_critic, 1.0)):
        for g in opt.param_groups:
          g["lr"] = lr * scale
      self.collect()
      wm_loss = self.fit_world_model()
      stats = {"wm_loss": wm_loss}
      if it > self.policy_start:
        stats.update(self.improve_policy())
      if it % cfg.log_interval == 0:
        elapsed = time.time() - t_start
        for k, v in stats.items():
          self.writer.add_scalar(f"train/{k}", v, it)
        line = [f"it {it}/{cfg.iterations}", f"{elapsed / 60:.1f} min",
                f"env_steps {self.env_steps / 1e6:.1f}M"]
        line += [f"{k} {v:.4g}" for k, v in stats.items()]
        line.append(f"archive {self.archive.size}")
        if self.nonfinite_steps:
          line.append(f"nonfinite_skipped {self.nonfinite_steps}")
        rates = self.term.kind_hit_rate.tolist()
        for name, r in zip(("ball", "low_bar", "high_bar"), rates):
          self.writer.add_scalar(f"Metrics/hit_rate/{name}", r, it)
        line.append("hit_rate " + "/".join(f"{r:.2f}" for r in rates))
        falls = self.ep_stats.get("Episode_Termination/fell_over", [])
        falls_mean = sum(falls) / len(falls) if falls else 0.0
        for k, vals in self.ep_stats.items():
          mean = sum(vals) / len(vals)
          self.writer.add_scalar(k, mean, it)
          if k in ("Metrics/target/tags", "Metrics/target/hits", "Metrics/target/dist",
                   "Metrics/target/obstacle_contact", "Episode_Termination/fell_over",
                   "Metrics/fast_yaw"):
            line.append(f"{k.split('/')[-1]} {mean:.3g}")
        self.ep_stats.clear()
        self.writer.add_scalar("time/minutes", elapsed / 60, it)
        print(" | ".join(line), flush=True)
        self._collapse_guard(falls_mean)
      if it % cfg.save_interval == 0 or it == cfg.iterations:
        if self.collapse_strikes == 0:
          self.last_ckpt = self.save(it)
          print(f"saved {self.last_ckpt}", flush=True)

  def _collapse_guard(self, falls: float) -> None:
    """Model exploitation shows up as many real falls: roll back and slow down."""
    if self.it <= self.actor_frozen_until or falls <= self.cfg.collapse_falls:
      self.collapse_strikes = 0
      return
    self.collapse_strikes += 1
    if self.collapse_strikes >= 2 and self.last_ckpt is not None:
      # The world model is kept: it has just learned from the real falls the actor
      # exploited, and reloading it would let the same exploit happen again.
      ckpt = torch.load(self.last_ckpt, map_location=self.dev, weights_only=False)
      self.actor.load_state_dict(ckpt["actor"])
      self.critic.load_state_dict(ckpt["critic"])
      self.lr_scale *= 0.5
      self.collapse_strikes = 0
      self.ramp_start = self.it
      # Fresh optimizer moments: the old ones carry the step that diverged.
      self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95))
      self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
      print(f"collapse guard: {falls:.1f} falls, rolled back to {self.last_ckpt}, "
            f"lr scale {self.lr_scale:.3g}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  defaults = TrainCfg()
  for name, val in asdict(defaults).items():
    if isinstance(val, tuple):
      ap.add_argument(f"--{name.replace('_', '-')}", type=int, nargs="+", default=list(val))
    elif isinstance(val, bool):  # type=bool would read "false" as True
      ap.add_argument(f"--{name.replace('_', '-')}", type=lambda x: x.lower() in ("1", "true"),
                      default=val)
    else:
      ap.add_argument(f"--{name.replace('_', '-')}", type=type(val), default=val)
  args = vars(ap.parse_args())
  cfg = TrainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()})
  Trainer(cfg).run()


if __name__ == "__main__":
  main()
