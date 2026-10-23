"""Evaluate a checkpoint, and optionally record env 0 with a status card.

  python scripts/eval.py Go2-Chase-Terrain --checkpoint <model.pt> --num-envs 256 --seconds 60
  python scripts/eval.py Go2-Chase-Terrain --checkpoint <model.pt> --mode both \
      --num-envs 1 --seconds 30 --video chase_dodge.mp4

Reports tags per minute, time near the chase ball, per-kind dodge hit rates with
the body height, highest foot and sideways speed while an obstacle is close,
time touching static obstacles, falls per minute and mean speed. Heights are above the ground under the base.
By default each task's own modes are used (terrain: chase only). In play mode an
episode never ends, so a video keeps the mode it starts with: use --mode both to
see chasing and dodging together.
"""

import argparse

import torch

import go2_supermimic.env  # noqa: F401
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.viz import DepthPanorama, lidar_frame, polish_renderer, status_card, status_lines
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

CATEGORY_NAMES = ("HIGH BALL", "MID BALL", "LOW BALL", "LOW BAR (jump)", "HIGH BAR (duck)")
MODE_PROBS = {"chase": (1.0, 0.0, 0.0), "dodge": (0.0, 1.0, 0.0), "both": (0.0, 0.0, 1.0),
              "mixed": (0.3, 0.3, 0.4)}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--mode", choices=list(MODE_PROBS), default=None)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--video", default=None, help="Write an mp4 of env 0 here.")
  ap.add_argument("--difficulty", type=float, default=1.0,
                  help="Curriculum progress in [0, 1] for the targets and terrain rows (1: hardest).")
  ap.add_argument("--terrain", default=None,
                  help="Put env 0 on this terrain type (e.g. stairs, boxes; see env/terrains.py).")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  args = ap.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  target_cfg = env_cfg.commands["target"]
  if args.mode:
    target_cfg.mode_probs = MODE_PROBS[args.mode]
  # Curricula run on env steps: put the step counter at the requested progress.
  steps = 10_000
  target_cfg.target.curriculum_steps = steps
  if "terrain_levels" in env_cfg.curriculum:
    env_cfg.curriculum["terrain_levels"].params["steps"] = steps
  if args.video:
    env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
    env_cfg.viewer.distance = 4.0
    # Look down more steeply among obstacles, so they do not hide the robot.
    env_cfg.viewer.elevation = -40.0 if target_cfg.target.obstacles.enabled else -20.0
    if target_cfg.target.obstacles.enabled:  # always some obstacles on camera
      target_cfg.target.obstacles.empty_prob = 0.0
    if target_cfg.terrain:  # closer and lower, so steps and slopes read
      env_cfg.viewer.distance, env_cfg.viewer.elevation = 3.2, -16.0
    for sensor in env_cfg.scene.sensors or ():
      if hasattr(sensor, "debug_vis"):
        sensor.debug_vis = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0",
                          render_mode="rgb_array" if args.video else None)
  ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
  if args.terrain:
    names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
    env.scene.terrain.terrain_types[0] = names.index(args.terrain)
  env.common_step_counter = int(args.difficulty * steps)
  obs, _ = env.reset()
  pano = None
  if args.video:
    polish_renderer(env)
    if target_cfg.target.obstacles.enabled:  # LiDAR view under the scene
      pano = DepthPanorama(env._offline_renderer._model)
  actor = Actor(obs["actor"].shape[-1], tuple(ckpt["cfg"]["actor_hidden"]), 12).to("cuda:0")
  actor.load_state_dict(ckpt["actor"])
  actor.eval()

  term = env.command_manager.get_term("target")
  n, dev = args.num_envs, env.device
  steps = int(args.seconds / env.step_dt)
  near, falls, speed, tags, contact = (torch.zeros(n, device=dev) for _ in range(5))
  # Per category (high / mid / low ball, low bar, high bar): throws, hits, and
  # body height, highest foot and sideways speed while one is within 0.8 m.
  nc = len(CATEGORY_NAMES)
  throws, hits = torch.zeros(nc, device=dev), torch.zeros(nc, device=dev)
  z_sum, foot_sum, side_sum, cnt = (torch.zeros(nc, device=dev) for _ in range(4))
  foot_ids, _ = term.robot.find_sites(("FR", "FL", "RR", "RL"), preserve_order=True)
  gen = term.gen

  def category():
    return torch.where(gen.kind == 0, gen.style.long(), gen.kind.long() + 2)

  prev_flying, prev_cat = gen.flying.clone(), category()
  frames = []
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, terminated, _, _ = env.step(actor(obs["actor"]))
      data = term.robot.data
      pos, vel = data.root_link_pos_w, data.root_link_lin_vel_w[:, :2]
      near += ((term.head_ball_dist() < term.cfg.near_radius) & term.has_chase).float()
      tags += gen.tagged.float()
      speed += vel.norm(dim=-1)
      falls += terminated.float()
      contact += (term.obstacle_penetration() > 0.05).float()

      g = term.ground_under_base()
      base_z = pos[:, 2] - g
      foot_z = data.site_pos_w[:, foot_ids, 2].max(-1).values - g
      d = gen.dodge_vel[:, :2] / gen.dodge_vel[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
      side = (vel * torch.stack([-d[:, 1], d[:, 0]], -1)).sum(-1).abs()
      close = term.has_dodge & gen.flying & ((gen.dodge_pos[:, :2] - pos[:, :2]).norm(dim=-1) < 0.8)
      launched = gen.flying & ~prev_flying & term.has_dodge
      cat = category()
      for k in range(nc):
        throws[k] += (launched & (cat == k)).sum()
        hits[k] += ((term.hit > 0) & (prev_cat == k)).sum()
        m = close & (cat == k)
        z_sum[k] += base_z[m].sum()
        foot_sum[k] += foot_z[m].sum()
        side_sum[k] += side[m].sum()
        cnt[k] += m.sum()
      prev_flying, prev_cat = gen.flying.clone(), cat
      if args.video:
        rgb = env.render()
        frames.append(lidar_frame(rgb, term, pano, env._offline_renderer._data) if pano
                      else status_card(rgb, status_lines(term)))

  minutes = args.seconds / 60.0
  chase = term.has_chase
  print(f"{args.task}  modes={target_cfg.mode_probs}  envs={n}  seconds={args.seconds}")
  if chase.any():
    print(f"  chase  tags per minute     {(tags[chase].mean() / minutes).item():.1f}")
    print(f"  chase  time within {term.cfg.near_radius} m  {(near[chase].mean() / steps).item():.2f}")
  if term.has_dodge.any():
    for k, name in enumerate(CATEGORY_NAMES):
      t, h, c = int(throws[k]), int(hits[k]), max(cnt[k].item(), 1)
      print(f"  dodge  {name:15s} throws {t:5d}  hit rate {h / max(t, 1):.2f}  "
            f"base {z_sum[k].item() / c:.2f} m  top foot {foot_sum[k].item() / c:.2f} m  "
            f"sideways {side_sum[k].item() / c:.2f} m/s")
  if target_cfg.target.obstacles.enabled:
    print(f"  obstacle contact (time)    {(contact.mean() / steps).item():.3f}")
  print(f"  falls per minute           {(falls.mean() / minutes).item():.3f}")
  print(f"  mean speed                 {(speed / steps).mean().item():.2f} m/s")
  if args.video:
    import mediapy

    mediapy.write_video(args.video, frames, fps=int(round(1.0 / env.step_dt)))
    print(f"  video: {args.video}")
  env.close()


if __name__ == "__main__":
  main()
