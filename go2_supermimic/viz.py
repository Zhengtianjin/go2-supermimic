"""Evaluation videos: renderer look, the status card, and for the obstacle task a
360 deg depth panorama at LiDAR height with the policy's LiDAR beams.

DepthPanorama renders four 90 deg MuJoCo depth views around the robot (its own
body hidden), resampled onto a cylinder (azimuth x elevation) and converted from
z-depth to range.
"""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

LIDAR_Z = 0.30  # height of the depth panorama's eye (the LiDAR plane) [m]
BG = (15, 18, 24)
FG = (235, 238, 242)
MUTED = (150, 158, 170)
GREEN = (110, 220, 120)
RED = (240, 95, 75)
YELLOW = (255, 210, 60)
KIND_NAMES = ("ball", "LOW BAR - jump", "HIGH BAR - duck")
BORDER_RGBA = (0.2, 0.2, 0.2, 1.0)  # mjlab's colour for the terrain border


def polish_renderer(env) -> None:
  """Render-only settings for videos (the simulation model is untouched):
  anti-aliasing, a sharper shadow map, a low sun so step faces are shaded, a
  dimmer headlight, distance fog, and the terrain border in the ground colour."""
  from go2_supermimic.env.terrains import GROUND_RGBA

  env.render()  # creates the offscreen renderer
  r = env._offline_renderer
  r.close()
  m = r._model
  m.vis.quality.offsamples = 8
  m.vis.quality.shadowsize = 8192
  m.vis.headlight.ambient[:] = (0.22, 0.22, 0.24)
  m.vis.headlight.diffuse[:] = (0.2, 0.2, 0.2)
  m.vis.headlight.specular[:] = (0.0, 0.0, 0.0)
  m.vis.map.fogstart, m.vis.map.fogend = 2.0, 6.0
  m.vis.rgba.fog[:] = (0.80, 0.86, 0.92, 1.0)
  sun = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "robot/sun")
  if sun >= 0:
    m.light_dir[sun] = (-0.6, -0.4, -0.6)  # low, so step faces are shaded
    m.light_diffuse[sun] = (0.75, 0.75, 0.71)
  border = np.all(np.isclose(m.geom_rgba, BORDER_RGBA), axis=1)
  m.geom_rgba[border] = GROUND_RGBA
  r.initialize()
  r.renderer._scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  try:
    return ImageFont.truetype(name, size)
  except OSError:
    return ImageFont.load_default(size=size)


def status_lines(term, i: int = 0) -> list[tuple[str, tuple[int, int, int]]]:
  """Card rows (text, colour) for env i of a TargetCommand."""
  chase, dodge = bool(term.has_chase[i]), bool(term.has_dodge[i])
  base = term.base_local()[i]
  rows = [("Chase + dodge" if chase and dodge else "Chase" if chase else "Dodge", FG)]
  rows.append((f"Robot     {term.robot.data.root_link_lin_vel_w[i, :2].norm().item():4.2f} m/s",
               YELLOW))
  if chase:
    d = (term.gen.chase_pos[i, :2] - base[:2]).norm().item()
    esc = "  breaking free" if term.gen.escape[i].item() > 0 else ""
    rows.append((f"Target    {term.gen.chase_vel[i].norm().item():4.2f} m/s   {d:4.2f} m away{esc}",
                 GREEN))
  if dodge:
    kind = KIND_NAMES[int(term.gen.kind[i].item())]
    speed = (f"{term.gen.dodge_vel[i].norm().item():4.2f} m/s" if bool(term.gen.flying[i])
             else "carried ")
    rows.append((f"Thrown    {speed}   {kind}", RED))
  rows.append((f"Base height  {base[2].item():4.2f} m", MUTED))
  return rows


def status_card(frame: np.ndarray, rows: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
  """The frame with a rounded status card (title row, then coloured rows)."""
  scene = Image.fromarray(frame).convert("RGBA")
  card = Image.new("RGBA", scene.size, (0, 0, 0, 0))
  d = ImageDraw.Draw(card)
  f_title, f_row = font(22, bold=True), font(20)
  pad, lh = 16, 28
  width = max(d.textlength(t, font=f_row) for t, _ in rows[1:]) + 2 * pad
  width = max(width, d.textlength(rows[0][0], font=f_title) + 2 * pad)
  d.rounded_rectangle([16, 16, 16 + width, 16 + 2 * pad + lh * len(rows)], radius=12,
                      fill=(12, 14, 18, 170))
  for j, (t, col) in enumerate(rows):
    d.text((16 + pad, 16 + pad + j * lh), t, fill=col, font=f_title if j == 0 else f_row)
  return np.asarray(Image.alpha_composite(scene, card).convert("RGB"))


class DepthPanorama:
  def __init__(self, model: mujoco.MjModel, width: int = 720, height: int = 120,
               elevation_deg: tuple[float, float] = (-20.0, 30.0), view_px: int = 256,
               far: float = 8.0):
    self.far = far
    self.model = copy.copy(model)
    self.model.vis.global_.fovy = 90.0
    g = self.model.vis.global_
    g.offwidth, g.offheight = max(g.offwidth, view_px), max(g.offheight, view_px)
    ext = self.model.stat.extent
    self.model.vis.map.znear = 0.02 / ext
    self.model.vis.map.zfar = far / ext
    self.renderer = mujoco.Renderer(self.model, view_px, view_px)
    self.renderer.enable_depth_rendering()
    self.opt = mujoco.MjvOption()
    self.opt.geomgroup[:] = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # hide the robot
    self.opt.sitegroup[:] = 0
    # Output pixel -> (view, source pixel, z-depth -> range factor).
    self.width, self.height = width, height
    az = math.pi - (np.arange(width) + 0.5) * (2 * math.pi / width)  # left edge = rear via left
    el = np.radians(elevation_deg[1]) - (np.arange(height) + 0.5) * (
      np.radians(elevation_deg[1] - elevation_deg[0]) / height
    )
    self.az, self.el = az, el
    self.el_range = np.radians(elevation_deg)
    view = np.round(az / (math.pi / 2)).astype(int) % 4  # view k looks along k * 90 deg
    local = az - view * (math.pi / 2)
    local = (local + math.pi) % (2 * math.pi) - math.pi
    tx = np.tan(local)[None, :]
    ty = np.tan(el)[:, None] / np.cos(local)[None, :]
    f = c = view_px / 2
    self.u = np.clip(np.round(c - f * tx), 0, view_px - 1).astype(int).repeat(height, 0)
    self.v = np.clip(np.round(c - f * ty), 0, view_px - 1).astype(int)
    self.view = np.broadcast_to(view[None, :], (height, width))
    self.factor = np.sqrt(1 + tx**2 + ty**2)

  def render(self, data: mujoco.MjData, base_xy: np.ndarray, yaw: float,
             eye_z: float = LIDAR_Z) -> np.ndarray:
    """Range panorama (height, width) [m] from (base_xy, eye_z); `far` where nothing is hit."""
    views = []
    for k in range(4):
      a = yaw + k * math.pi / 2
      cam = mujoco.MjvCamera()
      cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      eye = np.array([base_xy[0], base_xy[1], eye_z])
      cam.lookat[:] = eye + 0.01 * np.array([math.cos(a), math.sin(a), 0.0])
      cam.distance = 0.01
      cam.azimuth = math.degrees(a)
      cam.elevation = 0.0
      self.renderer.update_scene(data, camera=cam, scene_option=self.opt)
      views.append(self.renderer.render())
    depth = np.stack(views)[self.view, self.v, self.u] * self.factor
    return np.minimum(depth, self.far)

  def pixel(self, rel_world: np.ndarray, yaw: float) -> tuple[float, float] | None:
    """Panorama pixel of a world-frame offset from the eye, or None if outside."""
    dx, dy, dz = rel_world
    az = math.atan2(dy, dx) - yaw
    az = (az + math.pi) % (2 * math.pi) - math.pi
    el = math.atan2(dz, math.hypot(dx, dy))
    if not (self.el_range[0] <= el <= self.el_range[1]):
      return None
    x = (math.pi - az) / (2 * math.pi) * self.width
    y = (self.el_range[1] - el) / (self.el_range[1] - self.el_range[0]) * self.height
    return x, y


def _colorize(rng: np.ndarray, far: float) -> np.ndarray:
  cmap = colormaps["turbo_r"]
  rgb = (cmap(np.clip(rng / far, 0, 1))[..., :3] * 255).astype(np.uint8)
  rgb[rng >= far * 0.999] = BG
  return rgb


def lidar_frame(frame: np.ndarray, term, pano: DepthPanorama, data) -> np.ndarray:
  """Obstacle task: scene view with the status card over the depth panel (policy
  LiDAR beams, the chase ball and a thrown object marked)."""
  rd = term.robot.data
  base = rd.root_link_pos_w[0].cpu().numpy()
  yaw = rd.heading_w[0].item()
  rng = pano.render(data, base[:2], yaw)
  gen = term.gen
  markers = []
  if bool(term.has_chase[0]):
    markers.append((gen.chase_pos[0].cpu().numpy(), GREEN, True, "target", False))
  if bool(term.has_dodge[0]) and bool(gen.flying[0]):
    markers.append((gen.dodge_pos[0].cpu().numpy(), RED, True, "thrown", True))
  ocfg = term.cfg.target.obstacles
  return compose_panel(frame, pano, rng, status_lines(term), base, yaw, markers,
                       beams=term.lidar()[0].cpu().numpy(), beam_range=ocfg.lidar_range,
                       legend="\u25cf policy LiDAR beams (16)    \u25b2 target / thrown")


def compose_panel(frame, pano: DepthPanorama, rng: np.ndarray, status, base: np.ndarray,
                  yaw: float, markers, beams=None, beam_range: float = 4.0,
                  hfov_deg: float | None = None, legend: str = "",
                  title: str = "Depth panorama at LiDAR height (360\u00b0)",
                  eye_z: float = LIDAR_Z) -> np.ndarray:
  """Scene view with a status card, over a depth-panorama panel.

  markers: (world position (3,), colour, filled, label, label above) each.
  """
  h, w = frame.shape[:2]
  scene = Image.fromarray(status_card(frame, status))

  panel_h = 300
  panel = Image.new("RGB", (w, panel_h), BG)
  d = ImageDraw.Draw(panel, "RGBA")
  left, top, bar_w = 40, 58, 18
  pw = w - left - 120
  ph = int(pw / pano.width * pano.height)
  sx, sy = pw / pano.width, ph / pano.height
  img = Image.fromarray(_colorize(rng, pano.far)).resize((pw, ph), Image.BILINEAR)
  panel.paste(img, (left, top))
  if hfov_deg is not None and hfov_deg < 360:
    hfov = math.radians(hfov_deg) / 2
    x_l = left + (math.pi - hfov) / (2 * math.pi) * pw
    x_r = left + (math.pi + hfov) / (2 * math.pi) * pw
    d.rectangle([left, top, x_l, top + ph], fill=(0, 0, 0, 110))
    d.rectangle([x_r, top, left + pw, top + ph], fill=(0, 0, 0, 110))
    for x in (x_l, x_r):
      for y in range(top, top + ph, 10):
        d.line([x, y, x, min(y + 5, top + ph)], fill=(255, 255, 255, 160), width=1)
  y_h = top + (pano.el_range[1]) / (pano.el_range[1] - pano.el_range[0]) * ph
  d.line([left, y_h, left + pw, y_h], fill=(255, 255, 255, 70), width=1)
  cmap = colormaps["turbo_r"]
  if beams is not None:
    for j, r in enumerate(beams):
      az = 2 * math.pi * j / len(beams)
      az = (az + math.pi) % (2 * math.pi) - math.pi
      x = left + (math.pi - az) / (2 * math.pi) * pw
      hit = r < beam_range - 1e-3
      col = tuple(int(255 * c) for c in cmap(min(r / pano.far, 1.0))[:3]) if hit else BG
      d.ellipse([x - 6, y_h - 6, x + 6, y_h + 6], fill=col + (255,), outline=FG + (255,), width=2)
  eye = np.array([base[0], base[1], eye_z])
  f_tag = font(14)
  for pos, color, filled, label, above in markers:
    p = pano.pixel(pos - eye, yaw)
    if p is None:
      continue
    x, y = left + p[0] * sx, top + p[1] * sy
    tri = [(x, y - 11), (x - 9, y + 7), (x + 9, y + 7)]
    if filled:
      d.polygon(tri, fill=color + (255,), outline=(0, 0, 0, 255))
    else:
      d.polygon(tri, outline=color + (255,), width=3)
    tw = d.textlength(label, font=f_tag)
    ty = y - 36 if above else y + 14
    tx = min(max(x - tw / 2, left), left + pw - tw)
    d.rounded_rectangle([tx - 5, ty - 2, tx + tw + 5, ty + 18], radius=5, fill=(10, 12, 16, 210))
    d.text((tx, ty), label, fill=color + (255,), font=f_tag)
  f_ax, f_lab = font(15), font(17, bold=True)
  for az, name in ((math.pi, "rear"), (math.pi / 2, "left"), (0.0, "front"),
                   (-math.pi / 2, "right"), (-math.pi, "rear")):
    x = left + (math.pi - az) / (2 * math.pi) * pw
    d.line([x, top + ph, x, top + ph + 6], fill=MUTED + (255,), width=1)
    tw = d.textlength(name, font=f_ax)
    d.text((x - tw / 2, top + ph + 9), name, fill=MUTED + (255,), font=f_ax)
  d.text((left, 14), title, fill=FG + (255,), font=f_lab)
  d.text((left + 400, 17), legend, fill=MUTED + (255,), font=font(15))
  bx = left + pw + 30
  grad = (cmap(np.linspace(0, 1, ph))[:, :3] * 255).astype(np.uint8)
  panel.paste(Image.fromarray(np.repeat(grad[:, None, :], bar_w, 1)), (bx, top))
  for m in range(0, int(pano.far) + 1, 2):
    y = top + m / pano.far * ph
    d.line([bx + bar_w, y, bx + bar_w + 4, y], fill=MUTED + (255,))
    d.text((bx + bar_w + 7, y - 8), f"{m} m", fill=MUTED + (255,), font=f_ax)
  out = np.concatenate([np.asarray(scene), np.asarray(panel)], 0)
  return out[: out.shape[0] // 2 * 2]
