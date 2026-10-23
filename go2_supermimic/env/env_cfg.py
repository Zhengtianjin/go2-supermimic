"""Go2 chase / dodge environment: flat ground, static obstacles or rough terrain."""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from go2_supermimic.env import mdp
from go2_supermimic.env.command import TargetCommandCfg
from go2_supermimic.env.terrains import clean_look, rough_terrain
from go2_supermimic.robot.go2 import FOOT_GEOM_NAMES, GO2_ACTION_SCALE, get_go2_robot_cfg
from go2_supermimic.target import obstacles as ob

CMD = "target"
ACTOR_HISTORY = 5
FOOT_SITE_NAMES = ("FR", "FL", "RR", "RL")


def _quiet_solver_warnings(env, env_ids) -> None:
  """Startup event: silence per-step solver iteration-limit warnings (with
  thousands of worlds one hits the limit nearly every step)."""
  del env_ids
  from mujoco_warp import OverflowType

  opt = env.sim.wp_model.opt
  opt.warn_overflow = int(opt.warn_overflow) & ~int(
    OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS
  )
  # The flag is baked into the solver kernels when the CUDA graphs are captured.
  env.sim.create_graph()


def _obstacle_entity(half: tuple[float, float]) -> EntityCfg:
  """A tall static box, moved per env through its mocap body."""

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="block", mocap=True)
    body.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half[0], half[1], ob.OBSTACLE_HEIGHT / 2], rgba=[0.62, 0.5, 0.36, 1.0],
                  contype=1, conaffinity=1)
    return spec

  return EntityCfg(spec_fn=spec_fn, init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)))


def _configure_go2(cfg: ManagerBasedRlEnvCfg) -> None:
  """Go2 entity, sim settings, foot contact sensing and domain randomisation."""
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
  cfg.scene.sensors += (ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  ),)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = GO2_ACTION_SCALE

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Per-axis foot friction for condim 6, as in mjlab's Go1 config.
  del cfg.events["foot_friction"]
  for name, axis, dist, rng in (
    ("foot_friction_slide", 0, "uniform", (0.3, 1.5)),
    ("foot_friction_spin", 1, "log_uniform", (1e-4, 2e-2)),
    ("foot_friction_roll", 2, "log_uniform", (1e-5, 5e-3)),
  ):
    cfg.events[name] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "distribution": dist,
        "axes": [axis],
        "ranges": rng,
        "shared_random": True,
      },
    )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
  cfg.events["quiet_solver_warnings"] = EventTermCfg(
    mode="startup", func=_quiet_solver_warnings, params={}
  )
  # A deterministic world model cannot represent velocity kicks (SuperMimic A).
  cfg.events.pop("push_robot", None)

  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "fell_over": TerminationTermCfg(func=vel_mdp.bad_orientation,
                                    params={"limit_angle": math.radians(70.0)}),
  }


def go2_chase_env_cfg(play: bool = False, terrain: bool = False,
                      obstacles: bool = False) -> ManagerBasedRlEnvCfg:
  """terrain: generated rough terrain, a ground height scan for the actor, and
  chasing with thrown bars only (no balls).
  obstacles: static boxes, pillars and walls, and a planar LiDAR for the actor."""
  cfg = make_velocity_env_cfg()
  _configure_go2(cfg)
  if terrain:
    cfg.scene.terrain = rough_terrain()
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    clean_look(cfg.scene.terrain)
  target_cmd = TargetCommandCfg(resampling_time_range=(2.0, 5.0), debug_vis=True,
                                terrain=terrain)
  if terrain:
    # Chase only, or chase while jumping / ducking thrown bars (no balls).
    target_cmd.mode_probs = (0.6, 0.0, 0.4)
    target_cmd.target.kind_probs = (0.0, 0.5, 0.5)
    target_cmd.target.bar_speed = (1.2, 2.0)
    target_cmd.target.bar_speed_final = (2.0, 3.0)
    target_cmd.target.bars_follow_ground = True
    # The robot is a little slower on rough ground, so the chase ball is too.
    target_cmd.target.flee_ratio = (0.7, 0.85)
    target_cmd.target.flee_ratio_final = (0.8, 1.0)
    target_cmd.target.speed_max_final = (1.8, 2.6)
    target_cmd.target.escape_speed = (2.5, 3.5)
    target_cmd.target.ou_sigma = 3.5
  if obstacles:
    target_cmd.target.obstacles = ob.ObstacleCfg(enabled=True)
    # Running among obstacles the robot is slower, so throws are slower too.
    target_cmd.target.throw_speed = (2.5, 3.5)
    target_cmd.target.throw_speed_final = (3.0, 4.5)
    target_cmd.target.max_time_to_robot = 0.75
    for k, half in enumerate(ob.OBSTACLE_HALF):
      cfg.scene.entities[f"obstacle{k}"] = _obstacle_entity(half)
  cfg.commands = {CMD: target_cmd}

  # Observations. No linear velocity (not measured on hardware); a short
  # history lets the actor infer its own motion.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=vel_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=vel_mdp.joint_pos_rel, params={"biased": True}, noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=vel_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=vel_mdp.last_action),
    "target": ObservationTermCfg(func=vel_mdp.generated_commands, params={"command_name": CMD}),
  }
  if obstacles:
    actor_terms["lidar"] = ObservationTermCfg(
      func=mdp.lidar, params={"command_name": CMD}, noise=Unoise(n_min=-0.03, n_max=0.03)
    )
  if terrain:
    actor_terms["height_scan"] = ObservationTermCfg(
      func=mdp.height_scan, params={"command_name": CMD}, noise=Unoise(n_min=-0.02, n_max=0.02)
    )
  cfg.observations = {
    "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True,
                                 enable_corruption=not play, history_length=ACTOR_HISTORY),
  }

  # Rewards (each scaled by the step dt). Imagination re-implements them
  # (supermimic/imagine.py); the env copies are what the logs report.
  p = {"command_name": CMD}
  cfg.rewards = {
    # Chase.
    "chase_approach": RewardTermCfg(func=mdp.chase_approach, weight=1.5,
                                    params={**p, "max_speed": 5.0}),
    "chase_tag": RewardTermCfg(func=mdp.chase_tag, weight=15.0, params=p),
    "facing_target": RewardTermCfg(func=mdp.facing_target, weight=0.5, params=p),
    # Dodge.
    "evade_danger": RewardTermCfg(func=mdp.evade_danger, weight=-2.0, params={**p, "std": 0.4}),
    "hit": RewardTermCfg(func=mdp.hit, weight=-100.0, params=p),
    "evade_anticipate": RewardTermCfg(func=mdp.evade_anticipate, weight=-1.5,
                                      params={**p, "std": 0.3}),
    # Shared.
    "outside_arena": RewardTermCfg(func=mdp.outside_arena, weight=-2.0,
                                   params={**p, "margin": 0.5}),
    "upright": RewardTermCfg(func=mdp.upright, weight=1.0, params={"std": 0.4}),
    "tilt_rate_l2": RewardTermCfg(func=mdp.tilt_rate_l2, weight=-0.05),
    "hard_landing": RewardTermCfg(func=mdp.hard_landing, weight=-0.5),
    "crouch": RewardTermCfg(func=mdp.crouch, weight=-30.0, params=p),
    "joint_deviation_l2": RewardTermCfg(func=mdp.joint_deviation_l2, weight=-0.1),
    "base_height_low": RewardTermCfg(func=mdp.base_height_low, weight=-10.0,
                                     params={"min_height": 0.18}),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
    "joint_torques_l2": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-2e-5),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
  }
  if terrain:
    # Running fast over steps risks falls that flat ground never did; at the flat
    # weight the policy learned to accept them for a faster chase.
    cfg.rewards["terminated"].weight = -600.0
  if obstacles:
    cfg.rewards["obstacle_contact"] = RewardTermCfg(func=mdp.obstacle_contact, weight=-100.0,
                                                    params=p)
  # The generator runs its own target curriculum; terrain rows open with progress.
  cfg.curriculum = {}
  if terrain:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_by_progress, params={"steps": 12_000}
    )

  if play:
    cfg.episode_length_s = int(1e9)
  return cfg
