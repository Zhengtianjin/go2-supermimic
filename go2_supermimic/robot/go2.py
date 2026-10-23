"""Unitree Go2 model, actuators and collisions (after mjlab's Go1 constants)."""

from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_XML: Path = Path(__file__).parent / "go2.xml"
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_XML))


##
# Actuator config.
##

# Go2 uses the same rotor family as Go1 (GO-M8010-6). The rotor inertia is taken
# from the Go1 URDF; gear ratio 6.33, knee has an extra 1.5x linkage.
# Torque limits are the Menagerie/URDF values (23.7 Nm hip/thigh, 45.43 Nm knee).
# The torque-speed curve (flat up to half the no-load speed, then linear to zero)
# is an assumption to be calibrated on hardware.
ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6.33
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=30.1,
  effort_limit=23.7,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=20.07,
  effort_limit=45.43,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

# 0-2 physics steps (0-10 ms) of command latency between policy and motor.
ACTUATOR_DELAY_MAX_LAG = 2

GO2_HIP_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  saturation_effort=2.0 * HIP_ACTUATOR.effort_limit,
  velocity_limit=HIP_ACTUATOR.velocity_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)
GO2_KNEE_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  saturation_effort=2.0 * KNEE_ACTUATOR.effort_limit,
  velocity_limit=KNEE_ACTUATOR.velocity_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.29),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": -0.1,
    ".*L_hip_joint": 0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL"))
_foot_regex = "^[FR][LR]_foot_collision$"
_collision_regex = r".*_collision\d*"

# All collisions enabled; feet get condim 6 and their own friction.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1),
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_HIP_ACTUATOR_CFG,
    GO2_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


GO2_ACTION_SCALE: dict[str, float] = {}
for a in GO2_ARTICULATION.actuators:
  assert isinstance(a, DcMotorActuatorCfg)
  for n in a.target_names_expr:
    GO2_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())
